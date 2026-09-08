// Ported from: vllm/v1/engine/core.py @ e24d1b24
// (add_request, abort_requests, step — the T0 subset). See core.h for scope,
// deviations and deferrals.
#include "vllm/v1/engine/core.h"

#include <vector>
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include <cassert>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>

#include "vllm/v1/core/sched/output.h"          // GrammarOutput
#include "vllm/v1/structured_output/manager.h"  // StructuredOutputManager

namespace vllm::v1 {

void EngineCore::add_request(std::unique_ptr<Request> request) {
  // core.py:870-876 (preprocess_add_request): compile the request's grammar
  // before scheduling it. Upstream runs this in the input-processing thread; at
  // T0 it is synchronous here. No-op for a non-structured request or when no
  // manager is wired.
  if (structured_output_manager_ != nullptr &&
      request->use_structured_output()) {
    structured_output_manager_->grammar_init(*request);
  }
  // core.py:403 self.scheduler.add_request(request). The upstream request_id
  // type / pooling-task / kv_transfer validation and the abort_immediately
  // hook are deferred (see the file header).
  scheduler_.add_request(std::move(request));
}

void EngineCore::abort_requests(const std::vector<std::string>& request_ids) {
  // core.py:415 self.scheduler.finish_requests(request_ids,
  // RequestStatus.FINISHED_ABORTED). Our finish_requests takes one id, so
  // iterate (a no-op for unknown / already-finished ids, matching upstream).
  for (const std::string& request_id : request_ids) {
    scheduler_.finish_requests(request_id, RequestStatus::kFinishedAborted);
  }
}

namespace {

// PER-STEP PHASE TIMING (VT_ENGINE_STEP_STATS=1, default off, zero cost when
// off -- one predicted branch on a function-local static).
//
// WHY. The c=16 concurrency cell is the worst point on our whole curve against
// llama.cpp and the point where our OWN curve goes backwards, and a session of
// GEMM work established where it is NOT: the regression survives at both
// max_num_batched_tokens 128 and 256, so it is neither the matmul tactic nor a
// batching-denominator artifact. That leaves the serving layer, which so far has
// had NOT ONE number attached to it. This is the first instrument.
//
// WHAT IT MEASURES, stated precisely because the phases are not what they look
// like. The Vulkan backend batches dispatches and returns before the GPU has
// finished, so `execute` is the time to BUILD AND SUBMIT the step, not the time
// the device spends on it. That is exactly the point: if the serving layer is
// the problem, the cost appears in schedule/update, or in an execute that is
// large while the dispatch histogram says the GPU was busy for far less. Read
// this beside VT_VULKAN_DISPATCH_STATS, never on its own.
//
// Reported with p50 and p99 as well as a mean, because a scheduler pathology
// that bites on SOME steps -- a preemption, a queue refill, a chunk boundary --
// is invisible in a mean and obvious in a tail.
struct StepPhaseStats {
  std::vector<double> schedule_us, execute_us, update_us;
  bool enabled = false;

  StepPhaseStats() {
    const char* v = std::getenv("VT_ENGINE_STEP_STATS");
    enabled = v != nullptr && v[0] != 0 && v[0] != '0';
    if (enabled) {
      schedule_us.reserve(4096);
      execute_us.reserve(4096);
      update_us.reserve(4096);
    }
  }

  static void Report(const char* name, std::vector<double>& v) {
    if (v.empty()) return;
    std::vector<double> t = v;
    std::sort(t.begin(), t.end());
    double sum = 0.0;
    for (double x : t) sum += x;
    const size_t p50 = t.size() / 2;
    const size_t p99 = t.size() > 100 ? (t.size() * 99) / 100 : t.size() - 1;
    std::fprintf(stderr,
                 "[vt engine] %-9s n=%-6zu mean=%8.1f us  p50=%8.1f  p99=%8.1f  total=%8.1f ms\n",
                 name, t.size(), sum / static_cast<double>(t.size()), t[p50],
                 t[p99], sum / 1000.0);
  }

  ~StepPhaseStats() {
    if (!enabled) return;
    std::fprintf(stderr, "[vt engine] --- per-step phase timing ---\n");
    std::fprintf(stderr,
                 "[vt engine] execute is BUILD+SUBMIT, not GPU time -- read beside VT_VULKAN_DISPATCH_STATS\n");
    Report("schedule", schedule_us);
    Report("execute", execute_us);
    Report("update", update_us);
    // ★ DECILES, and they exist because the mean lied.
    //
    // execute read mean=302ms p50=139ms p99=2235ms. Dividing the TOTAL by the
    // step count and calling the quotient a per-step cost -- which is what was
    // done, twice -- is only valid on a distribution the mean describes, and this
    // one is not. 196 steps at the median would total 27 s; the run totalled 59.
    // So more than half the time lives in a handful of steps, and an average
    // spread over all of them names the wrong mechanism.
    //
    // Deciles separate the two populations without assuming which is which: a
    // decode step and a prefill step differ by more than an order of magnitude
    // here, and any fix aimed at the wrong one is aimed at nothing.
    if (!execute_us.empty()) {
      std::vector<double> t = execute_us;
      std::sort(t.begin(), t.end());
      std::fprintf(stderr, "[vt engine] execute deciles (ms):");
      for (int i = 0; i <= 10; ++i) {
        const size_t k = std::min(t.size() - 1, (t.size() * static_cast<size_t>(i)) / 10);
        std::fprintf(stderr, " %.1f", t[k] / 1000.0);
      }
      std::fprintf(stderr, "\n");
      double top = 0.0, all = 0.0;
      const size_t n90 = (t.size() * 9) / 10;
      for (size_t i = 0; i < t.size(); ++i) { all += t[i]; if (i >= n90) top += t[i]; }
      std::fprintf(stderr,
                   "[vt engine] ★ slowest 10%% of steps hold %.1f%% of execute time\n",
                   all > 0 ? 100.0 * top / all : 0.0);
    }
  }
};

StepPhaseStats& Steps() {
  static StepPhaseStats s;
  return s;
}

using StepClock = std::chrono::steady_clock;
inline double UsSince(StepClock::time_point t0) {
  return std::chrono::duration<double, std::micro>(StepClock::now() - t0).count();
}

}  // namespace

std::pair<std::map<int, EngineCoreOutputs>, bool> EngineCore::step() {
  // core.py:488 if not self.scheduler.has_requests(): return {}, False
  // Our Scheduler has no has_requests(); inline the interface.py default
  // (connector pending-push term deferred — no connector at T0).
  const bool has_requests = scheduler_.get_num_unfinished_requests() > 0 ||
                            scheduler_.has_finished_requests();
  if (!has_requests) {
    return {{}, false};
  }


  // core.py:490 scheduler_output = self.scheduler.schedule(...)
  // (the _should_throttle_prefills() arg is deferred — DP prefill balancing).
  StepPhaseStats& steps = Steps();
  const auto t_sched = StepClock::now();
  SchedulerOutput scheduler_output = scheduler_.schedule();
  if (steps.enabled) steps.schedule_us.push_back(UsSince(t_sched));
  const auto t_exec = StepClock::now();

  // core.py:491-499 execute the forward, then sample. The MRV2 runner's
  // execute_model returns None ("forward done"), so we always call sample_tokens.
  std::optional<ModelRunnerOutput> model_output =
      executor_.execute_model(scheduler_output);
  // Chunked-prefill progress, AFTER the forward so its elapsed_s is real wall
  // time rather than the cost of scheduling. No-op unless
  // VT_SERVER_PREFILL_PROGRESS / VT_SERVER_VERBOSE is on.
  scheduler_.LogPrefillAfterExecute(scheduler_output);
  // core.py:492 grammar_output = self.scheduler.get_grammar_bitmask(...). Nullopt
  // when no structured request is scheduled (or no manager is wired); threaded to
  // sample_tokens (Task 3 consumes it).
  const std::optional<GrammarOutput> grammar_output =
      scheduler_.get_grammar_bitmask(scheduler_output);
  if (!model_output.has_value()) {
    model_output = executor_.sample_tokens(grammar_output);
  }

  // core.py:503 self._process_aborts_queue() — deferred (no in-flight aborts at
  // T0; synchronous execution leaves no window).

  // core.py:504-506 engine_core_outputs = scheduler.update_from_output(...).
  // Our update_from_output returns a single flat EngineCoreOutputs (T0 single
  // client); wrap it in the per-client map to keep the upstream return shape.
  // Upstream builds dict[client_index, EngineCoreOutputs] only for clients that
  // produced outputs this step (the dict comprehension over `outputs.items()`),
  // so a 0-output step (e.g. a finished-req flush) yields an empty map. We drop
  // the finished_requests-only entries (that DP-signalling field is deferred),
  // so the entry is present iff there are token outputs.
  if (steps.enabled) steps.execute_us.push_back(UsSince(t_exec));
  const auto t_upd = StepClock::now();
  EngineCoreOutputs engine_core_outputs =
      scheduler_.update_from_output(scheduler_output, *model_output);
  // Attach this step's scheduler snapshot + the engine_core_timestamp the
  // frontend threads into IterationStats for TTFT/ITL (core.py builds
  // EngineCoreOutputs(scheduler_stats=scheduler.make_stats(), timestamp=...)).
  // Inert on the no-logger path (the frontend never reads it).
  engine_core_outputs.scheduler_stats = scheduler_.make_stats();
  engine_core_outputs.timestamp = MonotonicSeconds();
  std::map<int, EngineCoreOutputs> outputs_by_client;
  if (!engine_core_outputs.outputs.empty()) {
    const int client_index = engine_core_outputs.engine_index;
    outputs_by_client.emplace(client_index, std::move(engine_core_outputs));
  }

  // core.py:509-517 post_step: feed the drafter's out-of-band proposal back to
  // the scheduler for the next step. Inert unless a speculator is configured.
  if (steps.enabled) steps.update_us.push_back(UsSince(t_upd));

  const bool model_executed = scheduler_output.total_num_scheduled_tokens > 0;
  post_step(model_executed);

  // core.py:508 return outputs, scheduler_output.total_num_scheduled_tokens > 0.
  return {std::move(outputs_by_client), model_executed};
}

void EngineCore::post_step(bool model_executed) {
  // core.py:509-517 (:617 at 555967922). Under async scheduling the draft
  // token ids are updated in the worker process instead (SPEC-DFLASH2 W7,
  // #1824): the AsyncScheduler ships -1 placeholders and the runner fills them
  // from its own propose, so the out-of-band pull below must NOT run — it
  // would overwrite the placeholders with values the scheduler must never
  // carry under async. The guard cannot be "which step function called me":
  // EngineCoreProc's busy loop AND the depth-1 LLMEngine::step both reach
  // here whatever the resolution, so it mirrors upstream's
  // `not self.async_scheduling` (the scheduler class IS the resolved value,
  // model_loader.cpp::MakeScheduler).
  if (!check_for_draft_tokens_ || scheduler_.async_scheduling() ||
      !model_executed) {
    return;
  }
  std::optional<DraftTokenIds> draft_token_ids = executor_.take_draft_token_ids();
  static const bool spec_post_trace = std::getenv("VT_SPEC_TRACE") != nullptr;
  if (spec_post_trace) {
    std::fprintf(stderr, "[spec-post_step] pulled=%d rows=%zu\n",
                 draft_token_ids.has_value() ? 1 : 0,
                 draft_token_ids.has_value() ? draft_token_ids->req_ids.size() : 0);
  }
  if (draft_token_ids.has_value()) {
    scheduler_.update_draft_token_ids(*draft_token_ids);
  }
}

std::pair<std::map<int, EngineCoreOutputs>, bool>
EngineCore::step_with_batch_queue() {
  // core.py:519-632. Fulfilling the batch queue has priority over consuming an
  // output: schedule + execute a new batch (if room + work), then block on the
  // OLDEST queued batch. Our executor resolves eagerly, so "execute" here means
  // the forward + sample ran synchronously and the ModelRunnerOutput is already
  // in hand (the AsyncScheduler placeholder accounting is what lets step N+1 be
  // scheduled before N's output is consumed — see core.h).
  assert(batch_queue_.size() < static_cast<std::size_t>(batch_queue_size_) &&
         "step_with_batch_queue called with a full batch queue");

  bool model_executed = false;
  std::optional<SchedulerOutput> deferred_scheduler_output;

  const bool has_requests = scheduler_.get_num_unfinished_requests() > 0 ||
                            scheduler_.has_finished_requests();
  if (has_requests) {
    // core.py:547 schedule (non-blocking; may return an empty batch).
    SchedulerOutput scheduler_output = scheduler_.schedule();
    // core.py:549-551 execute_model(non_block=True). MRV2: nullopt (forward
    // done). A failed eager forward throws here, through the engine-fatal guard.
    std::optional<ModelRunnerOutput> exec_out =
        executor_.execute_model(scheduler_output);
    // Same diagnostic as the synchronous path above.
    scheduler_.LogPrefillAfterExecute(scheduler_output);
    // core.py:552-553 model_executed = total_num_scheduled_tokens > 0
    // (is_ec_consumer is always true for us — no EC transfer).
    model_executed = scheduler_output.total_num_scheduled_tokens > 0;

    std::unique_ptr<AsyncModelRunnerOutput> async_out;
    if (!model_executed) {
      // core.py:555-557 no sampling required — carry the (empty) forward result
      // as an already-materialized async output.
      async_out = std::make_unique<ReadyModelRunnerOutput>(
          exec_out.value_or(ModelRunnerOutput{}));
    } else if (!scheduler_output.pending_structured_output_tokens) {
      // core.py:559-567 not waiting on any tokens — get the grammar bitmask and
      // sample immediately. sample_tokens_async issues the sampled-id copy but,
      // under async scheduling, DEFERS the host wait to get_output() at consume
      // time (below), so the copy overlaps the next step's forward; a synchronous
      // runner returns an already-materialized ReadyModelRunnerOutput.
      const std::optional<GrammarOutput> grammar_output =
          scheduler_.get_grammar_bitmask(scheduler_output);
      async_out = executor_.sample_tokens_async(grammar_output);
    } else {
      // core.py:568-571 defer sampling until the prior step's output is
      // processed (a structured request is waiting on in-flight tokens).
      deferred_scheduler_output = scheduler_output;
    }

    if (!deferred_scheduler_output.has_value()) {
      // core.py:573-575 add this step's result to the queue (front).
      assert(async_out != nullptr);
      batch_queue_.push_front(
          BatchQueueItem{std::move(async_out), std::move(scheduler_output)});
      // core.py:576-581 don't block on the next result unless the queue is full
      // or there are no more requests to schedule.
      const bool more_requests =
          scheduler_.get_num_unfinished_requests() > 0 ||
          scheduler_.has_finished_requests();
      if (batch_queue_.size() < static_cast<std::size_t>(batch_queue_size_) &&
          (model_executed || more_requests)) {
        return {{}, model_executed};
      }
    }
  } else if (batch_queue_.empty()) {
    // core.py:583-587 queue empty and no requests — nothing to do.
    return {{}, false};
  }

  // core.py:589-590 block until the next result is available (pop the OLDEST).
  BatchQueueItem item = std::move(batch_queue_.back());
  batch_queue_.pop_back();

  // core.py:589-590 future.result() → get_output(): resolve the (possibly
  // deferred) sampled-id copy NOW — the ONE host block, and it is off the model's
  // critical path because it overlapped the batch we just scheduled+forwarded
  // above. A synchronous runner's output is already materialized (no wait).
  ModelRunnerOutput model_output = item.async_output->get_output();

  // core.py:604-607 update the scheduler from the popped batch's output.
  EngineCoreOutputs engine_core_outputs =
      scheduler_.update_from_output(item.scheduler_output, model_output);

  // core.py:609-630 grammar deferral: now that the prior output is processed,
  // compute the deferred batch's bitmask, sample it, and append it to the queue.
  // (The runner's execute_model stash from this step is still valid — no other
  // execute_model ran between it and here.)
  if (deferred_scheduler_output.has_value()) {
    // core.py:718-731 (SPEC-DFLASH2 W7, #1824): with drafts under async
    // scheduling, the deferred batch's scheduled_spec_decode_tokens still
    // holds the -1 placeholders (the worker fill patches only its own copy).
    // Pull the worker's real drafts and rewrite them into the deferred output
    // so the grammar bitmask below reads real token ids; a slot the worker
    // could not fill is -1-padded and recorded in num_invalid_spec_tokens for
    // the bitmask computation to skip. No-op without a speculator.
    if (check_for_draft_tokens_) {
      std::optional<DraftTokenIds> draft_token_ids =
          executor_.take_draft_token_ids();
      if (draft_token_ids.has_value()) {
        scheduler_.update_draft_token_ids_in_output(*draft_token_ids,
                                                    *deferred_scheduler_output);
      }
    }
    const std::optional<GrammarOutput> grammar_output =
        scheduler_.get_grammar_bitmask(*deferred_scheduler_output);
    std::unique_ptr<AsyncModelRunnerOutput> sampled =
        executor_.sample_tokens_async(grammar_output);
    batch_queue_.push_front(
        BatchQueueItem{std::move(sampled), std::move(*deferred_scheduler_output)});
  }

  // Attach this step's scheduler snapshot + engine-core timestamp, identically
  // to the synchronous step() above. Upstream stamps BOTH inside the path the
  // two step functions share — scheduler_stats in Scheduler.update_from_output
  // (scheduler.py:1938-1951) and timestamp in EngineCoreOutputs.__post_init__
  // (engine/__init__.py:249-251) — so upstream's step_with_batch_queue
  // (core.py:622-720) stamps nothing extra precisely because it already has
  // them.
  //
  // Until #277 this path stamped NEITHER: the async-scheduling serving stack
  // (LoadedEngine resolves max_concurrent_batches=2 whenever the runner
  // supports it) published a default-constructed SchedulerStats — every gauge
  // 0 — and a timestamp of 0.0, which turns every TTFT/e2e observation into
  // `-arrival_time`. The former VT_TTFT_DUMP-only timestamp stamp is subsumed
  // here; the diagnostic reads exactly the value it always did.
  engine_core_outputs.scheduler_stats = scheduler_.make_stats();
  engine_core_outputs.timestamp = MonotonicSeconds();

  std::map<int, EngineCoreOutputs> outputs_by_client;
  if (!engine_core_outputs.outputs.empty()) {
    const int client_index = engine_core_outputs.engine_index;
    outputs_by_client.emplace(client_index, std::move(engine_core_outputs));
  }
  // core.py:632 return engine_core_outputs, model_executed.
  return {std::move(outputs_by_client), model_executed};
}

}  // namespace vllm::v1
