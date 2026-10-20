// Ported from: vllm/v1/engine/__init__.py + vllm/v1/outputs.py @ e24d1b24
//
// The EngineCore I/O message types the V1 engine passes across its process
// boundary: EngineCoreRequest (frontend -> core), SamplerOutput /
// ModelRunnerOutput (runner -> scheduler), EngineCoreOutput / EngineCoreOutputs
// (core -> frontend). These are plain value carriers; field names, defaults and
// the one behavioral helper (finished/Finished) are mirrored 1:1 with upstream.
//
// Scope = the T0 subset the scheduler's update_from_output (M1.4) and the
// OutputProcessor (M1.8) actually consume. FinishReason is REUSED from
// vllm/v1/request.h (upstream defines it in vllm/v1/engine/__init__.py and
// request.py imports it); it is NOT redefined here.
//
// DEFERRED upstream fields, intentionally omitted — later units slot these in
// without reshaping the structs:
//   EngineCoreRequest: mm_features (multimodal) and pooling_params are now
//     PRESENT (pooling_params since ARCH-ONE-SURFACE ROW 6); lora_request,
//     cache_salt, data_parallel_rank, prompt_embeds, prompt_is_token_ids,
//     client_index, current_wave, trace_headers, resumable,
//     external_req_id, reasoning_ended / reasoning_parser_kwargs,
//     abort_immediately, and the params property. (eos_token_id is NOT a field
//     upstream either — it rides on sampling_params.eos_token_id.)
//   SamplerOutput: logprobs_tensors now carries the real LogprobsTensors payload
//     (vllm/v1/outputs.py, ported at M1.7); the sampler's gather_logprobs fills
//     it. It stays std::optional (None => no logprobs requested this step).
//   ModelRunnerOutput: logprobs (LogprobsLists), prompt_logprobs_dict (ROAD-V1-C7
//     SAMPLE-LOGPROBS) and pooler_output (ARCH-ONE-SURFACE ROW 6) are now
//     PRESENT; kv_connector_output / ec_connector_output (P/D KV transfer),
//     num_nans_in_logits, cudagraph_stats, routed_experts, and the
//     with_kv_conn_output_only / EMPTY_MODEL_RUNNER_OUTPUT helpers stay deferred.
//   EngineCoreOutput: new_logprobs / new_prompt_logprobs_tensors are now PRESENT
//     (ROAD-V1-C7); events (EngineCoreEvent) is now PRESENT
//     (SERVE-RESPONSE-METRICS — the per-request QUEUED/SCHEDULED/PREEMPTED
//     timing events the scheduler drains onto each output); pooling_output is
//     now PRESENT (ARCH-ONE-SURFACE ROW 6); prefill_stats is now PRESENT (the
//     cached/computed prompt-token split the frontend counters read);
//     kv_transfer_params, trace_headers, routed_experts, num_nans_in_logits
//     deferred.
//   EngineCoreOutputs: scheduler_stats (SchedulerStats), utility_output,
//     finished_requests, wave_complete / start_wave (DP wave signalling), and
//     the __post_init__ monotonic-timestamp default (the frontend stamps it).
//
// DEVIATIONS, recorded:
//   - SamplerOutput.sampled_token_ids is a torch.Tensor [num_reqs,
//     max_num_generated_tokens] upstream; here it is the list-of-lists form
//     (vector<vector<int32_t>>) matching ModelRunnerOutput.sampled_token_ids,
//     which is how the scheduler reads it. T0 non-spec decode is [num_reqs, 1].
//     PLACEHOLDER_TOKEN_ID (-1) padding is a device-tensor concern that does
//     not apply to the ragged list form.
//   - EngineCoreOutput.stop_reason is int | str | None upstream; represented
//     here as std::optional<std::string> (the OutputProcessor passes it through
//     to CompletionOutput.stop_reason). A stop_token_id match stringifies its
//     id; a stop-string match carries the string. A std::variant would be a
//     closer union but is heavier than a T0 value carrier warrants.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "vllm/model_executor/layers/pooler/pooling_params.h"  // PoolingParams (pooling seam)
#include "vllm/multimodal/inputs.h"  // multimodal::MultiModalFeatureSpec (mm seam)
#include "vllm/sampling_params.h"
#include "vllm/v1/metrics/stats.h"  // vllm::v1::SchedulerStats (per-step stats)
#include "vllm/v1/outputs.h"  // vllm::v1::LogprobsTensors (SamplerOutput payload)
#include "vllm/v1/request.h"  // vllm::v1::FinishReason (reused, not redefined)

namespace vllm::v1 {

// PLACEHOLDER_TOKEN_ID: upstream pads the SamplerOutput token tensor with -1.
// Kept for reference; the list-of-lists form here is ragged and unpadded.
inline constexpr int32_t kPlaceholderTokenId = -1;

// EngineCoreRequest: the frontend -> EngineCore message (T0 field subset).
// msgspec.Struct upstream; a plain value carrier here.
struct EngineCoreRequest {
  std::string request_id;
  // Upstream is list[int] | None; T0 pure-token path always has a list.
  std::vector<int32_t> prompt_token_ids;
  // Already PostInit'd / validated by the frontend (the InputProcessor runs
  // __post_init__ before building this message), so Request::FromEngineCoreRequest
  // does NOT re-validate. The model's EOS id rides on sampling_params.eos_token_id.
  SamplingParams sampling_params;
  double arrival_time = 0.0;
  // priority (engine/__init__.py EngineCoreRequest.priority): the scheduling
  // priority carried from the frontend request (OpenAI `priority` field) into
  // Request.priority. Default 0 keeps FCFS parity when unset.
  int priority = 0;
  // mm_features (EngineCoreRequest.mm_features): processed multimodal placeholder
  // specs (mm-hash + span + encoder input). EMPTY for text-only requests -> every
  // downstream mm hook is a no-op and the engine is byte-identical. Populated by
  // the mm input pipeline (M1); consumed by the encoder cache / vision tower (M2).
  std::vector<multimodal::MultiModalFeatureSpec> mm_features = {};
  // cache_salt (EngineCoreRequest.cache_salt): optional prefix-cache namespace
  // salt carried from the frontend (OpenAI `cache_salt`) into Request.cache_salt
  // and folded into the first block's hash. nullopt -> byte-identical text path.
  std::optional<std::string> cache_salt = std::nullopt;
  // lora_name (EngineCoreRequest.lora_request.lora_name): the LoRA adapter name,
  // carried into Request.lora_name for the prefix-cache extra-key path. nullopt
  // for a base-model request. Full LoRA runtime is LORA-RUNTIME.
  std::optional<std::string> lora_name = std::nullopt;
  // pooling_params (EngineCoreRequest.pooling_params, ARCH-ONE-SURFACE ROW 6):
  // set iff this is a POOLING-task request (upstream `params: SamplingParams |
  // PoolingParams` — the union collapses to sampling_params + this optional).
  // nullopt on every generation request -> byte-identical text path. Carried
  // into Request::pooling_params; the scheduler's pooling stop
  // (scheduler.py:1718-1721 mirror) fires only when it is set.
  std::optional<PoolingParams> pooling_params = std::nullopt;
};

// SamplerOutput (vllm/v1/outputs.py): the raw sampler result for a step.
struct SamplerOutput {
  // [num_reqs, max_num_generated_tokens]; T0 non-spec decode is [num_reqs, 1].
  std::vector<std::vector<int32_t>> sampled_token_ids;
  // logprobs_tensors upstream (LogprobsTensors | None): the sampler's
  // gather_logprobs payload, None when no logprobs were requested this step.
  std::optional<LogprobsTensors> logprobs_tensors;
  // ENG-ASYNC-SCHED W3: set when Sampler::forward was given a device out-tensor
  // and produced the sampled ids DEVICE-RESIDENT (all-greedy fast path) WITHOUT a
  // host download — `sampled_token_ids` is then intentionally empty and the
  // async-output D2H (AsyncGPUModelRunnerOutput) owns the single copy to host.
  // Upstream keeps sampler_output.sampled_token_ids on the GPU unconditionally
  // (async_utils.py:31); this flag records that we took the same device-resident
  // route on our greedy gate path (sync path leaves it false, byte-identical).
  bool sampled_on_device = false;
};

// ModelRunnerOutput (vllm/v1/outputs.py): the runner -> scheduler result,
// serialized across the process boundary. update_from_output reads req_ids
// (ordering), req_id_to_index and sampled_token_ids at T0.
struct ModelRunnerOutput {
  // [num_reqs] — the per-request ordering the scheduler iterates.
  std::vector<std::string> req_ids;
  // req_id -> index into the num_reqs-major arrays.
  std::map<std::string, int> req_id_to_index;
  // num_reqs x num_generated_tokens (ragged; per-request length can differ
  // under speculative/jump decode). T0 non-spec decode is [num_reqs, 1].
  std::vector<std::vector<int32_t>> sampled_token_ids;
  // logprobs (ModelRunnerOutput.logprobs, LogprobsLists | None): the batch-wide
  // sample logprobs the sampler's gather_logprobs produced this step (one row
  // per num_logits position, in req order). None when no request asked for
  // logprobs. The scheduler slices it per request (slice_request). Our
  // host-vector LogprobsTensors doubles as the LogprobsLists twin.
  std::optional<LogprobsTensors> logprobs;
  // prompt_logprobs_dict (ModelRunnerOutput.prompt_logprobs_dict): req_id ->
  // the prompt-position logprobs tensors, populated during prefill for requests
  // that asked for prompt_logprobs. Empty on the pure sample-logprobs path. The
  // tensor SOURCE (lm_head over prompt positions) is a runner/prefill addition
  // (SAMPLE-PROMPT-LOGPROBS); the OUTPUT plumbing below consumes it 1:1.
  std::map<std::string, LogprobsTensors> prompt_logprobs_dict;
  // pooler_output (ModelRunnerOutput.pooler_output, vllm/v1/outputs.py;
  // ARCH-ONE-SURFACE ROW 6): one entry per req in `req_ids` order on a POOLING
  // model's step — the pooled vector, or nullopt while the request is still
  // consuming prefill chunks (the is_valid=false rows,
  // pool/pooling_runner.py:40-41). EMPTY (size 0) on every generation step ->
  // byte-identical text path.
  std::vector<std::optional<std::vector<float>>> pooler_output;
};

// DraftTokenIds (vllm/v1/outputs.py:310-315): the drafter's proposal for the
// NEXT verify step, produced OUT-OF-BAND by the runner (take_draft_token_ids)
// and fed back to the scheduler by the engine core (core.py:511-517 ->
// Scheduler::update_draft_token_ids). Parallel arrays: draft_token_ids[i] is the
// list of draft ids for req_ids[i]. Frozen SPEC-MTP metadata ABI (I2) — I5's
// runner populates it, I3's rejection sampler verifies against it.
struct DraftTokenIds {
  // [num_reqs].
  std::vector<std::string> req_ids;
  // num_reqs x num_draft_tokens (ragged; per-request draft count can differ).
  std::vector<std::vector<int32_t>> draft_token_ids;
};

// EngineCoreOutput (vllm/v1/engine/__init__.py): the per-request core -> frontend
// delta. msgspec.Struct upstream; a plain value carrier here.
struct EngineCoreOutput {
  std::string request_id;
  std::vector<int32_t> new_token_ids;
  // finish_reason is None until the request finishes (reused FinishReason).
  std::optional<FinishReason> finish_reason;
  // int | str | None upstream — see DEVIATIONS in the file header.
  std::optional<std::string> stop_reason;
  // new_logprobs (EngineCoreOutput.new_logprobs, LogprobsLists | None): this
  // request's slice of the step's sample logprobs (scheduler.py:1821
  // logprobs.slice_request), None unless the request asked for logprobs.
  std::optional<LogprobsTensors> new_logprobs;
  // new_prompt_logprobs_tensors (EngineCoreOutput.new_prompt_logprobs_tensors,
  // LogprobsTensors | None): the prompt-position logprobs produced during
  // prefill for a prompt_logprobs request (scheduler.py:1827/1836).
  std::optional<LogprobsTensors> new_prompt_logprobs_tensors;
  // events (EngineCoreOutput.events, list[EngineCoreEvent] | None): the
  // per-request QUEUED/SCHEDULED/PREEMPTED events the scheduler recorded for
  // this request since the last step, drained here via Request::take_events in
  // update_from_output. nullopt (omit-default) when log_stats is off or no event
  // fired this step -> byte-identical no-stats path. Consumed by
  // IterationStats.update_from_events (stats.py:428-450) to fill the request's
  // queue/prefill/inference timing intervals + the preemption counter.
  std::optional<std::vector<EngineCoreEvent>> events;
  // pooling_output (EngineCoreOutput.pooling_output, ARCH-ONE-SURFACE ROW 6):
  // the pooled vector of a finished POOLING request (scheduler.py:1837
  // `pooling_output=pooler_output`). nullopt on every generation output ->
  // byte-identical text path.
  std::optional<std::vector<float>> pooling_output;
  // prefill_stats (EngineCoreOutput.prefill_stats, engine/__init__.py:211 @
  // 0.28.0): set on a request's first output only, from
  // Request::take_prefill_stats. The frontend folds it into
  // IterationStats.prompt_token_stats and the request's num_cached_tokens.
  std::optional<PrefillStats> prefill_stats;

  // finished (property): a request is finished iff finish_reason is set.
  bool Finished() const { return finish_reason.has_value(); }
};

// EngineCoreOutputs (vllm/v1/engine/__init__.py): the batched core -> frontend
// message for one engine step.
struct EngineCoreOutputs {
  int engine_index = 0;
  // [num_reqs]
  std::vector<EngineCoreOutput> outputs;
  // Monotonic timestamp; upstream __post_init__ stamps time.monotonic() when
  // left at 0.0. EngineCore::step() stamps it (the engine_core_timestamp the
  // frontend threads into IterationStats.update_from_output for TTFT/ITL).
  double timestamp = 0.0;
  // This step's scheduler snapshot (running/waiting/kv-usage + the per-step
  // prefix-cache delta). Upstream: EngineCoreOutputs.scheduler_stats
  // (vllm/v1/engine/__init__.py), filled by Scheduler.make_stats()
  // (vllm/v1/core/sched/scheduler.py:2399-2436). Default-constructed (all zero)
  // on the no-stats path so it is inert unless a stat logger consumes it.
  SchedulerStats scheduler_stats;
};

}  // namespace vllm::v1
