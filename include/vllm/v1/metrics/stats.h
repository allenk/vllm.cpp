// Ported from: vllm/v1/metrics/stats.py @ e24d1b24
//
// Scope: the CACHE-HIT statistics surface. Two ported types plus their shared
// base, mirroring upstream 1:1:
//   * BaseCacheStats            (stats.py:18-32)  — reset/requests/queries/hits
//   * PrefixCacheStats          (stats.py:115-142) — adds the three
//     mutually-exclusive preempted_* counters and record()
//   * CachingMetrics            (stats.py:35-111) — the hit-rate aggregator over
//     a sliding window of the most recent N REQUESTS (not steps, not tokens)
//
// WHY THIS EXISTS: before this port there were NO prefix-cache statistics at
// any level in the tree (`src/vllm/v1/core/kv_cache_manager.cpp:139` was a
// deferral comment). The benchmark protocol makes "prove cache hits in every
// arm" mandatory, so an arm without a hit counter is VOID — which made every
// caching / offloading benchmark unrunnable. These counters are the evidence
// surface for `KV-PREFIX-CACHE`, `KV-OFFLOAD` and
// `BACKEND-GATE-CUDA-SGLANG-PREFIX`.
//
// SEMANTICS THAT ARE EASY TO GET WRONG (all mirrored exactly):
//   - `queries` and `hits` count TOKENS, not requests or blocks. `queries` is
//     request.num_tokens (the whole prompt), `hits` is the number of tokens
//     served from cache (stats.py:236-239).
//   - The preempted_* counters are MUTUALLY EXCLUSIVE with requests/queries/
//     hits: a request that has been preempted at least once contributes to the
//     preempted_* triple ONLY (stats.py:134-142). This keeps a preempted
//     request's guaranteed second-pass hit from inflating the headline rate.
//   - CachingMetrics' window is measured in REQUESTS (default 1000), and the
//     eviction loop always keeps at least one entry, so the most recent
//     observation is never dropped (stats.py:96-104).
//   - An observation with requests == 0 is NOT appended, so idle steps cannot
//     flush useful history out of the window (stats.py:70-72).
//   - `stats.reset` (set by reset_prefix_cache) resets the aggregate BEFORE the
//     current observation is folded in (stats.py:65-68).
//
// DEVIATIONS: none in behaviour. Upstream's dataclass field defaults and the
// deque of (requests, queries, hits) triples are a plain struct and a
// std::deque here.
#ifndef VLLM_V1_METRICS_STATS_H_
#define VLLM_V1_METRICS_STATS_H_

#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <deque>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

// stats.py:12 `from vllm.v1.spec_decode.metrics import SpecDecodingStats` — the
// spec-decode aggregate is defined beside the counters that consume it, and
// SchedulerStats only carries it.
#include "vllm/v1/spec_decode/metrics.h"

namespace vllm::v1 {

// Seconds from a steady monotonic clock. The engine stamps EngineCoreOutputs
// with this (upstream: time.monotonic() in EngineCoreOutputs.__post_init__) and
// RequestState records arrival time from the same clock, so the TTFT / ITL /
// e2e intervals IterationStats derives are always non-negative. Wall-clock vs
// monotonic is elided (a single steady clock) — the intervals, not absolute
// timestamps, are what the histograms observe.
inline double MonotonicSeconds() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Upstream: @dataclass BaseCacheStats (vllm/v1/metrics/stats.py:18-32).
struct BaseCacheStats {
  // Whether the cache was reset since the last observation.
  bool reset = false;
  // The number of requests in this update.
  int64_t requests = 0;
  // The number of queried TOKENS in these requests.
  int64_t queries = 0;
  // The number of TOKENS served from cache in these requests.
  int64_t hits = 0;
};

// Upstream: @dataclass PrefixCacheStats (vllm/v1/metrics/stats.py:115-142).
struct PrefixCacheStats : BaseCacheStats {
  // The number of previously-preempted requests in this update.
  int64_t preempted_requests = 0;
  // The `queries` number for preempted requests.
  int64_t preempted_queries = 0;
  // The `hits` number for preempted requests.
  int64_t preempted_hits = 0;

  // Aggregate one request's lookup. `preempted` is
  // `request.num_preemptions > 0` at the call site. Upstream: record()
  // (stats.py:130-142).
  void record(int64_t num_tokens, int64_t num_hits, bool preempted) {
    if (preempted) {
      preempted_requests += 1;
      preempted_queries += num_tokens;
      preempted_hits += num_hits;
    } else {
      requests += 1;
      queries += num_tokens;
      hits += num_hits;
    }
  }
};

// Upstream: class CachingMetrics (vllm/v1/metrics/stats.py:35-111).
// Hit rate over the most recent `max_recent_requests` requests.
class CachingMetrics {
 public:
  explicit CachingMetrics(int64_t max_recent_requests = 1000)
      : max_recent_requests_(max_recent_requests) {}

  // Fold one take-and-swap observation into the window.
  // Upstream: observe() (stats.py:58-104).
  void observe(const BaseCacheStats& stats) {
    // reset_prefix_cache was invoked before the current update: reset the
    // metrics BEFORE aggregating the current stats.
    if (stats.reset) {
      reset();
    }
    // Do NOT append empty stats: an idle step would otherwise slide useful
    // history out of the window.
    if (stats.requests == 0) {
      return;
    }
    queue_.emplace_back(stats.requests, stats.queries, stats.hits);
    aggregated_requests_ += stats.requests;
    aggregated_query_total_ += stats.queries;
    aggregated_query_hit_ += stats.hits;

    // Drop the oldest observations until the window fits. The `size() > 1`
    // guard preserves the latest observation regardless of its size.
    while (queue_.size() > 1 && aggregated_requests_ > max_recent_requests_) {
      const auto& front = queue_.front();
      aggregated_requests_ -= std::get<0>(front);
      aggregated_query_total_ -= std::get<1>(front);
      aggregated_query_hit_ -= std::get<2>(front);
      queue_.pop_front();
    }
  }

  // Upstream: reset() (stats.py:106-111).
  void reset() {
    aggregated_requests_ = 0;
    aggregated_query_total_ = 0;
    aggregated_query_hit_ = 0;
    queue_.clear();
  }

  // Upstream: @property empty.
  bool empty() const { return aggregated_requests_ == 0; }

  // Upstream: @property hit_rate. 0.0 when nothing has been queried.
  double hit_rate() const {
    if (aggregated_query_total_ == 0) {
      return 0.0;
    }
    return static_cast<double>(aggregated_query_hit_) /
           static_cast<double>(aggregated_query_total_);
  }

  int64_t aggregated_requests() const { return aggregated_requests_; }
  int64_t aggregated_query_total() const { return aggregated_query_total_; }
  int64_t aggregated_query_hit() const { return aggregated_query_hit_; }

 private:
  int64_t max_recent_requests_;
  int64_t aggregated_requests_ = 0;
  int64_t aggregated_query_total_ = 0;
  int64_t aggregated_query_hit_ = 0;
  // (requests, queries, hits) for the most recent observations.
  std::deque<std::tuple<int64_t, int64_t, int64_t>> queue_;
};

// Ported from: @dataclass SchedulerStats (vllm/v1/metrics/stats.py:186-215).
// The always-on subset the Prometheus scheduler-state metrics read (loggers.py
// PrometheusStatLogger.record), plus spec_decoding_stats. The remaining advanced
// fields (DP wave, cudagraph, kv-connector, lora, eviction events) are deferred
// with their config-gated metric families and are not part of the SERVE-METRICS
// core.
struct SchedulerStats {
  // Number of requests currently in a model-execution batch.
  int64_t num_running_reqs = 0;
  // Length of the waiting request queue.
  int64_t num_waiting_reqs = 0;
  // KV-cache usage fraction in [0, 1]. 1.0 == 100% usage.
  double kv_cache_usage = 0.0;
  // This step's take-and-swap prefix-cache observation (TOKEN counts).
  PrefixCacheStats prefix_cache_stats;
  // stats.py:206 `spec_decoding_stats: SpecDecodingStats | None = None`.
  // SET ONLY on a step that verified at least one draft token (#2770): an
  // engine with no speculator never sets it, and neither does a speculative
  // engine's prefill step, which is why it is an optional rather than a
  // zero-valued aggregate. `PrometheusStatLogger::Record` folds it into the
  // spec_decode counter families when it is present (loggers.py:1140-1143).
  std::optional<spec_decode::SpecDecodingStats> spec_decoding_stats;
};

// Ported from: @dataclass FinishedRequestStats (vllm/v1/metrics/stats.py:236-259).
// One finished request's per-request timing/token breakdown. `finish_reason`
// is the stringified FinishReason ("stop"/"length"/"abort") that labels
// vllm:request_success_total.
struct FinishedRequestStats {
  std::string finish_reason = "stop";
  double e2e_latency = 0.0;
  int64_t num_prompt_tokens = 0;
  int64_t num_generation_tokens = 0;
  // Present when the request set max_tokens (< 0 == unset → not observed).
  int64_t max_tokens_param = -1;
  double queued_time = 0.0;
  double prefill_time = 0.0;
  double inference_time = 0.0;
  double decode_time = 0.0;
  double mean_time_per_output_token = 0.0;
  // The prompt tokens this request did not compute (its first output's
  // PrefillStats.num_cached_tokens). vllm:request_prefill_kv_computed_tokens
  // observes num_prompt_tokens minus this.
  int64_t num_cached_tokens = 0;
};

// Ported from: @dataclass PrefillStats (vllm/v1/metrics/stats.py:259-299 @
// 0.28.0). The breakdown of a request's first scheduled prefill. The scheduler
// sets it each time the request is looked up in the waiting queue before its
// first preemption (scheduler.py:898-905), so the lookup that admits the
// request is the one reported, and hands it to the frontend on the request's
// first output (scheduler.py:1962-1969).
//
// DEFERRED: num_cache_creation_tokens and finalize(). Their only consumer is
// RequestOutput.num_cache_creation_tokens, which is not ported.
struct PrefillStats {
  // Tokens to be prefilled.
  int64_t num_prompt_tokens = 0;
  // Tokens prefilled locally (actual compute work).
  int64_t num_computed_tokens = 0;
  // Tokens prefilled without compute work (local + external).
  int64_t num_cached_tokens = 0;
  // Tokens served from the local prefix cache.
  int64_t num_local_cached_tokens = 0;
  // Tokens served by an external KV transfer.
  int64_t num_external_cached_tokens = 0;

  // Upstream: set() (stats.py:278-291).
  void set(int64_t prompt_tokens, int64_t local_cached_tokens,
           int64_t external_cached_tokens) {
    const int64_t cached = local_cached_tokens + external_cached_tokens;
    assert(cached <= prompt_tokens);
    num_prompt_tokens = prompt_tokens;
    num_computed_tokens = prompt_tokens - cached;
    num_cached_tokens = cached;
    num_local_cached_tokens = local_cached_tokens;
    num_external_cached_tokens = external_cached_tokens;
  }
};

// Ported from: @dataclass PromptTokenStats (vllm/v1/metrics/stats.py:301-346 @
// 0.28.0). The prompt tokens of the requests whose first output arrived this
// step, by source. Invariants: computed + local_cache_hit +
// external_kv_transfer == total, and local_cache_hit + external_kv_transfer ==
// cached_tokens.
struct PromptTokenStats {
  // The `source` label values of vllm:prompt_tokens_by_source, in upstream's
  // ALL_SOURCES order.
  static constexpr std::array<std::string_view, 3> kAllSources = {
      "local_compute", "local_cache_hit", "external_kv_transfer"};

  int64_t computed = 0;
  int64_t local_cache_hit = 0;
  int64_t external_kv_transfer = 0;
  int64_t cached_tokens = 0;
  int64_t total = 0;

  // Upstream: update_from_output() (stats.py:328-335).
  void update_from_output(const PrefillStats& prefill_stats) {
    computed += prefill_stats.num_computed_tokens;
    cached_tokens += prefill_stats.num_cached_tokens;
    total += prefill_stats.num_prompt_tokens;
    local_cache_hit += prefill_stats.num_local_cached_tokens;
    external_kv_transfer += prefill_stats.num_external_cached_tokens;
  }

  // Upstream: get_by_source() (stats.py:337-346).
  int64_t get_by_source(std::string_view source) const {
    if (source == "local_compute") return computed;
    if (source == "local_cache_hit") return local_cache_hit;
    if (source == "external_kv_transfer") return external_kv_transfer;
    throw std::invalid_argument("Unknown source: " + std::string(source));
  }
};

// Ported from: IterationStats (vllm/v1/metrics/stats.py:349-371 @ 0.28.0), the
// frontend per-step aggregate consumed by PrometheusStatLogger.record. The
// always-on subset: token counters, preemptions, and the per-request TTFT/ITL/n
// samples plus finished-request breakdowns. A frontend builds one only for a
// step that has request outputs; a stats-only step records scheduler stats
// alone.
struct IterationStats {
  int64_t num_generation_tokens = 0;  // vllm:generation_tokens_total
  // vllm:prompt_tokens_total, _by_source and _cached. Its `computed` part, not
  // `total`, is the prompt share of vllm:iteration_tokens_total.
  PromptTokenStats prompt_token_stats;
  int64_t num_preempted_reqs = 0;     // vllm:num_preemptions_total
  // Per-request samples observed this step.
  std::vector<double> time_to_first_tokens_iter;     // seconds
  std::vector<double> inter_token_latencies_iter;    // seconds
  std::vector<int64_t> n_params_iter;                // request `n`
  std::vector<int64_t> max_num_generation_tokens_iter;
  std::vector<FinishedRequestStats> finished_requests;

  // Upstream @property num_prompt_tokens (stats.py:368-371).
  int64_t num_prompt_tokens() const { return prompt_token_stats.total; }
};

}  // namespace vllm::v1

#endif  // VLLM_V1_METRICS_STATS_H_
