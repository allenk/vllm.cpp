// SERVE-METRICS (ROAD-V1-C8) scrape gate. The executable spec is vLLM's own
// scrape assertion tests/entrypoints/serve/instrumentator/test_metrics.py:
//   * test_metrics_exist: `assert metric in response.text` for every string in
//     EXPECTED_METRICS_V1 (a SUBSTRING check over the /metrics exposition).
//   * _get_expected_values: the counter `_total` / histogram `_sum`/`_count`
//     values a fixed workload must produce.
// So this test asserts our PrometheusStatLogger exposition CONTAINS every
// EXPECTED_METRICS_V1 name (RED-first: dropping any family fails), that the
// label schema is {model_name, engine}, that the histogram buckets match vLLM's
// 1-2-5 / timing schedules, and that record() folds SchedulerStats +
// IterationStats into the right counters/histograms.
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "vllm/v1/metrics/loggers.h"
#include "vllm/v1/metrics/prometheus.h"
#include "vllm/v1/metrics/stats.h"

using vllm::v1::FinishedRequestStats;
using vllm::v1::IterationStats;
using vllm::v1::PrefillStats;
using vllm::v1::SchedulerStats;
using vllm::v1::metrics::Build1_2_5Buckets;
using vllm::v1::metrics::PrometheusStatLogger;
using vllm::v1::metrics::PromRegistry;

namespace {

// vLLM EXPECTED_METRICS_V1 verbatim (test_metrics.py:182-228), minus the
// config-gated cache_config_info Info metric which we assert separately.
const std::vector<std::string>& ExpectedMetricsV1() {
  static const std::vector<std::string> v = {
      "vllm:num_requests_running",
      "vllm:num_requests_waiting",
      "vllm:num_requests_waiting_by_reason",
      "vllm:kv_cache_usage_perc",
      "vllm:prefix_cache_queries",
      "vllm:prefix_cache_hits",
      "vllm:num_preemptions_total",
      "vllm:prompt_tokens_total",
      "vllm:generation_tokens_total",
      "vllm:iteration_tokens_total",
      "vllm:cache_config_info",
      "vllm:request_success_total",
      "vllm:request_prompt_tokens_sum",
      "vllm:request_prompt_tokens_bucket",
      "vllm:request_prompt_tokens_count",
      "vllm:request_generation_tokens_sum",
      "vllm:request_generation_tokens_bucket",
      "vllm:request_generation_tokens_count",
      "vllm:request_params_n_sum",
      "vllm:request_params_n_bucket",
      "vllm:request_params_n_count",
      "vllm:request_params_max_tokens_sum",
      "vllm:request_params_max_tokens_bucket",
      "vllm:request_params_max_tokens_count",
      "vllm:time_to_first_token_seconds_sum",
      "vllm:time_to_first_token_seconds_bucket",
      "vllm:time_to_first_token_seconds_count",
      "vllm:inter_token_latency_seconds_sum",
      "vllm:inter_token_latency_seconds_bucket",
      "vllm:inter_token_latency_seconds_count",
      "vllm:e2e_request_latency_seconds_sum",
      "vllm:e2e_request_latency_seconds_bucket",
      "vllm:e2e_request_latency_seconds_count",
      "vllm:request_queue_time_seconds_sum",
      "vllm:request_queue_time_seconds_bucket",
      "vllm:request_queue_time_seconds_count",
      "vllm:request_inference_time_seconds_sum",
      "vllm:request_inference_time_seconds_bucket",
      "vllm:request_inference_time_seconds_count",
      "vllm:request_prefill_time_seconds_sum",
      "vllm:request_prefill_time_seconds_bucket",
      "vllm:request_prefill_time_seconds_count",
      "vllm:request_decode_time_seconds_sum",
      "vllm:request_decode_time_seconds_bucket",
      "vllm:request_decode_time_seconds_count",
  };
  return v;
}

bool Contains(const std::string& hay, const std::string& needle) {
  return hay.find(needle) != std::string::npos;
}

// Extract the sample value for `line_prefix` (a full sample name + label set up
// to the space) from the exposition. Returns the trailing number as a string.
std::string SampleValue(const std::string& text, const std::string& sample) {
  const size_t p = text.find(sample);
  if (p == std::string::npos) return "";
  const size_t nl = text.find('\n', p);
  return text.substr(p + sample.size(),
                     (nl == std::string::npos ? text.size() : nl) -
                         (p + sample.size()));
}

}  // namespace

TEST_CASE("build_1_2_5_buckets mirrors vLLM's docstring example") {
  // loggers.py:1302 >>> build_1_2_5_buckets(100) == [1,2,5,10,20,50,100].
  const std::vector<double> b = Build1_2_5Buckets(100);
  CHECK(b == std::vector<double>{1, 2, 5, 10, 20, 50, 100});
  CHECK(Build1_2_5Buckets(0).empty());
  CHECK(Build1_2_5Buckets(3) == std::vector<double>{1, 2});
}

TEST_CASE(
    "SERVE-METRICS: /metrics exposition contains every EXPECTED_METRICS_V1 "
    "name (RED-first) with the {model_name, engine} label schema") {
  PrometheusStatLogger logger("test-model", /*max_model_len=*/32768);
  logger.SetCacheConfigInfo(/*kv_cache_size_tokens=*/4096,
                            /*kv_cache_max_concurrency=*/1.0);
  const std::string text = logger.Expose();

  // Mirror vLLM test_metrics_exist: substring presence for every family.
  for (const std::string& metric : ExpectedMetricsV1()) {
    CHECK_MESSAGE(Contains(text, metric),
                  "missing metric in /metrics exposition: " << metric);
  }

  // Label schema: the always-on families carry model_name + engine.
  CHECK(Contains(text,
                 "vllm:num_requests_running{model_name=\"test-model\",engine="
                 "\"0\"}"));
  CHECK(Contains(text,
                 "vllm:prompt_tokens_total{model_name=\"test-model\",engine="
                 "\"0\"}"));
  // request_success adds the finished_reason label (loggers.py:719).
  CHECK(Contains(text, "finished_reason=\"stop\""));
  // waiting_by_reason adds the reason label.
  CHECK(Contains(text, "vllm:num_requests_waiting_by_reason{model_name="
                       "\"test-model\",engine=\"0\",reason=\"capacity\"}"));

  // TYPE lines: counter/gauge/histogram parity.
  CHECK(Contains(text, "# TYPE vllm:prompt_tokens counter"));
  CHECK(Contains(text, "# TYPE vllm:num_requests_running gauge"));
  CHECK(Contains(text,
                 "# TYPE vllm:time_to_first_token_seconds histogram"));

  // The two always-on families vLLM 0.28.0 feeds from PrefillStats
  // (loggers.py:680-693, 962-973), primed for each prompt-token source.
  CHECK(Contains(text, "# TYPE vllm:prompt_tokens_by_source counter"));
  for (const char* source :
       {"local_compute", "local_cache_hit", "external_kv_transfer"}) {
    CHECK(Contains(text,
                   std::string("vllm:prompt_tokens_by_source_total{model_name="
                               "\"test-model\",engine=\"0\",source=\"") +
                       source + "\"} 0.0"));
  }
  CHECK(Contains(text,
                 "# TYPE vllm:request_prefill_kv_computed_tokens histogram"));

  // Histogram bucket schedule parity (timing + 1-2-5 count buckets). Whole
  // bucket bounds render as "N.0" exactly like prometheus_client floatToGoString.
  CHECK(Contains(text, "le=\"0.001\""));    // TTFT first bucket
  CHECK(Contains(text, "le=\"2560.0\""));   // TTFT last finite bucket
  CHECK(Contains(text, "le=\"+Inf\""));     // synthetic overflow bucket
  CHECK(Contains(text, "le=\"20000.0\""));  // 1-2-5 bucket at max_model_len 32768

  // cache_config_info Info metric carries non-empty required labels
  // (test_metrics.py:299-301).
  CHECK(Contains(text, "kv_cache_size_tokens=\"4096\""));
  CHECK(Contains(text, "kv_cache_max_concurrency=\"1\""));

  // content type is the prometheus text 0.0.4 media type.
  CHECK(std::string(vllm::v1::metrics::kContentTypeLatest) ==
        "text/plain; version=0.0.4; charset=utf-8");
}

TEST_CASE(
    "SERVE-METRICS: record() folds SchedulerStats + IterationStats into the "
    "counters and histograms with the expected values") {
  PrometheusStatLogger logger("m", /*max_model_len=*/1024);

  SchedulerStats s;
  s.num_running_reqs = 3;
  s.num_waiting_reqs = 2;
  s.kv_cache_usage = 0.5;
  s.prefix_cache_stats.queries = 90;
  s.prefix_cache_stats.hits = 40;

  // A 10-token prompt whose first 4 tokens came from the prefix cache.
  IterationStats it;
  PrefillStats prefill;
  prefill.set(/*prompt_tokens=*/10, /*local_cached_tokens=*/4,
              /*external_cached_tokens=*/0);
  it.prompt_token_stats.update_from_output(prefill);
  it.num_generation_tokens = 5;
  it.num_preempted_reqs = 1;
  it.time_to_first_tokens_iter = {0.02};
  it.inter_token_latencies_iter = {0.03, 0.03};
  FinishedRequestStats f;
  f.finish_reason = "length";
  f.e2e_latency = 1.25;
  f.num_prompt_tokens = 10;
  f.num_generation_tokens = 5;
  f.max_tokens_param = 5;
  f.queued_time = 0.1;
  f.prefill_time = 0.2;
  f.inference_time = 0.9;
  f.decode_time = 0.7;
  f.mean_time_per_output_token = 0.14;
  f.num_cached_tokens = 4;
  it.finished_requests.push_back(f);

  logger.Record(s, &it);
  const std::string text = logger.Expose();

  // Gauges reflect the latest scheduler state.
  CHECK(Contains(
      text,
      "vllm:num_requests_running{model_name=\"m\",engine=\"0\"} 3.0"));
  CHECK(Contains(
      text,
      "vllm:num_requests_waiting{model_name=\"m\",engine=\"0\"} 2.0"));
  CHECK(Contains(
      text, "vllm:kv_cache_usage_perc{model_name=\"m\",engine=\"0\"} 0.5"));

  // Counters accumulate.
  CHECK(Contains(
      text, "vllm:prompt_tokens_total{model_name=\"m\",engine=\"0\"} 10.0"));
  CHECK(Contains(
      text,
      "vllm:generation_tokens_total{model_name=\"m\",engine=\"0\"} 5.0"));
  CHECK(Contains(
      text, "vllm:num_preemptions_total{model_name=\"m\",engine=\"0\"} 1.0"));
  CHECK(Contains(
      text,
      "vllm:prefix_cache_queries_total{model_name=\"m\",engine=\"0\"} 90.0"));
  CHECK(Contains(
      text,
      "vllm:prefix_cache_hits_total{model_name=\"m\",engine=\"0\"} 40.0"));

  // request_success labeled by the finished reason.
  CHECK(Contains(text,
                 "vllm:request_success_total{model_name=\"m\",engine=\"0\","
                 "finished_reason=\"length\"} 1.0"));

  // Histogram counts: one e2e/prefill/decode observation this step.
  CHECK(SampleValue(text,
                    "vllm:e2e_request_latency_seconds_count{model_name=\"m\","
                    "engine=\"0\"}") == " 1.0");
  CHECK(SampleValue(text,
                    "vllm:time_to_first_token_seconds_count{model_name=\"m\","
                    "engine=\"0\"}") == " 1.0");
  CHECK(SampleValue(text,
                    "vllm:inter_token_latency_seconds_count{model_name=\"m\","
                    "engine=\"0\"}") == " 2.0");
  // e2e sum is the observed latency.
  CHECK(SampleValue(text,
                    "vllm:e2e_request_latency_seconds_sum{model_name=\"m\","
                    "engine=\"0\"}") == " 1.25");

  // The prompt split (loggers.py:1194-1208 @ 0.28.0): all 10 prompt tokens are
  // counted, 4 of them as cached, and the iteration-token histogram observes
  // the 6 computed ones plus the 5 generated.
  CHECK(Contains(
      text,
      "vllm:prompt_tokens_cached_total{model_name=\"m\",engine=\"0\"} 4.0"));
  CHECK(Contains(text,
                 "vllm:prompt_tokens_by_source_total{model_name=\"m\",engine="
                 "\"0\",source=\"local_compute\"} 6.0"));
  CHECK(Contains(text,
                 "vllm:prompt_tokens_by_source_total{model_name=\"m\",engine="
                 "\"0\",source=\"local_cache_hit\"} 4.0"));
  CHECK(Contains(text,
                 "vllm:prompt_tokens_by_source_total{model_name=\"m\",engine="
                 "\"0\",source=\"external_kv_transfer\"} 0.0"));
  CHECK(SampleValue(text,
                    "vllm:iteration_tokens_total_sum{model_name=\"m\","
                    "engine=\"0\"}") == " 11.0");
  // The finished request computed 10 - 4 prompt tokens (loggers.py:1240-1246).
  CHECK(SampleValue(text,
                    "vllm:request_prefill_kv_computed_tokens_sum{model_name="
                    "\"m\",engine=\"0\"}") == " 6.0");
}

TEST_CASE(
    "SERVE-METRICS: record() without IterationStats folds the scheduler state "
    "only") {
  // A step with no request outputs (loggers.py:1185-1186 @ 0.28.0): its
  // prefix-cache lookups and gauges count, and no iteration family moves.
  PrometheusStatLogger logger("m", /*max_model_len=*/1024);
  SchedulerStats s;
  s.num_running_reqs = 1;
  s.prefix_cache_stats.requests = 1;
  s.prefix_cache_stats.queries = 40;
  s.prefix_cache_stats.hits = 32;

  logger.Record(s, nullptr);
  const std::string text = logger.Expose();

  CHECK(Contains(
      text, "vllm:num_requests_running{model_name=\"m\",engine=\"0\"} 1.0"));
  CHECK(Contains(
      text,
      "vllm:prefix_cache_queries_total{model_name=\"m\",engine=\"0\"} 40.0"));
  CHECK(Contains(
      text, "vllm:prefix_cache_hits_total{model_name=\"m\",engine=\"0\"} 32.0"));
  CHECK(Contains(
      text, "vllm:prompt_tokens_total{model_name=\"m\",engine=\"0\"} 0.0"));
  CHECK(SampleValue(text,
                    "vllm:iteration_tokens_total_count{model_name=\"m\","
                    "engine=\"0\"}") == " 0.0");
}

TEST_CASE("SERVE-METRICS: registry histogram cumulative buckets are monotone") {
  PromRegistry reg;
  reg.RegisterHistogram("h", "help", {"l"}, {1.0, 2.0, 5.0});
  reg.Observe("h", {"a"}, 0.5);  // <=1
  reg.Observe("h", {"a"}, 1.5);  // <=2
  reg.Observe("h", {"a"}, 3.0);  // <=5
  reg.Observe("h", {"a"}, 9.0);  // +Inf only
  const std::string text = reg.Expose();
  CHECK(Contains(text, "h_bucket{l=\"a\",le=\"1.0\"} 1.0"));
  CHECK(Contains(text, "h_bucket{l=\"a\",le=\"2.0\"} 2.0"));
  CHECK(Contains(text, "h_bucket{l=\"a\",le=\"5.0\"} 3.0"));
  CHECK(Contains(text, "h_bucket{l=\"a\",le=\"+Inf\"} 4.0"));
  CHECK(Contains(text, "h_count{l=\"a\"} 4.0"));
  CHECK(Contains(text, "h_sum{l=\"a\"} 14.0"));
}
