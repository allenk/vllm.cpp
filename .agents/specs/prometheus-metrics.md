# Prometheus `/metrics` exposition (`SERVE-METRICS`, ROAD-V1-C8)

Live spec for the `SERVE-METRICS` engine-matrix row: the Prometheus text
exposition served at `GET /metrics`, mirroring vLLM 0.26.0.dev0
(`555967922`). The oldest open T0 serving debt — as of 2026-07-27 the endpoint
serves LIVE per-step values (W4 below), not just the primed catalog.

## Scope

Wire a self-contained Prometheus registry + text-format-0.0.4 exposition and the
always-on vLLM metric catalog (`PrometheusStatLogger`), and expose it at
`GET /metrics` behind an opt-in backing on the OpenAI `ApiServer`. The gate is
metric-NAME + label-schema parity with vLLM's own scrape spec. Config-gated
metric families (kv-connector, mm-cache, LoRA, kv-block-lifetime, corrupted,
engine_sleep_state) are OUT of the always-on core and stay INVENTORIED — vLLM
itself only registers them under the matching config, so their absence is
faithful, not a gap. (prompt_tokens_by_source was listed here; in vLLM 0.28.0 it
is always on, and W7 below registers it. Spec-decoding landed with its config.)

## Upstream chain

- `vllm/entrypoints/serve/instrumentator/metrics.py:82` — `make_asgi_app(registry)`
  mounts `/metrics`; `PrometheusResponse` sets `text/plain; version=0.0.4;
  charset=utf-8` (`:52-60`).
- `vllm/v1/metrics/loggers.py:480-1060` — `PrometheusStatLogger` registers every
  metric: names, documentation, type (Counter/Gauge/Histogram/Info), histogram
  buckets, and the `["model_name","engine"]` label schema; `build_buckets`
  /`build_1_2_5_buckets` (`:1284-1305`) build the count buckets.
- `vllm/v1/metrics/loggers.py:1100-1257` — `record(SchedulerStats,
  IterationStats)` folds one engine step into the metrics.
- `vllm/v1/metrics/stats.py:186-259` — `SchedulerStats`, `IterationStats`,
  `FinishedRequestStats`.
- The `prometheus_client` package provides the text format: counters export a
  `_total` sample, histograms export `_bucket{le=...}`/`_sum`/`_count`, Info
  exports `{labels} 1.0`. This project has no Python at runtime, so the format
  bytes are reimplemented.

## Our baseline

Before this row: only `include/vllm/v1/metrics/stats.h` existed (the prefix-cache
`PrefixCacheStats`/`CachingMetrics` from `KV-PREFIX-CACHE`); NO Prometheus
registry, NO metric catalog, and `GET /metrics` was ABSENT from
`src/vllm/entrypoints/openai/api_server.cpp` (only completions/chat/models/health/
version were registered).

## Port map

- `include/vllm/v1/metrics/prometheus.h` + `src/vllm/v1/metrics/prometheus.cpp` —
  `PromRegistry`: Counter/Gauge/Histogram/Info families with multi-label series
  and the text-0.0.4 `Expose()` formatter (`kContentTypeLatest`).
- `include/vllm/v1/metrics/stats.h` — added `SchedulerStats`, `IterationStats`,
  `FinishedRequestStats` (the always-on subset of stats.py).
- `include/vllm/v1/metrics/loggers.h` + `src/vllm/v1/metrics/loggers.cpp` —
  `PrometheusStatLogger`: registers the always-on catalog 1:1 (names/help/type/
  buckets/labels), `Build1_2_5Buckets`, `Record()`, `SetCacheConfigInfo()`,
  `Expose()`.
- `src/vllm/entrypoints/openai/api_server.cpp` + `.h` — `handle_metrics()` and the
  opt-in `set_metrics_logger()` + conditional `GET /metrics` route.

## Tests to port

Upstream `tests/entrypoints/serve/instrumentator/test_metrics.py` — its
`EXPECTED_METRICS_V1` list + `test_metrics_exist` (`assert metric in
response.text`, a substring check over the exposition) is the executable spec.
Re-expressed as `tests/vllm/v1/test_prometheus_metrics.cpp`: substring presence
for every `EXPECTED_METRICS_V1` name (RED-first), label-schema `{model_name,
engine}`, TYPE lines, histogram bucket schedules, `build_1_2_5_buckets`
docstring example, `record()` value folding, and cumulative-bucket monotonicity.
Endpoint-level cases in `tests/vllm/entrypoints/openai/test_api_server.cpp`.

## Gates

- Parity: the `/metrics` exposition CONTAINS every `EXPECTED_METRICS_V1` string
  with the vLLM label schema and bucket bounds (CPU, deterministic). RED-first:
  dropping any family fails the substring assertion.
- Inertness: opt-in — a server without `set_metrics_logger()` does not register
  `/metrics` and is byte-identical; the 22 pre-existing api_server cases stay
  green. Clean CPU `-Werror`.

## Dependencies

None new. Reuses `PrefixCacheStats` (already in stats.h). The scheduler exposes
`prefix_cache_metrics()` and running/waiting counts; the live per-step stats are
now fed into `Record()` at the sync `LLMEngine` step site (W4, done) via
`Scheduler::make_stats()` + the `OutputProcessor`-built `IterationStats`. Wiring
the same into the AsyncLLM production-serving loop is the remaining follow-on.

## Work breakdown

- W1: `PromRegistry` + text exposition + unit test — DONE.
- W2: `PrometheusStatLogger` always-on catalog + `Record()` + cache_config_info —
  DONE.
- W3: `ApiServer` `/metrics` route (opt-in) + endpoint test — DONE.
- W4: fold the live EngineCore/Scheduler per-step SchedulerStats +
  IterationStats into `Record()` — **DONE 2026-07-27 (`CLAIM-ROADMAP-C8-METRICS-WIRE`).**
  `EngineCoreOutputs` carries `scheduler_stats` (new `Scheduler::make_stats()`,
  `scheduler.py:2399-2436` — running/waiting/kv-usage + the per-step prefix-cache
  delta STASHED by `schedule()`, no second take-and-swap) + a stamped monotonic
  `timestamp`; `OutputProcessor::process_outputs` builds `IterationStats` (token
  counts, TTFT/ITL samples, finished-request breakdowns off new `RequestState`
  timing, `stats.py:377-475`); the sync `LLMEngine::step()` folds both into the
  attached logger's `Record()` guarded by outputs>0 (`llm_engine.py:308-329`).
  Opt-in `set_stat_logger` (null default) ⇒ byte-identical no-stats path.
  Behavioural CPU gate `test_llm_engine.cpp` case 6 (44 asserts, RED-first: 14
  flip 0→correct when `Record` is disabled).
- W5: the per-request queue/prefill/inference timing + preemption counter via
  EngineCoreEvents — **LANDED + CPU-GATED 2026-07-27 (`CLAIM-ROADMAP-C8-RESPONSE-METRICS`,
  `SERVE-RESPONSE-METRICS` → `ACTIVE`).** The scheduler records
  QUEUED/SCHEDULED/PREEMPTED events, the `OutputProcessor` folds them into the
  timing intervals + `num_preemptions`, so `vllm:request_{queue,prefill,inference}
  _time_seconds` + `vllm:num_preemptions_total` now carry real durations (they
  were observed at 0 before). See
  [per-request-response-metrics.md](per-request-response-metrics.md).
- W6 (residual): the AsyncLLM production-serving metric wiring; the config-gated
  families (spec-decode/kv-connector/mm/LoRA) as their configs land; the
  chat/completion RESPONSE-BODY per-request timing surface — OPEN.
- W7: the prompt-token split and stats-only steps, to vLLM 0.28.0 — **DONE
  2026-10-20.** Found by comparing a translation workload's scrape against vLLM
  0.28.0 with prefix caching on: `vllm:prompt_tokens_cached` read 0 while 92% of
  the prompt tokens hit, `vllm:iteration_tokens_total` counted the hits as
  computed, and about 430 requests' prefix-cache lookups were missing.
  - `PrefillStats` (`stats.py:259-299`) is set on `Request` in the waiting loop
    until the first preemption (`scheduler.py:898-905`) and taken onto the
    request's first `EngineCoreOutput` (`:1962-1969`). The output processor
    folds it into `IterationStats.prompt_token_stats` (`PromptTokenStats`,
    `stats.py:301-346`) and the request's `num_cached_tokens`.
  - `Record()` takes `const IterationStats*`; null is a step without request
    outputs, which records the scheduler state only (`loggers.py:1185-1186`).
    `vllm:prompt_tokens` counts `total`, `vllm:prompt_tokens_cached` counts
    `cached_tokens`, `vllm:iteration_tokens_total` observes `computed +
    num_generation_tokens`, and the always-on `vllm:prompt_tokens_by_source`
    and `vllm:request_prefill_kv_computed_tokens` are registered.
  - `EngineCore` returns a stats-only `EngineCoreOutputs` for a step without
    request outputs while `log_stats` is on (`scheduler.py:2121-2134`); AsyncLLM
    and LLMEngine record it. Under async scheduling a `schedule()`'s lookups are
    published by the update of the batch before it, which has no outputs when it
    was all partial prefills or the flush after a finish.
  - Deferred: `PrefillStats.finalize` / `num_cache_creation_tokens` and
    `RequestOutput.num_cached_tokens`, whose consumers are not ported.

## Risks/decisions

- Decision: mirror `prometheus_client` byte-for-byte for the format — counters
  emit `_total`, histograms emit cumulative `_bucket`/`_sum`/`_count`, whole
  numbers render as `N.0` (Go `floatToGoString`). We deliberately do NOT emit the
  optional `_created` series (prometheus_client can disable them; vLLM's scrape
  spec never asserts them).
- Decision: register only the ALWAYS-ON families; config-gated ones stay out
  until their config exists, matching vLLM exactly (honest, not a shortcut).
- Risk: live-engine value wiring (W4) must not perturb the hot path — it is an
  opt-in read of already-computed stats, so inert by construction.
