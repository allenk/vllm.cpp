# Mamba `align` mode: prefix caching for hybrid GDN models

Row `KV-MAMBA-ALIGN` (`.agents/engine-matrix.md`), feature-matrix row
"Mamba/GDN prefix-cache retention".

## Now

`ACTIVE`, phase 1 landed for Qwen3.5 / 3.6 behind explicit
`--enable-prefix-caching`. Phase 2 is named under Work breakdown.

## Scope

Prefix caching for a hybrid whose recurrent layers are GDN (Qwen3.5 / 3.6,
dense and MoE): the recurrent state is kept at block boundaries and a request
whose prompt starts with cached blocks resumes from the state of the last one.

Why it matters: vLLM turns prefix caching on for generative hybrids by default
(`vllm/config/model.py::is_prefix_caching_supported`) and, with it,
`mamba_cache_mode = "align"`. A translation workload with a shared system
prompt of about 1,100 tokens (1,139 tokens in, 39 out per request) hits on
91.6% of its prompt tokens in vLLM 0.28.0 and computes 0.63 M of 7.5 M prompt
tokens; this tree computed all 7.78 M and was 1.78 times slower end to end on
the same book.

Out of scope here: other hybrid families (refused, see Risks/decisions),
speculative decoding with align, partial-block hits (`prefix_match_unit`),
shared-prefix junctions, internal prefill checkpoints.

## Upstream chain

The deployed reference is vLLM 0.28.0; the local checkout is newer and is cited
where it is clearer.

| What | Upstream |
|---|---|
| align whenever prefix caching is on for a hybrid | `model_executor/models/config.py::MambaModelConfig.verify_and_update_config` |
| attention block raised to cover the mamba page | `platforms/interface.py::_align_hybrid_block_size` (`B = A * cdiv(mamba_page, A * attn_page_1tok)`, mamba page padded to `B * attn_page_1tok`) |
| block-aligned prefill chunks | `v1/core/sched/scheduler.py::_mamba_block_aligned_split` (0.28.0) |
| hit, allocate, free, cache in align mode | `v1/core/single_type_kv_cache_manager.py::MambaManager` |
| state index = block of `(seq_len - 1) // B` | `v1/attention/backends/utils.py::mamba_get_block_table_tensor` |
| copy the previous block's state into the new one before the forward | `v1/worker/mamba_utils.py::preprocess_mamba` |

Invariants carried over:

1. Block `p` of a request's GDN block table holds the state after exactly
   `(p + 1) * B` tokens. Non-final prefill chunks end on a multiple of `B`; a
   chunk that starts mid-block stops at the next boundary; no chunk runs past
   the last cacheable boundary.
2. The kernels read and write the state in place at block
   `bt[(seq_len - 1) // B]`, `seq_len = num_computed + num_scheduled`.
3. When that column moves, the previous column's state is copied into it
   before the forward. A fresh request starts at column -1 (zero state); a
   request with a `k`-block hit starts at column `k - 1`, `has_initial_state`
   true.
4. Hits are multiples of `B`, capped at `num_tokens - 1`; the full-attention
   hit is truncated to the GDN hit.

Worked numbers for Qwen3.6-35B-A3B: two KV heads of 256, bf16, give 2,048
bytes per token per attention layer; the GDN state is 49,152 bytes of bf16
conv (3 x 8,192) plus 2,097,152 bytes of fp32 recurrent state (32 x 128 x 128),
so `B = 16 * cdiv(2,146,304, 32,768) = 1056` (vLLM 0.28.0 logs "Setting
attention block size to 1056 tokens"). Qwen3.5-2B gets 544.

## Our baseline

Before phase 1: a hybrid defaulted prefix caching off, and forcing it on was
wrong rather than refused. The GDN spec was mode `"none"`, the runner gave each
request a recycled state slot keyed on its id, and a request admitted with
`num_computed_tokens = hit` resumed its GDN layers from zeros or from the slot's
previous occupant (on Qwen3.5-2B the hit requests continued the English source
instead of translating it). The scheduler-side MambaManager align bookkeeping
was already ported and matched upstream; no config path selected it and the
runner had no block-to-block state copy.

## Port map

| Upstream | Here |
|---|---|
| `MambaModelConfig` mode selection | `LoadedEngine::UsesMambaAlign` (model_loader.cpp), with `ModelInfo::supports_mamba_align` marking the hybrids that have the runner path |
| `_align_hybrid_block_size` | `LoadedEngine::ResolveEngineBlockSize` / `MambaAlignBlockSize`; the mode is set on the Mamba spec in `MakeKVCacheResolved` |
| `_mamba_block_aligned_split` | `Scheduler::mamba_block_aligned_split`, applied in the running and the waiting loop |
| `mamba_get_block_table_tensor` + `preprocess_mamba` | `GPUModelRunner::select_gdn_align_state_blocks` (replaces `remap_gdn_state_slots` in align mode) |
| mamba pages in the shared pool | `KVBytesPerBlock` counts align-mode GDN rows; `recurrent_state_bytes` skips them |

## Tests to port

- `tests/v1/core/test_mamba_align_chunk_split.py`: the aligned-split cases are
  covered by hand-worked cases in `tests/vllm/v1/test_scheduler.cpp`
  (`mamba_block_aligned_split: hand-worked cases at block 1056`, the sub-block
  re-alignment case) and a scheduler-level shared-prefix hit.
- `tests/v1/e2e/general/test_mamba_prefix_cache.py`: an end-to-end hit-vs-miss
  gate on real checkpoints (Gates, below), not yet a CI test.
- Upstream's MambaManager align tests: owed; the local manager cases still run
  mode `"none"`.

## Gates

- Unit (CPU): the aligned split against hand-worked cases; a scheduler-level
  hit at 1,056 tokens for a shared prefix; the mode selection and refusals; the
  block size on the shipped Qwen3.5 specs (1056, 1088 at alignment 64, 544);
  align rows counted per block.
- End to end, greedy, one request at a time on a fresh server: a request that
  hits the cached boundary produces the same tokens as the same request on a
  miss, and the same text as with caching off. Qwen3.5-2B and
  Qwen3.6-35B-A3B-FP8 both pass; the previous binary with caching forced on
  fails it (the hit requests continue the English source).
- Speed: a replica of the translation shape (1,172 prompt / 36 output tokens,
  c=8, temperature 0.2): 76.2 s without the cache, 44.6 s with it at a 90.1%
  hit rate, vLLM 0.28.0 in the same session 41.6 s. Then the owner's book.

## Dependencies

- `fix(metrics)` (`f90257d87`): prefix-cache lookups were dropped under async
  scheduling, so the hit counters this row is gated on read 0.
- Still owed for exact counters: steps with no output (a multi-chunk prefill
  alone in the batch) are not forwarded by EngineCore, so their lookups are not
  recorded; upstream sends stats-only outputs.

## Work breakdown

Phase 1 (landed): mode selection and refusals, block size, aligned split, state
row per block id with the column copy, KV byte accounting, tests above.

Phase 2 (named, not here):

1. Upstream's shared layout: attention plus three GDN groups of ten layers,
   tensors `shared_by` one layer of each, so a block id costs 10 pages
   (21.6 MB on the 35B) instead of its attention pages plus 30 GDN rows
   (86 MB).
2. Default on for hybrids, once the gates have run on the workload.
3. Speculative decoding with align (the accepted-token copy offset of
   `preprocess_mamba` / `postprocess_mamba`).
4. Partial hits, junctions, internal prefill checkpoints.

## Risks/decisions

- **D1, own state tensors in phase 1.** The GDN state stays in per-layer
  tensors with one row per block id, so each block id carries attention pages
  and GDN rows whichever group owns it: about 4x upstream's bytes per block on
  the 35B. Chosen because it reuses the runner's existing state views and keeps
  the change local; at `--kv-cache-memory 8GiB` the 35B gets 99 blocks of 1,056
  tokens, enough for the translation workload at c=8, not for many long
  contexts at once. Phase 2 item 1 removes it.
- **D2, opt-in.** Default prefix caching stays off for hybrids until the gates
  run on the owner's workload, because align mode changes the block size and
  the memory per block.
- **D3, refuse instead of degrade.** Prefix caching on a hybrid without an align
  path, or with speculative decoding, throws at load. Before, both admitted cache
  hits whose recurrent layers resumed from a state that was not theirs.
- **R1, previous column derived, not tracked.** Without speculation the column
  the state was last written to is `(num_computed - 1) // B`, so the runner
  derives it each step instead of keeping upstream's `mamba_state_idx` map; a
  request that leaves and re-enters the batch loses nothing. Speculation breaks
  this and is refused.
- **R2, coarse blocks.** A 1,056-token block holds every request's tail, so short
  requests pay a whole block of attention KV; same as upstream.
- **R3, budget below a block.** A prefill whose remaining step budget is below
  one block waits for a later step (upstream behaves the same).
