# Spec: APEX 27B greedy token-correctness gate vs the llama.cpp oracle

Row: `BACKEND-TENSTORRENT`
Issue: `ISSUE-LOCAL-01M2NMSNSS6R8T0EW1WYHHVA0Z`
State: ACTIVE (2026-09-16)
Git integration: one PR for spec and evidence (developer preference,
recorded for this row). The work is records + evidence; product code changes
only if the comparison needs a dump hook (named here first if so).
Base: `origin/main` (after #3212: the e2e completes without env knobs).

## Scope

The APEX-I-Nano 27B e2e completes (exit 0, 64/64 tokens) but nothing pins
the tokens CORRECT. This spec defines the greedy correctness gate against
the pinned `llama.cpp` b10451 oracle (`~/.agents/oracles/llama-cpp.md`,
registry id `llama-cpp`): the k-quant floor oracle for GGUF arms, built at
`~/Sources/llama.cpp/build/bin/llama-cli` (verified `build 10451,
commit 10bf611e5`, aarch64 CPU).

Workload identity (the whole gate rests on this):

- Artifact: `/mnt/models/mudler-qwen3.8-27B-APEX-gguf/Qwen3.8-27B-APEX-I-Nano.gguf`
  (10.7 GB), same file both sides, byte-identical (sha256 recorded in the
  evidence).
- Prompts: the bench generated-dataset builder (`examples/bench/bench_core.h:541`,
  `detail::BuildPrompt`) — words {hello, world, 1, 2}, `std::mt19937_64`
  seeded `seed + i` (i = 0, 1 for `--num-prompts 2`), appended 4 words per
  step until the tokenizer counts ≥ 128 tokens. The oracle side replicates
  the loop with the llama.cpp tokenizer (same GGUF vocab, so the same
  counts); the replication is verified by matching the bench's reported
  prompt token counts.
- Decoding: greedy (temp 0), 32 new tokens per prompt, no chat template on
  either side (`--skip-chat-template` on ours; completion mode on the
  oracle's).
- Comparison: token IDs where both sides can dump them; text equality where
  the oracle only prints text. Any mismatch names the first divergent
  position and both sides' ids.

Out of scope: throughput (axis #1003), captured decode, MTP/speculative.

## Design

1. Replicate the two prompts: a small Python driver seeds `random.Random`
   — no, `std::mt19937_64` — via a C++ one-off compiled against the bench
   header? NO: the honest replication uses OUR tokenizer. The bench prints
   the tokenized prompt counts; the driver iterates the word loop against
   `llama-tokenize` (same vocab) until the count matches. Cross-check: the
   bench's own reported counts must equal the driver's.
2. Oracle leg: `llama-cli -m <artifact> -f <prompt> -n 32 --temp 0 --seed 0
   -st` per prompt (single-turn completion, template suppressed), tokens
   captured.
3. Our leg: the bench recipe from #3212's gate plus
   `--output-token-ids /tmp/apex_ids.json`.
4. Compare; record in the issue and, on pass, this spec's `## Outcome`.
5. Disposition rules: exact match → gate PASS. Mismatch inside the
   documented TT quant-decode bit-exactness envelope (there is none — the
   keep-quant decode is pinned bit-exact vs `BlockToFloat`) → any mismatch
   is a finding, not a tolerance. The known admissible divergence: the
   oracle's own sampling-stack differences (softmax order) can flip a
   near-tie greedy token; if that happens, name it, measure the logit gap,
   and record the near-tie disposition per the repo's near-tie precedent.

## Tests

- The gate IS the test. Red-first proxy: the e2e previously could not
  generate at all (the OOM); there is no prior green to mutate.
- Evidence lines: sha256 of the artifact, both sides' prompt token counts,
  the 2×32 token ids (or texts), the comparison verdict, host and build
  info for both legs.

## Risks

- The oracle's qwen3.5 thinking template auto-activates in conversation
  mode; the gate must suppress it (`-no-cnv` / `-st` with a raw prompt) or
  the workloads differ. If `-no-cnv` cannot suppress it in b10451, fall
  back to feeding the RAW prompt through the completion path the oracle
  itself documents and record the harness adaptation.
- Our tokenizer and the oracle's may disagree on token COUNT near the
  128 boundary (the loop is count-driven); the driver verifies counts and
  the prompt strings are recorded, so any drift is visible.

## Owed

- The published bench record (only after this gate's disposition).
- The throughput axis (#1003).

## Stop conditions

- If the oracle cannot produce greedy output for the artifact at all, the
  gate is PENDING a named external authority, and the issue records that.
- If tokens diverge beyond a near-tie, stop and open the divergence as its
  own bug — do not widen tolerance.
