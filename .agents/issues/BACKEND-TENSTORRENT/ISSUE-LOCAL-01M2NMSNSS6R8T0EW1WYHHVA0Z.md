ID: ISSUE-LOCAL-01M2NMSNSS6R8T0EW1WYHHVA0Z
Title: APEX 27B e2e: greedy token-correctness gate vs the llama.cpp oracle
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-16
Updated: 2026-09-16
Closed: -

## Problem

The APEX-I-Nano 27B now completes the e2e bench (PR #3212: exit 0, 64/64 tokens) but nothing pins that the generated tokens are CORRECT. Before any bench record or the APEX e2e owed closes, the greedy decode needs a gate against the pinned llama.cpp b10451 oracle (the registry's k-quant floor oracle) on the identical workload: same GGUF artifact, same generated-dataset prompts (BuildPrompt, seeds 0 and 1, input_len 128), greedy temp 0, 32 output tokens, no chat template. Deliverable: recorded token/text comparison, disposition, and the recipe.

## Resolution

Evidence 2026-09-16: gate EXECUTED, verdict FAIL (the defect it exists to
catch is real; fix owned by ISSUE-LOCAL-01M2NSDATJQ1YNW1PA9ZBMAAM5).

- Workload: identical both legs — artifact sha256
  `47b627b7de17c2bcfa9cebb7cf2d9d81e68b4b61f8cad16cf85f839604c694cb`,
  BuildPrompt seeds 0/1 (351-byte filler prompts, byte-identical files
  /tmp/apex_prompt{0,1}.txt), greedy temp 0, 32 tokens, no chat template.
- Oracle leg: pinned llama.cpp b10451 `llama-server` /completion with
  `return_tokens` (/tmp/oracle_tokens.json) — real token ids, both prompts.
- Our leg: `vllm-bench --output-token-ids` (/tmp/apex_ids.json) — **all
  zeros**, 2x32. `VT_DEBUG_SAMPLED=1` confirms the sampler itself emits
  tok=0 at every step (also on a 1x16 / 4-token run, /tmp/apex_debug2.log).
- Attribution: the logits slot IS committed `tracked=1` at `[1,248320]`
  and `EnsureHost` downloads from it (TT-SLOT lines 3020/4481/5928/7375),
  so the zeros are IN the committed device tensor: the E=1 chunked
  keep-quant arm produces garbage at the lm_head decode shapes. The 27B
  e2e completions were structurally complete but token content garbage —
  the gate did its job before any bench record.
