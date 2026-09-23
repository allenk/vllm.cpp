ID: ISSUE-LOCAL-01M32475H9MZMMVVTCDP0EK7VT
Title: GSQ-RCO Qwen3.8-27B GGUF decodes degenerate blanks on TT keep-quant e2e; fails b10451 greedy parity
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-21
Updated: 2026-09-22
Closed: 2026-09-22

## Problem

The GSQ e2e gate (tenstorrent-gsq-keepquant.md gate 3) fails: on identical 65-token input (gsq_p0.i32, engine-tokenization verified identical to the oracle prompt), the engine produces [220,16]x16 on both GSQ-RCO files while the pinned llama.cpp b10451 oracle (oracle-dump --greedy 16, ctx=128) echoes prompt words. IQ3_XXS oracle: 23066 23066 1814 220 17 23066 220 16 220 16 23066 1814 220 17 1814 1814; IQ3_S oracle: 23066 220 16 220 16 23066 1814 220 17 1814 1814 23066 220 17 1814 220. Engine: [220,16]x16 both. 0/16 match. Prior session's gsq_iq3s.json (EXIT=0) was the same degenerate pattern - the gate was never token-green. Ruled out: prompt tokenization mismatch (engine ids == gsq_p0.i32), missing tensors (GGUF tensor names identical to APEX except the blk.64 MTP head APEX-only), env config (default env identical), metadata (only block_count 65->64, nextn_predict_layers 1->absent differ). APEX-I-Nano passes on the same build and prompts. Evidence: /tmp/gsq_iq3xxs_gate.log /tmp/gsq_iq3xxs_gate_tokens.json /tmp/gsq_iq3xxs_gate.dump /tmp/gsq_prompt_sharegpt.json. Next diagnostic: teacher-forced logits comparison at prompt positions 0..64 (oracle dump carries per-position f32 logits).

## Resolution

RESOLVED 2026-09-22, evidence 10: the reported e2e failure was measured against the pinned oracle's INCREMENTAL greedy, which disagrees with the oracle's own batch decode at the same causal positions (llama.cpp GDN-hybrid ubatch-boundary inconsistency at b10451). Against the batch denominator the engine is argmax-EXACT on both GSQ files: gap 0.0 mnats at all 16 generated positions (band 500). Gate 3/5 passes; no engine defect existed. The incremental-vs-batch inconsistency is a llama.cpp finding, recorded for optional upstream report.
