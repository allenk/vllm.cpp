ID: ISSUE-LOCAL-01M3265TQ26YSVKVA1HX6E28VC
Title: vllm-cli (C ABI path) and vllm-bench (AsyncLLM path) tokenize the same prompt differently for a bare .gguf model
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-21
Updated: 2026-09-21
Closed: 2026-09-21

## Problem

For /mnt/models/ISTA-DASLab-Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf, vllm-cli reports prompt_tokens=51 and vllm-bench reports 65 input tokens for the IDENTICAL prompt string (32 words, '1 2 world 1 hello ...'). The cli loads through the C ABI (vllm_engine_load, src/capi/vllm_c.cpp); the bench drives AsyncLLM directly. The two entry points resolve the tokenizer differently for a bare .gguf (no tokenizer.json sidecar), so any cli-based parity or benchmark number for GGUF models is suspect. Evidence: /tmp/gsq_iq3xxs_tt_cli.log vs /tmp/gsq_iq3xxs_gate.log (same string, 51 vs 65). Discovered 2026-09-21 during the GSQ e2e parity investigation (ISSUE-LOCAL-01M32475H9MZMMVVTCDP0EK7VT), where it invalidated the cli-based CPU tie-break leg.

## Resolution

FALSIFIED by the tree, 2026-09-21. The 51-vs-65 comparison compared two DIFFERENT strings, not two tokenizers: the cli leg was fed the 32-word string while the bench dataset contained the 41-word genprompt text. llama-tokenize at the pin gives 51 ids for the 32-word string (byte-identical to the engine cli's ids) and 65 ids for the 41-word text (byte-identical to gsq_p0.i32). The engine tokenizer, the bench pretokenizer, and the pinned llama.cpp all agree on both strings. No defect exists; the issue was an investigation artifact of feeding different prompts to the two entry points. Evidence: /tmp/gsq41_cpu.log and /tmp/gsq41_auto.log (prompt_tokens=65 on both devices for the 41-word prompt).
