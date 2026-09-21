ID: ISSUE-LOCAL-01M2YCW65A2K35KF4P4QS9KCDY
Title: gguf dequant census case list misses IQ1_M (ggml type 29)
Row: LOAD-GGUF
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-20
Updated: 2026-09-20
Closed: -

## Problem

DequantGgufTensorToF32's hand-maintained ggml-type case list (src/vllm/model_executor/model_loader/gguf_dequant.cpp:107-129) lacks case 29 (IQ1_M), so every tensor carrying it falls to default and refuses the checkpoint with 'unsupported ggml type 29 (IQ1_M) (Task 2/i-quant)'. Both GSQ-RCO Qwen3.8-27B arms (IQ3_S and IQ3_XXS GGUF) carry IQ1_M tensors and refuse to load, although vt has decoded IQ1_M since its landing. This is the third instance of the case-list drift behind BlockDTypeFromGgmlTypeId that the switch's own comment predicts.

## Resolution

-
