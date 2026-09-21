ID: ISSUE-LOCAL-01M2YMVYBHPZ7VZW96BWKGBRG7
Title: TT: keep-quant arms for the GSQ-RCO Qwen3.8-27B census
Row: BACKEND-TENSTORRENT
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-20
Updated: 2026-09-20
Closed: -

## Problem

The two GSQ-RCO Qwen3.8-27B GGUFs do not run on the Tenstorrent keep-quant path: the IQ3_XXS artifact's census carries 198 block-quantized tensors whose dtypes the TT admission set does not name — IQ3_S (97, ffn/attn), IQ4_XS (33, ssm_out/attn), IQ2_XS (32, ffn), Q2_K (28, ffn/embd), IQ1_M (4) and IQ1_S (4, ffn tail) — each refusing or falling back to expand-bf16. The TT set today is {Q4_K,Q5_K,Q6_K,Q8_0} plus the int8-dot IQ set {IQ3_XXS,IQ2_XXS,IQ2_S,Q3_K} (waves 1-3 of QUANT-GGUF-IQ-TENSTORRENT). This widens the int8-dot arm with the six missing b10451 decodes so both GSQ files run keep-quant end-to-end, token-verified against the pinned llama.cpp greedy oracle. Spec: .agents/specs/tenstorrent-gsq-keepquant.md

## Resolution

-
