ID: ISSUE-LOCAL-01M2XSWCP507W0M18EXHGR5TGS
Title: TT prefill M>1: RmsNormGated gate reshape fatals MeshBuffer, SigmoidGateBf16 then OOMs
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-19
Updated: 2026-09-20
Closed: 2026-09-20

## Problem

At concurrency>=2 on the Qwen3.8-27B-APEX TT vehicle, the PREFILL step still throws 'MeshBuffer must be large enough to hold the tensor' (assert.hpp:104) inside RmsNormGatedKernel's act reshape ([256,6144] TILE -> {12288,128}, equal numel AND equal tile count, yet tt::tt_metal::view refuses). The fatal is silently caught upstream; the degraded step then reaches SigmoidGateBf16 whose ttnn::multiply allocates a 36 GiB output (inputs carry a poisoned [12288,786432]-class spec) and the engine dies (bank_manager.cpp:495). Repro: vllm-bench --model Qwen3.8-27B-APEX-I-Nano.gguf --num-prompts 2 --input-len 128 --output-len 32 --concurrency 2 --temperature 0 on P150, VT_TT_AFFINE_F32=1 VT_TT_NORM_PAD=1 VT_TT_PROGRAM_CACHE=1. Evidence: /tmp/m2_rng2.log (VT_TT_RNG_TRACE stage prints bracket the fatal inside CaptureSafeReshape -> ttnn::reshape), /tmp/m2_bench_fixed.log (OOM backtrace through BinaryNgDeviceOperation).

## Resolution

2026-09-20: the fatal does not exist on the tenstorrent-grouped-m2 tip. A/B on the P150 with the issue's exact repro (Qwen3.8-27B-APEX-I-Nano, --num-prompts 2 --input-len 128 --concurrency 2 --temperature 0, VT_TT_AFFINE_F32=1 VT_TT_NORM_PAD=1 VT_TT_PROGRAM_CACHE=1): base main@1f46d44e1 enters the MeshBuffer TT_FATAL retry loop (/tmp/m2p_base_red32.log, 247+ fatals); the tip (1fec06683, PR #3230 applied) completes the same run clean three times — output-len 1, 8, and 32 — 0 fatals, 16/16/64 tokens (/tmp/m2p_bench_red.log, /tmp/m2p_bench_red8.log, /tmp/m2p_bench_red32.log). The filed evidence (/tmp/m2_rng2.log 22:48, /tmp/m2_bench_fixed.log 21:30) predates the #3230 commits (07b52bde8 23:44, 1fec06683 23:46) and came from the investigation's WIP debug build (TT-RNG2 prints), so the RmsNormGated gate-reshape attribution rode that degraded state; the clean-build defect is the same MeshBuffer fatal family 07b52bde8 fixed in the batched eager view paths. The [256,6144] committed gate shadow -> [12288,128] reshape itself is a legal tt::tt_metal metadata view (reshape_view this_is_view: equal last-dim tile divisibility) and the new focused test pins it: kRmsNormGated serves a [256,6144] committed gate shadow at the batched-prefill [12288,128] geometry (tests/vt/test_tenstorrent_backend.cpp, CPU-oracle diff within the 0.035 envelope). Full TT suite green with the test.
