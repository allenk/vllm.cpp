# Reproduce

| Benchmark | Entry point |
|---|---|
| vLLM online grid | `.agents/specs/competitive-benchmarks.md`, evidence under `dgx:~/work/vllm.cpp-online-gate/evidence/` |
| Clock-controlled pin series | `$HOME/gpu.lock` FIRST, then `sudo -n nvidia-smi -lgc 2190` under an always-fires `-rgc` trap; oracle by EXPLICIT PATH, identity asserted per leg; a `gpu_clock_state.py` window per leg |
| CPU vs llama.cpp | Same GGUF both arms, 3 reps under one `flock $HOME/gpu.lock`; `VT_GGUF_KEEP_F16=0` reproduces the pre-L7 baseline |
| Laguna NVFP4 decode | `flock $HOME/gpu.lock ./build-cuda/examples/laguna-gen --model ~/laguna-xs-nvfp4 --gpu` (that directory holds the S-2.1 checkpoint); `drop_caches` first, create the CUDA context before loading weights |
| DeepSeek-V4-Flash decode | `deepseek-v4-gen --gpu --kv-cache` on `ds4flash.gguf`, captured under tmux |
| Metal vs MLX-LM | Paired A/B harness, interleaved runs, cold legs discarded |
| Vulkan vs llama.cpp Vulkan | Same GGUF both arms: ours `-DVLLM_CPP_VULKAN=ON`, llama.cpp `-DGGML_VULKAN=ON` at `237ad9b96`, SUPERSEDED, via `llama-bench`; clean legs only, one `flock $HOME/gpu.lock`. GEMV sweep: `benchmarks/vulkan_gemv_ab.cpp` |
| Qwen3.8-27B EXL3 on GB10 | Step-by-step in [`qwen38-27b-exl3-gb10.md`](qwen38-27b-exl3-gb10.md#reproduce-this-run): both checkpoints pinned by revision and sha256, the 164-problem HumanEval set, and both arms on one binary. `VT_DFLASH_PAGED=0` is required ([#2274](https://github.com/mudler/vllm.cpp/issues/2274)) |
| Variadic serving load, either engine | `benchmarks/variadic/`, method in [`variadic-load-methodology.md`](variadic-load-methodology.md#running-it): `build_corpus.py` from three sha256-pinned corpora, `client.py` closed-loop at a chosen concurrency, `report.py` for the tables, `job.sh` for the whole sweep inside an `rc` lease. The report recomputes every number from the per-request records, so a corrected statistic needs no GPU |
| Which llama.cpp a figure ran | Three revisions on this page, all SUPERSEDED (#1003): fork `237ad9b96` (GB10 CPU, Vulkan, x86, kernel matrix), stock `b9892` (Pi 5), stock `7044859` (Muse Glimmer, #391). Pin is stock `b10451`, unbuilt (#857) |
| Android q8_0 repack, and the thread count a phone wants | `benchmarks/android/android_sweep_ondevice.sh` then `android_repack_ab_ondevice.sh`, both driven ON THE DEVICE (`adb forward` registers a mapping without creating a host listener; device-side curl answers at once, and it keeps the USB round trip out of the timing). Each refuses to start if any server exists or the port is held, requires exactly one server per leg, reads back that the kill engaged, and traps so an interrupt leaves no orphan -- an orphan both steals CPU and answers the readiness probe with none of the script's env. State the OUTPUT LENGTH: tok/s here divides output tokens by total wall, so the same configuration reads 6.94 / 11.55 / 12.54 at 32 / 128 / 256 output tokens |
| Jetson Orin Vulkan, reference axis | Both arms bf16 from one checkpoint -- ours the published safetensors (the GGUF arch allowlist has no `qwen3` dense), llama.cpp the BF16 GGUF. 1,024 in / 128 out, TOTAL tok/s, c=1, six legs each interleaved, unique prompt per request. Prompt length CALIBRATED from llama.cpp's own report (1,210) and used for both arms. Discard any leg whose window contains a first-run shader compile -- the driver cache write is timestamped and visible |
| Revisions repo-wide | **Five**, not three, enumerated in the [spec](../../.agents/specs/oracle-llamacpp-repin-stock.md). Absent here: stock `030ebb5` (NON-BINDING) and a Poolside fork BRANCH with no commit recorded, behind Laguna's `27.8 tok/s` |

Build flags, environment variables, and the full gate list are in
[BUILD.md](../BUILD.md) and [ENVIRONMENT.md](../ENVIRONMENT.md).
