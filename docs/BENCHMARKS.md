# Benchmarks

This is the public benchmark index. Each benchmark owns one detail file with
its workload, artifacts, commands, results, ratios, and limitations. Internal
measurement evidence remains in [the benchmark record](../.agents/benchmark-record.md).

| Benchmark ID | Subject | Disposition | Detail |
|---|---|---|---|
| `at-a-glance` | Current headline measurements and release evidence | Mixed | [Details](benchmarks/at-a-glance.md) |
| `vllm-online-serving` | Online serving compared with vLLM | Mixed | [Details](benchmarks/vllm-online-serving.md) |
| `memory` | Host and device memory measurements | Mixed | [Details](benchmarks/memory.md) |
| `llama-cpp-cpu` | CPU comparison with llama.cpp | Superseded | [Details](benchmarks/llama-cpp-cpu.md) |
| `cpu-q8_0-repack` | The q8_0 repack tier on three CPUs: x86-64, Arm with i8mm, Arm without | Measured | [Details](benchmarks/cpu-q8_0-repack.md) |
| `windows-rtx-pro-6000` | Windows x64 single-host matrix on a consumer Blackwell (sm_120) | Measured | [Details](benchmarks/windows-rtx-pro-6000.md) |
| `mlx-lm-apple-m4` | Apple M4 comparison with MLX-LM | Measured | [Details](benchmarks/mlx-lm-apple-m4.md) |
| `dwarfstar-gguf` | GGUF comparison with DwarfStar | Measured | [Details](benchmarks/dwarfstar-gguf.md) |
| `speculative-decoding` | Speculative decoding measurements | Mixed | [Details](benchmarks/speculative-decoding.md) |
| `qwen38-27b-exl3-gb10` | Qwen3.8-27B EXL3 3.5bpw with its DFlash2 draft, on GB10 | Measured | [Details](benchmarks/qwen38-27b-exl3-gb10.md) |
| `qwen38-27b-q4km-gfx1151` | Qwen3.8-27B Q4_K_M on Strix Halo: llama.cpp, vLLM and vllm.cpp, under a failing token gate | Mixed | [Details](benchmarks/qwen38-27b-q4km-gfx1151.md) |
| `qwen38-27b-exl3-variadic-gb10` | Qwen3.8-27B EXL3 3.5bpw under a mixed-length serving load, swept over concurrency, on GB10 | Measured | [Details](benchmarks/qwen38-27b-exl3-variadic-gb10.md) |
| `tt-capture-default-decode` | Tenstorrent decode rate, capture default vs opt-out, on the P150 | Measured | [Details](benchmarks/tt-capture-default-decode.md) |
| `tt-keepquant-27b-decode` | Tenstorrent keep-quant 27B Q4_K_M decode, first end-to-end completion on the P150 | Measured | [Details](benchmarks/tt-keepquant-27b-decode.md) |
| `how-we-measure` | Benchmark method and acceptance rules | Method | [Details](benchmarks/how-we-measure.md) |
| `variadic-load-methodology` | How the mixed-length, swept-concurrency serving benchmark works | Method | [Details](benchmarks/variadic-load-methodology.md) |
| [`vulkan-jetson-orin`](benchmarks/vulkan-jetson-orin.md) | Vulkan on an 8 GB Jetson Orin Nano against llama.cpp Vulkan: 2.41x on the reference axis, 1.65x on a decode-weighted one, and the backend's missing quantized compute tier |
| `open-gaps` | Pending, failed, void, and superseded measurements | Open | [Details](benchmarks/open-gaps.md) |
| `reproduce` | Reproduction commands and artifacts | Method | [Details](benchmarks/reproduce.md) |
