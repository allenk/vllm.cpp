# Windows, RTX PRO 6000 Blackwell (sm_120)

A single-host matrix on the platform this port was built for: Windows x64 on a
consumer-class Blackwell (compute capability 12.0), which is neither of the two
devices upstream measures on. Every cell here was taken on the same box, in one
session, under one lock.

## Scope, and what it is not

| Axis | This page |
|---|---|
| Host | Windows 11, Ryzen 9950X3D (16 cores / 32 threads), 191 GiB RAM, RTX PRO 6000 Blackwell Workstation (97,887 MiB, sm_120) |
| Workload | 1024-token prompt, 128 output tokens, greedy, concurrency 1/2/4/8/16/32 |
| Metric | Both output tok/s and TOTAL tok/s are recorded. TOTAL counts prefill, so it is not comparable to any figure scored on output alone |
| Not measured | Energy, cold-start, long-context, batch sizes above 32 |

**Deviations from `how-we-measure.md`, stated rather than silently taken.** That
protocol asks for three interleaved repetitions per point. Here, concurrency 1
through 8 run ONE repetition and 16 and 32 run three, and the repetitions are
sequential within one point rather than interleaved across legs.

**The spread column is not the variance that matters, and this page nearly said
it was.** Every c=1 to c=8 point reports a 0.0% spread, which was the original
justification for one repetition. It is the wrong number for that argument: it
measures agreement between repetitions inside ONE warm server process, so it says
the sampler is stable and nothing about whether the number reproduces.

Measured properly, by restarting the server three times and taking c=1 on each:

| | instance 1 | 2 | 3 | band |
|---|---:|---:|---:|---:|
| ours | 6.44 | 6.69 | 6.16 | 8.6% |
| llama.cpp | 27.88 | 31.53 | 31.70 | 13.7% |

All six instances reported 0.0% internally. **A first draft of this page called
the between-instance spread 28%, from two samples taken hours apart. Three clean
back-to-back restarts say 9 to 14%,** and the 28% was mostly one low outlier: the
`C1` leg below, taken inside the full matrix run rather than standalone, read 5.09,
which is 19% under the lowest of the three restarts while the standalone thread
sweep read 6.51, inside the band. Why a leg taken inside the matrix lands low is
not established. The tables below carry the matrix figure, because that is the run
they come from, and this band beside it.

Leg order is ours-then-reference in every table, so a slow thermal drift lands on
the reference leg, which is the direction that flatters the reference rather than
us.

## Thread count is measured per platform, and it does not transfer

Both platforms swept on the same short axis (128-in / 32-out, c=1), timing taken
from the harness alone. Output tok/s, Qwen3-0.6B, public tree.

| Threads | WSL2 | Windows |
|---:|---:|---:|
| 1 | 19.35 | 5.55 |
| 2 | 29.20 | 8.05 |
| 4 | 32.37 | 13.39 |
| 8 | **32.90** | 17.99 |
| 16 | 29.76 | **19.24** |
| 32 | 0.70 | 18.01 |

**The optima differ and so do the failure shapes.** WSL2 peaks at 8 and falls off
a cliff at 32 — a 47x collapse, not a decline. Windows peaks at 16 and loses only
6.4% at 32. Nothing about "32 threads is bad" is a property of the engine; it is
a property of one platform, and an earlier version of this page stated it as the
former on Linux-only evidence.

**n=16 is used for every CPU row on this page.** It is Windows' optimum, and on
WSL2 8 and 16 are inseparable on the handwritten path (24.71 against 24.48) while
16 is clearly better on the Triton path (25.76 against 21.49). It is the only
value that costs nothing on either platform or either path.

The knob is proven live by the sweep varying rather than by a `cpu/wall` column:
if `VLLM_CPP_CPU_THREADS` did nothing, all six points would read the same, and a
47x cliff falsifies that loudly. That substitution matters because Windows has no
cheap way to sample process CPU time — every candidate starts a process, and on
this host `powershell.exe -Command 1` costs 0.28 to 1.19 s, `tasklist.exe` 0.13 to
6.10 s, and a purpose-written 40-line native probe 0.098 to 2.03 s. The spread is
worse than the mean: a fixed cost could be subtracted, 46x of jitter cannot, and
it lands hardest on the short windows where the fast settings live.

An earlier sweep on this page did put `powershell.exe` inside the timed region.
It reported `cpu/wall = 0.46` at one thread — a process using half a core while
single-threaded, which is impossible — and that reading was the instrument
failing, not the platform being slow.

## CPU

**Baseline platform: WSL2**, because it is the only one that can run the whole
four-way comparison. The Triton-CPU provider's kernel loader is compiled out
under `_WIN32`, so a Windows Triton leg would register the provider, decline
every op, and produce a row byte-identical to the handwritten one. Windows is
kept below as a platform control rather than as the main table.

Qwen3-0.6B, 16 threads, 1024-in / 128-out, output tok/s. Ours reads the
safetensors checkpoint, llama.cpp the BF16 GGUF of the same weights; the mismatch
is forced, not chosen, because our GGUF loader rejects the plain `qwen3`
architecture outright. The two engines support disjoint GGUF sets at this size.

| Leg | c=1 | c=2 | c=4 | c=8 | c=16 † | c=32 |
|---|---:|---:|---:|---:|---:|---:|
| ours, lab tree, handwritten | 2.85 | 4.35 | 6.06 | 7.59 | 8.54 | 8.41 |
| ours, lab tree, **Triton** | 17.63 | 21.04 | 25.26 | 29.07 | 31.21 | 32.50 |
| ours, public tree, handwritten | 9.64 | 15.05 | 21.14 | 28.16 | 31.57 | 33.90 |
| ours, public tree, **Triton** | 17.92 | 21.52 | 24.70 | 29.64 | 32.13 | 33.70 |
| llama.cpp | **26.51** | **37.22** | **53.97** | **86.19** | 86.12 | 45.36 |

† Every leg is unstable at c=16, spreads of 66% to 308%. Four engine
configurations going unstable at the same point is a property of the host, not of
any one of them, and that row should not be quoted on its own.

**The Triton-CPU provider is worth 6.2x on the tree it was written against**
(2.85 to 17.63 at c=1). Same tree, same binary, one environment variable: the
kernels are dlopen'd at first dispatch, and the leg is only counted when the log
shows them loaded and running — eight loaded, seven serving ops.

**Upstream's three months of handwritten work is worth 3.4x** (lab 2.85 to public
9.64 at c=1) **and has not caught the provider at low concurrency**: 9.64 against
17.92, still 1.9x behind. It draws level at c=8 (28.16 against 29.64). Both trees'
Triton rows are nearly identical (17.63/17.92, 32.50/33.70), so the movement is
all on the handwritten path; the Triton path has been at this level throughout.

**We are behind llama.cpp, and the gap widens with concurrency**: 1.5x at c=1,
2.2x at c=4, 2.9x at c=8. The shape is the finding, not the ratio. llama.cpp
scales 3.2x from c=1 to c=8 (27.04 to 86.19) where our best row scales 1.65x
(17.92 to 29.64). **Triton lifted single-stream throughput and did not change the
scaling at all** — its curve has the same shape as the handwritten one — so the
scaling deficit is not kernel quality, and more kernels will not fix it.

That is consistent with what the x86 tier offers. All five ISA tiers in
`cpu_isa_x86.cpp` advertise the same capability string, `elementwise-gemm`; the
top one requires `avx512f/bw/vl` and not `avx512_bf16`, and no `dpbf16`
instruction appears anywhere under `src/vt/cpu/`. The Arm `kI8mm` tier advertises
`quant-dot,quant-repack`. Upstream's README positions this backend as
"Correctness / CI reference" with an "Arm i8mm tier", so the x86 gap is a
declared design position rather than a defect found here.

**llama.cpp collapses at c=32 and we do not**: 86.19 to 45.36 on WSL2, 93.97 to
56.92 on Windows, against four of our legs still climbing. Its sweet spot is c=8
to c=16.

### Windows, as a platform control

Same axis, same thread count. No Triton row exists here by construction.

| Leg | c=1 | c=2 | c=4 | c=8 | c=16 † | c=32 |
|---|---:|---:|---:|---:|---:|---:|
| ours, lab tree, handwritten | 3.89 | 5.70 | 7.57 | 9.25 | 10.49 | 10.62 |
| ours, public tree, handwritten | 6.03 | 10.09 | 13.69 | 17.37 | 19.82 | 20.52 |
| llama.cpp | 27.04 | 36.62 | 53.85 | 76.95 | 93.97 | 56.92 |

On the lab tree Windows leads WSL2 by 1.22x to 1.37x, narrowing as concurrency
rises — the shape of a fixed per-syscall virtualisation cost being amortised, and
an ordinary Hyper-V overhead rather than anything about the engine. **On the
public tree the direction reverses** (6.03 against 9.64 at c=1): upstream's
handwritten improvements land better under Linux than under Windows. That is
unexplained, and is the open item in this section.

### This section replaces an earlier one that was wrong in four ways

The previous CPU numbers were taken at 32 threads. On Linux 32 is a cliff, not a
plateau — 32.9 to 0.702 output tok/s, a 47x collapse — and the value was carried
over from a Windows sweep whose own timing ran through `powershell.exe`, which
costs 0.28 to 1.19 s on this host against a 1.43 s measured window. Four
conclusions came out of that, and none survived:

| Was published | Actually |
|---|---|
| Linux CPU is 2.8-8.5x slower than Windows | Windows leads by 1.22-1.37x on the lab tree, and TRAILS on the public tree |
| Windows thread optimum is 32 | 16; 32 costs only 6.4% there |
| 32 threads is a cliff, as an engine property | Linux only; Windows has no cliff |
| CPU is 4-5x behind llama.cpp | 1.5x to 2.9x; the old figure compared our cliff value against llama.cpp's optimum |

Method and the full post-mortem: `cpu-measurement-rules.md`, and
`docs/vllm-public-porting-plan.md` section 6l.

## Vulkan

Qwen3.6-27B BF16, c=1, TOTAL tok/s. Two artefacts and two binaries. The compat
binary is upstream `main` plus the three Windows commits that let it load a model
at all, and nothing else: no added shaders, no tuning. So the ours/compat pair
measures what the added kernels are worth, not what the tuning is worth.

| | ours | upstream + Windows fixes | ratio | ops on the portable CPU tier |
|---|---:|---:|---:|---|
| safetensors | **201.25** | 10.50 | **19.2x** | 1 vs 2 |
| GGUF | 7.17 | 3.30 | 2.2x | 1 vs 3 |

**The safetensors row is the result; the GGUF row is two broken paths compared to
each other.** Upstream drops `RopeCosSinCache`, `CausalConv1dFwd` and
`MoeSiluMul` to the portable CPU tier on GGUF and `CausalConv1dFwd` on
safetensors, which is what the 19.2x is: a decode loop that leaves the GPU once
per layer against one that does not. Adding those shaders is the port's main
Vulkan contribution.

**Our own GGUF path is a defect, not a slow cell, and it is not in the headline.**
It reads 7.17 against the same binary's 201.25 on safetensors, a 28x gap on
identical weights. What it is not: the kernels (`coopmat`, `gemv` and matmul-arm
selection log line-for-line identically on both artefacts, `bt=0` both), the
weights failing to reach the device (both reach about 60 GiB in flight), the port
(the lab tree reads 7.10, within noise of the public tree's 7.17), or the added
`MoeSiluMul` shader (byte-identical in both trees, and zero fallbacks are
logged). What it is, measured with `VT_VULKAN_DISPATCH_STATS=1` on one 8-token
request:

| | dispatches | elapsed | rate | flush reasons |
|---|---:|---:|---:|---|
| safetensors | 7300 | 95.8 s | 76/s | ring-full 14, copy-src 16, synchronize 16 |
| GGUF | 500 | 315.1 s | 2/s | synchronize 8 of 9, ring-full **0** |

Fewer dispatches and far more time. The safetensors path batches until the
command ring fills, up to 218 dispatches in one submit; the GGUF path never fills
a ring and almost every flush is the host waiting on the GPU. That is a broken
pipeline, not slow arithmetic, and it matches the per-step cost being flat at
about 1.28 s from c=1 to c=16.

An earlier record of this cell said "1.63x behind safetensors". That figure came
from a run whose own harness marked it `unstable` at a 47.3% spread across
repetitions of 18.4 / 11.8 / 13.9, and whose two sibling runs exited mid-sweep.
It is withdrawn.

### vs llama.cpp Vulkan, and this one does not go our way

Same weights, different container: our leg reads the BF16 safetensors, llama.cpp
reads the BF16 GGUF built from it. TOTAL tok/s.

| Concurrency | ours | llama.cpp Vulkan | ours/llama |
|---:|---:|---:|---:|
| 1 | 201.25 | **232.59** | 0.87x |
| 2 | 320.41 | **398.29** | 0.80x |
| 4 | 428.61 | **624.40** | 0.69x |
| 8 | 558.92 | **772.88** | 0.72x |
| 16 | **563.04** | 315.87 (310% spread) | 1.78x |
| 32 | **875.68** | 449.65 | 1.95x |

**llama.cpp is ahead by 1.15x to 1.46x wherever both engines are stable**, which
is c=1 through c=8. That is the row to read.

**The c=16 and c=32 rows are not a win and should not be quoted as one.**
llama.cpp's c=16 point carries a 310% spread, and its c=32 figure, 449.65, is
below its own c=8 figure of 772.88 — an engine that goes backwards under load is
not a denominator. Our arm is stable at both points, which is worth something,
but "1.95x faster at c=32" would be a claim about llama.cpp's concurrency
handling wearing the costume of a claim about ours.

This reproduces the shape recorded on 2026-09-08 on the same axis (0.926 / 0.878 /
0.849 / 0.813 then a spurious 2.15 and 2.03), with our c=4 slightly worse today,
0.69x against 0.849x. Two independent runs a day apart agreeing on both the
ranking and the instability is the useful part.

**Host memory: the Vulkan platform declares no residency policy.**
`src/vllm/platforms/vulkan.cpp` returns a default-constructed `ResidencyPolicy`,
so `release_host_weights_after_upload` is false where `CudaResidencyPolicy` sets
it true with the comment "a discrete GPU reclaims host RAM". Measured
consequence: a 50 GiB model, a 84 GiB server working set. The fix is not to copy
CUDA's values, because Vulkan also runs on integrated and unified-memory devices
where keeping the single copy is correct; it needs to branch on
`VkPhysicalDeviceType`. Whether this has ever caused a failure is not established
here.

## CUDA

Qwen3.6-27B NVFP4, output tok/s. One binary, one model, one server per leg; the
only difference is the four GDN Triton launchers, which default ON and are turned
off together for U1 (`VT_GDN_PACKED_DECODE_TRITON`, `VT_GDN_DELTAH_TRITON`,
`VT_GDN_CHUNKO_TRITON`, `VT_GDN_WU_TRITON` all `=0`). This model is hybrid, 288
`linear_attn` layers to 17 `full_attention`, so the GDN path is on the hot path
rather than dormant.

| Concurrency | handwritten | Triton AOT | ratio | spread (hw / AOT) |
|---:|---:|---:|---:|---|
| 1 | 22.18 | 21.71 | 0.98x | 0.0% / 0.0% |
| 2 | 40.01 | 40.12 | 1.00x | 0.0% / 0.0% |
| 4 | 70.23 | 71.21 | 1.01x | 0.0% / 0.0% |
| 8 | 118.02 | 115.09 | 0.98x | 0.0% / 0.0% |
| 16 | 167.36 | 158.75 | 0.95x | 1.4% / 3.9% |
| 32 | 193.70 | 190.60 | 0.98x | 3.8% / 1.7% |

**Parity.** Every ratio sits inside 5% and most inside the measured spread. The
vendored `sm_120a` AOT tree does not make this model faster on this card.

**What it does do is make the path reachable at all**, and that is the honest
claim for it. The runtime selector matches compute capability exactly and never
substitutes a near architecture (`triton_aot_arch_dispatch.h`: "a cubin is never
tried on a merely similar architecture", which is a property of cubins, not a
policy choice). Upstream ships `sm_80 sm_86 sm_89 sm_90a sm_100a sm_121a`; 12.0
is absent, so `TritonAotTreeIndex` returns -1 on this card and an upstream binary
runs the handwritten kernels with no diagnostic. Adding the tree changes that
from "silently skipped" to "measured, and worth nothing here". Both are useful to
know; only the second one can be written down.

### The pair is not token-identical, and the difference is reproducible

The AOT path exists for fidelity, not speed: the header states a mirror policy,
that vLLM runs this exact FLA kernel by default so we do too. The table above
measures throughput, which is not the claim. So the claim was checked.

Same binary, same checkpoint, same flags, greedy (`temperature 0`), 200 tokens,
three prompts, and each leg run twice.

The controls come first. Repeating the handwritten leg reproduces itself token
for token on all three prompts, and so does repeating the AOT leg. Neither leg is
noisy at this concurrency, so a difference between them cannot be run-to-run
variation.

Across the switch, two of the three prompts diverge. They agree for 61 tokens of
70 and for 97 tokens of 200 respectively, and the third agrees for all 200. The
repeat pair diverges at exactly the same two positions, so the divergence is a
property of the switch and not of the run.

What diverges is phrasing. One completion continues "scattering process gives
the ocean", the other "scattering process is what gives the ocean"; the other
says "divisibility by all prime numbers less than or equal to 31" against
"divisibility by the prime numbers", and both then list the same primes and reach
the same conclusion. That is the shape of an argmax flipping at a near-tie
between two float reduction orders, and no factual content differs on this
sample.

**One divergence is an exact tie and the other is not.** Separating "the two
paths ranked a coin flip differently" from "one of them computed something else"
needs the margin between the top two candidates at the divergent position. That
was unreadable when this section was first written, because the endpoint returned
null logprobs; two defects behind that have since been fixed and the margins are
now measured, at a third of the earlier KV footprint, with both divergence
positions reproducing token for token.

At the second prompt's token 97 the top two candidates are *exactly equal* on
both legs (0.000000 nats: ` all` against ` prime` on the handwritten leg,
` all` against ` the` on the AOT leg). Which one wins is settled by tie-break
order, so that divergence is a coin flip in the literal sense.

At the first prompt's token 61 it is not: the handwritten leg ranks ` gives`
above ` is` by 0.125 nats and the AOT leg ranks them the other way by 0.250, so
the two executions assign the same candidates values that differ by about 0.19
nats. Every candidate logprob here lands on a 1/8-nat grid, which is what NVFP4
weights and an fp8 KV cache leave of the resolution — coarse enough that exact
ties are common and that one or two steps of drift reorders a pair.

So: one tie-break, one precision-difference. Neither is a wrong answer, and
neither should be described as the other.

One caveat on the switch evidence. Only the delta-H launcher announces itself on
stderr (`TRITON DELTA-H BAILED: env-off` against `ENGAGED (first launch,
hv_n=48)`); the other three print nothing, so one of the four is directly
witnessed and three are inferred. An earlier version of this check counted
matching lines instead of reading them, got one line from each leg, and would
have reported the two legs as indistinguishable.

### vs vLLM on NVFP4, under a constrained configuration

llama.cpp cannot read NVFP4, so vLLM is the only reference this artefact has.
vLLM 0.28.0, same checkpoint, same 1024-in/128-out axis, TOTAL tok/s.

| Concurrency | ours (handwritten) | ours (Triton AOT) | vLLM 0.28.0 | ours/vLLM |
|---:|---:|---:|---:|---:|
| 1 | **199.93** | 195.72 | 170.97 | 1.17x |
| 2 | **360.69** | 361.69 | 264.44 | 1.36x |
| 4 | 633.18 | **641.98** | 585.29 | 1.08x |
| 8 | **1064.06** | 1037.65 | 1010.85 | 1.05x |
| 16 | 1508.81 | 1431.27 | **1646.59** | 0.92x |
| 32 | 1746.37 | 1718.40 | **2322.32** | 0.75x |

**We lead below c=16 and vLLM takes it from there, ending 1.33x ahead at c=32.**
vLLM's spreads at those two points are 4.7% and 1.6%, so the crossover is real and
not the collapse-under-load seen on the llama.cpp arms.

**Three non-default settings, all of which handicap vLLM, and two of them are
ours to own.** This row is "vLLM under constraint", not "vLLM":

| Setting | Why | Whose |
|---|---|---|
| `--enforce-eager` | no CUDA graph capture. Ours run with graphs | ours, to keep the card usable |
| `--attention-backend TRITON_ATTN` | FlashInfer is first in vLLM's own priority order and is not installed in this venv | environment |
| weight-only FP4 | vLLM reports "Your GPU does not have native support for FP4 computation" and takes the Marlin weight-only path; our build sets `VT_FP4_MMA_SM120A=1` | engine difference, not a setting |

The third one matters most: on the same NVFP4 file the two engines are **not
running the same kind of arithmetic**, so this is not a clean kernel-vs-kernel
comparison in either direction. And since the first two also cost vLLM speed, the
sub-c=16 lead is the softest number on this page: a full-configuration vLLM on a
card that does not also drive a display could plausibly take those rows too.

### Memory footprint, and why the legs differ

| Leg | device memory during run | weights alone |
|---|---:|---:|
| llama.cpp Vulkan BF16 | 68.0 GB | 53.8 GB |
| ours Vulkan safetensors BF16 | 72.1 GB | 55.6 GB |
| ours CUDA NVFP4 | 48.1 GB | 21.9 GB |
| ours CUDA Q4_K_M | 43.8 GB | 19.1 GB |
| vLLM NVFP4 | 42.9 GB | 20.0 GB |

**The 60-70 GB legs are the BF16 ones, and it is the artefact, not the cache
policy.** Within the CUDA column all three engines land at 43 to 48 GB, so vLLM's
42.9 GB is ordinary rather than starved.

⚠️ **These are spot readings, not a measured series.** They were taken at
different moments against a desktop baseline that itself moved between 6.7 and
12 GB, so the engine-only footprint cannot be derived from them and no peak-RSS
comparison is claimed here. A per-leg steady-state peak with its own baseline
sample is owed.

### The same-file CUDA pair, and the artefact rule that forces it

llama.cpp cannot read NVFP4, so the U1/U2 rows above have no llama.cpp
denominator and never will. The artefact both engines read is Q4_K_M, which is
also upstream's own filed plan for this gap (`open-gaps.md`,
`BACKEND-GATE-CUDA-LLAMACPP`: "**No run** ... Our Q4_K_M arm, then a same-file
same-box quant-matched run"). One file, `Qwen3.6-27B-Q4_K_M.gguf`, 19.10 GB, both
arms, TOTAL tok/s.

| Concurrency | ours | llama.cpp CUDA | llama/ours |
|---:|---:|---:|---:|
| 1 | 97.77 | **465.50** | 4.8x |
| 2 | 101.20 | **727.23** | 7.2x |
| 4 | 107.03 | **907.94** | 8.5x |
| 8 | 108.86 | **1250.53** | 11.5x |
| 16 | 107.77 | 366.08 (423% spread) | |
| 32 | 105.46 | 530.76 | |

**Our arm does not scale at all.** 10.84 output tok/s at c=1 and 11.70 at c=32,
flat across a 32x change in batch. The same binary on the same backend reading
NVFP4 goes 22.18 to 118.02 over c=1 to c=8, a 5.3x scale-up. Same binary, same
device, only the artefact changed.

**The device was named, not assumed.** This binary registers the Vulkan platform
as well ("`platform REGISTERED for kVULKAN`" appears in every CUDA leg's log), and
these legs ran with `--device auto`, so the placement was re-checked with
`--device cuda` spelled out: 10.62 and 11.80 against `auto`'s 10.84 and 12.07,
the same curve. `--device vulkan` is rejected outright by this build (`Unknown
device: vulkan (expected one of: auto, cpu, cuda)`), so `auto` could not have
selected it. The CUDA placement is confirmed rather than inferred.

**This retires the earlier localisation of the GGUF defect.** It was recorded as
a Vulkan dispatch-rhythm problem, on the evidence of flush reasons in the Vulkan
backend. But the flat curve now appears on CUDA too, with a different GGUF
quantization, in a backend that shares none of that dispatch code:

| | artefact | shape across concurrency |
|---|---|---|
| Vulkan | GGUF BF16 | per-step cost flat at ~1.28 s, c=1 to c=16 |
| CUDA | GGUF Q4_K_M | throughput flat at ~11.5 tok/s, c=1 to c=32 |
| CUDA | NVFP4 | 22.18 to 118.02, scales |

Two backends, two quantizations, one shape. The common term is the GGUF path, not
the Vulkan backend, and the Vulkan dispatch measurements are a symptom seen from
one side rather than the cause.

Roofline, for scale, on a device with roughly 1.8 TB/s: llama.cpp on Q4_K_M moves
986 GB/s of weights per second at c=1 (55% of it), our NVFP4 path 486 GB/s (27%),
our Q4_K_M path 207 GB/s (11%). Our GGUF path is below our own NVFP4 path, which
is itself well below the device.

**Scope of the A/B, stated because the switch and the evidence do not fully
overlap.** Four launchers are toggled together, and only the delta-H launcher
announces itself (`[vt gdn] TRITON DELTA-H ENGAGED (first launch, hv_n=48)`,
present in the AOT leg and absent in the handwritten one, which is what confirms
the toggle took). The other three print nothing either way, so this pair cannot
attribute the result among the four, and a per-kernel answer needs per-kernel
switching.

## The Linux control, and what it separates

Every GPU figure above is a Windows figure, which entangles "this engine is slow
here" with "on Windows" in a way no amount of care on the Windows side can undo.
So the same commit was built on the same box under WSL2 (Ubuntu, GCC, CUDA 13.2,
`sm_120a`, ccache) and the CUDA leg re-run with the checkpoint copied to ext4
rather than served over the NTFS bridge.

**CUDA is platform-neutral, and Linux is slightly ahead.**

| Concurrency | CUDA Windows (AOT) | CUDA Linux | Linux/Windows |
|---:|---:|---:|---:|
| 1 | 195.72 | **202.25** | 1.03x |
| 2 | 361.69 | **364.51** | 1.01x |
| 4 | 641.98 | **651.83** | 1.02x |
| 8 | 1037.65 | **1110.44** | 1.07x |
| 16 | 1431.27 | **1594.44** | 1.11x |
| 32 | 1718.40 | **1883.72** | 1.10x |

Output tok/s on Qwen3.6-27B NVFP4. Linux leads by 1 to 11%, widening with
concurrency, and its high-concurrency spreads are tighter (1.0% and 2.6% against
3.9% and 1.7%).

**What this rules out.** A general WSL2 penalty would land here, and it does not.
So the CPU section's platform comparison — Windows ahead by 1.22-1.37x on the lab
tree, behind on the public tree — is about the CPU path specifically, not about
the virtualisation layer being slow in general. Hyper-V overhead on a
compute-bound GPU workload is evidently near zero.

**An earlier version of this section reported the CPU side as "Linux is 4.1x to
6.7x behind Windows".** That was measured at 32 threads on both sides, which is
Windows' plateau and Linux's cliff. Corrected numbers are in the CPU section
above; the CUDA rows here were never affected, because the thread knob does not
apply to them.

**The c=32 exit did not reproduce on Linux.** The CPU configuration that ended
the Windows server with no diagnostic completed all three repetitions under Linux
with the server still alive afterwards. That is one crash and one clean run, n=1
on each side, so it supports "did not reproduce here" and not "is
Windows-specific".

## Open, and what flips first

| Flips first | Verdict | Exposure |
|---|---|---|
| 1 | CUDA `0.95x` at c=16 | the only ratio outside 2%, and its two legs carry 1.4% and 3.9% spreads. One re-take can move it either side of parity, and the parity verdict for the whole table leans on it |
| 2 | CPU `5.79x` at c=1 | our leg reads 19% under three clean restarts of the same configuration. On restart medians the same row is 4.9x. The direction is not in doubt at this size; the figure is |
| 3 | Vulkan `19.2x` | one repetition, one instance, no restart band taken. The mechanism behind it (three ops on the portable CPU tier upstream) is not in doubt, but the multiplier is a single sample |

### Open defects

**Silent server exit, at least three contexts.** No error line, log ends at
`listening on`. Seen at CPU c=32 on safetensors with a 0.6B model, on the Vulkan
arm during a c=1 GGUF sweep, and in two of three repetitions of a 2026-09-08 GGUF
run whose guard recorded 86 GiB of device memory free. It is not specific to a
backend, an artefact, or a model size, and the 0.6B case rules out weight memory
as the cause. No reproducer yet.

**The GGUF path does not scale with concurrency, on either backend.** An earlier
version of this list said "GGUF on Vulkan, localised to the dispatch rhythm". That
localisation is retired above: the same flat curve appears on CUDA with a different
quantization, so the common term is the GGUF path and not a backend. The call site
that forces a synchronize per step is still not found.

**The Vulkan platform declares no residency policy.** Measured cost is a 84 GiB
working set for a 50 GiB model. Fixing it needs a `VkPhysicalDeviceType` branch,
because the CUDA values are wrong for integrated and unified-memory devices.

### Not measured here, and why

| Cell | Why |
|---|---|
| Triton-CPU | `dlopen` is compiled out on Windows; the cell belongs to a Linux leg |
| vLLM as a CUDA reference | needs the NVFP4 checkpoint on Linux-native storage first |
| Upstream `main` on the historical axis | runnable, not run. The Windows-fix binary used for the compat legs IS upstream `main` plus three commits, so the comparison against an older stock data point is available and simply has not been taken |

| Energy | no instrumentation on this host |
