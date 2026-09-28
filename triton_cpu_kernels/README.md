# `triton_cpu_kernels/`: the CPU kernels the Triton-CPU provider loads

> **Added in this fork ([allenk/vllm.cpp](https://github.com/allenk/vllm.cpp)).
> Upstream ([mudler/vllm.cpp](https://github.com/mudler/vllm.cpp)) has no
> Triton-CPU provider.**

The engine ships **the ability to load a CPU kernel, not a kernel**. The
provider in [`src/vt/triton_cpu/`](../src/vt/triton_cpu/) is compiled into every
binary and is inert until you point it at a directory of compiled `.so` files.
This directory holds the sources those `.so` files are built from.

These are **not** the CUDA Triton kernels in [`triton_kernels/`](../triton_kernels/).
Those are AOT-compiled to cubins at build time under `VLLM_CPP_TRITON`. These are
CPU kernels, built separately, loaded at run time, and not part of any build or
release step.

## 1. Which kernels to load, because it is not all of them

Measured on the published `v0.0.3-vk.1` Linux archive, Qwen3-0.6B, 128-in /
32-out, c=1, 16 threads, two interleaved passes, each cell reporting the range
across passes:

```
hand-written GEMM + Triton paged_attn, rms_norm, rms_norm_resid, silu_and_mul
                                             28.32 - 29.37 out tok/s   <- best
all hand-written                             26.78 - 26.83
all eight Triton kernels                     24.02 - 24.99   <- worse than none
```

⚠️ **Loading everything is slower than loading nothing.** The difference between
the second and third rows is dominated by `matmul_bt`: it is a GEMV expression,
it ties with the hand-written path at M=1, and it loses on prefill calls with
large M that the hand-written path amortises across rows. `embedding`,
`greedy_argmax` and `reshape_and_cache` measured as neutral-to-negative on this
axis.

⇒ **Start with the four in the first row.** A directory is the unit of
selection, so building a directory that contains only those four is how you
select them.

⚠️ The magnitude does not travel. A different model or axis gives a different
number, and an earlier measurement on a different model showed a much larger
gain. Treat the RANKING as the result and re-measure the ratio on your own
workload.

## 2. What you need to build them

```
triton-cpu          the CPU backend of Triton, built from source.
                    NOT the triton package on PyPI, which has no CPU backend.
LLVM                whatever your triton-cpu revision pins.
a C toolchain       to link the compiled object into a shared library.
```

⚠️ **These sources are published without a pinned triton-cpu revision**, because
the artifacts here were built against a development checkout rather than a
tagged release. That is a real limitation: the kernels are plain Triton and
should compile against any recent triton-cpu, but "should" is not "was
measured". If you build them, the assertion in §4 is how you find out whether
your build behaves.

## 3. How the engine finds them

One directory, fixed file names, one symbol per file:

```
<dir>/silu_and_mul.so        silu_and_mul_kernel
<dir>/rms_norm.so            rms_norm_kernel
<dir>/rms_norm_resid.so      rms_norm_resid_kernel
<dir>/embedding.so           embedding_kernel
<dir>/reshape_and_cache.so   reshape_and_cache_kernel
<dir>/matmul_bt.so           matmul_bt_kernel
<dir>/greedy_argmax.so       greedy_argmax_kernel
<dir>/paged_attn.so          paged_attn_general_kernel
<dir>/manifest.txt           ⭐ REQUIRED, see below
```

```bash
VLLM_CPP_TRITON_CPU=1 VLLM_CPP_TRITON_CPU_DIR=/path/to/kernels ./vllm-server ...
```

Useful while you are checking your own build:

```bash
VT_OP_PROVIDER_STATS=1          # prints which provider won each op
VT_OP_PROVIDER_DISABLE=triton-cpu   # same binary, provider off, for an A/B
```

### ⭐ `manifest.txt` is not optional

```
style=loop
```

`paged_attn.py` here defines `paged_attn_general_kernel`, the shape-general
entry point. The provider only looks for that symbol when the manifest says
`style=loop`; **without a manifest it reads the directory as the older grid
style and looks up `paged_attn_kernel` instead, which does not exist here.**

The failure is silent in the only way that matters: the other kernels load, the
server runs, and the output is correct. You simply do not get the kernel that
carries most of the win. Copying `*.so` without the manifest is exactly how this
happened during the measurement above, and the only thing that caught it was a
count of loaded kernels not matching the count of files present.

⇒ **Check the count.** The provider logs `[triton-cpu] loaded <symbol> from
<path>` per kernel, and a `dlopen`/`dlsym` failure per miss. If loaded is fewer
than present, you are measuring something other than what you think.

## 4. What the provider re-checks at run time

The provider does not trust the artifact. Before dispatching it verifies dtype,
unit innermost stride, row strides and pointer alignment, and declines to the
built-in kernel if any of them disagree. Two consequences:

- A kernel compiled for assumptions your build does not meet **declines** rather
  than producing a wrong answer.
- An absent, unloadable or refused kernel leaves the output **byte-identical**
  to a build without this provider.

## 5. Platform limits

⚠️ **Windows: the provider is selected but never executes.** The kernel loader
is compiled out under `_WIN32`, so `VLLM_CPP_TRITON_CPU=1` makes
`VT_OP_PROVIDER_STATS` report `selected=triton-cpu` for seven ops while every
call forwards to the built-in kernel. Throughput is indistinguishable from the
hand-written path because it IS the hand-written path.

⚠️ **RISC-V: re-verify alignment.** These kernels rely on unaligned vector
access being safe, which was measured on x86-64 with AVX-512 (no aligned moves
emitted) but is implementation-defined elsewhere.

## 6. Not in CI, and not in any release

Nothing here is built, tested or packaged by this repository:

```
CMakeLists.txt    compiles the provider only, never a kernel
release.yml       packages no .so; the release archives contain none
ci.yml            tests the provider's LADDER STATES (present / asked-for but
                  absent / loaded), not the kernels themselves
```

⇒ **The correctness of these kernels is not covered by this project's CI.** They
are published so the feature is usable and inspectable, not as a supported
artifact. The run-time checks in §4 are what stands between a mismatched kernel
and a wrong answer.

## 7. Provenance

Each kernel mirrors a specific hand-written reference, named in its file header,
and is published **verbatim** from the sources the measured `.so` files were
built from. Only comments were translated; a line-by-line comparison of the code
after stripping comments confirms all eight are unchanged.
