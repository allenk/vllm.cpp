# triton_cpu_kernels/greedy_argmax.py
#
# out[i] = argmax(logits[i, :]), with the reference tie rule.
#
# Compiled AHEAD OF TIME into greedy_argmax.so and loaded at run time by the
# Triton-CPU provider (src/vt/triton_cpu/triton_cpu_provider.cpp) via dlopen,
# which looks up the symbol `greedy_argmax_kernel`. Nothing here runs inside the
# engine: the shipped binary contains no Python and no Triton.
#
# The provider declines back to the built-in kernel for any shape it cannot
# take, so an absent or unloadable .so leaves output byte-identical.
#
# Reference implementation this must match: src/vt/cpu/cpu_sample.cpp, GreedyArgmaxKernel
#
# See README.md in this directory for the build recipe, the toolchain it needs,
# and WHICH kernels are worth loading (measured: not all of them).

import triton
import triton.language as tl

# Two things make this heavier than it looks:
#   (a) a 151936-wide vocabulary does not fit one tile (608 KB), so it needs a
#       blocked loop with a running accumulator, which is the same structure
#       online softmax and paged attention need;
#   (b) it is the first kernel here that crosses dtypes: f32 in, i64 out.
#
# TIE RULE. The reference uses a strict `>`, so the FIRST (lowest-index) maximum
# wins and the result is bit-exact against torch.argmax. Both halves matter:
# inside a block take the lowest index with tl.min; across blocks merge with a
# strict `>` so a later block never displaces an equal earlier one.
@triton.jit
def greedy_argmax_kernel(out_ptr, logits_ptr, v, row_stride,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = logits_ptr + row * row_stride
    best_v = -float("inf")
    best_i = 0
    for start in range(0, v, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < v
        x = tl.load(base + cols, mask=mask, other=-float("inf"))
        m = tl.max(x, axis=0)
        idx = tl.min(tl.where(x == m, cols, v), axis=0)   # lowest index in-block
        take = m > best_v                                  # strict: a tie does not displace
        best_i = tl.where(take, idx, best_i)
        best_v = tl.where(take, m, best_v)
    tl.store(out_ptr + row, best_i.to(tl.int64))
