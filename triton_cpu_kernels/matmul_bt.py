# triton_cpu_kernels/matmul_bt.py
#
# out[m, n] = sum_k a[m, k] * b[n, k]. Expressed as a GEMV.
#
# Compiled AHEAD OF TIME into matmul_bt.so and loaded at run time by the
# Triton-CPU provider (src/vt/triton_cpu/triton_cpu_provider.cpp) via dlopen,
# which looks up the symbol `matmul_bt_kernel`. Nothing here runs inside the
# engine: the shipped binary contains no Python and no Triton.
#
# The provider declines back to the built-in kernel for any shape it cannot
# take, so an absent or unloadable .so leaves output byte-identical.
#
# Reference implementation this must match: src/vt/cpu/cpu_ops.cpp, MatmulBTKernel
#
# See README.md in this directory for the build recipe, the toolchain it needs,
# and WHICH kernels are worth loading (measured: not all of them).

import triton
import triton.language as tl

# NO tl.dot AND NO tl.trans. BM disappears entirely: m is a pure grid index, so
# that dimension is shape-general for free.
#
# BN != BK on purpose. A square tile triggers a full relayout the source never
# asked for; see docs/upstream/triton-cpu-square-tile-relayout.md.
#
# MEASURED WARNING, and it is the reason this file exists alongside a README:
# loading this kernel is a NET LOSS on the paths measured so far. Its cost is
# entirely in prefill, where it takes calls with large M that the hand-written
# path amortises across rows and this one does not. See README.md.
@triton.jit(do_not_specialize=["b_ptr", "M"])
def matmul_bt_kernel(out_ptr, a_ptr, b_ptr, M, N, K,
                     a_stride, b_stride, out_stride,
                     BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    nmask = offs_n < N
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        kmask = kk < K
        a = tl.load(a_ptr + pid_m * a_stride + kk, mask=kmask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offs_n[:, None] * b_stride + kk[None, :],
                    mask=nmask[:, None] & kmask[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(b * a[None, :], axis=1)
    tl.store(out_ptr + pid_m * out_stride + offs_n,
             acc.to(out_ptr.dtype.element_ty), mask=nmask)
