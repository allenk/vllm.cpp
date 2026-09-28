# triton_cpu_kernels/silu_and_mul.py
#
# SiLU-gated multiply: out[row, c] = silu(x[row, c]) * x[row, d + c].
#
# Compiled AHEAD OF TIME into silu_and_mul.so and loaded at run time by the
# Triton-CPU provider (src/vt/triton_cpu/triton_cpu_provider.cpp) via dlopen,
# which looks up the symbol `silu_and_mul_kernel`. Nothing here runs inside the
# engine: the shipped binary contains no Python and no Triton.
#
# The provider declines back to the built-in kernel for any shape it cannot
# take, so an absent or unloadable .so leaves output byte-identical.
#
# Reference implementation this must match: src/vt/cpu/cpu_ops.cpp, SiluAndMulKernel
#
# See README.md in this directory for the build recipe, the toolchain it needs,
# and WHICH kernels are worth loading (measured: not all of them).

import triton
import triton.language as tl

# Shape-general: the column range is walked by a loop rather than pinned by a
# compile-time tile, so one artifact serves every d.
@triton.jit
def silu_and_mul_kernel(out_ptr, x_ptr, d, x_stride, out_stride,
                        BLOCK: tl.constexpr):
    row = tl.program_id(0)
    for start in range(0, d, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < d
        gate = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0)
        up = tl.load(x_ptr + row * x_stride + d + cols, mask=mask, other=0.0)
        g = gate.to(tl.float32)
        u = up.to(tl.float32)
        y = (g / (1.0 + tl.exp(-g))) * u
        tl.store(out_ptr + row * out_stride + cols,
                 y.to(out_ptr.dtype.element_ty), mask=mask)
