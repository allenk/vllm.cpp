# triton_cpu_kernels/rms_norm_resid.py
#
# RMS normalisation fused with a residual add.
#
# Compiled AHEAD OF TIME into rms_norm_resid.so and loaded at run time by the
# Triton-CPU provider (src/vt/triton_cpu/triton_cpu_provider.cpp) via dlopen,
# which looks up the symbol `rms_norm_resid_kernel`. Nothing here runs inside the
# engine: the shipped binary contains no Python and no Triton.
#
# The provider declines back to the built-in kernel for any shape it cannot
# take, so an absent or unloadable .so leaves output byte-identical.
#
# Reference implementation this must match: src/vt/cpu/cpu_ops.cpp, RmsNormKernel (residual branch)
#
# See README.md in this directory for the build recipe, the toolchain it needs,
# and WHICH kernels are worth loading (measured: not all of them).

import triton
import triton.language as tl

# THE ROUNDING IS LOAD-BEARING. The reference stores x + r into the residual
# and then reads it BACK, so the sum of squares is taken over values already
# rounded to the residual dtype, not over the f32 sum. Skipping the round trip
# changes the output.
@triton.jit(do_not_specialize=["w_ptr"])
def rms_norm_resid_kernel(out_ptr, x_ptr, w_ptr, r_ptr, eps, n,
                          x_stride, out_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    sumsq = tl.zeros([], dtype=tl.float32)
    for start in range(0, n, BLOCK):   # pass 1: x+r, round, write back, accumulate
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n
        x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + row * out_stride + cols, mask=mask, other=0.0).to(tl.float32)
        v_r = (x + r).to(r_ptr.dtype.element_ty)                  # the rounding
        tl.store(r_ptr + row * out_stride + cols, v_r, mask=mask)
        v = v_r.to(tl.float32)                                    # use the rounded value
        sumsq += tl.sum(v * v, axis=0)
    inv = 1.0 / tl.sqrt(sumsq / n.to(tl.float32) + eps)
    for start in range(0, n, BLOCK):   # pass 2: v_r is already in r_ptr, re-read it
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n
        v = tl.load(r_ptr + row * out_stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * out_stride + cols,
                 (v * inv * w).to(out_ptr.dtype.element_ty), mask=mask)
