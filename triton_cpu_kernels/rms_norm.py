# triton_cpu_kernels/rms_norm.py
#
# RMS normalisation without a residual.
#
# Compiled AHEAD OF TIME into rms_norm.so and loaded at run time by the
# Triton-CPU provider (src/vt/triton_cpu/triton_cpu_provider.cpp) via dlopen,
# which looks up the symbol `rms_norm_kernel`. Nothing here runs inside the
# engine: the shipped binary contains no Python and no Triton.
#
# The provider declines back to the built-in kernel for any shape it cannot
# take, so an absent or unloadable .so leaves output byte-identical.
#
# Reference implementation this must match: src/vt/cpu/cpu_ops.cpp, RmsNormKernel (no-residual branch)
#
# See README.md in this directory for the build recipe, the toolchain it needs,
# and WHICH kernels are worth loading (measured: not all of them).

import triton
import triton.language as tl

# TWO PASSES, and the second one RE-READS x. A whole-row version holds the row
# in registers and reads it once; the blocked version has to compute the sum of
# squares before it can normalise anything. This op is memory-bound, so unlike
# the elementwise kernels the shape-general form here may genuinely cost
# something, which is a prediction worth measuring rather than assuming.
#
# do_not_specialize on w_ptr: the weight arrives through the mmap/repack path
# and is measured at 8-byte alignment, while Triton would otherwise assert
# divisibility by 16. Do not let it make that claim.
@triton.jit(do_not_specialize=["w_ptr"])
def rms_norm_kernel(out_ptr, x_ptr, w_ptr, eps, n, x_stride, out_stride,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = x_ptr + row * x_stride
    sumsq = tl.zeros([], dtype=tl.float32)
    for start in range(0, n, BLOCK):                       # pass 1: sum of squares
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n
        x = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)
    inv = 1.0 / tl.sqrt(sumsq / n.to(tl.float32) + eps)
    for start in range(0, n, BLOCK):                       # pass 2: normalise
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n
        x = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * out_stride + cols,
                 (x * inv * w).to(out_ptr.dtype.element_ty), mask=mask)
