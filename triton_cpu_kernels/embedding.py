# triton_cpu_kernels/embedding.py
#
# Token embedding gather: out[row, :] = table[ids[row], :].
#
# Compiled AHEAD OF TIME into embedding.so and loaded at run time by the
# Triton-CPU provider (src/vt/triton_cpu/triton_cpu_provider.cpp) via dlopen,
# which looks up the symbol `embedding_kernel`. Nothing here runs inside the
# engine: the shipped binary contains no Python and no Triton.
#
# The provider declines back to the built-in kernel for any shape it cannot
# take, so an absent or unloadable .so leaves output byte-identical.
#
# Reference implementation this must match: src/vt/cpu/cpu_ops.cpp, EmbeddingKernel
#
# See README.md in this directory for the build recipe, the toolchain it needs,
# and WHICH kernels are worth loading (measured: not all of them).

import triton
import triton.language as tl

# do_not_specialize on table_ptr for the same reason as rms_norm's weight: the
# table is a mapped weight, not an engine allocation.
@triton.jit(do_not_specialize=["table_ptr"])
def embedding_kernel(out_ptr, table_ptr, ids_ptr, h, table_stride, out_stride,
                     BLOCK: tl.constexpr):
    row = tl.program_id(0)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    for start in range(0, h, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < h
        x = tl.load(table_ptr + idx * table_stride + cols, mask=mask, other=0)
        tl.store(out_ptr + row * out_stride + cols, x, mask=mask)
