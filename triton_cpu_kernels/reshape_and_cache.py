# triton_cpu_kernels/reshape_and_cache.py
#
# Scatter one token of K and V into the paged KV cache.
#
# Compiled AHEAD OF TIME into reshape_and_cache.so and loaded at run time by the
# Triton-CPU provider (src/vt/triton_cpu/triton_cpu_provider.cpp) via dlopen,
# which looks up the symbol `reshape_and_cache_kernel`. Nothing here runs inside the
# engine: the shipped binary contains no Python and no Triton.
#
# The provider declines back to the built-in kernel for any shape it cannot
# take, so an absent or unloadable .so leaves output byte-identical.
#
# Reference implementation this must match: src/vt/cpu/cpu_cache.cpp, ReshapeAndCacheKernel
#
# See README.md in this directory for the build recipe, the toolchain it needs,
# and WHICH kernels are worth loading (measured: not all of them).

import triton
import triton.language as tl

@triton.jit
def reshape_and_cache_kernel(k_ptr, v_ptr, kc_ptr, vc_ptr, slots_ptr,
                             n_elems, block_size,
                             k_tok_stride, v_tok_stride,
                             k_block_stride, k_page_stride,
                             v_block_stride, v_page_stride,
                             BLOCK: tl.constexpr):
    t = tl.program_id(0)
    slot = tl.load(slots_ptr + t)
    if slot >= 0:                          # a padding token carries -1
        blk = slot // block_size
        off = slot % block_size
        for start in range(0, n_elems, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < n_elems
            kx = tl.load(k_ptr + t * k_tok_stride + cols, mask=mask, other=0)
            vx = tl.load(v_ptr + t * v_tok_stride + cols, mask=mask, other=0)
            tl.store(kc_ptr + blk * k_block_stride + off * k_page_stride + cols,
                     kx, mask=mask)
            tl.store(vc_ptr + blk * v_block_stride + off * v_page_stride + cols,
                     vx, mask=mask)
