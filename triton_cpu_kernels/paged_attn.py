# triton_cpu_kernels/paged_attn.py
#
# Paged attention over the KV cache, shape-general in the head dim.
#
# Compiled AHEAD OF TIME into paged_attn.so and loaded at run time by the
# Triton-CPU provider (src/vt/triton_cpu/triton_cpu_provider.cpp) via dlopen,
# which looks up the symbol `paged_attn_general_kernel`. Nothing here runs inside the
# engine: the shipped binary contains no Python and no Triton.
#
# The provider declines back to the built-in kernel for any shape it cannot
# take, so an absent or unloadable .so leaves output byte-identical.
#
# Reference implementation this must match: src/vt/cpu/cpu_paged_attn.cpp, PagedAttentionKernel
#
# See README.md in this directory for the build recipe, the toolchain it needs,
# and WHICH kernels are worth loading (measured: not all of them).

import triton
import triton.language as tl

# D is a RUN-TIME argument, walked by an inner loop, so one artifact serves
# every head dimension. The provider selects this entry point only when the
# artifact directory's manifest.txt says `style=loop`; a directory without that
# manifest is read as the older grid style and the provider looks up
# `paged_attn_kernel` instead, which this file does not define. A missing
# manifest therefore means this kernel silently never loads.
@triton.jit
def paged_attn_general_kernel(out_ptr, q_ptr, kc_ptr, vc_ptr, btab_ptr,
                              pos_ptr, slen_ptr, req_ptr,
                              qpk, block_size, scale, causal,
                              window_left, window_right,
                              q_stride_t, q_stride_h, out_stride_t, out_stride_h,
                              bt_row, bt_col,
                              kc_blk, kc_pg, kc_hd, vc_blk, vc_pg, vc_hd,
                              D,
                              BD: tl.constexpr, BJ: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    r = tl.load(req_ptr + t).to(tl.int32)
    p = tl.load(pos_ptr + t).to(tl.int32)
    seqlen = tl.load(slen_ptr + t).to(tl.int32)

    jmin = tl.where(window_left >= 0, tl.maximum(0, p - window_left), 0)
    jmax = tl.where(causal != 0, p, seqlen - 1)
    jmax = tl.where(window_right >= 0, tl.minimum(jmax, p + window_right), jmax)
    jmax = tl.minimum(jmax, seqlen - 1)

    if jmax >= jmin:
        g = h // qpk
        qbase = q_ptr + t * q_stride_t + h * q_stride_h

        # Pass A: online softmax. One sweep produces both the running maximum
        # and the denominator.
        m = -float("inf")
        denom = 0.0
        for j0 in range(jmin, jmax + 1, BJ):
            js = j0 + tl.arange(0, BJ)
            jm = js <= jmax
            blk = tl.load(btab_ptr + r * bt_row + (js // block_size) * bt_col,
                          mask=jm, other=0).to(tl.int64)
            off = js % block_size
            kb = blk * kc_blk + off * kc_pg + g * kc_hd
            s = tl.zeros((BJ,), dtype=tl.float32)
            for d0 in range(0, D, BD):                    # inner loop over D
                de = d0 + tl.arange(0, BD)
                dm = de < D
                q = tl.load(qbase + de, mask=dm, other=0.0).to(tl.float32)
                k = tl.load(kc_ptr + kb[:, None] + de[None, :],
                            mask=jm[:, None] & dm[None, :], other=0.0).to(tl.float32)
                s += tl.sum(k * q[None, :], axis=1)       # accumulate across D blocks
            s = s * scale
            s = tl.where(jm, s, -float("inf"))
            m_new = tl.maximum(m, tl.max(s, axis=0))
            # The old denominator is rescaled to the new maximum. This is the
            # whole of online softmax.
            denom = denom * tl.exp(m - m_new) + \
                tl.sum(tl.where(jm, tl.exp(s - m_new), 0.0), axis=0)
            m = m_new

        # Pass B: one sweep over j per D block, and s has to be recomputed
        # because a partial-D score is not a score.
        for db in range(0, D, BD):
            deo = db + tl.arange(0, BD)
            dmo = deo < D
            acc = tl.zeros((BD,), dtype=tl.float32)
            for j0 in range(jmin, jmax + 1, BJ):
                js = j0 + tl.arange(0, BJ)
                jm = js <= jmax
                blk = tl.load(btab_ptr + r * bt_row + (js // block_size) * bt_col,
                              mask=jm, other=0).to(tl.int64)
                off = js % block_size
                kb = blk * kc_blk + off * kc_pg + g * kc_hd
                vb = blk * vc_blk + off * vc_pg + g * vc_hd
                s = tl.zeros((BJ,), dtype=tl.float32)
                for d0 in range(0, D, BD):                # recompute s over full D
                    de = d0 + tl.arange(0, BD)
                    dm = de < D
                    q = tl.load(qbase + de, mask=dm, other=0.0).to(tl.float32)
                    k = tl.load(kc_ptr + kb[:, None] + de[None, :],
                                mask=jm[:, None] & dm[None, :], other=0.0).to(tl.float32)
                    s += tl.sum(k * q[None, :], axis=1)
                s = s * scale
                e = tl.where(jm, tl.exp(s - m), 0.0)
                v = tl.load(vc_ptr + vb[:, None] + deo[None, :],
                            mask=jm[:, None] & dmo[None, :], other=0.0).to(tl.float32)
                acc += tl.sum(e[:, None] * v, axis=0)
            tl.store(out_ptr + t * out_stride_t + h * out_stride_h + deo,
                     (acc / denom).to(out_ptr.dtype.element_ty), mask=dmo)
