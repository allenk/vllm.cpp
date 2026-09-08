// vt::Gdn{Prefill,Decode} — the SHARED body of the two gated-delta recurrences.
// Included by vt_gdn_prefill.comp and vt_gdn_decode.comp AFTER they declare
// their bindings and push block, because GLSL functions cannot take buffer
// blocks as parameters: the step below names the buffer instances directly, so
// both shaders must declare the SAME instance names at the SAME bindings. That
// constraint is why this is an include and not a copy — the two ops differ only
// in how they pick the state row and how many tokens they run.
//
// PORT PROVENANCE.
//   * PER-STEP ARITHMETIC, 1:1 from src/vt/cpu/cpu_ops.cpp:1280-1311
//     GdnHeadTokenStep (itself the vLLM-parity golden for
//     fla fused_recurrent_gated_delta_rule_fwd_kernel):
//         q' = q*scale;  S *= exp(g);  v' = (v - S@k)*beta;  S += outer(v',k);
//         out = S@q'      (k is NOT scaled)
//     Both contractions stay fused with the pass that produces them, in that
//     order, exactly as the CPU reference writes them.
//   * DISPATCH SHAPE AND STATE HANDLING, ported from our own CUDA kernel
//     src/vt/cuda/cuda_gdn.cu:2417-2503 GdnDecodeFusedKernel: one block per
//     (sequence, value-head, BV-value-tile); the [BV,Dk] state slice staged into
//     shared memory ONCE by a COALESCED load, updated in place, written back
//     ONCE; NW lanes per value row splitting the Dk contraction and reducing
//     their partial sums. The CUDA header explains why: the alternative streams
//     the [Dv,Dk] state from GLOBAL four times per token, strided by Dk, so
//     uncoalesced.
//
// WHY (sequence, value-head, VALUE-ROW) IS A LEGAL PARALLEL AXIS.
// cpu_ops.cpp:1275-1279 states heads and sequences: distinct heads own disjoint
// [Dv,Dk] state blocks and disjoint output rows. The VALUE ROW is independent
// too and the CUDA kernel already exploits it as its grid.x (cuda_gdn.cu:2421,
// the fla NV value tiling): row vi of S is only ever touched as
//     S[vi] <- S[vi]*decay + v'[vi]*k,   out[vi] = S[vi]·q',
//     v'[vi] = (v[vi] - S[vi]·k)*beta
// — every term is either row-local or a per-(token,head) broadcast. Only the
// SEQUENCE POSITION is sequential, and it stays sequential, in `tok`, inside the
// workgroup.
//
// WHY THE TILE IS LOADED ONCE PER *SEQUENCE* AND NOT PER TOKEN (the prefill
// lever). CUDA's fused kernel is a decode kernel: T==1, so its one coalesced
// load/store pair covers one step. Running the SAME tile through the whole
// token range amortises that pair over the sequence, which is the difference
// between streaming the [Dv,Dk] state through memory 4x per token and touching
// it twice per PROMPT. Nothing else changes; the recurrence is untouched.
//
// TILE GEOMETRY, and why these numbers.
// Vulkan only GUARANTEES 16384 bytes of maxComputeSharedMemorySize, and this
// backend probes no limits on purpose (vt_common.glsl § VT_TG). The workgroup is
// VT_TG == 128 lanes, split BV rows x NW lanes-per-row, so BV*NW == 128 and both
// are powers of two (the row reduction halves). The budget picks the largest such
// BV:
//     BV=32 -> 32*(128+1) + 2*128 + 128 = 4512 floats = 18048 B  > 16384  REJECT
//     BV=16 -> 16*(128+1) + 2*128 + 128 = 2448 floats =  9792 B  ok
// so BV=16, NW=8. MAX_DK bounds the tile's compile-time extent; the host DECLINES
// to the reference tier for a wider Dk rather than reading past the array
// (vulkan_ops.cpp GdnPrefillKernel). Qwen3.6-27B's linear_key_head_dim is 128.
//
// NUMERIC TIER: NMSE, not bit-exact. Two divergences from the CPU reference, both
// deliberate. (1) GLSL `exp` is a ~3-ULP builtin, not std::exp — and the decay it
// produces multiplies the carried state EVERY step, so the difference compounds
// along the sequence rather than staying local. (2) each Dk contraction is a sum
// of NW partial sums instead of one sequential accumulation, so the reduction
// order is a tree. Same tier as every other reducing kernel in this backend.
#ifndef VT_GDN_RECURRENCE_GLSL
#define VT_GDN_RECURRENCE_GLSL

// TILE ROWS, a specialization constant so the occupancy sweep runs in ONE binary.
// Default 16 reproduces every earlier build exactly.
//
// WHY IT IS SWEEPABLE. Measured 2026-09-06 on the 27B prefill: the grid is
// n * Hv * ceil(Dv/BV) workgroups, which at one sequence is 1 * 48 * 8 = 384 --
// on a card with 188 SMs, and each workgroup then walks the sequence
// SEQUENTIALLY because the recurrence is data-dependent in t. Feeding more
// sequences shows what that costs: per-sequence time falls 0.7396 -> 0.5185 ->
// 0.4643 ms as the grid goes 384 -> 768 -> 1536 and then FLATTENS, so the
// machine fills near 1536 and a single-sequence prefill is paying about 1.74x
// for running a quarter of it.
//
// BV is the only term in that grid we control without touching the algorithm:
// halving it doubles the workgroups at the same n. It is not free -- BV * NW
// == VT_TG is fixed, so a smaller tile means a WIDER lane group per row, a
// deeper reduction tree (log2 NW) and fewer Dk columns per lane. Which side
// wins is a measurement, not a derivation.
layout(constant_id = 7) const uint VT_GDN_BV = 16u;
#define VT_GDN_NW (VT_TG / VT_GDN_BV)  // lanes cooperating on one row
#define VT_GDN_MAX_DK 128u // compile-time bound on the tile's row width

// The [BV,Dk] state slice, rows padded to Dk+1 to break the bank conflict a
// power-of-two stride would give every lane of a row (cuda_gdn.cu:2445 does the
// same, and for the same reason).
shared float vt_gdn_sbh[VT_GDN_BV * (VT_GDN_MAX_DK + 1u)];  // sized by the spec constant
shared float vt_gdn_bq[VT_GDN_MAX_DK];  // q' = q*scale, broadcast to every lane
shared float vt_gdn_bk[VT_GDN_MAX_DK];  // k, likewise
shared float vt_gdn_red[VT_TG];         // per-row partial sums

// Per-invocation geometry, filled by vt_gdn_setup(). Plain globals: in a compute
// shader these are private to the invocation.
uint vt_gdn_tid;    // lane in the workgroup
uint vt_gdn_vi;     // value-state row within the tile
uint vt_gdn_wk;     // Dk-slice owner within this row's lane group
uint vt_gdn_vrow;   // the tile row's absolute value-state row
uint vt_gdn_c0;     // this lane's Dk column slice, [c0, c1)
uint vt_gdn_c1;
uint vt_gdn_sdk;    // padded shared row stride, Dk+1
uint vt_gdn_rbase;  // vt_gdn_vi * vt_gdn_sdk

void vt_gdn_setup(uint vbase) {
  vt_gdn_tid = gl_LocalInvocationID.x;
  vt_gdn_vi = vt_gdn_tid / VT_GDN_NW;
  vt_gdn_wk = vt_gdn_tid % VT_GDN_NW;
  vt_gdn_vrow = vbase + vt_gdn_vi;
  vt_gdn_sdk = p.dk + 1u;
  vt_gdn_rbase = vt_gdn_vi * vt_gdn_sdk;
  // Partition [0, Dk) across the row's NW lanes (cuda_gdn.cu:2469-2471). When Dk
  // is small enough that a lane's slice starts past the end, c1 < c0 and the
  // lane's loops are simply empty — it still reaches every barrier.
  uint ck = (p.dk + VT_GDN_NW - 1u) / VT_GDN_NW;
  vt_gdn_c0 = vt_gdn_wk * ck;
  vt_gdn_c1 = min(vt_gdn_c0 + ck, p.dk);
}

// Sum one value row's NW partial sums and broadcast the total to its NW lanes.
// Shared-memory halving tree rather than a subgroup shuffle (cuda_gdn.cu:2485
// uses __shfl_xor_sync): subgroup ARITHMETIC is an optional Vulkan 1.1 feature
// and this backend probes no optional features (vt_common.glsl § VT_TG).
//
// THE LEADING BARRIER IS LOAD-BEARING, for the reason vt_common.glsl's
// vt_tg_sum states: this is called twice per token, so without it a lane racing
// into the next call would overwrite a slot another lane had not yet read.
float vt_gdn_rowsum(float v) {
  barrier();
  vt_gdn_red[vt_gdn_tid] = v;
  barrier();
  for (uint s = VT_GDN_NW / 2u; s > 0u; s >>= 1) {
    if (vt_gdn_wk < s) { vt_gdn_red[vt_gdn_tid] += vt_gdn_red[vt_gdn_tid + s]; }
    barrier();
  }
  return vt_gdn_red[vt_gdn_tid - vt_gdn_wk];  // lane 0 of this row's group
}

// Stage state[state_row][h_v][vbase .. vbase+BV)[0 .. Dk) into shared. The read
// is flat over the tile so consecutive lanes read consecutive words — the
// coalesced load cuda_gdn.cu:2464-2465 exists for. Rows past Dv are ZEROED, not
// skipped: their lanes stay live through every barrier below and compute on the
// zeros, and the store-back leaves them out.
void vt_gdn_load_tile(uint state_row, uint h_v, uint vbase) {
  uint base = (p.state_off >> 2) + ((state_row * p.hv + h_v) * p.dv + vbase) * p.dk;
  uint valid = min(p.dv - vbase, VT_GDN_BV);
  for (uint e = vt_gdn_tid; e < VT_GDN_BV * p.dk; e += VT_TG) {
    uint r = e / p.dk;
    uint c = e - r * p.dk;
    float s = 0.0;
    if (r < valid) { s = uintBitsToFloat(S.v[base + e]); }
    vt_gdn_sbh[r * vt_gdn_sdk + c] = s;
  }
  barrier();
}

void vt_gdn_store_tile(uint state_row, uint h_v, uint vbase) {
  barrier();
  uint base = (p.state_off >> 2) + ((state_row * p.hv + h_v) * p.dv + vbase) * p.dk;
  uint valid = min(p.dv - vbase, VT_GDN_BV);
  for (uint e = vt_gdn_tid; e < valid * p.dk; e += VT_TG) {
    uint r = e / p.dk;
    uint c = e - r * p.dk;
    S.v[base + e] = floatBitsToUint(vt_gdn_sbh[r * vt_gdn_sdk + c]);
  }
}

// ONE token of the recurrence for this workgroup's (value-head, value-tile).
// cpu_ops.cpp:1285-1310, statement for statement; only the [0,Dk) loops are
// split across the row's NW lanes and closed with a reduction.
void vt_gdn_step(uint tok, uint h_v, uint h_k) {
  // q' and k for this (token, key-head), broadcast to every lane. Both loops are
  // over the FULL workgroup, so the barriers around them are uniformly reached.
  barrier();
  uint qkbase = (tok * p.hk + h_k) * p.dk;
  for (uint e = vt_gdn_tid; e < p.dk; e += VT_TG) {
    vt_gdn_bk[e] = VT_LOAD(K32, K16, VT_PC_QKV_DT, p.k_off, qkbase + e);
    vt_gdn_bq[e] = VT_LOAD(Q32, Q16, VT_PC_QKV_DT, p.q_off, qkbase + e) * p.scale;
  }
  barrier();

  uint hidx = tok * p.hv + h_v;
  float decay = exp(uintBitsToFloat(G.v[(p.g_off >> 2) + hidx]));
  float beta_t = uintBitsToFloat(BETA.v[(p.beta_off >> 2) + hidx]);

  // (S * exp(g)) @ k, fused with the decay pass (cpu_ops.cpp:1292-1301).
  float pdot = 0.0;
  for (uint c = vt_gdn_c0; c < vt_gdn_c1; ++c) {
    float s = vt_gdn_sbh[vt_gdn_rbase + c] * decay;
    vt_gdn_sbh[vt_gdn_rbase + c] = s;
    pdot += s * vt_gdn_bk[c];
  }
  float dot = vt_gdn_rowsum(pdot);

  float vv = 0.0;
  if (vt_gdn_vrow < p.dv) {
    vv = VT_LOAD(V32, V16, VT_PC_QKV_DT, p.v_off, hidx * p.dv + vt_gdn_vrow);
  }
  float vp = (vv - dot) * beta_t;

  // (S + outer(v',k)) @ q', fused with the rank-1 update (cpu_ops.cpp:1302-1310).
  float po = 0.0;
  for (uint c = vt_gdn_c0; c < vt_gdn_c1; ++c) {
    float s = vt_gdn_sbh[vt_gdn_rbase + c] + vp * vt_gdn_bk[c];
    vt_gdn_sbh[vt_gdn_rbase + c] = s;
    po += s * vt_gdn_bq[c];
  }
  float o = vt_gdn_rowsum(po);
  if (vt_gdn_vrow < p.dv && vt_gdn_wk == 0u) {
    VT_STORE(O32, O16, VT_PC_OUT_DT, p.out_off, hidx * p.dv + vt_gdn_vrow, o);
  }
}

#endif  // VT_GDN_RECURRENCE_GLSL
