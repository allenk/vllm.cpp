// Vulkan backend — op kernels: descriptor binding + dispatch of the committed
// SPIR-V in vulkan_spirv.h, plus the `RegisterOp` table entries. BACKEND-VULKAN,
// W0 skeleton. Self-registering TU, copying the `src/vt/cpu/cpu_ops.cpp`
// Registrar idiom exactly, so adding this backend edited NO existing kernel file.
//
// WHAT THIS TU COVERS (deliberately a SEAM PROOF, not a model):
//   kAdd, kRelu, kSiluAndMul, kCastBf16, kCastF32, kLayerNorm, kRmsNorm and the
//   single kFusedChain registration that inherits the portable fusion catalog.
// That set spans every structural class the seam has to get right: flat
// elementwise, a rank-1 broadcast, a dtype-converting copy, TWO different row
// reductions, an optional in-place residual stream, and the recipe interpreter.
// It matches the Metal skeleton's set exactly, so the two backends are directly
// comparable through tests/vt/test_backend_cross_device.cpp.
//
// SINCE THEN this TU has grown the dense path (both GEMM orientations plus the
// decode GEMV and coopmat tactics), the attention block (paged attention, the KV
// write, the QKV split, the rotary apply), the two ends of the model (embedding
// and greedy argmax), and — this row, BACKEND-VULKAN-GDN — six of the GDN /
// conv1d glue ops that Qwen3.6-27B needs.
//
// WHAT IS STILL NOT REGISTERED: the quant tier, MoE, the sampler beyond greedy
// argmax, the GDN recurrences themselves, and the ops listed in the
// BACKEND-VULKAN-GDN block comment further down. None of them THROW any more:
// since the portable reference tier landed, a missed GetOp on this
// unified-memory device installs the CPU kernel as a priority -1000 provider, so
// an unregistered op is CORRECT AND SLOW rather than fatal.
//
// BINDING MODEL: every tensor operand occupies TWO consecutive descriptor
// bindings onto the SAME VkBuffer — a uint32_t view and a uint16_t view — and
// its BYTE OFFSET travels in the push constants. See
// src/vt/vulkan/shaders/vt_common.glsl § STORAGE MODEL for why.
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <algorithm>
#include <cstdio>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "vulkan_buffers.h"
#include "vulkan_context.h"
#include "vt/ops.h"

// --- coopmat shape histogram (VT_VULKAN_SHAPE_STATS=1) ----------------------
namespace {
struct CoopShape { long long m, k, n; uint32_t tiles; };
std::mutex g_shape_mu;
std::map<std::string, uint64_t> g_shape_counts;
bool ShapeStatsOn() {
  static const bool on = std::getenv("VT_VULKAN_SHAPE_STATS") != nullptr;
  return on;
}
// `tactic` is what makes this histogram readable now that there is more than one
// coopmat path. External review 2026-09-06 (F2): the single call site sat AFTER
// the workgroup branch had already returned, so VT_VULKAN_SHAPE_STATS reported
// only the shapes that FELL THROUGH to the subgroup kernel -- and since the
// workgroup kernel became the main prefill path, the instrument was blind to
// exactly the set it exists to describe. A 1026xKxN prefill shape was simply
// absent, and any next-shape priority read off it would have been read off the
// leftovers.
//
// Recorded per tactic rather than merged: "which shapes are hot" and "which
// kernel got them" are the same question here, because the tactic is chosen FROM
// the shape. The label stays out of the dispatch name, so the pipeline cache is
// still keyed by specialization and not split per shape.
void RecordCoopmatShape(const char* tactic, long long m, long long k, long long n,
                        uint32_t tiles) {
  if (!ShapeStatsOn()) return;
  static bool reg = false;
  char buf[160];
  std::snprintf(buf, sizeof(buf), "%-9s m=%-6lld k=%-6lld n=%-8lld tiles=%u", tactic, m, k,
                n, tiles);
  std::lock_guard<std::mutex> g(g_shape_mu);
  if (!reg) {
    reg = true;
    std::atexit([] {
      std::lock_guard<std::mutex> g2(g_shape_mu);
      uint64_t tot = 0;
      for (const auto& kv : g_shape_counts) tot += kv.second;
      std::vector<std::pair<std::string, uint64_t>> v(g_shape_counts.begin(),
                                                      g_shape_counts.end());
      std::sort(v.begin(), v.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
      std::fprintf(stderr, "[vt vulkan] COOPMAT SHAPES  distinct=%zu  calls=%llu\n",
                   v.size(), static_cast<unsigned long long>(tot));
      for (const auto& kv : v) {
        std::fprintf(stderr, "[vt vulkan]   %-46s %8llu  %5.1f%%\n", kv.first.c_str(),
                     static_cast<unsigned long long>(kv.second),
                     tot ? 100.0 * kv.second / tot : 0.0);
      }
      std::fflush(stderr);
    });
  }
  ++g_shape_counts[buf];
}
}  // namespace


namespace vt::vulkan {
namespace {

// Gated on the same flag as the dispatch profile; costs nothing when unset.
const bool kCoopMatWhy = [] {
  const char* v = std::getenv("VT_VULKAN_DISPATCH_STATS");
  return v != nullptr && std::strcmp(v, "0") != 0;
}();

// Storage dtype -> the shader-side code (vt_common.glsl VT_DT_*).
uint32_t DtypeCode(DType d) {
  switch (d) {
    case DType::kF32: return 0;
    case DType::kF16: return 1;
    case DType::kBF16: return 2;
    default: break;
  }
  VT_CHECK(false, "vulkan: unsupported storage dtype (f32/f16/bf16 only in the W0 skeleton)");
  return 0;
}

// Collects the (buffer, byte-offset) pairs for a dispatch. Each Add() appends
// the SAME buffer twice — bindings 2k and 2k+1, the u32 and u16 views — and
// returns the byte offset for the push-constant block.
class Binder {
 public:
  uint32_t Add(const Tensor& t, const char* what) {
    Resolved r = Resolve(t.data, what);
    buffers_.push_back(r.buffer);
    buffers_.push_back(r.buffer);
    // f32 access indexes a uint32_t[] view, so a f32 operand's byte offset must
    // be 4-byte aligned; 16-bit access only needs 2. Tensor storage always
    // satisfies this (allocations are 64-byte aligned and views advance by whole
    // elements), but a violation would silently read shifted data.
    VT_CHECK(t.dtype != DType::kF32 || r.offset % 4 == 0,
             std::string("vulkan: ") + what + " has a byte offset that is not 4-byte aligned");
    VT_CHECK(r.offset % 2 == 0,
             std::string("vulkan: ") + what + " has an odd byte offset");
    return r.offset;
  }
  // A raw buffer bound ONCE (no 16-bit view): the fused-chain step list.
  void AddRaw(void* buffer) { buffers_.push_back(buffer); }

  // ONE MORE VIEW of an operand already added, for a shader that reads the same
  // memory at a THIRD width (vt_matmul_vec's 64-bit packed loads). Deliberately
  // separate from Add: this appends a SINGLE binding, and it must land after
  // every operand's u32/u16 pair, so the caller is choosing the descriptor index
  // rather than getting one implicitly.
  uint32_t AddAlias(const Tensor& t, const char* what) {
    Resolved r = Resolve(t.data, what);
    buffers_.push_back(r.buffer);
    return r.offset;
  }

  // A tensor bound through the uint32_t view ONLY, for shaders that declare a
  // single binding per operand because the operand is integer (embedding ids,
  // sampler token ids) or f32-by-contract (logits). Binding the unused 16-bit
  // view would need the shader to declare it too, and a descriptor a shader does
  // not declare must not be written.
  uint32_t AddU32Only(const Tensor& t, const char* what) {
    Resolved r = Resolve(t.data, what);
    buffers_.push_back(r.buffer);
    VT_CHECK(r.offset % 4 == 0,
             std::string("vulkan: ") + what + " has a byte offset that is not 4-byte aligned");
    return r.offset;
  }

  // A tensor bound through the uint32_t view ONLY, for BYTE-granular reads: an
  // i8 operand (GDN's has_initial_state) may legitimately start at any byte, so
  // AddU32Only's 4-byte assertion would reject a valid tensor. The shader
  // recovers the byte with a shift, which is exact because every buffer is bound
  // WHOLE at offset 0 and the returned offset is therefore a plain byte address
  // into it (vt_common.glsl § STORAGE MODEL). Deliberately NOT VK_KHR_8bit_storage:
  // this backend does not probe for it, and requiring it would narrow the set of
  // devices the backend registers on for the sake of one boolean array.
  uint32_t AddByteView(const Tensor& t, const char* what) {
    Resolved r = Resolve(t.data, what);
    buffers_.push_back(r.buffer);
    return r.offset;
  }

  // A tensor bound through the uint16_t view ONLY, the mirror of AddU32Only for
  // an operand every consumer reads at 16 bits: an fp16 scale vector, or the
  // EXL3 trellis, whose i8 BYTES are read as the uint32 pairs exl3_dq.cuh reads
  // and never as f32. Binding the unused 32-bit view would need the shader to
  // declare it too, and glslang strips a block nothing references — which the
  // generator then reports as a binding HOLE rather than passing silently.
  // The 2-byte assertion is Add()'s, and it is the real constraint here.
  uint32_t AddU16Only(const Tensor& t, const char* what) {
    Resolved r = Resolve(t.data, what);
    buffers_.push_back(r.buffer);
    VT_CHECK(r.offset % 2 == 0,
             std::string("vulkan: ") + what + " has an odd byte offset");
    return r.offset;
  }

  const void* const* data() const { return buffers_.data(); }
  uint32_t count() const { return static_cast<uint32_t>(buffers_.size()); }

 private:
  std::vector<const void*> buffers_;
};

// ---- Host mirrors of the shaders' push-constant blocks. Field order and types
// must match the GLSL declarations EXACTLY. GLSL `uint`/`float` are 4-byte with
// 4-byte alignment and every block below is a run of 4-byte scalars, so the std430
// push-constant layout coincides with the C++ layout with no padding surprises.
struct AddParams {
  uint32_t n, d, a_dt, b_dt, out_dt, bcast, a_off, b_off, out_off;
};
struct UnaryParams {
  uint32_t n, a_dt, out_dt, a_off, out_off;
};
// vt_cast carries its dtype pair in specialization constants instead, so its
// push block is only the shape and the two offsets.
struct CastParams {
  uint32_t n, a_off, out_off;
};
struct MatmulParams {
  uint32_t m, n, k, a_off, b_off, out_off;
};
struct EmbeddingParams {
  uint32_t t, h, table_off, ids_off, out_off;
};
struct ArgmaxParams {
  uint32_t n, v, logits_off, out_off;
};
struct QkvSplitParams {
  uint32_t tokens, q_dim, k_dim, v_dim, src_off, q_off, k_off, v_off;
};
struct RopeFromCacheParams {
  uint32_t tokens, half_dim, rotary_dim, hq, hk;
  uint32_t q_s0, q_s1, k_s0, k_s1;
  uint32_t q_off, k_off, c_off, p_off;
};
struct ReshapeAndCacheParams {
  uint32_t num_slots, n_elems, block_size;
  uint32_t k_blk, k_pg, v_blk, v_pg;
  uint32_t k_tok, v_tok;
  uint32_t k_off, v_off, kc_off, vc_off, sm_off;
};
struct PagedAttnParams {
  uint32_t total_q, hq, d, block_size, qpk, num_reqs;
  uint32_t causal;
  int32_t window_left, window_right;
  uint32_t kc_blk, kc_pg, kc_hd;
  uint32_t vc_blk, vc_pg, vc_hd;
  uint32_t bt_row, bt_col;
  uint32_t q_off, k_off, v_off, out_off;
  uint32_t bt_off, sl_off, qsl_off;
  float scale;
  float softcap;
};
// Pass 2 of the split-K attention. Small on purpose: the merge reads only the
  // partials and writes the row.
struct PagedAttnMergeParams {
  uint32_t total_q, hq, d, par_off, out_off;
};
struct SiluMulParams {
  uint32_t t, d, x_dt, out_dt, x_off, out_off;
};
// The SPLIT form of the same activation: gate and up in two buffers rather than
// as the halves of one. See vt_moe_silu_mul.comp for why both exist.
struct MoeSiluMulParams {
  uint32_t n, gate_dt, up_dt, out_dt, gate_off, up_off, out_off;
};
struct RmsParams {
  uint32_t t, h, x_dt, w_dt, out_dt, res_dt, has_res, gemma, x_off, w_off, out_off, res_off;
  float eps;
};
struct LayerNormParams {
  uint32_t rows, d, x_dt, w_dt, b_dt, out_dt, has_w, has_b, x_off, w_off, b_off, out_off;
  float eps;
};
struct FcParams {
  uint32_t t, h, nsteps, x_dt, w_dt, res_dt, out_dt, x_off, w_off, res_off, out_off;
  float eps;
};
// --- The GDN / conv1d family (BACKEND-VULKAN-GDN). Same rule as above: field
// order and types must match the GLSL push-constant blocks EXACTLY.
struct SigmoidGateParams {
  uint32_t n, a_off, g_off, o_off;
};
struct RmsNormGatedParams {
  uint32_t rows, d, x_dt, z_dt, w_dt, out_dt, sigmoid_gate, gate_group, gate_outer;
  uint32_t x_off, z_off, w_off, out_off;
  float eps;
};
struct GdnStateGatherParams {
  uint32_t rows, work_row, work_inner, cache_inner, cache_row, n_cache_rows;
  uint32_t work_dt, cache_dt, his_mode;
  uint32_t work_off, cache_off, idx_off, his_off;
};
struct GdnStateScatterParams {
  uint32_t rows, work_row, work_inner, cache_inner, cache_row, n_cache_rows;
  uint32_t work_dt, cache_dt;
  uint32_t cache_off, work_off, idx_off;
};
struct ConvUpdateParams {
  uint32_t batch, c_dim, k, width, state_len, x_rs, n_state_rows;
  uint32_t has_bias, has_idx, silu;
  uint32_t out_dt, x_dt, w_dt, bias_dt, st_dt;
  uint32_t out_off, x_off, w_off, bias_off, st_off, idx_off;
};
struct GdnPostConvParams {
  uint32_t t, hk, dk, hv, dv, key_dim, value_dim, conv_dim, a_rs, b_rs;
  uint32_t conv_off, q_off, k_off, v_off, a_off, b_off;
  uint32_t g_off, beta_off, alog_off, dtb_off;
  float eps;
};
// The fused full-attention preamble (BACKEND-VULKAN-QKNORM).
struct AttnQkNormRopeGateParams {
  uint32_t hq, hkv, dh, rot, half_dim, qgate_rs, kf_rs, gemma;
  uint32_t qg_off, kf_off, qo_off, ko_off, go_off, qn_off, kn_off, cs_off;
  float eps;
};
// ONE block for BOTH recurrences: vt_gdn_prefill and vt_gdn_decode share their
// step body through an include, so they must also share their push layout. The
// prefill shader ignores has_idx / n_state_rows (they are the decode state-row
// indirection) rather than each op carrying a block that drifts from the other.
struct GdnRecurrenceParams {
  uint32_t hk, dk, hv, dv, nv, ratio, has_idx, n_state_rows;
  uint32_t q_off, k_off, v_off, out_off, g_off, beta_off, state_off, meta_off;
  float scale;
};

// BACKEND-VULKAN-EXL3 (#2530). vt_exl3_had.comp and vt_exl3_gemm.comp.
struct Exl3HadParams {
  uint32_t nblocks, blocks_per_row, cols, total_groups, has_pre, has_post;
  uint32_t in_off, out_off, pre_off, post_off;
  float r_scale;
};
struct Exl3GemmParams {
  uint32_t m, k, n, bits, codebook, tiles_n, raw_off, ah_off, tr_off;
};

// Vulkan only GUARANTEES 128 bytes of push-constant space (maxPushConstantsSize);
// staying inside it is what keeps this backend portable without a probe.
static_assert(sizeof(RmsParams) <= 128, "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(LayerNormParams) <= 128, "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(FcParams) <= 128, "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(PagedAttnParams) <= 128,
              "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(ConvUpdateParams) <= 128,
              "push constants must fit the guaranteed 128 bytes");
// The widest block in the backend at 84 bytes: the fused post-conv carries ten
// operand offsets. If it ever needs an eleventh, the step list has to move to the
// scratch buffer the way vt_fused_chain's does.
static_assert(sizeof(GdnPostConvParams) <= 128,
              "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(RmsNormGatedParams) <= 128,
              "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(GdnStateGatherParams) <= 128,
              "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(GdnRecurrenceParams) <= 128,
              "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(AttnQkNormRopeGateParams) <= 128,
              "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(Exl3HadParams) <= 128, "push constants must fit the guaranteed 128 bytes");
static_assert(sizeof(Exl3GemmParams) <= 128, "push constants must fit the guaranteed 128 bytes");

template <typename P>
void Go(const char* name, const Binder& b, const P& p, uint32_t groups,
        const uint32_t* spec = nullptr, uint32_t spec_count = 0,
        const std::string& stats_label = std::string()) {
  VulkanContext::Get().Dispatch(name, b.data(), b.count(), &p, sizeof(P), groups, spec,
                                spec_count, stats_label);
}

// ---------------------------------------------------------------------------
// Kernels. Every argument was already validated by the vt:: wrapper in
// src/vt/ops.cpp before GetOp dispatched here, so these only translate.
// ---------------------------------------------------------------------------

// cpu_layernorm.cpp:87-99 AddKernel.
void AddKernel(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  const int64_t n = a.Numel();
  const int64_t d = a.rank == 0 ? 1 : a.shape[a.rank - 1];
  const bool bcast = b.rank == 1 && a.rank != 1;
  Binder bind;
  const uint32_t a_off = bind.Add(a, "add: a");
  const uint32_t b_off = bind.Add(b, "add: b");
  const uint32_t out_off = bind.Add(out, "add: out");
  AddParams p{static_cast<uint32_t>(n), static_cast<uint32_t>(d),
              DtypeCode(a.dtype),      DtypeCode(b.dtype),
              DtypeCode(out.dtype),    bcast ? 1u : 0u,
              a_off,                   b_off,
              out_off};
  Go("vt_add", bind, p, FlatGroupCount(n));
}

// cpu_layernorm.cpp:75-85 ReluKernel.
void ReluKernel(Queue&, Tensor& out, const Tensor& x) {
  const int64_t n = x.Numel();
  Binder bind;
  const uint32_t x_off = bind.Add(x, "relu: x");
  const uint32_t out_off = bind.Add(out, "relu: out");
  UnaryParams p{static_cast<uint32_t>(n), DtypeCode(x.dtype), DtypeCode(out.dtype), x_off,
                out_off};
  Go("vt_relu", bind, p, FlatGroupCount(n));
}

// cpu_ops.cpp:1436-1451 CastBf16Kernel / CastF32Kernel — one shader serves both
// (the CPU pair is likewise the same LoadF32/StoreF32 body twice).
void CastKernel(Queue&, Tensor& out, const Tensor& in) {
  const int64_t n = out.Numel();
  Binder bind;
  const uint32_t in_off = bind.Add(in, "cast: in");
  const uint32_t out_off = bind.Add(out, "cast: out");
  // The dtype pair rides SPECIALIZATION CONSTANTS rather than push constants, so
  // the per-element dtype branch is folded away at pipeline creation and each
  // (src, dst) pair is its own cached pipeline. Ascending constantID order, which
  // is what GetPipeline binds against the module's declared SpecIds.
  const uint32_t spec[2] = {DtypeCode(in.dtype), DtypeCode(out.dtype)};
  CastParams p{static_cast<uint32_t>(n), in_off, out_off};
  Go("vt_cast", bind, p, FlatGroupCount(n), spec, 2);
}

// ─── EXL3, BACKEND-VULKAN-EXL3 (#2530) ───────────────────────────────────────
//
// The fused chain, step for step as `Exl3GemmKernelCpu` runs it:
//   1. A_had = had_r_128(A, pre_scale = suh)
//   2. C_raw = A_had @ reconstruct(trellis)        [f32 accumulation]
//   3. C     = had_r_128(C_raw, post_scale = svh)
// Argument validation (shapes, dtypes, contiguity, the 16/128 multiples) is done
// once in `vt::Exl3Gemm` for every backend and is not repeated here. What IS
// checked is what selects device code: `bits` and `codebook` reach the shader as
// values and an out-of-range one would decode SILENTLY, because a shader cannot
// throw.
//
// See src/vt/vulkan/shaders/vt_exl3_gemm.comp for why the donor is the portable
// CPU reference and not cuda_exl3.cu, and .agents/specs/backend-vulkan-exl3.md
// for the byte-equality contract this arm carries instead of the CUDA arm's
// 1.0e-3 RMS bound.

// 1/sqrt(128) spelled as upstream spells it (hadamard.cu:107) and as
// cpu_exl3_kernels.cpp spells it. Kept as the literal and NEVER recomputed: a
// recomputed 1/sqrt(128) can differ in the last f32 bit, and that bit is the
// difference between this arm's byte gate passing and failing.
constexpr float kExl3InvSqrt128 = 0.088388347648f;

// The f32 [m, n] staging buffer between steps 2 and 3. The CPU arm holds it in a
// std::vector; a device needs the same bytes somewhere. GROW-ONLY and
// process-wide, which is the discipline rocm_exl3.hip's EnsureRawScratch
// documents.
//
// NAMED AS A COST rather than hidden. A fused kernel would not need it. A fused
// kernel is the later speed row; this one buys the end of the reference tier.
//
// THE FLUSH BEFORE A GROW IS LOAD-BEARING. Dispatches are BATCHED into an open
// command buffer, so the previous buffer may still be bound by work that has not
// executed. Freeing it under an open batch is a use-after-free with no error and
// no crash — the same hazard Backend::Copy's FlushIfBatchTouches exists for.
float* EnsureExl3RawScratch(size_t need) {
  static std::mutex mu;
  static void* base = nullptr;
  static size_t bytes = 0;
  std::lock_guard<std::mutex> lock(mu);
  if (need <= bytes) return static_cast<float*>(base);
  void* buffer = nullptr;
  void* memory = nullptr;
  void* fresh = VulkanContext::Get().AllocBuffer(need, &buffer, &memory);
  RegisterAllocation(fresh, need, buffer, memory);
  if (base != nullptr) {
    VulkanContext::Get().FlushBatch("exl3 raw scratch grow");
    void* old_buffer = nullptr;
    void* old_memory = nullptr;
    if (UnregisterAllocation(base, &old_buffer, &old_memory))
      VulkanContext::Get().FreeBuffer(old_buffer, old_memory);
  }
  base = fresh;
  bytes = need;
  return static_cast<float*>(base);
}

// One `had_r_128` dispatch. `pre` and `post` are optional and at most one is set,
// which is upstream's own instantiation (hadamard.cu:112-172) and what
// `vt::Exl3HadR128` checks; an absent one is bound aliasing `in`, because a
// descriptor the shader statically uses must be valid even on the path that
// never reads it.
void Exl3HadDispatch(void* out, DType out_dtype, const void* in, DType in_dtype,
                     const Tensor* pre, const Tensor* post, const Device& device, float r_scale,
                     int64_t rows, int64_t cols) {
  const int64_t blocks_per_row = cols / 128;
  const int64_t nblocks = rows * blocks_per_row;
  if (nblocks <= 0) return;
  const int64_t total_groups = (nblocks + 3) / 4;

  Tensor tin = Tensor::Contiguous(const_cast<void*>(in), in_dtype, device, {rows, cols});
  Tensor tout = Tensor::Contiguous(out, out_dtype, device, {rows, cols});
  Binder bind;
  const uint32_t in_off = bind.Add(tin, "exl3_had: in");
  const uint32_t out_off = bind.Add(tout, "exl3_had: out");
  const uint32_t pre_off =
      pre != nullptr ? bind.AddU16Only(*pre, "exl3_had: pre_scale") : bind.AddU16Only(tin, "exl3_had: in");
  const uint32_t post_off = post != nullptr ? bind.AddU16Only(*post, "exl3_had: post_scale")
                                            : bind.AddU16Only(tin, "exl3_had: in");
  const uint32_t spec[2] = {in_dtype == DType::kF16 ? 1u : 0u,
                            out_dtype == DType::kF16 ? 1u : 0u};
  Exl3HadParams p{static_cast<uint32_t>(nblocks),
                  static_cast<uint32_t>(blocks_per_row),
                  static_cast<uint32_t>(cols),
                  static_cast<uint32_t>(total_groups),
                  pre != nullptr ? 1u : 0u,
                  post != nullptr ? 1u : 0u,
                  in_off,
                  out_off,
                  pre_off,
                  post_off,
                  r_scale};
  // The shader carries a grid-stride loop, so the workgroup count is a
  // PERFORMANCE choice and never a correctness one. 65535 is the Vulkan-
  // guaranteed maxComputeWorkGroupCount[0], so this launch is valid on any
  // conformant device no matter how large the tensor is.
  const uint32_t groups = static_cast<uint32_t>(total_groups > 65535 ? 65535 : total_groups);
  Go("vt_exl3_had", bind, p, groups, spec, 2);
}

void Exl3GemmKernelVulkan(Queue& q, Tensor& c, const Tensor& a, const Tensor& trellis,
                          const Tensor& suh, const Tensor& svh, Tensor& a_had,
                          const Exl3GemmArgs& args) {
  const int64_t m = a.shape[0];
  const int64_t k = a.shape[1];
  const int64_t n = c.shape[1];
  if (m == 0 || k == 0 || n == 0) return;
  VT_CHECK(args.bits >= 1 && args.bits <= 8,
           "vt vulkan exl3: bits must be in [1, 8]; got " + std::to_string(args.bits));
  VT_CHECK(args.codebook >= 0 && args.codebook <= 2,
           "vt vulkan exl3: codebook " + std::to_string(args.codebook) +
               " is not implemented (0 == 3INST, 1 == MCG, 2 == mul1). Upstream defines "
               "no other value: `decode_3inst<cb>` (codebook.cuh:56-90) has arms for 0, "
               "1 and 2 and falls off the end for anything else.");

  // 1. the input transform, into the caller's scratch (which may alias A).
  Exl3HadDispatch(a_had.data, a_had.dtype, a.data, a.dtype, &suh, nullptr, q.device,
                  kExl3InvSqrt128, m, k);

  // 2. the matmul against the decoded trellis, f32 accumulators.
  float* raw = EnsureExl3RawScratch(static_cast<size_t>(m) * static_cast<size_t>(n) *
                                    sizeof(float));
  Tensor traw = Tensor::Contiguous(raw, DType::kF32, q.device, {m, n});
  Binder bind;
  const uint32_t raw_off = bind.AddU32Only(traw, "exl3_gemm: raw");
  const uint32_t ah_off = bind.AddU16Only(a_had, "exl3_gemm: a_had");
  const uint32_t tr_off = bind.AddU16Only(trellis, "exl3_gemm: trellis");
  Exl3GemmParams p{static_cast<uint32_t>(m),
                   static_cast<uint32_t>(k),
                   static_cast<uint32_t>(n),
                   static_cast<uint32_t>(args.bits),
                   static_cast<uint32_t>(args.codebook),
                   static_cast<uint32_t>(n / 16),
                   raw_off,
                   ah_off,
                   tr_off};
  // The grid is FLATTENED because Dispatch takes group_count_x only: one
  // workgroup per (16-column output tile, 8-row block), and the shader
  // decomposes the id with the same `tiles_n` it is handed here.
  const int64_t groups = (n / 16) * ((m + 7) / 8);
  Go("vt_exl3_gemm", bind, p, static_cast<uint32_t>(groups));

  // 3. the output transform — had_ff for an f32 C, had_fh for an fp16 one, the
  // same two arms the CPU reference picks between off `c.dtype`.
  Exl3HadDispatch(c.data, c.dtype, raw, DType::kF32, nullptr, &svh, q.device, kExl3InvSqrt128, m,
                  n);
}

// cpu_ops.cpp:252-264 SiluAndMulKernel.
void SiluAndMulKernel(Queue&, Tensor& out, const Tensor& x) {
  const int64_t t = x.shape[0], d = x.shape[1] / 2;
  Binder bind;
  const uint32_t x_off = bind.Add(x, "silu_and_mul: x");
  const uint32_t out_off = bind.Add(out, "silu_and_mul: out");
  SiluMulParams p{static_cast<uint32_t>(t), static_cast<uint32_t>(d), DtypeCode(x.dtype),
                  DtypeCode(out.dtype), x_off, out_off};
  Go("vt_silu_and_mul", bind, p, FlatGroupCount(t * d));
}

// cpu_ops.cpp:452-462 MoeSiluMulKernel -- out[i] = silu(gate[i]) * up[i] with
// gate and up as SEPARATE contiguous tensors (the GGUF weight layout). Without
// this registration the whole op fell to the portable CPU tier, which cost the
// GGUF path 44x end-to-end; the shader header carries the measurement.
void MoeSiluMulKernel(Queue&, Tensor& out, const Tensor& gate, const Tensor& up) {
  const int64_t n = out.Numel();
  Binder bind;
  const uint32_t gate_off = bind.Add(gate, "moe_silu_mul: gate");
  const uint32_t up_off = bind.Add(up, "moe_silu_mul: up");
  const uint32_t out_off = bind.Add(out, "moe_silu_mul: out");
  MoeSiluMulParams p{static_cast<uint32_t>(n), DtypeCode(gate.dtype), DtypeCode(up.dtype),
                     DtypeCode(out.dtype),     gate_off,              up_off,
                     out_off};
  Go("vt_moe_silu_mul", bind, p, FlatGroupCount(n));
}

// WHICH RmsNorm MODULE THIS DISPATCH USES.
//
// `vt_rms_norm` is the portable 128-invocation module; `vt_rms_norm_wide` is the
// same body at 1024 invocations with a subgroup reduction. The choice is a
// DEVICE CAPABILITY question (VulkanContext::wide_reduce), not a shape question:
// the wide module is never wrong, only unavailable.
//
// VulkanContext::rms_norm_override() is the A/B lever over that decision
// (VT_VULKAN_RMSNORM=base|wide, or the setter the unit gate uses); it exists so
// the two arms are switchable inside ONE binary, because a cross-BUILD comparison
// is the shape that produced a false 1.2x reading for the subgroup tactic earlier
// in this campaign (see vulkan_context.cpp § kRingDepth).
const char* RmsNormShader() {
  const VulkanContext& ctx = VulkanContext::Get();
  const int forced = ctx.rms_norm_override();
  if (forced < 0) return "vt_rms_norm";
  if (forced > 0 || ctx.wide_reduce()) {
    VT_CHECK(ctx.wide_reduce(),
             "vulkan: the wide RmsNorm module was forced but this device does not "
             "support 1024-invocation workgroups with compute subgroup arithmetic");
    return "vt_rms_norm_wide";
  }
  return "vt_rms_norm";
}

// cpu_ops.cpp:225-250 RmsNormKernel. One workgroup per token row.
void RmsNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                   const RmsNormArgs& args, Tensor* residual) {
  const int64_t t = x.shape[0], h = x.shape[1];
  Binder bind;
  const uint32_t x_off = bind.Add(x, "rmsnorm: x");
  const uint32_t w_off = bind.Add(w, "rmsnorm: weight");
  const uint32_t out_off = bind.Add(out, "rmsnorm: out");
  // Bindings 6/7 are always written: a descriptor a shader statically uses must
  // be valid even on the code path that never reads it. With has_res == 0 they
  // alias `out` and are dead.
  const uint32_t res_off =
      residual != nullptr ? bind.Add(*residual, "rmsnorm: residual") : bind.Add(out, "rmsnorm: out");
  RmsParams p{static_cast<uint32_t>(t),
              static_cast<uint32_t>(h),
              DtypeCode(x.dtype),
              DtypeCode(w.dtype),
              DtypeCode(out.dtype),
              residual != nullptr ? DtypeCode(residual->dtype) : 0u,
              residual != nullptr ? 1u : 0u,
              args.gemma ? 1u : 0u,
              x_off,
              w_off,
              out_off,
              res_off,
              args.eps};
  Go(RmsNormShader(), bind, p, static_cast<uint32_t>(t));
}

// cpu_layernorm.cpp:49-73 LayerNormKernel.
void LayerNormKernel(Queue&, Tensor& out, const Tensor& x, const Tensor* weight,
                     const Tensor* bias, const LayerNormArgs& args) {
  const int64_t d = x.shape[x.rank - 1];
  const int64_t rows = d == 0 ? 0 : x.Numel() / d;
  Binder bind;
  const uint32_t x_off = bind.Add(x, "layer_norm: x");
  const uint32_t w_off =
      weight != nullptr ? bind.Add(*weight, "layer_norm: weight") : bind.Add(x, "layer_norm: x");
  const uint32_t b_off =
      bias != nullptr ? bind.Add(*bias, "layer_norm: bias") : bind.Add(x, "layer_norm: x");
  const uint32_t out_off = bind.Add(out, "layer_norm: out");
  LayerNormParams p{static_cast<uint32_t>(rows),
                    static_cast<uint32_t>(d),
                    DtypeCode(x.dtype),
                    weight != nullptr ? DtypeCode(weight->dtype) : 0u,
                    bias != nullptr ? DtypeCode(bias->dtype) : 0u,
                    DtypeCode(out.dtype),
                    weight != nullptr ? 1u : 0u,
                    bias != nullptr ? 1u : 0u,
                    x_off,
                    w_off,
                    b_off,
                    out_off,
                    args.eps};
  Go("vt_layer_norm", bind, p, static_cast<uint32_t>(rows));
}

// cpu_ops.cpp:1649-1702 FusedChainInterpKernel — the Tier-1 interpreter. ONE
// registration; every Tier-1-able recipe in include/vt/recipes.h realizes
// through it, and every non-Tier-1 recipe realizes through the device-agnostic
// Tier-0 composite in src/vt/ops.cpp, which re-enters this backend's standalone
// ops. That is the whole "2 lines -> all 10 recipes" property the spike claims.
void FusedChainKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& weight,
                      Tensor* residual, const FusedRecipe& r, float eps) {
  const int64_t t = x.shape[0], h = x.shape[1];
  VT_CHECK(r.n >= 1 && r.n <= kMaxFusedSteps, "vulkan fused_chain: bad step count");

  // Words per step, matching VT_STEP_WORDS in vt_fused_chain.comp.
  constexpr uint32_t kStepWords = 5;
  std::vector<uint32_t> steps(static_cast<size_t>(r.n) * kStepWords, 0u);
  for (int s = 0; s < r.n; ++s) {
    const FStep& st = r.steps[s];
    uint32_t op = 0;
    switch (st.op) {
      case FOp::kAdd: op = 0; break;
      case FOp::kMul: op = 1; break;
      case FOp::kSilu: op = 2; break;
      case FOp::kSigmoid: op = 3; break;
      case FOp::kRmsNorm:
        // Mirrors the CPU interpreter's assertion (cpu_ops.cpp:1674): the shader
        // hard-codes the mean-square reduction, so any other kind must not reach it.
        VT_CHECK(st.reduce == FReduce::kMeanSquare,
                 "vulkan fused_chain: rmsnorm needs kMeanSquare");
        op = 4;
        break;
      default:
        VT_CHECK(false, "vulkan fused_chain: non-Tier-1 opcode reached the interpreter");
    }
    // Canonical operand indices (cpu_ops.cpp:1621-1643): 0=x 1=weight 2=residual
    // 3=out, with 2 and 3 the only writable slots.
    VT_CHECK(st.out == 2 || st.out == 3, "vulkan fused_chain: step writes a read-only operand");
    VT_CHECK(st.in[0] <= 3 && st.in[1] <= 3, "vulkan fused_chain: operand index out of range");
    VT_CHECK(residual != nullptr || (st.out != 2 && st.in[0] != 2 && st.in[1] != 2),
             "vulkan fused_chain: recipe touches the residual slot but none was bound");
    const size_t base = static_cast<size_t>(s) * kStepWords;
    steps[base + 0] = op;
    steps[base + 1] = st.out;
    steps[base + 2] = st.in[0];
    steps[base + 3] = st.in[1];
    steps[base + 4] = st.gemma ? 1u : 0u;
  }

  VulkanContext& ctx = VulkanContext::Get();
  const size_t step_bytes = steps.size() * sizeof(uint32_t);
  VT_CHECK(step_bytes <= VulkanContext::kScratchBytes,
           "vulkan fused_chain: step list exceeds the scratch buffer");
  std::memcpy(ctx.ScratchData(), steps.data(), step_bytes);

  Binder bind;
  const uint32_t x_off = bind.Add(x, "fused_chain: x");
  const uint32_t w_off = bind.Add(weight, "fused_chain: weight");
  const uint32_t res_off = residual != nullptr ? bind.Add(*residual, "fused_chain: residual")
                                               : bind.Add(out, "fused_chain: out");
  const uint32_t out_off = bind.Add(out, "fused_chain: out");
  bind.AddRaw(ctx.ScratchBuffer());
  FcParams p{static_cast<uint32_t>(t),
             static_cast<uint32_t>(h),
             static_cast<uint32_t>(r.n),
             DtypeCode(x.dtype),
             DtypeCode(weight.dtype),
             residual != nullptr ? DtypeCode(residual->dtype) : 0u,
             DtypeCode(out.dtype),
             x_off,
             w_off,
             res_off,
             out_off,
             eps};
  Go("vt_fused_chain", bind, p, static_cast<uint32_t>(t));
}

// cpu_ops.cpp:187-260 MatmulChunked / MatmulKernel / MatmulBTKernel. One
// invocation per OUTPUT ELEMENT with the whole K reduction on it, which is what
// the CPU kernel does too (it deliberately never splits a K reduction across
// threads), so the accumulation ORDER matches rather than merely the tolerance.
//
// The naive body is the portable correctness tier on purpose; the tiled and
// cooperative-matrix ports (llama.cpp mul_mm.comp / mul_mm_cm2.comp) are VK-C,
// which needs exactly this as its same-device A/B reference.
// TACTIC SELECTION (VK-C). Every condition below is a HARD requirement of the
// cooperative-matrix path, not a heuristic, and failing any one of them means the
// scalar kernel -- which is always correct -- runs instead:
//
//   * the device reports the EXACT configuration the committed coopmat SPIR-V is
//     written to (16x16x16, bf16/bf16/f32/f32, SUBGROUP). Vulkan matches
//     configurations exactly, so "close enough" does not exist;
//   * subgroup size is 32, because the shader's workgroup is a literal 32 (see
//     the shader for why the size cannot travel as a specialization constant at
//     this target);
//   * BOTH operands are bf16. Every configuration GB10 reports takes
//     bf16/f16/int8 inputs, so f32 operands can never use this path -- a hardware
//     constraint, not a policy;
//   * K is a multiple of 16. A ragged K tail cannot be masked inside a
//     cooperative-matrix load, and silently truncating it would drop terms from
//     the dot product. Ragged M and N are fine: the shader bounds-checks its
//     store.
//
// MEASURED: GB10 satisfies all four; llvmpipe -- the only Vulkan device CI can
// reach -- fails the first, so CI exercises the scalar tactic and this selection
// returning false is the property CI can actually gate.
bool CoopMatMatmulUsable(const Tensor& a, const Tensor& b, int64_t k, int64_t m, int64_t n) {
  // VT_VULKAN_COOPMAT=0 forces the scalar tactic. This exists for ONE reason: a
  // same-binary A/B. Comparing the two tactics across two builds would confound
  // the kernel with everything else that differs between them, and the project's
  // benchmark protocol wants the arms to differ in exactly one thing. Default is
  // ON -- absent or any value other than "0" leaves selection to the capability
  // probe, so production behaviour is unchanged by the lever's existence.
  static const bool kDisabled = [] {
    const char* v = std::getenv("VT_VULKAN_COOPMAT");
    return v != nullptr && std::strcmp(v, "0") == 0;
  }();
  if (kDisabled) return false;

  const VulkanContext& ctx = VulkanContext::Get();

  // WHY IT DECLINED, reported once per distinct reason under
  // VT_VULKAN_DISPATCH_STATS. A 27B prefill measured 99.9% of GPU time in the
  // UNTILED SCALAR kernel at ~96 GFLOP/s -- roughly 1% of what this device can do
  // -- because this predicate was returning false for every GEMM, and reading the
  // source did not reveal which clause. Shapes were whole tiles and activations
  // were bf16, so the obvious two candidates were both excluded by inspection and
  // the answer still had to be measured. A selection predicate that can silently
  // route an entire model onto the correctness tier should be able to say so.
  if (kCoopMatWhy) {
    const char* why = nullptr;
    if (!ctx.coopmat_bf16_f32()) why = "device reports no bf16->f32 16x16x16 SUBGROUP config";
    else if (ctx.subgroup_size() != 32) why = "subgroup size is not 32";
    else if (a.dtype != DType::kBF16) why = "operand a is not bf16";
    else if (b.dtype != DType::kBF16) why = "operand b is not bf16";
    else if (k % ctx.coopmat_tile_k() != 0) why = "K is not a multiple of the tile K";
    else if (m < static_cast<int64_t>(ctx.coopmat_tile_m())) why = "M is below one tile row";
    else if (n % static_cast<int64_t>(ctx.coopmat_tile_n()) != 0) why = "N is not a multiple of the tile N";
    if (why != nullptr) {
      static std::mutex seen_mu;
      static std::set<std::string> seen;
      std::string key = std::string(why) + "|" + std::to_string(static_cast<int>(a.dtype)) +
                        "," + std::to_string(static_cast<int>(b.dtype));
      std::lock_guard<std::mutex> g(seen_mu);
      if (seen.insert(key).second) {
        std::fprintf(stderr,
                     "[vt vulkan] coopmat DECLINED: %s  (a.dtype=%d b.dtype=%d "
                     "m=%lld k=%lld n=%lld)\n",
                     why, static_cast<int>(a.dtype), static_cast<int>(b.dtype),
                     (long long)m, (long long)k, (long long)n);
        std::fflush(stderr);
      }
    }
  }

  // COOPMAT LOWER BOUND ON M, as a knob, because the TACTIC ORDER makes the GEMV
  // bound alone unable to move anything.
  //
  // MatmulBT tries coopmat (line ~889) before GEMV (line ~1086), so coopmat claims
  // every shape with m >= tile_m and the GEMV upper bound never sees them. Raising
  // VT_VULKAN_GEMV_MAX_M to 17 therefore changed NOTHING at c=16 -- measured, and
  // the tactic readout caught it: the top kernel stayed vt_matmul_coopmat at 96.0%
  // in both arms. A knob that does not move the variable is not a control, and its
  // null is worthless.
  //
  // So the handover needs BOTH ends. This one raises the floor at which coopmat is
  // willing to take a shape; the other raises the ceiling at which GEMV is willing
  // to keep it. Default is the driver tile (16 here), which is today's behaviour.
  // DEFAULT 17, so m=16 goes to GEMV instead. This is the handover point, and it
  // moved because the old one was set before the batched decode GEMV existed.
  //
  // Evidence for moving it, all measured 2026-09-03:
  //   speed      c=16 goes 67.2 -> 75.8 tok/s, +12.8%, two interleaved rounds
  //              with the tactic identity checked on both arms
  //   numerics   GEMV at m=16 measures NMSE 1.93e-4 against the scalar reference,
  //              against coopmat's own 1.61e-4 on the same shape -- essentially
  //              the same distance, so this is not speed bought with accuracy.
  //              The token-id difference it produces is one argmax tie-break, and
  //              it took a raw-tensor comparator to tell those two apart.
  //   context    c=16 was the worst cell on the whole curve against llama.cpp
  //              (0.661x) AND the point where our own curve went backwards.
  //
  // ⚠️ Never below the driver's reported tile: on a device whose only bf16
  // configuration is 64x64x16 (an Adreno 840 reports exactly that for fp16),
  // a literal 17 would hand coopmat shapes it cannot load.
  static const int64_t kCoopMatMinM = [] {
    const char* v = std::getenv("VT_VULKAN_COOPMAT_MIN_M");
    const long n2 = v != nullptr ? std::strtol(v, nullptr, 10) : 17;
    return (n2 >= 2 && n2 <= 256) ? static_cast<int64_t>(n2) : 17;
  }();
  // Divisibility is against the extents the DRIVER reported, not a literal 16.
  // On this card they are 16; on a device reporting 64x32x16 the same predicate
  // asks for 64 and 32 with no code change. The reasoning below is unchanged --
  // only the numbers stop being assumptions.
  const uint32_t t_m = ctx.coopmat_tile_m();
  const uint32_t t_n = ctx.coopmat_tile_n();
  const uint32_t t_k = ctx.coopmat_tile_k();
  return ctx.coopmat_bf16_f32() && ctx.subgroup_size() == 32 &&
         a.dtype == DType::kBF16 && b.dtype == DType::kBF16 && k % t_k == 0 &&
         // M AND N MUST ALSO BE WHOLE TILES. `coopMatLoad` reads a FULL 16x16
         // tile with no masking, so a partial tile reads past the end of the
         // operand -- and the store being bounds-checked does not save it,
         // because the fault happens on the LOAD. MEASURED: lm_head at M=1
         // (single decode token) read 15 rows (~30 KB) past a small activation
         // buffer, faulted the GPU, and the fence NEVER SIGNALLED -- an infinite
         // vkWaitForFences, which presents as a hang, not as an error.
         //
         // The original correctness gate used M=20, N=12 precisely to exercise
         // ragged shapes and PASSED, because there the out-of-bounds read stayed
         // inside the allocation and its garbage rows were discarded by the
         // bounds-checked store. Raggedness alone was not enough; the read has to
         // leave the allocation to fault.
         //
         // M NEED ONLY BE AT LEAST ONE WHOLE TILE, not a multiple of one. The
         // shader slides a trailing tile back to start at M-16, so every read
         // stays in bounds and the shared rows recompute to identical values.
         //
         // Requiring m % 16 == 0 here is what fixed the original hang, and it
         // MEASURED as the entire prefill bottleneck: prompt length gives
         // m = tokens + 1, so 513 % 16 == 1 sent every 27B prefill GEMM to the
         // untiled scalar kernel -- 99.9% of GPU time at ~96 GFLOP/s, about 1% of
         // this device. N stays whole because a ragged N would need the same
         // treatment on the B operand and no shape in play needs it.
         m >= std::max(kCoopMatMinM, static_cast<int64_t>(t_m)) && n % static_cast<int64_t>(t_n) == 0;
}

// GEMV TACTIC SELECTION (VK-F). Same shape of contract as the coopmat predicate
// above -- every requirement is a hard one, and failing any of them runs the
// always-correct scalar kernel instead.
//
// The problem this solves is COALESCING, measured: vt_matmul was ~55% of all GPU
// time in an e2e decode run. It puts one invocation on each output element and
// loops K there, so for MatmulBT lane j reads b[j*k + q] and adjacent lanes land
// k*2 bytes apart -- each pulling its own cache line to use 2 bytes of it. The
// GEMV shader instead gives each output element a workgroup whose lanes stride K,
// so adjacent lanes read adjacent addresses.
//
//   * MatmulBT ONLY. In the other orientation vt_matmul reads b[q*n + j], which
//     is ALREADY coalesced across lanes; the GEMV shape would make that strided
//     and strictly worse. This is not a universally better kernel and the
//     predicate does not pretend otherwise.
//   * m == 1, the decode shape. One workgroup per output element is the right
//     trade only when there are few of them: at prefill m*n workgroups would each
//     do k/128 multiplies, and prefill is the coopmat tactic's job anyway.
//   * k >= the workgroup width, so the strided loop actually has work for every
//     lane. Below that most lanes contribute a zero partial and the reduction
//     costs more than the loop saves.
//
// ACCUMULATION ORDER: the K reduction becomes a tree, so this tactic does NOT
// share the CPU's accumulation order -- it sits in the NMSE tier alongside
// coopmat. That is why it is gated on a token-exactness run and not on an NMSE
// bound alone.
bool GemvMatmulUsable(bool bt, int64_t k, int64_t m) {
  // VT_VULKAN_GEMV=0 forces the scalar tactic, for the same single reason the
  // coopmat lever exists: a same-binary A/B, so the arms differ in exactly one
  // thing. Default ON.
  static const bool kDisabled = [] {
    const char* v = std::getenv("VT_VULKAN_GEMV");
    return v != nullptr && std::strcmp(v, "0") == 0;
  }();
  if (kDisabled) return false;

  // WHY IT DECLINED, once per distinct reason, under VT_VULKAN_DISPATCH_STATS.
  // Same reasoning as the coopmat predicate above: a 27B decode profile showed the
  // UNTILED SCALAR kernel still taking 256 calls at 12.53 ms -- one per output
  // token, and the largest single per-call cost in decode -- and no amount of
  // reading the source says WHICH clause sent it there.
  if (kCoopMatWhy) {
    const char* why = nullptr;
    if (!bt) why = "not MatmulBT (b is [K,N]; that layout is already coalesced)";
    else if (m != 1 && m >= 16) why = "M is at least 16 (coopmat territory)";
    else if (k < static_cast<int64_t>(kWorkgroupSize)) why = "K is below one workgroup width";
    if (why != nullptr) {
      static std::mutex gseen_mu;
      static std::set<std::string> gseen;
      std::lock_guard<std::mutex> g(gseen_mu);
      if (gseen.insert(std::string(why)).second) {
        std::fprintf(stderr, "[vt vulkan] gemv DECLINED: %s  (bt=%d m=%lld k=%lld)\n",
                     why, bt ? 1 : 0, (long long)m, (long long)k);
        std::fflush(stderr);
      }
    }
  }

  // THE M = 2..15 GAP (local change), behind VT_VULKAN_GEMV_SMALL_M.
  //
  // Three tactics serve MatmulBT and their M ranges do not meet:
  //     vt_matmul_vec      M == 1     0.074 ms/call   this kernel
  //     vt_matmul_coopmat  M >= 16    0.53  ms/call   tensor cores
  //     vt_matmul          anything   0.76  ms/call   the untiled scalar kernel
  // so M = 2..15 falls to the scalar kernel, 7-10x slower than either
  // neighbour -- and M = 2..15 is exactly what concurrency 2..15 produces.
  //
  // MEASURED on this box, 27B bf16, aggregate tok/s by concurrency:
  //     c1 22.5   c2 8.3   c4 11.1   c8 12.2   c16 77.8   c32 135.5
  // The dip is not scheduling and not batching: dispatch counts at c=1 and c=4
  // are IDENTICAL (28992), and the per-shader histogram shows the work moving
  // from vt_matmul_vec (985 ms, 53%) to vt_matmul (10201 ms, 96.5%).
  //
  // The shader needs no change. It already indexes m * n output elements flatly
  // and derives the output row as base / n, so M > 1 has always been expressible;
  // the m == 1 restriction is a HOST predicate, and it exists only because
  // VT_MM_ROWS > 1 gives a workgroup consecutive elements that must not straddle
  // two output rows. GemvRows already folds rows down to 1 when m != 1, so at
  // rows == 1 that reason does not apply.
  //
  // Bounded above at 16 because coopmat takes over there and is faster.
  static const bool kSmallM = [] {
    const char* v = std::getenv("VT_VULKAN_GEMV_SMALL_M");
    return v == nullptr || v[0] != '0';   // default ON, VT_VULKAN_GEMV_SMALL_M=0 to A/B
  }();
  if (!bt) return false;
  // THE UPPER BOUND IS NOW A KNOB, because the comment above claims coopmat
  // "takes over there and is faster" and that claim has expired.
  //
  // It was true when written. The numbers it cites -- c16 77.8, c32 135.5 -- come
  // from BEFORE the batched decode GEMV landed, which took c=4 from 34.5 to 83.5
  // by sharing one weight row across batch rows. That change made this tactic
  // substantially better at m > 1 and nobody re-measured where it stops winning.
  //
  // MEASURED 2026-09-03, same dispatch count (462,324) at both concurrencies:
  //     c=8  (m=8)   vt_matmul_vec     0.1328 ms/call   GPU span 38.4 s
  //     c=16 (m=16)  vt_matmul_coopmat 0.4089 ms/call   GPU span 94.5 s
  // Twice the rows for 2.46x the time, so coopmat is about 1.5x worse per unit of
  // work at the very shape this bound hands to it -- and the end-to-end curve
  // shows it: 86.2 tok/s at c=8 falling to 77.7 at c=16, non-monotonic, and the
  // worst cell against llama.cpp on the whole curve.
  //
  // A THRESHOLD MEASURED BEFORE AN OPTIMISATION IS NOT VALID AFTER IT. The knob
  // exists so the crossover can be found rather than assumed; the default stays
  // at 16 until a sweep says otherwise.
  static const int64_t kSmallMMax = [] {
    const char* v = std::getenv("VT_VULKAN_GEMV_MAX_M");
    const long n = v != nullptr ? std::strtol(v, nullptr, 10) : 17;
    return (n >= 2 && n <= 256) ? static_cast<int64_t>(n) : 17;
  }();
  if (m != 1 && !(kSmallM && m >= 2 && m < kSmallMMax)) return false;
  return k >= static_cast<int64_t>(kWorkgroupSize);
}

// GEMV VARIANT AXES (row BACKEND-VULKAN-GEMVROWS). Both are specialization
// constants on the SAME committed module, so every arm of an A/B lives in one
// binary; see shaders/vt_matmul_vec.comp for what each one can and cannot buy.
//
// Defaults are named here rather than spelled inline so a measured flip is one
// edit and the env lever keeps meaning "override the measured default".
//
// MEASURED, GB10, benchmarks/vulkan_gemv_ab.cpp over the 27B's own decode shapes,
// 9 arms x 4 rotated passes, each arm paired against the rows=1/pack=0 baseline
// measured IN THE SAME PASS (the box drifted 15.5% peak-to-peak between passes,
// so an unpaired ranking would have been noise):
//
//   rows=1 pack=0  1.000x     rows=2 pack=0  0.966x     rows=4 pack=0  0.968x
//   rows=1 pack=1  1.025x     rows=2 pack=1  1.017x     rows=4 pack=1  1.014x
//   rows=1 pack=2  1.086x     rows=2 pack=2  1.047x     rows=4 pack=2  1.039x
//
// ROWS > 1 IS A MEASURED LOSS ON THIS DEVICE, in all four passes and at every
// pack width, so it ships OFF. It is kept as an axis rather than deleted because
// llama.cpp makes exactly this knob device-dependent -- ggml-vulkan.cpp:4705-4719
// sets `rm_stdq` to 2 on AMD GCN and on Intel and leaves it 1 elsewhere -- and the
// board this campaign is waiting on (VK-I) is one of the two it raises it for.
// Deleting the axis would mean rediscovering it there.
// The likely reason it loses here: halving the workgroup count halves the number
// of independent sequential read streams the memory controller sees, and each
// surviving workgroup interleaves reads from rows k*2 bytes apart. The activation
// re-reads it saves were L2 hits, which were never the constraint.
constexpr uint32_t kGemvRowsDefault = 1;
constexpr uint32_t kGemvPackDefault = 2;

uint32_t EnvVariant(const char* name, uint32_t fallback, uint32_t lo, uint32_t hi) {
  const char* v = std::getenv(name);
  if (v == nullptr || *v == '\0') return fallback;
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(v, &end, 10);
  if (end == v || *end != '\0') return fallback;
  const uint32_t val = static_cast<uint32_t>(parsed);
  return (val >= lo && val <= hi) ? val : fallback;
}

// Once per distinct reason, exactly like the two DECLINED reporters above: a
// variant that silently stops applying is the failure mode this whole row has to
// be able to rule out, and reading the source cannot tell you which clause fired.
void VariantWhy(const char* kind, const char* why) {
  if (!kCoopMatWhy) return;
  static std::mutex mu;
  static std::set<std::string> seen;
  std::lock_guard<std::mutex> g(mu);
  if (seen.insert(std::string(kind) + "|" + why).second) {
    std::fprintf(stderr, "[vt vulkan] gemv %s DECLINED: %s\n", kind, why);
    std::fflush(stderr);
  }
}

// OUTPUT ELEMENTS PER WORKGROUP. Requires m == 1 (so the block cannot straddle
// two output rows) and n % rows == 0 (so the shader has no ragged tail to branch
// on in its inner loop). Both hold for every decode projection in the models this
// backend runs; when they do not, the request degrades to the next lower power of
// two rather than failing, and says so under VT_VULKAN_DISPATCH_STATS.
uint32_t GemvRows(int64_t m, int64_t n) {
  // ONLY 1, 2 AND 4 EXIST, and that is a correctness constraint rather than a
  // taste one: the shader materialises exactly four accumulator sets and guards
  // them with `VT_MM_ROWS >= 2` / `>= 4`, so a value of 3 would compute two
  // output elements while the host dispatched ceil(n/3) workgroups and silently
  // drop a third of the row. Anything else falls back to 1.
  static const uint32_t kWant = [] {
    const uint32_t v = EnvVariant("VT_VULKAN_GEMV_ROWS", kGemvRowsDefault, 1, 4);
    return (v == 1u || v == 2u || v == 4u) ? v : 1u;
  }();
  uint32_t rows = kWant;
  while (rows > 1u && (m != 1 || (n % static_cast<int64_t>(rows)) != 0)) rows >>= 1;
  if (rows < kWant) {
    VariantWhy("rows", m != 1 ? "M is not 1" : "N is not a multiple of the requested row count");
  }
  return rows;
}

// LOAD WIDTH for 16-bit operands: 0 = one element per load, 1 = two through the
// 32-bit view, 2 = four through the 64-bit view. Every requirement is a hard one
// and a failure DEGRADES to the next narrower width rather than declining, so a
// shape that cannot take the widest load still takes the one it can.
//   * both operands 16-bit -- an f32 operand is already one element per 32-bit
//     word, so there is nothing to pack.
//   * byte offsets aligned to the load width -- the wide views index at
//     (off >> 2) and (off >> 3), so a misaligned offset would read a word
//     straddling two elements.
//   * k divisible by the element count -- the shader's pair/quad count is exact
//     with no ragged element left over, and a row start j*k must itself be
//     aligned for every j, which k carries.
uint32_t GemvPack(const Tensor& a, const Tensor& b, int64_t k, uint32_t a_off,
                  uint32_t b_off) {
  static const uint32_t kWant = EnvVariant("VT_VULKAN_GEMV_PACK", kGemvPackDefault, 0, 2);
  if (kWant == 0) return 0;
  const bool a16 = a.dtype == DType::kF16 || a.dtype == DType::kBF16;
  const bool b16 = b.dtype == DType::kF16 || b.dtype == DType::kBF16;
  if (!a16 || !b16) {
    VariantWhy("pack", "an operand is not 16-bit (nothing to pack)");
    return 0;
  }
  uint32_t level = kWant;
  if (level >= 2 &&
      ((a_off % 8u) != 0u || (b_off % 8u) != 0u || (k % 4) != 0)) {
    VariantWhy("pack", "K or an operand byte offset is not 4-element aligned (8 B)");
    level = 1;
  }
  if (level >= 1 &&
      ((a_off % 4u) != 0u || (b_off % 4u) != 0u || (k % 2) != 0)) {
    VariantWhy("pack", "K or an operand byte offset is not 2-element aligned (4 B)");
    level = 0;
  }
  return level;
}

template <bool kBT>
void MatmulGeneric(Queue&, Tensor& out, const Tensor& a, const Tensor& b) {
  const int64_t m = a.shape[0], k = a.shape[1];
  const int64_t n = kBT ? b.shape[0] : b.shape[1];
  if (m == 0 || n == 0) return;
  Binder bind;
  const uint32_t a_off = bind.Add(a, "matmul: a");
  const uint32_t b_off = bind.Add(b, "matmul: b");
  const uint32_t out_off = bind.Add(out, "matmul: out");

  if (CoopMatMatmulUsable(a, b, k, m, n)) {
    // TILED TACTIC (local addition), selected by VT_VULKAN_COOPMAT_TILED=1.
    //
    // The default coopmat shader gives one 16x16 output tile to one subgroup and
    // reads both operands from global memory on every K step, so each tile
    // re-reads a whole A row-block and B column-block. MEASURED on an RTX PRO
    // 6000 Blackwell (1792 GB/s): at 512x5120x17408 that is 62x the necessary
    // traffic and the kernel lands at 1884 GB/s effective, i.e. ON the roofline
    // -- bandwidth-bound on redundant reads rather than short of tensor-core
    // throughput. At M=2048 the redundancy is 229x.
    //
    // vt_matmul_coopmat_tiled.comp stages A and B blocks in shared memory and
    // gives one 64x64 output tile to a workgroup of four subgroups, cutting the
    // tile count 16x before counting the reuse staging itself buys.
    //
    // Same-binary A/B, exactly as VT_VULKAN_COOPMAT is: comparing two builds
    // would confound the kernel with everything else that differs between them.
    //
    // ★ THE GATE AND THE MEASUREMENT ARE IN, 2026-09-06, AND IT STAYS OFF.
    //
    // Correctness first: teacher-forced against the untiled default at m=1026,
    // steps 0..2 all NMSE 0 -- bit-identical, so this is a speed question only.
    //
    // Speed, prefill shapes, same binary:
    //     input-len   untiled      tiled
    //     1024        1734.2 ms    2051.7 ms    tiled 1.18x slower
    //     2048        3498.2       3822.6       tiled 1.09x slower
    //     3072        5331.4       5992.8       tiled 1.12x slower
    //
    // IT IS NOT THAT THE KERNEL IS BAD. Against the baseline it was designed for
    // -- MR=1/NR=1, which was the default when it was written -- it wins
    // decisively: at m=1026, 9582 ms untiled against 3952 tiled, 2.42x. The
    // register-blocking work that landed the same day takes the untiled path to
    // 3756 ms, 2.55x, and that is the only reason tiled now loses. Two
    // independent attacks on the same redundant-traffic problem converged, and
    // the cheaper one -- relaxing an over-strict host guard and changing two
    // defaults -- came out marginally ahead.
    //
    // The note above predicts tiled should widen its lead as M grows (62x
    // redundancy at M=512, 229x at M=2048). The gap does narrow, 1.18 -> 1.09,
    // but it never crosses -- and M cannot be pushed further here anyway:
    // ResolveMaxNumBatchedTokens caps a dense-arch step at 2048 tokens, so m=2048
    // is the largest single-step GEMM this engine produces. The 3072 row is that
    // cap in action, chunked into 2048+1024, which is why its ms/call reads lower
    // than the 2048 row rather than higher.
    //
    // ⇒ Correct, well-built, and second by about ten percent on every shape this
    // engine actually runs. Keep it behind the env, and revisit if the token
    // budget rises or a model arrives with a larger natural M.
    static const bool kTiled = [] {
      const char* v = std::getenv("VT_VULKAN_COOPMAT_TILED");
      return v != nullptr && v[0] != '0';
    }();
    // REGISTER BLOCKING for the untiled coopmat kernel (VT_VULKAN_COOPMAT_MR /
    // _NR, both default 1 = the original kernel, byte-identical).
    //
    // The measurement that motivates this, at c=32 on 27B bf16: this shader is
    // 94.4% of GPU time, the gap between dispatches is 0.1%, and every hot shape
    // has m equal to the concurrency -- 32 rows against n in the tens of
    // thousands. With one 16x16 tile per workgroup that is 2 tile-rows and
    // thousands of tile-columns, each streaming its own operands, so the loads
    // come to 4.0x the useful bytes and the kernel delivers 254 GB/s of useful
    // traffic on a 1792 GB/s card.
    //
    // MR shares one B load across tile-rows; NR shares one A load across
    // tile-columns. Same idea as the batched decode GEMV that landed earlier in
    // this campaign -- reuse the operand you already paid to load -- applied to
    // the other kernel and the other operand.
    //
    // Blocking is used ONLY when it divides exactly. A partial block would need
    // ragged-N handling the cooperative-matrix load cannot express, and reading
    // past the operand faults the GPU as an infinite fence wait rather than an
    // error.
    // ★ MR DEFAULTS TO 2 AS OF 2026-09-06, and the reason it did not before is
    // that it could never reach the shapes where it pays.
    //
    // The host used to demand m % (16*MR) == 0, which every prefill shape here
    // fails (m=1026 from a 1,024-token prompt), so MR was forced to 1 exactly
    // where reuse across M matters most. The shader was always able to handle a
    // ragged M -- it slides its last block back -- so the guard was stricter than
    // the kernel needed. With the guard relaxed to "m must be large enough":
    //
    //   prefill (m=1026, teacher-forced, two rounds)
    //     MR=1  coopmat 8453.8 / 8506.0 ms   9.78 / 9.84 ms/call   regs 57
    //     MR=2  coopmat 4919.9 / 4925.4      5.69 / 5.70           regs 82
    //     -> 1.72x on the GEMM, 1.60x on GPU span, BIT-IDENTICAL output
    //   decode c=16 (m=16, where MR is forced to 1 regardless)
    //     MR=1  GPU 33528 / 34224     MR=2  GPU 31196 / 31731
    //     -> that 1.07x is the PREFILL inside that run, not decode changing
    //
    // Registers go 57 -> 82 with Stack Size 0, so nothing spills, and the
    // occupancy it costs is worth less than the traffic it saves at these shapes.
    // That is the opposite of the NR=4 result below, and not a contradiction: the
    // NR note measured decode shapes and said so.
    static const uint32_t kCmMR = [] {
      const char* v = std::getenv("VT_VULKAN_COOPMAT_MR");
      const long n2 = v != nullptr ? std::strtol(v, nullptr, 10) : 2;
      return n2 == 2 ? 2u : 1u;
    }();
    // NR DEFAULTS TO 2, measured. Four paired arms against an adjacent baseline
    // give 1.054-1.080, and a three-pair confirmation run lands at 1.057 with the
    // baselines reproducing to 0.24% and the arms to 0.56%. Token ids are
    // byte-identical across every arm, prefill included.
    //
    // Two is the optimum, not an end point: NR=4 measures 0.952 (SLOWER) and
    // MR=2/NR=4 only 1.031. The driver says why -- Register Count goes 40, 57,
    // 76, 122 across (1,1) (2,1) (1,4) (2,4) with Stack Size 0 throughout, so
    // nothing spills and what is lost is occupancy. Traffic and occupancy trade,
    // and the trade turns over between 2 and 4.
    //
    // Scope, stated rather than implied: measured on the DECODE shapes (m=32,
    // n in the thousands to tens of thousands) on one Blackwell card. Prefill
    // shapes are covered for CORRECTNESS by the same token-exact runs but their
    // performance is unmeasured. VT_VULKAN_COOPMAT_NR=1 is the rollback.
    static const uint32_t kCmNR = [] {
      const char* v = std::getenv("VT_VULKAN_COOPMAT_NR");
      if (v == nullptr) return 2u;
      const long n2 = std::strtol(v, nullptr, 10);
      return (n2 == 2 || n2 == 4) ? static_cast<uint32_t>(n2) : 1u;
    }();
    uint32_t cm_mr = kCmMR, cm_nr = kCmNR;
    // ★ NR IS SHAPE-DEPENDENT, because its optimum reverses between the two
    // shape families this kernel serves.
    //
    // The note above measured NR on DECODE shapes and found 4 slower than 2
    // (0.952), attributing it to occupancy: registers go 40/57/76/122 across
    // (MR,NR) = (1,1) (2,1) (1,4) (2,4) with nothing spilling. It also said, in
    // as many words, that prefill performance was unmeasured. It is measured now,
    // teacher-forced and bit-identical, at m=1026:
    //
    //     NR=1  coopmat 5763.3 ms   6.67 ms/call   regs 57
    //     NR=2  coopmat 4889/4900   5.66/5.67      regs 82
    //     NR=4  coopmat ★ 3537.0    ★ 4.09         regs 129
    //     -> ★ 1.38x over NR=2, 1.63x over NR=1, and Stack Size stays 0
    //
    // Same knob, opposite answer, because a prefill GEMM has enough work per tile
    // to pay for the registers and a decode GEMV does not. A single global default
    // has to be wrong for one of them.
    //
    // ⚠️ THE THRESHOLD IS NOT MEASURED, only bounded: 32 (decode at c=32) prefers
    // 2 and 1026 prefers 4, and nothing between has been tried. 256 sits well
    // above any concurrency we run and well below a real prompt, so it separates
    // the two families without pretending to be a tuned crossover.
    static const int64_t kCmNrBigM = [] {
      const char* v = std::getenv("VT_VULKAN_COOPMAT_NR4_MIN_M");
      const long n2 = v != nullptr ? std::strtol(v, nullptr, 10) : 256;
      return (n2 >= 1 && n2 <= 100000) ? static_cast<int64_t>(n2) : 256;
    }();
    if (std::getenv("VT_VULKAN_COOPMAT_NR") == nullptr && m >= kCmNrBigM) {
      cm_nr = 4u;
    }
    // ★ M ONLY HAS TO BE BIG ENOUGH, NOT DIVISIBLE. The shader slides its last
    // M block back (vt_matmul_coopmat.comp, "RAGGED M, HANDLED BY SHIFTING THE
    // LAST BLOCK BACK"), and that slide is written in terms of BM = VT_CM_M *
    // VT_MM_CM_MR, so it is already correct for MR > 1. Requiring divisibility
    // here was stricter than the kernel needs, and the cost fell entirely on
    // PREFILL: m=1026 from a 1,024-token prompt fails 1026 % 32, so cm_mr was
    // forced to 1 and the M-axis register blocking -- which the note above calls
    // "the optimisation" -- was switched off for exactly the shapes where reuse
    // across M matters most.
    //
    // N is NOT relaxed: the shader does not slide N back, and says so.
    if (cm_mr > 1u && m < 16 * static_cast<int64_t>(cm_mr)) cm_mr = 1u;
    if (cm_nr > 1u && n % (16 * static_cast<int64_t>(cm_nr)) != 0) cm_nr = 1u;
    // The tile extents come from the DRIVER now, not from a constant. On this
    // card there is one bf16 configuration and it is 16x16x16, so these are 16;
    // on a device reporting 64x32x16 they would be that, with no rebuild and no
    // second shader -- the dimensions are specialization constants, which was
    // verified to compile before any of this was written.
    const VulkanContext& cmctx = VulkanContext::Get();
    const uint32_t tile_m = cmctx.coopmat_tile_m();
    const uint32_t tile_n = cmctx.coopmat_tile_n();
    const uint32_t tile_k = cmctx.coopmat_tile_k();
    // Blocking multiplies the tile, so re-check divisibility against the REAL
    // extents rather than an assumed 16.
    // Same relaxation against the driver-reported tile: big enough, not divisible.
    if (cm_mr > 1u && m < static_cast<int64_t>(tile_m) * cm_mr) cm_mr = 1u;
    if (cm_nr > 1u && n % (static_cast<int64_t>(tile_n) * cm_nr) != 0) cm_nr = 1u;
    const uint32_t spec[7] = {kBT ? 1u : 0u, DtypeCode(out.dtype), cm_mr, cm_nr,
                              tile_m, tile_n, tile_k};
    MatmulParams p{static_cast<uint32_t>(m), static_cast<uint32_t>(n),
                   static_cast<uint32_t>(k), a_off, b_off, out_off};
    // Temporary diagnostic: the tiled path was never taken in the real model and
    // the gate did not say why. A predicate that can silently route every GEMM
    // back to the slower tactic should be able to report it -- the same
    // reasoning that put kCoopMatWhy on the coopmat predicate above.
    if (kTiled) {
      static std::mutex tw_mu;
      static std::set<std::string> tw_seen;
      const char* why = nullptr;
      if (m < 64) why = "M < 64";
      else if (n % 64 != 0) why = "N not a multiple of 64";
      else if (k % 32 != 0) why = "K not a multiple of 32";
      if (why != nullptr) {
        std::string key = std::string(why) + "|" + std::to_string((long long)m) + "," +
                          std::to_string((long long)k) + "," + std::to_string((long long)n);
        std::lock_guard<std::mutex> g(tw_mu);
        if (tw_seen.insert(key).second) {
          std::fprintf(stderr, "[vt vulkan] TILED declined: %s  (m=%lld k=%lld n=%lld)\n",
                       why, (long long)m, (long long)k, (long long)n);
          std::fflush(stderr);
        }
      }
    }
    if (kTiled && m >= 64 && n % 64 == 0 && k % 32 == 0) {
      // The tiled shader's own edge rules: it slides a trailing tile back the
      // way the untiled one does, so M only needs one whole 64-row tile, but N
      // and K must be whole because a partial staged block would read past the
      // operand -- and coopMatLoad faults on the LOAD, which presents as a hang.
      const uint32_t tiles =
          static_cast<uint32_t>(((m + 63) / 64) * ((n + 63) / 64));
      Go("vt_matmul_coopmat_tiled", bind, p, tiles, spec, 2);
      return;
    }
    // One workgroup (= one subgroup) per 16x16 OUTPUT TILE. Deliberately not
    // FlatGroupCount, which divides an element count by the workgroup size: here
    // the whole subgroup cooperates on one tile.
    // WORKGROUP-SCOPE TACTIC (VK_NV_cooperative_matrix2), opt-in via
    // VT_VULKAN_COOPMAT2=1 -- the same variable that asks the device for the
    // capability, so the extension cannot be enabled without the shader being
    // reachable or the reverse.
    //
    // Selected only on exact division. This kernel has NO ragged handling on
    // purpose: a cooperative-matrix load cannot mask, and reading past an operand
    // faults the GPU as an infinite vkWaitForFences rather than an error, so the
    // guard belongs here where it can be stated rather than in the shader where
    // it would be silent.
    //
    // ⚠️ NVIDIA ONLY. There is no KHR equivalent for workgroup scope, so this is
    // the per-vendor peak tier and the subgroup kernel remains the universal path.
    // Declining is the normal case, not the failure case.
    // N DEFAULTS TO 64, and it is a dominance argument rather than a speed one.
    // Swept against adjacent baselines: 32 gives 1.158, 64 gives 1.171, 128 gives
    // 1.171 -- so 64 and 128 tie inside noise while 32 is about two percent worse.
    // Between the two that tie, 64 wins on everything else: it spills half the
    // shared memory (8 KB against 16), which matters beside the 8 KB the driver
    // reserves for workgroup scope, and n % N == 0 is easier to satisfy, so more
    // shapes qualify on models other than this one. Equal speed, fewer resources,
    // wider applicability.
    static const uint32_t kWgN = [] {
      const char* v = std::getenv("VT_VULKAN_COOPMAT2_N");
      const long n2 = v != nullptr ? std::strtol(v, nullptr, 10) : 64;
      // 256 is admissible only because the spill buffer is 32*128 floats and a
      // 16-row tile at N=256 is exactly 4096 -- the same size. At wg_m=32 the
      // host clamps below, since 32x256 would be double the buffer.
      return (n2 == 32 || n2 == 64 || n2 == 128 || n2 == 256) ? static_cast<uint32_t>(n2) : 64u;
    }();
    // The workgroup shader now loads through tensorLayoutNV, so workgroup scope
    // alone is no longer enough to select it: the device must also have granted
    // cooperativeMatrixTensorAddressing. Before 2026-09-06 the shader used plain
    // coopMatLoad and this second condition did not exist; the shader changed and
    // the predicate has to change with it, or the tactic runs a kernel using a
    // capability nobody asked the device for.
    if (cmctx.coopmat2_workgroup() && cmctx.coopmat2_tensor_addressing()) {
      // TWO SHAPES, ONE MODULE. The device reports bf16 workgroup-scope
      // cooperative matrix at granularity 16x16x16 with 32 or 64 invocations and
      // 32x32x16 with 256, and the granularity is paired with the invocation
      // count -- so serving m=16 as well as m=32 means two workgroup sizes.
      //
      // m=16 is not an afterthought. It is the concurrency where our own curve
      // goes BACKWARDS (86.2 at c=8 falling to 77.7 at c=16) and where we sit
      // furthest behind llama.cpp on the whole curve. The alternative fix --
      // handing m=16 to the GEMV tactic -- is 12.8% faster and measured to move
      // one sequence in sixteen off the scalar reference, because GEMV sums in
      // pairs rather than sequentially. This path keeps the coopmat family's
      // accumulation, and the coopmat family is byte-identical to that reference
      // over 256 tokens.
      //
      // ⇒ So the two candidates are not equivalent: one buys speed with a
      // numerical change, the other tries to buy it without one.
      // ★ K DEPTH IS SWEEPABLE. External research (2026-09-06) put this first:
      // llama.cpp builds its bf16 coopmat2 pipelines with BK=64 across all three
      // size classes (small 64x64x64, medium 128x128x64, large 128x256x64 --
      // ggml-vulkan.cpp:4254 / :4664), while this path has had K pinned at 16.
      // The RFG-1 review reached the same place from the other direction, calling
      // the K slices narrow and the load/MMA loop unpipelined.
      //
      // K was not sweepable before the tensor layout landed: a deeper K step with
      // a plain coopMatLoad has the same bounds problem a wider M tile had.
      static const uint32_t kWgK = [] {
        const char* v = std::getenv("VT_VULKAN_COOPMAT2_WGK");
        // DEFAULT 64, measured at m=1026 and bit-identical at every step:
        //   K=16  wg 1991.5 ms  2.96 ms/call   GPU span 2749.4
        //   K=32  wg 1388.9     2.07           GPU span 2146.0
        //   K=64  wg 1114.7     1.66           GPU span 1874.9
        // Monotone, 1.79x over K=16 on the GEMM. This is the value llama.cpp uses.
        const long n2 = v != nullptr ? std::strtol(v, nullptr, 10) : 64;
        // 128 wired 2026-09-08: BK was the only monotonic axis in the joint
        // (BK, unroll) sweep and its next step had never been reachable.
        return (n2 == 16 || n2 == 32 || n2 == 64 || n2 == 128) ? static_cast<uint32_t>(n2)
                                                               : 64u;
      }();
      const uint32_t wg_k = kWgK;
      uint32_t wg_m = 0, wg_inv = 0;
      // wg_m=32 WAS QUARANTINED AS "MEASURED WRONG" AND IS NOW FIXED. The whole
      // episode is kept here because the failure mode is the interesting part.
      //
      // It measured NMSE 0.132 against the scalar reference at c=32 decode --
      // 264x this project's 5e-4 bar -- while producing token ids BYTE IDENTICAL
      // to the shipping path across seven interleaved rounds. A logit error that
      // argmax ranks the same way is invisible to every token-level gate, so md5
      // agreement proved nothing, and a +17.7% speedup was measured and nearly
      // shipped on a kernel computing the wrong thing.
      //
      // The cause was the N floor below, not this line: an illegal 16-wide tile
      // was dispatched for n=48 at a granularity that requires 32. With that
      // shape declined the same configuration measures 2.66e-4 and passes.
      //
      // ⇒ And the honest speedup is +10.7% (74.44/74.47 -> 82.40/82.57, two
      // interleaved rounds, tactic identity checked on both arms), not the 17.7%
      // measured before. The gap between the two numbers IS the cost of the shape
      // the old code was skipping illegally -- which is exactly why a retracted
      // number gets re-measured rather than reinstated.
      //
      // ★ AND THE OUTPUT IS BIT-IDENTICAL to the subgroup kernel it displaces.
      // Re-verified under teacher forcing (VT_VULKAN_PIN_TOKEN), which pins both
      // arms to the same token sequence so decode steps stay comparable -- the
      // earlier check only worked because the two arms happened to emit the same
      // ids, which is luck rather than a guarantee. With the kernel confirmed to
      // run (2360 dispatches, 20.4% of GPU time), the logit dumps hash the same:
      // 4370e3cbec04 for both, against cb6a486e56a3 for the scalar reference.
      //
      // ⇒ So this is not speed bought with accuracy. It is 10.7% for nothing.
      //
      // ⇒ It also explains the §22 puzzle better than the explanation given there.
      // When this path was broken, seven interleaved rounds of token md5 agreed,
      // and that was attributed to a rank-preserving error. The simpler and now
      // measured account is that the kernel is bit-identical in the normal case,
      // so only the illegal n=48 tile was corrupting anything, and it corrupted
      // by too little to move an argmax. Both stories fit the md5s; only one of
      // them was measured, and it was not the one written down.
      // ★ M NO LONGER HAS TO DIVIDE THE TILE. The shader now slides its last block
      // back the way the subgroup kernel already did, so a ragged m is exact
      // rather than silently truncated.
      //
      // WHY IT MATTERS: the previous `m % 32 == 0` excluded every PREFILL shape
      // this model produces -- a 1,024-token prompt gives m=1026 -- and prefill is
      // where the deficit actually is. Measured 2026-09-06 against llama.cpp on
      // the same bf16 weights and the same 350 W cap: our TTFT is 7.8x slower at
      // c=1 while decode-only reads 0.90x, near parity. Prefill runs 91.8% in
      // vt_matmul_coopmat at 9.33 ms/call, about 5.8 TFLOP/s, and that kernel is
      // one subgroup per workgroup with every operand fragment re-read from global
      // -- no workgroup-level tiling at all. This path has 8 subgroups and lets
      // the driver tile, which is the reason to reach for it here.
      //
      // The floor stays: sliding back needs m >= wg_m, or p.m - VT_WG_M underflows.
      // ★ VT_VULKAN_COOPMAT2_WGM lets the M tile be swept now that the tensor
      // layout clamps the edges: 64 is what llama.cpp uses (BM=64, BN=64, BK=16 in
      // mul_mm_cm2.comp) and the spill buffer holds 32*128 floats, so 64x64 fits
      // exactly. Before the clamp this could not be tried at all -- a larger tile
      // meant a larger ragged remainder and the slide could not express it.
      static const uint32_t kWgM = [] {
        const char* v = std::getenv("VT_VULKAN_COOPMAT2_WGM");
        // DEFAULT 64, measured: at m=1026 the workgroup GEMM goes 3627/3783 ms at
        // 32 to 2081 at 64 -- 1.74x, bit-identical. 64x64 is exactly the 32*128
        // float spill buffer, and it is the tile llama.cpp uses in mul_mm_cm2.
        const long n2 = v != nullptr ? std::strtol(v, nullptr, 10) : 64;
        return (n2 == 16 || n2 == 32 || n2 == 64 || n2 == 128) ? static_cast<uint32_t>(n2) : 64u;
      }();
      // ★ INVOCATION COUNT IS PAIRED WITH THE GRANULARITY, NOT FREE.
      //
      // The device reports bf16 workgroup scope at granularity 16x16x16 with 32 or
      // 64 invocations and 32x32x16 with 256 -- so an M of 16 is only legal at 64
      // invocations. Making kWgM sweepable wrote `wg_inv = 256` for every value it
      // could take, which meant VT_VULKAN_COOPMAT2_WGM=16 selected M=16 with 256
      // invocations: a configuration this device does not offer.
      //
      // Not caught by any bit-exactness run, because the shipped default is 64 and
      // the illegal pairing is only reachable through the override. Found by
      // external review (2026-09-06, P1) reading the selection against the device
      // table rather than running it.
      if (m >= static_cast<int64_t>(kWgM)) {
        wg_m = kWgM;
        wg_inv = (kWgM >= 32u) ? 256u : 64u;
      }
      else if (m >= 32) { wg_m = 32; wg_inv = 256; }
      else if (m >= 16) { wg_m = 16; wg_inv = 64; }
      // ⭐ THE N FLOOR IS THE GRANULARITY'S N, NOT A LITERAL 16.
      //
      // This is the defect that produced NMSE 0.132 and it is worth stating
      // exactly, because the intent was already written down three paragraphs
      // above and the code simply did not implement it: the granularity is PAIRED
      // WITH THE INVOCATION COUNT -- 16x16x16 at 32/64 invocations, 32x32x16 at
      // 256. So at wg_m=32 / wg_inv=256 the N dimension must be a multiple of 32,
      // and a tile of 16 is not a legal configuration on this device at all.
      //
      // The old clamp walked wg_n down to a floor of 16 regardless. For n=48 that
      // is what happened: 48 is not divisible by 64 or by 32, so the loop stopped
      // at 16, `n % wg_n == 0` was satisfied, and an ILLEGAL tile was dispatched.
      // The driver did not complain. It returned wrong numbers.
      //
      // HOW IT WAS FOUND, since the route matters more than the line: the shape
      // bisect over the decode N values came back with all seven arms BYTE
      // IDENTICAL and clean -- which is the signature of a knob that did not
      // engage, so it was checked, and the kernel had run. The seven arms were
      // clean because each one admitted a SINGLE n, and the guilty shape was not
      // among the seven I had enumerated: n=48 never appeared in the histogram I
      // read the list from. Restricting to n=48 alone then reproduced the defect
      // (step 1 NMSE 0.0264) while n=5120 alone stayed clean (2.66e-4).
      //
      // ⇒ So the null result was real AND misleading at the same time: the defect
      // needed one shape that my enumeration had missed, and the thing that
      // pointed at it was the arms being identical to EACH OTHER, not their
      // values.
      const uint32_t wg_gran_n = (wg_inv == 256) ? 32u : 16u;
      uint32_t wg_n = kWgN;
      // The spill buffer is the hard limit: wg_m * wg_n floats must fit 32*128.
      while (wg_n > wg_gran_n && wg_m * wg_n > 128u * 128u) wg_n /= 2;
      // And N must divide the shape. Below the granularity we DECLINE rather than
      // clamp -- `fits` requires n % wg_n == 0, which n=48 now fails at wg_n=32.
      while (wg_n > wg_gran_n && n % wg_n != 0) wg_n /= 2;
      // SHAPE BISECT (VT_VULKAN_COOPMAT2_ONLY_N=<n>, default off).
      //
      // The defect is now known to be in DECODE, not prefill: at c=32 the decline
      // log shows every prefill shape (m=1037) refused for M, so only the m=32
      // decode shapes reached this kernel. And decode step 0 measures a clean
      // 1.21e-4 while step 1 jumps 300x to 3.6e-2 and then grows smoothly to
      // 0.385 by step 5 -- with the token ids identical throughout.
      //
      // A clean first step followed by compounding growth, at constant token
      // sequence, points at PERSISTENT STATE rather than at the logits path: this
      // model carries a GDN recurrent state across steps, so a wrong matmul
      // feeding that state poisons every step after it while the step that wrote
      // it still looks fine.
      //
      // The decode shapes differ only in n (48, 1024, 5120, 6144, 10240, 12288,
      // 34816, 248320), so restricting the tactic to ONE n at a time says which
      // one carries it. That is a bisect over a list of nine, not a hunt.
      static const int64_t kOnlyN = [] {
        const char* v = std::getenv("VT_VULKAN_COOPMAT2_ONLY_N");
        return v != nullptr ? std::strtol(v, nullptr, 10) : 0;
      }();
      // ⭐ AND THE FINAL WORD IS THE DEVICE'S, NOT THIS FILE'S. Everything above
      // chooses a tile; coopmat2_wg_supports() checks that the chosen
      // (M, N, K, invocations) is a configuration this device actually reports,
      // by reading
      // vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV.
      //
      // The rules above were written from what THIS card reports, and an
      // external review found one of them already wrong (an override paired
      // M=16 with 256 invocations, which this card does not offer). A rule
      // transcribed from one device is a guess about every other device, and
      // this tactic is an NVIDIA-only optional path over a generic one -- so
      // when the transcription and the device disagree, the device wins and the
      // generic subgroup kernel takes the shape.
      const bool device_offers =
          cmctx.coopmat2_wg_supports(wg_m, wg_n, wg_k, wg_inv);
      const bool fits = wg_m != 0 && n % wg_n == 0 && k % wg_k == 0 &&
                        device_offers && (kOnlyN == 0 || n == kOnlyN);
      if (fits) {
        // Seven values now: the spill size is a specialization constant (id 5),
        // so each variant allocates only its own tile. The module declares ids
        // {0,1,2,3,4,5,9} in that order and the runtime checks the count, so a
        // mismatch is loud rather than a silently wrong workgroup size.
        // VT_VULKAN_COOPMAT2_DIRECT_STORE=1 replaces the shared-memory epilogue
        // with coopMatStoreTensorNV. Off by default until the bit-identity gate
        // says the bf16 conversion matches F32ToBF16; see the shader constant.
        static const uint32_t kDirectStore = [] {
          const char* v = std::getenv("VT_VULKAN_COOPMAT2_DIRECT_STORE");
          return (v != nullptr && v[0] == '1') ? 1u : 0u;
        }();
        // VT_VULKAN_COOPMAT2_GROUP_M groups row-blocks so B is not re-streamed
        // once per row-block. See the shader constant for the working-set
        // arithmetic that predicts which shapes it helps.
        // Default 32, measured. The sweep is monotonic in the B working set,
        // which is what the L2 model predicts and what makes it more than a
        // tuned constant:
        //     GROUP   n=34816 (B 357 MB)   k=17408 (178 MB)   n=10240 (105 MB)
        //     1            1.00x                1.00x              1.00x
        //     8            1.34x                1.09x              1.00x
        //     32           1.46x                1.18x              1.01x
        // Whole prefill GEMM 1.26x at 32. The shape whose B already fits gains
        // nothing and loses nothing, so there is no cell to trade away and no
        // shape gate is needed -- unlike the GDN tile and the attention split,
        // where the sweep had a crossover.
        //
        // 32 rather than higher because m=2048 gives 32 row-blocks: at 32 the
        // whole M dimension is one group and B is streamed once, which is the
        // floor. A larger value cannot help and the host clamps it anyway.
        static const uint32_t kKUnrollWg = [] {
          const char* v = std::getenv("VT_VULKAN_COOPMAT2_KUNROLL");
          if (v == nullptr) return 1u;
          const long u = std::strtol(v, nullptr, 10);
          return (u == 1 || u == 2 || u == 4 || u == 8) ? static_cast<uint32_t>(u) : 1u;
        }();
        static const uint32_t kGroupM = [] {
          const char* v = std::getenv("VT_VULKAN_COOPMAT2_GROUP_M");
          if (v == nullptr) return 32u;
          const long g = std::strtol(v, nullptr, 10);
          return (g >= 1 && g <= 64) ? static_cast<uint32_t>(g) : 32u;
        }();
        const uint32_t wg_spec[10] = {kBT ? 1u : 0u, DtypeCode(out.dtype),
                                     wg_m, wg_n, wg_k, wg_m * wg_n, wg_inv,
                                     kDirectStore, kGroupM, kKUnrollWg};
        const uint32_t wg_groups =
            // ceil on M, to match the shader's ragged-M block count.
            static_cast<uint32_t>(((m + wg_m - 1) / wg_m) * (n / wg_n));
        RecordCoopmatShape("wg", m, k, n, wg_groups);
        // Per-SHAPE time for the prefill GEMM, same stats-only label as GEMV.
        // The prefill/decode split measured 2026-09-07 puts 62-79% of the gap to
        // llama.cpp in PREFILL, and this kernel is 69.3% of prefill -- and its
        // efficiency had never been measured, because every "85-91% of peak"
        // figure in this campaign belongs to the GEMV path and is a BANDWIDTH
        // number. This one is compute-bound at m=1024; its roof is FLOPs.
        std::string wlbl;
        if (ShapeStatsOn()) {
          char b3[64];
          std::snprintf(b3, sizeof(b3), "vt_matmul_coopmat_wg k=%lld n=%lld", (long long)k,
                        (long long)n);
          wlbl = b3;
        }
        Go("vt_matmul_coopmat_wg", bind, p, wg_groups, wg_spec, 10, wlbl);
        return;
      }
      // Say why, once per distinct reason -- the same contract the other
      // predicates in this file carry, because a tactic that silently routes
      // everything back to the slower path is indistinguishable from one that is
      // not wired up at all.
      static std::mutex wg_mu;
      static std::set<std::string> wg_seen;
      // NAME THE TILE THAT WAS ACTUALLY REQUIRED, not a constant. External review
      // 2026-09-06 (F1): `fits` tests `k % wg_k == 0` and wg_k has defaulted to 64
      // since the K sweep, but this message still said "not a multiple of 16" --
      // so K=32 was refused for failing a test it passes, and the message sends
      // the reader looking for a bug that is not there. A diagnostic that names
      // the wrong constant is worse than no diagnostic, because it is believed.
      char why_buf[128];
      if (wg_m != 0 && n % wg_n == 0 && k % wg_k == 0 && !device_offers) {
        std::snprintf(why_buf, sizeof(why_buf),
                      "the device reports no %ux%ux%u workgroup config at %u invocations",
                      wg_m, wg_n, wg_k, wg_inv);
      } else if (wg_m == 0) {
        std::snprintf(why_buf, sizeof(why_buf), "M=%lld is below the smallest M tile (16)",
                      (long long)m);
      } else if (n % wg_n != 0) {
        std::snprintf(why_buf, sizeof(why_buf), "N=%lld is not a multiple of the N tile %u",
                      (long long)n, wg_n);
      } else {
        std::snprintf(why_buf, sizeof(why_buf), "K=%lld is not a multiple of the K tile %u",
                      (long long)k, wg_k);
      }
      const char* why = why_buf;
      std::string key = std::string(why) + "|" + std::to_string((long long)m) + "," +
                        std::to_string((long long)k) + "," + std::to_string((long long)n);
      std::lock_guard<std::mutex> g(wg_mu);
      if (wg_seen.insert(key).second) {
        std::fprintf(stderr, "[vt vulkan] COOPMAT2-WG declined: %s  (m=%lld k=%lld n=%lld)\n",
                     why, (long long)m, (long long)k, (long long)n);
        std::fflush(stderr);
      }
    }

    const int64_t bm = static_cast<int64_t>(tile_m) * cm_mr;
    const int64_t bn = static_cast<int64_t>(tile_n) * cm_nr;
    const uint32_t tiles =
        static_cast<uint32_t>(((m + bm - 1) / bm) * ((n + bn - 1) / bn));
    // SHAPE HISTOGRAM (VT_VULKAN_SHAPE_STATS=1, default off). The per-kernel
    // dispatch histogram says this shader owns 94.4% of GPU time at c=32 across
    // 222,128 calls, which is where the search has to go next -- but "one kernel"
    // is not an optimisation target, a SHAPE is. Tile size, K-blocking and
    // cooperative-matrix configuration are all shape-dependent, so reasoning
    // about any of them before knowing the m/k/n mix is guessing.
    // Counted here rather than in the dispatch path because m/k/n are local
    // here and opaque push-constant bytes there, and because decorating the
    // dispatch NAME would split the pipeline cache per shape.
    RecordCoopmatShape("subgroup", m, k, n, tiles);
    Go("vt_matmul_coopmat", bind, p, tiles, spec, 7);
    return;
  }

  if (GemvMatmulUsable(kBT, k, m)) {
    // ONE WORKGROUP PER `rows` OUTPUT ELEMENTS -- not FlatGroupCount, which would
    // divide the element count by the workgroup size and put the whole K
    // reduction back on a single lane. The workgroup cooperates on its elements.
    const uint32_t rows = GemvRows(m, n);
    uint32_t groups = static_cast<uint32_t>((m * n + rows - 1) / rows);
    // VT_VULKAN_GEMV_UNROLL=1 forces the un-unrolled body, for the same-binary A/B.
    static const uint32_t kUnroll = [] {
      const char* v = std::getenv("VT_VULKAN_GEMV_UNROLL");
      return (v != nullptr && std::strcmp(v, "1") == 0) ? 1u : 4u;
    }();
    // VT_VULKAN_GEMV_REDUCE=1 selects the subgroup reduction. Default 0 keeps the
    // shared-memory halving tree, which is bit-identical to every earlier build.
    //
    // ⚠️ THE REJECTION BELOW WAS MEASURED ON A BROKEN BUILD AND IS RETRACTED.
    // Kept off, but for a completely different and much weaker reason.
    //
    // What the gating run originally recorded, 27B bf16, c=16, teacher-forced:
    //
    //     step   0      1      2      3      4      5
    //     nmse   0    0.65   0.223  0.289  0.372  0.282
    //
    // against a 5e-4 bar -- roughly 1300x over, and read at the time as "the
    // variant is numerically wrong". ⇒ It was not. Those numbers were produced
    // by THIS SHADER'S OWN BUG: the subgroup path wrote accumulator slots 4..7
    // and never folded or stored them, so four of every eight rows were wrong
    // (fixed in 0e652a513, "GEMV subgroup 規約漏折 slots 4..7"). The verdict was
    // written before that fix and never revisited, so a defect was rejecting
    // the feature that contained it.
    //
    // ⭐ Caught by external review 2026-09-07, which noticed this comment still
    // cites a pre-fix verdict, and re-measured on the SAME harness:
    //
    //     step        0      1        2        3        4        5
    //     nmse        0   1.06e-4  3.49e-5  8.68e-5  3.27e-5  2.41e-5
    //     aggregate  4.08e-05, max_abs 0.158, rel_max 0.0061
    //
    // ⇒ Under the 5e-4 bar with 12x of margin. The correctness objection is gone.
    //
    // WHAT SURVIVES is the speed half, and only just. Interleaved, three rounds:
    //     c=16   0.2499/0.2510/0.2512  ->  0.2472/0.2481/0.2477   1.011-1.014x
    //     c=1    0.0719/0.0723         ->  0.0719/0.0719          1.000-1.006x
    // A consistent ~1.2% at c=16 and nothing at c=1. That does not justify moving
    // a default on the path that feeds the sampler, so it stays off -- but "no
    // measurable gain" is a far weaker reason than "numerically wrong", and the
    // difference matters: the subgroup path is NOT disqualified, so the barrier
    // -reduction variants that build on it are open questions rather than dead
    // ones.
    //
    // Step 0 still reads exactly 0 because step 0 is the PREFILL output and
    // prefill never reaches this kernel -- the blindness teacher forcing was
    // added to remove.
    static const uint32_t kReduce = [] {
      const char* v = std::getenv("VT_VULKAN_GEMV_REDUCE");
      return (v != nullptr && std::strcmp(v, "1") == 0) ? 1u : 0u;
    }();
    // VT_VULKAN_GEMV_BATCH -- the m > 1 axis, DEFAULT OFF (1 = old behaviour).
    //
    // At m > 1 each workgroup owns one (batch row, weight row) pair, so the SAME
    // weight row is fetched m times and this bandwidth-bound kernel scales with m.
    // Measured 27B bf16, same binary, c=1 vs c=4: dispatch count identical
    // (115584 vs 116724), submit count identical (515 vs 520) -- so batching
    // itself works -- yet ms/call went 0.0783 -> 0.2531. A 3.23x rise for 4x the
    // rows is the weight traffic, not the arithmetic.
    //
    // With BATCH the workgroup owns one weight row and B batch rows, so the weight
    // is read once per block. ROWS and BATCH are mutually exclusive: GemvRows
    // already folds to 1 when m != 1, which is exactly when BATCH applies.
    // DEFAULT 4 as of the sweep below; VT_VULKAN_GEMV_BATCH=1 restores the old
    // one-row-per-workgroup kernel for a same-binary A/B.
    //
    // 27B bf16, one binary, token ids compared per cell:
    //     c    BATCH=1   BATCH=4   ratio   ids
    //     1    26.3      26.3      1.00x   identical   (regression: m == 1 untouched)
    //     2    31.9      48.0      1.51x   identical
    //     4    33.8      81.9      2.42x   identical   (llama.cpp Vulkan: 86.9)
    //     8    35.4      89.7      2.54x   identical
    //     16   84.0      85.0      1.01x   identical   (coopmat still owns this)
    //
    // Bit-identical at every cell, which is the bar this kernel has to clear: it
    // feeds the sampler, where one low bit changes a token. The earlier 0.13%
    // divergence was a GLSL dot() in the quad path, not a property of batching.
    // 0 means AUTO -- distinct from an explicit 4. The first version made the
    // default 4 and then let any smaller explicit value cap the per-m choice; with
    // the default itself being 4, the cap fired on the unset path too and the
    // automatic 8 at m >= 8 never happened. The measurement caught it because the
    // unset leg matched the capped leg to two decimals instead of matching the
    // BATCH=8 leg taken ten minutes earlier.
    static const uint32_t kBatch = [] {
      const char* v = std::getenv("VT_VULKAN_GEMV_BATCH");
      if (v == nullptr) return 0u;   // auto
      const long n2 = std::strtol(v, nullptr, 10);
      return (n2 == 2 || n2 == 4 || n2 == 8) ? static_cast<uint32_t>(n2) : 1u;
    }();
    // BLOCK SIZE BY m, not one global default. Measured 27B bf16, both cells,
    // token-identical either way:
    //     c=4   BATCH=4 17.94 tok/s/stream (0.0863 ms/call)
    //           BATCH=8 17.33             (0.0923)   3.4% WORSE
    //     c=8   BATCH=4 11.43             (0.1639)
    //           BATCH=8 13.71             (0.1309)   20% better
    // Eight slots halve the weight re-reads at m >= 8, but at m == 4 half of them
    // idle while the register pressure is paid anyway. A knob that helps its own
    // cell and quietly costs the neighbouring one is not an improvement, so the
    // host picks the largest block that m actually fills.
    //
    // The 20% is also well short of the 2x the halved re-reads predict, which is
    // the register cost the shader's own comment warns about: 8 slots x 4
    // accumulators is 32 floats live. Recorded as a miss, not smoothed over.
    uint32_t batch = 1u;
    if ((kBatch == 0u || kBatch > 1u) && m > 1 && rows == 1u) {
      batch = (m >= 8) ? 8u : ((m >= 4) ? 4u : 2u);
      // Only an EXPLICIT request caps the automatic choice; kBatch == 0 is auto.
      if (kBatch != 0u && kBatch < batch) batch = kBatch;
    }
    // VT_VULKAN_GEMV_BROWS=2 turns on the second tile axis. Traffic per output is
    // 1/R + 1/B, so R=2,B=4 moves 1.5x less than R=1,B=8 for the same eight
    // accumulators.
    //
    // ★ THE A/B IS NOW ON RECORD, AND IT SAYS NO. Default stays 1.
    //
    // It could not be run before, and the reason is in the sweep table above: at
    // m=16 this kernel was not even reached, because coopmat owned that handover.
    // The handover moved to GEMV on 2026-09-04 (coopmat measures 1.84x slower on
    // the m=16 GEMM), which is what made the cell reachable at all.
    //
    // 27B bf16, c=16, interleaved, GPU-time metric rather than throughput:
    //     brows   GEMV kernel ms      ms/call     tok/s
    //     1       22847 / 23274       0.2798      49.33 / 49.68
    //     2       28419 / 26720       0.3481      44.46 / 46.99
    //     ->      1.15-1.24x SLOWER on 1.5x LESS traffic
    //
    // ★ RE-MEASURED 2026-09-06 ON A CLEAN ARM, and it still says no. The grid
    // below was overwriting the BROWS grid (VK-PREFILL-004), so every number in
    // the table above was taken with twice the workgroups BROWS needs. Fixed,
    // then re-run interleaved, three rounds, same 27B c=16 shape:
    //     brows   GEMV ms/call (r1 / r2 / r3)
    //     1       0.2398 / 0.2437 / 0.2471
    //     2       0.2801 / 0.2774 / 0.2809
    //     ->      1.14x SLOWER, and token ids are IDENTICAL to the R=1 arm
    //
    // ⇒ The confound was real and it was not the explanation. It accounts for
    // the top of the old 1.15-1.24x range and nothing more. The verdict stands
    // and is now clean, which is a better thing to own than the old one was.
    //
    // The traffic model is not wrong about the traffic; it is wrong about what this
    // kernel is limited by.
    //
    // ⚠️ AN EARLIER VERSION OF THIS NOTE SAID "so the family is LATENCY-bound, and
    // the next attempt should raise occupancy". That was written on this one
    // negative result and a sharper test refuted it the same day. Pipeline stats
    // -- now taken WITH the capture flag the query has always required, and
    // returning 64 / 47 / 40 unchanged, so the contract defect did not move
    // these particular numbers -- put this kernel at 64 registers and 4 KB of shared memory, so registers cap
    // residency and shared memory does not -- which predicts that LOWERING the
    // batch block should help, since the shader spends 8 slots x 4 accumulators.
    // Lowering it raises occupancy exactly as predicted and loses, monotonically:
    //
    //     batch   registers   GEMV kernel ms   ms/call
    //     8 (auto)   64        20859 / 21639   0.2555 / 0.2650
    //     4          47        26898           0.3294   1.27x SLOWER
    //     2          40        42716           0.5232   2.03x SLOWER
    //
    // ⇒ So it is not occupancy either. Both single-cause stories are dead, and
    // what survives has to explain BOTH results at once, because they pull in
    // opposite directions: along the B axis less traffic is faster, along the R
    // axis less traffic is slower.
    //
    // ⇒ ★ THE ACCESS SHAPE, not the byte count and not the residency. B-axis reuse
    // holds ONE contiguous weight stream and serves more batch rows from it, which
    // stays perfectly coalesced. R=2 halves the re-reads by keeping TWO weight rows
    // live, which doubles the number of concurrent streams and splits the
    // coalescing. Same bytes saved, opposite outcome, and stream count is the
    // variable that differs.
    //
    // ⇒ So the next attempt should widen or lengthen the single stream -- more
    // batch rows per fetch, wider per-thread loads -- and should NOT introduce a
    // second one. The 205 GB/s against 1792 GB/s of peak is still unexplained by
    // any of the three models, and is the honest open question here.
    static const uint32_t kBRows = [] {
      const char* v = std::getenv("VT_VULKAN_GEMV_BROWS");
      return (v != nullptr && std::strcmp(v, "2") == 0) ? 2u : 1u;
    }();
    // THE ACCUMULATOR BUDGET IS EIGHT SETS, so R * B <= 8. The traffic model wrote
    // that constraint down and this line is where it is actually enforced: with
    // batch auto-selecting 8 at m >= 8, brows = 2 asked for sixteen sets and the
    // reduction folded past the end -- half the sequences produced garbage from
    // the first token. A constraint that lives only in a comment is not enforced.
    //
    // R=2,B=4 is also the better point of the two that fit: traffic per output is
    // 1/R + 1/B, so 0.75 against 1.125 for R=1,B=8.
    //
    // AND IT IS GATED ON EFFECTIVE PACK==2, not on the request. External review
    // 2026-09-06 (VK-PREFILL-003): the shader's R x B body is guarded by
    // `VT_MM_BROWS > 1u && VT_MM_PACK == 2u`, so when pack downgrades -- an f32
    // operand, a K or byte offset that is not 4-element aligned -- the else
    // branch fills eight slots as EIGHT BATCH ROWS while the epilogue reads them
    // as (b, r) pairs. Wrong values, no crash. GemvPack applies those downgrades
    // silently by design, so asking it here is the only way to know which body
    // will actually run.
    const uint32_t pack_eff = GemvPack(a, b, k, a_off, b_off);
    uint32_t brows =
        (batch > 1u && kBRows == 2u && n % 2 == 0 && pack_eff == 2u) ? 2u : 1u;
    if (brows > 1u && batch > 4u) { batch = 4u; }
    if (brows > 1u && batch * brows > 8u) { brows = 1u; }
    const uint32_t spec[9] = {DtypeCode(a.dtype), DtypeCode(b.dtype),
                              DtypeCode(out.dtype), kUnroll, rows,
                              pack_eff, kReduce, batch,
                              brows};
    MatmulParams p{static_cast<uint32_t>(m), static_cast<uint32_t>(n),
                   static_cast<uint32_t>(k), a_off, b_off, out_off};
    // TWO EXTRA BINDINGS, and they are the same two buffers again. This shader
    // declares a 64-bit view of each input operand for the widest packed load;
    // a shader must declare every descriptor the host writes, so both views are
    // bound for every variant and the narrower ones simply never read them. The
    // Binder pushed a and b at bindings 0-3 already, so the aliases have to be
    // appended AFTER the output pair to land on 6 and 7.
    Binder wide = bind;
    wide.AddAlias(a, "matmul: a (64-bit view)");
    wide.AddAlias(b, "matmul: b (64-bit view)");
    // GRID. ROWS mode covers m*n elements in blocks of `rows`. BATCH mode covers
    // one COLUMN per workgroup across a block of B rows, so the count is
    // ceil(m/B) * n -- getting this wrong under-dispatches and leaves part of the
    // output untouched rather than failing loudly.
    //
    // ⚠️ AND THE COLUMN COUNT IS ceil(n/R), NOT n. External review 2026-09-06
    // (VK-PREFILL-004): brows set this grid forty lines earlier and THIS line
    // then overwrote it, unconditionally, because brows > 1 implies batch > 1.
    // So every BROWS=2 run dispatched twice the workgroups it needed and the
    // extra half early-returned after recomputing its index.
    //
    // ⇒ WHICH MEANS THE BROWS NEGATIVE ABOVE WAS MEASURED ON A CONFOUNDED ARM.
    // It is not thereby wrong -- launch overhead for groups that exit early is
    // not obviously worth 1.15-1.24x -- but it is no longer a clean result, and
    // the table above says the axis is the ACCESS SHAPE, which this does not
    // touch. Re-measured below rather than argued.
    if (batch > 1u) {
      const uint32_t jblocks = static_cast<uint32_t>((n + brows - 1) / brows);
      groups = static_cast<uint32_t>((m + batch - 1) / batch) * jblocks;
    }
    // GEMV SHAPES, under the same VT_VULKAN_SHAPE_STATS as the coopmat ones.
    //
    // Listed as an open item since the RFG-1 audit and still missing when the
    // WP0 breakdown made it the thing that matters: this kernel is 428 of the
    // 913 dispatches in a decode step and 84.6% of its GPU time, running at
    // ~80% of peak bandwidth -- and there was no way to see WHICH shapes make
    // up the remaining 20%. The coopmat path has had a histogram for weeks;
    // this one, the busier of the two at decode, had none.
    //
    // `tiles` carries the workgroup count so a shape's grid is visible beside
    // its call count, which is what the occupancy findings elsewhere in this
    // file all turned on.
    RecordCoopmatShape(brows > 1u ? "gemv-r2" : "gemv", m, k, n, groups);
    // Per-SHAPE time, not just per-shape call count. The histogram said n=48 is
    // 21.5% of decode's GEMV calls while carrying 0.1% of its bytes; whether
    // that costs anything is a different question and needs the clock.
    std::string lbl;
    if (ShapeStatsOn()) {
      char b2[64];
      std::snprintf(b2, sizeof(b2), "vt_matmul_vec k=%lld n=%lld", (long long)k, (long long)n);
      lbl = b2;
    }
    Go("vt_matmul_vec", wide, p, groups, spec, 9, lbl);
    return;
  }

  // Scalar tactic: the portable reference, and the only one whose accumulation
  // ORDER matches the CPU kernel's.
  //
  // COLUMN BLOCKING IS GATED TO bt == 0, and that gate is structural, not a
  // heuristic. It exists to give a workgroup a CONTIGUOUS run of b, and b is only
  // contiguous along the output columns in the [K,N] orientation. In [N,K]
  // contiguity runs along K instead -- which is exactly why MatmulBT at M=1 has
  // its own kernel -- so blocking columns there would stride every load and make
  // this strictly worse.
  //
  // It is NOT gated on m. Each of a lane's accumulators owns one output element
  // and runs the whole K reduction sequentially, so the result is bit-identical to
  // the flat body for every shape; there is no numeric tier to protect by
  // restricting it to the decode shape. Correctness of that claim is gated by a
  // bitwise memcmp of the two arms, not by a tolerance.
  // ⭐ THE VOCABULARY PROJECTION GETS 2, AND THE DEFAULT 4 WAS NEVER MEASURED ON IT.
  //
  // This kernel's only caller on this model is lm_head -- m=1, k=5120,
  // n=248320, which is 2.54 GB of bf16 weights read once. That gives it a hard
  // floor: 2.54 GB / 1.79 TB/s = 1.42 ms. Swept, same binary, all arms
  // bit-identical (verified by token ids, as the comment below has always
  // claimed):
  //
  //     ncols   ms/call   achieved      of peak
  //     1        2.0611   1234 GB/s       69%
  //     2        1.5307   1661 GB/s       93%   <- 1.08x off the physical floor
  //     4        2.4231   1049 GB/s       59%   <- the default
  //     8        2.5930    981 GB/s       55%
  //
  // So the shipped default was the second WORST arm for the one shape that
  // reaches this kernel in decode, and 1.58x off the best. The knob existed and
  // this shape had never been put through it -- our own "a backend's default is
  // not its capability", turned on our own code.
  //
  // Narrow on purpose. A vocabulary projection is a recognisable shape class in
  // any decoder -- one row against a very wide, very tall weight -- so the gate
  // names that class rather than raising the global default on the strength of
  // a single shape. Everything outside it keeps the default, because everything
  // outside it is unmeasured: no other shape reached this kernel on this model,
  // so there was nothing here to test the change against.
  const bool wide_projection = !kBT && m <= 8 && n >= 65536;
  const uint32_t ncols =
      kBT ? 1u
          : (wide_projection && !MatmulColumnsExplicit().load(std::memory_order_relaxed)
                 ? 2u
                 : MatmulColumnsPerLane());

  // WHICH ARM RAN, once per distinct orientation, under VT_VULKAN_DISPATCH_STATS.
  // Column blocking is a PERFORMANCE axis: every arm is bit-identical, so a run's
  // OUTPUT can never say which one served it, and the per-shader histogram only
  // reports the module name -- every arm is `vt_matmul`. A measurement block on
  // this box produced one 215 ms/call leg against three at 12.4 across two runs of
  // each arm; with no marker in the log there was no way to distinguish an arm
  // that had not taken the flag from a run that hit the GB10 residency lottery
  // (replication showed it was the lottery). An A/B whose arms are not
  // self-identifying is not an A/B.
  if (kCoopMatWhy) {
    static std::mutex sseen_mu;
    static std::set<std::string> sseen;
    const std::string key = std::to_string(kBT ? 1 : 0) + "|" + std::to_string(ncols);
    std::lock_guard<std::mutex> g(sseen_mu);
    if (sseen.insert(key).second) {
      std::fprintf(stderr,
                   "[vt vulkan] scalar matmul ARM: bt=%d ncols=%u (first shape m=%lld k=%lld "
                   "n=%lld)\n",
                   kBT ? 1 : 0, ncols, (long long)m, (long long)k, (long long)n);
      std::fflush(stderr);
    }
  }

  // ROW BLOCKING, so one workgroup's loaded b serves several output rows.
  //
  // Measured on lm_head (k=5120, n=248320): ms/call was 1.54 / 2.59 / 6.49 /
  // 13.36 at m = 1 / 2 / 4 / 8, which tracks "b re-read once per row"
  // (1.54 / 3.09 / 6.18 / 12.35) and not "b read once" (1.54 throughout). The
  // blocked body indexed workgroups as (row, column-block), so every row
  // re-streamed the whole 2.5 GB weight. At c=4 that is 4.95 ms of every decode
  // step, and it grows linearly with concurrency.
  //
  // MROWS * NCOLS <= 8 is the accumulator budget they share, enforced here
  // rather than in the shader -- a constraint that lives only in a comment is
  // not enforced, which this file has already learned once on the GEMV R x B
  // axis.
  static const uint32_t kMRowsCap = [] {
    const char* v = std::getenv("VT_VULKAN_MATMUL_MROWS");
    if (v == nullptr) return 8u;
    const long n = std::strtol(v, nullptr, 10);
    return (n >= 1 && n <= 8) ? static_cast<uint32_t>(n) : 8u;
  }();
  // Branch-free so the transposed instantiation, where `ncols` folds to the
  // literal 1, does not turn the guard into a constant condition (/WX).
  const int64_t nc = static_cast<int64_t>(ncols);
  const int64_t budget = nc > 1 ? 8 / nc : 1;
  const uint32_t mrows = static_cast<uint32_t>(
      std::max<int64_t>(1, std::min<int64_t>(std::min<int64_t>(m, budget),
                                             static_cast<int64_t>(kMRowsCap))));
  // Ascending constantID order: a dtype, b dtype, out dtype, orientation, ncols.
  // kunroll is id 5 and mrows is id 6; both are appended at the dispatch below.
  const uint32_t spec[5] = {DtypeCode(a.dtype), DtypeCode(b.dtype), DtypeCode(out.dtype),
                            kBT ? 1u : 0u, ncols};
  MatmulParams p{static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k),
                 a_off, b_off, out_off};
  // ONE WORKGROUP PER COLUMN BLOCK when blocked -- not FlatGroupCount, which
  // divides an ELEMENT count by the workgroup size. Here a workgroup owns
  // kWorkgroupSize*ncols consecutive columns of ONE output row, and the block
  // count is rounded UP because N is not required to be a multiple of that span
  // (248320 is not a multiple of 1024); the shader bounds-checks each column.
  const int64_t span = static_cast<int64_t>(kWorkgroupSize) * ncols;
  const uint32_t groups =
      ncols == 1u
          ? FlatGroupCount(m * n)
          : static_cast<uint32_t>(((m + mrows - 1) / mrows) * ((n + span - 1) / span));
  // VT_VULKAN_MATMUL_KUNROLL=4 puts four K positions in flight in the
  // column-blocked body. Default 1 (the rolled loop) until the A/B is recorded.
  // Bit-identical either way: the unroll hoists loads above the FMAs and leaves
  // the accumulation sequence untouched, which is what the memcmp test asserts.
  static const uint32_t kKUnroll = [] {
    const char* v = std::getenv("VT_VULKAN_MATMUL_KUNROLL");
    return (v != nullptr && std::strcmp(v, "4") == 0) ? 4u : 1u;
  }();
  const uint32_t spec7[7] = {spec[0], spec[1], spec[2], spec[3],
                             spec[4], kKUnroll, mrows};
  Go("vt_matmul", bind, p, groups, spec7, 7);
}

// cpu_ops.cpp:661-672 EmbeddingKernel. One output ELEMENT per invocation.
// The id dtype (i32 vs i64) is a specialization constant rather than a
// per-element branch; see the shader for why only the low 32 bits are read.
void EmbeddingKernel(Queue&, Tensor& out, const Tensor& table, const Tensor& ids) {
  const int64_t t = ids.shape[0], h = table.shape[1];
  if (t == 0 || h == 0) return;
  VT_CHECK(ids.dtype == DType::kI32 || ids.dtype == DType::kI64,
           "vulkan embedding: ids must be i32 or i64");
  Binder bind;
  const uint32_t table_off = bind.Add(table, "embedding: table");
  const uint32_t ids_off = bind.Add(ids, "embedding: ids");
  const uint32_t out_off = bind.Add(out, "embedding: out");
  const uint32_t spec[3] = {DtypeCode(table.dtype), DtypeCode(out.dtype),
                            ids.dtype == DType::kI64 ? 1u : 0u};
  EmbeddingParams p{static_cast<uint32_t>(t), static_cast<uint32_t>(h), table_off, ids_off,
                    out_off};
  Go("vt_embedding", bind, p, FlatGroupCount(t * h), spec, 3);
}

// cpu_sample.cpp:40-56 GreedyArgmaxKernel. ONE INVOCATION PER ROW, because the
// tie-break (strict `>`, so the first maximum wins) is part of the token-exact
// contract and a tree reduction would have to carry the index and break ties
// toward the lower one at every merge. Rows are few at decode; the vocabulary
// scan is the slow axis and is deliberately left for a later change.
// RAW-TENSOR DUMP (VT_VULKAN_LOGITS_DUMP=<path>, default off).
//
// WHY THIS EXISTS. Every correctness check on this backend so far has compared
// TOKEN IDS -- an md5 over what came out after argmax. That gate is asymmetric
// and this campaign has now hit both of its limits in one day: passing is strong
// evidence and failing says nothing about magnitude, because argmax is a
// discontinuous amplifier sitting between the numbers and the hash. A 1e-7
// difference and a catastrophic one hash differently in exactly the same way.
//
// The consequence, concretely: three cells are currently marked "different" with
// no way to say whether the difference matters -- GEMV at m=16, and both coopmat
// and coopmat2 on prefill shapes. "Different" is not a verdict.
//
// This project's own doctrine already says what to do instead: compare the RAW
// OUTPUT TENSOR, before postprocess, with a distance metric against an oracle
// (execution-faithfulness, 2026-07-13; NMSE <= 5e-4 vs the f64 oracle). The LM
// engine never grew that tool. This is it, hooked where the logits are complete
// and nothing has yet collapsed them: immediately before the argmax.
//
// Sampled by a fixed STRIDE rather than a prefix, because a prefix of a 248,320
// wide vocabulary tells you about one corner of it, and the elements have to
// MATCH across runs for a distance to mean anything -- a stride guarantees that
// while a top-k would not. Exact min/max/sum over the full row go alongside, so
// a difference outside the sample is still visible.
//
// The flush is what makes the read valid: this backend batches dispatches, and
// its allocations are host-visible and persistently mapped, so after a flush the
// logits are simply readable at logits.data. It changes TIMING, not values.
void DumpLogitsIfAsked(const Tensor& logits, int64_t n, int64_t v) {
  static const char* path = std::getenv("VT_VULKAN_LOGITS_DUMP");
  if (path == nullptr || logits.data == nullptr) return;
  static std::mutex mu;
  static int step = 0;
  static const int kMaxSteps = [] {
    const char* m = std::getenv("VT_VULKAN_LOGITS_DUMP_STEPS");
    const long x = m != nullptr ? std::strtol(m, nullptr, 10) : 8;
    return (x >= 1 && x <= 1000) ? static_cast<int>(x) : 8;
  }();
  std::lock_guard<std::mutex> g(mu);
  if (step >= kMaxSteps) return;
  VulkanContext::Get().FlushBatch("logits-dump");
  const float* f = static_cast<const float*>(logits.data);
  std::FILE* fp = std::fopen(path, "a");
  if (fp == nullptr) return;
  const int64_t kSamples = 4096;
  for (int64_t r = 0; r < n; ++r) {
    const float* row = f + r * v;
    double lo = row[0], hi = row[0], sum = 0.0;
    for (int64_t j = 0; j < v; ++j) {
      const double x = row[j];
      if (x < lo) lo = x;
      if (x > hi) hi = x;
      sum += x;
    }
    std::fprintf(fp, "step %d row %lld v %lld min %.9g max %.9g sum %.9g", step,
                 static_cast<long long>(r), static_cast<long long>(v), lo, hi, sum);
    const int64_t stride = v > kSamples ? v / kSamples : 1;
    for (int64_t j = 0; j < v; j += stride) std::fprintf(fp, " %.9g", row[j]);
    std::fprintf(fp, "\n");
  }
  std::fclose(fp);
  ++step;
}

// TEACHER FORCING (VT_VULKAN_PIN_TOKEN=1, default off).
//
// WHY IT HAD TO EXIST. The raw-tensor comparator hooks here, before the argmax,
// and its own rule says step 0 is the only unconditionally valid comparison --
// because once the two arms pick different tokens they are generating different
// sequences and every later step compares two different computations.
//
// That rule is correct and it has a consequence nobody had drawn: step 0 is the
// PREFILL output, and prefill runs shapes far above the GEMV handover. So a
// comparator restricted to step 0 is STRUCTURALLY BLIND to every kernel that
// only appears in decode -- which is the GEMV path, the whole coopmat2 workgroup
// path, and anything else selected by a small m.
//
// It showed up as a false pass, which is the dangerous direction: comparing the
// subgroup-reduction GEMV against the shared-memory one gave step 0 NMSE exactly
// 0, bit for bit. That reads as "the kernel is exact" and means "the kernel did
// not run".
//
// WHAT THIS DOES. Before the argmax reads them, overwrite one logit so the argmax
// is forced to a token chosen by STEP NUMBER rather than by value. Both arms then
// walk the same sequence, feed the same inputs to the same layers, and every step
// stays comparable for as long as the run lasts.
//
// The tokens are nonsense and the output text is garbage -- that is the point.
// This is a comparison harness, not a generation mode: what must match between
// the arms is the arithmetic, and holding the path fixed is what isolates it.
//
// The write lands in a persistently mapped host-visible allocation that the
// pending argmax dispatch reads, the same property the dump relies on to read
// them; DumpLogitsIfAsked has already flushed, so the ordering is: GPU writes
// logits -> flush -> we read (dump) -> we write (pin) -> GPU reads (argmax).
void PinTokenIfAsked(const Tensor& logits, int64_t n, int64_t v) {
  static const bool on = [] {
    const char* p = std::getenv("VT_VULKAN_PIN_TOKEN");
    return p != nullptr && p[0] != 0 && p[0] != '0';
  }();
  if (!on || logits.data == nullptr) return;
  static std::mutex mu;
  static int64_t step = 0;
  std::lock_guard<std::mutex> g(mu);
  VulkanContext::Get().FlushBatch("pin-token");
  float* f = static_cast<float*>(logits.data);
  // A stride that is coprime with any plausible vocabulary keeps the forced
  // tokens from landing on one small cycle, so the sequence exercises a spread of
  // embedding rows rather than the same one every step.
  const int64_t pick = (step * 7919) % v;
  for (int64_t r = 0; r < n; ++r) {
    float* row = f + r * v;
    double hi = row[0];
    for (int64_t j = 1; j < v; ++j) if (row[j] > hi) hi = row[j];
    row[pick] = static_cast<float>(hi + 1000.0);
  }
  ++step;
}

void GreedyArgmaxKernel(Queue&, Tensor& token_ids, const Tensor& logits) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  VT_CHECK(logits.dtype == DType::kF32, "vulkan greedy argmax: logits must be f32");
  VT_CHECK(token_ids.dtype == DType::kI64, "vulkan greedy argmax: token_ids must be i64");
  DumpLogitsIfAsked(logits, n, v);
  PinTokenIfAsked(logits, n, v);
  Binder bind;
  const uint32_t logits_off = bind.AddU32Only(logits, "argmax: logits");
  const uint32_t out_off = bind.AddU32Only(token_ids, "argmax: token_ids");
  ArgmaxParams p{static_cast<uint32_t>(n), static_cast<uint32_t>(v), logits_off, out_off};
  // ONE WORKGROUP PER ROW, matching vt_rms_norm's convention -- the shader
  // tree-reduces the vocabulary across the workgroup's lanes. NOT
  // FlatGroupCount(n), which would allot one INVOCATION per row and leave the
  // vocabulary scan serial; at decode n is 1, so that dispatched a single lane
  // and measured 10.03 ms per call.
  Go("vt_greedy_argmax", bind, p, static_cast<uint32_t>(n));
}

// cpu_paged_attn.cpp:52-171 PagedAttentionKernel. ONE WORKGROUP per (query
// token, query head), lanes splitting the head dimension; see the shader for why
// the CPU's three passes become one online-softmax recurrence (its `probs` array
// is one float per key in the window, which a shader cannot allocate).
//
// This is the only kernel in the backend with NO llama.cpp counterpart to port
// from: its Vulkan backend has no paged KV anywhere. The block-table indirection
// and windowing come from the CPU kernel above, the online-softmax skeleton from
// flash_attn.comp's shape.
void PagedAttentionKernel(Queue& q, Tensor& out, const Tensor& query, const Tensor& k_cache,
                          const Tensor& v_cache, const Tensor& block_table,
                          const Tensor& seq_lens, const Tensor& query_start_loc,
                          const PagedAttentionArgs& args) {
  const int64_t num_reqs = seq_lens.shape[0];
  const int64_t total_q = query.shape[0];
  const int64_t hq = query.shape[1], d = query.shape[2];
  const int64_t block_size = k_cache.shape[1];
  const int64_t num_kv_heads = k_cache.shape[2];

  // PER-CALL REFUSAL, not a silent regression. An fp8 KV cache stores 1-byte
  // pages that must be dequantised as Dequant(fp8) * k_scale|v_scale before the
  // f32 softmax (cpu_paged_attn.cpp:79-93). This shader reads f32/f16/bf16 only,
  // so rather than throw -- which would REMOVE a capability the portable
  // reference tier already provides -- it declines through the provider seam and
  // forwards to the next provider down, which is exactly what GetOpFallback is
  // for (op_provider.h:94-100: per-call refusal belongs in the kernel, because
  // GetOp has no shape or dtype to inspect).
  if (args.kv_cache_dtype != vt::Fp8KVCacheDataType::kAuto) {
    auto next = reinterpret_cast<PagedAttentionFn>(
        GetOpFallback(OpId::kPagedAttention, DeviceType::kVULKAN, kNativeProviderName));
    next(q, out, query, k_cache, v_cache, block_table, seq_lens, query_start_loc, args);
    return;
  }

  if (total_q == 0 || hq == 0 || d == 0) return;
  // The shader keeps its accumulator in VT_PA_ACC_MAX slots per lane, one per
  // head-dim element the lane owns. Asserted rather than trusted: overflowing it
  // would write past a local array.
  VT_CHECK(d <= 8 * static_cast<int64_t>(kWorkgroupSize),
           "vulkan paged attention: head dim " + std::to_string(d) +
               " exceeds the per-lane accumulator (8 * workgroup)");
  VT_CHECK(num_kv_heads > 0 && hq % num_kv_heads == 0,
           "vulkan paged attention: query heads must be a multiple of kv heads");

  Binder bind;
  const uint32_t q_off = bind.Add(query, "paged_attn: query");
  const uint32_t k_off = bind.Add(k_cache, "paged_attn: k_cache");
  const uint32_t v_off = bind.Add(v_cache, "paged_attn: v_cache");
  const uint32_t out_off = bind.Add(out, "paged_attn: out");
  const uint32_t bt_off = bind.AddU32Only(block_table, "paged_attn: block_table");
  const uint32_t sl_off = bind.AddU32Only(seq_lens, "paged_attn: seq_lens");
  const uint32_t qsl_off = bind.AddU32Only(query_start_loc, "paged_attn: query_start_loc");

  const int64_t wl = args.window_size.has_value() ? args.window_size->left : -1;
  const int64_t wr = args.window_size.has_value() ? args.window_size->right : -1;

  const uint32_t spec[4] = {DtypeCode(query.dtype), DtypeCode(k_cache.dtype),
                            DtypeCode(v_cache.dtype), DtypeCode(out.dtype)};
  PagedAttnParams p{static_cast<uint32_t>(total_q),
                    static_cast<uint32_t>(hq),
                    static_cast<uint32_t>(d),
                    static_cast<uint32_t>(block_size),
                    static_cast<uint32_t>(hq / num_kv_heads),
                    static_cast<uint32_t>(num_reqs),
                    args.causal ? 1u : 0u,
                    static_cast<int32_t>(wl),
                    static_cast<int32_t>(wr),
                    static_cast<uint32_t>(k_cache.stride[0]),
                    static_cast<uint32_t>(k_cache.stride[1]),
                    static_cast<uint32_t>(k_cache.stride[2]),
                    static_cast<uint32_t>(v_cache.stride[0]),
                    static_cast<uint32_t>(v_cache.stride[1]),
                    static_cast<uint32_t>(v_cache.stride[2]),
                    static_cast<uint32_t>(block_table.stride[0]),
                    static_cast<uint32_t>(block_table.stride[1]),
                    q_off,
                    k_off,
                    v_off,
                    out_off,
                    bt_off,
                    sl_off,
                    qsl_off,
                    args.scale,
                    args.logits_soft_cap};
    // SPLIT-K ACROSS WORKGROUPS, opt-in.
    //
    // The single-pass kernel dispatches total_q * hq workgroups, which for a
    // single-stream decode is 1 * hq -- sixteen on Qwen3-0.6B. Measured on this
    // card, throughput keeps scaling to about 256 workgroups of that shape, so one
    // stream used roughly 6% of the parallelism available, and vt_paged_attn was
    // 49.8% of all GPU time. Splitting inside the workgroup is already at the
    // hardware limit (1024 invocations), so the remaining axis is across
    // workgroups, which needs a place to put the partial results.
    //
    // DEFAULT OFF. VT_VULKAN_ATTN_SPLIT=<n> turns it on with n workgroups per
    // (token, head); unset or 1 keeps the single-pass path byte-for-byte.
    // VT_VULKAN_ATTN_SPLIT pins the split; unset means the shape-aware choice below.
    static const uint32_t kAttnSplitEnv = [] {
    const char* v = std::getenv("VT_VULKAN_ATTN_SPLIT");
    if (v == nullptr) return 0u;
    const long n = std::strtol(v, nullptr, 10);
    return (n >= 1 && n <= 64) ? static_cast<uint32_t>(n) : 0u;
    }();

    const uint32_t units = static_cast<uint32_t>(total_q * hq);

    // ⭐ SPLIT ONLY WHEN THE GRID IS TOO SMALL, and the target is measured.
    //
    // The single-pass kernel dispatches total_q * hq workgroups, which at one
    // decoding stream on this model is 1 * 24 = 24 -- on a card with 188 SMs. It
    // is also the WORST-EFFICIENCY kernel in a decode step by a wide margin:
    // measured 4.36 ms to move 67 MB of KV, which is 15 GB/s against a ~1790
    // GB/s card, i.e. 0.9% of peak, while the weight-streaming kernels beside it
    // run at 80%. Splitting across workgroups is the only axis left, since
    // splitting inside one is already at the 1024-invocation hardware limit.
    //
    // Swept on the 27B, decode, same binary. attn kernel time vs split=1:
    //
    //     C   grid(split=1)   best split   grid at best   gain on attn
    //     1        24              4            96           1.48x
    //     2        48              2            96           1.07x
    //     4        96              1            96           --   (2 costs 0.89x)
    //    16       384              1           384           --   (4 costs 0.67x)
    //
    // ⇒ Every row's optimum lands on ~96 workgroups, and past it the split's
    // extra pass plus the merge cost more than the parallelism returns. C=2 was
    // measured AFTER the rule was written down, as a prediction: it called
    // split=2, and split=2 won with 1 and 4 on either side of it.
    //
    // Whole-step effect at C=1, interleaved three rounds: 1.050-1.060x, saving
    // 1.64-1.73 ms of a ~40.5 ms step. Numerically it is a different reduction
    // order, not a bit-identical rearrangement, so it is gated on the project's
    // 5e-4 bar rather than on token equality: measured nmse 5.51e-05 aggregate
    // (per-step 1.12e-4 / 1.76e-4 / 4.77e-5 / 8.46e-5), and the token ids match.
    //
    // ⚠️ 96 IS FITTED TO ONE CARD AND ONE MODEL, exactly like the GDN tile's 512
    // and for the same reason -- Vulkan exposes no portable SM count. Env
    // override stays authoritative.
    //
    // NOTE: this path allocates a workspace, and until 2026-09-07 growing that
    // workspace freed the old buffer while batches could still reference it
    // (VK-PREFILL-005). Turning this on by default is only safe because that is
    // fixed.
    // The device's subgroup width, read ONCE for this dispatch: both attention
    // kernels size per-lane and per-split state from it, and it is a device fact
    // (32 on NVIDIA, 8 on lavapipe, 64 on AMD) rather than a constant.
    const uint32_t sgsz = VulkanContext::Get().subgroup_size();
    const uint32_t kAttnSplit = [&] {
      if (kAttnSplitEnv != 0u) return kAttnSplitEnv;
      if (units == 0u) return 1u;
      uint32_t want = 1u;
      while (want < 8u && units * want * 2u <= 96u) want *= 2u;
      return want;
    }();
    if (kAttnSplit > 1u) {
    // (m, l, acc[d]) per (unit, split), f32. Never read by the host.
    const size_t par_floats =
        static_cast<size_t>(units) * kAttnSplit * (static_cast<size_t>(d) + 2);
    VulkanContext& wctx = VulkanContext::Get();
      void* ws = wctx.Workspace(par_floats * sizeof(float));
    if (ws != nullptr) {
      // CAPTURE the offsets Add() returns. A tensor is a VIEW into a larger
      // buffer, so its data does not start at byte 0; discarding these and
      // passing zero makes every load read from the head of the allocation
      // instead of from the tensor, which produces plausible-looking garbage
      // rather than a crash. (It did: the first build of this path decoded
      // " dissurgenturgenturgent..." at full speed.)
      Binder sb;
      const uint32_t s_q = sb.Add(query, "paged_attn_split: query");
      const uint32_t s_k = sb.Add(k_cache, "paged_attn_split: k_cache");
      const uint32_t s_v = sb.Add(v_cache, "paged_attn_split: v_cache");
      sb.AddRaw(ws);
      const uint32_t s_bt = sb.AddU32Only(block_table, "paged_attn_split: block_table");
      const uint32_t s_sl = sb.AddU32Only(seq_lens, "paged_attn_split: seq_lens");
      const uint32_t s_qsl = sb.AddU32Only(query_start_loc, "paged_attn_split: query_start_loc");
      // Offsets are recomputed for this binder: it has one fewer 16-bit view
      // than the single-pass one (the partials are raw f32), so reusing the
      // other binder's offsets would silently address the wrong buffer.
      PagedAttnParams sp_p = p;
      sp_p.q_off = s_q;
      sp_p.k_off = s_k;
      sp_p.v_off = s_v;
      sp_p.out_off = 0;  // the workspace IS a dedicated buffer, so base 0 here
      sp_p.bt_off = s_bt;
      sp_p.sl_off = s_sl;
      sp_p.qsl_off = s_qsl;
      const uint32_t sspec4[4] = {DtypeCode(query.dtype), DtypeCode(k_cache.dtype),
                                 DtypeCode(v_cache.dtype), kAttnSplit};
      // Same two device-derived bounds as the single-pass kernel; see its dispatch.
      const uint32_t s_slots =
          sgsz > 0u ? (static_cast<uint32_t>(d) + sgsz - 1u) / sgsz : 8u;
      const uint32_t s_splits = sgsz > 0u ? (1024u / sgsz) : 32u;
      const uint32_t sspec[6] = {sspec4[0], sspec4[1], sspec4[2], sspec4[3],
                                 s_slots,   s_splits};
      Go("vt_paged_attn_split", sb, sp_p, units * kAttnSplit, sspec, 6);

      Binder mb;
      mb.AddRaw(ws);
      const uint32_t m_out = mb.Add(out, "paged_attn_merge: out");
      PagedAttnMergeParams mp{static_cast<uint32_t>(total_q), static_cast<uint32_t>(hq),
                              static_cast<uint32_t>(d), 0u, m_out};
      const uint32_t mspec[2] = {DtypeCode(out.dtype), kAttnSplit};
      Go("vt_paged_attn_merge", mb, mp, units, mspec, 2);
      return;
    }
    }

    // ⭐ MATRIX-FORM PREFILL ATTENTION, opt-in, and gated on the EXACT contract
    // it was written for rather than on a shape heuristic.
    //
    // The scalar kernel below runs the 27B prefill at 3.2 TFLOP/s, about 2.6% of
    // this card's compute peak, because it does a 256-wide scalar dot and a
    // subgroup reduction per (query, key) pair -- a reduction as deep as the
    // useful work. vt_paged_attn_cm2.comp does the same operator as three
    // cooperative-matrix multiplies with the online softmax kept in matrices.
    //
    // Every condition here was MEASURED before the kernel was written, not
    // assumed: the variant histogram reports one contract for the whole hot
    // prefill path (hq=24, causal, no window, no softcap) and the device reports
    // all seven cooperative_matrix2 features. Anything outside it falls through
    // to the scalar kernel, which stays correct for every shape.
    //
    // Bc == block_size is the load-bearing one: our KV is PAGED, and a tensor
    // layout cannot express a page table. One KV step per page turns the block
    // table back into a base offset.
    // DEFAULT ON, measured. Pure prefill, c=4, interleaved three rounds:
    //     attention   518.4 / 521.2 / 515.8 ms  ->  13.6 / 13.4 / 13.7
    //     whole GPU  3026 / 3046 / 3046        -> 2514 / 2500 / 2518
    //     = 37.6-38.9x on the operator, 1.20-1.22x on the entire prefill
    // Numerics against the scalar kernel: nmse 5.91e-05 aggregate (per step
    // 1.21e-4 / 1.39e-4 / 3.09e-5 / 4.82e-5), token ids identical. It is a
    // different reduction order, so the bar is the project's 5e-4, not bitwise.
    // VT_VULKAN_ATTN_CM2=0 forces the scalar kernel back.
    static const bool kAttnCm2 = [] {
      const char* v = std::getenv("VT_VULKAN_ATTN_CM2");
      return v == nullptr || v[0] != '0';
    }();
    const VulkanContext& facx = VulkanContext::Get();
    if (kAttnCm2 && facx.coopmat2_workgroup() && facx.coopmat2_tensor_addressing() &&
        // MEASURED, not assumed: the decline log reported q=0 out=0 k=2 v=2, so
        // this op's contract is f32 query and output with a bf16 KV cache. The
        // first version of this gate asked for bf16 everywhere and never fired
        // -- and its correctness check then compared two runs of the SAME
        // kernel and reported nmse 0, which is why the decline log exists.
        query.dtype == DType::kF32 && k_cache.dtype == DType::kBF16 &&
        v_cache.dtype == DType::kBF16 && out.dtype == DType::kF32 && d == 256 &&
        block_size == 32 && args.causal && !args.window_size.has_value() &&
        args.logits_soft_cap <= 0.0f && total_q > 1 && hq % num_kv_heads == 0) {
      const uint32_t br = 32u;
      const uint32_t fa_spec[3] = {br, 32u, 256u};
      // An upper bound on blocks per request, so a workgroup's rows never
      // straddle two requests. The excess returns before touching memory.
      const uint32_t blocks_ub =
          static_cast<uint32_t>((total_q + br - 1) / br);
      const uint32_t fa_groups =
          static_cast<uint32_t>(num_reqs) * blocks_ub * static_cast<uint32_t>(hq);
      RecordCoopmatShape("attn-cm2", total_q, d, hq, fa_groups);
      Go("vt_paged_attn_cm2", bind, p, fa_groups, fa_spec, 3);
      return;
    }
    if (kAttnCm2) {
      // SAY WHY, once. A tactic that silently declines is indistinguishable from
      // one that is not wired up -- and this file has already been bitten by a
      // correctness gate that compared two runs of the SAME kernel and reported
      // nmse 0 because the new one never dispatched.
      static std::once_flag fa_why;
      std::call_once(fa_why, [&] {
        std::fprintf(stderr,
                     "[vt vulkan] ATTN-CM2 declined: q=%d k=%d v=%d out=%d d=%lld "
                     "block=%lld causal=%d window=%d softcap=%.3f total_q=%lld "
                     "hq=%lld kv=%lld wg=%d tensor=%d\n",
                     static_cast<int>(DtypeCode(query.dtype)),
                     static_cast<int>(DtypeCode(k_cache.dtype)),
                     static_cast<int>(DtypeCode(v_cache.dtype)),
                     static_cast<int>(DtypeCode(out.dtype)), (long long)d,
                     (long long)block_size, args.causal ? 1 : 0,
                     args.window_size.has_value() ? 1 : 0, args.logits_soft_cap,
                     (long long)total_q, (long long)hq, (long long)num_kv_heads,
                     facx.coopmat2_workgroup() ? 1 : 0,
                     facx.coopmat2_tensor_addressing() ? 1 : 0);
        std::fflush(stderr);
      });
    }

    // GQA HEAD TILING WAS BUILT, MEASURED AND REVERTED. See
    // docs/lm-vulkan-attention-roofline.md: six query heads share a kv head, so
    // one workgroup serving all six reads K and V once instead of six times.
    // Traffic fell 6x and the time did not follow -- 9.55 -> 8.88 ms/call at
    // HT=6, best 7.89 at HT=3 -- and the restructure made the HT=1 path itself
    // 1.21-1.34x SLOWER across L. The re-reads were already being served by L2,
    // which is also what the pre-tile numbers were saying and I did not hear:
    // the computed rate EXCEEDED the card's DRAM peak at L=2048.
    //
    // One workgroup per (query token, head) -- NOT FlatGroupCount, which divides by the
    // workgroup size; here the whole workgroup cooperates on one output row.
    // Attention variant histogram, under VT_VULKAN_SHAPE_STATS. The 2026-09-08
    // gap table asks, BEFORE any kernel is written, whether the hot prefill path
    // is ONE combination or several: a matrix-form rewrite that has to cover
    // causal x window x softcap x dtype separately is a different piece of work
    // from one that covers a single variant. The third column packs the variant
    // (hq * 1000 + causal * 100 + window * 10 + softcap) so one histogram row is
    // one contract.
    RecordCoopmatShape(total_q > 1 ? "attn-pre" : "attn-dec",
                       static_cast<long long>(total_q), static_cast<long long>(d),
                       static_cast<long long>(hq) * 1000 +
                           (args.causal != 0 ? 100 : 0) +
                           (args.window_size.has_value() ? 10 : 0) +
                           (args.logits_soft_cap > 0.0f ? 1 : 0),
                       units);
    // SLOTS PER LANE, computed from THIS device rather than assumed. The shader
    // gives each lane a stride-gl_SubgroupSize slice of the head dimension, so it
    // needs ceil(d / subgroup_size) accumulator registers. A 32-wide subgroup at
    // head_dim 256 needs 8, which is what the shader used to hardcode; an 8-wide
    // one needs 32 and used to overflow the array silently.
    const uint32_t acc_slots =
        sgsz > 0u ? (static_cast<uint32_t>(d) + sgsz - 1u) / sgsz : 8u;
    // Split groups per workgroup, from the device rather than assumed. On a
    // 32-wide subgroup this is 32, which is what the shader used to hardcode, so
    // its shared-memory footprint there is unchanged by the portability fix.
    const uint32_t splits = sgsz > 0u ? (1024u / sgsz) : 32u;
    const uint32_t spec6[6] = {spec[0], spec[1], spec[2], spec[3], acc_slots, splits};
    Go("vt_paged_attn", bind, p, units, spec6, 6);
}

// cpu_cache.cpp:33-72 ReshapeAndCacheKernel. Pure BYTE MOVEMENT -- the CPU
// kernel is two memcpys per token and converts nothing -- so the dtype selects
// only the storage WIDTH to copy at, and the gate for it is bit-exactness.
void ReshapeAndCacheKernel(Queue&, const Tensor& k, const Tensor& v, Tensor& k_cache,
                           Tensor& v_cache, const Tensor& slot_mapping) {
  const int64_t num_slots = slot_mapping.shape[0];
  const int64_t block_size = k_cache.shape[1];
  const int64_t n_elems = k_cache.shape[2] * k_cache.shape[3];  // one token's page
  if (num_slots == 0 || n_elems == 0) return;
  VT_CHECK(slot_mapping.dtype == DType::kI64,
           "vulkan reshape_and_cache: slot_mapping must be i64");
  VT_CHECK(k.dtype == k_cache.dtype && v.dtype == v_cache.dtype,
           "vulkan reshape_and_cache: source and cache dtypes must match (this op "
           "moves bytes and converts nothing)");

  Binder bind;
  const uint32_t k_off = bind.Add(k, "reshape_and_cache: k");
  const uint32_t v_off = bind.Add(v, "reshape_and_cache: v");
  const uint32_t kc_off = bind.Add(k_cache, "reshape_and_cache: k_cache");
  const uint32_t vc_off = bind.Add(v_cache, "reshape_and_cache: v_cache");
  const uint32_t sm_off = bind.AddU32Only(slot_mapping, "reshape_and_cache: slot_mapping");

  const uint32_t spec[1] = {k.dtype == DType::kF32 ? 0u : 1u};
  ReshapeAndCacheParams p{static_cast<uint32_t>(num_slots),
                          static_cast<uint32_t>(n_elems),
                          static_cast<uint32_t>(block_size),
                          static_cast<uint32_t>(k_cache.stride[0]),
                          static_cast<uint32_t>(k_cache.stride[1]),
                          static_cast<uint32_t>(v_cache.stride[0]),
                          static_cast<uint32_t>(v_cache.stride[1]),
                          static_cast<uint32_t>(k.stride[0]),
                          static_cast<uint32_t>(v.stride[0]),
                          k_off,
                          v_off,
                          kc_off,
                          vc_off,
                          sm_off};
  Go("vt_reshape_and_cache", bind, p, FlatGroupCount(num_slots * n_elems), spec, 1);
}

// vt::RopeFromCache — the APPLY half of vLLM's rotary split.
// Upstream: rotary_embedding/base.py:160-252, common.py:145-185 @ e24d1b24fe96;
// our reference is cpu_ops.cpp RopeFromCacheKernel (:751-802).
//
// vLLM's RotaryEmbedding builds cos_sin_cache once in __init__ and the forward
// only applies it, so kRopeCosSinCache (the table, built in double) stays on the
// portable tier and this native kernel is the per-token apply. See the shader for
// why that boundary is also the right one numerically.
void RopeFromCacheKernel(Queue& queue, Tensor& qs, Tensor* ks, const Tensor& positions,
                         const Tensor& cache, const RopeArgs& args) {
  // MROPE DECLINES rather than throws. Multimodal RoPE selects a different
  // position AXIS per pair (cpu_ops.cpp:769-771 via MropeAxisForPair, mirroring
  // vLLM mrope.py), which this shader does not implement -- and throwing would
  // REMOVE a capability the portable reference tier already provides. Forwarded
  // through the provider seam, the same per-call refusal fp8 KV uses.
  if (positions.rank == 2) {
    auto next = reinterpret_cast<RopeFromCacheFn>(
        GetOpFallback(OpId::kRopeFromCache, DeviceType::kVULKAN, kNativeProviderName));
    next(queue, qs, ks, positions, cache, args);
    return;
  }

  const int64_t tokens = qs.shape[0];
  const int64_t hq = qs.shape[1];
  const int64_t hk = ks == nullptr ? 0 : ks->shape[1];
  const int64_t half = args.rotary_dim / 2;
  if (tokens == 0 || half == 0 || (hq + hk) == 0) return;
  VT_CHECK(positions.dtype == DType::kI32 || positions.dtype == DType::kI64,
           "vulkan rope_from_cache: positions must be i32 or i64");

  Binder bind;
  const uint32_t q_off = bind.Add(qs, "rope_from_cache: q");
  // Bindings 2/3 are declared by the shader whether or not k exists, and a
  // descriptor a shader statically uses must be valid even on the path that never
  // reads it -- so with hk == 0 they alias q and are dead. Same arrangement the
  // rmsnorm kernel already uses for its optional residual.
  const uint32_t k_off = ks != nullptr ? bind.Add(*ks, "rope_from_cache: k")
                                       : bind.Add(qs, "rope_from_cache: q");
  const uint32_t c_off = bind.Add(cache, "rope_from_cache: cos_sin_cache");
  const uint32_t p_off = bind.AddU32Only(positions, "rope_from_cache: positions");

  const uint32_t spec[5] = {DtypeCode(qs.dtype),
                            ks != nullptr ? DtypeCode(ks->dtype) : DtypeCode(qs.dtype),
                            DtypeCode(cache.dtype),
                            args.is_neox_style ? 1u : 0u,
                            positions.dtype == DType::kI64 ? 1u : 0u};
  RopeFromCacheParams p{static_cast<uint32_t>(tokens),
                        static_cast<uint32_t>(half),
                        static_cast<uint32_t>(args.rotary_dim),
                        static_cast<uint32_t>(hq),
                        static_cast<uint32_t>(hk),
                        static_cast<uint32_t>(qs.stride[0]),
                        static_cast<uint32_t>(qs.stride[1]),
                        static_cast<uint32_t>(ks != nullptr ? ks->stride[0] : 0),
                        static_cast<uint32_t>(ks != nullptr ? ks->stride[1] : 0),
                        q_off,
                        k_off,
                        c_off,
                        p_off};
  Go("vt_rope_from_cache", bind, p, FlatGroupCount(tokens * half * (hq + hk)), spec, 5);
}

// cpu_ops.cpp:2162-2176 QkvSplitKernel. Mirrors vLLM's QKVParallelLinear output
// split (qkv.split([q_size, kv_size, kv_size], dim=-1)); the three widths are
// independent because under GQA k and v are narrower than q. One invocation per
// OUTPUT element across all three destinations, so this is one dispatch.
void QkvSplitKernel(Queue&, Tensor& q_out, Tensor& k_out, Tensor& v_out, const Tensor& qkv) {
  const int64_t t = qkv.shape[0];
  if (t == 0) return;
  const int64_t q_dim = q_out.Numel() / t;
  const int64_t k_dim = k_out.Numel() / t;
  const int64_t v_dim = v_out.Numel() / t;
  VT_CHECK(q_out.dtype == k_out.dtype && k_out.dtype == v_out.dtype,
           "vulkan qkv_split: the three destinations must share a dtype");
  Binder bind;
  const uint32_t src_off = bind.Add(qkv, "qkv_split: qkv");
  const uint32_t q_off = bind.Add(q_out, "qkv_split: q");
  const uint32_t k_off = bind.Add(k_out, "qkv_split: k");
  const uint32_t v_off = bind.Add(v_out, "qkv_split: v");
  const uint32_t spec[2] = {DtypeCode(qkv.dtype), DtypeCode(q_out.dtype)};
  QkvSplitParams p{static_cast<uint32_t>(t),     static_cast<uint32_t>(q_dim),
                   static_cast<uint32_t>(k_dim), static_cast<uint32_t>(v_dim),
                   src_off,                      q_off,
                   k_off,                        v_off};
  Go("vt_qkv_split", bind, p, FlatGroupCount(t * (q_dim + k_dim + v_dim)), spec, 2);
}

// ===========================================================================
// The GDN / conv1d family (BACKEND-VULKAN-GDN). Qwen3.6-27B is a GDN hybrid, so
// before this row every one of these ops fell to the PORTABLE CPU REFERENCE TIER
// on Vulkan — correct, and running on the host against shared memory.
//
// The two RECURRENCES themselves (kGdnPrefill / kGdnDecode) landed in the
// follow-up row BACKEND-VULKAN-GDN-CORE and are at the bottom of this section.
//
// WHAT IS DELIBERATELY NOT HERE, so a later row does not have to re-derive it:
//   * kRopeCosSinCache — the rotary TABLE BUILD, which constructs its angles in
//     `double` (cpu_ops.cpp RopeCosSinCacheKernel). GLSL has no f64 here and
//     emulating it would be a numerics divergence in the one place vLLM itself
//     keeps the work off the device (its RotaryEmbedding builds the cache once in
//     __init__). Leaving it on the host MIRRORS upstream; "implementing" it would
//     be a regression, and the assertion in tests/vt/test_vulkan_backend.cpp says
//     so out loud.
//   * kCausalConv1dFwd — the PREFILL conv. It is the same arithmetic as the
//     update below but its state write-back reads the OLD state row while other
//     tokens of the same sequence are still reading it, so it needs either a
//     per-(sequence, channel) serial invocation over the whole token range or a
//     buffered old row; that is a different dispatch shape, not a wider push
//     block, and it is left for a follow-up rather than guessed at here.
// ===========================================================================

// cpu_ops.cpp:2272-2279 SigmoidGateBf16Kernel. Flat, one invocation per element.
void SigmoidGateBf16Kernel(Queue&, Tensor& out, const Tensor& attn, const Tensor& gate) {
  const int64_t n = out.Numel();
  if (n == 0) return;
  Binder bind;
  const uint32_t a_off = bind.Add(attn, "sigmoid_gate_bf16: attn");
  const uint32_t g_off = bind.Add(gate, "sigmoid_gate_bf16: gate");
  const uint32_t o_off = bind.Add(out, "sigmoid_gate_bf16: out");
  // Only the attention operand varies; `gate` is f32 and `out` is bf16 by the op
  // contract (src/vt/ops.cpp:3327-3334) and are compile-time constants in the
  // shader, so a violation fails the host VT_CHECK rather than silently working.
  const uint32_t spec[1] = {DtypeCode(attn.dtype)};
  SigmoidGateParams p{static_cast<uint32_t>(n), a_off, g_off, o_off};
  Go("vt_sigmoid_gate_bf16", bind, p, FlatGroupCount(n), spec, 1);
}

// cpu_ops.cpp:1210-1235 RmsNormGatedKernel. ONE WORKGROUP PER ROW (the workgroup
// tree-reduces the mean square), not FlatGroupCount — same convention as
// vt_rms_norm.
void RmsNormGatedKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& gate,
                        const Tensor& w, const RmsNormGatedArgs& args) {
  const int64_t d = x.shape[x.rank - 1];
  const int64_t rows = d == 0 ? 0 : x.Numel() / d;
  if (rows == 0 || d == 0) return;
  // The rank-3 gate is a padded-row [T,Hv,D] view of the merged qkvz z slice;
  // rank-2 degenerates to group 1 with outer stride d (cpu_ops.cpp:1218-1219).
  const int64_t gate_group = gate.rank == 3 ? gate.shape[1] : 1;
  const int64_t gate_outer = gate.stride[0];
  Binder bind;
  const uint32_t x_off = bind.Add(x, "rmsnorm_gated: x");
  const uint32_t z_off = bind.Add(gate, "rmsnorm_gated: gate");
  const uint32_t w_off = bind.Add(w, "rmsnorm_gated: weight");
  const uint32_t out_off = bind.Add(out, "rmsnorm_gated: out");
  RmsNormGatedParams p{static_cast<uint32_t>(rows),
                       static_cast<uint32_t>(d),
                       DtypeCode(x.dtype),
                       DtypeCode(gate.dtype),
                       DtypeCode(w.dtype),
                       DtypeCode(out.dtype),
                       args.sigmoid_gate ? 1u : 0u,
                       static_cast<uint32_t>(gate_group),
                       static_cast<uint32_t>(gate_outer),
                       x_off,
                       z_off,
                       w_off,
                       out_off,
                       args.eps};
  Go("vt_rms_norm_gated", bind, p, static_cast<uint32_t>(rows));
}

// Shared geometry of the two state-cache ops (cpu_ops.cpp:1676-1682 and
// :1723-1727 compute it identically). `mid` is the channels/heads per row and
// `cache_row` the row's PHYSICAL width, which exceeds work_row when the conv
// state has been widened for spec-decode rollback.
struct GdnStateGeom {
  int64_t rows, work_row, work_inner, cache_inner, cache_row;
};
GdnStateGeom GdnStateGeometry(const Tensor& working, const Tensor& cache,
                              const Tensor& state_idx) {
  GdnStateGeom g{};
  g.rows = state_idx.shape[0];
  if (g.rows == 0) return g;
  g.work_inner = working.shape[working.rank - 1];
  g.cache_inner = cache.shape[cache.rank - 1];
  g.work_row = working.Numel() / g.rows;
  const int64_t mid = g.work_inner == 0 ? 0 : g.work_row / g.work_inner;
  g.cache_row = mid * g.cache_inner;
  return g;
}

// cpu_ops.cpp:1666-1707 GdnStateGatherKernel.
void GdnStateGatherKernel(Queue&, Tensor& working, const Tensor& cache,
                          const Tensor& state_idx, const Tensor* has_initial_state) {
  const GdnStateGeom g = GdnStateGeometry(working, cache, state_idx);
  if (g.rows == 0 || g.work_row == 0) return;
  Binder bind;
  const uint32_t work_off = bind.Add(working, "gdn_state_gather: working");
  const uint32_t cache_off = bind.Add(cache, "gdn_state_gather: cache");
  const uint32_t idx_off = bind.AddU32Only(state_idx, "gdn_state_gather: state_idx");
  // Binding 5 is always written: a descriptor a shader statically uses must be
  // valid even on the path that never reads it. With his_mode == 0 it aliases
  // state_idx and is dead — the same arrangement vt_rms_norm uses for its
  // optional residual.
  const uint32_t his_off =
      has_initial_state != nullptr
          ? bind.AddByteView(*has_initial_state, "gdn_state_gather: has_initial_state")
          : bind.AddByteView(state_idx, "gdn_state_gather: state_idx");
  uint32_t his_mode = 0;
  if (has_initial_state != nullptr) {
    his_mode = has_initial_state->dtype == DType::kI8 ? 1u : 2u;
  }
  GdnStateGatherParams p{static_cast<uint32_t>(g.rows),
                         static_cast<uint32_t>(g.work_row),
                         static_cast<uint32_t>(g.work_inner),
                         static_cast<uint32_t>(g.cache_inner),
                         static_cast<uint32_t>(g.cache_row),
                         static_cast<uint32_t>(cache.shape[0]),
                         DtypeCode(working.dtype),
                         DtypeCode(cache.dtype),
                         his_mode,
                         work_off,
                         cache_off,
                         idx_off,
                         his_off};
  Go("vt_gdn_state_gather", bind, p, FlatGroupCount(g.rows * g.work_row));
}

// cpu_ops.cpp:1709-1745 GdnStateScatterKernel.
void GdnStateScatterKernel(Queue&, Tensor& cache, const Tensor& working,
                           const Tensor& state_idx) {
  const GdnStateGeom g = GdnStateGeometry(working, cache, state_idx);
  if (g.rows == 0 || g.work_row == 0) return;
  Binder bind;
  const uint32_t cache_off = bind.Add(cache, "gdn_state_scatter: cache");
  const uint32_t work_off = bind.Add(working, "gdn_state_scatter: working");
  const uint32_t idx_off = bind.AddU32Only(state_idx, "gdn_state_scatter: state_idx");
  GdnStateScatterParams p{static_cast<uint32_t>(g.rows),
                          static_cast<uint32_t>(g.work_row),
                          static_cast<uint32_t>(g.work_inner),
                          static_cast<uint32_t>(g.cache_inner),
                          static_cast<uint32_t>(g.cache_row),
                          static_cast<uint32_t>(cache.shape[0]),
                          DtypeCode(working.dtype),
                          DtypeCode(cache.dtype),
                          cache_off,
                          work_off,
                          idx_off};
  Go("vt_gdn_state_scatter", bind, p, FlatGroupCount(g.rows * g.work_row));
}

struct ConvFwdParams {
  uint32_t n, c_dim, k, width, x_rs;
  uint32_t max_t_len;
  uint32_t has_bias, his_is_i8, silu;
  uint32_t out_dt, x_dt, w_dt, bias_dt, st_dt;
  uint32_t out_off, x_off, w_off, bias_off, st_off, qsl_off, his_off;
};

// cpu_ops.cpp CausalConv1dFwdKernel, the GDN PREFILL conv. ONE INVOCATION PER
// (sequence, channel) -- the CPU kernel's own ForRows unit.
//
// Deliberately NOT llama.cpp's ssm_conv.comp shape, which also parallelises over
// tokens (BLOCK_SIZE=32, TOKENS_PER_WG=16). That works there because ggml's
// dataflow is SSA and advancing the conv state is a separate node; this op
// computes outputs from the OLD state and then overwrites that state, so
// splitting tokens across invocations would put those two against each other.
// Copying the old window into a private array first, as the CPU kernel does with
// `old_row`, keeps one invocation self-contained and needs no barrier.
//
// THE STATE ROW STRIDE IS `width`, NOT `state_len`. The update kernel beside
// this one uses state_len (conv_state.shape[2]) because a speculative-decode row
// can be wider than K-1; the FWD reference addresses with width
// (cpu_ops.cpp:1065 `s * c_dim * width`). Following the neighbouring kernel here
// would silently mis-address every state row whenever shape[2] > k-1.
void CausalConv1dFwdKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                           const Tensor* bias, Tensor& conv_state, const Tensor& qsl,
                           const Tensor& his, const CausalConv1dArgs& args) {
  const int64_t c_dim = x.shape[1], k = w.shape[1];
  const int64_t n = conv_state.shape[0];
  if (n == 0 || c_dim == 0) return;
  VT_CHECK(conv_state.dtype == DType::kF32 || conv_state.dtype == DType::kBF16,
           "vulkan causal_conv1d_fwd: conv_state must be f32 or bf16");
  // The shader keeps the carried window in a private array of this size, so a
  // wider kernel would read past it. Declining here sends those shapes to the
  // portable tier, which is correct, instead of computing garbage.
  VT_CHECK(k >= 1 && k - 1 <= 8,
           "vulkan causal_conv1d_fwd: kernel width above the shader's window");
  VT_CHECK(his.dtype == DType::kI8 || his.dtype == DType::kI32,
           "vulkan causal_conv1d_fwd: has_initial_state must be i8 or i32");
  // The two checks the CPU reference makes (cpu_ops.cpp:1051,1054). Not
  // cosmetic: the shader derives its output loop bound from
  // query_start_loc[s+1] - query_start_loc[s], and that subtraction is
  // unsigned, so a non-monotonic table turns into a ~2^32 iteration loop, a
  // dispatch that never completes and a fence that never signals -- a hang with
  // no error message. The shader also guards, but a bad table should be
  // reported here, where it can name itself.
  {
    const int32_t* qslp = qsl.Ptr<int32_t>();
    const int64_t total = x.shape[0];
    VT_CHECK(qslp[0] == 0 && qslp[n] == total,
             "vulkan causal_conv1d_fwd: bad query_start_loc bounds");
    for (int64_t si = 0; si < n; ++si) {
      VT_CHECK(qslp[si + 1] >= qslp[si] && qslp[si] >= 0,
               "vulkan causal_conv1d_fwd: query_start_loc not monotonic");
    }
  }

  Binder bind;
  const uint32_t out_off = bind.Add(out, "causal_conv1d_fwd: out");
  const uint32_t x_off = bind.Add(x, "causal_conv1d_fwd: x");
  const uint32_t w_off = bind.Add(w, "causal_conv1d_fwd: weight");
  const uint32_t bias_off = bias != nullptr ? bind.Add(*bias, "causal_conv1d_fwd: bias")
                                            : bind.Add(w, "causal_conv1d_fwd: weight");
  const uint32_t st_off = bind.Add(conv_state, "causal_conv1d_fwd: conv_state");
  const uint32_t qsl_off = bind.AddU32Only(qsl, "causal_conv1d_fwd: query_start_loc");
  // has_initial_state may be i8, which is not 4-byte aligned, so it goes through
  // the byte view and the shader unpacks it -- the same reasoning the update
  // kernel gives for aliasing a bf16 state through AddByteView.
  const uint32_t his_off = bind.AddByteView(his, "causal_conv1d_fwd: has_initial_state");

  ConvFwdParams p{static_cast<uint32_t>(n),
                  static_cast<uint32_t>(c_dim),
                  static_cast<uint32_t>(k),
                  static_cast<uint32_t>(k - 1),
                  static_cast<uint32_t>(x.stride[0]),
                  static_cast<uint32_t>(x.shape[0]),
                  bias != nullptr ? 1u : 0u,
                  his.dtype == DType::kI8 ? 1u : 0u,
                  args.silu_activation ? 1u : 0u,
                  DtypeCode(out.dtype),
                  DtypeCode(x.dtype),
                  DtypeCode(w.dtype),
                  bias != nullptr ? DtypeCode(bias->dtype) : DtypeCode(w.dtype),
                  DtypeCode(conv_state.dtype),
                  out_off,
                  x_off,
                  w_off,
                  bias_off,
                  st_off,
                  qsl_off,
                  his_off};
  // TOKEN BLOCKS, chosen so the grid reaches the point where this card stops
  // caring, and 1 (the original mapping) whenever it is already there.
  //
  // The measurement behind both the split and the target is in the shader's
  // VT_CONV_BLOCKS declaration: at one sequence this dispatches 80 workgroups
  // on 188 SMs, and 4x the work costs only 7% more time. kConvTargetGroups is
  // the same ~1536 saturation point the GDN tile sweep measured on this card,
  // and carries the same caveat -- fitted to one device, env-overridable, not a
  // statement about hardware in general.
  static const int64_t kConvTargetGroups = [] {
    const char* v = std::getenv("VT_VULKAN_CONV_TARGET_GROUPS");
    if (v == nullptr) return static_cast<int64_t>(1536);
    const long g = std::strtol(v, nullptr, 10);
    return (g >= 1 && g <= 65536) ? static_cast<int64_t>(g) : static_cast<int64_t>(1536);
  }();
  const int64_t base_groups = (n * c_dim + 127) / 128;
  // Never more blocks than there are tokens to give them: past that the extra
  // invocations are empty and only cost launch.
  const int64_t avg_len = n > 0 ? std::max<int64_t>(1, x.shape[0] / n) : 1;
  int64_t blocks = base_groups > 0 ? (kConvTargetGroups + base_groups - 1) / base_groups : 1;
  blocks = std::max<int64_t>(1, std::min<int64_t>(blocks, avg_len));
  const uint32_t conv_spec[1] = {static_cast<uint32_t>(blocks)};
  Go("vt_causal_conv1d_fwd", bind, p,
     FlatGroupCount(n * c_dim * blocks), conv_spec, 1);
}

// cpu_ops.cpp:1081-1127 CausalConv1dUpdateKernel. One invocation per
// (token, channel) — the CPU kernel's own row-chunking unit, and what makes the
// read-old-then-roll safe with no barrier.
void CausalConv1dUpdateKernel(Queue&, Tensor& out, const Tensor& x, const Tensor& w,
                              const Tensor* bias, Tensor& conv_state,
                              const Tensor* conv_state_indices,
                              const CausalConv1dArgs& args) {
  const int64_t batch = x.shape[0], c_dim = x.shape[1], k = w.shape[1];
  if (batch == 0 || c_dim == 0) return;
  // The state is read AND written in place through the dtype-erased pair of
  // views, so a COMPRESSED (bf16) cache needs no caller-side gather/upcast. That
  // is what SupportsCompressedConvState() advertises for this backend
  // (src/vt/ops.cpp CheckConvCommon), and what lets the model's indexed
  // state-I/O path — which hands the kernel the cache itself plus the slot
  // indices — replace two host memcpys per GDN layer per token.
  VT_CHECK(conv_state.dtype == DType::kF32 || conv_state.dtype == DType::kBF16,
           "vulkan causal_conv1d_update: conv_state must be f32 or bf16");
  Binder bind;
  const uint32_t out_off = bind.Add(out, "causal_conv1d_update: out");
  const uint32_t x_off = bind.Add(x, "causal_conv1d_update: x");
  const uint32_t w_off = bind.Add(w, "causal_conv1d_update: weight");
  // Bindings 6/7 alias the weight when there is no bias; see the note above.
  const uint32_t bias_off = bias != nullptr ? bind.Add(*bias, "causal_conv1d_update: bias")
                                            : bind.Add(w, "causal_conv1d_update: weight");
  const uint32_t st_off = bind.Add(conv_state, "causal_conv1d_update: conv_state");
  const uint32_t idx_off =
      conv_state_indices != nullptr
          ? bind.AddU32Only(*conv_state_indices, "causal_conv1d_update: conv_state_indices")
          // Alias the state when there are no indices. AddByteView, not
          // AddU32Only: a bf16 state is only 2-byte aligned, and AddU32Only's
          // 4-byte assertion would reject a perfectly valid tensor on the path
          // where the shader never reads this binding at all (p.has_idx == 0).
          : bind.AddByteView(conv_state, "causal_conv1d_update: conv_state");
  ConvUpdateParams p{static_cast<uint32_t>(batch),
                     static_cast<uint32_t>(c_dim),
                     static_cast<uint32_t>(k),
                     static_cast<uint32_t>(k - 1),
                     static_cast<uint32_t>(conv_state.shape[2]),
                     static_cast<uint32_t>(x.stride[0]),
                     static_cast<uint32_t>(conv_state.shape[0]),
                     bias != nullptr ? 1u : 0u,
                     conv_state_indices != nullptr ? 1u : 0u,
                     args.silu_activation ? 1u : 0u,
                     DtypeCode(out.dtype),
                     DtypeCode(x.dtype),
                     DtypeCode(w.dtype),
                     bias != nullptr ? DtypeCode(bias->dtype) : DtypeCode(w.dtype),
                     DtypeCode(conv_state.dtype),
                     out_off,
                     x_off,
                     w_off,
                     bias_off,
                     st_off,
                     idx_off};
  Go("vt_causal_conv1d_update", bind, p, FlatGroupCount(batch * c_dim));
}

// cpu_ops.cpp:2337-2417 GdnPostConvKernel. ONE WORKGROUP PER (token, head slot)
// over Hk + Hv slots — upstream's own (L, H+HV) grid — not FlatGroupCount: the
// q/k slots tree-reduce an L2 norm across the workgroup's lanes.
void GdnPostConvKernel(Queue&, Tensor& q_out, Tensor& k_out, Tensor& v_out, Tensor& g_out,
                       Tensor& beta_out, const Tensor& conv, const Tensor& araw,
                       const Tensor& braw, const Tensor& a_log, const Tensor& dt_bias,
                       const L2NormArgs& args) {
  const int64_t t = conv.shape[0];
  const int64_t hk = q_out.shape[1], dk = q_out.shape[2];
  const int64_t hv = v_out.shape[1], dv = v_out.shape[2];
  if (t == 0 || hk + hv == 0) return;
  const int64_t key_dim = hk * dk, value_dim = hv * dv;
  Binder bind;
  const uint32_t conv_off = bind.Add(conv, "gdn_post_conv: conv");
  const uint32_t q_off = bind.Add(q_out, "gdn_post_conv: q_out");
  const uint32_t k_off = bind.Add(k_out, "gdn_post_conv: k_out");
  const uint32_t v_off = bind.Add(v_out, "gdn_post_conv: v_out");
  const uint32_t a_off = bind.Add(araw, "gdn_post_conv: araw");
  const uint32_t b_off = bind.Add(braw, "gdn_post_conv: braw");
  // f32 BY CONTRACT (src/vt/ops.cpp:3459-3463), so one binding each rather than a
  // dtype-erased pair whose 16-bit half could never be taken.
  const uint32_t g_off = bind.AddU32Only(g_out, "gdn_post_conv: g_out");
  const uint32_t beta_off = bind.AddU32Only(beta_out, "gdn_post_conv: beta_out");
  const uint32_t alog_off = bind.AddU32Only(a_log, "gdn_post_conv: a_log");
  const uint32_t dtb_off = bind.AddU32Only(dt_bias, "gdn_post_conv: dt_bias");
  // Ascending constantID order: conv dtype, the shared q/k/v dtype, the shared
  // araw/braw dtype.
  const uint32_t spec[3] = {DtypeCode(conv.dtype), DtypeCode(q_out.dtype),
                            DtypeCode(araw.dtype)};
  GdnPostConvParams p{static_cast<uint32_t>(t),
                      static_cast<uint32_t>(hk),
                      static_cast<uint32_t>(dk),
                      static_cast<uint32_t>(hv),
                      static_cast<uint32_t>(dv),
                      static_cast<uint32_t>(key_dim),
                      static_cast<uint32_t>(value_dim),
                      static_cast<uint32_t>(2 * key_dim + value_dim),
                      static_cast<uint32_t>(araw.stride[0]),
                      static_cast<uint32_t>(braw.stride[0]),
                      conv_off,
                      q_off,
                      k_off,
                      v_off,
                      a_off,
                      b_off,
                      g_off,
                      beta_off,
                      alog_off,
                      dtb_off,
                      args.eps};
  Go("vt_gdn_post_conv", bind, p, static_cast<uint32_t>(t * (hk + hv)), spec, 3);
}

// ---------------------------------------------------------------------------
// The two GATED-DELTA RECURRENCES (BACKEND-VULKAN-GDN-CORE). These are not glue:
// they ARE Qwen3.6's linear-attention core, and with the glue above already
// native they were the whole of what the model still ran on the host — a 512
// token prompt spent ~280 s in kGdnPrefill on the reference tier.
//
// The shaders (src/vt/vulkan/shaders/vt_gdn_prefill.comp, vt_gdn_decode.comp and
// the shared vt_gdn_recurrence.glsl) carry the port provenance and the tile
// geometry. What the HOST has to get right is only the grid and the decline.
// ---------------------------------------------------------------------------

// Must equal VT_GDN_BV / VT_GDN_MAX_DK in vt_gdn_recurrence.glsl. Duplicated
// rather than shared because the shader constants are in GLSL and the SPIR-V is
// committed, so nothing can compute one from the other — the gate in
// tests/vt/test_vulkan_backend.cpp exercises a Dv that is NOT a multiple of the
// tile so a drift shows up as wrong numbers there rather than in a model run.
// VT_VULKAN_GDN_BV overrides it (16 / 8 / 4), and the SAME value goes to the
// shader as specialization id 7 -- so host and shader cannot disagree about the
// tile the way the duplicated constant above could. See the constant's
// declaration in vt_gdn_recurrence.glsl for the occupancy measurement that
// makes it worth sweeping.
const int64_t kGdnTileRowsMax = 16;
// VT_VULKAN_GDN_BV pins it (16 / 8 / 4); unset means the shape-aware choice below.
const int64_t kGdnTileRowsEnv = [] {
  const char* v = std::getenv("VT_VULKAN_GDN_BV");
  if (v == nullptr) return static_cast<int64_t>(0);
  const long n = std::strtol(v, nullptr, 10);
  return (n == 4 || n == 8 || n == 16) ? static_cast<int64_t>(n) : static_cast<int64_t>(0);
}();

// SMALLER TILE ONLY WHEN THE GRID IS TOO SMALL TO FILL THE MACHINE, and the
// threshold is measured, not reasoned.
//
// The grid is n * Hv * ceil(Dv/BV) and the recurrence is sequential in t, so at
// one sequence this model dispatches 1 * 48 * 8 = 384 workgroups. Halving the
// tile doubles that -- but it also halves the lanes' share of each row, which
// deepens the reduction tree, so it is a trade and not a free win. Measured on
// the 27B prefill, same binary, BV=16 against BV=8:
//
//     grid(BV=16)   BV16 ms/call   BV8 ms/call   ratio
//     384              0.7372        0.6394      1.15x   <- smaller tile wins
//     768              1.0182        1.1593      0.88x
//     1536             1.8386        2.1641      0.85x
//     1536             1.7194        2.0875      0.82x
//
// A sharp crossover: below ~512 groups the extra parallelism pays, above it the
// deeper reduction does not. Fixing BV=8 as the default would have LOST -- the
// end-to-end c=4 run that first tested it came back 1471 -> 1516 ms of GDN
// time, TTFT 3501 -> 3577. The isolated ms/call and the end-to-end run
// disagreed because they sat on opposite sides of this line.
//
// ⚠️ 512 IS FITTED TO ONE CARD (188 SMs) AND ONE MODEL SHAPE. It is the same
// class of constant as the coopmat2 pairing rules that were transcribed from
// this device and found wrong for another configuration; Vulkan exposes no
// portable SM count to derive it from, so it stays a measured default with an
// env override rather than a claim about hardware in general.
int64_t GdnTileRows(int64_t n, int64_t hv, int64_t dv) {
  if (kGdnTileRowsEnv != 0) return kGdnTileRowsEnv;
  const int64_t groups = n * hv * ((dv + kGdnTileRowsMax - 1) / kGdnTileRowsMax);
  return groups <= 512 ? 8 : kGdnTileRowsMax;
}
constexpr int64_t kGdnMaxDk = 128;

// Shared grid + binding setup for the two recurrences. Returns false when this
// backend cannot serve the shape and the caller must decline to the next
// provider.
bool GdnRecurrenceCommon(const Tensor& out, const Tensor& q_in, const Tensor& k,
                         const Tensor& v, const Tensor& g, const Tensor& beta,
                         const Tensor& state, Binder& bind, GdnRecurrenceParams& p,
                         uint32_t spec[3], float scale) {
  const int64_t hv = state.shape[1], dv = state.shape[2], dk = state.shape[3];
  const int64_t hk = q_in.shape[1];
  // PER-CALL REFUSAL rather than a throw, the same seam vt_paged_attn uses for an
  // fp8 KV cache: declining forwards to the portable reference tier, which is
  // correct for every shape, instead of REMOVING a capability the backend already
  // had. Two reasons to decline.
  //   * Dk beyond the shared tile's compile-time extent. The tile has to be sized
  //     at compile time against Vulkan's GUARANTEED 16 KB of shared memory, and
  //     the real gate dim is 128.
  //   * q/k/v disagreeing on dtype. One specialization constant covers the three
  //     (they come out of one GdnPostConv dispatch and always agree), and CUDA
  //     asserts the same thing (cuda_gdn.cu:2577).
  if (dk > kGdnMaxDk || dk <= 0) return false;
  if (k.dtype != q_in.dtype || v.dtype != q_in.dtype) return false;
  const uint32_t q_off = bind.Add(q_in, "gdn recurrence: q");
  const uint32_t k_off = bind.Add(k, "gdn recurrence: k");
  const uint32_t v_off = bind.Add(v, "gdn recurrence: v");
  const uint32_t out_off = bind.Add(out, "gdn recurrence: out");
  // g, beta and the state are f32 by the op contract on this device
  // (src/vt/ops.cpp:1629-1643 — a compressed state is CUDA-only), so one binding
  // each rather than a dtype-erased pair whose 16-bit half could never be taken.
  const uint32_t g_off = bind.AddU32Only(g, "gdn recurrence: g");
  const uint32_t beta_off = bind.AddU32Only(beta, "gdn recurrence: beta");
  const uint32_t state_off = bind.AddU32Only(state, "gdn recurrence: state");
  spec[0] = DtypeCode(q_in.dtype);
  spec[1] = DtypeCode(out.dtype);
  p.hk = static_cast<uint32_t>(hk);
  p.dk = static_cast<uint32_t>(dk);
  p.hv = static_cast<uint32_t>(hv);
  p.dv = static_cast<uint32_t>(dv);
  const int64_t tile_rows = GdnTileRows(state.shape[0], hv, dv);
  p.nv = static_cast<uint32_t>((dv + tile_rows - 1) / tile_rows);
  spec[2] = static_cast<uint32_t>(tile_rows);
  p.ratio = static_cast<uint32_t>(hv / hk);
  p.has_idx = 0;
  p.n_state_rows = static_cast<uint32_t>(state.shape[0]);
  p.q_off = q_off;
  p.k_off = k_off;
  p.v_off = v_off;
  p.out_off = out_off;
  p.g_off = g_off;
  p.beta_off = beta_off;
  p.state_off = state_off;
  p.meta_off = 0;
  p.scale = scale;
  return true;
}

// cpu_ops.cpp:1331-1366 GdnPrefillKernel. ONE WORKGROUP PER
// (sequence, value-head, value-tile) — the CPU kernel's own (SEQUENCE,
// VALUE-HEAD) chunking plus the value-row tile our CUDA kernel already uses as
// its grid.x (cuda_gdn.cu:2421). NOT FlatGroupCount: the whole workgroup
// cooperates on one tile, and the sequence stays sequential inside it.
void GdnPrefillKernel(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                      const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                      const Tensor& query_start_loc, const GdnArgs& args) {
  const int64_t n = state.shape[0], hv = state.shape[1], dv = state.shape[2];
  if (n == 0 || hv == 0 || dv == 0) return;
  Binder bind;
  GdnRecurrenceParams p{};
  uint32_t spec[3] = {0, 0, 0};
  if (!GdnRecurrenceCommon(out, q_in, k, v, g, beta, state, bind, p, spec, args.scale)) {
    auto next = reinterpret_cast<GdnPrefillFn>(
        GetOpFallback(OpId::kGdnPrefill, DeviceType::kVULKAN, kNativeProviderName));
    next(q, out, q_in, k, v, g, beta, state, query_start_loc, args);
    return;
  }
  p.meta_off = bind.AddU32Only(query_start_loc, "gdn_prefill: query_start_loc");
  // ⭐ REGISTER-STATE VARIANT, opt-in. llama runs the same sequential recurrence
  // 4.2x faster (11.77 ms/512 tokens against 49.67), and the difference is the
  // decomposition, not the algorithm: it keeps the state in per-lane registers
  // with a single subgroupAdd per reduction, while this path keeps it in shared
  // memory with barriers and a halving tree, 1024 times over. See the shader.
  //
  // Gated on Dk == Dv == 128 so one subgroup covers a row exactly; everything
  // else stays on the shared-state kernel, which is correct for every shape.
  // DEFAULT ON, measured. Interleaved three rounds, pure prefill, c=4:
  //     gdn   396.9 / 395.5 / 391.7 ms -> 245.9 / 242.8 / 237.1   = 1.61-1.65x
  //     whole GPU span 2510 / 2518 / 2482 -> 2432 / 2410 / 2374   = 1.03-1.05x
  // Numerics against the shared-state kernel: nmse 6.39e-05 (per step 8.93e-5 /
  // 1.38e-4 / 5.5e-5), token ids identical. The Dk contraction is summed across
  // 32 lanes instead of 8, so the reduction order differs and the bar is 5e-4.
  //
  // ⚠️ 1.64x of llama's 4.2x, so this closes about half the gap and the rest is
  // still unexplained. The decomposition now MATCHES theirs -- one subgroup per
  // value row, four state rows per lane in registers, one subgroupAdd per
  // reduction -- so whatever remains is elsewhere: their FLOAT_TYPE may be
  // narrower, and our VT_LOAD is a dtype-erased macro where theirs is a direct
  // typed access. Recorded rather than guessed at.
  // VT_VULKAN_GDN_REG=0 forces the shared-state kernel back.
  static const bool kGdnReg = [] {
    const char* v = std::getenv("VT_VULKAN_GDN_REG");
    return v == nullptr || v[0] != '0';
  }();
  if (kGdnReg && state.shape[3] == 128 && dv == 128) {
    const uint32_t rspec[3] = {spec[0], spec[1], 4u};
    const uint32_t rgroups = static_cast<uint32_t>(n * hv * dv);
    Go("vt_gdn_prefill_reg", bind, p, rgroups, rspec, 3);
    return;
  }
  Go("vt_gdn_prefill", bind, p, static_cast<uint32_t>(n * hv * p.nv), spec, 3);
}

// cpu_ops.cpp:1368-1396 GdnDecodeKernel, one step per batch token. Same grid with
// the sequence axis replaced by the batch — cuda_gdn.cu:2513's
// (NV, n*Hv) flattened.
void GdnDecodeKernel(Queue& q, Tensor& out, const Tensor& q_in, const Tensor& k,
                     const Tensor& v, const Tensor& g, const Tensor& beta, Tensor& state,
                     const Tensor* state_idx, const GdnArgs& args) {
  const int64_t batch = q_in.shape[0], hv = state.shape[1], dv = state.shape[2];
  if (batch == 0 || hv == 0 || dv == 0) return;
  Binder bind;
  GdnRecurrenceParams p{};
  uint32_t spec[3] = {0, 0, 0};
  if (!GdnRecurrenceCommon(out, q_in, k, v, g, beta, state, bind, p, spec, args.scale)) {
    auto next = reinterpret_cast<GdnDecodeFn>(
        GetOpFallback(OpId::kGdnDecode, DeviceType::kVULKAN, kNativeProviderName));
    next(q, out, q_in, k, v, g, beta, state, state_idx, args);
    return;
  }
  // Binding 11 is always written: a descriptor a shader statically uses must be
  // valid even on the path that never reads it. With has_idx == 0 it aliases the
  // state buffer and is dead.
  if (state_idx != nullptr) {
    p.has_idx = 1;
    p.meta_off = bind.AddU32Only(*state_idx, "gdn_decode: state_idx");
  } else {
    p.meta_off = bind.AddU32Only(state, "gdn_decode: state");
  }
  Go("vt_gdn_decode", bind, p, static_cast<uint32_t>(batch * hv * p.nv), spec, 3);
}

// ---------------------------------------------------------------------------
// The FUSED FULL-ATTENTION PREAMBLE (BACKEND-VULKAN-QKNORM).
//
// WHY THIS OP AND NOT ANOTHER. With the GDN family native, a 27B decode step was
// MEASURED at ~30 command-buffer flushes per token of which ~28 were
// reference-tier: op_provider.cpp drains the whole recorded batch (submit +
// blocking fence) before it can hand a host kernel device memory, so every
// reference-tier op costs a full round trip regardless of how little arithmetic
// it does. The declines named exactly three ops, and only this one runs in
// DECODE on every step: kCausalConv1dFwd is prefill-only and kRopeCosSinCache is
// deliberately host-side (the double-precision table; see vt_rope_from_cache.comp).
// Qwen3.6-27B has 64 layers of which 48 are linear-attention, so this fires 16
// times per token.
//
// NO DECLINE PATH. Every dtype the op contract admits (ops.cpp:1524-1541) is a
// specialization axis here, the packed row strides are read from stride[0], and
// rotary_dim < Dh is the ordinary case rather than a special one — so there is no
// shape this kernel has to hand back. cpu_ops.cpp:956-1010 is the oracle.
void AttnQkNormRopeGateKernel(Queue&, Tensor& q_out, Tensor& k_out, Tensor& gate_out,
                              const Tensor& qgate, const Tensor& kf, const Tensor& q_norm,
                              const Tensor& k_norm, const Tensor& cos_sin,
                              const RmsNormArgs& na, const RopeArgs& ra) {
  const int64_t t = q_out.shape[0], hq = q_out.shape[1], dh = q_out.shape[2];
  const int64_t hkv = k_out.shape[1];
  if (t == 0 || hq + hkv == 0 || dh == 0) return;
  Binder bind;
  // Binding order IS the descriptor order: the shader declares qgate, kf, then
  // the three outputs as dtype-erased PAIRS, then the three f32-by-contract
  // operands as single bindings.
  const uint32_t qg_off = bind.Add(qgate, "attn_qk_norm_rope_gate: qgate");
  const uint32_t kf_off = bind.Add(kf, "attn_qk_norm_rope_gate: kf");
  const uint32_t qo_off = bind.Add(q_out, "attn_qk_norm_rope_gate: q_out");
  const uint32_t ko_off = bind.Add(k_out, "attn_qk_norm_rope_gate: k_out");
  const uint32_t go_off = bind.Add(gate_out, "attn_qk_norm_rope_gate: gate_out");
  const uint32_t qn_off = bind.AddU32Only(q_norm, "attn_qk_norm_rope_gate: q_norm");
  const uint32_t kn_off = bind.AddU32Only(k_norm, "attn_qk_norm_rope_gate: k_norm");
  const uint32_t cs_off = bind.AddU32Only(cos_sin, "attn_qk_norm_rope_gate: cos_sin");
  // Ascending constantID order: the shared qgate/kf dtype, the shared q/k out
  // dtype, the gate out dtype (its own axis — the FA-2 prefill combo keeps the
  // gate f32 while q/k are bf16).
  const uint32_t spec[3] = {DtypeCode(qgate.dtype), DtypeCode(q_out.dtype),
                            DtypeCode(gate_out.dtype)};
  AttnQkNormRopeGateParams p{static_cast<uint32_t>(hq),
                             static_cast<uint32_t>(hkv),
                             static_cast<uint32_t>(dh),
                             static_cast<uint32_t>(ra.rotary_dim),
                             static_cast<uint32_t>(ra.rotary_dim / 2),
                             static_cast<uint32_t>(qgate.stride[0]),
                             static_cast<uint32_t>(kf.stride[0]),
                             na.gemma ? 1u : 0u,
                             qg_off,
                             kf_off,
                             qo_off,
                             ko_off,
                             go_off,
                             qn_off,
                             kn_off,
                             cs_off,
                             na.eps};
  // ONE WORKGROUP PER (token, head) over Hq + Hkv slots — CUDA's dim3(t, hq+hkv)
  // grid (cuda_ops.cu:1394) flattened. NOT FlatGroupCount: the workgroup
  // tree-reduces one head row's mean square.
  Go("vt_attn_qk_norm_rope_gate", bind, p, static_cast<uint32_t>(t * (hq + hkv)), spec, 3);
}

struct Registrar {
  Registrar() {
    // Same guard as the backend registrar: a Vulkan-enabled build on a host with
    // no loader or no conformant device registers nothing, so GetOp throws its
    // normal not-registered error.
    if (!VulkanContext::Available()) return;
    // static_cast against the ops.h aliases ties every kernel signature to the
    // registration contract at COMPILE time (the cpu_ops.cpp idiom).
    RegisterOp(OpId::kQkvSplit, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<QkvSplitFn>(&QkvSplitKernel)));
    RegisterOp(OpId::kRopeFromCache, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<RopeFromCacheFn>(&RopeFromCacheKernel)));
    RegisterOp(OpId::kReshapeAndCache, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<ReshapeAndCacheFn>(&ReshapeAndCacheKernel)));
    RegisterOp(OpId::kPagedAttention, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<PagedAttentionFn>(&PagedAttentionKernel)));
    RegisterOp(OpId::kEmbedding, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<EmbeddingFn>(&EmbeddingKernel)));
    RegisterOp(OpId::kGreedyArgmax, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<GreedyArgmaxFn>(&GreedyArgmaxKernel)));
    RegisterOp(OpId::kMatmul, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulGeneric<false>)));
    RegisterOp(OpId::kMatmulBT, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulGeneric<true>)));
    RegisterOp(OpId::kAdd, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<AddFn>(&AddKernel)));
    RegisterOp(OpId::kRelu, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<ReluFn>(&ReluKernel)));
    RegisterOp(OpId::kSiluAndMul, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<SiluAndMulFn>(&SiluAndMulKernel)));
    RegisterOp(OpId::kMoeSiluMul, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<MoeSiluMulFn>(&MoeSiluMulKernel)));
    RegisterOp(OpId::kCastBf16, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<CastBf16Fn>(&CastKernel)));
    RegisterOp(OpId::kCastF32, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<CastF32Fn>(&CastKernel)));
    // BACKEND-VULKAN-EXL3 V1 (#2530). The THIRD sibling, and the same kernel: the
    // dtype pair is a specialization constant, `DtypeCode` already maps
    // DType::kF16 to VT_DT_F16, and vt_common.glsl's f16 codec is an integer
    // transcription of src/vt/dtype.cpp -- so this registration adds a shader
    // variant, not a shader. It was missing while its two siblings were present,
    // which is why an EXL3 checkpoint's f16 activation cast was one of the two
    // ops S1 measured falling to the portable CPU tier on a Vulkan queue.
    //
    // THE SHARED LIMITATION IS NAMED RATHER THAN INHERITED IN SILENCE. `vt::CastF16`
    // and `vt::CastBf16` both TOLERATE a packed strided input (the merged-QKV view)
    // whose rows are dense while the row stride spans a parent tensor, and
    // `CastKernel` above indexes FLAT from one byte offset, so it would read such an
    // input as contiguous. That is pre-existing for the two siblings and is not
    // widened here; src/vt/ops.cpp says the merge is CUDA-only, and
    // .agents/specs/backend-vulkan-exl3.md `## Owed` carries it.
    RegisterOp(OpId::kCastF16, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<CastF16Fn>(&CastKernel)));
    // BACKEND-VULKAN-EXL3 V2 (#2530). The other op S1 measured on the reference
    // tier, and the whole EXL3 forward. `kExl3HadR128` is deliberately NOT
    // registered: the shader that performs it ships here as steps 1 and 3 of this
    // GEMM, but no dense forward path calls that op, and a registration nothing
    // reaches is what `.agents/reachability.md` exists to prevent. The spec's
    // `## Owed` carries it.
    RegisterOp(OpId::kExl3Gemm, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<Exl3GemmFn>(&Exl3GemmKernelVulkan)));
    RegisterOp(OpId::kLayerNorm, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<LayerNormFn>(&LayerNormKernel)));
    RegisterOp(OpId::kRmsNorm, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<RmsNormFn>(&RmsNormKernel)));
    RegisterOp(OpId::kFusedChain, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<FusedChainFn>(&FusedChainKernel)));
    // BACKEND-VULKAN-GDN: the GDN glue family. kCausalConv1dFwd (the prefill
    // conv) and kRopeCosSinCache (the double-precision rotary table, deliberately
    // host-side) stay on the portable reference tier; see the block comment above
    // these kernels.
    RegisterOp(OpId::kSigmoidGateBf16, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<SigmoidGateBf16Fn>(&SigmoidGateBf16Kernel)));
    RegisterOp(OpId::kRmsNormGated, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<RmsNormGatedFn>(&RmsNormGatedKernel)));
    RegisterOp(OpId::kGdnStateGather, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<GdnStateGatherFn>(&GdnStateGatherKernel)));
    RegisterOp(OpId::kGdnStateScatter, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<GdnStateScatterFn>(&GdnStateScatterKernel)));
    RegisterOp(
        OpId::kCausalConv1dUpdate, DeviceType::kVULKAN,
        reinterpret_cast<void*>(static_cast<CausalConv1dUpdateFn>(&CausalConv1dUpdateKernel)));
    // VT_VULKAN_CONV_FWD=0 leaves this op on the portable CPU reference tier,
    // which is where it lived before this kernel existed. Same-binary A/B, the
    // same shape as VT_VULKAN_COOPMAT and VT_VULKAN_COOPMAT_TILED: a before/after
    // across two BUILDS would confound the kernel with everything else that
    // differs between them, and this op's effect is large enough (49 s to 2 s on
    // a 576-token prefill) that it must be attributable to the kernel alone.
    {
      const char* conv_env = std::getenv("VT_VULKAN_CONV_FWD");
      if (conv_env == nullptr || conv_env[0] != '0') {
        RegisterOp(OpId::kCausalConv1dFwd, DeviceType::kVULKAN,
                   reinterpret_cast<void*>(static_cast<CausalConv1dFwdFn>(&CausalConv1dFwdKernel)));
      }
    }
    RegisterOp(OpId::kGdnPostConv, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<GdnPostConvFn>(&GdnPostConvKernel)));
    // BACKEND-VULKAN-GDN-CORE: the two recurrences.
    RegisterOp(OpId::kGdnPrefill, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<GdnPrefillFn>(&GdnPrefillKernel)));
    RegisterOp(OpId::kGdnDecode, DeviceType::kVULKAN,
               reinterpret_cast<void*>(static_cast<GdnDecodeFn>(&GdnDecodeKernel)));
    // BACKEND-VULKAN-QKNORM: the fused full-attention preamble, the last
    // per-decode-step reference-tier op on the 27B.
    RegisterOp(
        OpId::kAttnQkNormRopeGate, DeviceType::kVULKAN,
        reinterpret_cast<void*>(static_cast<AttnQkNormRopeGateFn>(&AttnQkNormRopeGateKernel)));
  }
} registrar;

}  // namespace

// OUTPUT COLUMNS PER LANE for the scalar matmul tactic (VK-G). 4 by default;
// VT_VULKAN_MATMUL_NCOLS=1|2|4|8 selects an arm.
//
// WHY 4 AND NOT 8, MEASURED on GB10 over three interleaved triples at 27B decode
// (ms/call for the lm_head, k=5120 n=248320): ncols 1 = 12.55, ncols 4 = 11.63,
// ncols 8 = 12.81. Blocking is a TRADE, not a monotone win -- it buys a longer
// contiguous run per workgroup and pays for it in workgroups. At 8 the dispatch
// is only ceil(248320/1024) = 243 workgroups of 128 lanes, about 31k threads,
// and the device runs out of work to hide memory latency with faster than the
// longer run buys back.
//
// The lever exists for ONE reason, the same one the coopmat and GEMV levers cite:
// a SAME-BINARY A/B. The factor is a specialization constant, so every arm is the
// same committed SPIR-V and two runs differ in exactly one thing; comparing two
// BUILDS is what produced a false 1.2x reading for the subgroup tactic earlier in
// this campaign. It is NOT a correctness switch -- every arm is bit-identical,
// which tests/vt/test_vulkan_backend.cpp asserts with memcmp.
//
// Atomic because ops dispatch from whichever thread the engine runs on; relaxed
// because nothing else is ordered against it -- a stale read would pick the other
// arm of a performance A/B, never a wrong result.
std::atomic<uint32_t>& MatmulColumnsSlot() {
  static std::atomic<uint32_t> slot{[] {
    const char* v = std::getenv("VT_VULKAN_MATMUL_NCOLS");
    if (v == nullptr) return 4u;
    if (std::strcmp(v, "1") == 0) return 1u;
    if (std::strcmp(v, "2") == 0) return 2u;
    if (std::strcmp(v, "8") == 0) return 8u;
    return 4u;
  }()};
  return slot;
}

uint32_t MatmulColumnsPerLane() { return MatmulColumnsSlot().load(std::memory_order_relaxed); }

std::atomic<bool>& MatmulColumnsExplicit() {
  static std::atomic<bool> f{std::getenv("VT_VULKAN_MATMUL_NCOLS") != nullptr};
  return f;
}

void SetMatmulColumnsPerLane(uint32_t ncols) {
  // 1, 2, 4 or 8 only. The shader sizes its accumulator array with a COMPILE-TIME
  // bound of 8 -- a specialization constant cannot size it without the array
  // spilling to scratch memory, which would defeat the point -- so a larger value
  // would read past the array rather than fail.
  VT_CHECK(ncols == 1u || ncols == 2u || ncols == 4u || ncols == 8u,
           "vulkan: matmul columns-per-lane must be 1, 2, 4 or 8 (the shader's "
           "accumulator array is bounded at 8)");
  MatmulColumnsSlot().store(ncols, std::memory_order_relaxed);
  // AN EXPLICIT REQUEST OUTRANKS THE SHAPE RULE. Without this the wide-projection
  // default silently overrode SetMatmulColumnsPerLane, so a caller asking for a
  // specific arm -- the A/B in tests/vt/test_vulkan_backend.cpp asks for 4 and
  // then 1 -- got 2 both times and neither pipeline it was checking for existed.
  // Caught by the suite, which is the run I had skipped.
  MatmulColumnsExplicit().store(true, std::memory_order_relaxed);
}

}  // namespace vt::vulkan
