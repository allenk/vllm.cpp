// Flint on the paged CUDA engine. See flint_paged.h for the shape of the
// forward and the KV-cache layout; this file composes the shared pieces:
//   * dense_attn::AttnBlock (merged NVFP4 qkv, per-head q/k RMSNorm, NeoX RoPE,
//     paged attention, NVFP4 o_proj) with the per-layer window, sinks and NoPE
//     options (AttnBlockLayerOpts);
//   * the NVFP4 W4A16 Marlin dense GEMMs (dense_nvfp4_gemm.h) for the SwiGLU
//     MLPs and the shared expert;
//   * the Marlin grouped MoE GEMM over a per-layer arena, the arrangement
//     nemotron_h_device.cpp uses for its sigmoid router and ungated shared
//     expert, with gate|up fused into one w13 operand as qwen3_5.cpp's Marlin
//     MoE does (vLLM fused_marlin_moe: one w13 GEMM, silu_and_mul, one w2 GEMM).
//   * the Qwen3MoeDecodeGraph capture/replay state machine.
#include "vllm/model_executor/models/flint_paged.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vllm/model_executor/layers/attention/attention.h"  // ResolveAttentionWindow
#include "vllm/model_executor/models/decode_graph_sizes.h"
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/dense_nvfp4_gemm.h"
#include "vllm/model_executor/models/flint_device.h"
#include "vllm/model_executor/models/kv_cache_route.h"  // IsFp8KvCache
#include "vllm/model_executor/models/qwen3_5_internal.h"  // detail::*DeviceTokenIds
#include "vllm/model_executor/models/step_token_ids.h"
#include "vllm/platforms/interface.h"
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/worker/gpu/async_runner_flag.h"
#include "vt/backend.h"
#include "vt/breakable_graph.h"
#include "vt/ops.h"
#include "vt/persistent_step_input.h"
// DSR-ALLOW(MODEL-TEXT-FLINT): TYPES, not behaviour -- vt/cuda/marlin_repack.h is built only under VT_MARLIN_NVFP4, so this include cannot be resolved by a runtime query (the nemotron_h_device.cpp A2-Q2a shape).
#ifdef VT_MARLIN_NVFP4
#include "vt/cuda/marlin_repack.h"
#endif

namespace vllm {
namespace {

using namespace dense_attn;
using v1::CommonAttentionMetadata;

// ─── the routed experts' Marlin arena, one per MoE layer ─────────────────────
// w13 holds gate|up concatenated along N (size_n = 2I) so one grouped GEMM
// produces both halves for SiluAndMul; w2 is the down projection. Sizes at
// Flint's geometry (H 2048, I 512, E 256): 302 MB + 151 MB per layer, 21.3 GB
// for the 47 MoE layers.
struct FlintMoeMarlin {
  void* w13 = nullptr;  // i32 [E, H/16, 2I*2]
  void* s13 = nullptr;  // fp8 [E, H/16, 2I]
  void* g13 = nullptr;  // f32 [E]
  void* w2 = nullptr;   // i32 [E, I/16, H*2]
  void* s2 = nullptr;   // fp8 [E, I/16, H]
  void* g2 = nullptr;   // f32 [E]
  void* workspace = nullptr;  // i32 [sms*4] reduction locks, zeroed once
  int sms = 0;
  bool ready = false;
};

// The resident state a weight owns (the qwen3_5.cpp / nemotron_h_device.cpp
// accessor), keyed on the weight's slot rather than its address.
template <typename R>
R& ResidentIn(const ResidentSlot& slot) {
  static std::mutex mu;
  std::lock_guard<std::mutex> lk(mu);
  if (!slot.state) slot.state = std::make_shared<R>();
  return *static_cast<R*>(slot.state.get());
}

// DSR-ALLOW(MODEL-TEXT-FLINT): the Marlin repack of the routed experts calls marlin_repack.h primitives, built only under VT_MARLIN_NVFP4; the runtime question, OpRegistered(kMoeGroupedGemmNvfp4Marlin, device), is asked in PrepareFlintPaged.
#ifdef VT_MARLIN_NVFP4
// Repacks every routed expert of one layer from its HOST bytes through one
// reused staging pair, then releases those host bytes: after this the arena is
// the only copy the forward reads. The combined scale factors follow vLLM's
// marlin_utils_fp4: w13 over every expert's gate and up scales, w2 over the
// downs; the fused w13 needs one global per expert, which the converter writes
// (gate == up for every expert) and anything else is refused.
void BuildFlintMoeMarlin(Dev d, const FlintPagedMoe& m, int64_t hidden, int64_t inter,
                         FlintMoeMarlin& mr) {
  if (mr.ready) return;
  const int E = static_cast<int>(m.expert_gate.size());
  const int K = static_cast<int>(hidden), N = static_cast<int>(inter);
  VT_CHECK(E > 0 && m.expert_up.size() == m.expert_gate.size() &&
               m.expert_down.size() == m.expert_gate.size(),
           "flint paged: the MoE layer must carry gate, up and down for every expert");
  mr.sms = vt::cuda::MarlinDeviceSms(d.q.device.index);

  const size_t w13_i32 = static_cast<size_t>(K / 16) * (static_cast<size_t>(N) * 2 * 2);
  const size_t s13_b = static_cast<size_t>(K / 16) * (static_cast<size_t>(N) * 2);
  const size_t w2_i32 = static_cast<size_t>(N / 16) * (static_cast<size_t>(K) * 2);
  const size_t s2_b = static_cast<size_t>(N / 16) * static_cast<size_t>(K);
  mr.w13 = d.b.Alloc(static_cast<size_t>(E) * w13_i32 * 4);
  mr.s13 = d.b.Alloc(static_cast<size_t>(E) * s13_b);
  mr.g13 = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
  mr.w2 = d.b.Alloc(static_cast<size_t>(E) * w2_i32 * 4);
  mr.s2 = d.b.Alloc(static_cast<size_t>(E) * s2_b);
  mr.g2 = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
  mr.workspace = d.b.Alloc(static_cast<size_t>(mr.sms) * 4 * sizeof(int32_t));

  std::vector<const uint8_t*> b13, b2;
  std::vector<size_t> l13, l2;
  for (int e = 0; e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    for (const Nvfp4Weight* w : {&m.expert_gate[se], &m.expert_up[se]}) {
      b13.push_back(reinterpret_cast<const uint8_t*>(w->scale.bytes.data()));
      l13.push_back(w->scale.bytes.size());
    }
    b2.push_back(reinterpret_cast<const uint8_t*>(m.expert_down[se].scale.bytes.data()));
    l2.push_back(m.expert_down[se].scale.bytes.size());
  }
  const float sf13 = vt::cuda::MarlinNvfp4CombinedScaleFactor(b13, l13);
  const float sf2 = vt::cuda::MarlinNvfp4CombinedScaleFactor(b2, l2);

  // One shard's bytes; the down projection is the same size (N*K/2 and N*K/16).
  const size_t pk_b = static_cast<size_t>(N) * static_cast<size_t>(K) / 2;
  const size_t sc_b = static_cast<size_t>(N) * static_cast<size_t>(K) / 16;
  auto* tmp_w = static_cast<uint8_t*>(d.b.Alloc(2 * pk_b));
  auto* tmp_s = static_cast<uint8_t*>(d.b.Alloc(2 * sc_b));
  std::vector<float> g13(static_cast<size_t>(E)), g2(static_cast<size_t>(E));
  for (int e = 0; e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    const Nvfp4Weight& g = m.expert_gate[se];
    const Nvfp4Weight& u = m.expert_up[se];
    const Nvfp4Weight& dn = m.expert_down[se];
    VT_CHECK(g.scale2 == u.scale2,
             "flint paged: an expert's gate and up global scales differ, so they "
             "cannot share one fused w13 operand");
    VT_CHECK(g.packed.bytes.size() == pk_b && u.packed.bytes.size() == pk_b &&
                 dn.packed.bytes.size() == pk_b && g.scale.bytes.size() == sc_b &&
                 u.scale.bytes.size() == sc_b && dn.scale.bytes.size() == sc_b,
             "flint paged: an expert's packed or scale bytes do not match [N, K/2] / [N, K/16]");
    // All copies and repacks share one stream, so each expert's repack has read
    // the staging bytes before the next copy overwrites them.
    d.b.Copy(d.q, tmp_w, g.packed.bytes.data(), pk_b);
    d.b.Copy(d.q, tmp_w + pk_b, u.packed.bytes.data(), pk_b);
    vt::cuda::MarlinRepackExpertWeight(d.q.handle, d.q.device.index,
                                       static_cast<uint32_t*>(mr.w13) + se * w13_i32, tmp_w,
                                       K, 2 * N);
    d.b.Copy(d.q, tmp_s, g.scale.bytes.data(), sc_b);
    d.b.Copy(d.q, tmp_s + sc_b, u.scale.bytes.data(), sc_b);
    vt::cuda::MarlinProcessExpertScales(d.q.handle, tmp_s,
                                        static_cast<uint8_t*>(mr.s13) + se * s13_b, K, 2 * N,
                                        sf13);
    g13[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(g.scale2, sf13);
    d.b.Copy(d.q, tmp_w, dn.packed.bytes.data(), pk_b);
    vt::cuda::MarlinRepackExpertWeight(d.q.handle, d.q.device.index,
                                       static_cast<uint32_t*>(mr.w2) + se * w2_i32, tmp_w, N,
                                       K);
    d.b.Copy(d.q, tmp_s, dn.scale.bytes.data(), sc_b);
    vt::cuda::MarlinProcessExpertScales(d.q.handle, tmp_s,
                                        static_cast<uint8_t*>(mr.s2) + se * s2_b, N, K, sf2);
    g2[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(dn.scale2, sf2);
  }
  d.b.Copy(d.q, mr.g13, g13.data(), g13.size() * sizeof(float));
  d.b.Copy(d.q, mr.g2, g2.data(), g2.size() * sizeof(float));
  d.b.Memset(d.q, mr.workspace, 0, static_cast<size_t>(mr.sms) * 4 * sizeof(int32_t));
  d.b.Synchronize(d.q);
  d.b.Free(tmp_w);
  d.b.Free(tmp_s);
  for (int e = 0; e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    for (const Nvfp4Weight* w : {&m.expert_gate[se], &m.expert_up[se], &m.expert_down[se]}) {
      w->packed.ReleaseHost();
      w->scale.ReleaseHost();
    }
  }
  mr.ready = true;
}
#endif  // VT_MARLIN_NVFP4

// silu(x @ gate^T) * (x @ up^T): one fused Marlin gate|up GEMM when the pair can
// share an operand (vLLM's merged gate_up_proj), else two W4A16 GEMMs.
DBuf GateUpAct(Dev d, const Tensor& x, const Nvfp4Weight& gate, const Nvfp4Weight& up) {
  // DSR-ALLOW(MODEL-TEXT-FLINT): dense_nvfp4::GateUpFusedMarlinD is declared only under VT_MARLIN_NVFP4 (dense_nvfp4_gemm.h); MarlinW4A16Selects is the runtime question inside.
#ifdef VT_MARLIN_NVFP4
  if (dense_nvfp4::MarlinW4A16Selects(d, x.dtype) && dense_nvfp4::GateUpFusedEligible(gate, up))
    return dense_nvfp4::GateUpFusedMarlinD(d, x, gate, up);
#endif
  DBuf go = dense_nvfp4::MatmulNvfp4W4A16D(d, x, gate, DType::kBF16);
  DBuf uo = dense_nvfp4::MatmulNvfp4W4A16D(d, x, up, DType::kBF16);
  DBuf act(d, DType::kBF16, {x.shape[0], gate.n});
  vt::MoeSiluMul(d.q, act.t(), go.t(), uo.t());
  return act;
}

DBuf DenseMlp(Dev d, const FlintPagedLayer& lw, const Tensor& x) {
  DBuf act = GateUpAct(d, x, lw.mlp_gate, lw.mlp_up);
  return dense_nvfp4::MatmulNvfp4W4A16D(d, act.t(), lw.mlp_down, DType::kBF16);
}

// The kFlint glue kernels (flint_device.h), or nullptr when this build or device
// has none, or VT_FLINT_FUSED=0 asks for the composed shared ops (a same-binary
// A/B: the composed attention rotates with the bf16 cos/sin cache).
const flint::FlintDeviceKernels* FusedKernels(Dev d) {
  static const bool on = [] {
    const char* e = std::getenv("VT_FLINT_FUSED");
    return !(e != nullptr && e[0] == '0');
  }();
  if (!on || !flint::FlintDeviceKernelsAvailable(d.q.device.type)) return nullptr;
  return flint::FlintDevice(d.q.device.type);
}

// The decode graph's prologue (flint_device.h decode_prologue), or nullptr. It
// only moves bytes, so it does not follow VT_FLINT_FUSED; VT_FLINT_GRAPH_PROLOGUE=0
// keeps the per-input staged copies and the embedding outside the graph.
const flint::FlintDeviceKernels* PrologueKernels(Dev d) {
  static const bool on = [] {
    const char* e = std::getenv("VT_FLINT_GRAPH_PROLOGUE");
    return !(e != nullptr && e[0] == '0');
  }();
  if (!on || !flint::FlintDeviceKernelsAvailable(d.q.device.type)) return nullptr;
  return flint::FlintDevice(d.q.device.type);
}

// One attention block. With the glue kernels it is vLLM's FlintAttention on SM120:
// the merged qkv GEMM, then ONE kernel for the per-head q/k RMSNorm, the RoPE in
// f32 from the f32 cos|sin table (sliding layers only) and the K/V cache write
// (vLLM's qk_norm_rope_cache), then paged attention with the window and sinks,
// then o_proj. Without them it is dense_attn::AttnBlock with the layer options.
DBuf Attention(Dev d, const FlintPagedLayer& lw, const FlintParams& p, const HfConfig& cfg,
               const Tensor& x, const StepInputs& si, const CommonAttentionMetadata& meta,
               const PagedKvCache& kv, int64_t T, const flint::FlintDeviceKernels* fk) {
  AttnBlockLayerOpts opts;
  if (lw.sliding) {
    // vLLM per_layer_sliding_window = sliding_window, i.e. FlashAttention's
    // (W-1, 0); --disable-sliding-window turns it off as for every model.
    opts.window = ResolveAttentionWindow(std::nullopt, p.sliding_window,
                                         v1::AttentionType::kDecoder,
                                         DisableSlidingWindowActive());
  }
  opts.sinks = &lw.sinks;
  opts.rope = lw.rope;
  // The kernel takes bf16 norm weights; anything else is the composed path.
  if (fk == nullptr || kv.dtype != DType::kBF16 || IsFp8KvCache(kv) || !lw.attn.IsNvfp4() ||
      lw.attn.q_norm.dtype != DType::kBF16 || lw.attn.k_norm.dtype != DType::kBF16)
    return AttnBlock(d, lw.attn, cfg, x, si, meta, kv, T, nullptr, opts);

  const int64_t Hq = p.num_attention_heads, Hkv = p.num_key_value_heads, Dh = p.head_dim;
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "flint paged: KV cache head dims mismatch the config");
  DBuf qkv = dense_nvfp4::MatmulNvfp4W4A16D(d, x, lw.attn.qkv_proj_fp4, DType::kBF16);
  DBuf q(d, DType::kBF16, {T, Hq, Dh});
  Tensor k_cache = KvSlice(kv, d.q.device, 0);
  Tensor v_cache = KvSlice(kv, d.q.device, 1);
  Tensor q_w = ResidentWeight(d, lw.attn.q_norm, {Dh});
  Tensor k_w = ResidentWeight(d, lw.attn.k_norm, {Dh});
  fk->qk_norm_rope_kv_cache(d.q, q.ptr(), qkv.ptr(), (Hq + 2 * Hkv) * Dh, q_w.data, k_w.data,
                            lw.rope ? si.cos_sin.t().Ptr<float>() : nullptr, k_cache.data,
                            v_cache.data, k_cache.stride[0], k_cache.stride[1],
                            k_cache.stride[2], kv.block_size, si.slot_mapping.t().Ptr<int64_t>(),
                            T, Hq, Hkv, Dh, static_cast<float>(p.rms_norm_eps));

  vt::PagedAttentionArgs pa{1.0F / std::sqrt(static_cast<float>(Dh)), meta.causal};
  pa.query_start_loc_host = meta.query_start_loc.data();
  pa.max_seq_len = meta.max_seq_len;
  pa.uniform_spec_query_len = meta.uniform_spec_query_len;
  pa.window_size = opts.window;
  // f32 sinks upload as they are (vLLM keeps them f32); bf16 ones widen.
  Tensor sinks = lw.sinks.dtype == DType::kF32 ? ResidentWeight(d, lw.sinks, {Hq})
                                               : ResidentWeightF32(d, lw.sinks, {Hq});
  pa.sinks = &sinks;
  DBuf attn(d, DType::kBF16, {T, Hq, Dh});
  vt::PagedAttention(d.q, attn.t(), q.t(), k_cache, v_cache, si.block_table.t(),
                     si.seq_lens.t(), si.query_start_loc.t(), pa);
  return dense_nvfp4::MatmulNvfp4W4A16D(d, Reshape(attn.t(), {T, Hq * Dh}), lw.attn.o_proj_fp4,
                                        DType::kBF16);
}

// The shared expert and its aux stream serve only the Marlin MoE block below; a
// build without Marlin refuses the MoE and would leave them unused.
// DSR-ALLOW(MODEL-TEXT-FLINT): the shared expert and its aux stream serve only the Marlin MoE block below and would be unused functions without it.
#ifdef VT_MARLIN_NVFP4
// The ungated shared expert (vLLM's merged gate_up_proj, then down_proj).
DBuf SharedExpert(Dev d, const FlintPagedMoe& m, const Tensor& x) {
  DBuf act = GateUpAct(d, x, m.shared_gate, m.shared_up);
  return dense_nvfp4::MatmulNvfp4W4A16D(d, act.t(), m.shared_down, DType::kBF16);
}

// The second stream the shared expert runs on while the routed experts run on
// the main one: vLLM's SharedExperts MULTI_STREAM_OVERLAPPED order, taken for
// steps of at most VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD (256) tokens. The
// switches are the Qwen3.5 driver's (qwen3_5.cpp MoeSharedAuxStreamEnabled):
// VT_MOE_SHARED_AUX_STREAM=0 keeps the serial order, VT_MOE_SHARED_AUX_THRESHOLD
// overrides the token cap (0 never overlaps). Flint keeps vLLM's 256 as its
// default where the Qwen3.5 driver calibrated 128 on GB10. One queue and two
// events per device, created at the first (eager) use, before any capture.
struct FlintAuxStream {
  vt::Queue q;
  vt::Event fork;  // recorded on main after the block input; aux waits it
  vt::Event done;  // recorded on aux after the shared expert; main waits it
};
FlintAuxStream* SharedExpertAuxStream(Dev d, int64_t T) {
  static const bool on = [] {
    const char* e = std::getenv("VT_MOE_SHARED_AUX_STREAM");
    return !(e != nullptr && e[0] == '0');
  }();
  static const int64_t cap = [] {
    const char* e = std::getenv("VT_MOE_SHARED_AUX_THRESHOLD");
    return e != nullptr ? static_cast<int64_t>(std::atoll(e)) : int64_t{256};
  }();
  if (!on || T > cap || !d.b.SupportsAuxStream()) return nullptr;
  static std::mutex mu;
  static std::unordered_map<int, FlintAuxStream> streams;  // by device index
  std::lock_guard<std::mutex> lk(mu);
  auto it = streams.find(d.q.device.index);
  if (it == streams.end()) {
    FlintAuxStream a{d.b.CreateQueue(), d.b.CreateEvent(), d.b.CreateEvent()};
    it = streams.emplace(d.q.device.index, a).first;
  }
  return &it->second;
}
#endif  // VT_MARLIN_NVFP4

// One MoE block: bf16 router logits -> sigmoid top-6 renormalised in f32 (the
// kFlint kernel, or the grouped router with a single group, which is plain
// top-k) -> moe_align -> w13 GEMM -> SiluAndMul (vLLM's act_and_mul with the
// kFlint kernels) -> w2 GEMM -> the ungated shared expert -> weighted combine.
// For small steps the shared expert runs on the aux stream concurrently with the
// router and the routed experts (SharedExpertAuxStream); both read x and write
// disjoint buffers, so the output is the serial one.
// The router GEMM stores bf16 because that is what vLLM's GateLinear does on
// SM120: its fp32-output tiers are gated on SM90 / SM100 (family 100 only) or
// on an out_dtype Flint does not set, so it falls to F.linear in the model dtype.
DBuf MoeBlock(Dev d, const FlintPagedMoe& m, const FlintParams& p, const Tensor& x, int64_t T,
              const flint::FlintDeviceKernels* fk) {
  // DSR-ALLOW(MODEL-TEXT-FLINT): the Marlin MoE forward reads the FlintMoeMarlin arena built above under the same gate; a build without Marlin refuses the MoE by name.
#ifdef VT_MARLIN_NVFP4
  const int64_t H = p.hidden_size, E = p.num_experts, Kk = p.num_experts_per_tok;
  const int64_t I = p.moe_intermediate_size, P = T * Kk;
  VT_CHECK(x.dtype == DType::kBF16, "flint paged: the Marlin MoE takes a bf16 activation");
  FlintMoeMarlin& mr = ResidentIn<FlintMoeMarlin>(m.marlin);
  if (!mr.ready) BuildFlintMoeMarlin(d, m, H, I, mr);

  std::optional<DBuf> shared;
  FlintAuxStream* ax = SharedExpertAuxStream(d, T);
  if (ax != nullptr) {
    d.b.RecordEvent(ax->fork, d.q);
    d.b.QueueWaitEvent(ax->q, ax->fork);
    // Its scratch from AuxPool, so no block is live on both streams at once.
    ActivePoolScope guard(&AuxPool(d.b));
    shared.emplace(SharedExpert(Dev{d.b, ax->q}, m, x));
    d.b.RecordEvent(ax->done, ax->q);
  }

  Tensor router = ResidentWeight(d, m.router, {E, H});
  DBuf logits(d, DType::kBF16, {T, E});
  vt::MatmulBT(d.q, logits.t(), x, router);
  DBuf topk_w(d, DType::kF32, {T, Kk});
  DBuf topk_id(d, DType::kI32, {T, Kk});
  if (fk != nullptr) {
    fk->sigmoid_topk(d.q, static_cast<float*>(topk_w.ptr()), static_cast<int32_t*>(topk_id.ptr()),
                     logits.ptr(), T, E, Kk);
  } else {
    vt::MoeRouterTopKArgs args;
    args.top_k = static_cast<int>(Kk);
    args.renormalize = true;  // norm_topk_prob
    args.scoring_func = vt::MoeScoringFunc::kSigmoid;
    // Sigmoid scoring lives on the grouped path; one group of all experts is
    // the ungrouped top-k (no routing bias, scaling factor 1).
    args.num_expert_group = 1;
    args.topk_group = 1;
    args.routed_scaling_factor = 1.0F;
    vt::MoeRouterTopK(d.q, topk_w.t(), topk_id.t(), logits.t(), args);
  }

  const int Ti = static_cast<int>(T), Hi = static_cast<int>(H), Ii = static_cast<int>(I);
  const int tki = static_cast<int>(Kk), Ei = static_cast<int>(E), Pi = static_cast<int>(P);
  const int block = vt::cuda::MarlinMoeAlignBlockSizeSelect(Ti, tki, Ei);
  int max_tok = 0, max_blk = 0;
  vt::cuda::MarlinMoeAlignSizes(Ti, tki, Ei, block, &max_tok, &max_blk);
  DBuf sorted(d, DType::kI32, {max_tok});
  DBuf expert_ids(d, DType::kI32, {max_blk});
  DBuf npad(d, DType::kI32, {1});
  vt::cuda::MarlinMoeAlignBlockSize(d.q.handle, static_cast<const int32_t*>(topk_id.t().data), Ti,
                                    tki, Ei, block, static_cast<int32_t*>(sorted.t().data),
                                    static_cast<int32_t*>(expert_ids.t().data),
                                    static_cast<int32_t*>(npad.t().data));
  Tensor ws = MakeTensor(mr.workspace, DType::kI32, d.q.device, {mr.sms * 4});

  DBuf gu(d, DType::kBF16, {P, 2 * I});
  {
    Tensor wq = MakeTensor(mr.w13, DType::kI32, d.q.device, {E, H / 16, 2 * I * 2});
    Tensor sc = MakeTensor(mr.s13, DType::kI8, d.q.device, {E, H / 16, 2 * I});
    Tensor gg = MakeTensor(mr.g13, DType::kF32, d.q.device, {E});
    vt::MoeGroupedGemmNvfp4Marlin(d.q, gu.t(), x, wq, sc, gg, ws, sorted.t(), expert_ids.t(),
                                  npad.t(), topk_w.t(),
                                  vt::MoeMarlinArgs{block, tki, Ti, 2 * Ii, Hi, false});
  }
  DBuf act(d, DType::kBF16, {P, I});
  if (fk != nullptr) {
    fk->routed_silu_and_mul(d.q, act.ptr(), gu.ptr(), P, I);
  } else {
    vt::SiluAndMul(d.q, act.t(), gu.t());
  }
  DBuf down(d, DType::kBF16, {P, H});
  {
    Tensor wq = MakeTensor(mr.w2, DType::kI32, d.q.device, {E, I / 16, H * 2});
    Tensor sc = MakeTensor(mr.s2, DType::kI8, d.q.device, {E, I / 16, H});
    Tensor gg = MakeTensor(mr.g2, DType::kF32, d.q.device, {E});
    vt::MoeGroupedGemmNvfp4Marlin(d.q, down.t(), act.t(), wq, sc, gg, ws, sorted.t(),
                                  expert_ids.t(), npad.t(), topk_w.t(),
                                  vt::MoeMarlinArgs{block, 1, Pi, Hi, Ii, false});
  }

  if (ax != nullptr) {
    d.b.QueueWaitEvent(d.q, ax->done);  // join before the combine reads it
  } else {
    shared.emplace(SharedExpert(d, m, x));
  }

  DBuf out(d, DType::kBF16, {T, H});
  Tensor eo = Reshape(down.t(), {T, Kk, H});
  vt::MoeCombine(d.q, out.t(), eo, topk_w.t(), &shared->t());
  return out;
#else
  (void)d;
  (void)m;
  (void)p;
  (void)x;
  (void)T;
  (void)fk;
  VT_CHECK(false, "flint paged: the MoE needs the Marlin NVFP4 build (VLLM_CPP_MARLIN)");
  return DBuf();
#endif
}

bool FusedNorm(const flint::FlintDeviceKernels* fk, const Tensor& x, const Tensor& w) {
  return fk != nullptr && x.dtype == DType::kBF16 && w.dtype == DType::kBF16 && x.IsContiguous();
}

// res := bf16(x + res); out = RMSNorm(x + res) * w. With the glue kernels this is
// vLLM's Flint fused_add_rms_norm (the norm taken from the unrounded f32 sum, one
// launch); without them the shared fusion catalog's kFusedAddRmsNormStd, which
// normalises the rounded res and is byte-identical to the standalone vt::RmsNorm
// it falls back to under VT_FUSED_CHAIN_ADOPT=0.
// `first`: the residual stream starts at this norm. The fused kernel then stores x
// as the residual (vLLM's `residual = hidden_states`); the composed fallback adds
// into the residual ForwardLayers zeroed for it.
void AddRmsNorm(Dev d, DBuf& out, const Tensor& x, DBuf& res, const Tensor& w, float eps,
                const flint::FlintDeviceKernels* fk, bool first = false) {
  if (FusedNorm(fk, x, w)) {
    (first ? fk->rms_norm_store_residual : fk->add_rms_norm)(
        d.q, out.ptr(), x.data, res.ptr(), w.data, x.shape[0], x.shape[1], eps);
    return;
  }
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, out.t(), x, w, &res.t(), vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, out.t(), x, w, vt::RmsNormArgs{eps, false}, &res.t());
  }
}

// One decoder layer (vLLM FlintDecoderLayer): fused add + input RMSNorm ->
// attention -> fused add + post RMSNorm -> dense MLP or MoE. The bf16 residual
// stream accumulates in `res`; `hidden` becomes this layer's output delta.
void RunLayer(Dev d, const FlintPagedLayer& lw, const FlintParams& p, const HfConfig& cfg,
              Tensor& hidden, std::shared_ptr<void>& hidden_hold, DBuf& res,
              const StepInputs& si, const CommonAttentionMetadata& meta,
              const PagedKvCache& kv, int64_t T, const flint::FlintDeviceKernels* fk,
              bool first) {
  const int64_t H = p.hidden_size;
  const float eps = static_cast<float>(p.rms_norm_eps);
  Tensor w_in = ResidentWeight(d, lw.input_layernorm, {H});
  DBuf normed(d, DType::kBF16, {T, H});
  AddRmsNorm(d, normed, hidden, res, w_in, eps, fk, first);

  DBuf attn = Attention(d, lw, p, cfg, normed.t(), si, meta, kv, T, fk);

  Tensor w_post = ResidentWeight(d, lw.post_attention_layernorm, {H});
  DBuf normed2(d, DType::kBF16, {T, H});
  AddRmsNorm(d, normed2, attn.t(), res, w_post, eps, fk);

  DBuf out = lw.dense ? DenseMlp(d, lw, normed2.t()) : MoeBlock(d, lw.moe, p, normed2.t(), T, fk);
  hidden = out.t();
  hidden_hold = out.ReleaseShared();
}

void GatherRows(Dev d, void* dst, const Tensor& src, const std::vector<int32_t>& idx,
                int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(src.dtype);
  auto* dp = static_cast<char*>(dst);
  const auto* sp = static_cast<const char*>(src.data);
  for (size_t s = 0; s < idx.size(); ++s)
    d.b.Copy(d.q, dp + s * rb, sp + static_cast<size_t>(idx[s]) * rb, rb);
}

// vt::Embedding stays outside the decode graph: it reports bad ids through a
// device flag it copies back with an event per call (and allocates on a ring
// slot's first use), which a capture cannot hold. Inside the graph the decode
// prologue gathers the rows instead.
Tensor EmbedTable(Dev d, const FlintPagedWeights& w) {
  return ResidentWeight(d, w.embed_tokens, {w.params.vocab_size, w.params.hidden_size});
}

void EmbedInto(Dev d, DBuf& hidden, const Tensor& ids, const FlintPagedWeights& w) {
  vt::Embedding(d.q, hidden.t(), EmbedTable(d, w), ids);
}

void EmbedInto(Dev d, DBuf& hidden, const std::vector<int32_t>& token_ids,
               const FlintPagedWeights& w) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  DBuf ids(d, DType::kI32, {T}, token_ids.data());
  detail::ApplyDeviceTokenIds(d.b, d.q, ids.ptr(), T, "flint embed");
  EmbedInto(d, hidden, ids.t(), w);
}

// The capturable region: the layers, the final norm and the head, returning
// [n_out, vocab] f32 on the device. Without `persistent` the step inputs are
// uploaded here from the host vectors; the decode graph instead passes device
// buffers it refreshed itself OUTSIDE the capture, because H2D copies baked into
// a graph read pageable memory and stalled each replay (measured: 6 copies and
// ~120 us of GPU idle per decode step before the first kernel). The RoPE table
// is a function of the positions, so it is rebuilt here either way.
DBuf ForwardLayers(Dev d, const Tensor& hidden_in, const std::vector<int32_t>& positions,
                   const CommonAttentionMetadata& meta, const std::vector<PagedKvCache>& kv,
                   const FlintPagedWeights& w, const HfConfig& cfg,
                   const std::vector<int32_t>& logits_indices,
                   StepInputs* persistent = nullptr) {
  const FlintParams& p = w.params;
  const int64_t T = hidden_in.shape[0], H = p.hidden_size, V = p.vocab_size;
  const float eps = static_cast<float>(p.rms_norm_eps);
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "flint paged: positions length must match token_ids");
  VT_CHECK(static_cast<int64_t>(kv.size()) == p.num_hidden_layers,
           "flint paged: one PagedKvCache per layer required");
  VT_CHECK(d.b.GetResidualNormPolicy() == vt::ResidualNormPolicy::kMaterialized,
           "flint paged: only the materialized residual-norm policy is wired");

  DBuf res(d, DType::kBF16, {T, H});
  const flint::FlintDeviceKernels* fk = FusedKernels(d);
  // The first layer's fused norm writes the residual; only the composed one adds
  // into it.
  if (w.layers.empty() ||
      !FusedNorm(fk, hidden_in, ResidentWeight(d, w.layers.front().input_layernorm, {H})))
    res.Zero(d);
  StepInputs built;
  if (persistent == nullptr) {
    built = BuildStepInputs(d, positions, meta, cfg);
  } else {
    vt::RopeCosSinCache(d.q, persistent->cos_sin.t(), persistent->positions.t(),
                        MakeRopeArgs(cfg));
    if (fk == nullptr && RopeCacheEnabled())
      vt::CastBf16(d.q, persistent->cos_sin_bf16.t(), persistent->cos_sin.t());
  }
  const StepInputs& si = persistent != nullptr ? *persistent : built;
  Tensor hidden = hidden_in;
  std::shared_ptr<void> hidden_hold;
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    RunLayer(d, w.layers[static_cast<size_t>(l)], p, cfg, hidden, hidden_hold, res, si, meta,
             kv[static_cast<size_t>(l)], T, fk, /*first=*/l == 0);
  }
  DBuf normed(d, DType::kBF16, {T, H});
  AddRmsNorm(d, normed, hidden, res, ResidentWeight(d, w.final_norm, {H}), eps, fk);

  const bool gather =
      !logits_indices.empty() && static_cast<int64_t>(logits_indices.size()) < T;
  const int64_t n_out = gather ? static_cast<int64_t>(logits_indices.size()) : T;
  Tensor src = normed.t();
  DBuf gathered(d, DType::kBF16, gather ? std::vector<int64_t>{n_out, H}
                                        : std::vector<int64_t>{1, 1});
  if (gather) {
    GatherRows(d, gathered.ptr(), normed.t(), logits_indices, H);
    src = gathered.t();
  }
  // vLLM's LogitsProcessor: the bf16 head output times the head's global scale
  // in bf16 (`logits *= self.scale`), widened to f32 for the sampler.
  Tensor head_w = ResidentWeight(d, w.lm_head, {V, H});
  DBuf head(d, DType::kBF16, {n_out, V});
  vt::MatmulBT(d.q, head.t(), src, head_w);
  if (w.lm_head_scale != 1.0F) {
    DBuf scaled(d, DType::kBF16, {n_out, V});
    vt::MulScalar(d.q, scaled.t(), head.t(), static_cast<double>(w.lm_head_scale));
    head = std::move(scaled);
  }
  DBuf logits(d, DType::kF32, {n_out, V});
  vt::CastF32(d.q, logits.t(), head.t());
  return logits;
}

DBuf ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                 const std::vector<int32_t>& positions, const CommonAttentionMetadata& meta,
                 const std::vector<PagedKvCache>& kv, const FlintPagedWeights& w,
                 const HfConfig& cfg, const std::vector<int32_t>& logits_indices) {
  DBuf hidden(d, DType::kBF16, {static_cast<int64_t>(token_ids.size()), w.params.hidden_size});
  EmbedInto(d, hidden, token_ids, w);
  return ForwardLayers(d, hidden.t(), positions, meta, kv, w, cfg, logits_indices);
}

ForwardLogits WrapDeviceLogits(DBuf&& logits, int64_t rows, int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = logits.t();
  fl.device_storage = logits.ReleaseShared();
  return fl;
}

ForwardLogits ViewDeviceLogits(void* base, vt::Device device, int64_t rows, int64_t vocab) {
  ForwardLogits fl;
  fl.rows = rows;
  fl.vocab = vocab;
  fl.device_tensor = MakeTensor(base, DType::kF32, device, {rows, vocab});
  fl.device_storage = std::shared_ptr<void>(base, [](void*) {});
  fl.non_owning_view = true;  // releasing it frees nothing: the runner may skip its drain
  return fl;
}

// VT_ASYNC_EXECUTOR, on by default for a CUDA queue (qwen3_5.cpp
// DecodeGraphDoubleBufferEnabled, the runner's parse): the runner skips its
// pre-forward drain after a graphed step, and its depth-2 loop enqueues
// sample(i-1) after forward(i), so each padded size keeps two independent slots
// and alternates them; otherwise replay i+1 would overwrite logits i before they
// are sampled. VT_ASYNC_EXECUTOR_NO_DBUF=1 is the same test-only escape hatch.
bool DecodeGraphDoubleBufferEnabled(vt::DeviceType device) {
  if (!v1::AsyncExecutorFlagIsOn(std::getenv("VT_ASYNC_EXECUTOR"),
                                 platforms::GetPlatform(device).async_executor_default_on()))
    return false;
  const char* nod = std::getenv("VT_ASYNC_EXECUTOR_NO_DBUF");
  return !(nod != nullptr && nod[0] == '1' && nod[1] == '\0');
}

template <typename T>
void CopyInPlace(std::vector<T>& dst, const std::vector<T>& src) {
  if (dst.size() != src.size()) {
    dst = src;
  } else {
    std::copy(src.begin(), src.end(), dst.begin());
  }
}

// The pool the decode graph's retained step inputs come from (qwen3_5.cpp's
// PersistentDecodeInputPool): per device, separate from the scratch pool.
DevicePool& FlintInputPool(vt::Backend& b) {
  static detail::PoolTable table;
  return table.For(b);
}

// The S-padded pure-decode inputs (qwen3_moe.cpp BuildPaddedDecodeAttn): the
// padding rows write no KV (slot -1) and read block 0 for one position, and
// their logits are never returned.
void BuildPaddedDecode(int64_t S, const std::vector<int32_t>& tok,
                       const std::vector<int32_t>& pos, const CommonAttentionMetadata& am,
                       std::vector<int32_t>& tok_out, std::vector<int32_t>& pos_out,
                       CommonAttentionMetadata& am_out) {
  const int64_t cols = am.block_table_num_cols;
  tok_out.assign(static_cast<size_t>(S), 0);
  pos_out.assign(static_cast<size_t>(S), 0);
  std::copy(tok.begin(), tok.end(), tok_out.begin());
  std::copy(pos.begin(), pos.end(), pos_out.begin());
  am_out = am;
  am_out.num_reqs = static_cast<int>(S);
  am_out.num_actual_tokens = static_cast<int>(S);
  am_out.max_query_len = 1;
  am_out.uniform_spec_query_len = 0;
  am_out.slot_mapping.assign(static_cast<size_t>(S), -1);
  std::copy(am.slot_mapping.begin(), am.slot_mapping.end(), am_out.slot_mapping.begin());
  am_out.seq_lens.assign(static_cast<size_t>(S), 1);
  std::copy(am.seq_lens.begin(), am.seq_lens.end(), am_out.seq_lens.begin());
  am_out.block_table_tensor.assign(static_cast<size_t>(S * cols), 0);
  std::copy(am.block_table_tensor.begin(), am.block_table_tensor.end(),
            am_out.block_table_tensor.begin());
  am_out.query_start_loc.resize(static_cast<size_t>(S + 1));
  for (int64_t i = 0; i <= S; ++i)
    am_out.query_start_loc[static_cast<size_t>(i)] = static_cast<int32_t>(i);
}

}  // namespace

void PrepareFlintPaged(const FlintPagedWeights& weights, vt::Queue& queue) {
  // DSR-ALLOW(MODEL-TEXT-FLINT): the load-time Marlin build; its first line asks the runtime question, OpRegistered(kMoeGroupedGemmNvfp4Marlin, device).
#ifdef VT_MARLIN_NVFP4
  if (!vt::OpRegistered(vt::OpId::kMoeGroupedGemmNvfp4Marlin, queue.device.type)) return;
  Dev d{vt::GetBackend(queue.device.type), queue};
  for (const FlintPagedLayer& lw : weights.layers) {
    if (lw.dense) continue;
    FlintMoeMarlin& mr = ResidentIn<FlintMoeMarlin>(lw.moe.marlin);
    BuildFlintMoeMarlin(d, lw.moe, weights.params.hidden_size,
                        weights.params.moe_intermediate_size, mr);
  }
#else
  (void)weights;
  (void)queue;
#endif
}

v1::KVCacheConfig MakeFlintPagedKVCache(const HfConfig& config, int block_size,
                                        int num_blocks) {
  const FlintParams p = ParseFlintParams(config);
  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<v1::FullAttentionSpec>(block_size,
                                              static_cast<int>(p.num_key_value_heads),
                                              static_cast<int>(p.head_dim),
                                              v1::ResolveKvCacheDType()));
  return kv;
}

ForwardLogits FlintPagedModel::ForwardDevice(const std::vector<int32_t>& token_ids,
                                             const std::vector<int32_t>& positions,
                                             const CommonAttentionMetadata& attn_meta,
                                             const std::vector<PagedKvCache>& attn_kv,
                                             const FlintPagedWeights& weights,
                                             const HfConfig& config, vt::Queue& queue,
                                             const std::vector<int32_t>& logits_indices) {
  Dev d{vt::GetBackend(queue.device.type), queue};
  DBuf logits = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, weights, config,
                            logits_indices);
  const int64_t rows = logits.t().shape[0];
  return WrapDeviceLogits(std::move(logits), rows, weights.params.vocab_size);
}

// ─── FlintPagedDecodeGraph ────────────────────────────────────────────────────
// The Qwen3MoeDecodeGraph driver (qwen3_moe.cpp) with Flint's forward in the
// captured region: the MoE arenas and dense Marlin residents are built before
// capture, and the FA-2 decode scratch (keyed by shape and split count) is filled
// by an eager forward right before each capture.
//
// The step's inputs reach the graph through its first node, the decode prologue:
// the host writes them into the slot's pinned step block and the prologue reads
// it in place, so a replay is the only thing the step submits. In front of each
// replay there used to be ten small operations of Flint's (seven staged copies;
// the embedding's flag memset, gather and flag copy back), and on Windows the GPU
// idled between them and the runner's uploads: median 240 us and p90 1.4 ms per
// decode step at c=1, although the host had issued all of it milliseconds early.
// Without the prologue (VT_FLINT_GRAPH_PROLOGUE=0, or no device alias for pinned
// memory) the copies and the embedding stay outside the graph as before.
//
// The DevicePool needs the 35B driver's discipline (qwen3_5.cpp #1380 / #2274),
// which Qwen3MoeDecodeGraph does not carry: a capture must be served entirely
// from free blocks, and under async scheduling the previous step's logits are
// still held when the next size captures, so the free list can be one block short
// in some size class even though the cold step ran the same forward. Measured
// here at c=16 as `cudaMalloc: operation not permitted when stream is capturing`,
// which stopped the engine. So the cold step records the pool demand it made, the
// capture pre-grows to it outside the captured region, and the blocks the graph
// baked are pinned out of the free list until the graph goes away.
struct FlintPagedDecodeGraph::Impl {
  Impl(const FlintPagedWeights& w, const HfConfig& c, vt::Queue q, int64_t max_reqs)
      : weights(w), config(c), queue(q), max_num_reqs(max_reqs) {
    const char* local = std::getenv("VT_FLINT_CUDAGRAPH");
    const bool local_on = local == nullptr || local[0] != '0';
    Backend& b = vt::GetBackend(queue.device.type);
    enabled = vt::GraphCaptureEnabled() && local_on &&
              platforms::GetPlatform(queue.device.type).support_static_graph_mode() &&
              !platforms::GetPlatform(queue.device.type).static_graph_requires_opt_in() &&
              b.SupportsGraphCapture();
    dbuf = DecodeGraphDoubleBufferEnabled(queue.device.type);
    if (enabled && weights.params.hidden_size % 8 == 0)
      prologue = PrologueKernels(Dev{b, queue});
  }
  ~Impl() {
    Backend& b = vt::GetBackend(queue.device.type);
    for (auto& kv : slots) {
      for (SizeSlot& s : kv.second.slot) {
        Pool(b).UnpinForGraph(b, s.pinned);
        s.pinned.clear();
        s.UnbindInputs(b);
        if (s.staged.handle != nullptr) b.DestroyEvent(s.staged);
      }
    }
  }

  struct SizeSlot {
    std::vector<int32_t> token_ids;
    std::vector<int32_t> positions;
    CommonAttentionMetadata attn_meta;
    std::unique_ptr<DBuf> hidden;
    std::unique_ptr<DBuf> logits;
    StepTokenIds ids;
    vt::BreakableGraph graph;
    DevicePool::StepDemand demand;                  // the cold step's pool demand
    std::vector<std::pair<size_t, void*>> pinned;  // blocks the captured graph baked
    // Device step inputs the captured graph reads, refreshed every step by the
    // prologue inside the graph, or without it by staged copies before the replay.
    // From FlintInputPool so retaining them never empties a size class the
    // captured scratch needs.
    std::unique_ptr<StepInputs> dev;
    // With the prologue: the pinned step block it reads, and that block's device
    // alias. Otherwise the per-input staged cells.
    void* block = nullptr;
    void* block_dev = nullptr;
    flint::FlintStepLayout layout{};
    vt::PersistentStepInput in_positions, in_slots, in_block_table, in_seq_lens, in_qsl;
    // Recorded after the last use of this slot's pinned staging (the copies, or
    // the replay whose prologue reads the step block); the next refresh of this
    // slot waits on it, so the staging is not rewritten while that use may still
    // be queued (the async runner overlaps steps).
    vt::Event staged{};
    bool staged_pending = false;
    int fa_cols = -1;
    bool warm = false;
    int64_t replays = 0;

    // Waits out a use still reading the pinned staging before freeing it.
    void UnbindInputs(Backend& b) {
      if (staged_pending) {
        b.SynchronizeEvent(staged);
        staged_pending = false;
      }
      if (block != nullptr) b.FreePinned(block);
      block = nullptr;
      block_dev = nullptr;
      in_positions.Unbind();
      in_slots.Unbind();
      in_block_table.Unbind();
      in_seq_lens.Unbind();
      in_qsl.Unbind();
      dev.reset();
    }

    void Refresh(const std::vector<int32_t>& tok, const std::vector<int32_t>& pos,
                 const CommonAttentionMetadata& am) {
      CopyInPlace(token_ids, tok);
      CopyInPlace(positions, pos);
      CopyInPlace(attn_meta.slot_mapping, am.slot_mapping);
      CopyInPlace(attn_meta.block_table_tensor, am.block_table_tensor);
      CopyInPlace(attn_meta.seq_lens, am.seq_lens);
      CopyInPlace(attn_meta.query_start_loc, am.query_start_loc);
      attn_meta.num_reqs = am.num_reqs;
      attn_meta.num_actual_tokens = am.num_actual_tokens;
      attn_meta.max_query_len = am.max_query_len;
      attn_meta.max_seq_len = am.max_seq_len;
      attn_meta.block_table_num_cols = am.block_table_num_cols;
      attn_meta.causal = am.causal;
    }
  };

  // Builds this slot's retained device inputs (first capture at this size, or
  // after a block-table width change) and binds the pinned staging cells to them.
  void BindInputs(Dev d, SizeSlot& s, int64_t S) {
    s.UnbindInputs(d.b);
    {
      ActivePoolScope scope(&FlintInputPool(d.b));
      s.dev = std::make_unique<StepInputs>(BuildStepInputs(d, s.positions, s.attn_meta, config));
    }
    const auto cols = static_cast<size_t>(s.attn_meta.block_table_num_cols);
    const auto rows = static_cast<size_t>(S);
    if (s.staged.handle == nullptr) s.staged = d.b.CreateEvent(/*blocking=*/true);
    s.staged_pending = false;
    if (prologue != nullptr && EmbedTable(d, weights).dtype == DType::kBF16) {
      s.layout = flint::MakeFlintStepLayout(S, static_cast<int64_t>(cols));
      s.block = d.b.AllocPinned(s.layout.bytes);
      s.block_dev = prologue->device_alias(s.block);
      if (s.block_dev != nullptr) {
        std::memset(s.block, 0, s.layout.bytes);
        static_cast<flint::FlintStepHeader*>(s.block)->bad_row = -1;
        return;
      }
      d.b.FreePinned(s.block);
      s.block = nullptr;
    }
    s.in_positions.Bind(d.b, s.dev->positions.ptr(), sizeof(int32_t) * rows);
    s.in_slots.Bind(d.b, s.dev->slot_mapping.ptr(), sizeof(int64_t) * rows);
    s.in_block_table.Bind(d.b, s.dev->block_table.ptr(), sizeof(int32_t) * rows * cols);
    s.in_seq_lens.Bind(d.b, s.dev->seq_lens.ptr(), sizeof(int32_t) * rows);
    s.in_qsl.Bind(d.b, s.dev->query_start_loc.ptr(), sizeof(int32_t) * (rows + 1));
  }

  // Writes this step's padded inputs into the slot's step block once the replay
  // that last read it has run, and raises an id that replay found out of range
  // (vt::Embedding's contract, which also reports at the following call).
  void StageBlock(Dev d, SizeSlot& s, int64_t S, const detail::DeviceTokenIds& ov) {
    if (s.staged_pending) {
      d.b.SynchronizeEvent(s.staged);
      s.staged_pending = false;
    }
    auto* base = static_cast<char*>(s.block);
    auto* hdr = static_cast<flint::FlintStepHeader*>(s.block);
    if (hdr->bad_row >= 0) {
      const int32_t id = hdr->bad_id;
      hdr->bad_row = -1;
      throw std::runtime_error("flint decode: embedding id " + std::to_string(id) +
                               " out of range [0, " +
                               std::to_string(weights.params.vocab_size) + ")");
    }
    const auto& am = s.attn_meta;
    const auto rows = static_cast<size_t>(S);
    VT_CHECK(s.token_ids.size() == rows && s.positions.size() == rows &&
                 am.seq_lens.size() == rows && am.query_start_loc.size() == rows + 1 &&
                 am.slot_mapping.size() == rows &&
                 am.block_table_tensor.size() ==
                     rows * static_cast<size_t>(am.block_table_num_cols),
             "flint decode: the padded step inputs do not match the slot's step block");
    VT_CHECK(ov.ids == nullptr || ov.count <= S,
             "flint decode: the runner's device ids are longer than this step's padded input");
    const flint::FlintStepLayout& l = s.layout;
    hdr->override_ids = ov.ids;
    hdr->override_count = ov.ids != nullptr ? static_cast<int32_t>(ov.count) : 0;
    std::memcpy(base + l.token_ids, s.token_ids.data(), rows * sizeof(int32_t));
    std::memcpy(base + l.positions, s.positions.data(), rows * sizeof(int32_t));
    std::memcpy(base + l.seq_lens, am.seq_lens.data(), rows * sizeof(int32_t));
    std::memcpy(base + l.query_start_loc, am.query_start_loc.data(), (rows + 1) * sizeof(int32_t));
    std::memcpy(base + l.slots, am.slot_mapping.data(), rows * sizeof(int64_t));
    std::memcpy(base + l.block_table, am.block_table_tensor.data(),
                am.block_table_tensor.size() * sizeof(int32_t));
  }

  // The prologue over the slot's step block (eagerly, or as the captured graph's
  // first node).
  void LaunchPrologue(Dev d, SizeSlot& s, int64_t S) {
    StepInputs& si = *s.dev;
    prologue->decode_prologue(
        d.q, s.block_dev, static_cast<int32_t*>(si.positions.ptr()),
        static_cast<int64_t*>(si.slot_mapping.ptr()), static_cast<int32_t*>(si.block_table.ptr()),
        static_cast<int32_t*>(si.seq_lens.ptr()), static_cast<int32_t*>(si.query_start_loc.ptr()),
        s.hidden->ptr(), EmbedTable(d, weights).data, S, s.attn_meta.block_table_num_cols,
        weights.params.hidden_size, weights.params.vocab_size);
  }

  // The step block is free to rewrite once the work queued so far has run.
  void MarkStaged(Dev d, SizeSlot& s) {
    d.b.RecordEvent(s.staged, d.q);
    s.staged_pending = true;
  }

  // Copies this step's padded host inputs into the slot's device buffers.
  void StageInputs(Dev d, SizeSlot& s) {
    if (s.staged_pending) d.b.SynchronizeEvent(s.staged);
    const auto& am = s.attn_meta;
    s.in_positions.RefreshFromHost(d.q, s.positions.data(), sizeof(int32_t) * s.positions.size());
    s.in_slots.RefreshFromHost(d.q, am.slot_mapping.data(), sizeof(int64_t) * am.slot_mapping.size());
    s.in_block_table.RefreshFromHost(d.q, am.block_table_tensor.data(),
                                     sizeof(int32_t) * am.block_table_tensor.size());
    s.in_seq_lens.RefreshFromHost(d.q, am.seq_lens.data(), sizeof(int32_t) * am.seq_lens.size());
    s.in_qsl.RefreshFromHost(d.q, am.query_start_loc.data(),
                             sizeof(int32_t) * am.query_start_loc.size());
    d.b.RecordEvent(s.staged, d.q);
    s.staged_pending = true;
  }

  const FlintPagedWeights& weights;
  const HfConfig& config;
  vt::Queue queue;
  int64_t max_num_reqs = 0;
  bool enabled = false;
  bool dbuf = false;  // two slots per size, alternated (VT_ASYNC_EXECUTOR=1)
  const flint::FlintDeviceKernels* prologue = nullptr;
  // With dbuf off only slot[0] is used and slot[1] stays empty.
  struct SlotRing {
    SizeSlot slot[2];
    int next = 0;
  };
  std::map<int64_t, SlotRing> slots;
  int64_t replays = 0;
  bool any_captured = false;
};

FlintPagedDecodeGraph::FlintPagedDecodeGraph(const FlintPagedWeights& weights,
                                             const HfConfig& config, vt::Queue queue,
                                             int64_t max_num_reqs)
    : impl_(std::make_unique<Impl>(weights, config, queue, max_num_reqs)) {}

FlintPagedDecodeGraph::~FlintPagedDecodeGraph() = default;

bool FlintPagedDecodeGraph::captured() const { return impl_->any_captured; }
int64_t FlintPagedDecodeGraph::replay_count() const { return impl_->replays; }

ForwardLogits FlintPagedDecodeGraph::Step(const std::vector<int32_t>& token_ids,
                                          const std::vector<int32_t>& positions,
                                          const CommonAttentionMetadata& attn_meta,
                                          const std::vector<PagedKvCache>& attn_kv) {
  const int64_t B = static_cast<int64_t>(token_ids.size());
  Backend& b = vt::GetBackend(impl_->queue.device.type);
  Dev d{b, impl_->queue};
  // A fresh demand measurement per step; the cold step's profile is what the
  // capture at its size pre-grows to.
  Pool(b).MarkStepBoundary();
  const int64_t V = impl_->weights.params.vocab_size;
  const int64_t H = impl_->weights.params.hidden_size;
  const std::vector<int32_t> kNoGather;
  const int64_t S = PadToCaptureSize(B, impl_->max_num_reqs);
  if (!impl_->enabled || S < 0) {
    DBuf lg = ForwardBody(d, token_ids, positions, attn_meta, attn_kv, impl_->weights,
                          impl_->config, kNoGather);
    return WrapDeviceLogits(std::move(lg), B, V);
  }

  Impl::SlotRing& ring = impl_->slots[S];
  Impl::SizeSlot& s = ring.slot[impl_->dbuf ? ring.next : 0];
  if (impl_->dbuf) ring.next ^= 1;
  const int cols = attn_meta.block_table_num_cols;
  std::vector<int32_t> ptok, ppos;
  CommonAttentionMetadata pam;
  BuildPaddedDecode(S, token_ids, positions, attn_meta, ptok, ppos, pam);
  // A block-table width change moves the persistent block table the captured
  // copy reads, so that size re-warms and re-captures.
  const bool cols_changed = s.fa_cols != -1 && s.fa_cols != cols;
  s.Refresh(ptok, ppos, pam);
  s.fa_cols = cols;
  const detail::DeviceTokenIds ov = detail::TakeDeviceTokenIds();
  // The embedding outside the graph: the cold step, and every step without the
  // prologue.
  const auto embed_outside = [&] {
    s.ids.Ensure(d, S);
    s.ids.Refresh(d, s.token_ids, ov.ids, ov.count);
    EmbedInto(d, *s.hidden, s.ids.t(), impl_->weights);
  };
  if (cols_changed) {
    if (s.graph.captured()) {
      s.graph.Reset();
      Pool(b).UnpinForGraph(b, s.pinned);  // no graph, nothing baked
      s.pinned.clear();
      s.warm = false;
    }
    s.UnbindInputs(b);  // the block table it was sized for is gone
  }

  if (s.graph.captured()) {
    if (s.block != nullptr) {
      impl_->StageBlock(d, s, S, ov);
      s.graph.Replay(impl_->queue);
      impl_->MarkStaged(d, s);
    } else {
      impl_->StageInputs(d, s);
      embed_outside();
      s.graph.Replay(impl_->queue);
    }
    ++s.replays;
    ++impl_->replays;
    return ViewDeviceLogits(s.logits->ptr(), d.q.device, B, V);
  }

  if (s.warm) {
    // Outside the captured region, where a cudaMalloc is legal: the retained step
    // inputs (allocated once per size and width), this step's values staged into
    // them, and the scratch pool grown to the cold step's demand (a no-op once the
    // free list is deep enough).
    if (s.dev == nullptr) impl_->BindInputs(d, s, S);
    if (s.block != nullptr) {
      impl_->StageBlock(d, s, S, ov);
      impl_->LaunchPrologue(d, s, S);
    } else {
      impl_->StageInputs(d, s);
      embed_outside();
    }
    // This step's logits come from an EAGER forward over the same inputs, and the
    // capture after it only records. The order is what makes the capture safe:
    // the FA-2 decode scratch is keyed by the split count, which follows the
    // longest sequence (one level per 128 tokens), so a sequence that crossed a
    // level since the cold step meets a key that does not exist yet INSIDE the
    // capture, where it cannot be allocated (measured at c=16 with the double-
    // buffered slots, whose cold step is two steps back). The eager pass creates
    // exactly the key the capture will use.
    DBuf eager = ForwardLayers(d, s.hidden->t(), s.positions, s.attn_meta, attn_kv,
                               impl_->weights, impl_->config, kNoGather, s.dev.get());
    Pool(b).PreGrowForCapture(b, s.demand);
    std::optional<DBuf> lg;
    {
      // kFull: a decode capture is one segment with attention inside, vLLM's
      // FULL decode graph.
      vt::GraphCaptureScope scope(b, impl_->queue, s.graph, vt::GraphCaptureMode::kFull);
      if (s.block != nullptr) impl_->LaunchPrologue(d, s, S);
      lg = ForwardLayers(d, s.hidden->t(), s.positions, s.attn_meta, attn_kv, impl_->weights,
                         impl_->config, kNoGather, s.dev.get());
    }
    // After the eager prologue's read of the step block (the capture ran nothing).
    if (s.block != nullptr) impl_->MarkStaged(d, s);
    if (!s.graph.captured()) {
      s.warm = false;
      if (s.graph.capture_failed()) {
        // Under a failed capture nothing ran, so *lg holds no logits.
        const std::exception_ptr err = s.graph.capture_error();
        s.graph.Reset();
        if (err) std::rethrow_exception(err);
        VT_CHECK(false,
                 "FlintPagedDecodeGraph: the decode capture was abandoned and its logits "
                 "were never computed");
      }
    } else {
      // Take the blocks the graph baked back out of the free list (the pre-grow
      // left exactly those on top), BEFORE the old logits are freed into it.
      if (s.pinned.empty()) s.pinned = Pool(b).PinForGraph(b, s.demand);
      s.logits = std::make_unique<DBuf>(std::move(*lg));
      impl_->any_captured = true;
      s.replays = 0;
      if (std::getenv("VT_DECODE_GRAPH_STATS") != nullptr)
        std::fprintf(stderr,
                     "[FlintPagedDecodeGraph] captured decode graph for padded size S=%lld "
                     "(real B=%lld)\n",
                     static_cast<long long>(S), static_cast<long long>(B));
    }
    ForwardLogits fl = WrapDeviceLogits(std::move(eager), B, V);
    fl.device_tensor = MakeTensor(fl.device_storage.get(), DType::kF32, d.q.device, {B, V});
    return fl;
  }

  // Cold size: one eager step warms the pool, the residents and the FA-2 scratch
  // for S; the next step at this size captures.
  s.hidden = std::make_unique<DBuf>(d, DType::kBF16, std::vector<int64_t>{S, H});
  embed_outside();
  DBuf lg = ForwardLayers(d, s.hidden->t(), s.positions, s.attn_meta, attn_kv, impl_->weights,
                          impl_->config, kNoGather);
  s.warm = true;
  // The per-class peak this exact forward had live, taken before the logits leave.
  s.demand = Pool(b).StepDemandProfile();
  ForwardLogits fl = WrapDeviceLogits(std::move(lg), B, V);
  fl.device_tensor = MakeTensor(fl.device_storage.get(), DType::kF32, d.q.device, {B, V});
  return fl;
}

}  // namespace vllm
