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

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/layers/attention/attention.h"  // ResolveAttentionWindow
#include "vllm/model_executor/models/decode_graph_sizes.h"
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/dense_nvfp4_gemm.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"  // detail::*DeviceTokenIds
#include "vllm/model_executor/models/step_token_ids.h"
#include "vllm/platforms/interface.h"
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/backend.h"
#include "vt/breakable_graph.h"
#include "vt/ops.h"
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

// One MoE block: bf16 router logits -> sigmoid top-6 renormalised in f32 (the
// grouped router with a single group, which is plain top-k) -> moe_align -> w13
// GEMM -> SiluAndMul -> w2 GEMM -> the ungated shared expert -> weighted combine.
// The router GEMM stores bf16 because that is what vLLM's GateLinear does on
// SM120: its fp32-output tiers are gated on SM90 / SM100 (family 100 only) or
// on an out_dtype Flint does not set, so it falls to F.linear in the model dtype.
DBuf MoeBlock(Dev d, const FlintPagedMoe& m, const FlintParams& p, const Tensor& x, int64_t T) {
#ifdef VT_MARLIN_NVFP4
  const int64_t H = p.hidden_size, E = p.num_experts, Kk = p.num_experts_per_tok;
  const int64_t I = p.moe_intermediate_size, P = T * Kk;
  VT_CHECK(x.dtype == DType::kBF16, "flint paged: the Marlin MoE takes a bf16 activation");
  FlintMoeMarlin& mr = ResidentIn<FlintMoeMarlin>(m.marlin);
  if (!mr.ready) BuildFlintMoeMarlin(d, m, H, I, mr);

  Tensor router = ResidentWeight(d, m.router, {E, H});
  DBuf logits(d, DType::kBF16, {T, E});
  vt::MatmulBT(d.q, logits.t(), x, router);
  DBuf topk_w(d, DType::kF32, {T, Kk});
  DBuf topk_id(d, DType::kI32, {T, Kk});
  {
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
  vt::SiluAndMul(d.q, act.t(), gu.t());
  DBuf down(d, DType::kBF16, {P, H});
  {
    Tensor wq = MakeTensor(mr.w2, DType::kI32, d.q.device, {E, I / 16, H * 2});
    Tensor sc = MakeTensor(mr.s2, DType::kI8, d.q.device, {E, I / 16, H});
    Tensor gg = MakeTensor(mr.g2, DType::kF32, d.q.device, {E});
    vt::MoeGroupedGemmNvfp4Marlin(d.q, down.t(), act.t(), wq, sc, gg, ws, sorted.t(),
                                  expert_ids.t(), npad.t(), topk_w.t(),
                                  vt::MoeMarlinArgs{block, 1, Pi, Hi, Ii, false});
  }

  DBuf shared_act = GateUpAct(d, x, m.shared_gate, m.shared_up);
  DBuf shared = dense_nvfp4::MatmulNvfp4W4A16D(d, shared_act.t(), m.shared_down, DType::kBF16);

  DBuf out(d, DType::kBF16, {T, H});
  Tensor eo = Reshape(down.t(), {T, Kk, H});
  vt::MoeCombine(d.q, out.t(), eo, topk_w.t(), &shared.t());
  return out;
#else
  (void)d;
  (void)m;
  (void)p;
  (void)x;
  (void)T;
  VT_CHECK(false, "flint paged: the MoE needs the Marlin NVFP4 build (VLLM_CPP_MARLIN)");
  return DBuf();
#endif
}

// One decoder layer (vLLM FlintDecoderLayer): fused add + input RMSNorm ->
// attention -> fused add + post RMSNorm -> dense MLP or MoE. The bf16 residual
// stream accumulates in `res`; `hidden` becomes this layer's output delta.
void RunLayer(Dev d, const FlintPagedLayer& lw, const FlintParams& p, const HfConfig& cfg,
              Tensor& hidden, std::shared_ptr<void>& hidden_hold, DBuf& res,
              const StepInputs& si, const CommonAttentionMetadata& meta,
              const PagedKvCache& kv, int64_t T) {
  const int64_t H = p.hidden_size;
  const float eps = static_cast<float>(p.rms_norm_eps);
  Tensor w_in = ResidentWeight(d, lw.input_layernorm, {H});
  DBuf normed(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, normed.t(), hidden, w_in, vt::RmsNormArgs{eps, false}, &res.t());

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
  DBuf attn = AttnBlock(d, lw.attn, cfg, normed.t(), si, meta, kv, T, nullptr, opts);

  Tensor w_post = ResidentWeight(d, lw.post_attention_layernorm, {H});
  DBuf normed2(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, normed2.t(), attn.t(), w_post, vt::RmsNormArgs{eps, false}, &res.t());

  DBuf out = lw.dense ? DenseMlp(d, lw, normed2.t()) : MoeBlock(d, lw.moe, p, normed2.t(), T);
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

// The embedding stays outside the decode graph: vt::Embedding allocates a device
// flag and syncs (the qwen3_moe.cpp EmbedInto note).
void EmbedInto(Dev d, DBuf& hidden, const Tensor& ids, const FlintPagedWeights& w) {
  Tensor table = ResidentWeight(d, w.embed_tokens, {w.params.vocab_size, w.params.hidden_size});
  vt::Embedding(d.q, hidden.t(), table, ids);
}

void EmbedInto(Dev d, DBuf& hidden, const std::vector<int32_t>& token_ids,
               const FlintPagedWeights& w) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  DBuf ids(d, DType::kI32, {T}, token_ids.data());
  detail::ApplyDeviceTokenIds(d.b, d.q, ids.ptr(), T, "flint embed");
  EmbedInto(d, hidden, ids.t(), w);
}

// The capturable region: the layers, the final norm and the head, returning
// [n_out, vocab] f32 on the device. Per-step inputs are read from host vectors
// the decode graph keeps at fixed addresses.
DBuf ForwardLayers(Dev d, const Tensor& hidden_in, const std::vector<int32_t>& positions,
                   const CommonAttentionMetadata& meta, const std::vector<PagedKvCache>& kv,
                   const FlintPagedWeights& w, const HfConfig& cfg,
                   const std::vector<int32_t>& logits_indices) {
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
  res.Zero(d);
  const StepInputs si = BuildStepInputs(d, positions, meta, cfg);
  Tensor hidden = hidden_in;
  std::shared_ptr<void> hidden_hold;
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    RunLayer(d, w.layers[static_cast<size_t>(l)], p, cfg, hidden, hidden_hold, res, si, meta,
             kv[static_cast<size_t>(l)], T);
  }
  DBuf normed(d, DType::kBF16, {T, H});
  vt::RmsNorm(d.q, normed.t(), hidden, ResidentWeight(d, w.final_norm, {H}),
              vt::RmsNormArgs{eps, false}, &res.t());

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
  return fl;
}

template <typename T>
void CopyInPlace(std::vector<T>& dst, const std::vector<T>& src) {
  if (dst.size() != src.size()) {
    dst = src;
  } else {
    std::copy(src.begin(), src.end(), dst.begin());
  }
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
// captured region: the embedding runs outside, the MoE arenas and dense Marlin
// residents are built before capture, and the FA-2 decode scratch is keyed per
// shape and filled by the cold step.
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
  }
  ~Impl() {
    Backend& b = vt::GetBackend(queue.device.type);
    for (auto& kv : slots) {
      Pool(b).UnpinForGraph(b, kv.second.pinned);
      kv.second.pinned.clear();
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
    int fa_cols = -1;
    bool warm = false;
    int64_t replays = 0;

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

  const FlintPagedWeights& weights;
  const HfConfig& config;
  vt::Queue queue;
  int64_t max_num_reqs = 0;
  bool enabled = false;
  std::map<int64_t, SizeSlot> slots;
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

  Impl::SizeSlot& s = impl_->slots[S];
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
  s.ids.Ensure(d, S);
  s.ids.Refresh(d, s.token_ids, ov.ids, ov.count);
  if (cols_changed && s.graph.captured()) {
    s.graph.Reset();
    Pool(b).UnpinForGraph(b, s.pinned);  // no graph, nothing baked
    s.pinned.clear();
    s.warm = false;
  }

  if (s.graph.captured()) {
    EmbedInto(d, *s.hidden, s.ids.t(), impl_->weights);
    s.graph.Replay(impl_->queue);
    ++s.replays;
    ++impl_->replays;
    return ViewDeviceLogits(s.logits->ptr(), d.q.device, B, V);
  }

  if (s.warm) {
    // Outside the captured region, where a cudaMalloc is legal; a no-op once the
    // free list is deep enough.
    Pool(b).PreGrowForCapture(b, s.demand);
    EmbedInto(d, *s.hidden, s.ids.t(), impl_->weights);
    std::optional<DBuf> lg;
    {
      // kFull: a decode capture is one segment with attention inside, vLLM's
      // FULL decode graph.
      vt::GraphCaptureScope scope(b, impl_->queue, s.graph, vt::GraphCaptureMode::kFull);
      lg = ForwardLayers(d, s.hidden->t(), s.positions, s.attn_meta, attn_kv, impl_->weights,
                         impl_->config, kNoGather);
    }
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
      ForwardLogits drained = WrapDeviceLogits(std::move(*lg), B, V);
      drained.device_tensor =
          MakeTensor(drained.device_storage.get(), DType::kF32, d.q.device, {B, V});
      return drained;
    }
    // Take the blocks the graph baked back out of the free list (the pre-grow
    // left exactly those on top), BEFORE the old logits are freed into it.
    if (s.pinned.empty()) s.pinned = Pool(b).PinForGraph(b, s.demand);
    s.logits = std::make_unique<DBuf>(std::move(*lg));
    impl_->any_captured = true;
    if (std::getenv("VT_DECODE_GRAPH_STATS") != nullptr)
      std::fprintf(stderr,
                   "[FlintPagedDecodeGraph] captured decode graph for padded size S=%lld "
                   "(real B=%lld)\n",
                   static_cast<long long>(S), static_cast<long long>(B));
    s.graph.Replay(impl_->queue);
    s.replays = 1;
    ++impl_->replays;
    return ViewDeviceLogits(s.logits->ptr(), d.q.device, B, V);
  }

  // Cold size: one eager step warms the pool, the residents and the FA-2 scratch
  // for S; the next step at this size captures.
  s.hidden = std::make_unique<DBuf>(d, DType::kBF16, std::vector<int64_t>{S, H});
  EmbedInto(d, *s.hidden, s.ids.t(), impl_->weights);
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
