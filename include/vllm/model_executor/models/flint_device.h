// Flint CUDA glue table (OpId::kFlint), the laguna_device.h pattern: the small
// fused kernels vLLM runs for Flint on SM120, registered by
// src/vt/cuda/cuda_flint.cu on kCUDA and resolved here. No CUDA code; on a build
// without the kernels FlintDeviceKernelsAvailable() is false and the paged arm
// composes the shared ops instead.
#pragma once

#include <cstdint>

#include "vt/device.h"  // vt::Queue

namespace vllm::flint {

struct FlintDeviceKernels {
  // vLLM models/flint/ops/qk_norm_rope.py, `_qk_norm_rope_cache_kernel`, with its
  // numerics: per-head RMSNorm of q and k computed in f32 and rounded to bf16
  // (bf16(x * rsqrt(mean(x^2) + eps) * w)); NeoX RoPE in f32 from an f32 cos|sin
  // table, rounded to bf16 once; v copied unchanged. One launch replaces
  // QkvSplit + RmsNorm(q) + RmsNorm(k) + RoPE + ReshapeAndCache.
  //   qkv      [T, (Hq + 2 Hkv) * D] bf16, row stride `qkv_stride` elements
  //   q_out    [T, Hq, D] bf16 (contiguous)
  //   q_w, k_w [D] bf16
  //   cos_sin  [T, D] f32, token-indexed (cos half then sin half), or nullptr for
  //            a NoPE layer
  //   k_cache, v_cache  bf16 pages addressed as
  //            (slot / block_size) * stride_block + (slot % block_size) *
  //            stride_slot + kv_head * stride_head (elements, innermost dim 1)
  //   slots    [T] i64; a negative slot writes nothing (padding rows)
  void (*qk_norm_rope_kv_cache)(vt::Queue&, void* q_out, const void* qkv, int64_t qkv_stride,
                                const void* q_w, const void* k_w, const float* cos_sin,
                                void* k_cache, void* v_cache, int64_t stride_block,
                                int64_t stride_slot, int64_t stride_head, int64_t block_size,
                                const int64_t* slots, int64_t num_tokens, int64_t num_q_heads,
                                int64_t num_kv_heads, int64_t head_dim, float eps);
  // Sigmoid top-k with renormalisation over bf16 router logits [T, E], vLLM's
  // fused_topk with scoring_func="sigmoid", renormalize=True: scores =
  // sigmoid(logits) in f32, the k largest (ties to the lower expert index),
  // weights = score / sum of the k scores. One warp per token; E <= 512, k <= 16.
  //   weights [T, k] f32, ids [T, k] i32
  void (*sigmoid_topk)(vt::Queue&, float* weights, int32_t* ids, const void* logits,
                       int64_t num_tokens, int64_t num_experts, int64_t top_k);
  // vLLM models/flint/ops/norm.py `_rms_norm_kernel`: s = x (+ residual) in f32;
  // with a residual, residual := bf16(s); out = bf16(s * rsqrt(mean(s^2) + eps) * w)
  // from the UNROUNDED s, rsqrt being rsqrt.approx.ftz.f32. bf16 [rows, hidden]
  // (contiguous), w [hidden]; hidden % 8 == 0 and <= 8192. out may alias x.
  void (*add_rms_norm)(vt::Queue&, void* out, const void* x, void* residual, const void* w,
                       int64_t rows, int64_t hidden, float eps);
  // vLLM csrc/activation_kernels.cu act_and_mul<silu>, what fused_marlin_moe runs
  // between the routed w13 and w2 GEMMs: out = bf16(bf16(silu(g)) * u), the SiLU
  // rounded to bf16 BEFORE the product (silu_kernel returns the element type);
  // precise expf. The dense MLP and the shared expert use Flint's own
  // silu_and_mul instead, bf16(silu(g) * u) with one rounding (vt::SiluAndMul).
  //   gu [rows, 2 d] bf16 (gate half, then up half), out [rows, d]; d % 8 == 0.
  void (*routed_silu_and_mul)(vt::Queue&, void* out, const void* gu, int64_t rows, int64_t d);
};

// The kCUDA table, or throws when nothing registered it.
const FlintDeviceKernels* FlintDevice();
bool FlintDeviceKernelsAvailable();

}  // namespace vllm::flint
