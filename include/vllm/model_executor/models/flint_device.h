// Flint CUDA glue table (OpId::kFlint), the laguna_device.h pattern: the small
// fused kernels vLLM runs for Flint on SM120, registered by
// src/vt/cuda/cuda_flint.cu on kCUDA and resolved here by the caller's device. No
// CUDA code; on a build or device without the kernels
// FlintDeviceKernelsAvailable(device) is false and the paged arm composes the
// shared ops instead.
#pragma once

#include <cstddef>
#include <cstdint>

#include "vt/device.h"  // vt::Queue

namespace vllm::flint {

// One decode step's inputs for the decode graph's prologue, in pinned host
// memory the prologue reads through unified addressing: this header, then the
// arrays at the FlintStepLayout offsets for the slot's (rows, cols).
struct FlintStepHeader {
  int32_t bad_row;         // -1, or written by the prologue: a row whose id is out of range
  int32_t bad_id;          // that id (clamped for the gather)
  int32_t override_count;  // rows [0, override_count) read their id from override_ids
  int32_t reserved;
  const int32_t* override_ids;  // the runner's device input ids, or null
};

struct FlintStepLayout {
  size_t token_ids, positions, seq_lens, query_start_loc, slots, block_table, bytes;
};

inline FlintStepLayout MakeFlintStepLayout(int64_t rows, int64_t cols) {
  const auto r = static_cast<size_t>(rows);
  const auto up = [](size_t x) { return (x + 15) & ~size_t{15}; };
  FlintStepLayout l{};
  l.token_ids = up(sizeof(FlintStepHeader));
  l.positions = up(l.token_ids + r * sizeof(int32_t));
  l.seq_lens = up(l.positions + r * sizeof(int32_t));
  l.query_start_loc = up(l.seq_lens + r * sizeof(int32_t));
  l.slots = up(l.query_start_loc + (r + 1) * sizeof(int32_t));
  l.block_table = up(l.slots + r * sizeof(int64_t));
  l.bytes = up(l.block_table + r * static_cast<size_t>(cols) * sizeof(int32_t));
  return l;
}

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
  // The first layer's form, vLLM's `residual = hidden_states` followed by the norm
  // without a residual: residual := x unchanged, out = bf16(x * rsqrt(mean(x^2) +
  // eps) * w). Saves zeroing a residual to add into (a memset node in the graph).
  void (*rms_norm_store_residual)(vt::Queue&, void* out, const void* x, void* residual,
                                  const void* w, int64_t rows, int64_t hidden, float eps);
  // vLLM csrc/activation_kernels.cu act_and_mul<silu>, what fused_marlin_moe runs
  // between the routed w13 and w2 GEMMs: out = bf16(bf16(silu(g)) * u), the SiLU
  // rounded to bf16 BEFORE the product (silu_kernel returns the element type);
  // precise expf. The dense MLP and the shared expert use Flint's own
  // silu_and_mul instead, bf16(silu(g) * u) with one rounding (vt::SiluAndMul).
  //   gu [rows, 2 d] bf16 (gate half, then up half), out [rows, d]; d % 8 == 0.
  void (*routed_silu_and_mul)(vt::Queue&, void* out, const void* gu, int64_t rows, int64_t d);
  // The device address of a pinned host block (what decode_prologue reads), or
  // nullptr when this device cannot read it in place.
  void* (*device_alias)(void* pinned);
  // The decode graph's first node. Copies one step's inputs from `block` (the
  // device alias of a pinned FlintStepHeader block laid out by
  // MakeFlintStepLayout(rows, cols)) into the device buffers the captured layers
  // read, and gathers each row's bf16 embedding: the id is override_ids[row] for
  // row < override_count, else the block's own. An id outside [0, vocab) is
  // clamped and reported in the header (bad_row / bad_id), which the host reads
  // once the step has run. Inside the graph, so a replay makes no copy of its own.
  //   embed [vocab, hidden] bf16, hidden_out [rows, hidden] bf16; hidden % 8 == 0
  void (*decode_prologue)(vt::Queue&, void* block, int32_t* positions, int64_t* slots,
                          int32_t* block_table, int32_t* seq_lens, int32_t* query_start_loc,
                          void* hidden_out, const void* embed, int64_t rows, int64_t cols,
                          int64_t hidden, int64_t vocab);
};

// The table registered for `device`, or throws when nothing registered it
// there. Only cuda_flint.cu registers one, so a CPU or Vulkan queue finds none.
const FlintDeviceKernels* FlintDevice(vt::DeviceType device);
bool FlintDeviceKernelsAvailable(vt::DeviceType device);

}  // namespace vllm::flint
