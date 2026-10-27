// RecursiveSI Flint (`FlintForCausalLM`) on the paged CUDA engine: many
// sequences per step over the runner's KV cache, NVFP4 W4A16 weights kept in
// FP4, and a decode CUDA graph. The CPU reference in flint.h stays the oracle;
// this arm is what the server runs on CUDA.
//
// Grounding: vLLM nightly `vllm/models/flint/model.py` (FlintAttention,
// FlintSparseMoeBlock, FlintDecoderLayer) and `gguf_nvfp4.py` (the official
// NVFP4 GGUF read as ModelOpt NVFP4, run W4A16), with the stock SM120 layers:
// Marlin NVFP4 for the linears and the MoE, FlashInfer attention with sinks.
//
// Shape of the forward, per layer:
//   input RMSNorm -> merged qkv (NVFP4) -> per-head q/k RMSNorm -> NeoX RoPE on
//   sliding layers only -> paged attention (window 513 on sliding layers, the
//   learned per-head sinks on every layer) -> o_proj -> post RMSNorm ->
//   layers 0-1: SwiGLU MLP; the rest: sigmoid top-6 renormalised router over 256
//   experts (Marlin grouped GEMM, fused gate|up) plus the ungated shared expert.
//
// KV cache: ONE full-attention group for all 49 layers, with the window applied
// in the attention kernel. That is how Gemma3/Gemma4/Laguna run here; the
// runner does not forward a second group's block table and slot mapping to a
// model. The sliding layers therefore keep their whole history (2 KiB per token
// per layer); vLLM's hybrid allocator frees it. Same attention result, more KV.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "vllm/model_executor/models/flint.h"
#include "vllm/model_executor/models/qwen3.h"           // Qwen3DenseAttnWeights, PagedKvCache
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor, Nvfp4Weight, ResidentSlot
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"  // CommonAttentionMetadata
#include "vt/device.h"

namespace vllm {

class GgufFile;

// One MoE layer: bf16 router, 256 NVFP4 experts and the NVFP4 shared expert.
// The routed experts are repacked into one Marlin arena at prepare time
// (`marlin`), after which their host bytes are released.
struct FlintPagedMoe {
  OwnedTensor router;                    // bf16 [E, H] (MatmulBT B operand)
  std::vector<Nvfp4Weight> expert_gate;  // [E] of [I, H]
  std::vector<Nvfp4Weight> expert_up;    // [E] of [I, H]
  std::vector<Nvfp4Weight> expert_down;  // [E] of [H, I]
  Nvfp4Weight shared_gate;               // [Is, H]
  Nvfp4Weight shared_up;                 // [Is, H]
  Nvfp4Weight shared_down;               // [H, Is]
  ResidentSlot marlin;
};

struct FlintPagedLayer {
  bool sliding = false;  // causal window of sliding_window tokens
  bool rope = false;     // NeoX RoPE (sliding layers); full layers are NoPE
  bool dense = false;    // layers 0-1: SwiGLU MLP instead of the MoE
  OwnedTensor input_layernorm;           // bf16 [H]
  OwnedTensor post_attention_layernorm;  // bf16 [H]
  // qkv_proj_fp4 is q|k|v merged by rows (the file gives the three one shared
  // global scale per layer, so the merge is exact); q_norm/k_norm bf16 [Dh].
  Qwen3DenseAttnWeights attn;
  OwnedTensor sinks;  // f32 [Hq]
  Nvfp4Weight mlp_gate;  // dense layers: [I_dense, H]
  Nvfp4Weight mlp_up;    // [I_dense, H]
  Nvfp4Weight mlp_down;  // [H, I_dense]
  FlintPagedMoe moe;
};

struct FlintPagedWeights {
  FlintParams params;
  OwnedTensor embed_tokens;  // bf16 [V, H]
  OwnedTensor final_norm;    // bf16 [H]
  // The NVFP4 head decoded to bf16 WITHOUT its global scale, which is exact
  // (e2m1 x e4m3 fits a bf16 mantissa), and the global scale applied to the f32
  // logits — vLLM's `_GGUF_DEQUANTIZED` lm_head and `logits_processor.scale`.
  OwnedTensor lm_head;  // bf16 [V, H] (MatmulBT B operand)
  float lm_head_scale = 1.0F;
  std::vector<FlintPagedLayer> layers;
};

// Reads the official NVFP4 GGUF (RecursiveSI/flint-nvfp4-gguf) keeping every
// NVFP4 tensor packed; bf16 and f32 tensors are taken as they are.
FlintPagedWeights LoadFlintPagedFromGguf(const GgufFile& gguf, const HfConfig& config);

// Repacks every MoE layer's routed experts into its Marlin arena and releases
// their host bytes. Idempotent; run once before the first forward so no
// allocation lands inside a CUDA-graph capture.
void PrepareFlintPaged(const FlintPagedWeights& weights, vt::Queue& queue);

// The single full-attention KV group every layer shares (see the file comment).
v1::KVCacheConfig MakeFlintPagedKVCache(const HfConfig& config, int block_size,
                                        int num_blocks);

class FlintPagedModel {
 public:
  // Batched paged forward over the flattened step tokens; attn_kv holds one
  // PagedKvCache per layer. Returns device f32 logits [n_out, vocab].
  static ForwardLogits ForwardDevice(const std::vector<int32_t>& token_ids,
                                     const std::vector<int32_t>& positions,
                                     const v1::CommonAttentionMetadata& attn_meta,
                                     const std::vector<PagedKvCache>& attn_kv,
                                     const FlintPagedWeights& weights,
                                     const HfConfig& config, vt::Queue& queue,
                                     const std::vector<int32_t>& logits_indices = {});
};

// Pure-decode CUDA-graph driver, the Qwen3MoeDecodeGraph state machine (cold
// eager step, then capture, then replay per padded batch size). Enabled when
// the backend captures; VLLM_CPP_CUDAGRAPH=0 or VT_FLINT_CUDAGRAPH=0 run every
// decode step eagerly.
class FlintPagedDecodeGraph {
 public:
  FlintPagedDecodeGraph(const FlintPagedWeights& weights, const HfConfig& config,
                        vt::Queue queue, int64_t max_num_reqs);
  ~FlintPagedDecodeGraph();
  FlintPagedDecodeGraph(const FlintPagedDecodeGraph&) = delete;
  FlintPagedDecodeGraph& operator=(const FlintPagedDecodeGraph&) = delete;

  ForwardLogits Step(const std::vector<int32_t>& token_ids,
                     const std::vector<int32_t>& positions,
                     const v1::CommonAttentionMetadata& attn_meta,
                     const std::vector<PagedKvCache>& attn_kv);

  bool captured() const;
  int64_t replay_count() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vllm
