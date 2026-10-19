// KV-MAMBA-ALIGN (.agents/specs/mamba-align-retention.md): when prefix caching
// puts a hybrid's recurrent layers in Mamba "align" mode, and the attention
// block size that mode needs.
//
// Ported from vllm @ 0.28.0:
//   model_executor/models/config.py::MambaModelConfig.verify_and_update_config
//     (align whenever prefix caching is on for a hybrid)
//   platforms/interface.py::_align_hybrid_block_size
//     (B = A * cdiv(mamba_page, A * attn_page_per_token))
// The shipped specs come from MakeQwen3_5KVCacheSpec, the registry helper the
// loader builds Qwen3.5 / 3.6 KV configs with, so the arithmetic is checked on
// the pages the runner allocates.
#include <doctest/doctest.h>

#include <optional>
#include <stdexcept>

#include "vllm/config/speculative.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/qwen3_5_common.h"

using vllm::HfConfig;
using vllm::entrypoints::EngineParams;
using vllm::entrypoints::LoadedEngine;

namespace {

// The recurrent and attention geometry of a Qwen3.5 / 3.6 checkpoint; only the
// fields MakeQwen3_5KVCacheSpec reads.
HfConfig GdnGeometry(int64_t linear_num_value_heads) {
  HfConfig c;
  c.num_key_value_heads = 2;
  c.head_dim = 256;
  c.linear_num_key_heads = 16;
  c.linear_key_head_dim = 128;
  c.linear_num_value_heads = linear_num_value_heads;
  c.linear_value_head_dim = 128;
  c.linear_conv_kernel_dim = 4;
  c.mamba_ssm_dtype = "float32";
  return c;
}

}  // namespace

TEST_CASE("Mamba align: prefix caching on a supporting hybrid selects it") {
  EngineParams params;
  vllm::ModelInfo decoder;
  vllm::ModelInfo hybrid;
  hybrid.is_hybrid = true;
  vllm::ModelInfo align_hybrid = hybrid;
  align_hybrid.supports_mamba_align = true;
  const std::optional<vllm::SpeculativeConfig> no_spec;

  // Default: hybrids keep prefix caching off, so nothing is aligned or refused.
  CHECK_FALSE(LoadedEngine::UsesMambaAlign(params, align_hybrid, no_spec));
  CHECK_FALSE(LoadedEngine::UsesMambaAlign(params, hybrid, no_spec));

  params.enable_prefix_caching = true;
  CHECK_FALSE(LoadedEngine::UsesMambaAlign(params, decoder, no_spec));
  CHECK(LoadedEngine::UsesMambaAlign(params, align_hybrid, no_spec));
  // A hybrid that cannot keep its state at block boundaries is refused rather
  // than resumed from another request's state.
  CHECK_THROWS_AS(LoadedEngine::UsesMambaAlign(params, hybrid, no_spec),
                  std::invalid_argument);
  // So is align together with speculative decoding (phase 1).
  const std::optional<vllm::SpeculativeConfig> mtp =
      vllm::SpeculativeConfig::ResolveMtp(/*mtp_num_hidden_layers=*/1,
                                          /*num_speculative_tokens=*/1);
  CHECK_THROWS_AS(LoadedEngine::UsesMambaAlign(params, align_hybrid, mtp),
                  std::invalid_argument);
}

TEST_CASE("Mamba align: the attention block holds one recurrent state page") {
  // Qwen3.6-35B-A3B: 2,048 attention bytes per token; 49,152 bytes of bf16
  // conv state plus 2,097,152 of fp32 recurrent state. vLLM 0.28.0 logs
  // "Setting attention block size to 1056 tokens" for this checkpoint.
  const auto qwen36_35b = vllm::MakeQwen3_5KVCacheSpec(GdnGeometry(32), /*block_size=*/16,
                                                       /*num_blocks=*/1, /*num_spec=*/0);
  CHECK(LoadedEngine::MambaAlignBlockSize(qwen36_35b, 16) == 1056);
  // A coarser alignment rounds to its own multiple (64 * 17).
  CHECK(LoadedEngine::MambaAlignBlockSize(qwen36_35b, 64) == 1088);
  // Qwen3.5-2B: 16 value heads halve the recurrent state.
  const auto qwen35_2b = vllm::MakeQwen3_5KVCacheSpec(GdnGeometry(16), /*block_size=*/16,
                                                      /*num_blocks=*/1, /*num_spec=*/0);
  CHECK(LoadedEngine::MambaAlignBlockSize(qwen35_2b, 16) == 544);
}
