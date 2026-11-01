// A GGUF's own generation config: the stop ids and sampling defaults that an
// HF checkpoint keeps in generation_config.json.
//
// Upstream reads that file through ModelConfig.try_get_generation_config
// (config/model.py) for two things: the extra stop ids that
// SamplingParams.update_from_generation_config merges into stop_token_ids
// (sampling_params.py:775-803), and the sampling defaults that
// get_diff_sampling_param narrows. A `.gguf` has no sibling file, so a GGUF
// load had neither. For Flint that is fatal to chat: its assistant turn has no
// end marker and ends when the model writes the next role token (<|user|>
// after an answer, <|observation|> after a tool call), while the GGUF's
// tokenizer eos is <|end_of_text|> alone, so every chat turn ran to max_tokens.
//
// The file carries both halves itself:
//   - `tokenizer.ggml.eog_token_ids`, the end-of-generation ids. The published
//     Flint NVFP4 GGUF lists 150001/150003/150005/150018, the same list as the
//     checkpoint's generation_config.json eos_token_id.
//   - `general.sampling.*`, the sampling keys llama.cpp reads as defaults.
//     The Flint, Qwen3.6 and Qwen3.8 GGUFs carry temp/top_p/top_k equal to
//     their checkpoints' generation_config.json.
#pragma once

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// Fills `config.generation_config_eos_ids` (sorted, unique) and every unset
// field of `config.generation_config_sampling` from `gguf`. What the
// architecture's builder already set is kept. A missing or wrong-typed key
// leaves its field unset and never throws, the same polarity as the sibling
// generation_config.json read in hf_config.cpp.
void ApplyGgufGenerationConfig(const GgufFile& gguf, HfConfig& config);

}  // namespace vllm
