// Ported from: vllm/v1/structured_output/backend_xgrammar.py:36 @ pin 555967922.
// See include/vllm/v1/structured_output/backend_xgrammar.h for scope, the §9
// reuse-the-native-matcher decision, and the parity rationale.
#include "vllm/v1/structured_output/backend_xgrammar.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/v1/structured_output/backend_native.h"
#include "vllm/v1/structured_output/backend_types.h"
#include "vllm/v1/structured_output/xgrammar_json_schema.h"

namespace vllm::v1 {

XgrammarStructuredOutputBackend::XgrammarStructuredOutputBackend(
    const tok::Tokenizer& tokenizer, int vocab_size, bool disable_any_whitespace,
    std::vector<int32_t> stop_token_ids)
    : inner_(std::make_unique<NativeStructuredOutputBackend>(
          tokenizer, vocab_size, std::move(stop_token_ids))),
      disable_any_whitespace_(disable_any_whitespace) {}

XgrammarStructuredOutputBackend::~XgrammarStructuredOutputBackend() = default;

std::unique_ptr<StructuredOutputGrammar>
XgrammarStructuredOutputBackend::compile_grammar(
    StructuredOutputOptions request_type, const std::string& grammar_spec,
    const std::vector<int32_t>& stop_token_ids) {
  // backend_xgrammar.py:78-126. any_whitespace = not disable_any_whitespace
  // (backend_xgrammar.py:83).
  const bool any_whitespace = !disable_any_whitespace_;
  switch (request_type) {
    case StructuredOutputOptions::kJson: {
      // compile_json_schema(grammar_spec, any_whitespace=...) (py:82-84). The
      // xgrammar-faithful schema->EBNF converter (declaration key order +
      // basic_* + ws), then the native matcher over the emitted EBNF.
      const std::string ebnf =
          XgrammarJsonSchemaToEbnf(grammar_spec, any_whitespace);
      return inner_->compile_grammar(StructuredOutputOptions::kGrammar, ebnf,
                                     stop_token_ids);
    }
    case StructuredOutputOptions::kJsonObject: {
      // compile_json_schema('{"type": "object"}', ...) (py:85-88).
      const std::string ebnf = XgrammarJsonObjectEbnf(any_whitespace);
      return inner_->compile_grammar(StructuredOutputOptions::kGrammar, ebnf,
                                     stop_token_ids);
    }
    case StructuredOutputOptions::kGrammar:
    case StructuredOutputOptions::kRegex:
    case StructuredOutputOptions::kChoice:
    case StructuredOutputOptions::kStructuralTag:
      // compile_grammar / compile_regex / structural_tag (py:89-110): xgrammar
      // accepts the same EBNF/GBNF dialect + regex + structural-tag surface our
      // native paths already implement — delegate. (JSON is the W1
      // xgrammar-distinct brick; these stay on the proven native lowering.)
      return inner_->compile_grammar(request_type, grammar_spec, stop_token_ids);
  }
  throw std::runtime_error("xgrammar backend: unsupported request type");
}

TokenBitmask XgrammarStructuredOutputBackend::allocate_token_bitmask(
    int max_num_seqs) {
  // xgr.allocate_token_bitmask(max_num_seqs, vocab_size) (py:128-129) — same
  // [max_num_seqs, ceil(vocab/32)] int32 layout the native backend allocates.
  return inner_->allocate_token_bitmask(max_num_seqs);
}

void XgrammarStructuredOutputBackend::destroy() { inner_->destroy(); }

std::function<std::unique_ptr<StructuredOutputBackend>()>
MakeXgrammarBackendFactory(const tok::Tokenizer& tokenizer, int vocab_size,
                           bool disable_any_whitespace,
                           std::vector<int32_t> stop_token_ids) {
  return [&tokenizer, vocab_size, disable_any_whitespace, stop_token_ids]() {
    return std::make_unique<XgrammarStructuredOutputBackend>(
        tokenizer, vocab_size, disable_any_whitespace, stop_token_ids);
  };
}

std::string ResolveStructuredOutputBackend(const std::string& configured) {
  // config/structured_outputs.py:13 supported names. `auto` -> `xgrammar`
  // (sampling_params.py:1031); an explicit backend passes through.
  if (configured == "auto" || configured.empty() || configured == "xgrammar") {
    return "xgrammar";
  }
  if (configured == "guidance" || configured == "outlines" ||
      configured == "lm-format-enforcer") {
    // Separate INVENTORIED rows (TOOLS-GUIDANCE-OUTLINES); until implemented
    // they run on the native engine. Named, not silent.
    return "native";
  }
  throw std::runtime_error("Unsupported structured output backend: " +
                           configured);
}

std::function<std::unique_ptr<StructuredOutputBackend>()>
MakeStructuredOutputBackendFactory(const std::string& configured,
                                   const tok::Tokenizer& tokenizer,
                                   int vocab_size, bool disable_any_whitespace,
                                   std::vector<int32_t> stop_token_ids) {
  const std::string resolved = ResolveStructuredOutputBackend(configured);
  if (resolved == "xgrammar") {
    return MakeXgrammarBackendFactory(tokenizer, vocab_size,
                                      disable_any_whitespace,
                                      std::move(stop_token_ids));
  }
  return MakeNativeBackendFactory(tokenizer, vocab_size,
                                  std::move(stop_token_ids));
}

}  // namespace vllm::v1
