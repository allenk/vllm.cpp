// Ported from: vllm/v1/structured_output/backend_xgrammar.py:36 (XgrammarBackend)
// @ pin 555967922 (vLLM 0.26.0.dev0). See .agents/specs/xgrammar-backend.md.
//
// The xgrammar structured-output backend — vLLM's DEFAULT (`auto`) backend
// (sampling_params.py:1031 resolves `auto` -> `xgrammar`). It is a SECOND
// registerable StructuredOutputBackend behind the same seam as the native one
// (backend_types.h), so the manager (manager.cpp) drives it identically.
//
// DECISION (porting-inventory §9): we do NOT vendor the xgrammar C++ library.
// xgrammar IS "grammar -> pushdown automaton -> per-step token bitmask"; our
// native engine (backend_native.cpp) already implements that algorithm portably
// (push-down FSM over a byte-level grammar + a token-byte trie for sub-O(vocab)
// fill). So the xgrammar backend REUSES the native matcher and supplies only the
// xgrammar-FAITHFUL front-end that the two backends differ on:
//   - JSON schema -> EBNF via XgrammarJsonSchemaToEbnf (xgrammar_json_schema.h),
//     which preserves property DECLARATION order and emits the `any_whitespace`
//     rule + `basic_*` set VERBATIM as xgrammar does — closing the
//     whitespace/key-order/exotic-schema parity gap the feature-gap analysis
//     flags (vllm-feature-gap-analysis.md, TOOLS-XGRAMMAR).
//   - GRAMMAR (EBNF) / REGEX / CHOICE / STRUCTURAL_TAG delegate to the native
//     compile paths (xgrammar accepts the same GBNF/EBNF dialect); JSON is the
//     W1 xgrammar-distinct brick.
//
// `disable_any_whitespace` mirrors XgrammarBackend.__post_init__ reading
// vllm_config.structured_outputs_config.disable_any_whitespace
// (backend_xgrammar.py:38-40); compile_json_schema is called with
// any_whitespace = not disable_any_whitespace (backend_xgrammar.py:83). vLLM's
// default is any-whitespace ON.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "vllm/v1/structured_output/backend_native.h"
#include "vllm/v1/structured_output/backend_types.h"

namespace vllm::tok {
class Tokenizer;
}

namespace vllm::v1 {

// XgrammarBackend (backend_xgrammar.py:36). Engine-level backend; compiles a
// grammar spec into a per-request StructuredOutputGrammar (the native FSM).
class XgrammarStructuredOutputBackend : public StructuredOutputBackend {
 public:
  // `disable_any_whitespace` == structured_outputs_config.disable_any_whitespace
  // (default false => any-whitespace ON, the vLLM default). `stop_token_ids` as
  // for the native backend (empty => the tokenizer's EOS).
  XgrammarStructuredOutputBackend(const tok::Tokenizer& tokenizer,
                                  int vocab_size,
                                  bool disable_any_whitespace = false,
                                  std::vector<int32_t> stop_token_ids = {});
  ~XgrammarStructuredOutputBackend() override;

  // compile_grammar (backend_xgrammar.py:78-126). JSON/JSON_OBJECT go through the
  // xgrammar-faithful schema->EBNF converter; the rest delegate to the native
  // compile paths (same EBNF/GBNF dialect).
  // The request's stop tokens pass through to the native grammar.
  std::unique_ptr<StructuredOutputGrammar> compile_grammar(
      StructuredOutputOptions request_type, const std::string& grammar_spec,
      const std::vector<int32_t>& stop_token_ids = {}) override;

  TokenBitmask allocate_token_bitmask(int max_num_seqs) override;
  void destroy() override;

 private:
  std::unique_ptr<NativeStructuredOutputBackend> inner_;  // the shared matcher
  bool disable_any_whitespace_;
};

// Factory helper mirroring MakeNativeBackendFactory — wires the
// StructuredOutputManager's BackendFactory to the xgrammar backend. Same
// tokenizer-lifetime contract (referenced only during construction).
std::function<std::unique_ptr<StructuredOutputBackend>()>
MakeXgrammarBackendFactory(const tok::Tokenizer& tokenizer, int vocab_size,
                           bool disable_any_whitespace = false,
                           std::vector<int32_t> stop_token_ids = {});

// Backend-name resolution mirroring vLLM's selection logic
// (config/structured_outputs.py:13 supported list; sampling_params.py:932-949
// engine-level pin; __init__.py:133-165 dispatch). Given the engine-configured
// backend name (`auto` | `xgrammar` | `guidance` | `outlines` |
// `lm-format-enforcer`), returns the CONCRETE backend name this engine will run.
//
// vllm.cpp scope: we ship `xgrammar` (this row) and the `native` engine (which is
// also what `guidance`/`outlines`/`lm-format-enforcer` currently fall back to —
// those backends are separate INVENTORIED rows, TOOLS-GUIDANCE-OUTLINES). `auto`
// resolves to `xgrammar` exactly as vLLM does (sampling_params.py:1031). An
// unknown name throws (mirrors __init__.py:164 ValueError).
std::string ResolveStructuredOutputBackend(const std::string& configured);

// Construct the manager BackendFactory for a resolved backend name. `xgrammar`
// (and `auto`) -> the xgrammar backend; anything else vllm.cpp implements today
// -> the native backend. Mirrors __init__.py:133-165's if/elif dispatch over the
// resolved `_backend`.
std::function<std::unique_ptr<StructuredOutputBackend>()>
MakeStructuredOutputBackendFactory(const std::string& configured,
                                   const tok::Tokenizer& tokenizer,
                                   int vocab_size,
                                   bool disable_any_whitespace = false,
                                   std::vector<int32_t> stop_token_ids = {});

}  // namespace vllm::v1
