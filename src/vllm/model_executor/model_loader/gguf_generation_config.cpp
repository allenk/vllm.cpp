// See include/vllm/model_executor/model_loader/gguf_generation_config.h.
#include "vllm/model_executor/model_loader/gguf_generation_config.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <set>
#include <variant>
#include <vector>

namespace vllm {
namespace {

// An integer kv of any GGUF integer width; nullopt for anything else.
std::optional<int64_t> AsInt(const GgufValue& v) {
  switch (v.TypeId()) {
    case kGgufU8:
      return std::get<uint8_t>(v.v);
    case kGgufI8:
      return std::get<int8_t>(v.v);
    case kGgufU16:
      return std::get<uint16_t>(v.v);
    case kGgufI16:
      return std::get<int16_t>(v.v);
    case kGgufU32:
      return std::get<uint32_t>(v.v);
    case kGgufI32:
      return std::get<int32_t>(v.v);
    case kGgufI64:
      return std::get<int64_t>(v.v);
    case kGgufU64: {
      const uint64_t u = std::get<uint64_t>(v.v);
      if (u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        return std::nullopt;
      return static_cast<int64_t>(u);
    }
    default:
      return std::nullopt;
  }
}

std::optional<int32_t> AsInt32(const GgufValue& v) {
  const std::optional<int64_t> i = AsInt(v);
  if (!i || *i < std::numeric_limits<int32_t>::min() ||
      *i > std::numeric_limits<int32_t>::max())
    return std::nullopt;
  return static_cast<int32_t>(*i);
}

// The converter stores these keys as f32, so the 0.95 in generation_config.json
// arrives as 0.949999988. Upstream parses that JSON 0.95 as a double; taking
// the shortest decimal that round-trips the f32 hands the sampler the same
// double a safetensors load of the same checkpoint does.
double ShortestDecimal(float f) {
  char buf[32];
  for (int precision = 1; precision <= 9; ++precision) {
    std::snprintf(buf, sizeof(buf), "%.*g", precision, static_cast<double>(f));
    if (std::strtof(buf, nullptr) == f) return std::strtod(buf, nullptr);
  }
  return static_cast<double>(f);
}

std::optional<double> AsReal(const GgufValue& v) {
  if (v.TypeId() == kGgufF32) {
    const float f = std::get<float>(v.v);
    if (!std::isfinite(f)) return std::nullopt;
    return ShortestDecimal(f);
  }
  if (v.TypeId() == kGgufF64) {
    const double d = std::get<double>(v.v);
    if (!std::isfinite(d)) return std::nullopt;
    return d;
  }
  // generation_config.json may spell `temperature: 1`; accept integers too.
  if (const std::optional<int64_t> i = AsInt(v)) return static_cast<double>(*i);
  return std::nullopt;
}

// Same polarity as ReadGenerationConfigEosIds: an element that is not a
// non-negative int32 is skipped, the rest are kept, sorted and unique.
std::vector<int32_t> EogTokenIds(const GgufFile& gguf) {
  const GgufValue* v = gguf.FindKv("tokenizer.ggml.eog_token_ids");
  if (v == nullptr || v->TypeId() != kGgufArray) return {};
  std::set<int32_t> ids;
  for (const GgufValue& e : std::get<GgufArray>(v->v).elems) {
    const std::optional<int32_t> id = AsInt32(e);
    if (id && *id >= 0) ids.insert(*id);
  }
  return std::vector<int32_t>(ids.begin(), ids.end());
}

void FillReal(const GgufFile& gguf, const char* key,
              std::optional<double>& field) {
  if (field.has_value()) return;
  if (const GgufValue* v = gguf.FindKv(key)) field = AsReal(*v);
}

}  // namespace

void ApplyGgufGenerationConfig(const GgufFile& gguf, HfConfig& config) {
  if (config.generation_config_eos_ids.empty())
    config.generation_config_eos_ids = EogTokenIds(gguf);

  // The keys with the same meaning as generation_config.json's. Not read:
  // `penalty_repeat`, because llama.cpp applies it over the last
  // `penalty_last_n` tokens while upstream's repetition_penalty covers the
  // whole prompt and output, and the mirostat / xtc / sequence keys, which
  // have no counterpart in get_diff_sampling_param.
  GenerationConfigSampling& s = config.generation_config_sampling;
  FillReal(gguf, "general.sampling.temp", s.temperature);
  FillReal(gguf, "general.sampling.top_p", s.top_p);
  FillReal(gguf, "general.sampling.min_p", s.min_p);
  if (!s.top_k.has_value()) {
    if (const GgufValue* v = gguf.FindKv("general.sampling.top_k"))
      s.top_k = AsInt32(*v);
  }
}

}  // namespace vllm
