// vllm.cpp ORIGINAL component (§9 deviation). See json_schema_to_gbnf.h for the
// supported subset, deviations and limits.
//
// The lowering builds a GBNF string in two parts:
//   1. a fixed PRIMITIVES block (ws / string / number / integer / boolean /
//      null / value / object / array / member) — the JSON grammar the Task-4
//      GbnfParser consumes;
//   2. a `root ::= <expr>` plus any schema-specific rules (closed objects,
//      typed arrays, enum / const alternations) generated recursively.
#include "vllm/v1/structured_output/json_schema_to_gbnf.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm::v1 {
namespace {

// The JSON primitive rules, in GBNF the Task-4 parser accepts. `ws` is an
// optional-whitespace rule (correctness-grade: accepts any JSON whitespace,
// including none, between structural tokens). `string` allows any non-control,
// non-quote, non-backslash byte (so arbitrary UTF-8 passes) plus the standard
// JSON escapes. `number` is the full JSON number grammar; `integer` its
// integer-only restriction. `value` / `object` / `array` are the generic
// (any-JSON) rules, referenced by json_object and by untyped sub-values.
const char* kPrimitivesBlock = R"GBNF(
ws ::= [ \t\n\r]*
hex ::= [0-9a-fA-F]
strchar ::= [^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)
string ::= "\"" strchar* "\""
integer ::= "-"? ("0" | [1-9] [0-9]*)
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
boolean ::= "true" | "false"
null ::= "null"
value ::= object | array | string | number | boolean | null
member ::= string ws ":" ws value
object ::= "{" ws "}" | "{" ws member (ws "," ws member)* ws "}"
array ::= "[" ws "]" | "[" ws value (ws "," ws value)* ws "]"
)GBNF";

// A JSON value (string / number / bool / null / object / array) -> a GBNF string
// literal matching the value's exact serialized bytes. dump() produces the
// canonical JSON serialization; we re-escape it for a GBNF double-quoted literal
// (escaping " and \ and control bytes so the literal matches byte-for-byte).
std::string JsonValueToGbnfLiteral(const nlohmann::json& value) {
  const std::string serialized = value.dump();
  std::string out = "\"";
  for (const char ch : serialized) {
    const auto b = static_cast<unsigned char>(ch);
    if (ch == '"' || ch == '\\') {
      out += '\\';
      out += ch;
    } else if (b < 0x20) {
      // A control byte: emit a \xHH GBNF escape (the parser decodes \x).
      static const char* kHex = "0123456789abcdef";
      out += "\\x";
      out += kHex[(b >> 4) & 0xF];
      out += kHex[b & 0xF];
    } else {
      out += ch;
    }
  }
  out += '"';
  return out;
}

std::string RawStringToGbnfLiteral(const std::string& raw);

// The recursive converter. Accumulates schema-specific rules in rules_ (name,
// body) and returns, for any schema node, a GBNF *element* string (a rule-ref
// name, a primitive name, a quoted literal, or a parenthesized group) that the
// caller can drop straight into a sequence.
class Converter {
 public:
  // `prefix` is prepended to every generated rule name, so multiple Converters
  // sharing ONE primitives block (the structural-tag composition) do not collide
  // (default "" == the standalone behavior).
  explicit Converter(std::string prefix = "") : prefix_(std::move(prefix)) {}

  std::string Convert(const nlohmann::json& schema) {
    const std::string root_expr = Visit(schema);
    std::string out = kPrimitivesBlock;
    out += "\nroot ::= ";
    out += root_expr;
    out += "\n";
    for (const auto& rule : rules_) {
      out += rule.first;
      out += " ::= ";
      out += rule.second;
      out += "\n";
    }
    return out;
  }

  // Lower `schema` to a GBNF element expression WITHOUT emitting the primitives
  // block or a `root` wrapper; the generated schema-specific rules are exposed
  // via rules(). Used by StructuralTagToGbnf, which emits ONE primitives block
  // and stitches several lowered bodies together (a per-tag `prefix` keeps their
  // generated rule names disjoint).
  std::string LowerElement(const nlohmann::json& schema) { return Visit(schema); }
  const std::vector<std::pair<std::string, std::string>>& rules() const {
    return rules_;
  }

  // The glm_xml argument surface for a tool's arguments schema (see
  // StructuralTagBody in json_schema_to_gbnf.h); the rules mirror the EBNF
  // xgrammar 0.2.3 builds for its glm_4_7 structural tag.
  std::string LowerGlmXmlArguments(const nlohmann::json& schema,
                                   const std::vector<std::string>& order) {
    if (schema.is_boolean() && !schema.get<bool>()) {
      throw std::runtime_error(
          "JsonSchemaToGbnf: a `false` arguments schema matches nothing");
    }
    if (!schema.is_object() || schema.empty()) return GlmAnyArguments();
    if (!schema.contains("properties")) return GlmWs();  // closed, no keys
    const nlohmann::json& props = schema["properties"];
    if (!props.is_object()) {
      throw std::runtime_error(
          "JsonSchemaToGbnf: `properties` must be an object");
    }
    std::vector<std::string> required;
    if (schema.contains("required") && schema["required"].is_array()) {
      for (const auto& r : schema["required"]) {
        if (r.is_string()) required.push_back(r.get<std::string>());
      }
    }
    for (const std::string& r : required) {
      if (!props.contains(r)) {
        throw std::runtime_error(
            "json schema -> gbnf: required key '" + r +
            "' is not declared in 'properties' (unsupported — would not be "
            "enforced)");
      }
    }
    std::vector<std::string> names;
    for (const std::string& n : order) {
      if (props.contains(n) &&
          std::find(names.begin(), names.end(), n) == names.end()) {
        names.push_back(n);
      }
    }
    for (auto it = props.begin(); it != props.end(); ++it) {
      if (std::find(names.begin(), names.end(), it.key()) == names.end()) {
        names.push_back(it.key());
      }
    }
    const std::string ws = GlmWs();
    std::string body;
    for (const std::string& n : names) {
      const std::string item =
          ws + " " + RawStringToGbnfLiteral("<arg_key>" + n + "</arg_key>") +
          " " + ws + " \"<arg_value>\" " + ws + " " + GlmXmlValue(props[n]) +
          " " + ws + " \"</arg_value>\"";
      const bool req =
          std::find(required.begin(), required.end(), n) != required.end();
      body += req ? item : "(" + item + ")?";
      body += " ";
    }
    body += ws;
    return NewRule(body);
  }

 private:
  // xgrammar's glm_xml whitespace: [ \n\t]* (the JSON `ws` also takes \r).
  std::string GlmWs() {
    if (glm_ws_.empty()) glm_ws_ = NewRule("[ \\n\\t]*");
    return glm_ws_;
  }

  // Raw text that never contains "</arg_value>" (xgrammar's xml_string, a
  // TagDispatch excluding it). The rules are the states of a matcher for that
  // word: state k has just read its first k bytes. `<` occurs in it only at
  // the start, so a mismatch always falls back to state 0, or to state 1 on a
  // `<`. Every state may end the value.
  std::string GlmXmlString() {
    if (!glm_xs_.empty()) return glm_xs_;
    static const std::string kClose = "</arg_value>";
    const std::size_t n = kClose.size();
    std::vector<std::string> state(n);
    for (std::size_t k = 0; k < n; ++k) {
      state[k] = prefix_ + "xs" + std::to_string(k);
    }
    for (std::size_t k = 0; k < n; ++k) {
      std::string b = "\"\"";
      if (k == 0) {
        b += " | [^<] " + state[0];
      } else {
        const char next = kClose[k];
        if (k + 1 < n) {
          b += " | " + RawStringToGbnfLiteral(std::string(1, next)) + " " +
               state[k + 1];
        }
        b += std::string(" | [^<") + next + "] " + state[0];
      }
      b += " | \"<\" " + state[1];
      rules_.emplace_back(state[k], std::move(b));
    }
    glm_xs_ = state[0];
    return glm_xs_;
  }

  // xgrammar's xml_any (an untyped value): raw text, or a JSON array / object.
  std::string GlmXmlAny() {
    return "(" + GlmXmlString() + " | array | object)";
  }

  // `true` arguments: any number of [a-zA-Z_][a-zA-Z0-9_]* keys, each with an
  // untyped value (xgrammar's xml_object).
  std::string GlmAnyArguments() {
    const std::string ws = GlmWs();
    return NewRule("(" + ws + " \"<arg_key>\" [a-zA-Z_] [a-zA-Z0-9_]* " +
                   "\"</arg_key>\" " + ws + " \"<arg_value>\" " + ws + " " +
                   GlmXmlAny() + " " + ws + " \"</arg_value>\")* " + ws);
  }

  // One property's VALUE: string-typed values are raw text, the rest JSON.
  std::string GlmXmlValue(const nlohmann::json& schema) {
    if (!schema.is_object()) {
      if (schema.is_boolean() && !schema.get<bool>()) {
        throw std::runtime_error(
            "JsonSchemaToGbnf: a `false` schema (matches nothing) is unsupported");
      }
      return GlmXmlAny();
    }
    for (const char* key : {"anyOf", "oneOf"}) {
      if (schema.contains(key) && schema[key].is_array() &&
          !schema[key].empty()) {
        std::string body;
        for (const auto& sub : schema[key]) {
          if (!body.empty()) body += " | ";
          body += GlmXmlValue(sub);
        }
        return "(" + body + ")";
      }
    }
    const auto literal = [](const nlohmann::json& v) {
      return v.is_string() ? RawStringToGbnfLiteral(v.get<std::string>())
                           : JsonValueToGbnfLiteral(v);
    };
    if (schema.contains("const")) return literal(schema["const"]);
    if (schema.contains("enum") && schema["enum"].is_array() &&
        !schema["enum"].empty()) {
      std::string body;
      for (const auto& v : schema["enum"]) {
        if (!body.empty()) body += " | ";
        body += literal(v);
      }
      return "(" + body + ")";
    }
    if (schema.contains("type")) {
      const nlohmann::json& t = schema["type"];
      if (t.is_string()) {
        return t.get<std::string>() == "string" ? GlmXmlString() : Visit(schema);
      }
      if (t.is_array() && !t.empty()) {
        std::string body;
        for (const auto& tv : t) {
          if (!tv.is_string()) {
            throw std::runtime_error(
                "JsonSchemaToGbnf: `type` array entries must be strings");
          }
          if (!body.empty()) body += " | ";
          body += tv.get<std::string>() == "string"
                      ? GlmXmlString()
                      : VisitType(tv.get<std::string>(), schema);
        }
        return "(" + body + ")";
      }
    }
    if (schema.contains("properties") || schema.contains("items")) {
      return Visit(schema);
    }
    return GlmXmlAny();
  }

  // Register a fresh named rule with the given body; return its name. Names are
  // "<prefix>r<N>" — disjoint from every primitive rule name (none of which start
  // "r" followed by a digit) and from "root".
  std::string NewRule(std::string body) {
    std::string name = prefix_ + "r" + std::to_string(counter_++);
    rules_.emplace_back(name, std::move(body));
    return name;
  }

  // The GBNF element for a single (non-union, non-enum, non-const) type name.
  std::string VisitType(const std::string& type, const nlohmann::json& schema) {
    if (type == "object") return VisitObject(schema);
    if (type == "array") return VisitArray(schema);
    if (type == "string") return "string";
    if (type == "integer") return "integer";
    if (type == "number") return "number";
    if (type == "boolean") return "boolean";
    if (type == "null") return "null";
    throw std::runtime_error(
        "JsonSchemaToGbnf: unsupported JSON-schema type '" + type + "'");
  }

  std::string Visit(const nlohmann::json& schema) {
    // A boolean schema: true == any value, false == nothing (unsupported).
    if (schema.is_boolean()) {
      if (schema.get<bool>()) return "value";
      throw std::runtime_error(
          "JsonSchemaToGbnf: a `false` schema (matches nothing) is unsupported");
    }
    if (!schema.is_object()) {
      // A non-object, non-boolean schema node (e.g. an empty/absent schema) ->
      // any JSON value.
      return "value";
    }
    // Unsupported combinators: fail loud rather than mis-constrain.
    for (const char* key : {"$ref", "allOf", "not"}) {
      if (schema.contains(key)) {
        throw std::runtime_error(
            std::string("JsonSchemaToGbnf: unsupported schema keyword '") + key +
            "'");
      }
    }
    // `anyOf` / `oneOf` -> a GBNF alternation of the sub-schemas (a natural GBNF
    // fit: `( A | B | ... )`). Each alternative is lowered recursively (it
    // registers its own rules) and combined with `|` inside a group. NOTE:
    // `oneOf` ("exactly one matches") is APPROXIMATED as a union — when the
    // sub-schemas overlap the grammar accepts a value matching more than one; for
    // the DISJOINT schemas this powers (tool_choice=required over tools with
    // distinct `name` consts) the union is exact. See json_schema_to_gbnf.h.
    for (const char* key : {"anyOf", "oneOf"}) {
      if (schema.contains(key)) {
        const nlohmann::json& alts = schema[key];
        if (!alts.is_array() || alts.empty()) {
          throw std::runtime_error(std::string("JsonSchemaToGbnf: `") + key +
                                   "` must be a non-empty array");
        }
        std::string body;
        bool first = true;
        for (const auto& sub : alts) {
          if (!first) body += " | ";
          first = false;
          body += Visit(sub);
        }
        return "(" + body + ")";
      }
    }
    if (schema.contains("const")) {
      return JsonValueToGbnfLiteral(schema["const"]);
    }
    if (schema.contains("enum")) {
      const nlohmann::json& e = schema["enum"];
      if (!e.is_array() || e.empty()) {
        throw std::runtime_error(
            "JsonSchemaToGbnf: `enum` must be a non-empty array");
      }
      std::string body;
      bool first = true;
      for (const auto& v : e) {
        if (!first) body += " | ";
        first = false;
        body += JsonValueToGbnfLiteral(v);
      }
      return "(" + body + ")";
    }
    if (schema.contains("type")) {
      const nlohmann::json& t = schema["type"];
      if (t.is_string()) {
        return VisitType(t.get<std::string>(), schema);
      }
      if (t.is_array()) {
        // A union of types -> alternation.
        if (t.empty()) {
          throw std::runtime_error("JsonSchemaToGbnf: empty `type` array");
        }
        std::string body;
        bool first = true;
        for (const auto& tv : t) {
          if (!tv.is_string()) {
            throw std::runtime_error(
                "JsonSchemaToGbnf: `type` array entries must be strings");
          }
          if (!first) body += " | ";
          first = false;
          body += VisitType(tv.get<std::string>(), schema);
        }
        return "(" + body + ")";
      }
      throw std::runtime_error(
          "JsonSchemaToGbnf: `type` must be a string or array of strings");
    }
    // No type / enum / const: infer from structure, else any value.
    if (schema.contains("properties")) return VisitObject(schema);
    if (schema.contains("items")) return VisitArray(schema);
    return "value";
  }

  std::string VisitObject(const nlohmann::json& schema) {
    if (!schema.contains("properties")) {
      // No declared properties -> any JSON object.
      return "object";
    }
    const nlohmann::json& props = schema["properties"];
    if (!props.is_object()) {
      throw std::runtime_error(
          "JsonSchemaToGbnf: `properties` must be an object");
    }
    if (props.empty()) return "object";

    // Which keys are required (a JSON array of names).
    std::vector<std::string> required;
    if (schema.contains("required")) {
      const nlohmann::json& req = schema["required"];
      if (!req.is_array()) {
        throw std::runtime_error(
            "JsonSchemaToGbnf: `required` must be an array");
      }
      for (const auto& r : req) {
        if (r.is_string()) required.push_back(r.get<std::string>());
      }
    }
    auto is_required = [&](const std::string& key) {
      for (const std::string& r : required) {
        if (r == key) return true;
      }
      return false;
    };

    // A `required` key that is NOT declared in `properties` cannot be turned into
    // a grammar member, so its requiredness would be silently unenforced —
    // over-permitting schema-INVALID JSON (missing that key). Per this converter's
    // "throw rather than mis-constrain" invariant (json_schema_to_gbnf.h), reject
    // it loudly instead. (In practice function-calling / response_format schemas
    // always list every required key in properties.)
    for (const std::string& r : required) {
      if (!props.contains(r)) {
        throw std::runtime_error(
            "json schema -> gbnf: required key '" + r +
            "' is not declared in 'properties' (unsupported — would not be "
            "enforced)");
      }
    }

    // Per property (in schema/key order): the "member" element
    //   "\"<key>\"" ws ":" ws <value-expr>
    // and whether it is required.
    struct Prop {
      std::string member_expr;
      bool required;
    };
    std::vector<Prop> ordered;
    for (auto it = props.begin(); it != props.end(); ++it) {
      const std::string key_literal =
          JsonValueToGbnfLiteral(nlohmann::json(it.key()));
      const std::string value_expr = Visit(it.value());
      Prop p;
      p.member_expr = key_literal + " ws \":\" ws " + value_expr;
      p.required = is_required(it.key());
      ordered.push_back(std::move(p));
    }

    // Build, from the tail forward, two rule families over the ordered members:
    //   first[i] : members i.. rendered with NO leading separator (the object's
    //              opening `"{" ws` precedes the first emitted member),
    //   rest[i]  : members i.. rendered when a member was ALREADY emitted (each
    //              emitted member here is prefixed by `ws "," ws`).
    // A required member is always emitted; an optional member may be skipped.
    // This yields any ordered subset of the optional keys (required always
    // present) with correct comma placement. `""` (an empty literal) is the
    // sentinel for "nothing more" (matches the empty string).
    const std::size_t n = ordered.size();
    std::vector<std::string> first(n + 1);
    std::vector<std::string> rest(n + 1);
    first[n] = "\"\"";
    rest[n] = "\"\"";
    for (std::size_t k = n; k-- > 0;) {
      const std::string& mem = ordered[k].member_expr;
      const std::string next_first = first[k + 1];
      const std::string next_rest = rest[k + 1];
      std::string first_body;
      std::string rest_body;
      if (ordered[k].required) {
        first_body = mem + " " + next_rest;
        rest_body = "ws \",\" ws " + mem + " " + next_rest;
      } else {
        first_body = mem + " " + next_rest + " | " + next_first;
        rest_body = "ws \",\" ws " + mem + " " + next_rest + " | " + next_rest;
      }
      first[k] = NewRule(std::move(first_body));
      rest[k] = NewRule(std::move(rest_body));
    }

    const std::string body =
        "\"{\" ws " + first[0] + " ws \"}\"";
    return NewRule(body);
  }

  std::string VisitArray(const nlohmann::json& schema) {
    if (!schema.contains("items")) {
      // No item schema -> any JSON array.
      return "array";
    }
    const nlohmann::json& items = schema["items"];
    if (items.is_array()) {
      // Tuple validation (positional items) is not constrained at T0 -> any
      // array. (Documented limit; still schema-valid output would need per-index
      // rules; deferred.)
      return "array";
    }
    const std::string item_expr = Visit(items);
    const std::string body = "\"[\" ws \"]\" | \"[\" ws " + item_expr +
                             " (ws \",\" ws " + item_expr + ")* ws \"]\"";
    return NewRule(body);
  }

  std::vector<std::pair<std::string, std::string>> rules_;
  int counter_ = 0;
  std::string prefix_;
  std::string glm_ws_;  // the glm_xml rules, built once per converter
  std::string glm_xs_;
};

// Escape a RAW byte string into a GBNF double-quoted string literal matching
// those exact bytes (escapes `"` and `\`, emits control bytes as `\xHH`, passes
// through the rest). Unlike JsonValueToGbnfLiteral this takes the raw bytes
// directly (no JSON serialization) — used for the structural-tag begin/end
// delimiters (e.g. `<tool_call>\n`).
std::string RawStringToGbnfLiteral(const std::string& raw) {
  std::string out = "\"";
  for (const char ch : raw) {
    const auto b = static_cast<unsigned char>(ch);
    if (ch == '"' || ch == '\\') {
      out += '\\';
      out += ch;
    } else if (b < 0x20) {
      static const char* kHex = "0123456789abcdef";
      out += "\\x";
      out += kHex[(b >> 4) & 0xF];
      out += kHex[b & 0xF];
    } else {
      out += ch;
    }
  }
  out += '"';
  return out;
}

}  // namespace

std::string JsonSchemaToGbnf(const nlohmann::json& schema) {
  Converter converter;
  return converter.Convert(schema);
}

std::string JsonObjectGbnf() {
  // The permissive any-JSON grammar: the primitives block + `root ::= value`.
  std::string out = kPrimitivesBlock;
  out += "\nroot ::= value\n";
  return out;
}

std::string WrapSchemaAsToolCallGbnf(
    const nlohmann::json& tool_call_object_schema) {
  // Lower the tool-call object schema. Convert() emits exactly one `root ::=`
  // line (the primitives block defines no rule named "root"); it is the JSON
  // value the constrained decode would emit BARE. Rename it to an inner rule so
  // we can wrap it with the `<tool_call>` literals in a fresh `root`.
  std::string inner = JsonSchemaToGbnf(tool_call_object_schema);
  const std::string marker = "\nroot ::= ";
  const std::size_t pos = inner.find(marker);
  if (pos == std::string::npos) {
    throw std::runtime_error(
        "WrapSchemaAsToolCallGbnf: lowered grammar has no 'root' rule to wrap");
  }
  // The name "_toolcallbody" starts with '_' (a valid GBNF rule-name start) and
  // does not collide with any primitive rule or with "root".
  inner.replace(pos, marker.size(), "\n_toolcallbody ::= ");

  // Prepend the wrapping root. The GBNF string literals carry the exact Hermes
  // wrapper bytes: `<tool_call>\n` (begin) and `\n</tool_call>` (end). The `\n`
  // are GBNF escapes (parsed to newline); `<`, `>`, `/` are literal bytes the
  // string-literal path emits verbatim. The Hermes parser's
  // `<tool_call>(.*?)</tool_call>` (DOTALL) captures `\n{...}\n`, then `.strip()`
  // trims the leading/trailing newline before json.loads — so the inner JSON is
  // extracted verbatim.
  std::string out =
      "root ::= \"<tool_call>\\n\" _toolcallbody \"\\n</tool_call>\"\n";
  out += inner;
  return out;
}

std::string StructuralTagToGbnf(const std::vector<StructuralTagBody>& tags,
                                bool stop_after_first) {
  if (tags.empty()) {
    throw std::runtime_error(
        "StructuralTagToGbnf: at least one tag is required");
  }
  // ONE shared primitives block; each tag lowers its content schema under a
  // per-tag rule-name prefix so the generated rules never collide.
  std::string out = kPrimitivesBlock;
  std::string root_alt;   // "tag0 | tag1 | ..."
  std::string tag_rules;  // "tagI ::= ...\n" + each tag's schema rules
  for (std::size_t i = 0; i < tags.size(); ++i) {
    const StructuralTagBody& t = tags[i];

    // The tag BODY element: any-JSON `value` for content_schema == true, else the
    // schema lowered to an element (its rules carried under the "t<i>_" prefix).
    std::string body_expr;
    std::vector<std::pair<std::string, std::string>> body_rules;
    if (t.content_style == "glm_xml") {
      Converter conv("t" + std::to_string(i) + "_");
      body_expr = conv.LowerGlmXmlArguments(t.content_schema, t.property_order);
      body_rules = conv.rules();
    } else if (t.content_style != "json") {
      throw std::runtime_error("StructuralTagToGbnf: unknown content_style '" +
                               t.content_style + "'");
    } else if (t.content_schema.is_boolean()) {
      if (!t.content_schema.get<bool>()) {
        throw std::runtime_error(
            "StructuralTagToGbnf: content_schema `false` matches nothing");
      }
      body_expr = "value";  // any well-formed JSON value
    } else {
      Converter conv("t" + std::to_string(i) + "_");
      body_expr = conv.LowerElement(t.content_schema);
      body_rules = conv.rules();
    }

    const std::string begin_lit = RawStringToGbnfLiteral(t.begin);
    const std::string end_lit = RawStringToGbnfLiteral(t.end);
    const std::string tag_name = "tag" + std::to_string(i);
    tag_rules += tag_name + " ::= " + begin_lit + " " + body_expr + " " +
                 end_lit + "\n";
    for (const auto& r : body_rules) {
      tag_rules += r.first + " ::= " + r.second + "\n";
    }
    if (i != 0) root_alt += " | ";
    root_alt += tag_name;
  }

  // stop_after_first => exactly one tag; else (at_least_one) => one-or-more.
  out += "\nroot ::= (";
  out += root_alt;
  out += stop_after_first ? ")\n" : ")+\n";
  out += tag_rules;
  return out;
}

}  // namespace vllm::v1
