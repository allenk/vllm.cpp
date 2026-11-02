#!/usr/bin/env python3
# vllm.cpp ORIGINAL -- accept/reject goldens for the GLM-4.7 structural tag.
#
# The pinned vLLM (555967922) constrains a glm45 / glm47 tool call with
# xgrammar's builtin `glm_4_7` structural tag (tool_parsers/
# structural_tag_registry.py: XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS, called
# with reasoning=False from parser/abstract_parser.py:_apply_structural_tag)
# whenever tool_choice is `required`, a named function, or `auto` with a strict
# tool. This script asks that xgrammar whether each candidate output is a
# complete match, so test_glm47_structural_tag can require the native backend
# to give the same answer.
#
# Needs xgrammar only (CPU); the pin requires xgrammar >= 0.2.1 and the vLLM
# 0.28.0 environment it was generated in carries 0.2.3.
#
# Usage: python3 scripts/gen-glm47-structural-tag-goldens.py \
#          tests/fixtures/glm47_structural_tag_goldens.json
import json
import sys
from importlib.metadata import version

from xgrammar import Grammar, get_model_structural_tag
from xgrammar.testing import _is_grammar_accept_string

out_path = sys.argv[1]

WEATHER = {"type": "function", "function": {
    "name": "get_weather",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "days": {"type": "integer"},
            "units": {"type": "string", "enum": ["c", "f"]},
            "flags": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["city"],
    }}}
NOOP = {"type": "function", "function": {
    "name": "noop", "parameters": {"type": "object", "properties": {}}}}
ECHO = {"type": "function", "function": {
    "name": "echo",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                   "required": ["text"]}}}
ANYARGS = {"type": "function", "function": {
    "name": "anyargs", "parameters": {"type": "object"}}}
# No `parameters`, and `strict: false`: xgrammar's _get_function_parameters maps
# both to the `True` schema (any keys, any values).
NOPARAMS = {"type": "function", "function": {"name": "noparams"}}
LOOSE = {"type": "function", "function": {
    "name": "loose", "strict": False,
    "parameters": {"type": "object", "properties": {"q": {"type": "integer"}},
                   "required": ["q"]}}}
TOOLS = [WEATHER, NOOP, ECHO, ANYARGS, NOPARAMS, LOOSE]

W = "<tool_call>get_weather"
CITY = "<arg_key>city</arg_key><arg_value>Taipei</arg_value>"
E = "</tool_call>"

CALLS = {
    "city only": W + CITY + E,
    "city and days": W + CITY + "<arg_key>days</arg_key><arg_value>3</arg_value>" + E,
    "all four in order": W + CITY + "<arg_key>days</arg_key><arg_value>3</arg_value>"
        "<arg_key>units</arg_key><arg_value>c</arg_value>"
        "<arg_key>flags</arg_key><arg_value>[1, 2]</arg_value>" + E,
    "optional skipped in the middle": W + CITY + "<arg_key>flags</arg_key><arg_value>[]</arg_value>" + E,
    "out of order": W + "<arg_key>days</arg_key><arg_value>3</arg_value>" + CITY + E,
    "required missing": W + "<arg_key>days</arg_key><arg_value>3</arg_value>" + E,
    "enum value outside": W + CITY + "<arg_key>units</arg_key><arg_value>k</arg_value>" + E,
    "integer as text": W + CITY + "<arg_key>days</arg_key><arg_value>three</arg_value>" + E,
    "array item of wrong type": W + CITY + "<arg_key>flags</arg_key><arg_value>[1, \"a\"]</arg_value>" + E,
    "unknown key": W + CITY + "<arg_key>zone</arg_key><arg_value>x</arg_value>" + E,
    "whitespace between parts": W + "\n<arg_key>city</arg_key>\n<arg_value> Taipei </arg_value>\n" + E,
    "string with markup inside": W + "<arg_key>city</arg_key><arg_value>a<b>c</arg_valu</arg_value>" + E,
    "string holding the close tag": W + "<arg_key>city</arg_key><arg_value>a</arg_value>b</arg_value>" + E,
    "string ending in a partial close tag": W + "<arg_key>city</arg_key><arg_value>a</arg_v</arg_value>" + E,
    "non-ASCII string": W + "<arg_key>city</arg_key><arg_value>臺北市</arg_value>" + E,
    "empty string value": W + "<arg_key>city</arg_key><arg_value></arg_value>" + E,
    "noop": "<tool_call>noop" + E,
    "noop with an argument": "<tool_call>noop<arg_key>x</arg_key><arg_value>1</arg_value>" + E,
    "echo with JSON-looking text": "<tool_call>echo<arg_key>text</arg_key><arg_value>{\"k\": 1}</arg_value>" + E,
    "anyargs with free keys": "<tool_call>anyargs<arg_key>a</arg_key><arg_value>x</arg_value>"
        "<arg_key>b_2</arg_key><arg_value>[1]</arg_value>" + E,
    "anyargs with nothing": "<tool_call>anyargs" + E,
    "noparams with free keys": "<tool_call>noparams<arg_key>a</arg_key><arg_value>x y</arg_value>"
        "<arg_key>_b2</arg_key><arg_value>{\"k\": [1]}</arg_value>" + E,
    "noparams with a bad key name": "<tool_call>noparams<arg_key>1a</arg_key><arg_value>x</arg_value>" + E,
    "noparams with nothing": "<tool_call>noparams" + E,
    "loose ignores its schema": "<tool_call>loose<arg_key>z</arg_key><arg_value>text</arg_value>" + E,
    "unknown tool": "<tool_call>other" + CITY + E,
    "two calls": W + CITY + E + "<tool_call>noop" + E,
    "text before the call": "ok " + W + CITY + E,
    "text after the call": W + CITY + E + " done",
    "unterminated": W + CITY,
    "plain text": "It is sunny in Taipei.",
    "empty": "",
}


def choice_for(mode):
    if mode == "named":
        return {"type": "function", "function": {"name": "get_weather"}}
    return mode


out = {"generator": f"xgrammar {version('xgrammar')} get_model_structural_tag(model='glm_4_7', reasoning=False)",
       "modes": {}}
for mode in ("required", "named", "auto"):
    tools = TOOLS
    if mode == "auto":
        # vLLM only constrains `auto` when a tool is strict; mark one so.
        tools = [dict(WEATHER, function=dict(WEATHER["function"], strict=True))] + TOOLS[1:]
    choice = choice_for(mode)
    st = get_model_structural_tag(model="glm_4_7", tools=tools, tool_choice=choice,
                                  reasoning=False)
    grammar = Grammar.from_structural_tag(st)
    out["modes"][mode] = {
        "tools": tools, "tool_choice": choice,
        "cases": [{"name": name, "text": text,
                   "accept": bool(_is_grammar_accept_string(grammar, text))}
                  for name, text in CALLS.items()]}

with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
    f.write("\n")
for mode, m in out["modes"].items():
    print(mode, sum(c["accept"] for c in m["cases"]), "accepted of", len(m["cases"]))
