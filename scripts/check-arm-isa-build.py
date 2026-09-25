#!/usr/bin/env python3
"""Audit W4 aarch64 CPU tier isolation in CMake compile commands."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


BASELINE_SOURCES = (
    "src/vt/cpu/cpu_isa_arm.cpp",
    "src/vt/cpu/cpu_matmul_elem.cpp",
    "src/vt/cpu/cpu_quant_dot.cpp",
    "src/vt/cpu/cpu_quant_repack.cpp",
)
TIER_MARCH = {
    "src/vt/cpu/cpu_quant_dot_sdot.cpp": "-march=armv8.2-a+dotprod+fp16",
    "src/vt/cpu/cpu_quant_dot_arm.cpp": "-march=armv8.2-a+i8mm+dotprod",
    # The repack tier's file is cpu_quant_repack_SIMD.cpp. This table said
    # _arm.cpp, a name the tree does not contain, so the audit demanded a
    # compile command that could never exist and the scheduled ci lane has
    # failed on it 35 times running since 2026-09-16. CMakeLists.txt:1536-1538
    # applies these exact flags to cpu_quant_dot_arm.cpp and
    # cpu_quant_repack_simd.cpp together.
    "src/vt/cpu/cpu_quant_repack_simd.cpp": "-march=armv8.2-a+i8mm+dotprod",
}


def relative_source(value: str) -> str | None:
    normalized = value.replace("\\", "/")
    marker = "/src/"
    if marker not in normalized:
        return None
    return "src/" + normalized.split(marker, 1)[1]


def command_arguments(entry: dict[str, object]) -> list[str]:
    arguments = entry.get("arguments")
    if isinstance(arguments, list):
        return [str(value) for value in arguments]
    return shlex.split(str(entry.get("command", "")))


def march_flags(arguments: list[str]) -> list[str]:
    return [argument for argument in arguments if argument.startswith("-march=")]


def validate_compile_commands(path: Path) -> list[str]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    commands: dict[str, list[str]] = {}
    errors: list[str] = []
    for entry in entries:
        arguments = command_arguments(entry)
        source = relative_source(str(entry.get("file", "")))
        label = source or str(entry.get("file", "<unknown>"))
        if "-march=native" in arguments:
            errors.append(f"{label}: forbidden -march=native")
        if source is not None:
            commands[source] = arguments

    for source in BASELINE_SOURCES:
        if source not in commands:
            errors.append(f"missing Arm compile command for {source}")
            continue
        leaked = march_flags(commands[source])
        if leaked:
            errors.append(f"{source}: baseline source leaks architecture flags {leaked}")

    for source, required in TIER_MARCH.items():
        if source not in commands:
            errors.append(f"missing Arm compile command for {source}")
            continue
        actual = march_flags(commands[source])
        if actual != [required]:
            errors.append(f"{source}: march flags {actual} != required {[required]}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-commands", type=Path, required=True)
    args = parser.parse_args()

    errors = validate_compile_commands(args.compile_commands)
    if errors:
        print("Arm ISA build audit FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Arm ISA build audit: portable baseline and exact DotProd/i8mm tiers OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
