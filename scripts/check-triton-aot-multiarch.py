#!/usr/bin/env python3
"""Audit W2 Triton AOT tree identities and collision-free archive symbols."""

from __future__ import annotations

import argparse
import re
import struct
import subprocess
import sys
from pathlib import Path


TREE_FLAGS = {
    "sm_80": 0x00500550,
    "sm_86": 0x00560556,
    "sm_89": 0x00590559,
    "sm_90a": 0x005A0D5A,
    "sm_100a": 0x0600640A,
    # FORK: this fork vendors an sm_120a tree (Blackwell, the arch
    # build-windows-release.ps1 names as VLLM_CPP_CUDA_ARCHITECTURES=120a).
    # Upstream has six trees and no sm_120a, so the tree was added here without
    # this table, and the audit failed on the SET before it ever checked a flag.
    # Value read off the vendored cubins, not chosen: all 20 carry 0x06007802.
    "sm_120a": 0x06007802,
    "sm_121a": 0x0600790A,
}
CUBIN_ARRAY_RE = re.compile(
    r"unsigned\s+char\s+CUBIN_NAME\s*\[[^]]+\]\s*=\s*\{(.*?)\};",
    re.DOTALL,
)
BYTE_RE = re.compile(r"0x([0-9a-fA-F]{2})")
SYMBOL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")


def manifest_bases(path: Path) -> set[str]:
    bases: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("base "):
            bases.add(line.split(maxsplit=2)[1])
    return bases


def cubin_flags(path: Path) -> int | None:
    match = CUBIN_ARRAY_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        return None
    data = bytes(int(value, 16) for value in BYTE_RE.findall(match.group(1)))
    if len(data) < 52 or data[:4] != b"\x7fELF":
        return None
    return struct.unpack_from("<I", data, 48)[0]


def validate_trees(root: Path) -> tuple[list[str], set[str]]:
    errors: list[str] = []
    actual_trees = {path.name for path in root.iterdir() if path.is_dir()}
    expected_trees = set(TREE_FLAGS)
    if actual_trees != expected_trees:
        errors.append(
            "Triton AOT tree set "
            f"{sorted(actual_trees)} != expected {sorted(expected_trees)}"
        )

    canonical_bases: set[str] | None = None
    for tree, expected_flags in TREE_FLAGS.items():
        directory = root / tree
        manifest = directory / "MANIFEST"
        if not manifest.is_file():
            errors.append(f"{tree}: missing MANIFEST")
            continue
        lines = manifest.read_text(encoding="utf-8").splitlines()
        if f"arch {tree}" not in lines:
            errors.append(f"{tree}: MANIFEST arch mismatch")
        bases = manifest_bases(manifest)
        if not bases:
            errors.append(f"{tree}: MANIFEST has no base declarations")
        if canonical_bases is None:
            canonical_bases = bases
        elif bases != canonical_bases:
            errors.append(f"{tree}: base set differs from the other trees")

        cubins = sorted(directory.glob("*.*.c"))
        if not cubins:
            errors.append(f"{tree}: no embedded cubin launchers")
        for source in cubins:
            actual_flags = cubin_flags(source)
            if actual_flags != expected_flags:
                shown = "unreadable" if actual_flags is None else hex(actual_flags)
                errors.append(
                    f"{tree}/{source.name}: wrong cubin ELF flags {shown}; "
                    f"expected {hex(expected_flags)}"
                )
    return errors, canonical_bases or set()


def validate_symbols(nm_listing: str, bases: set[str]) -> list[str]:
    symbols = set(SYMBOL_RE.findall(nm_listing))
    expected = set()
    for tree in TREE_FLAGS:
        for base in bases:
            expected.add(f"vt_aot_{tree}_{base}_default")
            expected.add(f"vt_aot_{tree}_load_{base}")
    missing = sorted(expected.difference(symbols))
    unnamespaced = sorted(
        symbol
        for base in bases
        for symbol in (f"{base}_default", f"load_{base}")
        if symbol in symbols
    )
    errors: list[str] = []
    if missing:
        errors.append(f"AOT symbol namespace is missing: {','.join(missing)}")
    if unnamespaced:
        errors.append(f"AOT symbol namespace leaked: {','.join(unnamespaced)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendored-root", type=Path, required=True)
    symbols = parser.add_mutually_exclusive_group(required=True)
    symbols.add_argument("--library", type=Path)
    symbols.add_argument("--nm-list", type=Path)
    args = parser.parse_args()

    errors, bases = validate_trees(args.vendored_root)
    if args.library is not None:
        result = subprocess.run(
            ["nm", "--defined-only", "--extern-only", str(args.library)],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            errors.append(f"nm failed: {result.stderr.strip()}")
            listing = ""
        else:
            listing = result.stdout
    else:
        listing = args.nm_list.read_text(encoding="utf-8")
    errors.extend(validate_symbols(listing, bases))

    if errors:
        print("Triton AOT multi-arch audit FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(
        f"Triton AOT multi-arch audit: {len(TREE_FLAGS)} exact trees "
        "and namespaces OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
