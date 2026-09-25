#!/usr/bin/env python3
"""Generate W9 CPU release manifest, VERSION, notices, licenses, and SPDX."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import release_manifest  # noqa: E402


ARTIFACT_HOST = {
    "linux-x86_64-glibc-cpu": ("linux", "x86_64", "glibc", "static-core"),
    "linux-aarch64-glibc-cpu": ("linux", "aarch64", "glibc", "static-core"),
    "linux-x86_64-musl-cpu-static": (
        "linux",
        "x86_64",
        "musl",
        "literal-static",
    ),
}
REQUIRED_FLAGS = (
    "MLX_ROOT",
    "VLLM_CPP_BUILD_EXAMPLES",
    "VLLM_CPP_BUILD_TESTS",
    "VLLM_CPP_CUDA",
    "VLLM_CPP_CUDA_ARCHITECTURES",
    "VLLM_CPP_HIP",
    "VLLM_CPP_HIP_ARCHITECTURES",
    "VLLM_CPP_LITERAL_STATIC",
    "VLLM_CPP_METAL",
    "VLLM_CPP_MLX",
    "VLLM_CPP_SERVER",
    "VLLM_CPP_TRITON",
    "VLLM_CPP_VULKAN",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence(state: str, command: str, result: str, url: str, reason: str = "") -> dict[str, str]:
    return {
        "command": command,
        "reason": reason,
        "result": result,
        "state": state,
        "url": url,
    }


def passed(command: str, url: str) -> dict[str, str]:
    return evidence("passed", command, "exit 0", url)


def absent(reason: str) -> dict[str, str]:
    return evidence("absent", "", "", "", reason)


def parse_cache(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("//", "#")) or "=" not in line or ":" not in line.split("=", 1)[0]:
            continue
        key_type, value = line.split("=", 1)
        key, _ = key_type.split(":", 1)
        values[key] = value
    return values


def cmake_bool(value: str, name: str) -> bool:
    if value in {"ON", "TRUE", "1", "YES"}:
        return True
    if value in {"OFF", "FALSE", "0", "NO", ""}:
        return False
    raise ValueError(f"release CMake flag {name} must resolve explicitly ON or OFF, got {value!r}")


def backend_flags(cache: dict[str, str]) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FLAGS if name not in cache]
    if missing:
        raise ValueError(f"CMake cache is missing release flags: {missing}")
    flags: dict[str, Any] = {}
    for name in REQUIRED_FLAGS:
        value = cache[name]
        if name == "MLX_ROOT":
            flags[name] = value
        elif name.endswith("ARCHITECTURES"):
            flags[name] = [item for item in value.split(";") if item]
        else:
            flags[name] = cmake_bool(value, name)
    return flags


def parse_needed(server: Path) -> list[str]:
    result = subprocess.run(
        ["readelf", "-dW", str(server)], text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise ValueError(f"readelf dependency inspection failed: {result.stderr.strip()}")
    return re.findall(r"\(NEEDED\).*?\[([^]]+)\]", result.stdout)


def dependencies(server: Path, abi: str, abi_version: str) -> list[dict[str, Any]]:
    if abi == "musl":
        return [
            {
                "bundled": True,
                "kind": "library",
                "linkage": "static",
                "name": "musl-libc",
                "role": "build-time",
                "version": abi_version,
            }
        ]
    needed = parse_needed(server)
    if not needed:
        raise ValueError("glibc release server has no recorded dynamic dependencies")
    return [
        {
            "bundled": False,
            "kind": "library",
            "linkage": "dynamic",
            "name": name,
            "role": "runtime",
            "version": "system",
        }
        for name in needed
    ]


def load_tier_report(path: Path, arch: str, stable: bool) -> tuple[list[dict[str, Any]], str, list[str]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("schema") != "vllm.cpp.cpu-tier-report.v1":
        raise ValueError("CPU tier report has the wrong schema")
    policy = release_manifest.CPU_TIER_POLICY[arch]
    rows = report.get("tiers")
    if not isinstance(rows, dict) or list(rows) != list(policy["tiers"]):
        raise ValueError("CPU tier report must contain every compiled tier in policy order")
    selected = report.get("selected_tier")
    if selected not in rows:
        raise ValueError("CPU tier report selected_tier is not compiled")
    commands = report.get("commands")
    if not isinstance(commands, list) or not commands or any(not isinstance(item, str) or not item for item in commands):
        raise ValueError("CPU tier report commands must be a non-empty string array")
    compiled: list[dict[str, Any]] = []
    for name in policy["tiers"]:
        execution = rows[name]
        if not isinstance(execution, dict):
            raise ValueError(f"CPU tier report {name} evidence must be an object")
        if stable and execution.get("state") != "passed":
            raise ValueError(f"stable CPU bundle requires passed execution evidence for {name}")
        state_policy = policy["os_state"][name]
        os_state = state_policy["linux"] if isinstance(state_policy, dict) else state_policy
        compiled.append(
            {
                "execution_evidence": execution,
                "kernel_families": sorted(policy["kernel_families"][name]),
                "name": name,
                "required_cpu_bits": sorted(policy["bits"][name]),
                "required_os_state": sorted(os_state),
            }
        )
    return compiled, selected, commands


def spdx_document(
    artifact_id: str,
    version: str,
    source_commit: str,
    server: Path,
    dependency_rows: list[dict[str, Any]],
    bundled_files: list[tuple[str, Path, str]] | None = None,
    server_relative: str = "bin/vllm-server",
) -> dict[str, Any]:
    binary_digest = sha256(server)
    packages = [
        {
            "SPDXID": "SPDXRef-Package-vllm-cpp",
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": True,
            "licenseConcluded": "Apache-2.0",
            "licenseDeclared": "Apache-2.0",
            "name": "vllm.cpp",
            "versionInfo": version,
        }
    ]
    for index, dependency in enumerate(dependency_rows):
        packages.append(
            {
                "SPDXID": f"SPDXRef-Dependency-{index}",
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "name": dependency["name"],
                "versionInfo": dependency["version"],
            }
        )
    files = [
        {
            "SPDXID": "SPDXRef-File-vllm-server",
            "checksums": [{"algorithm": "SHA256", "checksumValue": binary_digest}],
            "copyrightText": "NOASSERTION",
            "fileName": f"./{server_relative}",
            "licenseConcluded": "Apache-2.0",
        }
    ]
    for index, (relative, path, license_id) in enumerate(bundled_files or (), 1):
        files.append(
            {
                "SPDXID": f"SPDXRef-Bundled-File-{index}",
                "checksums": [{"algorithm": "SHA256", "checksumValue": sha256(path)}],
                "copyrightText": "NOASSERTION",
                "fileName": f"./{relative}",
                "licenseConcluded": license_id,
            }
        )
    return {
        "SPDXID": "SPDXRef-DOCUMENT",
        "creationInfo": {
            "created": "1970-01-01T00:00:00Z",
            "creators": ["Organization: vllm.cpp"],
        },
        "dataLicense": "CC0-1.0",
        "documentNamespace": f"https://github.com/mudler/vllm.cpp/spdx/{source_commit}/{artifact_id}",
        "files": files,
        "name": artifact_id,
        "packages": packages,
        "spdxVersion": "SPDX-2.3",
    }


def prepare_cpu_metadata(args: argparse.Namespace) -> dict[str, Any]:
    if args.artifact_id not in ARTIFACT_HOST:
        raise ValueError(f"unsupported W9 CPU artifact {args.artifact_id!r}")
    os_name, arch, abi, static_boundary = ARTIFACT_HOST[args.artifact_id]
    stable = args.channel == "stable"
    server = args.stage_dir / "bin/vllm-server"
    if not server.is_file():
        raise ValueError("staged bin/vllm-server is missing")
    flags = backend_flags(parse_cache(args.build_dir / "CMakeCache.txt"))
    compiled_tiers, selected_tier, test_commands = load_tier_report(
        args.tier_report, arch, stable
    )
    dependency_rows = dependencies(server, abi, args.abi_version)
    gate_command = " && ".join(test_commands)
    facts: dict[str, Any] = {
        "artifact": {
            "c_abi_version": args.c_abi_version,
            "channel": args.channel,
            "id": args.artifact_id,
            "kind": "primary",
            "static_boundary": static_boundary,
            "version": args.version,
        },
        "backend": {
            "flags": flags,
            "gpu_driver_boundary": "not-applicable",
            "name": "cpu",
        },
        "build": {
            "compiler": args.compiler,
            "resolved_cmake_options": flags,
            "source_clean": args.source_clean,
            "source_commit": args.source_commit,
            "test_commands": [*test_commands, "python3 scripts/validate-release-archive.py"],
            "toolchain": args.toolchain,
        },
        "cpu": {
            "baseline": release_manifest.CPU_TIER_POLICY[arch]["baseline"],
            "compiled_tiers": compiled_tiers,
            "selected_tier": selected_tier,
        },
        "dependencies": dependency_rows,
        "evidence": {
            "archive_smoke": passed("extracted vllm-server --help && --version", args.evidence_url),
            "build": passed("cmake --build <build> --target server", args.evidence_url),
            "correctness": passed(gate_command, args.evidence_url),
            "dependency_audit": passed("readelf -dW && ldd/lddtree", args.evidence_url),
            "performance": absent("release packaging does not imply a performance claim"),
            "runtime": passed(gate_command, args.evidence_url),
        },
        "host": {
            "abi": abi,
            "abi_version": args.abi_version,
            "arch": arch,
            "os": os_name,
        },
        "supply_chain": {
            "archive_checksum": passed("sha256sum <final-archive>", args.evidence_url),
            "licenses": passed("validate THIRD_PARTY_NOTICES and share/licenses", args.evidence_url),
            "provenance": passed("validate detached in-toto SLSA subject digest", args.evidence_url),
            "sbom": passed("validate SPDX-2.3 server checksum and dependencies", args.evidence_url),
        },
    }
    schema = release_manifest.load_schema(args.repo_root / "release/manifest-v1.schema.json")
    manifest = release_manifest.generate_manifest(facts, args.repo_root, schema)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "release-manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
    version_values = {
        "version": args.version,
        "commit": args.source_commit,
        "artifact_id": args.artifact_id,
        "backend": "cpu",
        "host_os": os_name,
        "host_arch": arch,
        "host_abi": abi,
        "source_clean": "true" if args.source_clean else "false",
        "c_abi_version": str(args.c_abi_version),
    }
    (output / "VERSION").write_text(
        "".join(f"{key}={value}\n" for key, value in version_values.items()),
        encoding="utf-8",
    )
    (output / "sbom.spdx.json").write_text(
        canonical_json(spdx_document(args.artifact_id, args.version, args.source_commit, server, dependency_rows)),
        encoding="utf-8",
    )
    notices = ["vllm.cpp release dependency notices", ""]
    notices.extend(f"- {row['name']} {row['version']} ({row['linkage']})" for row in dependency_rows)
    (output / "THIRD_PARTY_NOTICES").write_text("\n".join(notices) + "\n", encoding="utf-8")
    license_dir = output / "share/licenses/vllm.cpp"
    license_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.repo_root / "LICENSE", license_dir / "LICENSE")
    return manifest


def windows_dependencies(report_path: Path, backend: str) -> list[dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("schema") != "vllm.cpp.pe-audit.v1":
        raise ValueError("PE audit report has the wrong schema")
    if str(report.get("machine", "")).upper() not in {"8664", "AMD64", "X64"}:
        raise ValueError("PE audit must prove an AMD64 executable")
    imports = report.get("imports")
    if not isinstance(imports, list) or not imports or any(not isinstance(item, str) for item in imports):
        raise ValueError("PE audit imports must be a non-empty string array")
    forbidden = [
        name for name in imports
        if re.match(r"(?i)^(?:msys-|mingw|libgcc|libstdc\+\+|vcruntime|msvcp|msvcr|ucrtbase|concrt|api-ms-win-crt-)", name)
    ]
    if forbidden:
        raise ValueError(f"PE audit violates native static-CRT policy: {sorted(forbidden)}")
    debug_paths = report.get("debug_paths")
    if not isinstance(debug_paths, list) or any(not isinstance(item, str) for item in debug_paths):
        raise ValueError("PE audit debug_paths must be a string array")
    if any(re.match(r"^[A-Za-z]:[\\/]", item) for item in debug_paths):
        raise ValueError("PE audit embeds a developer-drive debug path")
    rows = [
        {
            "bundled": False,
            "kind": "library",
            "linkage": "dynamic",
            "name": name,
            "role": "runtime",
            "version": "windows-2022",
        }
        for name in sorted(set(imports), key=str.upper)
    ]
    # FORK: the cuda row exists because this fork added a Windows CUDA lane that
    # upstream does not have. Its runtime is external and never bundled, exactly
    # like Vulkan's -- validate-release-archive.py REFUSES a CUDA payload in a
    # Windows archive, so declaring these as host-supplied is not a choice, it is
    # the only description of the artifact that is true.
    externals = {
        "vulkan": (
            ("vulkan-loader", "library"),
            ("vulkan-icd", "library"),
            ("vulkan-driver", "driver"),
        ),
        "cuda": (
            ("cuda-driver", "driver"),
            ("cudart", "library"),
            ("cublas", "library"),
            ("cublasLt", "library"),
        ),
    }
    if backend in externals:
        rows.extend(
            {
                "bundled": False,
                "kind": kind,
                "linkage": "external",
                "name": name,
                "role": "external-runtime",
                "version": "host",
            }
            for name, kind in externals[backend]
        )
    return rows


def prepare_windows_metadata(args: argparse.Namespace) -> dict[str, Any]:
    expected_id = f"windows-x86_64-msvc-{args.backend}"
    if args.artifact_id != expected_id or args.backend not in {"cpu", "vulkan", "cuda"}:
        raise ValueError(f"unsupported Windows artifact {args.artifact_id!r}")
    server = args.stage_dir / "bin/vllm-server.exe"
    if not server.is_file():
        raise ValueError("staged bin/vllm-server.exe is missing")
    flags = backend_flags(parse_cache(args.build_dir / "CMakeCache.txt"))
    if flags["VLLM_CPP_VULKAN"] is not (args.backend == "vulkan"):
        raise ValueError("resolved Vulkan flag disagrees with Windows artifact")
    # The same check for CUDA, and it is not symmetry for its own sake: without
    # it a cuda-labelled artifact built with VLLM_CPP_CUDA=OFF would describe
    # itself as a CUDA build and say so in a signed manifest.
    if flags["VLLM_CPP_CUDA"] is not (args.backend == "cuda"):
        raise ValueError("resolved CUDA flag disagrees with Windows artifact")
    compiled_tiers, selected_tier, test_commands = load_tier_report(
        args.tier_report, "x86_64", False
    )
    dependency_rows = windows_dependencies(args.pe_report, args.backend)
    # A runner with no GPU cannot produce runtime evidence, and the manifest must
    # say so rather than inherit the CPU lane's passing command line. The Windows
    # CUDA lane proves BUILD and PACKAGING only -- release.yml says that in its
    # own comment, and this is where that sentence becomes a machine-readable
    # fact instead of a promise in YAML.
    no_runtime = {
        "vulkan": "no real extracted-archive Vulkan ICD probe was executed",
        "cuda": "the release runner has no NVIDIA GPU; this lane proves build and "
                "packaging and never execution",
    }
    runtime = (absent(no_runtime[args.backend])
               if args.backend in no_runtime
               else passed(" && ".join(test_commands), args.evidence_url))
    facts: dict[str, Any] = {
        "artifact": {
            "c_abi_version": args.c_abi_version,
            "channel": "preview",
            "id": args.artifact_id,
            "kind": "primary",
            "static_boundary": "static-core",
            "version": args.version,
        },
        "backend": {
            "flags": flags,
            "gpu_driver_boundary": "not-applicable" if args.backend == "cpu" else "external-host-never-bundled",
            "name": args.backend,
        },
        "build": {
            "compiler": args.compiler,
            "resolved_cmake_options": flags,
            "source_clean": args.source_clean,
            "source_commit": args.source_commit,
            "test_commands": [*test_commands, "python scripts/validate-release-archive.py --archive-format zip"],
            "toolchain": args.toolchain,
        },
        "dependencies": dependency_rows,
        "evidence": {
            "archive_smoke": passed("extracted vllm-server.exe --help && --version", args.evidence_url),
            "build": passed("cmake --build <build> --config Release --target server", args.evidence_url),
            "correctness": passed(" && ".join(test_commands), args.evidence_url) if args.backend == "cpu" else absent("no real Vulkan ICD correctness probe was executed"),
            "dependency_audit": passed("dumpbin /headers /dependents /rawdata", args.evidence_url),
            "performance": absent("release packaging does not imply a performance claim"),
            "runtime": runtime,
        },
        "host": {
            "abi": "msvc", "abi_version": args.abi_version, "arch": "x86_64",
            "os": "windows", "toolset_version": args.toolset_version,
            "ucrt_version": args.ucrt_version,
        },
        "supply_chain": {
            "archive_checksum": passed("SHA256 final ZIP", args.evidence_url),
            "licenses": passed("validate notices and licenses", args.evidence_url),
            "provenance": passed("validate detached in-toto SLSA subject digest", args.evidence_url),
            "sbom": passed("validate SPDX-2.3 server checksum and dependencies", args.evidence_url),
        },
    }
    if args.backend == "cpu":
        facts["cpu"] = {
            "baseline": release_manifest.CPU_TIER_POLICY["x86_64"]["baseline"],
            "compiled_tiers": compiled_tiers,
            "selected_tier": selected_tier,
        }
    if args.backend == "cuda":
        # A CUDA artifact must declare the SMs it was compiled for; without them
        # the generator refuses it outright ("CUDA generation input requires
        # cuda.compiled_sms"). This is the SAME function the Linux CUDA lanes
        # publish from, so the two cannot end up describing the same ten
        # architectures differently.
        import release_accelerator_metadata
        facts["cuda"] = release_accelerator_metadata.cuda_evidence(args.evidence_url)
    schema = release_manifest.load_schema(args.repo_root / "release/manifest-v1.schema.json")
    manifest = release_manifest.generate_manifest(facts, args.repo_root, schema)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "release-manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
    values = {
        "version": args.version, "commit": args.source_commit,
        "artifact_id": args.artifact_id, "backend": args.backend,
        "host_os": "windows", "host_arch": "x86_64", "host_abi": "msvc",
        "source_clean": "true" if args.source_clean else "false",
        "c_abi_version": str(args.c_abi_version),
    }
    (output / "VERSION").write_text("".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8")
    (output / "sbom.spdx.json").write_text(
        canonical_json(spdx_document(args.artifact_id, args.version, args.source_commit, server, dependency_rows, server_relative="bin/vllm-server.exe")),
        encoding="utf-8",
    )
    (output / "THIRD_PARTY_NOTICES").write_text(
        "vllm.cpp release dependency notices\n\n" + "".join(
            f"- {row['name']} {row['version']} ({row['linkage']})\n" for row in dependency_rows
        ), encoding="utf-8"
    )
    license_dir = output / "share/licenses/vllm.cpp"
    license_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.repo_root / "LICENSE", license_dir / "LICENSE")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tier-report", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--channel", choices=("stable", "preview", "experimental-preview"), required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--c-abi-version", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-clean", action="store_true")
    parser.add_argument("--abi-version", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--toolchain", required=True)
    parser.add_argument("--evidence-url", required=True)
    parser.add_argument("--backend", choices=("cpu", "vulkan", "cuda"), default="cpu")
    parser.add_argument("--pe-report", type=Path)
    parser.add_argument("--toolset-version")
    parser.add_argument("--ucrt-version")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.artifact_id.startswith("windows-"):
            if args.pe_report is None or not args.toolset_version or not args.ucrt_version:
                raise ValueError("Windows metadata requires PE report, toolset version, and UCRT version")
            prepare_windows_metadata(args)
        else:
            prepare_cpu_metadata(args)
    except (OSError, json.JSONDecodeError, KeyError, ValueError, release_manifest.ManifestError) as exc:
        print(f"release metadata error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
