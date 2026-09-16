#!/usr/bin/env python3
"""W15 native Windows release metadata contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/release_metadata.py"
SHA = "0123456789abcdef0123456789abcdef01234567"


def load():
    spec = importlib.util.spec_from_file_location("release_windows_metadata", TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TOOL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def passed(command: str) -> dict[str, str]:
    return {"command": command, "reason": "", "result": "exit 0", "state": "passed", "url": "https://example.invalid/run"}


class WindowsMetadataContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = load()

    def fixture(self, root: Path, backend: str) -> argparse.Namespace:
        build, stage, output = root / "build", root / "stage", root / "metadata"
        build.mkdir(parents=True)
        (stage / "bin").mkdir(parents=True)
        (stage / "bin/vllm-server.exe").write_bytes(b"MZ fixture")
        flags = {
            "MLX_ROOT": ("PATH", ""), "VLLM_CPP_BUILD_EXAMPLES": ("BOOL", "ON"),
            "VLLM_CPP_BUILD_TESTS": ("BOOL", "ON"), "VLLM_CPP_CUDA": ("BOOL", "OFF"),
            "VLLM_CPP_CUDA_ARCHITECTURES": ("STRING", ""), "VLLM_CPP_HIP": ("BOOL", "OFF"),
            "VLLM_CPP_HIP_ARCHITECTURES": ("STRING", ""), "VLLM_CPP_LITERAL_STATIC": ("BOOL", "OFF"),
            "VLLM_CPP_METAL": ("BOOL", "OFF"), "VLLM_CPP_MLX": ("BOOL", "OFF"),
            "VLLM_CPP_SERVER": ("BOOL", "ON"), "VLLM_CPP_TRITON": ("BOOL", "OFF"),
            "VLLM_CPP_VULKAN": ("BOOL", "ON" if backend == "vulkan" else "OFF"),
        }
        (build / "CMakeCache.txt").write_text("".join(f"{key}:{kind}={value}\n" for key, (kind, value) in flags.items()), encoding="utf-8")
        policy = self.tool.release_manifest.CPU_TIER_POLICY["x86_64"]
        report = {"schema": "vllm.cpp.cpu-tier-report.v1", "selected_tier": "avx2-f16c", "commands": ["portable", "avx2"], "tiers": {name: passed(name) for name in policy["tiers"]}}
        tier_report = root / "tiers.json"
        tier_report.write_text(json.dumps(report), encoding="utf-8")
        pe_report = root / "pe.json"
        pe_report.write_text(json.dumps({"schema": "vllm.cpp.pe-audit.v1", "machine": "8664", "imports": ["KERNEL32.dll", "WS2_32.dll"], "debug_paths": []}), encoding="utf-8")
        return argparse.Namespace(
            abi_version="14.38", artifact_id=f"windows-x86_64-msvc-{backend}",
            backend=backend, build_dir=build, c_abi_version=17, channel="preview",
            compiler="MSVC 19.38.33135", evidence_url="https://example.invalid/run",
            output_dir=output, pe_report=pe_report, repo_root=ROOT, source_clean=True,
            source_commit=SHA, stage_dir=stage, tier_report=tier_report,
            toolchain="Visual Studio 2022 v143 /MT", toolset_version="14.38.33130",
            ucrt_version="10.0.20348.0", version="0.0.3-vk.1",
        )

    def test_cpu_metadata_is_native_preview_and_sbom_names_exe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary), "cpu")
            manifest = self.tool.prepare_windows_metadata(args)
            self.assertEqual(manifest["host"]["toolset_version"], "14.38.33130")
            self.assertEqual(manifest["host"]["ucrt_version"], "10.0.20348.0")
            self.assertEqual(manifest["evidence"]["runtime"]["state"], "passed")
            sbom = json.loads((args.output_dir / "sbom.spdx.json").read_text())
            self.assertEqual(sbom["files"][0]["fileName"], "./bin/vllm-server.exe")

    def test_vulkan_runtime_stays_absent_without_real_icd_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary), "vulkan")
            manifest = self.tool.prepare_windows_metadata(args)
            self.assertEqual(manifest["evidence"]["runtime"]["state"], "absent")
            self.assertEqual(
                {row["name"] for row in manifest["dependencies"] if row["linkage"] == "external"},
                {"vulkan-loader", "vulkan-icd", "vulkan-driver"},
            )
            self.assertNotIn(
                "vulkan-runtime-passed",
                (ROOT / "scripts/release_metadata.py").read_text(encoding="utf-8"),
            )

    def test_contract_test_precedes_required_release_environment(self) -> None:
        script = (ROOT / "scripts/build-windows-release.ps1").read_text(encoding="utf-8")
        contract = script.index("if ($ContractTest)")
        contract_exit = script.index("exit 0", contract)
        environment = script.index(
            'foreach ($name in @("SOURCE_SHA", "VERSION", "EVIDENCE_URL", "SOURCE_DATE_EPOCH"))'
        )
        self.assertLess(contract, contract_exit)
        self.assertLess(contract_exit, environment)

    def test_pe_report_rejects_msys_debug_crt_and_developer_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary), "cpu")
            for mutation in (
                {"imports": ["MSYS-2.0.dll"]},
                {"imports": ["VCRUNTIME140D.dll"]},
                {"debug_paths": [r"C:\\agent\\build\\server.pdb"]},
            ):
                report = json.loads(args.pe_report.read_text())
                report.update(mutation)
                args.pe_report.write_text(json.dumps(report), encoding="utf-8")
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    self.tool.prepare_windows_metadata(args)

            report = json.loads(args.pe_report.read_text())
            report.update({"imports": ["api-ms-win-crt-runtime-l1-1-0.dll"],
                           "debug_paths": []})
            args.pe_report.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(ValueError):
                self.tool.prepare_windows_metadata(args)

    def test_vulkan_loader_uses_wide_win32_api_and_releases_invalid_dll(self) -> None:
        source = (ROOT / "src/vt/vulkan/vulkan_loader.cpp").read_text(encoding="utf-8")
        self.assertIn("LoadLibraryW(name)", source)
        self.assertIn("GetProcAddress(reinterpret_cast<HMODULE>(handle), name)", source)
        self.assertIn("FreeLibrary(reinterpret_cast<HMODULE>(handle))", source)
        self.assertIn('ops.lookup(ops.context, handle, "vkGetInstanceProcAddr")', source)
        self.assertIn("Win32LibraryShutdown", source)

    # RENAMED, because the property it asserted stopped being true. The cache
    # is no longer EXCLUDED on Windows: cuda_matmul_nvfp4_cutlass.cu references
    # its symbols unconditionally, so a Windows CUDA build cannot link without
    # it, and CMakeLists.txt now carries a _WIN32 branch with the same contract
    # (write to a temporary, then rename) instead of `if(NOT WIN32)`.
    #
    # What still holds, and is what this case now pins: the source is added
    # exactly once, inside a platform-conditional block, and the TEST stays
    # Linux-only in tests/CMakeLists.txt.
    def test_persistent_cache_is_platform_conditional_and_added_once(self) -> None:
        root_cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        tests_cmake = (ROOT / "tests/CMakeLists.txt").read_text(encoding="utf-8")
        source = "target_sources(vllm PRIVATE src/vt/cuda/nvfp4_persistent_cache.cpp)"
        # Added exactly once and NOT inside a platform conditional. The source
        # used to sit in an `if(NOT WIN32)` block; it cannot, because
        # cuda_matmul_nvfp4_cutlass.cu references its symbols unconditionally.
        self.assertEqual(root_cmake.count(source), 1)
        # Anchored to a LINE START. The bare substring also matched the
        # comment in CMakeLists.txt that explains why if(TRUE) was removed,
        # so the assertion failed on its own documentation.
        self.assertNotIn("\nif(TRUE)", root_cmake)
        block_start = tests_cmake.index("if(NOT WIN32)\n", tests_cmake.index("test_ops_nvfp4_fp4"))
        target = tests_cmake.index("vllm_cpp_add_test(test_nvfp4_persistent_cache", block_start)
        block_end = tests_cmake.index("endif()", target)
        self.assertLess(block_start, target)
        self.assertLess(target, block_end)

    def test_windows_script_smokes_the_final_zip_server_lifecycle(self) -> None:
        script = (ROOT / "scripts/build-windows-release.ps1").read_text(encoding="utf-8")
        package = script.index('"scripts/package-server.py"')
        expand = script.index("Expand-Archive", package)
        archive_server = script.index('bin/vllm-server.exe', expand)
        lifecycle = script.index("$smokeHarness, $archiveServer", archive_server)
        validate = script.index('"scripts/validate-release-archive.py"', lifecycle)
        self.assertLess(package, expand)
        self.assertLess(expand, archive_server)
        self.assertLess(archive_server, lifecycle)
        self.assertLess(lifecycle, validate)


if __name__ == "__main__":
    unittest.main()
