#!/usr/bin/env python3
"""End-to-end contract for the W6 installed server archive.

This test catches a package target that copies a build-tree executable, keeps
the historical ``server`` filename, or leaves the extracted executable coupled
to a build-tree ``libvllm``.  It configures the real CPU server, builds the real
archive target, extracts it into a second empty tree, and invokes that binary.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILD_JOBS = os.environ.get("VLLM_CPP_PACKAGE_TEST_JOBS", "2")
# The archive carries VLLM_CPP_BUILD_VERSION, which defaults to the CMake
# project version; release_pipeline.py holds that equal to `project_version`,
# so the expected name follows a version bump instead of pinning one release.
PROJECT_VERSION = json.loads(
    (ROOT / "release" / "release-version.json").read_text(encoding="utf-8")
)["project_version"]


def run(*command: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}"
        )
    return result


class ServerPackageTest(unittest.TestCase):
    def test_archive_is_deterministic_installed_and_runnable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vllm-server-package-") as temporary:
            scratch = Path(temporary)
            configured_build = os.environ.get("VLLM_CPP_PACKAGE_TEST_BUILD_DIR")
            if configured_build:
                build = Path(configured_build).resolve()
                self.assertTrue(
                    (build / "CMakeCache.txt").is_file(),
                    "VLLM_CPP_PACKAGE_TEST_BUILD_DIR must name a configured build",
                )
            else:
                build = scratch / "build"
                run(
                    "cmake",
                    "-S",
                    str(ROOT),
                    "-B",
                    str(build),
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DVLLM_CPP_BUILD_TESTS=OFF",
                    "-DVLLM_CPP_BUILD_EXAMPLES=ON",
                    "-DVLLM_CPP_SERVER=ON",
                    "-DVLLM_CPP_CUDA=OFF",
                    "-DVLLM_CPP_METAL=OFF",
                    "-DVLLM_CPP_VULKAN=OFF",
                    "-DVLLM_CPP_HIP=OFF",
                )
            run(
                "cmake",
                "--build",
                str(build),
                "--target",
                "vllm-server-archive",
                f"-j{BUILD_JOBS}",
            )

            archives = sorted((build / "release").glob("vllm.cpp-*.tar.gz"))
            self.assertEqual(len(archives), 1, "the package target must emit one archive")
            archive = archives[0]
            self.assertEqual(
                archive.name,
                f"vllm.cpp-{PROJECT_VERSION}-linux-{platform.machine()}-glibc-cpu.tar.gz",
            )
            first_digest = hashlib.sha256(archive.read_bytes()).hexdigest()

            # A second package invocation must reproduce the bytes, not merely a
            # staging tree with equivalent contents.
            archive.unlink()
            run(
                "cmake",
                "--build",
                str(build),
                "--target",
                "vllm-server-archive",
                f"-j{BUILD_JOBS}",
            )
            self.assertEqual(first_digest, hashlib.sha256(archive.read_bytes()).hexdigest())

            extracted = scratch / "extracted"
            extracted.mkdir()
            with tarfile.open(archive, "r:gz") as bundle:
                bundle.extractall(extracted, filter="data")

            executable = extracted / "bin" / (
                "vllm-server.exe" if platform.system() == "Windows" else "vllm-server"
            )
            self.assertTrue(executable.is_file(), "archive must use the canonical server name")
            self.assertTrue(os.access(executable, os.X_OK), "installed server must be executable")
            self.assertTrue(
                (build / "release" / "stage" / "bin" / executable.name).is_file(),
                "the package target must retain its installed staging tree",
            )

            clean_env = os.environ.copy()
            clean_env.pop("LD_LIBRARY_PATH", None)
            clean_env.pop("DYLD_LIBRARY_PATH", None)
            help_result = run(str(executable), "--help", env=clean_env)
            self.assertIn("--max-num-seqs N", help_result.stdout)
            self.assertIn("--max-num-batched-tokens N", help_result.stdout)
            version_result = run(str(executable), "--version", env=clean_env)
            self.assertRegex(version_result.stdout, r"^vllm\.cpp \d+\.\d+\.\d+(?:\+cuda)? c-abi=\d+\n$")

            if platform.system() == "Linux" and shutil.which("ldd"):
                dependencies = run("ldd", str(executable), env=clean_env).stdout
                self.assertNotIn("libvllm", dependencies)

            # W6 adds a server component without replacing the existing library
            # installation contract.
            install_prefix = scratch / "library-install"
            run(
                "cmake",
                "--build",
                str(build),
                "--target",
                "vllm_shared",
                f"-j{BUILD_JOBS}",
            )
            run("cmake", "--install", str(build), "--prefix", str(install_prefix))
            self.assertTrue((install_prefix / "include" / "vllm.h").is_file())
            self.assertTrue(any((install_prefix / "lib").glob("libvllm.a")))
            shared_libraries = [
                *install_prefix.glob("lib/libvllm.so*"),
                *install_prefix.glob("lib/libvllm*.dylib"),
                *install_prefix.glob("bin/vllm*.dll"),
            ]
            self.assertTrue(shared_libraries, "the existing shared-library install must remain")


if __name__ == "__main__":
    unittest.main()
