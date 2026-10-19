#!/usr/bin/env python3
"""W13 generated release index contract."""

from __future__ import annotations

import importlib.util
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts/release_index.py"
FIXTURE = ROOT / "tests/scripts/fixtures/release_manifest/v1/cpu-manifest.json"


def load():
    spec = importlib.util.spec_from_file_location("release_index", TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TOOL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseIndexContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = load()

    def assets(self, scratch: Path, windows: bool = False) -> tuple[Path, dict]:
        assets = scratch / "assets"
        assets.mkdir()
        manifest = json.loads(FIXTURE.read_text(encoding="utf-8"))
        archive_format = "zip" if windows else "tar.gz"
        if windows:
            manifest["artifact"].update({"id": "windows-x86_64-msvc-cpu", "channel": "preview"})
            manifest["host"].update({"os": "windows", "arch": "x86_64", "abi": "msvc"})
        artifact_id = manifest["artifact"]["id"]
        archive = assets / (
            f"vllm.cpp-{manifest['artifact']['version']}-{artifact_id}.{archive_format}"
        )
        manifest_path = scratch / "release-manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        if windows:
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.write(manifest_path, arcname="release-manifest.json")
        else:
            with tarfile.open(archive, "w:gz") as bundle:
                bundle.add(manifest_path, arcname="release-manifest.json")
        digest = self.tool.sha256(archive)
        (assets / f"{archive.name}.sha256").write_text(f"{digest}  {archive.name}\n")
        (assets / f"{archive.name}.provenance.json").write_text("{}\n")
        handoff = {
            "artifacts": [{
                "archive_format": archive_format,
                "channel": manifest["artifact"]["channel"],
                "id": artifact_id,
                "required": True,
            }],
            "files": [
                {
                    "artifact_id": artifact_id,
                    "name": path.name,
                    "sha256": self.tool.sha256(path),
                    "size": path.stat().st_size,
                }
                for path in sorted(assets.iterdir())
            ],
            "release_tag": f"v{manifest['artifact']['version']}",
            "prerelease": "-" in manifest["artifact"]["version"],
            "project_version": manifest["artifact"]["version"].split("-", 1)[0],
            "retention": {
                "ci_artifacts_days": 7,
                "github_release": "maintainer-deletion-only",
            },
            "source_sha": manifest["build"]["source_commit"],
            "verified": True,
            "version": manifest["artifact"]["version"],
        }
        return assets, handoff

    def test_index_is_derived_from_archive_manifests_and_byte_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets, handoff = self.assets(root)
            json_out = root / "release-index.json"
            markdown_out = root / "RELEASE_INDEX.md"
            self.tool.generate_index(assets, handoff, json_out, markdown_out)
            index = json.loads(json_out.read_text())
            self.assertEqual(index["schema"], "vllm.cpp.release-index.v1")
            self.assertTrue(index["prerelease"])
            self.assertEqual(index["project_version"], "0.1.0")
            self.assertEqual(index["version"], "0.1.0-test")
            self.assertEqual(index["artifacts"][0]["id"], "linux-x86_64-glibc-cpu")
            self.assertEqual(index["artifacts"][0]["cpu_tiers"], [
                "portable-sse2", "sse2-f16c", "avx2-f16c", "avx512f"
            ])
            self.assertIn("linux-x86_64-glibc-cpu", markdown_out.read_text())

    def test_markdown_links_are_download_urls_when_a_repository_is_given(self) -> None:
        # The Markdown doubles as the GitHub release body, where ./<name>
        # resolves against .../releases/tag/<tag> and 404s.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets, handoff = self.assets(root)
            markdown_out = root / "RELEASE_INDEX.md"
            index = self.tool.generate_index(
                assets, handoff, root / "i.json", markdown_out, repository="allenk/vllm.cpp"
            )
            text = markdown_out.read_text()
            base = f"https://github.com/allenk/vllm.cpp/releases/download/{index['release_tag']}/"
            self.assertIn(f"]({base}{index['artifacts'][0]['archive']})", text)
            self.assertNotIn("](./", text)
            with self.assertRaises(ValueError):
                self.tool.generate_index(
                    assets, handoff, root / "j.json", root / "j.md", repository="not a slug"
                )

    def test_index_reads_windows_zip_from_explicit_handoff_format(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets, handoff = self.assets(root, windows=True)
            index = self.tool.generate_index(
                assets, handoff, root / "index.json", root / "index.md"
            )
            self.assertEqual(index["artifacts"][0]["archive"], "vllm.cpp-0.1.0-test-windows-x86_64-msvc-cpu.zip")

    def test_manifest_identity_or_handoff_digest_drift_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets, handoff = self.assets(root)
            handoff["files"][0]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                self.tool.generate_index(assets, handoff, root / "i.json", root / "i.md")

    def test_manifest_channel_must_match_authoritative_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets, handoff = self.assets(root)
            handoff["artifacts"][0]["channel"] = "stable"
            with self.assertRaises(ValueError):
                self.tool.generate_index(assets, handoff, root / "i.json", root / "i.md")


if __name__ == "__main__":
    unittest.main()
