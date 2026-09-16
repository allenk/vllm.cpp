#!/usr/bin/env python3
"""Offline contract tests for the authenticated post-publication audit."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "scripts/release_postpublish_audit.py"
MATRIX = ROOT / "release/release-matrix.json"
VERSION = ROOT / "release/release-version.json"
INDEX = ROOT / "scripts/release_index.py"
SHA = "0123456789abcdef0123456789abcdef01234567"
RUN_ID = "31415926535"
REPO = "mudler/vllm.cpp"


def load():
    spec = importlib.util.spec_from_file_location("release_postpublish_audit", AUDIT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {AUDIT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_index():
    spec = importlib.util.spec_from_file_location("release_index", INDEX)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {INDEX}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PostPublishAuditContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = load()
        cls.indexer = load_index()
        cls.matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
        cls.version = json.loads(VERSION.read_text(encoding="utf-8"))

    def fixture(self):
        remote_bytes: dict[str, bytes] = {}
        attestations = {}
        for item in self.matrix["artifacts"]:
            archive = self.audit.canonical_archive_name(
                self.version["version"], item["id"], item["archive_format"]
            )
            artifact_id = item["id"]
            parts = artifact_id.split("-")
            os_name = "macos" if artifact_id.startswith("macos-") else parts[0]
            arch = "aarch64" if "arm64" in artifact_id or "aarch64" in artifact_id else "x86_64"
            abi = "msvc" if os_name == "windows" else (
                "darwin" if os_name == "macos" else ("musl" if "musl" in artifact_id else "glibc")
            )
            backend = next(
                (name for name in ("cuda", "vulkan", "metal", "mlx") if name in artifact_id),
                "cpu",
            )
            manifest = {
                "artifact": {
                    "channel": item["channel"], "id": artifact_id,
                    "static_boundary": "literal-static" if "static" in artifact_id else "static-core",
                    "version": self.version["version"],
                },
                "backend": {
                    "gpu_driver_boundary": "external-driver" if backend != "cpu" else "not-applicable",
                    "name": backend,
                },
                "build": {"source_commit": SHA},
                "cpu": {"compiled_tiers": [{"name": "portable-sse2"}] if arch == "x86_64" else [{"name": "portable-neon"}]},
                "cuda": {"compiled_sms": ["80", "90a"] if backend == "cuda" else []},
                "dependencies": ([{"name": "vulkan-1.dll", "role": "external-runtime"}] if backend == "vulkan" else []),
                "host": {"abi": abi, "abi_version": "fixture-1", "arch": arch, "os": os_name},
            }
            manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
            stream = io.BytesIO()
            if item["archive_format"] == "zip":
                with zipfile.ZipFile(stream, "w") as bundle:
                    bundle.writestr("release-manifest.json", manifest_bytes)
            else:
                with tarfile.open(fileobj=stream, mode="w:gz") as bundle:
                    info = tarfile.TarInfo("release-manifest.json")
                    info.size = len(manifest_bytes)
                    bundle.addfile(info, io.BytesIO(manifest_bytes))
            body = stream.getvalue()
            digest = hashlib.sha256(body).hexdigest()
            checksum = f"{digest}  {archive}\n".encode()
            provenance = json.dumps(
                {"subject": [{"name": archive, "digest": {"sha256": digest}}]}
            ).encode()
            for name, data in (
                (archive, body),
                (archive + ".sha256", checksum),
                (archive + ".provenance.json", provenance),
            ):
                remote_bytes[name] = data
            attestations[archive] = [{
                "digest": digest,
                "repository": REPO,
                "source_sha": SHA,
                "run_id": RUN_ID,
                "verified": True,
            }]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = root / "assets"
            assets.mkdir()
            files = []
            for item in self.matrix["artifacts"]:
                archive = self.audit.canonical_archive_name(
                    self.version["version"], item["id"], item["archive_format"]
                )
                for name in (archive, archive + ".sha256", archive + ".provenance.json"):
                    data = remote_bytes[name]
                    (assets / name).write_bytes(data)
                    files.append({
                        "artifact_id": item["id"], "name": name,
                        "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
                    })
            handoff = {
                "artifacts": self.matrix["artifacts"], "files": files,
                "prerelease": True, "project_version": "0.0.3",
                "release_tag": "v0.0.3-vk.1",
                "retention": self.matrix["retention"], "source_sha": SHA,
                "verified": True, "version": "0.0.3-vk.1",
            }
            index_path = root / "release-index.json"
            markdown_path = root / "RELEASE_INDEX.md"
            self.indexer.generate_index(assets, handoff, index_path, markdown_path)
            remote_bytes["release-index.json"] = index_path.read_bytes()
            remote_bytes["RELEASE_INDEX.md"] = markdown_path.read_bytes()
        assets = [
            {
                "id": index + 100,
                "name": name,
                "size": len(data),
                "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
            }
            for index, (name, data) in enumerate(sorted(remote_bytes.items()))
        ]
        snapshot = {
            "tag": {"object": {"sha": SHA, "type": "commit"}},
            "release": {
                "draft": False,
                "prerelease": True,
                "tag_name": "v0.0.3-vk.1",
                "assets": assets,
            },
            "run": {
                "head_sha": SHA,
                "id": int(RUN_ID),
                "path": ".github/workflows/release.yml",
                "conclusion": "success",
            },
            "jobs": [
                {"name": name, "conclusion": "success"}
                for name in self.audit.REQUIRED_RELEASE_JOBS
            ],
        }
        return snapshot, remote_bytes, attestations

    @staticmethod
    def replace_remote(snapshot, remote_bytes, name, data) -> None:
        remote_bytes[name] = data
        row = next(item for item in snapshot["release"]["assets"] if item["name"] == name)
        row["size"] = len(data)
        row["digest"] = "sha256:" + hashlib.sha256(data).hexdigest()

    def test_complete_authenticated_remote_fixture_passes(self) -> None:
        snapshot, remote_bytes, attestations = self.fixture()
        result = self.audit.validate_remote_release(
            snapshot, remote_bytes, attestations, self.matrix, self.version,
            REPO, SHA, RUN_ID,
        )
        self.assertEqual(result["asset_count"], 35)
        self.assertEqual(result["archive_count"], 11)

    def test_every_remote_invariant_fails_closed(self) -> None:
        mutations = {}
        snapshot, remote_bytes, attestations = self.fixture()
        mutations["tag SHA"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["tag SHA"][0]["tag"]["object"]["sha"] = "f" * 40
        mutations["release prerelease"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["release prerelease"][0]["release"]["prerelease"] = False
        mutations["release draft"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["release draft"][0]["release"]["draft"] = True
        mutations["run SHA"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["run SHA"][0]["run"]["head_sha"] = "e" * 40
        mutations["job conclusion"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["job conclusion"][0]["jobs"][0]["conclusion"] = "failure"
        mutations["extra asset"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["extra asset"][0]["release"]["assets"].append({"id": 999, "name": "extra", "size": 0, "digest": "sha256:" + "0" * 64})
        mutations["duplicate asset"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["duplicate asset"][0]["release"]["assets"].append(copy.deepcopy(snapshot["release"]["assets"][0]))
        mutations["API digest"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["API digest"][0]["release"]["assets"][0]["digest"] = None
        mutations["download bytes"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        first_name = mutations["download bytes"][0]["release"]["assets"][0]["name"]
        mutations["download bytes"][1][first_name] += b"drift"
        archive = next(name for name in remote_bytes if name.endswith((".tar.gz", ".zip")))
        mutations["checksum"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["checksum"][1][archive + ".sha256"] = ("0" * 64 + f"  {archive}\n").encode()
        mutations["provenance"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["provenance"][1][archive + ".provenance.json"] = b'{"subject": []}'
        mutations["index"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        index = json.loads(mutations["index"][1]["release-index.json"])
        index["source_sha"] = "d" * 40
        mutations["index"][1]["release-index.json"] = json.dumps(index).encode()
        mutations["attestation missing"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["attestation missing"][2][archive] = []
        mutations["attestation duplicate"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["attestation duplicate"][2][archive] *= 2
        mutations["attestation identity"] = (copy.deepcopy(snapshot), dict(remote_bytes), copy.deepcopy(attestations))
        mutations["attestation identity"][2][archive][0]["source_sha"] = "c" * 40

        for name, (mutant_snapshot, mutant_bytes, mutant_attestations) in mutations.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.audit.validate_remote_release(
                    mutant_snapshot, mutant_bytes, mutant_attestations,
                    self.matrix, self.version, REPO, SHA, RUN_ID,
                )

    def test_downloaded_json_index_is_exactly_canonical(self) -> None:
        row_mutations = {
            "artifact id": ("id", "wrong-id"),
            "backend": ("backend", "wrong-backend"),
            "channel": ("channel", "wrong-channel"),
            "host": ("host", {"os": "wrong", "arch": "wrong", "abi": "wrong", "abi_version": "wrong"}),
            "cpu tiers": ("cpu_tiers", ["wrong-tier"]),
            "driver boundary": ("driver_boundary", "wrong-boundary"),
            "limitations": ("limitations", ["invented limitation"]),
            "CUDA SMs": ("sms", ["999"]),
            "static boundary": ("static_boundary", "wrong-static-boundary"),
        }
        accepted = []
        for name, (field, value) in row_mutations.items():
            snapshot, remote_bytes, attestations = self.fixture()
            index = json.loads(remote_bytes["release-index.json"])
            index["artifacts"][0][field] = value
            data = (json.dumps(index, indent=2, sort_keys=True) + "\n").encode()
            self.replace_remote(snapshot, remote_bytes, "release-index.json", data)
            try:
                self.audit.validate_remote_release(
                    snapshot, remote_bytes, attestations, self.matrix, self.version,
                    REPO, SHA, RUN_ID,
                )
            except ValueError:
                continue
            accepted.append(name)

        for name, mutate in (
            ("retention", lambda index: index.__setitem__("retention", {"ci_artifacts_days": 99, "github_release": "forever"})),
            ("unexpected row field", lambda index: index["artifacts"][0].__setitem__("invented", True)),
            ("unexpected top field", lambda index: index.__setitem__("invented", True)),
            ("missing row field", lambda index: index["artifacts"][0].pop("backend")),
            ("missing top field", lambda index: index.pop("retention")),
            ("reordered rows", lambda index: index["artifacts"].reverse()),
        ):
            snapshot, remote_bytes, attestations = self.fixture()
            index = json.loads(remote_bytes["release-index.json"])
            mutate(index)
            data = (json.dumps(index, indent=2, sort_keys=True) + "\n").encode()
            self.replace_remote(snapshot, remote_bytes, "release-index.json", data)
            try:
                self.audit.validate_remote_release(
                    snapshot, remote_bytes, attestations, self.matrix, self.version,
                    REPO, SHA, RUN_ID,
                )
            except ValueError:
                continue
            accepted.append(name)
        self.assertEqual(accepted, [], f"audit accepted noncanonical JSON index mutations: {accepted}")

    def test_downloaded_markdown_index_is_byte_exact_canonical_output(self) -> None:
        snapshot, remote_bytes, _ = self.fixture()
        canonical = remote_bytes["RELEASE_INDEX.md"].decode()
        rows = [line for line in canonical.splitlines() if line.startswith("| [")]
        malformed = (
            f"v0.0.3-vk.1 {SHA}\n" + "\n".join(
                self.audit.canonical_archive_name(
                    self.version["version"], item["id"], item["archive_format"]
                )
                for item in self.matrix["artifacts"]
            ) + "\n"
        )
        mutations = {
            "extra row": canonical + rows[0] + "\n",
            "duplicate row": canonical.replace(rows[1], rows[0], 1),
            "reordered rows": canonical.replace(rows[0], "ROW-SWAP", 1).replace(rows[1], rows[0], 1).replace("ROW-SWAP", rows[1], 1),
            "false prerelease state": canonical.replace("Prerelease: `true`", "Prerelease: `false`", 1),
            "wrong checksum link": canonical.replace(".sha256)", ".wrong.sha256)", 1),
            "wrong artifact id": canonical.replace(self.matrix["artifacts"][0]["id"], "wrong-id", 1),
            "wrong channel": canonical.replace(f"| {self.matrix['artifacts'][0]['channel']} |", "| wrong-channel |", 1),
            "malformed substring-only document": malformed,
            "no table": canonical.replace("| Artifact |", "Artifact ", 1),
        }
        accepted = []
        for name, markdown in mutations.items():
            mutant_snapshot, mutant_bytes, mutant_attestations = self.fixture()
            self.replace_remote(
                mutant_snapshot, mutant_bytes, "RELEASE_INDEX.md", markdown.encode()
            )
            try:
                self.audit.validate_remote_release(
                    mutant_snapshot, mutant_bytes, mutant_attestations,
                    self.matrix, self.version, REPO, SHA, RUN_ID,
                )
            except ValueError:
                continue
            accepted.append(name)
        self.assertEqual(accepted, [], f"audit accepted noncanonical Markdown mutations: {accepted}")

    def test_live_collector_uses_mocked_authenticated_apis_and_downloads(self) -> None:
        snapshot, remote_bytes, _ = self.fixture()
        release = copy.deepcopy(snapshot["release"])
        by_id = {row["id"]: row["name"] for row in release["assets"]}

        def json_side_effect(args):
            endpoint = args[-1]
            if endpoint == f"repos/{REPO}/releases/tags/v0.0.3-vk.1":
                return release
            if endpoint == f"repos/{REPO}/actions/runs/{RUN_ID}":
                return snapshot["run"]
            if endpoint.endswith("/jobs?per_page=100"):
                return [{"jobs": snapshot["jobs"]}]
            if endpoint == f"repos/{REPO}/git/ref/tags/v0.0.3-vk.1":
                return {"object": {"sha": SHA, "type": "commit"}}
            if args[:2] == ["attestation", "verify"]:
                self.assertIn(f"{REPO}/.github/workflows/release.yml", args)
                self.assertIn(SHA, args)
                self.assertIn("refs/tags/v0.0.3-vk.1", args)
                return [{"verificationResult": {"statement": {"subject": []}, "certificate": {
                    "sourceRepositoryURI": f"https://github.com/{REPO}",
                    "sourceRepositoryDigest": SHA,
                    "runInvocationURI": f"https://github.com/{REPO}/actions/runs/{RUN_ID}",
                }}}]
            raise AssertionError(args)

        def bytes_side_effect(args):
            asset_id = int(args[-1].rsplit("/", 1)[1])
            return remote_bytes[by_id[asset_id]]

        with mock.patch.object(self.audit, "gh_json", side_effect=json_side_effect), mock.patch.object(
            self.audit, "gh_bytes", side_effect=bytes_side_effect
        ):
            observed, downloads, attestations = self.audit.collect_remote(
                REPO, "v0.0.3-vk.1", SHA, RUN_ID
            )
        self.assertEqual(downloads, remote_bytes)
        result = self.audit.validate_remote_release(
            observed, downloads, attestations, self.matrix, self.version,
            REPO, SHA, RUN_ID,
        )
        self.assertEqual(result["asset_count"], 35)

    def test_annotated_and_lightweight_tags_resolve_to_the_commit(self) -> None:
        with mock.patch.object(
            self.audit, "gh_json", return_value={"object": {"sha": SHA, "type": "commit"}}
        ):
            self.assertEqual(
                self.audit.resolve_tag(REPO, "v0.0.3-vk.1"),
                {"sha": SHA, "type": "commit"},
            )
        responses = iter((
            {"object": {"sha": "a" * 40, "type": "tag"}},
            {"tag": "v0.0.3-vk.1", "object": {"sha": SHA, "type": "commit"}},
        ))
        with mock.patch.object(self.audit, "gh_json", side_effect=lambda args: next(responses)):
            self.assertEqual(
                self.audit.resolve_tag(REPO, "v0.0.3-vk.1"),
                {"sha": SHA, "type": "commit"},
            )


if __name__ == "__main__":
    unittest.main()
