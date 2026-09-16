#!/usr/bin/env python3
"""Authenticate a published binary release from remote GitHub state and bytes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


_INDEX_SPEC = importlib.util.spec_from_file_location(
    "vllm_release_index", Path(__file__).with_name("release_index.py")
)
if _INDEX_SPEC is None or _INDEX_SPEC.loader is None:
    raise RuntimeError("cannot load the canonical release index generator")
release_index_tool = importlib.util.module_from_spec(_INDEX_SPEC)
_INDEX_SPEC.loader.exec_module(release_index_tool)


REQUIRED_RELEASE_JOBS = (
    "plan", "cpu_x86", "cpu_arm64", "cpu_musl", "cuda_x86", "cuda_arm64",
    "vulkan_x86", "metal_arm64", "mlx_arm64", "cpu_windows",
    "vulkan_windows", "build", "verify", "attest", "publish",
)
ARCHIVE_FORMATS = {"tar.gz", "zip"}
SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical_archive_name(version: str, artifact_id: str, archive_format: str) -> str:
    if archive_format not in ARCHIVE_FORMATS:
        raise ValueError("invalid archive format")
    return f"vllm.cpp-{version}-{artifact_id}.{archive_format}"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def expected_names(matrix: dict[str, Any], version: str) -> tuple[set[str], set[str]]:
    if set(matrix) != {"artifacts", "release_ready", "retention", "schema"} or (
        matrix.get("schema") != "vllm.cpp.release-matrix.v1"
        or matrix.get("release_ready") is not True
        or matrix.get("retention") != {
            "ci_artifacts_days": 7,
            "github_release": "maintainer-deletion-only",
        }
    ):
        raise ValueError("audit requires the exact authoritative release matrix")
    artifacts = matrix.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 10:
        raise ValueError("audit requires the canonical ten-artifact matrix")
    archives: set[str] = set()
    names = {"release-index.json", "RELEASE_INDEX.md"}
    ids: set[str] = set()
    for item in artifacts:
        if (
            not isinstance(item, dict)
            or set(item) != {"archive_format", "channel", "id", "required"}
            or item.get("required") is not True
            or not isinstance(item.get("channel"), str)
        ):
            raise ValueError("every audited artifact must be required")
        artifact_id = item.get("id")
        archive_format = item.get("archive_format")
        if not isinstance(artifact_id, str) or artifact_id in ids:
            raise ValueError("audited artifact IDs must be unique")
        archive = canonical_archive_name(version, artifact_id, archive_format)
        ids.add(artifact_id)
        archives.add(archive)
        names.update((archive, archive + ".sha256", archive + ".provenance.json"))
    if len(names) != 32:
        raise ValueError("audited release must have exactly 32 canonical assets")
    return names, archives


def _canonical_index_bytes(
    remote_bytes: dict[str, bytes], matrix: dict[str, Any], declaration: dict[str, Any],
    source_sha: str,
) -> tuple[bytes, bytes]:
    """Regenerate both indexes from authenticated downloaded triplet bytes."""
    with tempfile.TemporaryDirectory(prefix="vllm-release-index-audit-") as temporary:
        root = Path(temporary)
        assets = root / "assets"
        assets.mkdir()
        files: list[dict[str, Any]] = []
        for item in matrix["artifacts"]:
            archive = canonical_archive_name(
                declaration["version"], item["id"], item["archive_format"]
            )
            for name in (archive, archive + ".sha256", archive + ".provenance.json"):
                data = remote_bytes[name]
                (assets / name).write_bytes(data)
                files.append({
                    "artifact_id": item["id"],
                    "name": name,
                    "sha256": digest(data),
                    "size": len(data),
                })
        handoff = {
            "artifacts": matrix["artifacts"],
            "files": files,
            "prerelease": declaration["prerelease"],
            "project_version": declaration["project_version"],
            "release_tag": declaration["tag"],
            "retention": matrix["retention"],
            "source_sha": source_sha,
            "verified": True,
            "version": declaration["version"],
        }
        json_output = root / "release-index.json"
        markdown_output = root / "RELEASE_INDEX.md"
        try:
            release_index_tool.generate_index(
                assets, handoff, json_output, markdown_output,
                matrix["retention"]["ci_artifacts_days"],
            )
        except (
            OSError,
            KeyError,
            json.JSONDecodeError,
            release_index_tool.tarfile.TarError,
            release_index_tool.zipfile.BadZipFile,
            ValueError,
        ) as exc:
            raise ValueError(f"downloaded release triplets cannot regenerate indexes: {exc}") from exc
        return json_output.read_bytes(), markdown_output.read_bytes()


def _single_subject(provenance: Any, archive: str, expected_digest: str) -> None:
    subjects = provenance.get("subject") if isinstance(provenance, dict) else None
    matches = [
        subject for subject in subjects or []
        if isinstance(subject, dict) and subject.get("name") == archive
        and subject.get("digest") == {"sha256": expected_digest}
    ]
    if len(matches) != 1 or len(subjects or []) != 1:
        raise ValueError(f"provenance subject does not bind {archive}")


def validate_remote_release(
    snapshot: dict[str, Any], remote_bytes: dict[str, bytes],
    attestations: dict[str, list[dict[str, Any]]], matrix: dict[str, Any],
    declaration: dict[str, Any], repo: str, source_sha: str, run_id: str,
) -> dict[str, Any]:
    """Validate only authenticated remote observations; local publish inputs are absent."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha) or not run_id.isdigit():
        raise ValueError("audit source SHA and run ID must be exact")
    version = declaration.get("version")
    project_version = declaration.get("project_version")
    tag = declaration.get("tag")
    if set(declaration) != {"prerelease", "project_version", "schema", "tag", "version"} or (
        declaration.get("schema") != "vllm.cpp.release-version.v1"
        or version != "0.0.3-vk.1" or project_version != "0.0.3"
        or tag != "v0.0.3-vk.1" or declaration.get("prerelease") is not True
    ):
        raise ValueError("audit release identity must be the authorized pre-alpha")
    names, archives = expected_names(matrix, version)

    tag_object = snapshot.get("tag", {}).get("object", {})
    if tag_object != {"sha": source_sha, "type": "commit"}:
        raise ValueError("release tag does not resolve exactly to the expected commit")
    release = snapshot.get("release")
    if not isinstance(release, dict) or (
        release.get("tag_name") != tag or release.get("draft") is not False
        or release.get("prerelease") is not True
    ):
        raise ValueError("GitHub release is not the exact non-draft prerelease")
    run = snapshot.get("run")
    if not isinstance(run, dict) or (
        str(run.get("id")) != run_id or run.get("head_sha") != source_sha
        or run.get("path") != ".github/workflows/release.yml"
        or run.get("conclusion") != "success"
    ):
        raise ValueError("release workflow run identity or conclusion is invalid")
    jobs = snapshot.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("release workflow jobs are missing")
    by_job: dict[str, list[Any]] = {}
    for job in jobs:
        if isinstance(job, dict):
            by_job.setdefault(str(job.get("name")), []).append(job.get("conclusion"))
    for required in REQUIRED_RELEASE_JOBS:
        if by_job.get(required) != ["success"]:
            raise ValueError(f"required release job {required} did not succeed exactly once")

    asset_rows = release.get("assets")
    if not isinstance(asset_rows, list) or len(asset_rows) != 32:
        raise ValueError("GitHub release must report exactly 32 assets")
    by_name: dict[str, dict[str, Any]] = {}
    for row in asset_rows:
        name = row.get("name") if isinstance(row, dict) else None
        if not isinstance(name, str) or name in by_name:
            raise ValueError("GitHub release asset names must be unique")
        by_name[name] = row
    if set(by_name) != names or set(remote_bytes) != names:
        raise ValueError("remote release assets do not match the canonical 32 names")
    for name, data in remote_bytes.items():
        row = by_name[name]
        api_digest = row.get("digest")
        if row.get("size") != len(data) or api_digest != "sha256:" + digest(data):
            raise ValueError(f"GitHub API digest/size disagrees with downloaded bytes: {name}")

    for archive in archives:
        archive_digest = digest(remote_bytes[archive])
        checksum_name = archive + ".sha256"
        provenance_name = archive + ".provenance.json"
        checksum = remote_bytes[checksum_name].decode("utf-8").split()
        if checksum != [archive_digest, archive]:
            raise ValueError(f"checksum sidecar does not match downloaded {archive}")
        _single_subject(json.loads(remote_bytes[provenance_name]), archive, archive_digest)
        proofs = attestations.get(archive)
        expected_proof = {
            "digest": archive_digest, "repository": repo, "source_sha": source_sha,
            "run_id": run_id, "verified": True,
        }
        if proofs != [expected_proof]:
            raise ValueError(f"{archive} must have exactly one valid bound attestation")
    expected_json, expected_markdown = _canonical_index_bytes(
        remote_bytes, matrix, declaration, source_sha
    )
    if remote_bytes["release-index.json"] != expected_json:
        raise ValueError("downloaded JSON release index is not exact canonical output")
    if remote_bytes["RELEASE_INDEX.md"] != expected_markdown:
        raise ValueError("downloaded Markdown release index is not exact canonical output")
    if set(attestations) != archives:
        raise ValueError("indexes or attestations omit a canonical archive")
    return {"archive_count": len(archives), "asset_count": len(names), "run_id": run_id}


def gh_json(args: list[str]) -> Any:
    result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def gh_bytes(args: list[str]) -> bytes:
    return subprocess.run(["gh", *args], check=True, capture_output=True).stdout


def resolve_tag(repo: str, tag: str) -> dict[str, str]:
    value = gh_json(["api", f"repos/{repo}/git/ref/tags/{tag}"])["object"]
    seen: set[str] = set()
    while value.get("type") == "tag":
        sha = value.get("sha")
        if not isinstance(sha, str) or sha in seen:
            raise ValueError("annotated tag resolution is cyclic or malformed")
        seen.add(sha)
        annotated = gh_json(["api", f"repos/{repo}/git/tags/{sha}"])
        if annotated.get("tag") != tag:
            raise ValueError("annotated tag name does not match release tag")
        value = annotated.get("object")
    if not isinstance(value, dict):
        raise ValueError("release tag object is malformed")
    return value


def _all_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _all_strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in _all_strings(child)]
    return []


def collect_remote(repo: str, tag: str, source_sha: str, run_id: str):
    release = gh_json(["api", f"repos/{repo}/releases/tags/{tag}"])
    run = gh_json(["api", f"repos/{repo}/actions/runs/{run_id}"])
    job_pages = gh_json([
        "api", "--paginate", "--slurp", f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100"
    ])
    jobs = [job for page in job_pages for job in page.get("jobs", [])]
    remote_bytes: dict[str, bytes] = {}
    attestations: dict[str, list[dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory(prefix="vllm-release-audit-") as temporary:
        root = Path(temporary)
        for asset in release.get("assets", []):
            name = asset.get("name")
            asset_id = asset.get("id")
            if not isinstance(name, str) or Path(name).name != name or not isinstance(asset_id, int):
                raise ValueError("release API returned an unsafe asset")
            data = gh_bytes([
                "api", "-H", "Accept: application/octet-stream",
                f"repos/{repo}/releases/assets/{asset_id}",
            ])
            remote_bytes[name] = data
            (root / name).write_bytes(data)
        for name, data in remote_bytes.items():
            if not name.endswith((".tar.gz", ".zip")):
                continue
            verified = gh_json([
                "attestation", "verify", str(root / name),
                "--repo", repo,
                "--signer-workflow", f"{repo}/.github/workflows/release.yml",
                "--source-digest", source_sha,
                "--source-ref", f"refs/tags/{tag}",
                "--format", "json",
            ])
            normalized = []
            for proof in verified if isinstance(verified, list) else []:
                strings = _all_strings(proof)
                normalized.append({
                    "digest": digest(data),
                    "repository": repo if any(repo in item for item in strings) else "",
                    "source_sha": source_sha if source_sha in strings else "",
                    "run_id": run_id if any(f"/actions/runs/{run_id}" in item for item in strings) else "",
                    "verified": True,
                })
            attestations[name] = normalized
    return {
        "tag": {"object": resolve_tag(repo, tag)},
        "release": {
            "assets": release.get("assets"), "draft": release.get("draft"),
            "prerelease": release.get("prerelease"), "tag_name": release.get("tag_name"),
        },
        "run": {
            "conclusion": run.get("conclusion"), "head_sha": run.get("head_sha"),
            "id": run.get("id"), "path": run.get("path"),
        },
        "jobs": [{"name": job.get("name"), "conclusion": job.get("conclusion")} for job in jobs],
    }, remote_bytes, attestations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--release-version", type=Path, required=True)
    args = parser.parse_args()
    try:
        matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
        declaration = json.loads(args.release_version.read_text(encoding="utf-8"))
        if args.tag != declaration.get("tag"):
            raise ValueError("requested audit tag disagrees with the release declaration")
        snapshot, remote_bytes, attestations = collect_remote(
            args.repo, args.tag, args.sha, args.run_id
        )
        result = validate_remote_release(
            snapshot, remote_bytes, attestations, matrix, declaration,
            args.repo, args.sha, args.run_id,
        )
        print(json.dumps(result, sort_keys=True))
    except (OSError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"post-publication audit error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
