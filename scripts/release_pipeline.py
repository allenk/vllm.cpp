#!/usr/bin/env python3
"""Plan and verify immutable W8 release workflow handoffs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


PLAN_SCHEMA = "vllm.cpp.release-plan.v1"
HANDOFF_SCHEMA = "vllm.cpp.release-handoff.v1"
MATRIX_SCHEMA = "vllm.cpp.release-matrix.v1"
RELEASE_INDEX_SCHEMA = "vllm.cpp.release-index.v1"
RELEASE_VERSION_SCHEMA = "vllm.cpp.release-version.v1"
CHANNELS = {"stable", "preview", "experimental-preview"}
ARCHIVE_FORMATS = {"tar.gz", "zip"}
PRIMARY_ARTIFACT_FORMATS = {
    "linux-x86_64-glibc-cpu": "tar.gz",
    "linux-aarch64-glibc-cpu": "tar.gz",
    "linux-x86_64-musl-cpu-static": "tar.gz",
    "linux-x86_64-glibc-cuda": "tar.gz",
    "linux-aarch64-glibc-cuda": "tar.gz",
    "macos-arm64-metal": "tar.gz",
    "macos-arm64-metal-mlx": "tar.gz",
    "linux-x86_64-glibc-vulkan": "tar.gz",
    "windows-x86_64-msvc-cpu": "zip",
    "windows-x86_64-msvc-vulkan": "zip",
    # FORK: cuda_windows is this fork's job -- upstream has no Windows CUDA lane.
    # It is wired into build.needs and its triplet is downloaded into release-assets,
    # so collate() would raise "not declared by the matrix" without this entry.
    "windows-x86_64-msvc-cuda": "zip",
}
RELEASE_TAG = re.compile(r"v[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?")
SEMANTIC_VERSION = re.compile(
    r"(?P<project>[0-9]+\.[0-9]+\.[0-9]+)(?P<suffix>[-+][0-9A-Za-z.-]+)?"
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_archive_name(version: str, artifact_id: str, archive_format: str) -> str:
    if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?", version) is None:
        raise ValueError("archive version must be a semantic version")
    if re.fullmatch(r"[a-z0-9][a-z0-9_.-]+", artifact_id) is None:
        raise ValueError("archive artifact ID is unsafe")
    if archive_format not in ARCHIVE_FORMATS:
        raise ValueError(f"archive format must be one of {sorted(ARCHIVE_FORMATS)}")
    return f"vllm.cpp-{version}-{artifact_id}.{archive_format}"


def validate_release_version(declaration: dict[str, Any]) -> dict[str, Any]:
    if set(declaration) != {
        "prerelease", "project_version", "schema", "tag", "version"
    }:
        raise ValueError("release version declaration has unknown or missing fields")
    if declaration.get("schema") != RELEASE_VERSION_SCHEMA:
        raise ValueError(f"release version schema must be {RELEASE_VERSION_SCHEMA}")
    version = declaration.get("version")
    project_version = declaration.get("project_version")
    tag = declaration.get("tag")
    prerelease = declaration.get("prerelease")
    match = SEMANTIC_VERSION.fullmatch(version) if isinstance(version, str) else None
    if match is None:
        raise ValueError("release version must be a semantic version")
    if not isinstance(project_version, str) or re.fullmatch(
        r"[0-9]+\.[0-9]+\.[0-9]+", project_version
    ) is None:
        raise ValueError("project version must be numeric CMake SemVer")
    if match.group("project") != project_version:
        raise ValueError("release version must extend the declared project version")
    if not isinstance(tag, str) or tag != f"v{version}":
        raise ValueError("release tag must exactly match the declared version")
    expected_prerelease = "-" in version.split("+", 1)[0]
    if type(prerelease) is not bool or prerelease is not expected_prerelease:
        raise ValueError("release prerelease state does not match its semantic version")
    return dict(declaration)


def validate_matrix(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    if set(matrix) != {"artifacts", "release_ready", "retention", "schema"}:
        raise ValueError("release matrix has unknown or missing fields")
    if matrix.get("schema") != MATRIX_SCHEMA:
        raise ValueError(f"release matrix schema must be {MATRIX_SCHEMA}")
    if type(matrix.get("release_ready")) is not bool:
        raise ValueError("release matrix release_ready must be boolean")
    if matrix.get("retention") != {
        "ci_artifacts_days": 7,
        "github_release": "maintainer-deletion-only",
    }:
        raise ValueError("release matrix retention policy is invalid")
    artifacts = matrix.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("release matrix artifacts must be a non-empty array")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(artifacts):
        if not isinstance(item, dict) or set(item) != {
            "archive_format", "id", "channel", "required"
        }:
            raise ValueError(f"release matrix artifact {index} has unknown or missing fields")
        artifact_id = item.get("id")
        if not isinstance(artifact_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9_.-]+", artifact_id) is None:
            raise ValueError(f"release matrix artifact {index} has unsafe id")
        if item.get("channel") not in CHANNELS or type(item.get("required")) is not bool:
            raise ValueError(f"release matrix artifact {artifact_id} has invalid policy")
        if item.get("archive_format") not in ARCHIVE_FORMATS:
            raise ValueError(f"release matrix artifact {artifact_id} has invalid archive format")
        normalized.append(dict(item))
    ids = [item["id"] for item in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("release matrix artifact ids must be unique")
    formats = {item["id"]: item["archive_format"] for item in normalized}
    if formats != PRIMARY_ARTIFACT_FORMATS:
        raise ValueError("release matrix must declare the exact primary tuple/archive-format policy")
    return normalized


def make_plan(
    event: str,
    ref: str,
    source_sha: str,
    declaration: dict[str, Any],
    matrix: dict[str, Any],
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("source SHA must be a full lowercase 40-hex commit")
    release = validate_release_version(declaration)
    version = release["version"]
    artifacts = validate_matrix(matrix)
    if event == "workflow_dispatch":
        if not ref.startswith("refs/heads/"):
            raise ValueError("manual dry runs must resolve a branch ref")
        release_tag = f"dry-run-{source_sha[:12]}"
        publish = False
    elif event == "push":
        release_tag = release["tag"]
        if ref != f"refs/tags/{release_tag}":
            raise ValueError(f"release tag must exactly equal {release_tag}")
        publish = matrix["release_ready"]
    else:
        raise ValueError(f"unsupported release event {event!r}")
    return {
        "artifacts": artifacts,
        "event": event,
        "prerelease": release["prerelease"],
        "project_version": release["project_version"],
        "publish": publish,
        "release_tag": release_tag,
        "retention": matrix["retention"],
        "schema": PLAN_SCHEMA,
        "source_sha": source_sha,
        "version": version,
    }


def inventory_assets(plan: dict[str, Any], assets_dir: Path) -> list[dict[str, Any]]:
    if not assets_dir.is_dir():
        raise ValueError(f"asset directory does not exist: {assets_dir}")
    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("plan artifacts are invalid")
    expected: dict[str, str] = {}
    required_sets: dict[str, set[str]] = {}
    version = plan.get("version")
    if not isinstance(version, str):
        raise ValueError("plan version is invalid")
    for item in artifacts:
        artifact_id = item["id"]
        archive = canonical_archive_name(version, artifact_id, item["archive_format"])
        names = {
            archive,
            f"{archive}.sha256",
            f"{archive}.provenance.json",
        }
        required_sets[artifact_id] = names
        expected.update({name: artifact_id for name in names})
    actual_paths = sorted(assets_dir.iterdir(), key=lambda path: path.name)
    for path in actual_paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"release asset must be a regular file: {path.name}")
        if path.name not in expected:
            raise ValueError(f"release asset is not declared by the matrix: {path.name}")
    actual = {path.name for path in actual_paths}
    for item in artifacts:
        names = required_sets[item["id"]]
        present = names & actual
        if present and present != names:
            raise ValueError(f"release asset triplet is incomplete for {item['id']}")
        if plan.get("publish") is True and item["required"] and present != names:
            raise ValueError(f"publish-ready handoff is missing required artifact {item['id']}")
    return [
        {
            "artifact_id": expected[path.name],
            "name": path.name,
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
        }
        for path in actual_paths
    ]


def handoff_value(plan_path: Path, assets_dir: Path) -> dict[str, Any]:
    plan = read_json(plan_path)
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("handoff input is not a release plan")
    return {
        "artifacts": plan.get("artifacts"),
        "files": inventory_assets(plan, assets_dir),
        "plan_sha256": file_sha256(plan_path),
        "prerelease": plan.get("prerelease"),
        "project_version": plan.get("project_version"),
        "publish": plan.get("publish"),
        "release_tag": plan.get("release_tag"),
        "retention": plan.get("retention"),
        "schema": HANDOFF_SCHEMA,
        "source_sha": plan.get("source_sha"),
        "version": plan.get("version"),
    }


def make_handoff(plan_path: Path, assets_dir: Path, output: Path) -> None:
    write_json(output, handoff_value(plan_path, assets_dir))


def verify_handoff(
    plan_path: Path,
    handoff_path: Path,
    assets_dir: Path,
    output: Path,
    expected_sha: str,
) -> None:
    plan = read_json(plan_path)
    handoff = read_json(handoff_path)
    expected = handoff_value(plan_path, assets_dir)
    expected["source_sha"] = expected_sha
    if handoff != expected or plan.get("source_sha") != expected_sha:
        raise ValueError("release handoff does not match the immutable plan and workflow SHA")
    write_json(output, {**handoff, "verified": True})


def publish_release(
    handoff_path: Path,
    assets_dir: Path,
    index_json: Path,
    index_markdown: Path,
    tag: str,
) -> None:
    """Publish only the regular files authenticated by a verified handoff."""
    handoff = read_json(handoff_path)
    if handoff.get("verified") is not True or handoff.get("publish") is not True:
        raise ValueError("release publication requires a verified publish handoff")
    if RELEASE_TAG.fullmatch(tag) is None or handoff.get("release_tag") != tag:
        raise ValueError("release tag does not match the verified handoff")
    prerelease = handoff.get("prerelease")
    project_version = handoff.get("project_version")
    version = handoff.get("version")
    if type(prerelease) is not bool or not isinstance(project_version, str):
        raise ValueError("verified handoff has no authenticated release state")
    validate_release_version({
        "prerelease": prerelease,
        "project_version": project_version,
        "schema": RELEASE_VERSION_SCHEMA,
        "tag": tag,
        "version": version,
    })
    if not assets_dir.is_dir():
        raise ValueError(f"asset directory does not exist: {assets_dir}")
    files = handoff.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("verified handoff has no release files")

    expected: dict[str, dict[str, Any]] = {}
    for item in files:
        if not isinstance(item, dict) or set(item) < {"name", "sha256", "size"}:
            raise ValueError("verified handoff file inventory is malformed")
        name = item["name"]
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in expected
        ):
            raise ValueError("verified handoff contains an unsafe or duplicate file name")
        expected[name] = item

    actual_paths = sorted(assets_dir.iterdir(), key=lambda path: path.name)
    actual_names = {path.name for path in actual_paths}
    if actual_names != set(expected):
        raise ValueError("release assets do not exactly match the verified handoff")
    for path in actual_paths:
        item = expected[path.name]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"release asset must be a regular file: {path.name}")
        if file_sha256(path) != item.get("sha256") or path.stat().st_size != item.get("size"):
            raise ValueError(f"release asset drifted after verification: {path.name}")
    for path in (index_json, index_markdown):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"release index must be a regular file: {path}")
    index = read_json(index_json)
    if (
        index.get("schema") != RELEASE_INDEX_SCHEMA
        or index.get("release_tag") != tag
        or index.get("source_sha") != handoff.get("source_sha")
        or index.get("version") != version
        or index.get("project_version") != project_version
        or index.get("prerelease") is not prerelease
    ):
        raise ValueError("release index identity does not match the verified handoff")
    index_rows = index.get("artifacts")
    if not isinstance(index_rows, list):
        raise ValueError("release index artifacts must be an array")
    expected_archives = {
        name: item
        for name, item in expected.items()
        if name.endswith((".tar.gz", ".zip"))
    }
    declared_artifacts = handoff.get("artifacts")
    if not isinstance(declared_artifacts, list) or not declared_artifacts:
        raise ValueError("verified handoff has no explicit artifact formats")
    artifact_formats: dict[str, str] = {}
    for item in declared_artifacts:
        if (not isinstance(item, dict) or item.get("archive_format") not in ARCHIVE_FORMATS
                or not isinstance(item.get("id"), str) or item["id"] in artifact_formats):
            raise ValueError("verified handoff artifact format is invalid")
        artifact_formats[item["id"]] = item["archive_format"]
    indexed_archives: set[str] = set()
    for row in index_rows:
        if not isinstance(row, dict):
            raise ValueError("release index artifact row is malformed")
        archive = row.get("archive")
        artifact_id = row.get("id")
        archive_format = artifact_formats.get(artifact_id)
        if (
            not isinstance(archive, str)
            or not isinstance(artifact_id, str)
            or not isinstance(version, str)
            or archive_format not in ARCHIVE_FORMATS
            or archive != canonical_archive_name(version, artifact_id, archive_format)
            or archive not in expected_archives
            or archive in indexed_archives
            or row.get("sha256") != expected_archives[archive].get("sha256")
        ):
            raise ValueError("release index does not match the verified archive inventory")
        indexed_archives.add(archive)
    if indexed_archives != set(expected_archives):
        raise ValueError("release index does not enumerate every verified archive")
    markdown = index_markdown.read_text(encoding="utf-8")
    required_markdown = [tag, str(handoff.get("source_sha", "")), *indexed_archives]
    if any(not value or value not in markdown for value in required_markdown):
        raise ValueError("release Markdown index does not match the verified handoff")

    create = [
        "gh",
        "release",
        "create",
        tag,
        *(str(path) for path in actual_paths),
        str(index_json),
        str(index_markdown),
        "--verify-tag",
        "--title",
        tag,
        "--notes-file",
        str(index_markdown),
    ]
    if prerelease:
        create.append("--prerelease")
    subprocess.run(create, check=True)
    result = subprocess.run(
        ["gh", "release", "view", tag, "--json", "isDraft,isPrerelease,tagName"],
        check=True,
        capture_output=True,
        text=True,
    )
    state = json.loads(result.stdout)
    if state != {"isDraft": False, "isPrerelease": prerelease, "tagName": tag}:
        raise ValueError("published GitHub release state does not match the verified handoff")


def write_outputs(path: Path, values: dict[str, str | bool]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            rendered = str(value).lower() if isinstance(value, bool) else value
            if "\n" in rendered:
                raise ValueError("workflow outputs must be single-line values")
            handle.write(f"{key}={rendered}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--event", required=True)
    plan.add_argument("--ref", required=True)
    plan.add_argument("--sha", required=True)
    plan.add_argument("--release-version", type=Path, required=True)
    plan.add_argument("--matrix", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--github-output", type=Path)
    handoff = commands.add_parser("handoff")
    handoff.add_argument("--plan", type=Path, required=True)
    handoff.add_argument("--assets-dir", type=Path, required=True)
    handoff.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--handoff", type=Path, required=True)
    verify.add_argument("--assets-dir", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--sha", required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--handoff", type=Path, required=True)
    publish.add_argument("--assets-dir", type=Path, required=True)
    publish.add_argument("--index-json", type=Path, required=True)
    publish.add_argument("--index-markdown", type=Path, required=True)
    publish.add_argument("--tag", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "plan":
            plan = make_plan(
                args.event,
                args.ref,
                args.sha,
                read_json(args.release_version),
                read_json(args.matrix),
            )
            write_json(args.output, plan)
            if args.github_output:
                write_outputs(
                    args.github_output,
                    {
                        "artifact_name": f"release-plan-{args.sha}",
                        "publish": plan["publish"],
                        "release_tag": plan["release_tag"],
                        "version": plan["version"],
                    },
                )
        elif args.command == "handoff":
            make_handoff(args.plan, args.assets_dir, args.output)
        elif args.command == "verify":
            verify_handoff(args.plan, args.handoff, args.assets_dir, args.output, args.sha)
        else:
            publish_release(
                args.handoff,
                args.assets_dir,
                args.index_json,
                args.index_markdown,
                args.tag,
            )
    except (OSError, json.JSONDecodeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"release pipeline error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
