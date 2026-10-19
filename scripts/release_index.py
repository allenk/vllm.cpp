#!/usr/bin/env python3
"""Generate W13 JSON/Markdown indexes exclusively from verified release bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "vllm.cpp.release-index.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(archive: Path, archive_format: str) -> dict[str, Any]:
    if archive_format == "tar.gz":
        with tarfile.open(archive, "r:gz") as bundle:
            members = [member for member in bundle.getmembers() if member.name == "release-manifest.json"]
            if len(members) != 1 or not members[0].isfile() or members[0].size > 4 * 1024 * 1024:
                raise ValueError(f"{archive.name} must contain one bounded release-manifest.json")
            if PurePosixPath(members[0].name).is_absolute():
                raise ValueError(f"{archive.name} has an unsafe manifest member")
            handle = bundle.extractfile(members[0])
            if handle is None:
                raise ValueError(f"{archive.name} manifest cannot be read")
            raw = handle.read()
    elif archive_format == "zip":
        with zipfile.ZipFile(archive) as bundle:
            members = [info for info in bundle.infolist() if info.filename == "release-manifest.json"]
            if len(members) != 1 or members[0].is_dir() or members[0].file_size > 4 * 1024 * 1024:
                raise ValueError(f"{archive.name} must contain one bounded release-manifest.json")
            raw = bundle.read(members[0])
    else:
        raise ValueError(f"unsupported explicit archive format {archive_format!r}")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{archive.name} manifest must be an object")
    return value


def generate_index(
    assets_dir: Path,
    handoff: dict[str, Any],
    json_output: Path,
    markdown_output: Path,
    retention_days: int = 7,
    repository: str | None = None,
) -> dict[str, Any]:
    if handoff.get("verified") is not True:
        raise ValueError("release index requires a verified handoff")
    if repository is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("release index repository must be owner/name")
    retention = handoff.get("retention")
    if retention != {
        "ci_artifacts_days": retention_days,
        "github_release": "maintainer-deletion-only",
    }:
        raise ValueError("release index retention does not match the verified handoff")
    files = handoff.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("verified handoff has no release files")
    by_name: dict[str, dict[str, Any]] = {}
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("verified handoff file inventory is malformed")
        if item["name"] in by_name or Path(item["name"]).name != item["name"]:
            raise ValueError("verified handoff has a duplicate or unsafe file name")
        path = assets_dir / item["name"]
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"verified release file is missing: {item['name']}")
        if sha256(path) != item.get("sha256") or path.stat().st_size != item.get("size"):
            raise ValueError(f"verified release file drifted: {item['name']}")
        by_name[item["name"]] = item
    actual_names = {
        path.name for path in assets_dir.iterdir() if path.is_file() or path.is_symlink()
    }
    if actual_names != set(by_name):
        raise ValueError("release assets do not exactly match the verified handoff")
    version = handoff.get("version")
    if not isinstance(version, str):
        raise ValueError("verified handoff has no release version")
    project_version = handoff.get("project_version")
    prerelease = handoff.get("prerelease")
    if not isinstance(project_version, str) or type(prerelease) is not bool:
        raise ValueError("verified handoff has no authenticated release state")
    declared = handoff.get("artifacts")
    if not isinstance(declared, list):
        raise ValueError("verified handoff has no explicit artifact formats")
    formats: dict[str, str] = {}
    channels: dict[str, str] = {}
    expected_archives: dict[str, str] = {}
    for item in declared:
        if not isinstance(item, dict) or item.get("archive_format") not in {"tar.gz", "zip"}:
            raise ValueError("verified handoff artifact format is invalid")
        artifact_id = item.get("id")
        if not isinstance(artifact_id, str) or artifact_id in formats:
            raise ValueError("verified handoff artifact ID is invalid or duplicated")
        formats[artifact_id] = item["archive_format"]
        channel = item.get("channel")
        if not isinstance(channel, str):
            raise ValueError("verified handoff artifact channel is invalid")
        channels[artifact_id] = channel
        expected_archives[f"vllm.cpp-{version}-{artifact_id}.{item['archive_format']}"] = artifact_id
    archives = sorted(name for name in by_name if name in expected_archives)
    rows: list[dict[str, Any]] = []
    for archive_name in archives:
        declared_id = expected_archives[archive_name]
        manifest = read_manifest(assets_dir / archive_name, formats[declared_id])
        artifact = manifest.get("artifact", {})
        artifact_id = artifact.get("id")
        expected_archive = f"vllm.cpp-{version}-{artifact_id}.{formats.get(artifact_id)}"
        if (
            not isinstance(artifact_id, str)
            or artifact.get("version") != version
            or artifact.get("channel") != channels.get(artifact_id)
            or archive_name != expected_archive
            or by_name[archive_name].get("artifact_id") != artifact_id
        ):
            raise ValueError(f"archive manifest identity mismatch for {archive_name}")
        required_names = {
            archive_name,
            f"{archive_name}.sha256",
            f"{archive_name}.provenance.json",
        }
        if not required_names <= by_name.keys():
            raise ValueError(f"verified asset triplet is incomplete for {artifact_id}")
        if manifest.get("build", {}).get("source_commit") != handoff.get("source_sha"):
            raise ValueError(f"archive source identity mismatch for {archive_name}")
        external = [
            dependency["name"]
            for dependency in manifest.get("dependencies", [])
            if isinstance(dependency, dict) and dependency.get("role") == "external-runtime"
        ]
        limitations: list[str] = []
        channel = manifest["artifact"]["channel"]
        if channel != "stable":
            limitations.append(f"{channel}: see per-gate evidence in the embedded manifest")
        if external:
            limitations.append("external runtime: " + ", ".join(external))
        if manifest["artifact"]["static_boundary"] == "literal-static":
            limitations.append("experimental CPU-only literal-static feasibility lane")
        rows.append(
            {
                "archive": archive_name,
                "backend": manifest["backend"]["name"],
                "channel": channel,
                "checksum": f"{archive_name}.sha256",
                "cpu_tiers": [
                    tier["name"] for tier in manifest.get("cpu", {}).get("compiled_tiers", [])
                ],
                "driver_boundary": manifest["backend"]["gpu_driver_boundary"],
                "host": manifest["host"],
                "id": artifact_id,
                "limitations": limitations,
                "provenance": f"{archive_name}.provenance.json",
                "sha256": by_name[archive_name]["sha256"],
                "size": by_name[archive_name]["size"],
                "sms": manifest.get("cuda", {}).get("compiled_sms", []),
                "static_boundary": manifest["artifact"]["static_boundary"],
            }
        )
    index = {
        "artifacts": rows,
        "prerelease": prerelease,
        "project_version": project_version,
        "release_tag": handoff.get("release_tag"),
        "retention": {
            "ci_artifacts_days": retention_days,
            "github_release": retention["github_release"],
        },
        "schema": SCHEMA,
        "source_sha": handoff.get("source_sha"),
        "version": version,
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# vllm.cpp {index['release_tag']} binary index",
        "",
        f"Source: `{index['source_sha']}`",
        f"Prerelease: `{str(prerelease).lower()}`",
        "",
        "| Artifact | Channel | Host ABI | Backend | CPU tiers / CUDA SMs | Boundary and limitations |",
        "|---|---|---|---|---|---|",
    ]
    # The Markdown is the GitHub release BODY as well as an asset. On the release
    # page a relative `./<name>` resolves against .../releases/tag/<tag> and
    # 404s, which is what every link of v0.0.3-vk.1 and v0.0.4-vk.1 did. With a
    # repository the links are the release's own download URLs; without one
    # (an offline index beside the archives) they stay relative.
    base = (
        f"https://github.com/{repository}/releases/download/{index['release_tag']}/"
        if repository is not None else "./"
    )
    for row in rows:
        host = row["host"]
        capabilities = ", ".join(row["cpu_tiers"] or row["sms"] or ["platform-native"])
        boundary = "; ".join([row["static_boundary"], row["driver_boundary"], *row["limitations"]])
        lines.append(
            f"| [{row['id']}]({base}{row['archive']}) ([sha256]({base}{row['checksum']})) "
            f"| {row['channel']} | {host['os']}/{host['arch']}/{host['abi']} {host['abi_version']} "
            f"| {row['backend']} | {capabilities} | {boundary} |"
        )
    lines.extend(
        (
            "",
            f"CI handoff artifacts are retained for {retention_days} days. Published GitHub release assets follow the {retention['github_release']} policy.",
            "",
        )
    )
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text("\n".join(lines), encoding="utf-8")
    return index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--retention-days", type=int, default=7)
    parser.add_argument("--repository", default=None,
                        help="owner/name; makes the Markdown links absolute download URLs")
    args = parser.parse_args()
    try:
        handoff = json.loads(args.handoff.read_text(encoding="utf-8"))
        generate_index(
            args.assets_dir,
            handoff,
            args.json_output,
            args.markdown_output,
            args.retention_days,
            args.repository,
        )
    except (OSError, json.JSONDecodeError, tarfile.TarError, zipfile.BadZipFile, ValueError) as exc:
        print(f"release index error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
