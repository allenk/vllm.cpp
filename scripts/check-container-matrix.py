#!/usr/bin/env python3
"""Gate release/container-matrix.json against docker/Dockerfile.

ENG-RELEASE-CONTAINERS ships three lanes from one GHCR package with the lane in
the tag. Two files have to agree about that, and nothing else in the tree makes
them: the matrix is what the publish workflow reads, the Dockerfile is what
actually builds. A lane that exists in one and not the other, or a base image
pinned by digest in one and floating in the other, is exactly the drift this
checker exists to refuse.

It is deliberately a CROSS-file checker rather than a schema validator. A schema
would prove the matrix is well-formed while the Dockerfile built something else
entirely -- consistency with itself, which proves nothing.

Usage: scripts/check-container-matrix.py [--matrix PATH] [--dockerfile PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATRIX = ROOT / "release/container-matrix.json"
DOCKERFILE = ROOT / "docker/Dockerfile"

SCHEMA = "vllm.cpp.container-matrix.v1"
PACKAGE = "ghcr.io/mudler/vllm.cpp"
# FORK: where THIS source lives. The OCI spec reads image.source as "URL to get
# source code for building the image", so on a fork it must name the fork -- the
# revision label beside it is a commit that exists only here, and a source label
# pointing upstream makes that pair internally impossible. It is also what GHCR
# follows to decide which repository, and which README, a package belongs to.
#
# The upstream project is credited by io.vllm-cpp.upstream instead, so the
# derivation stays visible rather than being erased by the correction.
#
# Asserted by VALUE, not by presence: the label was present and wrong for the
# whole of v0.0.3-vk.1's development, and a checker that only asks whether a
# name appears anywhere in the file cannot see that.
FORK_SOURCE = "https://github.com/allenk/vllm.cpp"
UPSTREAM_SOURCE = "https://github.com/mudler/vllm.cpp"
CHANNELS = frozenset({"stable", "preview"})
PLATFORMS = frozenset({"linux/amd64", "linux/arm64"})

DIGEST_PINNED = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
DOCKERFILE_ARG = re.compile(r"^ARG\s+(?P<name>[A-Z0-9_]+)=(?P<value>\S+)\s*$", re.MULTILINE)
DOCKERFILE_TARGET = re.compile(r"^FROM\s+\S+\s+AS\s+(?P<name>[A-Za-z0-9_.-]+)\s*$", re.MULTILINE)
LABEL_SOURCE = re.compile("org[.]opencontainers[.]image[.]source=\"([^\"]*)\"")
LABEL_UPSTREAM = re.compile("io[.]vllm-cpp[.]upstream=\"([^\"]*)\"")
DOCKERFILE_FROM = re.compile(r"^FROM\s+(?P<image>\S+)", re.MULTILINE)

# Every lane's runtime stage must carry these, because an image with no lane
# label is unattributable once it is pulled and its tag has moved on.
REQUIRED_LABELS = (
    "org.opencontainers.image.source",
    "org.opencontainers.image.revision",
    "org.opencontainers.image.version",
    "io.vllm-cpp.lane",
    "io.vllm-cpp.channel",
)


def dockerfile_instructions(text: str) -> list[tuple[str, str]]:
    """Yield (INSTRUCTION, body) pairs with comments dropped and continuations joined."""
    instructions: list[tuple[str, str]] = []
    pending: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        continued = line.endswith("\\")
        pending.append(line[:-1] if continued else line)
        if continued:
            continue
        joined = " ".join(part.strip() for part in pending)
        pending = []
        keyword, _, body = joined.partition(" ")
        instructions.append((keyword.upper(), body))
    if pending:
        joined = " ".join(part.strip() for part in pending)
        keyword, _, body = joined.partition(" ")
        instructions.append((keyword.upper(), body))
    return instructions


def load_matrix(path: Path) -> tuple[dict, list[str]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except FileNotFoundError:
        return {}, [f"{path} does not exist"]
    except json.JSONDecodeError as error:
        return {}, [f"{path} is not valid JSON: {error}"]


def check_shape(matrix: dict) -> list[str]:
    errors: list[str] = []

    if matrix.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}, found {matrix.get('schema')!r}")
    if matrix.get("package") != PACKAGE:
        errors.append(f"package must be {PACKAGE!r}, found {matrix.get('package')!r}")

    bases = matrix.get("bases")
    if not isinstance(bases, dict) or not bases:
        errors.append("bases must be a non-empty object of pinned base images")
    else:
        for name, reference in sorted(bases.items()):
            if not isinstance(reference, str) or not DIGEST_PINNED.match(reference):
                errors.append(
                    f"base {name!r} must be pinned by digest (name@sha256:<64 hex>), "
                    f"found {reference!r}; a floating tag makes the image unreproducible"
                )

    lanes = matrix.get("lanes")
    if not isinstance(lanes, list) or not lanes:
        errors.append("lanes must be a non-empty list")
        return errors

    seen_ids: set[str] = set()
    moving_owners: dict[str, str] = {}
    for lane in lanes:
        if not isinstance(lane, dict):
            errors.append(f"lane entries must be objects, found {lane!r}")
            continue
        lane_id = lane.get("id")
        if not isinstance(lane_id, str) or not lane_id:
            errors.append(f"lane is missing a string id: {lane!r}")
            continue
        if lane_id in seen_ids:
            errors.append(f"lane {lane_id!r} is declared more than once")
        seen_ids.add(lane_id)

        if lane.get("channel") not in CHANNELS:
            errors.append(
                f"lane {lane_id!r} channel must be one of {sorted(CHANNELS)}, "
                f"found {lane.get('channel')!r}"
            )
        if lane.get("target") != lane_id:
            errors.append(
                f"lane {lane_id!r} must build the Dockerfile target of the same name, "
                f"found target {lane.get('target')!r}"
            )

        version_tag = lane.get("version_tag")
        if not isinstance(version_tag, str) or "{version}" not in version_tag:
            errors.append(
                f"lane {lane_id!r} version_tag must interpolate {{version}}, "
                f"found {version_tag!r}"
            )
        elif not version_tag.endswith(f"-{lane_id}"):
            errors.append(
                f"lane {lane_id!r} version_tag must end in -{lane_id} so the lane is "
                f"in the tag, found {version_tag!r}"
            )

        moving = lane.get("moving_tags")
        if not isinstance(moving, list) or not moving:
            errors.append(f"lane {lane_id!r} must declare at least one moving tag")
        else:
            if f"latest-{lane_id}" not in moving:
                errors.append(
                    f"lane {lane_id!r} must own the moving tag latest-{lane_id}"
                )
            for tag in moving:
                if tag in moving_owners:
                    errors.append(
                        f"moving tag {tag!r} is claimed by both {moving_owners[tag]!r} "
                        f"and {lane_id!r}; a moving tag has exactly one owner"
                    )
                else:
                    moving_owners[tag] = lane_id

        architectures = lane.get("architectures")
        if not isinstance(architectures, list) or len(architectures) < 2:
            errors.append(
                f"lane {lane_id!r} must publish a multi-arch manifest; declare both "
                f"{sorted(PLATFORMS)}"
            )
            continue
        platforms = set()
        for entry in architectures:
            if not isinstance(entry, dict):
                errors.append(f"lane {lane_id!r} architecture entries must be objects")
                continue
            platform = entry.get("platform")
            if platform not in PLATFORMS:
                errors.append(
                    f"lane {lane_id!r} platform must be one of {sorted(PLATFORMS)}, "
                    f"found {platform!r}"
                )
            platforms.add(platform)
            if not isinstance(entry.get("runner"), str) or not entry.get("runner"):
                errors.append(
                    f"lane {lane_id!r} platform {platform!r} must name its native runner; "
                    "arm64 is never cross-built from an amd64 runner here"
                )
            if not isinstance(entry.get("runtime_evidence"), bool):
                errors.append(
                    f"lane {lane_id!r} platform {platform!r} must state runtime_evidence "
                    "as a boolean; a build result is not a runtime result"
                )
        if platforms != PLATFORMS:
            errors.append(
                f"lane {lane_id!r} must declare exactly {sorted(PLATFORMS)}, "
                f"found {sorted(p for p in platforms if p)}"
            )

    default_lane = matrix.get("default_lane")
    if default_lane not in seen_ids:
        errors.append(
            f"default_lane {default_lane!r} is not a declared lane; the bare :latest "
            "tag has to resolve to one"
        )
    elif moving_owners.get("latest") != default_lane:
        errors.append(
            f"the bare :latest tag must be owned by the default lane {default_lane!r}, "
            f"found owner {moving_owners.get('latest')!r}"
        )

    for key in ("blocked", "not_containerizable"):
        entries = matrix.get(key)
        if not isinstance(entries, list):
            errors.append(f"{key} must be a list, so the boundary is recorded not implied")
            continue
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("id"):
                errors.append(f"{key} entries need an id: {entry!r}")
                continue
            if not entry.get("reason"):
                errors.append(
                    f"{key} entry {entry['id']!r} needs a reason; an unexplained "
                    "exclusion reads as pending work"
                )
            if entry["id"] in seen_ids:
                errors.append(
                    f"{entry['id']!r} is both a published lane and listed under {key}"
                )

    retention = matrix.get("retention")
    if not isinstance(retention, dict):
        errors.append("retention must be an object")
    else:
        for key in ("version_tags", "moving_tags", "untagged_digests_days"):
            if key not in retention:
                errors.append(f"retention must state {key}")
        if retention.get("version_tags") != "maintainer-deletion-only":
            errors.append(
                "retention.version_tags must be maintainer-deletion-only: a version "
                "tag is immutable and is never garbage-collected"
            )

    return errors


def check_dockerfile(matrix: dict, dockerfile: Path) -> list[str]:
    errors: list[str] = []
    try:
        text = dockerfile.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [f"{dockerfile} does not exist"]

    targets = set(DOCKERFILE_TARGET.findall(text))
    args = {match.group("name"): match.group("value") for match in DOCKERFILE_ARG.finditer(text)}

    for lane in matrix.get("lanes", []):
        if not isinstance(lane, dict):
            continue
        target = lane.get("target")
        if target and target not in targets:
            errors.append(
                f"lane {lane.get('id')!r} names Dockerfile target {target!r}, which "
                f"{dockerfile} does not define"
            )

    declared_bases = set(matrix.get("bases", {}).values()) if isinstance(matrix.get("bases"), dict) else set()
    arg_bases = {value for name, value in args.items() if "BASE" in name}
    for reference in sorted(arg_bases):
        if reference not in declared_bases:
            errors.append(
                f"{dockerfile} builds from {reference!r}, which release/container-matrix.json "
                "does not record; the matrix is what the publish workflow reads"
            )
    for reference in sorted(declared_bases - arg_bases):
        errors.append(
            f"release/container-matrix.json records base {reference!r} that "
            f"{dockerfile} never uses"
        )

    for image in DOCKERFILE_FROM.findall(text):
        if image.startswith("${") or image in targets:
            continue
        if not DIGEST_PINNED.match(image):
            errors.append(
                f"{dockerfile} has an unpinned FROM {image!r}; every base is pinned by "
                "digest so a rebuild of a version tag cannot silently change base"
            )

    for label in REQUIRED_LABELS:
        if label not in text:
            errors.append(f"{dockerfile} never sets the required label {label}")

    lanes = len(matrix.get("lanes") or ())
    sources = LABEL_SOURCE.findall(text)
    if len(sources) != lanes:
        errors.append(
            f"{dockerfile} sets org.opencontainers.image.source {len(sources)} times "
            f"but the matrix declares {lanes} lanes; every lane's runtime stage "
            "carries its own"
        )
    for value in sources:
        if value != FORK_SOURCE:
            errors.append(
                f"{dockerfile} sets org.opencontainers.image.source={value!r}; this "
                f"fork ships from {FORK_SOURCE!r} and must not appear to ship from "
                "anywhere else. Credit the origin with io.vllm-cpp.upstream"
            )
    upstreams = LABEL_UPSTREAM.findall(text)
    if len(upstreams) != lanes or any(v != UPSTREAM_SOURCE for v in upstreams):
        errors.append(
            f"{dockerfile} must record io.vllm-cpp.upstream={UPSTREAM_SOURCE!r} on "
            "every lane, so correcting the source label never erases the origin"
        )

    # The driver is the host's. An image that installs one is claiming support it
    # cannot honour, and the boundary is the single most repeated line in the spec.
    #
    # Only RUN/COPY/ADD are scanned, because only those can put a driver in the
    # image. A comment explaining that libcuda.so.1 comes from the host, and a
    # LABEL declaring which host driver is required, are the image DOCUMENTING
    # the boundary -- the opposite of violating it. A checker that cannot tell an
    # action from metadata would forbid saying the true thing.
    for instruction, body in dockerfile_instructions(text):
        if instruction not in {"RUN", "COPY", "ADD"}:
            continue
        for forbidden in ("cuda-drivers", "nvidia-driver", "libcuda.so"):
            if forbidden in body:
                errors.append(
                    f"{dockerfile} {instruction} installs or copies {forbidden!r}: the "
                    "GPU driver stays on the host and is never bundled"
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--dockerfile", type=Path, default=DOCKERFILE)
    args = parser.parse_args()

    matrix, errors = load_matrix(args.matrix)
    if not errors:
        errors = check_shape(matrix) + check_dockerfile(matrix, args.dockerfile)

    if errors:
        print("ERROR: the container matrix and the Dockerfile do not agree:")
        for error in errors:
            print(f"  - {error}")
        return 1

    lanes = ", ".join(sorted(lane["id"] for lane in matrix["lanes"]))
    print(f"container matrix OK: {matrix['package']} lanes {lanes}, bases digest-pinned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
