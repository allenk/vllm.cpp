#!/usr/bin/env python3
"""Mutation suite for scripts/check-container-matrix.py.

Every case mutates one guarantee the checker claims and asserts it fails FOR
THAT REASON. A suite that only proves the shipped files pass would go green
against a checker whose body had been deleted.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts/check-container-matrix.py"
MATRIX = ROOT / "release/container-matrix.json"
DOCKERFILE = ROOT / "docker/Dockerfile"


def load_module():
    spec = importlib.util.spec_from_file_location("check_container_matrix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = load_module()


def shipped_matrix() -> dict:
    return json.loads(MATRIX.read_text(encoding="utf-8"))


def errors_for(matrix: dict) -> list[str]:
    return checker.check_shape(matrix)


def dockerfile_errors(matrix: dict, text: str) -> list[str]:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "Dockerfile"
        path.write_text(text, encoding="utf-8")
        return checker.check_dockerfile(matrix, path)


def lane(matrix: dict, lane_id: str) -> dict:
    return next(entry for entry in matrix["lanes"] if entry["id"] == lane_id)


class ShippedRecordTests(unittest.TestCase):
    def test_the_shipped_matrix_passes_its_own_checker(self):
        matrix = shipped_matrix()
        self.assertEqual(errors_for(matrix), [])
        self.assertEqual(checker.check_dockerfile(matrix, DOCKERFILE), [])

    def test_every_shipped_lane_names_a_real_dockerfile_target(self):
        text = DOCKERFILE.read_text(encoding="utf-8")
        targets = set(checker.DOCKERFILE_TARGET.findall(text))
        for entry in shipped_matrix()["lanes"]:
            self.assertIn(entry["target"], targets)


class SchemaMutationTests(unittest.TestCase):
    def test_a_wrong_schema_is_rejected(self):
        matrix = shipped_matrix()
        matrix["schema"] = "vllm.cpp.container-matrix.v2"
        self.assertTrue(any("schema must be" in e for e in errors_for(matrix)))

    def test_a_renamed_package_is_rejected(self):
        matrix = shipped_matrix()
        matrix["package"] = "ghcr.io/someone-else/vllm.cpp"
        self.assertTrue(any("package must be" in e for e in errors_for(matrix)))

    def test_a_floating_base_tag_is_rejected(self):
        matrix = shipped_matrix()
        matrix["bases"]["runtime"] = "ubuntu:24.04"
        self.assertTrue(
            any("pinned by digest" in e for e in errors_for(matrix)),
            "an unpinned base makes a version tag unreproducible",
        )

    def test_a_truncated_digest_is_rejected(self):
        matrix = shipped_matrix()
        matrix["bases"]["runtime"] = "ubuntu:24.04@sha256:deadbeef"
        self.assertTrue(any("pinned by digest" in e for e in errors_for(matrix)))


class TagMutationTests(unittest.TestCase):
    def test_a_version_tag_without_the_lane_is_rejected(self):
        matrix = shipped_matrix()
        lane(matrix, "cuda")["version_tag"] = "{version}"
        self.assertTrue(
            any("must end in -cuda" in e for e in errors_for(matrix)),
            "the lane lives in the tag, not the package name",
        )

    def test_a_version_tag_that_does_not_interpolate_the_version_is_rejected(self):
        matrix = shipped_matrix()
        lane(matrix, "cpu")["version_tag"] = "release-cpu"
        self.assertTrue(any("interpolate" in e for e in errors_for(matrix)))

    def test_two_lanes_cannot_claim_the_same_moving_tag(self):
        matrix = shipped_matrix()
        lane(matrix, "cuda")["moving_tags"].append("latest")
        self.assertTrue(
            any("exactly one owner" in e for e in errors_for(matrix)),
            "a moving tag pushed by two lanes races on every release",
        )

    def test_the_bare_latest_must_follow_the_default_lane(self):
        matrix = shipped_matrix()
        matrix["default_lane"] = "cuda"
        self.assertTrue(
            any("owned by the default lane" in e for e in errors_for(matrix)),
            "pulling :latest on a box with no GPU must not get the cuda image",
        )

    def test_an_unknown_default_lane_is_rejected(self):
        matrix = shipped_matrix()
        matrix["default_lane"] = "rocm"
        self.assertTrue(any("is not a declared lane" in e for e in errors_for(matrix)))

    def test_a_lane_missing_its_own_moving_tag_is_rejected(self):
        matrix = shipped_matrix()
        lane(matrix, "vulkan")["moving_tags"] = ["nightly-vulkan"]
        self.assertTrue(any("latest-vulkan" in e for e in errors_for(matrix)))


class ArchitectureMutationTests(unittest.TestCase):
    def test_a_single_architecture_lane_is_rejected(self):
        matrix = shipped_matrix()
        entry = lane(matrix, "cuda")
        entry["architectures"] = entry["architectures"][:1]
        self.assertTrue(
            any("multi-arch" in e for e in errors_for(matrix)),
            "arm64 is first-class: GB10, Thor and Orin are all arm64",
        )

    def test_dropping_the_native_runner_is_rejected(self):
        matrix = shipped_matrix()
        lane(matrix, "cpu")["architectures"][1].pop("runner")
        self.assertTrue(any("native runner" in e for e in errors_for(matrix)))

    def test_runtime_evidence_must_be_a_boolean_not_a_string(self):
        matrix = shipped_matrix()
        lane(matrix, "cuda")["architectures"][0]["runtime_evidence"] = "pending"
        self.assertTrue(
            any("runtime_evidence" in e for e in errors_for(matrix)),
            "a build result is not a runtime result and cannot be fudged into one",
        )

    def test_an_unknown_platform_is_rejected(self):
        matrix = shipped_matrix()
        lane(matrix, "cpu")["architectures"][0]["platform"] = "linux/riscv64"
        self.assertTrue(any("platform must be one of" in e for e in errors_for(matrix)))


class BoundaryMutationTests(unittest.TestCase):
    def test_a_blocked_entry_without_a_reason_is_rejected(self):
        matrix = shipped_matrix()
        matrix["blocked"][0].pop("reason")
        self.assertTrue(
            any("needs a reason" in e for e in errors_for(matrix)),
            "an unexplained exclusion reads as pending work",
        )

    def test_a_not_containerizable_entry_without_a_reason_is_rejected(self):
        matrix = shipped_matrix()
        matrix["not_containerizable"][0].pop("reason")
        self.assertTrue(any("needs a reason" in e for e in errors_for(matrix)))

    def test_publishing_a_blocked_lane_is_rejected(self):
        matrix = shipped_matrix()
        matrix["lanes"].append(
            {
                "id": "rocm",
                "target": "rocm",
                "channel": "preview",
                "version_tag": "{version}-rocm",
                "moving_tags": ["latest-rocm"],
                "architectures": lane(matrix, "cpu")["architectures"],
            }
        )
        self.assertTrue(
            any("both a published lane and listed under blocked" in e for e in errors_for(matrix))
        )

    def test_mutable_version_tags_are_rejected(self):
        matrix = shipped_matrix()
        matrix["retention"]["version_tags"] = "pruned-after-90-days"
        self.assertTrue(
            any("maintainer-deletion-only" in e for e in errors_for(matrix)),
            "a version tag is immutable; garbage-collecting it breaks every pin",
        )

    def test_missing_retention_is_rejected(self):
        matrix = shipped_matrix()
        matrix["retention"].pop("untagged_digests_days")
        self.assertTrue(any("untagged_digests_days" in e for e in errors_for(matrix)))


class DockerfileMutationTests(unittest.TestCase):
    def test_a_lane_whose_target_does_not_exist_is_rejected(self):
        matrix = shipped_matrix()
        errors = dockerfile_errors(matrix, "FROM scratch AS cpu\nLABEL x=y\n")
        self.assertTrue(any("does not define" in e for e in errors))

    def test_an_unpinned_from_is_rejected(self):
        matrix = shipped_matrix()
        errors = dockerfile_errors(matrix, "FROM ubuntu:24.04 AS cpu\n")
        self.assertTrue(any("unpinned FROM" in e for e in errors))

    def test_a_base_the_matrix_does_not_record_is_rejected(self):
        matrix = shipped_matrix()
        text = DOCKERFILE.read_text(encoding="utf-8").replace(
            matrix["bases"]["runtime"], "debian:12@sha256:" + "0" * 64
        )
        errors = dockerfile_errors(matrix, text)
        self.assertTrue(
            any("does not record" in e for e in errors),
            "swapping the base in one file only is precisely the drift this catches",
        )

    def test_a_missing_required_label_is_rejected(self):
        matrix = shipped_matrix()
        text = DOCKERFILE.read_text(encoding="utf-8").replace("io.vllm-cpp.lane", "io.x.lane")
        errors = dockerfile_errors(matrix, text)
        self.assertTrue(any("io.vllm-cpp.lane" in e for e in errors))

    def test_installing_the_gpu_driver_is_rejected(self):
        matrix = shipped_matrix()
        text = DOCKERFILE.read_text(encoding="utf-8") + "\nRUN apt-get install -y cuda-drivers\n"
        errors = dockerfile_errors(matrix, text)
        self.assertTrue(
            any("never bundled" in e for e in errors),
            "the driver is the host's; bundling one is a support claim we cannot honour",
        )

    def test_copying_the_driver_out_of_the_builder_is_rejected(self):
        matrix = shipped_matrix()
        text = DOCKERFILE.read_text(encoding="utf-8") + (
            "\nCOPY --from=build-cuda /usr/local/cuda/lib64/libcuda.so.1 /opt/vllm/lib/\n"
        )
        errors = dockerfile_errors(matrix, text)
        self.assertTrue(any("never bundled" in e for e in errors))

    def test_documenting_the_driver_boundary_is_NOT_rejected(self):
        """The inverse of the two cases above, and the reason they are narrow.

        The image must be able to SAY that libcuda.so.1 comes from the host, in a
        comment and in its min-driver label. A checker that fired on the prose
        would push the boundary out of the file that has to honour it.
        """
        matrix = shipped_matrix()
        text = DOCKERFILE.read_text(encoding="utf-8") + (
            '\n# libcuda.so.1 is the host driver, never bundled\n'
            'LABEL io.vllm-cpp.note="needs host libcuda.so.1"\n'
        )
        errors = dockerfile_errors(matrix, text)
        self.assertEqual([e for e in errors if "never bundled" in e], [])


class AttributionMutationTests(unittest.TestCase):
    """image.source must name THIS fork, and must not erase the origin.

    v0.0.3-vk.1 was one tag away from shipping three images whose
    image.source said mudler/vllm.cpp while the image.revision beside it was
    a commit that exists only in allenk/vllm.cpp -- a pair no reader can
    reconcile, and the label GHCR follows when it decides which repository,
    and which README, a package belongs to. The label was present the whole
    time; only its VALUE was wrong, which is exactly what a presence check
    cannot see.
    """

    def test_an_upstream_source_label_is_rejected(self):
        matrix = shipped_matrix()
        text = DOCKERFILE.read_text(encoding="utf-8").replace(
            checker.FORK_SOURCE, checker.UPSTREAM_SOURCE, 1
        )
        errors = dockerfile_errors(matrix, text)
        self.assertTrue(
            any("must not appear to ship from" in e for e in errors),
            "one lane pointing upstream has to be refused",
        )

    def test_every_lane_carries_its_own_source_label(self):
        matrix = shipped_matrix()
        text = DOCKERFILE.read_text(encoding="utf-8").replace(
            '      org.opencontainers.image.source="%s" %s%s'
            % (checker.FORK_SOURCE, chr(92), chr(10)),
            "",
            1,
        )
        errors = dockerfile_errors(matrix, text)
        self.assertTrue(any("declares 3 lanes" in e for e in errors))

    def test_correcting_the_source_may_not_erase_the_origin(self):
        matrix = shipped_matrix()
        text = DOCKERFILE.read_text(encoding="utf-8").replace(
            '      io.vllm-cpp.upstream="%s" %s%s'
            % (checker.UPSTREAM_SOURCE, chr(92), chr(10)),
            "",
        )
        errors = dockerfile_errors(matrix, text)
        self.assertTrue(
            any("never erases the origin" in e for e in errors),
            "this fork is derived from upstream and the images must say so",
        )

    def test_the_two_constants_are_not_the_same_repository(self):
        self.assertNotEqual(checker.FORK_SOURCE, checker.UPSTREAM_SOURCE)
        self.assertIn("allenk", checker.FORK_SOURCE)
        self.assertIn("mudler", checker.UPSTREAM_SOURCE)


class InstructionParserTests(unittest.TestCase):
    def test_line_continuations_are_joined_into_one_instruction(self):
        parsed = checker.dockerfile_instructions("RUN apt-get install \\\n      ffmpeg\n")
        self.assertEqual(parsed, [("RUN", "apt-get install ffmpeg")])

    def test_comments_are_dropped(self):
        parsed = checker.dockerfile_instructions("# RUN rm -rf /\nFROM scratch\n")
        self.assertEqual(parsed, [("FROM", "scratch")])

    def test_a_driver_hidden_across_a_continuation_is_still_seen(self):
        parsed = checker.dockerfile_instructions("RUN apt-get install -y \\\n  cuda-drivers\n")
        self.assertIn("cuda-drivers", parsed[0][1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
