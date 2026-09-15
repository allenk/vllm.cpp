# How this fork is branched, released, and contributed back

This file exists only in this fork. `upstream/main` has no `FORK.md`, so nothing
here can ever conflict on a merge from upstream — which is the same reason the
fork's identity lives in `release/release-version.json` rather than in a file
upstream also edits.

It is the standing rule for `allenk/vllm.cpp`. Follow it rather than re-deriving
it.

---

## 1. The branches and what each one is for

```
main            OURS, and the DEFAULT branch. The release branch.
                Everything that ships has landed here. A visitor to the
                repository sees this README, this file tree, and this work.

upstream-sync   A pure mirror of mudler/vllm.cpp. ZERO local commits, ever.
                Only ever fast-forwarded. It is the base that `row/<ID>`
                branches are cut from.

port/vulkan     Implementation branches, one per platform or work line.
compat/windows  They are where the work actually happens; they are merged or
...             cherry-picked into `main` when they are ready. They are not
                release surfaces and nothing is published from them.
```

Flow, in one line:

```
upstream-sync ──(fetch, fast-forward only)── mudler/vllm.cpp
port/vulkan, compat/*, ... ──(merge or cherry-pick when ready)──> main ──(tag)──> release
upstream-sync ──(cut row/<ID>, cherry-pick the one fix)──> PR to mudler
```

### Why `main` is ours and not the upstream mirror

Because the default branch is what the world sees. GitHub Releases are attached
to **tags** and the Releases page is repository-level, so artifacts are reachable
whatever branch the tag points into — but the landing page renders the DEFAULT
branch's README. If `main` mirrored upstream, a visitor would read upstream's
README and see no sign that this fork has the Vulkan backend, the Windows lanes,
the Triton CPU provider or ten execution providers.

The mirror still has to exist, for cutting clean PR branches. It just does not
get to be the front door. That is the whole reason it is a separate branch
instead of `main`.

---

## 2. Releasing

Releases are cut from `main`, by pushing a tag. Two facts that are easy to get
wrong, both verified against this repository's own pipeline:

```
push a branch     updates the branch. Publishes NOTHING.
                  release.yml triggers only on tags: ['v*'] and workflow_dispatch
workflow_dispatch a DRY RUN. Builds and validates every lane, publish=false,
                  creates no tag and no release. Use it before every tag.
push a tag        the release. Triggers release.yml AND containers.yml, so it
                  publishes a GitHub Release AND a GHCR image in one step.
```

`release_pipeline.py` enforces `tag == "v" + version`, and that the version's
numeric core equals `project_version` (which must equal `CMakeLists.txt`). The
fork identity therefore rides in the version's **pre-release field**:

```
project_version   0.0.3          tracks CMakeLists.txt, unchanged
version           0.0.3-vk.1     ours
tag               v0.0.3-vk.1    == "v" + version, so the validator is untouched
assets            vllm.cpp-0.0.3-vk.1-<platform>.<ext>
images            ghcr.io/allenk/vllm.cpp-vk:0.0.3-vk.1-<lane>
```

One field makes the tag, every asset filename and every image tag impossible for
upstream to emit, with no change to any file upstream also maintains.

**Always dry-run before tagging.** All artifacts are `required: true`, so one red
lane means no release at all — and a tag, once pushed, is public and cannot be
taken back. The first dry run on this pipeline had eight of eleven lanes red.

---

## 3. Contributing back to mudler/vllm.cpp

PR work is deliberately separate from the work above and does not block it.

```
1  cut row/<ID> from upstream-sync, NOT from main or an implementation branch
2  cherry-pick the ONE fix. A branch cut from port/vulkan drags 26 commits of
   context into a PR that wants one
3  the commit message MUST carry the protocol paragraph and trailers, because
   scripts/check-commit-trailers.py gates the landed range and a squash-merge
   composes its body from the branch's messages:

       FOLLOWING_AGENTS_PROTOCOL

       Following-Agents-Protocol: true
       AI-Assisted: true
       Assisted-by: Claude:Opus-5 [Claude Code]

   A hand-written `Co-authored-by:` for an AI is REJECTED by that gate; only the
   forge's own domain is exempt. This happens to agree with this project's own
   rule of never adding a Co-Authored-By trailer.
4  squash-merge. check-role-discipline.py wants the subject to name the row
   branch or the PR as `(#N)`, and a squash writes `(#N)` for free
```

### What must NEVER be sent upstream

Anything that names this fork or its hardware:

```
release/release-version.json     the -vk version
.github/workflows/containers.yml REGISTRY_PACKAGE -> ghcr.io/allenk/...
.github/workflows/release.yml    the fork guard, the three Windows jobs
release/release-matrix.json      windows-x86_64-msvc-cuda
scripts/check-triton-aot-multiarch.py   the sm_120a row
this file
```

---

## 4. The failure mode this fork keeps hitting

Four separate times, the same shape:

```
this fork ADDS a capability      and the gate that COUNTS it stays at upstream's number
  sm_120a AOT tree               check-triton-aot-multiarch.py knew six trees, disk had seven
  two decode-GEMV shaders        test_vulkan_backend.cpp asserted 36 modules, disk had 38
  the Windows CUDA lane          release-matrix.json never declared the id it downloads
  the cuda_windows job itself    PRIMARY_ARTIFACT_FORMATS never learned it either
```

Upstream does not have these capabilities, so upstream's counters never grow on
their own, and every one of them failed CLOSED — loudly, at release time, on a
lane nobody runs locally.

⇒ **When you add a tree, a shader, a lane or an artifact, grep for the number
that counts it in the same commit.** Better: make the gate compute the count
rather than hardcode it, as `check-triton-aot-multiarch.py` now does.

---

## 5. Operational traps, each paid for once

```
gh resolves to UPSTREAM        `gh workflow run` in this tree dispatched to
                               mudler/vllm.cpp; only a 403 stopped it. Pass
                               --repo allenk/vllm.cpp on every gh command that
                               has a side effect. git is safe: remotes are named.
container: is Linux-only       "Container operations are only supported on Linux
                               runners". Linux CUDA lanes get nvcc from
                               `container: nvidia/cuda:...-devel`; Windows cannot
                               and must install a toolkit instead.
dumpbin needs the dev prompt   It is an MSVC tool, on PATH only inside a Developer
                               Command Prompt. It resolves on a developer box and
                               not on a bare runner.
two working trees              The Windows checkout is origin; the tree that BUILDS
                               is a WSL ext4 clone. Confirm which one you are
                               editing before touching src/.
comments outlive code          `RefuseDflash2PathWalk` was described in a comment
                               long after the function was deleted. Check the code.
```
