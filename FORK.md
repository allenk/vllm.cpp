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

### ⚠ The dry run covers ONE of the two workflows a tag fires

A `workflow_dispatch` on `release` proves `release.yml` and says nothing at all
about `containers.yml`, which the same tag also fires. Reading the box above and
stopping there is the trap: it names both workflows and teaches you to rehearse
only the first.

**`containers.yml` cannot be rehearsed the same way.** Its dispatch stops after
`verify` by design, so it can prove a lane builds and can never prove `publish`,
`manifest`, `attest` or `promote`. Those four are gated on every publish lane
being green, so a single red lane skips them silently — and on 2026-09-23 that
was the standing state: **five consecutive runs red, and `manifest`, `attest`
and `promote` had never executed on this fork at all.** A tag would have been
their debut.

What rehearses it instead is the **nightly cron on `main`** (`0 4 * * *`), which
runs the whole chain including the registry writes. So before tagging:

```
1. dispatch `release`        rehearses release.yml           (every lane green)
2. wait for a nightly        rehearses containers.yml        (manifest/attest/
   `containers` run on main                                   promote EXECUTED,
                                                              not skipped)
3. only then push the tag
```

⚠ And note **why the nightly is the thing to wait for rather than a push**:
`containers.yml`'s push trigger is deliberately narrow — `docker/**`, the
container matrix, four named scripts and its own file — because `main` takes
dozens of pushes a day. A fix elsewhere, `scripts/release_manifest.py` included,
does NOT rebuild containers. The cron is what closes that window, and its own
comment says so: *"a main image is never more than a day behind the tree."* That
is the design working; widening the filter buys hours at a standing cost.

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
cmake/TritonAOT*Test.cmake       the seven-tree counts
tests/scripts/test_check_triton_aot_multiarch.py   the sm_120a fixture
.github/workflows/ci.yml         the upstream-only guard on audit-live-rows
release/release-version.json + the eight files that spell 0.0.3-vk.1
scripts/env-doc-allowlist.txt    our 42 Vulkan/Triton-CPU tuning knobs
this file
```

---

## 3b. Merging upstream: three outcomes, and how to tell them apart

Measured on the first big merge: 248 upstream commits, three conflicts.

```
upstream made the SAME fix     take upstream wholesale, drop ours. TWO of the
                               three were this. Ours was not wasted -- it shipped
                               while upstream had none -- it is merely redundant
                               now, and keeping it would be a delta in a shared
                               file for nothing.
upstream made a BETTER fix     also take upstream. speech_engine.h: we deleted
                               the copy pair, upstream deleted copy AND move,
                               which is the more consistent version because
                               std::mutex is not movable either.
upstream UNDID something only  the only case that needs real work: take
WE need                        upstream's change and re-apply ours on top of it.
```

The third is the dangerous one, because it looks like a clean take.

`src/vt/cuda/cuda_qwen4_exp.cu`: upstream rewrote the kernel (one block per group
instead of one thread, f32 with a warp-shuffle tree instead of a serial double)
AND renamed a parameter back to `hyper`. That name is a Windows build break --
CCCL, reached through `<cub/cub.cuh>`, defines `hyper` as a macro expanding to
`__int64`, so any `.cu` including both cub and `vt/ops.h` fails with
"expected a )". Upstream does not build on Windows and cannot see it.

Resolved as upstream's algorithm plus our rename: 20 bare `hyper` identifiers in
code became `hyper_state`, and prose comments about the hyper-connection concept
were left alone.

> **Before taking a conflicted hunk wholesale, ask what OUR side of it was FOR.**
> If the answer is a platform upstream does not build, then upstream cannot have
> preserved it, and "theirs" silently reintroduces the bug it fixed.

## 4. The failure mode this fork keeps hitting

Five separate times, the same shape:

```
this fork ADDS a capability      and the gate that COUNTS it stays at upstream's number
  sm_120a AOT tree               check-triton-aot-multiarch.py knew six trees, disk had seven
  two decode-GEMV shaders        test_vulkan_backend.cpp asserted 36 modules, disk had 38
  the Windows CUDA lane          release-matrix.json never declared the id it downloads
  the cuda_windows job itself    PRIMARY_ARTIFACT_FORMATS never learned it either
  the sm_120 dispatch tree       upstream's NEW test asserted six trees, header had seven
```

The fifth one arrived from the opposite direction and is worth separating: we
did not forget a counter, the 248-commit merge DELIVERED one. Upstream wrote a
dispatch test encoding its own six-tree table, and on this tree it did not even
compile -- and three of its assertions were wrong rather than merely mis-sized,
including one that would have asserted the AOT fast path is OFF on the only card
this project develops on.

=> **Every merge from upstream can import a gate that counts something this fork
has more of.** A merge that compiles is not evidence of this; that one happened
not to.

Upstream does not have these capabilities, so upstream's counters never grow on
their own, and every one of them failed CLOSED — loudly, at release time, on a
lane nobody runs locally.

⇒ **When you add a tree, a shader, a lane or an artifact, grep for the number
that counts it in the same commit.** Better: make the gate compute the count
rather than hardcode it, as `check-triton-aot-multiarch.py` now does.

---

## 5. Operational traps, each paid for once

```
a NEW lane has never run       Before adding a release lane, check whether that
                               COMBINATION has ever been built anywhere. ci.yml
                               has two Windows jobs and neither builds CUDA, and
                               cuda-fat-build is ubuntu-latest, so Windows x CUDA
                               was new the day the lane was. "Windows passed" and
                               "CUDA passed" do not compose. Its five straight
                               reds were a first look, not a regression.
a dry run is HALF a rehearsal   `workflow_dispatch` on `release` says nothing
                               about `containers.yml`, which the same tag fires.
                               See the box in §2. Found the hard way: containers
                               had been red for five runs and three of its stages
                               had never executed.
two tables, one updated        The most expensive shape of red on this pipeline.
                               The sm_120a AOT tree was vendored and
                               `check-triton-aot-multiarch.py` was taught about
                               it; `release_manifest.py` was not. The build audit
                               then printed "7 exact trees and namespaces OK" and
                               the archive validator refused the same binary for
                               "fabricates unavailable AOT namespace for sm_120a"
                               — two hours in, because everything passes until
                               the last gate. If a fact lives in two files, write
                               the test that computes their agreement.
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
