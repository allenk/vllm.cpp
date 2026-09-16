cmake_minimum_required(VERSION 3.24)

# BUILD-TRITON-DEFAULT-ON (#219) — the resolved default of VLLM_CPP_TRITON.
#
# `vt_triton_aot_computed_default` is the whole rule, kept pure so this matrix
# runs under `cmake -P` with no compiler, no CUDA and no GPU (exactly like
# TritonAOTMultiArchTest.cmake). The configure-level cells that this cannot
# reach — a real CUDA configure resolving the option, and the fat build still
# configuring clean — are proven on the gate host and recorded in the row's
# spec.
#
# Scratch fixtures live under a unique temporary root OUTSIDE the checkout:
# `cmake -P` sets CMAKE_CURRENT_BINARY_DIR to the invoker's cwd, which under CI
# is the repository root. `_expect` also RECORDS failures instead of raising
# them, so the run always reaches its own cleanup and always reports every
# failing cell rather than only the first one; the single FATAL_ERROR comes
# after the fixtures are gone.

get_filename_component(_here "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
get_filename_component(_root "${_here}/.." ABSOLUTE)
include("${_here}/TritonAOTMultiArch.cmake")

set(_vendored "${_root}/src/vt/cuda/triton_aot_vendored")
if(DEFINED ENV{TMPDIR})
  set(_tmp_root "$ENV{TMPDIR}")
else()
  set(_tmp_root "/tmp")
endif()
string(RANDOM LENGTH 10 ALPHABET "abcdefghijklmnopqrstuvwxyz0123456789" _tag)
set(_scratch "${_tmp_root}/vllm-triton-aot-default-test-${_tag}")
file(REMOVE_RECURSE "${_scratch}")

set_property(GLOBAL PROPERTY VT_TRITON_DEFAULT_TEST_FAILURES "")

function(_fail MESSAGE)
  # A recorded message can quote a decline reason, and that reason JOINs several
  # tree defects with "; ". CMake lists are semicolon-separated, so a literal
  # semicolon would split one failure into several entries and inflate the
  # count; flatten it before appending.
  string(REPLACE ";" "," _message "${MESSAGE}")
  set_property(GLOBAL APPEND PROPERTY VT_TRITON_DEFAULT_TEST_FAILURES "${_message}")
endfunction()

function(_expect CASE EXPECTED_DEFAULT EXPECTED_REASON_SUBSTRING
         CUDA_ENABLED VENDORED_DIR SOURCE_ROOT REGEN)
  vt_triton_aot_computed_default(_default _reason
    "${CUDA_ENABLED}" "${VENDORED_DIR}" "${SOURCE_ROOT}" "${REGEN}")
  if(NOT _default STREQUAL EXPECTED_DEFAULT)
    _fail("${CASE}: expected default ${EXPECTED_DEFAULT}, got '${_default}'")
    return()
  endif()
  if(EXPECTED_REASON_SUBSTRING STREQUAL "")
    if(NOT _reason STREQUAL "")
      _fail("${CASE}: an enabling cell must carry no decline reason, got '${_reason}'")
    endif()
    return()
  endif()
  if(_reason STREQUAL "")
    _fail("${CASE}: a declining cell must name the condition that declined it")
    return()
  endif()
  string(FIND "${_reason}" "${EXPECTED_REASON_SUBSTRING}" _pos)
  if(_pos LESS 0)
    _fail("${CASE}: decline reason must mention "
      "'${EXPECTED_REASON_SUBSTRING}', got '${_reason}'")
  endif()
endfunction()

# 1. The vendored artifacts are CUDA cubins. Every non-CUDA backend (CPU,
#    Metal/MSL, Vulkan, ROCm) resolves OFF and must keep configuring clean.
_expect("cuda-off" OFF "VLLM_CPP_CUDA is OFF" OFF "${_vendored}" "${_root}" OFF)

# 2. A CUDA build with the vendored trees present is the case this row exists
#    for: the artifacts are pre-generated cubins embedded in plain C, so
#    consuming them needs a C compiler and nothing else. This cell also runs the
#    full precondition against the SHIPPED trees, so an inconsistent vendored
#    tree in the repository fails here rather than in a CUDA configure.
_expect("cuda-vendored-present" ON "" ON "${_vendored}" "${_root}" OFF)

# 3. The arch COUNT is deliberately not a condition. The builder path embeds
#    every vendored tree (TritonAOT.cmake `_triton_aot_arch_names`) and
#    dispatches by exact SM at runtime, which is what the shipped ten-SM
#    release archive and the cuda-fat-gencode CI job already build with
#    -DVLLM_CPP_TRITON=ON. The default therefore cannot depend on
#    VLLM_CPP_CUDA_ARCHITECTURES, and the function takes no such argument;
#    this cell pins that the single-arch and fat answers are the same object.
_expect("cuda-fat-is-the-same-answer" ON "" ON "${_vendored}" "${_root}" OFF)

# 4. Maintainer regeneration pins ONE arch tree and runs the Python toolchain.
#    It is an explicit act, so the default never selects it for you.
_expect("regen" OFF "VLLM_CPP_TRITON_REGEN" ON "${_vendored}" "${_root}" ON)

# 5. A tree that is not there cannot be consumed: the builder path would
#    FATAL_ERROR in `_triton_aot_arch_names`. The default declines instead.
file(MAKE_DIRECTORY "${_scratch}/empty")
_expect("no-vendored-tree" OFF "sm_121a" ON "${_scratch}/empty" "${_root}" OFF)

vt_triton_aot_available_arches(_arches)
list(LENGTH _arches _n_arches)
# SEVEN here, not upstream's six: sm_120a is this fork's tree, vendored for the
# consumer Blackwell this project develops on. Pinning the COUNT is the point of
# this check, so the literal stays a literal -- it just has to be the right one.
# THIRD gate that counted six: scripts/check-triton-aot-multiarch.py (c567a3760),
# cmake/TritonAOTMultiArchTest.cmake (18a6912c7), and this. Each was found by a
# separate CI run because each earlier fix looked for gates and did not sweep.
if(NOT _n_arches EQUAL 7)
  _fail("expected seven vendored trees, got ${_n_arches}")
endif()
file(MAKE_DIRECTORY "${_scratch}/partial")
foreach(_arch IN LISTS _arches)
  if(NOT _arch STREQUAL "sm_90a")
    file(MAKE_DIRECTORY "${_scratch}/partial/${_arch}")
    file(WRITE "${_scratch}/partial/${_arch}/MANIFEST" "arch ${_arch}\n")
  endif()
endforeach()
_expect("one-missing-tree" OFF "sm_90a" ON "${_scratch}/partial" "${_root}" OFF)

# ── The precondition must match what the BUILDER path actually demands ───────
#
# The default may not turn a configure that used to succeed into an error, and
# MANIFEST-existence alone is far weaker than the builder's own preconditions:
# TritonAOT.cmake FATAL_ERRORs on a missing generated file (:253-263), on an
# artifact whose bytes drift from its MANIFEST hash (:602-625) and on a
# triton_kernels/*.py edited since regeneration (:627-664). Each cell below
# reproduces exactly one of those, on a fixture that is otherwise VALID, so the
# single mutation is what declines it.
#
# `_synth_fixture` writes a self-contained source root (its own copies of
# triton_kernels/*.py and scripts/triton-aot-compile.py) plus a minimal but
# fully consistent tree per arch. Cell "synth-valid" pins that the unmutated
# fixture resolves ON.
function(_synth_fixture ROOT)
  file(REMOVE_RECURSE "${ROOT}")
  file(MAKE_DIRECTORY "${ROOT}/scripts" "${ROOT}/triton_kernels")
  file(COPY "${_root}/scripts/triton-aot-compile.py" DESTINATION "${ROOT}/scripts")
  file(GLOB _srcs "${_root}/triton_kernels/*.py")
  file(COPY ${_srcs} DESTINATION "${ROOT}/triton_kernels")
  file(SHA256 "${ROOT}/scripts/triton-aot-compile.py" _generator)
  file(GLOB _copied "${ROOT}/triton_kernels/*.py")
  list(SORT _copied)
  vt_triton_aot_available_arches(_all)
  foreach(_arch IN LISTS _all)
    string(REGEX REPLACE "^sm_([0-9]+)a?$" "\\1" _cc "${_arch}")
    set(_dir "${ROOT}/vendored/${_arch}")
    file(MAKE_DIRECTORY "${_dir}")
    file(WRITE "${_dir}/demo.c" "/* generated launcher for ${_arch} */\n")
    file(WRITE "${_dir}/demo.h" "/* generated header for ${_arch} */\n")
    file(SHA256 "${_dir}/demo.c" _demo_c)
    file(SHA256 "${_dir}/demo.h" _demo_h)
    set(_manifest "arch ${_arch}\ntriton_target cuda:${_cc}:32\nline_info disabled\n")
    string(APPEND _manifest
      "generator scripts/triton-aot-compile.py sha256=${_generator}\n")
    foreach(_py IN LISTS _copied)
      get_filename_component(_name "${_py}" NAME)
      file(SHA256 "${_py}" _hash)
      string(APPEND _manifest "source ${_name} sha256=${_hash}\n")
    endforeach()
    string(APPEND _manifest "base demo py=demo.py kernel=demo grid=1,1,1\n")
    string(APPEND _manifest "artifact demo.c sha256=${_demo_c}\n")
    string(APPEND _manifest "artifact demo.h sha256=${_demo_h}\n")
    file(WRITE "${_dir}/MANIFEST" "${_manifest}")
  endforeach()
endfunction()

set(_fixture "${_scratch}/fixture")

_synth_fixture("${_fixture}")
_expect("synth-valid" ON "" ON "${_fixture}/vendored" "${_fixture}" OFF)

# Reproduction 1 (reviewer F2): append one comment line to a kernel source. The
# builder path FATAL_ERRORs on the staleness comparison, so with a
# MANIFEST-existence-only precondition every CUDA configure that used to succeed
# with the option OFF now fails.
_synth_fixture("${_fixture}")
file(APPEND "${_fixture}/triton_kernels/solve_tril.py" "# one more comment\n")
_expect("synth-kernel-edited" OFF "solve_tril.py" ON
  "${_fixture}/vendored" "${_fixture}" OFF)

# A kernel source that no vendored tree pins is the same defect seen from the
# other side, and the builder rejects it in the same place.
_synth_fixture("${_fixture}")
file(WRITE "${_fixture}/triton_kernels/brand_new.py" "# never regenerated\n")
_expect("synth-kernel-added" OFF "brand_new.py" ON
  "${_fixture}/vendored" "${_fixture}" OFF)

# Reproduction 2 (reviewer F2): delete a generated file that the MANIFEST pins.
_synth_fixture("${_fixture}")
file(REMOVE "${_fixture}/vendored/sm_80/demo.c")
_expect("synth-artifact-deleted" OFF "demo.c" ON
  "${_fixture}/vendored" "${_fixture}" OFF)

# A generated file whose bytes drift from the MANIFEST hash.
_synth_fixture("${_fixture}")
file(APPEND "${_fixture}/vendored/sm_89/demo.h" "/* hand-edited */\n")
_expect("synth-artifact-corrupted" OFF "demo.h" ON
  "${_fixture}/vendored" "${_fixture}" OFF)

# A generated file the MANIFEST does not pin at all.
_synth_fixture("${_fixture}")
file(WRITE "${_fixture}/vendored/sm_86/stray.c" "/* unpinned */\n")
_expect("synth-artifact-unpinned" OFF "stray.c" ON
  "${_fixture}/vendored" "${_fixture}" OFF)

# The generator shim the trees were produced with.
_synth_fixture("${_fixture}")
file(APPEND "${_fixture}/scripts/triton-aot-compile.py" "# shim changed\n")
_expect("synth-generator-changed" OFF "triton-aot-compile.py" ON
  "${_fixture}/vendored" "${_fixture}" OFF)

# The path-independent line-info policy the builder requires of every tree.
_synth_fixture("${_fixture}")
file(READ "${_fixture}/vendored/sm_100a/MANIFEST" _m)
string(REPLACE "line_info disabled\n" "" _m "${_m}")
file(WRITE "${_fixture}/vendored/sm_100a/MANIFEST" "${_m}")
_expect("synth-line-info-missing" OFF "line_info" ON
  "${_fixture}/vendored" "${_fixture}" OFF)

# A MANIFEST that belongs to another architecture.
_synth_fixture("${_fixture}")
file(READ "${_fixture}/vendored/sm_121a/MANIFEST" _m)
string(REPLACE "arch sm_121a\n" "arch sm_90a\n" _m "${_m}")
file(WRITE "${_fixture}/vendored/sm_121a/MANIFEST" "${_m}")
_expect("synth-arch-mismatch" OFF "sm_121a" ON
  "${_fixture}/vendored" "${_fixture}" OFF)

file(REMOVE_RECURSE "${_scratch}")

get_property(_failures GLOBAL PROPERTY VT_TRITON_DEFAULT_TEST_FAILURES)
if(_failures)
  list(LENGTH _failures _n_failures)
  list(JOIN _failures "\n  " _failures_text)
  message(FATAL_ERROR
    "Triton AOT default matrix: ${_n_failures} FAILED\n  ${_failures_text}")
endif()

message(STATUS "Triton AOT default matrix: ALL PASS")
