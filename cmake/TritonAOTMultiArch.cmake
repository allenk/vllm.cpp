# W2 helpers for embedding all available Triton AOT cubin trees in one binary.
# Kept separate so the exact matrix and symbol namespace can be tested with
# `cmake -P` without enabling a compiler or CUDA.

# sm_120a added here: the list is HARD-CODED, so generating a tree with
# VLLM_CPP_TRITON_REGEN is not enough -- the runtime dispatch table is built from
# this list, and a tree missing from it makes TritonAotAvailableOnCurrentDevice()
# return false on that device. Measured: an RTX PRO 6000 Blackwell reports
# compute_cap 12.0 (sm_120), which was in NO shipped tree, so the Triton fast path
# could never fire on it -- and the failure is silent, indistinguishable from
# "Triton is not faster" unless the bail-out reason is instrumented.
set(VT_TRITON_AOT_AVAILABLE_ARCHES
  sm_80 sm_86 sm_89 sm_90a sm_100a sm_120a sm_121a)

function(vt_triton_aot_available_arches OUT_VAR)
  set(${OUT_VAR} "${VT_TRITON_AOT_AVAILABLE_ARCHES}" PARENT_SCOPE)
endfunction()

# vt_triton_aot_tree_defect(OUT_REASON ARCH VENDORED_DIR SOURCE_ROOT)
#
# Empty OUT_REASON iff <VENDORED_DIR>/<ARCH> satisfies every TREE-LEVEL
# precondition the builder path enforces, each mirrored from the place that
# FATAL_ERRORs on it in TritonAOT.cmake:
#
#   * the tree and its MANIFEST exist          (`_triton_aot_arch_names`)
#   * `arch`/`triton_target`/`line_info`       (triton_aot_finalize)
#   * the generator shim's sha256              (triton_aot_finalize)
#   * every `artifact` line names a file that exists and still hashes to its
#     pinned sha256, and the tree holds no *.c/*.h the MANIFEST does not pin
#     (triton_aot_finalize; this is also what makes the per-base
#     `<base>.c`/`<base>.h`/`<base>.<hash>.c` existence checks in
#     `add_triton_kernel` unreachable)
#   * every `source` line matches triton_kernels/<name>.py byte-for-byte, and
#     no triton_kernels/*.py is unpinned (the staleness gate)
#
# Everything here is derivable from the vendored tree plus the source root, so
# it can run BEFORE the option exists. See the note on the residual in
# `vt_triton_aot_computed_default`.
function(vt_triton_aot_tree_defect OUT_REASON ARCH VENDORED_DIR SOURCE_ROOT)
  set(_dir "${VENDORED_DIR}/${ARCH}")
  if(NOT EXISTS "${_dir}/MANIFEST")
    set(${OUT_REASON} "tree ${ARCH} is absent from ${VENDORED_DIR}" PARENT_SCOPE)
    return()
  endif()
  if(NOT ARCH MATCHES "^sm_([0-9]+)a?$")
    set(${OUT_REASON}
      "tree ${ARCH} is not one sm_<capability> directory" PARENT_SCOPE)
    return()
  endif()
  set(_capability "${CMAKE_MATCH_1}")
  file(STRINGS "${_dir}/MANIFEST" _lines)
  if(NOT "arch ${ARCH}" IN_LIST _lines)
    set(${OUT_REASON}
      "tree ${ARCH} carries a MANIFEST generated for another architecture"
      PARENT_SCOPE)
    return()
  endif()
  if(NOT "triton_target cuda:${_capability}:32" IN_LIST _lines)
    set(${OUT_REASON}
      "tree ${ARCH} was generated for another Triton target" PARENT_SCOPE)
    return()
  endif()
  if(NOT "line_info disabled" IN_LIST _lines)
    set(${OUT_REASON}
      "tree ${ARCH} lacks the path-independent line_info policy" PARENT_SCOPE)
    return()
  endif()
  set(_shim "${SOURCE_ROOT}/scripts/triton-aot-compile.py")
  if(NOT EXISTS "${_shim}")
    set(${OUT_REASON}
      "scripts/triton-aot-compile.py is missing, so tree ${ARCH} cannot be \
matched to its generator" PARENT_SCOPE)
    return()
  endif()
  file(SHA256 "${_shim}" _shim_hash)
  if(NOT "generator scripts/triton-aot-compile.py sha256=${_shim_hash}"
     IN_LIST _lines)
    set(${OUT_REASON}
      "scripts/triton-aot-compile.py differs from the generator tree ${ARCH} \
was produced with" PARENT_SCOPE)
    return()
  endif()

  set(_pinned "")
  foreach(_line IN LISTS _lines)
    if(_line MATCHES "^artifact ([^ ]+) sha256=([0-9a-f]+)$")
      set(_name "${CMAKE_MATCH_1}")
      set(_want "${CMAKE_MATCH_2}")
      list(APPEND _pinned "${_name}")
      if(NOT EXISTS "${_dir}/${_name}")
        set(${OUT_REASON}
          "tree ${ARCH} is missing the generated file ${_name} its MANIFEST pins"
          PARENT_SCOPE)
        return()
      endif()
      file(SHA256 "${_dir}/${_name}" _have)
      if(NOT _have STREQUAL _want)
        set(${OUT_REASON}
          "tree ${ARCH} file ${_name} no longer matches its MANIFEST sha256"
          PARENT_SCOPE)
        return()
      endif()
    endif()
  endforeach()
  if(NOT _pinned)
    set(${OUT_REASON}
      "tree ${ARCH} MANIFEST pins no generated artifacts" PARENT_SCOPE)
    return()
  endif()
  file(GLOB _present RELATIVE "${_dir}" "${_dir}/*.c" "${_dir}/*.h")
  set(_unpinned "${_present}")
  if(_unpinned)
    list(REMOVE_ITEM _unpinned ${_pinned})
  endif()
  if(_unpinned)
    list(JOIN _unpinned " " _unpinned_text)
    set(${OUT_REASON}
      "tree ${ARCH} holds generated file(s) [${_unpinned_text}] its MANIFEST \
does not pin" PARENT_SCOPE)
    return()
  endif()

  set(_pinned_sources "")
  foreach(_line IN LISTS _lines)
    if(_line MATCHES "^source ([^ ]+) sha256=([0-9a-f]+)$")
      set(_name "${CMAKE_MATCH_1}")
      set(_want "${CMAKE_MATCH_2}")
      list(APPEND _pinned_sources "${_name}")
      set(_py "${SOURCE_ROOT}/triton_kernels/${_name}")
      if(NOT EXISTS "${_py}")
        set(${OUT_REASON}
          "triton_kernels/${_name} is pinned by tree ${ARCH} but missing from \
the source tree" PARENT_SCOPE)
        return()
      endif()
      file(SHA256 "${_py}" _have)
      if(NOT _have STREQUAL _want)
        set(${OUT_REASON}
          "triton_kernels/${_name} was edited since tree ${ARCH} was \
regenerated" PARENT_SCOPE)
        return()
      endif()
    endif()
  endforeach()
  file(GLOB _sources "${SOURCE_ROOT}/triton_kernels/*.py")
  foreach(_py IN LISTS _sources)
    get_filename_component(_name "${_py}" NAME)
    if(NOT _name IN_LIST _pinned_sources)
      set(${OUT_REASON}
        "triton_kernels/${_name} is not pinned by tree ${ARCH}" PARENT_SCOPE)
      return()
    endif()
  endforeach()

  set(${OUT_REASON} "" PARENT_SCOPE)
endfunction()

# vt_triton_aot_computed_default(OUT_DEFAULT OUT_REASON CUDA_ENABLED
#                                VENDORED_DIR SOURCE_ROOT REGEN)
#
# The default value of VLLM_CPP_TRITON (BUILD-TRITON-DEFAULT-ON, #219). ON
# wherever the build can actually consume the vendored artifacts; otherwise OFF
# with a one-line reason naming the condition that declined it.
#
# The artifacts are pre-generated cubins embedded in plain C, so consuming them
# needs a C compiler and nothing else — the opt-in default guarded no
# dependency, it only made the shipped kernels easy to omit by accident. The
# gate profile has required `-DVLLM_CPP_TRITON=ON` by hand for that reason
# (.agents/environment.md, MANDATORY gate-build flags).
#
# NOT a condition: the number of CUDA architectures. The builder path embeds
# EVERY vendored tree (`_triton_aot_arch_names` in TritonAOT.cmake) and the
# runtime picks a cubin by exact SM, which is precisely what the shipped ten-SM
# release archive (scripts/build-linux-accelerator-release.sh) and the
# cuda-fat-gencode CI job (.github/workflows/ci.yml) already build with
# -DVLLM_CPP_TRITON=ON. The single-arch FATAL_ERROR in `_triton_aot_arch_name`
# guards the MAINTAINER regen flow, which is why REGEN below declines instead.
#
# WHAT THE QUIET DECLINE IS AND IS NOT WORTH. Testing MANIFEST existence alone
# would be much weaker than what the builder demands, and the gap was not
# theoretical: appending one comment line to a triton_kernels/*.py, or deleting
# a generated file from one vendored tree, both left the default ON and then
# FATAL_ERRORed in the builder — turning a configure that used to succeed with
# the option OFF into a hard failure. `vt_triton_aot_tree_defect` above closes
# that by running the builder's own TREE-LEVEL preconditions here.
#
# ONE residual remains, and it is stated rather than claimed away: the builder
# also compares each MANIFEST `base` line against the build contract accumulated
# from the `add_triton_kernel` calls in CMakeLists.txt (TritonAOT.cmake, the
# per-base check in `add_triton_kernel` and the set check in
# `triton_aot_finalize`). Those parameters do not exist yet when the option is
# defined, so editing a kernel's signature, grid, warps or stages WITHOUT
# regenerating still fails a defaulted-ON configure. That is a repository-local
# edit to a tracked file made by whoever is changing the kernel contract, never
# something a downstream builder can hit, and it is the only way in. Everything
# a checkout can present to `cmake` is covered.
#
# An explicit -DVLLM_CPP_TRITON=ON keeps every existing diagnostic, unchanged:
# asking for something impossible stays an error, and only the default degrades
# quietly.
function(vt_triton_aot_computed_default OUT_DEFAULT OUT_REASON CUDA_ENABLED
         VENDORED_DIR SOURCE_ROOT REGEN)
  if(NOT CUDA_ENABLED)
    set(${OUT_DEFAULT} OFF PARENT_SCOPE)
    set(${OUT_REASON}
      "the vendored artifacts are CUDA cubins and VLLM_CPP_CUDA is OFF"
      PARENT_SCOPE)
    return()
  endif()
  if(REGEN)
    set(${OUT_DEFAULT} OFF PARENT_SCOPE)
    set(${OUT_REASON}
      "VLLM_CPP_TRITON_REGEN is ON — maintainer regeneration pins ONE arch tree \
and runs the Python toolchain, so it is never selected for you"
      PARENT_SCOPE)
    return()
  endif()
  vt_triton_aot_available_arches(_arches)
  set(_defects "")
  foreach(_arch IN LISTS _arches)
    vt_triton_aot_tree_defect(_defect "${_arch}" "${VENDORED_DIR}" "${SOURCE_ROOT}")
    if(_defect)
      list(APPEND _defects "${_defect}")
    endif()
  endforeach()
  if(_defects)
    list(JOIN _defects "; " _defects_text)
    set(${OUT_DEFAULT} OFF PARENT_SCOPE)
    set(${OUT_REASON}
      "the vendored artifacts cannot be consumed as they stand: ${_defects_text}"
      PARENT_SCOPE)
    return()
  endif()
  set(${OUT_DEFAULT} ON PARENT_SCOPE)
  set(${OUT_REASON} "" PARENT_SCOPE)
endfunction()

function(vt_triton_aot_arch_tree OUT_VAR ARCH)
  set(_tree "sm_${ARCH}")
  if(_tree IN_LIST VT_TRITON_AOT_AVAILABLE_ARCHES)
    set(${OUT_VAR} "${_tree}" PARENT_SCOPE)
  else()
    set(${OUT_VAR} "" PARENT_SCOPE)
  endif()
endfunction()

function(vt_triton_aot_namespace_token OUT_VAR ARCH TOKEN)
  if(NOT ARCH MATCHES "^sm_[0-9]+a?$")
    message(FATAL_ERROR "invalid Triton AOT namespace architecture '${ARCH}'")
  endif()
  if(NOT TOKEN MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
    message(FATAL_ERROR "invalid Triton AOT namespace token '${TOKEN}'")
  endif()
  set(${OUT_VAR} "vt_aot_${ARCH}_${TOKEN}" PARENT_SCOPE)
endfunction()

# Return one namespaced declaration without its trailing semicolon. CMake uses
# semicolons as list separators, so storing raw C declarations in a GLOBAL
# property can drop the terminator and concatenate two declarations into a
# function definition. The final header writer adds exactly one semicolon.
function(vt_triton_aot_namespace_declaration OUT_VAR ARCH TOKEN DECLARATION)
  vt_triton_aot_namespace_token(_namespaced "${ARCH}" "${TOKEN}")
  string(REPLACE "${TOKEN}" "${_namespaced}" _declaration "${DECLARATION}")
  # file(STRINGS) escapes a source semicolon as `\;` in a CMake list. Remove
  # both characters before this value enters another list/property.
  string(REGEX REPLACE "[\\\\;]+$" "" _declaration "${_declaration}")
  set(${OUT_VAR} "${_declaration}" PARENT_SCOPE)
endfunction()

# Prefix every generated external identifier containing BASE before compiling a
# vendored C launcher. This includes stable dispatchers, hash dispatchers,
# module/function globals, cubin arrays, and load/unload functions. The embedded
# kernel name string remains untouched, as required by cuModuleGetFunction.
function(vt_triton_aot_namespace_sources ARCH BASE OUTPUT_DIR OUT_VAR)
  set(_sources ${ARGN})
  if(NOT _sources)
    message(FATAL_ERROR "no Triton AOT sources to namespace for ${ARCH}/${BASE}")
  endif()
  file(MAKE_DIRECTORY "${OUTPUT_DIR}")
  set(_tokens)
  foreach(_source IN LISTS _sources)
    file(STRINGS "${_source}" _symbol_lines REGEX "${BASE}")
    string(JOIN "\n" _symbol_text ${_symbol_lines})
    string(REGEX MATCHALL
      "[A-Za-z_][A-Za-z0-9_]*${BASE}[A-Za-z0-9_]*"
      _prefixed_tokens "${_symbol_text}")
    string(REGEX MATCHALL "${BASE}[A-Za-z0-9_]*"
      _base_tokens "${_symbol_text}")
    list(APPEND _tokens ${_prefixed_tokens} ${_base_tokens})
  endforeach()
  list(REMOVE_DUPLICATES _tokens)
  list(SORT _tokens)
  if(NOT _tokens)
    message(FATAL_ERROR "no generated symbols found for ${ARCH}/${BASE}")
  endif()

  set(_header "${OUTPUT_DIR}/${ARCH}_${BASE}_namespace.h")
  file(WRITE "${_header}"
    "/* Generated by TritonAOTMultiArch.cmake: collision-free W2 symbols. */\n")
  foreach(_token IN LISTS _tokens)
    vt_triton_aot_namespace_token(_namespaced "${ARCH}" "${_token}")
    file(APPEND "${_header}" "#define ${_token} ${_namespaced}\n")
  endforeach()
  set_property(SOURCE ${_sources} APPEND PROPERTY
    # PORTABILITY. `-include<file>` is the GCC/Clang force-include spelling; MSVC
    # spells it `/FI<file>` and SILENTLY IGNORES the GCC form. On Windows that made
    # the per-arch renaming a no-op: every tree's objects kept the UNPREFIXED
    # symbols while cuda_gdn.cu referenced vt_aot_<arch>_load_*, so VLLM_CPP_TRITON=ON
    # failed to link with 11 unresolved externals. Confirmed with dumpbin: sm_90a and
    # sm_121a objects were byte-for-byte identical in their exported names.
    COMPILE_OPTIONS "$<IF:$<CXX_COMPILER_ID:MSVC>,/FI${_header},-include${_header}>")
  set(${OUT_VAR} "${_sources}" PARENT_SCOPE)
endfunction()
