# cmake/TritonAOT.cmake — Triton AOT kernels, VENDORED per-arch (CUDA-only, gated).
#
# SANCTIONED CUDA-ONLY accelerator (user decision). @triton.jit kernels are
# compiled to cubins that are EMBEDDED (as C byte arrays) in generated C
# launchers and linked into libvllm. At runtime there is NO Triton/Python — the
# launcher uses the CUDA driver API (cuModuleLoadData + cuLaunchKernel). The hand
# C++/CUDA path is always kept as the fallback; a Triton realization is selected
# only behind both a compile gate (VLLM_CPP_TRITON) and a runtime toggle.
#
# ── Two modes ────────────────────────────────────────────────────────────────
#
# BUILDER (VLLM_CPP_TRITON=ON, the default flow): the generated .c/.h artifacts
# are VENDORED in-repo under src/vt/cuda/triton_aot_vendored/<arch>/ (like the
# Marlin / FlashAttention-2 vendored kernels). The build consumes them directly:
# NO Python, NO Triton, NO GPU needed at build time — only a C compiler. If the
# target arch has no vendored artifacts, configuration FAILS with instructions.
#
# MAINTAINER (VLLM_CPP_TRITON_REGEN=ON): runs the original Python toolchain
# (triton.tools.compile / triton.tools.link, needs Python+Triton+ptxas) AND
# refreshes the vendored
# tree + its MANIFEST, so `scripts/regen-triton-aot.sh` + `git diff` is the
# whole regen workflow. The build then compiles the refreshed vendored files
# (exactly what a builder will compile).
#
# Mechanism of the generated code (proven on a dgx PoC):
#   python -m triton.tools.compile <kernel.py> --kernel-name <fn>
#       --out-name <base> --out-path <dir>/<base>
#       --signature "<sig>" --grid "gx,gy,gz" --num-warps N --num-stages S
#   emits, per specialization:
#     <base>.<hash>_<suffix>.c  launcher + EMBEDDED cubin (plain C; no Triton at
#                               runtime). The grid gx/gy/gz is baked in from
#                               --grid (a C expression over the signature scalars);
#                               num_warps/num_stages are baked into the cubin.
#     <base>.<hash>_<suffix>.h  declares void load_<sym>(void); void unload_<sym>(void);
#                               and CUresult <sym>(CUstream, <typed args...>) where
#                               <sym> is HASHED (<base>_<hash>_<suffix>).
#   python -m triton.tools.link <base>.*.h -o <dir>/<base>
#   emits STABLE-named dispatchers into <base>.{c,h}:
#     CUresult <base>(CUstream, <full args...>, int algo_id);
#     CUresult <base>_default(CUstream, <full args...>);   // algo_id = 0
#     void load_<base>(void); void unload_<base>(void);
#   so our C++ calls a STABLE symbol (<base>_default) and never touches the hash.
#   We ALWAYS run the linker (even for a single specialization): it yields the
#   same stable ABI for 1..N specs, so no hash-parsing is ever required.
#
# Staleness: the vendored MANIFEST records the sha256 of every
# triton_kernels/*.py plus the exact generation parameters per kernel base.
# Consuming a vendored tree whose hashes/parameters no longer match the working
# tree FAILS configuration. Shipping source and embedded cubins as an atomic
# unit is the builder contract; regen remains a maintainer task.
#
# VLLM_CPP_TRITON=OFF => this file only DEFINES the options + cache
# vars + the functions and does nothing else. No Python is invoked, no sources
# are added, no compile definitions change: a build without it is byte-identical
# to a tree that never included this file.

option(VLLM_CPP_TRITON_REGEN
  "MAINTAINER: regenerate target-pinned vendored Triton AOT artifacts with Python+Triton+ptxas; see scripts/regen-triton-aot.sh" OFF)

include("${CMAKE_CURRENT_LIST_DIR}/TritonAOTMultiArch.cmake")

# Where the vendored artifacts live, one subdir per arch (sm_121a today;
# sm_90/sm_80/gfx* slots in later). Each arch dir holds the generated .c/.h for
# every kernel base plus a MANIFEST (generator versions + source hashes +
# generation parameters).
set(VLLM_CPP_TRITON_VENDORED_DIR "${CMAKE_SOURCE_DIR}/src/vt/cuda/triton_aot_vendored"
  CACHE PATH "Root of the vendored Triton AOT artifacts (per-arch subdirs)")

# The default SHIPS the vendored kernels (BUILD-TRITON-DEFAULT-ON, #219): it is
# computed, not hardcoded, so it can be ON exactly where the build can consume
# the artifacts and OFF — quietly, naming the condition — everywhere else. The
# rule and the matrix that pins it live in TritonAOTMultiArch.cmake /
# TritonAOTDefaultTest.cmake. `option()` still yields to a command-line
# -DVLLM_CPP_TRITON=..., and an EXPLICIT request keeps every existing
# diagnostic below unchanged: only the default is allowed to degrade quietly.
vt_triton_aot_computed_default(_vt_triton_aot_default _vt_triton_aot_decline
  "${VLLM_CPP_CUDA}" "${VLLM_CPP_TRITON_VENDORED_DIR}" "${CMAKE_SOURCE_DIR}"
  "${VLLM_CPP_TRITON_REGEN}")
option(VLLM_CPP_TRITON
  "Link the vendored Triton AOT kernels (cubins embedded in C) into libvllm (CUDA-only; no Python needed)"
  ${_vt_triton_aot_default})
# Report the RESOLVED value, not the computed one. A cache entry or an explicit
# -DVLLM_CPP_TRITON=ON overrides the default, and printing "default OFF" beside
# a cache that reads ON is simply false — which is what every maintainer regen
# saw, since scripts/regen-triton-aot.sh passes VLLM_CPP_TRITON=ON together
# with VLLM_CPP_TRITON_REGEN=ON.
if(_vt_triton_aot_decline AND NOT VLLM_CPP_TRITON)
  message(STATUS
    "Triton AOT: default OFF — ${_vt_triton_aot_decline}")
endif()

# Arch subdir name. Empty => derived from VLLM_CPP_CUDA_ARCHITECTURES as
# "sm_<arch>" (e.g. 121a -> sm_121a). Set explicitly for cross-arch layouts the
# derivation cannot guess (a multi-arch fat build, or a future gfx* backend).
set(VLLM_CPP_TRITON_VENDORED_ARCH "" CACHE STRING
  "Vendored Triton AOT arch subdir (empty = sm_<VLLM_CPP_CUDA_ARCHITECTURES>)")

# The Python interpreter that has Triton installed. Used ONLY at configure time,
# ONLY when VLLM_CPP_TRITON_REGEN=ON (the maintainer path). Normal builds never
# touch it. Defaults to the project's oracle venv on dgx.
set(VLLM_CPP_TRITON_PYTHON "$ENV{HOME}/venvs/vllm-oracle/bin/python"
  CACHE FILEPATH "Python interpreter with Triton installed (REGEN only)")

# Optional explicit Triton target "<backend>:<arch>:<warp-size>" (e.g.
# cuda:121:32 for GB10 sm_121). Empty => derive it from the single vendored arch
# directory. An explicit value must match that directory; autodetect is forbidden
# because it can silently put an sm_90 cubin in sm_121a on a generic GPU runner.
set(VLLM_CPP_TRITON_TARGET "" CACHE STRING
  "Triton AOT target '<backend>:<arch>:<warpsize>' (REGEN only; empty = derive from vendored arch)")

# Fresh build-declaration accumulator per configure run. Every
# add_triton_kernel call appends its exact manifest line in both builder and
# regen modes; finalize compares it with the canonical contract and MANIFEST.
set_property(GLOBAL PROPERTY VLLM_TRITON_AOT_EXPECTED_BASE_LINES "")
set_property(GLOBAL PROPERTY VLLM_TRITON_AOT_DISPATCH_DECLS "")

# _triton_aot_arch_name(OUTVAR) -> the ONE vendored arch directory this build
# regenerates into. REACHED ONLY FROM THE MAINTAINER REGEN FLOW: the builder path
# goes through `_triton_aot_arch_names` below, which embeds every available tree
# and leaves the choice to exact-SM runtime dispatch. A fat builder configure is
# therefore fine (the shipped ten-SM release archive is exactly that), and it is
# only regeneration — writing a cubin into one directory — that needs a single
# arch. The FATAL_ERROR below is that regen guard and its wording is unchanged.
#
# MULTI-ARCH (fat-binary) REGENERATION IS NOT SUPPORTED BY THE AOT TREE, and that
# is a PROPERTY OF CUBINS, not a missing feature here. Every vendored artifact embeds
# a cubin compiled for exactly one `sm_<cc>` target; `cuModuleLoadData` rejects it
# on any other SM. A fat CUDA build (e.g. `-DVLLM_CPP_CUDA_ARCHITECTURES="120a;121a"`)
# therefore has NO single correct AOT tree: pinning it to one arch would silently
# ship a binary whose Triton realizations fault on the other target. The
# derivation used to produce a nonexistent joined name ("sm_120a_121a") and fail
# downstream with a misleading "regenerate this arch" hint; it now says exactly
# what is wrong and what the two honest options are. The hand C++/CUDA kernels are
# always-available fallbacks, so `-DVLLM_CPP_TRITON=OFF` degrades gracefully with
# no loss of correctness (see .agents/specs/cuda-arch-additivity.md).
function(_triton_aot_arch_name OUTVAR)
  if(VLLM_CPP_TRITON_VENDORED_ARCH)
    set(_a "${VLLM_CPP_TRITON_VENDORED_ARCH}")
  else()
    list(LENGTH VLLM_CPP_CUDA_ARCHITECTURES _n_archs)
    if(_n_archs GREATER 1)
      message(FATAL_ERROR
        "VLLM_CPP_TRITON=ON is incompatible with the multi-architecture (fat) CUDA "
        "build VLLM_CPP_CUDA_ARCHITECTURES='${VLLM_CPP_CUDA_ARCHITECTURES}'.\n"
        "Vendored Triton AOT artifacts embed a cubin built for ONE sm_<cc> target; "
        "the CUDA driver refuses to load it on any other SM, so no single vendored "
        "tree is correct for this build.\n"
        "Options:\n"
        "  * build the fat binary with -DVLLM_CPP_TRITON=OFF (the portable C++/CUDA\n"
        "    kernels are the always-available fallback; correctness is unaffected), or\n"
        "  * build one single-arch binary per target with -DVLLM_CPP_TRITON=ON, or\n"
        "  * set -DVLLM_CPP_TRITON_VENDORED_ARCH=sm_<cc> to pin ONE tree, which is\n"
        "    only sound when every device you will run on matches that cubin.")
    endif()
    set(_a "sm_${VLLM_CPP_CUDA_ARCHITECTURES}")
  endif()
  set(${OUTVAR} "${_a}" PARENT_SCOPE)
endfunction()

# _triton_aot_arch_dir(OUTVAR) -> absolute path of the active vendored arch dir.
function(_triton_aot_arch_dir OUTVAR)
  _triton_aot_arch_name(_a)
  set(${OUTVAR} "${VLLM_CPP_TRITON_VENDORED_DIR}/${_a}" PARENT_SCOPE)
endfunction()

# _triton_aot_arch_names(OUTVAR) -> artifact trees embedded by a normal build.
# W2 deliberately embeds every complete vendored tree. Runtime selection is
# exact-SM, so sm_87/sm_103/sm_110/sm_120 use the portable CUDA fallback and
# can never attempt a neighboring cubin. Maintainer regeneration remains a
# one-tree operation selected by VLLM_CPP_TRITON_VENDORED_ARCH.
function(_triton_aot_arch_names OUTVAR)
  if(VLLM_CPP_TRITON_REGEN)
    _triton_aot_arch_name(_single)
    set(_arches "${_single}")
  else()
    vt_triton_aot_available_arches(_arches)
  endif()
  foreach(_arch IN LISTS _arches)
    if(NOT EXISTS "${VLLM_CPP_TRITON_VENDORED_DIR}/${_arch}/MANIFEST")
      message(FATAL_ERROR
        "Triton AOT W2 matrix names missing tree ${_arch}; regenerate or repair "
        "${VLLM_CPP_TRITON_VENDORED_DIR}")
    endif()
  endforeach()
  set(${OUTVAR} "${_arches}" PARENT_SCOPE)
endfunction()

# Resolve and validate the code-generation target from the vendored destination.
# The current AOT toolchain is CUDA-only and accepts one target per artifact tree.
function(_triton_aot_resolved_target_for_arch OUTVAR _arch)
  if(NOT _arch MATCHES "^sm_([0-9]+)a?$")
    message(FATAL_ERROR
      "Cannot derive a single CUDA Triton target from vendored arch '${_arch}'. "
      "Use one sm_<capability> directory per artifact tree.")
  endif()
  set(_derived "cuda:${CMAKE_MATCH_1}:32")
  if(VLLM_CPP_TRITON_TARGET AND NOT VLLM_CPP_TRITON_TARGET STREQUAL _derived)
    message(FATAL_ERROR
      "VLLM_CPP_TRITON_TARGET='${VLLM_CPP_TRITON_TARGET}' does not match "
      "vendored arch '${_arch}' (required '${_derived}'). Refusing to write a "
      "cubinary into the wrong architecture directory.")
  endif()
  set(${OUTVAR} "${_derived}" PARENT_SCOPE)
endfunction()

function(_triton_aot_resolved_target OUTVAR)
  _triton_aot_arch_name(_arch)
  # SCOPING FIX. The helper's `set(${OUTVAR} ... PARENT_SCOPE)` lands in THIS
  # function's scope, not the caller's, so without the propagation below the
  # caller reads an empty string -- and CMake drops the empty list element, which
  # reaches Triton as a bare `--target` and fails with
  # "argument --target/-t: expected one argument". Only the REGEN path calls this,
  # which is why a vendored-artifact build never sees it.
  _triton_aot_resolved_target_for_arch(_vt_resolved_target "${_arch}")
  set(${OUTVAR} "${_vt_resolved_target}" PARENT_SCOPE)
endfunction()

# add_triton_kernel(RESULT_VAR KERNEL_PY KERNEL_NAME OUT_BASE SIGNATURE GRID
#                   [NUM_WARPS] [NUM_STAGES])
#
# Provides the AOT build of KERNEL_NAME (a @triton.jit fn defined in KERNEL_PY):
# from the vendored tree (default) or freshly regenerated (REGEN). SIGNATURE is
# one Triton signature, or several separated by '|' to build multiple
# specializations of the SAME kernel (they share OUT_BASE so triton.tools.link
# groups them under one stable dispatcher that picks the right specialization at
# runtime). GRID is the "gx,gy,gz" launch-grid expression (may reference the
# signature's scalar args, e.g. "n_rows,1,1"). Defaults: NUM_WARPS=4, NUM_STAGES=3.
#
# On return, in the CALLER's scope:
#   ${RESULT_VAR}              -> the generated C sources (add to a target)
#   ${RESULT_VAR}_INCLUDE_DIR  -> the dir holding <OUT_BASE>.h (the stable header)
#   ${RESULT_VAR}_HEADER       -> "<OUT_BASE>.h"
#
# Runs at CONFIGURE time. Only ever called under VLLM_CPP_TRITON (guard at call
# site); call triton_aot_finalize() once after the last add_triton_kernel().
function(add_triton_kernel RESULT_VAR KERNEL_PY KERNEL_NAME OUT_BASE SIGNATURE GRID)
  set(_num_warps 4)
  set(_num_stages 3)
  if(ARGC GREATER 6)
    set(_num_warps "${ARGV6}")
  endif()
  if(ARGC GREATER 7)
    set(_num_stages "${ARGV7}")
  endif()

  if(NOT EXISTS "${KERNEL_PY}")
    message(FATAL_ERROR "add_triton_kernel: kernel file not found: ${KERNEL_PY}")
  endif()
  get_filename_component(_py_name "${KERNEL_PY}" NAME)

  # The MANIFEST line that describes THIS generation request. Written on regen;
  # compared against the vendored MANIFEST when consuming (a mismatch means the
  # pins/signature changed without a regen -> stale cubins).
  set(_manifest_line
    "base ${OUT_BASE} py=${_py_name} kernel=${KERNEL_NAME} warps=${_num_warps} stages=${_num_stages} grid=${GRID} signature=${SIGNATURE}")
  set_property(GLOBAL APPEND PROPERTY VLLM_TRITON_AOT_EXPECTED_BASE_LINES
    "${_manifest_line}")

  if(NOT VLLM_CPP_TRITON_REGEN)
    # ── BUILDER path: consume and namespace all seven vendored trees. ────────
    _triton_aot_arch_names(_arch_names)
    set(_all_sources)
    set(_dispatch_dir "${CMAKE_BINARY_DIR}/triton_aot_multiarch")
    foreach(_arch IN LISTS _arch_names)
      set(_adir "${VLLM_CPP_TRITON_VENDORED_DIR}/${_arch}")
      if(NOT EXISTS "${_adir}/${OUT_BASE}.c" OR
         NOT EXISTS "${_adir}/${OUT_BASE}.h")
        message(FATAL_ERROR
          "Vendored Triton AOT tree ${_adir} is missing '${OUT_BASE}'. "
          "Regenerate: scripts/regen-triton-aot.sh")
      endif()
      file(GLOB _spec_sources "${_adir}/${OUT_BASE}.*.c")
      if(NOT _spec_sources)
        message(FATAL_ERROR
          "Vendored Triton AOT tree ${_adir} has no ${OUT_BASE}.<hash>.c")
      endif()
      set(_tree_sources ${_spec_sources} "${_adir}/${OUT_BASE}.c")

      set(_found_line "")
      file(STRINGS "${_adir}/MANIFEST" _mlines)
      foreach(_line IN LISTS _mlines)
        string(FIND "${_line}" "base ${OUT_BASE} " _pos)
        if(_pos EQUAL 0)
          set(_found_line "${_line}")
        endif()
      endforeach()
      if(NOT _found_line STREQUAL _manifest_line)
        message(FATAL_ERROR
          "STALE VENDORED TRITON AOT ARTIFACTS: '${OUT_BASE}' differs in "
          "${_adir}/MANIFEST.\n  expected: ${_manifest_line}\n"
          "  vendored: ${_found_line}")
      endif()

      vt_triton_aot_namespace_sources(
        "${_arch}" "${OUT_BASE}" "${_dispatch_dir}/namespaces"
        _tree_sources_namespaced ${_tree_sources})
      list(APPEND _all_sources ${_tree_sources_namespaced})

      file(STRINGS "${_adir}/${OUT_BASE}.h" _default_decl
        REGEX "^CUresult ${OUT_BASE}_default\\(")
      list(LENGTH _default_decl _decl_count)
      if(NOT _decl_count EQUAL 1)
        message(FATAL_ERROR
          "${_adir}/${OUT_BASE}.h must declare one ${OUT_BASE}_default")
      endif()
      vt_triton_aot_namespace_declaration(
        _default_decl "${_arch}" "${OUT_BASE}_default" "${_default_decl}")
      set_property(GLOBAL APPEND PROPERTY VLLM_TRITON_AOT_DISPATCH_DECLS
        "${_default_decl}")
      file(STRINGS "${_adir}/${OUT_BASE}.h" _load_decl
        REGEX "^void load_${OUT_BASE}\\(")
      list(LENGTH _load_decl _load_decl_count)
      if(NOT _load_decl_count EQUAL 1)
        message(FATAL_ERROR
          "${_adir}/${OUT_BASE}.h must declare one load_${OUT_BASE}")
      endif()
      vt_triton_aot_namespace_declaration(
        _load_decl "${_arch}" "load_${OUT_BASE}" "${_load_decl}")
      set_property(GLOBAL APPEND PROPERTY VLLM_TRITON_AOT_DISPATCH_DECLS
        "${_load_decl}")
      vt_triton_aot_namespace_token(
        _namespaced_default "${_arch}" "${OUT_BASE}_default")
      message(STATUS
        "Triton AOT: ${OUT_BASE} <- ${_arch} as ${_namespaced_default}")
    endforeach()
    set(_result_include_dir "${_dispatch_dir}")
  else()
    # ── MAINTAINER path: regenerate with the Python toolchain, then refresh ──
    # the vendored tree and compile THOSE files (what a builder will compile).
    if(NOT EXISTS "${VLLM_CPP_TRITON_PYTHON}")
      message(FATAL_ERROR
        "VLLM_CPP_TRITON_REGEN=ON but VLLM_CPP_TRITON_PYTHON does not exist:\n"
        "  ${VLLM_CPP_TRITON_PYTHON}\n"
        "Point -DVLLM_CPP_TRITON_PYTHON=<path> at a Python that has Triton.")
    endif()

    _triton_aot_arch_dir(_adir)
    set(_outdir "${CMAKE_BINARY_DIR}/triton_aot")
    file(MAKE_DIRECTORY "${_outdir}")

    # Wipe any stale artifacts for this base so a reconfigure regenerates cleanly
    # and the header glob below never picks up leftovers (per-spec OR the linked
    # <base>.{c,h}). Matches <base>.* which covers both.
    file(GLOB _stale "${_outdir}/${OUT_BASE}.*")
    if(_stale)
      file(REMOVE ${_stale})
    endif()

    _triton_aot_resolved_target(_resolved_target)
    set(_target_args --target "${_resolved_target}")

    # Split SIGNATURE into one-or-more specializations on '|'.
    string(REPLACE "|" ";" _sigs "${SIGNATURE}")

    # 1) Compile each specialization -> <base>.<hash>_<suffix>.{c,h}
    foreach(_sig IN LISTS _sigs)
      string(STRIP "${_sig}" _sig)
      message(STATUS "Triton AOT: compile ${KERNEL_NAME} [${_sig}] grid(${GRID}) "
                     "warps=${_num_warps} stages=${_num_stages}")
      execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env TRITON_DISABLE_LINE_INFO=1
                "${VLLM_CPP_TRITON_PYTHON}"
                "${CMAKE_SOURCE_DIR}/scripts/triton-aot-compile.py"
                "${KERNEL_PY}"
                --kernel-name "${KERNEL_NAME}"
                --out-name "${OUT_BASE}"
                --out-path "${_outdir}/${OUT_BASE}"
                --signature "${_sig}"
                --grid "${GRID}"
                --num-warps "${_num_warps}"
                --num-stages "${_num_stages}"
                ${_target_args}
        WORKING_DIRECTORY "${_outdir}"
        RESULT_VARIABLE _rc
        OUTPUT_VARIABLE _out
        ERROR_VARIABLE _err)
      if(NOT _rc EQUAL 0)
        message(FATAL_ERROR
          "Triton AOT compile failed (${KERNEL_NAME} / '${_sig}'):\n${_out}\n${_err}")
      endif()
    endforeach()

    # 2) Link all per-spec headers -> stable <base>.{c,h}. The glob runs BEFORE the
    #    linker writes <base>.h (which lacks a middle '.' and would not match the
    #    <base>.*.h pattern anyway), so it only sees the per-spec headers.
    file(GLOB _spec_headers "${_outdir}/${OUT_BASE}.*.h")
    if(NOT _spec_headers)
      message(FATAL_ERROR "Triton AOT: no per-spec headers generated for ${OUT_BASE}")
    endif()
    execute_process(
      COMMAND "${VLLM_CPP_TRITON_PYTHON}" -m triton.tools.link
              ${_spec_headers} -o "${_outdir}/${OUT_BASE}"
      WORKING_DIRECTORY "${_outdir}"
      RESULT_VARIABLE _rc
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "Triton AOT link failed (${OUT_BASE}):\n${_out}\n${_err}")
    endif()

    # 3) Refresh the vendored tree: drop this base's old files (hash-named files
    #    can change name across regens), copy the fresh ones with a provenance
    #    header prepended. Explicit target selection plus disabled line info
    #    removes GPU/path sensitivity; exact bytes are pinned in the MANIFEST.
    file(MAKE_DIRECTORY "${_adir}")
    file(GLOB _old_vendored "${_adir}/${OUT_BASE}.*")
    if(_old_vendored)
      file(REMOVE ${_old_vendored})
    endif()
    string(CONCAT _hdr
      "/*\n"
      " * vllm.cpp VENDORED Triton AOT artifact — GENERATED CODE. DO NOT EDIT.\n"
      " *\n"
      " * Generated by `scripts/triton-aot-compile.py` / `triton.tools.link` from\n"
      " * triton_kernels/${_py_name} (kernel ${KERNEL_NAME}); the embedded byte\n"
      " * array (if any) is the compiled cubin for the arch named by this\n"
      " * directory. Generator versions, kernel-source hashes and the exact\n"
      " * generation parameters are recorded in the MANIFEST next to this file.\n"
      " *\n"
      " * Regenerate (MAINTAINER task; needs Python + Triton + ptxas):\n"
      " *   scripts/regen-triton-aot.sh\n"
      " *\n"
      " * The generating kernel source (triton_kernels/${_py_name}) is ported\n"
      " * verbatim from flash-linear-attention via vLLM (MIT; see its header);\n"
      " * this generated artifact is distributed under the repository's\n"
      " * Apache-2.0 license (see LICENSE and NOTICE).\n"
      " */\n")
    file(GLOB _gen_files "${_outdir}/${OUT_BASE}.*")
    set(_all_sources)
    foreach(_f IN LISTS _gen_files)
      get_filename_component(_fname "${_f}" NAME)
      file(READ "${_f}" _content)
      # Triton's linker emits whitespace-only lines in generated headers. Keep
      # the vendored text canonical so new artifacts pass git diff --check and
      # repeated regeneration does not require hand-editing generated files.
      string(REGEX REPLACE "[ \t]+\n" "\n" _content "${_content}")
      string(REGEX REPLACE "[ \t]+$" "" _content "${_content}")
      string(REGEX REPLACE "\n+$" "\n" _content "${_content}")
      file(WRITE "${_adir}/${_fname}" "${_hdr}${_content}")
      if(_fname MATCHES "\\.c$")
        list(APPEND _all_sources "${_adir}/${_fname}")
      endif()
    endforeach()
    message(STATUS "Triton AOT: ${OUT_BASE} regenerated -> ${_adir}")
    set(_result_include_dir "${_adir}")
  endif()

  # This is GENERATED codegen, not our code: compile with warnings suppressed
  # (-w). The launch stub deliberately has a missing-return on its empty-grid
  # branch; without -w that would trip a warning. The files end in .c, so CMake
  # compiles them as C automatically (enable_language(C) at the call site).
  set_property(SOURCE ${_all_sources} APPEND PROPERTY COMPILE_OPTIONS "-w")

  set(${RESULT_VAR} "${_all_sources}" PARENT_SCOPE)
  set(${RESULT_VAR}_INCLUDE_DIR "${_result_include_dir}" PARENT_SCOPE)
  set(${RESULT_VAR}_HEADER "${OUT_BASE}.h" PARENT_SCOPE)
  message(STATUS
    "Triton AOT: ${OUT_BASE} -> stable symbol ${OUT_BASE}_default()")
endfunction()

# triton_aot_finalize()
#
# Call ONCE after the last add_triton_kernel() (only under VLLM_CPP_TRITON).
#  * REGEN: writes the vendored MANIFEST — generator versions (triton, its
#    bundled ptxas, python, the CUDA toolkit configured here), the sha256 of
#    every triton_kernels/*.py, the generation date, and one 'base' line per
#    kernel (accumulated by the add_triton_kernel calls).
#  * builder: verifies target, line-info policy, source hashes, generation
#    declarations, artifact inventory, and artifact hashes; any drift is fatal.
function(triton_aot_finalize)
  if(VLLM_CPP_TRITON_REGEN)
    _triton_aot_arch_dir(_adir)
    _triton_aot_arch_name(_arch_name)
    _triton_aot_resolved_target(_resolved_target)
  else()
    _triton_aot_arch_names(_arch_names)
  endif()

  get_property(_expected_bases GLOBAL PROPERTY VLLM_TRITON_AOT_EXPECTED_BASE_LINES)
  list(SORT _expected_bases)
  get_property(_contract_bases GLOBAL PROPERTY VLLM_TRITON_AOT_CONTRACT_LINES)
  list(SORT _contract_bases)
  if(NOT _expected_bases STREQUAL _contract_bases)
    list(JOIN _expected_bases "\n  " _expected_text)
    list(JOIN _contract_bases "\n  " _contract_text)
    message(FATAL_ERROR
      "Triton AOT build declarations differ from cmake/TritonAOTKernels.cmake.\n"
      "Build declarations:\n  ${_expected_text}\n"
      "Canonical contract:\n  ${_contract_text}\n"
      "Update the canonical contract and build calls together, then regenerate.")
  endif()

  if(VLLM_CPP_TRITON_REGEN)
    # Generator versions. Triton compiles the cubin with its OWN bundled ptxas
    # (triton/backends/nvidia/bin/ptxas), not the toolkit's — record that one.
    execute_process(
      COMMAND "${VLLM_CPP_TRITON_PYTHON}" -c
"import os, sys, subprocess, triton
print(triton.__version__)
print('%d.%d.%d' % sys.version_info[:3])
p = os.path.join(os.path.dirname(triton.__file__), 'backends', 'nvidia', 'bin', 'ptxas')
if os.path.exists(p):
    out = subprocess.run([p, '--version'], capture_output=True, text=True).stdout
    print(out.strip().splitlines()[-1].strip())
else:
    print('unknown (no bundled ptxas)')"
      RESULT_VARIABLE _rc OUTPUT_VARIABLE _vers ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "Triton AOT: version probe failed:\n${_err}")
    endif()
    string(REGEX REPLACE "\n+$" "" _vers "${_vers}")
    string(REPLACE "\n" ";" _vlist "${_vers}")
    list(GET _vlist 0 _triton_ver)
    list(GET _vlist 1 _python_ver)
    list(GET _vlist 2 _ptxas_ver)
    string(TIMESTAMP _date "%Y-%m-%d" UTC)
    set(_m "")
    string(APPEND _m
      "# vllm.cpp vendored Triton AOT artifacts — MANIFEST. GENERATED, do not edit.\n"
      "# Regenerate: scripts/regen-triton-aot.sh (maintainer task; Python+Triton+ptxas).\n"
      "# 'source' lines are the sha256 of the generating triton_kernels/*.py at regen\n"
      "# time; 'generator' pins the repository-owned Triton 3.6 target shim;\n"
      "# 'base' lines are the exact generation parameters per dispatcher;\n"
      "# 'artifact' lines pin every generated file byte-for-byte.\n")
    string(APPEND _m "arch ${_arch_name}\n")
    string(APPEND _m "generated ${_date}\n")
    string(APPEND _m "triton ${_triton_ver}\n")
    string(APPEND _m "python ${_python_ver}\n")
    string(APPEND _m "ptxas ${_ptxas_ver}\n")
    string(APPEND _m "cuda_toolkit ${CMAKE_CUDA_COMPILER_VERSION}\n")
    string(APPEND _m "triton_target ${_resolved_target}\n")
    string(APPEND _m "line_info disabled\n")
    file(SHA256 "${CMAKE_SOURCE_DIR}/scripts/triton-aot-compile.py" _generator_hash)
    string(APPEND _m
      "generator scripts/triton-aot-compile.py sha256=${_generator_hash}\n")
    file(GLOB _pys "${CMAKE_SOURCE_DIR}/triton_kernels/*.py")
    list(SORT _pys)
    foreach(_py IN LISTS _pys)
      get_filename_component(_pn "${_py}" NAME)
      file(SHA256 "${_py}" _h)
      string(APPEND _m "source ${_pn} sha256=${_h}\n")
    endforeach()
    foreach(_b IN LISTS _expected_bases)
      string(APPEND _m "${_b}\n")
    endforeach()
    file(GLOB _artifacts "${_adir}/*.c" "${_adir}/*.h")
    list(SORT _artifacts)
    foreach(_artifact IN LISTS _artifacts)
      get_filename_component(_artifact_name "${_artifact}" NAME)
      file(SHA256 "${_artifact}" _artifact_hash)
      string(APPEND _m "artifact ${_artifact_name} sha256=${_artifact_hash}\n")
    endforeach()
    file(WRITE "${_adir}/MANIFEST" "${_m}")
    message(STATUS "Triton AOT: MANIFEST written -> ${_adir}/MANIFEST "
                   "(triton ${_triton_ver}, ptxas '${_ptxas_ver}'). "
                   "Review + commit the vendored tree: git diff ${_adir}")
  else()
    get_property(_dispatch_decls GLOBAL PROPERTY
      VLLM_TRITON_AOT_DISPATCH_DECLS)
    list(REMOVE_DUPLICATES _dispatch_decls)
    list(SORT _dispatch_decls)
    set(_dispatch_dir "${CMAKE_BINARY_DIR}/triton_aot_multiarch")
    file(MAKE_DIRECTORY "${_dispatch_dir}")
    set(_dispatch_header "${_dispatch_dir}/triton_aot_multiarch.h")
    file(WRITE "${_dispatch_header}"
      "/* Generated W2 declarations for six collision-free AOT trees. */\n"
      "#pragma once\n#include <cuda.h>\n#ifdef __cplusplus\nextern \"C\" {\n#endif\n")
    foreach(_decl IN LISTS _dispatch_decls)
      file(APPEND "${_dispatch_header}" "${_decl};\n")
    endforeach()
    file(APPEND "${_dispatch_header}"
      "#ifdef __cplusplus\n}\n#endif\n")

    foreach(_arch_name IN LISTS _arch_names)
    set(_adir "${VLLM_CPP_TRITON_VENDORED_DIR}/${_arch_name}")
    _triton_aot_resolved_target_for_arch(_resolved_target "${_arch_name}")
    # Exact dispatcher/signature set for the vendored tree we just consumed.
    file(STRINGS "${_adir}/MANIFEST" _mlines)
    if(NOT "arch ${_arch_name}" IN_LIST _mlines)
      message(FATAL_ERROR
        "Triton AOT MANIFEST arch does not match destination '${_arch_name}'")
    endif()
    if(NOT "triton_target ${_resolved_target}" IN_LIST _mlines)
      message(FATAL_ERROR
        "Triton AOT MANIFEST target does not match '${_resolved_target}'")
    endif()
    if(NOT "line_info disabled" IN_LIST _mlines)
      message(FATAL_ERROR
        "Triton AOT MANIFEST lacks the path-independent line-info policy; regenerate")
    endif()
    file(SHA256 "${CMAKE_SOURCE_DIR}/scripts/triton-aot-compile.py" _generator_hash)
    if(NOT
       "generator scripts/triton-aot-compile.py sha256=${_generator_hash}" IN_LIST _mlines)
      # Print BOTH hashes. This fired once with neither of them in the message,
      # and the cause was not a stale tree at all: the file is committed as LF
      # and a Windows checkout hashed it as CRLF. "They differ" sends the reader
      # to regenerate cubins; "they differ, and here are the two values" sends
      # them to compare the bytes, which is where the answer was.
      message(FATAL_ERROR
        "Triton AOT MANIFEST generator shim differs from scripts/triton-aot-compile.py.\n"
        "  working tree: ${_generator_hash}\n"
        "  MANIFEST (${_adir}/MANIFEST) records a different one.\n"
        "If the tree is unchanged, check line endings before regenerating: this "
        "hash is over the file as CHECKED OUT, so CRLF and LF give different "
        "answers for identical content (.gitattributes pins it to LF).")
    endif()
    set(_manifest_bases "")
    foreach(_line IN LISTS _mlines)
      if(_line MATCHES "^base ")
        list(APPEND _manifest_bases "${_line}")
      endif()
    endforeach()
    list(SORT _manifest_bases)
    if(NOT _manifest_bases STREQUAL _expected_bases)
      list(JOIN _expected_bases "\n  " _expected_text)
      list(JOIN _manifest_bases "\n  " _manifest_text)
      message(FATAL_ERROR
        "STALE VENDORED TRITON AOT ARTIFACT SET (${_adir}).\n"
        "Expected from the build contract:\n  ${_expected_text}\n"
        "Vendored MANIFEST contains:\n  ${_manifest_text}\n"
        "Regenerate: scripts/regen-triton-aot.sh")
    endif()

    set(_manifest_artifacts "")
    foreach(_line IN LISTS _mlines)
      if(_line MATCHES "^artifact ([^ ]+) sha256=([0-9a-f]+)$")
        set(_artifact_name "${CMAKE_MATCH_1}")
        set(_artifact_hash "${CMAKE_MATCH_2}")
        list(APPEND _manifest_artifacts "${_artifact_name}")
        if(NOT EXISTS "${_adir}/${_artifact_name}")
          message(FATAL_ERROR
            "Triton AOT MANIFEST names missing artifact ${_artifact_name}")
        endif()
        file(SHA256 "${_adir}/${_artifact_name}" _actual_hash)
        if(NOT _actual_hash STREQUAL _artifact_hash)
          message(FATAL_ERROR
            "Triton AOT artifact hash mismatch: ${_artifact_name}")
        endif()
      endif()
    endforeach()
    file(GLOB _actual_artifacts RELATIVE "${_adir}" "${_adir}/*.c" "${_adir}/*.h")
    list(SORT _actual_artifacts)
    list(SORT _manifest_artifacts)
    if(NOT _actual_artifacts STREQUAL _manifest_artifacts)
      message(FATAL_ERROR
        "Triton AOT MANIFEST artifact set does not match ${_adir}; regenerate")
    endif()

    # Source staleness for the vendored tree.
    set(_manifest_pys "")
    set(_stale "")
    foreach(_line IN LISTS _mlines)
      if(_line MATCHES "^source ([^ ]+) sha256=([0-9a-f]+)$")
        set(_pn "${CMAKE_MATCH_1}")
        set(_want "${CMAKE_MATCH_2}")
        list(APPEND _manifest_pys "${_pn}")
        set(_py "${CMAKE_SOURCE_DIR}/triton_kernels/${_pn}")
        if(NOT EXISTS "${_py}")
          list(APPEND _stale "${_pn}: in MANIFEST but missing from triton_kernels/")
        else()
          file(SHA256 "${_py}" _have)
          if(NOT _have STREQUAL _want)
            list(APPEND _stale "${_pn}: sha256 differs (kernel edited since regen)")
          endif()
        endif()
      endif()
    endforeach()
    file(GLOB _pys "${CMAKE_SOURCE_DIR}/triton_kernels/*.py")
    foreach(_py IN LISTS _pys)
      get_filename_component(_pn "${_py}" NAME)
      if(NOT _pn IN_LIST _manifest_pys)
        list(APPEND _stale "${_pn}: present in triton_kernels/ but not in MANIFEST")
      endif()
    endforeach()
    if(_stale)
      list(JOIN _stale "\n  " _stale_txt)
      message(FATAL_ERROR
        "STALE VENDORED TRITON AOT ARTIFACTS (${_adir}):\n"
        "  ${_stale_txt}\n"
        "The triton_kernels/*.py sources changed but the vendored cubins were NOT "
        "regenerated. Refusing to build the old kernels. Regenerate "
        "(maintainer task; Python+Triton+ptxas): scripts/regen-triton-aot.sh")
    else()
      message(STATUS "Triton AOT: vendored tree ${_adir} matches triton_kernels/ "
                     "(MANIFEST hashes OK)")
    endif()
    endforeach()
    message(STATUS
      "Triton AOT W2: embedded trees [${_arch_names}]; exact runtime dispatch "
      "header ${_dispatch_header}")
  endif()
endfunction()

# Convenience build-system entry point for the maintainer regen task:
#   cmake --build <build-dir> --target regen-triton-aot
# (equivalent to running scripts/regen-triton-aot.sh from the source root; the
# drift twin `check-triton-aot-drift` needs no Python/GPU and is what CI runs.)
if(NOT TARGET regen-triton-aot)
  add_custom_target(regen-triton-aot
    COMMAND bash "${CMAKE_SOURCE_DIR}/scripts/regen-triton-aot.sh"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Regenerating vendored Triton AOT artifacts (maintainer task: Python+Triton+GPU)"
    USES_TERMINAL)
  add_custom_target(check-triton-aot-drift
    COMMAND bash "${CMAKE_SOURCE_DIR}/scripts/check-triton-aot-drift.sh"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Checking vendored Triton AOT artifacts vs kernel sources (no Python/GPU)")
endif()
