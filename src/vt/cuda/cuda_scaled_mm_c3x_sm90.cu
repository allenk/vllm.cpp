// vllm.cpp — Hopper (sm_90a) CUTLASS C3x scaled-mm FP8 (W8A8) GEMM, wgmma/TMA
// fast-path BUILD-VERIFY translation unit (BACKEND-CUDA-SM090, ROAD-V1-D1-CUDA,
// datacenter fast-path §9 DC2 — the Hopper C3x scaled-mm leg).
//
// SIGNAL: DERIVED+BUILD-VERIFIED (testing-welcome). This TU is a faithful 1:1
// type-port of vLLM's Sm90 C3x fp8 scaled-mm kernel — it compiles + emits real
// sm_90a SASS on the existing GB10 box (single-arch cross-compile), but NO
// H100/H200/sm_90 board has ever RUN it here. A green compile + a cuobjdump SASS
// proof is NOT execution evidence, NOT runtime support, and NOT vLLM-competitive.
// See .agents/specs/cuda-arch-datacenter-fastpath.md §0/§6a/§9 (DC2) and
// backend-matrix.md row BACKEND-CUDA-SM090.
//
// GROUNDING (ground-every-impl rule). The GEMM type structure is ported 1:1 from
//   vLLM @ 5559679229 (0.26.0.dev0)
//   csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8_dispatch.cuh
//     :22-107  (template cutlass_3x_gemm_sm90_fp8 — the collective structure)
//     :109-205 (the sm90_fp8_config_{default,M8192_K6144,M128,M64_N1280,
//               M64_N8192,M16_N1280,M16_N8192} tile/cluster/schedule configs)
//   entry csrc/.../c3x/scaled_mm_sm90_fp8.cu:1-33 (cutlass_scaled_mm_sm90_fp8).
// The Hopper 4th-gen tensor-core collective (wgmma warpgroup-async MMA + TMA +
// the warp-specialized producer/consumer pipeline) lives INSIDE CUTLASS 4.5.0,
// selected by
//   ArchTag = cutlass::arch::Sm90 + OpClassTensorOp +
//   KernelSchedule = KernelTmaWarpSpecialized{Pingpong,Cooperative,}FP8FastAccum
//   EpilogueSchedule = TmaWarpSpecialized{,Cooperative}
//   (cutlass::gemm::collective::CollectiveBuilder picks the wgmma/TMA mainloop) —
// this is the DISTINCT leg vs our shipped consumer sm_12x body, which uses
// ArchTag = Sm120 (src/vt/cuda/cuda_matmul_fp8_cutlass.cu, cutlass_3x_gemm_sm120).
// The tile/cluster/schedule shapes are vLLM's exact choices (spec §10: the
// ArchTag alone is insufficient; the schedule + tile/cluster config is
// load-bearing — it is what makes CUTLASS emit the FP8-fast-accum wgmma pipeline).
//
// DEVIATION (recorded, mirrors cuda_matmul_fp8_cutlass.cu). vLLM applies the two
// per-tensor scales through its ScaledEpilogue EVT (out = scale_a·(scale_b·acc)),
// which pulls in vLLM's cutlass_extensions epilogue headers (not vendored here).
// For a BUILD-VERIFY TU whose purpose is to emit the wgmma/TMA collective SASS,
// we instantiate the plain CUTLASS LinearCombination epilogue (alpha_ptr on
// device) instead — the two per-tensor scalars collapse to one accumulator
// multiply out = alpha·acc, exactly the fold our shipped sm_12x fp8 drop-in
// already uses. The EVT is vLLM DISPATCH glue; the wgmma/TMA MAINLOOP collective
// — the thing this TU exists to build-verify — is byte-for-byte the Sm90 config.
//
// The whole body is guarded on BOTH the build gate VT_SCALEDMM_C3X_SM90 (set by
// the `scaledmm-c3x-sm90` FEATURE-TABLE cell, enabled ONLY for 90a) and the
// CUTLASS capability macro CUTLASS_ARCH_MMA_SM90_SUPPORTED (the analog of vLLM's
// enable_sm90_or_later guard). On the gate arch sm_121a NEITHER is defined, so
// this TU is not even compiled (the CMake source is added only for a 90a target)
// → byte-zero impact on GB10.
#if defined(VT_SCALEDMM_C3X_SM90)

#include <cuda_runtime.h>

#include <cstddef>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "vt/cuda/cutlass_params_launch.cuh"

namespace vt::cuda::scaled_mm_c3x_sm90 {

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using namespace cute;

// vLLM scaled_mm_sm90_fp8_dispatch.cuh:109-205 — the seven tile/cluster/schedule
// configs vLLM selects by GEMM shape. We port a covering subset that exercises
// EVERY distinct (KernelSchedule, EpilogueSchedule) pair vLLM uses on Hopper, so
// the archive carries the wgmma/TMA SASS of all three warp-specialized mainloop
// variants:
//   * _default        (M in (128, inf))        — Pingpong FP8FastAccum
//   * _cooperative     (M >= 8192, K >= 6144)   — Cooperative FP8FastAccum
//   * _small_m         (M <= 64, small tiles)   — plain FP8FastAccum + swap-AB shape
// The four remaining vLLM configs (M128, M64_N1280, M16_N1280, M16_N8192) reuse
// one of these exact schedule pairs with a different tile/cluster and add nothing
// to the emitted collective family, so they are documented here rather than
// separately instantiated.
struct sm90_fp8_config_default {
  // vLLM sm90_fp8_config_default:109-127
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
};

struct sm90_fp8_config_cooperative {
  // vLLM sm90_fp8_config_M8192_K6144:129-147
  using KernelSchedule =
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileShape = Shape<_256, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
};

struct sm90_fp8_config_small_m {
  // vLLM sm90_fp8_config_M64_N8192:168-186 (the low-M swap-AB tile)
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
  using TileShape = Shape<_64, _64, _256>;
  using ClusterShape = Shape<_1, _1, _1>;
};

// vLLM scaled_mm_sm90_fp8_dispatch.cuh:22-107 — cutlass_3x_gemm_sm90_fp8. e4m3
// A/B, RowMajor A / ColumnMajor B, f32 accumulate, ArchTag=Sm90, OpClassTensorOp;
// CUTLASS's CollectiveBuilder emits the wgmma/TMA warp-specialized mainloop +
// epilogue for the requested schedule. (Plain LinearCombination epilogue — see
// the DEVIATION note above; the mainloop collective is vLLM's Sm90 config 1:1.)
template <typename Config, typename OutType>
struct Cutlass3xGemmSm90Fp8 {
  using ElementAB = cutlass::float_e4m3_t;
  using ElementD = OutType;
  using ElementC = OutType;
  using ElementAcc = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  static constexpr int AlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD =
      128 / cutlass::sizeof_bits<ElementD>::value;

  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using TileShape = typename Config::TileShape;
  using ClusterShape = typename Config::ClusterShape;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc,
          ElementAcc, ElementC, LayoutC, AlignmentCD, ElementD, LayoutD,
          AlignmentCD, typename Config::EpilogueSchedule>::CollectiveOp;

  using Stages = cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementAB, LayoutA, AlignmentAB, ElementAB,
          LayoutB, AlignmentAB, ElementAcc, TileShape, ClusterShape, Stages,
          typename Config::KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Host-side workspace probe — instantiates the collective's host path.
template <typename Config>
size_t WorkspaceProbe(int m, int n, int k) {
  using Gemm = typename Cutlass3xGemmSm90Fp8<Config, cutlass::bfloat16_t>::Gemm;
  typename Gemm::Arguments args;
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = make_shape(m, n, k, 1);
  return Gemm::get_workspace_size(args);
}

// Reference the full run/can_implement path so the wgmma/TMA device kernel is
// EMITTED into the archive (not just host-side size math) — this is the
// DERIVED+BUILD-VERIFIED SASS proof (cuobjdump -lelf). NEVER called on GB10 (the
// TU is compiled only into a single-arch 90a archive). A real Hopper board
// upgrades the SIGNAL to RUNTIME-VERIFIED by wiring this into the fp8 scaled-mm
// tactic registry + gating token-exact (spec §5/§6b).
template <typename Config>
cutlass::Status RunProbe(void* d, const void* a, const void* b,
                         const float* alpha, int m, int n, int k,
                         void* workspace, cudaStream_t stream) {
  using GemmT = typename Cutlass3xGemmSm90Fp8<Config, cutlass::bfloat16_t>::Gemm;
  using ElementD = typename GemmT::ElementD;
  using StrideA = typename GemmT::GemmKernel::StrideA;
  using StrideB = typename GemmT::GemmKernel::StrideB;
  using StrideC = typename GemmT::GemmKernel::StrideC;
  using StrideD = typename GemmT::GemmKernel::StrideD;

  StrideA dA = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
  StrideB dB = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
  StrideC dC = cutlass::make_cute_packed_stride(StrideC{}, make_shape(m, n, 1));
  StrideD dD = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));

  typename GemmT::Arguments args;
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = make_shape(m, n, k, 1);
  args.mainloop.ptr_A = static_cast<const cutlass::float_e4m3_t*>(a);
  args.mainloop.dA = dA;
  args.mainloop.ptr_B = static_cast<const cutlass::float_e4m3_t*>(b);
  args.mainloop.dB = dB;
  args.epilogue.thread.alpha_ptr = alpha;
  args.epilogue.ptr_C = static_cast<const ElementD*>(d);
  args.epilogue.dC = dC;
  args.epilogue.ptr_D = static_cast<ElementD*>(d);
  args.epilogue.dD = dD;

  cutlass::Status s = GemmT::can_implement(args);
  if (s != cutlass::Status::kSuccess) return s;
  // Launched through the bytes entry point, like the sm_12x TUs: the adapter's
  // by-value kernel parameter cannot be compiled by MSVC once CUtensorMap keeps
  // its alignment (cutlass_params_launch.cuh). The 2x1x1 configs go through
  // ClusterLauncher there.
  const cudaError_t err =
      cutlass_launch::Run<typename GemmT::GemmKernel>(args, workspace, stream, &s);
  if (s != cutlass::Status::kSuccess) return s;
  return err == cudaSuccess ? cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
}

// Explicit instantiations across the three schedule variants so each wgmma/TMA
// kernel variant lands in the archive with sm_90a SASS.
template size_t WorkspaceProbe<sm90_fp8_config_default>(int, int, int);
template size_t WorkspaceProbe<sm90_fp8_config_cooperative>(int, int, int);
template size_t WorkspaceProbe<sm90_fp8_config_small_m>(int, int, int);
template cutlass::Status RunProbe<sm90_fp8_config_default>(
    void*, const void*, const void*, const float*, int, int, int, void*,
    cudaStream_t);
template cutlass::Status RunProbe<sm90_fp8_config_cooperative>(
    void*, const void*, const void*, const float*, int, int, int, void*,
    cudaStream_t);
template cutlass::Status RunProbe<sm90_fp8_config_small_m>(
    void*, const void*, const void*, const float*, int, int, int, void*,
    cudaStream_t);

#endif  // CUTLASS_ARCH_MMA_SM90_SUPPORTED

}  // namespace vt::cuda::scaled_mm_c3x_sm90

#endif  // VT_SCALEDMM_C3X_SM90
