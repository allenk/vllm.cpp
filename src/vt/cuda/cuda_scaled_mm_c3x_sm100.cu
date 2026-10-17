// vllm.cpp — datacenter-Blackwell (sm_100a) CUTLASS C3x scaled-mm FP8 (W8A8)
// GEMM, tcgen05 fast-path BUILD-VERIFY translation unit (BACKEND-CUDA-SM100,
// ROAD-V1-D1-CUDA, datacenter fast-path §9 DC3 — the Blackwell C3x scaled-mm leg).
//
// SIGNAL: DERIVED+BUILD-VERIFIED (testing-welcome). This TU is a faithful 1:1
// type-port of vLLM's Sm100 C3x fp8 scaled-mm kernel — it compiles + emits real
// sm_100a SASS on the existing GB10 box (single-arch cross-compile), but NO
// B200/sm_100 board has ever RUN it here. A green compile + a cuobjdump SASS
// proof is NOT execution evidence, NOT runtime support, and NOT vLLM-competitive.
// See .agents/specs/cuda-arch-datacenter-fastpath.md §0/§6a/§9 (DC3) and
// backend-matrix.md row BACKEND-CUDA-SM100.
//
// GROUNDING (ground-every-impl rule). The GEMM type structure is ported 1:1 from
//   vLLM (0.26.0.dev0, source tree @ a4e3cb40 — the on-box `~/vllm-src` checkout;
//   the DC1/DC2 bricks cite the pinned oracle build @ 5559679229, the Sm100 C3x
//   fp8 kernel is structurally identical across the two)
//   csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_sm100_fp8_dispatch.cuh
//     :18-97   (template cutlass_3x_gemm_sm100_fp8 — the collective structure)
//     :99-194  (the sm100_fp8_config_{default,M256,M64_swap_ab,M64,M16_swap_ab}
//               tile/cluster configs)
//   entry csrc/.../c3x/scaled_mm_sm100_fp8.cu:1-40 (cutlass_scaled_mm_sm100_fp8).
// The datacenter 5th-gen tensor-core collective (tcgen05 UMMA + dedicated tensor
// memory + 1SM/2SM cluster MMA + TMA warp-specialized producer/consumer pipeline)
// lives INSIDE CUTLASS 4.5.0, selected by
//   ArchTag = cutlass::arch::Sm100 + OpClassTensorOp +
//   KernelSchedule   = cutlass::gemm::collective::KernelScheduleAuto
//   EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto
//   (cutlass::gemm::collective::CollectiveBuilder picks the tcgen05 mainloop) —
// this is the DISTINCT leg both vs our shipped consumer sm_12x body (ArchTag=Sm120
// + KernelTmaWarpSpecializedCooperative, cuda_matmul_fp8_cutlass.cu) AND vs the
// Hopper leg (ArchTag=Sm90 + explicit KernelTmaWarpSpecialized*FP8FastAccum wgmma,
// cuda_scaled_mm_c3x_sm90.cu). On Sm100 vLLM does NOT name an explicit FP8FastAccum
// schedule — it hands KernelScheduleAuto to the CollectiveBuilder, which is what
// makes CUTLASS emit the Blackwell tcgen05 block-scaled-less fp8 mainloop; the
// tile/cluster shapes are vLLM's exact choices (spec §10: ArchTag alone is
// insufficient; the 1SM/2SM ClusterShape is load-bearing — the datacenter default
// is a 2x2 cluster, the tcgen05 2SM MMA, NOT the Hopper 2x1). This mirrors the
// sm_100a NVFP4 tcgen05 brick's lesson (cuda_matmul_nvfp4_sm100.cu): port vLLM's
// Sm100 KernelScheduleAuto structure, not the Sm90 config.
//
// DEVIATION (recorded, mirrors cuda_scaled_mm_c3x_sm90.cu / cuda_matmul_fp8_cutlass.cu).
// vLLM applies the two per-tensor scales through its ScaledEpilogue EVT
// (out = scale_a·(scale_b·acc)), which pulls in vLLM's cutlass_extensions epilogue
// headers (not vendored here). For a BUILD-VERIFY TU whose purpose is to emit the
// tcgen05 collective SASS, we instantiate the plain CUTLASS LinearCombination
// epilogue (alpha_ptr on device — the default the CollectiveBuilder selects when no
// EVT is supplied, exactly as the Sm90 and NVFP4-Sm100 build-verify TUs do)
// instead — the two per-tensor scalars collapse to one accumulator multiply
// out = alpha·acc, exactly the fold our shipped sm_12x fp8 drop-in already uses.
// The EVT is vLLM DISPATCH glue; the tcgen05 MAINLOOP collective — the thing this
// TU exists to build-verify — is byte-for-byte the Sm100 config. We likewise omit
// vLLM's compile-time swap_ab operand transpose (a low-M throughput dispatch knob,
// not a distinct collective): every config here is the non-swapped operand order,
// which exercises the identical KernelScheduleAuto tcgen05 mainloop.
//
// The whole body is guarded on BOTH the build gate VT_SCALEDMM_C3X_SM100 (set by
// the `scaledmm-c3x-sm100` FEATURE-TABLE cell, enabled ONLY for 100a) and the
// CUTLASS capability macro CUTLASS_ARCH_MMA_SM100_SUPPORTED (the analog of vLLM's
// enable_sm100_to_sm120 guard, and the same macro the NVFP4-Sm100 TU uses). On the
// gate arch sm_121a NEITHER is defined, so this TU is not even compiled (the CMake
// source is added only for a 100a target) → byte-zero impact on GB10.
#if defined(VT_SCALEDMM_C3X_SM100)

#include <cuda_runtime.h>

#include <cstddef>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "vt/cuda/cutlass_params_launch.cuh"

namespace vt::cuda::scaled_mm_c3x_sm100 {

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

using namespace cute;

// vLLM scaled_mm_sm100_fp8_dispatch.cuh:99-194 — the five tile/cluster configs
// vLLM selects by GEMM shape. On Sm100 EVERY config uses the SAME
// (KernelScheduleAuto, EpilogueScheduleAuto) pair — the only variation is the
// tile/cluster shape (and, for the low-M configs, the swap_ab operand transpose we
// omit per the DEVIATION note). We port a covering subset that exercises BOTH the
// datacenter 2SM cluster (ClusterShape<_2,_2,_1> / <_2,_1,_1>) and the 1SM cluster
// (<_1,_1,_1>) so the archive carries the tcgen05 SASS of both cluster modes:
//   * _default   (M in (256, inf))   — 2x2 cluster, the tcgen05 2SM MMA
//   * _M256      (M in (64, 256])     — 2x1 cluster
//   * _M64       (M = 64, K < 4096)   — 1x1 cluster, 1SM MMA
// The two remaining vLLM configs (M64_swap_ab, M16_swap_ab) reuse this exact
// schedule pair with a different tile/cluster and the swap_ab transpose, adding
// nothing to the emitted collective family, so they are documented here rather
// than separately instantiated.
struct sm100_fp8_config_default {
  // vLLM sm100_fp8_config_default:99-115
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_256, _128, _128>;
  using ClusterShape = Shape<_2, _2, _1>;
};

struct sm100_fp8_config_M256 {
  // vLLM sm100_fp8_config_M256:117-133
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
};

struct sm100_fp8_config_M64 {
  // vLLM sm100_fp8_config_M64:156-173 (the 1SM low-M tile, no swap-AB)
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_64, _64, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
};

// vLLM scaled_mm_sm100_fp8_dispatch.cuh:18-97 — cutlass_3x_gemm_sm100_fp8. e4m3
// A/B, RowMajor A / ColumnMajor B, f32 accumulate, ArchTag=Sm100, OpClassTensorOp;
// CUTLASS's CollectiveBuilder emits the tcgen05 warp-specialized mainloop +
// epilogue for the KernelScheduleAuto/EpilogueScheduleAuto pair. (Plain
// LinearCombination epilogue + non-swapped operands — see the DEVIATION note
// above; the mainloop collective is vLLM's Sm100 config 1:1.)
template <typename Config, typename OutType>
struct Cutlass3xGemmSm100Fp8 {
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

  using ArchTag = cutlass::arch::Sm100;
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

  // vLLM's GemmKernel wraps this in enable_sm100_to_sm120<> (an __CUDA_ARCH__
  // guard) with a `void` TileScheduler; this TU compiles only into a single-arch
  // 100a archive, so we use GemmUniversal directly with the same `void` scheduler.
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Host-side workspace probe — instantiates the collective's host path.
template <typename Config>
size_t WorkspaceProbe(int m, int n, int k) {
  using Gemm = typename Cutlass3xGemmSm100Fp8<Config, cutlass::bfloat16_t>::Gemm;
  typename Gemm::Arguments args;
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = make_shape(m, n, k, 1);
  return Gemm::get_workspace_size(args);
}

// Reference the full run/can_implement path so the tcgen05 device kernel is
// EMITTED into the archive (not just host-side size math) — this is the
// DERIVED+BUILD-VERIFIED SASS proof (cuobjdump -lelf). NEVER called on GB10 (the
// TU is compiled only into a single-arch 100a archive). A real B200 board upgrades
// the SIGNAL to RUNTIME-VERIFIED by wiring this into the fp8 scaled-mm tactic
// registry + gating token-exact (spec §5/§6b).
template <typename Config>
cutlass::Status RunProbe(void* d, const void* a, const void* b,
                         const float* alpha, int m, int n, int k,
                         void* workspace, cudaStream_t stream) {
  using GemmT = typename Cutlass3xGemmSm100Fp8<Config, cutlass::bfloat16_t>::Gemm;
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
  // its alignment (cutlass_params_launch.cuh). The 2x2x1 / 2x1x1 configs go
  // through ClusterLauncher there.
  const cudaError_t err =
      cutlass_launch::Run<typename GemmT::GemmKernel>(args, workspace, stream, &s);
  if (s != cutlass::Status::kSuccess) return s;
  return err == cudaSuccess ? cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
}

// Explicit instantiations across the three cluster-mode variants so each tcgen05
// kernel variant lands in the archive with sm_100a SASS.
template size_t WorkspaceProbe<sm100_fp8_config_default>(int, int, int);
template size_t WorkspaceProbe<sm100_fp8_config_M256>(int, int, int);
template size_t WorkspaceProbe<sm100_fp8_config_M64>(int, int, int);
template cutlass::Status RunProbe<sm100_fp8_config_default>(
    void*, const void*, const void*, const float*, int, int, int, void*,
    cudaStream_t);
template cutlass::Status RunProbe<sm100_fp8_config_M256>(
    void*, const void*, const void*, const float*, int, int, int, void*,
    cudaStream_t);
template cutlass::Status RunProbe<sm100_fp8_config_M64>(
    void*, const void*, const void*, const float*, int, int, int, void*,
    cudaStream_t);

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

}  // namespace vt::cuda::scaled_mm_c3x_sm100

#endif  // VT_SCALEDMM_C3X_SM100
