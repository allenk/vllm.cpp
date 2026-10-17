// vllm.cpp — datacenter-Blackwell (sm_100a) NVFP4 block-scaled GEMM, tcgen05
// fast-path BUILD-VERIFY translation unit (BACKEND-CUDA-SM100, ROAD-V1-D1-CUDA).
//
// SIGNAL: DERIVED+BUILD-VERIFIED (testing-welcome). This TU is a faithful 1:1
// type-port of vLLM's datacenter NVFP4 scaled-mm kernel — it compiles + emits
// real sm_100a SASS on the existing GB10 box (single-arch cross-compile), but NO
// B200/sm_100 board has ever RUN it here. A green compile + a cuobjdump SASS
// proof is NOT execution evidence, NOT runtime support, and NOT vLLM-competitive.
// See .agents/specs/cuda-arch-datacenter-fastpath.md §0/§6a and backend-matrix.md
// row BACKEND-CUDA-SM100.
//
// GROUNDING (ground-every-impl rule). The GEMM type structure is ported 1:1 from
//   vLLM @ 5559679229 (0.26.0.dev0)
//   csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_kernels.cu:44-138
//   (struct sm100_fp4_config_{default,M256,M16} + template Fp4GemmSm100).
// The datacenter tcgen05 collective (5th-gen tensor core: UMMA + dedicated tensor
// memory + 1SM/2SM cluster MMA) lives inside CUTLASS 4.5.0, selected by
//   ArchTag = cutlass::arch::Sm100 + KernelScheduleAuto/EpilogueScheduleAuto
//   (cutlass::gemm::collective::CollectiveBuilder picks the tcgen05 mainloop) —
// this is the DISTINCT leg vs our shipped consumer sm_12x body, which uses
// ArchTag = Sm120 + KernelTmaWarpSpecializedCooperative
// (src/vt/cuda/nvfp4_cutlass_tactic_impl.cuh:35-80, FlashInfer sm120 template).
// e2m1 packing + ue4m3 block-scale (SFA/SFB) layout via Sm1xxBlkScaledConfig is
// SHARED between the two legs (.agents/specs/cuda-arch-datacenter-fastpath.md §2).
// The tile/cluster/PerSm shapes are vLLM's exact choices (spec §10 risk #1: the
// ArchTag alone is insufficient; the tile/cluster config is load-bearing).
//
// The whole body is guarded on BOTH the build gate VT_CUTLASS_NVFP4_SM100 (set by
// the `cutlass-nvfp4-sm100` FEATURE-TABLE cell, enabled ONLY for 100a) and the
// CUTLASS capability macro CUTLASS_ARCH_MMA_SM100_SUPPORTED (vLLM's own guard).
// On the gate arch sm_121a NEITHER is defined, so this TU is not even compiled
// (the CMake source is added only for a 100a target) → byte-zero impact on GB10.
#if defined(VT_CUTLASS_NVFP4_SM100)

#include <cuda_runtime.h>

#include <cstddef>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "vt/cuda/cutlass_params_launch.cuh"

namespace vt::cuda::nvfp4_sm100 {

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

using namespace cute;

// vLLM nvfp4_scaled_mm_kernels.cu:44-72 — the three datacenter tile configs. The
// tcgen05 collective is 2SM-cluster (ClusterShape<_2,_1,_1>) for the large-M
// configs, so the per-SM tile is half the MMA tile in M (PerSmTileShape_MNK).
struct sm100_fp4_config_default {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileScheduler = cutlass::gemm::PersistentScheduler;
  using TileShape = Shape<_256, _256, _256>;
  using ClusterShape = Shape<_2, _1, _1>;
  using PerSmTileShape_MNK = Shape<_128, _256, _256>;
};

struct sm100_fp4_config_M256 {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileScheduler = void;
  using TileShape = Shape<_256, _128, _256>;
  using ClusterShape = Shape<_2, _1, _1>;
  using PerSmTileShape_MNK = Shape<_128, _128, _256>;
};

struct sm100_fp4_config_M16 {
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileScheduler = void;
  using TileShape = Shape<_128, _128, _256>;
  using ClusterShape = Shape<_1, _1, _1>;
  using PerSmTileShape_MNK = Shape<_128, _128, _256>;
};

// vLLM nvfp4_scaled_mm_kernels.cu:74-138 — Fp4GemmSm100<Config, OutType>. e2m1
// A/B, ue4m3 block-scale, RowMajor/ColumnMajor, ArchTag=Sm100,
// OpClassBlockScaledTensorOp; CUTLASS builds the tcgen05 mainloop/epilogue.
template <typename Config, typename OutType>
struct Fp4GemmSm100 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementD = OutType;
  using ElementC = OutType;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Config::TileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
          typename Config::EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB, LayoutBTag, AlignmentB,
          ElementAccumulator, MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          typename Config::KernelSchedule>::CollectiveOp;

  using TileScheduler = typename Config::TileScheduler;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                          CollectiveMainloop, CollectiveEpilogue,
                                                          TileScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Materialize the datacenter tcgen05 GEMM device kernels so the TU carries real
// sm_100a SASS (the DERIVED+BUILD-VERIFIED proof, cuobjdump -lelf). This entry is
// NEVER called on GB10 (the TU is compiled only into a single-arch 100a archive);
// it exists so the CUTLASS kernel symbol is instantiated and emitted. A real
// board upgrades the SIGNAL to RUNTIME-VERIFIED by wiring this into the NVFP4
// tactic registry + gating token-exact (spec §5/§6b).
template <typename Config>
size_t WorkspaceProbe(int m, int n, int k) {
  using Gemm = typename Fp4GemmSm100<Config, cutlass::bfloat16_t>::Gemm;
  typename Gemm::Arguments args;
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = make_shape(m, n, k, 1);
  return Gemm::get_workspace_size(args);
}

// Reference the full run/can_implement path of every config so the tcgen05
// collective's device kernel is emitted (not just host-side size math).
template <typename Config>
cutlass::Status RunProbe(void* d, const void* a, const void* b, const void* a_sf, const void* b_sf,
                         const float* alpha, int m, int n, int k, void* workspace,
                         cudaStream_t stream) {
  using GemmT = typename Fp4GemmSm100<Config, cutlass::bfloat16_t>::Gemm;
  using ElementD = typename GemmT::ElementD;
  typename GemmT::Arguments args;
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.epilogue.thread.alpha_ptr = alpha;
  args.problem_shape = make_shape(m, n, k, 1);
  args.mainloop.ptr_A = static_cast<const cutlass::float_e2m1_t*>(a);
  args.mainloop.ptr_B = static_cast<const cutlass::float_e2m1_t*>(b);
  args.mainloop.ptr_SFA = static_cast<const cutlass::float_ue4m3_t*>(a_sf);
  args.mainloop.ptr_SFB = static_cast<const cutlass::float_ue4m3_t*>(b_sf);
  args.epilogue.ptr_C = static_cast<const ElementD*>(d);
  args.epilogue.ptr_D = static_cast<ElementD*>(d);
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

// Explicit instantiations across the three datacenter configs so each tcgen05
// kernel variant lands in the archive with sm_100a SASS.
template size_t WorkspaceProbe<sm100_fp4_config_default>(int, int, int);
template size_t WorkspaceProbe<sm100_fp4_config_M256>(int, int, int);
template size_t WorkspaceProbe<sm100_fp4_config_M16>(int, int, int);
template cutlass::Status RunProbe<sm100_fp4_config_default>(void*, const void*, const void*,
                                                            const void*, const void*, const float*,
                                                            int, int, int, void*, cudaStream_t);
template cutlass::Status RunProbe<sm100_fp4_config_M256>(void*, const void*, const void*,
                                                         const void*, const void*, const float*, int,
                                                         int, int, void*, cudaStream_t);
template cutlass::Status RunProbe<sm100_fp4_config_M16>(void*, const void*, const void*, const void*,
                                                        const void*, const float*, int, int, int,
                                                        void*, cudaStream_t);

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED

}  // namespace vt::cuda::nvfp4_sm100

#endif  // VT_CUTLASS_NVFP4_SM100
