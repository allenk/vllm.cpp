// vllm.cpp — launch a CUTLASS 3.x kernel through an entry point whose formal
// parameter is plain bytes, so the same source builds and runs under GCC and
// MSVC.
//
// WHY THIS EXISTS. Every sm90/sm120 CUTLASS GEMM keeps its TMA descriptors
// (`CUtensorMap`) inside `GemmKernel::Params` and reads them straight out of
// kernel-parameter space; the first thing the mainloop does is
// `prefetch.tensormap [&params.tma_load_a]`, which needs a 64-byte aligned
// address. On MSVC that fails twice over, for the reasons spelled out in
// `tma_kernel_param.cuh`:
//
//  1. Without `/Zc:__cplusplus`, `CUtensorMap` loses `alignas(128)` and the
//     descriptors land at 0x18 / 0xa0 / 0x248 instead of 0x80 / 0x180 / 0x480.
//     The build succeeds and the kernel faults ("misaligned address", which
//     CUTLASS reports as `Error Internal`). CUTLASS's own CMakeLists adds the
//     flag for MSVC; this tree consumes CUTLASS as headers only, so CMakeLists
//     adds it per TU (`_VT_CUTLASS_TMA_SOURCES`).
//  2. With the flag, `cutlass::device_kernel<Op>(Params const params)` cannot be
//     compiled: its host stub takes a 128-aligned parameter by value (C2719).
//
// `ParamsBytesKernel` is `cutlass::device_kernel` with the parameter declared as
// bytes of `sizeof(Params)`, aligned like `Params` in the device pass and to 8 in
// the host pass. `Run` replaces `GemmUniversalAdapter::initialize` + `run`, which
// are the only adapter members that name `cutlass::device_kernel`; the adapter's
// static `can_implement` and `get_workspace_size` remain usable. GCC builds take
// the same path, so there is one launch path on both platforms.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "cutlass/cutlass.h"
#include "vt/cuda/tma_kernel_param.cuh"

namespace vt::cuda::cutlass_launch {

// Below 64 means this TU was built without `/Zc:__cplusplus` on MSVC, and every
// TMA kernel in it would carry misaligned descriptors.
static_assert(alignof(CUtensorMap) >= 64,
              "CUtensorMap is not 64-byte aligned in this translation unit: on MSVC it must be "
              "compiled with -Xcompiler=/Zc:__cplusplus, otherwise cuda.h drops alignas(128) and "
              "every TMA descriptor in a CUTLASS Params lands at a misaligned offset");

template <class Kernel>
struct alignas(kKernelParamAlign<alignof(typename Kernel::Params)>) ParamsBytes {
  unsigned char bytes[sizeof(typename Kernel::Params)];
};

template <class Kernel>
__global__ void
#ifdef __CUDACC__
__launch_bounds__(Kernel::MaxThreadsPerBlock, Kernel::MinBlocksPerMultiprocessor)
#endif
ParamsBytesKernel(const __grid_constant__ ParamsBytes<Kernel> raw) {
  using Params = typename Kernel::Params;
  // The declaration above makes the driver honour this; trap rather than run on
  // a misaligned descriptor if that ever stops being true.
  if (reinterpret_cast<uintptr_t>(&raw) % alignof(Params) != 0) __trap();
  extern __shared__ char smem[];
  Kernel op;
  op(*reinterpret_cast<const Params*>(&raw), smem);
}

// `GemmUniversalAdapter::initialize` + `run` for a kernel launched without a
// cluster and without PDL. Returns the CUDA error instead of folding it into
// `Status::kErrorInternal`, so a failure names its cause; `*status` carries the
// CUTLASS status of workspace initialisation.
template <class Kernel>
cudaError_t Run(const typename Kernel::Arguments& args, void* workspace, cudaStream_t stream,
                cutlass::Status* status) {
  using Params = typename Kernel::Params;
  static_assert(sizeof(ParamsBytes<Kernel>) == sizeof(Params), "blob must mirror Params exactly");
  static_assert(Kernel::ArchTag::kMinComputeCapability != 90,
                "sm90 kernels launch with a cluster; this path does not handle cluster launch");

  *status = Kernel::initialize_workspace(args, workspace, stream);
  if (*status != cutlass::Status::kSuccess) return cudaSuccess;

  const Params params = Kernel::to_underlying_arguments(args, workspace);
  const dim3 grid = Kernel::get_grid_shape(params);
  const dim3 block = Kernel::get_block_shape();
  const int smem_size = Kernel::SharedStorageSize;
  const void* fn = reinterpret_cast<const void*>(&ParamsBytesKernel<Kernel>);

  if (smem_size >= (48 << 10)) {
    const cudaError_t err =
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (err != cudaSuccess) return err;
  }
  void* kernel_args[] = {const_cast<Params*>(&params)};
  return cudaLaunchKernel(fn, grid, block, kernel_args, static_cast<size_t>(smem_size), stream);
}

}  // namespace vt::cuda::cutlass_launch
