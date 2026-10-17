// vllm.cpp — kernel parameters that must sit on a 64/128-byte boundary in
// parameter space (TMA descriptors, or structs that contain them), declared so
// that both GCC and MSVC builds compile them and the driver places them right.
//
// THE CONSTRAINT. `prefetch.tensormap` and `cp.async.bulk.tensor` read a
// `CUtensorMap` straight from kernel-parameter space, and the hardware requires
// that address to be 64-byte aligned. A misaligned one faults the kernel on its
// first descriptor access ("misaligned address").
//
// WHY A KERNEL PARAMETER CANNOT SIMPLY BE `CUtensorMap` ON WINDOWS.
//  1. `cuda.h` gives `CUtensorMap` `alignas(128)` only when
//     `__cplusplus >= 201103L`. cl.exe reports 199711L unless the TU is built
//     with `/Zc:__cplusplus`, and nvcc's device front end follows it, so by
//     default the descriptor is 8-aligned on both sides: the build succeeds and
//     the kernel faults. (`GdnChunkDeltaHTmaKernel` put its descriptors at
//     parameter offsets 0x68 / 0xe8 this way.)
//  2. With `/Zc:__cplusplus`, x64 cl.exe rejects any by-value formal parameter
//     aligned above 8 (C2719), and the host stub nvcc generates for a kernel is
//     exactly that.
//
// THE DECLARATION THAT WORKS IN BOTH. The parameter type is plain bytes whose
// alignment differs between nvcc's two compilation passes:
//  - DEVICE pass (`__CUDA_ARCH__` defined): the real alignment. The kernel's
//    parameter metadata carries it, and the driver places the parameter on that
//    boundary for every launch, CUDA graph kernel nodes included. Relying on the
//    parameter buffer happening to start aligned is NOT enough: measured on
//    Windows, a direct launch landed on 128 and a graph-captured one did not.
//  - HOST pass: at most 8, so the generated stub is legal under cl.exe. The host
//    never reads that alignment; `<<<>>>` and `cudaLaunchKernel` copy the bytes
//    to wherever the device-side metadata says.
// The size is the same in both passes, so the two passes agree on every offset.
#pragma once

#include <cuda.h>

#include <cstddef>

namespace vt::cuda {

template <size_t DeviceAlign>
inline constexpr size_t kKernelParamAlign =
#if defined(__CUDA_ARCH__)
    DeviceAlign;
#else
    (DeviceAlign < 8 ? DeviceAlign : 8);
#endif

// A `CUtensorMap` passed by value to a kernel. Device code hands `&param` to the
// TMA instruction; host code builds a `CUtensorMap` and passes `AsTmaParam(map)`.
struct alignas(kKernelParamAlign<128>) TmaDescriptorParam {
  unsigned char bytes[128];
};
static_assert(sizeof(TmaDescriptorParam) == sizeof(CUtensorMap),
              "TmaDescriptorParam must mirror CUtensorMap byte for byte");

inline const TmaDescriptorParam& AsTmaParam(const CUtensorMap& map) {
  return *reinterpret_cast<const TmaDescriptorParam*>(&map);
}

}  // namespace vt::cuda
