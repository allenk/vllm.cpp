// The FlashAttention-2 head dims this build instantiated.
//
// CMakeLists.txt defines VT_FA2_HDIM_<d>=1 for every entry of
// VLLM_CPP_FA2_HEAD_DIMS (64;128;192;256 by default) and compiles only the
// vendored kernel sources of those dims, so a dim it leaves out has no kernel to
// launch. Every FA-2 lane gate asks Fa2HeadDimBuilt() beside its head-dim test
// and falls through to the lane a build without FA-2 takes; the launchers in
// cuda_flash_attn_fa2.cu refuse a dim that is not built, so reaching one is a
// gate defect rather than a silent wrong answer.
#pragma once

#include <cstdint>

#ifndef VT_FA2_HDIM_64
#define VT_FA2_HDIM_64 0
#endif
#ifndef VT_FA2_HDIM_128
#define VT_FA2_HDIM_128 0
#endif
#ifndef VT_FA2_HDIM_192
#define VT_FA2_HDIM_192 0
#endif
#ifndef VT_FA2_HDIM_256
#define VT_FA2_HDIM_256 0
#endif

namespace vt::cuda {

constexpr bool Fa2HeadDimBuilt(int64_t d) {
  return (d == 64 && VT_FA2_HDIM_64 != 0) || (d == 128 && VT_FA2_HDIM_128 != 0) ||
         (d == 192 && VT_FA2_HDIM_192 != 0) || (d == 256 && VT_FA2_HDIM_256 != 0);
}

}  // namespace vt::cuda
