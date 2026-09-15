// Exact runtime selector for the vendored Triton AOT CUDA trees shipped in the
// release fat binary. The remaining release SMs use the portable CUDA
// implementation; a cubin is never tried on a merely similar architecture.
//
// This fork ships SEVEN trees where upstream ships six: sm_120 (12.0) is ours,
// for the RTX PRO 6000 Blackwell this project develops on. The count is named
// ONCE below and every array arity derives from it, because the arity being
// written out by hand is exactly how upstream's own dispatch test arrived
// asserting six on a tree that has seven.
#pragma once

#include <array>
#include <cstddef>
#include <utility>

namespace vt::cuda {

// The number of vendored trees. TritonAotTreeIndex returns [0, kTritonAotTreeCount).
inline constexpr std::size_t kTritonAotTreeCount = 7;

inline constexpr int TritonAotTreeIndex(int major, int minor) {
  if (major == 8 && minor == 0) return 0;
  if (major == 8 && minor == 6) return 1;
  if (major == 8 && minor == 9) return 2;
  if (major == 9 && minor == 0) return 3;
  if (major == 10 && minor == 0) return 4;
  // sm_120 (compute_cap 12.0) added alongside sm_121. This table is a SECOND
  // hard-coded arch list, independent of VT_TRITON_AOT_AVAILABLE_ARCHES in
  // cmake/TritonAOTMultiArch.cmake, and nothing forces the two to agree: adding a
  // tree to the CMake list embeds its cubins but leaves this returning -1, so
  // TritonAotAvailableOnCurrentDevice() stays false and the fast path is silently
  // skipped on that device. Measured on an RTX PRO 6000 Blackwell (12.0).
  if (major == 12 && minor == 0) return 5;
  if (major == 12 && minor == 1) return 6;
  return -1;
}

template <typename Result, typename Function, typename... Args>
Result DispatchTritonAot(int major, int minor, Result fallback,
                         const std::array<Function, kTritonAotTreeCount>& trees,
                         Args&&... args) {
  const int index = TritonAotTreeIndex(major, minor);
  if (index < 0) return fallback;
  return trees[static_cast<std::size_t>(index)](
      std::forward<Args>(args)...);
}

template <typename Function, typename... Args>
bool DispatchTritonAotVoid(int major, int minor,
                           const std::array<Function, kTritonAotTreeCount>& trees,
                           Args&&... args) {
  const int index = TritonAotTreeIndex(major, minor);
  if (index < 0) return false;
  trees[static_cast<std::size_t>(index)](std::forward<Args>(args)...);
  return true;
}

}  // namespace vt::cuda
