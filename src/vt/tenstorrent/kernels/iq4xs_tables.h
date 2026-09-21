// SPDX-License-Identifier: Apache-2.0
//
// IQ4_XS codebook tables for the TT int8-dot device kernel
// (tenstorrent-gsq-keepquant wave 2). The device include path is this
// kernels/ directory ONLY (KeepQuantKernelIncludeDir), so the table lives
// here as a copy of the CPU original:
//
//   kValuesIq4nl[16]    src/vt/cpu/cpu_quant_iq_tables.h:1254
//
// (kKmaskIq2xs / kKsignsIq2xs are NOT needed: IQ4_XS has no sign array — the
// nibble is a plain codebook index.)
//
// DRIFT GUARD: there is no shared header (the CPU header is not on the
// device include path), so the copy is pinned byte-for-byte by the op-level
// oracle test — tests/vt/test_tenstorrent_backend.cpp's int8-dot sweep and
// the default-path decode leg run the device dot against
// vt::cpu::VecDotIQ4_XSQ8_K bit-exactly; any table divergence in either
// direction reddens them.

#pragma once

#include <cstdint>

// The 16-entry IQ4_NL non-linear codebook, kvalues_iq4nl (llama.cpp @ b10451
// ggml-common.h:1120; CPU original src/vt/cpu/cpu_quant_iq_tables.h:1254,
// which carries the extraction provenance). The nibble is an INDEX here, not
// a quant. Extracted mechanically from the pinned blob, not transcribed.
inline constexpr int8_t kValuesIq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};
