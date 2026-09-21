// Q8_0 repack tier (CIQ G7): the repacked-layout GEMM, one driver, kernels per ISA.
//
// Renamed from cpu_quant_repack_arm.cpp 2026-09-20. The file used to hold the
// Arm kernels AND the driver inside `#if __aarch64__ && __ARM_FEATURE_MATMUL_INT8`,
// which made a second architecture cost a copy of ~80 architecture-free lines.
// The driver now sits outside; an ISA supplies `GemmTileQ8_0` (prefill, 4x4) and
// `GemvRowQ8_0` (decode, M=1) and nothing else.
//
// The Arm kernels are unchanged, line for line: the tier-0 bit-identity contract
// (non-fused vmla) lives in their arithmetic, and a retype would not show up in
// review.
//
// The x86 kernels build the same 4x4 from `maddubs`, since x86 has no `vmmlaq_s32`:
// broadcasting one activation row across the four 64-bit lanes lines it up
// against the four interleaved weight rows in one operation. Validated against
// the plain path before landing -- integer dots exact on 196,608 of 196,608
// blocks, tile within 2.8e-07 of AVX2-plain (fp32 ordering). Measured there:
// decode 1.40x, prefill(M=4) 1.66x over an AVX2 plain path, single-threaded and
// cache-resident, so the engine-level number will be smaller.
//
// x86 IS OPT-IN (`VT_CPU_QUANT_REPACK=1`). Landing the tier and changing the
// default are separate decisions: repack COPIES the weight slice rather than
// borrowing the mmap, so enabling it raises the loader's peak footprint, and
// that peak already measures ~2.4x the model on a 7 GB board.

#include "vt/quant.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cpu_quant_blocks.h"
#include "cpu_quant_repack.h"
#include "cpu_threadpool.h"
#include "vt/tensor.h"

#if defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
#define VT_REPACK_ARM 1
#include <arm_neon.h>
#include "vt/cpu/cpu_isa_arm.h"
#elif defined(__AVX2__)
#define VT_REPACK_X86 1
#include <immintrin.h>
#endif

namespace vt::cpu {
namespace {

float LoadActF32(const Tensor& t, int64_t elem_offset) {
  switch (t.dtype) {
    case DType::kF32: return t.Ptr<float>()[elem_offset];
    case DType::kF16: return F16ToF32(t.Ptr<uint16_t>()[elem_offset]);
    case DType::kBF16: return BF16ToF32(t.Ptr<uint16_t>()[elem_offset]);
    default:
      VT_CHECK(false, "quant_repack: unsupported activation dtype");
      return 0.0f;
  }
}

void StoreOutF32(const Tensor& t, int64_t elem_offset, float v) {
  switch (t.dtype) {
    case DType::kF32: t.Ptr<float>()[elem_offset] = v; break;
    case DType::kBF16: t.Ptr<uint16_t>()[elem_offset] = F32ToBF16(v); break;
    default: VT_CHECK(false, "quant_repack: unsupported output dtype");
  }
}

#if defined(VT_REPACK_ARM)

// ggml_gemm_q8_0_4x8_q8_0 (NEON+i8mm branch, repack.cpp:5091-5155), restricted
// to a range of weight COLUMN-groups [xg0, xg1) so the driver can parallelize
// over them. Runs the full set of activation ROW-groups (mgroups). `wgroups` and
// `agroups` are the repacked weight / interleaved activation, laid out as
// group-major (each group = `nblocks` BlockQ8_0x4). `s` is f32 [*, N] row stride
// `bs`. Non-fused vmla — see the file header (bit-identity).
void GemmTileQ8_0(float* s, int64_t bs, const BlockQ8_0x4* wgroups,
                  const BlockQ8_0x4* agroups, int64_t mgroups, int64_t nblocks,
                  int64_t xg0, int64_t xg1) {
  for (int64_t yg = 0; yg < mgroups; ++yg) {
    const BlockQ8_0x4* a_base = agroups + yg * nblocks;
    for (int64_t xg = xg0; xg < xg1; ++xg) {
      const BlockQ8_0x4* b_ptr = wgroups + xg * nblocks;
      const BlockQ8_0x4* a_ptr = a_base;

      float32x4_t acc_f32[4];
      for (int i = 0; i < 4; i++) acc_f32[i] = vdupq_n_f32(0);

      for (int64_t bi = 0; bi < nblocks; ++bi) {
        int32x4_t acc[4];
        for (int i = 0; i < 4; i++) acc[i] = vdupq_n_s32(0);

        for (int chunk = 0; chunk < 4; chunk++) {
          int8x16_t a01 = vld1q_s8(a_ptr->qs + chunk * 32);
          int8x16_t a23 = vld1q_s8(a_ptr->qs + chunk * 32 + 16);
          int8x16_t b01 = vld1q_s8(b_ptr->qs + chunk * 32);
          int8x16_t b23 = vld1q_s8(b_ptr->qs + chunk * 32 + 16);

          acc[0] = vmmlaq_s32(acc[0], a01, b01);
          acc[1] = vmmlaq_s32(acc[1], a01, b23);
          acc[2] = vmmlaq_s32(acc[2], a23, b01);
          acc[3] = vmmlaq_s32(acc[3], a23, b23);
        }

        int32x4_t row0 = vcombine_s32(vget_low_s32(acc[0]), vget_low_s32(acc[1]));
        int32x4_t row1 = vcombine_s32(vget_high_s32(acc[0]), vget_high_s32(acc[1]));
        int32x4_t row2 = vcombine_s32(vget_low_s32(acc[2]), vget_low_s32(acc[3]));
        int32x4_t row3 = vcombine_s32(vget_high_s32(acc[2]), vget_high_s32(acc[3]));

        float32x4_t a_d = vcvt_f32_f16(vld1_f16((const __fp16*)a_ptr->d));
        float32x4_t b_d = vcvt_f32_f16(vld1_f16((const __fp16*)b_ptr->d));

        acc_f32[0] = vmlaq_f32(acc_f32[0], vcvtq_f32_s32(row0), vmulq_laneq_f32(b_d, a_d, 0));
        acc_f32[1] = vmlaq_f32(acc_f32[1], vcvtq_f32_s32(row1), vmulq_laneq_f32(b_d, a_d, 1));
        acc_f32[2] = vmlaq_f32(acc_f32[2], vcvtq_f32_s32(row2), vmulq_laneq_f32(b_d, a_d, 2));
        acc_f32[3] = vmlaq_f32(acc_f32[3], vcvtq_f32_s32(row3), vmulq_laneq_f32(b_d, a_d, 3));

        a_ptr++;
        b_ptr++;
      }

      const int64_t x = xg * 4;
      for (int row = 0; row < 4; row++) {
        vst1q_f32(s + (yg * 4 + row) * bs + x, acc_f32[row]);
      }
    }
  }
}

// ggml_gemv_q8_0_4x8_q8_0 (NEON+dotprod branch, repack.cpp:1779-1820) for ONE
// activation row (plain BlockQ8_0, NOT interleaved), restricted to weight
// column-groups [xg0, xg1). `s_row` is the f32 output row. Non-fused vmla.
void GemvRowQ8_0(float* s_row, const BlockQ8_0x4* wgroups,
                 const BlockQ8_0* a_row, int64_t nblocks, int64_t xg0,
                 int64_t xg1) {
  for (int64_t xg = xg0; xg < xg1; ++xg) {
    const BlockQ8_0x4* b_ptr = wgroups + xg * nblocks;
    const BlockQ8_0* a_ptr = a_row;
    float32x4_t acc = vdupq_n_f32(0);

    for (int64_t bi = 0; bi < nblocks; ++bi) {
      int8x16x4_t b_low = vld1q_s8_x4((const int8_t*)b_ptr->qs);
      int8x16x4_t b_high = vld1q_s8_x4((const int8_t*)b_ptr->qs + 64);
      float16x4_t bd = vld1_f16((const __fp16*)b_ptr->d);

      int8x8x4_t a_chunks = vld1_s8_x4(a_ptr->qs);
      int8x16_t a0 = vcombine_s8(a_chunks.val[0], a_chunks.val[0]);
      int8x16_t a1 = vcombine_s8(a_chunks.val[1], a_chunks.val[1]);
      int8x16_t a2 = vcombine_s8(a_chunks.val[2], a_chunks.val[2]);
      int8x16_t a3 = vcombine_s8(a_chunks.val[3], a_chunks.val[3]);
      float16x4_t ad = vld1_dup_f16((const __fp16*)&a_ptr->d);

      int32x4_t ret0 = vdupq_n_s32(0);
      int32x4_t ret1 = vdupq_n_s32(0);
      ret0 = vdotq_s32(ret0, b_low.val[0], a0);
      ret1 = vdotq_s32(ret1, b_low.val[1], a0);
      ret0 = vdotq_s32(ret0, b_low.val[2], a1);
      ret1 = vdotq_s32(ret1, b_low.val[3], a1);
      ret0 = vdotq_s32(ret0, b_high.val[0], a2);
      ret1 = vdotq_s32(ret1, b_high.val[1], a2);
      ret0 = vdotq_s32(ret0, b_high.val[2], a3);
      ret1 = vdotq_s32(ret1, b_high.val[3], a3);

      int32x4_t ret = vpaddq_s32(ret0, ret1);
      acc = vmlaq_f32(acc, vcvtq_f32_s32(ret),
                      vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
      a_ptr++;
      b_ptr++;
    }
    vst1q_f32(s_row + xg * 4, acc);
  }
}

#elif defined(VT_REPACK_X86)

// int8 dot of one 32-byte chunk. `aw` is |w| read as UNSIGNED bytes and `sa` is
// `a` carrying w's sign -- which is exactly the (unsigned, signed) operand pair
// VNNI's vpdpbusd wants, so the VNNI path is a drop-in rather than a rewrite.
//
// WHY THIS BRANCH EXISTS. Measured 2026-09-21: this engine emitted ZERO
// vpdpbusd while llama.cpp's CPU backend emitted 677, on a host whose
// /proc/cpuinfo reports both avx_vnni and avx512_vnni. llama.cpp's own
// tinyBLAS carries the identical ladder (llamafile/sgemm.cpp:1754-1764
// `updot`), and ITS `#else` fallback is byte-for-byte the sequence below -- so
// we were running llama.cpp's slow path while it ran its fast one. The prefill
// half of the gap measured 2.76x at the time.
//
// The cause was a flag in our own CMakeLists pinning this TU to -mavx2, with a
// comment justifying it for machines that lack AVX2. That reasoning is sound
// for "do not crash on old hardware" and says nothing about "do not leave new
// hardware unused" -- this project's own rule that a backend's default is not
// its capability, applied to a default we set.
//
// -mavxvnni AND NOT -mavx512vnni, deliberately: the 512-bit flag would also let
// the compiler re-vectorise the whole TU at 512 bits, which changes two things
// at once. The VEX-encoded 256-bit form keeps the vector width identical so the
// only variable is the dot instruction itself.
//
// BIT-IDENTICAL, not merely close. maddubs accumulates into int16 and can
// saturate; dpbusd accumulates into int32 and cannot. For q8_0 the operands are
// bounded by 127, so a pair of products reaches at most 2*127*127 = 32258 <
// 32767 and the int16 path never saturates -- the two routes agree exactly
// rather than approximately.
inline __m256i DotChunk(__m256i w, __m256i a) {
  const __m256i aw = _mm256_sign_epi8(w, w);
  const __m256i sa = _mm256_sign_epi8(a, w);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
  return _mm256_dpbusd_epi32(_mm256_setzero_si256(), aw, sa);
#elif defined(__AVXVNNI__)
  return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), aw, sa);
#else
  const __m256i p16 = _mm256_maddubs_epi16(aw, sa);
  return _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
#endif
}

void GemvRowQ8_0(float* s_row, const BlockQ8_0x4* wgroups,
                 const BlockQ8_0* a_row, int64_t nblocks, int64_t xg0,
                 int64_t xg1) {
  for (int64_t g = xg0; g < xg1; ++g) {
    const BlockQ8_0x4* bp = wgroups + g * nblocks;
    __m128 acc = _mm_setzero_ps();
    for (int64_t b = 0; b < nblocks; ++b, ++bp) {
      const __m256i* wq = reinterpret_cast<const __m256i*>(bp->qs);
      const int64_t* aq = reinterpret_cast<const int64_t*>(a_row[b].qs);
      __m256i s32 = _mm256_setzero_si256();
      for (int c = 0; c < 4; ++c) {
        s32 = _mm256_add_epi32(
            s32, DotChunk(_mm256_loadu_si256(wq + c), _mm256_set1_epi64x(aq[c])));
      }
      // dwords [r0a r0b r1a r1b | r2a r2b r3a r3b] -> [r0 r1 r2 r3]
      const __m256i h = _mm256_hadd_epi32(s32, s32);
      const __m128i lo = _mm256_castsi256_si128(h);              // [r0 r1 r0 r1]
      const __m128i hi = _mm256_extracti128_si256(h, 1);         // [r2 r3 r2 r3]
      const __m128i d4 = _mm_unpacklo_epi64(lo, hi);             // [r0 r1 r2 r3]
      const __m128 wd = _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(bp->d)));
      const __m128 scale = _mm_mul_ps(wd, _mm_set1_ps(F16ToF32(a_row[b].d)));
      acc = _mm_add_ps(acc, _mm_mul_ps(_mm_cvtepi32_ps(d4), scale));
    }
    _mm_storeu_ps(s_row + g * kQ8_0xNrowsInterleaved, acc);
  }
}

void GemmTileQ8_0(float* s, int64_t bs, const BlockQ8_0x4* wgroups,
                        const BlockQ8_0x4* agroups, int64_t mgroups,
                        int64_t nblocks, int64_t xg0, int64_t xg1) {
  for (int64_t yg = 0; yg < mgroups; ++yg) {
    const BlockQ8_0x4* a_base = agroups + yg * nblocks;
    for (int64_t xg = xg0; xg < xg1; ++xg) {
      const BlockQ8_0x4* bp = wgroups + xg * nblocks;
      const BlockQ8_0x4* ap = a_base;
      __m128 accf[4] = {_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(),
                        _mm_setzero_ps()};
      for (int64_t bi = 0; bi < nblocks; ++bi, ++ap, ++bp) {
        __m256i acc[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(),
                          _mm256_setzero_si256(), _mm256_setzero_si256()};
        for (int c = 0; c < 4; ++c) {
          const __m256i bv =
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bp->qs + c * 32));
          const int64_t* arow = reinterpret_cast<const int64_t*>(ap->qs + c * 32);
          for (int i = 0; i < 4; ++i) {
            acc[i] = _mm256_add_epi32(
                acc[i], DotChunk(bv, _mm256_set1_epi64x(arow[i])));
          }
        }
        const __m128 bd =
            _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(bp->d)));
        const __m128 ad =
            _mm_cvtph_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(ap->d)));
        alignas(16) float adv[4];
        _mm_store_ps(adv, ad);
        for (int i = 0; i < 4; ++i) {
          const __m256i h = _mm256_hadd_epi32(acc[i], acc[i]);
          const __m128i d4 = _mm_unpacklo_epi64(_mm256_castsi256_si128(h),
                                                _mm256_extracti128_si256(h, 1));
          accf[i] = _mm_add_ps(
              accf[i], _mm_mul_ps(_mm_cvtepi32_ps(d4),
                                  _mm_mul_ps(bd, _mm_set1_ps(adv[i]))));
        }
      }
      for (int i = 0; i < 4; ++i) _mm_storeu_ps(s + (yg * 4 + i) * bs + xg * 4, accf[i]);
    }
  }
}

#endif

}  // namespace

bool QuantRepackActive() {
#if defined(VT_REPACK_ARM)
  static const bool v = [] {
    const char* e = std::getenv("VT_CPU_QUANT_REPACK");
    bool enabled = false;
    std::string error;
    VT_CHECK(ResolveArmIsaToggle(DetectArmIsaCaps(), ArmIsaTier::kI8mm,
                                e == nullptr ? "auto" : e, &enabled, &error),
             error);
    return enabled;
  }();
  return v;
#elif defined(VT_REPACK_X86)
  // OPT-IN on x86 until the loader-footprint cost of repack-at-load has an
  // engine-level measurement. Same env grammar as the Arm arm.
  static const bool v = [] {
    const char* e = std::getenv("VT_CPU_QUANT_REPACK");
    if (e == nullptr) return false;
    const std::string s(e);
    return s == "1" || s == "on" || s == "true" || s == "x86" || s == "avx2";
  }();
  return v;
#else
  return false;
#endif
}

#if defined(VT_REPACK_ARM) || defined(VT_REPACK_X86)

void QuantRepackMatmul(Tensor& out, const Tensor& a, const Tensor& b) {
  VT_CHECK(b.repacked && b.dtype == DType::kQ8_0,
           "quant_repack_matmul: weight must be a repacked q8_0 tensor");
  const int64_t m = a.shape[0];
  const int64_t k = a.shape[1];
  const int64_t n = b.shape[0];
  const int64_t nblocks = k / kQK8_0;
  const int64_t a_rs = a.stride[0];
  VT_CHECK(n % kQ8_0xNrowsInterleaved == 0 && k % kQK8_0 == 0,
           "quant_repack_matmul: N%4 and K%32 required (repack eligibility)");

  const FromFloatFn from_float = QuantTraits(DType::kQ8_0).from_float;
  const BlockQ8_0x4* wgroups = reinterpret_cast<const BlockQ8_0x4*>(b.data);

  // Output: write f32 directly when possible, else into an f32 scratch that is
  // converted to bf16 at the end. MatmulBTQuant guarantees `out` contiguous, so
  // the f32 row stride is exactly N.
  const bool out_is_f32 = (out.dtype == DType::kF32);
  std::vector<float> out_scratch;
  float* s;
  if (out_is_f32) {
    s = out.Ptr<float>();
  } else {
    out_scratch.resize(static_cast<size_t>(m) * n);
    s = out_scratch.data();
  }
  const int64_t bs = n;
  const int64_t ncol_groups = n / kQ8_0xNrowsInterleaved;

  // Prefill: activation rows in groups of 4 -> the i8mm gemm.
  const int64_t mgroups = m / kQ8_0xNrowsInterleaved;
  if (mgroups > 0) {
    // Interleave the first mgroups*4 activation rows into group-major
    // BlockQ8_0x4, quantizing each row with the SAME from_float as tier-0.
    std::vector<uint8_t> act(static_cast<size_t>(mgroups) * nblocks *
                             sizeof(BlockQ8_0x4));
    BlockQ8_0x4* actb = reinterpret_cast<BlockQ8_0x4*>(act.data());
    std::vector<BlockQ8_0> tmp(static_cast<size_t>(kQ8_0xNrowsInterleaved) * nblocks);
    std::vector<float> row(static_cast<size_t>(k));
    for (int64_t g = 0; g < mgroups; ++g) {
      for (int r = 0; r < kQ8_0xNrowsInterleaved; ++r) {
        const int64_t i = g * kQ8_0xNrowsInterleaved + r;
        for (int64_t p = 0; p < k; ++p) row[p] = LoadActF32(a, i * a_rs + p);
        from_float(row.data(), tmp.data() + r * nblocks, k);
      }
      InterleaveQ8_0Rows4(tmp.data() + 0 * nblocks, tmp.data() + 1 * nblocks,
                          tmp.data() + 2 * nblocks, tmp.data() + 3 * nblocks,
                          actb + g * nblocks, nblocks);
    }
    ParallelForRows(CurrentThreadpool(), ncol_groups,
                    [&](int64_t xg0, int64_t xg1) {
                      GemmTileQ8_0(s, bs, wgroups, actb, mgroups, nblocks, xg0,
                                   xg1);
                    });
  }

  // Leftover rows (m % 4, includes decode m == 1): the i8mm/dotprod gemv, one
  // plain-q8_0 activation row at a time.
  const int64_t mg_rows = mgroups * kQ8_0xNrowsInterleaved;
  if (mg_rows < m) {
    for (int64_t i = mg_rows; i < m; ++i) {
      std::vector<BlockQ8_0> arow(static_cast<size_t>(nblocks));
      std::vector<float> row(static_cast<size_t>(k));
      for (int64_t p = 0; p < k; ++p) row[p] = LoadActF32(a, i * a_rs + p);
      from_float(row.data(), arow.data(), k);
      float* s_row = s + i * bs;
      ParallelForRows(CurrentThreadpool(), ncol_groups,
                      [&](int64_t xg0, int64_t xg1) {
                        GemvRowQ8_0(s_row, wgroups, arow.data(), nblocks, xg0,
                                    xg1);
                      });
    }
  }

  if (!out_is_f32) {
    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        StoreOutF32(out, i * n + j, out_scratch[static_cast<size_t>(i) * n + j]);
      }
    }
  }
}

#else  // no repack kernels on this target; the loader never repacks.

void QuantRepackMatmul(Tensor&, const Tensor&, const Tensor&) {
  VT_CHECK(false, "quant_repack_matmul: repack tier is not built on this target");
}

#endif

}  // namespace vt::cpu
