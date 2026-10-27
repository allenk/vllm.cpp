// The kFlint glue kernels (flint_device.h, src/vt/cuda/cuda_flint.cu) against a
// host model of the same arithmetic. The host repeats every rounding point of the
// vLLM kernel each one ports (for qk_norm_rope_cache: bf16 after the norm, bf16
// after the rotation); what can still differ is rsqrt / expf's last ulp, FMA
// contraction and the reduction order, so outputs may move by one bf16 ulp on a
// small share of elements, and pure copies (v, the residual) must be exact.
// Skips without a CUDA backend.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "vllm/model_executor/models/flint_device.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace {

using vt::Backend;
using vt::DeviceType;

bool HasFlintKernels() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
  } catch (const std::runtime_error&) {
    return false;
  }
  return vllm::flint::FlintDeviceKernelsAvailable();
}

float Bf16ToF32(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
float RoundBf16(float x) { return Bf16ToF32(vt::F32ToBF16(x)); }

std::vector<float> Rand(size_t n, uint32_t seed, float lo, float hi) {
  std::vector<float> v(n);
  uint32_t s = seed;
  for (float& x : v) {
    s = s * 1664525u + 1013904223u;
    x = lo + (hi - lo) * static_cast<float>(s >> 8) / static_cast<float>(1u << 24);
  }
  return v;
}

struct Buf {
  Backend& b;
  void* p;
  size_t n;
  Buf(Backend& be, vt::Queue& q, const void* host, size_t bytes) : b(be), p(be.Alloc(bytes)), n(bytes) {
    if (host != nullptr) b.Copy(q, p, host, bytes);
  }
  ~Buf() { b.Free(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
};

// Ulp distance between two bf16 words on the ordered line through zero (so -0 and
// +0 coincide and a value that rounds across zero is a small distance).
int UlpDiff(uint16_t a, uint16_t b) {
  auto key = [](uint16_t x) {
    const int mag = static_cast<int>(x & 0x7fffu);
    return (x & 0x8000u) != 0 ? -mag : mag;
  };
  return std::abs(key(a) - key(b));
}

}  // namespace

TEST_CASE("flint qk_norm_rope_kv_cache matches the host model of vLLM's kernel") {
  if (!HasFlintKernels()) {
    MESSAGE("no CUDA backend or no kFlint kernels; skipping");
    return;
  }
  const int64_t T = 5, Hq = 32, Hkv = 4, D = 128, half = D / 2, rows = Hq + 2 * Hkv;
  const int64_t block_size = 16, num_blocks = 6;
  const float eps = 1e-6f;
  // Flint's q/k come out of the projection at a few units; the norm weights near 1.
  std::vector<uint16_t> qkv(static_cast<size_t>(T * rows * D));
  {
    const auto f = Rand(qkv.size(), 11, -4.f, 4.f);
    for (size_t i = 0; i < f.size(); ++i) qkv[i] = vt::F32ToBF16(f[i]);
  }
  std::vector<uint16_t> qw(static_cast<size_t>(D)), kw(static_cast<size_t>(D));
  {
    const auto a = Rand(qw.size(), 12, 0.5f, 1.5f), c = Rand(kw.size(), 13, 0.5f, 1.5f);
    for (int64_t i = 0; i < D; ++i) {
      qw[static_cast<size_t>(i)] = vt::F32ToBF16(a[static_cast<size_t>(i)]);
      kw[static_cast<size_t>(i)] = vt::F32ToBF16(c[static_cast<size_t>(i)]);
    }
  }
  // Token-indexed cos|sin for positions 100..104 at theta 1e4.
  std::vector<float> cos_sin(static_cast<size_t>(T * D));
  for (int64_t t = 0; t < T; ++t)
    for (int64_t i = 0; i < half; ++i) {
      const double ang = static_cast<double>(100 + t) *
                         std::pow(10000.0, -2.0 * static_cast<double>(i) / static_cast<double>(D));
      cos_sin[static_cast<size_t>(t * D + i)] = static_cast<float>(std::cos(ang));
      cos_sin[static_cast<size_t>(t * D + half + i)] = static_cast<float>(std::sin(ang));
    }
  // The runner's combined (num_blocks, 2, block, Hkv, D) layout; slot -1 is padding.
  const std::vector<int64_t> slots = {3, 17, -1, 40, 95};
  const int64_t within = block_size * Hkv * D;
  const int64_t sb = 2 * within, ss = Hkv * D, sh = D;

  for (const bool rope : {true, false}) {
    CAPTURE(rope);
    Backend& b = vt::GetBackend(DeviceType::kCUDA);
    vt::Queue q = b.CreateQueue();
    {
      std::vector<uint16_t> cache(static_cast<size_t>(num_blocks * 2 * within), 0x7fc0u);  // NaN fill
      Buf dqkv(b, q, qkv.data(), qkv.size() * 2), dqw(b, q, qw.data(), qw.size() * 2),
          dkw(b, q, kw.data(), kw.size() * 2), dcs(b, q, cos_sin.data(), cos_sin.size() * 4),
          dslots(b, q, slots.data(), slots.size() * 8), dcache(b, q, cache.data(), cache.size() * 2),
          dq(b, q, nullptr, static_cast<size_t>(T * Hq * D) * 2);
      auto* kc = static_cast<uint16_t*>(dcache.p);
      vllm::flint::FlintDevice()->qk_norm_rope_kv_cache(
          q, dq.p, dqkv.p, rows * D, dqw.p, dkw.p, rope ? static_cast<const float*>(dcs.p) : nullptr,
          kc, kc + within, sb, ss, sh, block_size, static_cast<const int64_t*>(dslots.p), T, Hq, Hkv,
          D, eps);
      std::vector<uint16_t> got_q(static_cast<size_t>(T * Hq * D));
      b.Copy(q, got_q.data(), dq.p, got_q.size() * 2);
      b.Copy(q, cache.data(), dcache.p, cache.size() * 2);
      b.Synchronize(q);

      // Host model, element for element.
      size_t compared = 0, off_one = 0, off_more = 0, v_mismatch = 0, untouched_bad = 0;
      for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < rows; ++h) {
          const uint16_t* x = &qkv[static_cast<size_t>((t * rows + h) * D)];
          const bool is_q = h < Hq, is_v = h >= Hq + Hkv;
          const int64_t slot = slots[static_cast<size_t>(t)];
          const int64_t kvh = is_v ? h - Hq - Hkv : h - Hq;
          const size_t base =
              slot < 0 ? 0
                       : static_cast<size_t>((slot / block_size) * sb + (slot % block_size) * ss +
                                             kvh * sh + (is_v ? within : 0));
          if (is_v) {
            if (slot < 0) continue;
            for (int64_t i = 0; i < D; ++i) v_mismatch += cache[base + static_cast<size_t>(i)] != x[i];
            continue;
          }
          double ssq = 0.0;
          for (int64_t i = 0; i < D; ++i) ssq += static_cast<double>(Bf16ToF32(x[i])) * Bf16ToF32(x[i]);
          const float rs = 1.0f / std::sqrt(static_cast<float>(ssq) / static_cast<float>(D) + eps);
          const uint16_t* w = is_q ? qw.data() : kw.data();
          for (int64_t i = 0; i < half; ++i) {
            float n1 = RoundBf16(Bf16ToF32(x[i]) * rs * Bf16ToF32(w[i]));
            float n2 = RoundBf16(Bf16ToF32(x[half + i]) * rs * Bf16ToF32(w[half + i]));
            if (rope) {
              const float c = cos_sin[static_cast<size_t>(t * D + i)];
              const float s = cos_sin[static_cast<size_t>(t * D + half + i)];
              const float r1 = n1 * c - n2 * s, r2 = n2 * c + n1 * s;
              n1 = r1;
              n2 = r2;
            }
            const uint16_t want[2] = {vt::F32ToBF16(n1), vt::F32ToBF16(n2)};
            for (int j = 0; j < 2; ++j) {
              const int64_t col = i + j * half;
              uint16_t got;
              if (is_q) {
                got = got_q[static_cast<size_t>((t * Hq + h) * D + col)];
              } else {
                if (slot < 0) continue;
                got = cache[base + static_cast<size_t>(col)];
              }
              const int u = UlpDiff(got, want[j]);
              ++compared;
              off_one += u == 1;
              off_more += u > 1;
            }
          }
        }
      }
      // Pages no slot names keep their fill.
      for (int64_t blk = 0; blk < num_blocks; ++blk)
        for (int64_t o = 0; o < block_size; ++o) {
          const int64_t slot = blk * block_size + o;
          if (std::find(slots.begin(), slots.end(), slot) != slots.end()) continue;
          for (int64_t e = 0; e < Hkv * D; ++e)
            untouched_bad += cache[static_cast<size_t>(blk * sb + o * ss + e)] != 0x7fc0u;
        }
      INFO("compared " << compared << ", off by one ulp " << off_one);
      CHECK(compared == static_cast<size_t>(T * Hq * D + 4 * Hkv * D));
      CHECK(off_more == 0);
      CHECK(off_one * 100 <= compared);  // at most 1%
      CHECK(v_mismatch == 0);
      CHECK(untouched_bad == 0);
    }
    b.DestroyQueue(q);
  }
}

TEST_CASE("flint sigmoid_topk picks the top-k sigmoid scores and renormalises them") {
  if (!HasFlintKernels()) {
    MESSAGE("no CUDA backend or no kFlint kernels; skipping");
    return;
  }
  const int64_t E = 256, K = 6;
  for (const int64_t T : {int64_t{1}, int64_t{7}, int64_t{33}}) {
    CAPTURE(T);
    std::vector<uint16_t> logits(static_cast<size_t>(T * E));
    const auto f = Rand(logits.size(), 21 + static_cast<uint32_t>(T), -6.f, 6.f);
    for (size_t i = 0; i < f.size(); ++i) logits[i] = vt::F32ToBF16(f[i]);
    // Exact ties in the first token: experts 9 and 200 share the largest logit,
    // and the lower index must come first.
    logits[9] = logits[200] = vt::F32ToBF16(7.5f);
    Backend& b = vt::GetBackend(DeviceType::kCUDA);
    vt::Queue q = b.CreateQueue();
    {
      Buf dl(b, q, logits.data(), logits.size() * 2), dw(b, q, nullptr, static_cast<size_t>(T * K) * 4),
          di(b, q, nullptr, static_cast<size_t>(T * K) * 4);
      vllm::flint::FlintDevice()->sigmoid_topk(q, static_cast<float*>(dw.p), static_cast<int32_t*>(di.p),
                                               dl.p, T, E, K);
      std::vector<float> w(static_cast<size_t>(T * K));
      std::vector<int32_t> ids(static_cast<size_t>(T * K));
      b.Copy(q, w.data(), dw.p, w.size() * 4);
      b.Copy(q, ids.data(), di.p, ids.size() * 4);
      b.Synchronize(q);
      for (int64_t t = 0; t < T; ++t) {
        std::vector<float> s(static_cast<size_t>(E));
        for (int64_t e = 0; e < E; ++e)
          s[static_cast<size_t>(e)] = 1.f / (1.f + std::exp(-Bf16ToF32(logits[static_cast<size_t>(t * E + e)])));
        std::vector<int> order(static_cast<size_t>(E));
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int c) {
          return s[static_cast<size_t>(a)] > s[static_cast<size_t>(c)];
        });
        float sum = 0.f;
        for (int64_t k = 0; k < K; ++k) sum += s[static_cast<size_t>(order[static_cast<size_t>(k)])];
        for (int64_t k = 0; k < K; ++k) {
          const int want = order[static_cast<size_t>(k)];
          CHECK(ids[static_cast<size_t>(t * K + k)] == want);
          CHECK(w[static_cast<size_t>(t * K + k)] ==
                doctest::Approx(s[static_cast<size_t>(want)] / sum).epsilon(1e-5));
        }
      }
      CHECK(ids[0] == 9);
      CHECK(ids[1] == 200);
    }
    b.DestroyQueue(q);
  }
}

TEST_CASE("flint add_rms_norm matches the host model of vLLM's fused add + RMSNorm") {
  if (!HasFlintKernels()) {
    MESSAGE("no CUDA backend or no kFlint kernels; skipping");
    return;
  }
  const float eps = 1e-6f;
  // 8192 fills every register vector; 2880 leaves the last one partial.
  for (const int64_t H : {int64_t{1024}, int64_t{2880}, int64_t{8192}}) {
    for (const int64_t R : {int64_t{1}, int64_t{5}}) {
      for (const bool with_res : {true, false}) {
        for (const bool in_place : {false, true}) {
          CAPTURE(H);
          CAPTURE(R);
          CAPTURE(with_res);
          CAPTURE(in_place);
          const size_t n = static_cast<size_t>(R * H);
          std::vector<uint16_t> x(n), res(n), w(static_cast<size_t>(H));
          {
            const auto fx = Rand(n, 31 + static_cast<uint32_t>(H), -3.f, 3.f);
            const auto fr = Rand(n, 32 + static_cast<uint32_t>(H), -20.f, 20.f);
            const auto fw = Rand(w.size(), 33, 0.2f, 2.f);
            for (size_t i = 0; i < n; ++i) {
              x[i] = vt::F32ToBF16(fx[i]);
              res[i] = vt::F32ToBF16(fr[i]);
            }
            for (size_t i = 0; i < w.size(); ++i) w[i] = vt::F32ToBF16(fw[i]);
          }
          Backend& b = vt::GetBackend(DeviceType::kCUDA);
          vt::Queue q = b.CreateQueue();
          {
            Buf dx(b, q, x.data(), n * 2), dr(b, q, res.data(), n * 2), dw(b, q, w.data(), w.size() * 2),
                dout(b, q, nullptr, n * 2);
            void* out = in_place ? dx.p : dout.p;
            vllm::flint::FlintDevice()->add_rms_norm(q, out, dx.p, with_res ? dr.p : nullptr, dw.p, R, H,
                                                     eps);
            std::vector<uint16_t> got(n), got_res(n);
            b.Copy(q, got.data(), out, n * 2);
            b.Copy(q, got_res.data(), dr.p, n * 2);
            b.Synchronize(q);

            size_t off_one = 0, off_more = 0, res_bad = 0;
            for (int64_t r = 0; r < R; ++r) {
              std::vector<float> s(static_cast<size_t>(H));
              double ssq = 0.0;
              for (int64_t i = 0; i < H; ++i) {
                const size_t k = static_cast<size_t>(r * H + i);
                float v = Bf16ToF32(x[k]);
                if (with_res) {
                  v += Bf16ToF32(res[k]);
                  res_bad += got_res[k] != vt::F32ToBF16(v);
                } else {
                  res_bad += got_res[k] != res[k];
                }
                s[static_cast<size_t>(i)] = v;
                ssq += static_cast<double>(v) * v;
              }
              const float rs = 1.0f / std::sqrt(static_cast<float>(ssq) / static_cast<float>(H) + eps);
              for (int64_t i = 0; i < H; ++i) {
                const size_t k = static_cast<size_t>(r * H + i);
                const uint16_t want =
                    vt::F32ToBF16(s[static_cast<size_t>(i)] * rs * Bf16ToF32(w[static_cast<size_t>(i)]));
                const int u = UlpDiff(got[k], want);
                off_one += u == 1;
                off_more += u > 1;
              }
            }
            INFO("off by one ulp " << off_one << " of " << n);
            CHECK(res_bad == 0);
            CHECK(off_more == 0);
            CHECK(off_one * 100 <= n);
          }
          b.DestroyQueue(q);
        }
      }
    }
  }
}

TEST_CASE("flint routed_silu_and_mul rounds the SiLU to bf16 before the product") {
  if (!HasFlintKernels()) {
    MESSAGE("no CUDA backend or no kFlint kernels; skipping");
    return;
  }
  // Flint's moe_intermediate_size and an odd row count (P = T * top_k rows).
  const int64_t rows = 7, d = 1024;
  std::vector<uint16_t> gu(static_cast<size_t>(rows * 2 * d));
  {
    const auto f = Rand(gu.size(), 41, -8.f, 8.f);
    for (size_t i = 0; i < f.size(); ++i) gu[i] = vt::F32ToBF16(f[i]);
  }
  Backend& b = vt::GetBackend(DeviceType::kCUDA);
  vt::Queue q = b.CreateQueue();
  {
    Buf dgu(b, q, gu.data(), gu.size() * 2), dout(b, q, nullptr, static_cast<size_t>(rows * d) * 2);
    vllm::flint::FlintDevice()->routed_silu_and_mul(q, dout.p, dgu.p, rows, d);
    std::vector<uint16_t> got(static_cast<size_t>(rows * d));
    b.Copy(q, got.data(), dout.p, got.size() * 2);
    b.Synchronize(q);
    size_t off = 0, off_more = 0, distinct = 0, distinct_matched = 0;
    for (int64_t r = 0; r < rows; ++r)
      for (int64_t c = 0; c < d; ++c) {
        const float g = Bf16ToF32(gu[static_cast<size_t>(r * 2 * d + c)]);
        const float u = Bf16ToF32(gu[static_cast<size_t>(r * 2 * d + d + c)]);
        const float silu = g / (1.0f + std::exp(-g));
        const uint16_t want = vt::F32ToBF16(RoundBf16(silu) * u);
        const uint16_t one_rounding = vt::F32ToBF16(silu * u);
        const uint16_t v = got[static_cast<size_t>(r * d + c)];
        const int ul = UlpDiff(v, want);
        off += ul != 0;
        off_more += ul > 2;
        if (want != one_rounding) {
          ++distinct;
          distinct_matched += v == want;
        }
      }
    INFO("off " << off << "; double vs single rounding differ on " << distinct << ", matched "
                << distinct_matched);
    CHECK(off_more == 0);
    CHECK(off * 100 <= static_cast<size_t>(rows * d));
    // Where the two roundings disagree the kernel takes vLLM's, so a
    // single-rounding kernel cannot pass.
    CHECK(distinct > 0);
    CHECK(distinct_matched * 100 >= distinct * 99);
  }
  b.DestroyQueue(q);
}
