// vllm.cpp original (vt runtime); no upstream mirror.
// vt::MatmulFp8BlockScaledGemv, the decode arm of the block-wise W8A8 FP8 linear:
// QuantFp8Group's activation quant fused into a GEMV. Checked against the math it
// claims, in double: the CPU QuantFp8Group's bytes and scales (byte-identical to the
// CUDA op's, test_ops_quant_fp8_group_cpu G6/G7), then
//   out[m, n] = sum_kb a_s[m, kb] * w_s[n / bn, kb] * sum_{k in kb} q[m, k] * w[n, k]
// to the output's bf16 rounding. Covers every row-count instantiation (M = 1, 2, 3,
// 5, 8), f32 and bf16 x and out, a ragged N, a group at the eps floor beside one a
// million times larger, and the shapes it refuses. Where CUTLASS takes the shape, the
// QuantFp8Group + MatmulFp8BlockScaled pair it replaces is measured against the
// same reference beside it. CUDA-only; skips cleanly with no GPU.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/fp8_kv.h"
#include "vt/ops.h"

namespace {

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Cpu() { return Device{DeviceType::kCPU, 0}; }

Tensor MakeTensor(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

struct QueueGuard {
  Backend& b;
  Queue q;
  explicit QueueGuard(Backend& backend) : b(backend), q(backend.CreateQueue()) {}
  ~QueueGuard() { b.DestroyQueue(q); }
  QueueGuard(const QueueGuard&) = delete;
  QueueGuard& operator=(const QueueGuard&) = delete;
};

class DeviceBuffer {
 public:
  DeviceBuffer(Backend& b, Queue& q, DType dt, const std::vector<int64_t>& shape,
               const void* host = nullptr)
      : b_(b) {
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    p_ = b_.Alloc(bytes_ == 0 ? 1 : bytes_);
    if (host != nullptr) b_.Copy(q, p_, host, bytes_);
    t_ = MakeTensor(p_, dt, q.device, shape);
  }
  ~DeviceBuffer() { b_.Free(p_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  Tensor& tensor() { return t_; }
  void Download(Queue& q, void* dst) {
    b_.Copy(q, dst, p_, bytes_);
    b_.Synchronize(q);
  }

 private:
  Backend& b_;
  void* p_ = nullptr;
  size_t bytes_ = 0;
  Tensor t_;
};

int64_t CDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Any finite e4m3fn code (the two NaN codes excluded).
uint8_t E4m3Code(std::mt19937& rng) {
  for (;;) {
    const auto b = static_cast<uint8_t>(rng() & 0xffu);
    if ((b & 0x7fu) != 0x7fu) return b;
  }
}

struct Case {
  int64_t m = 1, n = 384, k = 2048;
  int64_t bn = 128, bk = 128;
  bool x_bf16 = true, out_bf16 = true;
  uint32_t seed = 1;
  bool zero_group = false;   // row 0's first group all zero: the eps floor
  bool huge_group = false;   // row 0's last group a million times larger than the rest
};

struct Fixture {
  Case c;
  int64_t nb = 0, kb = 0;
  std::vector<float> xf;     // x in f32 (the bf16 values themselves when x_bf16)
  std::vector<uint16_t> xb;  // x in bf16 when x_bf16
  std::vector<uint8_t> w;    // [N, K] e4m3fn
  std::vector<float> ws;     // [nb, kb]
  std::vector<double> ref;   // [M, N]
};

Fixture Build(const Case& c) {
  Fixture f;
  f.c = c;
  f.nb = CDiv(c.n, c.bn);
  f.kb = c.k / c.bk;
  std::mt19937 rng(c.seed);
  std::normal_distribution<float> act(0.0f, 1.0f);
  std::uniform_real_distribution<float> scale(0.5f, 2.0f);
  f.xf.resize(static_cast<size_t>(c.m * c.k));
  // A few outliers so the groups' scales differ.
  for (auto& v : f.xf) v = act(rng) * ((rng() % 16u) == 0u ? 8.0f : 1.0f);
  if (c.zero_group)
    for (int64_t i = 0; i < c.bk; ++i) f.xf[static_cast<size_t>(i)] = 0.0f;
  if (c.huge_group)
    for (int64_t i = c.k - c.bk; i < c.k; ++i) f.xf[static_cast<size_t>(i)] *= 1.0e6f;
  if (c.x_bf16) {
    f.xb.resize(f.xf.size());
    for (size_t i = 0; i < f.xf.size(); ++i) {
      f.xb[i] = vt::F32ToBF16(f.xf[i]);
      f.xf[i] = vt::BF16ToF32(f.xb[i]);
    }
  }
  f.w.resize(static_cast<size_t>(c.n * c.k));
  for (auto& b : f.w) b = E4m3Code(rng);
  f.ws.resize(static_cast<size_t>(f.nb * f.kb));
  for (auto& s : f.ws) s = scale(rng) / 256.0f;

  // The bytes and scales the op must reproduce: the CPU QuantFp8Group.
  std::vector<uint8_t> q(static_cast<size_t>(c.m * c.k));
  std::vector<float> as(static_cast<size_t>(c.m * f.kb));
  Queue cq{Cpu(), nullptr};
  Tensor tx = c.x_bf16 ? MakeTensor(f.xb.data(), DType::kBF16, Cpu(), {c.m, c.k})
                       : MakeTensor(f.xf.data(), DType::kF32, Cpu(), {c.m, c.k});
  Tensor tq = MakeTensor(q.data(), DType::kI8, Cpu(), {c.m, c.k});
  Tensor ts = MakeTensor(as.data(), DType::kF32, Cpu(), {c.m, f.kb});
  vt::QuantFp8Group(cq, tq, ts, tx, static_cast<int>(c.bk));

  f.ref.assign(static_cast<size_t>(c.m * c.n), 0.0);
  for (int64_t r = 0; r < c.m; ++r) {
    for (int64_t n = 0; n < c.n; ++n) {
      double acc = 0.0;
      for (int64_t b = 0; b < f.kb; ++b) {
        double part = 0.0;
        for (int64_t k = b * c.bk; k < (b + 1) * c.bk; ++k)
          part += static_cast<double>(vt::F8E4M3ToF32(q[static_cast<size_t>(r * c.k + k)])) *
                  static_cast<double>(vt::F8E4M3ToF32(f.w[static_cast<size_t>(n * c.k + k)]));
        acc += part * static_cast<double>(as[static_cast<size_t>(r * f.kb + b)]) *
               static_cast<double>(f.ws[static_cast<size_t>((n / c.bn) * f.kb + b)]);
      }
      f.ref[static_cast<size_t>(r * c.n + n)] = acc;
    }
  }
  return f;
}

std::vector<float> Download(DeviceBuffer& out, Queue& q, bool bf16, size_t count) {
  std::vector<float> got(count);
  if (bf16) {
    std::vector<uint16_t> raw(count);
    out.Download(q, raw.data());
    for (size_t i = 0; i < count; ++i) got[i] = vt::BF16ToF32(raw[i]);
  } else {
    out.Download(q, got.data());
  }
  return got;
}

// The op under test, or (cutlass) the QuantFp8Group + MatmulFp8BlockScaled pair.
std::vector<float> Run(const Fixture& f, bool cutlass = false) {
  const Case& c = f.c;
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  DeviceBuffer dx(gpu, gq.q, c.x_bf16 ? DType::kBF16 : DType::kF32, {c.m, c.k},
                  c.x_bf16 ? static_cast<const void*>(f.xb.data())
                           : static_cast<const void*>(f.xf.data()));
  DeviceBuffer dw(gpu, gq.q, DType::kI8, {c.n, c.k}, f.w.data());
  DeviceBuffer ds(gpu, gq.q, DType::kF32, {f.nb, f.kb}, f.ws.data());
  DeviceBuffer dout(gpu, gq.q, c.out_bf16 ? DType::kBF16 : DType::kF32, {c.m, c.n});
  if (cutlass) {
    DeviceBuffer dq(gpu, gq.q, DType::kI8, {c.m, c.k});
    DeviceBuffer das(gpu, gq.q, DType::kF32, {c.m, f.kb});
    vt::QuantFp8Group(gq.q, dq.tensor(), das.tensor(), dx.tensor(), static_cast<int>(c.bk));
    vt::MatmulFp8BlockScaled(gq.q, dout.tensor(), dq.tensor(), das.tensor(), dw.tensor(),
                             ds.tensor(), static_cast<int>(c.bn), static_cast<int>(c.bk));
    return Download(dout, gq.q, c.out_bf16, static_cast<size_t>(c.m * c.n));
  }
  vt::MatmulFp8BlockScaledGemv(gq.q, dout.tensor(), dx.tensor(), dw.tensor(), ds.tensor(), c.bn,
                               c.bk);
  return Download(dout, gq.q, c.out_bf16, static_cast<size_t>(c.m * c.n));
}

// |got - ref| <= rtol * |ref| + atol per element. The output is rounded through
// bf16 (2^-9 relative at most); the rest is f32 summation order. atol is tied to
// the output's RMS because the order error scales with the terms, not with a sum
// that may have cancelled.
size_t CountBad(const Fixture& f, const std::vector<float>& got, double* worst = nullptr) {
  REQUIRE(got.size() == f.ref.size());
  double ss = 0.0;
  for (double v : f.ref) ss += v * v;
  const double rms = std::sqrt(ss / static_cast<double>(f.ref.size()));
  constexpr double kRtol = 1.0 / 128.0, kAtolOfRms = 1e-3;
  size_t bad = 0;
  double w = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double err = std::fabs(static_cast<double>(got[i]) - f.ref[i]);
    w = std::max(w, err / (std::fabs(f.ref[i]) + rms));
    if (!(err <= kRtol * std::fabs(f.ref[i]) + kAtolOfRms * rms)) ++bad;
  }
  if (worst != nullptr) *worst = w;
  return bad;
}

void CheckCase(const Case& c) {
  CAPTURE(c.m);
  CAPTURE(c.n);
  CAPTURE(c.k);
  const Fixture f = Build(c);
  double worst = 0.0;
  const size_t bad = CountBad(f, Run(f), &worst);
  CAPTURE(worst);
  CHECK(bad == 0u);
}

}  // namespace

TEST_CASE("GPU matmul_fp8_block_scaled_gemv matches the W8A8 block reference at every row count") {
  if (!HasCuda()) {
    MESSAGE("CUDA backend unavailable; skipping");
    return;
  }
  for (const int64_t m : {1, 2, 3, 5, 8}) {  // the 1-, 2-, 4- and 8-row instantiations
    Case c;
    c.m = m;
    c.seed = 100 + static_cast<uint32_t>(m);
    CheckCase(c);
  }
}

TEST_CASE("GPU matmul_fp8_block_scaled_gemv takes f32 and bf16 x and out, and a ragged N") {
  if (!HasCuda()) {
    MESSAGE("CUDA backend unavailable; skipping");
    return;
  }
  Case a;
  a.m = 4; a.x_bf16 = false; a.out_bf16 = false; a.k = 4096; a.seed = 200;
  CheckCase(a);
  Case b;
  b.m = 2; b.out_bf16 = false; b.seed = 210;
  CheckCase(b);
  Case d;
  d.m = 1; d.x_bf16 = false; d.seed = 220;
  CheckCase(d);
  // N = 200: the second scale row covers 72 columns.
  Case r;
  r.m = 3; r.n = 200; r.seed = 230;
  CheckCase(r);
}

TEST_CASE("GPU matmul_fp8_block_scaled_gemv quantizes an all-zero group and a far larger one") {
  if (!HasCuda()) {
    MESSAGE("CUDA backend unavailable; skipping");
    return;
  }
  Case c;
  c.m = 2; c.zero_group = true; c.huge_group = true; c.seed = 300;
  CheckCase(c);
}

TEST_CASE("GPU matmul_fp8_block_scaled_gemv agrees with the quant + CUTLASS pair it replaces") {
  if (!HasCuda()) {
    MESSAGE("CUDA backend unavailable; skipping");
    return;
  }
  if (!vt::OpRegistered(vt::OpId::kMatmulFp8BlockScaled, DeviceType::kCUDA)) {
    MESSAGE("no CUDA MatmulFp8BlockScaled in this build; the pair is not measured");
    return;
  }
  for (const int64_t m : {1, 8}) {
    Case c;
    c.m = m;
    c.n = 512;
    c.seed = 400 + static_cast<uint32_t>(m);
    const Fixture f = Build(c);
    const std::vector<float> gemv = Run(f);
    const std::vector<float> pair = Run(f, /*cutlass=*/true);
    double wg = 0.0, wp = 0.0;
    const size_t bg = CountBad(f, gemv, &wg), bp = CountBad(f, pair, &wp);
    size_t same = 0;
    for (size_t i = 0; i < gemv.size(); ++i) same += gemv[i] == pair[i] ? 1u : 0u;
    MESSAGE("M=" << m << ": worst error over (|ref| + rms) gemv " << wg << ", quant + CUTLASS " << wp
                 << "; " << same << " / " << gemv.size() << " outputs bit-equal");
    CHECK(bg == 0u);
    CHECK(bp == 0u);
  }
}

TEST_CASE("matmul_fp8_block_scaled_gemv takes only the shapes it can hold") {
  CHECK(vt::MatmulFp8BlockScaledGemvApplies(1, 2048, 128));
  CHECK(vt::MatmulFp8BlockScaledGemvApplies(8, 4096, 128));
  CHECK(vt::MatmulFp8BlockScaledGemvApplies(2, 17408, 128));    // 27B's down projection
  CHECK_FALSE(vt::MatmulFp8BlockScaledGemvApplies(3, 17408, 128));  // past 48 KiB
  CHECK_FALSE(vt::MatmulFp8BlockScaledGemvApplies(9, 2048, 128));   // more than 8 rows
  CHECK_FALSE(vt::MatmulFp8BlockScaledGemvApplies(1, 2056, 128));   // K % block_k
  CHECK_FALSE(vt::MatmulFp8BlockScaledGemvApplies(1, 2048, 64));    // a group other than 128
  CHECK_FALSE(vt::MatmulFp8BlockScaledGemvApplies(0, 2048, 128));
  if (!HasCuda()) return;
  Case c;
  c.m = 9;
  c.seed = 500;
  const Fixture f = Build(c);
  CHECK_THROWS_AS(Run(f), std::runtime_error);
}

TEST_CASE("GPU matmul_fp8_block_scaled_gemv is bit-reproducible run to run") {
  if (!HasCuda()) {
    MESSAGE("CUDA backend unavailable; skipping");
    return;
  }
  Case c;
  c.m = 5;
  c.seed = 600;
  const Fixture f = Build(c);
  const std::vector<float> first = Run(f);
  for (int rep = 0; rep < 3; ++rep) CHECK(Run(f) == first);
}
