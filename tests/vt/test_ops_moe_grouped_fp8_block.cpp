// vllm.cpp original (vt runtime); no upstream mirror.
// The grouped MoE GEMM over BLOCK-WISE FP8 experts (vt::MoeGroupedGemmFp8Block),
// the layout `Qwen/Qwen3.6-35B-A3B-FP8` ships: per expert, fp8-e4m3fn [N, K] bytes
// beside an f32 [cdiv(N, bn), cdiv(K, bk)] scale grid. Output row p must equal the
// single-expert reference
//   out[p, n] = sum_kb s_e[n / bn, kb] * sum_{k in kb} w_e[n, k] * act[row(p), k]
// computed on the host in double. The kernel keeps the activation in bf16 and the
// weight exact (W8A16), so the only differences are f32 summation order and the
// bf16 rounding of a bf16 output. Covers every launch regime of
// cuda_matmul_nvfp4.cu `LaunchGroupedFp8Block`:
//   * K or block_k not a multiple of 16          -> one thread per output;
//   * P <= kFp8BlockWarpMaxP(64)                  -> one warp per output;
//   * 64 < P <= kMoeDecodeMaxP(512)               -> BM=16 WMMA tile;
//   * P > kMoeDecodeMaxP                          -> BM=64 WMMA tile;
// with ragged N and K (partial last scale block and partial last K-tile), both
// routings (row map, identity) and both output dtypes. CUDA-only; skips cleanly
// with no GPU.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
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
  void* ptr() { return p_; }
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

// An e4m3fn code. `full_range` draws any finite code (subnormals and +-448
// included, the two NaN codes excluded); otherwise the exponent field is 4..9,
// a magnitude in [2^-3, 7.5], so long dot products stay well scaled.
uint8_t E4m3Code(std::mt19937& rng, bool full_range) {
  if (full_range) {
    for (;;) {
      const auto b = static_cast<uint8_t>(rng() & 0xffu);
      if ((b & 0x7fu) != 0x7fu) return b;
    }
  }
  const uint32_t h = rng();
  const auto sign = static_cast<uint8_t>(h & 1u);
  const auto exp = static_cast<uint8_t>(4 + (h >> 1) % 6);
  const auto mant = static_cast<uint8_t>((h >> 8) & 7u);
  return static_cast<uint8_t>((sign << 7) | (exp << 3) | mant);
}

struct Case {
  int64_t e_count = 0, t_rows = 0, top_k = 0, k_dim = 0, n_cols = 0;
  int64_t block_n = 128, block_k = 128;
  uint32_t seed = 0;
  bool use_row_map = true;
  bool bf16_out = false;
  bool full_range = false;
};

struct Fixture {
  Case c;
  int64_t P = 0, act_rows = 0, kb = 0, nb = 0;
  std::vector<std::vector<uint8_t>> w;   // E x [N, K] e4m3fn
  std::vector<std::vector<float>> s;     // E x [nb, kb] f32
  std::vector<uint16_t> act;             // [act_rows, K] bf16
  std::vector<int32_t> expert_ids, row_map;
  std::vector<double> ref;               // [P, N]
};

Fixture Build(const Case& c) {
  Fixture f;
  f.c = c;
  f.P = c.t_rows * c.top_k;
  f.act_rows = c.use_row_map ? c.t_rows : f.P;
  f.kb = CDiv(c.k_dim, c.block_k);
  f.nb = CDiv(c.n_cols, c.block_n);
  std::mt19937 rng(c.seed);
  std::uniform_real_distribution<float> unit(-2.0f, 2.0f);
  std::uniform_real_distribution<float> scale(0.25f, 2.0f);
  for (int64_t e = 0; e < c.e_count; ++e) {
    std::vector<uint8_t> we(static_cast<size_t>(c.n_cols * c.k_dim));
    for (auto& b : we) b = E4m3Code(rng, c.full_range);
    std::vector<float> se(static_cast<size_t>(f.nb * f.kb));
    for (auto& x : se) x = scale(rng) / 64.0f;
    f.w.push_back(std::move(we));
    f.s.push_back(std::move(se));
  }
  f.act.resize(static_cast<size_t>(f.act_rows * c.k_dim));
  for (auto& x : f.act) x = vt::F32ToBF16(unit(rng));
  f.expert_ids.resize(static_cast<size_t>(f.P));
  f.row_map.resize(static_cast<size_t>(f.P));
  for (int64_t p = 0; p < f.P; ++p) {
    f.row_map[static_cast<size_t>(p)] = static_cast<int32_t>(p / c.top_k);
    // Deterministic pseudo-routing; a co-prime stride gives ragged per-expert counts.
    f.expert_ids[static_cast<size_t>(p)] = static_cast<int32_t>((p * 7 + 3) % c.e_count);
  }
  f.ref.assign(static_cast<size_t>(f.P * c.n_cols), 0.0);
  for (int64_t p = 0; p < f.P; ++p) {
    const auto e = static_cast<size_t>(f.expert_ids[static_cast<size_t>(p)]);
    const int64_t r = c.use_row_map ? f.row_map[static_cast<size_t>(p)] : p;
    for (int64_t n = 0; n < c.n_cols; ++n) {
      double acc = 0.0;
      for (int64_t b = 0; b < f.kb; ++b) {
        const int64_t k0 = b * c.block_k;
        const int64_t k1 = std::min(k0 + c.block_k, c.k_dim);
        double part = 0.0;
        for (int64_t k = k0; k < k1; ++k)
          part += static_cast<double>(vt::F8E4M3ToF32(f.w[e][static_cast<size_t>(n * c.k_dim + k)])) *
                  static_cast<double>(vt::BF16ToF32(f.act[static_cast<size_t>(r * c.k_dim + k)]));
        acc += part * static_cast<double>(f.s[e][static_cast<size_t>((n / c.block_n) * f.kb + b)]);
      }
      f.ref[static_cast<size_t>(p * c.n_cols + n)] = acc;
    }
  }
  return f;
}

std::vector<float> Run(const Fixture& f) {
  const Case& c = f.c;
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  DeviceBuffer dact(gpu, gq.q, DType::kBF16, {f.act_rows, c.k_dim}, f.act.data());
  DeviceBuffer deids(gpu, gq.q, DType::kI32, {f.P}, f.expert_ids.data());
  DeviceBuffer drow(gpu, gq.q, DType::kI32, {f.P}, f.row_map.data());
  std::vector<std::unique_ptr<DeviceBuffer>> bufs;
  std::vector<int64_t> wptrs, sptrs;
  for (int64_t e = 0; e < c.e_count; ++e) {
    bufs.push_back(std::make_unique<DeviceBuffer>(
        gpu, gq.q, DType::kI8, std::vector<int64_t>{c.n_cols, c.k_dim},
        f.w[static_cast<size_t>(e)].data()));
    wptrs.push_back(reinterpret_cast<int64_t>(bufs.back()->ptr()));
    bufs.push_back(std::make_unique<DeviceBuffer>(
        gpu, gq.q, DType::kF32, std::vector<int64_t>{f.nb, f.kb},
        f.s[static_cast<size_t>(e)].data()));
    sptrs.push_back(reinterpret_cast<int64_t>(bufs.back()->ptr()));
  }
  DeviceBuffer dwp(gpu, gq.q, DType::kI64, {c.e_count}, wptrs.data());
  DeviceBuffer dsp(gpu, gq.q, DType::kI64, {c.e_count}, sptrs.data());
  DeviceBuffer dout(gpu, gq.q, c.bf16_out ? DType::kBF16 : DType::kF32, {f.P, c.n_cols});
  vt::MoeGroupedGemmFp8Block(gq.q, dout.tensor(), dact.tensor(), deids.tensor(),
                             c.use_row_map ? &drow.tensor() : nullptr, dwp.tensor(),
                             dsp.tensor(), c.block_n, c.block_k);
  std::vector<float> got(static_cast<size_t>(f.P * c.n_cols));
  if (c.bf16_out) {
    std::vector<uint16_t> raw(got.size());
    dout.Download(gq.q, raw.data());
    for (size_t i = 0; i < got.size(); ++i) got[i] = vt::BF16ToF32(raw[i]);
  } else {
    dout.Download(gq.q, got.data());
  }
  return got;
}

// Element-wise |got - ref| <= rtol * |ref| + atol, with atol tied to the output's
// RMS: f32 order differences are relative to the size of the terms, not of the
// (possibly cancelled) sum. A bf16 output costs one 8-bit-mantissa rounding.
void CheckAgainstReference(const Fixture& f, const std::vector<float>& got) {
  REQUIRE(got.size() == f.ref.size());
  double ss = 0.0;
  for (double v : f.ref) ss += v * v;
  const double rms = std::sqrt(ss / static_cast<double>(f.ref.size()));
  const double rtol = f.c.bf16_out ? 8e-3 : 1e-3;
  const double atol = (f.c.bf16_out ? 8e-3 : 1e-3) * rms;
  size_t bad = 0, first_bad = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double err = std::fabs(static_cast<double>(got[i]) - f.ref[i]);
    if (!(err <= rtol * std::fabs(f.ref[i]) + atol)) {
      if (bad == 0) first_bad = i;
      ++bad;
    }
  }
  if (bad != 0) {
    CAPTURE(bad);
    CAPTURE(first_bad);
    CAPTURE(got[first_bad]);
    CAPTURE(f.ref[first_bad]);
    CAPTURE(rms);
  }
  CHECK(bad == 0);
}

void RunCase(const Case& c) {
  const Fixture f = Build(c);
  CheckAgainstReference(f, Run(f));
}

}  // namespace

// K = 40 is not a multiple of 16 -> one thread per output. block_k = 24 leaves a
// short last scale block (40 = 24 + 16); N = 33 with block_n = 16 a short last row
// block.
TEST_CASE("GPU moe_grouped_gemm_fp8_block one-thread path matches the block-scaled reference") {
  if (!HasCuda()) {
    MESSAGE("CUDA backend unavailable; skipping");
    return;
  }
  Case c;
  c.e_count = 4; c.t_rows = 5; c.top_k = 2; c.k_dim = 40; c.n_cols = 33;
  c.block_n = 16; c.block_k = 24; c.seed = 100;
  RunCase(c);
}

// P = 16 <= 64 -> one warp per output: the decode kernel, at the Qwen block size
// with N = 96 inside one partial 128-row block and K = 256 two full K-blocks.
TEST_CASE("GPU moe_grouped_gemm_fp8_block warp path matches the block-scaled reference") {
  if (!HasCuda()) {
    MESSAGE("CUDA backend unavailable; skipping");
    return;
  }
  Case c;
  c.e_count = 5; c.t_rows = 4; c.top_k = 4; c.k_dim = 256; c.n_cols = 96; c.seed = 200;
  RunCase(c);
  // A short last K-block (176 = 2 * 64 + 48), identity routing, bf16 out: the
  // down projection's call shape.
  Case d = c;
  d.k_dim = 176; d.block_k = 64; d.block_n = 32; d.use_row_map = false; d.bf16_out = true;
  d.seed = 210;
  RunCase(d);
}

// Every finite e4m3 code, subnormals and +-448 included, through the hardware
// conversion on sm_89+ (the header's software path below it).
TEST_CASE("GPU moe_grouped_gemm_fp8_block decodes the whole e4m3fn code range") {
  if (!HasCuda()) {
    MESSAGE("CUDA backend unavailable; skipping");
    return;
  }
  Case c;
  c.e_count = 3; c.t_rows = 4; c.top_k = 2; c.k_dim = 128; c.n_cols = 64; c.seed = 300;
  c.full_range = true;
  RunCase(c);  // warp path
  Case d = c;
  d.t_rows = 40; d.seed = 310;  // P = 80 -> BM=16 tile
  RunCase(d);
}

// 64 < P = 160 <= 512 -> BM=16 WMMA tile. N = 192 spans two scale-row blocks
// (three 64-column tiles), K = 256 two K-blocks of four 32-wide tiles each.
TEST_CASE("GPU moe_grouped_gemm_fp8_block BM=16 tile matches the block-scaled reference") {
  if (!HasCuda()) {
    MESSAGE("CUDA backend unavailable; skipping");
    return;
  }
  Case c;
  c.e_count = 7; c.t_rows = 20; c.top_k = 8; c.k_dim = 256; c.n_cols = 192; c.seed = 400;
  RunCase(c);
  // K = 208: the last K-tile is half full and the last 64-wide K-block holds 16.
  Case d = c;
  d.k_dim = 208; d.block_k = 64; d.block_n = 64; d.seed = 410;
  RunCase(d);
}

// P = 1024 > 512 -> BM=64 WMMA tile. N = 130 leaves a 2-column last tile, K = 384
// three K-blocks; identity routing and bf16 out on the second call.
TEST_CASE("GPU moe_grouped_gemm_fp8_block BM=64 tile matches the block-scaled reference") {
  if (!HasCuda()) {
    MESSAGE("CUDA backend unavailable; skipping");
    return;
  }
  Case c;
  c.e_count = 9; c.t_rows = 128; c.top_k = 8; c.k_dim = 384; c.n_cols = 130;
  c.block_n = 64; c.seed = 500;
  RunCase(c);
  Case d = c;
  d.use_row_map = false; d.bf16_out = true; d.seed = 510;
  RunCase(d);
}

// Greedy decode must be reproducible: the warp kernel's lane order and butterfly
// are fixed, and the tile computes each output row on its own whatever position
// the counting sort's atomics give it.
TEST_CASE("GPU moe_grouped_gemm_fp8_block is bit-reproducible run to run") {
  if (!HasCuda()) {
    MESSAGE("CUDA backend unavailable; skipping");
    return;
  }
  for (const int64_t t_rows : {2, 40}) {  // warp path, then the BM=16 tile
    Case c;
    c.e_count = 6; c.t_rows = t_rows; c.top_k = 8; c.k_dim = 512; c.n_cols = 128;
    c.seed = 600 + static_cast<uint32_t>(t_rows);
    const Fixture f = Build(c);
    const std::vector<float> first = Run(f);
    CheckAgainstReference(f, first);
    for (int rep = 0; rep < 3; ++rep) CHECK(Run(f) == first);
  }
}
