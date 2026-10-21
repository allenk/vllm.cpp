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

#include "support/test_env.h"  // SetEnv/UnsetEnv — MSVC has no setenv (#603)
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/fp8_kv.h"
#include "vt/ops.h"
#ifdef VLLM_CPP_CUDA
#include "vt/cuda/cuda_device_caps.h"
#endif

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

// e4m3 MMAs (the W8A8 path) are sm_89+.
bool HasFp8Mma() {
#ifdef VLLM_CPP_CUDA
  if (!HasCuda()) return false;
  const vt::cuda::DeviceCaps& caps = vt::cuda::GetDeviceCaps();
  return caps.valid && caps.sm_arch() >= 89;
#else
  return false;
#endif
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

// --- vt::MoeGroupedGemmFp8BlockGateUpSilu --------------------------------------
// `w`'s weights against `a`'s activation and routing, in double: the up
// projection of a gate/up pair whose activation came from the gate fixture.
std::vector<double> RefSums(const Fixture& w, const Fixture& a) {
  const Case& c = a.c;
  std::vector<double> ref(static_cast<size_t>(a.P * c.n_cols), 0.0);
  for (int64_t p = 0; p < a.P; ++p) {
    const auto e = static_cast<size_t>(a.expert_ids[static_cast<size_t>(p)]);
    const int64_t r = c.use_row_map ? a.row_map[static_cast<size_t>(p)] : p;
    for (int64_t n = 0; n < c.n_cols; ++n) {
      double acc = 0.0;
      for (int64_t b = 0; b < a.kb; ++b) {
        const int64_t k0 = b * c.block_k;
        const int64_t k1 = std::min(k0 + c.block_k, c.k_dim);
        double part = 0.0;
        for (int64_t k = k0; k < k1; ++k)
          part += static_cast<double>(vt::F8E4M3ToF32(w.w[e][static_cast<size_t>(n * c.k_dim + k)])) *
                  static_cast<double>(vt::BF16ToF32(a.act[static_cast<size_t>(r * c.k_dim + k)]));
        acc += part * static_cast<double>(w.s[e][static_cast<size_t>((n / c.block_n) * a.kb + b)]);
      }
      ref[static_cast<size_t>(p * c.n_cols + n)] = acc;
    }
  }
  return ref;
}

struct GateUpOut {
  std::vector<uint16_t> triplet;  // two grouped GEMMs into bf16, then MoeSiluMul
  std::vector<uint16_t> fused;    // MoeGroupedGemmFp8BlockGateUpSilu
};

// One activation and routing (g's), gate weights from g, up weights from u.
GateUpOut RunGateUp(const Fixture& g, const Fixture& u) {
  const Case& c = g.c;
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  QueueGuard gq(gpu);
  DeviceBuffer dact(gpu, gq.q, DType::kBF16, {g.act_rows, c.k_dim}, g.act.data());
  DeviceBuffer deids(gpu, gq.q, DType::kI32, {g.P}, g.expert_ids.data());
  DeviceBuffer drow(gpu, gq.q, DType::kI32, {g.P}, g.row_map.data());
  std::vector<std::unique_ptr<DeviceBuffer>> bufs;
  auto tables = [&](const Fixture& f, std::vector<int64_t>& wp, std::vector<int64_t>& sp) {
    for (int64_t e = 0; e < c.e_count; ++e) {
      bufs.push_back(std::make_unique<DeviceBuffer>(
          gpu, gq.q, DType::kI8, std::vector<int64_t>{c.n_cols, c.k_dim},
          f.w[static_cast<size_t>(e)].data()));
      wp.push_back(reinterpret_cast<int64_t>(bufs.back()->ptr()));
      bufs.push_back(std::make_unique<DeviceBuffer>(
          gpu, gq.q, DType::kF32, std::vector<int64_t>{f.nb, f.kb},
          f.s[static_cast<size_t>(e)].data()));
      sp.push_back(reinterpret_cast<int64_t>(bufs.back()->ptr()));
    }
  };
  std::vector<int64_t> gw, gs, uw, us;
  tables(g, gw, gs);
  tables(u, uw, us);
  DeviceBuffer dgw(gpu, gq.q, DType::kI64, {c.e_count}, gw.data());
  DeviceBuffer dgs(gpu, gq.q, DType::kI64, {c.e_count}, gs.data());
  DeviceBuffer duw(gpu, gq.q, DType::kI64, {c.e_count}, uw.data());
  DeviceBuffer dus(gpu, gq.q, DType::kI64, {c.e_count}, us.data());
  const Tensor* rmap = c.use_row_map ? &drow.tensor() : nullptr;

  DeviceBuffer dg(gpu, gq.q, DType::kBF16, {g.P, c.n_cols});
  DeviceBuffer du(gpu, gq.q, DType::kBF16, {g.P, c.n_cols});
  DeviceBuffer dtrip(gpu, gq.q, DType::kBF16, {g.P, c.n_cols});
  vt::MoeGroupedGemmFp8Block(gq.q, dg.tensor(), dact.tensor(), deids.tensor(), rmap,
                             dgw.tensor(), dgs.tensor(), c.block_n, c.block_k);
  vt::MoeGroupedGemmFp8Block(gq.q, du.tensor(), dact.tensor(), deids.tensor(), rmap,
                             duw.tensor(), dus.tensor(), c.block_n, c.block_k);
  vt::MoeSiluMul(gq.q, dtrip.tensor(), dg.tensor(), du.tensor());
  DeviceBuffer dfused(gpu, gq.q, DType::kBF16, {g.P, c.n_cols});
  vt::MoeGroupedGemmFp8BlockGateUpSilu(gq.q, dfused.tensor(), dact.tensor(), deids.tensor(),
                                       rmap, dgw.tensor(), dgs.tensor(), duw.tensor(),
                                       dus.tensor(), c.block_n, c.block_k);
  GateUpOut out;
  out.triplet.resize(static_cast<size_t>(g.P * c.n_cols));
  out.fused.resize(out.triplet.size());
  dtrip.Download(gq.q, out.triplet.data());
  dfused.Download(gq.q, out.fused.data());
  return out;
}

// silu(bf16(g)) * bf16(u) from the double sums, against the fused output. The
// bf16 rounding of each sum sits on f32 sums that differ from the double ones in
// the last bits, so one of them can land a bf16 ulp away; the tolerance allows a
// few.
void CheckGateUpAgainstReference(const Fixture& g, const Fixture& u,
                                 const std::vector<uint16_t>& got) {
  const std::vector<double> uref = RefSums(u, g);
  std::vector<double> ref(got.size());
  double ss = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double gb = vt::BF16ToF32(vt::F32ToBF16(static_cast<float>(g.ref[i])));
    const double ub = vt::BF16ToF32(vt::F32ToBF16(static_cast<float>(uref[i])));
    ref[i] = gb / (1.0 + std::exp(-gb)) * ub;
    ss += ref[i] * ref[i];
  }
  const double rms = std::sqrt(ss / static_cast<double>(ref.size()));
  size_t bad = 0, first_bad = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double v = vt::BF16ToF32(got[i]);
    if (!(std::fabs(v - ref[i]) <= 3e-2 * std::fabs(ref[i]) + 3e-2 * rms)) {
      if (bad == 0) first_bad = i;
      ++bad;
    }
  }
  if (bad != 0) {
    CAPTURE(bad);
    CAPTURE(first_bad);
    CAPTURE(vt::BF16ToF32(got[first_bad]));
    CAPTURE(ref[first_bad]);
    CAPTURE(rms);
  }
  CHECK(bad == 0);
}

// The up fixture: the same shape and routing, other weights (its own activation
// is not used).
Fixture BuildUp(const Case& c) {
  Case uc = c;
  uc.seed = c.seed + 7777;
  return Build(uc);
}

void RunGateUpCase(const Case& c, bool expect_bit_identical) {
  const Fixture g = Build(c);
  const Fixture u = BuildUp(c);
  const GateUpOut out = RunGateUp(g, u);
  CheckGateUpAgainstReference(g, u, out.fused);
  if (expect_bit_identical) {
    size_t diff = 0;
    for (size_t i = 0; i < out.fused.size(); ++i) diff += out.fused[i] != out.triplet[i] ? 1 : 0;
    CAPTURE(diff);
    CHECK(diff == 0);
  }
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

// P = 1024 with the small-tile bound at 512 -> BM=64 WMMA tile. N = 130 leaves a
// 2-column last tile, K = 384 three K-blocks; identity routing and bf16 out on the
// second call.
TEST_CASE("GPU moe_grouped_gemm_fp8_block BM=64 tile matches the block-scaled reference") {
  if (!HasCuda()) {
    MESSAGE("CUDA backend unavailable; skipping");
    return;
  }
  vllm_test::SetEnv("VT_MOE_FP8_BLOCK_SMALL_TILE_MAX_P", "512");
  Case c;
  c.e_count = 9; c.t_rows = 128; c.top_k = 8; c.k_dim = 384; c.n_cols = 130;
  c.block_n = 64; c.seed = 500;
  RunCase(c);
  Case d = c;
  d.use_row_map = false; d.bf16_out = true; d.seed = 510;
  RunCase(d);
  vllm_test::UnsetEnv("VT_MOE_FP8_BLOCK_SMALL_TILE_MAX_P");
}

// The pipelined BM=16 tile (BK = 128, loads in flight under the MMAs) against the
// BM=16 tile it replaces, which VT_MOE_FP8_BLOCK_PIPE=0 selects: one table and the
// gate+up pair, bit for bit, at the Qwen3.6-35B-A3B prefill-chunk shape (about 4
// pairs per expert, K = 2048 for gate/up and 512 for down, 128-wide scale blocks),
// at a two-scale-block K, and at P = 1024, which the 16,384 bound now gives this tile.
TEST_CASE("GPU moe_grouped_gemm_fp8_block pipelined BM=16 tile is the BM=16 tile bit for bit") {
  if (!HasCuda()) {
    MESSAGE("CUDA backend unavailable; skipping");
    return;
  }
  constexpr const char* kKnob = "VT_MOE_FP8_BLOCK_PIPE";
  Case gate;
  gate.e_count = 64; gate.t_rows = 32; gate.top_k = 8; gate.k_dim = 2048; gate.n_cols = 128;
  gate.seed = 1100;
  Case down = gate;
  down.k_dim = 512; down.n_cols = 192; down.use_row_map = false; down.bf16_out = true;
  down.seed = 1110;
  Case two = gate;
  two.e_count = 9; two.t_rows = 128; two.k_dim = 384; two.n_cols = 130; two.block_n = 64;
  two.block_k = 256; two.seed = 1120;
  for (const Case& c : {gate, down, two}) {
    CAPTURE(c.seed);
    const Fixture f = Build(c);
    const Fixture u = BuildUp(c);
    vllm_test::SetEnv(kKnob, "0");
    const std::vector<float> old_out = Run(f);
    const std::vector<uint16_t> old_fused = RunGateUp(f, u).fused;
    vllm_test::UnsetEnv(kKnob);
    const std::vector<float> pipe_out = Run(f);
    CheckAgainstReference(f, pipe_out);
    CHECK(pipe_out == old_out);
    CHECK(RunGateUp(f, u).fused == old_fused);
  }
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

// The fused gate+up+SwiGLU op is the decode triplet bit for bit wherever the
// triplet's GEMMs run their warp or one-thread kernels (P <= 64), and is correct
// at any P.
TEST_CASE("GPU moe_grouped_gemm_fp8_block_gate_up_silu equals the gate, up and silu-mul triplet") {
  if (!HasCuda()) {
    MESSAGE("CUDA backend unavailable; skipping");
    return;
  }
  // The Qwen decode shape at a small scale: P = 16 pairs, block 128, two K-blocks.
  Case c;
  c.e_count = 5; c.t_rows = 4; c.top_k = 4; c.k_dim = 256; c.n_cols = 96; c.seed = 700;
  RunGateUpCase(c, /*expect_bit_identical=*/true);
  // A short last K-block (176 = 2 * 64 + 48), a 32-row scale block, identity routing.
  Case d = c;
  d.k_dim = 176; d.block_k = 64; d.block_n = 32; d.use_row_map = false; d.seed = 710;
  RunGateUpCase(d, true);
  // K = 40, block_k = 24: the one-thread kernel on both sides.
  Case n = c;
  n.k_dim = 40; n.n_cols = 33; n.block_n = 16; n.block_k = 24; n.seed = 720;
  RunGateUpCase(n, true);
  // Every finite e4m3 code.
  Case r = c;
  r.full_range = true; r.seed = 730;
  RunGateUpCase(r, true);
  // P = 64, the largest step the MoE block sends here (c=8 decode).
  Case w = c;
  w.t_rows = 8; w.top_k = 8; w.k_dim = 512; w.n_cols = 128; w.seed = 740;
  RunGateUpCase(w, true);
}

// Above the warp range the fused op runs the grouped tile for both tables, at the
// height the triplet's GEMMs pick, so it is the triplet bit for bit there too.
TEST_CASE("GPU moe_grouped_gemm_fp8_block_gate_up_silu equals the triplet above the warp range") {
  if (!HasCuda()) {
    MESSAGE("CUDA backend unavailable; skipping");
    return;
  }
  // P = 160: the BM=16 tile; N = 192 spans two scale-row blocks.
  Case c;
  c.e_count = 7; c.t_rows = 20; c.top_k = 8; c.k_dim = 256; c.n_cols = 192; c.seed = 800;
  RunGateUpCase(c, /*expect_bit_identical=*/true);
  // P = 1024: the pipelined BM=16 tile, then with the small-tile bound at 512 the
  // BM=64 tile; N = 130 leaves a 2-column last tile, K = 384 three K-blocks,
  // identity routing.
  Case b;
  b.e_count = 9; b.t_rows = 128; b.top_k = 8; b.k_dim = 384; b.n_cols = 130; b.block_n = 64;
  b.use_row_map = false; b.seed = 810;
  RunGateUpCase(b, true);
  vllm_test::SetEnv("VT_MOE_FP8_BLOCK_SMALL_TILE_MAX_P", "512");
  RunGateUpCase(b, true);
  vllm_test::UnsetEnv("VT_MOE_FP8_BLOCK_SMALL_TILE_MAX_P");
  // The Qwen3.6-35B-A3B prefill-chunk shape at reduced width: about 4 pairs per
  // expert over 64 experts, K = 2048 in 128-wide blocks.
  Case q;
  q.e_count = 64; q.t_rows = 32; q.top_k = 8; q.k_dim = 2048; q.n_cols = 128; q.seed = 820;
  RunGateUpCase(q, true);
}

// The warp kernels batch their weight loads (several output columns per warp,
// several chunks in flight per lane) without touching the order of any sum, so
// every layout must give the bits of the one-column, one-load layout that
// VT_MOE_FP8_BLOCK_WARP_LOADS=0 selects. K = 2048 runs one column with four
// loads, 1024 two columns with two, 512 and 256 four columns with one; N = 100
// leaves a short last column group.
TEST_CASE("GPU moe_grouped_gemm_fp8_block warp layouts are bit-identical to one load per lane") {
  if (!HasCuda()) {
    MESSAGE("CUDA backend unavailable; skipping");
    return;
  }
  constexpr const char* kKnob = "VT_MOE_FP8_BLOCK_WARP_LOADS";
  for (const int64_t k : {2048, 1024, 512, 256}) {
    CAPTURE(k);
    Case c;
    c.e_count = 5; c.t_rows = 2; c.top_k = 8; c.k_dim = k; c.n_cols = 100; c.block_n = 32;
    c.seed = 1000 + static_cast<uint32_t>(k);
    const Fixture f = Build(c);
    const Fixture u = BuildUp(c);
    Case cb = c;
    cb.bf16_out = true;
    const Fixture fb = Build(cb);
    vllm_test::SetEnv(kKnob, "0");
    const std::vector<float> one = Run(f);
    const std::vector<float> one_bf16 = Run(fb);
    const std::vector<uint16_t> one_fused = RunGateUp(f, u).fused;
    vllm_test::UnsetEnv(kKnob);
    const std::vector<float> batched = Run(f);
    CheckAgainstReference(f, batched);
    CHECK(batched == one);
    CHECK(Run(fb) == one_bf16);
    CHECK(RunGateUp(f, u).fused == one_fused);
  }
}

// --- W8A8 (VT_MOE_FP8_BLOCK_W8A8=1) ---------------------------------------------
// The activation rows quantized per 128-wide group by the CPU QuantFp8Group arm, which
// the CUDA kernel the W8A8 path calls matches byte for byte
// (test_ops_quant_fp8_group_cpu.cpp G6).
struct QuantAct {
  std::vector<uint8_t> q;  // [act_rows, K] e4m3
  std::vector<float> s;    // [act_rows, K / 128]
};

QuantAct QuantizeActCpu(const Fixture& f) {
  const Case& c = f.c;
  QuantAct qa;
  qa.q.resize(static_cast<size_t>(f.act_rows * c.k_dim));
  qa.s.resize(static_cast<size_t>(f.act_rows * (c.k_dim / 128)));
  const Device cpu{DeviceType::kCPU, 0};
  Queue cq{cpu, nullptr};
  Tensor x = MakeTensor(const_cast<uint16_t*>(f.act.data()), DType::kBF16, cpu,
                        {f.act_rows, c.k_dim});
  Tensor tq = MakeTensor(qa.q.data(), DType::kI8, cpu, {f.act_rows, c.k_dim});
  Tensor ts = MakeTensor(qa.s.data(), DType::kF32, cpu, {f.act_rows, c.k_dim / 128});
  vt::QuantFp8Group(cq, tq, ts, x, 128);
  return qa;
}

// out[p, n] = sum_kb a_s[row, kb] * s_e[n / 128, kb] * sum_{k in kb} a_q[row, k] * w_e[n, k]
// in double: what the W8A8 kernel computes, up to the f32 order of its sums.
std::vector<double> W8A8Reference(const Fixture& f, const QuantAct& qa) {
  const Case& c = f.c;
  const int64_t kb = c.k_dim / 128;
  std::vector<double> ref(static_cast<size_t>(f.P * c.n_cols), 0.0);
  for (int64_t p = 0; p < f.P; ++p) {
    const auto e = static_cast<size_t>(f.expert_ids[static_cast<size_t>(p)]);
    const int64_t r = c.use_row_map ? f.row_map[static_cast<size_t>(p)] : p;
    for (int64_t n = 0; n < c.n_cols; ++n) {
      double acc = 0.0;
      for (int64_t b = 0; b < kb; ++b) {
        double part = 0.0;
        for (int64_t k = b * 128; k < (b + 1) * 128; ++k)
          part += static_cast<double>(vt::F8E4M3ToF32(qa.q[static_cast<size_t>(r * c.k_dim + k)])) *
                  static_cast<double>(vt::F8E4M3ToF32(f.w[e][static_cast<size_t>(n * c.k_dim + k)]));
        acc += part * static_cast<double>(qa.s[static_cast<size_t>(r * kb + b)]) *
               static_cast<double>(f.s[e][static_cast<size_t>((n / c.block_n) * f.kb + b)]);
      }
      ref[static_cast<size_t>(p * c.n_cols + n)] = acc;
    }
  }
  return ref;
}

// Element-wise |got - ref| <= rtol * (|ref| + rms(ref)).
void CheckNear(const std::vector<float>& got, const std::vector<double>& ref, double rtol) {
  REQUIRE(got.size() == ref.size());
  double ss = 0.0;
  for (double v : ref) ss += v * v;
  const double rms = std::sqrt(ss / static_cast<double>(ref.size()));
  size_t bad = 0, first_bad = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double err = std::fabs(static_cast<double>(got[i]) - ref[i]);
    if (!(err <= rtol * (std::fabs(ref[i]) + rms))) {
      if (bad == 0) first_bad = i;
      ++bad;
    }
  }
  if (bad != 0) {
    CAPTURE(bad);
    CAPTURE(first_bad);
    CAPTURE(got[first_bad]);
    CAPTURE(ref[first_bad]);
    CAPTURE(rms);
  }
  CHECK(bad == 0);
}

// Above the warp range the switch runs the W8A8 tile, which must be the W8A8 reference
// to the MMA's summation (and one bf16 rounding on a bf16 output), at both tile heights
// and bit-identically between them. N = 192 leaves a half 128-column tile; P = 320
// over 9 experts (about 36 rows each) picks the 64-row tile and over 16 experts (20
// each) the 32-row one; K = 512 is four scale blocks.
TEST_CASE("GPU moe_grouped_gemm_fp8_block W8A8 matches the W8A8 block-scaled reference") {
  if (!HasFp8Mma()) {
    MESSAGE("no sm_89+ CUDA device; skipping");
    return;
  }
  constexpr const char* kW8A8 = "VT_MOE_FP8_BLOCK_W8A8";
  constexpr const char* kBM = "VT_MOE_FP8_BLOCK_W8A8_BM";
  for (const int64_t experts : {int64_t{9}, int64_t{16}}) {
    for (const bool bf16_out : {false, true}) {
      CAPTURE(experts);
      CAPTURE(bf16_out);
      Case c;
      c.e_count = experts; c.t_rows = 40; c.top_k = 8; c.k_dim = 512; c.n_cols = 192;
      c.bf16_out = bf16_out; c.use_row_map = !bf16_out;
      c.seed = 1200 + static_cast<uint32_t>(experts) + (bf16_out ? 50u : 0u);
      const Fixture f = Build(c);
      const std::vector<double> ref = W8A8Reference(f, QuantizeActCpu(f));
      vllm_test::SetEnv(kW8A8, "1");
      const std::vector<float> got = Run(f);
      vllm_test::SetEnv(kBM, "32");
      const std::vector<float> got32 = Run(f);
      vllm_test::SetEnv(kBM, "64");
      const std::vector<float> got64 = Run(f);
      vllm_test::UnsetEnv(kBM);
      vllm_test::UnsetEnv(kW8A8);
      // f32: the MMA's own accumulation order and precision, as CheckAgainstReference allows
      CheckNear(got, ref, bf16_out ? 8e-3 : 1e-3);
      CHECK(got32 == got64);
      CHECK(got == got64);
    }
  }
}

// The fused W8A8 gate+up runs both tables in one launch over one sort, then MoeSiluMul:
// the separate W8A8 gate and up GEMMs and MoeSiluMul, bit for bit.
TEST_CASE("GPU moe_grouped_gemm_fp8_block_gate_up_silu W8A8 equals the W8A8 triplet") {
  if (!HasFp8Mma()) {
    MESSAGE("no sm_89+ CUDA device; skipping");
    return;
  }
  vllm_test::SetEnv("VT_MOE_FP8_BLOCK_W8A8", "1");
  Case c;
  c.e_count = 9; c.t_rows = 40; c.top_k = 8; c.k_dim = 512; c.n_cols = 256; c.seed = 1300;
  const Fixture g = Build(c);
  const Fixture u = BuildUp(c);
  const GateUpOut out = RunGateUp(g, u);
  vllm_test::UnsetEnv("VT_MOE_FP8_BLOCK_W8A8");
  size_t diff = 0;
  for (size_t i = 0; i < out.fused.size(); ++i) diff += out.fused[i] != out.triplet[i] ? 1 : 0;
  CAPTURE(diff);
  CHECK(diff == 0);
}

// W8A8 against the exact-activation reference: what quantizing the activation to e4m3
// (3 mantissa bits, a scale per 128 values) costs, bounded so a wrong scale or a
// misplaced fragment cannot pass. Off, the same call is the W8A16 tile.
TEST_CASE("GPU moe_grouped_gemm_fp8_block W8A8 stays within e4m3 activation error") {
  if (!HasFp8Mma()) {
    MESSAGE("no sm_89+ CUDA device; skipping");
    return;
  }
  Case c;
  c.e_count = 9; c.t_rows = 40; c.top_k = 8; c.k_dim = 512; c.n_cols = 192; c.seed = 1400;
  const Fixture f = Build(c);
  const std::vector<float> w8a16 = Run(f);
  vllm_test::SetEnv("VT_MOE_FP8_BLOCK_W8A8", "1");
  const std::vector<float> w8a8 = Run(f);
  vllm_test::UnsetEnv("VT_MOE_FP8_BLOCK_W8A8");
  CheckAgainstReference(f, w8a16);
  double se = 0.0, ss = 0.0;
  for (size_t i = 0; i < w8a8.size(); ++i) {
    se += (static_cast<double>(w8a8[i]) - f.ref[i]) * (static_cast<double>(w8a8[i]) - f.ref[i]);
    ss += f.ref[i] * f.ref[i];
  }
  const double rel_rms = std::sqrt(se / ss);
  MESSAGE("W8A8 vs exact activation: relative RMS error " << rel_rms);
  CHECK(rel_rms > 1e-4);  // it did quantize
  CHECK(rel_rms < 6e-2);
}

TEST_CASE("GPU moe_grouped_gemm_fp8_block_gate_up_silu is bit-reproducible run to run") {
  if (!HasCuda()) {
    MESSAGE("CUDA backend unavailable; skipping");
    return;
  }
  Case c;
  c.e_count = 6; c.t_rows = 2; c.top_k = 8; c.k_dim = 512; c.n_cols = 128; c.seed = 900;
  const Fixture g = Build(c);
  const Fixture u = BuildUp(c);
  const std::vector<uint16_t> first = RunGateUp(g, u).fused;
  for (int rep = 0; rep < 3; ++rep) CHECK(RunGateUp(g, u).fused == first);
}
