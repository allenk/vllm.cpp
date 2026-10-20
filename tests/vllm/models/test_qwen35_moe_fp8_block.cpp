// Block-wise FP8 routed and shared experts on the Qwen3.5 MoE block -- the
// scheme `Qwen/Qwen3.6-35B-A3B-FP8` ships (per-expert `.weight` F8_E4M3 beside
// a `.weight_scale_inv` 128x128 grid, dynamic activation scales).
//
// WHAT IS PINNED. `MoeBlock`'s block-FP8 arm (the per-expert reference loop
// through `ExpertMlpFp8Block`) and the block arm of `SharedExpertUngated`,
// against the SAME block run through the bf16 arm with weights that are the
// EXACT dequantization of the fp8 ones. Power-of-two block scales keep every
// e4m3 value times its scale representable in bf16, so the two blocks hold the
// same weights and differ only in what the block GEMM does to the ACTIVATION:
// per-token, per-128-group e4m3 quantization with a dynamic scale, which is
// upstream's scheme (`per_token_group_quant_fp8`), not an error of this arm.
//
// That bounds the difference statistically rather than bitwise. The bounds
// below leave room for the activation rounding and none for a transposed
// weight, a block scale applied twice or not at all, or an expert bound to the
// wrong id: each of those moves the output by O(1), which a relative error of a
// few percent cannot hide. The subcases bind two of them -- swapped expert ids
// and a foreign down projection -- by requiring the same comparison to FAIL.
//
// Neither arm can be silently skipped: the fp8 block carries NO bf16 or fp4
// expert or shared-expert field, so any other arm would read an empty weight
// and throw.

#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/qwen3_5_moe_block.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace {

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;
using vllm::Fp8BlockWeight;
using vllm::MoeBlockWeights;
using vllm::OwnedTensor;

// Whole 128-blocks on every axis the block GEMM tiles: gate/up reduce over
// K = H = 2 blocks, down over K = I = 1 block, and every N is a multiple of 128.
constexpr int64_t kBlock = 128;
constexpr int64_t kH = 256;
constexpr int64_t kI = 128;
constexpr int64_t kIs = 128;
constexpr int64_t kE = 4;
constexpr int64_t kTopK = 2;
constexpr int64_t kT = 5;

uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// An e4m3fn code with exponent field 4..8, so the value is a normal number in
// [2^-3, 2^1 * 1.875] in magnitude: never NaN (field 15), never subnormal.
uint8_t E4m3Code(uint64_t seed) {
  const uint64_t h = Mix(seed);
  const auto sign = static_cast<uint8_t>((h >> 63) & 1u);
  const auto exp = static_cast<uint8_t>(4 + (h % 5));
  const auto mant = static_cast<uint8_t>((h >> 8) & 7u);
  return static_cast<uint8_t>((sign << 7) | (exp << 3) | mant);
}

float E4m3Value(uint8_t code) {
  const int sign = (code >> 7) & 1;
  const int exp = (code >> 3) & 0xF;
  const int mant = code & 7;
  const float v = std::ldexp(1.0F + static_cast<float>(mant) / 8.0F, exp - 7);
  return sign != 0 ? -v : v;
}

// A power of two per block, 2^-3 .. 2^-6: exact in f32, and times any e4m3
// value above it is exact in bf16 (3 mantissa bits never need more than 7).
float BlockScale(uint64_t seed) {
  return std::ldexp(1.0F, -3 - static_cast<int>(Mix(seed) % 4));
}

OwnedTensor Owned(DType dt, const std::vector<int64_t>& shape, size_t bytes) {
  OwnedTensor o;
  o.dtype = dt;
  o.rank = static_cast<int>(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) o.shape[i] = shape[i];
  o.bytes.resize(bytes);
  return o;
}

OwnedTensor CopyOf(const OwnedTensor& t) {
  std::vector<int64_t> shape(t.shape, t.shape + t.rank);
  OwnedTensor o = Owned(t.dtype, shape, t.bytes.size());
  std::memcpy(o.bytes.data(), t.bytes.data(), t.bytes.size());
  o.nk = t.nk;
  return o;
}

// One projection `[N=out, K=in]` in both residencies: the raw fp8 block weight
// the fp8 arm reads, and its exact bf16 dequantization in the Matmul-B `[K, N]`
// orientation the bf16 arm reads.
struct Projection {
  Fp8BlockWeight fp8;
  OwnedTensor bf16;
};

Projection MakeProjection(int64_t n, int64_t k, uint64_t seed) {
  Projection p;
  p.fp8.n = n;
  p.fp8.k = k;
  p.fp8.block_n = kBlock;
  p.fp8.block_k = kBlock;
  p.fp8.packed = Owned(DType::kI8, {n, k}, static_cast<size_t>(n * k));
  const int64_t rows = (n + kBlock - 1) / kBlock;
  const int64_t cols = (k + kBlock - 1) / kBlock;
  p.fp8.scale = Owned(DType::kF32, {rows, cols},
                      static_cast<size_t>(rows * cols) * sizeof(float));
  auto* scale = reinterpret_cast<float*>(p.fp8.scale.bytes.data());
  for (int64_t c = 0; c < rows * cols; ++c)
    scale[c] = BlockScale(seed * 1000003ULL + static_cast<uint64_t>(c));
  p.bf16 = Owned(DType::kBF16, {k, n}, static_cast<size_t>(n * k) * 2);
  auto* packed = p.fp8.packed.bytes.data();
  auto* dq = reinterpret_cast<uint16_t*>(p.bf16.bytes.data());
  for (int64_t r = 0; r < n; ++r) {
    for (int64_t c = 0; c < k; ++c) {
      const uint8_t code =
          E4m3Code(seed * 7919ULL + static_cast<uint64_t>(r * k + c));
      packed[r * k + c] = code;
      const float s = scale[(r / kBlock) * cols + (c / kBlock)];
      const float v = E4m3Value(code) * s;
      const uint16_t b = vt::F32ToBF16(v);
      REQUIRE(vt::BF16ToF32(b) == v);  // the twin is EXACT, or the test is moot
      dq[c * n + r] = b;               // [N,K] -> Matmul-B [K,N]
    }
  }
  return p;
}

OwnedTensor Bf16Matrix(int64_t rows, int64_t cols, uint64_t seed, float amp) {
  OwnedTensor o = Owned(DType::kBF16, {rows, cols},
                        static_cast<size_t>(rows * cols) * 2);
  auto* p = reinterpret_cast<uint16_t*>(o.bytes.data());
  for (int64_t i = 0; i < rows * cols; ++i) {
    const double u = static_cast<double>(Mix(seed + static_cast<uint64_t>(i)) >> 11) /
                     static_cast<double>(1ULL << 53);
    p[i] = vt::F32ToBF16(static_cast<float>((2.0 * u - 1.0) * amp));
  }
  return o;
}

struct Blocks {
  MoeBlockWeights fp8;
  MoeBlockWeights bf16;
};

// The same MoE block twice. `swap_experts` binds expert 0's fp8 weights to id 1
// and vice versa in the FP8 block only -- the "wrong id" defect.
// `foreign_down` replaces the bf16 twin of expert 0's down projection with an
// unrelated one -- what a loader that bound the wrong tensor would produce.
Blocks MakeBlocks(bool swap_experts = false, bool foreign_down = false) {
  Blocks b;
  b.fp8.router_gate = Bf16Matrix(kH, kE, 11, 0.25F);
  b.fp8.shared_gate = Bf16Matrix(kH, 1, 12, 0.25F);
  b.bf16.router_gate = CopyOf(b.fp8.router_gate);
  b.bf16.shared_gate = CopyOf(b.fp8.shared_gate);
  for (int64_t e = 0; e < kE; ++e) {
    const auto s = static_cast<uint64_t>(100 + e * 10);
    Projection g = MakeProjection(kI, kH, s + 1);
    Projection u = MakeProjection(kI, kH, s + 2);
    Projection d = MakeProjection(kH, kI, s + 3);
    b.bf16.expert_gate.push_back(std::move(g.bf16));
    b.bf16.expert_up.push_back(std::move(u.bf16));
    b.bf16.expert_down.push_back(std::move(d.bf16));
    b.fp8.expert_gate_fp8_block.push_back(std::move(g.fp8));
    b.fp8.expert_up_fp8_block.push_back(std::move(u.fp8));
    b.fp8.expert_down_fp8_block.push_back(std::move(d.fp8));
  }
  if (swap_experts) {
    std::swap(b.fp8.expert_gate_fp8_block[0], b.fp8.expert_gate_fp8_block[1]);
    std::swap(b.fp8.expert_up_fp8_block[0], b.fp8.expert_up_fp8_block[1]);
    std::swap(b.fp8.expert_down_fp8_block[0], b.fp8.expert_down_fp8_block[1]);
  }
  if (foreign_down) {
    Projection d = MakeProjection(kH, kI, 777);
    b.bf16.expert_down[0] = std::move(d.bf16);
  }
  Projection sg = MakeProjection(kIs, kH, 201);
  Projection su = MakeProjection(kIs, kH, 202);
  Projection sd = MakeProjection(kH, kIs, 203);
  b.bf16.shared_gate_proj = std::move(sg.bf16);
  b.bf16.shared_up_proj = std::move(su.bf16);
  b.bf16.shared_down_proj = std::move(sd.bf16);
  b.fp8.shared_gate_proj_fp8_block = std::move(sg.fp8);
  b.fp8.shared_up_proj_fp8_block = std::move(su.fp8);
  b.fp8.shared_down_proj_fp8_block = std::move(sd.fp8);
  return b;
}

vllm::HfConfig Config() {
  vllm::HfConfig c;
  c.hidden_size = kH;
  c.num_experts = kE;
  c.num_experts_per_tok = kTopK;
  c.moe_intermediate_size = kI;
  c.shared_expert_intermediate_size = kIs;
  return c;
}

std::vector<uint16_t> Hidden() {
  const OwnedTensor h = Bf16Matrix(kT, kH, 31, 1.0F);
  std::vector<uint16_t> x(static_cast<size_t>(kT * kH));
  std::memcpy(x.data(), h.bytes.data(), x.size() * 2);
  return x;
}

bool HasBackend(DeviceType t) {
  try {
    vt::GetBackend(t);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Tensor MakeT(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
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

std::vector<float> RunBlock(DeviceType dt, const MoeBlockWeights& w,
                            const std::vector<uint16_t>& x) {
  Backend& b = vt::GetBackend(dt);
  QueueGuard qg(b);
  const size_t bytes = x.size() * sizeof(uint16_t);
  void* p = b.Alloc(bytes);
  b.Copy(qg.q, p, x.data(), bytes);
  const Tensor dh = MakeT(p, DType::kBF16, Device{dt, 0}, {kT, kH});
  std::vector<uint16_t> out(static_cast<size_t>(kT * kH));
  try {
    vllm::MoeBlockOutput r = vllm::RunMoeBlock(qg.q, w, Config(), dh, kT);
    b.Copy(qg.q, out.data(), r.tensor.data, out.size() * sizeof(uint16_t));
    b.Synchronize(qg.q);
  } catch (...) {
    b.Free(p);
    throw;
  }
  b.Free(p);
  std::vector<float> f(out.size());
  for (size_t i = 0; i < out.size(); ++i) f[i] = vt::BF16ToF32(out[i]);
  return f;
}

struct Agreement {
  double cosine = 0.0;
  double rel_l2 = 0.0;
  double norm_ratio = 0.0;
};

Agreement Compare(const std::vector<float>& got, const std::vector<float>& ref) {
  REQUIRE(got.size() == ref.size());
  double dot = 0.0, ng = 0.0, nr = 0.0, nd = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    REQUIRE(std::isfinite(got[i]));
    dot += static_cast<double>(got[i]) * ref[i];
    ng += static_cast<double>(got[i]) * got[i];
    nr += static_cast<double>(ref[i]) * ref[i];
    const double d = static_cast<double>(got[i]) - ref[i];
    nd += d * d;
  }
  REQUIRE(nr > 0.0);
  Agreement a;
  a.cosine = dot / std::sqrt(ng * nr);
  a.rel_l2 = std::sqrt(nd / nr);
  a.norm_ratio = std::sqrt(ng / nr);
  return a;
}

// The activation-rounding allowance. e4m3 keeps 3 mantissa bits, so one
// quantized activation is within 2^-4 of its value and a K-long dot product of
// independent roundings lands near 2-3% relative; two quantized GEMMs in series
// (gate/up, then down) roughly double it. Measured on this fixture: see the
// MESSAGE lines. A wrong weight binding is O(1).
constexpr double kMaxRelL2 = 0.08;
constexpr double kMinCosine = 0.995;

void CheckArmsAgree(DeviceType dt, const char* what) {
  const Blocks b = MakeBlocks();
  const std::vector<uint16_t> x = Hidden();
  const std::vector<float> fp8 = RunBlock(dt, b.fp8, x);
  const std::vector<float> bf16 = RunBlock(dt, b.bf16, x);
  const Agreement a = Compare(fp8, bf16);
  MESSAGE(std::string(what) << ": cosine " << a.cosine << ", relative L2 " << a.rel_l2
               << ", norm ratio " << a.norm_ratio);
  CHECK(a.cosine > kMinCosine);
  CHECK(a.rel_l2 < kMaxRelL2);
  CHECK(std::fabs(a.norm_ratio - 1.0) < kMaxRelL2);

  // Run-to-run: the same block on the same input is bit-identical.
  const std::vector<float> again = RunBlock(dt, b.fp8, x);
  CHECK(std::memcmp(again.data(), fp8.data(), fp8.size() * sizeof(float)) == 0);
}

void CheckDefectIsVisible(DeviceType dt, bool swap, bool foreign_down,
                          const char* what) {
  const Blocks b = MakeBlocks(swap, foreign_down);
  const std::vector<uint16_t> x = Hidden();
  const Agreement a = Compare(RunBlock(dt, b.fp8, x), RunBlock(dt, b.bf16, x));
  MESSAGE(std::string(what) << " (must disagree): cosine " << a.cosine << ", relative L2 "
               << a.rel_l2);
  CHECK(a.rel_l2 > kMaxRelL2);
}

}  // namespace

TEST_CASE("qwen3_5 moe: block-wise FP8 experts and shared expert match the exact bf16 twin (CPU)") {
  CheckArmsAgree(DeviceType::kCPU, "CPU");
  SUBCASE("a routed expert bound to the wrong id is visible") {
    CheckDefectIsVisible(DeviceType::kCPU, /*swap=*/true, false, "CPU swapped ids");
  }
  SUBCASE("a down projection that is not the twin's is visible") {
    CheckDefectIsVisible(DeviceType::kCPU, false, /*foreign_down=*/true,
                         "CPU wrong down");
  }
}

TEST_CASE("qwen3_5 moe: block-wise FP8 experts and shared expert match the exact bf16 twin (CUDA)") {
  if (!HasBackend(DeviceType::kCUDA)) {
    std::printf("[SKIP] no CUDA backend: the block-FP8 MoE arm NOT exercised on CUDA\n");
    return;
  }
  CheckArmsAgree(DeviceType::kCUDA, "CUDA");
  SUBCASE("a routed expert bound to the wrong id is visible") {
    CheckDefectIsVisible(DeviceType::kCUDA, /*swap=*/true, false, "CUDA swapped ids");
  }
}
