// Flint's small fused CUDA kernels: the kFlint glue table (flint_device.h).
// Ported from vLLM nightly `vllm/models/flint/ops/qk_norm_rope.py` (the Triton
// `_qk_norm_rope_cache_kernel`), `ops/norm.py` (`_rms_norm_kernel`), the sigmoid
// arm of vLLM's fused_topk and csrc act_and_mul<silu>, which are what vLLM runs
// for this model on SM120.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "vllm/model_executor/models/flint_device.h"
#include "vt/ops.h"  // OpId, RegisterOp, DeviceType

namespace vllm::flint {
namespace {

using vt::DeviceType;
using vt::OpId;
using vt::Queue;
using vt::RegisterOp;

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("flint cuda: ") + what + ": " + cudaGetErrorString(err));
  }
}

__device__ __forceinline__ float RoundBf16(float x) {
  return __bfloat162float(__float2bfloat16(x));
}

// One block per (token, head); thread i owns the rotation pair (i, i + D/2), so
// a head's sum of squares is a block reduction over D/2 threads. Heads
// [0, Hq) are q, [Hq, Hq+Hkv) k, the rest v (copied unchanged).
template <int kHalf>
__global__ void QkNormRopeKvCacheKernel(__nv_bfloat16* __restrict__ q_out,
                                        const __nv_bfloat16* __restrict__ qkv, int64_t qkv_stride,
                                        const __nv_bfloat16* __restrict__ q_w,
                                        const __nv_bfloat16* __restrict__ k_w,
                                        const float* __restrict__ cos_sin,
                                        __nv_bfloat16* __restrict__ k_cache,
                                        __nv_bfloat16* __restrict__ v_cache, int64_t stride_block,
                                        int64_t stride_slot, int64_t stride_head,
                                        int64_t block_size, const int64_t* __restrict__ slots,
                                        int hq, int hkv, float eps) {
  constexpr int kD = 2 * kHalf;
  const int64_t t = blockIdx.x;
  const int head = static_cast<int>(blockIdx.y);
  const int i = static_cast<int>(threadIdx.x);
  const bool is_q = head < hq;
  const bool is_v = head >= hq + hkv;
  const __nv_bfloat16* row = qkv + t * qkv_stride + static_cast<int64_t>(head) * kD;
  const __nv_bfloat16 b1 = row[i];
  const __nv_bfloat16 b2 = row[kHalf + i];
  const int64_t slot = is_q ? -1 : slots[t];
  const int kv_head = is_v ? head - hq - hkv : head - hq;
  if (is_v) {
    if (slot >= 0) {
      __nv_bfloat16* dst = v_cache + (slot / block_size) * stride_block +
                           (slot % block_size) * stride_slot + kv_head * stride_head;
      dst[i] = b1;
      dst[kHalf + i] = b2;
    }
    return;
  }
  const float x1 = __bfloat162float(b1);
  const float x2 = __bfloat162float(b2);
  float sum_sq = x1 * x1 + x2 * x2;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, o);
  __shared__ float warp_sum[kHalf / 32];
  if ((i & 31) == 0) warp_sum[i >> 5] = sum_sq;
  __syncthreads();
  float total = 0.f;
#pragma unroll
  for (int w = 0; w < kHalf / 32; ++w) total += warp_sum[w];
  const float rs = rsqrtf(total / static_cast<float>(kD) + eps);
  const __nv_bfloat16* weight = is_q ? q_w : k_w;
  // bf16(x * rs * w), then back to f32 for the rotation, as the Triton kernel.
  float n1 = RoundBf16(x1 * rs * __bfloat162float(weight[i]));
  float n2 = RoundBf16(x2 * rs * __bfloat162float(weight[kHalf + i]));
  if (cos_sin != nullptr) {
    const float c = cos_sin[t * kD + i];
    const float s = cos_sin[t * kD + kHalf + i];
    const float r1 = n1 * c - n2 * s;
    const float r2 = n2 * c + n1 * s;
    n1 = r1;
    n2 = r2;
  }
  const __nv_bfloat16 o1 = __float2bfloat16(n1);
  const __nv_bfloat16 o2 = __float2bfloat16(n2);
  if (is_q) {
    __nv_bfloat16* dst = q_out + (t * hq + head) * kD;
    dst[i] = o1;
    dst[kHalf + i] = o2;
  } else if (slot >= 0) {
    __nv_bfloat16* dst = k_cache + (slot / block_size) * stride_block +
                         (slot % block_size) * stride_slot + kv_head * stride_head;
    dst[i] = o1;
    dst[kHalf + i] = o2;
  }
}

template <int kHalf>
void LaunchQkNormRopeKvCache(cudaStream_t s, void* q_out, const void* qkv, int64_t qkv_stride,
                             const void* q_w, const void* k_w, const float* cos_sin,
                             void* k_cache, void* v_cache, int64_t stride_block,
                             int64_t stride_slot, int64_t stride_head, int64_t block_size,
                             const int64_t* slots, int64_t num_tokens, int64_t hq, int64_t hkv,
                             float eps) {
  const dim3 grid(static_cast<unsigned>(num_tokens), static_cast<unsigned>(hq + 2 * hkv));
  QkNormRopeKvCacheKernel<kHalf><<<grid, kHalf, 0, s>>>(
      static_cast<__nv_bfloat16*>(q_out), static_cast<const __nv_bfloat16*>(qkv), qkv_stride,
      static_cast<const __nv_bfloat16*>(q_w), static_cast<const __nv_bfloat16*>(k_w), cos_sin,
      static_cast<__nv_bfloat16*>(k_cache), static_cast<__nv_bfloat16*>(v_cache), stride_block,
      stride_slot, stride_head, block_size, slots, static_cast<int>(hq), static_cast<int>(hkv),
      eps);
}

void QkNormRopeKvCacheLaunch(Queue& q, void* q_out, const void* qkv, int64_t qkv_stride,
                             const void* q_w, const void* k_w, const float* cos_sin,
                             void* k_cache, void* v_cache, int64_t stride_block,
                             int64_t stride_slot, int64_t stride_head, int64_t block_size,
                             const int64_t* slots, int64_t num_tokens, int64_t num_q_heads,
                             int64_t num_kv_heads, int64_t head_dim, float eps) {
  if (num_tokens == 0) return;
  auto s = static_cast<cudaStream_t>(q.handle);
  switch (head_dim) {
    case 64:
      LaunchQkNormRopeKvCache<32>(s, q_out, qkv, qkv_stride, q_w, k_w, cos_sin, k_cache, v_cache,
                                  stride_block, stride_slot, stride_head, block_size, slots,
                                  num_tokens, num_q_heads, num_kv_heads, eps);
      break;
    case 128:
      LaunchQkNormRopeKvCache<64>(s, q_out, qkv, qkv_stride, q_w, k_w, cos_sin, k_cache, v_cache,
                                  stride_block, stride_slot, stride_head, block_size, slots,
                                  num_tokens, num_q_heads, num_kv_heads, eps);
      break;
    case 256:
      LaunchQkNormRopeKvCache<128>(s, q_out, qkv, qkv_stride, q_w, k_w, cos_sin, k_cache, v_cache,
                                   stride_block, stride_slot, stride_head, block_size, slots,
                                   num_tokens, num_q_heads, num_kv_heads, eps);
      break;
    default:
      throw std::runtime_error("flint cuda: qk_norm_rope_kv_cache supports head_dim 64, 128 or "
                               "256, got " + std::to_string(head_dim));
  }
  Check(cudaGetLastError(), "qk_norm_rope_kv_cache launch");
}

constexpr int kTopKMaxPerLane = 16;  // E <= 512
constexpr int kTopKMax = 16;
constexpr int kTopKWarpsPerBlock = 4;

// One warp per token. Lane l holds experts l, l+32, ...; each round takes the warp
// maximum (ties to the lower expert index), and its owner drops it. A taken or
// absent expert reads -1, below every sigmoid score.
__global__ void SigmoidTopKKernel(float* __restrict__ weights, int32_t* __restrict__ ids,
                                  const __nv_bfloat16* __restrict__ logits, int64_t num_tokens,
                                  int num_experts, int top_k) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * kTopKWarpsPerBlock + threadIdx.x / 32;
  if (t >= num_tokens) return;
  const int lane = static_cast<int>(threadIdx.x & 31);
  const __nv_bfloat16* row = logits + t * num_experts;
  float score[kTopKMaxPerLane];
#pragma unroll
  for (int j = 0; j < kTopKMaxPerLane; ++j) {
    const int e = j * 32 + lane;
    score[j] = e < num_experts ? 1.f / (1.f + expf(-__bfloat162float(row[e]))) : -1.f;
  }
  float picked[kTopKMax];
  int picked_id[kTopKMax];
  float sum = 0.f;
  for (int k = 0; k < top_k; ++k) {
    float best = -1.f;
    int best_e = 0x7fffffff;
#pragma unroll
    for (int j = 0; j < kTopKMaxPerLane; ++j) {
      if (score[j] > best) {
        best = score[j];
        best_e = j * 32 + lane;
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, best, o);
      const int oe = __shfl_xor_sync(0xffffffffu, best_e, o);
      if (ob > best || (ob == best && oe < best_e)) {
        best = ob;
        best_e = oe;
      }
    }
#pragma unroll
    for (int j = 0; j < kTopKMaxPerLane; ++j)
      if (j * 32 + lane == best_e) score[j] = -1.f;
    picked[k] = best;
    picked_id[k] = best_e;
    sum += best;
  }
  if (lane == 0) {
    for (int k = 0; k < top_k; ++k) {
      weights[t * top_k + k] = picked[k] / sum;
      ids[t * top_k + k] = picked_id[k];
    }
  }
}

void SigmoidTopKLaunch(Queue& q, float* weights, int32_t* ids, const void* logits,
                       int64_t num_tokens, int64_t num_experts, int64_t top_k) {
  if (num_experts < 1 || num_experts > 32 * kTopKMaxPerLane || top_k < 1 || top_k > kTopKMax ||
      top_k > num_experts) {
    throw std::runtime_error("flint cuda: sigmoid_topk needs 1 <= top_k <= 16, top_k <= E <= 512");
  }
  if (num_tokens == 0) return;
  const auto blocks =
      static_cast<unsigned>((num_tokens + kTopKWarpsPerBlock - 1) / kTopKWarpsPerBlock);
  SigmoidTopKKernel<<<blocks, 32 * kTopKWarpsPerBlock, 0, static_cast<cudaStream_t>(q.handle)>>>(
      weights, ids, static_cast<const __nv_bfloat16*>(logits), num_tokens,
      static_cast<int>(num_experts), static_cast<int>(top_k));
  Check(cudaGetLastError(), "sigmoid_topk launch");
}

__device__ __forceinline__ float RsqrtApproxFtz(float x) {
  float y;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

constexpr int kNormThreads = 256;
constexpr int kNormMaxVec = 4;  // 8 bf16 per vector: hidden <= 256 * 4 * 8 = 8192

union Bf16x8 {
  uint4 raw;
  __nv_bfloat16 h[8];
};

// One block per row; thread i owns the 8-element vectors i, i + 256, ... and keeps
// its f32 sums s in registers, so the output comes from the unrounded s while the
// residual is stored rounded, as the Triton kernel does. kStore: the residual is
// written with x and not read (the first layer).
template <bool kStore>
__global__ void AddRmsNormKernel(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* x,
                                 __nv_bfloat16* residual, const __nv_bfloat16* __restrict__ w,
                                 int hidden, float eps) {
  const int64_t base = static_cast<int64_t>(blockIdx.x) * hidden;
  const int nvec = hidden / 8;
  const auto* xv = reinterpret_cast<const uint4*>(x + base);
  float s[kNormMaxVec][8];
  float sum_sq = 0.f;
#pragma unroll
  for (int j = 0; j < kNormMaxVec; ++j) {
    const int v = static_cast<int>(threadIdx.x) + j * kNormThreads;
    if (v >= nvec) break;
    Bf16x8 a;
    a.raw = xv[v];
#pragma unroll
    for (int k = 0; k < 8; ++k) s[j][k] = __bfloat162float(a.h[k]);
    if constexpr (kStore) {
      reinterpret_cast<uint4*>(residual + base)[v] = a.raw;
    } else if (residual != nullptr) {
      auto* rv = reinterpret_cast<uint4*>(residual + base);
      Bf16x8 r;
      r.raw = rv[v];
      Bf16x8 o;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        s[j][k] += __bfloat162float(r.h[k]);
        o.h[k] = __float2bfloat16(s[j][k]);
      }
      rv[v] = o.raw;
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) sum_sq += s[j][k] * s[j][k];
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, o);
  __shared__ float warp_sum[kNormThreads / 32];
  const int lane = static_cast<int>(threadIdx.x & 31), warp = static_cast<int>(threadIdx.x >> 5);
  if (lane == 0) warp_sum[warp] = sum_sq;
  __syncthreads();
  float total = 0.f;
#pragma unroll
  for (int i = 0; i < kNormThreads / 32; ++i) total += warp_sum[i];
  const float rs = RsqrtApproxFtz(total / static_cast<float>(hidden) + eps);
  const auto* wv = reinterpret_cast<const uint4*>(w);
  auto* ov = reinterpret_cast<uint4*>(out + base);
#pragma unroll
  for (int j = 0; j < kNormMaxVec; ++j) {
    const int v = static_cast<int>(threadIdx.x) + j * kNormThreads;
    if (v >= nvec) break;
    Bf16x8 wt;
    wt.raw = wv[v];
    Bf16x8 o;
#pragma unroll
    for (int k = 0; k < 8; ++k) o.h[k] = __float2bfloat16(s[j][k] * rs * __bfloat162float(wt.h[k]));
    ov[v] = o.raw;
  }
}

template <bool kStore>
void LaunchAddRmsNorm(Queue& q, void* out, const void* x, void* residual, const void* w,
                      int64_t rows, int64_t hidden, float eps) {
  if (hidden % 8 != 0 || hidden > 8 * kNormThreads * kNormMaxVec) {
    throw std::runtime_error("flint cuda: add_rms_norm needs hidden % 8 == 0 and hidden <= 8192");
  }
  if (kStore && residual == nullptr) {
    throw std::runtime_error("flint cuda: rms_norm_store_residual needs a residual");
  }
  if (rows == 0) return;
  AddRmsNormKernel<kStore><<<static_cast<unsigned>(rows), kNormThreads, 0,
                             static_cast<cudaStream_t>(q.handle)>>>(
      static_cast<__nv_bfloat16*>(out), static_cast<const __nv_bfloat16*>(x),
      static_cast<__nv_bfloat16*>(residual), static_cast<const __nv_bfloat16*>(w),
      static_cast<int>(hidden), eps);
  Check(cudaGetLastError(), "add_rms_norm launch");
}

void AddRmsNormLaunch(Queue& q, void* out, const void* x, void* residual, const void* w,
                      int64_t rows, int64_t hidden, float eps) {
  LaunchAddRmsNorm<false>(q, out, x, residual, w, rows, hidden, eps);
}

void RmsNormStoreResidualLaunch(Queue& q, void* out, const void* x, void* residual,
                                const void* w, int64_t rows, int64_t hidden, float eps) {
  LaunchAddRmsNorm<true>(q, out, x, residual, w, rows, hidden, eps);
}

// Grid-stride over 8-element vectors of the [rows, d] output.
__global__ void RoutedSiluAndMulKernel(__nv_bfloat16* __restrict__ out,
                                       const __nv_bfloat16* __restrict__ gu, int64_t nvec,
                                       int64_t dvec) {
  const auto* in = reinterpret_cast<const uint4*>(gu);
  auto* o = reinterpret_cast<uint4*>(out);
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t v = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; v < nvec;
       v += step) {
    const int64_t row = v / dvec, c = v - row * dvec;
    Bf16x8 g, u, r;
    g.raw = in[row * 2 * dvec + c];
    u.raw = in[row * 2 * dvec + dvec + c];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float x = __bfloat162float(g.h[k]);
      const __nv_bfloat16 silu = __float2bfloat16(x / (1.0f + expf(-x)));
      r.h[k] = __float2bfloat16(__bfloat162float(silu) * __bfloat162float(u.h[k]));
    }
    o[v] = r.raw;
  }
}

void RoutedSiluAndMulLaunch(Queue& q, void* out, const void* gu, int64_t rows, int64_t d) {
  if (d % 8 != 0) throw std::runtime_error("flint cuda: routed_silu_and_mul needs d % 8 == 0");
  const int64_t dvec = d / 8, nvec = rows * dvec;
  if (nvec == 0) return;
  constexpr int kThreads = 256;
  const int64_t blocks = std::min<int64_t>((nvec + kThreads - 1) / kThreads, 4096);
  RoutedSiluAndMulKernel<<<static_cast<unsigned>(blocks), kThreads, 0,
                           static_cast<cudaStream_t>(q.handle)>>>(
      static_cast<__nv_bfloat16*>(out), static_cast<const __nv_bfloat16*>(gu), nvec, dvec);
  Check(cudaGetLastError(), "routed_silu_and_mul launch");
}

void* DeviceAlias(void* pinned) {
  void* dev = nullptr;
  if (cudaHostGetDevicePointer(&dev, pinned, 0) != cudaSuccess) {
    (void)cudaGetLastError();  // not mapped: clear the sticky error, report "no alias"
    return nullptr;
  }
  return dev;
}

constexpr int kPrologueThreads = 256;

// One block per row. Thread 0 resolves the row's id and copies its scalars; the
// block copies the row's block-table entries and then its embedding row.
__global__ void DecodePrologueKernel(char* __restrict__ block, FlintStepLayout lay,
                                     int32_t* __restrict__ positions, int64_t* __restrict__ slots,
                                     int32_t* __restrict__ block_table,
                                     int32_t* __restrict__ seq_lens,
                                     int32_t* __restrict__ query_start_loc,
                                     __nv_bfloat16* __restrict__ hidden_out,
                                     const __nv_bfloat16* __restrict__ embed, int rows, int cols,
                                     int hidden, int vocab) {
  const int r = static_cast<int>(blockIdx.x);
  auto* hdr = reinterpret_cast<FlintStepHeader*>(block);
  __shared__ int32_t id;
  if (threadIdx.x == 0) {
    int32_t t = r < hdr->override_count ? hdr->override_ids[r]
                                        : reinterpret_cast<const int32_t*>(block + lay.token_ids)[r];
    if (t < 0 || t >= vocab) {
      hdr->bad_row = r;
      hdr->bad_id = t;
      __threadfence_system();
      t = t < 0 ? 0 : vocab - 1;  // keep the gather in bounds, as vt::Embedding does
    }
    id = t;
    positions[r] = reinterpret_cast<const int32_t*>(block + lay.positions)[r];
    seq_lens[r] = reinterpret_cast<const int32_t*>(block + lay.seq_lens)[r];
    slots[r] = reinterpret_cast<const int64_t*>(block + lay.slots)[r];
    const auto* qsl = reinterpret_cast<const int32_t*>(block + lay.query_start_loc);
    query_start_loc[r] = qsl[r];
    if (r == rows - 1) query_start_loc[rows] = qsl[rows];
  }
  const auto* bt = reinterpret_cast<const int32_t*>(block + lay.block_table) +
                   static_cast<int64_t>(r) * cols;
  for (int c = static_cast<int>(threadIdx.x); c < cols; c += kPrologueThreads)
    block_table[static_cast<int64_t>(r) * cols + c] = bt[c];
  __syncthreads();
  const auto* src = reinterpret_cast<const uint4*>(embed + static_cast<int64_t>(id) * hidden);
  auto* dst = reinterpret_cast<uint4*>(hidden_out + static_cast<int64_t>(r) * hidden);
  for (int v = static_cast<int>(threadIdx.x); v < hidden / 8; v += kPrologueThreads) dst[v] = src[v];
}

void DecodePrologueLaunch(Queue& q, void* block, int32_t* positions, int64_t* slots,
                          int32_t* block_table, int32_t* seq_lens, int32_t* query_start_loc,
                          void* hidden_out, const void* embed, int64_t rows, int64_t cols,
                          int64_t hidden, int64_t vocab) {
  if (hidden % 8 != 0) throw std::runtime_error("flint cuda: decode_prologue needs hidden % 8 == 0");
  if (vocab <= 0) throw std::runtime_error("flint cuda: decode_prologue with an empty vocabulary");
  if (rows == 0) return;
  DecodePrologueKernel<<<static_cast<unsigned>(rows), kPrologueThreads, 0,
                         static_cast<cudaStream_t>(q.handle)>>>(
      static_cast<char*>(block), MakeFlintStepLayout(rows, cols), positions, slots, block_table,
      seq_lens, query_start_loc, static_cast<__nv_bfloat16*>(hidden_out),
      static_cast<const __nv_bfloat16*>(embed), static_cast<int>(rows), static_cast<int>(cols),
      static_cast<int>(hidden), static_cast<int>(vocab));
  Check(cudaGetLastError(), "decode_prologue launch");
}

const FlintDeviceKernels kFlintKernels = {
    &QkNormRopeKvCacheLaunch, &SigmoidTopKLaunch,      &AddRmsNormLaunch,
    &RmsNormStoreResidualLaunch, &RoutedSiluAndMulLaunch, &DeviceAlias,
    &DecodePrologueLaunch};

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kFlint, DeviceType::kCUDA,
               const_cast<void*>(static_cast<const void*>(&kFlintKernels)));
  }
} registrar;

}  // namespace
}  // namespace vllm::flint
