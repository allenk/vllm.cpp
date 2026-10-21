// Ported from: vllm/v1/sample/ops/topk_topp_sampler.py + vllm/v1/sample/sampler.py @ e24d1b24.
//
// Correctness-grade CUDA kernels for the V1 sampling ops (M1.7 Task 2), mirroring
// the CPU reference (src/vt/cpu/cpu_sample.cpp) element for element:
//   - apply_temperature: grid-stride, per-row temp with the eps greedy guard.
//   - greedy_argmax / random_sample: ONE BLOCK per row, single-threaded scan so
//     the lowest-index tie-break matches torch.argmax / the CPU reference exactly.
//   - compute_probs / compute_logprobs: block-per-row max-subtracted softmax.
//   - apply_top_k_top_p: SORT-FREE block-cooperative pivot-bracket THRESHOLD
//     search (one block per row), mirroring flashinfer's TopK/TopPRenormProb
//     (sampling.cuh). It replaces the old per-row full-vocab thrust::stable_sort
//     + <<<n,1>>> single-thread top-p cumsum + blocking cudaStreamSynchronize.
//     The kept set is identical to apply_top_k_top_p_pytorch for distinct logits
//     (validated by the scalar mirror + cross-check in tests/vt/test_ops_sample.cpp).
// NOTE: this file is built + verified on dgx.casa (the CI box is CPU-only); the
// CUDA parity tests are HasCuda-guarded.
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <limits>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/sample_common.h"

namespace vt::cuda {
namespace {

constexpr int kBlock = 256;
// -INFINITY rather than the macro: MSVC defines INFINITY as a double-typed
// expression that overflows, so `constexpr float x = -INFINITY;` is rejected
// with "floating-point value does not fit in required floating-point type".
// The numeric_limits form is the portable spelling of the same value.
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

void Check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("vt cuda: ") + what + ": " + cudaGetErrorString(err));
  }
}

cudaStream_t AsStream(const Queue& q) { return static_cast<cudaStream_t>(q.handle); }

unsigned GridFor(int64_t n) {
  const int64_t blocks = (n + kBlock - 1) / kBlock;
  return static_cast<unsigned>(blocks < 4096 ? blocks : 4096);
}

// The RNG and the argmax reduce come from vt/sample_common.h, which cpu_sample.cpp
// also includes -- so "bit-identical to the CPU reference" is a property of the
// build rather than of two copies staying in step.
using vt::sample::ArgReduce;
using vt::sample::GumbelScore;
using vt::sample::kArgSentinel;

// --- apply_temperature ------------------------------------------------------
__global__ void ApplyTemperatureKernel(float* logits, const float* temp, int64_t n, int64_t v,
                                       bool all_random) {
  const int64_t total = n * v;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += step) {
    const int64_t row = idx / v;
    float t = temp[row];
    if (!all_random && t < kSamplingEps) t = 1.0f;
    logits[idx] /= t;
  }
}

void ApplyTemperatureCuda(Queue& q, Tensor& logits, const Tensor& temp, bool all_random) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  ApplyTemperatureKernel<<<GridFor(n * v), kBlock, 0, AsStream(q)>>>(
      logits.Ptr<float>(), temp.Ptr<float>(), n, v, all_random);
  Check(cudaGetLastError(), "apply_temperature launch");
}

// --- greedy_argmax (two-pass multi-block reduction, exact lowest-index tie) ---
// The old kernel was a single block of a single thread scanning the whole vocab
// (~151k) serially -- ~7.5 ms/token on the decode path (a single-SM, latency-
// bound scan). Here we grid-stride the vocab across many blocks (pass 1 -> per-
// block partials) and reduce the partials (pass 2), keeping torch.argmax
// semantics: the maximum value wins; on ties the LOWEST index wins. The reduce
// operator compares the true global index (not thread/block order) so the tie-
// break is order-independent. Unfilled lanes carry (-inf, INT64_MAX) so a real
// index -- even one whose logit is -inf (all-masked row) -- always beats an empty
// lane, yielding index 0 for an all-(-inf) row, exactly like the CPU reference.
//
// PARAMETERISED ON THE SCORE (#1984). The reduction below is the same whether
// the value at (row, j) is a logit or a Gumbel score, and it is the ONLY
// order-independent argmax in this file, so `random_sample` instantiates it
// rather than growing a second hand-written copy that could drift from it. The
// greedy instantiation compiles the same expression it always did.
struct LogitScore {
  const float* logits;
  int64_t v;
  __device__ float operator()(int64_t row, int64_t j) const { return logits[row * v + j]; }
};

// probs / q with q ~ Exp(1), i.e. one element of upstream's `probs.div_(q)`
// (topk_topp_sampler.py::sample_with_exponential_noise). Computed per element
// rather than materialised, exactly as the serial kernel below does, so the
// float the reduction sees is the SAME float -- this is what makes the parallel
// path bit-identical to the serial one rather than merely close.
struct GumbelScore2D {
  const float* probs;
  const int64_t* seeds;
  int64_t v;
  __device__ float operator()(int64_t row, int64_t j) const {
    return GumbelScore(probs[row * v + j], static_cast<uint64_t>(seeds[row]), row, j);
  }
};

template <typename Score>
__global__ void ArgmaxPartialKernel(float* part_val, int64_t* part_idx, Score score, int64_t v,
                                    int blocks_per_row) {
  const int64_t row = blockIdx.y;
  const int blk = blockIdx.x;
  __shared__ float sv[kBlock];
  __shared__ int64_t si[kBlock];

  float bv = kNegInf;
  int64_t bi = kArgSentinel;
  const int64_t stride = static_cast<int64_t>(blocks_per_row) * blockDim.x;
  for (int64_t j = static_cast<int64_t>(blk) * blockDim.x + threadIdx.x; j < v; j += stride)
    ArgReduce(bv, bi, score(row, j), j);

  sv[threadIdx.x] = bv;
  si[threadIdx.x] = bi;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s)
      ArgReduce(sv[threadIdx.x], si[threadIdx.x], sv[threadIdx.x + s], si[threadIdx.x + s]);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    part_val[row * blocks_per_row + blk] = sv[0];
    part_idx[row * blocks_per_row + blk] = si[0];
  }
}

__global__ void ArgmaxFinalKernel(int64_t* out, const float* part_val, const int64_t* part_idx,
                                  int blocks_per_row) {
  const int64_t row = blockIdx.x;
  const float* pv = part_val + row * blocks_per_row;
  const int64_t* pi = part_idx + row * blocks_per_row;
  __shared__ float sv[kBlock];
  __shared__ int64_t si[kBlock];

  float bv = kNegInf;
  int64_t bi = kArgSentinel;
  for (int j = threadIdx.x; j < blocks_per_row; j += blockDim.x) ArgReduce(bv, bi, pv[j], pi[j]);

  sv[threadIdx.x] = bv;
  si[threadIdx.x] = bi;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s)
      ArgReduce(sv[threadIdx.x], si[threadIdx.x], sv[threadIdx.x + s], si[threadIdx.x + s]);
    __syncthreads();
  }
  if (threadIdx.x == 0) out[row] = (si[0] == kArgSentinel) ? 0 : si[0];
}

// Persistent scratch for the reduction partials -- grown on demand and kept
// alive (a few KB), so the decode path never pays a cudaMalloc/cudaFree per
// token. `n * blocks_per_row` elements, and blocks_per_row is capped at kBlock,
// so at num_reqs 32 this is 8192 entries: 32 KiB + 64 KiB.
struct ArgScratch {
  float* val = nullptr;
  int64_t* idx = nullptr;
  size_t cap = 0;  // capacity in elements
};

void EnsureArgScratch(ArgScratch& s, size_t elems, const char* val_what, const char* idx_what) {
  if (elems <= s.cap) return;
  if (s.val) cudaFree(s.val);
  if (s.idx) cudaFree(s.idx);
  Check(cudaMalloc(&s.val, elems * sizeof(float)), val_what);
  Check(cudaMalloc(&s.idx, elems * sizeof(int64_t)), idx_what);
  s.cap = elems;
}

// greedy_argmax and random_sample get SEPARATE scratches. A mixed greedy/random
// batch runs both inside one Sampler::sample() call, and sharing one buffer
// would make correctness depend on stream ordering that nothing in the type
// system enforces. Two allocations of tens of KiB is not a reason to accept that.
ArgScratch g_argmax_scratch;
ArgScratch g_sample_scratch;

// Legacy single-block single-thread argmax (bit-exact reference). Retained behind
// VT_FAST_ARGMAX=0 for same-binary A/B against the two-pass kernel above.
__global__ void GreedyArgmaxKernelSlow(int64_t* out, const float* logits, int64_t v) {
  const int64_t row = blockIdx.x;
  if (threadIdx.x != 0) return;
  const float* r = logits + row * v;
  int64_t best = 0;
  float best_v = r[0];
  for (int64_t j = 1; j < v; ++j) {
    if (r[j] > best_v) {
      best_v = r[j];
      best = j;
    }
  }
  out[row] = best;
}

bool FastArgmaxEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FAST_ARGMAX");
    return e == nullptr || (e[0] != '0');
  }();
  return on;
}

void GreedyArgmaxCuda(Queue& q, Tensor& token_ids, const Tensor& logits) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  cudaStream_t s = AsStream(q);

  if (!FastArgmaxEnabled()) {
    GreedyArgmaxKernelSlow<<<static_cast<unsigned>(n), 1, 0, s>>>(token_ids.Ptr<int64_t>(),
                                                                 logits.Ptr<float>(), v);
    Check(cudaGetLastError(), "greedy_argmax launch (slow)");
    return;
  }

  // One block per kBlock vocab elements, capped so pass 2 fits a single block.
  const int bpr = vt::sample::ArgBlocksPerRow(v, kBlock);

  EnsureArgScratch(g_argmax_scratch, static_cast<size_t>(n) * bpr, "argmax scratch val",
                   "argmax scratch idx");
  dim3 grid1(static_cast<unsigned>(bpr), static_cast<unsigned>(n));
  ArgmaxPartialKernel<<<grid1, kBlock, 0, s>>>(g_argmax_scratch.val, g_argmax_scratch.idx,
                                               LogitScore{logits.Ptr<float>(), v}, v, bpr);
  ArgmaxFinalKernel<<<static_cast<unsigned>(n), kBlock, 0, s>>>(
      token_ids.Ptr<int64_t>(), g_argmax_scratch.val, g_argmax_scratch.idx, bpr);
  Check(cudaGetLastError(), "greedy_argmax launch");
}

// --- compute_probs / compute_logprobs (block-per-row softmax) ---------------
__global__ void SoftmaxKernel(float* out, const float* logits, int64_t v, bool log_softmax) {
  const int64_t row = blockIdx.x;
  const float* r = logits + row * v;
  float* o = out + row * v;
  __shared__ float red[kBlock];

  float m = kNegInf;
  for (int64_t j = threadIdx.x; j < v; j += blockDim.x) m = fmaxf(m, r[j]);
  red[threadIdx.x] = m;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  const float mx = red[0];
  __syncthreads();

  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < v; j += blockDim.x) acc += expf(r[j] - mx);
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  const float sum = red[0];
  const float lse = mx + logf(sum);
  __syncthreads();

  for (int64_t j = threadIdx.x; j < v; j += blockDim.x)
    o[j] = log_softmax ? (r[j] - lse) : expf(r[j] - mx) / sum;
}

void ComputeProbsCuda(Queue& q, Tensor& probs, const Tensor& logits) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  SoftmaxKernel<<<static_cast<unsigned>(n), kBlock, 0, AsStream(q)>>>(
      probs.Ptr<float>(), logits.Ptr<float>(), v, /*log_softmax=*/false);
  Check(cudaGetLastError(), "compute_probs launch");
}

void ComputeLogprobsCuda(Queue& q, Tensor& logprobs, const Tensor& logits) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  SoftmaxKernel<<<static_cast<unsigned>(n), kBlock, 0, AsStream(q)>>>(
      logprobs.Ptr<float>(), logits.Ptr<float>(), v, /*log_softmax=*/true);
  Check(cudaGetLastError(), "compute_logprobs launch");
}

// --- random_sample (two-pass reduction over probs/q; #1984) ------------------
// Upstream is `probs.div_(q).argmax(dim=-1)` over the whole tensor
// (topk_topp_sampler.py::sample_with_exponential_noise), i.e. fully parallel.
// Ours was `<<<n, 1>>>` with `if (threadIdx.x != 0) return;` and a serial walk
// of the vocabulary -- the SAME single-lane shape recorded twenty lines above at
// ~7.5 ms/token for a ~151k greedy scan, at 1.64x the vocab and with an f64
// `log` and two 64-bit mixes on top. It now reuses the greedy reduction with the
// Gumbel score substituted for the logit.
//
// The output is BIT-IDENTICAL, not merely equivalent, and the reason is worth
// stating because the gate asserts equality rather than agreement: every
// element's score is `GumbelScore(probs[row][j], seed, row, j)` on both paths,
// evaluated by the same device libm, so the reduction sees the same floats and
// differs only in the order it combines them -- and ArgReduce is
// order-independent (vt/sample_common.h).
//
// The serial kernel below is RETAINED, reachable as VT_FAST_RANDOM_SAMPLE=0,
// mirroring the VT_FAST_ARGMAX lever the greedy rewrite kept. It is what makes
// the equality gate a same-binary A/B, which AGENTS.md requires before a
// performance result is accepted -- and what #1929/#1975 cost when a sampling
// kernel landed on a per-kernel figure alone.
__global__ void RandomSampleKernelSlow(int64_t* out, const float* probs, const int64_t* seeds,
                                       int64_t v) {
  const int64_t row = blockIdx.x;
  if (threadIdx.x != 0) return;
  const float* r = probs + row * v;
  const uint64_t seed = static_cast<uint64_t>(seeds[row]);
  int64_t best = 0;
  float best_v = kNegInf;
  for (int64_t j = 0; j < v; ++j) {
    const float score = GumbelScore(r[j], seed, row, j);
    if (score > best_v) {
      best_v = score;
      best = j;
    }
  }
  out[row] = best;
}

bool FastRandomSampleEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FAST_RANDOM_SAMPLE");
    return e == nullptr || (e[0] != '0');
  }();
  return on;
}

void RandomSampleCuda(Queue& q, Tensor& token_ids, const Tensor& probs, const Tensor& seeds) {
  const int64_t n = probs.shape[0], v = probs.shape[1];
  if (n == 0 || v == 0) return;
  cudaStream_t s = AsStream(q);

  if (!FastRandomSampleEnabled()) {
    RandomSampleKernelSlow<<<static_cast<unsigned>(n), 1, 0, s>>>(
        token_ids.Ptr<int64_t>(), probs.Ptr<float>(), seeds.Ptr<int64_t>(), v);
    Check(cudaGetLastError(), "random_sample launch (slow)");
    return;
  }

  const int bpr = vt::sample::ArgBlocksPerRow(v, kBlock);
  EnsureArgScratch(g_sample_scratch, static_cast<size_t>(n) * bpr,
                   "random_sample scratch val", "random_sample scratch idx");
  dim3 grid1(static_cast<unsigned>(bpr), static_cast<unsigned>(n));
  ArgmaxPartialKernel<<<grid1, kBlock, 0, s>>>(
      g_sample_scratch.val, g_sample_scratch.idx,
      GumbelScore2D{probs.Ptr<float>(), seeds.Ptr<int64_t>(), v}, v, bpr);
  ArgmaxFinalKernel<<<static_cast<unsigned>(n), kBlock, 0, s>>>(
      token_ids.Ptr<int64_t>(), g_sample_scratch.val, g_sample_scratch.idx, bpr);
  Check(cudaGetLastError(), "random_sample launch");
}

// --- apply_top_k_top_p (SORT-FREE block-cooperative threshold search) --------
// Each row is masked by ONE block. The top-k value-threshold (k-th largest logit)
// and the top-p exp-threshold are found with the two-pivot bracket search that
// flashinfer's TopK/TopPRenormProb use (sampling.cuh): each iteration evaluates
// count(>pivot) / sum(>pivot) cooperatively and snaps the bracket to real array
// values via min_gt_low / max_le_high, so it converges to an EXACT array-value
// threshold (empirically <=17 iters over a 152k vocab). No sort, no serial scan,
// no host round-trip, and (unlike the old path) no blocking Synchronize — the op
// is fully async on the stream, ordered before the downstream softmax/sample.
//
// Kept set == apply_top_k_top_p_pytorch for DISTINCT logits (validated by the
// scalar mirror + cross-check in tests/vt/test_ops_sample.cpp). The only
// divergence is the measure-zero exact-tie-straddling-boundary case, where this
// path keeps/drops whole tie groups atomically (like flashinfer) rather than
// splitting them by stable index order; real f32 logits are effectively
// continuous, so this never fires.
constexpr int kThreshMaxIter = 64;  // >3x the observed worst case; converges by pinning

// Block reductions over a kBlock-thread block (kBlock is a power of two). Each
// returns the reduced value broadcast to all threads. `s` is a kBlock-wide shared
// scratch; callers must not hold a live value in it across the call.
__device__ inline float BlockRedMaxF(float v, float* s) {
  const int t = threadIdx.x;
  s[t] = v;
  __syncthreads();
  for (int o = kBlock / 2; o > 0; o >>= 1) {
    if (t < o) s[t] = fmaxf(s[t], s[t + o]);
    __syncthreads();
  }
  const float r = s[0];
  __syncthreads();
  return r;
}
__device__ inline float BlockRedMinF(float v, float* s) {
  const int t = threadIdx.x;
  s[t] = v;
  __syncthreads();
  for (int o = kBlock / 2; o > 0; o >>= 1) {
    if (t < o) s[t] = fminf(s[t], s[t + o]);
    __syncthreads();
  }
  const float r = s[0];
  __syncthreads();
  return r;
}
__device__ inline float BlockRedSumF(float v, float* s) {
  const int t = threadIdx.x;
  s[t] = v;
  __syncthreads();
  for (int o = kBlock / 2; o > 0; o >>= 1) {
    if (t < o) s[t] += s[t + o];
    __syncthreads();
  }
  const float r = s[0];
  __syncthreads();
  return r;
}
__device__ inline int BlockRedSumI(int v, int* s) {
  const int t = threadIdx.x;
  s[t] = v;
  __syncthreads();
  for (int o = kBlock / 2; o > 0; o >>= 1) {
    if (t < o) s[t] += s[t + o];
    __syncthreads();
  }
  const int r = s[0];
  __syncthreads();
  return r;
}

// The two thresholds a row is masked with -- keep r >= thr_k, then keep
// exp(r - mx) > low (low < 0 keeps every top-k survivor) -- and the row max. One
// block per row. ApplyTopKTopPRowKernel masks with them; the fused sampler below
// computes them for the rows its candidates cannot decide.
struct TopKTopPThresholds {
  float thr_k;
  float low;
  float mx;
};

__device__ TopKTopPThresholds RowTopKTopPThresholds(const float* r, int64_t v, bool has_k,
                                                    int32_t k, bool has_p, float p, float* red,
                                                    int* redi, float* sh_thr_k, float* sh_low) {
  const int t = threadIdx.x;

  // Row max (softmax reference) and min over FINITE logits (the top-k search
  // lower bracket must be finite — upstream masks (min_p / bad_words / allowed
  // ids) can leave -inf logits, and an -inf bracket would stall the pivot
  // arithmetic). mn == global min in the common (no pre-masked) case.
  float lmax = kNegInf, lmin = INFINITY;
  for (int64_t j = t; j < v; j += kBlock) {
    const float x = r[j];
    lmax = fmaxf(lmax, x);
    if (x != kNegInf) lmin = fminf(lmin, x);
  }
  const float mx = BlockRedMaxF(lmax, red);
  const float mn = BlockRedMinF(lmin, red);  // +inf if every logit is -inf

  // ---- top-k: thr_k = k-th largest logit; keep {r[j] >= thr_k} (ties kept) ----
  // Active only for 1 <= k < v (k>=v is a no-op; k<1 invalid, guarded). Mirrors
  // apply_top_k_only / the torch top_k_mask (logits_sort < sorted[v-k]).
  const bool topk_active = has_k && (k >= 1) && (static_cast<int64_t>(k) < v);
  float thr_k = kNegInf;  // -inf => every token survives
  if (topk_active) {
    int lc = 0;
    for (int64_t j = t; j < v; j += kBlock)
      if (r[j] > mn) ++lc;
    const int cnt_gt_min = BlockRedSumI(lc, redi);
    if (cnt_gt_min < k) {
      thr_k = mn;  // k-th largest == global min => masks nothing
    } else {
      // Find `low` = largest value with count(r>low) >= k; the k-th largest is the
      // smallest value strictly greater than `low` (min_gt_low at convergence).
      float low = mn, high = mx, min_gt_low = mx, max_le_high = mn;
      for (int iter = 0; iter < kThreshMaxIter; ++iter) {
        const float p0 = (2.0f * low + high) / 3.0f;
        const float p1 = (low + 2.0f * high) / 3.0f;
        int l0 = 0, l1 = 0;
        float lmglow = high, lmleh = low;
        for (int64_t j = t; j < v; j += kBlock) {
          const float x = r[j];
          if (x > p0) ++l0;
          if (x > p1) ++l1;
          if (x > low) lmglow = fminf(lmglow, x);
          if (x <= high) lmleh = fmaxf(lmleh, x);
        }
        const int c0 = BlockRedSumI(l0, redi);
        const int c1 = BlockRedSumI(l1, redi);
        min_gt_low = BlockRedMinF(lmglow, red);
        max_le_high = BlockRedMaxF(lmleh, red);
        if (c1 >= k) {
          low = p1;
        } else if (c0 >= k) {
          low = p0;
          high = fminf(p1, max_le_high);
        } else {
          high = fminf(p0, max_le_high);
        }
        if (min_gt_low == max_le_high) break;
      }
      thr_k = min_gt_low;
    }
  }
  if (t == 0) *sh_thr_k = thr_k;
  __syncthreads();
  thr_k = *sh_thr_k;
  // survivor(j) := r[j] >= thr_k

  // ---- top-p over survivors: keep {survivor && e_j > low}, e_j = exp(r-mx) ----
  // denom / target normalize over the TOP-K SURVIVORS (matching the torch path,
  // whose top-p softmax runs on the top-k-masked logits). p >= 1 is a no-op.
  float low = -1.0f;  // on e in (0,1]; -1 => keep every survivor
  const bool topp_active = has_p && (p < 1.0f);
  if (topp_active) {
    float lden = 0.0f;
    for (int64_t j = t; j < v; j += kBlock)
      if (r[j] >= thr_k) lden += expf(r[j] - mx);
    const float denom = BlockRedSumF(lden, red);
    const float target = p * denom;  // survivor mass to keep (>= p fraction)
    float lo = 0.0f, hi = 1.0f, min_gt_low = 1.0f, max_le_high = 0.0f;
    for (int iter = 0; denom > 0.0f && iter < kThreshMaxIter; ++iter) {
      const float p0 = (2.0f * lo + hi) / 3.0f;
      const float p1 = (lo + 2.0f * hi) / 3.0f;
      float la0 = 0.0f, la1 = 0.0f, lmglow = hi, lmleh = lo;
      for (int64_t j = t; j < v; j += kBlock) {
        if (r[j] < thr_k) continue;  // non-survivor: excluded from the top-p mass
        const float e = expf(r[j] - mx);
        if (e > p0) la0 += e;
        if (e > p1) la1 += e;
        if (e > lo) lmglow = fminf(lmglow, e);
        if (e <= hi) lmleh = fmaxf(lmleh, e);
      }
      const float a0 = BlockRedSumF(la0, red);
      const float a1 = BlockRedSumF(la1, red);
      min_gt_low = BlockRedMinF(lmglow, red);
      max_le_high = BlockRedMaxF(lmleh, red);
      if (a1 >= target) {
        lo = p1;
      } else if (a0 >= target) {
        lo = p0;
        hi = fminf(p1, max_le_high);
      } else {
        hi = fminf(p0, max_le_high);
      }
      if (min_gt_low == max_le_high) break;
    }
    low = lo;
  }
  if (t == 0) *sh_low = low;
  __syncthreads();
  low = *sh_low;
  return {thr_k, low, mx};
}

__global__ void ApplyTopKTopPRowKernel(float* logits, const int32_t* k_arr, const float* p_arr,
                                       int64_t v) {
  const int64_t row = blockIdx.x;
  float* r = logits + row * v;
  const int t = threadIdx.x;

  __shared__ float red[kBlock];
  __shared__ int redi[kBlock];
  __shared__ float sh_thr_k;
  __shared__ float sh_low;

  const bool has_k = (k_arr != nullptr);
  const bool has_p = (p_arr != nullptr);
  const int32_t k = has_k ? k_arr[row] : 0;
  const float p = has_p ? p_arr[row] : 1.0f;
  const TopKTopPThresholds th =
      RowTopKTopPThresholds(r, v, has_k, k, has_p, p, red, redi, &sh_thr_k, &sh_low);

  // ---- final in-place mask ----
  for (int64_t j = t; j < v; j += kBlock) {
    const float x = r[j];
    const bool keep = (x >= th.thr_k) && (th.low < 0.0f || expf(x - th.mx) > th.low);
    if (!keep) r[j] = kNegInf;
  }
}

void ApplyTopKTopPCuda(Queue& q, Tensor& logits, const Tensor* k, const Tensor* p) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  // Both k-only and k+p go through the same threshold kernel; k-only is the torch
  // apply_top_k_only set (k-th largest, ties kept). Fully async — no Synchronize.
  ApplyTopKTopPRowKernel<<<static_cast<unsigned>(n), kBlock, 0, AsStream(q)>>>(
      logits.Ptr<float>(), k != nullptr ? k->Ptr<int32_t>() : nullptr,
      p != nullptr ? p->Ptr<float>() : nullptr, v);
  Check(cudaGetLastError(), "top_k_top_p launch");
}

// --- top_k_top_p_sample: ApplyTopKTopP + ComputeProbs + RandomSample in one ---
// Those three ops each walk the whole vocabulary with one block per row, so a
// decode step of n rows runs on n SMs. On Qwen3.6-35B-A3B (248,320 logits) that
// was 1.2 ms per step, against 0.15 ms for the FlashInfer kernels vLLM 0.28.0
// samples with. With top-k on every row (1 <= k <= kFusedMaxK) the kept set is
// the k-th largest logit plus ties, so:
//   1. FusedTopKCandidatesKernel, grid (chunks, rows): each block radix-selects
//      its chunk's k-th largest logit and writes every logit above it, then the
//      ties at it while kFusedCap lasts. Every row survivor is among them.
//   2. FusedTopKTopPSampleKernel, one block per row: the exact k-th largest of
//      the candidates, then top-p, the softmax and the Gumbel draw over the
//      survivors only.
// Step 2 computes the same floats as the three ops: a survivor j is summed by
// thread j % kBlock in increasing j, as ApplyTopKTopPRowKernel and SoftmaxKernel
// sum it, and the per-thread sums go through the same tree, so the drawn token
// is the token they draw. A masked token's probability is 0, so its Gumbel score
// is 0 and the draw is decided by the survivors. The k-th largest is exact where
// ApplyTopKTopPRowKernel's bracket search is exact (it converges, <= 17
// iterations observed against a cap of 64). A row the candidates cannot decide
// (a chunk dropped ties at or above the threshold, fewer than k finite logits,
// a NaN, too many survivors) runs the three ops' arithmetic on its full row in
// the same block. The logits are read, not masked.
constexpr int kFusedChunk = 4096;
constexpr int kFusedPerThread = kFusedChunk / kBlock;
constexpr int kFusedMaxK = kTopKTopPSampleMaxK;
constexpr int kFusedCap = 128;  // > kFusedMaxK: the logits above the threshold always fit
constexpr int kFusedMaxSurvivors = 512;
constexpr int kFusedMaxPerThread = 8;
constexpr int kFusedMaxChunks = 256;
static_assert(kFusedCap > kFusedMaxK, "a chunk must fit its k - 1 logits above the threshold");
static_assert(static_cast<int64_t>(kFusedMaxChunks) * kFusedChunk >= kTopKTopPSampleMaxVocab,
              "the finish kernel keeps one written count per chunk in shared memory");

// Flags of a chunk's candidate list.
constexpr int kFusedTiesDropped = 1;
constexpr int kFusedNaN = 2;
constexpr int kFusedBadK = 4;

struct FusedChunkMeta {
  int written;  // candidates written, <= kFusedCap
  int finite;   // logits other than -inf (NaN counts, and also sets kFusedNaN)
  float thr;    // the chunk's k-th largest logit
  int flags;
};

// Order-preserving map of a float to an unsigned key. -0 maps to +0's key, so a
// tie the way `>=` sees it is a tie in the key order too.
__device__ inline uint32_t FusedKey(float x) {
  const uint32_t b = __float_as_uint(x == 0.0f ? 0.0f : x);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ inline float FusedKeyValue(uint32_t key) {
  return __uint_as_float((key & 0x80000000u) ? (key & 0x7fffffffu) : ~key);
}

__device__ inline int WarpSumI(int v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// One radix-select step, run by warp 0 after `hist` holds the 8-bit digit at
// `shift` of every key that matches the prefix fixed so far: find the digit
// holding the *sh_krem-th largest key, fix it into *sh_prefix and leave in
// *sh_krem the rank still wanted inside that digit. Lane l owns digits
// 255 - 8l down to 248 - 8l.
__device__ void RadixPickDigit(const int* hist, int shift, uint32_t* sh_prefix, int* sh_krem) {
  const int lane = threadIdx.x;
  int bins[8];
  int sum = 0;
  for (int i = 0; i < 8; ++i) {
    bins[i] = hist[255 - lane * 8 - i];
    sum += bins[i];
  }
  int incl = sum;
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += y;
  }
  const int krem = *sh_krem;
  __syncwarp();
  const int excl = incl - sum;
  if (excl < krem && krem <= incl) {
    int before = excl;
    for (int i = 0; i < 8; ++i) {
      if (krem <= before + bins[i]) {
        *sh_prefix |= static_cast<uint32_t>(255 - lane * 8 - i) << shift;
        *sh_krem = krem - before;
        break;
      }
      before += bins[i];
    }
  }
}

// Exclusive block scan of one int per thread; *total gets the block sum.
__device__ int BlockExclusiveScanI(int v, int* warp_tot, int* total) {
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  int incl = v;
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += y;
  }
  if (lane == 31) warp_tot[warp] = incl;
  __syncthreads();
  int before = 0, all = 0;
  for (int w = 0; w < kBlock / 32; ++w) {
    if (w < warp) before += warp_tot[w];
    all += warp_tot[w];
  }
  __syncthreads();
  *total = all;
  return before + incl - v;
}

__global__ void FusedTopKCandidatesKernel(const float* logits, const int32_t* k_arr,
                                          float* cand_val, int32_t* cand_idx,
                                          FusedChunkMeta* meta, int chunks, int64_t v) {
  const int c = blockIdx.x;
  const int64_t row = blockIdx.y;
  const int t = threadIdx.x;
  const int32_t k = k_arr[row];
  const float* r = logits + row * v;
  const int64_t base = static_cast<int64_t>(c) * kFusedChunk;
  const int64_t slot = row * chunks + c;

  __shared__ int hist[256];
  __shared__ uint32_t sh_prefix;
  __shared__ int sh_krem;
  __shared__ int sh_count[3];  // valid, finite, NaN
  __shared__ int warp_tot[kBlock / 32];

  if (k < 1 || k > kFusedMaxK) {
    if (t == 0) meta[slot] = FusedChunkMeta{0, 0, kNegInf, kFusedBadK};
    return;
  }

  // Element e of this thread is j = base + e * kBlock + t: coalesced loads, and
  // the valid ones are e < n_valid.
  float vals[kFusedPerThread];
  uint32_t keys[kFusedPerThread];
  int n_valid = 0, n_finite = 0, n_nan = 0;
  for (int e = 0; e < kFusedPerThread; ++e) {
    const int64_t j = base + static_cast<int64_t>(e) * kBlock + t;
    if (j < v) {
      const float x = r[j];
      vals[e] = x;
      keys[e] = FusedKey(x);
      ++n_valid;
      if (x != kNegInf) ++n_finite;
      if (x != x) ++n_nan;
    }
  }
  if (t < 3) sh_count[t] = 0;
  __syncthreads();
  const int wv = WarpSumI(n_valid), wf = WarpSumI(n_finite), wn = WarpSumI(n_nan);
  if ((t & 31) == 0) {
    atomicAdd(&sh_count[0], wv);
    atomicAdd(&sh_count[1], wf);
    atomicAdd(&sh_count[2], wn);
  }
  __syncthreads();
  const int total = sh_count[0];

  // The chunk's k-th largest key; with fewer than k logits every logit is a
  // candidate (key 0 is below every float's key).
  uint32_t thr = 0u;
  if (total >= k) {
    if (t == 0) {
      sh_prefix = 0u;
      sh_krem = k;
    }
    uint32_t mask = 0u;
    for (int shift = 24; shift >= 0; shift -= 8) {
      for (int i = t; i < 256; i += kBlock) hist[i] = 0;
      __syncthreads();
      const uint32_t prefix = sh_prefix;
      for (int e = 0; e < n_valid; ++e)
        if ((keys[e] & mask) == prefix) atomicAdd(&hist[(keys[e] >> shift) & 255u], 1);
      __syncthreads();
      if (t < 32) RadixPickDigit(hist, shift, &sh_prefix, &sh_krem);
      __syncthreads();
      mask |= 0xffu << shift;
    }
    thr = sh_prefix;
  }

  // Fewer than k keys lie above the threshold, so they always fit; the ties at
  // it are written while the list has room.
  int n_gt = 0, n_eq = 0;
  for (int e = 0; e < n_valid; ++e) {
    if (keys[e] > thr) {
      ++n_gt;
    } else if (keys[e] == thr) {
      ++n_eq;
    }
  }
  int tot_gt = 0, tot_eq = 0;
  int pg = BlockExclusiveScanI(n_gt, warp_tot, &tot_gt);
  int pe = tot_gt + BlockExclusiveScanI(n_eq, warp_tot, &tot_eq);
  float* cv = cand_val + slot * kFusedCap;
  int32_t* ci = cand_idx + slot * kFusedCap;
  for (int e = 0; e < n_valid; ++e) {
    const int32_t j = static_cast<int32_t>(base + static_cast<int64_t>(e) * kBlock + t);
    if (keys[e] > thr) {
      cv[pg] = vals[e];
      ci[pg] = j;
      ++pg;
    } else if (keys[e] == thr) {
      if (pe < kFusedCap) {
        cv[pe] = vals[e];
        ci[pe] = j;
      }
      ++pe;
    }
  }
  if (t == 0) {
    const int all = tot_gt + tot_eq;
    meta[slot] = FusedChunkMeta{all < kFusedCap ? all : kFusedCap, sh_count[1],
                                total >= k ? FusedKeyValue(thr) : kNegInf,
                                (all > kFusedCap ? kFusedTiesDropped : 0) |
                                    (sh_count[2] > 0 ? kFusedNaN : 0)};
  }
}

// The three ops' arithmetic on one full row: ApplyTopKTopPRowKernel's
// thresholds, SoftmaxKernel's max and sum over the masked row, and
// RandomSample's argmax of the Gumbel scores. A row with finite logits keeps its
// max, so every masked token's score is 0 and only the kept tokens are scored;
// any other row scores every token, as RandomSample does.
__device__ void FusedFullRowSample(int64_t* out, const float* r, int64_t v, int64_t row,
                                   int32_t k, bool has_p, float p, uint64_t seed, float* red,
                                   int* redi, int64_t* si, float* sh_thr_k, float* sh_low,
                                   bool row_has_nan) {
  const int t = threadIdx.x;
  const TopKTopPThresholds th =
      RowTopKTopPThresholds(r, v, /*has_k=*/true, k, has_p, p, red, redi, sh_thr_k, sh_low);
  float m = kNegInf;
  for (int64_t j = t; j < v; j += kBlock) {
    const float x = r[j];
    const bool keep = (x >= th.thr_k) && (th.low < 0.0f || expf(x - th.mx) > th.low);
    m = fmaxf(m, keep ? x : kNegInf);
  }
  const float mm = BlockRedMaxF(m, red);
  float acc = 0.0f;
  for (int64_t j = t; j < v; j += kBlock) {
    const float x = r[j];
    const bool keep = (x >= th.thr_k) && (th.low < 0.0f || expf(x - th.mx) > th.low);
    acc += expf((keep ? x : kNegInf) - mm);
  }
  const float sum = BlockRedSumF(acc, red);
  const bool kept_only = !row_has_nan && isfinite(mm) && isfinite(sum) && sum > 0.0f;
  float bv = kNegInf;
  int64_t bi = kArgSentinel;
  for (int64_t j = t; j < v; j += kBlock) {
    const float x = r[j];
    const bool keep = (x >= th.thr_k) && (th.low < 0.0f || expf(x - th.mx) > th.low);
    if (kept_only && !keep) continue;
    ArgReduce(bv, bi, GumbelScore(expf((keep ? x : kNegInf) - mm) / sum, seed, row, j), j);
  }
  red[t] = bv;
  si[t] = bi;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s >>= 1) {
    if (t < s) ArgReduce(red[t], si[t], red[t + s], si[t + s]);
    __syncthreads();
  }
  if (t == 0) out[row] = (si[0] == kArgSentinel) ? 0 : si[0];
}

__global__ void FusedTopKTopPSampleKernel(int64_t* out, const float* logits, const int32_t* k_arr,
                                          const float* p_arr, const int64_t* seeds,
                                          const float* cand_val, const int32_t* cand_idx,
                                          const FusedChunkMeta* meta, int chunks, int64_t v) {
  const int64_t row = blockIdx.x;
  const int t = threadIdx.x;
  const int32_t k = k_arr[row];
  const bool has_p = (p_arr != nullptr);
  const float p = has_p ? p_arr[row] : 1.0f;
  const uint64_t seed = static_cast<uint64_t>(seeds[row]);
  const float* r = logits + row * v;
  const FusedChunkMeta* m = meta + row * chunks;
  const float* cv = cand_val + row * chunks * kFusedCap;
  const int32_t* ci = cand_idx + row * chunks * kFusedCap;

  __shared__ float red[kBlock];
  __shared__ int redi[kBlock];
  __shared__ int64_t si[kBlock];
  __shared__ int hist[256];
  __shared__ uint32_t sh_prefix;
  __shared__ int sh_krem;
  __shared__ int sh_nsurv;
  __shared__ float sh_thr_k, sh_low;
  __shared__ float surv_val[kFusedMaxSurvivors];
  __shared__ int32_t surv_idx[kFusedMaxSurvivors];
  __shared__ float sorted_val[kFusedMaxSurvivors];
  __shared__ int32_t sorted_idx[kFusedMaxSurvivors];
  __shared__ int written[kFusedMaxChunks];

  int fin = 0, flags = 0;
  for (int c = t; c < chunks; c += kBlock) {
    fin += m[c].finite;
    flags |= m[c].flags;
    written[c] = m[c].written;
  }
  const int total_finite = BlockRedSumI(fin, redi);
  const bool row_has_nan = BlockRedSumI(flags & kFusedNaN, redi) != 0;
  const bool bad_k = BlockRedSumI(flags & kFusedBadK, redi) != 0;
  if (bad_k || row_has_nan || static_cast<int64_t>(k) >= v || total_finite < k) {
    FusedFullRowSample(out, r, v, row, k, has_p, p, seed, red, redi, si, &sh_thr_k, &sh_low,
                       row_has_nan);
    return;
  }

  // The exact k-th largest logit of the row: the candidates hold every chunk's
  // top k, ties at a chunk's threshold aside.
  if (t == 0) {
    sh_prefix = 0u;
    sh_krem = k;
  }
  // Every block thread walks the flat (chunk, slot) space; slot i of chunk c
  // holds a candidate when i < written[c].
  const int span = chunks * kFusedCap;
  uint32_t mask = 0u;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int i = t; i < 256; i += kBlock) hist[i] = 0;
    __syncthreads();
    const uint32_t prefix = sh_prefix;
    for (int f = t; f < span; f += kBlock) {
      if (f % kFusedCap >= written[f / kFusedCap]) continue;
      const uint32_t key = FusedKey(cv[f]);
      if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1);
    }
    __syncthreads();
    if (t < 32) RadixPickDigit(hist, shift, &sh_prefix, &sh_krem);
    __syncthreads();
    mask |= 0xffu << shift;
  }
  const float thr_k = FusedKeyValue(sh_prefix);

  // A chunk that dropped ties at or above thr_k may hide survivors.
  int dropped = 0;
  for (int c = t; c < chunks; c += kBlock)
    if ((m[c].flags & kFusedTiesDropped) && m[c].thr >= thr_k) dropped = 1;
  if (t == 0) sh_nsurv = 0;
  __syncthreads();
  for (int f = t; f < span; f += kBlock) {
    if (f % kFusedCap >= written[f / kFusedCap]) continue;
    const float x = cv[f];
    if (x >= thr_k) {
      const int pos = atomicAdd(&sh_nsurv, 1);
      if (pos < kFusedMaxSurvivors) {
        surv_val[pos] = x;
        surv_idx[pos] = ci[f];
      }
    }
  }
  const bool undecided = BlockRedSumI(dropped, redi) != 0;
  const int ns = sh_nsurv;
  if (undecided || ns > kFusedMaxSurvivors) {
    FusedFullRowSample(out, r, v, row, k, has_p, p, seed, red, redi, si, &sh_thr_k, &sh_low,
                       false);
    return;
  }

  // Survivors in increasing j, then the ones the unfused kernels give this
  // thread (j % kBlock), still in increasing j.
  for (int s = t; s < ns; s += kBlock) {
    const int32_t j = surv_idx[s];
    int rank = 0;
    for (int u = 0; u < ns; ++u) rank += (surv_idx[u] < j) ? 1 : 0;
    sorted_val[rank] = surv_val[s];
    sorted_idx[rank] = j;
  }
  __syncthreads();
  float own_x[kFusedMaxPerThread];
  int32_t own_j[kFusedMaxPerThread];
  int own = 0;
  for (int s = 0; s < ns; ++s) {
    if ((sorted_idx[s] & (kBlock - 1)) == t) {
      if (own < kFusedMaxPerThread) {
        own_x[own] = sorted_val[s];
        own_j[own] = sorted_idx[s];
      }
      ++own;
    }
  }
  if (BlockRedSumI(own > kFusedMaxPerThread ? 1 : 0, redi) != 0) {
    FusedFullRowSample(out, r, v, row, k, has_p, p, seed, red, redi, si, &sh_thr_k, &sh_low,
                       false);
    return;
  }

  // The row max is the largest survivor.
  float lmx = kNegInf;
  for (int e = 0; e < own; ++e) lmx = fmaxf(lmx, own_x[e]);
  const float mx = BlockRedMaxF(lmx, red);
  if (!isfinite(mx)) {
    FusedFullRowSample(out, r, v, row, k, has_p, p, seed, red, redi, si, &sh_thr_k, &sh_low,
                       false);
    return;
  }

  // Top-p over the survivors: RowTopKTopPThresholds' loop, where each thread's
  // sums run over its survivors only (it skips the other tokens).
  float low = -1.0f;
  if (has_p && p < 1.0f) {
    float lden = 0.0f;
    for (int e = 0; e < own; ++e) lden += expf(own_x[e] - mx);
    const float denom = BlockRedSumF(lden, red);
    const float target = p * denom;
    float lo = 0.0f, hi = 1.0f, min_gt_low = 1.0f, max_le_high = 0.0f;
    for (int iter = 0; denom > 0.0f && iter < kThreshMaxIter; ++iter) {
      const float p0 = (2.0f * lo + hi) / 3.0f;
      const float p1 = (lo + 2.0f * hi) / 3.0f;
      float la0 = 0.0f, la1 = 0.0f, lmglow = hi, lmleh = lo;
      for (int e = 0; e < own; ++e) {
        const float ev = expf(own_x[e] - mx);
        if (ev > p0) la0 += ev;
        if (ev > p1) la1 += ev;
        if (ev > lo) lmglow = fminf(lmglow, ev);
        if (ev <= hi) lmleh = fmaxf(lmleh, ev);
      }
      const float a0 = BlockRedSumF(la0, red);
      const float a1 = BlockRedSumF(la1, red);
      min_gt_low = BlockRedMinF(lmglow, red);
      max_le_high = BlockRedMaxF(lmleh, red);
      if (a1 >= target) {
        lo = p1;
      } else if (a0 >= target) {
        lo = p0;
        hi = fminf(p1, max_le_high);
      } else {
        hi = fminf(p0, max_le_high);
      }
      if (min_gt_low == max_le_high) break;
    }
    low = lo;
  }

  // SoftmaxKernel's sum over the masked row (a masked token adds exp(-inf) = 0),
  // then RandomSample's argmax of probs / Exp(1) over the kept tokens.
  bool keep[kFusedMaxPerThread];
  float acc = 0.0f;
  for (int e = 0; e < own; ++e) {
    keep[e] = (low < 0.0f || expf(own_x[e] - mx) > low);
    if (keep[e]) acc += expf(own_x[e] - mx);
  }
  const float sum = BlockRedSumF(acc, red);
  float bv = kNegInf;
  int64_t bi = kArgSentinel;
  for (int e = 0; e < own; ++e)
    if (keep[e]) ArgReduce(bv, bi, GumbelScore(expf(own_x[e] - mx) / sum, seed, row, own_j[e]),
                           own_j[e]);
  red[t] = bv;
  si[t] = bi;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s >>= 1) {
    if (t < s) ArgReduce(red[t], si[t], red[t + s], si[t + s]);
    __syncthreads();
  }
  if (t == 0) out[row] = (si[0] == kArgSentinel) ? 0 : si[0];
}

// Candidate lists, grown on demand and kept (n * chunks * kFusedCap entries:
// about 3.9 MiB at 8 rows of a 248,320 vocab).
struct FusedSampleScratch {
  float* val = nullptr;
  int32_t* idx = nullptr;
  FusedChunkMeta* meta = nullptr;
  size_t slots = 0;  // (row, chunk) pairs
};
FusedSampleScratch g_fused_scratch;

void TopKTopPSampleCuda(Queue& q, Tensor& token_ids, const Tensor& logits, const Tensor& k,
                        const Tensor* p, const Tensor& seeds) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  const int chunks = static_cast<int>((v + kFusedChunk - 1) / kFusedChunk);
  const size_t slots = static_cast<size_t>(n) * static_cast<size_t>(chunks);
  FusedSampleScratch& s = g_fused_scratch;
  if (slots > s.slots) {
    if (s.val) cudaFree(s.val);
    if (s.idx) cudaFree(s.idx);
    if (s.meta) cudaFree(s.meta);
    Check(cudaMalloc(&s.val, slots * kFusedCap * sizeof(float)), "top_k_top_p_sample scratch val");
    Check(cudaMalloc(&s.idx, slots * kFusedCap * sizeof(int32_t)),
          "top_k_top_p_sample scratch idx");
    Check(cudaMalloc(&s.meta, slots * sizeof(FusedChunkMeta)), "top_k_top_p_sample scratch meta");
    s.slots = slots;
  }
  cudaStream_t st = AsStream(q);
  FusedTopKCandidatesKernel<<<dim3(static_cast<unsigned>(chunks), static_cast<unsigned>(n)), kBlock,
                              0, st>>>(logits.Ptr<float>(), k.Ptr<int32_t>(), s.val, s.idx,
                                       s.meta, chunks, v);
  FusedTopKTopPSampleKernel<<<static_cast<unsigned>(n), kBlock, 0, st>>>(
      token_ids.Ptr<int64_t>(), logits.Ptr<float>(), k.Ptr<int32_t>(),
      p != nullptr ? p->Ptr<float>() : nullptr, seeds.Ptr<int64_t>(), s.val, s.idx, s.meta,
      chunks, v);
  Check(cudaGetLastError(), "top_k_top_p_sample launch");
}

// --- top-k that EMITS the surviving (id, value) pairs -----------------------
// SPEC-DFLASH2 W3 / D2 (#1314). The CUDA arm of vt::TopKValuesIndices, whose
// contract, tie-break and upstream anchor live on `TopKValuesIndicesArgs`
// (include/vt/ops.h). `src/vt/cpu/cpu_ops.cpp::TopKValuesIndicesKernel` is the
// authoritative reference.
//
// This is the SAME pivot-bracket threshold search ApplyTopKTopPRowKernel above
// runs -- deliberately, because D2 rejects porting FlashInfer's 3380-line radix
// kernel for a shape that is K=16 over 248320 for about 224 rows. The difference
// is what happens after `thr_k`: that kernel masks everything below it in place
// and returns no indices, and this one COMPACTS the survivors and orders exactly
// k of them.
//
// THE TIE IS THE WHOLE DIFFICULTY. The bracket search converges to an exact
// array VALUE, so `{x >= thr_k}` keeps whole tie groups and can hold MORE than k
// elements. Everything strictly greater than `thr_k` survives unconditionally
// (there are at most k-1 such elements, by the definition of the k-th largest),
// and the remaining `k - cnt_gt` slots go to the LOWEST-INDEXED elements equal
// to `thr_k` -- which is the (value DESC, index ASC) order the CPU reference
// pins. The equal-valued slots are filled by repeated block-wide min-index
// passes rather than by a scan, because a single-threaded scan of a 248320-wide
// row would cost more than the search that produced the threshold.
//
// NOT gated on this host: there is no `nvcc` here, so this kernel has never
// compiled. See `## Owed` of .agents/specs/dflash2-spec-decode.md.
__device__ inline int BlockRedMinI(int v, int* s) {
  const int t = threadIdx.x;
  s[t] = v;
  __syncthreads();
  for (int o = kBlock / 2; o > 0; o >>= 1) {
    if (t < o) s[t] = min(s[t], s[t + o]);
    __syncthreads();
  }
  const int r = s[0];
  __syncthreads();
  return r;
}

__global__ void TopKValuesIndicesRowKernel(float* out_values, int64_t* out_indices,
                                           const float* logits, int64_t v, int64_t usable,
                                           int k) {
  const int64_t row = blockIdx.x;
  const float* r = logits + row * v;
  const int t = threadIdx.x;

  __shared__ float red[kBlock];
  __shared__ int redi[kBlock];
  __shared__ float sh_thr;
  __shared__ int sh_count;
  extern __shared__ float pairs[];  // k floats of value, then k ints of index
  int* pair_idx = reinterpret_cast<int*>(pairs + k);

  // Bracket over the USABLE columns only: `usable = V - num_org_vocab_padding`,
  // so an org-vocab padding tail can never contribute a candidate. This is
  // upstream's `logits[..., -num_pad:] = -inf`, done by restricting the search
  // rather than by writing to a read-only input.
  float lmax = kNegInf, lmin = INFINITY;
  for (int64_t j = t; j < usable; j += kBlock) {
    const float x = r[j];
    lmax = fmaxf(lmax, x);
    if (x != kNegInf) lmin = fminf(lmin, x);
  }
  const float mx = BlockRedMaxF(lmax, red);
  const float mn = BlockRedMinF(lmin, red);

  float thr = mn;  // k-th largest; every usable element is >= mn by construction
  {
    int lc = 0;
    for (int64_t j = t; j < usable; j += kBlock)
      if (r[j] > mn) ++lc;
    const int cnt_gt_min = BlockRedSumI(lc, redi);
    if (cnt_gt_min >= k) {
      float low = mn, high = mx, min_gt_low = mx, max_le_high = mn;
      for (int iter = 0; iter < kThreshMaxIter; ++iter) {
        const float p0 = (2.0f * low + high) / 3.0f;
        const float p1 = (low + 2.0f * high) / 3.0f;
        int l0 = 0, l1 = 0;
        float lmglow = high, lmleh = low;
        for (int64_t j = t; j < usable; j += kBlock) {
          const float x = r[j];
          if (x > p0) ++l0;
          if (x > p1) ++l1;
          if (x > low) lmglow = fminf(lmglow, x);
          if (x <= high) lmleh = fmaxf(lmleh, x);
        }
        const int c0 = BlockRedSumI(l0, redi);
        const int c1 = BlockRedSumI(l1, redi);
        min_gt_low = BlockRedMinF(lmglow, red);
        max_le_high = BlockRedMaxF(lmleh, red);
        if (c1 >= k) {
          low = p1;
        } else if (c0 >= k) {
          low = p0;
          high = fminf(p1, max_le_high);
        } else {
          high = fminf(p0, max_le_high);
        }
        if (min_gt_low == max_le_high) break;
      }
      thr = min_gt_low;
    }
  }
  if (t == 0) sh_thr = thr;
  __syncthreads();
  thr = sh_thr;

  // Everything STRICTLY greater than the threshold survives: at most k-1 of
  // them, so the shared pair buffer of k entries cannot overflow.
  if (t == 0) sh_count = 0;
  __syncthreads();
  for (int64_t j = t; j < usable; j += kBlock) {
    if (r[j] > thr) {
      const int slot = atomicAdd(&sh_count, 1);
      if (slot < k) {
        pairs[slot] = r[j];
        pair_idx[slot] = static_cast<int>(j);
      }
    }
  }
  __syncthreads();
  int filled = sh_count < k ? sh_count : k;

  // The remaining slots go to the LOWEST-INDEXED elements equal to `cur`, one
  // block-wide min-index pass each. `k - filled` is 1 in the ordinary case
  // (exactly one element attains the k-th largest value), so this loop is one
  // pass and not k of them.
  //
  // `cur` starts at the threshold and DROPS when that value's tie group is
  // exhausted. It has to: a row whose usable columns hold fewer than k values
  // at or above `thr` -- a heavily -inf-masked row is the real case -- would
  // otherwise return fewer than k pairs, while the CPU reference always returns
  // exactly k. Dropping `cur` to the largest value strictly below it and
  // continuing keeps the two arms answering the same thing on such a row instead
  // of only on the ordinary one.
  float cur = thr;
  int taken = -1;
  while (filled < k) {
    int local = INT_MAX;
    for (int64_t j = t; j < usable; j += kBlock)
      if (r[j] == cur && static_cast<int>(j) > taken) {
        local = min(local, static_cast<int>(j));
        break;  // ascending j: the first match in this thread's stride is its min
      }
    const int pick = BlockRedMinI(local, redi);
    if (pick != INT_MAX) {
      if (t == 0) {
        pairs[filled] = cur;
        pair_idx[filled] = pick;
      }
      taken = pick;
      ++filled;
      __syncthreads();
      continue;
    }
    // This value is exhausted: drop to the largest one strictly below it.
    float below = kNegInf;
    for (int64_t j = t; j < usable; j += kBlock)
      if (r[j] < cur) below = fmaxf(below, r[j]);
    const float next = BlockRedMaxF(below, red);
    if (next == kNegInf && cur == kNegInf) break;  // nothing left at all
    cur = next;
    taken = -1;
  }

  // Order the (at most k) pairs: value DESCENDING, ties by index ASCENDING. k is
  // 16 on both published checkpoints, so an O(k^2) selection sort in one thread
  // is cheaper than any cooperative alternative.
  if (t == 0) {
    for (int a = 0; a < filled; ++a) {
      int best = a;
      for (int b = a + 1; b < filled; ++b) {
        const bool better = (pairs[b] > pairs[best]) ||
                            (pairs[b] == pairs[best] && pair_idx[b] < pair_idx[best]);
        if (better) best = b;
      }
      const float fv = pairs[a];
      const int fi = pair_idx[a];
      pairs[a] = pairs[best];
      pair_idx[a] = pair_idx[best];
      pairs[best] = fv;
      pair_idx[best] = fi;
    }
  }
  __syncthreads();
  for (int j = t; j < k; j += kBlock) {
    out_values[row * k + j] = j < filled ? pairs[j] : kNegInf;
    out_indices[row * k + j] = j < filled ? static_cast<int64_t>(pair_idx[j]) : 0;
  }
}

void TopKValuesIndicesCuda(Queue& q, Tensor& values, Tensor& indices, const Tensor& logits,
                           const TopKValuesIndicesArgs& args) {
  const int64_t rows = logits.shape[0], v = logits.shape[1];
  if (rows == 0) return;
  const int k = static_cast<int>(args.k);
  const size_t shared = static_cast<size_t>(k) * (sizeof(float) + sizeof(int));
  TopKValuesIndicesRowKernel<<<static_cast<unsigned>(rows), kBlock, shared, AsStream(q)>>>(
      values.Ptr<float>(), indices.Ptr<int64_t>(), logits.Ptr<float>(), v,
      v - args.num_org_vocab_padding, k);
  Check(cudaGetLastError(), "topk_values_indices launch");
}

// --- apply_penalties (fused repetition + frequency + presence) --------------
__global__ void ApplyPenaltiesKernel(float* logits, const int8_t* prompt_mask,
                                     const int32_t* output_bin_counts, const int8_t* output_mask,
                                     const float* freq, const float* pres, const float* rep,
                                     int64_t n, int64_t v) {
  const int64_t total = n * v;
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += step) {
    const int64_t row = idx / v;
    float x = logits[idx];
    if (prompt_mask[idx] || output_mask[idx]) {
      const float r = rep[row];
      x = x > 0.0f ? x / r : x * r;
    }
    x -= freq[row] * static_cast<float>(output_bin_counts[idx]);
    x -= pres[row] * static_cast<float>(output_mask[idx]);
    logits[idx] = x;
  }
}

void ApplyPenaltiesCuda(Queue& q, Tensor& logits, const Tensor& prompt_mask,
                        const Tensor& output_bin_counts, const Tensor& output_mask,
                        const Tensor& frequency_penalties, const Tensor& presence_penalties,
                        const Tensor& repetition_penalties) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  ApplyPenaltiesKernel<<<GridFor(n * v), kBlock, 0, AsStream(q)>>>(
      logits.Ptr<float>(), prompt_mask.Ptr<int8_t>(), output_bin_counts.Ptr<int32_t>(),
      output_mask.Ptr<int8_t>(), frequency_penalties.Ptr<float>(), presence_penalties.Ptr<float>(),
      repetition_penalties.Ptr<float>(), n, v);
  Check(cudaGetLastError(), "apply_penalties launch");
}

// --- apply_min_p (block-per-row softmax, thread 0 threshold + mask) ----------
__global__ void ApplyMinPKernel(float* logits, const float* min_p, int64_t v) {
  const int64_t row = blockIdx.x;
  const float m = min_p[row];
  if (m <= 0.0f) return;
  float* r = logits + row * v;
  __shared__ float red[kBlock];

  float mx = kNegInf;
  for (int64_t j = threadIdx.x; j < v; j += blockDim.x) mx = fmaxf(mx, r[j]);
  red[threadIdx.x] = mx;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  const float rowmax = red[0];
  __syncthreads();

  float acc = 0.0f;
  for (int64_t j = threadIdx.x; j < v; j += blockDim.x) acc += expf(r[j] - rowmax);
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s /= 2) {
    if (static_cast<int>(threadIdx.x) < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  const float sum = red[0];
  __syncthreads();

  // max prob == exp(rowmax - rowmax)/sum == 1/sum, so threshold = m / sum.
  const float thr = m / sum;
  for (int64_t j = threadIdx.x; j < v; j += blockDim.x)
    if (expf(r[j] - rowmax) / sum < thr) r[j] = kNegInf;
}

void ApplyMinPCuda(Queue& q, Tensor& logits, const Tensor& min_p) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  ApplyMinPKernel<<<static_cast<unsigned>(n), kBlock, 0, AsStream(q)>>>(logits.Ptr<float>(),
                                                                        min_p.Ptr<float>(), v);
  Check(cudaGetLastError(), "apply_min_p launch");
}

// --- sparse scatter ops (logit-bias add / -inf token mask) ------------------
__global__ void ApplyLogitBiasKernel(float* logits, const int32_t* rows, const int32_t* cols,
                                     const float* biases, int64_t v, int64_t m) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < m; k += step)
    logits[static_cast<int64_t>(rows[k]) * v + cols[k]] += biases[k];
}

void ApplyLogitBiasCuda(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols,
                        const Tensor& biases) {
  const int64_t v = logits.shape[1], m = rows.shape[0];
  if (m == 0) return;
  ApplyLogitBiasKernel<<<GridFor(m), kBlock, 0, AsStream(q)>>>(
      logits.Ptr<float>(), rows.Ptr<int32_t>(), cols.Ptr<int32_t>(), biases.Ptr<float>(), v, m);
  Check(cudaGetLastError(), "apply_logit_bias launch");
}

__global__ void ApplyTokenMaskKernel(float* logits, const int32_t* rows, const int32_t* cols,
                                     int64_t v, int64_t m) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t k = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; k < m; k += step)
    logits[static_cast<int64_t>(rows[k]) * v + cols[k]] = kNegInf;
}

void ApplyTokenMaskCuda(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols) {
  const int64_t v = logits.shape[1], m = rows.shape[0];
  if (m == 0) return;
  ApplyTokenMaskKernel<<<GridFor(m), kBlock, 0, AsStream(q)>>>(logits.Ptr<float>(),
                                                              rows.Ptr<int32_t>(),
                                                              cols.Ptr<int32_t>(), v, m);
  Check(cudaGetLastError(), "apply_token_mask launch");
}

// --- apply_allowed_token_ids (masked_fill, mask TRUE == exclude) -------------
__global__ void ApplyAllowedTokenIdsKernel(float* logits, const int8_t* mask, int64_t total) {
  const int64_t step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += step)
    if (mask[idx]) logits[idx] = kNegInf;
}

void ApplyAllowedTokenIdsCuda(Queue& q, Tensor& logits, const Tensor& mask) {
  const int64_t n = logits.shape[0], v = logits.shape[1];
  if (n == 0 || v == 0) return;
  ApplyAllowedTokenIdsKernel<<<GridFor(n * v), kBlock, 0, AsStream(q)>>>(logits.Ptr<float>(),
                                                                         mask.Ptr<int8_t>(), n * v);
  Check(cudaGetLastError(), "apply_allowed_token_ids launch");
}

// --- Greedy spec-decode rejection sampling (SPEC-REJECTION I3) --------------
// Mirrors the CPU reference (cpu_sample.cpp GreedyRejectionSampleKernel) element
// for element, and upstream's TWO-PHASE decomposition: a per-expanded-row argmax
// (upstream `_compute_local_logits_stats_kernel` per-vocab-block partials +
// `_compute_global_target_argmax`, rejection_sampler_utils.py:923-946) followed by
// a one-thread-per-request sequential accept walk (`_rejection_kernel`, launched
// `num_warps=1` at :1032-1067). Bit-exact vs the CPU reference on the accepted
// token ids: the argmax uses the SAME ArgReduce lowest-index tie-break as
// GreedyArgmax, and the accept walk is pure integer equality.

// Phase 1: target_argmax[row] = argmax(logits[row]) for every expanded row.
__global__ void RejectionRowArgmaxKernel(int32_t* out, const float* logits, int64_t v) {
  const int64_t row = blockIdx.x;
  const float* r = logits + row * v;
  __shared__ float sv[kBlock];
  __shared__ int64_t si[kBlock];

  float bv = kNegInf;
  int64_t bi = kArgSentinel;
  for (int64_t j = threadIdx.x; j < v; j += blockDim.x) ArgReduce(bv, bi, r[j], j);

  sv[threadIdx.x] = bv;
  si[threadIdx.x] = bi;
  __syncthreads();
  for (int s = kBlock / 2; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s)
      ArgReduce(sv[threadIdx.x], si[threadIdx.x], sv[threadIdx.x + s], si[threadIdx.x + s]);
    __syncthreads();
  }
  if (threadIdx.x == 0)
    out[row] = static_cast<int32_t>((si[0] == kArgSentinel) ? 0 : si[0]);
}

// Phase 2: one thread per request walks its draft positions, accepting while the
// draft equals the target argmax; stores the accepted stream + accepted_length+1.
__global__ void GreedyRejectAcceptKernel(int32_t* sampled, int32_t* num_sampled,
                                         const int32_t* target_argmax,
                                         const int32_t* draft_sampled, const int32_t* cu_num_logits,
                                         int64_t width) {
  const int64_t req = blockIdx.x;
  const int64_t start = cu_num_logits[req];
  const int64_t end = cu_num_logits[req + 1];
  const int64_t num_draft_tokens = end - start - 1;
  int32_t* row = sampled + req * width;
  for (int64_t j = 0; j < width; ++j) row[j] = -1;  // PLACEHOLDER_TOKEN_ID pad

  bool accepted = true;
  int64_t accepted_length = 0;
  for (int64_t i = 0; i < num_draft_tokens; ++i) {
    if (!accepted) break;  // upstream `elif accepted:` guard
    const int32_t ta = target_argmax[start + i];
    // +1: draft token i is the input id at the NEXT expanded row (:534). A -1
    // placeholder can never equal an argmax (>= 0) => rejected, no OOB read.
    const int32_t draft = draft_sampled[start + i + 1];
    accepted = (ta == draft);
    row[i] = accepted ? draft : ta;
    accepted_length += accepted ? 1 : 0;
  }
  if (accepted_length == num_draft_tokens) {
    // Bonus token (greedy resample == argmax of the bonus row).
    row[accepted_length] = target_argmax[start + accepted_length];
  }
  num_sampled[req] = static_cast<int32_t>(accepted_length) + 1;
}

// The per-row argmax buffer between the two kernels is the CALLER's, and it used
// to be a file-scope grow-only global here (SPEC-DFLASH2 A2-2, #2802). It could
// not stay one. This function RETURNS with both kernels still queued, so the
// buffer the accept kernel reads is live after the call; a later call with more
// rows grew the global and `cudaFree`d exactly that buffer. The accept kernel
// then read freed memory, and because speculative decoding is lossless the loud
// outcome (an illegal access) is the lucky one — the quiet outcome is a garbage
// accept prefix and silently wrong emitted ids. The grow path's `cudaFree` was
// also a device-wide synchronize inside a function that advertises waiting on
// nothing. Both go away with the global: the scratch is now a parameter whose
// lifetime the caller ties to the in-flight window.
void GreedyRejectionSampleCuda(Queue& q, Tensor& sampled, Tensor& num_sampled,
                               Tensor& target_argmax, const Tensor& logits,
                               const Tensor& draft_sampled, const Tensor& cu_num_logits) {
  const int64_t num_logits = logits.shape[0], v = logits.shape[1];
  const int64_t num_reqs = cu_num_logits.shape[0] - 1;
  if (num_reqs == 0 || num_logits == 0 || v == 0) return;
  cudaStream_t s = AsStream(q);

  int32_t* argmax = target_argmax.Ptr<int32_t>();
  RejectionRowArgmaxKernel<<<static_cast<unsigned>(num_logits), kBlock, 0, s>>>(
      argmax, logits.Ptr<float>(), v);
  Check(cudaGetLastError(), "greedy_rejection_sample argmax launch");
  GreedyRejectAcceptKernel<<<static_cast<unsigned>(num_reqs), 1, 0, s>>>(
      sampled.Ptr<int32_t>(), num_sampled.Ptr<int32_t>(), argmax,
      draft_sampled.Ptr<int32_t>(), cu_num_logits.Ptr<int32_t>(), sampled.shape[1]);
  Check(cudaGetLastError(), "greedy_rejection_sample accept launch");
}

struct Registrar {
  Registrar() {
    RegisterOp(OpId::kApplyTemperature, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ApplyTemperatureFn>(&ApplyTemperatureCuda)));
    RegisterOp(OpId::kGreedyArgmax, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<GreedyArgmaxFn>(&GreedyArgmaxCuda)));
    RegisterOp(OpId::kTopKValuesIndices, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<TopKValuesIndicesFn>(&TopKValuesIndicesCuda)));
    RegisterOp(OpId::kApplyTopKTopP, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ApplyTopKTopPFn>(&ApplyTopKTopPCuda)));
    RegisterOp(OpId::kComputeProbs, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ComputeProbsFn>(&ComputeProbsCuda)));
    RegisterOp(OpId::kComputeLogprobs, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ComputeLogprobsFn>(&ComputeLogprobsCuda)));
    RegisterOp(OpId::kRandomSample, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<RandomSampleFn>(&RandomSampleCuda)));
    RegisterOp(OpId::kTopKTopPSample, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<TopKTopPSampleFn>(&TopKTopPSampleCuda)));
    RegisterOp(OpId::kGreedyRejectionSample, DeviceType::kCUDA,
               reinterpret_cast<void*>(
                   static_cast<GreedyRejectionSampleFn>(&GreedyRejectionSampleCuda)));
    RegisterOp(OpId::kApplyPenalties, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ApplyPenaltiesFn>(&ApplyPenaltiesCuda)));
    RegisterOp(OpId::kApplyMinP, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ApplyMinPFn>(&ApplyMinPCuda)));
    RegisterOp(OpId::kApplyLogitBias, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ApplyLogitBiasFn>(&ApplyLogitBiasCuda)));
    RegisterOp(OpId::kApplyTokenMask, DeviceType::kCUDA,
               reinterpret_cast<void*>(static_cast<ApplyTokenMaskFn>(&ApplyTokenMaskCuda)));
    RegisterOp(
        OpId::kApplyAllowedTokenIds, DeviceType::kCUDA,
        reinterpret_cast<void*>(static_cast<ApplyAllowedTokenIdsFn>(&ApplyAllowedTokenIdsCuda)));
  }
} registrar;

}  // namespace
}  // namespace vt::cuda
