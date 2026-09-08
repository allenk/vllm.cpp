// Vulkan decode-GEMV variant A/B: rows-per-workgroup and packed 32-bit loads
// against the shipped one-element-per-workgroup, 16-bit-load kernel.
// BACKEND-VULKAN work row BACKEND-VULKAN-GEMVROWS.
//
// WHY AN ISOLATED SWEEP EXISTS AT ALL. vt_matmul_vec is 89% of 27B decode GPU
// time, and a 27B e2e run costs ~50 GB of weights and minutes per leg. Deciding
// which variant is worth an e2e run from e2e runs is the expensive way round; a
// sweep over the SAME (k, n) shapes the model actually dispatches answers it in
// seconds and, unlike an e2e number, isolates the kernel from every other cost.
//
// SAME BINARY, ONE VARIABLE. Every arm runs this executable; the arms differ only
// in VT_VULKAN_GEMV_ROWS / VT_VULKAN_GEMV_PACK / VT_VULKAN_GEMV_UNROLL, which are
// specialization constants on ONE committed SPIR-V module. Comparing two builds
// is what produced a false 1.2x reading for the subgroup tactic earlier in this
// campaign, and .agents/benchmark-protocol.md rules it out.
//
// EACH ARM PROVES WHICH VARIANT IT RAN, by printing the pipeline cache KEY --
// module name plus specialization values. Every variant computes the same
// numbers, so timings alone can never tell the arms apart, and a variant that
// silently declined would post a perfectly plausible result.
//
// IT ALSO VERIFIES EVERY SHAPE against a host f64 oracle, so a variant cannot be
// fast by being wrong.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/vulkan/vulkan_context.h"

namespace {

using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

double MedianMs(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[v.size() / 2];
}

// FNV-1a over the RAW BITS of the whole output vector. The rows-per-workgroup
// axis is supposed to be BIT-IDENTICAL across its values -- each lane accumulates
// the same strided subset of K in the same order, and only the assignment of
// output elements to workgroups moves -- while the packed-load axis is supposed
// NOT to be, because it repartitions K. A spot-checked relative error cannot tell
// those two claims apart; a checksum of every output bit can, and it is the
// cheapest possible evidence that "bit-identical" is a fact rather than an
// argument about the source.
uint64_t BitHash(const std::vector<float>& v) {
  uint64_t h = 1469598103934665603ull;
  for (float f : v) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    for (int b = 0; b < 4; ++b) {
      h ^= static_cast<uint64_t>((bits >> (8 * b)) & 0xFFu);
      h *= 1099511628211ull;
    }
  }
  return h;
}

struct Shape {
  int64_t k;
  int64_t n;
  const char* what;
};

// The decode GEMV shapes Qwen3.6-27B dispatches (hidden 5120, intermediate
// 17408, head_dim 256, 24 q heads with an output gate, 4 kv heads; GDN layers
// carry 16x128 key and 48x128 value heads). Read from the model's own config
// rather than invented, because a sweep over unrepresentative shapes answers a
// question nobody asked.
constexpr Shape kShapes[] = {
    {5120, 34816, "mlp gate+up (merged)"},
    {17408, 5120, "mlp down"},
    {5120, 12288, "attn q + output gate"},
    {5120, 2048, "attn kv"},
    {6144, 5120, "attn/gdn out"},
    {5120, 8192, "gdn in (qk)"},
    {5120, 6144, "gdn in (v)"},

    // LOCAL ADDITION (not upstream). The decode GEMV shapes of Qwen3-0.6B, the
    // model actually present on this box, read from the safetensors header
    // rather than derived: hidden 1024, intermediate 3072, head_dim 128, 16 q
    // heads, 8 kv heads, 28 layers, vocab 151936, BF16 throughout.
    //
    // WHY BOTH TABLES. vulkan-dispatch-floor measured a flat ~0.075 ms per
    // dispatch on this GPU from 256 elements out to 1e6, so anything under a
    // few million elements reports submit cost, not kernel quality. Every 0.6B
    // per-layer weight lands at 1-3M elements and is therefore INSIDE that
    // floor; the 27B shapes above are 10-178M and are not. The one 0.6B shape
    // that clears the floor is lm_head at 155.6M, which also happens to be
    // 311 MB and so cannot sit in L2 the way the smaller repeated shapes do.
    // It is the only shape here that is simultaneously from a model we can run,
    // above the dispatch floor, and immune to the cache artefact.
    {1024, 2048, "0.6B attn q"},
    {1024, 1024, "0.6B attn k"},
    {1024, 1024, "0.6B attn v"},
    {2048, 1024, "0.6B attn out"},
    {1024, 3072, "0.6B mlp gate"},
    {1024, 3072, "0.6B mlp up"},
    {3072, 1024, "0.6B mlp down"},
    {1024, 151936, "0.6B lm_head"},
};

}  // namespace

// GPU-timestamp milliseconds accumulated for one shader, or -1 when the context
// is not profiling (VT_VULKAN_DISPATCH_STATS unset, or the device cannot
// timestamp). This is the SAME clock the 27B two-length decode profile used, so a
// number from here is comparable with one from there; the wall clock is not,
// because it also contains submit and fence-wait cost that batched decode
// amortises over hundreds of dispatches.
double GpuMsFor(const vt::vulkan::VulkanContext& ctx, const char* name) {
  for (const auto& kv : ctx.DispatchTimeMs()) {
    if (kv.first == name) return kv.second;
  }
  return 0.0;
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 40;
  const int warmup = argc > 2 ? std::atoi(argv[2]) : 8;
  // GB10's published unified-memory bandwidth. Reported as a denominator so a
  // variant's number is readable as a fraction of the roof rather than in
  // isolation; override for another box.
  const double roof_gbs = argc > 3 ? std::atof(argv[3]) : 273.0;

  if (!vt::vulkan::VulkanDeviceAvailable()) {
    std::fprintf(stderr, "no Vulkan device\n");
    return 2;
  }
  auto& ctx = vt::vulkan::VulkanContext::Get();
  vt::Backend& vk = vt::GetBackend(DeviceType::kVULKAN);
  Queue q = vk.CreateQueue();
  const Device d{DeviceType::kVULKAN, 0};

  auto env = [](const char* name) {
    const char* v = std::getenv(name);
    return v ? v : "(unset)";
  };
  std::printf("device               : %s\n", ctx.device_name().c_str());
  std::printf("subgroup size        : %u\n", ctx.subgroup_size());
  std::printf("VT_VULKAN_GEMV       : %s\n", env("VT_VULKAN_GEMV"));
  std::printf("VT_VULKAN_GEMV_ROWS  : %s\n", env("VT_VULKAN_GEMV_ROWS"));
  std::printf("VT_VULKAN_GEMV_PACK  : %s\n", env("VT_VULKAN_GEMV_PACK"));
  std::printf("VT_VULKAN_GEMV_UNROLL: %s\n", env("VT_VULKAN_GEMV_UNROLL"));
  std::printf("reps                 : %d (warmup %d), roof %.1f GB/s\n\n", reps, warmup,
              roof_gbs);
  std::printf("(gpu ms is the GPU timestamp when VT_VULKAN_DISPATCH_STATS is set,\n"
              " else a copy of wall; wall includes submit + fence amortised over 8)\n\n");
  std::printf("%-22s %7s %7s %10s %10s %10s %8s %18s  %s\n", "shape", "K", "N", "gpu ms",
              "wall ms", "GB/s", "%roof", "outbits", "pipeline key");

  int failures = 0;
  double total_ms = 0.0;
  double total_bytes = 0.0;

  for (const Shape& s : kShapes) {
    // bf16 operands: the 27B's storage dtype, and the one the packed-load axis
    // applies to at all.
    // FULL-MANTISSA OPERANDS, and that is the point. The obvious small-integer
    // pattern (0.5*(i%7-3) and friends) sums EXACTLY in f32, so every variant
    // produces the same bits no matter how it repartitions K -- which would make
    // the checksum column look like proof of bit-identity when it proved only
    // that the test data could not tell. A cheap LCG over [0.5, 1.5) puts eight
    // real mantissa bits in every bf16 element, so ROWS (same partition of K,
    // same order) must still match the baseline bit for bit while PACK (a
    // different partition) must not. Kept POSITIVE so the k-long dot product does
    // not cancel: a near-zero sum would make the relative-error check below
    // meaningless without saying so.
    std::vector<uint16_t> ha(static_cast<size_t>(s.k));
    std::vector<uint16_t> hb(static_cast<size_t>(s.n * s.k));
    uint32_t rng = 0x9E3779B9u;
    auto next = [&rng] {
      rng = rng * 1664525u + 1013904223u;
      return 0.5f + static_cast<float>(rng >> 9) / static_cast<float>(1 << 23);
    };
    for (int64_t i = 0; i < s.k; ++i) ha[static_cast<size_t>(i)] = vt::F32ToBF16(next());
    for (int64_t i = 0; i < s.n * s.k; ++i) hb[static_cast<size_t>(i)] = vt::F32ToBF16(next());

    void* da = vk.Alloc(ha.size() * sizeof(uint16_t));
    void* db = vk.Alloc(hb.size() * sizeof(uint16_t));
    auto* dout = static_cast<float*>(vk.Alloc(static_cast<size_t>(s.n) * sizeof(float)));
    vk.Copy(q, da, ha.data(), ha.size() * sizeof(uint16_t));
    vk.Copy(q, db, hb.data(), hb.size() * sizeof(uint16_t));
    vk.Synchronize(q);

    Tensor ta = Tensor::Contiguous(da, DType::kBF16, d, {1, s.k});
    Tensor tb = Tensor::Contiguous(db, DType::kBF16, d, {s.n, s.k});
    Tensor to = Tensor::Contiguous(dout, DType::kF32, d, {1, s.n});

    const std::vector<std::string> keys_before = ctx.PipelineKeysFor("vt_matmul_vec");
    for (int i = 0; i < warmup; ++i) vt::MatmulBT(q, to, ta, tb);
    vk.Synchronize(q);

    // WHICH VARIANT RAN. The key added by this shape (or, if an earlier shape
    // already built the same specialization, the single key that matches it).
    const std::vector<std::string> keys_after = ctx.PipelineKeysFor("vt_matmul_vec");
    std::string key;
    for (const std::string& kk : keys_after) {
      if (std::find(keys_before.begin(), keys_before.end(), kk) == keys_before.end()) key = kk;
    }
    if (key.empty()) key = keys_after.size() == 1 ? keys_after.front() : "(reused)";

    // BATCHED, NOT ONE-AT-A-TIME. Decode records hundreds of dispatches into one
    // command buffer and submits once; a harness that fences after every dispatch
    // measures a regime the model never runs in, and it showed: fenced-per-call
    // this sweep read 45% of the bandwidth roof where the e2e two-length profile
    // of the same kernel read 89%. `inner` dispatches per timed rep puts the
    // kernel back in the batched regime. The backend emits a memory barrier
    // between recorded dispatches, so they still execute one at a time on the GPU
    // exactly as they do in decode -- what is amortised is the SUBMIT, not the
    // kernel.
    const int inner = 8;
    std::vector<double> ms;
    ms.reserve(static_cast<size_t>(reps));
    const double gpu0 = GpuMsFor(ctx, "vt_matmul_vec");
    for (int i = 0; i < reps; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int j = 0; j < inner; ++j) vt::MatmulBT(q, to, ta, tb);
      vk.Synchronize(q);
      const auto t1 = std::chrono::steady_clock::now();
      ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / inner);
    }
    const double gpu_per_call =
        (GpuMsFor(ctx, "vt_matmul_vec") - gpu0) / (static_cast<double>(reps) * inner);

    std::vector<float> got(static_cast<size_t>(s.n), 0.0f);
    vk.Copy(q, got.data(), dout, got.size() * sizeof(float));
    vk.Synchronize(q);
    // Host oracle in f64, so no device accumulation order is privileged. Spot
    // rows only: a full n-row oracle at n=34816, k=5120 is 178M host FMAs per
    // shape and would dominate the run.
    double worst = 0.0;
    for (int64_t j = 0; j < std::min<int64_t>(s.n, 48); ++j) {
      double acc = 0.0;
      for (int64_t c = 0; c < s.k; ++c) {
        acc += static_cast<double>(vt::BF16ToF32(ha[static_cast<size_t>(c)])) *
               static_cast<double>(vt::BF16ToF32(hb[static_cast<size_t>(j * s.k + c)]));
      }
      worst = std::max(worst, std::fabs(acc - static_cast<double>(got[static_cast<size_t>(j)])) /
                                  std::max(1.0, std::fabs(acc)));
    }
    // Also check the LAST row, which is the one a rows-per-workgroup bug at the
    // end of the dispatch would corrupt while every early row stayed right.
    {
      const int64_t j = s.n - 1;
      double acc = 0.0;
      for (int64_t c = 0; c < s.k; ++c) {
        acc += static_cast<double>(vt::BF16ToF32(ha[static_cast<size_t>(c)])) *
               static_cast<double>(vt::BF16ToF32(hb[static_cast<size_t>(j * s.k + c)]));
      }
      worst = std::max(worst, std::fabs(acc - static_cast<double>(got[static_cast<size_t>(j)])) /
                                  std::max(1.0, std::fabs(acc)));
    }

    const double wall = MedianMs(ms);
    // GPU time when the context is profiling, wall otherwise. Stated per row so a
    // reader can see how much of each figure is submit overhead.
    const double med = gpu_per_call > 0 ? gpu_per_call : wall;
    // WEIGHT bytes only. The activation vector is k*2 bytes and every workgroup
    // re-reads it, but it is L2-resident by inspection (10 KB at k=5120), so
    // counting it would inflate the figure past the DRAM roof and mean nothing.
    const double bytes = static_cast<double>(s.n) * static_cast<double>(s.k) * 2.0;
    const double gbs = med > 0 ? bytes / (med / 1e3) / 1e9 : 0.0;
    total_ms += med;
    total_bytes += bytes;
    std::printf("%-22s %7lld %7lld %10.4f %10.4f %10.1f %7.1f%% %018llx  %s%s\n", s.what,
                (long long)s.k, (long long)s.n, med, wall, gbs, 100.0 * gbs / roof_gbs,
                (unsigned long long)BitHash(got), key.c_str(),
                worst < 1e-3 ? "" : "   *** WRONG ***");
    if (!(worst < 1e-3)) {
      ++failures;
      std::printf("  worst rel err: %.3e\n", worst);
    }

    vk.Free(da);
    vk.Free(db);
    vk.Free(dout);
  }

  const double agg = total_ms > 0 ? total_bytes / (total_ms / 1e3) / 1e9 : 0.0;
  std::printf("\nTOTAL                  %26.4f %21.1f %7.1f%%\n", total_ms, agg,
              100.0 * agg / roof_gbs);

  vk.DestroyQueue(q);
  return failures == 0 ? 0 : 1;
}
