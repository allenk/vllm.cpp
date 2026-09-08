// Triton-CPU acceleration provider — SHELL (T-29 Step 2).
//
// WHAT THIS IS. A provider registered on DeviceType::kCPU under the name
// "triton-cpu", at a priority above the built-in "vt-native" kernels. Every
// kernel here currently DECLINES: it forwards to the next provider down via
// GetOpFallback(), which is vt-native. So with this provider selected the engine
// produces BYTE-IDENTICAL output to a build without it.
//
// WHY A SHELL FIRST. This separates "the wiring is wrong" from "the kernel is
// wrong". A first failure with a real kernel in place is ambiguous; a first
// failure here is not. Same discipline as the ncnn video patch in the sibling
// project, whose OFF state had to be bit-identical before any ON state counted.
//
// WHY A PROVIDER AND NOT A NEW DeviceType. op_provider.h already supplies every
// mechanism a Triton backend needs, and building a parallel DeviceType would
// duplicate them:
//   * DEVICE-level gate      OpProvider::supports(caps)
//   * PER-CALL refusal       the kernel calls GetOpFallback() and forwards,
//                            because GetOp has no shape to inspect
//   * same-binary A/B        VT_OP_PROVIDER_DISABLE=triton-cpu
//   * proof of selection     VT_OP_PROVIDER_STATS=1
// A new DeviceType would also need a Platform, a device.h enum change and CMake
// device wiring, and would still have to re-derive the shape-gated fallback.
//
// NAMING. Deliberately "triton-cpu", NOT "triton": this tree already spells
// VLLM_CPP_TRITON for the CUDA Triton-AOT cubins (cmake/TritonAOT.cmake), and a
// second unqualified "triton" next to it is a near-miss for both humans and
// grep. Same reasoning their Tenstorrent spec used to reject kBLACKHOLE.
//
// DEFAULT OFF. supports() returns false unless VLLM_CPP_TRITON_CPU is set to a
// non-zero value, so linking this file changes nothing until asked. Selection is
// cached, so this is a process-lifetime switch, not a per-call one.
#include <atomic>
#include <chrono>
#include <climits>
#include <fstream>
#include <set>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <cmath>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <string>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include "vt/op_provider.h"
#include "vt/ops.h"
#include "vt/tensor.h"

// vt-native's SiluAndMul runs on this pool; so must ours, or every later A/B is a
// comparison against a crippled control. Private to src/vt/cpu, but we are in the
// same library — and reaching for it is the honest choice here.
#include "../cpu/cpu_threadpool.h"

namespace vt::triton_cpu {
namespace {

constexpr const char* kProviderName = "triton-cpu";
// Above vt-native (0). Nothing else registers on kCPU at this priority today;
// if something does, the strcmp tiebreak makes the winner deterministic anyway.
constexpr int kProviderPriority = 10;

bool EnabledOnce() {
  static const bool on = [] {
    const char* e = std::getenv("VLLM_CPP_TRITON_CPU");
    return e != nullptr && std::strcmp(e, "0") != 0 && e[0] != '\0';
  }();
  return on;
}

// Device-level gate. Cheap and side-effect free, per the OpProviderSupportsFn
// contract. Shape/dtype refusal does NOT belong here — it belongs in the kernel,
// which is what the forwarding below is.
bool TritonCpuSupports(const ProviderCaps&) { return EnabledOnce(); }

// ---------------------------------------------------------------------------
// SHAPE PROBE (VLLM_CPP_TRITON_CPU_PROBE=1). Prints the first call to each op,
// then goes quiet.
//
// WHY THIS EXISTS. Writing a Triton kernel means committing to a dtype, a tile
// size and — the part that bites — a set of SPECIALIZATIONS. Triton bakes any
// argument it believes is divisible by 16 into the compiled code as an alignment
// assumption, and an argument that is 1 at compile time disappears entirely
// (Task 0, §0d). Both are silent correctness bugs if the runtime value differs.
// So the alignment column below is not decoration: it is the input to the shape
// gate that decides when the kernel may run at all.
//
// Guessing the dtype instead of measuring it would also be the same mistake this
// project keeps writing down: assume, build, and discover the assumption was
// wrong after the fact.
bool ProbeEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VLLM_CPP_TRITON_CPU_PROBE");
    return e != nullptr && std::strcmp(e, "0") != 0 && e[0] != '\0';
  }();
  return on;
}

const char* DTypeName(DType d) {
  switch (d) {
    case DType::kF32: return "f32";
    case DType::kF16: return "f16";
    case DType::kBF16: return "bf16";
    case DType::kI8: return "i8";
    case DType::kI32: return "i32";
    case DType::kI64: return "i64";
    default: return "other/quant";
  }
}

void Probe(const char* op,
           std::initializer_list<std::pair<const char*, const Tensor*>> ts) {
  std::fprintf(stderr, "[triton-cpu probe] %s\n", op);
  for (const auto& [name, t] : ts) {
    if (t == nullptr) {
      std::fprintf(stderr, "    %-6s (null)\n", name);
      continue;
    }
    // Largest power-of-two alignment of the data pointer, capped at 64. This is
    // what decides whether a Triton `:16` divisibility specialization is safe.
    std::size_t align = 1;
    const auto addr = reinterpret_cast<std::uintptr_t>(t->data);
    if (addr != 0) {
      while (align < 64 && (addr % (align * 2)) == 0) align *= 2;
    }
    std::fprintf(stderr,
                 "    %-6s dtype=%-4s rank=%d shape=[%lld,%lld,%lld,%lld] "
                 "stride=[%lld,%lld,%lld,%lld] align=%zuB\n",
                 name, DTypeName(t->dtype), t->rank,
                 (long long)t->shape[0], (long long)t->shape[1],
                 (long long)t->shape[2], (long long)t->shape[3],
                 (long long)t->stride[0], (long long)t->stride[1],
                 (long long)t->stride[2], (long long)t->stride[3], align);
  }
}

// One line per call site. `static bool` initialization is thread-safe in C++11;
// the store is not, so a racing first call can print twice. That is deliberate:
// a duplicated diagnostic line is cheaper than a mutex on a path that is
// supposed to disappear once the shapes are known.
#define VT_TRITON_CPU_PROBE(NAME, ...)                                       \
  do {                                                                        \
    if (ProbeEnabled()) {                                                     \
      static bool probed = false;                                             \
      if (!probed) {                                                          \
        probed = true;                                                        \
        Probe(NAME, __VA_ARGS__);                                             \
      }                                                                       \
    }                                                                         \
  } while (0)

// ---------------------------------------------------------------------------
// STEP 3 — the first real Triton kernel: SiluAndMul.
//
// WHY THIS OP AND NOT kPagedAttention. Paged attention is where a Triton kernel
// is actually worth writing (no vendor library ships one), but that makes it the
// DESTINATION of this arc, not its first step. It carries online softmax, a block
// table walk and masking all at once, so a first failure there cannot be
// attributed. SiluAndMul is the smallest real op in the model — elementwise, one
// dimension — and it still exercises the mechanism that matters: it goes through
// `tl.exp`, which is the `vec_lib` path. Ladder from here: SiluAndMul (mechanism)
// -> RmsNorm (reduction) -> MatmulBT (tl.dot) -> PagedAttention.
//
// HOW THE KERNEL ARRIVES. dlopen, not a link dependency — rung 3 of the project's
// linking ladder, the same rung the TFLite EP sits on. Absent .so => this op
// declines and the engine is unchanged, so the Triton artifact is a deployment
// question rather than a build one. Path comes from VLLM_CPP_TRITON_CPU_LIB.
//
// THE ABI. Read off the compiled artifact, not assumed:
//   define void @silu_and_mul_kernel(ptr, ptr, i32, i32, i32, i32,i32,i32, i32,i32,i32)
//                                    out  x    d    xs   os   px py pz    gx gy gz
// Triton's CPU backend compiles ONE program instance; the CALLER writes the grid
// loop. Pointer arithmetic inside the kernel is in ELEMENTS, which is also the
// unit of Tensor::stride, so the strides pass through unconverted.
using SiluKernelFn = void (*)(void* out, const void* x, std::int32_t d,
                              std::int32_t x_stride, std::int32_t out_stride,
                              std::int32_t px, std::int32_t py, std::int32_t pz,
                              std::int32_t gx, std::int32_t gy, std::int32_t gz);

using RmsNormKernelFn = void (*)(void* out, const void* x, const void* w, float eps,
                                 std::int32_t n, std::int32_t x_stride,
                                 std::int32_t out_stride, std::int32_t px,
                                 std::int32_t py, std::int32_t pz, std::int32_t gx,
                                 std::int32_t gy, std::int32_t gz);

using RmsNormResidKernelFn = void (*)(void* out, const void* x, const void* w, void* r,
                                      float eps, std::int32_t n, std::int32_t x_stride,
                                      std::int32_t out_stride, std::int32_t px,
                                      std::int32_t py, std::int32_t pz, std::int32_t gx,
                                      std::int32_t gy, std::int32_t gz);

// Must match build_kernels.py. These are baked into the GATE, not the kernel: the
// kernels take n/d at run time, but they were COMPILED at these values, and Triton
// turns "divisible by 16" into a specialization on arguments it saw that way.
// Accepting a size the kernel never saw would be a silent wrong answer, not a crash.
constexpr std::int32_t kSiluD = 3072;    // intermediate_size
constexpr std::int32_t kSiluBlock = 1024;
constexpr std::int32_t kRmsN = 1024;     // hidden_size

// One directory, fixed file names — the interface stopped being "a .so" the moment
// there was more than one kernel. Absent directory => every op declines, which is
// the dlopen rung degrading, and is a tested state rather than an assumed one.
void* LoadKernelSymbol(const char* so_name, const char* symbol) {
#ifdef _WIN32
  (void)so_name; (void)symbol;
  return nullptr;
#else
  const char* dir = std::getenv("VLLM_CPP_TRITON_CPU_DIR");
  if (dir == nullptr || dir[0] == '\0') return nullptr;
  const std::string path = std::string(dir) + "/" + so_name;
  void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (h == nullptr) {
    std::fprintf(stderr, "[triton-cpu] dlopen(%s) failed: %s\n", path.c_str(),
                 dlerror());
    return nullptr;
  }
  void* f = dlsym(h, symbol);
  if (f == nullptr) {
    std::fprintf(stderr, "[triton-cpu] dlsym(%s) failed: %s\n", symbol, dlerror());
    return nullptr;
  }
  std::fprintf(stderr, "[triton-cpu] loaded %s from %s\n", symbol, path.c_str());
  return f;
#endif
}

// ⭐ BLOCK is a property of the ARTIFACT, not of this file. The kernel source is
// blocked (`cols = program_id(1) * BLOCK + arange(0, BLOCK)`), so the length is
// absorbed by how many grid cells the caller emits — but the caller has to know how
// wide one cell is, and that number was frozen into the .so at compile time.
//
// ⚠️ Guessing it is not symmetric. Guess too SMALL and the kernel is called more
// times than needed: every extra cell is fully masked, so the answer stays right and
// only the work is wasted. Guess too LARGE and `col_blocks` comes out too small, the
// tail is never visited, and the output keeps whatever was in the buffer — a silent
// wrong answer, not a crash. So this is read off a manifest the build writes, and
// only falls back to the compiled-in default when there is no manifest to read.
std::string ManifestValue(const char* key) {
  const char* dir = std::getenv("VLLM_CPP_TRITON_CPU_DIR");
  if (dir == nullptr || dir[0] == '\0') return {};
  std::ifstream f(std::string(dir) + "/manifest.txt");
  std::string line;
  const std::string k = std::string(key) + "=";
  while (std::getline(f, line)) {
    if (line.compare(0, k.size(), k) == 0) return line.substr(k.size());
  }
  return {};
}

std::int32_t SiluBlockWidth() {
  static const std::int32_t v = [] {
    const std::string s = ManifestValue("silu_block");
    if (s.empty()) return kSiluBlock;
    const long n = std::strtol(s.c_str(), nullptr, 10);
    // A manifest that says something impossible is worse than no manifest.
    if (n <= 0 || (n & (n - 1)) != 0 || n > (1 << 20)) return kSiluBlock;
    std::fprintf(stderr, "[triton-cpu] manifest: silu_block=%ld\n", n);
    return static_cast<std::int32_t>(n);
  }();
  return v;
}

// ⭐ THE ARTIFACT DECLARES ITS OWN SHAPE CONTRACT. Two kernel styles now exist and
// one binary has to be able to load either, so the difference cannot be a build-time
// assumption on this side — it has to travel with the .so.
//
//   grid  (default, and what the original set is)
//         `cols = program_id(1) * BLOCK + arange(0, BLOCK)`. One tile per grid cell,
//         so the CALLER walks the columns. Requires knowing BLOCK, and the size the
//         kernel was compiled at is load-bearing for the ops that never got a second
//         grid dimension at all.
//
//   loop  `for start in range(0, n, BLOCK)`. The KERNEL walks the columns, so the
//         caller emits exactly one cell and does not need to know BLOCK. Reductions
//         can only be written this way — a running accumulator cannot survive across
//         grid cells, which have no shared state.
//
// ⚠️ Getting this wrong is not symmetric, and neither direction is a crash:
//   loop artifact + grid caller  → the kernel runs its whole loop once per cell, so
//                                  the answer is right and the work is done N times.
//   grid artifact + loop caller  → only the first tile is ever visited. The tail keeps
//                                  whatever was in the buffer. ★ Silently wrong.
// So the flag is read off the artifact rather than inferred, and its absence means
// `grid`, which is what every previously-built directory is.
bool ArtifactIsLoopStyle() {
  static const bool v = [] {
    const std::string s = ManifestValue("style");
    const bool loop = (s == "loop");
    if (!s.empty())
      std::fprintf(stderr, "[triton-cpu] manifest: style=%s%s\n", s.c_str(),
                   loop ? "  (shape-general)" : "");
    return loop;
  }();
  return v;
}

// ⭐ THE CALLING CONVENTION AND THE GATE POLICY ARE TWO DIFFERENT DECISIONS, and
// collapsing them made the first A/B unattributable. Swapping in the loop-style
// artifacts changed both the kernel AND who executes the calls the old gate used to
// refuse — 896 of them moved from the native implementation to ours — so a difference
// in tokens/s had two candidate causes and no way to tell them apart.
//
// The convention MUST follow the artifact: a loop kernel driven by a grid-walking
// caller repeats the whole row per cell, and the reverse never visits the tail. But
// the gate is free. Forcing it strict against a loop artifact gives the missing cell:
// same division of labour as the pinned run, only the kernel differs.
//
//   VLLM_CPP_TRITON_CPU_STRICT_GATES=1
//
// ⚠️ Only meaningful with a loop-style artifact — against a pinned one the gates are
// already strict and this is a no-op.
bool GatesRelaxed() {
  static const bool v = [] {
    if (!ArtifactIsLoopStyle()) return false;
    const char* e = std::getenv("VLLM_CPP_TRITON_CPU_STRICT_GATES");
    const bool strict = (e != nullptr && e[0] == '1');
    if (strict)
      std::fprintf(stderr,
                   "[triton-cpu] ★ STRICT_GATES=1 — loop kernels, pinned-era size "
                   "checks (the isolating cell)\n");
    return !strict;
  }();
  return v;
}

SiluKernelFn SiluKernel() {
  static auto fn = reinterpret_cast<SiluKernelFn>(
      LoadKernelSymbol("silu_and_mul.so", "silu_and_mul_kernel"));
  return fn;
}
RmsNormKernelFn RmsKernel() {
  static auto fn = reinterpret_cast<RmsNormKernelFn>(
      LoadKernelSymbol("rms_norm.so", "rms_norm_kernel"));
  return fn;
}
RmsNormResidKernelFn RmsResidKernel() {
  static auto fn = reinterpret_cast<RmsNormResidKernelFn>(
      LoadKernelSymbol("rms_norm_resid.so", "rms_norm_resid_kernel"));
  return fn;
}

using MatmulBTKernelFn = void (*)(void* out, const void* a, const void* b,
                                  std::int32_t M, std::int32_t N, std::int32_t K,
                                  std::int32_t a_stride, std::int32_t b_stride,
                                  std::int32_t out_stride, std::int32_t px,
                                  std::int32_t py, std::int32_t pz, std::int32_t gx,
                                  std::int32_t gy, std::int32_t gz);

// Must match build_kernels.py. BM=1 is deliberate, not lazy: tl.dot on the CPU
// backend has no minimum tile (unlike the GPU path's 16), the probe showed
// GreedyArgmax takes logits of [1, 151936] so lm_head is ALWAYS M=1 even during
// prefill, and at that shape a BM of 16 would turn the op from memory-bound
// (311 MB of weights, ~3.1 ms) into compute-bound (5.0 GFLOP, ~10 ms). Fitting the
// shape is worth real time here, not neatness.
constexpr std::int32_t kMmBM = 1;
constexpr std::int32_t kMmBNDefault = 64;

MatmulBTKernelFn MatmulBTKernel() {
  static auto fn = reinterpret_cast<MatmulBTKernelFn>(
      LoadKernelSymbol("matmul_bt.so", "matmul_bt_kernel"));
  return fn;
}

// ⭐ THE ARTIFACT'S BN IS MEASURED, NOT DECLARED.
//
// This used to be `constexpr kMmBN = 64`, which silently assumed every artifact this
// binary would ever load had been built at 64. That held only because one machine
// built every artifact. It stops holding the moment a second platform builds its own
// set — and RISC-V has to, because a wide bf16 tile there either fails to compile at
// all (no zvfbfmin, >240 s) or compiles into scalarized code.
//
// ⚠️ And the failure is silent, not a crash, and it is not symmetric:
//   artifact BN=16, caller assumes 64 → n_blocks = N/64 cells, each covering 16
//                                      columns ⇒ only N/4 of the output is ever
//                                      written and the rest keeps whatever was in
//                                      the buffer. ★ A wrong answer that looks fine.
//   artifact BN=64, caller assumes 16 → 4x the cells, everything past N masked ⇒
//                                      right answer, wasted work.
// Same asymmetry ArtifactIsLoopStyle() documents for the style flag, one layer down.
//
// A manifest key would fix the mechanism but not the trust: it is the build script
// asserting something about the .so. This is measurable instead, so it is measured —
// fill the output with a sentinel, run ONLY grid cell (0, 0), and count how many
// leading columns changed. That count IS the BN the kernel was compiled at, because
// the kernel writes `offs_n = pid_n * BN + arange(0, BN)` masked to `< N`.
//
// Cost is one kernel call with ~160 KB of scratch, once, on first use.
std::int32_t MmBlockWidth() {
  static const std::int32_t v = [] {
    MatmulBTKernelFn fn = MatmulBTKernel();
    if (fn == nullptr) return kMmBNDefault;

    // Wider than any BN worth building: the block sweep turned up a U-curve whose
    // right-hand side was already 10x the optimum well before this.
    constexpr std::int32_t kProbeN = 8192;
    // ⚠️⚠️ EVERY INTEGER ARGUMENT HERE MUST BE DIVISIBLE BY 16, and the first
    // version of this probe was not. Triton specializes integer arguments it sees
    // as divisible by 16 (`tt.divisibility=16`) and emits aligned vector accesses
    // on that promise. The artifacts are built at K=1024 with 16-divisible strides,
    // so a probe passing K=8 breaks a contract the .so was compiled against and
    // segfaults inside the kernel. The BN=64 artifact happened to survive the same
    // violation, which was luck about page boundaries, not safety.
    //
    // ⇒ This is a FOURTH layer of shape binding, alongside kernel source, launcher
    //   and gate: the specialization the artifact was compiled under. It is unstated
    //   anywhere and it silently constrains every caller, not just this probe.
    constexpr std::int32_t kProbeK = 16;
    constexpr std::uint16_t kSentinel = 0x7FC0;  // bf16 NaN, never a valid product
    static_assert(kProbeN % 16 == 0 && kProbeK % 16 == 0,
                  "probe arguments must honour the artifact's tt.divisibility=16");

    std::vector<std::uint16_t> out(kProbeN, kSentinel);
    // Over-allocated on purpose. The kernel's masked loads compute addresses across
    // the whole BK tile before masking, and BK is exactly what we do not know yet.
    std::vector<std::uint16_t> a(4096, 0);
    std::vector<std::uint16_t> b(static_cast<std::size_t>(kProbeN) * kProbeK + 4096, 0);

    fn(out.data(), a.data(), b.data(), /*M=*/1, kProbeN, kProbeK,
       /*a_stride=*/kProbeK, /*b_stride=*/kProbeK, /*out_stride=*/kProbeN,
       /*px=*/0, /*py=*/0, /*pz=*/0, /*gx=*/1, /*gy=*/1, /*gz=*/1);

    std::int32_t n = 0;
    while (n < kProbeN && out[n] != kSentinel) ++n;

    // Every column written ⇒ the kernel walks N itself rather than taking a column
    // block per cell. Reporting kProbeN would make the caller emit one cell, which is
    // correct for that style, but we have never built matmul that way and silently
    // adopting it would hide a build mistake. Refuse instead.
    if (n <= 0 || n >= kProbeN) {
      std::fprintf(stderr,
                   "[triton-cpu] ⚠️ matmul BN probe returned %d of %d columns — "
                   "expected a strict column block. Falling back to %d.\n",
                   n, kProbeN, kMmBNDefault);
      return kMmBNDefault;
    }

    // The manifest is now a cross-check, not the source of truth. A disagreement
    // means the build script and the .so have drifted apart, which is worth saying
    // out loud even though the measured value is the one we use.
    const std::string declared = ManifestValue("mm_bn");
    if (!declared.empty()) {
      const long d = std::strtol(declared.c_str(), nullptr, 10);
      if (d != n)
        std::fprintf(stderr,
                     "[triton-cpu] ⚠️ manifest says mm_bn=%ld but the artifact "
                     "measures %d — using the measured value.\n", d, n);
    }
    std::fprintf(stderr, "[triton-cpu] matmul BN probed from artifact: %d\n", n);
    return n;
  }();
  return v;
}

using ReshapeAndCacheKernelFn = void (*)(
    const void* k, const void* v, void* kc, void* vc, const void* slots,
    std::int32_t n_elems, std::int32_t block_size, std::int32_t k_tok_stride,
    std::int32_t v_tok_stride, std::int32_t k_block_stride,
    std::int32_t k_page_stride, std::int32_t v_block_stride,
    std::int32_t v_page_stride, std::int32_t px, std::int32_t py, std::int32_t pz,
    std::int32_t gx, std::int32_t gy, std::int32_t gz);

constexpr std::int32_t kRcElems = 1024;  // num_kv_heads(8) × head_size(128)

ReshapeAndCacheKernelFn RcKernel() {
  static auto fn = reinterpret_cast<ReshapeAndCacheKernelFn>(
      LoadKernelSymbol("reshape_and_cache.so", "reshape_and_cache_kernel"));
  return fn;
}

using EmbeddingKernelFn = void (*)(void* out, const void* table, const void* ids,
                                   std::int32_t h, std::int32_t table_stride,
                                   std::int32_t out_stride, std::int32_t px,
                                   std::int32_t py, std::int32_t pz, std::int32_t gx,
                                   std::int32_t gy, std::int32_t gz);

constexpr std::int32_t kEmbH = 1024;  // hidden_size

EmbeddingKernelFn EmbKernel() {
  static auto fn = reinterpret_cast<EmbeddingKernelFn>(
      LoadKernelSymbol("embedding.so", "embedding_kernel"));
  return fn;
}

using GreedyArgmaxKernelFn = void (*)(void* out, const void* logits,
                                      std::int32_t v, std::int32_t row_stride,
                                      std::int32_t px, std::int32_t py,
                                      std::int32_t pz, std::int32_t gx,
                                      std::int32_t gy, std::int32_t gz);

GreedyArgmaxKernelFn ArgmaxKernel() {
  static auto fn = reinterpret_cast<GreedyArgmaxKernelFn>(
      LoadKernelSymbol("greedy_argmax.so", "greedy_argmax_kernel"));
  return fn;
}

// ⭐ The destination of this arc. Note the arity: the Python kernel has 26 user
// parameters and the compiled artifact has 24 — `causal` and `bt_col` were both 1 at
// compile time, so Triton folded them into constants and DROPPED them from the
// signature. That is the most dangerous specialization there is: at run time a
// non-causal call would not merely compute the wrong thing, every argument after it
// would shift. The gate therefore REQUIRES causal and a unit block-table column
// stride — and the folded `causal` is a small mercy, because it turns "wrong answer"
// into "declined".
using PagedAttnKernelFn = void (*)(
    void* out, const void* q, const void* kc, const void* vc, const void* btab,
    const void* pos, const void* slen, const void* req, std::int32_t qpk,
    std::int32_t block_size, float scale, std::int32_t window_left,
    std::int32_t window_right, std::int32_t q_stride_t, std::int32_t q_stride_h,
    std::int32_t out_stride_t, std::int32_t out_stride_h, std::int32_t bt_row,
    std::int32_t kc_blk, std::int32_t kc_pg, std::int32_t kc_hd, std::int32_t vc_blk,
    std::int32_t vc_pg, std::int32_t vc_hd, std::int32_t px, std::int32_t py,
    std::int32_t pz, std::int32_t gx, std::int32_t gy, std::int32_t gz);

constexpr std::int32_t kPaD = 128;  // head_size the kernel was compiled for

// ⭐ The general form takes the head width at run time, so its signature has exactly
// one more argument — `D`, sitting where the folded constexpr used to be, right after
// the cache strides and before the grid. Verified off the compiled IR rather than
// assumed: the pinned kernel takes 30 parameters, this one takes 31.
//
// ⚠️ Two entry points, two symbol names, and no way to tell them apart from the
// pointer — so which one to load, and which signature to call it through, both come
// off the manifest. Calling the wrong one shifts every argument after `vc_hd`.
using PagedAttnGeneralFn = void (*)(
    void* out, const void* q, const void* kc, const void* vc, const void* btab,
    const void* pos, const void* slen, const void* req, std::int32_t qpk,
    std::int32_t block_size, float scale, std::int32_t window_left,
    std::int32_t window_right, std::int32_t q_stride_t, std::int32_t q_stride_h,
    std::int32_t out_stride_t, std::int32_t out_stride_h, std::int32_t bt_row,
    std::int32_t kc_blk, std::int32_t kc_pg, std::int32_t kc_hd, std::int32_t vc_blk,
    std::int32_t vc_pg, std::int32_t vc_hd,
    std::int32_t d,                                     // ★ the added argument
    std::int32_t px, std::int32_t py, std::int32_t pz,
    std::int32_t gx, std::int32_t gy, std::int32_t gz);

PagedAttnKernelFn PaKernel() {
  static auto fn = ArtifactIsLoopStyle()
                       ? nullptr
                       : reinterpret_cast<PagedAttnKernelFn>(LoadKernelSymbol(
                             "paged_attn.so", "paged_attn_kernel"));
  return fn;
}
PagedAttnGeneralFn PaKernelGeneral() {
  static auto fn = ArtifactIsLoopStyle()
                       ? reinterpret_cast<PagedAttnGeneralFn>(LoadKernelSymbol(
                             "paged_attn.so", "paged_attn_general_kernel"))
                       : nullptr;
  return fn;
}

bool Aligned(const void* p, std::uintptr_t n) {
  return p != nullptr && (reinterpret_cast<std::uintptr_t>(p) % n) == 0;
}

// Everything the compiled kernel assumed is re-checked here, because none of it is
// checked anywhere else: dtype, unit innermost stride, row strides, and pointer
// alignment.
//
// ⭐ THE SIZE USED TO BE ONE OF THOSE ASSUMPTIONS, AND IT NO LONGER IS. This gate
// once required `d == kSiluD` (3072). That looked like it was protecting the kernel,
// but the kernel never needed it: it is written in the blocked form and builds its
// own mask from the runtime `d`. What actually required 3072 was the LAUNCHER below,
// which fed the constant in place of the tensor's own extent. Fixing that turned the
// size from an assumption into an ordinary argument.
//
// ⚠️ What survives is the ONE assumption that is real: Triton marked `d` with
// `tt.divisibility = 16` because it was compiled at a value divisible by 16. That is
// a promise the generated code may rely on, so it is still checked — and it is a
// property of the number, not a specific number, which is the whole difference.
//
// ⚠️ And this relaxation is NOT transferable to the other ops. RmsNorm's kernel uses
// `cols = arange(0, BLOCK)` with no second grid dimension, so at n = 5120 its mask is
// all-true over the first 1024 elements and its `tl.sum` silently reduces over a
// prefix. There the strict size check is the only thing standing between us and a
// wrong answer. Relaxing a gate is safe exactly when the kernel was already general.
bool SiluGateOk(const Tensor& out, const Tensor& x) {
  if (out.dtype != DType::kBF16 || x.dtype != DType::kBF16) return false;
  if (out.rank != 2 || x.rank != 2) return false;
  if (out.shape[0] != x.shape[0] || out.shape[0] <= 0) return false;
  const std::int64_t d = out.shape[1];
  if (d <= 0 || d > INT32_MAX) return false;
  if (x.shape[1] != 2 * d) return false;
  if (d % 16 != 0) return false;                        // tt.divisibility, see above
  if (out.stride[1] != 1 || x.stride[1] != 1) return false;
  if (out.stride[0] != d || x.stride[0] != 2 * d) return false;
  return Aligned(out.data, 64) && Aligned(x.data, 64);
}

// ⭐ RmsNorm is where the alignment story stops being theoretical. `w` is a WEIGHT:
// it comes from the loader (mmap, possibly repacked), not from the StepArena, and
// the probe measured it at 8-BYTE alignment while every other tensor here is 64.
// A gate written once for the whole provider would reject this op forever.
//
// 8 bytes is safe here, and that is a MEASURED claim, not an assumption: the
// generated x86 code contains ZERO aligned vector accesses through a caller-supplied
// pointer (every `vmovaps`/`vmovdqa64` in it is either a stack spill or a
// register-to-register move). build_kernels.py asserts this on every rebuild, so a
// toolchain change that starts emitting aligned loads fails the build instead of
// faulting inside the engine. ⚠️ It must be re-verified per target — unaligned
// vector access is implementation-defined on RISC-V.
bool RmsGateOk(const Tensor& out, const Tensor& x, const Tensor& w,
               const RmsNormArgs& args, const Tensor* residual) {
  if (args.gemma) return false;  // the (1 + w) variant was not compiled
  // ⭐ THE SIZE CHECK IS NOW CONDITIONAL ON WHAT THE ARTIFACT CLAIMS. The pinned
  // kernel walks the row with a single `arange(0, BLOCK)`, so at n > BLOCK its mask
  // is all-true over a prefix and the `tl.sum` quietly reduces over part of the row —
  // wrong, not slow, and invisible. The loop-style kernel carries the accumulator
  // across blocks itself, so any n is fine. Same gate, two contracts, and which one
  // applies is read off the artifact rather than assumed.
  const std::int64_t n = out.shape[1];
  const bool size_ok = GatesRelaxed() ? (n > 0 && n % 16 == 0) : (n == kRmsN);
  // ⚠️ Diagnostic, one line per distinct newly-admitted size. Relaxing this gate made
  // the shadow audit jump from 0.02% differing to 59%, and guessing which call shape
  // was responsible wasted a round — so the gate reports what it let through.
  if (ProbeEnabled() && size_ok && n != kRmsN) {
    static std::mutex m;
    static std::set<std::int64_t> seen;
    std::lock_guard<std::mutex> lk(m);
    if (seen.insert(n).second)
      std::fprintf(stderr,
                   "[triton-cpu] RmsNorm gate ADMITTED n=%lld  (rows=%lld "
                   "x.stride0=%lld out.stride0=%lld w.shape0=%lld resid=%s "
                   "★ in_place=%s)\n",
                   (long long)n, (long long)out.shape[0],
                   (long long)x.stride[0], (long long)out.stride[0],
                   (long long)w.shape[0], residual ? "yes" : "no",
                   out.data == x.data ? "YES" : "no");
  }
  const bool base =
      out.dtype == DType::kBF16 && x.dtype == DType::kBF16 &&
      w.dtype == DType::kBF16 && out.rank == 2 && x.rank == 2 && w.rank == 1 &&
      out.shape[0] == x.shape[0] && out.shape[0] > 0 &&
      size_ok && x.shape[1] == n && w.shape[0] == n && n <= INT32_MAX &&
      out.stride[1] == 1 && x.stride[1] == 1 && w.stride[0] == 1 &&
      out.stride[0] == n && x.stride[0] == n &&
      Aligned(out.data, 64) && Aligned(x.data, 64) &&
      Aligned(w.data, 2);  // element alignment only — see the comment above
  if (!base) return false;
  if (residual == nullptr) return true;
  return residual->dtype == DType::kBF16 && residual->rank == 2 &&
         residual->shape[0] == x.shape[0] && residual->shape[1] == n &&
         residual->stride[1] == 1 && residual->stride[0] == n &&
         Aligned(residual->data, 64);
}

// Selection is proven by VT_OP_PROVIDER_STATS, but that proves the provider was
// CHOSEN, not that this gate accepted. These counters are the difference — without
// them a fully-declining build and a fully-accepting one look identical from
// outside. Reported under the probe env.
struct OpCounters {
  const char* name;
  std::atomic<std::uint64_t> triton{0};
  std::atomic<std::uint64_t> declined{0};
  // ⭐ The second thermometer. End-to-end tok/s is the number that counts, but one op
  // out of seven may not move it above run-to-run noise — and "below noise" read off a
  // single instrument is indistinguishable from "no effect". This measures the op
  // directly so the two can disagree, which is the only way to notice that they do.
  //
  // ⚠️ It is WALL time around the whole ParallelForRows, so on a multi-core run it
  // measures the span, not the CPU work — deliberately, because the span is what the
  // caller waits for. Only armed under the probe env; zero cost otherwise.
  std::atomic<std::uint64_t> ns{0};
  // ⭐ THE COMPARISON WE HAD NEVER MADE. Everything measured so far has been one of
  // our variants against another of our variants — never against the thing we are
  // supposedly improving on. That reference is not hypothetical and not far away:
  // this tree's CPU path is hand-written C++ with AVX-512, AVX2 and F16C variants
  // selected at run time (src/vt/cpu/cpu_matmul_elem_avx512.cpp, cpu_isa_x86.cpp).
  //
  // And the shadow audit ALREADY runs it, on the same inputs, immediately after ours,
  // in the same process — so timing it costs nothing beyond two clock reads and gives
  // a paired comparison with the thermal state held as close to constant as it gets.
  //
  // ⚠️ Only populated under shadow, and shadow makes both sides slower (snapshots,
  // restores, an extra full execution). Read the RATIO, never the absolute numbers,
  // and never compare these against the non-shadow timings above.
  std::atomic<std::uint64_t> native_ns{0};
  std::atomic<std::uint64_t> native_calls{0};
};

// Times the reference implementation inside the shadow path. Separate from
// ScopedOpTimer because it wraps one call rather than a function body.
class NativeTimer {
 public:
  explicit NativeTimer(OpCounters& c)
      : m_c(c), m_t0(std::chrono::steady_clock::now()) {}
  ~NativeTimer() {
    const auto dt = std::chrono::steady_clock::now() - m_t0;
    m_c.native_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count()),
        std::memory_order_relaxed);
    m_c.native_calls.fetch_add(1, std::memory_order_relaxed);
  }
  NativeTimer(const NativeTimer&) = delete;
  NativeTimer& operator=(const NativeTimer&) = delete;

 private:
  OpCounters& m_c;
  std::chrono::steady_clock::time_point m_t0;
};

// RAII so an early return cannot skip the stop. Reads the clock twice per call, which
// is why it is gated: at ~1800 accepted calls per run that is noise, but the gate keeps
// it honest for anyone who turns the provider on without the probe.
class ScopedOpTimer {
 public:
  explicit ScopedOpTimer(OpCounters& c, bool on)
      : m_c(c), m_on(on),
        m_t0(on ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{}) {}
  // ⚠️ Must be called before the shadow block in ops that have one. The destructor
  // fires at end of function, and under shadow that function body continues on into
  // snapshotting, restoring inputs and running the reference — charging all of it to
  // us. A Triton-vs-native ratio built on that would be measuring the harness.
  void stop() {
    if (!m_on || m_stopped) return;
    m_stopped = true;
    const auto dt = std::chrono::steady_clock::now() - m_t0;
    m_c.ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count()),
        std::memory_order_relaxed);
  }
  ~ScopedOpTimer() { stop(); }
  ScopedOpTimer(const ScopedOpTimer&) = delete;
  ScopedOpTimer& operator=(const ScopedOpTimer&) = delete;

 private:
  OpCounters& m_c;
  bool m_on;
  bool m_stopped = false;
  std::chrono::steady_clock::time_point m_t0;
};
// ⭐ MatmulBT's gate carries two things the earlier ops did not.
//
// (1) DIVISIBILITY. The emitted binary was compiled with N=1024 and K=3072, and the
//     Triton IR marks BOTH `tt.divisibility = 16`. That is a promise the kernel is
//     allowed to rely on, so a shape that breaks it is a silent wrong answer. Every
//     real shape in this model satisfies it (N in 1024/2048/3072/4096/6144/151936,
//     K in 1024/2048/3072) but the gate checks rather than trusts.
//     ⚠️ Found only after re-reading the IR: my first grep for the specialization
//     list used a lowercase-only pattern and silently hid `N` and `K`.
//
// (2) REPACKED WEIGHTS. `b` is a weight, and this tree rewrites weight BYTES at load
//     in three different ways while leaving the SHAPE alone (`repacked` for the CPU
//     i8mm interleave, `q8_0_aligned` for the CUDA layout, `elem_kn_repacked` for a
//     [N,K]-shaped tensor whose bytes are [K,N]). A kernel that reads the shape and
//     ignores those flags reads the right extents out of the wrong bytes.
bool MatmulBTGateOk(const Tensor& out, const Tensor& a, const Tensor& b) {
  if (a.repacked || a.q8_0_aligned || a.elem_kn_repacked) return false;
  if (b.repacked || b.q8_0_aligned || b.elem_kn_repacked) return false;
  if (out.dtype != DType::kBF16 || a.dtype != DType::kBF16 ||
      b.dtype != DType::kBF16)
    return false;
  if (out.rank != 2 || a.rank != 2 || b.rank != 2) return false;
  const std::int64_t M = a.shape[0], K = a.shape[1];
  const std::int64_t N = b.shape[0];
  if (M <= 0 || N <= 0 || K <= 0) return false;
  if (out.shape[0] != M || out.shape[1] != N || b.shape[1] != K) return false;
  if (N % 16 != 0 || K % 16 != 0) return false;             // see (1)
  if (M > INT32_MAX || N > INT32_MAX || K > INT32_MAX) return false;
  if (out.stride[1] != 1 || a.stride[1] != 1 || b.stride[1] != 1) return false;
  if (a.stride[0] != K || b.stride[0] != K || out.stride[0] != N) return false;
  // out/a come from the StepArena (64 B); `b` is a weight off the loader, which the
  // probe measured at 8 B for RmsNorm's — element alignment only, justified by the
  // same build-time assertion that no aligned access goes through a caller pointer.
  return Aligned(out.data, 64) && Aligned(a.data, 64) && Aligned(b.data, 2);
}

// ⭐ The first op here that touches the PAGED KV cache — and deliberately one that
// only WRITES. It separates "can a Triton kernel address a paged layout" from "can a
// Triton kernel do attention", so a failure is attributable. Same reason SiluAndMul
// came before RmsNorm.
//
// ⚠️ Two things this gate must not get wrong.
// (1) The destination strides come from the TENSORS, never from the shape. Their own
//     comment states why: one allocation is (num_blocks, 2, block_size, H, D) and
//     K/V are its two dim-1 unbind slices, so the block stride is 2*bs*H*D. Deriving
//     it from the shape gives half, which is correct for the first block and wrong
//     for every one after — an error that passes a short smoke test.
// (2) ALL THIRTEEN user arguments are marked `tt.divisibility = 16`, including the
//     five pointers. The KV cache pointers measured exactly 16-byte aligned (they
//     come from a std::vector, not the 64-byte arena), so that assumption is
//     satisfied with ZERO margin — hence the explicit check rather than a comment.
bool RcGateOk(const Tensor& k, const Tensor& v, const Tensor& kc, const Tensor& vc,
              const Tensor& slots) {
  if (k.dtype != DType::kBF16 || v.dtype != DType::kBF16 ||
      kc.dtype != DType::kBF16 || vc.dtype != DType::kBF16)
    return false;
  if (slots.dtype != DType::kI64 || slots.rank != 1) return false;
  if (k.rank != 3 || v.rank != 3 || kc.rank != 4 || vc.rank != 4) return false;
  const std::int64_t nh = kc.shape[2], hd = kc.shape[3];
  // Same split as RmsNorm: pinned artifact means the compiled BLOCK has to cover the
  // whole per-token run; the loop-style one walks it.
  if (GatesRelaxed() ? (nh * hd <= 0 || (nh * hd) % 16 != 0)
                          : (nh * hd != kRcElems))
    return false;
  if (k.shape[1] != nh || k.shape[2] != hd) return false;
  if (v.shape[1] != nh || v.shape[2] != hd) return false;
  if (k.shape[0] != slots.shape[0] || v.shape[0] != slots.shape[0]) return false;
  // The reference's fast path assumes each token's page is one dense run.
  if (kc.stride[3] != 1 || vc.stride[3] != 1) return false;
  if (kc.stride[2] != hd || vc.stride[2] != hd) return false;
  if (k.stride[2] != 1 || v.stride[2] != 1) return false;
  if (k.stride[1] != hd || v.stride[1] != hd) return false;
  const std::int64_t vals[] = {kc.shape[1],  k.stride[0],  v.stride[0],
                               kc.stride[0], kc.stride[1], vc.stride[0],
                               vc.stride[1]};
  for (std::int64_t x : vals) {
    if (x % 16 != 0 || x > INT32_MAX) return false;       // see (2)
  }
  return Aligned(k.data, 16) && Aligned(v.data, 16) && Aligned(kc.data, 16) &&
         Aligned(vc.data, 16) && Aligned(slots.data, 16);
}

// ⭐ Embedding is the first GATHER: the offset is READ rather than computed, which is
// the same move paged attention makes when it reads the block table. Getting it right
// somewhere simple first is the point of doing this op before that one.
//
// ⚠️ The reference does `VT_CHECK(id >= 0 && id < vocab)`. A Triton kernel cannot
// throw, so that precondition has nowhere to live except here — the recurring theme
// of this whole exercise, that a contract the kernel cannot express moves to the
// caller. Checking `tokens` ids is trivially cheap next to the gather itself.
//
// The table pointer measured 8-byte aligned (it is a weight, off the loader, not the
// 64-byte arena). Rather than argue that Triton's divisibility claim is harmless
// here, the kernel is compiled with `do_not_specialize=["table_ptr"]` so the claim is
// never made — the artifact's assumptions and the runtime's facts now agree.
bool EmbGateOk(const Tensor& out, const Tensor& table, const Tensor& ids) {
  if (out.dtype != DType::kBF16 || table.dtype != DType::kBF16) return false;
  if (ids.dtype != DType::kI32 || ids.rank != 1) return false;
  if (out.rank != 2 || table.rank != 2) return false;
  const std::int64_t h = table.shape[1];
  if (out.shape[1] != h) return false;
  // Same split as RmsNorm — see GatesRelaxed().
  if (GatesRelaxed() ? (h <= 0 || h % 16 != 0 || h > INT32_MAX)
                          : (h != kEmbH))
    return false;
  if (out.shape[0] != ids.shape[0] || ids.shape[0] <= 0) return false;
  if (out.stride[1] != 1 || table.stride[1] != 1 || ids.stride[0] != 1) return false;
  if (out.stride[0] != h || table.stride[0] != h) return false;
  if (h % 16 != 0) return false;
  const std::int64_t vocab = table.shape[0];
  if (vocab <= 0 || vocab * h > INT32_MAX) return false;  // idx*stride stays in range
  const auto* p = static_cast<const std::int32_t*>(ids.data);
  for (std::int64_t i = 0; i < ids.shape[0]; ++i) {
    if (p[i] < 0 || p[i] >= vocab) return false;          // the reference's VT_CHECK
  }
  return Aligned(out.data, 64) && Aligned(ids.data, 16) && Aligned(table.data, 2);
}

// ⭐ The first kernel here whose BLOCK is a TILING parameter rather than a capacity
// limit. The vocabulary is 151,936 f32 — 608 KB, far past anything a single tile
// holds — so the kernel loops over blocks carrying a running (max, index) pair.
// Consequently the gate does not have to pin the size: any vocabulary works. That is
// the shape-general form the earlier kernels are not, and it is the same structure
// online softmax needs, which is why this op comes before paged attention.
//
// ⚠️ Tie-breaking is load-bearing. The reference uses a strict `>` so the FIRST
// (lowest-index) maximum wins, and states that is bit-exact against torch.argmax. So
// the kernel picks the lowest index within a block AND merges across blocks with a
// strict `>` — a later block that merely equals the running max must not replace it.
// Verified at build time with a deliberate tie at indices 100 and 90,000, which land
// in different blocks.
//
// f32 in, i64 out — also the first op here that changes dtype across the boundary.
bool ArgmaxGateOk(const Tensor& out, const Tensor& logits) {
  if (out.dtype != DType::kI64 || logits.dtype != DType::kF32) return false;
  if (out.rank != 1 || logits.rank != 2) return false;
  const std::int64_t n = logits.shape[0], v = logits.shape[1];
  if (n <= 0 || v <= 0 || out.shape[0] != n) return false;
  if (v % 16 != 0 || v > INT32_MAX) return false;
  if (out.stride[0] != 1 || logits.stride[1] != 1) return false;
  if (logits.stride[0] != v) return false;
  return Aligned(out.data, 16) && Aligned(logits.data, 16);
}

// Everything the kernel cannot express, checked here. Beyond the folded `causal` and
// `bt_col` above: no soft-cap (not compiled), no fp8 KV (a different element path),
// head_size fixed at what the kernel saw, and a GQA ratio that divides evenly.
bool PaGateOk(const Tensor& out, const Tensor& query, const Tensor& kc,
              const Tensor& vc, const Tensor& bt, const Tensor& sl, const Tensor& qs,
              const PagedAttentionArgs& args) {
  if (!args.causal) return false;                      // folded to a constant
  if (args.logits_soft_cap > 0.0f) return false;       // not compiled
  if (args.kv_cache_dtype != vt::Fp8KVCacheDataType::kAuto) return false;
  if (out.dtype != DType::kBF16 || query.dtype != DType::kBF16 ||
      kc.dtype != DType::kBF16 || vc.dtype != DType::kBF16)
    return false;
  if (bt.dtype != DType::kI32 || sl.dtype != DType::kI32 || qs.dtype != DType::kI32)
    return false;
  if (out.rank != 3 || query.rank != 3 || kc.rank != 4 || vc.rank != 4) return false;
  if (bt.rank != 2 || sl.rank != 1 || qs.rank != 1) return false;
  const std::int64_t hq = query.shape[1], d = query.shape[2];
  const std::int64_t hkv = kc.shape[2];
  // ⭐ The last of the seven. Same split as the others — but note the asymmetry
  // inside this one op: the SEQUENCE dimension was never pinned, because it grows by
  // one every step and a kernel that fixed it could not finish its own model. Only
  // the head width was, because that stops changing once a model is loaded.
  //
  // ⚠️ The relaxed bound is not `% 16` here. `tl.arange` needs a power of two, so the
  // head width has to be one — and real ones that are not (80, 96) are covered by the
  // next power up plus masking, which the kernel already does.
  if (GatesRelaxed() ? (d <= 0 || d > (1 << 16) || (d & (d - 1)) != 0)
                     : (d != kPaD))
    return false;
  if (kc.shape[3] != d || vc.shape[3] != d) return false;
  if (hkv <= 0 || hq % hkv != 0) return false;         // GQA ratio must divide
  if (out.shape[0] != query.shape[0] || out.shape[1] != hq || out.shape[2] != d)
    return false;
  if (bt.stride[1] != 1) return false;                 // folded to a constant
  if (out.stride[2] != 1 || query.stride[2] != 1) return false;
  if (kc.stride[3] != 1 || vc.stride[3] != 1) return false;
  const std::int64_t vals[] = {query.stride[0], query.stride[1], out.stride[0],
                               out.stride[1],   bt.stride[0],    kc.stride[0],
                               kc.stride[1],    kc.stride[2],    vc.stride[0],
                               vc.stride[1],    vc.stride[2],    kc.shape[1]};
  for (std::int64_t x : vals) {
    if (x > INT32_MAX) return false;
  }
  return Aligned(out.data, 16) && Aligned(query.data, 16) && Aligned(kc.data, 16) &&
         Aligned(vc.data, 16) && Aligned(bt.data, 16) && Aligned(sl.data, 16) &&
         Aligned(qs.data, 16);
}

OpCounters g_silu{"SiluAndMul"};
OpCounters g_rms{"RmsNorm"};
OpCounters g_mm{"MatmulBT"};
OpCounters g_rc{"ReshapeAndCache"};
OpCounters g_emb{"Embedding"};
OpCounters g_ga{"GreedyArgmax"};
OpCounters g_pa{"PagedAttention"};

void NoteAccept(OpCounters& c) {
  if (c.triton.fetch_add(1, std::memory_order_relaxed) == 0) {
    std::fprintf(stderr, "[triton-cpu] %s: RUNNING the Triton kernel\n", c.name);
  }
}

// ---------------------------------------------------------------------------
// SHADOW MODE (VLLM_CPP_TRITON_CPU_SHADOW=1) — the faithfulness audit.
//
// WHY. Swapping in the Triton RmsNorm changes the generated text from token 6 on.
// The available story is that its vector reduction differs in ORDER from
// vt-native's deliberately-sequential one (their comment: "each row's f32 variance
// reduction stays SEQUENTIAL on one thread — bit-identical"), so the two answers
// differ by around one bf16 ULP and greedy decoding amplifies that at a near-tie.
//
// That story is plausible, deterministic and consistent with the divergence point.
// It is also exactly the kind of story this project has been wrong about before, so
// it does not get believed until measured — and the build-time check does NOT
// measure it: it compares against torch on random data, not against the kernel it
// actually replaced on the activations that actually occur.
//
// WHAT THIS DOES. Runs BOTH kernels on the same inputs and reports the difference,
// while leaving vt-native's result in place — so a shadow run's output is
// byte-identical to a run without the provider, and the measurement costs nothing
// but time. For the residual variant the residual is snapshotted and restored
// between the two, because vt-native mutates it in place and a double-apply would
// measure a bug we invented rather than the one we are looking for.
bool ShadowEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VLLM_CPP_TRITON_CPU_SHADOW");
    return e != nullptr && std::strcmp(e, "0") != 0 && e[0] != '\0';
  }();
  return on;
}

// bf16 is the high half of an f32, so widening is a shift — exact, no library.
float Bf16ToF32(std::uint16_t bits) {
  const std::uint32_t w = static_cast<std::uint32_t>(bits) << 16;
  float f;
  std::memcpy(&f, &w, sizeof(f));
  return f;
}

struct ShadowStats {
  explicit ShadowStats(const char* n) : name(n) {}
  const char* name;
  std::atomic<std::uint64_t> calls{0};
  std::atomic<std::uint64_t> elems{0};
  std::atomic<std::uint64_t> mismatched{0};   // differing bf16 bit patterns
  std::atomic<std::uint64_t> max_ulp{0};      // |bit distance|, bf16 ULPs
  double max_abs = 0.0;                       // guarded by the mutex below
  std::mutex mu;
};
ShadowStats g_shadow_rms{"RmsNorm"};
ShadowStats g_shadow_silu{"SiluAndMul"};
ShadowStats g_shadow_mm{"MatmulBT"};
ShadowStats g_shadow_pa{"PagedAttention"};

void ShadowCompare(ShadowStats& s, const void* a, const void* b, std::size_t n) {
  const auto* pa = static_cast<const std::uint16_t*>(a);
  const auto* pb = static_cast<const std::uint16_t*>(b);
  std::uint64_t mism = 0, ulp = 0;
  double worst = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (pa[i] == pb[i]) continue;
    ++mism;
    const std::uint64_t d = pa[i] > pb[i] ? pa[i] - pb[i] : pb[i] - pa[i];
    if (d > ulp) ulp = d;
    const double diff = std::fabs(static_cast<double>(Bf16ToF32(pa[i])) -
                                  static_cast<double>(Bf16ToF32(pb[i])));
    if (diff > worst) worst = diff;
  }
  s.calls.fetch_add(1, std::memory_order_relaxed);
  s.elems.fetch_add(n, std::memory_order_relaxed);
  s.mismatched.fetch_add(mism, std::memory_order_relaxed);
  std::uint64_t prev = s.max_ulp.load(std::memory_order_relaxed);
  while (ulp > prev &&
         !s.max_ulp.compare_exchange_weak(prev, ulp, std::memory_order_relaxed)) {}
  std::lock_guard<std::mutex> lk(s.mu);
  if (worst > s.max_abs) s.max_abs = worst;
}

void ShadowReport(ShadowStats& s) {
  const std::uint64_t e = s.elems.load(), m = s.mismatched.load();
  if (e == 0) return;
  std::fprintf(stderr,
               "[triton-cpu shadow] %s: calls=%llu elems=%llu differing=%llu "
               "(%.4f%%) max_bf16_ulp=%llu max_abs=%.3e\n",
               s.name, (unsigned long long)s.calls.load(),
               (unsigned long long)e, (unsigned long long)m,
               100.0 * static_cast<double>(m) / static_cast<double>(e),
               (unsigned long long)s.max_ulp.load(), s.max_abs);
}

// ---------------------------------------------------------------------------
// THREAD WITNESS (VLLM_CPP_TRITON_CPU_PROBE=1) — T-21 §2.
//
// The counters below are written with atomics, which is correct whether or not the
// engine dispatches an op concurrently. But "correct under concurrency" and
// "concurrency actually happens" are different claims, and only the first was ever
// established. This records which threads reach each op so the second stops being an
// assumption. If dispatch turns out to be single-threaded, the atomics on a path
// taken thousands of times per decode step are pure overhead — worth knowing.
//
// ⚠️ Deliberately LOCK-FREE. A mutex here would be simpler, but a mutex is also a
// serializer: it could suppress the very concurrency it was added to detect. Fixed
// slots plus compare-exchange keeps the observation from changing the thing observed.
constexpr int kWitnessSlots = 8;

struct ThreadWitness {
  const char* name;
  std::atomic<std::uint64_t> slot[kWitnessSlots];
  std::atomic<std::uint64_t> overflow{0};   // more distinct threads than slots
  std::atomic<std::uint64_t> calls{0};
  explicit ThreadWitness(const char* n) : name(n) {
    for (auto& s : slot) s.store(0, std::memory_order_relaxed);
  }
};

void WitnessThread(ThreadWitness& w) {
  w.calls.fetch_add(1, std::memory_order_relaxed);
  // +1 so a hash of 0 never collides with the "empty" marker.
  const std::uint64_t id =
      static_cast<std::uint64_t>(
          std::hash<std::thread::id>{}(std::this_thread::get_id())) |
      1ull;
  for (auto& s : w.slot) {
    std::uint64_t cur = s.load(std::memory_order_relaxed);
    if (cur == id) return;
    if (cur == 0 &&
        s.compare_exchange_strong(cur, id, std::memory_order_relaxed)) {
      return;
    }
    if (cur == id) return;  // lost the race to an equal id
  }
  w.overflow.fetch_add(1, std::memory_order_relaxed);
}

ThreadWitness g_w_silu{"SiluAndMul"};
ThreadWitness g_w_rms{"RmsNorm"};
ThreadWitness g_w_mm{"MatmulBT"};
ThreadWitness g_w_rc{"ReshapeAndCache"};
ThreadWitness g_w_emb{"Embedding"};
ThreadWitness g_w_ga{"GreedyArgmax"};
ThreadWitness g_w_pa{"PagedAttention"};

// The INNER witness: one shared record for whatever runs inside ParallelForRows.
// Dispatch concurrency and kernel concurrency are different questions and this keeps
// them apart — the pool is expected to show several threads by construction, so it
// doubles as a control that the witness itself works.
ThreadWitness g_w_inner{"(inner: SiluAndMul)"};
ThreadWitness g_w_inner_pa{"(inner: PagedAttention)"};

struct WitnessDump {
  ~WitnessDump() {
    if (!ProbeEnabled()) return;
    for (const ThreadWitness* w :
         {&g_w_silu, &g_w_rms, &g_w_mm, &g_w_rc, &g_w_emb, &g_w_ga, &g_w_pa,
          &g_w_inner, &g_w_inner_pa}) {
      int n = 0;
      for (const auto& s : w->slot) {
        if (s.load(std::memory_order_relaxed) != 0) ++n;
      }
      if (w->calls.load() == 0) continue;
      std::fprintf(stderr,
                   "[triton-cpu threads] %-26s calls=%llu distinct_threads=%d%s\n",
                   w->name, (unsigned long long)w->calls.load(), n,
                   w->overflow.load() ? " (+overflow)" : "");
    }
  }
} witness_dump;

struct CounterDump {
  ~CounterDump() {
    if (ShadowEnabled()) {
      ShadowReport(g_shadow_silu);
      ShadowReport(g_shadow_rms);
      ShadowReport(g_shadow_mm);
      ShadowReport(g_shadow_pa);
    }
    if (!ProbeEnabled()) return;
    for (const OpCounters* c : {&g_silu, &g_rms, &g_mm, &g_rc, &g_emb, &g_ga, &g_pa}) {
      const std::uint64_t n = c->triton.load(), ns = c->ns.load();
      const std::uint64_t nn = c->native_calls.load(), nns = c->native_ns.load();
      std::fprintf(stderr,
                   "[triton-cpu] %-16s triton=%-6llu declined=%-6llu "
                   "total=%8.3f ms  per_call=%8.3f us",
                   c->name, (unsigned long long)n,
                   (unsigned long long)c->declined.load(), ns / 1e6,
                   n ? ns / 1e3 / static_cast<double>(n) : 0.0);
      // ★ Only under shadow. The ratio is the point; the absolutes are inflated on
      // both sides by the audit and must not be compared with the numbers above.
      if (nn > 0) {
        const double ours = n ? ns / static_cast<double>(n) : 0.0;
        const double theirs = nns / static_cast<double>(nn);
        std::fprintf(stderr,
                     "  |  native=%8.3f ms per_call=%8.3f us  "
                     "★ ours/native=%.2fx",
                     nns / 1e6, theirs / 1e3, theirs > 0 ? ours / theirs : 0.0);
      }
      std::fprintf(stderr, "\n");
    }
  }
} counter_dump;

// Every kernel in the shell is the same three lines: resolve the provider below
// us ONCE, count the decline, forward. `NoteOpDecline` exists precisely so a
// shape-gated provider does not re-walk the provider stack on the hot path — a
// decode run declines tens of thousands of times.
#define VT_TRITON_CPU_DECLINE(FN_TYPE, OP_ID, ...)                          \
  do {                                                                       \
    static FN_TYPE next = reinterpret_cast<FN_TYPE>(                         \
        GetOpFallback(OP_ID, DeviceType::kCPU, kProviderName));              \
    NoteOpDecline(OP_ID, DeviceType::kCPU);                                  \
    next(__VA_ARGS__);                                                       \
  } while (0)

void Matmul(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  VT_TRITON_CPU_PROBE("Matmul", {{"out", &out}, {"a", &a}, {"b", &b}});
  VT_TRITON_CPU_DECLINE(MatmulFn, OpId::kMatmul, q, out, a, b);
}
void MatmulBT(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  if (ProbeEnabled()) WitnessThread(g_w_mm);
  VT_TRITON_CPU_PROBE("MatmulBT", {{"out", &out}, {"a", &a}, {"b", &b}});

  MatmulBTKernelFn kernel = MatmulBTKernel();
  if (kernel == nullptr || !MatmulBTGateOk(out, a, b)) {
    g_mm.declined.fetch_add(1, std::memory_order_relaxed);
    VT_TRITON_CPU_DECLINE(MatmulFn, OpId::kMatmulBT, q, out, a, b);
    return;
  }
  NoteAccept(g_mm);
  ScopedOpTimer timer(g_mm, ProbeEnabled());

  const auto M = static_cast<std::int32_t>(a.shape[0]);
  const auto N = static_cast<std::int32_t>(b.shape[0]);
  const auto K = static_cast<std::int32_t>(a.shape[1]);
  const std::int32_t bn = MmBlockWidth();
  const std::int32_t n_blocks = (N + bn - 1) / bn;
  void* out_p = out.data;
  const void* a_p = a.data;
  const void* b_p = b.data;

  std::vector<std::uint8_t> shadow_out;
  const auto n_elems = static_cast<std::size_t>(M) * static_cast<std::size_t>(N);

  // Parallelize over the FLATTENED (m, n-block) grid rather than over m alone.
  // At decode M is 1 while n_blocks is 2374 for lm_head, so splitting on m would
  // leave 15 of 16 threads idle on the single most expensive op in the step. This
  // is the same choice their BatchedMatmulKernel documents: "parallelized over the
  // flattened (batch, row) output space".
  vt::cpu::ParallelForRows(
      vt::cpu::CurrentThreadpool(), static_cast<std::int64_t>(M) * n_blocks,
      [&](std::int64_t r0, std::int64_t r1) {
        for (std::int64_t r = r0; r < r1; ++r) {
          const auto pm = static_cast<std::int32_t>(r / n_blocks);
          const auto pn = static_cast<std::int32_t>(r % n_blocks);
          kernel(out_p, a_p, b_p, M, N, K, K, K, N, pm, pn, 0, M, n_blocks, 1);
        }
      });

  timer.stop();  // ★ 停在 shadow 之前 —— 見 ScopedOpTimer::stop()
  if (!ShadowEnabled()) return;
  shadow_out.assign(static_cast<const std::uint8_t*>(out.data),
                    static_cast<const std::uint8_t*>(out.data) + n_elems * 2);
  static MatmulFn native = reinterpret_cast<MatmulFn>(
      GetOpFallback(OpId::kMatmulBT, DeviceType::kCPU, kProviderName));
  { NativeTimer nt(g_mm);
  native(q, out, a, b); }
  ShadowCompare(g_shadow_mm, shadow_out.data(), out.data, n_elems);
}
void RmsNorm(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
             const RmsNormArgs& args, Tensor* residual) {
  if (ProbeEnabled()) WitnessThread(g_w_rms);
  VT_TRITON_CPU_PROBE("RmsNorm",
                      {{"out", &out}, {"x", &x}, {"w", &w}, {"resid", residual}});

  RmsNormKernelFn plain = residual == nullptr ? RmsKernel() : nullptr;
  RmsNormResidKernelFn resid = residual == nullptr ? nullptr : RmsResidKernel();
  if ((plain == nullptr && resid == nullptr) ||
      !RmsGateOk(out, x, w, args, residual)) {
    g_rms.declined.fetch_add(1, std::memory_order_relaxed);
    VT_TRITON_CPU_DECLINE(RmsNormFn, OpId::kRmsNorm, q, out, x, w, args, residual);
    return;
  }
  NoteAccept(g_rms);
  ScopedOpTimer timer(g_rms, ProbeEnabled());

  // Shadow: snapshot what vt-native would need to see, so it can run afterwards on
  // untouched inputs and its answer — not ours — is what leaves this function.
  const std::int32_t n = static_cast<std::int32_t>(out.shape[1]);
  const std::size_t n_elems =
      static_cast<std::size_t>(out.shape[0]) * static_cast<std::size_t>(n);
  std::vector<std::uint8_t> shadow_out, shadow_resid, shadow_x;
  if (ShadowEnabled()) {
    if (residual != nullptr) {
      shadow_resid.assign(static_cast<const std::uint8_t*>(residual->data),
                          static_cast<const std::uint8_t*>(residual->data) +
                              n_elems * 2);
    }
    // ⭐ A BLIND SPOT IN OUR OWN AUDITOR, found by relaxing a gate. The whole method
    // is "keep our answer, restore the inputs, let the reference recompute, compare" —
    // and it quietly assumed `out` and `x` are different buffers. QK-norm calls this
    // op IN PLACE (measured: out.data == x.data, n=128). On those, our kernel had
    // already overwritten the input, so the reference ran on ALREADY-NORMALISED data
    // and its answer disagreed with ours by 59% of elements and a max_abs of 965 —
    // which reads exactly like a broken kernel and is not one.
    //
    // ⚠️ The audit could not see this before because the pinned gate declined every
    // in-place call, so the case never arose. The instrument was wrong the whole time;
    // widening what we accept is what exposed it.
    if (out.data == x.data) {
      shadow_x.assign(static_cast<const std::uint8_t*>(x.data),
                      static_cast<const std::uint8_t*>(x.data) + n_elems * 2);
    }
  }

  const std::int32_t rows = static_cast<std::int32_t>(out.shape[0]);
  const std::int32_t xs = static_cast<std::int32_t>(x.stride[0]);
  const std::int32_t os = static_cast<std::int32_t>(out.stride[0]);
  void* out_p = out.data;
  const void* x_p = x.data;
  const void* w_p = w.data;
  void* r_p = residual == nullptr ? nullptr : residual->data;
  const float eps = args.eps;

  // One program per ROW — the variance reduction spans the whole row, so unlike
  // SiluAndMul there is no column blocking to do.
  vt::cpu::ParallelForRows(
      vt::cpu::CurrentThreadpool(), rows, [&](std::int64_t r0, std::int64_t r1) {
        for (std::int64_t r = r0; r < r1; ++r) {
          const auto pid = static_cast<std::int32_t>(r);
          if (resid != nullptr) {
            resid(out_p, x_p, w_p, r_p, eps, n, xs, os, pid, 0, 0, rows, 1, 1);
          } else {
            plain(out_p, x_p, w_p, eps, n, xs, os, pid, 0, 0, rows, 1, 1);
          }
        }
      });

  timer.stop();  // ★ 停在 shadow 之前 —— 見 ScopedOpTimer::stop()
  if (!ShadowEnabled()) return;
  // Keep our answer, restore the inputs, let vt-native overwrite `out`, compare.
  shadow_out.assign(static_cast<const std::uint8_t*>(out.data),
                    static_cast<const std::uint8_t*>(out.data) + n_elems * 2);
  if (residual != nullptr) {
    std::memcpy(residual->data, shadow_resid.data(), shadow_resid.size());
  }
  // Restore the input the in-place call destroyed — see the snapshot above. Without
  // this the reference is handed our output and asked to treat it as its input.
  if (!shadow_x.empty()) {
    std::memcpy(const_cast<void*>(x.data), shadow_x.data(), shadow_x.size());
  }
  static RmsNormFn native = reinterpret_cast<RmsNormFn>(
      GetOpFallback(OpId::kRmsNorm, DeviceType::kCPU, kProviderName));
  { NativeTimer nt(g_rms);
  native(q, out, x, w, args, residual); }
  ShadowCompare(g_shadow_rms, shadow_out.data(), out.data, n_elems);
}
void SiluAndMul(Queue& q, Tensor& out, const Tensor& x) {
  if (ProbeEnabled()) WitnessThread(g_w_silu);
  VT_TRITON_CPU_PROBE("SiluAndMul", {{"out", &out}, {"x", &x}});

  SiluKernelFn kernel = SiluKernel();
  if (kernel == nullptr || !SiluGateOk(out, x)) {
    g_silu.declined.fetch_add(1, std::memory_order_relaxed);
    VT_TRITON_CPU_DECLINE(SiluAndMulFn, OpId::kSiluAndMul, q, out, x);
    return;
  }
  NoteAccept(g_silu);
  ScopedOpTimer timer(g_silu, ProbeEnabled());

  const std::int32_t rows = static_cast<std::int32_t>(out.shape[0]);
  // ⭐ Both of these used to be `kSiluD`. The kernel takes `d` at run time and masks
  // with it; feeding the constant made a general kernel behave like a pinned one.
  const std::int32_t d = static_cast<std::int32_t>(out.shape[1]);
  // SiluAndMul is the one op that exists in BOTH styles, so it is the one that has to
  // branch. `loop` artifacts walk the columns themselves — emitting more than one cell
  // would repeat the entire row's work per cell, which stays correct and wastes it.
  const std::int32_t col_blocks =
      ArtifactIsLoopStyle() ? 1 : (d + SiluBlockWidth() - 1) / SiluBlockWidth();
  void* out_p = out.data;
  const void* x_p = x.data;

  // The grid loop. It runs on THEIR threadpool, not a private one: vt-native's
  // SiluAndMul is `ForRows` over the same pool, and a Triton leg that was single
  // threaded would make every later A/B a comparison against a crippled control —
  // the failure mode this project already wrote down once (a knob that does not
  // move the variable is not a control).
  vt::cpu::ParallelForRows(
      vt::cpu::CurrentThreadpool(), rows, [&](std::int64_t r0, std::int64_t r1) {
        if (ProbeEnabled()) WitnessThread(g_w_inner);
        for (std::int64_t r = r0; r < r1; ++r) {
          for (std::int32_t cb = 0; cb < col_blocks; ++cb) {
            kernel(out_p, x_p, d, static_cast<std::int32_t>(x.stride[0]),
                   static_cast<std::int32_t>(out.stride[0]),
                   static_cast<std::int32_t>(r), cb, 0, rows, col_blocks, 1);
          }
        }
      });
}
void Embedding(Queue& q, Tensor& out, const Tensor& table, const Tensor& ids) {
  if (ProbeEnabled()) WitnessThread(g_w_emb);
  VT_TRITON_CPU_PROBE("Embedding", {{"out", &out}, {"table", &table}, {"ids", &ids}});
  EmbeddingKernelFn kernel = EmbKernel();
  if (kernel == nullptr || !EmbGateOk(out, table, ids)) {
    g_emb.declined.fetch_add(1, std::memory_order_relaxed);
    VT_TRITON_CPU_DECLINE(EmbeddingFn, OpId::kEmbedding, q, out, table, ids);
    return;
  }
  NoteAccept(g_emb);
  ScopedOpTimer timer(g_emb, ProbeEnabled());

  const auto rows = static_cast<std::int32_t>(ids.shape[0]);
  const auto h = static_cast<std::int32_t>(table.shape[1]);   // was kEmbH
  const auto ts = static_cast<std::int32_t>(table.stride[0]);
  const auto os = static_cast<std::int32_t>(out.stride[0]);
  void* out_p = out.data;
  const void* tab_p = table.data;
  const void* ids_p = ids.data;
  vt::cpu::ParallelForRows(
      vt::cpu::CurrentThreadpool(), rows, [&](std::int64_t r0, std::int64_t r1) {
        for (std::int64_t r = r0; r < r1; ++r) {
          kernel(out_p, tab_p, ids_p, h, ts, os,
                 static_cast<std::int32_t>(r), 0, 0, rows, 1, 1);
        }
      });
}
void RopeNeox(Queue& q, Tensor& qt, Tensor& kt, const Tensor& pos, const RopeArgs& args) {
  VT_TRITON_CPU_PROBE("RopeNeox", {{"q", &qt}, {"k", &kt}, {"pos", &pos}});
  VT_TRITON_CPU_DECLINE(RopeFn, OpId::kRopeNeox, q, qt, kt, pos, args);
}
void ReshapeAndCache(Queue& q, const Tensor& k, const Tensor& v, Tensor& kc, Tensor& vc,
                     const Tensor& slot_mapping) {
  // kc/vc are the paged KV cache itself — this probe is where the block layout
  // of `docs/kv-cache-anatomy.md` §3.1 becomes an observation instead of a
  // reading of somebody else's kernel.
  if (ProbeEnabled()) WitnessThread(g_w_rc);
  VT_TRITON_CPU_PROBE("ReshapeAndCache", {{"k", &k},
                                          {"v", &v},
                                          {"k$cache", &kc},
                                          {"v$cache", &vc},
                                          {"slots", &slot_mapping}});
  ReshapeAndCacheKernelFn kernel = RcKernel();
  if (kernel == nullptr || !RcGateOk(k, v, kc, vc, slot_mapping)) {
    g_rc.declined.fetch_add(1, std::memory_order_relaxed);
    VT_TRITON_CPU_DECLINE(ReshapeAndCacheFn, OpId::kReshapeAndCache, q, k, v, kc, vc,
                          slot_mapping);
    return;
  }
  NoteAccept(g_rc);
  ScopedOpTimer timer(g_rc, ProbeEnabled());

  const auto tokens = static_cast<std::int32_t>(slot_mapping.shape[0]);
  const auto i32 = [](std::int64_t x) { return static_cast<std::int32_t>(x); };
  const void* k_p = k.data;
  const void* v_p = v.data;
  void* kc_p = kc.data;
  void* vc_p = vc.data;
  const void* s_p = slot_mapping.data;
  const std::int32_t bs = i32(kc.shape[1]);
  const std::int32_t rc_elems = i32(kc.shape[2] * kc.shape[3]);   // was kRcElems
  const std::int32_t kts = i32(k.stride[0]), vts = i32(v.stride[0]);
  const std::int32_t kbs = i32(kc.stride[0]), kps = i32(kc.stride[1]);
  const std::int32_t vbs = i32(vc.stride[0]), vps = i32(vc.stride[1]);

  // One program per token. No shadow comparison here: the op is a pure copy, its
  // bit-exactness is asserted at build time against a reference that includes a
  // padded (-1) slot and a block-crossing pair, and snapshotting the destination
  // would mean copying the whole 896 MiB KV pool per call. The end-to-end text is
  // also self-checking — a misplaced write makes attention read the wrong page and
  // the output collapses immediately rather than drifting.
  vt::cpu::ParallelForRows(
      vt::cpu::CurrentThreadpool(), tokens, [&](std::int64_t r0, std::int64_t r1) {
        for (std::int64_t r = r0; r < r1; ++r) {
          kernel(k_p, v_p, kc_p, vc_p, s_p, rc_elems, bs, kts, vts, kbs, kps, vbs,
                 vps, static_cast<std::int32_t>(r), 0, 0, tokens, 1, 1);
        }
      });
}
void PagedAttention(Queue& q, Tensor& out, const Tensor& query, const Tensor& kc,
                    const Tensor& vc, const Tensor& block_table, const Tensor& seq_lens,
                    const Tensor& query_start, const PagedAttentionArgs& args) {
  if (ProbeEnabled()) WitnessThread(g_w_pa);
  VT_TRITON_CPU_PROBE("PagedAttention", {{"out", &out},
                                         {"query", &query},
                                         {"k$cache", &kc},
                                         {"v$cache", &vc},
                                         {"blk$tbl", &block_table},
                                         {"seqlen", &seq_lens},
                                         {"qstart", &query_start}});
  PagedAttnKernelFn kernel = PaKernel();
  PagedAttnGeneralFn kernel_g = PaKernelGeneral();
  if ((kernel == nullptr && kernel_g == nullptr) ||
      !PaGateOk(out, query, kc, vc, block_table, seq_lens, query_start, args)) {
    g_pa.declined.fetch_add(1, std::memory_order_relaxed);
    VT_TRITON_CPU_DECLINE(PagedAttentionFn, OpId::kPagedAttention, q, out, query, kc,
                          vc, block_table, seq_lens, query_start, args);
    return;
  }
  NoteAccept(g_pa);
  ScopedOpTimer timer(g_pa, ProbeEnabled());

  const auto i32 = [](std::int64_t x) { return static_cast<std::int32_t>(x); };
  const std::int64_t num_reqs = seq_lens.shape[0];
  const std::int64_t total_q = query.shape[0];
  const auto hq = i32(query.shape[1]);
  const auto qpk = i32(query.shape[1] / kc.shape[2]);

  // Same precomputation the reference does on the caller: flatten (request, local
  // token) into the global token index, so the kernel needs only three arrays and
  // the grid is one embarrassingly parallel axis.
  std::vector<std::int32_t> pos(static_cast<std::size_t>(total_q), 0);
  std::vector<std::int32_t> slen(static_cast<std::size_t>(total_q), 0);
  std::vector<std::int32_t> req(static_cast<std::size_t>(total_q), 0);
  const auto* qsl = static_cast<const std::int32_t*>(query_start.data);
  const auto* sls = static_cast<const std::int32_t*>(seq_lens.data);
  for (std::int64_t r = 0; r < num_reqs; ++r) {
    const std::int64_t q0 = qsl[r], q1 = qsl[r + 1];
    const std::int64_t qlen = q1 - q0;
    if (qlen <= 0) continue;
    const std::int64_t ctx = sls[r] - qlen;
    for (std::int64_t l = 0; l < qlen; ++l) {
      pos[static_cast<std::size_t>(q0 + l)] = i32(ctx + l);
      slen[static_cast<std::size_t>(q0 + l)] = sls[r];
      req[static_cast<std::size_t>(q0 + l)] = i32(r);
    }
  }

  void* out_p = out.data;
  const void* q_p = query.data;
  const void* kc_p = kc.data;
  const void* vc_p = vc.data;
  const void* bt_p = block_table.data;
  const float scale = args.scale;
  const auto wl = args.window_size.has_value() ? i32(args.window_size->left) : -1;
  const auto wr = args.window_size.has_value() ? i32(args.window_size->right) : -1;
  const auto bs = i32(kc.shape[1]);
  const auto qst = i32(query.stride[0]), qsh = i32(query.stride[1]);
  const auto ost = i32(out.stride[0]), osh = i32(out.stride[1]);
  const auto btr = i32(block_table.stride[0]);
  const auto kb = i32(kc.stride[0]), kp = i32(kc.stride[1]), kh = i32(kc.stride[2]);
  const auto vb = i32(vc.stride[0]), vp = i32(vc.stride[1]), vh = i32(vc.stride[2]);
  const auto dh = i32(query.shape[2]);   // ★ was kPaD, baked in at build time
  const std::int32_t* pos_p = pos.data();
  const std::int32_t* sl_p = slen.data();
  const std::int32_t* rq_p = req.data();

  // Grid is (token, q-head). The reference parallelizes over tokens and loops heads
  // inside; splitting both gives more work items, which matters at decode where
  // total_q is 1 and hq is 16.
  vt::cpu::ParallelForRows(
      vt::cpu::CurrentThreadpool(), total_q * hq,
      [&](std::int64_t r0, std::int64_t r1) {
        if (ProbeEnabled()) WitnessThread(g_w_inner_pa);
        for (std::int64_t r = r0; r < r1; ++r) {
          const auto t = i32(r / hq);
          const auto h = i32(r % hq);
          // ⚠️ The two entry points differ by one argument, and it sits in the middle —
          // so this is a branch on which symbol was loaded, not an optional trailing
          // parameter. Getting it wrong shifts the grid indices into the strides.
          if (kernel_g != nullptr) {
            kernel_g(out_p, q_p, kc_p, vc_p, bt_p, pos_p, sl_p, rq_p, qpk, bs, scale,
                     wl, wr, qst, qsh, ost, osh, btr, kb, kp, kh, vb, vp, vh,
                     dh,                                   // ★ head width at run time
                     t, h, 0, i32(total_q), hq, 1);
          } else {
            kernel(out_p, q_p, kc_p, vc_p, bt_p, pos_p, sl_p, rq_p, qpk, bs, scale, wl,
                   wr, qst, qsh, ost, osh, btr, kb, kp, kh, vb, vp, vh, t, h, 0,
                   i32(total_q), hq, 1);
          }
        }
      });

  timer.stop();  // ★ 停在 shadow 之前 —— 見 ScopedOpTimer::stop()
  if (!ShadowEnabled()) return;
  // The op reads the KV cache but writes only `out`, so unlike ReshapeAndCache the
  // snapshot is cheap and the audit is worth having — this is the kernel that most
  // deserves one, since it is the only place a deliberate two-pass-instead-of-three
  // deviation was accepted.
  const auto n_elems =
      static_cast<std::size_t>(total_q) * static_cast<std::size_t>(hq) * dh;
  std::vector<std::uint8_t> mine(
      static_cast<const std::uint8_t*>(out.data),
      static_cast<const std::uint8_t*>(out.data) + n_elems * 2);
  static PagedAttentionFn native = reinterpret_cast<PagedAttentionFn>(
      GetOpFallback(OpId::kPagedAttention, DeviceType::kCPU, kProviderName));
  { NativeTimer nt(g_pa);
  native(q, out, query, kc, vc, block_table, seq_lens, query_start, args); }
  ShadowCompare(g_shadow_pa, mine.data(), out.data, n_elems);
}
void GreedyArgmax(Queue& q, Tensor& out, const Tensor& logits) {
  if (ProbeEnabled()) WitnessThread(g_w_ga);
  VT_TRITON_CPU_PROBE("GreedyArgmax", {{"out", &out}, {"logits", &logits}});
  GreedyArgmaxKernelFn kernel = ArgmaxKernel();
  if (kernel == nullptr || !ArgmaxGateOk(out, logits)) {
    g_ga.declined.fetch_add(1, std::memory_order_relaxed);
    VT_TRITON_CPU_DECLINE(GreedyArgmaxFn, OpId::kGreedyArgmax, q, out, logits);
    return;
  }
  NoteAccept(g_ga);
  ScopedOpTimer timer(g_ga, ProbeEnabled());

  const auto rows = static_cast<std::int32_t>(logits.shape[0]);
  const auto v = static_cast<std::int32_t>(logits.shape[1]);
  void* out_p = out.data;
  const void* lg_p = logits.data;
  vt::cpu::ParallelForRows(
      vt::cpu::CurrentThreadpool(), rows, [&](std::int64_t r0, std::int64_t r1) {
        for (std::int64_t r = r0; r < r1; ++r) {
          kernel(out_p, lg_p, v, v, static_cast<std::int32_t>(r), 0, 0, rows, 1, 1);
        }
      });
}

#undef VT_TRITON_CPU_DECLINE
#undef VT_TRITON_CPU_PROBE

// Registration is table fill only: it never allocates, never throws, and does
// not depend on another TU's constructor having run (op_provider.h contract).
// So doing it from a static initializer is safe by design.
struct Registrar {
  Registrar() {
    OpProvider p;
    p.name = kProviderName;
    p.priority = kProviderPriority;
    p.supports = &TritonCpuSupports;

#define VT_TRITON_CPU_REGISTER(OP_ID, FN_TYPE, FN)                              \
  p.fn = reinterpret_cast<void*>(static_cast<FN_TYPE>(&FN));                     \
  RegisterOpProvider(OP_ID, DeviceType::kCPU, p);

    VT_TRITON_CPU_REGISTER(OpId::kMatmul,          MatmulFn,          Matmul)
    VT_TRITON_CPU_REGISTER(OpId::kMatmulBT,        MatmulFn,          MatmulBT)
    VT_TRITON_CPU_REGISTER(OpId::kRmsNorm,         RmsNormFn,         RmsNorm)
    VT_TRITON_CPU_REGISTER(OpId::kSiluAndMul,      SiluAndMulFn,      SiluAndMul)
    VT_TRITON_CPU_REGISTER(OpId::kEmbedding,       EmbeddingFn,       Embedding)
    VT_TRITON_CPU_REGISTER(OpId::kRopeNeox,        RopeFn,            RopeNeox)
    VT_TRITON_CPU_REGISTER(OpId::kReshapeAndCache, ReshapeAndCacheFn, ReshapeAndCache)
    VT_TRITON_CPU_REGISTER(OpId::kPagedAttention,  PagedAttentionFn,  PagedAttention)
    VT_TRITON_CPU_REGISTER(OpId::kGreedyArgmax,    GreedyArgmaxFn,    GreedyArgmax)

#undef VT_TRITON_CPU_REGISTER
  }
} registrar;

}  // namespace
}  // namespace vt::triton_cpu
