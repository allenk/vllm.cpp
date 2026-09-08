// Vulkan backend — the `vt::Backend` implementation + its static registrar, plus
// the allocation registry declared in vulkan_buffers.h. BACKEND-VULKAN, W0
// skeleton. vllm.cpp original (vt runtime, inventory deviation §9.1): vLLM has
// no Vulkan platform, so there is no upstream mirror; the SHAPE is the CPU
// reference `src/vt/cpu/cpu_backend.cpp` (the 6 pure virtuals, 36 lines) and the
// Vulkan resource handling is ported from llama.cpp `ggml/src/ggml-vulkan/
// ggml-vulkan.cpp` @ 237ad9b96 (see vulkan_buffers.h and vulkan_context.h).
//
// SCOPE / STUBS — stated plainly so nothing here reads as more than it is:
//   * Every op is dispatched SYNCHRONOUSLY (record, submit, wait on a fence, see
//     vulkan_context.cpp). `Synchronize` is therefore a no-op and `Copy`/`Memset`
//     are host memcpy/memset over the persistently mapped, host-coherent
//     allocation. That is CORRECT but not fast; async command-buffer pipelining,
//     a transfer queue and a staging path for non-host-visible memory are all
//     deferred.
//   * `SupportsGraphCapture()` stays FALSE. A pre-recorded VkCommandBuffer is
//     the eventual mapping (include/vt/backend.h:92 already names it) and is NOT
//     implemented here.
//   * The async-output primitives (AllocPinned / events) inherit the
//     `vt::Backend` defaults, which src/vt/backend.cpp documents as already
//     correct for UNIFIED-memory backends — which GB10 is (one 89.72 GiB
//     DEVICE_LOCAL|HOST_VISIBLE heap), and which is in any case correct for ANY
//     backend whose transfers are synchronous host memcpy, as these are.
//   * `DeviceCapabilityMajor/Minor` report the VULKAN API VERSION; see
//     vulkan_context.h for why that is the right analogue of CUDA's sm_XY.
#include <cstring>
#include <map>
#include <mutex>
#include <string>

#include "vulkan_buffers.h"
#include "vulkan_context.h"
#include "vt/backend.h"

namespace vt::vulkan {
namespace {

struct AllocEntry {
  size_t bytes = 0;
  void* buffer = nullptr;  // packed VkBuffer
  void* memory = nullptr;  // packed VkDeviceMemory
};

// Interval map keyed by allocation base address. Guarded because allocation can
// legitimately race with record on different threads.
std::mutex& AllocMutex() {
  static std::mutex m;
  return m;
}
std::map<uintptr_t, AllocEntry>& AllocMap() {
  static std::map<uintptr_t, AllocEntry> m;
  return m;
}

class VulkanBackend final : public Backend {
 public:
  void* Alloc(size_t bytes) override {
    void* buffer = nullptr;
    void* memory = nullptr;
    void* base = VulkanContext::Get().AllocBuffer(bytes, &buffer, &memory);
    RegisterAllocation(base, bytes == 0 ? 1 : bytes, buffer, memory);
    return base;
  }

  void Free(void* p) override {
    if (p == nullptr) return;
    void* buffer = nullptr;
    void* memory = nullptr;
    VT_CHECK(UnregisterAllocation(p, &buffer, &memory),
             "vulkan: Free() on a pointer this backend did not allocate");
    VulkanContext::Get().FreeBuffer(buffer, memory);
  }

  // Host-coherent, persistently mapped storage + synchronous dispatch => the
  // host may touch the bytes directly. Both are BIT-EXACT byte operations, which
  // is why the gate keeps bit-exactness for pure copy/layout paths while
  // reductions only owe NMSE.
  //
  // BOTH FLUSH FIRST (VK-A2). These are host memcpy/memset over the persistently
  // mapped allocation, so with command-buffer batching a pending dispatch's
  // writes may not have executed yet -- the host would read STALE bytes with no
  // error and no crash. The flush is a no-op when nothing is pending, and when
  // batching is off there is never anything pending.
  void Memset(Queue&, void* p, int value, size_t bytes) override {
    VulkanContext::Get().FlushIfBatchTouches(TryResolve(p).buffer, "memset");
    std::memset(p, value, bytes);
  }
  // DRAIN ONLY ON A REAL DEPENDENCY. MEASURED: this unconditional flush was 2,081
  // of 2,193 flushes in a 27B decode -- 95% -- averaging 3.2 dispatches each, so
  // command-buffer batching was almost entirely defeated and every host memcpy
  // paid a full submit-plus-blocking-fence round trip. Ring exhaustion and the
  // batch cap never fired once.
  //
  // A drain is required exactly when the copy touches memory the OPEN batch is
  // using: reading a buffer the batch writes would see bytes the GPU has not
  // written, and writing one the batch reads would change operands mid-flight.
  // If the batch never bound the buffer -- the common case, since activations
  // flow forward into freshly allocated ones -- neither hazard exists and the
  // batch can keep accumulating. A pointer outside every Vulkan allocation is
  // plain host memory and can never alias a bound buffer.
  // EXPERIMENT (VT_VULKAN_ASYNC_READBACK), default OFF.
  //
  // The base answer is false, and the runner's comment explains what a wrong
  // true costs: the async input-combine leg HOST-DEREFERENCES the sampled id
  // instead of going through Copy, so on a backend where that read is not
  // valid it silently returns stale bytes -- the "!" tokens seen on ROCm.
  //
  // For THIS backend the question is genuinely open rather than settled,
  // because its Copy is a plain memcpy: device allocations here are host
  // addressable. What that does NOT establish is whether the value is READY,
  // since a direct dereference bypasses FlushIfBatchTouches. So it is wired
  // as an opt-in experiment behind a token-exactness gate, not as a default.
  bool SupportsAsyncSampledTokenReadback() const override {
    static const bool on = std::getenv("VT_VULKAN_ASYNC_READBACK") != nullptr;
    return on;
  }

  void Copy(Queue&, void* dst, const void* src, size_t bytes) override {
    auto& ctx = VulkanContext::Get();
    // DIAGNOSTIC, off unless VT_VULKAN_COPY_LOG is set. Every drain in a decode
    // run is reported as "copy-src" -- 1025 of them for 512 tokens, two per
    // token, each a full submit and fence. WHICH two the reason string cannot
    // say, because every host readback shares it. The byte count separates
    // them: a sampled-id readback is a handful of bytes, a logits gather is
    // megabytes.
    static const bool kCopyLog = std::getenv("VT_VULKAN_COPY_LOG") != nullptr;
    if (kCopyLog && (ctx.BatchTouches(TryResolve(dst).buffer) ||
                     ctx.BatchTouches(TryResolve(src).buffer))) {
      const int dst_dev = (TryResolve(dst).buffer != nullptr) ? 1 : 0;
      const int src_dev = (TryResolve(src).buffer != nullptr) ? 1 : 0;
      std::fprintf(stderr,
                   "[vt vulkan] COPY-FLUSH bytes=%zu dst_dev=%d src_dev=%d\n",
                   bytes, dst_dev, src_dev);
    }
    // DEVICE-TO-DEVICE goes to the GPU, not through the host. Both sides being
    // Vulkan allocations means nothing has to be visible to the CPU, so the copy
    // can be recorded into the batch already in flight instead of draining it.
    // Off by default until the A/B is on record; VT_VULKAN_DEV_COPY=1 turns it on.
    static const bool kDevCopy = [] {
      const char* v = std::getenv("VT_VULKAN_DEV_COPY");
      return v != nullptr && v[0] == '1';
    }();
    if (kDevCopy) {
      const auto rd = TryResolve(dst);
      const auto rs = TryResolve(src);
      if (rd.buffer != nullptr && rs.buffer != nullptr &&
          ctx.TryDeviceCopy(rd.buffer, rd.offset, rs.buffer, rs.offset, bytes)) {
        return;
      }
    }
    ctx.FlushIfBatchTouches(TryResolve(dst).buffer, "copy-dst");
    ctx.FlushIfBatchTouches(TryResolve(src).buffer, "copy-src");
    std::memcpy(dst, src, bytes);
  }
  // Synchronize was a no-op while every dispatch waited on its own fence. With
  // batching it is the caller's explicit "make results readable" point, and it is
  // what the tests and the engine use before touching device memory.
  void Synchronize(Queue&) override { VulkanContext::Get().FlushBatch("synchronize"); }
  // THE REFERENCE-TIER SAFETY HOOK (backend.h:44-49, the seam Metal already
  // implements for M3c-1). op_provider.cpp calls this before running a PORTABLE
  // CPU kernel, which reads and writes this backend's device memory DIRECTLY
  // because Vulkan here is unified-memory. Without it, a batched-but-unsubmitted
  // dispatch's writes would be invisible to that host kernel -- stale bytes, no
  // error, no crash. This is what lets command-buffer batching be the DEFAULT
  // rather than an opt-in lever.
  void FlushPending() override { VulkanContext::Get().FlushBatch("reference-tier"); }

  // One process-wide VkQueue is shared by every vt::Queue: the queue handle is
  // the ORDERING domain and, with synchronous dispatch, every op is already
  // ordered. `id` still makes each vt::Queue a distinct identity for the
  // workspace-key machinery (src/vt/ops.cpp MakeWorkspaceKey).
  Queue CreateQueue() override {
    return Queue{Device{DeviceType::kVULKAN, 0}, VulkanContext::Get().queue_handle()};
  }

  bool UnifiedMemory() const override { return VulkanContext::Get().unified_memory(); }

  // Every allocation is HOST_VISIBLE|HOST_COHERENT and persistently mapped by
  // AllocBuffer, and Copy/Memset above are already a plain host memcpy/memset
  // over exactly that pointer. So this is not a new claim -- it NAMES the
  // property this backend has always relied on, so the weight loader can rely
  // on it too instead of keeping a redundant host mirror.
  bool DeviceMemoryIsHostAddressable() const override { return true; }

  // vt_causal_conv1d_update binds the state through the dtype-erased 32/16-bit
  // view pair and rounds once on store, so a bf16 conv_state is read and written
  // in place. Without this the caller must gather the cache into an f32 working
  // copy and scatter it back — two host memcpys per GDN layer per token, each
  // one intersecting the open command batch and forcing a drain.
  bool SupportsCompressedConvState() const override { return true; }

  int DeviceCapabilityMajor() const override { return VulkanContext::Get().api_major(); }
  int DeviceCapabilityMinor() const override { return VulkanContext::Get().api_minor(); }
};

struct Registrar {
  Registrar() {
    // A Vulkan-ENABLED build may still run where there is no loader or no
    // conformant device (a headless CI container). Probing first keeps that a
    // clean "kVULKAN not registered" — the state src/vt/ops.cpp:104-111 already
    // treats as supported — instead of a throw during static initialization,
    // which would abort the process.
    if (!VulkanContext::Available()) return;
    static VulkanBackend backend;
    RegisterBackend(DeviceType::kVULKAN, &backend);
  }
} registrar;

}  // namespace

void RegisterAllocation(void* base, size_t bytes, void* buffer, void* memory) {
  std::lock_guard<std::mutex> g(AllocMutex());
  AllocMap()[reinterpret_cast<uintptr_t>(base)] = AllocEntry{bytes, buffer, memory};
}

bool UnregisterAllocation(void* base, void** out_buffer, void** out_memory) {
  std::lock_guard<std::mutex> g(AllocMutex());
  auto& m = AllocMap();
  auto it = m.find(reinterpret_cast<uintptr_t>(base));
  if (it == m.end()) return false;
  *out_buffer = it->second.buffer;
  *out_memory = it->second.memory;
  m.erase(it);
  return true;
}

Resolved TryResolve(const void* ptr) {
  const auto addr = reinterpret_cast<uintptr_t>(ptr);
  std::lock_guard<std::mutex> g(AllocMutex());
  auto& m = AllocMap();
  auto it = m.upper_bound(addr);
  if (it != m.begin()) {
    --it;
    if (addr >= it->first && addr < it->first + it->second.bytes) {
      return Resolved{it->second.buffer, static_cast<uint32_t>(addr - it->first)};
    }
  }
  return Resolved{};
}

Resolved Resolve(const void* ptr, const char* what) {
  const auto addr = reinterpret_cast<uintptr_t>(ptr);
  std::lock_guard<std::mutex> g(AllocMutex());
  auto& m = AllocMap();
  // upper_bound gives the first allocation starting AFTER addr; the candidate
  // container is its predecessor. Same interior-pointer walk as llama.cpp's
  // ggml_vk_host_get (ggml-vulkan.cpp:7416).
  auto it = m.upper_bound(addr);
  if (it != m.begin()) {
    --it;
    if (addr >= it->first && addr < it->first + it->second.bytes) {
      return Resolved{it->second.buffer, static_cast<uint32_t>(addr - it->first)};
    }
  }
  VT_CHECK(false, std::string("vulkan: ") + what +
                      " points outside every Vulkan allocation — Vulkan kernels can only "
                      "bind memory obtained from vt::GetBackend(DeviceType::kVULKAN).Alloc()");
  return Resolved{};
}

}  // namespace vt::vulkan
