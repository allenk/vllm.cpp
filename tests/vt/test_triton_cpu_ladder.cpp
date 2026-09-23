// vllm.cpp original (vt runtime); no upstream mirror.
//
// The Triton-CPU provider's LADDER STATES, which the tree asserts in prose and
// has never asserted in CI.
//
// `CMakeLists.txt` says of `src/vt/triton_cpu/triton_cpu_provider.cpp`: "a build
// that does not enable it is byte-identical to one without this file", and the
// provider's own header says "DEFAULT OFF. supports() returns false unless
// VLLM_CPP_TRITON_CPU is set to a non-zero value, so linking this file changes
// nothing until asked." Both are load-bearing claims -- that translation unit is
// compiled into every platform's binary unconditionally, Windows included -- and
// until this file neither was checked by anything.
//
// WHAT IS UNDER TEST:
//
//   1. NOT VACUOUS. A gate that would also pass with the provider ABSENT proves
//      nothing, so the suite first requires that `triton-cpu` is actually in the
//      provider stack. That check is the reason to trust the other two.
//   2. INERT WITHOUT KERNELS. With no `VLLM_CPP_TRITON_CPU_DIR`, every one of the
//      nine registered ops must still bind `vt-native`. This is the dlopen
//      rung degrading, and it is ALSO the permanent Windows state: there
//      `LoadKernelSymbol` returns nullptr by `#ifdef`, so every op declines by
//      construction. One assertion covers both, which is why this suite is not
//      guarded to Linux.
//   3. THE A/B LEVER MOVES THE VARIABLE. `VT_OP_PROVIDER_DISABLE=triton-cpu` is
//      how every same-binary comparison of this provider is taken. This project
//      has a standing rule that a knob which does not move the variable is not a
//      control, learned when `taskset` was applied to an engine that pins its own
//      threads, so the lever is checked rather than assumed.
//
// The environment is READ AND REPORTED rather than assumed, because selection is
// cached for the process lifetime: one process can only observe one state, and a
// green result has to name the conditions it was green under. CMake registers
// this binary twice, once with `VLLM_CPP_TRITON_CPU=1`, so both states run.
#include <doctest/doctest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vt/op_provider.h"
#include "vt/ops.h"

namespace {

using vt::DeviceType;
using vt::OpId;

// Exactly the set `VT_TRITON_CPU_REGISTER` covers in the provider.
const std::vector<std::pair<OpId, const char*>>& TritonOps() {
  static const std::vector<std::pair<OpId, const char*>> kOps = {
      {OpId::kMatmul, "Matmul"},
      {OpId::kMatmulBT, "MatmulBT"},
      {OpId::kRmsNorm, "RmsNorm"},
      {OpId::kSiluAndMul, "SiluAndMul"},
      {OpId::kEmbedding, "Embedding"},
      {OpId::kRopeNeox, "RopeNeox"},
      {OpId::kReshapeAndCache, "ReshapeAndCache"},
      {OpId::kPagedAttention, "PagedAttention"},
      {OpId::kGreedyArgmax, "GreedyArgmax"},
  };
  return kOps;
}

std::string ProviderList(OpId op) {
  std::string out;
  const int n = vt::OpProviderCount(op, DeviceType::kCPU);
  for (int i = 0; i < n; ++i) {
    const char* nm = vt::OpProviderNameAt(op, DeviceType::kCPU, i);
    if (nm == nullptr) continue;
    if (!out.empty()) out += ", ";
    out += nm;
  }
  return out;
}

bool ListHas(OpId op, const char* want) {
  const int n = vt::OpProviderCount(op, DeviceType::kCPU);
  for (int i = 0; i < n; ++i) {
    const char* nm = vt::OpProviderNameAt(op, DeviceType::kCPU, i);
    if (nm != nullptr && std::strcmp(nm, want) == 0) return true;
  }
  return false;
}

const char* EnvOr(const char* name, const char* fallback) {
  const char* v = std::getenv(name);
  return (v != nullptr && v[0] != '\0') ? v : fallback;
}

}  // namespace

TEST_CASE("triton-cpu: the gate is not vacuous -- the provider is in the stack") {
  // WITHOUT this, cases 2 and 3 would pass just as green on a build where the
  // translation unit was never compiled, and would be measuring nothing.
  // ONE argument: doctest's MESSAGE does not concatenate a list, and the first
  // draft of this line reported the environment as SET while it was unset. A
  // diagnostic that lies is worse than none, in a case whose entire job is to
  // let a green result name the conditions it was green under.
  const std::string env_line = std::string("VLLM_CPP_TRITON_CPU=") +
      EnvOr("VLLM_CPP_TRITON_CPU", "(unset)") + "  VLLM_CPP_TRITON_CPU_DIR=" +
      EnvOr("VLLM_CPP_TRITON_CPU_DIR", "(unset)");
  // NOT doctest MESSAGE: it converted a const char* to bool and printed "1",
  // twice, in the one place whose job is to report the conditions truthfully.
  std::fprintf(stderr, "[ladder] %s\n", env_line.c_str());
  for (const auto& [op, name] : TritonOps()) {
    const std::string op_name(name);
    CAPTURE(op_name);
    const std::string line = op_name + " providers: " + ProviderList(op);
    std::fprintf(stderr, "[ladder] %s\n", line.c_str());
    REQUIRE(vt::OpProviderCount(op, DeviceType::kCPU) >= 2);
    CHECK(ListHas(op, vt::kNativeProviderName));
    // THE anti-vacuity assertion: without this the two cases below
    // would be just as green on a build where the provider's
    // translation unit was never compiled.
    CHECK(ListHas(op, "triton-cpu"));
  }
}

TEST_CASE("triton-cpu: what binds depends on being ASKED, and only on that") {
  // ⚠️ THIS CASE WAS WRONG ON ITS FIRST RUN, and the mistake is worth keeping in
  // the comment because it is easy to repeat. The first draft asserted that with
  // no kernel directory every op binds `vt-native` IN BOTH STATES. It does not:
  // with VLLM_CPP_TRITON_CPU=1 all nine bind `triton-cpu` and then decline
  // per-call. BINDING AND RUNNING ARE DIFFERENT THINGS -- op_provider.h is
  // explicit that `last_selected` names the BOUND provider while `declines`
  // counts the forwards -- and a gate that conflates them asserts the wrong
  // invariant while looking perfectly reasonable.
  //
  // So the claim under test is narrower and exactly what the source promises:
  // "linking this file changes nothing UNTIL ASKED". Unasked, the binding must
  // be vt-native. Asked, the binding is triton-cpu whether or not kernels exist,
  // because the decline happens further down, per call.
  REQUIRE(std::getenv("VLLM_CPP_TRITON_CPU_DIR") == nullptr);
  const char* asked = std::getenv("VLLM_CPP_TRITON_CPU");
  const bool is_asked = (asked != nullptr && asked[0] != 0 &&
                         std::strcmp(asked, "0") != 0);
  const char* expect = is_asked ? "triton-cpu" : vt::kNativeProviderName;
  const char* state_line =
      is_asked ? "state: ASKED (VLLM_CPP_TRITON_CPU set, no kernel dir)"
               : "state: UNASKED (VLLM_CPP_TRITON_CPU unset)";
  std::fprintf(stderr, "[ladder] %s\n", state_line);

  vt::EnableOpProviderCallStats(true);
  for (const auto& [op, name] : TritonOps()) {
    const std::string op_name(name);
    CAPTURE(op_name);
    vt::ResetOpProviderStats(op, DeviceType::kCPU);
    REQUIRE(vt::GetOp(op, DeviceType::kCPU) != nullptr);
    const vt::OpProviderStats st = vt::GetOpProviderStats(op, DeviceType::kCPU);
    REQUIRE(st.last_selected != nullptr);
    CHECK(std::strcmp(st.last_selected, expect) == 0);
  }
  vt::EnableOpProviderCallStats(false);
}

TEST_CASE("triton-cpu: the same-binary A/B lever actually moves") {
  // A knob that does not move the variable is not a control.
  REQUIRE_FALSE(vt::OpProviderDisabled("triton-cpu"));
  vt::DisableOpProvider("triton-cpu", true);
  CHECK(vt::OpProviderDisabled("triton-cpu"));

  // With triton-cpu disabled the binding must fall to vt-native REGARDLESS of
  // whether it was asked for -- that is what the lever is for, and it is the
  // one place in this file where the expected value does not depend on the
  // environment.
  vt::EnableOpProviderCallStats(true);
  for (const auto& [op, name] : TritonOps()) {
    const std::string op_name(name);
    CAPTURE(op_name);
    vt::ResetOpProviderStats(op, DeviceType::kCPU);
    REQUIRE(vt::GetOp(op, DeviceType::kCPU) != nullptr);
    const vt::OpProviderStats st = vt::GetOpProviderStats(op, DeviceType::kCPU);
    REQUIRE(st.last_selected != nullptr);
    CHECK(std::strcmp(st.last_selected, vt::kNativeProviderName) == 0);
  }
  vt::EnableOpProviderCallStats(false);

  vt::DisableOpProvider("triton-cpu", false);
  CHECK_FALSE(vt::OpProviderDisabled("triton-cpu"));
}
