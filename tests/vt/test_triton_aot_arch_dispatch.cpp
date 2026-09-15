#include <array>
#include <cstdint>

#include "doctest/doctest.h"
#include "vt/cuda/triton_aot_arch_dispatch.h"

namespace {

int Tree0(int value) { return value + 0; }
int Tree1(int value) { return value + 10; }
int Tree2(int value) { return value + 20; }
int Tree3(int value) { return value + 30; }
int Tree4(int value) { return value + 40; }
int Tree5(int value) { return value + 50; }
int Tree6(int value) { return value + 60; }

int loaded_tree = -1;
void Load0() { loaded_tree = 0; }
void Load1() { loaded_tree = 1; }
void Load2() { loaded_tree = 2; }
void Load3() { loaded_tree = 3; }
void Load4() { loaded_tree = 4; }
void Load5() { loaded_tree = 5; }
void Load6() { loaded_tree = 6; }

}  // namespace

// SEVEN, not upstream's six: sm_120 (12.0) is this fork's tree, and adding it
// moved 12.1 from index 5 to 6. Upstream cannot have written these rows.
TEST_CASE("Triton AOT selects only the exact seven vendored SM trees") {
  const std::array trees{&Tree0, &Tree1, &Tree2, &Tree3,
                         &Tree4, &Tree5, &Tree6};
  static_assert(trees.size() == vt::cuda::kTritonAotTreeCount);
  CHECK(vt::cuda::TritonAotTreeIndex(8, 0) == 0);
  CHECK(vt::cuda::TritonAotTreeIndex(8, 6) == 1);
  CHECK(vt::cuda::TritonAotTreeIndex(8, 9) == 2);
  CHECK(vt::cuda::TritonAotTreeIndex(9, 0) == 3);
  CHECK(vt::cuda::TritonAotTreeIndex(10, 0) == 4);
  CHECK(vt::cuda::TritonAotTreeIndex(12, 0) == 5);
  CHECK(vt::cuda::TritonAotTreeIndex(12, 1) == 6);

  CHECK(vt::cuda::DispatchTritonAot(8, 0, -1, trees, 7) == 7);
  CHECK(vt::cuda::DispatchTritonAot(8, 6, -1, trees, 7) == 17);
  CHECK(vt::cuda::DispatchTritonAot(8, 9, -1, trees, 7) == 27);
  CHECK(vt::cuda::DispatchTritonAot(9, 0, -1, trees, 7) == 37);
  CHECK(vt::cuda::DispatchTritonAot(10, 0, -1, trees, 7) == 47);
  CHECK(vt::cuda::DispatchTritonAot(12, 0, -1, trees, 7) == 57);
  CHECK(vt::cuda::DispatchTritonAot(12, 1, -1, trees, 7) == 67);
}

TEST_CASE("Triton AOT falls back for the three release SMs without trees") {
  const std::array trees{&Tree0, &Tree1, &Tree2, &Tree3,
                         &Tree4, &Tree5, &Tree6};
  static_assert(trees.size() == vt::cuda::kTritonAotTreeCount);
  // {12, 0} left this list when this fork vendored its tree. A stale entry here
  // would assert the fast path is OFF on the very card we develop on.
  for (const auto [major, minor] :
       std::array<std::array<int, 2>, 3>{{{8, 7}, {10, 3}, {11, 0}}}) {
    CHECK(vt::cuda::TritonAotTreeIndex(major, minor) == -1);
    CHECK(vt::cuda::DispatchTritonAot(major, minor, -99, trees, 7) == -99);
  }
}

TEST_CASE("Triton AOT wrong-tree mutation is observable") {
  const std::array swapped{&Tree1, &Tree0, &Tree2, &Tree3,
                           &Tree4, &Tree5, &Tree6};
  CHECK(vt::cuda::DispatchTritonAot(8, 0, -1, swapped, 7) != 7);
  CHECK(vt::cuda::DispatchTritonAot(8, 6, -1, swapped, 7) != 17);
}

TEST_CASE("Triton AOT loader dispatch is exact and unavailable trees are no-op") {
  const std::array loaders{&Load0, &Load1, &Load2, &Load3,
                           &Load4, &Load5, &Load6};
  loaded_tree = -1;
  CHECK(vt::cuda::DispatchTritonAotVoid(9, 0, loaders));
  CHECK(loaded_tree == 3);
  loaded_tree = -1;
  CHECK_FALSE(vt::cuda::DispatchTritonAotVoid(11, 0, loaders));
  CHECK(loaded_tree == -1);
  // ... while 12.0, which upstream used for this no-op case, now dispatches.
  CHECK(vt::cuda::DispatchTritonAotVoid(12, 0, loaders));
  CHECK(loaded_tree == 5);
}
