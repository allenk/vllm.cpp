// Ported from:
//   tests/plugins_tests/test_oot_registration_offline.py:14-45
//     (test_plugin — VLLM_PLUGINS="" ⇒ arch unsupported;
//      test_oot_registration_text_generation — VLLM_PLUGINS names the plugin ⇒
//      the out-of-tree arch resolves and runs)
//   vllm/plugins/__init__.py:36-90 (load_plugins_by_group allowlist +
//     load_general_plugins load-once idempotence + failure isolation)
// @ 555967922 (vLLM 0.26.0.dev0).
//
// ENG-PLUGIN-SYSTEM W1 CPU brick (CLAIM-PLUGIN-SYSTEM). The extensibility proof:
// an OUT-OF-CORE translation unit (toy_model_plugin.cpp) registers a toy model
// architecture through the public seam, and the engine picks it up ONLY after
// LoadGeneralPlugins runs — with ZERO edit to any engine-core file. RED-first is
// proven inline: before LoadGeneralPlugins (or when the VLLM_PLUGINS allowlist
// blocks the plugin) the toy arch throws "are not supported for now"; after, it
// resolves to the plugin's factory.
#include <doctest/doctest.h>

#include <cstdlib>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "support/test_env.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/plugins/plugins.h"

namespace toy_plugin {
int ToyRegisterCalls();
bool BrokenPluginRan();
std::string_view ToyArch();
}  // namespace toy_plugin

using vllm::HfConfig;
using vllm::ModelRegistration;
using vllm::ModelRegistry;

namespace {

HfConfig ToyConfig() {
  HfConfig config;
  config.architectures = {std::string(toy_plugin::ToyArch())};
  return config;
}

bool ToyArchRegistered() {
  for (const ModelRegistration& r : ModelRegistry::Registrations()) {
    if (r.architecture == toy_plugin::ToyArch()) return true;
  }
  return false;
}

// Set / clear VLLM_PLUGINS for one scenario. Empty-string sets it to "" (the
// upstream "disable all" posture that parses to {""}); nullptr unsets it (load
// all). Always followed by ResetLoadedForTesting so the next LoadGeneralPlugins
// re-reads the env under a cleared load-once latch.
void SetAllowlist(const char* value) {
  // NOT vllm_test::SetEnv, and this is the exception its own header describes:
  // "A test that genuinely needs a defined-but-empty variable cannot use this
  // helper and has to say so at its own call site." This is that call site.
  //
  // The two states are DIFFERENT here and the shim collapses them. VLLM_PLUGINS
  // UNSET means load every plugin (see the `// load all` call below);
  // VLLM_PLUGINS="" means an allowlist of {""}, which matches no plugin name and
  // therefore disables all of them -- Phase 1 below is built on exactly that
  // distinction. SetEnv maps an empty value to a DELETE on both platforms, so
  // routing through it turned "disable everything" into "load everything" and
  // the toy and broken plugins both ran.
  //
  // Found the expensive way: the shim looked like the right answer, its header
  // even names the three files that grew private _putenv_s branches, and the
  // caveat is in that same header two paragraphs down. Three CPU lanes went red.
  if (value == nullptr) {
    ::unsetenv("VLLM_PLUGINS");
  } else {
    ::setenv("VLLM_PLUGINS", value, /*overwrite=*/1);
  }
  vllm::plugins::ResetLoadedForTesting();
}

}  // namespace

TEST_CASE(
    "ENG-PLUGIN-SYSTEM: out-of-core plugin registers a model only via "
    "LoadGeneralPlugins (RED-first + VLLM_PLUGINS allowlist + idempotence + "
    "failure isolation)") {
  // The toy arch must NOT be in the base registry: its plugin TU is out-of-core
  // and its register() has not run. This is the RED baseline — resolution fails.
  REQUIRE_FALSE(ToyArchRegistered());
  CHECK_THROWS_WITH_AS(ModelRegistry::Resolve(ToyConfig()),
                       doctest::Contains("are not supported for now"),
                       std::runtime_error);
  CHECK(toy_plugin::ToyRegisterCalls() == 0);

  // ── Phase 1 — VLLM_PLUGINS="" DISABLES every plugin (ports test_plugin). ────
  // LoadGeneralPlugins runs, but the empty allowlist {""} matches no plugin
  // name, so neither the toy nor the broken plugin executes. The toy arch stays
  // unresolved: still "are not supported for now".
  SetAllowlist("");
  REQUIRE_NOTHROW(vllm::plugins::LoadGeneralPlugins());
  CHECK(vllm::plugins::PluginsLoaded());
  CHECK(toy_plugin::ToyRegisterCalls() == 0);
  CHECK_FALSE(toy_plugin::BrokenPluginRan());
  CHECK_FALSE(ToyArchRegistered());
  CHECK_THROWS_WITH_AS(ModelRegistry::Resolve(ToyConfig()),
                       doctest::Contains("are not supported for now"),
                       std::runtime_error);

  // ── Phase 2 — allowlist NAMES only the toy plugin (ports the
  // test_oot_registration_text_generation posture: VLLM_PLUGINS="register_..."
  // ⇒ that plugin loads). The broken plugin is not named ⇒ not run. ───────────
  SetAllowlist("register_toy_model");
  REQUIRE_NOTHROW(vllm::plugins::LoadGeneralPlugins());
  CHECK(toy_plugin::ToyRegisterCalls() == 1);
  CHECK_FALSE(toy_plugin::BrokenPluginRan());  // not on the allowlist

  // GREEN: the out-of-core arch now resolves to the plugin's factory — the
  // engine picked it up with ZERO core edit. Full "complete factory" contract.
  REQUIRE(ToyArchRegistered());
  const HfConfig toy_config = ToyConfig();
  const ModelRegistration& reg = ModelRegistry::Resolve(toy_config);
  CHECK(reg.architecture == toy_plugin::ToyArch());
  REQUIRE(reg.factory != nullptr);
  CHECK(reg.factory->parse_config != nullptr);
  CHECK(reg.factory->load_weights != nullptr);
  CHECK(reg.factory->prepare != nullptr);
  CHECK(reg.factory->forward != nullptr);
  CHECK(reg.factory->make_kv_cache != nullptr);
  CHECK(reg.info.is_text_generation_model);

  // ── Phase 3 — IDEMPOTENCE (plugins/__init__.py:82-85, "one process loads
  // plugins once"). A second LoadGeneralPlugins WITHOUT a reset is a no-op: the
  // toy register() does not run again. ───────────────────────────────────────
  CHECK(vllm::plugins::PluginsLoaded());
  vllm::plugins::LoadGeneralPlugins();  // latch set ⇒ no-op
  CHECK(toy_plugin::ToyRegisterCalls() == 1);

  // Even when the latch is cleared and load re-runs, the plugin is written to be
  // safely loadable multiple times (its own get_supported_archs guard): the
  // callback runs again but registers NO duplicate arch.
  SetAllowlist(nullptr);  // load all
  REQUIRE_NOTHROW(vllm::plugins::LoadGeneralPlugins());
  CHECK(toy_plugin::ToyRegisterCalls() == 2);  // callback re-ran
  int toy_entries = 0;
  for (const ModelRegistration& r : ModelRegistry::Registrations()) {
    if (r.architecture == toy_plugin::ToyArch()) ++toy_entries;
  }
  CHECK(toy_entries == 1);  // no duplicate despite the re-run

  // ── Phase 4 — FAILURE ISOLATION (plugins/__init__.py:68-72). The "load all"
  // run above also invoked the broken plugin, which throws. LoadGeneralPlugins
  // caught + skipped it (the REQUIRE_NOTHROW above proves it never propagated),
  // and the good toy arch is still resolvable. ───────────────────────────────
  CHECK(toy_plugin::BrokenPluginRan());
  CHECK(ToyArchRegistered());
  CHECK_NOTHROW(ModelRegistry::Resolve(ToyConfig()));

  // Restore a clean env for any later suite in the process.
  SetAllowlist(nullptr);
}
