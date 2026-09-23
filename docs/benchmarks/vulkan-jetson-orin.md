# Vulkan on a Jetson Orin Nano: it ports, and what it costs

The Vulkan backend runs on an 8 GB Orin Nano — platform probe, load, inference,
output, all of it. This page is what that costs against `llama.cpp`'s Vulkan on
the same board, and the one structural thing the measurement turned up.

Desktop Vulkan lives in [`windows-rtx-pro-6000.md`](windows-rtx-pro-6000.md).
This page is the edge board.

---

## Scope, and what it is not

**Device.** Jetson Orin Nano 8 GB (sm_87), `NVIDIA Tegra Orin (nvgpu)`,
Vulkan 1.4.329. Measured host bandwidth **27.6 GB/s** by STREAM triad — measured,
not taken from the 102 GB/s on the spec sheet.

**Not measured, and named rather than omitted:** concurrency above 1, tok/W on
this backend, and **Android Vulkan, which is an empty cell entirely**. Adreno and
Mali expose Vulkan and nothing else, so §3 is the gate on that whole tier.

**⛔ One figure from an earlier draft is withdrawn.** A `9.15x` circulated for
this board. It is not citable: its three legs were on three different axes. See
§4.

---

## 1. On the reference axis

`1,024 in / 128 out`, **TOTAL tok/s**, concurrency 1, the axis the rest of this
directory uses. Qwen3-0.6B bf16 — ours reads the published safetensors, llama.cpp
reads the BF16 GGUF converted from the same checkpoint (§5 says why the
containers differ). Unique prompt per request; six legs each, interleaved.
Prompt length is **calibrated, not assumed**: llama.cpp reports what it actually
evaluated, and that count (1,210) is used for both engines.

| engine | wall median | **TOTAL tok/s** | legs | spread |
|---|---:|---:|---|---:|
| ours | 14.954 s | **89.51** | 14.798 / 15.216 / 14.971 / 14.936 / 14.864 / 15.392 | 4.0% |
| llama.cpp | 6.198 s | **215.94** | 6.217 / 6.214 / 6.167 / 6.201 / 6.196 / 6.155 | 1.0% |

**llama.cpp is ahead 2.41x.**

## 2. And on a decode-weighted axis, where it is 1.65x

The same two engines and the same file at `128 in / 32 out` scored on **output
tokens only** — a different quantity, kept because it is the one that isolates
decode:

| engine | tok/s | spread | GPU busy | VmRSS |
|---|---:|---:|---:|---:|
| ours | 18.72 / 18.72 | **0.1–0.2%** | 82% | **1.53 GiB** |
| llama.cpp | 31.03 / 30.82 | 3.8–4.0% | 92–95% | 2.04 GiB |

**1.65x**, we hold less memory, and our run-to-run spread is an order of
magnitude tighter.

⭐ **The two axes disagree in a way that is informative rather than
contradictory.** Moving from a decode-weighted axis to a prefill-heavy one takes
the gap from 1.65x to 2.41x, which says the deficit is **in prefill**. That is
the same decomposition the desktop shows — prefill 2.05x, decode 1.03x — now
reproduced on a second device.

⚠️ The 82% GPU-busy figure is the clean one. Its sibling leg read 47% and is
discarded: the driver's shader cache was written *inside* that window
(1,408,646 bytes, timestamp inside the leg), so 4.7 s of first-run compilation
sat in the sample. See §6.

---

## 3. ⭐ The Vulkan backend has no quantized compute tier

Every Vulkan number here, and every Vulkan number this fork has published, is a
**bf16** number. Four independent checks:

1. `src/vt/vulkan/vulkan_ops.cpp` — `DtypeCode` accepts `f32/f16/bf16` and calls
   `VT_CHECK(false, "vulkan: unsupported storage dtype")` on anything else. Not a
   loader preference: **the backend cannot hold a quantized tensor from any
   source.**
2. `grep -rn kMatmulBTQuant src/vt/vulkan/` is empty. Nor is `kMatmulNvfp4`.
3. The loader decides with
   `keep_quant = OpRegistered(kMatmulBTQuant, dev)` — **the container format
   never enters that expression.** A q8_0 GGUF and an NVFP4 safetensors are both
   expanded for this backend, while CUDA keeps both.
4. `DeviceKeepQuantSupported` enumerates `kROCM` and `kTENSTORRENT`. There is no
   `kVULKAN` row.

| backend | `kMatmulBTQuant` |
|---|---|
| cpu · cuda · rocm · tenstorrent | yes |
| metal · triton_cpu · **vulkan** | **—** |

⭐ Vulkan is the only *portable* backend in that list.

**Measured cost.** Same file, same binary, `--device` the only variable
(Qwen3.5-2B-Q8_0, 2,012,012,800 bytes):

| device | VmRSS | RssAnon | RssFile |
|---|---:|---:|---:|
| Vulkan | 3.93 GiB | **3.70 GiB** | 232 MiB |
| CPU | 2.15 GiB | 257 MiB | **1.90 GiB** |

The CPU arm's `RssFile` is the GGUF mapped in place. The Vulkan arm's `RssAnon`
is the same weights expanded to bf16. Decode is bandwidth-bound, so **2.00 bytes
per weight against 1.06 is a 1.88x handicap before any kernel question** — and
it is what filled a 7.31 GiB board to 130 MB free, which is the real cause of an
allocation failure previously attributed to the allocator.

**The desktop never met this** because it compares bf16 against bf16: format
decides whether the question is asked, the device decides the answer.

---

## 4. ⛔ Why `9.15x` is withdrawn

An earlier draft compared three legs on the 2B q8_0 file:

```
llama.cpp Vulkan   native q8_0         1.06 B/weight    1.9 GB/token
ours CPU           q8_0 mmapped        1.06 B/weight    1.9 GB/token
ours Vulkan        expanded to bf16    2.00 B/weight    3.8 GB/token
```

Under one filename our Vulkan arm was running a model **1.9x larger in bytes
streamed**. The ratio that produced is a property of the harness, not the
engine. §1 and §2 are the replacements, on axes that are stated.

---

## 5. Two engine boundaries this run walked into

**The GGUF arch allowlist has eight rows and `qwen3` dense is not one.**
`model_loader.cpp`'s `kGgufArchArms` lists `deepseek4, muse-glimmer, qwen35,
qwen35moe, qwen3next, qwen4exp, glm5next, glm-dsa`. `qwen3.cpp` registers
`Qwen3ForCausalLM` on the **safetensors** path only. Reproduced on an Android
device too, so it is a capability boundary rather than one board's quirk — and
it is why §1 and §2 feed our arm safetensors.

⚠️ **The container is then a second variable, and it is not inert.**
`VT_VULKAN_DISPATCH_STATS` shows the GGUF path logging `gemv DECLINED` plus two
`scalar matmul` arms while the safetensors path logs neither. **Our own GGUF
orientation makes the backend decline its own optimised GEMV** — the code's
stated ground is that `[K,N]` "is already coalesced", while the same comment
block records that the scalar fallback is *"the LARGEST SINGLE PER-CALL COST in
decode"*. That assumption now has a counterexample.

⇒ §1 and §2 are **deployment** comparisons — each engine on its best path for
its own container — and not kernel-for-kernel ones.

---

## 6. Method notes

- **Cooperative matrix is present and is declined by design.** `vulkaninfo`
  reports `VK_KHR_cooperative_matrix` rev 2 and `VK_NV_cooperative_matrix2`
  rev 1. The log line is `coopmat DECLINED: M is below one tile row (m=1)` —
  decode is m=1, a tile needs M >= its height, and **the desktop declines it for
  the same reason**. Not a portability gap.
- **No engine-side pipeline cache.** A first run pays ~4.7 s of shader
  compilation; the driver's cache (`~/.nv/GLCache`) absorbs it afterwards. It
  also **biases any GPU-busy comparison** whose two engines are not equally cold,
  which is exactly what happened to the discarded 47% leg above.
- **Unique prompt per request.** llama.cpp's server prefix-caches; ours reports
  `prefix caching disabled`. Sending one prompt twice puts the two engines on
  different workloads — once measured 495x off, when the scored requests
  evaluated **4** tokens against a warm-up's 1,980.
- **Denominators are measured.** 27.6 GB/s by triad, not the 102 on the sheet.
  ⚠️ And that denominator is a *lower* bound: llama.cpp's 0.6B leg implies about
  54 GB/s of read streaming, so every "% of roofline" on this board is an
  order-of-magnitude indicator, not a figure. The **1.88x** in §3 is unaffected,
  being a ratio of two streams rather than a fraction of a ceiling.
- **Reproduce:** `benchmarks/android/` holds the sibling on-device scripts and the
  guards they use. The Jetson legs run the same shape over ssh.

---

## 7. Open

| | |
|---|---|
| **Quantized compute tier for Vulkan** | The gate on this whole tier. Build it on a desktop card first: developing a memory-saving feature on a board already out of memory is the wrong order |
| `[K,N]` GEMV, or transposing at GGUF load | Independent of the above and cheaper; §5 has the counterexample |
| Engine-side `VkPipelineCache` | 4.7 s cold, and it contaminates GPU-busy comparisons |
| **Android Vulkan** | Empty. No CUDA fallback exists there, so the first row is the gate |
| Concurrency above 1, and tok/W | Not measured on this backend |
