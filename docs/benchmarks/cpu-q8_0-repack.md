# The q8_0 repack tier: one portable layout, one kernel pair per ISA

Measured on three CPUs — x86-64 (Zen 5), Arm with i8mm (Snapdragon 8 Gen 3), and
Arm without it (Jetson Orin Nano, Cortex-A78AE) — against `llama.cpp` on the
same model file, the same axis, and the same thread count.

**The short version.** Serving several requests at once should cost little more
than serving one, because they can share the walk through the weights. Without
the repack tier this engine does not share: throughput is flat from about four
concurrent requests, and adding more buys nothing. With it, the same binary
scales **1.80x → 3.74x** from one to thirty-two concurrent requests on x86, and
**1.18x → 1.72x** on a phone. The layout that makes that possible is entirely
portable; what is not portable is the small kernel pair that reads it, and that
pair is where a platform gets served or does not.

---

## 1. What the tier does

A quantised weight matrix in `q8_0` stores 32 int8 values plus one fp16 scale
per block. The plain path computes one output row per pass over the weights —
`nrc == 1` — so every loaded weight element feeds exactly **one** multiply, and
an activation row that wants the same element loads it again.

`block_q8_0x4` interleaves four output rows in eight-byte chunks:

```
qs = [ r0[0:8]  r1[0:8]  r2[0:8]  r3[0:8]  r0[8:16]  r1[8:16] ... ]
```

so one 32-byte load carries eight elements from each of four weight rows. The
tile kernel pairs that with four activation rows at a time, and a single fetched
element then feeds **sixteen** multiply-accumulates instead of one.

The transform is a byte permutation and nothing else. `cpu_quant_repack.cpp`
says so about itself — *"portable transform + eligibility (arch-independent)"* —
and the layout matches what ggml's `ggml_repack_get_optimal_repack_type` selects
for `q8_0`, so the two ecosystems agree on the bytes.

### It is a register-reuse win, not a traffic win

An earlier draft of this page described the plain path as re-reading the model
from main memory once per request. That is wrong, and reading the loop settles
it: in `QuantChunk` the **weight row is the outer loop and the activation row is
the inner one**, so a row — about 2.1 KB at k=2048 — is fetched once and stays
in L1 while every activation is dotted against it.

The arithmetic agrees. With the tier off and 32 concurrent requests, x86 reads
25.07 tok/s on a 2.012 GB model, which is **1.58 GB/s of weight traffic** — an
order of magnitude below anything this class of machine supplies. Main memory
was never the constraint.

What is missing is reuse *inside the core*, and that distinction has a
consequence worth stating: **a faster memory bus would not have helped, and a
change that only reduces traffic would not have helped either.** The measurements
in §5 bear that out from the other direction.

---

## 2. x86-64: the consumer was simply absent

The layout was portable and written; the GEMM that consumes it lived entirely
inside `#if __aarch64__ && __ARM_FEATURE_MATMUL_INT8`. On x86 the tier was
therefore eligible for nothing, and the engine took the `nrc == 1` path.

Adding it meant writing **two kernels** — `GemmTileQ8_0` (prefill, 4x4) and
`GemvRowQ8_0` (decode, M=1) — from `maddubs`, since x86 has no `vmmlaq_s32`.
The ~80 lines of surrounding machinery turned out to contain no architecture at
all once they were lifted out of the `#if`.

Linux/GCC, `Qwen3.5-2B-Q8_0`, 16 threads (logical/2), 128-in/128-out,
interleaved off/on/off/on behind a **discarded warmup leg**, every bracket
inside 1.7%:

| concurrency | 1 | 2 | 4 | 8 | 16 | 32 | **scaling** |
|---|---:|---:|---:|---:|---:|---:|---:|
| repack off | 13.9 | 20.5 | 23.8 | 24.8 | 25.1 | 25.1 | **1.84x** |
| repack on | 16.6 | 19.8 | 39.5 | 53.2 | 61.2 | 64.3 | **3.88x** |

The gain grows with concurrency exactly as the mechanism predicts: at one
request there is one activation row and the interleave buys only a better load
pattern; as more arrive, four output rows per fetch is worth more and more.

A second run of the same axis, this time with `llama.cpp` built fresh from
`ca3d5a3e1` as a third arm and two legs per arm:

| arm | scaling mean | leg-to-leg spread |
|---|---:|---:|
| repack off | **1.80x** | 2.8% |
| repack on | **3.74x** | 2.6% |
| llama.cpp | **5.63x** | 1.4% |

So the tier recovers **2.08x of a 3.13x** concurrency deficit. It does not close
it; §5 is about what is left.

---

## 3. Arm with i8mm: the same A/B, on the device

Snapdragon 8 Gen 3 (Cortex-X4, i8mm present), clocks pinned, same switch in the
same process:

| concurrency | 1 | 4 | 16 | **scaling** |
|---|---:|---:|---:|---:|
| repack off | 7.32 / 7.07 | 8.27 / 8.17 | 8.47 / 8.54 | **1.18x** |
| repack on | 9.97 / 9.90 | 16.39 / 16.23 | 17.21 / 17.04 | **1.72x** |

With the tier off the phone falls to **1.18x**, which is where x86 and the
Jetson sit when they are also on the plain path. Three very different CPUs
converge on the same number as soon as they run the same kernel — which is the
cleanest evidence available that the kernel, and not the silicon, is what the
figure describes.

The prediction written into the script before that run was "repack off drops
this to about 1.1x". It did.

### 3a. The axis of the table above, and the part of it that was not recorded

Stated plainly because a benchmark without its axis cannot be checked, and this
index promises reproducibility in [reproduce.md](reproduce.md).

**Recorded:** Snapdragon 8 Gen 3 (Cortex-X4, `i8mm` present), clocks pinned, one
process with the switch toggled inside it, concurrency 1 / 4 / 16, two legs per
cell.

**NOT recorded, and not recoverable from what the run left behind:** the thread
count, the prompt and output lengths, and the name of the switch. The two server
logs kept from that session are byte-identical, so they do not say which arm was
which. There is no harness output and there was no script.

⚠️ **Output length is not a detail here, it is the axis.** This engine's tok/s
divides OUTPUT tokens by TOTAL wall, so prefill sits in the denominator. On the
same phone, same model, same switch position, only the output length varying:

| `max_tokens` | 32 | 128 | 256 |
|---|---:|---:|---:|
| tok/s | 6.94 | 11.55 | 12.54 |

The `9.97` in the table above falls between the 32 and the 128 reading. A
comparison against it therefore has to state its own output length or it is not
a comparison.

### 3b. A re-measurement that does carry its axis (2026-09-23)

Not a reproduction of the table above -- a different axis, stated so it can be
checked and repeated. Script: `benchmarks/android/android_repack_ab_ondevice.sh`, driven on
the device, A/B interleaved, guards described in §8.

**Axis:** Qwen3.5-2B-Q8_0 · `~128 prompt tokens / 32 output` · concurrency 1 ·
**6 threads** · `VT_CPU_QUANT_REPACK=0|1` · clocks pinned (all eight cores on
`performance` at their maximum frequency, verified per core) · unique prompt per
request · two legs per arm, interleaved.

| `VT_CPU_QUANT_REPACK` | tok/s | cpu/wall | RssAnon | RssFile |
|---|---:|---:|---:|---:|
| **1** | 5.44 / 5.19 | 5.85 / 6.12 | **2752 / 2760 MB** | 21 / 21 MB |
| **0** | 3.47 / 3.44 | 5.78 / 5.84 | 845 / 844 MB | **1925 / 1925 MB** |

**1.54x**, and the switch is proven by residency rather than by throughput: with
the tier on the weights are a private anonymous copy, with it off they are
borrowed from the mapping. §4 shows only the off state, on a board where the
tier cannot engage at all; this shows both states on one device.

`cpu/wall` is 5.8-6.1 on both arms, so the two differ in the kernel and not in
how much parallelism they got.

**Why six threads and not the usual "logical cores / 2".** That rule was set on
a homogeneous desktop, where naming a count names a set. This part is 1+5+2 --
one X4 at 3.30 GHz, five A720 at 2.96-3.15, two A520 at 2.27 -- and swept here:

| threads | 1 | 2 | 4 | 6 | 8 |
|---|---:|---:|---:|---:|---:|
| tok/s | 2.53 | 3.94 | 5.14 | **5.43** | 4.81 |
| cpu/wall | 1.01 | 1.90 | 4.14 | 5.86 | 7.37 |

The optimum is six, exactly the non-LITTLE core count; eight is 12% *worse*,
because the two A520 join the parallel region and then hold the rest at the
barrier. Confirmed interleaved (8/6/8/6 reading 4.94 / 5.42 / 4.56 / 5.18)
rather than trusted to a sweep whose last leg was also its hottest.

⚠️ **Absolutes are not quotable on this device.** The same configuration read
5.32 tok/s inside a four-leg A/B and 6.94 run fresh: about 25% drift from heat
and back-to-back legs. The ratio survives that only because the two arms are
adjacent and interleaved; a blocked run would fold the drift into the result.

---

## 4. Arm without i8mm: eligible upstream, not eligible here

The Jetson Orin Nano's Cortex-A78AE reports `asimddp` and **not** `i8mm`. Our
`QuantRepackActive()` gates on the i8mm tier, so the tier is dead on that board.
Verified rather than assumed, two ways:

- with the tier off the model sits in **`RssFile` 2,013,176 kB** — borrowed from
  the mmap, not repacked into anonymous memory;
- forcing it on fails loudly: `fatal: unsupported forced Arm ISA tier 'i8mm'`.

**Upstream is not in that position.** ggml's `q8_0` branch offers a *second*
repack arm for exactly this class of chip — `neon && dotprod -> q8_0_4x4` — and
the `libggml-cpu.so` running on that board exports
`ggml_gemv_q8_0_4x4_q8_0` and `ggml_gemm_q8_0_4x4_q8_0` and contains **1044
`sdot` instructions and zero `smmla`**.

⇒ **We ported one arm of a two-arm branch.** Everything measured on that board —
including a 2.87x energy-per-token deficit under real power rails — is a
repacked path measured against a non-repacked one, which is the same comparison
that reads 2.44x on x86 between the switch's two positions, except that on this
board there is no switch to turn on.

The fix is named and has a worked precedent one architecture over: a
**NEON + dotprod (`vdotq_s32`) 4x4 kernel**. The `block_q8_0x4` layout needs no
change. *The layout is portable; the consumer is where the hardware shows up.*

---

## 5. What the tier is NOT, with each alternative priced

Four candidate explanations for the residual gap against `llama.cpp` were tested
rather than argued. All four are small, and knowing that is what keeps effort
off them.

| candidate | worth | how it was settled |
|---|---:|---|
| a larger tile | **1.00x** | `tinyBLAS_Q0_AVX` caps at `gemm<4,4>` — the same 4x4 |
| AVX-VNNI (`vpdpbusd`) | **1.025x** | implemented, instruction-verified, four interleaved legs |
| parallelising the serial activation prep | **≤1.09x** | Amdahl fit over a 1→16 thread sweep: serial fraction 7.4–8.6% |
| deferring the in-loop reduction | **1.22x** | single-variable microbenchmark, both arms bit-agreeing |

The VNNI result is the informative one. Collapsing two ALU instructions into one
bought essentially nothing — which, read together with repack's own 2.40x on
prefill, says the kernel is **not ALU-bound**: fixing load reuse mattered,
fixing arithmetic did not. Neither measurement says that alone.

A thread sweep also showed this engine's prefill scaling is *better* than
`llama.cpp`'s (6.99x vs 5.59x from 1 to 16 threads; serial fraction 8.6% vs
about 12%), so the residual is per-thread kernel throughput — 3.27x at one
thread, narrowed to 2.62x at sixteen by the better scaling. **Reading only the
many-thread cell understates it.** That remainder is open and deliberately
unattributed.

---

## 6. How we know the switch engaged

Throughput is not evidence that a flag took effect — it is the thing the flag is
supposed to change, so using it as proof is circular. Three independent signals:

- **Resident-set split.** Repack copies the weight slice into anonymous memory
  and drops the mmap's file pages, so the *total* is designed to stay the same
  and only the columns move. Measured: `anon 1,144,204 → 3,096,224 kB`,
  `file 1,976,216 → 24,940 kB`, **total within 0.024%** — 1.95 GB crossing in
  each direction, reproducing exactly across legs.
- **Emitted instructions.** For the VNNI A/B: `off: vpdpbusd=0 vpmaddubsw=20`,
  `on: vpdpbusd=20 vpmaddubsw=0`, read from the binaries before anything was
  timed.
- **Assertion count.** The byte-identity test adds 195 comparisons when it finds
  the tier available: `110 → 305`. Had the flag been ignored, both runs would
  print the same number.

The first of those was not the check originally planned. The plan was "repack
copies, so RSS should rise"; it does not, by design, and the code says why. The
signal that does discriminate came from reading that code rather than from the
note about it.

---

## 7. Correctness, and the defaults

The repacked GEMM is **byte-identical** to the plain path, not approximately
equal — `tests/CMakeLists.txt` states that contract and the suite enforces it.
Before any of it was wired in, the standalone probe matched the plain path's
integer dot products on **196,608 of 196,608 blocks exactly**; the residual
1.2e-07 on the float side is the order additions happen in.

**x86 is opt-in** (`VT_CPU_QUANT_REPACK=1`). Landing a tier and changing a
default are decisions with different blast radii: repack copies the weight slice
instead of borrowing the mmap, and the loader's peak footprint under it has no
engine-level measurement yet. **AVX-VNNI is opt-in too** (`VT_X86_VNNI`, off),
kept because it is free where the hardware has it and 2.5% is measured rather
than assumed.

⚠️ **One env-var trap worth knowing.** `VT_CPU_QUANT_REPACK` is a boolean on
x86 but an **ISA tier name** on Arm, where forcing a tier the CPU lacks fails
loudly by design. `=1` is therefore not a portable A/B knob: on an Arm box
without i8mm it will stop the server rather than turn the tier off.

---

## 8. Method notes

Anything below changed a published number at some point in this work, so it is
recorded rather than assumed.

- **Axis alignment before rigour.** The engine's scaling and a `llama.cpp`
  figure could not be compared until both were re-measured on one axis: the
  *same binary* with the tier off reads **1.11x on a 128-in/32-out axis and
  1.84x on 128-in/128-out**. Same code, 1.66x apart. Sharing a unit is not
  sharing a quantity.
- **A discarded warmup leg, and legs interleaved** rather than run as blocks —
  the first leg of an A/B lands on a rested machine, and block ordering puts the
  second arm in a different thermal state.
- **Brackets.** Each axis opens and closes at one request so within-leg drift is
  visible. On this host the *ratio* is reproducible to ~3% while absolute levels
  drift ~10% between legs, so conclusions are stated as ratios with the spread
  shown.
- **`llama.cpp` is built fresh**, not taken from whatever binary was on disk; a
  four-month-old comparison binary is a wrong denominator wearing a small number.
- **Denominators are measured, not looked up.** Host bandwidth on the x86 box
  measures 41.67 GB/s by STREAM triad against a spec-sheet expectation nearly
  twice that.
- **Write the axis down, not just the number.** §3's table was measured
  carefully and cannot be checked, because the thread count, the prompt and
  output lengths and the switch name were never written and the run's own logs
  are byte-identical between arms. §3a states what is known and §3b adds a
  re-measurement that carries its axis. A number whose axis is missing is not a
  weaker result; it is an unverifiable one.
- **On a heterogeneous CPU, a thread COUNT does not name a thread SET.** The
  optimum on the 1+5+2 phone is the non-LITTLE core count, and the homogeneous
  rule undershoots it. Sweep on the platform, and record the choice.
