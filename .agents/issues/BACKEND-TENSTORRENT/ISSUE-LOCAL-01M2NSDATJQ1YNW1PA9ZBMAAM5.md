ID: ISSUE-LOCAL-01M2NSDATJQ1YNW1PA9ZBMAAM5
Title: APEX/TT: keep-quant E=1 chunked arm emits all-zero logits at lm_head decode shapes
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-16
Updated: 2026-09-16
Closed: 2026-09-18

## Problem

The token-correctness gate (ISSUE-LOCAL-01M2NMSNSS6R8T0EW1WYHHVA0Z) caught: the E=1 chunked keep-quant matmul arm (MatmulBTQuantGroupedKernel, production dense path) commits all-zero logits at the 27B lm_head [1,248320] Q6_K shape (4 commits tracked=1, EnsureHost downloads them, argmax = id 0 every step, VT_DEBUG_SAMPLED tok=0 x4). Every sampled token of the 27B e2e is 0. Suspects: ttnn matmul with M in {1,2} against the chunked [chunk,K] bf16 tiles returning zeros, the partials concat, or a TTReclaimPlanes use-after-free inside the chunk loop (a reclaimed plane still aliased). The 0.8B vehicle never hit this because its bf16 head routes through MatmulBf16LogitsF32D (dense non-quant matmul), so the quant-head e2e path was never validated until now. Needs: a red unit test at [1,K]x[248320,5120] Q6_K (or scaled-down V) vs the CPU oracle, then the fix.

## Resolution

-

## CORRECTION (2026-09-16, later same day)

The nb=20 "garbage" was a TEST BUG: the discriminator allocated packed(N*2)
blocks while declaring the tensor {N, K=5120} (1280 blocks) — an out-of-bounds
read, whose drifting 1e9 values are exactly the uninit-memory signature. With
the allocation fixed, the chunked arm is CORRECT at unit scale across chunk
overrides 4 and 20, M in {1,2}, nb in {2,20} (worst_abs 1354, in envelope),
and the decode is bit-exact at rows=4/nb=20 (B=80). The keep-quant kernel and
the TTReclaimPlanes chunk loop are exonerated at unit scale.

The e2e all-zero logits stand (VT_DEBUG_SAMPLED tok=0, committed device tensor
zero). Remaining repro axes: production scale (N=248320, 76 chunks of 3276
rows, B=65520 decode, 76-way concat) or the engine context around the head
(DBuf pooled logits, ResidentWeight staging of the real lm_head pointer, the
real activation source — a zero hidden state would also produce zero logits
through a correct kernel).

## Original narrowed-repro section (SUPERSEDED by the correction above)

Unit-scale discriminator (`-tc="*lm_head decode shapes*"`):
- M=1/2, N=64, nb=2 (K=512), chunk override 4 (16 chunks): CORRECT (worst_abs 239, within envelope), stable across 5 repeated calls.
- M=1/2, N=64, nb=20 (K=5120), chunk override 4: **GARBAGE** — worst_abs 5.7e9 / 1.2e10, values drift run to run (uninitialized/reused memory signature).
- M=1, N=64, nb=20, SINGLE chunk (override 0): CORRECT magnitude (worst_abs 1031, K=5120 bf16 dot).

So the defect is the MULTI-CHUNK path at nb=20, not the K=5120 matmul itself, not repeated calls, not M. Per chunk: B = chunk_rows x nb = 80 words; the single-chunk case takes the sl_alias branch (slice returns the whole tensor) while multi-chunk slices + TTReclaimPlanes({&sl,&wf,&wbf}). Prime suspect: a plane reclaimed while still aliased in the chunk loop, or the slice of the word shadow at row offsets c0*nb.

## Engine attribution (2026-09-16, debug hooks VT_DEBUG_SAMPLED=2 / VT_TT_TRACE_DEBUG)

- The logits are EXACTLY 0.0 across all 248320 entries at every step
  ([TT-LOGITS] nz=0/248320) — the signature of a zero activation OR a zero
  weight, not numeric corruption.
- Activations entering quant matmuls are LIVE for the first ~9 calls
  (through block 2) and ZERO from block 3 on. The death point in the op
  timeline: block 3's MoE down-proj enters MatmulBTQuant and never reaches
  MatmulBTQuantGrouped — it dispatched to the int8-dot kernel (Q3_K is
  served unconditionally by int8-dot at tenstorrent_ops.cpp:2862).
- BUT forcing VT_TT_KEEPQUANT_INT8DOT=1 (all encodings through int8-dot)
  ALSO yields all-zero logits. Both arms fail in-engine; the grouped arm is
  proven correct at unit AND production scale (N=32760, chunk=3276,
  B=65520, 10-way concat: sane, worst_abs 2253).
- Slot trace on the 17x5120 hidden slot: committed 4x per forward
  (dc=1), EnsureHost downloads from the right slot — first forward reads
  live bytes, later forwards read zeros. The producers write zeros.

Next suspects, in order: (1) the word-shadow STAGING of specific weights —
if a weight's staged i32 words are zeros (staged from a stale/zeroed host
master, e.g. an mmap'd GGUF tensor never read into the staging source), every
matmul through it emits exact zeros while the kernel stays correct; (2) the
engine's weight-view plumbing for this GGUF (ResidentWeight for the
lm_head/down weights); (3) the residual path around the first dead op.
Probe: checksum the staged `words` (row 0) inside the kernels behind
VT_DEBUG_SAMPLED=2, and checksum the packed host master before staging.

## Staging + kernel exonerations (2026-09-16, continued)

- Weight staging is LIVE: all 353 stagings across the artifact's encoding mix
  (iq2_s x88, iq2_xxs x44, iq3_xxs x68, q3_K x78, q4_K x74, q6_K x1 — the
  lm_head staged live at rows=248320). No zero masters, no zero word shadows.
- The int8-dot kernel is BIT-EXACT at the production block-3 shape
  (q3_K, M=17, N=5120, K=17408): 87040/87040 exact vs the CPU vec_dot oracle
  (test "int8-dot Q3_K at the 27B down-proj shape"). Both keep-quant kernels
  are now exonerated in isolation at production shapes.

## Remaining suspect: the ServeActF32 / SigmoidGateBf16 geometry path

The death sits right after the attention block (AttnQkNormRopeGate ->
CastBf16 -> ReshapeAndCache -> TryPagedAttentionDeviceDecode ->
SigmoidGateBf16). SigmoidGateBf16 serves its operands at a [1, n] geometry
through ServeActF32 (tenstorrent_ops.cpp:7607-7653), whose eager
geometry-mismatch arm calls CaptureSafeReshape — the FREE ttnn::reshape —
on a TILE shadow whose native geometry may be [17, 87040]. This is the
exact doctrine violation NormalizeDevF32Tile's comment warns about
(free reshape of a TILE at a changed shape; the #3206 review's F2 owed
item named kSigmoidGateBf16 as a hardcoded-geometry consumer predating
the doctrine). Probe: checksum dev_attn/dev_gate inside
SigmoidGateBf16Kernel behind VT_DEBUG_SAMPLED=2; read the serve path at
:7613 against the AttnQkNormRopeGate doctrine.

## SigmoidGateBf16 exonerated as the CAUSE (2026-09-16, probe + fix attempt)

- Probe: the gate's operands are LIVE at the first full-attention layer and
  ALREADY ZERO at the next one — the zeros arrive from upstream, the serve
  path faithfully serves them.
- A doctrine-aligned fix of the serve path's eager reshape (ROW_MAJOR
  round-trip) changed nothing — reverted as unproven. SigmoidGateBf16 is a
  victim, not the cause.
- Refined timeline: full-attention layers are RARE in this hybrid (the op
  trace shows one AttnQkNormRopeGate+SigmoidGateBf16 sequence among GDN
  blocks). The FIRST zero anywhere in the stream is the KQACT call 10
  (block 3's 17x5120 GDN in-proj activation). Between KQACT call 9 (block 2
  down-proj input, live) and call 10 sit: block 2's down-proj (grouped arm),
  its commit, the residual add, and block 3's pre-GDN RmsNorm.
- Next probes: checksum the down-proj OUTPUT commit (out stats behind the
  probe in MatmulBTQuantGroupedKernel), and find the residual-add op (no Add
  op appears in the TT op trace — it may run on device through a fused op,
  on host, or inside RmsNorm; whichever it is, it consumed zeros or
  committed zeros).

## Grouped outputs LIVE; residual add located (2026-09-16, final probes)

- All probed grouped down-proj outputs are LIVE pre-commit
  ([TT-KQOUT] nz=full on every call incl. the 17x5120 hidden-stream shapes).
  The keep-quant arm commits live data through the death window.
- The residual add has NO op of its own: it lives INSIDE RmsNormKernel —
  the TT RmsNorm takes a residual parameter (residual += x; out =
  rms_norm(residual)), reached from the model's block boundary and via
  FusedChainKernel (tenstorrent_ops.cpp:4188-4212, which dispatches
  kFusedAddRmsNormStd to the same RmsNormKernel call). That is why no Add
  appears in the TT op trace.
- Next probe: checksum x and *residual inside RmsNormKernel when residual
  is non-null (VT_DEBUG_SAMPLED=2), at the block 2->3 boundary. The
  suspects are the ttnn add at the [17,5120] non-tile-aligned M (TILE pads
  M to 32), or the residual commit/download geometry after the add.

## Residual add exonerated; corruption narrowed to rms_norm output/commit (2026-09-16)

- [TT-RESADD] probe: every residual add's operands AND sum are LIVE through
  the whole run (387 probe lines), yet the KQACT call 10 (the next RmsNorm's
  output feeding the GDN in-proj) still reads zero and the logits stay zero.
- The corruption is therefore INSIDE RmsNormKernel's device path AFTER the
  residual add: either the ttnn::rms_norm output is zero (at [17,5120]
  non-tile-aligned M?), or the CommitDevice2D of `out` records a wrong/empty
  device tensor so the consumer's EnsureHost downloads zeros.
- The same norm consumed a LIVE add output (probed), so the input side is
  clean.
- NEXT (first thing): checksum the rms_norm device output in RmsNormKernel
  before CommitDevice2D, behind VT_DEBUG_SAMPLED=2. Zero output => ttnn
  rms_norm at this shape (compare the captured 0.8B arm, which pins rms_norm
  at other shapes); live output => CommitDevice2D/slot — diff what
  CommitDevice2D stores vs what EnsureHost downloads for that slot.

## rms_norm outputs LIVE; the death is in the model plumbing between (2026-09-16)

- [TT-NORMOUT] probe: EVERY rms_norm device output is live pre-commit
  (nz~87040/87040, all norms, whole run). ttnn::rms_norm is fine at
  [17,5120].
- DECISIVE: the consumer's activation pointer (0xfff0c811fdc0 in the
  KQACT runs) is a DIFFERENT TENSOR than the norm's out slot (0xffef9011fdc0
  in the NORMOUT runs). There is an intermediate copy/view between the norm
  output and the matmul input, and THAT copy produces the zeros.
- Next: run with VT_TT_SLOT_TRACE=1 + VT_DEBUG_SAMPLED=2 and correlate the
  act tensor's pointer to its producer events (register/ensure/commit) to
  find the copy op; then read the model plumbing (qwen3_5.cpp) for how the
  norm output reaches the matmul (a DBuf/ view/ CastBf16?) and probe it.
  The drift signature (values differ run to run) says the copy reads
  unallocated/reused memory.

## EnsureDevice2D same-numel arm exonerated too (2026-09-16, last probe)

Fixing the eager same-numel reshape arm (ROW_MAJOR round-trip instead of the
free TILE reshape) changed nothing — decode-step acts still read zero and the
logits stay zero. Reverted as unproven. The poisoning is NOT (only) the slot
reshape.

## Where the investigation stands (handoff)

Eliminated by direct probes: grouped arm (unit + production scale),
int8-dot (production shape bit-exact), weight staging (all live),
SigmoidGateBf16/ServeActF32, residual add ([TT-RESADD] all live),
ttnn::rms_norm + its commit ([TT-NORMOUT] all live pre-commit),
EnsureDevice2D slot reshape.

The one hard fact still standing: the SAME tensor slot commits live data and
the consumer later EnsureHost-reads zeros from that very slot, with values
that drift run to run (uninit/reuse signature). Something between the last
live probe and the read replaces or rebinds the slot's device tensor without
a probe on it. The next session should bisect the slot-history timeline:
with VT_TT_SLOT_TRACE=1, take the zero slot's pointer, list EVERY event line
in order, and probe/inspect each mutation (FindSlot writes outside the
probed ops — e.g. EnsureHostBytes, DownloadToHost, the recycled-address
drop, or a second tensor aliasing the same host pointer). Also worth
checking: the probe itself (to_vector downloads) perturbs the run — compare
a probe-off run's sampled tokens to confirm the death is not probe-induced
(one run, VT_DEBUG_SAMPLED unset, --output-token-ids).

## Probe-off control: the death is real (2026-09-16)

A run with NO probes (`--output-token-ids`, no env) still dumps all-zero ids.
The defect is independent of the measurement.

## Converged hypothesis: slot holds a force-freed buffer

Everything else is eliminated (see the probe chain above). The remaining
explanation that fits ALL observations — live commits, zero later downloads,
values drifting run-to-run, decode dying earlier than prefill (less work
before the first reuse), probes not required — is: a TTReclaimPlanes
force-deallocate frees a buffer that a BufferSlot still references (via an
aliased view stored in s->device — e.g. the EnsureDevice2D reshape arms store
reshaped views, the exact-shape arm stores the producer tensor itself), the
allocator recycles the block, and the slot's next EnsureHost download reads
recycled (zeroed) memory while the slot still says device_current.

Next instrumentation: log every TTReclaimPlanes force-deallocate (plane
logical shape + ~size) behind VT_TT_SLOT_TRACE=2, and log the same identity
for the slot's stored tensor in CommitDevice2D/EnsureHost download. Correlate
a zero slot's stored-tensor identity against the free list. Then the fix is
one of: (a) not storing aliased views in slots, (b) reclaim checking the
slot map before force-freeing, or (c) slot invalidation on dealloc.

## Slot-history instrumentation results (2026-09-16, VT_TT_SLOT_TRACE=2)

- 433 downloads, 314 read zeros (73%).
- The FIRST zero download (log line 306) is slot 0xffeea102f280's FIRST-EVER
  commit: [1,30720] bf16, committed at line 302, downloaded at 306 — zero.
  No [TT-FREE] before it, no earlier zero downloads. So the producing op
  computed a zero tensor from its first use: the zeros ORIGINATE in the
  producer's operands or its own compute, and propagate downstream (the 73%
  are inheritance, not independent corruption).
- Shape [1,30720] bf16 matches the SigmoidGateBf16 commit form
  (CommitDeviceLogical2D(out, dev_y, 1, n)); earlier SIGGATE probes showed
  its first call LIVE and later calls' operands zero — so the origin is at
  or before that op's operands.
- Next: rerun with VT_TT_TRACE_DEBUG=1 + VT_TT_SLOT_TRACE=2 to name the
  committing op of the first zero; then walk the operand chain one op at a
  time (the SIGGATE probe showed attn/gate live at call 1 — find which
  operand of which call is the first zero and read its producer).

## The first zero commit is in the GDN state path (2026-09-16, op-named run)

Op-named walk-back (/tmp/walkback.log): the first zero [1,30720] bf16
commit sits between the ops `GdnStateScatter` and `GdnPostConv` of a GDN
block — i.e. the GDN state write or the post-conv preamble. The op
sequence per block: RmsNorm, MatmulBTQuant x2, MatmulBT x2 (in-projs),
GdnStateGather, CausalConv1dFwd, GdnStateScatter, [FIRST ZERO COMMIT],
GdnPostConv, GdnStateGather, GdnPrefill, ...

Interpretation: if the scatter writes the updated recurrent state as zeros
(or the conv state output is zero), every later block's gather reads zeros,
GdnPrefill computes on zeros, and the whole stream past that block dies —
exactly the propagation observed.

NEXT (mechanical): probe the outputs of CausalConv1dFwd and GdnStateScatter
(checksum behind VT_DEBUG_SAMPLED=2), and compare the chain against the CPU
oracle at the in-engine shape (T=17). The GDN ops are pinned by the suite at
test shapes (185/185) — the failing configuration is the real [1,30720]-class
state plane and/or the state-cache slot addressing in-engine.

## GDN conv/scatter exonerated; GdnPostConv is the origin candidate (2026-09-16)

[TT-GDN] probes: conv_out, conv_state, conv_x, scatter newc — ALL LIVE at
every probed call. The [1,30720] bf16 zero commit is NOT the scatter (its
newc is f32 and live). The remaining producer at that position:
**GdnPostConv**, whose five outputs (q_out/k_out/v_out/g_out/beta_out) are
unprobed, and whose v/beta commit shape class matches [1,30720] bf16.

NEXT: checksum GdnPostConv's five outputs behind VT_DEBUG_SAMPLED=2 at the
T=17 recipe (same gdn_checksum pattern; the outputs are vt::Tensors —
use the host_ck LoadElemF32 helper). Which output is zero names the
broken slice/gather inside the kernel; compare against cpu_ops
GdnPostConvKernel at T=17 for the fix. Note the kernel reads its conv
input via EnsureDevice2D/EnsureHost — check whether the zero output's
input view (a row-strided view of the conv tensor) is being served from
stale host bytes.

## DEFECT LOCALIZED: GdnPostConvKernel's first in-engine call (2026-09-16)

[TT-GPC] probe (/tmp/gpc_probe.log):
- Call 1: q live (17408/34816), **k nz=2/34816 (ALL ZEROS)**, **v
  nz=8170/104448 with max=50,593,792 (uninitialized memory garbage)**, g/beta
  live. The k zeros + v garbage propagate: the hidden stream dies and every
  later read is zero.
- Calls 2+: q live, **k EXACTLY half-zero (17408/34816)**, v exactly
  half-live — a consistent half-corruption (a buffer half never written or
  read at the wrong offset).

The defect is inside GdnPostConvKernel (tenstorrent_ops.cpp:~4424) on this
APEX in-engine shape: k (the L2Norm device chain over the k2 columns) and v
(the plain slice copy of the conv tensor) are broken while q (same L2Norm
chain over q2) is live — the q/k asymmetry localizes it further: k2's slice
or its L2Norm differs from q2's.

NEXT:
1. Rerun with VT_DUMP_TRUST=<dir> — the kernel already has TrustDump at the
   commit site (call 0 dumps k_conv_in/q/k/v/g/beta + g-chain intermediates
   as .f32). Diff k/v against the cpu_ops GdnPostConvKernel oracle at T=17.
2. Compare the k2 slice derivation vs q2's (offsets: key_dim vs
   key_dim+2*key_dim... the comment says q/k are [0,key_dim) and
   [key_dim,2*key_dim) slices, v the rest) and the l2() geometry (t*hk, dk).
3. The 27B head counts differ from the 0.8B vehicle the kernel was written
   and pinned against — check hk/hv/dk/dv and the [17, ...] non-tile-aligned
   M in the reshape/l2 chain for an offset that only breaks at this config.

## CORRECTION: the [TT-GPC] probe was invalid; dtype mismatch is the prime suspect (2026-09-16)

The TrustDump rerun shows call-0 q/k/v ALL LIVE at the commit site (device
truth via Backend::Copy after EnsureHostBytes: k_q 34816/34816, k_k
34816/34816, k_v 104447/104448, conv_in 174079/174080; shapes [17,16,128] /
[17,48,128] — hk=16 dk=128 hv=48 dv=128, key_dim 2048 tile-aligned, so the
slice-misalignment theory is also out). The [TT-GPC] probe read raw HOST
bytes via LoadElemF32 WITHOUT EnsureHost — after CommitDeviceLogical2D those
are legitimately stale (host_current=false), so its zeros were an artifact.
Lesson recorded: probes must download device truth (to_vector) or EnsureHost
first.

The VALID device-truth signal stands: the first zero [TT-DL] is the GDN
STATE CACHE slot (0xffeea102f280, [1,30720] **bf16**, dt=1) — committed by
GdnStateScatter's CommitDeviceLogical2D(cache, newc, ...). But the scatter's
own [TT-GDN] probe showed newc LIVE as F32. A live f32 newc committing into
a bf16 cache slot (or a bf16 reinterpreted as f32 on the way back) is the
prime suspect: a dtype mismatch in the GDN state-cache path. GdnPrefill then
reads zero state, and the stream dies past the first GDN block.

NEXT: read GdnStateScatterKernel/GdnStateGatherKernel/EnsureGdnCacheDevice
for the cache dtype handling (cache bf16 vs newc f32), compare with
cpu_ops, and red unit test the scatter at the in-engine cache dtype.

## Zero-download census names k_out (2026-09-16, walkback.log correlation)

Scripted correlation of every zero [TT-DL] to its committing op (walkback
log, op+slot interleaved):
- **86x GdnPostConv [272x128] — the k_out tensor. THE ORIGIN.** (Consistent
  with the GPC probe's call-1 k nz=2/34816; only that probe's v-garbage
  reading was an artifact.)
- 48x GdnStateGather [1x30720] — the state cache; likely legit (initial
  state zeros + propagated dead state), inherited.
- 43x MatmulBTQuant [17x10240] / RmsNormGated [816x128] / CastBf16 /
  AttnQkNormRopeGate / MatmulBTQuantGrouped — all downstream inheritance.
- NO slot goes live->zero without an intervening commit: the slot machinery
  and the force-reclaims are exonerated. The producers write zeros.

Config (from the TrustDump headers): hk=16 dk=128 (key_dim 2048,
tile-aligned), hv=48 dv=128, conv_dim 10240, t=17. The k2 column slice
[2048,4096) of the TILE dev_conv IS aligned, so plain misalignment is out;
the zero is in k2's content (the conv tensor's k region as seen on device)
or in l2(k2).

NEXT: probe k2 and q2 INSIDE GdnPostConvKernel (to_vector device downloads)
— if k2 is zero while the conv tensor's host bytes are live (they are:
conv_out probed 174079/174080), the defect is DeviceRows/EnsureDevice2D
serving the conv slot's stale/zero device tensor to the slice, or the slice
reading the wrong columns. Then fix per the doctrine and red-unit-test at
[T=17, hk=16, dk=128, hv=48, dv=128].

## Slices live; the zero appears BETWEEN commit and the next-layer read (2026-09-16)

[TT-KQ2] probe: conv/q2/k2/v2 ALL LIVE on device at every GdnPostConv call
(k2 34816/34816). Combined with TrustDump (call-0 k_k LIVE through
EnsureHostBytes at the commit site): the l2 chain, the commit, and the slot
download at the commit moment are all correct.

The zeros appear when a LATER step downloads the same k_out slot
([TT-DL] 86x zero). So something between step N's commit and step N+k's
download frees or rebinds the slot's stored device tensor — the freed-buffer
hypothesis, now narrowed to the k_out slot specifically.

NEXT (deterministic, no new probes needed):
1. Take a k_out slot pointer from a zero [TT-DL] in /tmp/walkback.log (which
   has TT-FREE + commits + downloads interleaved) and list EVERY event for
   that pointer in order: the commits (live), the frees ([TT-FREE] shape
   matches), and the zero download. The freeing op names itself.
2. The likely mechanism: l2()'s intermediates (rows/sq/s/denom/inv) die
   normally at return, but some OTHER op's TTReclaimPlanes force-frees a
   plane whose memory the k_out tensor's buffer aliases (allocator reuse of
   the same block), killing the slot's bytes. The [TT-FREE] shape+time
   correlation will name it.
3. Fix candidates once the freeing op is named: drop the aliasing reclaim,
   or re-order, or store a fresh copy in the slot.

## The origin moves to the GDN in-proj output; async-recycle hypothesis (2026-09-16)

The dying block's timeline (walkback.log ~3419-3495, k slot
0xfff1c524e3c0): its k_out commits were LIVE through block N, and the first
zero commit coincides with the block whose **[17,10240] GDN in-proj output
downloads as zero immediately after CausalConv1dFwd** — before the k_out
commit. The conv, state, k/v/q of that block and everything after are
inheritance.

Pattern: commit -> FIRST download already zero. Run-dependent (probes shift
it, control run dies without them), values drift (uninit). This is the
ASYNC-RECYCLE class: an output buffer is reclaimed (normal refcount
destruction, no finish()) before the queued op executes — the allocator
hands the block to a later allocation, and the op's output lands in memory
now owned elsewhere (or its input is read after its bytes were recycled).
The keep-quant chunk loop's TTReclaimPlanes calls finish() before
force-freeing, but NORMAL destruction of expression temporaries (e.g. the
l2() chain, or the model's DBuf pools) does not sync.

NEXT:
1. Probe the [17,10240] in-proj output immediately after ITS commit
   (to_vector download) — live-then-zero-later confirms recycle-after-commit;
   zero-immediately means its input was already dead (keep walking back —
   the same first-download-zero test per op).
2. If recycle confirmed: instrument normal ttnn tensor destruction around
   the suspect ops, or test the fix candidate — a queue finish() before the
   in-proj output's last reference dies (or holding it until the consumer
   syncs). Compare tt-metal's allocator contract for async-safe reuse.

## Norm-weight exonerated; the remaining paradox, precisely stated (2026-09-16, final)

[TT-NORMW]: the norm weight is live at ALL 129 calls (nz=5120/5120). And
[TT-NORMOUT] re-read: 117 of 129 norm outputs are ZERO AT COMMIT (the
earlier "all live" reading only covered the first six calls).

So: rms_norm(to_norm, eps, dev_w) emits exact zeros from call ~7 on, where
to_norm was probed live (RESADD) and dev_w is probed live (NORMW). Under
synchronous execution that is impossible — so one of: (a) to_norm's buffer
is written between the add and the rms_norm execution (an aliasing write —
find who), (b) execution is NOT synchronous (fd command queue batching) and
to_norm's buffer is recycled before rms_norm executes (the async-recycle
class), or (c) the NORMOUT/RESADD probes themselves shift the race (control
run says the death is real regardless).

NEXT SESSION — a clean minimal discriminator:
1. In RmsNormKernel, download to_norm AGAIN immediately before the rms_norm
   call (after the probes), and checksum dev_y immediately after — bracket
   the rms_norm with input+output truth in the SAME run. If input live and
   output zero in one bracketed pair, the op itself (or its dispatch) is the
   defect — take it to a minimal tt-metal repro (add -> rms_norm, same
   shapes [17,5120] bf16/f32) outside the engine.
2. Check dispatch: whether these ops run through fd_mesh_command_queue
   batching (async) — if so, test a finish() before the rms_norm as a
   workaround to confirm ordering.
3. The minimal tt-metal repro is also the red-first artifact for the fix PR.

Everything else is eliminated (see the probe chain above). The probes and
the correlation scripts are all committed on this branch.

## THE OP IS CONVICTED: ttnn::rms_norm reads a stale address from the program cache (2026-09-16)

[TT-NORMBR] bracket probe (same run, same tensors): 129 brackets —
**117 with the input downloaded LIVE immediately before rms_norm and the
output ALL ZERO immediately after**; zero brackets with a dead input; the
first live-in/zero-out at bracket #13, permanent from there (brackets 1-12
are live->live).

Mechanism that fits: the rms_norm program is cached on first use with the
input's device address; calls 2-12 reused the same buffer (addresses
stable); from call 13 the allocator hands the input a different block and
the cached program still reads the OLD, recycled (zeroed) address. This
also explains why the 0.8B suite passes (light allocator pressure, stable
addresses) and why the death is run-dependent with exact zeros (recycled
DRAM).

NEXT (red-first artifact + fix):
1. Minimal repro outside the engine: allocate A -> rms_norm(A) -> free A ->
   allocate B (different address) -> rms_norm(B) -> expect zeros (red).
2. Confirm against pinned tt-metal: check rms_norm's program-cache address
   handling (whether the input runtime arg is updated on cache hit), and
   check our call for anything pinning the input as a constant.
3. Fix candidates: bypass/update the program cache for this op at changed
   input addresses, or upstream the tt-metal fix. The engine-side workaround
   (hold the input buffer stable) is not acceptable as the fix.

## Minimal repro GREEN: the failure needs engine context (2026-09-16)

The standalone A→free→filler→B→rms_norm repro (/tmp/rms_repro, compiled
against the pinned tt-metal) is GREEN: both rms_norm calls return live
output. Simple buffer recycling at a changed address does not reproduce.

The failing configuration therefore needs more of the engine: ~12+ other
program types interleaved between rms_norm calls (program cache holding
many entries), the real op mix, or the engine's queue state. Note the
program cache is the remaining suspect: 12 rms_norm calls correct, then
permanent zeros after the cache has absorbed the first ~6 blocks' worth of
other programs — an eviction/collision/address-update bug in the cache-hit
path under cache pressure.

NEXT: reproduce with cache pressure — extend the repro (or write a second
one) that interleaves M other distinct-shape ops (add/matmul/slice at
varied shapes) between rms_norm calls until the output goes zero, then
bisect the interleaved set. Alternatively instrument the pinned tt-metal's
rms_norm compute program cache-hit path directly (print the input buffer
address baked into the program vs the actual input address at each call)
— one instrumented tt-metal build answers it definitively.

## Cache-pressure repro v2 GREEN with cache ON (2026-09-16)

Repro v2 (program cache enabled + 6 varied-shape zero/add ops per block, 32
blocks) stays green. The remaining engine ingredients the repro lacks, in
likelihood order:
1. **Dtype mirror**: the engine calls rms_norm(f32 or mixed input,
   **bf16 weight** via EnsureAffine1D's raw-bf16 cache) — v2 used f32/f32.
   Mirror: bf16 weight, input = the ADD output (ttnn::add result, not a
   from_vector upload).
2. The engine's to_norm is a chained op output (add of two device tensors),
   possibly with a slot-commit between the add and the norm.
3. The full engine program mix (quant kernels' many programs) — grow the
   interleaved set toward the real op census.

Next: mirror (1)+(2) in repro v3 (bf16 weight via from_vector of bf16,
to_norm = ttnn::add(dev_x, dev_r) with both device tensors). If v3 is still
green, bisect toward the real mix or instrument the tt-metal rms_norm
cache-hit path (baked input address vs actual) directly — that answers it
in one instrumented run.

## Repro v3 (bf16 weight + chained add) GREEN (2026-09-16)

With the corrected bf16 unit weight, v3 runs 32 blocks green. The dtype
mirror alone is insufficient. Remaining differentiator: the engine's full
program mix (keep-quant kernels' many programs) and/or its slot-commit
interleaving.

The efficient endgame: instrument the pinned tt-metal's rms_norm
compute cache-hit path directly (tt_metal/ttnn normalization rmsnorm
program factory: print the input buffer address baked into the cached
program vs the actual input address at each call). One instrumented
tt-metal build answers definitively whether the cached program reads a
stale address; then the fix lands tt-metal-side or as a cache-management
workaround. Revert the instrumentation after.

Interim user-facing note: VT_TT_KEEPQUANT... no — the workaround to test on
the engine is disabling the rms_norm program cache hit (device
program_cache off for the norm, or a cache-avoiding rms_norm variant) —
if the e2e then produces live tokens, the diagnosis is confirmed end to
end before any tt-metal change.

## Program cache exonerated (2026-09-16, VT_TT_PROGRAM_CACHE=0 lever)

Added an env opt-out around all three enable_program_cache() sites. The e2e
probe with the program cache DISABLED still produces all-zero ids. The
program cache (enablement, eviction, address staleness on cache hits) is
exonerated.

The remaining suspects are narrowed to the rms_norm EXECUTION itself on this
config/input: (a) the ttnn rms_norm compute at [17,5120] f32-in + bf16
weight + gemma... — note the engine passes a BF16 weight where the pinned
tt-metal rms_norm may expect the weight in the input's dtype or f32 (a
dtype-agnostic kernel reading bf16 bytes as f32 would produce near-zero
values: bf16 1.0 = 0x3F80 read as f32 is a denormal ~2.4e-41 ≈ 0!). CHECK
THIS FIRST: whether the engine's rms_norm weight is passed as BF16 to a
kernel whose weight dtype must match the input (f32) — a dtype mismatch
reads near-zero garbage weights → exact-zero outputs. This fits EVERYTHING
(live input, live host weight bytes, exact zeros, first calls fine if the
first calls took a different warm path).

## Weight-dtype state (2026-09-16): EnsureAffine1D stages bf16; tt-metal may support mixed

Confirmed: EnsureAffine1D (tenstorrent_ops.cpp:4016) uploads the norm weight
as BF16, so the engine calls rms_norm(f32 input, bf16 weight) — mixed
dtypes. The pinned tt-metal factory derives gamma's DataFormat from the
gamma tensor's own dtype (layernorm_op_multi_core.cpp:244) — so mixed MAY be
supported at the factory level; the kernel-level reality is unverified.

DECISIVE EXPERIMENT (one build + one ~7 min run): gate EnsureAffine1D's
staging dtype behind VT_DEBUG_SAMPLED=2-style env (or VT_TT_AFFINE_F32=1) to
upload F32 instead of BF16, rerun the probe. Live tokens ⇒ the mixed-dtype
call is the defect (fix: stage the weight in the input's dtype, or per the
tt-metal contract). Still zero ⇒ the weight dtype is exonerated too, and the
instrumented tt-metal rms_norm cache-hit path is the remaining endgame.

## Weight dtype exonerated (2026-09-16, VT_TT_AFFINE_F32=1)

The e2e probe with the affine staged F32 still produces all-zero ids. The
mixed-dtype theory is dead. The eliminations now cover every engine-side
actor around the norm: input (live), weight (live, both dtypes), program
cache, staging, commits, slot machinery.

The endgame is unambiguous: instrument the pinned tt-metal rms_norm compute
path directly — print the input buffer address baked into the dispatched
program vs the actual input address, and the compute-kernel view of the
input, at each call, in one instrumented tt-metal build, running the APEX
probe. Revert the instrumentation after. That answers whether tt-metal's
rms_norm itself computes zeros at [17,5120] from live data (an upstream
bug to fix or work around at the pin) — the only hypothesis left standing
after the full elimination chain.

## Norm pad exonerated (2026-09-16, VT_TT_NORM_PAD=1)

Padding the norm input rows to the tile height (concat + slice) before
rms_norm changes nothing — ids still all-zero. Non-tile-aligned M is
exonerated.

## SANCTION REQUEST: instrument the pinned tt-metal

Every engine-side lever is exhausted (see the full elimination chain). The
one remaining path is instrumenting ~/Sources/tt/tt-metal's rms_norm device
code (print the input buffer address baked into the dispatched program vs
the actual input address, and the kernel's view of the input, per call),
running the APEX probe once, and reverting the instrumentation. This touches
the pinned oracle source and needs the developer's go-ahead.

## tt-metal pin bump: upstream dev20260916 HANGS on basic dispatch (2026-09-17)

The minimal rms_norm repro (compiled against the new pin) HANGS on the P150 —
zero output, killed at 120s, while the same program takes ~2s on the pinned
dev20260911. The hang is in basic op dispatch, independent of our engine
entirely. Our two dispatch patches are exonerated (reverted, same hang).
Upstream regression in tenstorrent/tt-metal between v0.79.0-dev20260911 and
v0.79.0-dev20260916 (or aarch64-specific). Also: dev20260917 is aarch64-
BROKEN (x86-only streaming profiler in mesh_device.cpp, #55967).

State: branch pin-v0.79.0-dev20260916 in ~/Sources/tt/tt-metal built in
build_new (local debug mods recorded: SFPI version check → warning, SFPI
7.77.0 aarch64-debian tarball staged, tracy WASM gated via
TT_BUILD_TRACY_WASM=OFF, copy cache-hit patch ported to the new
program-factory API). Old pin preserved: branch old-pin-20260916-backup +
build_old_pin; build_Release is a symlink to build_new (swap back by
removing the symlink and restoring).

NEXT (pin-bump path): bisect upstream dev20260911..dev20260916 for the
dispatch hang (build+run per step, each ~10-20 min), or report the hang
upstream to tt-metal with this finding. INTERIM: the zero-logits bug
investigation continues on the WORKING old pin — instrument its rms_norm
path (baked vs actual input address per call) as planned.

## ROOT CAUSE FOUND: rms_norm (layernorm) program factory has no address-patching hook (2026-09-17)

The pinned tt-metal's framework contract (ttnn/api/ttnn/device_operation.hpp:~286-300):
on a program-cache HIT the framework calls `WorkloadFactory::apply_descriptor` if the
factory implements it, ELSE `WorkloadFactory::override_runtime_arguments` — if the
factory implements NEITHER, the cached program's runtime args are NEVER updated.

`LayerNormMultiCoreProgramFactory` (and the sharded factory) implement NEITHER
(ttnn/cpp/ttnn/operations/normalization/layernorm/device/ — no matches for either
hook). So the cached rms_norm program keeps the FIRST call's input/gamma/output
addresses forever. Every later call with a different input buffer reads the first
call's long-recycled (zeroed) memory → exact zeros.

This explains every observation:
- 0.8B suite green: light allocator pressure → stable addresses → first-call
  addresses stay valid.
- 27B zeros from norm call ~13: allocator pressure crosses a threshold, the input
  buffer moves, the cached program reads the old zeroed block.
- Exact zeros, run-to-run drift, first calls live.

THE FIX (tt-metal-side): implement `override_runtime_arguments` on
LayerNormMultiCoreProgramFactory (and LayerNormShardedProgramFactory) patching the
reader/writer/compute runtime args with the current input/gamma/output buffer
addresses — mirroring the copy-op patch we already carry (scratch commit
f3088579db0, ported example in the pin-v0.79.0-dev20260916 branch's copy factory).
Red-first artifact: a unit test calling rms_norm twice with inputs at different
addresses (the /tmp/rms_repro3.cpp pattern) — red on the pin, green after.
Consider reporting upstream: this operation-contract violation affects every
layernorm-family user of the framework whose buffers move.

## Fix attempt 1: the override compiled but was never called (2026-09-17)

Added `override_runtime_arguments` to LayerNormMultiCoreProgramFactory (both
declaration and definition; tt-metal rebuilt, fresh JIT cache). The APEX probe
still produced all-zero ids. Two possibilities:
1. The adapter's dispatch never reaches the factory's override: the framework
   probes specific signatures (see mesh_device_operation_adapter.hpp's
   apply_override_runtime_arguments: form (a) `cached_program_t&` + 4 args if
   the factory defines cached_program_t; form (b) `(program, shared_vars,
   attrs, tensor_args, ret)` — my 5-arg copy-style signature
   (program, attrs, tensor_args, ret, coord) matches NEITHER probe, so the
   adapter treats the factory as hookless).
2. Or the override ran and the addresses were patched but zeros persist (less
   likely — no [COPY-OVERRIDE]-style evidence was printed; my override has no
   print yet).

NEXT: give the factory a `shared_variables_t` (empty struct) and match probe
form (b)'s exact signature/order — `override_runtime_arguments(Program&,
shared_variables_t&, const LayerNormParams&, const LayerNormInputs&, Tensor&)`
— plus a debug print inside; one probe run confirms the call fires; then
UpdateProgramRunArgs with the fresh tensor args is the fix. Alternatively add
`cached_program_t` and use form (a). The m2 artifact's run_params rebuild for
the override may also need the kernel_run_args (per-core scalars) — rotary's
override shows tensor-only run_args is accepted by UpdateProgramRunArgs.

## Attempt 2: the factory override is NEVER called (2026-09-17, final)

Re-signatured the layernorm override to the adapter's form (b) —
(program, shared_variables_t&, attrs, tensor_args, ret) with an empty
shared_variables_t on the factory. Built, ran: **zero [TT-LN-OVR] prints** —
the adapter's cache-hit path never dispatches to the factory's override for
this operation. The zeros are not a missing factory refresh.

NEXT: identify which adapter variant layernorm resolves to
(resolve_program_factory in mesh_device_operation_adapter.hpp) and trace
THAT variant's cache-hit handling of tensor addresses — the refresh for this
op happens (or fails) in the adapter/framework layer, not the factory. The
program_run_args.cpp UpdateTensorArgs instrumentation (TT_METAL_TA_DEBUG=1)
never printed either, so the cache-hit path for this op reaches neither the
factory override nor UpdateTensorArgs — i.e. the cached program is enqueued
WITHOUT any runtime-arg refresh, which IS the zero-output mechanism; find
the dispatch branch that skips the refresh and why layernorm takes it.

## Attempt 3 (final, 2026-09-17): properly-built lib — the override STILL never fires

With a verified-fresh _ttnncpp.so (ninja-rebuilt after touch, containing both
the [TT-LN-OVR] print and the [TT-TA] UpdateTensorArgs instrumentation): the
probe run prints NEITHER. The layernorm factory's override_runtime_arguments
is genuinely never invoked by the framework's cache-hit path, and
UpdateTensorArgs is never reached for this op either. All-zero ids persist.

This isolates the defect precisely: the cache-hit dispatch for the rms_norm
program (LayerNormDeviceOperation via the mesh adapter) reaches NEITHER the
factory's override hook NOR the tensor-args refresh — the cached program is
enqueued with the FIRST call's runtime args, reading the first call's buffers
(recycled → zeros). The missing piece is which code path the dispatch takes
for this operation and why it skips the refresh hooks that other operations
(copy, rotary embedding-indexed) implement.

NEXT SESSION: trace the variant dispatch — mesh_device_operation_adapter.hpp
defines multiple inner adapter variants (DirectDescriptorFactory,
ProgramSpecMeshWorkloadFactoryAdapter ~line 769, CustomProgramSpec...,
MeshWorkloadSpecFactoryAdapter ~line 1049+); identify which variant
LayerNormDeviceOperation's factory resolves to (it has create_program_artifacts
+ program_factory_t variant), then read THAT variant's cache-hit path. The fix
goes wherever the refresh is skipped. All probe/build tooling is committed.

## Final narrowing (2026-09-17): rms_norm dispatches through a path with NO cache-hit refresh at all

- The standalone repro (same ttnn::rms_norm, same factory) WORKS with the same
  never-called hooks — because the allocator accidentally reuses the same
  buffer every call (fresh input allocated at the same block). The engine's
  buffers MOVE (in_addr differs per call: 470816512/475245312/471746304) →
  the never-refreshed cached program reads recycled zeros.
- The standalone repro's [TT-LN-OVR] print NEVER fires even there — the
  factory override is not part of this op's call path at all on the old pin.
- The engine installs no GraphTracker hooks (graph_capture_blocks_dispatch()
  = false), so the framework's refresh at device_operation.hpp:296 SHOULD
  run — it doesn't. Conclusion: ttnn::rms_norm on the old pin takes a
  LAUNCH PATH where neither apply_descriptor nor override_runtime_arguments
  is invoked on cache hits, and LayerNormDeviceOperation implements no
  override of its own.

NEXT SESSION:
1. Determine the exact launch path: does ttnn::rms_norm at this pin go through
   launch_operation_with_adapter (mesh adapter) or the legacy operation launch?
   (grep the rmsnorm.cpp call chain; add a temporary print in both paths.)
2. Add the address refresh in the correct place: either the DeviceOperation
   override (legacy path) or the factory hook (adapter path). The refresh body:
   patch reader/writer runtime args with the current input/gamma/output
   addresses (the copy-op pattern from f3088579).
3. Red-first artifact: /tmp/rms_repro3_oldpin extended — two rms_norm calls
   with the input at DIFFERENT addresses (allocate a filler between to force
   the move; v1's filler trick) → second call returns zeros = red. Green after
   the refresh is added.

## DECISIVE (2026-09-17, late): rms_norm does NOT take the mesh-adapter path

With a verified-fresh lib64/_ttnncpp.so containing the TT-DISPATCH print in
handle_mesh_adapter_cache_hit: zero prints. ttnn::rms_norm never enters
handle_mesh_adapter_cache_hit — the earlier "hooks never called" results were
partly a STALE-LIB64 artifact (lib64/_ttnncpp.so was a Sep-12 copy; the build
outputs land in ttnn/ and lib64 is only updated by the install/copy step),
but the conclusion stands corrected: rms_norm takes the LEGACY
tt-metal operation launch path (ttnn::prim::layer_norm ->
ttnn::device_operation::launch<LayerNormDeviceOperation> — or an older
dispatch), whose cache-hit handling differs and never refreshes runtime args
for this op.

NEXT SESSION (mechanical):
1. Instrument the legacy launch path: tt_metal/impl/program/program_cache +
   the legacy operation launch (ttnn::device_operation::launch) — print
   whether the program cache is consulted and whether any
   override_runtime_arguments is invoked on hits.
2. LayerNormDeviceOperation on this pin implements NO override; add one in
   the form the legacy path expects (check the legacy contract — likely
   `static void override_runtime_arguments(program, cached_variables, attrs,
   tensor_args, ret)` on the OPERATION class or its factory), mirroring the
   copy-op patch.
3. Red-first: the rms_norm-twice-at-different-addresses repro (/tmp/rms_repro3.cpp
   + a filler allocation between calls to force divergence).

## Adapter-variant census (2026-09-17, dispatch instrumentation)

With TT_METAL_DISPATCH_DEBUG=1: 54,388 cache-hit dispatches, ALL taking
`apply_descriptor` — but `UpdateTensorArgs` is NEVER called for ANY op in the
run (TT-TA total = 0). So every op in this engine resolves to the
`CustomProgramSpecMeshWorkloadFactoryAdapter` variant (or similar) whose
`apply_descriptor` calls the FACTORY's own `override_runtime_arguments(attrs,
tensor_args, ret, coord)` — and the layernorm factory's re-signatured form
`override_runtime_arguments(program, shared_vars, attrs, tensor_args, ret)`
does not match the CustomProgramSpec probe's expected signature
`(attrs, tensor_args, ret, coord)` — the CustomProgramSpecFactoryConcept
requires the factory to define override_runtime_arguments in the 4-arg form.

THE FIX IS NOW EXACT: re-signature LayerNormMultiCoreProgramFactory's
override to the CustomProgramSpec form —
`override_runtime_arguments(const LayerNormParams& attrs, const
LayerNormInputs& tensor_args, Tensor& ret, const
std::optional<MeshCoordinate>& coord)` returning void (or ProgramRunArgs —
check CustomProgramSpecFactoryConcept's exact return requirement at
mesh_device_operation_adapter.hpp:1009+), with the body building
ProgramRunArgs (tensor-only) and calling UpdateProgramRunArgs(program,
run_args). The shared_variables_t form-b signature was wrong for this
adapter; delete it.

## Attempt 4 (final, 2026-09-18): form-(b) override with shared_variables_t — STILL never called

Added `struct shared_variables_t {}` to LayerNormMultiCoreProgramFactory and
re-signatured the override to the exact form-(b) shape
(program, shared_vars, attrs, tensor_args, ret) with a [TT-LN-OVR] print:
still zero prints, ids still all-zero. The adapter's dispatch never reaches
the factory's override for this op, even with the correct form-(b) shape and
a fresh lib64.

The remaining trace (next session, fresh context): WHY the dispatch skips the
factory override. Candidates: (1) the resolve_program_factory/dispatch visits
map layernorm's factory to a variant whose adapter has a no-op or different
override (read the full visitor list in device_operation.hpp:239-262 and the
variant element types actually instantiated — print typeid names); (2) the
op's cached program is enqueued via a path that never calls
dispatch_to_mesh_workload_factory's hook block at all (e.g. trace-replay
enqueue, or the enqueue_mesh_workload tail re-applies the ORIGINAL run_params
— whose tensor_args hold the first call's mesh_tensor handles, whose
addresses are re-read AT ENQUEUE from the stored handles — i.e. the handles
point at the first call's buffers: the addresses are re-derived from STALE
HANDLES, which is the defect itself).

## FINAL STATE (2026-09-18, session limit): the dispatch chain has one missing link

Confirmed by TT-DISPATCH instrumentation: handle_mesh_adapter_cache_hit runs
(54,388 dispatches, all taking apply_descriptor) — but UpdateTensorArgs never
runs for ANY op, and the layernorm factory's form-(b) override never fires.
The dispatch chain is: handle_mesh_adapter_cache_hit → adapter's
override_runtime_arguments (mesh_device_operation_adapter.hpp:~285) →
apply_override_runtime_arguments (mesh_device_operation_utils.hpp:97) →
user factory's form-(a)/(b) override. One link between the adapter's
override and the user factory silently fails for layernorm.

NEXT SESSION (30 min of work, fresh context):
1. Print inside the ADAPTER's override_runtime_arguments
   (mesh_device_operation_adapter.hpp:~285) — does it fire for layernorm?
2. Print inside apply_override_runtime_arguments — which form is selected,
   and does the form-(b) call reach my LayerNormMultiCoreProgramFactory
   override?
3. The mismatch will be in the requires-probe conditions or the
   shared_variables_t resolution — fix the factory's hook to match, rebuild
   tt-metal (ninja ttnn tt_metal + COPY ttnn/_ttnncpp.so lib64/ — THE STALE
   LIB64 TRAP), and rerun.
4. Then: APEX gate rerun expecting REAL tokens → suite 77/77 → PRs → close
   ISSUE-LOCAL-01M2NSDATJQ1YNW1PA9ZBMAAM5 → bench record → stage 2 → #1003.

## Resolution (2026-09-18)

Root cause, two layers:

1. `MatmulBTQuantInt8DotKernel` bf16 arm staged the activation through
   `EnsureDevice2D` + `to_layout`, whose slot-residency fast path serves the
   producer's committed resident buffer with stale bytes at kernel read time
   (inf in the down-projection, all-zero logits downstream).
2. `MatmulBTQuantGroupedKernel` (q4_K/q6_K path) carried the identical
   `EnsureDevice2D` + `to_layout` staging, so the corruption survived the
   int8-dot fix on every grouped projection.

Fix: both arms now stage the host master directly via
`ttnn::Tensor::from_span` (bf16 span, ROW_MAJOR), bypassing slot staging.
The grouped arm caches the staged TILE tensor in `GroupedActShadow` for
trace capture; eager mode always stages fresh, and the shadow drops with
the host buffer via `UnregisterHostBuffer`. An intermediate `from_span`
attempt that passed `ttsl::Span<const uint16_t>` was itself a regression:
`from_span` derives the tensor dtype from the span element type, so
`UINT16 != BFLOAT16` triggered a numeric `to_dtype` re-encode of every
bf16 bit pattern; the span must be `const bfloat16`.

Evidence:
- Unit suite 78/78 pass (524,515 assertions), including the new bit-exact
  27B prefill-shapes test (M=256, down-proj K=17408 N=5120 and gate/up
  K=5120 N=17408, Q3_K + IQ3_XXS: exact=4456448/4456448, infs=0).
- E2e Qwen3.8-27B-APEX-I-Nano, 128-in/8-out greedy: tokens now varied
  [220,16,220,17,220,17,220,16]; the previous all-220 degenerate loop is
  gone; no inf/nan.
- Token-exact parity with the llama.cpp oracle is NOT yet achieved and
  stays open under the gate spec (ISSUE-LOCAL-01M2NMSNSS6R8T0EW1WYHHVA0Z);
  this issue closes on the all-zero-logits symptom it named.
