# DAWN-SRW Experiment Notebook

This is the canonical, Git-tracked record of material DAWN-SRW experiments.
Dates are KST. Experiment IDs are stable as `YYYY-MM-DD/#NN`.

## Current State

- Current goal: continue v4174 exact backward optimization without changing model mathematics, parameter/checkpoint schema, canonical shapes, grouped Q/K/V forward execution, or the accepted precision boundary.
- Repository state at the 2026-07-26 audit: branch `codex/v4167-poc`, GitHub HEAD `75a4fa9d3c4fdf21dbd6968427197ea4eb3ae17d`; local uncommitted changes in `models/dawn_srw_v4174.py` and `scripts/benchmark_srw_tpu.py` belong to a parallel active-edge experiment and are not part of the accepted result or this notebook-only publication.
- Active optimization config: `spatial-r1-v4.1.7.4`, `configs/train_config_v4174_400M_c4_40B_v4_64_space24_top2_bundle4.yaml`, batch 1024, sequence 512, mesh `16x2` on 32 devices. The accepted measurements below are from-scratch speed checks or benchmarks with no restored checkpoint.
- Best measured accepted result: exact selected-top-2 backward with compact exact overflow and `Tcap=3072` at commit `8187c00d4b767a743c3178a5bcb8caac1c6124c5` reached 10.181 s/it at step 10, 10.223 s/it at step 20, and 10.924 s/it with 47,902 tokens/s at step 50. This is about 32.0% faster than the matched pre-exact-backward 16.072 s/it result.
- Current conclusion: bundle4 forward remains about 2.423 s and is accepted. Exact top-2 space backward and compact overflow are also accepted. The measured QKV/RST forward-plus-backward kernels still add about 5.557 s beyond their forward calls, so the remaining target is the numerical score/gate/write gradient rather than sort, packing, or scan control.
- Current blocker: the first active-edge block-VJP lowering was exact but catastrophically slow, at 398.835 s for the detailed split. A separate owner-batched active-edge worktree is in progress and must not be attributed to a completed experiment until its owning workstream records a finished result.
- Next experiment: evaluate an exact fused active-edge backward that avoids per-edge `[edge,R]` materialization and serial task loops; retain commit `8187c00d` as the accepted fallback until a complete TPU result beats it.

## Fixed Context

- Preserve model equations, residual order, attention-then-RST routing, parameter tree/count, checkpoint and optimizer schema, loss, and all live-gradient paths.
- Preserve Q/K pairing, one grouped Q/K/V executor and writeback, `M=24`, `K=2`, `D=2048`, `R=256`, operator counts, chunk semantics, metric-only behavior, and logging cadence unless a later user decision explicitly changes them.
- FP32 control boundary: rho/tau/margin/gates, gate mass, global denominator, control scaling, and residual/state updates.
- BF16 representation boundary: read/write GEMM operands, P/U projection operands, route output, and the final model-axis D-output psum; cast the psum result back to FP32.
- Production is U-before-psum plus BF16 D-output psum. U-before-psum plus FP32 D-output psum is a numerical reference, not a configurable production mode.
- Do not solve HBM pressure by reducing batch size, sequence length, operator count, `M`, `K`, or chunk semantics, or by introducing dispatch/sparse execution, unless the user explicitly opens that scope.
- The user explicitly opened exact backward-only dispatch scope: forward remains bundle4 dense, while backward may use exact selected-top-2 spaces and exact active-edge support. No selected space, active edge, or overflow work may be dropped.
- Short TPU benchmarks and 50-step runs are sufficient for optimization screening. Require a longer steady window before production acceptance, and always report the measured window separately from compile, first-step, metric, validation, and checkpoint time.
- TPU names recorded below are historical evidence, not authorization for a future task. `AGENTS.md` controls TPU authorization and prohibits TPU creation or allocation.
- Evidence labels used below: `TPU measured`, `local validated`, `user-provided`, and `not measured`.

## Experiment Log

### 2026-07-21

#### #01 — Increase v4174 tau-summary broadcast capacity

- Status: accepted; TPU failure diagnosed, local fix validated.
- Question: can the full per-address Q/K/V/RST tau calibration summary be retained without startup failure?
- Change: raised only the tau-summary host broadcast limit from 16,384 to 65,536 bytes.
- Run: the fresh calibration summary was 22,713 bytes.
- Result: TPU had failed with `String too long for broadcast: 22713 > 16384`; a 22,713-byte local broadcast round-trip passed after the change.
- Learned: fresh v4174 calibration broadcasts the full per-address summary, while resume restores tau from checkpoint parameters and bypasses calibration.
- Decision/next: keep the full summary and the larger limit; continue through factory construction.
- Evidence: commit `13f71cf0215d9240a5fc2446bc137f642b454a80`; `scripts/train_jax.py`.

#### #02 — Preserve wrapped factory signatures

- Status: accepted; local validated.
- Question: why did sharded construction pass an unsupported `analysis` argument?
- Change: added `functools.wraps` to v4174 profile and dense-address factory wrappers.
- Result: the inherited callable signatures became visible again and production, diagnostics, and analysis-related sharded construction succeeded locally.
- Learned: generic `(*args, **kwargs)` wrappers break trainer signature inspection and can falsely advertise optional arguments.
- Decision/next: preserve underlying signatures instead of adding one-off argument aliases.
- Evidence: commit `fae0820b4966b824d18b9530b94e17274106126c`; `models/dawn_srw_v4174.py`.

#### #03 — Map v4174 diagnostics to the shared DirectTau contract

- Status: accepted; TPU error reproduced and local path validated.
- Question: why did regular logging fail after the first optimizer step?
- Change: added canonical Q/K/QK/V/RST metric mapping, QK aggregation, norms, composition aliases, and v4174 address metrics.
- Result: local sharded production/diagnostics parity passed with 48 required DirectTau metrics, 28 address metrics, and 13 composition metrics.
- Learned: a successful train step does not validate the regular console and compact logging contracts.
- Decision/next: validate startup, diagnostics, regular metrics, and compact JSONL together before deployment.
- Evidence: commit `d5969cac7256bea3f4990f62314437220ca85c32`; `models/dawn_srw_v4174.py`, `scripts/train_jax.py`.

#### #04 — Select a v4174-specific compact JSONL schema

- Status: accepted in source; local validated after a TPU logging failure.
- Question: why did compact JSONL still require v4171-only fields after the regular console block succeeded?
- Change: added v4174-specific record/output key sets and an explicit v4174 schema branch.
- Result: local production/diagnostics parity and compact JSONL generation passed. At the end of the original attempt the fix was still local while the TPU had `d5969cac`; it was subsequently committed.
- Learned: local validation, Git publication, remote deployment, and TPU execution are separate evidence states.
- Decision/next: always verify remote SHA before rerunning.
- Evidence: commit `abed852efb4ff75c6ab00d647beb95edb5c41390`; `scripts/train_jax.py`.

### 2026-07-24

#### #01 — Separate differentiable production RW from metric-only work

- Status: partial at this stage; local/static validated, canonical TPU HBM not yet measured.
- Question: can activation lifetime be reduced without changing grouped execution or model mathematics?
- Change: moved always-on differentiable RW work into a production path carrying `raw_out` and `gate_mass`; moved diagnostics into a compact stop-gradient metric-only scan without RW write GEMM, U writeback, residual, attention, or loss work.
- Result: removed full differentiable RW results from the dynamic metric conditional while preserving grouped Q/K/V reduction and writeback.
- Learned: the prior conditional retained very large route/space/token/operator tensors; metric collection must not own production-sized live buffers.
- Decision/next: retain the split and address remaining per-chunk lifetime with rematerialization.
- Evidence: commit `dc743ae3cdd33b0000e1b021d26e980ec7c91e4f`; `models/dawn_srw_v4174.py`.

#### #02 — Rematerialize every production RW chunk

- Status: accepted policy; local validated, TPU effect not yet isolated at this stage.
- Question: was conditional multi-chunk rematerialization sufficient for the remaining approximately 60 MiB shortage?
- Change: first tried `multi_chunk_only`, then fixed Q/K, V, and RST production scans to `jax.checkpoint(production_step, prevent_cse=False)` with policy `always`.
- Result: all production RW chunk bodies were consistently rematerialized while the grouped Q/K/V path and `[3,24,32768,256]` grouped result shape remained unchanged.
- Learned: a small final HBM shortage should be addressed through buffer lifetime before changing topology or shrinking the experiment.
- Decision/next: keep always-remat and clean startup-only executable/cache lifetime.
- Evidence: commits `4d1d43999635e89097e8d3b29462fbc3d8da8c9b`, `81ec117efd6cb7306f44708f0cd7e0108249368f`; `models/dawn_srw_v4174.py`.

#### #03 — Make startup cleanup and OOM probing donation-safe

- Status: partial; local and synthetic compile validation only at this stage.
- Question: can startup HBM be reduced without consuming donated live training state?
- Change: synchronized final state before releasing scratch/restore trees, added one-shot `gc.collect(); jax.clear_caches(); gc.collect()` before first train-step compilation, and changed the v4174 OOM probe to `train_step_fn.lower(...).compile()`.
- Result: the compile-only probe retained the compiled executable without executing or consuming donated params/optimizer state. Existing `donate_argnums=(0, 1)` was preserved.
- Learned: cache cleanup must occur before the final executable is loaded, and a dummy donated step is not a safe compile probe.
- Decision/next: require canonical TPU compile/HBM evidence before claiming an OOM fix.
- Evidence: commit `8b557f0aeeeb704e3c0d0f259c3e3dd9448c7f7d`; `scripts/train_jax.py`.

### 2026-07-25

#### #01 — Move FP32 model psum after U writeback

- Status: accepted as the FP32 numerical reference; local validated, TPU performance not measured for this variant.
- Question: can the FP32 `[route,M,T,R]` collective be replaced by an equivalent FP32 `[route,T,D]` collective?
- Change: kept global FP32 gate-mass psum and denominator, performed local normalization, space weighting, route scaling, and grouped U writeback, then psummed the completed FP32 D-output.
- Result: Q/K/V/RST/logit max-absolute difference was at most `1e-5`; loss difference at most `1e-6`; gradient cosine at least `0.99999`; optimizer updates passed `atol=1e-7, rtol=1e-6`; model-axis 1/2 parity passed. Two-device lowered HLO removed the R-axis numerator all-reduce.
- Learned: the reorder is valid by linearity because denominator, space weights, route scales, and U are common across model shards.
- Decision/next: keep this path as reference and lower only the completed representation collective to BF16 for production.
- Evidence: commit `d73692e6dfb55f75392837898bbfb0a5a73a1dc0`; focused local validation included 15 standard and 2 forced two-device checks.

#### #02 — Use BF16 D-output psum as canonical production

- Status: accepted; local validated and TPU measured.
- Question: does the BF16 representation boundary retain numerical/training stability while reducing communication and step time?
- Change: cast local completed D-output to BF16 immediately before model psum, then cast the summed output back to FP32. Gate mass and denominator remained FP32. FP32 D-output psum remained reference-only.
- Run: commit `14b07415f73f735e15ca9cb3098ea91923c86d38`; `spatial-se-400m`; canonical direct-read config; mesh `16x2`; 100 training steps.
- Result: local core suite passed `6 passed in 126.09s`; FP32 reorder and BF16 forward/loss/gradient/update thresholds passed. TPU completed 100 steps without NaN, Inf, OOM, or traceback; step 100 loss was 7.3192 and grad norm 2.42.
- Performance: matched raw steady windows were 28.4181 s/it at baseline steps 1401-1500 and 24.3255 s/it at production steps 51-100, a 14.40% reduction. Logged throughput increased from 18,419.1 to 21,519.1 tokens/s, or 16.83%. Compile time was 60.4 s.
- Communication/HBM: representation payload fell from 3.0 to 0.5 GiB/layer; with FP32 gate mass included, total logical payload fell from 3,084 to 524 MiB/layer. XLA buffer assignment was 29.33 of 33.01 GiB, leaving 3.68 GiB.
- Learned: the next bottleneck is not the forward representation collective. Largest operator-score buffers were approximately V 4.10 GiB, Q/K 2.62 GiB, and RST 2.06 GiB, with rematerialized backward kernels dominating cycles.
- Decision/next: retain BF16 output psum; optimize score/scan lifetime and backward execution without changing dense all-space mathematics.
- Evidence: commit `14b07415f73f735e15ca9cb3098ea91923c86d38`; remote logs `/home/madst0614/train.v4174_bf16_20260725T0245Z.log` and `/home/madst0614/train.pre_u_before_psum_20260725T024404Z.log`.
- Operational note: an initial `spatial-400m`/`spatial-se-400m` naming mistake briefly stopped the former run; its log was preserved and its original run folder was restarted before all canonical validation continued only on `spatial-se-400m`.

#### #03 — Remove deleted operator modes from v4174 resume validation

- Status: accepted; local validated, committed, pushed, and passed the former TPU schema blocker.
- Question: why did an unchanged v4174 checkpoint require legacy `operator_key_mode` and `operator_query_mode` fields?
- Change: v4174 resume compatibility now requires `d_route` but not the two deleted mode fields; v4171-v4173 checks remain unchanged.
- Result: local validation accepted a v4174 checkpoint without the legacy modes and still rejected a v4173 checkpoint missing its required mode. No config option or alias was added.
- Learned: version-owned checkpoint schema must not inherit deleted predecessor fields.
- Decision/next: retry the real step-1500 abstract resume.
- Evidence: commit `de48c42edfd1b74b606326f9fee925134064d84b`; archived failure log `/home/madst0614/train.v4174_resume_schema_mismatch_20260725T0356Z.log`.

#### #04 — Remove a Python float conversion from abstract tau initialization

- Status: accepted; local abstract restore validated, committed, pushed, and passed the former TPU initializer blocker.
- Question: why did abstract checkpoint-target construction fail after the schema check was fixed?
- Change: removed the outer Python `float()` conversion around the JAX scalar returned by `_raw_tau_init_from_cosine_tau`.
- Result: the full v4174 abstract parameter tree constructed successfully with exactly 393,755,652 parameters.
- Learned: shape/dtype-only abstract restore paths must not materialize traced JAX scalars through Python numeric conversion.
- Decision/next: retry the same checkpoint without changing model math or checkpoint values.
- Evidence: commit `b046f3bb560c02e8209a7c644dafb26e0807688b`; archived failure log `/home/madst0614/train.v4174_abstract_resume_error_20260725T0405Z.log`.

#### #05 — Resume the canonical checkpoint at step 1500

- Status: partial; restore and multihost agreement measured, first resumed optimizer step not captured in the available evidence.
- Question: do the two minimal resume fixes restore the existing canonical run without altering state?
- Run: `spatial-se-400m`, 8 hosts/32 devices, commit `b046f3bb`, canonical direct-read config, Orbax step 1500.
- Result: params, full optimizer state, and RNG restored directly into final mesh shardings; `global_step=1500`, optimizer count 1500, consumed tokens 786,432,000, and host agreement passed across 8 hosts. Parameter count remained 393,755,652 and all training processes reached first train-step XLA compilation.
- Learned: schema and abstract-target construction were the blockers; checkpoint contents and model topology did not require migration.
- Decision/next: a later entry must record the first completed resumed step and subsequent trend before calling the resumed run fully validated.
- Evidence: run folder `gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4174_400M_c4_40B_v4_64_space24_top2_direct_read/run_vspatial-r1-v4.1.7.4_20260724_223010_3201`; observed resume log timestamp `20260725_041101`.

#### #06 — Introduce exact four-space compact bundle execution

- Status: accepted structural baseline; TPU measured.
- Question: can the union of selected top-2 spaces be executed as fixed four-space bundles while preserving exact forward mathematics and grouped Q/K/V behavior?
- Change: added the canonical bundle4 config and compact executor, grouped Q/K/V packing/writeback, bundle metrics, trainer support, and benchmark profiling.
- Run identity: branch `codex/v4167-poc`; effective source commit `213001e3967d76d4c625b85390d2d2e6b0f8f7ee`; bundle4 config; from scratch, no checkpoint; batch 1024, sequence 512, mesh `16x2`/32 devices; TPU checkout was dirty from direct file deployment.
- Result: physical compute fell to about 31.1% of the all-space dense path. With block 2048, forward was approximately 2.632 s and training was approximately 17 s/it. Increasing block 1024 to 2048 improved forward by about 6.3% and train-step time by about 18% with little HBM change.
- Learned: the compact topology reduced arithmetic successfully, but packing, scatter, block control, and rematerialized bundle work still dominated the remaining gap.
- Decision/next: retain bundle4 and test prefix-count packing, scan unroll, and compact residual save policies before writing a custom kernel.
- Evidence: commit `213001e3967d76d4c625b85390d2d2e6b0f8f7ee`; the original `~/train.log` was overwritten by later pre-notebook runs, so program HBM for this exact attempt is not preserved.

#### #07 — Remove stable sort and test scan unroll/remat residuals

- Status: accepted only for prefix-count packing and compact RW residual save; unroll rejected.
- Question: are stable sort, packing metadata recomputation, or scan control the dominant remaining costs?
- Change: replaced stable sorting with exact prefix-count packing, reused saved metadata, tested block/RST scan unroll, and saved only compact `raw_out`/`gate_mass` at the remat boundary.
- Run identity: branch `codex/v4167-poc`; final effective source commit `ff62aed8f529b6bf1e02be1ff9ae052b9291fa4a`; same bundle4 config, from scratch/no checkpoint, batch 1024, sequence 512, mesh `16x2`/32 devices; intermediate unroll variants were dirty direct deployments.
- Result: sort removal improved trainer time by only about 0.5-0.95%. Preserved fast-profile results were: block-unroll 1 forward 2.4291 s, QKV 1.1220 s, RST 0.9304 s; block-unroll 2 forward 2.4607 s; RST-unroll 1 forward 2.4309 s. The preserved unroll-2 train benchmark averaged 15.0989 s over its configured short window.
- Learned: generic sort and scan control were not the primary cost after block 2048; unrolling could worsen live ranges and scheduling.
- Decision/next: keep prefix-count packing, unroll 1, and compact RW save policy; move to a backward-only exact selected-space executor.
- Evidence: commit `ff62aed8f529b6bf1e02be1ff9ae052b9291fa4a`; remote JSONL files under `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_prefix_block_u1_fast/`, `v4174_prefix_u2_fast/`, `v4174_prefix_u2_rst_u1_fast/`, and `v4174_prefix_u2_train/`.

#### #08 — Build exact selected-top-2 backward and reject serial schedules

- Status: exact mathematics accepted locally; first two TPU schedules rejected.
- Question: can forward remain bundle4 dense while backward computes gradients only for the exactly selected two spaces per token?
- Change: added a custom VJP with exact prefix-packed token-space pairs, exact routing-weight scatter, shared Q/K/V projection VJP, and static 24-space padded dense buckets.
- Run identity: branch `codex/v4167-poc`; final implementation commit `09c39397789d85d3e0546a48fa7ace9b355b1d6c`; same bundle4 config, from scratch/no checkpoint, batch 1024, sequence 512, mesh `16x2`/32 devices. The serial and fixed-overflow schedules were intermediate dirty deployments whose exact diffs were not preserved.
- Local validation: forced `Tcap=2` overflow produced attention output max error 0, attention gradient max error `2.384e-7`, RST output max error `3.725e-9`, and RST gradient max error `1.863e-7`; no pair was dropped.
- TPU result: a serial exact-space scan reached 14.883 s/it at step 10. Batched primary `Tcap=4096` with the old fixed 56-block overflow scan reached 9.328 s/it at step 5 but degraded to 14.627 s/it at step 10 whenever overflow activated the full scan.
- Learned: exact selected-space arithmetic was beneficial, but any fixed global overflow scan erased the gain.
- Decision/next: retain the exact custom VJP and replace fixed overflow with compact exact overflow tasks.
- Evidence: commit `09c39397789d85d3e0546a48fa7ace9b355b1d6c`; the intermediate raw `~/train.log` was overwritten before the notebook policy, so these two schedules are recorded from the TPU task transcript only.

#### #09 — Use compact exact overflow with `Tcap=4096`

- Status: accepted improvement; TPU measured through step 50.
- Question: does scheduling only actual overflow restore the fast primary batched path as routing skews?
- Change: replaced the unconditional 56-block overflow scan with compact exact overflow tasks processed four at a time; overflow remained lossless.
- Run identity: branch `codex/v4167-poc`; commit `09c39397789d85d3e0546a48fa7ace9b355b1d6c`; same bundle4 config, from scratch/no checkpoint, batch 1024, sequence 512, mesh `16x2`/32 devices; effective model matched the commit while the TPU checkout remained dirty from direct deployment.
- Result: step times were 10.515, 9.211, 10.305, 10.414, and 11.308 s/it at steps 1, 5, 10, 20, and 50. Step-50 throughput was 46,275.5 tokens/s; attention/RST top-1 fractions were 0.768/0.805.
- Learned: compact overflow removed the 14.627 s collapse, but `Tcap=4096` lost efficiency as the selected-space distribution skewed.
- Decision/next: retain compact exact overflow and A/B rematerialization plus smaller bucket capacities.
- Evidence: commit `09c39397789d85d3e0546a48fa7ace9b355b1d6c`; raw training log later overwritten under the pre-notebook workflow.

#### #10 — Reject no-remat variants and select `Tcap=3072`

- Status: `Tcap=3072` with remat accepted; no-remat variants rejected.
- Question: can saved dense residuals remove recomputation, and which bucket capacity best handles the long-run selected-space skew?
- Run identity: branch `codex/v4167-poc`; final source commit `8187c00d4b767a743c3178a5bcb8caac1c6124c5`; same bundle4 config, from scratch/no checkpoint, batch 1024, sequence 512, mesh `16x2`/32 devices; Python 3.10.12, JAX 0.6.2, Flax 0.10.7.
- Failed variants: all-pool no-remat at `Tcap=4096` failed compilation at 33.10 GiB used versus a 30.75 GiB limit, with 30.33 GiB program HBM and a 2.06 GiB RST stack allocation. QKV-only no-remat passed but slowed step 10 to 10.658 s. All-pool no-remat at `Tcap=2048` passed but reached only 12.247 s at step 5 because of excess overflow waves.
- Accepted result: rematerialized `Tcap=3072` reached 11.994, 10.237, 10.181, 10.223, and 10.924 s/it at steps 1, 5, 10, 20, and 50. Step-50 throughput was 47,902.3 tokens/s, loss 8.7007, and grad norm 8.77. It was 3.4% faster than `Tcap=4096` at step 50 and about 32.0% faster than the matched pre-exact-backward 16.072 s/it result.
- Learned: retaining remat is necessary; `Tcap=3072` balances primary MXU density against compact overflow better than 2048 or 4096 in the measured skewed regime.
- Decision/next: make commit `8187c00d` the accepted fallback and profile the exact numerical backward before attempting active-edge lowering.
- Evidence: commit `8187c00d4b767a743c3178a5bcb8caac1c6124c5`; raw training log later overwritten under the pre-notebook workflow.

#### #11 — Measure exact QKV/RST forward and backward kernels

- Status: accepted benchmark instrumentation; TPU measured.
- Question: after exact selected-space packing, how much time remains in actual QKV/RST numerical backward?
- Change: added opt-in `--fast --backward-profile`, using real custom VJPs and a device-side full-gradient checksum to prevent gradient DCE or host transfer.
- Run identity: branch `codex/v4167-poc`; model commit `8187c00d4b767a743c3178a5bcb8caac1c6124c5`, profiler commit `507d3a4a9bf9abc8a439b8ed525e8b7313dcfd98`; same bundle4 config, no checkpoint, batch 1024, sequence 512, mesh `16x2`/32 devices; Python 3.10.12, JAX 0.6.2, Flax 0.10.7.
- Result: full forward was 2.4226 s. Across 18 layers, QKV forward was 1.1236 s and forward-plus-backward 3.7754 s; RST forward was 0.9314 s and forward-plus-backward 3.8370 s. The independently inferred backward increments were 2.6518 s and 2.9056 s, or 5.5574 s combined. Detailed split time was 9.9006 s.
- Memory: runtime live-array peak was 4.9606 GiB; XLA program HBM was not measured by this benchmark and must not be inferred from that telemetry.
- Learned: sort, packing, and control are closed as primary hypotheses. The remaining gap is score/gate/write plus P/RW/U gradient execution.
- Decision/next: test exact active-edge backward only if it avoids per-edge feature materialization and serial reductions.
- Evidence: commit `507d3a4a9bf9abc8a439b8ed525e8b7313dcfd98`; `/home/madst0614/benchmark_v4174_backward.jsonl`.

#### #12 — Reject block-serial active-edge custom VJP

- Status: rejected; TPU measured.
- Question: can exact positive-margin edge support reduce the dense selected-space backward further?
- Change: a dirty experimental custom VJP packed active edges and replayed them through operator/token edge blocks. The exact dirty source was subsequently evolved and was not committed, so this attempt is not reproducible from a Git SHA.
- Run identity: branch `codex/v4167-poc`; base history through profiler commit `507d3a4a`; dirty worktree; same bundle4 config, no checkpoint, batch 1024, sequence 512, mesh `16x2`/32 devices; Python 3.10.12, JAX 0.6.2, Flax 0.10.7.
- Result: forward remained 2.4231 s, but detailed split time rose to 398.8351 s. QKV forward-plus-backward totaled 206.2843 s and RST 190.2427 s across 18 layers; compile took 597.8101 s. Runtime live-array peak was 5.0804 GiB.
- Learned: edge-count reduction is irrelevant if implemented as serial dynamic gather/scatter and segment work; the lowering was roughly two orders of magnitude slower than the accepted dense exact-top-2 VJP.
- Decision/next: reject block-serial edge replay. Any successor must use owner-batched/fused reductions and receive a short single-layer TPU gate before a complete 18-layer profile.
- Evidence: `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_active_edge_block_vjp/benchmark_metrics.jsonl`; source identity limitation recorded above.

### 2026-07-26

#### #01 — Parallel owner-batched active-edge attempt remains in progress

- Status: in progress and owned by a parallel workstream; no completed result is accepted here.
- Run identity at notebook audit: branch `codex/v4167-poc`, HEAD `75a4fa9d3c4fdf21dbd6968427197ea4eb3ae17d`, dirty `models/dawn_srw_v4174.py` SHA-256 `e27de9ab404a620f502394b271cb34703f619226351d751eaf43ce2abc01eebe`, dirty `scripts/benchmark_srw_tpu.py` SHA-256 `287194cd19ab4a069610f221bdc5c81767a6193b6e5842919f00a1d124dd7e10`, config SHA-256 `4e0dc1188c6489ac28e22f710f685c7b4554f0869557876f4205671ba150ef8f`; no checkpoint, batch 1024, sequence 512, mesh `16x2`/32 devices; Python 3.10.12, JAX 0.6.2, Flax 0.10.7.
- Partial evidence: forward measured 2.4248 s. A later-layer QKV forward-plus-backward call was observed at 41.537 s, but the full benchmark did not complete, so aggregate performance, exactness, and program HBM are `not measured`.
- Operational correction: during this notebook audit the parallel benchmark was mistakenly interrupted. Source files were not changed; the attempted Git restore failed before modifying them. The partial log and dirty source snapshot were preserved under `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_active_edge_dual_owner_xla/`. Restart was not performed because the user restricted this task to recordkeeping.
- Decision/next: the owning parallel workstream must decide whether to resume or replace this attempt and append a later correction/result entry. Do not infer acceptance or rejection from the partial timing alone.

## Backfill Boundary

- The initial notebook covered evidence-backed work from 2026-07-21 through the step-1500 resume attempt.
- This update backfills compact-bundle through exact-top-2 backward, bucket selection, detailed backward profiling, and the first rejected active-edge lowering.
- Intermediate dirty variants whose exact diffs or raw logs were overwritten before this policy are explicitly marked as such instead of being presented as fully reproducible runs.
