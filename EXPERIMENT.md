# DAWN-SRW Experiment Notebook

This is the canonical, Git-tracked record of material DAWN-SRW experiments.
Dates are KST. Experiment IDs are stable as `YYYY-MM-DD/#NN`.

## Current State

- Current goal: screen end-to-end P-RW-U rematerialization inside the accepted exact selected-top-2 dense backward without changing model mathematics, parameter/checkpoint schema, canonical shapes, grouped Q/K/V execution, or the accepted precision boundary.
- Publication state: branch `codex/v4167-poc`, GitHub HEAD `1d74f359b6bc082224ff93f10e1cead7465fba94`; rejected exact-owner source remains preserved at commit `4cacfe39`, fusion source is at `fd7e5e32`, and the current HEAD also pins the remote SSH user. The candidate is not accepted until TPU measurement.
- Active optimization config: `spatial-r1-v4.1.7.4`, `configs/train_config_v4174_400M_c4_40B_v4_64_space24_top2_bundle4.yaml`, batch 1024, sequence 512, mesh `16x2` on 32 devices. The accepted measurements below are from-scratch speed checks or benchmarks with no restored checkpoint.
- Best measured accepted result: exact selected-top-2 backward with compact exact overflow and `Tcap=3072` at commit `8187c00d4b767a743c3178a5bcb8caac1c6124c5` reached 10.181 s/it at step 10, 10.223 s/it at step 20, and 10.924 s/it with 47,902 tokens/s at step 50. This is about 32.0% faster than the matched pre-exact-backward 16.072 s/it result.
- Current conclusion: bundle4 dense forward and exact selected-top-2 space backward remain accepted. Exact operator sparsity did not translate into TPU speed: dual-owner XLA tasks were extremely slow, Mosaic Pallas could not lower the required indirect gathers/reductions, static grouped XLA was slower still, and saving read scores duplicated enough metadata traffic to regress both QKV and RST.
- Current blocker: the fixed-user retry reached the correct clean commit as UID 2001 but exited because the failed UID 2002 process had left `/tmp/libtpu_lockfile`; the exact stale lock was removed on all workers after verifying zero accelerator holders.
- Next experiment: relaunch the accepted checkpoint with holder-verified stale-lock cleanup, confirm one completed restored step, then require a bounded one-to-two-layer fusion detailed-profile gate against `8187c00d`; reject the candidate on a material regression and leave accepted checkpoint training resumed afterward.

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

#### #02 — Complete the dual-owner audit and reject its XLA task schedule

- Status: rejected for performance; local exactness validated and TPU partially measured before external interruption.
- Hypothesis/change: prefix-pack positive-margin support into one shared capacity, compute read/write gradients in operator-owned dense blocks, compute local/tau gradients in token-owned dense blocks, and replay every overflowing owner exactly with drop count zero.
- Run identity: TPU checkout branch `codex/v4167-poc`, remote base HEAD `b046f3bb560c02e8209a7c644dafb26e0807688b`, dirty model SHA-256 `e27de9ab404a620f502394b271cb34703f619226351d751eaf43ce2abc01eebe`, benchmark SHA-256 `287194cd19ab4a069610f221bdc5c81767a6193b6e5842919f00a1d124dd7e10`, config SHA-256 `4e0dc1188c6489ac28e22f710f685c7b4554f0869557876f4205671ba150ef8f`; no checkpoint, batch 1024, sequence 512, mesh `16x2`/32 devices; remote Python 3.10.12, JAX 0.6.2, Flax 0.10.7.
- Local validation: in `.venv-jax062` with Python 3.12.7, JAX 0.6.2, and Flax 0.10.7, `margin == 0` matched dense output and all gradients exactly. A forced overflow with capacity 24 for 256 dense edges matched output exactly with maximum gradient relative error `5.5334e-7`.
- TPU result: forward remained 2.4248 s. First QKV/RST forward-plus-backward compile-and-run calls were 128.806/205.477 s. Cached QKV calls across completed layers were 41.125-41.559 s with mean 41.341 s; cached RST calls were 72.062-72.417 s with mean 72.202 s. The run reached layer 9 QKV and was interrupted during layer 9 RST, so a full 18-layer aggregate and program HBM are `not measured`.
- Lesson: exact edge reduction alone is insufficient; the dual owner task graph, gathers, loops, and reductions dominate TPU execution.
- Decision/next: reject this schedule and keep accepted commit `8187c00d` as the performance fallback. Preserve the exact source as a reproducible experimental checkpoint, not as an accepted training result.
- Evidence: `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_active_edge_dual_owner_xla/interrupted_20260726_004200_kst/`.

#### #03 — Reject Mosaic Pallas owner kernels at lowering

- Status: rejected; TPU compiler probes failed before a valid performance measurement.
- Hypothesis/change: retain dense bundle forward and replace only the exact-owner backward row-dot and owner reductions with operator-major/token-major Mosaic Pallas kernels.
- Run identity: same config, no checkpoint, batch 1024, sequence 512, mesh `16x2`/32 devices, remote Python 3.10.12, JAX 0.6.2, Flax 0.10.7; the probes were dirty direct deployments on remote base HEAD `b046f3bb`. Exact intermediate source hashes were overwritten before notebook finalization.
- Result: successive lowerings failed on the TPU last-two-dimension block-alignment rule, a Triton-only implicit `broadcast_to`, `Cannot do int indexing on TPU` for indirect edge gathers, a BF16 minor-singleton shape cast, `SupportsVectorPermuteBetweenSublane`, and finally `limits[i] <= dim(i) (128 vs. 1)` for a singleton MXU output. Retiling to 128 did not remove the sublane failure. Runtime speed and HBM are `not measured`.
- Lesson: this owner layout requires indirect gathers and reduction shapes that Mosaic TPU cannot lower efficiently or legally; this is a deterministic compiler/layout limitation rather than compile-cache corruption.
- Decision/next: reject the Pallas path and test only contiguous/static XLA owner blocks as a fallback.
- Evidence: the experimental Pallas implementation is preserved in `models/dawn_srw_v4174.py`; the transient raw `~/train.log` was overwritten before this entry, so the exact error signatures above are the surviving evidence.

#### #04 — Reject contiguous static grouped-XLA owner blocks

- Status: rejected; TPU measured with a finite two-layer gate.
- Hypothesis/change: move all indirect edge gathering back to XLA and execute contiguous static owner blocks, avoiding Mosaic indexing while retaining exact shared-capacity overflow.
- Run identity: remote base HEAD `b046f3bb560c02e8209a7c644dafb26e0807688b`, dirty model SHA-256 `471ae4f480aad48d2bf4340efa74463eb1b7c70704d7fcb7e514beed9a26fe05`, benchmark SHA-256 `03c32e30be954caee0b18264824fdeed3ba7d2e13d0e6f0e0cb839e38c8ff58f`, config SHA-256 `4e0dc1188c6489ac28e22f710f685c7b4554f0869557876f4205671ba150ef8f`; no checkpoint, batch 1024, sequence 512, mesh `16x2`/32 devices; Python 3.10.12, JAX 0.6.2, Flax 0.10.7.
- Result: forward was 2.4282 s. First QKV/RST compile-and-run calls were 139.690/245.434 s and cached layer-1 calls were 59.354/99.108 s. The measured two-layer QKV/RST forward-plus-backward totals were 118.9738/197.8045 s, detailed compile was 576.619 s, and runtime live-array peak was 5.530 GiB. Per layer this was about 43.9%/37.0% slower than the dual-owner XLA schedule.
- Lesson: converting sparse ownership to static XLA scans increases executed task tiles and live state enough to overwhelm any indexing simplification.
- Decision/next: reject static grouped XLA and isolate whether saving the dense-forward read score can remove the dominant repeated dot without changing the exact owner schedule.
- Evidence: `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_active_edge_static_group_xla_gate/completed_20260726_023203_kst/`.

#### #05 — Reject duplicated saved-read metadata

- Status: rejected; local exactness and TPU two-layer performance measured.
- Hypothesis/change: preserve dense forward, pack the forward read score beside each operator-major and token-major primary edge, and skip backward read-dot reconstruction while retaining exact overflow recomputation.
- Run identity: remote base HEAD `b046f3bb560c02e8209a7c644dafb26e0807688b`, dirty model SHA-256 `413efea55e67711bfa5460969887854b5940eeb172cb5fbcc637d30270c5989d`, benchmark SHA-256 `03c32e30be954caee0b18264824fdeed3ba7d2e13d0e6f0e0cb839e38c8ff58f`, config SHA-256 `4e0dc1188c6489ac28e22f710f685c7b4554f0869557876f4205671ba150ef8f`; no checkpoint, batch 1024, sequence 512, mesh `16x2`/32 devices; Python 3.10.12, JAX 0.6.2, Flax 0.10.7.
- Local validation: `margin == 0` matched dense output and all gradients exactly; forced exact overflow had maximum relative gradient error `5.5334e-7`; a representative BF16 production-shape comparison had minimum gradient cosine `0.999993`.
- TPU result: forward remained 2.4264 s. First QKV/RST compile-and-run calls were 111.111/217.945 s; cached layer-1 calls were 48.402/78.710 s. Two-layer QKV/RST forward-plus-backward totals were 96.8341/157.2556 s, detailed compile was 489.425 s, and runtime live-array peak was 5.365 GiB. Cached QKV/RST regressed about 17.1%/9.0% versus the unsaved dual-owner schedule.
- Lesson: duplicating FP32 read scores into both owner orderings costs more residual packing and memory traffic than recomputing the compact read dots.
- Decision/next: reject saved-read metadata and restore exact-owner source SHA-256 `e27de9ab404a620f502394b271cb34703f619226351d751eaf43ce2abc01eebe` on the local checkout and all eight TPU workers.
- Evidence: `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_active_edge_exact_owner_saved_read_gate/completed_20260726_saved_read/`.

#### #06 — Retain bounded detailed-profile gates

- Status: accepted benchmark instrumentation; used in completed TPU gates.
- Hypothesis/change: add `--detailed-profile-layers N` with `0` meaning all layers and emit start/done timing for each compile/profile operator so a failing schedule can finish a bounded gate without manual termination.
- Run identity: benchmark SHA-256 `03c32e30be954caee0b18264824fdeed3ba7d2e13d0e6f0e0cb839e38c8ff58f`; exercised by experiments #04 and #05 with the same config, no checkpoint, batch 1024, sequence 512, and mesh `16x2`/32 devices.
- Result: both two-layer gates completed normally, emitted separate QKV/RST compile and cached timings, wrote metrics JSONL, and exited without leaving tmux sessions, benchmark processes, or TPU holders.
- Lesson: bounded layer gates expose compile/runtime regressions early and avoid ambiguous manual kills during long detailed profiles.
- Decision/next: retain the profiler change. Active-edge screening is closed; any future operator-sparse design must first pass this two-layer gate before a full profile.
- Evidence: `scripts/benchmark_srw_tpu.py` and the completed archive directories recorded in experiments #04 and #05.

#### #07 — Relaunch the accepted fallback from the latest canonical checkpoint

- Status: in progress; launcher completed, first restored optimizer step not yet measured.
- Hypothesis/change: keep `spatial-se-400m` occupied with the fastest accepted code while the next candidate is prepared, rather than leaving the pod idle on the rejected experimental HEAD.
- Run identity: existing TPU `spatial-se-400m`; branch `codex/v4174-accepted-8187`; commit `8187c00d4b767a743c3178a5bcb8caac1c6124c5`; bundle4 config; resume source `gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4174_400M_c4_40B_v4_64_space24_top2_direct_read/run_vspatial-r1-v4.1.7.4_20260724_223010_3201`; latest checkpoint step `1681`; batch 1024, sequence 512, mesh `16x2`/32 devices.
- Result: all eight SSH preflights and cleanup checks passed, dependency/repository setup completed, and `scripts/train_jax.py` was started in `tmux train` with combined output in `~/train.log`. A real restored step is still `not measured`.
- Lesson: the previously dirty remote checkout at base `b046f3bb` was unsuitable for continuity; the exact accepted commit is now available as a clean remote branch.
- Decision/next: confirm restore identity and at least one completed donated optimizer step before replacing this run with the bounded fusion gate.
- Evidence: worker logs `/home/madst0614/train.log`; overwritten prelaunch logs preserved as `/home/madst0614/train.pre_fusion_resume_20260726T0326KST.log`.

#### #08 — Build an end-to-end P-RW-U rematerialized exact backward candidate

- Status: candidate; minimum local validation passed, TPU performance not measured.
- Hypothesis/change: retain the exact top-2 space buckets, dense MXU-friendly RW arithmetic, compact exact overflow, and all gradients, but wrap each attention/RST exact-space P-RW-U primal in one `nothing_saveable` rematerialization boundary. Exact overflow task groups use the same policy so projection, RW, denominator, U writeback, and their pullback intermediates are recomputed together instead of retained across AD boundaries.
- Run identity: branch `codex/v4167-poc`; parent GitHub HEAD `4cacfe3987e881c21881373f49fcb5903ee9c916`; model restored from accepted commit `8187c00d` before the fusion-only edit; candidate model SHA-256 `709a0571ab3c510748c80a95e87d4caec788e1bce134f67b4e7b60f063278ae2`; canonical bundle4 config; local Python 3.12.7, JAX 0.6.2, Flax 0.10.7.
- Result: `.venv-jax062` `py_compile`, import, policy identity (`nothing_saveable`), and `git diff --check` passed. Numerical equivalence and TPU speed/HBM are `not measured`.
- Lesson: this candidate changes residual lifetime only. It does not introduce active-edge packing, indirect gather, a new parameter/config field, or a claim that XLA emits one physical kernel.
- Decision/next: publish the exact source, run the bounded TPU detailed-profile gate, and continue to a short training screen only if compile/runtime are competitive with `8187c00d`.
- Evidence: `models/dawn_srw_v4174.py`; this notebook entry.

#### #09 — Pin the required remote SSH user in training and benchmark control paths

- Status: root cause confirmed; local launcher validation passed, corrected relaunch pending.
- Hypothesis/change: the launcher, benchmark launcher, watcher, and failure grep must address `madst0614@<TPU>` explicitly instead of allowing gcloud to infer the Windows caller name.
- Run identity: existing TPU `spatial-se-400m`; failed accepted-resume source commit `8187c00d4b767a743c3178a5bcb8caac1c6124c5`; local control source parent `fd7e5e32f6c45ea421df19ecb0df18feb207f1a4`; bundle4 config; requested resume checkpoint step `1681`; eight hosts/32 devices.
- Result: the first launch started all eight processes as UID 2002 user `MADST`, while `/tmp/tpu_logs` and the canonical home/checkpoint environment belonged to UID 2001 `madst0614`. Every host repeated `Could not open ... /tmp/tpu_logs/...MADST...: Permission denied`; no restored optimizer step completed. Fixed-user edits in `launch_tpu_pod.sh`, `launch_srw_benchmark_tpu_pod.sh`, `watch_tpu_logs.sh`, and `grep_tpu_logs.sh` passed Git Bash `bash -n` and `git diff --check`.
- Lesson: supplying `madst0614@` in ad hoc inspection commands is insufficient when the launcher and watcher internally call gcloud with a bare TPU name; the user identity must be fixed at every current training/benchmark control boundary.
- Decision/next: preserve the failed log, terminate only the UID 2002 run, publish the control-path fix, and relaunch the accepted branch as UID 2001 before any fusion benchmark.
- Evidence: bad-user logs under `/home/MADST/train.log`; UID/ownership inspection on worker 0 at 2026-07-26 KST; the four launcher/watcher files above.

#### #10 — Remove a stale libtpu lock only after holder verification

- Status: root cause confirmed; live cleanup succeeded, corrected launcher code pending publication and relaunch.
- Hypothesis/change: process cleanup must verify that `/dev/accel*` has no remaining holder and only then remove `/tmp/libtpu_lockfile`; both training and benchmark launchers should abort if a holder remains.
- Run identity: existing TPU `spatial-se-400m`; accepted branch `codex/v4174-accepted-8187` at `8187c00d4b767a743c3178a5bcb8caac1c6124c5`; control source parent `1d74f359b6bc082224ff93f10e1cead7465fba94`; bundle4 config; requested checkpoint step `1681`; eight hosts/32 devices.
- Result: the UID 2001 retry cloned a clean repository and printed the exact accepted SHA, but every worker exited from JAX initialization with `ABORTED: The TPU is already in use by another process`. Inspection showed no `/dev/accel0` holder and a zero-byte `/tmp/libtpu_lockfile` owned by the failed UID 2002 user from 18:33 UTC. After closing only the failed `tmux train` wrappers, all eight workers independently verified zero holders and removed that exact lockfile.
- Lesson: killing the previous process is insufficient when libtpu leaves its lockfile behind; lock removal is safe only after an explicit zero-holder gate.
- Decision/next: publish the fail-loud cleanup sequence, relaunch accepted training, and require one real restored step before the fusion gate.
- Evidence: `/home/madst0614/train.log` on the fixed-user retry; live `lsof` and lock ownership inspection on all workers; `scripts/launch_tpu_pod.sh`; `scripts/launch_srw_benchmark_tpu_pod.sh`.

#### #11 — Make the bounded v4174 backward gate a first-class launcher path

- Status: implemented and locally shell-validated; TPU execution pending completion of the accepted-resume control.
- Hypothesis/change: the existing benchmark launcher should carry `--backward-profile` and `--detailed-profile-layers` directly, accept the canonical v4174 model identity, and require an explicit existing TPU name instead of retaining a default resource target. The training launcher now also rejects a missing `--tpu`.
- Run identity: branch `codex/v4167-poc`; source parent `521be1be9c2eb4de53fdc1f12d9dae34265f0fd5`; no model/config/checkpoint execution in this implementation entry.
- Result: Git Bash `bash -n` passed for both launchers and `git diff --check` passed. No Python model test was added or run.
- Lesson: a bounded TPU profile must be expressible through the same fixed-user, `tmux train`, `~/train.log`, fail-loud cleanup path as training; an implicit resource target is incompatible with explicit experiment identity.
- Decision/next: publish this launcher path, then run the fusion candidate with `--fast --backward-profile --detailed-profile-layers 2 --no-xla-dump`.
- Evidence: `scripts/launch_tpu_pod.sh`; `scripts/launch_srw_benchmark_tpu_pod.sh`; local shell validation on 2026-07-26 KST.

#### #12 — Verify the accepted checkpoint resume before profiling

- Status: control passed; intentionally replaced by the matched baseline gate after 19 completed optimizer steps.
- Hypothesis/change: the fixed-user and holder-verified lock cleanup should restore the canonical accepted run directly into final mesh shardings and advance beyond emergency checkpoint step 1681 without a resume-only HBM failure.
- Run identity: existing TPU `spatial-se-400m`; branch `codex/v4174-accepted-8187`; commit `8187c00d4b767a743c3178a5bcb8caac1c6124c5`; config `configs/train_config_v4174_400M_c4_40B_v4_64_space24_top2_bundle4.yaml`; resume folder `gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4174_400M_c4_40B_v4_64_space24_top2_direct_read/run_vspatial-r1-v4.1.7.4_20260724_223010_3201`; clean checkout; batch 1024, sequence 512, mesh `16x2`/32 devices.
- Result: all eight hosts initialized as `madst0614`; process 0 was worker 6. Orbax restored `global_step=1681`, `opt_state_count=1681`, RNG, and final parameter/optimizer shardings consistently across all hosts. Training reached step 1700 with loss/CE 5.1633, grad norm 28.98, and no OOM or process loss. The reported 28.045 s/it and 18,658 tok/s cover a 533.952 s cold window containing first JIT, so they are not a steady-state performance measurement.
- Lesson: the canonical accepted checkpoint and resume path are healthy once remote-user identity and stale libtpu state are corrected; the fusion decision can now be isolated to matched profiler evidence.
- Decision/next: compare clean accepted model commit `001c2c27fd7baf6998937530cdca3e55e57fccf0` and fusion commit `1ff2e57b811175a875795c85f7c2f14a7d71bb0b` with identical real-data two-layer forward/backward gates.
- Evidence: `/home/madst0614/train.accepted_resume_control_20260726T040810KST.log` on all eight workers; primary worker 6 `~/train.log` before the gate launch.

## Backfill Boundary

- The initial notebook covered evidence-backed work from 2026-07-21 through the step-1500 resume attempt.
- This update backfills compact-bundle through exact-top-2 backward, bucket selection, detailed backward profiling, and the first rejected active-edge lowering.
- Intermediate dirty variants whose exact diffs or raw logs were overwritten before this policy are explicitly marked as such instead of being presented as fully reproducible runs.
