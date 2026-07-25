# DAWN-SRW Experiment Notebook

This is the canonical, Git-tracked record of material DAWN-SRW experiments.
Dates are KST. Experiment IDs are stable as `YYYY-MM-DD/#NN`.

## Current State

- Current goal: continue v4174 speed and HBM optimization without changing model mathematics, parameter/checkpoint schema, canonical shapes, grouped Q/K/V execution, or the accepted precision boundary.
- Repository state at this backfill: branch `codex/v4167-poc`, GitHub HEAD `507d3a4a9bf9abc8a439b8ed525e8b7313dcfd98`; uncommitted changes in `models/dawn_srw_v4174.py` and `scripts/benchmark_srw_tpu.py` are in progress and are not attributed to a completed experiment here.
- Canonical model/config: `spatial-r1-v4.1.7.4`, `configs/train_config_v4174_400M_c4_40B_v4_64_space24_top2_direct_read.yaml`, batch 1024, sequence 512, mesh `16x2` on 32 devices.
- Best measured result: U-before-psum with BF16 final D-output psum at commit `14b07415f73f735e15ca9cb3098ea91923c86d38` ran 100 steps on `spatial-se-400m`; steady step time was 24.316 s/it and throughput 21,519 tokens/s versus 28.409 s/it and 18,419 tokens/s for the matched baseline.
- Current conclusion: the BF16 representation collective is accepted production behavior. It reduced the SRW representation payload from 3.0 GiB/layer to 0.5 GiB/layer and improved steady step time by about 14.4%; the next dominant cost is materialized operator-score state and rematerialized backward work.
- Current blocker: later exact-backward/profile commits and the current dirty worktree have not yet been connected to complete run evidence in this notebook.
- Next experiment: identify the exact code/config represented by the current dirty worktree, record it as in progress, then compare its 100-step steady window and XLA program HBM against the accepted `14b07415` result.

## Fixed Context

- Preserve model equations, residual order, attention-then-RST routing, parameter tree/count, checkpoint and optimizer schema, loss, and all live-gradient paths.
- Preserve Q/K pairing, one grouped Q/K/V executor and writeback, `M=24`, `K=2`, `D=2048`, `R=256`, operator counts, chunk semantics, metric-only behavior, and logging cadence unless a later user decision explicitly changes them.
- FP32 control boundary: rho/tau/margin/gates, gate mass, global denominator, control scaling, and residual/state updates.
- BF16 representation boundary: read/write GEMM operands, P/U projection operands, route output, and the final model-axis D-output psum; cast the psum result back to FP32.
- Production is U-before-psum plus BF16 D-output psum. U-before-psum plus FP32 D-output psum is a numerical reference, not a configurable production mode.
- Do not solve HBM pressure by reducing batch size, sequence length, operator count, `M`, `K`, or chunk semantics, or by introducing dispatch/sparse execution, unless the user explicitly opens that scope.
- A 100-step run is sufficient for the present trend decision. Report a steady non-validation/non-checkpoint window separately from compile, first-step, metric, and logging time.
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

## Backfill Boundary

- This initial notebook records evidence-backed Codex work from 2026-07-21 through the step-1500 resume attempt on 2026-07-25.
- Commits after `b046f3bb` other than the measured `14b07415` production result, including compact-bundle, exact-top-2 backward, bucket, and profiling work, still need separate evidence-backed entries.
- The current uncommitted model and benchmark changes are intentionally not described as completed experiments.
