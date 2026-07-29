# DAWN-SRW Experiment Notebook

This is the canonical, Git-tracked record of material DAWN-SRW experiments.
Dates are KST. Experiment IDs are stable as `YYYY-MM-DD/#NN`.

## Current State

- Current goal: use `codex/v4167-poc` as the sole canonical integration, baseline-source, and final experiment-reporting branch for all POC work. Experimental branches may isolate candidate code and runs, but every material result and current decision must be synchronized back to this notebook on POC and pushed. Only accepted source changes are integrated into POC.
- Paper-analysis track: `configs/paper_checkpoint_registry.yaml` is the descriptive source for the user-designated DAWN-SRW v4172 400M/1.3B and baseline-JAX 400M/1.3B run roots. The 400M DAWN and baseline are the matched 40B-token performance pair; DAWN 1.3B is available for 20B-token scale evaluation and selected mechanistic replication. The baseline-JAX 1.3B run is still training and is excluded from evaluation and comparison until the user declares it complete.
- Paper execution state: the v4172 400M Phase 0 gate is complete. Step 76,293 identity, clean restore, and immutable input contracts passed in `2026-07-29/#03`; two independent fixed-batch runs in `2026-07-29/#04` passed deterministic production/adapter parity. The full unlimited stock zero-shot suite completed as a comparable paper result in `2026-07-29/#05`, and the locked 10M-token C4 validation completed in `2026-07-29/#06`. The preregistered behavioral mechanistic screen completed in `2026-07-29/#07`: IOI, MCQA, ARC, and RAVEL passed the minimum-known-correct gate in discovery, validation, and test, while arithmetic did not. Promote IOI, ARC, and RAVEL to task-specific causal analysis; retain MCQA as a reserve/appendix task. Causal claims remain gated on localization, intervention, controls, restoration, and held-out confirmation. DAWN 1.3B work is sequential and selective, and no 1.3B baseline work is currently allowed.
- v4175 40M POC state: a route-separated `spatial-r1-v4.1.7.5` reference is locally implemented directly on `codex/v4167-poc` from exact clean parent/origin commit `7de5fc391608d337d9835d0cb9759487a6affd8d`. One shared eight-space atlas owns `space_read_proj`/`space_write_proj`; Q, K, V, and RST own independent selector projections and space keys, independently choose top-2 atlas entries, retain separate tau maps and operator banks, and use metric-complete dense all-space execution. The exact 40M tree is 40,344,160 parameters, 544 (`0.00135%`) above the 40,343,616 Transformer reference. These changes and this notebook entry remain an uncommitted local dirty worktree and are not yet canonical published source.
- v4175 evidence boundary: local Python/JAX/Flax `3.12.7/0.6.2/0.10.7` abstract-tree, route-factory, calibration, JIT forward/backward, compact-metric, reduced canonical train-step, and v4174 regression checks passed. TPU compile, HBM, step time, throughput, and training stability are `not measured`; execution optimization is intentionally deferred until after the 40M semantic reference runs.
- Publication state: accepted QKV+RST outer analytic-pullback source is integrated by merge commit `62417b239d5b79c5dfa8684563532958ed212385` on `codex/v4167-poc`. User-approved QK/V/RST initializer target fractions `0.08/0.08/0.05` are integrated by merge commit `bb2a3ada9a0c281df9439043162dde8045d62bfd` and measured at clean POC commit `ae12416b40607ef8dd21fe79a64d86b4335e18b3`. Generation-`g5` registry and the rejected matched chunk4 report are centralized by POC commit `4e896c00b63f9b0523bb95ebe8357c37cc824740`. The chunk4 source remains only on `codex/v4174-g4-frac885-rst-chunk4` at `a3879f21dcd3b786bdc2cde270bcca2a382796a4`.
- Active optimization config: `spatial-r1-v4.1.7.4`, `configs/train_config_v4174_400M_c4_20B_v4_64.yaml`, QK/V/RST initializer target fractions `0.08/0.08/0.05`, 393,803,872 parameters, batch 1024, sequence 512, mesh `16x2` on 32 devices, and 38,146 full global steps. It is 3,164 parameters above the v4172 393,800,708 reference.
- Best optimization result: active generation `g5` keeps the accepted generation-`g4` execution graph, Q/K/V/RST chunks `1/1/2/8`, and new initializer fractions. At clean POC commit `ae12416b`, its matched two-layer combined F+B is `0.628789` s and its three-warmup/five-measure full train mean is `8.943757` s/it (`58,621.5` tokens/s). This is runtime-equivalent to historical `g4` (`+0.0412%` combined F+B and `+0.0039%` full-step mean).
- Current conclusion: use generation `g5` with RST chunk8 for production. The matched chunk4 candidate produced `0.634587` s combined F+B (`0.9222%` slower than the matched chunk8 control) and failed the mandatory `0.565677` s gate. Its full-step mean was `8.923707` s/it, only `0.2242%` faster than chunk8 and below the required `2%`; p90 was `0.2863%` worse. The noise-level mean does not override the failed gates, so chunk4 is rejected.
- Baseline state: `docs/v4174_dense_rw_baselines.yaml` generation `g5` is active for QK/V/RST fractions `0.08/0.08/0.05` and chunks `1/1/2/8`. Generation `g4` is superseded historical evidence under the prior initializer fingerprint; generations `g1` through `g3` remain historical.
- Last verified v4174 production state (2026-07-27): fresh generation-`g5` production was running on the then-explicitly-authorized existing TPU `spatial-se-400m` from exact POC commit `4e896c00`. The GCS run folder is `gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4174_400M_c4_20B_v4_64/run_vspatial-r1-v4.1.7.4_20260727_002935_3201`; all eight workers then retained one `scripts/train_jax.py --from-scratch` process in `tmux train`. Step 20 was finite at loss `9.3893`, and its latest 10-step timing window was `8.530` s/it (`61,343.2` tokens/s). No live status refresh was authorized or performed for this v4175 task.
- Current blocker: no local implementation blocker. A v4175 TPU run requires the user to name the exact existing TPU resource in the current task; none has been named, so no gcloud, SSH, launcher, watcher, or remote process command was run.
- Next experiment: review and publish the exact v4175 source/config/notebook state, then run the fresh 40M config on an explicitly named existing TPU. First establish compile, HBM, finite step-1/short-window behavior, and route-specific selection evidence; optimize the dense route kernels only after that semantic reference is measured.

## Paper Writing Ledger

This section is the manuscript-facing source of truth. A paper writer should use
it for framing, methods, table cells, figure inputs, evidence boundaries, and
remaining-work decisions; the dated experiment log below is the audit trail.
`Measured` means a result is eligible for a paper table, `validated` means a
supporting contract passed but is not itself a headline result, and `pending`
means that no result may be inferred or filled from a training log.
Scientifically negative or null outcomes from a completed valid protocol are
results and must be reported: this includes a preregistered threshold miss, no
localized effect, no control separation, a non-significant effect, or a
rejected hypothesis. Authentication, launcher, dependency, or code failures
that produce no interpretable scientific measurement are operational incidents,
not paper results; rerun them correctly and report the valid completed outcome.
Paper storage is aggregate-first: per-example IDs, logits, margins, correctness
masks, raw capture rows, operator schedules, and effect vectors are not durable
paper artifacts. Retain aggregate statistics, content hashes, immutable
benchmark/protocol identity, exact code/config/checkpoint identity, and compact
run summaries. Do not classify or remove JSON by file size alone; write normal
structured results, then audit their field semantics and purge only files that
contain raw per-example vectors.

### Paper thesis and contribution structure

- Working one-sentence thesis: DAWN-SRW realizes language-model state
  transitions through input-conditioned read/write operators, enabling
  operator-level computation to be localized, intervened on, restored, and
  tested for reuse.
- Evidence order: `(A) language-model performance retention -> (B) causal RW
  operator units -> (C) cross-example circuit reuse -> (D) selected 1.3B
  replication`.
- The novelty claim is the causal and reusable operator unit, not absolute
  benchmark leadership. Performance establishes that the analyzed checkpoint
  is a functioning language model.
- The 400M and 1.3B checkpoints are not a scaling curve: they differ in both
  parameter count and training tokens (`40B` versus `20B`). The 1.3B result is
  a replication at a larger scale.

| Claim | Required evidence | Current status | Manuscript wording now allowed |
|---|---|---|---|
| A. Performance retention | Matched C4 validation and six-task zero-shot versus dense 400M | DAWN 400M C4 and zero-shot measured; dense 400M pending | Report DAWN absolute C4 and zero-shot results; do not yet claim parity with the dense baseline |
| B. Causal RW unit | Localized operators, suppression, matched controls, restoration, held-out confirmation | Behavioral eligibility measured for IOI, MCQA, ARC, and RAVEL; causal analysis pending | Report which tasks are eligible for analysis; do not yet claim causal importance |
| C. Reusable circuit | Discovery/confirmatory separation and transfer to matched held-out inputs with negative controls | IOI, ARC, and RAVEL promoted to confirmatory causal analysis; transfer evidence pending | Describe the frozen follow-up design, not circuit reuse as a result |
| D. Scale replication | 1.3B performance and selected causal effect with the same control design | DAWN 1.3B checkpoint available; evaluation pending | Describe the planned replication only |

### Study matrix and comparison roles

| Model | Parameters | Training tokens | Checkpoint state | Paper role |
|---|---:|---:|---|---|
| DAWN-SRW v4172 400M | `393,800,708` measured | `39,999,504,384` in the evaluation manifest | Step `76,293` resolved, restored, parity-validated, C4 and zero-shot measured | Discovery model, full mechanistic analysis, primary causal result |
| Dense Transformer 400M | Config estimate `~393.9M`; exact restore count pending | `40B` configured | User-designated run available; exact numeric step/evaluation pending | Matched 400M performance control |
| DAWN-SRW v4172 1.3B | `1,272,038,404` configured; restore verification pending | `20B` configured | User-designated run available; exact numeric step/evaluation pending | Larger-scale performance and selected causal replication |
| Dense Transformer 1.3B | Pending completed training | `20B` configured | Training in progress; excluded by user decision | No current paper result |

The machine-readable run roots and availability states are in
`configs/paper_checkpoint_registry.yaml`. Every paper result must additionally
resolve one numeric checkpoint step and preserve its exact evaluation commit.

### DAWN-SRW v4172 400M method specification

| Component | Exact paper model setting |
|---|---|
| Identity | `spatial-r1-v4.1.7.2`; generalized coordinate-wise bilinear RW operator keys |
| Backbone dimensions | Vocabulary `30,522`; width `2,304`; 18 blocks; 36 attention heads; head width 64; maximum sequence length 512 |
| Operator coordinates | Route width `d_route=256`; direct state-to-operation query projection |
| Shared operator pools | Q/K pool `4,600`; V pool `14,500`; post-attention RST pool `29,300`; total stored pool rows `48,400` |
| Operator representation | Each pool row owns model-width read and write vectors. Normalized read/write directions are projected through shared read/write probes; their coordinate-wise product forms the live generalized-bilinear route key |
| Selection/composition | DirectTau boundary; `linear_angular` composition; denominator powers QK/V/RST `0.5/1.0/1.2` |
| Block computation | Q, K, and V operator compositions form causal self-attention; a separately routed RST composition updates the post-attention residual before the next block |
| Regularization/runtime | Model and router dropout `0`; gradient checkpointing enabled; QK/V/RST pool chunks `1/1/4` |
| Parameters | `393,800,708` at the restored checkpoint |

Training used C4 with seed 1, global batch 1,024, sequence length 512, one
gradient-accumulation step, TPU mesh `data=16, model=2`, learning rate `3e-4`,
8% warmup, weight decay `0.1`, tau learning-rate multiplier `0.001`, and
quantile initialization targets QK/V/RST `0.05/0.03/0.01`. The optimizer is the
repository canonical Adam-style transform with `b2=0.95`, decoupled
base/pool weight decay, warmup plus cosine schedule, and optional global
gradient clipping. The paper checkpoint is the concrete Orbax directory
`checkpoints/000000076293` under the registered v4172 400M run.

The tokenizer is `bert-base-uncased`, logical/effective vocabulary 30,522,
lower-casing enabled, pad ID 0, CLS ID 101, SEP ID 102, no BOS/EOS token, and
no automatic special-token insertion during zero-shot scoring. The zero-shot
manifest records tokenizer vocabulary hash
`68564f66edb91d6593d0a740d21bfa1fdfcf2db08573f06e17616315c797f572`.

### Matched 400M dense-control contract

The configured dense control uses the same C4 40B data, seed 1, global batch
1,024, sequence length 512, 18 layers, dropout 0, mesh `16x2`, learning rate
`3e-4`, 8% warmup, and weight decay `0.1`. Its architecture is width 1,280,
FFN width 5,120, and 20 attention heads. The config estimates approximately
393.9M parameters. Exact parameter count, numeric checkpoint step, full C4
validation, tokenizer parity, and the identical zero-shot protocol remain
required before any DAWN-versus-dense performance statement.

### Evaluation protocol locked for paper tables

- C4 validation must report cross-entropy, perplexity, token accuracy, valid
  target-token count, split/data hash, batch, sequence length, dtype, and exact
  checkpoint. The 16,352-token Phase 0 batch is a forward-parity contract, not
  the final C4 validation row.
- Zero-shot uses stock `lm-eval==0.4.2`, zero few-shot examples, protocol
  `mamba_table3_zero_shot_v1`, no optimizer or checkpoint write, global batch
  32, maximum length 512, and length buckets 64/128/256/512.
- Table metrics are normalized accuracy for HellaSwag, PIQA, ARC-Easy, and
  ARC-Challenge; raw accuracy for LAMBADA and WinoGrande; LAMBADA perplexity is
  reported separately. Every accuracy includes its standard error.
- `limit=1` executions are parity-only and must never populate a paper
  performance table. The eligible v4172 400M run has `limit=null`,
  `smoke_test_only=false`, all six tasks complete, and `comparable=true`.

### Table-ready v4172 400M C4 validation

| Model | Validation protocol | Valid target tokens | Cross-entropy | Perplexity | Token accuracy |
|---|---|---:|---:|---:|---:|
| DAWN-SRW v4172 400M | packed C4 validation, 10M-token cap | `9,974,720` | `2.940212858` | `18.919873132` | `0.424911877` (`42.4912%`) |

The locked `packed_c4_validation_v1` run evaluated all 610 complete global
batches admitted by the 10M-token cap: 19,520 sequences at length 512 and
global batch 32. The capped dataset view contained 19,531 sequences; the final
11 could not form a complete global batch and were excluded before scoring.
The result is `comparable=true` and uses the same fixed protocol required for
the dense 400M and DAWN 1.3B rows.

The paper-eligible C4 result root is
`gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4172_400M_c4_40B_v4_64_ver1_den_qk0p5_v1p0_rst1p2/run_vspatial-r1-v4.1.7.2_20260715_133004_3201/side_analysis/run_analysis_000000076293_c4_validation_20260729T044410Z-37fa6082`.
Use `validation_summary.json` for table cells, `run_manifest.json` for the
source, software, data, mesh, checkpoint, and hash contract, and `summary.log`
for the canonical text summary. This run used clean evaluation commit
`37fa60822af9ae13a8cd2fddb361b563cbdc3304`.

### Table-ready v4172 400M performance

| Benchmark | Split (`n`) | Metric | Score | Standard error |
|---|---:|---|---:|---:|
| LAMBADA OpenAI | test (`5,153`) | accuracy | `0.352027945` | `0.006653938` |
| LAMBADA OpenAI | test (`5,153`) | perplexity | `24.077383687` | `0.800538623` |
| HellaSwag | validation (`10,042`) | normalized accuracy | `0.458374826` | `0.004972460` |
| PIQA | validation (`1,838`) | normalized accuracy | `0.712187160` | `0.010563250` |
| ARC-Easy | test (`2,376`) | normalized accuracy | `0.484006734` | `0.010254534` |
| ARC-Challenge | test (`1,172`) | normalized accuracy | `0.256825939` | `0.012766924` |
| WinoGrande | validation (`1,267`) | accuracy | `0.513812155` | `0.014047123` |
| Six-task aggregate | six primary accuracy metrics | arithmetic mean | `0.462872460` | not aggregated |

The paper-eligible zero-shot result root is
`gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4172_400M_c4_40B_v4_64_ver1_den_qk0p5_v1p0_rst1p2/run_vspatial-r1-v4.1.7.2_20260715_133004_3201/side_analysis/run_analysis_000000076293_zero_shot_20260729T041547Z-6232963d`.
Use `backends/stock_zero_shot/results_summary.json` for table cells,
`backends/stock_zero_shot/run_manifest.json` for software, tokenizer, split,
fingerprint, protocol, and source identity, and `items/zero_shot/*.json` for
per-task evidence. This run used clean evaluation commit
`623eaee8fc30471db809cb7093d3ed28594bff35`.

### Manuscript-ready performance result

At the fixed step-76,293 checkpoint, the 393.8M-parameter DAWN-SRW model
obtained C4 validation cross-entropy 2.9402, perplexity 18.9199, and next-token
accuracy 42.49% over 9.97M valid target tokens. On the six stock zero-shot
tasks, it obtained mean accuracy 46.29%, with its strongest absolute result on
PIQA (71.22%) and lower results on ARC-Challenge (25.68%) and LAMBADA
(35.20%). These measurements establish that the checkpoint is a functioning
language model suitable for causal analysis. They do not yet establish
performance retention relative to a dense Transformer, because the matched
400M dense-control cells remain unmeasured.

### Mechanistic screening result and locked causal design

The preregistered first-stage screen tested behavioral eligibility for IOI,
MCQA, arithmetic, ARC reasoning, and RAVEL before any circuit search. A row is
mechanistically eligible only when the frozen checkpoint answers both the base
example and its labeled source behavior correctly where the source label is
defined. Every discovery, validation, and test phase must contain at least 32
independent eligible units. MIB candidate scores use summed log probability;
RAVEL uses mean log probability per token. The screen performs no optimizer or
parameter update and, by itself, supports no localization or causal claim.

| Task | Rows D/V/T | Base accuracy D/V/T | Source accuracy D/V/T | Pair accuracy D/V/T | Independent eligible units D/V/T | All-phase gate | Follow-up decision |
|---|---:|---:|---:|---:|---:|---|---|
| IOI | `128/128/128` | `0.984/0.977/0.992` | `0.945/0.984/0.898` | `0.930/0.961/0.891` | `119/123/114` | Pass | Primary MIB causal task |
| MCQA | `110/50/50` | `1.000/1.000/1.000` | `0.727/0.680/0.640` | `0.727/0.680/0.640` | `80/34/32` | Pass | Reserve/appendix; test split is exactly at the threshold |
| Arithmetic | `128/128/128` | `0.438/0.445/0.500` | `0.516/0.555/0.461` | `0.094/0.086/0.063` | `12/11/8` | Fail | Exclude from causal claims at this checkpoint |
| ARC | `128/128/128` | `0.977/0.969/0.992` | `0.414/0.430/0.453` | `0.391/0.398/0.445` | `50/51/57` | Pass | Secondary MIB causal task |
| RAVEL | `512/512/512` | `0.678/0.699/0.676` | `0.699/0.633/0.660` | `0.578/0.574/0.568` | `142/154/146` | Pass | Primary variable-localization and transfer task |

`D/V/T` denotes discovery/validation/test. For RAVEL, paired-correct row counts
were `296/294/291`; the independent-unit counts in the table collapse repeated
rows from the same official causal unit. The task selection is fixed before
localization: IOI supplies the strongest MIB paired behavior, ARC adds a
distinct reasoning task with a comfortable all-phase sample buffer, and RAVEL
supplies the largest independent causal-variable sample. MCQA remains valid
but is not used to select the main result because validation/test contain only
`34/32` units. Arithmetic is not analyzed causally because it never meets the
minimum sample gate.

The second-stage split contract is:

- Discovery split: select layers, pools, operators, circuit size, and
  intervention specification.
- Confirmatory validation split: freeze and apply the discovery specification;
  it cannot be reused to select operators or thresholds.
- Held-out test: evaluate only the already-frozen specification and report it
  separately from discovery and validation.
- Promotion beyond behavioral eligibility requires a localized and repeatable
  effect, the same effect direction on held-out examples, and clear separation
  from random and matched controls.

For each promoted task, the full causal sequence is:

1. Localize layer, Q/K/V/RST pool, operator, and multilayer trajectory.
2. Suppress the frozen circuit and measure correct-answer logit, logit
   difference, loss, exact accuracy, prediction flips, and unrelated outputs.
3. Compare with equal-count random, same-layer random, activation-magnitude
   matched, route-frequency matched, and other-task circuit controls.
4. Restore the suppressed circuit and test recovery toward intact behavior.
5. Transfer a source circuit to same-template held-out inputs, different
   wording with the same computation, and a different-computation negative
   control.
6. Treat mediation as supporting analysis only; suppression, matched controls,
   restoration, and held-out transfer carry the main causal claim.

Run the task-specific IOI, ARC, and RAVEL analyses before the fail-closed full
`scientific` claim aggregation. For 1.3B, repeat only one or two strongest
400M tasks and interventions with the same controls. Identical operator
indices across scales are not required; the effect direction and computational
organization are the replication target.

The paper-eligible screening result root is
`gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4172_400M_c4_40B_v4_64_ver1_den_qk0p5_v1p0_rst1p2/run_vspatial-r1-v4.1.7.2_20260715_133004_3201/side_analysis/run_analysis_000000076293_mechanistic_screen_20260729T051149Z-7d3be0d9`.
Use `summary.json` and `summary.log` for the run-level decision, and
the aggregate task table below for the benchmark-level decision. The former
per-example behavioral JSON vectors were deliberately purged under the
aggregate-only paper-storage policy; reproducibility remains bound by the
immutable benchmark manifest, protocol hash, exact code/checkpoint identity,
aggregate statistics, and content hashes. The clean analysis commit is
`469c2d12caf1dabf578e16d0bd1ab9cc7bf70ea4`, and the protocol hash is
`e55551954702df3ae628798f5a208681e1eed68dbbda0d967967af6865107adc`.

### Manuscript table and figure map

| Output | Required content | Current evidence source |
|---|---|---|
| Table 1: model/training specification | Parameters, tokens, dimensions, pools, tokenizer, training conditions | This ledger; `2026-07-29/#03`, `#05`, and `#06`; baseline rows pending |
| Table 2: C4 validation | Loss, perplexity, token accuracy for DAWN/dense 400M and DAWN 1.3B | DAWN 400M row complete in `#06`; comparison rows pending |
| Table 3: zero-shot | Six tasks, mean, standard errors, model comparison | DAWN 400M row complete in `#05`; comparison rows pending |
| Figure 1: architecture | State -> route query -> selected RW read/write composition -> attention/RST residual update | v4172 method specification above |
| Figure 2: performance | C4 and benchmark comparison with uncertainty | DAWN 400M C4 and zero-shot complete; comparison cells pending |
| Figure 3: circuit localization | Layer x operator causal contribution, trajectory, held-out overlap | Behavioral task-selection table complete in `#07`; IOI/ARC/RAVEL causal localization pending |
| Figure 4: causal intervention | Intact, suppressed, matched control, restored | Pending scientific |
| Figure 5: transfer | Source circuit, matched target, paraphrased target, negative control | Pending scientific |
| Figure 6: scale replication | 400M versus 1.3B effect size and normalized layer trajectory | Pending selected 1.3B replication |

### Interpretation and exclusion rules

- Supported now: the exact v4172 400M checkpoint restores reproducibly; the
  production and analysis forwards agree within `3.31e-7` CE; the locked
  10M-token C4 validation and full unlimited stock six-task results above are
  reproducible and paper-eligible; and IOI, MCQA, ARC, and RAVEL have enough
  paired-correct independent examples for mechanistic follow-up under the
  preregistered threshold.
- Not supported yet: matched dense performance retention, causal importance,
  restoration, reuse, scale replication, pruning-quality tradeoffs, or
  hardware speedup.
- Do not infer a full validation loss from the Phase 0 fixed batch or from a
  training-log summary. Do not use `limit=1` task scores.
- Do not call 400M-to-1.3B a scaling curve. Do not include baseline 1.3B until
  the user declares training complete and a numeric checkpoint is resolved.
- Preserve completed null and negative results in the manuscript ledger and
  result tables. Do not silently drop a task because its valid result misses a
  threshold; identify the failed gate and restrict the corresponding claim.
- Do not promote setup, authentication, launcher, dependency, or implementation
  failures into scientific evidence when no valid protocol result was produced.
- Do not retain raw per-example vectors as paper artifacts. Keep aggregate
  statistics and deterministic content hashes, and audit completed result files
  by field semantics rather than treating a large JSON file as invalid merely
  because of its size.
- Structural sparsity may be reported from measured operator usage. Hardware
  efficiency requires matched device, mesh, batch, length, dtype,
  compilation, logging, and checkpointing conditions; without that protocol,
  restrict language to estimated executed operator work.
- A strong causal statement requires repeat activation, selective suppression,
  matched-control separation, restoration, held-out transfer, and scale
  replication. Any missing link must remain explicit in the paper.

## Fixed Context

- `codex/v4167-poc` is the canonical integration branch, canonical baseline-source branch, and central final-reporting branch for every POC. Work and measurements may occur on candidate branches, but every material result and current-state decision must be synchronized to this notebook on POC and pushed.
- New canonical baselines must be measured from a clean exact commit reachable from `origin/codex/v4167-poc`. Historical measurements retain the branch and commit actually executed; synchronization does not rewrite their identity.
- Rejected and inconclusive candidate source is not synchronized merely for ancestry completeness. Its report is synchronized here as a POC reporting-only commit, while the accepted source on POC excludes the candidate code.
- Canonical v4174 parameter roots are `space_selector`, `space_interface`, `operator_controller`, and `operator_bank`; pre-schema v4174 checkpoints remain intentionally incompatible with no migration mode or parameter alias. Checkpoints already using these roots are shape-compatible, but restore their saved projection values and therefore cannot test a new initializer.
- The selected hard-top-k ReLU-squared space weights are L1-normalized across the selected-space axis. Their sum is one; an all-zero selected mass uses a deterministic top-1 one-hot fallback. Raw space mass remains diagnostic, not an absolute update-scale control.
- Preserve residual order, attention-then-RST reselection, Q/K pairing, one grouped Q/K/V executor and writeback, `M=24`, `K=2`, `D=2048`, `R=256`, chunk semantics, metric-only behavior, logging cadence, loss, and all live-gradient paths unless a later user decision explicitly changes them.
- The parameter-matched config uses `n_q=n_k=21,504`, `n_v=134,016`, and `n_rst=270,000`, for 393,803,872 parameters. Exact equality with the v4172 393,800,708 reference is impossible under 48-operator shard increments; this is the closest valid count.
- FP32 control boundary: rho/tau/margin/gates, gate mass, global denominator, control scaling, and residual/state updates.
- BF16 representation boundary: read/write GEMM operands, P/U projection operands, route output, and the final model-axis D-output psum; cast the psum result back to FP32.
- Production is U-before-psum plus BF16 D-output psum. U-before-psum plus FP32 D-output psum is a numerical reference, not a configurable production mode.
- Do not solve HBM pressure by reducing batch size, sequence length, operator count, `M`, `K`, or chunk semantics, or by introducing dispatch/sparse execution, unless the user explicitly opens that scope.
- The user explicitly opened exact backward-only dispatch scope: forward remains bundle4 dense, while backward may use exact selected-top-2 spaces and exact active-edge support. No selected space, active edge, or overflow work may be dropped.
- Canonical generation `g4` removes the generic outer autodiff graph for both QKV and RST linear-angular paths: U writeback, selected-space routing weight, global denominator, RW chunk, tau, local normalization, and P projection are transposed analytically. Other composition modes retain the generic VJP fallback.
- Short TPU benchmarks and 50-step runs are sufficient for optimization screening. Require a longer steady window before production acceptance, and always report the measured window separately from compile, first-step, metric, validation, and checkpoint time.
- TPU names recorded below are historical evidence, not authorization for a future task. `AGENTS.md` controls TPU authorization and prohibits TPU creation or allocation.
- Evidence labels used below: `TPU measured`, `local validated`, `user-provided`, and `not measured`.

## Baseline Registry

- Machine-readable source: `docs/v4174_dense_rw_baselines.yaml`.
- Canonical source branch: `codex/v4167-poc`. Candidate branches may produce comparison evidence, but a promoted implementation and its baseline decision must be synchronized to POC; the next canonical baseline run must use an exact clean POC-reachable commit.
- Normal path: reuse the active generation's stored control and run only the candidate with the same two-layer gate. Continue to the matched three-warmup/five-measure full train benchmark only after the gate passes.
- Promotion path: when a candidate passes both protocols and is accepted, its measurements become the next generation baseline directly. Promotion by itself never triggers a duplicate control run.
- Rebaseline path: create one new baseline generation when accepted structure/shapes, forward or overflow semantics, precision/collectives, performance config or seed, data/batch/sequence, TPU/mesh, Python/JAX/Flax/Orbax/libtpu release line, or benchmark/profiler semantics changes. If accepted source changed without both protocol results, measure that accepted source once before testing the next candidate.
- Noise path: near a threshold, repeat the candidate first. Repeat the control only after an invariant mismatch or concrete baseline-drift evidence.
- Acceptance thresholds: at least 10% for combined two-layer QKV+RST forward-plus-backward, at least 2% for five-step mean time, no more than 5% compile regression, no more than 0.05 GiB runtime peak-HBM regression, and no OOM/NaN/Inf/process loss. Signed compile and HBM deltas are always reported.

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

#### #13 — Reject end-to-end P-RW-U rematerialization after matched TPU gates

- Status: rejected for production; isolated backward improved, but full training did not.
- Hypothesis/change: wrapping the accepted exact-space attention/RST P-RW-U primal and compact overflow groups in a `nothing_saveable` rematerialization boundary should reduce retained backward intermediates without changing model equations, shapes, grouped Q/K/V execution, precision, parameters, or checkpoint schema.
- Run identity: existing TPU `spatial-se-400m`; accepted gate branch `codex/v4174-accepted-profile-gate` at clean commit `001c2c27fd7baf6998937530cdca3e55e57fccf0`; candidate gate branch `codex/v4174-p-rw-u-profile-gate` at clean commit `1ff2e57b811175a875795c85f7c2f14a7d71bb0b`; candidate model SHA-256 `709a0571ab3c510748c80a95e87d4caec788e1bce134f67b4e7b60f063278ae2`; bundle4 config; real C4 data; no checkpoint; batch 1024, sequence 512, mesh `16x2`/32 devices; Python 3.10.12, JAX 0.6.2, Flax 0.10.7.
- Two-layer result: accepted/candidate forward was 2.4222/2.4224 s with identical printed loss 28.8470. QKV forward-plus-backward was 419.739/403.785 ms and RST was 426.555/412.014 ms; the combined 846.294/815.799 ms favored the candidate by 3.60%. Detailed split was 1.1242/1.0935 s and runtime peak HBM was 4.961/4.960 GiB. Candidate detailed compile regressed from 129.420 to 146.149 s (+12.93%).
- Full train-step result: both runs used three warmups and five measured forward+backward+AdamW steps. Accepted/candidate mean was 10.7882/10.7895 s, median 10.8422/10.8453 s, throughput 48,610.4/48,603.7 tok/s, and runtime peak HBM 7.938/7.940 GiB. The candidate was 0.012% slower, effectively equal. Initial compile was 102.660/106.593 s, a 3.83% candidate regression.
- Exactness evidence: all printed loss values matched to four decimals except one `9.3902` versus `9.3901`; the largest printed grad-norm delta across measured steps was 0.046. No OOM, NaN, invalid benchmark flag, or process loss occurred.
- Lesson: the explicit boundary changes the isolated custom-VJP schedule, but its small gain disappears inside the complete 18-layer optimizer graph and does not reduce observed runtime HBM. Runtime telemetry is not XLA program-memory evidence, so no program-HBM reduction is claimed.
- Decision/next: do not promote the P-RW-U candidate. Preserve it at `fd7e5e32`/`1ff2e57b`, retain `8187c00d` as the accepted fallback, and resume the canonical step-1681 training run.
- Evidence: `/home/madst0614/train.v4174_p_rw_u_gate_accepted_001c2c27_20260726T041649KST.log`; `/home/madst0614/train.v4174_p_rw_u_gate_candidate_1ff2e57b_20260726T042523KST.log`; `/home/madst0614/train.v4174_p_rw_u_trainstep_candidate_1ff2e57b_20260726T043455KST.log`; `/home/madst0614/train.v4174_p_rw_u_trainstep_accepted_001c2c27_20260726T044420KST.log`; matching JSONL directories under `/home/madst0614/DAWN-SRW/benchmark_runs/`.

#### #14 — Consolidate the accepted v4174 400M run into one 20B config

- Status: accepted configuration change; local validation passed; TPU execution not requested.
- Hypothesis/change: replacing the direct-read and bundle4-named 40B YAMLs with one canonical `train_config_v4174_400M_c4_20B_v4_64.yaml`, explicitly selecting bundle4 execution, and restoring the exact accepted model source should remove launch ambiguity while halving the requested token budget without changing topology, parameters, precision, grouped execution, or checkpoint schema.
- Run identity: branch `codex/v4167-poc`; source parent `7316b5b2e4a0472d6f4528cc33873cdf12d8edd1`; model Git blob `bfbe6eef04d1047d43625b68e0a2e3976da064e2`, identical to `8187c00d4b767a743c3178a5bcb8caac1c6124c5:models/dawn_srw_v4174.py`; config SHA-256 `31a4a57f0044aef100c66e8519a9386ad4122289920805412e82d57b697679df`; batch 1024, sequence 512, mesh `16x2`/32 devices; local Python 3.12.7, JAX 0.6.2, Flax 0.10.7.
- Result: the canonical config retains `gs://dawn-tpu-data-c4/c4_train_40B` as the source but caps it at 20,000,000,000 tokens, yielding 38,146 full batches of 524,288 tokens, 19,999,490,048 consumed tokens, and a 509,952-token dropped remainder. It selects `bundle_dense`, bundle size 4, token block 2048, exact selected-top-2 backward with bucket capacity 3072 from the accepted model, and 393,755,652 symbolic parameters. Checkpoints and logs now use `dawn_srw_v4174_400M_c4_20B_v4_64`; both superseded 40B config files were removed.
- Validation: `.venv-jax062` loaded the YAML and model, checked the 20B step arithmetic, accepted execution fields, parameter count, and checkpoint suffix; the working model Git blob matched the accepted `8187c00d` blob; `git diff --check` passed. No TPU command, GCS write, runtime step, or performance measurement was performed.
- Lesson: renaming the checkpoint base does not migrate the existing step-1681 run, and the strict resume path intentionally uses checkpoint `full_config`; passing the old 40B run to `--resume-from` would retain the 40B schedule. A 20B continuation therefore needs an explicit migration/fork mechanism or a fresh run rather than an implicit config override.
- Decision/next: publish only the repository change as requested and leave the current `spatial-se-400m` process untouched. The user will perform the execution replacement and choose the runtime checkpoint transition.
- Evidence: `configs/train_config_v4174_400M_c4_20B_v4_64.yaml`; accepted model blob identity above; local validation output from 2026-07-26 KST.

#### #15 — Make operation spaces explicit and normalize selected-space weights

- Status: accepted implementation; local validated; publication and TPU runtime pending.
- Hypothesis/change: separate space addressing, global-to-space read/write interfaces, per-space tau control, and RW operator banks into explicit parameter roots; remove the `M <= R` route-key restriction; add round-trip and cross-space geometry diagnostics; and make the hard top-k ReLU-squared selector control only relative space mixture by L1-normalizing selected weights to sum to one. An all-zero selected mass uses a deterministic top-1 one-hot fallback. No compatibility mode or legacy parameter alias was added.
- Run identity: branch `codex/v4167-poc`; source parent `462202863a91b9f000c7725d16794215de854b86`; dirty model SHA-256 `39edbe7a6d3169500b011e846e65dbc03941b152b8196ce073e9383de45be003`; trainer SHA-256 `a36a9f534ff8d30a63136892e41c90ce19e4e995cceccc61711e908aa8271fdb`; benchmark SHA-256 `4fed504d141cbf281d47a14bdfb7664c295e80af8da006834c2e56db0c65023a`; config SHA-256 `2945e84a0b3e989ec50da337f322ec20bd0ab2f3933e3887c6ce41f2353fcc68`; existing test-file SHA-256 `8b2066a76981e7c76e718a8e03523b371148eb5f402b2b8625a443a4e2227289`. The tree is dirty only in the five implementation/config/consumer files plus this notebook; no checkpoint; target runtime is the user-authorized existing TPU `spatial-se-400m`, not contacted in this entry.
- Parameter result: the fresh tree has roots `space_selector`, `space_interface`, `operator_controller`, and `operator_bank`; route tau kernel/bias shapes are `[24,256,1]` and `[24,1]`. With `n_rst` increased from 269,952 to 270,000, the abstract and symbolic counts both equal 393,803,872, which is 3,164 parameters or 0.0008% above the v4172 393,800,708 reference. All Q/K/V/RST counts are divisible by `24*2=48`; the RST-only count change is 0.01778% and minimally perturbs the QK:V:RST ratio.
- Local validation: repository environment Python 3.12.7, JAX 0.6.2, Flax 0.10.7. The complete existing v4174 regression file passed `20 passed, 2 skipped in 139.70s`; the two skipped checks require two local devices. Canonical abstract init matched symbolic count exactly; bundle QKV/RST benchmark forward and exact custom-VJP backward returned finite gradients; quantile calibration produced per-space biases; `py_compile`, `git diff --check`, and Git Bash syntax checks for the training launcher/setup/watcher/grep scripts passed.
- Validation corrections: the first single-device detailed-profile construction correctly rejected missing canonical vocab-parallel functions, which are only built for `mesh_model>1`; local stand-ins were then used solely to exercise the new profile parameter consumers. The changed output scale made mixed-vs-FP32 RST active decisions differ by two of 512 at the first step while all 20-step loss and gradient parity gates passed; the existing check now expresses the discrete one-decision Q/K/V and two-decision post-attention RST budget instead of an impossible sub-decision fixed threshold. A bracket typo in that existing check failed collection once and was corrected before the full pass.
- Lesson: raw space score mass and within-space RW mass normalize different axes. With selected-space L1 normalization, the selector no longer closes the whole residual update when initial raw mass is small; the RW denominator and fixed route/residual scales retain responsibility for absolute update magnitude.
- Decision/next: commit and push this exact tree, launch the canonical 20B config from scratch on `spatial-se-400m`, and record remote commit/config/mesh plus first real optimizer-step evidence in a later entry.
- Evidence: `models/dawn_srw_v4174.py`; `scripts/train_jax.py`; `scripts/benchmark_srw_tpu.py`; `configs/train_config_v4174_400M_c4_20B_v4_64.yaml`; `tests/test_v4174_operation_spaces.py`; local validation output from 2026-07-26 KST.

#### #16 — Publish and start the canonical v4174 run from scratch

- Status: implementation published; TPU measured through optimizer step 5; training remains active.
- Hypothesis/change: publish the explicit operation-space tree and selected-space L1 gate without a compatibility mode, then replace the existing workload on the user-authorized `spatial-se-400m` with a fresh canonical 20B run. A real step with raw space mass still near initialization should demonstrate that selector mass no longer closes the whole update.
- Run identity: existing READY `v4-64` TPU `spatial-se-400m`; branch `codex/v4167-poc`; clean source commit `db16ffdf9ecb1a65cdf5ea0cfe3702b790430055` on all eight workers; config `configs/train_config_v4174_400M_c4_20B_v4_64.yaml`; selected full-config SHA-256 `b3b7995ec09571e80bf2da54d1d8234f3fb1e1c6ae91ba5bbec764fe0b39cd38`; `--from-scratch`; no restore checkpoint; batch 1024, sequence 512, mesh `16x2`/32 devices; Python 3.10.12, JAX 0.6.2, Flax 0.10.7, Orbax 0.11.24.
- Launch: the first local launcher invocation stopped before TPU contact because Git Bash selected unsupported gcloud Python 3.9. Re-running the same fixed launcher with `CLOUDSDK_PYTHON=C:\Users\MADST\gcloud\google-cloud-sdk\platform\bundledpython\python.exe` passed all eight SSH preflights and verified no live accelerator holder before starting `scripts/train_jax.py` in `tmux train` at 2026-07-26 13:55:32 KST. Each worker uses `~/train.log`; the JAX primary is gcloud worker 6 / process 0.
- Runtime result: scratch initialization produced exactly 393,803,872 concrete and symbolic parameters, matching the config expectation. It created `gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4174_400M_c4_20B_v4_64/run_vspatial-r1-v4.1.7.4_20260726_135537_3201`, completed per-space quantile calibration, passed the eight-host `global_step=0` consistency barrier, and entered canonical shard-map training without HBM or compilation failure.
- Step evidence: step 1 completed with loss/CE 10.7297, finite grad norm 11.74, attention/RST raw space mass both 0.024, selected active-space count 2.000, zero-mass fallback rate 0.000, attention norm 0.197, RST norm 1.989, and residual norm 9.581. The first compiled step took 113.163 s. Step 5 completed with loss 10.6744 and a four-step window of 11.534 s/it / 45,368.7 tokens/s; the process was still live with no fatal error.
- Lesson: the new tree initializes and trains on the canonical 32-device topology. The key scale check is satisfied: despite raw selector mass of only about 0.024, normalized top-2 routing remains fully active and produces non-negligible attention/RST updates, while raw mass remains an observable diagnostic.
- Decision/next: keep this from-scratch run active. Treat step 5 only as startup evidence, not a steady-state performance acceptance; inspect step 100, validation, and checkpoint boundaries before drawing longer-run conclusions.
- Evidence: remote `~/train.log` on all eight `spatial-se-400m` workers; primary log on gcloud worker 6; GCS run folder above; local untracked launcher captures `launch_spatial-se-400m_20260726_135522_worker_{0..7}.log` are raw operational evidence and must not be committed.

#### #17 — Confirm continued early stability through step 20

- Status: TPU measured; the same from-scratch process remains active.
- Run identity: unchanged from `2026-07-26/#16`; `spatial-se-400m`, commit `db16ffdf9ecb1a65cdf5ea0cfe3702b790430055`, canonical 20B config, GCS run `run_vspatial-r1-v4.1.7.4_20260726_135537_3201`, mesh `16x2`/32 devices, and no checkpoint restore.
- Result: step 10 completed at loss 10.5675, 10.195 s/it, and 51,323.1 tokens/s. Step 20 completed at loss 10.3221, finite grad norm 17.11, 10.079 s/it, and 51,917.6 tokens/s. Attention/RST raw space mass remained 0.024 while both selected active-space counts remained 2.000; attention, RST, and residual norms were 0.291, 2.027, and 9.990. The primary `train_jax.py --from-scratch` process was still live and no fatal error was present.
- Decision/next: leave training untouched and wait for the regular step-100 report before treating throughput or numerical behavior as a stable window.
- Evidence: primary `~/train.log` on gcloud worker 6, observed 2026-07-26 KST.

#### #18 — Record the canonical step-100 routing window

- Status: TPU measured; accepted training remained live after observation.
- Run identity: unchanged from `2026-07-26/#16`; existing TPU `spatial-se-400m`, clean commit `db16ffdf9ecb1a65cdf5ea0cfe3702b790430055`, canonical 20B config, GCS run `run_vspatial-r1-v4.1.7.4_20260726_135537_3201`, batch 1024, sequence 512, and mesh `16x2`/32 devices.
- Result: step 100 completed with loss/CE 7.5920, finite grad norm 5.12, an 11.086 s/it 50-step window, and 47,199.0 tokens/s. Attention/RST selected-space active counts remained exactly 2.000 with zero fallback. Top-1 fractions were 0.795/0.839; bundle primary token counts ranged from 1,012 to 29,353 for attention and 461 to 30,484 for RST, with mean counts 10,715.8/10,757.5 and padding 6.76%/5.78%.
- Lesson: routing had become strongly skewed by step 100 while exact selected-space execution remained complete. The accepted `Tcap=3072` primary bucket therefore still requires compact overflow replay; no-drop semantics remain mandatory for the analytic candidate.
- Decision/next: preserve this accepted state with the cooperative SIGTERM emergency-save path before using the same TPU for the bounded analytic VJP gate.
- Evidence: primary `~/train.log` on gcloud worker 6, observed 2026-07-26 KST.

#### #19 — Implement a dense RW analytic chunk pullback

- Status: candidate implemented and locally validated; TPU not yet measured.
- Hypothesis/change: retain the accepted bundle4 forward, exact top-2 bucket/overflow topology, dense read/write GEMMs, precision boundaries, normalization, tau map, denominator, and every gradient, but replace general autodiff through each canonical linear-angular RW chunk with an explicit pullback for write GEMM, gate/tau algebra, read GEMM, and read/write normalization.
- Run identity: branch `codex/v4174-dense-rw-analytic-vjp`; source parent `d478168f88d844514529af4e8d517bbd40762fa8`; dirty candidate model SHA-256 `4bba4a64915d95925afeef46cc88a4f740b099ee7e7e341f0b214e096c9de020`; config SHA-256 `2945e84a0b3e989ec50da337f322ec20bd0ab2f3933e3887c6ce41f2353fcc68`; no checkpoint or TPU execution in this entry. Local environment: Python 3.12.7, JAX 0.6.2, Flax 0.10.7.
- Local validation: dense synthetic cases covered FP32/BF16 throughput dots, padded chunks, token masks, pruning at 0 and 0.05, and an exact `margin == 0` boundary. Forward outputs were bit-identical; maximum gradient absolute difference versus the former autodiff path was `9.5367e-7`, with all gradient cosines effectively 1.0. Existing checkpoint/JIT/diagnostics and 20-step mixed-precision paths passed `3 passed in 36.92s`; `py_compile` and `git diff --check` passed.
- Lesson: the accepted BF16 primal's transpose rounds dot-result cotangents at the BF16 cast boundary without quantizing the upstream FP32 cotangent. Matching that exact order is required for near-bitwise parity; simply reusing the forward BF16 einsum helper in reverse is not equivalent.
- Decision/next: publish this isolated candidate and run the real-data two-layer `--fast --backward-profile --detailed-profile-layers 2 --no-xla-dump` gate against the clean accepted control. Require at least 10% combined QKV+RST improvement and no compile/runtime-HBM regression before any full train-step A/B.
- Evidence: `models/dawn_srw_v4174.py`; this notebook entry; focused local validation output from 2026-07-26 KST.

#### #20 — Accept the dense RW analytic pullback and establish reusable baseline generations

- Status: accepted; local validated, TPU measured, and promoted to active baseline generation `g2`.
- Hypothesis/change: keep the accepted dense primal and replace only the canonical linear-angular RW chunk pullback with explicit analytic derivatives. Formalize control reuse so later candidates do not rerun a stable baseline, while forcing one new baseline when experiment-defining structure, config, runtime, TPU/mesh, data, or profiler invariants change.
- Matched run identity: existing TPU `spatial-se-400m`; real C4 data; no checkpoint; seed 1; config `configs/train_config_v4174_400M_c4_20B_v4_64.yaml` at Git blob `d1b0e0895a7238eb01d6f513cb8ed340ea697bf5` / SHA-256 `2945e84a0b3e989ec50da337f322ec20bd0ab2f3933e3887c6ce41f2353fcc68`; batch 1024, sequence 512, mesh `16x2`/32 devices; benchmark blob `e3454ca75b8b0bfd08a8e92c5265b84c73f4078e`; launcher blob `0f26c5865c6b1b382074143b2d8c7495c82b6db6`; setup blob `e74e14f95071371a8aa374b92deb7c42b1d27f82`; Python 3.10.12, JAX 0.6.2, Flax 0.10.7, Orbax 0.11.24; clean remote checkouts.
- Control/candidate identity: generation `g1` was branch `codex/v4167-poc`, commit `d478168f88d844514529af4e8d517bbd40762fa8`, model blob `82c490399bf05888a59c18c209f599ff4586efc0`. Generation `g2` is branch `codex/v4174-dense-rw-analytic-vjp`, commit `7b14a45668e6d262eede1fe43c9344d0b7ddcd86`, model blob `893816a4d01737055275c485139304f5626abed3`.
- Two-layer gate: control/candidate forward was 2.459801/2.458540 s. Combined QKV+RST forward-plus-backward was 0.855218/0.725111 s, a 15.21% candidate improvement; subtracting the matched forward totals gave 0.626484/0.496151 s of backward-only work, a 20.80% improvement. Detailed split improved 1.132877/1.003114 s, or 11.45%; detailed compile improved 133.636/98.256 s, or 26.47%; runtime peak HBM was 4.961921/4.958973 GiB.
- Full train result: with three warmups and five measured forward+backward+AdamW steps, control/candidate mean was 11.032429/9.781903 s, an 11.33% reduction; median was 10.997462/9.754085 s; throughput was 47,523.7/53,599.0 tokens/s, a 12.78% increase. Initial compile improved 101.858/99.594 s, or 2.22%; runtime peak HBM was 7.942044/7.939323 GiB. Both summaries were valid and no OOM, NaN, Inf, or process loss occurred.
- Numerical evidence: focused local FP32/BF16, padding, mask, pruning, and exact-boundary parity produced bit-identical outputs and maximum gradient absolute difference `9.5367e-7`; all gradient cosines were effectively 1.0. TPU losses and gradients remained finite through both matched five-step trajectories.
- Lesson: the analytic pullback removes enough generic autodiff overhead to survive the full 18-layer optimizer graph; unlike the earlier end-to-end rematerialization candidate, the isolated backward win produces a material full-step win without observed compile or runtime-HBM cost.
- Baseline decision: promote the already measured candidate to active generation `g2`; do not rerun it merely to call it a control. Later candidates compare against the stored `g2` measurements unless `docs/v4174_dense_rw_baselines.yaml` declares an invariant invalid. If an accepted structural change lacks both canonical measurements, measure it once and create the next generation before continuing.
- Evidence: `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_dense_rw_analytic_control_d478/benchmark_metrics_host_6.jsonl`; `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_dense_rw_analytic_candidate_7b14a456/benchmark_metrics_host_6.jsonl`; `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_dense_rw_fullstep_control_d478/benchmark_metrics_host_6.jsonl`; `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_dense_rw_fullstep_candidate_7b14a456/benchmark_metrics_host_6.jsonl`; `docs/v4174_dense_rw_baselines.yaml`.

#### #21 — Leave the accepted generation running as the non-idle workload

- Status: active; launcher, distributed startup, and first optimizer step passed.
- Intent: replace the completed benchmark with a disposable canonical workload so the explicitly authorized existing TPU `spatial-se-400m` is not left idle while the user prepares a later initialization change. This run is not used to remeasure or promote the baseline.
- Run identity: existing READY TPU-v4-64 `spatial-se-400m`; branch `codex/v4174-dense-rw-analytic-vjp`; clean tracked source commit `7b14a45668e6d262eede1fe43c9344d0b7ddcd86`; remote worktree has only untracked raw `benchmark_runs/`; config `configs/train_config_v4174_400M_c4_20B_v4_64.yaml`; selected full-config SHA-256 `b3b7995ec09571e80bf2da54d1d8234f3fb1e1c6ae91ba5bbec764fe0b39cd38`; `--from-scratch`; seed 1; batch 1024, sequence 512, mesh `16x2`/32 devices; Python 3.10.12, JAX 0.6.2, Flax 0.10.7, Orbax 0.11.24.
- Launch/result: all eight worker SSH preflights and holder-verified cleanup passed. The standard launcher started `scripts/train_jax.py` in `tmux train` on every worker at 2026-07-26 15:05 KST with combined stdout/stderr in `~/train.log`. Process 0 is gcloud worker 6; `jax.distributed` reported 8 processes and 32 global devices, created run folder `gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4174_400M_c4_20B_v4_64/run_vspatial-r1-v4.1.7.4_20260726_150551_3201`, initialized 393,803,872 parameters, completed tau calibration and Orbax/barrier setup, verified `global_step=0` across all hosts, and entered the canonical minimal training loop. Step 1 completed with loss/CE 10.7297, finite grad norm 11.74, attention/RST selected-space counts 2.000, zero fallback, and no OOM, NaN, Inf, traceback, or process loss.
- Decision/next: leave this process running. Its 109.624 s/it and 4,773.3 tokens/s first-step window includes cold JIT and is not performance evidence; replacing the run for the planned initialization change is explicitly allowed.
- Evidence: primary worker 6 `/home/madst0614/train.log`; archived final candidate benchmark log `/home/madst0614/train.v4174_dense_rw_fullstep_candidate_7b14a456_20260726T1454KST.log`; local untracked launcher captures `launch_spatial-se-400m_20260726_150537_worker_{0..7}.log` are raw operational evidence and must not be committed.

#### #22 — Consolidate the accepted analytic VJP into POC and make space initialization independent

- Status: implemented, locally validated, committed, and pushed; TPU not measured.
- Hypothesis/change: first fast-forward `codex/v4167-poc` from `d478168f` through `ee326ae0` so it contains the measured generation `g2` analytic pullback and baseline registry. On that exact best source, replace the shared semi-orthogonal base plus per-space row signs with one independently keyed semi-orthogonal `D x R` matrix per operation space. Preserve `P_m.T @ P_m ~= I`, initialize each write map as the exact transpose, retain independent read/write training, and keep all parameter shapes/counts and execution math unchanged. Define the forward fallback once as `mass <= SPACE_GATE_L1_EPS`, report both exact-zero and actual-fallback fractions, add `q/k/v_global_interface_norm` and `rst_global_update_norm`, and retain the former names plus `space_zero_gate_frac` as temporary compatibility aliases.
- Run identity: branch `codex/v4167-poc`; source parent `ee326ae0b2e8b2575d460cb57f04d1b134f84389`; dirty tracked files were `models/dawn_srw_v4174.py` SHA-256 `fefaa2536d741b521aa6d308126c5e7c3e36f5f823b4c55fc23b90a303242691`, `scripts/train_jax.py` SHA-256 `bba79b438222cb355a919fecc9b55dd8cb89fb45634c516429e595b7f7b6713b`, `tests/test_v4174_operation_spaces.py` SHA-256 `9a0d47b51fdf77429297dfcdd90a314c112cf6e056b8bcd7899e7921f17242b5`, `tests/test_minimal_pretraining_path.py` SHA-256 `87f3e17b21688e8fbd06411da64b98d343d59498be8bf9b3a1f31dc15affdc92`, `docs/v4174_dense_rw_baselines.yaml` SHA-256 `1f65909e86d4705d009b3bb983b4347bfe64bfe5f55bb85524333d1d53b9fe15`, and this notebook. The canonical config remained Git blob `d1b0e0895a7238eb01d6f513cb8ed340ea697bf5` / SHA-256 `2945e84a0b3e989ec50da337f322ec20bd0ab2f3933e3887c6ce41f2353fcc68`; no checkpoint and no TPU/GCS execution. Local environment: Python 3.12.7, JAX 0.6.2, Flax 0.10.7.
- Local validation: same seed reproduced the complete parameter tree exactly, a different seed changed the read bases, different spaces no longer shared absolute matrix entries, every per-space Gram matrix matched identity, and write maps exactly equaled read transposes. The fallback contract covered mass `0`, `0 < mass < 1e-8`, and `mass > 1e-8`; exact-zero was `1/3` while actual fallback was `2/3`, including the sharded production metric vector. The complete v4174 regression passed `21 passed, 2 skipped in 140.62s`; skips require two local devices. The minimal-pretraining contract passed `6 passed in 1.93s`; `py_compile` and `git diff --check` passed. Symbolic/tree checks retained 393,803,872 parameters.
- Validation corrections: the first full regression exposed one mixed/FP32 active-fraction difference equal to exactly one discrete gate decision and an invalid comparison of gradient norms after the mixed and FP32 optimizers had already followed different trajectories. The contract now budgets the exact one-decision resolution and enforces gradient parity at step 1 where parameters are identical, while retaining finite 20-step trajectories and loss bounds. The first minimal-path pass also exposed stale source assertions for the newly fast-forwarded analytic VJP and renamed grouped intermediates; those assertions now verify the accepted custom analytic pullback and current grouped writeback contract. No production math was loosened or changed to satisfy either test.
- Historical boundary: every recorded v4174 run through generation `g2`, including the last observed active workload, used the old shared-base/row-sign-derived initialization. Pulling this source into an already initialized process changes nothing, and resuming any old checkpoint restores its saved read/write matrices rather than invoking the new initializer. Any learning comparison for this change must therefore be a new from-scratch run with the same seed/config/data/mesh.
- Decision/next: the consolidated implementation is published on `origin/codex/v4167-poc`. Keep generation `g2` as pinned historical performance evidence, but require a fresh next-generation run before attributing training behavior to independent operation-space languages. `scripts/benchmark_srw_tpu.py` has no hard-coded consumers of the renamed norms or selector zero metric, so compatibility aliases make a benchmark edit unnecessary.
- Evidence: implementation commit `fe3d82deb814675b1e7b27e60037727d67c29795`; `models/dawn_srw_v4174.py`; `scripts/train_jax.py`; `tests/test_v4174_operation_spaces.py`; `tests/test_minimal_pretraining_path.py`; `docs/v4174_dense_rw_baselines.yaml`; this notebook; local validation output from 2026-07-26 KST.

#### #23 — Make POC the mandatory integration and baseline branch

- Status: repository policy recorded, remote branch and accepted-source audit passed, committed, and pushed; TPU not used.
- Hypothesis/change: make the existing operating rule explicit and durable: `codex/v4167-poc` is the source of truth for every POC and canonical baseline. Experimental branches remain candidate-only, and every accepted final structure, parameter/config schema, variable/metric name, runtime/launcher change, optimization, and baseline decision must be synchronized back to POC and pushed before completion.
- Run identity: branch `codex/v4167-poc`; audited tracked-clean source and `origin/codex/v4167-poc` were both `ec744a2a095227b4d928cdc058816976d10e98f0` after `git fetch origin --prune`. Pre-existing untracked local launcher logs were present and excluded. No checkpoint, TPU, GCS, model execution, or runtime measurement was involved.
- Audit result: every remote branch tip except `origin/codex/v4174-accepted-profile-gate` was already an ancestor of POC. That sole non-ancestor commit `001c2c27fd7baf6998937530cdca3e55e57fccf0` is a comparison-control snapshot whose `models/dawn_srw_v4174.py` is identical to accepted ancestor `8187c00d4b767a743c3178a5bcb8caac1c6124c5`; it contains no missing accepted work. The rejected P-RW-U remat candidate remains intentionally unpromoted.
- Accepted-source result: compact bundle commit `213001e3`, accepted exact-top-2 source `8187c00d`, explicit operation-space refactor `db16ffdf`, dense RW analytic VJP `7b14a456`, and independent initialization/metric contract `fe3d82de` are all ancestors of POC. Current source contains the analytic custom pullback and bundle checkpoint policy, not the rejected end-to-end `nothing_saveable` P-RW-U remat. It also contains the four canonical parameter roots, global-interface variable/metric names with temporary old aliases, and exact-zero versus epsilon-fallback metrics.
- Baseline boundary: generation `g2` remains historical measured evidence with its actual experimental-branch commit and old sign-derived initializer. It is not relabeled as a POC run. The next canonical generation must be measured from a clean exact POC-reachable commit because the current independent initializer changed the fingerprint.
- Decision/next: enforce the policy in `AGENTS.md`, this notebook, and the machine-readable v4174 registry. For every future accepted branch experiment, synchronize the final decision to POC before declaring it complete or using it as the canonical baseline.
- Evidence: policy commit `4e2ebedf2a88f63c274265efa1930dc26f5d85c9`; `git branch --remotes --no-merged origin/codex/v4167-poc`; ancestry checks for `213001e3`, `8187c00d`, `db16ffdf`, `7b14a456`, and `fe3d82de`; model-blob comparison between `8187c00d` and `001c2c27`; `AGENTS.md`; `docs/v4174_dense_rw_baselines.yaml`.

#### #24 — Establish generation g3 from the independent-basis POC source

- Status: TPU measured, recorded, committed, and published as the active canonical baseline generation.
- Hypothesis/change: the independently keyed per-space semi-orthogonal initializer changes the accepted initialization fingerprint, so historical generation `g2` cannot remain the current-source control. Measure the exact latest clean POC source once from scratch with the canonical two-layer gate and three-warmup/five-measure full-step protocols before beginning exact top-2 forward work.
- Run identity: existing READY TPU-v4-64 `spatial-se-400m`; branch `codex/v4167-poc`; clean tracked source and `origin/codex/v4167-poc` commit `8313cb07711b8a29163503abefde28227588e73a`; model blob `4fe109a5508343275f806373d94d5930dfdfc090` / SHA-256 `fefaa2536d741b521aa6d308126c5e7c3e36f5f823b4c55fc23b90a303242691`; config `configs/train_config_v4174_400M_c4_20B_v4_64.yaml`, blob `d1b0e0895a7238eb01d6f513cb8ed340ea697bf5` / SHA-256 `2945e84a0b3e989ec50da337f322ec20bd0ab2f3933e3887c6ce41f2353fcc68`; no checkpoint / fresh seed-1 initialization; real `gs://dawn-tpu-data-c4/c4_train_40B` data; batch 1024, sequence 512, mesh `16x2` on 32 devices/eight hosts; Python 3.10.12, JAX 0.6.2, Flax 0.10.7, Orbax 0.11.24. The benchmark, launcher, and setup blobs were `e3454ca7`, `0f26c586`, and `e74e14f9`. Remote tracked state was clean with pre-existing untracked `benchmark_runs/`; local tracked state was clean with only untracked raw launcher captures before this record.
- Launch continuity: the old-sign `g2` training workload was inspected at step 400, with 9.950 s/it in its latest 100-step logging window, then archived on every worker at `/home/madst0614/train.g2_old_sign_step400_before_g3_20260726T162022KST.log` and replaced as authorized. The first local launcher call failed before remote cleanup because Git Bash selected unsupported gcloud Python 3.9; it made no TPU mutation and left `g2` running. Retrying with the bundled gcloud Python succeeded; all eight SSH preflights, holder-verified cleanup, exact-commit checkout, dependency setup, and `tmux train` launches passed.
- Two-layer gate: fast forward compile/measure were 51.515928/2.442341 s. QKV forward and F+B were 0.125157/0.373928 s; RST forward and F+B were 0.105272/0.352597 s. Combined forward, F+B, and inferred backward were 0.230429/0.726526/0.496097 s. Detailed split/compile were 1.005572/96.882180 s and runtime peak HBM was 4.966570496 GiB. Relative to historical `g2`, F+B was 0.1950% slower, inferred backward 0.0110% faster, detailed compile 1.3987% faster, and HBM increased 0.007598 GiB.
- Full-step result: compile was 96.460215 s. The five measured steps were 9.931496, 9.865839, 9.939950, 9.857402, and 9.882364 s, for mean/median/p90 9.895410/9.882364/9.936568 s and 52,983.6 tokens/s. Peak HBM was 7.94692096 GiB. Relative to historical `g2`, mean time was 1.1604% slower, throughput 1.1482% lower, compile 3.1468% faster, and HBM increased 0.007598 GiB. The summary was valid; losses and gradients were finite, with no OOM, NaN, Inf, traceback, or process loss.
- Lesson/decision: the initializer rebaseline is operationally matched and shows no material execution regression; its small timing shift is now part of the current fingerprint rather than an optimization comparison. Promote this run to active generation `g3`, mark `current_source_requires_new_generation: false`, retain `g2` as superseded historical evidence, and use only stored `g3` measurements for the next candidate control.
- Next: implement exact selected-top-2 forward with `Tcap=3072` plus compact exact overflow while retaining the accepted analytic backward. Require at least 10% two-layer F+B improvement, 2% full-step improvement, at most 5% compile regression, at most 0.05 GiB HBM regression, and retained benefit under step-100-skew routing.
- Evidence: gate JSONL `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_dense_rw_g3_independent_control_8313cb07_gate/benchmark_metrics_host_6.jsonl`; full-step JSONL `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_dense_rw_g3_independent_control_8313cb07_fullstep/benchmark_metrics_host_6.jsonl`; archived gate/full logs `/home/madst0614/train.v4174_dense_rw_g3_independent_control_8313cb07_gate_20260726T1629KST.log` and `/home/madst0614/train.v4174_dense_rw_g3_independent_control_8313cb07_fullstep_20260726T1637KST.log`; local untracked launcher captures `launch_srw_benchmark_spatial-se-400m_20260726_162428_worker_{0..7}.log` and `launch_srw_benchmark_spatial-se-400m_20260726_163232_worker_{0..7}.log`; `docs/v4174_dense_rw_baselines.yaml`; this notebook.

#### #25 — Implement exact selected-top-2 forward with compact overflow

- Status: implemented and locally validated on a candidate branch; TPU not yet measured.
- Hypothesis/change: after the analytic pullback reduced backward, bundle4 forward still executes four physical spaces for each routed bundle and commonly computes 4–8 spaces per token despite semantic top-2 selection. Add a distinct `exact_top2` production execution mode that directly reuses the accepted per-space `Tcap=3072` bucket and compact exact overflow executor for both primal forward and analytic backward. Preserve `bundle_dense` and `dense_all_space`, all parameter roots/shapes/counts, routing weights, Q/K pairing, Q/K/V grouped execution, RST reselection, BF16/FP32 boundaries, no-drop overflow, and the final output collective.
- Implementation: `models/dawn_srw_v4174.py` adds fail-loud `exact_top2` schema validation, exact attention/RST factories, primal exact packing, the existing custom-VJP exact pullback, exact-capacity packing metrics through the stable metric keys, and forward/backward kernel contracts. `scripts/train_jax.py` selects and reports the new mode. The candidate config replaces only bundle execution fields with `operation_space_execution_mode: exact_top2`; the canonical g3 config blob remains pinned historically in the baseline registry.
- Local identity: branch `codex/v4174-exact-top2-forward`; clean parent `300a8966` on canonical POC; dirty model/trainer/config SHA-256 values `1a83e4c3fccfb605a66a3bd1342c8bda34e72f9f7009b7853f5a30dbe0c7f5ca`, `74c4305d0935561ace35b772598f760186346a17bab1d78d39f78e76b827347c`, and `052dfda7315297ab93c8a8e2b9628c2fe16a0584fcba64f7cd34ee6e1c5c6b84`; no checkpoint or TPU/GCS execution. Local environment: Python 3.12.7, JAX 0.6.2, Flax 0.10.7.
- Local validation: `py_compile` passed for the model, trainer, and benchmark; config materialization selected `exact_top2` with no legacy bundle fields and retained 393,803,872 symbolic parameters; one-device factory construction reported exact forward and backward contracts for attention and RST. Small FP32 bundle-versus-exact attention outputs differed by at most `2.3283e-10` / `6.8530e-8` relative, while RST was bit-identical. With forced overflow (`Tcap=2`, maximum selected-space count 8), attention/RST loss differences were `3.4925e-10` and `7.2760e-11`; all input/parameter gradients were bit-identical with zero relative L2 difference. The focused canonical count and trainer-factory contracts passed `2 passed in 49.28s`; `git diff --check` passed.
- Decision/next: publish the candidate branch and run only the real-data two-layer `--fast --backward-profile --detailed-profile-layers 2 --no-xla-dump` gate against stored `g3`. Require at least 10% combined QKV+RST F+B improvement with no compile/HBM/process failure before any full-step or skew-state run.
- Evidence: `models/dawn_srw_v4174.py`; `scripts/train_jax.py`; `configs/train_config_v4174_400M_c4_20B_v4_64.yaml`; local validation output from 2026-07-26 KST; this notebook.

#### #26 — Screen the first exact selected-top-2 forward candidate

- Status: TPU measured and rejected at the mandatory first gate; full-step and step-100-skew protocols intentionally not run.
- Hypothesis/change: replace bundle4 primal execution with exact selected-top-2 `Tcap=3072` primary buckets plus compact exact overflow while retaining the accepted analytic custom-VJP backward. The expected forward work reduction must improve the stored generation-`g3` combined two-layer QKV+RST F+B time by at least 10% before any broader benchmark.
- Run identity: existing READY TPU-v4-64 `spatial-se-400m`; branch `codex/v4174-exact-top2-forward`; clean tracked source and `origin/codex/v4174-exact-top2-forward` commit `1a1b8c7180891aa0704072264434cf0434df0326`; model/trainer/config blobs `2d44c778c84c070c0313cdbd08a8594ea6ca3416`, `4bc09037a282cd904a682b69dcc71cd4fe1dc5e3`, and `6bd72b7337468fa73f8d7e7875cf5d5fd0191743`; model/trainer/config SHA-256 values `1a83e4c3fccfb605a66a3bd1342c8bda34e72f9f7009b7853f5a30dbe0c7f5ca`, `74c4305d0935561ace35b772598f760186346a17bab1d78d39f78e76b827347c`, and `052dfda7315297ab93c8a8e2b9628c2fe16a0584fcba64f7cd34ee6e1c5c6b84`; no checkpoint / fresh seed-1 initialization; real `gs://dawn-tpu-data-c4/c4_train_40B` data; batch 1024, sequence 512, mesh `16x2` on 32 devices/eight hosts; Python 3.10.12, JAX 0.6.2, Flax 0.10.7, Orbax 0.11.24. Every worker verified the exact commit before launch; remote tracked state was clean with pre-existing untracked `benchmark_runs/`.
- Gate result: the summary was valid with no OOM, NaN, Inf, traceback, or process loss. QKV forward and F+B were 0.134832/0.374316 s; RST forward and F+B were 0.070643/0.319990 s. Combined forward and F+B were 0.205475 and 0.694306 s. Against stored `g3`, forward improved 10.8295%, but F+B improved only 4.4347%; QKV F+B regressed 0.1037% while RST F+B improved 9.2477%. The mandatory 10% F+B threshold is 0.653873 s or less, leaving the candidate 0.040433 s above the gate.
- Secondary diagnostics: fast-forward compile improved 61.10% from 51.515928 to 20.038446 s; detailed compile improved 9.04% from 96.882180 to 88.120281 s. Runtime peak HBM increased 0.010758 GiB from 4.966570496 to 4.977328128 GiB, within the 0.05-GiB limit. These secondary passes do not override the failed primary F+B gate.
- Lesson/decision: exact bucketing alone is insufficient because the all-24-space primary batch regresses QKV forward despite reducing physical selected-space work; only RST realizes a material forward gain. Reject this exact execution shape and do not run its five-step or skew-state protocols. Preserve the candidate branch as evidence and try one forward-only four-space primary grouping refinement while leaving the accepted analytic backward unchanged.
- Evidence: JSONL `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_exact_top2_forward_candidate_1a1b8c71_gate/benchmark_metrics_host_6.jsonl`; archived raw log `/home/madst0614/train.v4174_exact_top2_forward_candidate_1a1b8c71_gate_20260726T1711KST.log` on all workers; local untracked launcher captures `launch_srw_benchmark_spatial-se-400m_20260726_170406_worker_{0..7}.log`; this notebook.

#### #27 — Group exact forward primary buckets in four-space MXU batches

- Status: implemented and locally validated on the candidate branch; TPU not yet measured.
- Hypothesis/change: the first exact forward's all-24-space primary batch lost QKV efficiency even though it reduced physical space-token work. Keep the same exact per-space `Tcap=3072` packing and compact overflow, but scan primary buckets in six groups of four spaces so each dense contraction retains the accepted bundle4 MXU batch shape while every token-space pair remains semantically exact. This grouping is forward-only: the custom-VJP backward still invokes the existing all-space exact primal and accepted analytic RW pullback.
- Local identity: branch `codex/v4174-exact-top2-forward`; clean parent `9a203dd0ca2732f8af0c0f90efd1883799e6feae`; dirty model SHA-256 `0baade943743fe5795d0e5f9cec10a7a5fe41f14104ca440b239cd986ae195d5`; trainer and config remain unchanged at SHA-256 `74c4305d0935561ace35b772598f760186346a17bab1d78d39f78e76b827347c` and `052dfda7315297ab93c8a8e2b9628c2fe16a0584fcba64f7cd34ee6e1c5c6b84`. No checkpoint, TPU, GCS, or remote execution was involved. Local environment: Python 3.12.7, JAX 0.6.2, Flax 0.10.7.
- Implementation: attention and RST exact executors share their primary-space arithmetic between the unchanged all-space backward path and a static forward grouping path. The new forward path slices exact bucket metadata and every matching interface/controller/operator tensor in groups of four, executes no unselected space, accumulates exact writeback, then uses the unchanged compact overflow executor. Runtime contracts expose and fail-loud validate forward primary group size 4; `M=24`, `K=2`, bucket capacity, overflow grouping, parameter shapes/count, precision policy, collectives, routing, and backward remain unchanged.
- Local validation: `py_compile` passed for the model, trainer, and TPU benchmark. On one-device FP32 reference kernels with `M=24`, group4 versus the previous group24 exact forward differed by at most `2.3842e-7`, with attention/RST relative L2 differences `5.9600e-8` and `4.6498e-8`. Linear losses were identical and every attention/RST input, controller, interface, operator-bank, routing, and scale gradient was bit-identical (`max_abs=0`, relative L2 `0`). The focused canonical parameter-count and trainer-factory contracts passed `2 passed in 44.85s`; `git diff --check` passed. No test file was created or modified.
- Decision/next: publish the refinement, retain the two-layer result as a diagnostic, and run the matched three-warmup/five-measure full-step protocol as the primary comparison. Do not infer keep/discard from either metric; report the evidence for user decision.
- Evidence: `models/dawn_srw_v4174.py`; local numerical/gradient validation output from 2026-07-26 KST; this notebook.

#### #28 — Move exact-forward candidate judgment to the full-step protocol

- Status: user-specified evaluation policy recorded; no new runtime measurement in this entry.
- User decision/correction: the earlier 10% two-layer F+B threshold is no longer an automatic discard gate. Use matched three-warmup/five-measure full-step behavior as the primary candidate criterion, report results only, and reserve every keep, hold, or discard decision for the user.
- Consequence: experiment #26 remains the exact historical diagnostic result, but its procedural `rejected` label does not constitute a candidate disposition. The all-24-space exact candidate remains available and must receive its missing full-step measurement. The group4 forward refinement is a separate follow-on candidate and will also be reported without an agent-authored disposition.
- Reporting boundary: include two-layer QKV/RST timing, full-step mean/median/p90 and throughput, compile, peak HBM, validity/finiteness/process status, and step-100-skew behavior when measured. Distinguish unmeasured evidence explicitly and do not promote, merge to POC, or delete candidate work without a later user decision.
- Next: run commit `1a1b8c7180891aa0704072264434cf0434df0326` with three warmups and five measured full steps, then measure the group4 refinement under the same protocol.
- Evidence: user instruction in the active 2026-07-26 task; this notebook.

#### #29 — Measure the all-24-space exact candidate by full-step

- Status: TPU measured and reported without a keep/discard disposition.
- Run identity: existing READY TPU-v4-64 `spatial-se-400m`; fixed comparison branch `codex/v4174-exact-top2-forward-all24`; clean tracked source and remote commit `1a1b8c7180891aa0704072264434cf0434df0326`; model/trainer/config blobs `2d44c778c84c070c0313cdbd08a8594ea6ca3416`, `4bc09037a282cd904a682b69dcc71cd4fe1dc5e3`, and `6bd72b7337468fa73f8d7e7875cf5d5fd0191743`; no checkpoint / fresh seed-1 initialization; real `gs://dawn-tpu-data-c4/c4_train_40B` data; batch 1024, sequence 512, mesh `16x2` on 32 devices/eight hosts; Python 3.10.12, JAX 0.6.2, Flax 0.10.7, Orbax 0.11.24. All workers reported the exact commit before execution; remote tracked state was clean with pre-existing untracked benchmark artifacts.
- Full-step result: compile was 72.310812 s. After three warmups, the five measured steps were 10.730821, 10.839554, 10.726553, 10.718062, and 10.766030 s, for mean/median/p90 10.756204/10.730821/10.810144 s and 48,743.7 tokens/s. Peak HBM was 7.954259968 GiB. The summary was valid; all reported losses and gradients were finite, with no OOM, NaN, Inf, traceback, or process loss.
- Comparison to stored `g3`: full-step mean/median/p90 were 8.6989%/8.5856%/8.7915% higher and throughput was 8.0023% lower. Compile was 25.0356% shorter and peak HBM increased 0.007339 GiB. The already recorded two-layer diagnostic improved combined forward 10.8295% and combined F+B 4.4347%; these are diagnostics, not an automatic disposition.
- Reporting boundary: no step-100-skew run was performed, and no keep/discard conclusion is attached. The fixed all24 branch remains published for user-directed follow-up.
- Evidence: JSONL `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_exact_top2_forward_candidate_1a1b8c71_fullstep/benchmark_metrics_host_6.jsonl`; archived raw log `/home/madst0614/train.v4174_exact_top2_forward_candidate_1a1b8c71_fullstep_20260726T1734KST.log` on all workers; local untracked launcher captures `launch_srw_benchmark_spatial-se-400m_20260726_172914_worker_{0..7}.log`; experiment #26 diagnostic evidence; this notebook.

#### #30 — Measure the group4 exact-forward refinement

- Status: TPU gate and full-step measured and reported without a keep/discard disposition.
- Run identity: existing READY TPU-v4-64 `spatial-se-400m`; branch `codex/v4174-exact-top2-forward`; clean tracked source and remote commit `73f04c4894d656144c934a39bc43b5d854bb47a8`; model/trainer/benchmark/config blobs `055c779d7d0428aff423119dbd308624edaacf99`, `4bc09037a282cd904a682b69dcc71cd4fe1dc5e3`, `e3454ca75b8b0bfd08a8e92c5265b84c73f4078e`, and `6bd72b7337468fa73f8d7e7875cf5d5fd0191743`; no checkpoint / fresh seed-1 initialization; real C4 data; batch 1024, sequence 512, mesh `16x2` on 32 devices/eight hosts; Python 3.10.12, JAX 0.6.2, Flax 0.10.7, Orbax 0.11.24. All workers reported the exact commit before execution; remote tracked state was clean with pre-existing untracked benchmark artifacts.
- Two-layer diagnostic: QKV forward/F+B were 0.133412/0.382820 s and RST forward/F+B were 0.096549/0.353259 s, for combined forward/F+B 0.229962/0.736079 s. Relative to `g3`, combined forward improved 0.2028% while combined F+B increased 1.3150%. Fast/detailed compile were 19.732852/90.198896 s, 61.6956%/6.8984% shorter than `g3`; peak HBM was 4.976562176 GiB, up 0.009992 GiB. The summary was valid with no process or numerical failure.
- Full-step result: compile was 72.806075 s. After three warmups, the five measured steps were 10.861779, 10.830850, 10.739313, 10.753277, and 10.771562 s, for mean/median/p90 10.791356/10.771562/10.849407 s and 48,585.0 tokens/s. Peak HBM was 7.952774656 GiB. The summary was valid; all reported losses and gradients were finite, with no OOM, NaN, Inf, traceback, or process loss.
- Comparison: versus stored `g3`, full-step mean/median/p90 increased 9.0542%/8.9978%/9.1867%, throughput decreased 8.3018%, compile shortened 24.5222%, and peak HBM increased 0.005854 GiB. Versus the all24 candidate, group4 full-step mean increased 0.3268% and two-layer F+B increased 6.0165%.
- Reporting boundary: no step-100-skew run was performed, and no keep/discard conclusion is attached. The group4 branch remains published for user-directed follow-up.
- Evidence: gate/full JSONL `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_exact_top2_forward_group4_73f04c48_gate/benchmark_metrics_host_6.jsonl` and `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_exact_top2_forward_group4_73f04c48_fullstep/benchmark_metrics_host_6.jsonl`; archived raw logs `/home/madst0614/train.v4174_exact_top2_forward_group4_73f04c48_gate_20260726T1741KST.log` and `/home/madst0614/train.v4174_exact_top2_forward_group4_73f04c48_fullstep_20260726T1750KST.log` on all workers; local untracked launcher captures `launch_srw_benchmark_spatial-se-400m_20260726_173754_worker_{0..7}.log` and `launch_srw_benchmark_spatial-se-400m_20260726_174416_worker_{0..7}.log`; this notebook.

#### #31 — Audit step-100-skew replay availability

- Status: read-only evidence audit completed; skew execution not measured.
- Audit: generation `g3` was established by benchmark-only from-scratch protocols and produced no checkpoint. The earlier recorded step-100 routing window came from source commit `db16ffdf` and GCS run `gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4174_400M_c4_20B_v4_64/run_vspatial-r1-v4.1.7.4_20260726_135537_3201`; read-only `gcloud storage ls` showed its `checkpoints/` and `best_checkpoints/` prefixes contain no saved step. Its logs retain aggregate maxima 29,353 attention and 30,484 RST tokens, but not the exact per-token selected-space metadata required for matched replay.
- Boundary: restoring an older unrelated checkpoint would change the initializer/projection fingerprint and would not be a matched `g3` comparison. Aggregate counts alone are insufficient to reconstruct exact no-drop overflow work. Therefore step-100 skew is explicitly `not measured`, not silently treated as passed or failed.
- Next: because the user changed the current candidate criterion to full-step and reserved disposition, do not start a new long step-100 training/replay workflow without their follow-up direction.
- Evidence: experiment #18 aggregate routing record; read-only GCS listing on 2026-07-26 KST; generation-`g3` benchmark identity in experiment #24; this notebook.

#### #32 — Close exact forward and implement a QKV P/RW/U analytic pullback

- Status: exact-forward line rejected by user; QKV-only successor implemented and locally parity-validated; TPU not yet measured.
- Disposition: discard all-24 exact forward commit `1a1b8c71` and group4 exact forward commit `73f04c48`; retain canonical bundle4 forward and generation `g3`. Do not run step-100 skew for either rejected candidate. The missing skew result remains `not measured`; the expectation that added routing skew and overflow would worsen exact forward is inference only. Candidate source remains on `origin/codex/v4174-exact-top2-forward`, while its detailed implementation and runtime evidence is centralized in experiments #25–#31 above.
- Hypothesis/change: preserve bundle4 primal execution and exact top-2 backward packing, but remove the generic outer QKV autodiff graph. The new linear-angular path directly transposes the BF16/FP32 output collective, U writeback, routing weights and shared Q/K scale, global composition denominator, tau projections, shared local norm, and P projection; it invokes the already accepted analytic RW chunk pullback for Q/K and V, sums their local-state cotangents, and performs P backward once per exact primary or overflow group. RST and every non-linear composition mode retain their previous generic VJP paths.
- Local identity: branch `codex/v4174-p-rw-u-analytic-vjp`; clean canonical parent and current `origin/codex/v4167-poc` commit `300a8966f9261904fa1a7e685b8e4e648f89a1f2`; dirty model SHA-256 `d6fd0b9b5bb14036b9b55c489c7c0b11f8d4c604c8c368d15d89a63df4606a5c`; unchanged canonical config Git blob `d1b0e0895a7238eb01d6f513cb8ed340ea697bf5`; no checkpoint, TPU, GCS, or remote process action. Local environment: Python 3.12.7, JAX 0.6.2, Flax 0.10.7. Pre-existing untracked launcher captures were preserved and excluded.
- Numerical validation: against a detached clean-parent generic-VJP reference, forced overflow with exact bucket capacity 2 passed for the complete attention kernel. FP32 forward loss was identical; all state, selector/routing, P, U, Q/K/V tau kernel/bias, Q/K/V read/write bank, and scale gradients had maximum absolute difference `7.4506e-9` and minimum cosine `0.99999982`. BF16 production boundaries were identical in forward loss with maximum gradient difference `3.7253e-9` and minimum cosine `0.99999988`. A two-device model-axis BF16 collective comparison retained identical loss, maximum gradient difference `2.0489e-8`, and minimum cosine `0.99999976`. Invalid primary/overflow padding was exercised by the compact task masks. A constructed exact `rho == tau == 0.5000001192` / `margin == 0` FP32 case was bit-identical in every gradient (`max_abs=0`).
- Focused validation: model `py_compile` and `git diff --check` passed; the existing minimal-pretraining contract passed `6 passed in 3.19s`; canonical config/schema and symbolic-count checks passed `2 passed in 3.06s`. No test or debug file was created or modified.
- Decision/next: commit and publish this isolated candidate, then run the stored-`g3` two-layer gate on `spatial-se-400m`. Require combined F+B at or below `0.653873` s, compile regression at or below 5%, peak-HBM increase at or below 0.05 GiB, and no numerical/process failure before any RST implementation or full-step run.
- Evidence: `models/dawn_srw_v4174.py`; detached clean-parent comparison at `C:\tmp\dawn-poc-ref`; focused local validation output from 2026-07-26 KST; this notebook.

#### #33 — Reject the QKV P/RW/U analytic candidate at the combined gate

- Status: TPU measured; valid run; mandatory two-layer gate failed; RST extension and full-step intentionally not run.
- Run identity: existing READY TPU-v4-64 `spatial-se-400m`; branch `codex/v4174-p-rw-u-analytic-vjp`; clean tracked source and `origin/codex/v4174-p-rw-u-analytic-vjp` commit `7123dffa9400ba8ddf79b979a3dd6e3f79f1d229`; model/config/benchmark/launcher/setup blobs `231fcc185fe444b98f39f8f5147927f9feeee4b2`, `d1b0e0895a7238eb01d6f513cb8ed340ea697bf5`, `e3454ca75b8b0bfd08a8e92c5265b84c73f4078e`, `0f26c5865c6b1b382074143b2d8c7495c82b6db6`, and `e74e14f95071371a8aa374b92deb7c42b1d27f82`; no checkpoint / fresh seed-1 initialization; real C4 data; batch 1024, sequence 512, mesh `16x2` on 32 devices/eight hosts; Python 3.10.12, JAX 0.6.2, Flax 0.10.7, Orbax 0.11.24. Every worker verified the exact commit and a clean tracked checkout; pre-existing untracked benchmark artifacts remained.
- Launch continuity: the completed group4 benchmark wrappers were inspected before replacement. All eight individual SSH preflights and holder-verified cleanup checks passed, the standard launcher cleared and reused `~/train.log`, and every worker started `scripts/benchmark_srw_tpu.py` in `tmux train` with `--fast --backward-profile --detailed-profile-layers 2 --no-xla-dump`.
- Gate result: QKV forward/F+B were `0.124946434`/`0.325455888` s and unchanged RST forward/F+B were `0.105301555`/`0.352556237` s. Combined forward/F+B were `0.230247989`/`0.678012125` s, with inferred combined backward `0.447764136` s. Fast forward compile/measure were `53.001948`/`2.442862` s; detailed split/compile were `0.957305`/`96.819576` s; profile peak HBM was `4.962752` GiB. The summary was valid with finite loss/gradient checks and no OOM, NaN, Inf, traceback, or process loss.
- Comparison to stored `g3`: QKV F+B improved 12.9630% and inferred QKV backward improved 19.4000%; RST F+B improved only 0.0116%. Combined F+B improved 6.6775% and inferred combined backward improved 9.7426%. Fast-forward compile regressed 2.8846%, detailed compile improved 0.0646%, and peak HBM decreased 0.003818 GiB; all secondary limits passed.
- Gate decision: the required combined F+B was at most `0.653873` s. The candidate was `0.024139` s above it, so do not extend the analytic boundary to RST and do not run the full-step protocol. Preserve the branch as rejected evidence and keep `g3` on canonical POC unchanged. With RST fixed at the measured control, QKV F+B would need to reach at most `0.301276` s for a revised QKV-only candidate to pass.
- Evidence correction: the local path `C:\tmp\dawn-poc-ref` named in experiment #32 was a temporary detached validation worktree and was removed after parity passed. Its reproducible source identity is clean parent commit `300a8966f9261904fa1a7e685b8e4e648f89a1f2`.
- Evidence: JSONL `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_p_rw_u_analytic_qkv_7123dffa_gate/benchmark_metrics_host_6.jsonl`; archived raw log `/home/madst0614/train.v4174_p_rw_u_analytic_qkv_7123dffa_gate_20260726T1854KST.log` on all eight workers; local untracked launcher captures `launch_srw_benchmark_spatial-se-400m_20260726_185134_worker_{0..7}.log`; this notebook.

#### #34 — Centralize all final experiment reporting on POC

- Status: repository policy corrected and historical candidate reports consolidated; reporting-only change; no TPU or model execution.
- User decision/change: experiments may be implemented and measured on their own branches, but final reporting must be centralized and pushed from `codex/v4167-poc`. This applies to accepted, rejected, blocked, and inconclusive attempts. Candidate code remains isolated unless separately accepted.
- Source identity: branch `codex/v4167-poc`; clean tracked parent and fetched `origin/codex/v4167-poc` commit `300a8966f9261904fa1a7e685b8e4e648f89a1f2`. Exact-forward reports were read from `origin/codex/v4174-exact-top2-forward` at `37fd6c3cbad1bbb78b9d80e6913df5e060ddbe3d`; QKV analytic reports were read from `origin/codex/v4174-p-rw-u-analytic-vjp` at `46f0a77eb3d09eb6385f580dc55484479591b952`. Pre-existing untracked launcher captures were preserved and excluded.
- Consolidation: copied stable experiments #25–#31 and #32–#33 into the POC notebook, updated `Current State` and `Fixed Context`, and made the reporting-only POC workflow explicit in `AGENTS.md`. No model, trainer, config, benchmark, test, or candidate source file was integrated.
- Boundary: experiment identities continue to name the exact candidate branches and commits actually run. Centralizing their reports does not relabel those measurements as POC runs, make candidate commits POC ancestors, or promote rejected source.
- Decision/next: after each future material attempt, first preserve the exact candidate evidence, then append and push the final report on POC. Merge candidate code into POC only after an explicit acceptance decision.
- Evidence: `AGENTS.md`; this notebook; fetched remote branch tips and tracked-file diff on 2026-07-26 KST.

#### #35 — Extend the P/RW/U analytic pullback through RST and pass the bounded gate

- Status: accepted candidate implementation; local parity validated, committed/pushed on its experiment branch, and TPU two-layer gate passed.
- User decision/change: supersede experiment #33's stop decision and extend the already successful QKV outer analytic pullback through RST instead of forcing QKV alone to carry the combined gate. Preserve bundle4 forward, attention-then-RST reselection, exact selected-top-2 backward packing, `Tcap=3072`, compact exact overflow, BF16/FP32 boundaries, and every routing/tau/P/U/operator/state gradient.
- Implementation: added the linear-angular RST group pullback and connected global output cotangent through the BF16/FP32 collective boundary, U writeback, selected-space weight and global-denominator pullbacks, the accepted RW chunk pullback, tau kernel/bias, local normalization, and P projection. Non-linear composition modes retain the generic VJP fallback. The QKV implementation from commit `7123dffa` is unchanged.
- Candidate identity: branch `codex/v4174-p-rw-u-analytic-vjp-rst`; clean tracked commit `aa7ed52117d5dc4427454815b70ca6533dab3589`; model Git blob `aaee2b34815a5867b9cc0d8f3b718fb71a0d56d8` / committed-content SHA-256 `735c2967a535e6afdbdc6952a7d4ab734c5f8c7e92677b44c7b67d4beef516d8`; config/benchmark/launcher/setup blobs `d1b0e0895a7238eb01d6f513cb8ed340ea697bf5`, `e3454ca75b8b0bfd08a8e92c5265b84c73f4078e`, `0f26c5865c6b1b382074143b2d8c7495c82b6db6`, and `e74e14f95071371a8aa374b92deb7c42b1d27f82`. No checkpoint / fresh seed-1 initialization; real C4 data; batch 1024, sequence 512, mesh `16x2` on 32 devices/eight hosts; remote Python 3.10.12, JAX 0.6.2, Flax 0.10.7, Orbax 0.11.24. Remote tracked state was clean with only pre-existing untracked `benchmark_runs/`.
- Launch continuity: the first foreground launcher invocation reached the sequential SSH preflight but exceeded the local 60-second wait before any remote replacement; a read-only check confirmed that the prior completed wrapper and commit were unchanged. The same standard launcher was then started as a hidden background process, completed all eight preflights, and placed every worker on exact commit `aa7ed521`.
- Local parity: `.venv-jax062` used Python 3.12.7, JAX 0.6.2, and Flax 0.10.7. FP32 and BF16 forced-overflow/padding comparisons covered selector, P, U, tau kernel/bias, read/write banks, state, and control gradients; forward/loss matched and maximum gradient absolute error was at most `2.9802322e-8`, with minimum cosine `1.0`. A corrected normalization-aware `rho == tau == 0.5000001192` / exact `margin == 0` fixture had maximum gradient error `3.7252903e-9`; the two-device BF16 collective case had maximum error `1.4901161e-8`, both with minimum cosine `1.0`. Model `py_compile`, `git diff --check`, the existing minimal contract (`6 passed in 1.78s`), and config/schema/symbolic-count checks (`2 passed in 3.18s`) passed. No test or debug file was created or modified.
- Gate result: fast forward compile/measure were `50.159320`/`2.442458` s. QKV forward/F+B were `0.125113`/`0.326193` s and RST forward/F+B were `0.105428`/`0.302336` s. Combined forward/F+B/inferred backward were `0.230541`/`0.628530`/`0.397989` s; detailed split/compile were `0.907842`/`95.060018` s; profile peak HBM was `4.95752448` GiB. The summary was valid and no OOM, NaN, Inf, traceback, or process loss occurred.
- Comparison/decision: versus stored `g3`, QKV F+B improved `12.7658%`, RST F+B improved `14.2545%`, combined F+B improved `13.4883%`, and inferred combined backward improved `19.7760%`. The `0.653873` s gate passed by `0.025343` s. Fast and detailed compile changed `-2.6334%` and `-1.8808%`, while peak HBM changed `-0.009046` GiB. All mandatory limits passed, so proceed to the matched three-warmup/five-measure full-step protocol.
- Evidence: gate JSONL `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_p_rw_u_analytic_qkv_rst_aa7ed521_gate/benchmark_metrics_host_6.jsonl`; archived raw log `/home/madst0614/train.v4174_p_rw_u_analytic_qkv_rst_aa7ed521_gate_20260726T2005KST.log` on all eight workers; local untracked launcher captures `launch_srw_benchmark_spatial-se-400m_20260726_200302_worker_{0..7}.log`; this notebook.

#### #36 — Accept the QKV+RST analytic candidate and promote generation g4

- Status: accepted; TPU full-step measured, source integrated into canonical POC, baseline promoted to `g4`, and final evidence centralized.
- Full-step identity: existing READY TPU-v4-64 `spatial-se-400m`; branch `codex/v4174-p-rw-u-analytic-vjp-rst`; clean tracked source and commit `aa7ed52117d5dc4427454815b70ca6533dab3589`; the same model/config/benchmark/launcher/setup blobs and runtime fingerprint as experiment #35; no checkpoint / fresh seed-1 initialization; real C4 data; batch 1024, sequence 512, mesh `16x2` on 32 devices/eight hosts. Protocol arguments were `--steps 5 --warmup-steps 3 --forward-profile-steps 0 --module-profile-steps 0 --no-xla-dump`.
- Result: explicit compile was `94.137387` s. Warmups were `99.745533`, `8.800138`, and `8.915595` s; the first warmup contained deferred compilation and was excluded by protocol. The five measured steps were `9.006934`, `8.934873`, `8.937077`, `8.939692`, and `8.898478` s, for mean/median/p90 `8.943411`/`8.937077`/`8.980037` s and `58,623.7` tokens/s. Peak HBM was `7.929470464` GiB.
- Comparison to stored `g3`: full-step mean/median/p90 improved `9.6206%`/`9.5654%`/`9.6264%`, and throughput improved `10.6451%`. Compile changed `-2.4081%` and peak HBM changed `-0.017450` GiB. The exact 2% mean-time threshold was `9.697502` s, so the candidate passed by `0.754091` s. The summary was valid; every measured loss and gradient was finite, with no OOM, NaN, Inf, traceback, or process loss.
- Completion audit: all eight workers reported branch `codex/v4174-p-rw-u-analytic-vjp-rst` and exact commit `aa7ed521`; actual benchmark `python3` process count was zero on every worker; literal Traceback, RESOURCE_EXHAUSTED, Killed, NaN, and ProcessLost counts were all zero; every archived log was non-empty. An earlier audit was discarded because it matched the idle tmux wrapper's embedded command text and lost regex quoting; the corrected process/pattern audit above is the evidence used for completion.
- Promotion/integration: merge commit `62417b239d5b79c5dfa8684563532958ed212385` synchronizes the accepted QKV and RST implementation history into `codex/v4167-poc` without integrating either rejected exact-forward branch. `docs/v4174_dense_rw_baselines.yaml` directly promotes the already measured candidate as active generation `g4`; no duplicate control or promoted-candidate run is required by the registry promotion path. Generation `g3` becomes superseded matched control.
- Canonical integration validation: the POC model hashes to the exact candidate Git blob `aaee2b34815a5867b9cc0d8f3b718fb71a0d56d8`; model `py_compile`, registry parse and promotion-threshold assertions, `git diff --check`, the existing minimal contract (`6 passed in 1.71s`), and config/schema/symbolic-count checks (`2 passed in 3.16s`) passed in `.venv-jax062`. No test or debug file was created or modified.
- Boundary/next: this is an optimization-screen acceptance based on the canonical bounded gate and five-step protocol. A longer steady training window remains `not measured` and is still required before a production-acceptance claim.
- Evidence: full-step JSONL `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_p_rw_u_analytic_qkv_rst_aa7ed521_fullstep/benchmark_metrics_host_6.jsonl`; archived raw log `/home/madst0614/train.v4174_p_rw_u_analytic_qkv_rst_aa7ed521_fullstep_20260726T2017KST.log` on all eight workers; local untracked launcher captures `launch_srw_benchmark_spatial-se-400m_20260726_201149_worker_{0..7}.log`; canonical integration commit `62417b23`; baseline registry; this notebook.

#### #37 — Reject the shared operator-bank row-norm cache at the bounded gate

- Status: locally validated and TPU measured, but rejected at the mandatory two-layer gate. The candidate source remains on its experiment branch, generation `g4` stays active, no `g5` was created, and the full-step protocol is `not measured`.
- Hypothesis/change: because one raw Q/K/V/RST operator bank is shared by all 18 layers, compute only its FP32 scalar read/write row norms once before the layer scan, apply `stop_gradient` to that cache, reuse the denominators in bundle4 primal execution, and use the cached norm in the existing analytic normalization pullback while returning raw operator gradients directly. The cache contains eight scalar arrays ending in dimension one, not normalized vectors; bundle4 forward, exact top-2 backward packing, compact overflow, model equations, parameter schema, optimizer geometry, and non-linear generic-VJP fallback remain unchanged.
- Candidate identity: branch `codex/v4174-operator-bank-row-norm-cache`; canonical parent `3fde111530aa69c1c04e6c27f72b937aa2baf45d`; model implementation commit `024b474fd95290a7437fbd1933b921435fe1bfd4`; final measured commit `737b699b1906ee96396b2b27023b86dc87b43475`. Final model Git blob / committed SHA-256 were `8c576a121740133a2f77b94439f1b07e5bc3d199` / `bd89927d002ddf135b934a3dace40de806eb702a9951ee2ce98d055490f78fe6`; benchmark blob / SHA-256 were `972f41e4d2bd77edd24cae39d9f1a265adb45473` / `14a6b42b7113c140a91ec773a63f3e4ba8e275fe791a9209c131ae047a915994`. Config, launcher, and setup blobs remained `d1b0e0895a7238eb01d6f513cb8ed340ea697bf5`, `0f26c5865c6b1b382074143b2d8c7495c82b6db6`, and `e74e14f95071371a8aa374b92deb7c42b1d27f82`.
- Local validation: repository Python 3.12.7, JAX 0.6.2, and Flax 0.10.7. Against the clean generation-`g4` parent source, both FP32 and BF16 two-device QKV/RST comparisons exercised forced compact overflow with exact bucket capacity 1 or 2, token and operator padding, zero-norm rows, selector/P/U/tau/read/write/state/control gradients, and returned bit-identical loss and gradients (`max_abs=0`, cosine effectively `1`). A constructed `margin == 0` case with operator padding and token masking was also bit-identical. A canonical 24-space two-layer model completed finite scan forward, its whole parameter-gradient tree abstract-traced successfully, and the corrected detailed-profiler QKV/RST forward and F+B calls abstract-traced with the cache contract. Model/benchmark `py_compile` and `git diff --check` passed; no test or debug file was created.
- First TPU attempt and correction: existing READY TPU-v4-64 `spatial-se-400m` ran implementation commit `024b474f` with real C4 data, fresh seed-1 initialization, batch 1024, sequence 512, and mesh `16x2` on 32 devices/eight hosts. The complete 18-layer forward succeeded at about `2.4146` s, but the detailed profiler still called the fused kernel with the old 24-argument contract and could not measure the mandatory F+B gate (`in_specs` length 30 versus call length 24). Commit `737b699b` changed only that benchmark consumer: it builds the same scalar cache once, records its setup separately, and passes it to QKV/RST forward and F+B calls. The first run's `~/train.log` was superseded by the intentional rerun before archival, so no remote raw-log archive is claimed for it; its JSONL and local launcher captures remain evidence.
- Final gate identity/result: all eight workers used clean tracked commit `737b699b`; no checkpoint was restored; remote Python/JAX/Flax/Orbax were 3.10.12/0.6.2/0.10.7/0.11.24. Fast-forward compile/measure were `49.910688`/`2.415130` s. One shared cache setup took `0.001131830` s. QKV forward/F+B were `0.123070885`/`0.322923347` s; RST forward/F+B were `0.102966336`/`0.303022257` s. The raw grouped combined forward/F+B/inferred backward were `0.226037221`/`0.625945604`/`0.399908383` s; including the once-per-scan cache setup makes the fair combined F+B `0.627077434` s. Detailed split/compile were `0.907378852`/`91.094544` s and profile peak HBM was `4.959370240` GiB. The final summary was valid and finite.
- Comparison/decision: versus stored `g4`, fast forward improved `1.1189%`, grouped two-layer forward improved `1.9536%`, QKV F+B improved `1.0025%`, and RST F+B regressed `0.2269%`. Raw grouped combined F+B improved only `0.4111%`; after the shared cache setup it improved only `0.2311%`, while inferred backward regressed `0.4824%`. The fair result was `0.061400654` s above the fixed `0.565676780` s gate. Fast and detailed compile changed `-0.4957%` and `-4.1715%`, while peak HBM changed `+0.001845760` GiB; all secondary limits passed, but the primary 10% gate failed decisively. Do not run the `8.764543` s full-step gate, do not merge this source into POC, and do not continue speculative analytic-boundary widening.
- Completion/evidence: worker-by-worker audit found branch/commit agreement, zero tracked changes, zero live benchmark `python3` processes, non-empty archived logs, and zero literal `Traceback`, `RESOURCE_EXHAUSTED`, `Killed`, `NaN`, or `ProcessLost` matches. Final JSONL is `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_operator_bank_row_norm_cache_737b699b_gate/benchmark_metrics_host_6.jsonl`; final raw logs are `/home/madst0614/train.v4174_operator_bank_row_norm_cache_737b699b_gate_20260726T2137KST.log` on all workers; final local launcher captures are `launch_srw_benchmark_spatial-se-400m_20260726_213215_worker_{0..7}.log`. The first incomplete attempt is preserved at `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_operator_bank_row_norm_cache_024b474f_gate/benchmark_metrics_host_6.jsonl` and local captures `launch_srw_benchmark_spatial-se-400m_20260726_212152_worker_{0..7}.log`. Next, profile generation `g4` into exact packing, P, tau, RW read/gate/write, denominator/collective, U, and scatter before changing chunk count or the overflow executor.

#### #38 — Add a benchmark-only g4 stage-attribution factory

- Status: benchmark-only implementation and local validation complete; two-layer TPU attribution complete; one next bottleneck selected; no production source change and no full-step.
- Hypothesis/scope: do not change or speculate on generation `g4`. Attribute its remaining time with independent benchmark executables for QKV/RST, forward/backward, bundle/primary/overflow, and each configured RW chunk. Production timing is not inferred by summing split calls, and no full-step run belongs to this attribution phase.
- Implementation: added opt-in `--stage-attribution`, valid only with `--fast --backward-profile`, plus launcher passthrough. The benchmark-only factory retains production bundle4/2048-token forward blocks, exact 24-space/3072-token primary backward buckets, compact four-task overflow groups, Q/K `448x1`, V `1396x2`, and RST `704x8` local chunks. It records compile-plus-first-execution and already-compiled runtime separately for selection, packing, P, tau, RW read/gate/write, denominator/model psum, U, primary/bundle scatter, compact overflow, and routing-weight scatter; each row includes direction, executor, pool, chunk, layer, and HBM boundary data. Exact-pack metadata records primary capacity, active overflow tasks, overflow padding, and routing-count skew.
- Identity: benchmark branch `codex/v4174-g4-stage-attribution`; clean tracked commit `f3859cc61d00cd303172b9f4337a2b44364ff8a7`; exact POC parent `8b9a0fcec45e835e1a0570ddb64cab17bc03cdf4`; accepted production source remains integration commit `62417b239d5b79c5dfa8684563532958ed212385`. Benchmark and launcher Git blobs are `ffea5719e03cc4f4b1f7b208985aab4d406df7fc` and `eed13cf5dfe60d9a3bd4137f12c97c6e8f28bb4d`. `models/dawn_srw_v4174.py` and `scripts/train_jax.py` have no candidate diff. Pre-existing untracked launcher logs were preserved and excluded.
- Local parity: repository Python/JAX/Flax were 3.12.7/0.6.2/0.10.7. On reduced canonical bundle4 fixtures, staged QKV and RST forward outputs were bit-identical to the production kernels. Production-versus-staged state-gradient maximum absolute error was `2.9103830e-11`; cosine was at least `0.9999988675`. One-, two-, and four-device runs covered independent data, model, and `2x2` data/model meshes with finite QKV/RST forward and backward outputs.
- Overflow/shape validation: a forced `T=4096` skew fixture produced 6,144 primary pairs, 2,048 exact overflow pairs, two active tasks in a four-task capacity, and an overflow padding fraction of `0.8333333`; compact overflow materialization and P execution were finite. Canonical abstract trace used local `T=32768`, `M=24`, `D=2048`, and `R=256`; it confirmed bundle capacity 77,824, exact primary `[24,3072]`, compact task capacity 24, and the configured Q/K/V/RST chunk shapes. The complete benchmark detailed-profile integration ran two reduced canonical layers and emitted runtime, compile, metadata, and bottleneck summaries.
- Focused checks: benchmark `py_compile`, launcher `bash -n`, `git diff --check`, the existing minimal trainer contract (`6 passed in 1.80s`), and config/schema/symbolic-count checks (`2 passed, 21 deselected in 3.27s`) passed. No test or debug file was created or modified.
- TPU identity/baseline reproduction: existing `spatial-se-400m` v4-64, real C4 data, batch 1024, sequence 512, mesh `16x2`, fresh initialization, branch/commit `codex/v4174-g4-stage-attribution`/`f3859cc61d00cd303172b9f4337a2b44364ff8a7`, and no XLA dump. Fast-forward compile/measure were `50.124660`/`2.444341` s. The ordinary two-layer detailed profiler reproduced QKV forward/F+B `0.125285181`/`0.325259618` s and RST forward/F+B `0.106219244`/`0.303271099` s, for combined F+B `0.628530717` s. This is only `0.0001529%` slower than stored `g4` `0.628529756` s.
- Runtime attribution: split-stage totals are intentionally not additive production latency. Within those isolated calls, QKV forward was led by output scatter `29.83%` and RW read/write/gate `47.99%`; QKV backward was led by RW read/gate/write `54.96%`, exact scatter `13.93%`, and denominator/model psum `13.56%`. RST forward was led by RW read/write/gate `81.73%`; RST backward was led by RW read `28.66%`, gate `31.78%`, and write `16.58%`, totaling `77.0263%`. RST per-chunk isolated read+gate+write totals were tightly repeated: forward `237.375`-`262.099` ms, primary backward `32.270`-`34.391` ms, and overflow backward `77.467`-`86.664` ms for each of chunks 0-7.
- Executor attribution: QKV backward split time was shared/primary/overflow `3.14%`/`37.09%`/`59.77%`; RST was `2.03%`/`29.62%`/`68.36%`. Overflow is clearly expensive, but its dominant portion is again the repeated RW chunk body: RST overflow RW read/gate/write totaled `647.797` ms, while overflow metadata/scatter bookkeeping was much smaller. Therefore this result selects chunk geometry before executor metadata changes.
- Exact-pack metadata: QKV layers 0/1 had 32,030/30,120 overflow pairs, 67/60 active tasks out of 384, and `97.2848%`/`97.4467%` task-token padding. RST layers 0/1 had 30,700/24,883 overflow pairs, 61/55 active tasks, and `97.3975%`/`97.8906%` padding. Mean overflow-pair shares were only `2.9635%` QKV and `2.6504%` RST. This remains a strong later executor candidate, but packing, compact-executor bookkeeping, and routing scatter did not outrank the RST RW body in this attribution.
- Compile/HBM attribution: independent compile-plus-first-call totals were QKV forward/backward `31.847`/`89.363` s and RST forward/backward `8.184`/`33.313` s. The cumulative profiler peak was `15.021344256` GiB; because the allocator is shared across all independently compiled stages, per-record peaks are boundary observations, not isolated allocation requirements. The largest single observed used-memory delta was `+1.811939` GiB at QKV primary state scatter. No OOM occurred.
- Decision/next: choose exactly one candidate, RST RW chunks `8 -> 4`; keep Q/K `1`, V `2`, bundle4 forward, exact top-2 backward, compact no-drop overflow, all math, and production analytic boundaries unchanged. Validate locally, then require combined F+B `<= 0.565676780` s, compile `<= +5%`, HBM `<= +0.05` GiB, and no OOM/NaN/Inf/process loss. Run the matched full-step gate only if the two-layer gate passes.
- Completion/evidence: all eight workers reported clean tracked `f3859cc6`, the exact branch, zero `python3` processes, non-empty logs, and zero `Traceback`, `RESOURCE_EXHAUSTED`, `Killed`, `NaN`, or `ProcessLost` matches. Canonical JSONL is `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_g4_stage_attribution_f3859cc6/benchmark_metrics_host_6.jsonl`; archived raw logs are `/home/madst0614/train.v4174_g4_stage_attribution_f3859cc6_20260726T2303KST.log` on all workers; local captures are `launch_srw_benchmark_spatial-se-400m_20260726_230320_worker_{0..7}.log`. The JSONL contains 5,230 stage rows, 20 ordinary profile rows, four exact-pack metadata rows, and one final summary.

#### #39 — Test only RST chunk count `8 -> 4`

- Status: one-line candidate implementation and local validation complete; candidate committed and pushed; TPU gate and full-step are `not measured`.
- Hypothesis/change: the stage attribution selected the uniformly repeated RST RW body. Change only `training.n_chunks_rst` from eight to four in the canonical v4174 400M config. Q/K/V chunks remain `1/1/2`; bundle4 forward, exact top-2 analytic backward, `Tcap=3072`, compact exact overflow, parameter shapes, optimizer geometry, and all model code remain unchanged.
- Identity: branch `codex/v4174-g4-rst-chunk4`; clean tracked candidate commit `e99bda13149fedab1e05de2a9638db7703c5bbb1`; exact POC parent `77db242774ba7cfdcfbb3fcad4705358af4f1dda`; config blob `f8623ee48f28750feb79e96103aedd6511bde658`. The diff is exactly one YAML value; model, trainer, benchmark, and test files have no candidate diff.
- Canonical shape trace: local Python/JAX/Flax were 3.12.7/0.6.2/0.10.7. On a two-device model-axis mesh, symbolic parameters remained `393,803,872`; per-space/model-shard pool sizes remained Q/K/V/RST `448/448/2792/5625`; maximum chunk shapes became `448/448/1396/1407`. Abstract execution retained output `[32768,2048]`, `bundle_dense`, and `exact_top2_space_batched_dense_with_overflow`.
- Numerical parity: a two-device padded fixture with 57 local RST rows per space compared eight `max_chunk=8` calls with four `max_chunk=15` calls. FP32 output/gradient maximum absolute differences were `1.16415322e-10`/`1.30385160e-08`, gradient cosine `1.000000000`; BF16 output maximum difference was zero, gradient maximum difference `4.57763672e-05`, cosine `0.999998748`; all values were finite.
- Exact-overflow parity: a finite non-tied routing fixture with 3,200 tokens selected the same two spaces for every token and produced 6,144 primary plus 256 exact overflow pairs in four-task capacity. Eight-versus-four-chunk BF16 output maximum difference was zero, gradient maximum difference `6.86645508e-05`, cosine `0.999997437`; all outputs and gradients were finite. A discarded all-zero-query fixture landed exactly on a top-k selector tie and produced non-finite selector gradients, so it was not used as parity evidence. Two earlier setup-only failures corrected fixture tau shapes and selected-weight shape before kernel execution.
- Focused checks: `git diff --check`, canonical YAML/schema/symbolic/abstract checks, the minimal trainer contract (`6 passed in 1.70s`), and the v4174 trainer-schema/direct-diagnostics contract (`1 passed, 22 deselected in 45.20s`) passed. No test or debug file was created or modified.
- Decision/next: publish the candidate but do not promote it. Run `--fast --backward-profile --detailed-profile-layers 2 --no-xla-dump` on `spatial-se-400m`. Gate against stored `g4`: combined F+B `<= 0.565676780` s, fast and detailed compile no more than `+5%`, profile peak HBM no more than `+0.05` GiB, and no OOM/NaN/Inf/process loss. Run the five-step full train gate only if this two-layer gate passes.

#### #40 — Set QK/V/RST initializer target fractions to `8/8/5`

- Status: user-approved canonical config change integrated into POC and locally validated; TPU full-step and production launch are pending.
- Change: set only `model.tau_init_target_qk_frac`, `model.tau_init_target_v_frac`, and `model.tau_init_target_rst_frac` to `0.08`, `0.08`, and `0.05`. The production implementation remains accepted generation `g4`, with Q/K/V/RST chunks `1/1/2/8`; bundle4 forward, exact top-2 analytic backward, parameter shapes, model math, precision boundaries, and optimizer geometry are unchanged.
- Identity: isolated branch `codex/v4174-g4-tau-frac885`, exact parent `1ce88b06d52a0c1fce499f44f17daf1d9de49c7f`, frac-only commit `ce0e92699e49c793a29279fb0aecc62a195b8a47`, and POC integration merge `bb2a3ada9a0c281df9439043162dde8045d62bfd`. The integrated config Git blob is `33b63502bfce873a34965c2846709f39b4102a3a`.
- Local validation: repository Python/JAX/Flax are 3.12.7/0.6.2/0.10.7. The trainer parsed the three targets exactly as `0.08/0.08/0.05`. Continuous quantile calibration produced Q/K/V fractions `0.0799999982` and RST `0.0499999970`; the existing canonical config/schema/symbolic contract passed (`1 passed, 22 deselected in 2.95 s`). `git diff --check` passed, and no test or debug file was created or modified.
- Boundary/next: the old `8 -> 4` candidate was built before this initializer decision and is not production. First measure a clean exact POC `1/1/2/8` full-step control under the new fractions. Then create a one-line POC child changing only RST chunks to four, run its two-layer and full-step comparisons under the same fractions, and select the faster valid implementation for fresh production. TPU behavior is currently `not measured`.

### 2026-07-27

#### #01 — Rebaseline frac `8/8/5` and reject matched RST chunk4

- Status: local validation and all requested TPU protocols complete. Generation `g5` is the active production baseline with the accepted generation-`g4` execution graph, Q/K/V/RST chunks `1/1/2/8`, and initializer target fractions `0.08/0.08/0.05`. The matched RST chunk4 candidate failed both acceptance gates and is rejected.
- Matched control identity: branch `codex/v4167-poc`, clean tracked commit `ae12416b40607ef8dd21fe79a64d86b4335e18b3`, config/model/benchmark/launcher/setup Git blobs `33b63502bfce873a34965c2846709f39b4102a3a`, `aaee2b34815a5867b9cc0d8f3b718fb71a0d56d8`, `e3454ca75b8b0bfd08a8e92c5265b84c73f4078e`, `0f26c5865c6b1b382074143b2d8c7495c82b6db6`, and `e74e14f95071371a8aa374b92deb7c42b1d27f82`. The config SHA-256 was `0cea96f49f5dda5153f27f32a50acbd9cbd6af41b7d5df10182538b5bda898cb`.
- Common runtime identity: explicitly authorized existing TPU `spatial-se-400m`, TPU-v4-64, 32 devices/eight hosts, mesh `16x2`, real C4 data at `gs://dawn-tpu-data-c4/c4_train_40B`, batch 1024, sequence 512, fresh seed-1 initialization, no checkpoint, no XLA dump, and Python/JAX/Flax/Orbax `3.10.12/0.6.2/0.10.7/0.11.24`. Every worker verified the exact branch/commit and zero tracked changes; pre-existing untracked launcher captures were preserved and excluded.
- Local control validation: repository Python/JAX/Flax were `3.12.7/0.6.2/0.10.7`. The trainer parsed the targets exactly as `0.08/0.08/0.05`; continuous quantile calibration achieved Q/K/V `0.0799999982` and RST `0.0499999970`. The existing canonical symbolic config contract passed (`1 passed, 22 deselected in 2.95 s`).
- Generation-`g5` two-layer result: fast-forward compile/measure were `55.873984281`/`2.444473727` s. QKV forward/F+B were `0.125081325`/`0.326300229` s; RST forward/F+B were `0.105245936`/`0.302488350` s. Combined F+B/backward-only were `0.628788579`/`0.398461318` s; detailed split/compile were `0.907630818`/`92.705203505` s; peak HBM was `4.957524480` GiB. Versus historical `g4`, combined F+B changed `+0.041179%`, backward-only `+0.118783%`, detailed compile `-2.477187%`, and HBM `+0.000000` GiB. The one-off fast compile changed `+11.393025%`, while the unchanged graph and all runtime metrics establish runtime equivalence.
- Generation-`g5` full-step result: compile was `97.442984990` s. After warmups `101.053756421`, `8.790992591`, and `8.916142256` s, measured steps were `9.008026133`, `8.935588836`, `8.939958846`, `8.936091605`, and `8.899121047` s. Mean/median/p90 were `8.943757293`/`8.936091605`/`8.980799218` s, throughput was `58,621.468264` tokens/s, and peak HBM was `7.929470464` GiB. Versus historical `g4`, mean changed `+0.003876%`, median `-0.011023%`, p90 `+0.008488%`, throughput `-0.003855%`, compile `+3.511462%`, and HBM `+0.000000` GiB.
- Chunk4 candidate identity/change: branch `codex/v4174-g4-frac885-rst-chunk4`, exact POC parent `ae12416b40607ef8dd21fe79a64d86b4335e18b3`, clean tracked commit `a3879f21dcd3b786bdc2cde270bcca2a382796a4`, config Git blob `0e90516d0fb4d323a2e81103c07b02646158b775`, and config SHA-256 `97795dcee5ccf665d6d9bb0545273ce7c0185a9a89a3308fb27f54c82d496968`. Its only source change is `n_chunks_rst: 8 -> 4`; Q/K/V chunks, fractions, source, checkpoint, runtime, shapes, math, and launcher are matched.
- Chunk4 local validation: the canonical config reported fractions `0.08/0.08/0.05`, chunks `1/1/2/4`, 393,803,872 parameters, and local maximum Q/K/V/RST chunk shapes `448/448/1396/1407`. The existing trainer-schema/direct-diagnostics contract passed (`1 passed, 22 deselected in 51.63 s`). Earlier exact chunk4 geometry parity covered FP32/BF16, padding, forced compact overflow, all differentiable banks/state, and two-device model-axis execution.
- Chunk4 two-layer result: fast-forward compile/measure were `51.247547352`/`2.445010806` s. QKV forward/F+B were `0.125066766`/`0.325663698` s; RST forward/F+B were `0.105138766`/`0.308923610` s. Combined F+B/backward-only were `0.634587308`/`0.404381776` s; detailed split/compile were `0.931801207`/`95.577959738` s; peak HBM was `4.956194304` GiB. Versus the matched generation-`g5` control, combined F+B regressed `0.922206%`, backward-only regressed `1.485830%`, RST F+B regressed `2.127440%`, detailed compile regressed `3.098800%`, and HBM changed `-0.001330176` GiB. It missed the fixed `0.565676780` s primary gate by `0.068910528` s.
- Chunk4 full-step result: although the normal protocol stops after the failed gate, the user explicitly requested a full run to close the work. Compile was `97.776012519` s. After warmups `100.394233271`, `8.881683934`, and `8.949756052` s, measured steps were `9.050204528`, `8.940970612`, `8.868499475`, `8.897446214`, and `8.861415415` s. Mean/median/p90 were `8.923707249`/`8.897446214`/`9.006510962` s, throughput was `58,755.766787` tokens/s, and peak HBM was `7.929745920` GiB.
- Chunk4 full-step comparison/decision: versus matched chunk8, mean improved only `0.224179%`, median improved `0.432464%`, p90 regressed `0.286297%`, throughput improved `0.229094%`, compile regressed `0.341767%`, and HBM changed `+0.000275456` GiB. The required 2% line was `8.764882148` s; chunk4 missed it by `0.158825101` s. Its noise-level mean improvement cannot override the failed 10% two-layer gate, failed 2% full-step gate, and worse p90. Keep the candidate isolated and use chunk8 for production.
- Safety/evidence: every completed run had finite loss/gradient summaries and zero literal `Traceback`, `RESOURCE_EXHAUSTED`, `Killed`, `NaN`, or `ProcessLost` matches on all eight workers. Generation-`g5` gate/full JSONL paths are `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_g4_frac885_ae12416b_gate/benchmark_metrics_host_6.jsonl` and `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_g4_frac885_ae12416b_fullstep/benchmark_metrics_host_6.jsonl`; archived logs are `/home/madst0614/train.v4174_g4_frac885_ae12416b_gate_20260727T0018KST.log` and `/home/madst0614/train.v4174_g4_frac885_ae12416b_fullstep_20260726T2351KST.log` on all workers. Chunk4 gate/full JSONL paths are `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_g4_frac885_rst_chunk4_a3879f21_gate/benchmark_metrics_host_6.jsonl` and `/home/madst0614/DAWN-SRW/benchmark_runs/v4174_g4_frac885_rst_chunk4_a3879f21_fullstep/benchmark_metrics_host_6.jsonl`; archived logs are `/home/madst0614/train.v4174_g4_frac885_rst_chunk4_a3879f21_gate_20260727T0001KST.log` and `/home/madst0614/train.v4174_g4_frac885_rst_chunk4_a3879f21_fullstep_20260727T0010KST.log` on all workers.
- Next: publish this centralized registry/notebook decision on POC, then launch a fresh generation-`g5` production run from the exact pushed POC commit on `spatial-se-400m`.

#### #02 — Launch generation-`g5` fresh production

- Status: production launched and running; startup, fresh initializer calibration, distributed source identity, and the first optimizer step are verified.
- Source/run identity: standard `scripts/launch_tpu_pod.sh` launched branch `codex/v4167-poc`, exact pushed commit `4e896c00b63f9b0523bb95ebe8357c37cc824740`, config `configs/train_config_v4174_400M_c4_20B_v4_64.yaml`, and default trainer `scripts/train_jax.py` with `--from-scratch`. The existing authorized TPU was `spatial-se-400m` in `dawn-486218/us-central2-b`; no TPU allocation or topology change occurred. The new run folder is `gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4174_400M_c4_20B_v4_64/run_vspatial-r1-v4.1.7.4_20260727_002935_3201`, and its saved selected full-config SHA-256 is `6d66cf63cc86b295a2a9debc12a56db0c7bf35bb549b066bb0ba95f687c181bf`.
- Startup contract: the launcher cleared and reused `~/train.log`, replaced the completed benchmark holders, and started the production task in `tmux train` on all eight workers. The trainer explicitly reported `Starting from scratch (--from-scratch)`, `global_step=0` consistent across eight hosts, mesh `16x2` on 32 devices, 393,803,872 parameters, real C4 input, and Q/K/V/RST chunks `1/1/2/8`.
- Fresh tau calibration: the live config reported target fractions QK/V/RST `0.080/0.080/0.050`. Estimated active pool/local fractions were Q/K `0.080000`, V `0.080000` or `0.080001` by row, and RST `0.050000`; Q and K split summaries were both `0.080000`. This verifies that the fresh production parameters use the new initializer fingerprint rather than restored tau values.
- First optimizer step: step 1 completed after initial compilation with finite `loss=10.7252`, `ce=10.7252`, gradient norm `3.57`, and learning rate `3.01e-05`. Observed active fractions were Q `7.98%`, K `8.01%`, combined QK `7.99%`, V `8.00%`, and RST `4.99%`. The first-step wall time `104.608` s/it is compilation warmup and is not a steady-state performance measurement.
- Early production window: step 20 remained finite at `loss=9.3893`, `ce=9.3893`, gradient norm `73.53`, and learning rate `3.18e-05`. Its latest 10-step host window was `85.385` s, or `8.530` s/it and `61,343.2` tokens/s; the preceding short window reported `8.403` s/it. These are live early-training observations, not a replacement for the canonical benchmark or a long steady acceptance window.
- All-worker audit: every worker reported branch `codex/v4167-poc`, exact commit `4e896c00b63f9b0523bb95ebe8357c37cc824740`, tracked-clean status, a live `tmux train` session, and exactly one `python3 scripts/train_jax.py --from-scratch` process. Every live `~/train.log` had zero literal `Traceback`, `RESOURCE_EXHAUSTED`, `Killed`, `NaN`, `Inf`, and `ProcessLost` matches at audit time.
- Boundary/next: the production job is intentionally left running. A step-100 steady window, validation, and the first checkpoint are `not measured` in this entry; monitor them without relaunching or changing the exact source. Live evidence is `~/train.log` on every worker, primary host worker 6, the GCS run folder above, and local launcher captures `launch_spatial-se-400m_20260727_002920_worker_{0..7}.log`.

### 2026-07-29

#### #01 — Implement the v4175 40M shared-atlas, route-separated reference

- Status: requested architecture implemented directly on POC and locally validated; source/config/notebook remain uncommitted and unpushed; TPU behavior and performance are `not measured`.
- Hypothesis/semantic contract: an operation space, rather than a Q/K/V/RST route, must own its read/write coordinate system. Q, K, V, and RST should each choose top-2 entries from the same space atlas with an independent selector, while the selected atlas entry supplies the same space-owned `D -> R` read coordinates and `R -> D` write coordinates to every route. Q/K/V select independently from the pre-attention state; RST independently reselects from the post-attention residual. Route tau maps and Q/K/V/RST operator banks remain separate.
- Source identity: branch `codex/v4167-poc`; exact clean parent and `origin/codex/v4167-poc` commit `7de5fc391608d337d9835d0cb9759487a6affd8d`; local dirty files `models/dawn_srw_v4175.py`, `scripts/train_jax.py`, `configs/train_config_v4175_40M_c4_5B.yaml`, and this notebook. At entry finalization, SHA-256 values were model `e11e946382d33679431198e42ce35fced03c830b07f7ddba69bfccae4bd1a301`, trainer `b9133a32bb7cdac75529e014fd024413273e18a03c48fc959bead9d6a9b151ef`, and config `4ff8f9e57563c4af85ba717cb4d02c5f61b2e3373561fba66b8683742ba93b90`.
- Implementation: added the fresh-only version `spatial-r1-v4.1.7.5` with no legacy alias or migration. `space_selector/route_query_proj` has shape `[4,D,R]` and `space_selector/route_space_keys` has shape `[4,M,R]`; `space_interface/space_read_proj [M,D,R]` and `space_interface/space_write_proj [M,R,D]` are shared atlas parameters. The trainer registry, constructor, quantile calibration, resume validation, optimizer/sharding classification, compact logging, and startup reporting now recognize the exact v4175 schema. Production construction requires four distinct `q/k/v/rst_space_dense` kernels and rejects the v4174 shared-routing fused attention executor.
- Initial execution boundary: the first implementation deliberately evaluates every space densely and applies each route's hard-top-2 L1-normalized routing weights at shared-atlas writeback. An output-only direct kernel was first considered but omitted 57 required DirectTau/composition metrics, so the reference production builder uses the metric-complete direct dense kernel. Its model-axis `pmax` diagnostic initially had no reverse-mode rule; applying `stop_gradient` to the local diagnostic before `pmax` fixed the actual training-gradient path. Bundle/compact execution and throughput optimization are out of scope for this first 40M reference.
- Parameter match: the Transformer reference is 40,343,616 parameters and v4174 40M is 40,343,136 (`-480`). Three additional complete selectors cost 168,960 parameters. Scaling the operator pools proportionally from v4174 to `n_q=n_k=4,432`, `n_v=27,592`, and `n_rst=55,584` removes 656 operators, or 167,936 parameters at `2R=256` parameters per operator. The resulting v4175 tree is 40,344,160, only `+544` (`+0.00135%`) versus Transformer and `+1,024` versus v4174. Because every pool must be divisible by `M=8`, the total operator-bank step is 2,048 parameters; the adjacent lower total would be 40,342,112 (`-1,504`), so `+544` is the closest valid match.
- 40M config: `configs/train_config_v4175_40M_c4_5B.yaml` uses `D=432`, `R=128`, 12 layers, six heads, `M=8`, top-2, batch 128, sequence 512, mesh `4x1`, and Q/K/V/RST chunks `1/1/2/8`. Per-space pools are `554/554/3,449/6,948`; top-2 exposes `1,108/1,108/6,898/13,896` operators. Initial target fractions `0.1805054/0.1805054/0.0869817/0.0863558` correspond to approximately `200/200/600/1,200` active Q/K/V/RST operators per token.
- Local validation: repository Python/JAX/Flax were `3.12.7/0.6.2/0.10.7`. `py_compile` and `git diff --check` passed. The 40M abstract Flax parameter tree and symbolic counter both returned exactly 40,344,160; selector shapes were `[4,432,128]` and `[4,8,128]`, while shared interface shapes were `[8,432,128]` and `[8,128,432]`. Canonical construction produced and validated all four route kernels. A reduced JIT forward/backward returned finite loss `3.4499819`, finite gradients, nonzero selector projection/key gradient norms `0.00633876/0.00335040`, all 24 route-selector metrics, and `train_metrics_collected=1`. Reduced quantile calibration returned finite, distinct Q/K/V/RST per-space taus and `[M,1]` biases; a reduced canonical optimizer step passed the internal 98-field v4175 compact schema and updated selector parameters. The unchanged v4174 40M path still produced matching actual/symbolic count 40,343,136 and its original fused attention/RST production factories.
- Decision/next: accept this dirty worktree as the locally validated semantic reference, not as a published canonical commit or a performance result. No test or debug file was created. After user review, commit and push the exact POC state; then, only after an exact existing TPU is named, launch the fresh 40M config and record deployed SHA/config/fingerprint, compile/HBM, finite early steps, and route-specific selector behavior before attempting optimization.

#### #02 — Freeze the paper checkpoint registry and first execution scope

- Status: planning/setup complete and locally validated; checkpoint contents, numeric Orbax steps, restore parity, validation, zero-shot, and mechanistic results are `not measured`.
- User decision: use the explicitly authorized existing TPU-v4-64. Follow the paper evidence order `performance retention -> causal RW operator unit -> selected scale replication`. Use v4172 400M for discovery and primary mechanistic evidence, DAWN v4172 1.3B for performance and selected replication, baseline-JAX 400M as the matched performance control, and exclude baseline-JAX 1.3B because its user-provided run is still training.
- Checkpoint identities (`user-provided`, not remotely verified in this entry): DAWN 400M `gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4172_400M_c4_40B_v4_64_ver1_den_qk0p5_v1p0_rst1p2/run_vspatial-r1-v4.1.7.2_20260715_133004_3201`; DAWN 1.3B `gs://dawn-tpu-data-c4/checkpoints/train_config_v4172_1p3B_c4_20B_v4_64_ver1_den_qk0p5_v1p0_rst1p2/run_vspatial-r1-v4.1.7.2_20260717_173556_3201`; baseline 400M `gs://dawn-tpu-data-c4/checkpoints/baseline_400M_c4_40B_v4_64/run_vbaseline-JAX_20260713_194029_3201`; in-progress baseline 1.3B `gs://dawn-tpu-data-c4/checkpoints/baseline_jax_1p3B_c4_20B_v4_64/run_vbaseline-JAX_20260729_093206_3201`.
- Local identity: branch `codex/v4167-poc`; exact clean parent/origin commit `58575cbedc764fda0d40950ec02765b54108ec6d`; intended dirty files are `AGENTS.md`, `configs/paper_checkpoint_registry.yaml`, and this notebook. `configs/paper_eval_manifest.yaml` content is identical to its parent Git blob and is not part of the change. Local diagnostic Python/JAX/Flax were `3.12.13/0.6.2/0.10.7`; registry paths, config existence, model-version labels, token budgets, and AGENTS path retention passed. Remote GCS and TPU facts are `not measured`.
- Execution gates: Phase 0 must pin one committed numeric step, checkpoint file/UUID identity, full config, parameter count, tokenizer/vocab, dataset revision, Python/JAX/jaxlib/Flax, TPU type, mesh, batch, sequence, precision, evaluation commit, deterministic repeat, and production/analysis forward parity. Any restore or parity failure stops later analysis. Then run v4172 400M C4 validation plus stock zero-shot, followed by `mechanistic_screen`; do not launch full `scientific` analysis until screen eligibility and held-out/control separation are reviewed.
- Reporting rule: append each material restore, parity, performance, zero-shot, and screen attempt here immediately, including failures and blocked attempts. Separate `user-provided`, `local validated`, `TPU measured`, and `not measured` evidence. Raw logs remain in `~/train.log` and checkpoint-side `side_analysis`; only stable identities and conclusions enter Git.
- Next: publish this registry/reporting setup on POC, verify the authorized TPU and remote canonical checkout, resolve the exact v4172 400M checkpoint step, and start the Phase 0 gate.

#### #03 — Resolve and restore the v4172 400M paper checkpoint

- Result: `TPU measured`; the registered v4172 400M run restored successfully at numeric step `76293` with 393,800,708 parameters. `mib_ioi`, `mib_mcqa`, `mib_arithmetic`, `mib_arc`, and `ravel` input contracts all returned `ready`.
- Reproducibility identity: clean `codex/v4167-poc` commit `70302ebb59d0e7e0b4c39a8583fb03e507523962`; config `configs/train_config_v4172_400M_c4_40B_v4_64_ver1_den_qk0p5_v1p0_rst1p2.yaml`; checkpoint run `run_vspatial-r1-v4.1.7.2_20260715_133004_3201`; mesh `16x2`; checkpoint identity `a7ce8afcd0242bc4e9b567c9e5066c36ca223461eaa6ae6f251e6525d1f91c17`; parameter-schema hash `d5cbfc00422932c3921211ee4f00b001eb872b4f5382b13b4249311a478c41af`.
- Evaluation identity: tokenizer revision `86b5e0934494bd15c9632b12f734a8a67f723594`, vocab hash `de054a2243ead40e8e1bcc06edf510f87bdff3f2d57fd0788f2715a7b2e0bd38`, benchmark manifest hash `f3fc05c12ac26c83e0cdf8089ef46bae7f26b517265545521605d767a8eafcb6`, protocol hash `8926b105b704fd6eff577685da8e61862c5dadf6a20ce27ebb6dda89549ea15e`.
- Evidence/decision: `gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4172_400M_c4_40B_v4_64_ver1_den_qk0p5_v1p0_rst1p2/run_vspatial-r1-v4.1.7.2_20260715_133004_3201/side_analysis/run_analysis_000000076293_contract_20260729T020722Z-5643bf32`. Accept step `76293` as the v4172 400M paper checkpoint.

#### #04 — Pass v4172 400M deterministic restore and forward parity

- Result: `TPU measured`; Phase 0 passed. Two independent restores scored the same 16,352 C4 target tokens and returned identical values: production CE `2.8010849952697754`, adapter CE `2.8010853258561417`, absolute difference `3.3058636628169324e-7` versus tolerance `5e-5`.
- Reproducibility identity: clean canonical commits `267e8638816870febfba4b9afa3cb155be5498ad` and `84ea0e34c07d9477d29e817c6899dcbc127cdde1`; target/config/checkpoint/mesh `v4172_400M` / `configs/train_config_v4172_400M_c4_40B_v4_64_ver1_den_qk0p5_v1p0_rst1p2.yaml` / step `76293` / `16x2`; identical parameter identity hash `37039db095829caae9b2812d77947cedaf084f39d2f5ac36c0bf0138386cb8ca`; both worktrees clean.
- Protocol/runtime: parity used one fixed global batch of 32 C4 sequences at maximum length 512 on one TPU v4-64 pod, eight hosts, 32 devices, and mesh `data=16, model=2`. Remote software was `/usr/bin/python3` 3.10.6, JAX/jaxlib 0.6.2, Flax 0.10.7, lm-eval 0.4.2, Transformers 4.40.2, and PEFT 0.10.0. The accompanying six-task executions used `limit=1`; they completed provenance, row-count, and request-consensus checks, but their task accuracies are parity-only and excluded from paper comparisons.
- Evidence/decision: `gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4172_400M_c4_40B_v4_64_ver1_den_qk0p5_v1p0_rst1p2/run_vspatial-r1-v4.1.7.2_20260715_133004_3201/side_analysis/run_analysis_000000076293_zero_shot_20260729T040956Z-e9c5305f` and `gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4172_400M_c4_40B_v4_64_ver1_den_qk0p5_v1p0_rst1p2/run_vspatial-r1-v4.1.7.2_20260715_133004_3201/side_analysis/run_analysis_000000076293_zero_shot_20260729T041237Z-a5eeaf0f`. Accept deterministic restore and production/analysis forward parity.

#### #05 — Measure v4172 400M full stock zero-shot performance

- Result: `TPU measured`; the complete primary suite finished with `limit=null`, zero few-shot examples, `smoke_test_only=false`, and `comparable=true`. Under the pinned `mamba_table3_zero_shot_v1` protocol, normalized accuracy was HellaSwag `0.458374826 +/- 0.004972460`, PIQA `0.712187160 +/- 0.010563250`, ARC-Easy `0.484006734 +/- 0.010254534`, and ARC-Challenge `0.256825939 +/- 0.012766924`; raw accuracy was LAMBADA `0.352027945 +/- 0.006653938` and WinoGrande `0.513812155 +/- 0.014047123`. The six-task mean accuracy was `0.462872460`; LAMBADA perplexity was `24.077383687 +/- 0.800538623`.
- Reproducibility identity: clean `codex/v4167-poc` commit `623eaee8fc30471db809cb7093d3ed28594bff35`; target `v4172_400M`; model `spatial-r1-v4.1.7.2`; config `configs/train_config_v4172_400M_c4_40B_v4_64_ver1_den_qk0p5_v1p0_rst1p2.yaml`; concrete checkpoint `checkpoints/000000076293`; 393,800,708 parameters; 39,999,504,384 pretraining tokens; checkpoint-config hash `08733ae4fefdfcda2bb8e61e51a6e6fce40c0b0e4d84cb80d715085da645039b`; parameter identity hash `37039db095829caae9b2812d77947cedaf084f39d2f5ac36c0bf0138386cb8ca`; tracked worktree clean.
- Runtime/protocol: one TPU v4-64 pod, eight hosts, 32 devices, mesh `data=16, model=2`, global evaluation batch 32, maximum sequence length 512, length buckets `64/128/256/512`, and float32 evaluation. Remote software was `/usr/bin/python3` 3.10.6, JAX/jaxlib 0.6.2, Flax 0.10.7, Orbax 0.11.24, lm-eval 0.4.2, datasets 2.19.2, Transformers 4.40.2, tokenizers 0.19.1, and PEFT 0.10.0. Evaluation splits/rows were LAMBADA test 5,153; HellaSwag validation 10,042; PIQA validation 1,838; ARC-Easy test 2,376; ARC-Challenge test 1,172; and WinoGrande validation 1,267. Exact dataset fingerprints and task-config hashes are preserved in the run manifest.
- Integrity/result boundary: all six item statuses are `ready`; no request was left-truncated or exceeded the continuation limit. The run repeated the Phase 0 C4 forward check on 16,352 target tokens: production CE `2.8010849952697754`, adapter CE `2.8010853258561417`, absolute difference `3.3058636628169324e-7` versus tolerance `5e-5`. This is the first v4172 400M zero-shot run eligible for paper tables; the two preceding `limit=1` runs remain parity-only.
- Evidence/decision: result root `gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4172_400M_c4_40B_v4_64_ver1_den_qk0p5_v1p0_rst1p2/run_vspatial-r1-v4.1.7.2_20260715_133004_3201/side_analysis/run_analysis_000000076293_zero_shot_20260729T041547Z-6232963d`; aggregate metrics `backends/stock_zero_shot/results_summary.json`; exact environment/protocol/dataset manifest `backends/stock_zero_shot/run_manifest.json`; task artifacts `items/zero_shot/{lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande}.json`; canonical text summary `summary.log`. Accept these six scores as the v4172 400M stock zero-shot paper result and proceed to matched C4 validation.

#### #06 — Measure v4172 400M packed C4 validation performance

- Result: `TPU measured`; the complete locked `packed_c4_validation_v1` protocol finished with `comparable=true`. Across 610/610 global batches and 9,974,720 valid target tokens, cross-entropy was `2.94021285810529`, perplexity was `18.919873132016434`, and next-token accuracy was `0.4249118772256264` (`42.49118772256264%`).
- Reproducibility identity: clean `codex/v4167-poc` commit `37fa60822af9ae13a8cd2fddb361b563cbdc3304`; target/model `v4172_400M` / `spatial-r1-v4.1.7.2`; config `configs/train_config_v4172_400M_c4_40B_v4_64_ver1_den_qk0p5_v1p0_rst1p2.yaml`; concrete checkpoint `checkpoints/000000076293`; 393,800,708 parameters; 39,999,504,384 pretraining tokens; checkpoint-config hash `08733ae4fefdfcda2bb8e61e51a6e6fce40c0b0e4d84cb80d715085da645039b`; parameter identity hash `37039db095829caae9b2812d77947cedaf084f39d2f5ac36c0bf0138386cb8ca`; tracked worktree clean.
- Runtime/protocol: one TPU v4-64 pod, eight hosts, 32 devices, mesh `data=16, model=2`, global evaluation batch 32, sequence length 512, float32 evaluation, token reduction chunk 32,768, and no optimizer or checkpoint write. The `production_diagnostics` forward was used to report both loss and token accuracy; Phase 0 had already established its loss parity with the production-minimal forward. Remote software was `/usr/bin/python3` 3.10.6, JAX/jaxlib 0.6.2, Flax 0.10.7, Orbax 0.11.24, NumPy 2.2.6, fsspec 2024.3.1, and gcsfs 2024.3.1.
- Validation data: `gs://dawn-tpu-data-c4/c4_val.bin` with metadata `gs://dawn-tpu-data-c4/c4_val_meta.json`, tokenizer `bert-base-uncased`, vocabulary 30,522, and metadata hash `4c2fc359ddd7542052442e5deab2eecd37f5cdf7f3dca19a0943ea47eaf4aaf7`. The 10M-token capped view contained 19,531 packed sequences; 19,520 complete-batch sequences were evaluated and 11 incomplete-batch sequences were excluded. The underlying validation artifact contains 50,000,384 tokens and records 2,210 discarded tail tokens during packing.
- Evidence/decision: result root `gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4172_400M_c4_40B_v4_64_ver1_den_qk0p5_v1p0_rst1p2/run_vspatial-r1-v4.1.7.2_20260715_133004_3201/side_analysis/run_analysis_000000076293_c4_validation_20260729T044410Z-37fa6082`; table cells `validation_summary.json`; exact source/software/data/protocol manifest `run_manifest.json`; canonical text summary `summary.log`; result hash `e32a86cec7f1901b4688775d77aacf92e9649344e7cd170a0f45c8c5b9cef893`. Accept this as the v4172 400M C4 paper row and proceed to the 400M `mechanistic_screen`; matched dense performance claims remain gated on the dense 400M run.

#### #07 — Screen v4172 400M tasks for mechanistic eligibility

- Result: `TPU measured`; the complete preregistered `mechanistic_screen` finished with all five immutable input contracts and all five behavioral-eligibility items `ready`, no interpretation blocker, and zero artifact warnings. IOI, MCQA, ARC, and RAVEL passed the `minimum_known_correct=32` gate independently in discovery, validation, and test. Arithmetic returned only `12/11/8` independent paired-correct units and is excluded from causal analysis at this checkpoint.
- Table values: IOI independent eligible units were `119/123/114`, with base/source/pair accuracy `0.984/0.945/0.930`, `0.977/0.984/0.961`, and `0.992/0.898/0.891` over discovery/validation/test. MCQA units were `80/34/32`, ARC units were `50/51/57`, and RAVEL independent units were `142/154/146`; RAVEL paired-correct row counts before independent-unit collapsing were `296/294/291`. The full manuscript-facing table, including every phase accuracy, is in the Paper Writing Ledger.
- Reproducibility identity: clean `codex/v4167-poc` commit `469c2d12caf1dabf578e16d0bd1ab9cc7bf70ea4`; target/model `v4172_400M` / `spatial-r1-v4.1.7.2`; config `configs/train_config_v4172_400M_c4_40B_v4_64_ver1_den_qk0p5_v1p0_rst1p2.yaml`; concrete checkpoint `checkpoints/000000076293`; 393,800,708 parameters; checkpoint-config hash `08733ae4fefdfcda2bb8e61e51a6e6fce40c0b0e4d84cb80d715085da645039b`; checkpoint identity `a7ce8afcd0242bc4e9b567c9e5066c36ca223461eaa6ae6f251e6525d1f91c17`; parameter-schema hash `d5cbfc00422932c3921211ee4f00b001eb872b4f5382b13b4249311a478c41af`; model-config hash `053a0b3d7cb8bdfffa2aa24441012b809dbd5f450728af6df0269645b266f52f`; tracked worktree clean.
- Runtime/protocol: one TPU v4-64 pod, eight hosts, 32 devices, mesh `data=16, model=2`, seed 4172, non-RAVEL phase cap 128, RAVEL phase cap 512, and no optimizer or parameter write. Remote software was `/usr/bin/python3` 3.10.6, JAX/jaxlib 0.6.2, Flax 0.10.7, Optax 0.2.8, and NumPy 2.2.6. The immutable benchmark build was `f056bb4f133cde4fc0ae02f9`, manifest hash `f3fc05c12ac26c83e0cdf8089ef46bae7f26b517265545521605d767a8eafcb6`, tokenizer revision `86b5e0934494bd15c9632b12f734a8a67f723594`, tokenizer vocab hash `de054a2243ead40e8e1bcc06edf510f87bdff3f2d57fd0788f2715a7b2e0bd38`, and protocol hash `e55551954702df3ae628798f5a208681e1eed68dbbda0d967967af6865107adc`.
- Evidence/decision: result root `gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4172_400M_c4_40B_v4_64_ver1_den_qk0p5_v1p0_rst1p2/run_vspatial-r1-v4.1.7.2_20260715_133004_3201/side_analysis/run_analysis_000000076293_mechanistic_screen_20260729T051149Z-7d3be0d9`; run identity/status `run_manifest.json`; paper summary `summary.json` and `summary.log`; aggregate benchmark values are transcribed in the Paper Writing Ledger. The former per-example behavioral JSON vectors were deliberately purged under the aggregate-only paper-storage policy; benchmark/protocol/code/checkpoint identity and the manuscript-facing aggregate result remain retained. Promote IOI, ARC, and RAVEL to task-specific causal analysis, retain MCQA as reserve/appendix evidence, and make no localization or causal claim from this screen alone.

#### #08 — Purge raw-vector artifacts and make paper storage aggregate-only

- Hypothesis/change: raw per-example behavioral vectors and native-program schedule/effect arrays are not required for manuscript claims once exact run identity, immutable inputs, aggregate metrics, and deterministic content hashes are retained. Source commit `a7d3175fdb8ba9d5aad492848a9e09c9ed9f9257` removes the known behavioral vectors from persisted item JSON while retaining scalar results, per-field aggregate summaries, and one exact payload hash; native-program schedules and effect vectors are now hashed in memory and represented by aggregate metadata without writing per-example NPZ artifacts. JSON remains governed by field semantics rather than a file-size deletion rule; no new hard size cutoff was added.
- Storage result: `storage audited`; under the v4172 400M `side_analysis` root, seven `behavioral_eligibility.json` files containing raw per-example vectors and twenty old native-program schedule NPZ files were permanently removed. The audited footprint changed from 110 objects / 197,447,261 bytes to 83 objects / 765,425 bytes, reclaiming 196,681,836 bytes (`187.57 MiB`). Compact run manifests, summaries, manuscript metrics, protocol/checkpoint identity, aggregate native-program JSON, and logs were retained.
- Validation identity/result: `local validated` on `codex/v4167-poc` source commit `a7d3175fdb8ba9d5aad492848a9e09c9ed9f9257`; the only tracked worktree difference was this reporting-only `EXPERIMENT.md` edit. `C:\Users\USER\Desktop\DAWN-SRW\.venv\Scripts\python.exe` resolved Python with JAX `0.6.2`, Flax `0.10.7`, and NumPy `2.2.6`; module compilation passed, synthetic semantic compaction preserved scalar paper metrics while removing all ten known raw-vector fields and retaining a 64-hex content hash, effect-vector persistence was disabled, and three focused native-program/protocol/runner contract tests passed.
- Lesson/decision/next: use aggregate statistics and hashes as durable paper evidence, and inspect completed artifacts by field meaning after each run; do not delete a normal structured JSON merely because it is large. Re-audit the next causal-analysis result root after completion and purge only confirmed raw per-example vector artifacts while preserving valid detailed scientific results.
- Evidence: audited root `gs://dawn-tpu-data-c4/checkpoints/dawn_srw_v4172_400M_c4_40B_v4_64_ver1_den_qk0p5_v1p0_rst1p2/run_vspatial-r1-v4.1.7.2_20260715_133004_3201/side_analysis`; source `analysis/operator_interpretability/runner.py`; manuscript storage policy and retained result boundary in the Paper Writing Ledger above.

## Backfill Boundary

- The initial notebook covered evidence-backed work from 2026-07-21 through the step-1500 resume attempt.
- This update backfills compact-bundle through exact-top-2 backward, bucket selection, detailed backward profiling, and the first rejected active-edge lowering.
- Intermediate dirty variants whose exact diffs or raw logs were overwritten before this policy are explicitly marked as such instead of being presented as fully reproducible runs.
