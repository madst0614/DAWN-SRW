# DAWN-SRW Train Analysis Pool

This document describes the reusable item pool used by
`scripts/analyze_dawn_srw_v4166.py --train-analysis`.

The analyzer entrypoint retains its historical script name for compatibility,
but the registry is shared by v4166, v4171, and v4172. The canonical source of
truth is `TRAIN_ANALYSIS_ITEM_DEFS` in
`analysis/dawn_train_analysis_items.py`. It currently contains **35 canonical
items**; the inventory below groups that complete set by runtime section.

The full analysis pipeline still uses stage names such as `eval`, `prune`,
`geometry`, `usage`, `trace`, `ablation`, and `report`. The item pool below is
only for the lightweight checkpoint-state `train_analysis` mode.

## Canonical Item Inventory and Version Support

Version labels map to these exact checkpoint identities:

| Label | `model.model_version` |
| --- | --- |
| v4166 | `spatial-r1-v4.1.6.6` |
| v4171 | `spatial-r1-v4.1.7.1` |
| v4172 | `spatial-r1-v4.1.7.2` |

"Supported" means that the item has a valid model/runtime path for that
version. An item that is merely selectable but returns `not_applicable` is not
listed as supported. v417x-only items are not silently aliased to a v4166
implementation.

### Checkpoint, prompt, and generation items (14)

| Item | Supported versions |
| --- | --- |
| `target_ratio` | v4166, v4171, v4172 |
| `layer_selectivity` | v4166, v4171, v4172 |
| `target_quantile_gap` | v4166, v4171, v4172 |
| `calibration_state` | v4166, v4171, v4172 |
| `qk_split` | v4166, v4171, v4172 |
| `concentration_max` | v4166, v4171, v4172 |
| `prune_breakdown` | v4166, v4171, v4172 |
| `execution_profile` | v4166, v4171, v4172 |
| `num_health` | v4166, v4171, v4172 |
| `composition_health` | v4171, v4172 |
| `prompt_trace` | v4166, v4171, v4172 |
| `prompt_decision` | v4166, v4171, v4172 |
| `generation_samples` | v4166, v4171, v4172 |
| `decision_reason` | v4166, v4171, v4172 |

`composition_health` is intentionally `not_applicable` on v4166 because that
checkpoint family does not emit v417x composition-denominator telemetry.

### Dataset-backed operator items (11)

| Item | Supported versions |
| --- | --- |
| `operator_dataset_manifest` | v4166, v4171, v4172 |
| `operator_behavior_eval` | v4171, v4172 |
| `ravel_operator_disentanglement` | v4171, v4172 |
| `ioi_operator_circuit` | v4171, v4172 |
| `blimp_operator_grammar` | v4171, v4172 |
| `lama_counterfact_factual_recall` | v4171, v4172 |
| `synthetic_binding_sanity` | v4171, v4172 |
| `operator_function_reuse` | v4171, v4172 |
| `operator_route_specificity` | v4171, v4172 |
| `operator_causal_specificity` | v4171, v4172 |
| `operator_analysis_summary` | v4171, v4172 |

The manifest item is model-independent. The other ten items require the
v417x production tracing and operator-suppression hooks.

### Transition and causal items (10)

| Item | Supported versions |
| --- | --- |
| `global_router_audit` | v4171, v4172 |
| `trajectory_trace` | v4171, v4172 |
| `context_divergence` | v4171, v4172 |
| `state_transition_decoupling` | v4171, v4172 |
| `causal_intervention` | v4171, v4172 |
| `causal_rerouting_trace` | v4171, v4172 |
| `causal_recovery_trace` | v4171, v4172 |
| `operator_functional_graph` | v4171, v4172 |
| `group_causal_intervention` | v4171, v4172 |
| `causal_ranking_calibration` | v4171, v4172 |

These ten items share the v417x transition implementation. An incompatible
checkpoint produces a failed item status instead of a legacy fallback.

Total: **14 + 11 + 10 = 35 canonical items**. v4171 and v4172 support all 35;
v4166 supports 14 items (13 common items plus
`operator_dataset_manifest`).

## Log Shape

Progress logs are emitted in the same order as the selected item list:

```text
TRAIN_ANALYSIS ACTIVE START ...
TRAIN_ANALYSIS ACTIVE batch=...
TRAIN_ANALYSIS PRUNE START ...
TRAIN_ANALYSIS PRUNE SUMMARY ...
TRAIN_ANALYSIS ITEMS START count=N
TRAIN_ANALYSIS ITEM 01/N id=target_ratio title='Target ratio' status=ready
TRAIN_ANALYSIS ITEM 02/N id=layer_selectivity title='Layer selectivity' status=ready
TRAIN_ANALYSIS ITEMS DONE
```

The final text summary uses the same item ids:

```text
Analysis item summaries:

ITEM target_ratio: Target ratio
  summary: TARGET_RATIO rows identify which pool is under/over target.
  ...

ITEM layer_selectivity: Layer selectivity
  summary: LAYER_SELECTIVITY shows whether a pool is globally closed or layer-local, and why.
  ...
```

Saved artifacts:

`train_analysis_latest.txt`
: overwritten with the newest human-readable summary.

`train_analysis_history.txt`
: append-only human-readable summaries, one block per analysis run.

`train_analysis_latest.json`
: overwritten with the newest full structured summary.

`train_analysis.jsonl`
: append-only scalar trend rows used by `Recent trend`.

## Commands

List the available presets, aliases, and items:

```bash
python3 -u scripts/analyze_dawn_srw_v4166.py --list-train-analysis-items
```

Prepare the immutable operator-analysis datasets on a Google Cloud VM:

```bash
python3 -u scripts/prepare_operator_analysis_datasets.py --publish-latest
```

Default dataset root:

```text
gs://dawn-tpu-data-c4/dataset/v4171_operator_analysis_v2
```

Run with a preset:

```bash
python3 -u scripts/analyze_dawn_srw_v4166.py \
  --train-analysis \
  --train-analysis-preset qk_closed
```

Run with only the items you need:

```bash
python3 -u scripts/analyze_dawn_srw_v4166.py \
  --train-analysis \
  --train-analysis-items target_ratio,layer_selectivity,prune_breakdown
```

## Presets

`minimal`
: `target_ratio`, `prune_breakdown`, `decision_reason`

`qk_closed`
: Default diagnosis preset for checking whether QK is closed and why.
  Items: `target_ratio`, `layer_selectivity`, `target_quantile_gap`,
  `calibration_state`, `qk_split`, `concentration_max`, `prune_breakdown`,
  `execution_profile`, `decision_reason`

`compute`
: Compute-focused preset for checking where sparse execution is saved or lost.
  Items: `target_ratio`, `prune_breakdown`, `execution_profile`,
  `decision_reason`

`health`
: Selector and numerical-health preset.
  Items: `target_ratio`, `layer_selectivity`, `num_health`,
  `composition_health`, `decision_reason`

`prompt_debug`
: Prompt-side selector diagnosis.
  Items: `target_ratio`, `layer_selectivity`, `prompt_trace`,
  `prompt_decision`, `decision_reason`

`sample`
: Short qualitative generation check.
  Items: `target_ratio`, `generation_samples`, `decision_reason`

`operator_datasets`
: Path-contract check for the mirrored public/operator datasets.
  Items: `operator_dataset_manifest`

`operator_analysis`
: Complete dataset-backed operator analysis for v4171 and v4172.
  Items: `operator_dataset_manifest`, `operator_behavior_eval`,
  `ravel_operator_disentanglement`, `ioi_operator_circuit`,
  `blimp_operator_grammar`, `lama_counterfact_factual_recall`,
  `synthetic_binding_sanity`, `operator_function_reuse`,
  `operator_route_specificity`, `operator_causal_specificity`,
  `operator_analysis_summary`

`v4171`, `v4172`
: Shared v417x checkpoint/prompt base preset.
  Items: all 14 checkpoint, prompt, and generation items listed above.

`v4171_self_organization`, `v4172_self_organization`
: Core v417x transition preset.
  Items: `global_router_audit`, `trajectory_trace`, `context_divergence`,
  `state_transition_decoupling`, `causal_intervention`

`v4171_operator_family`, `v4172_operator_family`
: Complete v417x transition and causal preset.
  Items: all 10 v417x transition and causal items listed above.

`v4171_self_organization_extended`, `v4172_self_organization_extended`
: Compatibility names for the corresponding `*_operator_family` preset.

`v4171_operator_monitor`, `v4172_operator_monitor`
: Core five self-organization items plus all 11 dataset-backed operator items
  (16 items total).

`v4171_complete`, `v4172_complete`
: Every canonical item in the pool (35 items). Use `v4172_complete` when a
  v4172 run must explicitly request full item coverage.

`v4166_1b`
: Default item preset selected by launcher `--preset v4166-1B`.
  Items: `target_ratio`, `layer_selectivity`, `target_quantile_gap`,
  `calibration_state`, `qk_split`, `concentration_max`, `prune_breakdown`,
  `execution_profile`, `prompt_trace`,
  `prompt_decision`, `generation_samples`, `decision_reason`

`deep`
: Compatibility preset with the same 12 items as `v4166_1b`.

`full`
: Every canonical item in the pool (35 items), identical to
  `v4171_complete` and `v4172_complete`. Use only with v4171 or v4172;
  v4166 does not support the v417x transition and production-suppression items.

## Aliases

Old item ids still work, but they are canonicalized before execution and output:

`per_layer_active`
: alias for `layer_selectivity`

`select_dist`
: alias for `layer_selectivity`

`concentration`
: alias for `execution_profile`

`exec_counts`
: alias for `execution_profile`

`trace`
: alias for `prompt_trace`

`prompt`
: alias for `prompt_trace`

`decision_probe`
: alias for `prompt_decision`

`generation`
: alias for `generation_samples`

`samples`
: alias for `generation_samples`

`dataset_manifest`
: alias for `operator_dataset_manifest`

`operator_manifest`
: alias for `operator_dataset_manifest`

`ravel`
: alias for `ravel_operator_disentanglement`

`ioi`
: alias for `ioi_operator_circuit`

`blimp`
: alias for `blimp_operator_grammar`

`lama`
: alias for `lama_counterfact_factual_recall`

`counterfact`
: alias for `lama_counterfact_factual_recall`

`synthetic`
: alias for `synthetic_binding_sanity`

## Items

### target_ratio

Measures target, active_tau, admission, effective, and effective/target by
pool. Use this first when you want to know whether QK/V/RST is under or over
the configured selector target.

Summary example:

```text
TARGET_RATIO:
  pool  target   active_tau admission effective eff/target status
  qk    0.080    0.026      0.033     0.025     0.31       TOO_CLOSED
```

### layer_selectivity

Merged item replacing `per_layer_active` and `select_dist`.

Measures per-layer active/admission/effective/top1, plus layer-level
score/tau/margin distributions. Use this when you need to distinguish global
pool closure from a few bad layers or threshold/score separation issues.

Summary example:

```text
PER_LAYER_SUMMARY:
  pool  eff_min eff_p10 eff_mean eff_p90 eff_max dead closed
  qk    0.004   0.012   0.025    0.041   0.058   2    7

PER_LAYER_ACTIVE:
  lyr qk_act qk_adm qk_eff qk_top1 | v_act v_adm v_eff v_top1 | rst_act rst_adm rst_eff rst_top1
  00  ...

SELECT_DIST:
  pool score_p50 score_p90 score_max tau_p50 tau_p90 margin_p50 margin_p90 pos_margin_frac
  qk   ...
```

### target_quantile_gap

Measures the approximate score quantile needed to hit the configured active
target and the current candidate target, then compares those score levels with
the current tau. Use this when `SELECT_DIST` says score is below tau and you
want the size of the boundary mismatch in one row.

The current implementation uses layer-level score summaries, so the source is
reported as `layer_score_quantile_approx`. If exact operator histograms are
added later, the item id can stay the same.

Summary example:

```text
TARGET_QUANTILE_GAP:
  source=layer_score_quantile_approx
  pool target candidate q@target q@candidate tau gap_target gap_candidate
  qk   0.080  0.160     -0.0450  -0.0650     0.0715 +0.1165   +0.1365
```

### calibration_state

Measures target, candidate, observed admission, admission error, tau, and
whether dynamic tau updates are visible from the checkpoint/config. Use this
when selection calibration is enabled but observed admission is not moving
toward `candidate_now`.

`tau_before`, `tau_delta`, `clamp`, and `stopgrad` remain `n/a` unless those
values are emitted by the training/checkpoint path. This is intentional: `n/a`
is safer than a misleading zero.

Summary example:

```text
CALIBRATION_STATE:
  pool target candidate observed_adm error tau_before tau_after tau_delta clamp stopgrad mode
  qk   0.080  0.160     0.030        -0.130 n/a        0.0715    n/a       n/a   n/a      enabled/static_tau_lr0
```

### qk_split

Separates Q and K sparse activity instead of reporting only merged QK. Use this
when QK is thin and prompt traces suggest Q and K are behaving differently.

Summary example:

```text
QK_SPLIT_SUMMARY:
  side active_tau admission effective active_ops effective_ops eff_min eff_mean eff_max
  q    ...
  k    ...
  q/k_effective_balance=...

QK_SPLIT_LAYERS:
  lyr q_act q_adm q_eff q_ops | k_act k_adm k_eff k_ops | qk_eff
  00  ...
```

### concentration_max

Finds the layer with the largest per-layer top1 concentration for each pool and
prints the local active/effective context. Use this when `execution_profile`
shows a high `top1_max` and you need to know which layer to inspect.

Summary example:

```text
CONCENTRATION_MAX:
  pool layer layer_top1_max global_top1_max layer_top1_mean active effective active_ops effective_ops operator_id
  rst  04    0.204          0.206           ...
```

### prune_breakdown

Measures eps-wise pruned eval loss delta, total compute, pool-level compute,
pool-level effective estimates, raw gate-mass diagnostic, and no-active fraction.
Use this when you want to see whether pruned eval is actually saving compute
and which pool is responsible.

Summary example:

```text
PRUNE_BREAKDOWN:
  eps   val_loss delta_ce total qk_cmp v_cmp rst_cmp | qk_eff v_eff rst_eff saved gate_mass_raw no_active
  0     ...
  1e-2  ...
  1e-1  ...
```

### execution_profile

Merged item replacing `concentration` and `exec_counts`.

Measures pool size, active/effective fractions, active/effective operator
counts, top1 concentration, max top1 concentration, and effective operator
ratio. Use this when sparse percentages need to be translated into actual
operator counts and collapse indicators.

Summary example:

```text
EXECUTION_PROFILE:
  pool pool_size active_frac active_ops effective_frac effective_ops top1 top1_max eff_ratio
  qk   7048      0.026       183.2      0.025          176.2         ...
```

### num_health

Measures residual norms, q/k/v norms, attention logit scale, softmax entropy,
output norm, and NaN status. Use this when selector behavior looks suspicious
and you need to rule out scale drift or numerical instability.

Summary example:

```text
NUM_HEALTH:
  metric                         mean     p90      max
  residual_norm                  ...
  attn_logit_max                 ...
  no_nan                         True
```

### composition_health

Measures v417x admission mass, composition-denominator minimum/maximum,
floor-hit fraction, and configured denominator power for QK, V, and RST.
Use this to verify that generalized operator composition remains finite and is
not pinned to its numerical floor. On model versions without composition
telemetry, the item reports the unavailable fields explicitly instead of
silently substituting zeros.

### prompt_trace

Measures version-matched prompt-token routing without generating text. It
reuses the compact top-k trace path and reports pool-size-normalized q/k/v/rst
active fractions, gate mass, top1 concentration, top operator ids, and
attention/RST output norms.

Summary example:

```text
PROMPT_TRACE:
  boundary_power=3.000
  prompt_id       len q_frac k_frac v_frac rst_frac q_top1 k_top1 v_top1 rst_top1 rst/attn
  text-000000     5   ...
```

### prompt_decision

Derived from `prompt_trace`. It flags prompt-local route imbalance using qk/v/rst
pool-size-normalized activity balance, selector concentration, and
RST-vs-attention norm ratio. This is the lightweight replacement for older
broad decision-probe scripts.

Summary example:

```text
PROMPT_DECISION:
  prompt_id       status qk_frac v_frac rst_frac rst_top1 rst/attn reason
  text-000000     ok     ...
```

### generation_samples

Uses the selected model version's `prefill` and `decode_step` KV-cache inference
API to generate a few short continuations. Defaults are intentionally small and deterministic:
three prompts, seeded top-k sampling with temperature 0.8, and 64 new tokens.
If the tokenizer is not available on the TPU worker, it falls back to validation
token prompts and prints generated token ids instead of skipping the item.
The summary prints the prompt, generated continuation, prompt+continuation, and
the first-step top token distribution so punctuation loops are visible.

Summary example:

```text
GENERATION_SAMPLES:
  decode_mode=text max_new=64 temp=0.80 top_k=50 seed=123 boundary_power=3.000
  gen-000000   new_tok=64  tok/s=...
    prompt   : ...
    generated: ...
    full     : ...
    first_top: ...
```

### operator_dataset_manifest

Reports the configured dataset root, the manifest path, and the per-dataset GCS
roots prepared by `scripts/prepare_operator_analysis_datasets.py`. The
historical `v4171` path segment is the immutable dataset-build identity; it is
not a restriction on v4172 checkpoints.

Default root:

```text
gs://dawn-tpu-data-c4/dataset/v4171_operator_analysis_v2
```

Summary example:

```text
OPERATOR_DATASET_MANIFEST:
  root     : gs://dawn-tpu-data-c4/dataset/v4171_operator_analysis_v2
  pointer  : gs://dawn-tpu-data-c4/dataset/v4171_operator_analysis_v2/LATEST.json
  manifest : .../builds/<build-id>/manifest.json
  prepare  : python3 -u scripts/prepare_operator_analysis_datasets.py --publish-latest
  id          root
  ravel       .../builds/<build-id>/ravel
```

### operator_behavior_eval

Runs the restored production checkpoint on every selected prepared behavior
row. It reports teacher-forced full-sequence and continuation margins,
accuracy, known-correct subsets, and logical-example bootstrap intervals.
Behavior competence is measured before route or causal claims are accepted.

### ravel_operator_disentanglement

Dataset-backed plan for RAVEL attribute operator disentanglement. The prepared
paths include the raw `data.tgz` plus HuggingFace parquet splits for
`city_entity` and `city_prompt`. The intended behavior metric is answer logit
margin over attribute-value distractors; the operator metric is same-attribute
overlap and cross-attribute causal drop.

### ioi_operator_circuit

Generated IOI prompts for clean/corrupt name swaps. The intended behavior
metric is correct indirect-object logit minus distractor subject logit, and the
operator metric is clean/corrupt gate delta plus QK/V/RST causal drop.

### blimp_operator_grammar

Prepared BLiMP parquet files for all 67 phenomena. The intended behavior metric
is good-sentence likelihood minus bad-sentence likelihood, grouped by
phenomenon and critical token.

### lama_counterfact_factual_recall

Prepared LAMA `data.zip` and CounterFact JSON. The intended behavior metric is
known-fact/rewrite object logit margin, with relation-specific QK/V/RST
operator ablations.

### synthetic_binding_sanity

Generated controlled binding-retrieval examples. This is a sanity-check
dataset for entity binding, attribute query, and residual write decomposition.

### operator_function_reuse

Compares same-function transition-path similarity against a length-matched
cross-function random null. It reports the reuse effect and a logical-pair
bootstrap interval rather than treating raw path overlap as sufficient.

### operator_route_specificity

Measures within-group versus between-group routing and transition overlap,
captured mass, the specificity gap, and enriched operators. Route evidence is
reported only when its captured-mass qualification passes.

### operator_causal_specificity

Measures task-margin drops for selected, matched-active, active-random,
inactive-random, and cross-function suppression strategies. It uses production
contribution subtraction and blocks the result if the zero-vector parity check
fails.

### operator_analysis_summary

Combines behavior competence, function reuse, route specificity, causal
specificity, validity checks, and limitations into the artifact-backed
cross-dataset result.

### global_router_audit

Audits restored router and pool parameter paths, hidden block-local routing
parameters, global sharing, geometry, and composition settings. It fails
loudly if a v417x checkpoint does not expose one global router and one global
operator-pool tree.

### trajectory_trace

Captures target-span residual/query trajectories, sparse operator ids,
captured mass, read responses, coefficients, Q/K/V SRW features, and actual
attention/RST residual updates. It reuses one target-only transition cache and
does not return the full gate tensor to the host.

### context_divergence

Compares controlled same-surface pairs across state, query, sparse gate, and
update similarity. It reports first divergence, maximum divergence, and late
reconvergence using captured-mass-qualified gate metrics.

### state_transition_decoupling

Tests whether distant representation states can share transition paths. It
reports state/query/gate/delta/path similarities, data-quantile quadrants,
random-null percentiles, correlations, effect size, and a bootstrap interval
without relying on a fixed arbitrary threshold.

### causal_intervention

Runs canonical target-token/layer/pool interventions for top contribution,
top gate, active/inactive random, and matched-active controls. It subtracts a
selected post-denominator production contribution and blocks on exact
zero-vector parity.

### causal_rerouting_trace

Compares same-forward baseline and intervention traces across residual, query,
sparse routing, transition, attention update, and RST update. It reports
divergence AUC, reconvergence, paired controls, and final causal effects without
transferring full gate tensors to the host.

### causal_recovery_trace

Measures immediate target-state damage, downstream recovery or amplification,
and final residual/logit effects. Distribution-relative judgments distinguish
small local effects from later compensation.

### operator_functional_graph

Builds seed-local reciprocal neighborhoods from rank-1 RW functional
similarity, address similarity, and sparse activation/contribution profiles.
Transitive-component percolation remains a separate diagnostic and is not
treated as a causal operator family.

### group_causal_intervention

Runs fixed-width family, address, coactivation, and random group suppression.
It reports dose response, recovery, and synergy against available single
effects to test whether weak single interventions reflect functional
redundancy.

### causal_ranking_calibration

Tests whether local gate and contribution rankings predict final causal
importance. It reports gate/contribution/immediate/final causal Spearman
inference, strategy win rates, and bootstrap/permutation judgments.

### decision_reason

Prints the numeric guardrails behind the final keep/watch/change decision. Use
this in almost every preset because it records why the summary made its call.

Summary example:

```text
Decision:
  - Current run needs attention: QK too closed.
  - Reason: qk QK too closed: eff/target=0.31, admission=0.033.
  - No obvious collapse: top1_max=0.024 < 0.100.
```

## Not Yet Implemented

`top5/top10 concentration`
: The model forward currently reports top1-style concentration. Exact top5 and
  top10 metrics need additional pool-gate reductions.

`prune_layer_sensitivity`
: The current prune item reports pool-level prune breakdown. Layer-level compute
  drop needs per-layer sparsity arrays from the prune eval path.

`update_ratio`
: Per-subtree `grad_norm`, `update_norm`, and `update/param` need optimizer or
  gradient state. A checkpoint-only side analysis cannot compute this exactly.
