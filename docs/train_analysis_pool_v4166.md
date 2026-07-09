# DAWN-SRW v4166 Train Analysis Pool

This document describes the reusable item pool used by
`scripts/analyze_dawn_srw_v4166.py --train-analysis`.

The full analysis pipeline still uses stage names such as `eval`, `prune`,
`geometry`, `usage`, `trace`, `ablation`, and `report`. The item pool below is
only for the lightweight checkpoint-state `train_analysis` mode.

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
  Items: `target_ratio`, `layer_selectivity`, `num_health`, `decision_reason`

`prompt_debug`
: v4166 prompt-side selector diagnosis.
  Items: `target_ratio`, `layer_selectivity`, `prompt_trace`,
  `prompt_decision`, `decision_reason`

`sample`
: Short qualitative generation check.
  Items: `target_ratio`, `generation_samples`, `decision_reason`

`v4166_1b`
: Default item preset selected by launcher `--preset v4166-1B`.
  Items: `target_ratio`, `layer_selectivity`, `target_quantile_gap`,
  `calibration_state`, `qk_split`, `concentration_max`, `prune_breakdown`,
  `execution_profile`, `prompt_trace`,
  `prompt_decision`, `generation_samples`, `decision_reason`

`deep`
: Broad checkpoint-state, prompt-route, and sample-generation check.
  Items: `target_ratio`, `layer_selectivity`, `target_quantile_gap`,
  `calibration_state`, `qk_split`, `concentration_max`, `prune_breakdown`,
  `execution_profile`, `prompt_trace`,
  `prompt_decision`, `generation_samples`, `decision_reason`

`full`
: Every canonical item in the pool.

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

### prompt_trace

Measures v4166 prompt-token routing without generating text. It reuses the
compact top-k trace path and reports pool-size-normalized q/k/v/rst active
fractions, gate mass, top1 concentration, top operator ids, and attention/RST
output norms.

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
RST-vs-attention norm ratio. This is the lightweight v4166 replacement for
older broad decision-probe scripts.

Summary example:

```text
PROMPT_DECISION:
  prompt_id       status qk_frac v_frac rst_frac rst_top1 rst/attn reason
  text-000000     ok     ...
```

### generation_samples

Uses the v4166 `prefill` and `decode_step` KV-cache inference API to generate a
few short continuations. Defaults are intentionally small and deterministic:
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
