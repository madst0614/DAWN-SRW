# Interpretability benchmark data contract

This document is the source of truth for the data consumed by
`analysis/operator_interpretability`. The contract was rebuilt from the full
official datasets and the pinned official MIB implementation. It does not read,
migrate, or alias the former `v4171_operator_analysis_v2` artifacts or the old
generated IOI/synthetic probe format.

Contract identity:

```text
schema: dawn_interpretability_source_contract
schema_version: 1
primary datasets: mib_ioi, mib_mcqa, mib_arithmetic, mib_arc, ravel
auxiliary datasets: blimp, counterfact
official MIB repository: https://github.com/aaronmueller/MIB
official MIB revision: b69dabe9899251d4a8fe90789afa4d655afc84c7
official circuit-track revision: b759df34433c9e31043ba9e02908ce0bf20e894f
```

## Pinned sources and splits

The preparation command downloads every official split before adapting rows.
It rejects a source if its resolved commit, row count, required columns, or
feature schema differs from the registry.

| Item | Source and pinned revision | Original split rows | DAWN phase mapping | Official counterfactual used |
|---|---|---:|---|---|
| `mib_ioi` | `mib-bench/ioi@e5f3468f3af4c0883be35cd3bced8c711c95d286` | train 10,000; validation 10,000; test 1,000 | discovery=train; validation=validation; test=test | `s2_io_flip_counterfactual` |
| `mib_mcqa` | `mib-bench/copycolors_mcqa@682676b0e80d3a80e847a21810157710c1f23e27`, config `4_answer_choices` | train 110; validation 50; test 50 | discovery=train; validation=validation; test=test | `symbol_counterfactual` |
| `mib_arithmetic` | `mib-bench/arithmetic_addition@d56d68b7b1d8f9c7e9a262ec1f89331ed52c7516` | train 34,440; validation 4,920; test 1,000 | discovery=train; validation=validation; test=test | `random_counterfactual` |
| `mib_arc` | `mib-bench/arc_easy@be6999b975e8348387cc9a21f3c8b6be8021e7d6` | train 2,251; validation 570; test 1,188 | discovery=train; validation=validation; test=test | `symbol_counterfactual` |
| `ravel` | `mib-bench/ravel@53dc7a4dbe84276567b895d8fb608b3f169b9276` | train 100,347; val 15,950; test 1,000 | discovery=train; validation=val; test=test | `attribute_counterfactual`, `wikipedia_counterfactual` |
| `blimp` | `alexwarstadt/blimp@3e56b06fcabca9b30822fc66435fca6b1aa40bb1` | 67 files x 1,000 rows | deterministic 70/15/15 content-hash split | official good/bad minimal pair |
| `counterfact` | `counterfact.json@sha256:d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f` | 21,919 rows | deterministic 70/15/15 content-hash split | `target_true` versus `target_new` |

RAVEL follows the official baseline's bounded source selection: the complete
split is downloaded and audited, then shuffled with seed 42 and capped at
10,000 source rows. Therefore discovery uses 10,000 of 100,347 train rows,
validation uses 10,000 of 15,950 val rows, and test uses all 1,000 rows. Both
the official and selected counts are stored in the manifest.

## Actual raw forms

The feature schemas below were read from all seven pinned datasets on
2026-07-18. Nested counterfactual values are official dataset objects, not
prompts synthesized by DAWN.

### MIB IOI

```text
template: string
metadata: {
  indirect_object: string, subject: string, object: string, place: string,
  random_a: string, random_b: string, random_c: string
}
prompt: string
choices: list[string]
answerKey: int64
<counterfactual>: {prompt: string, choices: list[string], answerKey: int64}
```

Top-level counterfactual columns are
`abc_counterfactual`, `random_names_counterfactual`,
`s1_io_flip_counterfactual`, `s2_io_flip_counterfactual`,
`random_names_s1_ioi_flip_counterfactual`,
`random_names_s2_ioi_flip_counterfactual`,
`s1_ioi_flip_s2_ioi_flip_counterfactual`, and
`random_names_s1_ioi_flip_s2_ioi_flip_counterfactual`. The circuit contract
uses only the official default `s2_io_flip_counterfactual`. Across the pinned
source, the base answer is `metadata.indirect_object` and the selected source
answer is `metadata.subject`.

### MIB CopyColors MCQA

```text
dataset, dataset_specific_id, counterfact_dataset_specific_id: string
question: string
elements, counterfact_elements: list[string]
choices: {label: list[string], text: list[string]}
answerKey: int64
prompt: string
symbol_counterfactual: {
  prompt: string,
  choices: {label: list[int64], text: list[string]},
  answerKey: int64
}
idx: int64
```

The selected config always has four choices. Base choice labels are strings;
the official symbol-counterfactual labels are integers 1 through 4. The
adapter resolves `answerKey` against the actual label list instead of assuming
one label type.

### MIB arithmetic addition

```text
idx: int64
template, prompt, label, operator: string
num_digit: int64
operand1, operand2: int64
<counterfactual>: {
  label: int64, operand1: int64, operand2: int64, prompt: string
}
```

The counterfactual columns are `random_counterfactual`,
`ones_op1_counterfactual`, `ones_op2_counterfactual`,
`tens_op1_counterfactual`, `tens_op2_counterfactual`,
`ones_carry_counterfactual`, and `tens_carry_counterfactual`. The official
circuit default is `random_counterfactual`.

`idx` is not a row-unique identifier: each arithmetic case is repeated across
six official prompt templates. The pinned split has 5,740 unique train indices,
820 unique validation indices, and 622 unique test indices. DAWN therefore
never uses raw `idx` as the artifact identity. One-digit rows are retained:
train contains 420 one-digit and 34,020 two-digit rows; validation contains 60
and 4,860; test contains 14 and 986.

### MIB ARC Easy

```text
arc_id, question, prompt, label: string
choices: {text: list[string], label: list[string]}
answerKey, idx: int64
<counterfactual>: {
  question: string,
  choices: {text: list[string], label: list[string]},
  prompt: string, label: string, answerKey: int64
}
```

Choice count is not fixed to four. The distributions are:

| Split | 3 choices | 4 choices | 5 choices |
|---|---:|---:|---:|
| train | 6 | 2,241 | 4 |
| validation | 1 | 567 | 2 |
| test | 4 | 1,182 | 2 |

All official rows are retained. The adapter resolves the answer against each
row's actual choice structure.

### MIB RAVEL

```text
template, prompt, entity, attribute: string
Continent, Country, Language: string
prompt_template_counterfactual: {
  template, prompt, entity, attribute,
  Continent, Country, Language: string
}
attribute_counterfactual: same nested form
wikipedia_counterfactual: same nested form
```

The official baseline uses `attribute_counterfactual` and
`wikipedia_counterfactual`; `prompt_template_counterfactual` is not a causal
source in this contract. Target variables are exactly `Continent`, `Country`,
and `Language`.

For every selected base row, source kind, and target variable:

```text
pair_type = cause      if base.attribute == target_variable
pair_type = isolation  otherwise

expected patched answer = source[target_variable]
                          if pair_type == cause
                          else base[base.attribute]
```

The attribute source is a labeled behavioral query and must be solved by the
checkpoint. The Wikipedia source has official `raw_output == ""`; it is kept as
an intervention source but is explicitly marked `source_behavior_required =
false`. No artificial Wikipedia answer is created.

One source row can yield at most six DAWN rows: two official source kinds times
three target variables. Missing values and indistinguishable positive/negative
contrasts are counted as attrition, never silently relabeled.

### BLiMP

The pinned repository contains 67 JSONL files with exactly 1,000 rows each.
Each raw row has:

```text
UID, field, linguistics_term, phenomenon: string
pairID: integer
sentence_good, sentence_bad: string
simple_LM_method, one_prefix_method, two_prefix_method: flags
lexically_identical: boolean
```

The source file name is recorded as `phenomenon`. BLiMP has no official
train/validation/test division, so every full canonical row ID is assigned once
to a stable 70/15/15 hash bucket. DAWN finds the shared token prefix and scores
the two official continuations. A pair that diverges at the first model token
cannot be represented by the current non-empty-prompt continuation scorer and
is excluded explicitly; no replacement prompt is generated.

BLiMP is auxiliary behavior evidence only. It has no causal source prompt, so
`source_behavior_required=false`, and no BLiMP result can select an operator
family or enter the scientific claim gate.

### CounterFact

The content-pinned published JSON object has 21,919 rows with top-level fields:

```text
case_id, pararel_idx
requested_rewrite
paraphrase_prompts, neighborhood_prompts, attribute_prompts, generation_prompts
```

The adapter reads the official `requested_rewrite.prompt`, `subject`,
`relation_id`, `target_true.str`, and `target_new.str`. It formats the subject
through the published prompt and compares the true and counterfactual target
strings. CounterFact also has no official phase split, so every canonical row
ID is assigned once to the stable 70/15/15 hash partition.

Target strings may have different token lengths. Their contrast uses mean token
log-probability, and all 21,919 pinned rows are retained. CounterFact is
auxiliary behavior evidence only and has no causal source or claim-gate role.

## Prepared row contract

Each accepted row contains all of the following:

```text
benchmark_id, example_id, phase
base_prompt, source_prompt
positive_answer, negative_answer
intervention_positive_answer, intervention_negative_answer
causal_variable, pair_type, source_behavior_required
trace_position_base, trace_position_source
input_ids_base, input_ids_source
positive_ids, negative_ids
source_positive_ids, source_negative_ids
intervention_positive_ids, intervention_negative_ids
metadata
```

`example_id` includes a SHA-256 hash of the full canonical raw row. Human source
IDs such as `idx`, `arc_id`, or `case_id` may appear only as a readable prefix.
This prevents the arithmetic template collision that occurs when `idx` alone
is used. Phase names are not added to hide content collisions: an identical
canonical row appearing in two official splits is rejected as leakage.

Tokenizer identity is part of the immutable build: repository name, requested
revision, resolved commit, vocabulary hash, vocabulary size, pad token, and
`add_special_tokens=false` are recorded. Runtime loading must reproduce the
same vocabulary hash and checkpoint logical vocabulary size.

MIB circuit candidates use their task-specific official output form. IOI,
CopyColors, and ARC require equal-length candidate tokenizations; arithmetic
requires one model token per candidate. Rows whose positive and negative
candidates tokenize identically cannot define a contrast and are excluded with
an explicit reason.

RAVEL values are naturally variable-length. They are not filtered to equal
token length. Behavioral and intervention contrasts use mean token
log-probability per candidate, recorded as
`mean_log_probability_per_token`. This is a frozen, teacher-forced causal
contrast for DAWN; it is not claimed to be identical to the official MIB
generation-and-string-checker metric.

## Physical phase and manifest contract

Every benchmark is written as three separate immutable files:

```text
builds/<build_id>/<benchmark_id>/discovery.jsonl
builds/<build_id>/<benchmark_id>/validation.jsonl
builds/<build_id>/<benchmark_id>/test.jsonl
builds/<build_id>/manifest.json
LATEST.json
```

The manifest records a path, SHA-256, and row count for each phase shard. The
loader rejects:

- an old or unknown schema;
- a missing physical phase;
- a shard containing a row from another phase;
- a shard hash or row-count mismatch;
- source revision, split mapping, or required-column drift;
- a missing source-level identity audit, duplicate source row, or split overlap;
- duplicate content identities;
- missing RAVEL cause/isolation evidence for either official source kind;
- incomplete tokenizer or intervention answer fields.

Discovery performs localization. Validation selects circuits/families. Test is
loaded as a physically separate shard and cannot be used for selection.

At runtime, RAVEL is sampled by a deterministic round-robin over all 12 strata
(3 variables x cause/isolation x 2 source kinds). Source variants sharing one
`pair_group_id` are not counted twice within cause or isolation. The
pre-registered RAVEL cap is 512 prepared rows per phase, yielding up to 256
independent cause units before behavioral filtering. Other primary benchmarks
retain the 128-row cap.

RAVEL localization rank stability is computed separately for `Continent`,
`Country`, and `Language`. Independent `pair_group_id` units are deterministically
balanced within each causal-variable x official-counterfactual-column stratum.
The persisted top-level RAVEL stability is the minimum of the three variable
scores; the old pooled-variable score is audit-only. Functional-family discovery,
causal mediation, and held-out trajectory analysis do not execute unless every
variable reaches the pre-registered 0.80 threshold. This is a runtime protocol
change and does not require rebuilding an immutable dataset artifact that
already contains at least the requested rows.

## Full preparation validation

The complete seven-dataset preparation was executed locally on 2026-07-18 with
`bert-base-uncased@86b5e0934494bd15c9632b12f734a8a67f723594`, vocabulary hash
`de054a2243ead40e8e1bcc06edf510f87bdff3f2d57fd0788f2715a7b2e0bd38`, and
`max_seq_len=512`.

```text
build_id: 940a91d6044c9cbb257e1245
mib_ioi:        21,000 accepted; 0 excluded
mib_mcqa:          210 accepted; 0 excluded
mib_arithmetic: 40,024 accepted; 336 identical-candidate exclusions
mib_arc:         4,009 accepted; 0 excluded
ravel:         109,346 accepted
blimp:          59,016 accepted; 7,977 no-shared-prefix exclusions;
                              7 no-divergent-continuation exclusions
counterfact:    21,919 accepted; 0 excluded
```

RAVEL accepted phase counts were discovery 51,939, validation 52,241, and test
5,166. Its source-form attrition was 12,009 indistinguishable base contrasts,
4,566 indistinguishable labeled-source contrasts, 25 missing target-variable
values, 6 missing source values for the base query, 45 missing labeled-source
values, and 3 tokenizer-identical candidate contrasts. All 36 required RAVEL
strata were present: 3 phases times 3 variables times cause/isolation times 2
official source kinds.

The pre-selection, pre-tokenization source identity audit covered 21,000 IOI,
210 CopyColors, 40,360 arithmetic, 4,009 ARC, 117,297 RAVEL, 67,000 BLiMP,
and 21,919 CounterFact raw rows. Every source had zero within-split duplicate
canonical rows and zero cross-split overlap.

These accepted counts are evidence for the tokenizer revision above, not a
replacement for the invariant official source row counts.

## Publishing

Run on a normal VM with network and GCS credentials:

```bash
python3 -u scripts/prepare_interpretability_benchmarks.py \
  --output-root gs://dawn-tpu-data-c4/dataset/operator_interpretability \
  --benchmarks all \
  --tokenizer bert-base-uncased \
  --tokenizer-revision main \
  --max-seq-len 512 \
  --publish-latest
```

`main` is only the requested tokenizer revision. Preparation resolves it to an
exact commit and stores that commit in the build; analysis reloads the exact
resolved commit. Publishing is immutable: an existing build path is reused only
when every content hash agrees.
