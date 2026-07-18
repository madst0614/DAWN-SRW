# DAWN stock zero-shot evaluation

This path evaluates a frozen DAWN or dense decoder checkpoint with the stock
task definitions and aggregations from `lm-eval==0.4.2`.

Protocol: `mamba_table3_zero_shot_v1`, `num_fewshot=0`.

Tasks: `lambada_openai`, `hellaswag`, `piqa`, `arc_easy`,
`arc_challenge`, and `winogrande`.

The evaluator restores only the Orbax `params` subtree, preserves the source
mesh, disables dropout, and never constructs an optimizer. A run/checkpoints
directory is resolved by host 0 exactly once and broadcast as a committed
numeric step before metadata, params, tasks, or output paths are initialized.

The evaluator loads the tokenizer id already recorded by the pretokenized C4
metadata. If that metadata also records a revision it is used automatically;
otherwise the resolved Hugging Face revision and vocabulary hash are captured
in the run manifest. It never adds CLS, SEP, BOS, EOS, or padding tokens. If
the source tokenizer has no causal EOT/BOS, the six primary tasks remain valid
because their contexts are non-empty, while rolling/empty-context calls fail
until an explicitly verified `--eot-token-id` is supplied.

Smoke run:

```bash
bash scripts/launch_zero_shot_eval_tpu_pod.sh \
  --tpu <TPU_NAME> \
  --branch <BRANCH> \
  --init-from gs://.../checkpoints/<CONCRETE_STEP> \
  --limit 32
```

Full comparable run:

```bash
bash scripts/launch_zero_shot_eval_tpu_pod.sh \
  --tpu <TPU_NAME> \
  --branch <BRANCH> \
  --init-from gs://.../checkpoints/<CONCRETE_STEP>
```

Every output directory contains `results_harness_raw.json`,
`results_summary.json`, `results_summary.csv`, `samples.jsonl`,
`run_manifest.json`, and `eval.log`. Runs with `--limit` are marked
`comparable=false` and `smoke_test_only=true`.

Local/VM verification after installing the pinned requirements:

```bash
python3 -m pip install -r requirements_zero_shot_eval.txt
python3 -m pytest -q \
  tests/test_zero_shot_protocol.py \
  tests/test_lm_eval_dawn_adapter.py \
  tests/test_lm_eval_stock_tasks.py
python3 -m py_compile \
  scripts/zero_shot_eval_jax.py \
  dawn/eval/*.py
bash -n scripts/launch_zero_shot_eval_tpu_pod.sh
```
