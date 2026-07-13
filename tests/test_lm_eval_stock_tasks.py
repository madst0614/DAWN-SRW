import importlib.metadata
import math

import pytest

lm_eval = pytest.importorskip(
    "lm_eval", reason="requires pinned lm-eval==0.4.2 and benchmark datasets")


def test_pinned_stock_task_loading_and_metric_keys():
    assert importlib.metadata.version("lm_eval") == "0.4.2"
    from lm_eval.tasks import TaskManager, get_task_dict

    tasks = (
        "lambada_openai", "hellaswag", "piqa", "arc_easy",
        "arc_challenge", "winogrande")
    task_dict = get_task_dict(list(tasks), TaskManager(verbosity="ERROR"))
    assert set(task_dict) == set(tasks)
    expected = {
        "lambada_openai": {"perplexity", "acc"},
        "hellaswag": {"acc", "acc_norm"},
        "piqa": {"acc", "acc_norm"},
        "arc_easy": {"acc", "acc_norm"},
        "arc_challenge": {"acc", "acc_norm"},
        "winogrande": {"acc"},
    }
    for name, value in task_dict.items():
        task = value[1] if isinstance(value, tuple) else value
        assert set(task._metric_fn_list) >= expected[name]
        assert task.dump_config()["task"] == name


def test_stock_synthetic_choice_aggregation_and_stderr():
    from lm_eval.api.metrics import stderr_for_metric
    from lm_eval.api.task import MultipleChoiceTask

    class SyntheticChoiceTask(MultipleChoiceTask):
        def download(self, *args, **kwargs):
            del args, kwargs

        def has_training_docs(self):
            return False

        def has_validation_docs(self):
            return True

        def has_test_docs(self):
            return False

        def validation_docs(self):
            return []

        def doc_to_text(self, doc):
            return doc["prompt"]

    task = SyntheticChoiceTask()
    first = task.process_results(
        {"prompt": "p", "choices": ["a", "bbbb"], "gold": 1},
        [(-1.0, False), (-2.0, False)],
    )
    second = task.process_results(
        {"prompt": "q", "choices": ["aa", "bbbb"], "gold": 0},
        [(-0.1, True), (-0.9, False)],
    )
    assert first == {"acc": 0.0, "acc_norm": 1.0}
    assert second == {"acc": 1.0, "acc_norm": 1.0}
    aggregation = task.aggregation()
    acc_values = [first["acc"], second["acc"]]
    assert aggregation["acc"](acc_values) == 0.5
    assert aggregation["acc_norm"](
        [first["acc_norm"], second["acc_norm"]]) == 1.0
    stderr = stderr_for_metric(
        metric=aggregation["acc"], bootstrap_iters=1000)(acc_values)
    assert stderr == pytest.approx(
        math.sqrt(sum((value - 0.5) ** 2 for value in acc_values))
        / math.sqrt(len(acc_values) * (len(acc_values) - 1)))
