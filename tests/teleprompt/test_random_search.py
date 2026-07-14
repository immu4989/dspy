import pytest

import dspy
from dspy import Example
from dspy.predict import Predict
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.events import (
    CandidateEvaluated,
    CandidateProposed,
    CandidateSelected,
    OptimizerScore,
    RandomSearchCandidateProposed,
)
from dspy.utils.callback import ACTIVE_CALL_ID, BaseCallback
from dspy.utils.dummies import DummyLM


class SimpleModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    def forward(self, **kwargs):
        return self.predictor(**kwargs)


def simple_metric(example, prediction, trace=None):
    return example.output == prediction.output


class IdentityModule(dspy.Module):
    def forward(self, input):
        return dspy.Prediction(output=input)


def identity_metric(example, prediction, trace=None):
    return float(example.output == prediction.output)


class RandomSearchCallback(BaseCallback):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.optimizer_call_id = None
        self.records = []
        self.evaluate_inputs = []

    def on_optimizer_start(self, call_id, instance, inputs):
        if instance is self.optimizer:
            self.optimizer_call_id = call_id
            self.records.append(("optimizer_start", call_id, None))

    def on_optimizer_event(self, call_id, instance, event):
        if instance is self.optimizer:
            self.records.append(("optimizer_event", call_id, event))

    def on_optimizer_end(self, call_id, outputs, exception):
        if call_id == self.optimizer_call_id:
            self.records.append(("optimizer_end", call_id, exception))

    def on_evaluate_start(self, call_id, instance, inputs):
        self.evaluate_inputs.append(inputs)
        self.records.append(("evaluate_start", ACTIVE_CALL_ID.get(), call_id))

    def on_evaluate_end(self, call_id, outputs, exception):
        self.records.append(("evaluate_end", self.optimizer_call_id, call_id))


def make_identity_trainset():
    return [
        Example(input="correct", output="correct").with_inputs("input"),
        Example(input="incorrect", output="different").with_inputs("input"),
    ]


def test_basic_workflow():
    """Test to ensure the basic compile flow runs without errors."""
    student = SimpleModule("input -> output")
    teacher = SimpleModule("input -> output")

    lm = DummyLM(
        [
            "Initial thoughts",
            "Finish[blue]",  # Expected output for both training and validation
        ]
    )
    dspy.configure(lm=lm)

    optimizer = BootstrapFewShotWithRandomSearch(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1)
    trainset = [
        Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
        Example(input="What does the fox say?", output="Ring-ding-ding-ding-dingeringeding!").with_inputs("input"),
    ]
    optimizer.compile(student, teacher=teacher, trainset=trainset)


def test_restrict_matching_no_candidate_seed_raises_clear_error():
    """restrict that matches no candidate seed should raise ValueError, not UnboundLocalError."""
    student = SimpleModule("input -> output")
    teacher = SimpleModule("input -> output")

    lm = DummyLM(["Initial thoughts", "Finish[blue]"])
    dspy.configure(lm=lm)

    optimizer = BootstrapFewShotWithRandomSearch(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1)
    trainset = [
        Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
    ]

    with pytest.raises(ValueError, match="restrict"):
        optimizer.compile(student, teacher=teacher, trainset=trainset, restrict=[999])


def test_restrict_as_single_use_iterator_still_matches_a_valid_seed():
    """A single-use iterable `restrict` must not be exhausted by the upfront validation check."""
    student = SimpleModule("input -> output")
    teacher = SimpleModule("input -> output")

    lm = DummyLM(["Initial thoughts", "Finish[blue]"])
    dspy.configure(lm=lm)

    optimizer = BootstrapFewShotWithRandomSearch(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1)
    trainset = [
        Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
    ]

    # -3 (zero-shot) is a valid seed; a plain iterator is single-use.
    result = optimizer.compile(student, teacher=teacher, trainset=trainset, restrict=iter([-3]))
    assert result is not None


def test_random_search_emits_typed_candidate_evaluation_and_selection_events():
    optimizer = BootstrapFewShotWithRandomSearch(metric=identity_metric, num_threads=2)
    callback = RandomSearchCallback(optimizer)
    dspy.configure(callbacks=[callback])

    result = optimizer.compile(IdentityModule(), trainset=make_identity_trainset(), restrict=[-3])

    assert [record[0] for record in callback.records] == [
        "optimizer_start",
        "optimizer_event",
        "evaluate_start",
        "evaluate_end",
        "optimizer_event",
        "optimizer_event",
        "optimizer_end",
    ]
    run_id = callback.optimizer_call_id
    assert run_id is not None
    assert all(record[1] == run_id for record in callback.records)

    proposed, evaluated, selected = [
        record[2] for record in callback.records if record[0] == "optimizer_event"
    ]
    assert proposed == RandomSearchCandidateProposed(
        candidate_id="candidate:0",
        candidate_index=0,
        seed=-3,
        kind="zero_shot",
    )
    assert isinstance(proposed, CandidateProposed)
    assert evaluated == CandidateEvaluated(
        candidate_id="candidate:0",
        candidate_index=0,
        evaluation_id="candidate:0:evaluation:0",
        metric_key="eval_full",
        score=OptimizerScore(value=50.0, unit="percent", aggregation="mean"),
        example_scores=(
            OptimizerScore(value=1.0, unit="raw_metric", aggregation="identity"),
            OptimizerScore(value=0.0, unit="raw_metric", aggregation="identity"),
        ),
    )
    assert selected == CandidateSelected(
        candidate_id="candidate:0",
        candidate_index=0,
        score=OptimizerScore(value=50.0, unit="percent", aggregation="mean"),
    )
    assert all(not hasattr(event, "program") for event in (proposed, evaluated, selected))
    assert callback.evaluate_inputs[0]["callback_metadata"] == {
        "metric_key": "eval_full",
        "candidate_id": "candidate:0",
        "candidate_index": 0,
        "evaluation_id": "candidate:0:evaluation:0",
    }
    assert [set(candidate) for candidate in result.candidate_programs] == [
        {"score", "subscores", "seed", "program"}
    ]


def test_random_search_events_preserve_sparse_seed_identity_and_first_tie_winner():
    optimizer = BootstrapFewShotWithRandomSearch(metric=identity_metric)
    callback = RandomSearchCallback(optimizer)
    dspy.configure(callbacks=[callback])

    result = optimizer.compile(IdentityModule(), trainset=make_identity_trainset(), restrict=[-3, -1])

    events = [record[2] for record in callback.records if record[0] == "optimizer_event"]
    proposed = [event for event in events if isinstance(event, RandomSearchCandidateProposed)]
    assert [event.candidate_id for event in proposed] == [
        "candidate:0",
        "candidate:1",
    ]
    assert [event.seed for event in proposed] == [-3, -1]
    assert events[-1] == CandidateSelected(
        candidate_id="candidate:0",
        candidate_index=0,
        score=OptimizerScore(value=50.0, unit="percent", aggregation="mean"),
    )
    assert [candidate["seed"] for candidate in result.candidate_programs] == [-3, -1]


def test_random_search_selection_event_matches_early_stop_winner():
    optimizer = BootstrapFewShotWithRandomSearch(metric=identity_metric, stop_at_score=50.0)
    callback = RandomSearchCallback(optimizer)
    dspy.configure(callbacks=[callback])

    result = optimizer.compile(IdentityModule(), trainset=make_identity_trainset(), restrict=[-3, -2])

    events = [record[2] for record in callback.records if record[0] == "optimizer_event"]
    assert events[-1] == CandidateSelected(
        candidate_id="candidate:0",
        candidate_index=0,
        score=OptimizerScore(value=50.0, unit="percent", aggregation="mean"),
    )
    assert [candidate["seed"] for candidate in result.candidate_programs] == [-3]


def test_random_search_event_callback_failure_does_not_change_result():
    class FailingCallback(BaseCallback):
        def on_optimizer_event(self, call_id, instance, event):
            raise ValueError("tracker failed")

    dspy.configure(callbacks=[FailingCallback()])
    optimizer = BootstrapFewShotWithRandomSearch(metric=identity_metric)

    result = optimizer.compile(IdentityModule(), trainset=make_identity_trainset(), restrict=[-3])

    assert result.candidate_programs[0]["seed"] == -3


def test_random_search_does_not_coerce_metric_scores_without_event_listeners():
    class NonCoercibleScore(float):
        def __float__(self):
            raise RuntimeError("instrumentation must not coerce metric scores")

    def metric(example, prediction, trace=None):
        return NonCoercibleScore(example.output == prediction.output)

    dspy.configure(callbacks=[BaseCallback()])
    optimizer = BootstrapFewShotWithRandomSearch(metric=metric)

    result = optimizer.compile(IdentityModule(), trainset=make_identity_trainset(), restrict=[-3])

    assert result.candidate_programs[0]["seed"] == -3
