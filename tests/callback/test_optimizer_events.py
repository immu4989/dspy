import asyncio
from typing import get_type_hints

import pytest

import dspy
from dspy.teleprompt.events import CandidateSelected, OptimizerScore
from dspy.utils.callback import ACTIVE_CALL_ID, BaseCallback, OptimizerEvent, emit_optimizer_event


@pytest.fixture(autouse=True)
def reset_settings():
    original_settings = dspy.settings.copy()
    yield
    dspy.configure(**original_settings)


class OptimizerCallback(BaseCallback):
    def __init__(self):
        self.calls = []

    def on_optimizer_start(self, call_id, instance, inputs):
        self.calls.append(
            {
                "handler": "on_optimizer_start",
                "call_id": call_id,
                "parent_call_id": ACTIVE_CALL_ID.get(),
                "instance": instance,
                "inputs": inputs,
            }
        )

    def on_optimizer_end(self, call_id, outputs, exception):
        self.calls.append(
            {
                "handler": "on_optimizer_end",
                "call_id": call_id,
                "outputs": outputs,
                "exception": exception,
            }
        )


@pytest.mark.asyncio
async def test_task_cannot_emit_optimizer_event_after_optimizer_end():
    gate = asyncio.Event()
    event = CandidateSelected(
        candidate_id="candidate:0",
        candidate_index=0,
        score=OptimizerScore(value=1.0, unit="raw_metric", aggregation="identity"),
    )

    class Callback(OptimizerCallback):
        def on_optimizer_event(self, call_id, instance, event):
            self.calls.append({"handler": "on_optimizer_event", "call_id": call_id})

    class Optimizer(dspy.Teleprompter):
        async def compile(self, student, *, trainset):
            async def emit_later():
                await gate.wait()
                emit_optimizer_event(self, event)

            return student, asyncio.create_task(emit_later())

    callback = Callback()
    with dspy.context(callbacks=[callback]):
        _, task = await Optimizer().compile(dspy.Predict("question -> answer"), trainset=[])
        assert [call["handler"] for call in callback.calls] == ["on_optimizer_start", "on_optimizer_end"]
        gate.set()
        await task

    assert [call["handler"] for call in callback.calls] == ["on_optimizer_start", "on_optimizer_end"]


def test_optimizer_event_uses_optimizer_id_inside_nested_module():
    expected_event = CandidateSelected(
        candidate_id="candidate:0",
        candidate_index=0,
        score=OptimizerScore(value=1.0, unit="raw_metric", aggregation="identity"),
    )

    class Callback(OptimizerCallback):
        def on_optimizer_event(self, call_id, instance, event):
            self.calls.append(
                {
                    "handler": "on_optimizer_event",
                    "call_id": call_id,
                    "event": event,
                }
            )

    class Emitter(dspy.Module):
        def __init__(self, optimizer):
            self.optimizer = optimizer

        def forward(self):
            emit_optimizer_event(self.optimizer, expected_event)

    class Optimizer(dspy.Teleprompter):
        def compile(self, student, *, trainset):
            Emitter(self)()
            return student

    callback = Callback()
    dspy.configure(callbacks=[callback])

    Optimizer().compile(dspy.Predict("question -> answer"), trainset=[])

    optimizer_start, optimizer_event, optimizer_end = [
        call for call in callback.calls if call["handler"].startswith("on_optimizer")
    ]
    assert optimizer_event["call_id"] == optimizer_start["call_id"]
    assert optimizer_event["event"] is expected_event
    assert optimizer_end["call_id"] == optimizer_start["call_id"]


def test_optimizer_events_use_lifecycle_callback_snapshot():
    expected_event = CandidateSelected(
        candidate_id="candidate:0",
        candidate_index=0,
        score=OptimizerScore(value=1.0, unit="raw_metric", aggregation="identity"),
    )

    class Callback(OptimizerCallback):
        def on_optimizer_event(self, call_id, instance, event):
            self.calls.append({"handler": "on_optimizer_event", "call_id": call_id})

    replacement = Callback()

    class Optimizer(dspy.Teleprompter):
        def compile(self, student, *, trainset):
            dspy.configure(callbacks=[replacement])
            emit_optimizer_event(self, expected_event)
            return student

    original = Callback()
    dspy.configure(callbacks=[original])

    Optimizer().compile(dspy.Predict("question -> answer"), trainset=[])

    assert [call["handler"] for call in original.calls] == [
        "on_optimizer_start",
        "on_optimizer_event",
        "on_optimizer_end",
    ]
    assert replacement.calls == []


def test_optimizer_event_type_is_runtime_resolvable():
    assert get_type_hints(BaseCallback.on_optimizer_event)["event"] is OptimizerEvent
    assert get_type_hints(emit_optimizer_event)["event"] is OptimizerEvent


def test_optimizer_event_without_listener_is_a_noop():
    event = CandidateSelected(
        candidate_id="candidate:0",
        candidate_index=0,
        score=OptimizerScore(value=1.0, unit="raw_metric", aggregation="identity"),
    )

    class Optimizer(dspy.Teleprompter):
        def compile(self, student, *, trainset):
            emit_optimizer_event(self, event)
            return student

    student = dspy.Predict("question -> answer")
    with dspy.context(callbacks=[]):
        assert Optimizer().compile(student, trainset=[]) is student


def test_optimizer_event_with_listener_requires_active_lifecycle():
    event = CandidateSelected(
        candidate_id="candidate:0",
        candidate_index=0,
        score=OptimizerScore(value=1.0, unit="raw_metric", aggregation="identity"),
    )

    class Callback(BaseCallback):
        def on_optimizer_event(self, call_id, instance, event):
            pass

    optimizer = dspy.Teleprompter()
    with dspy.context(callbacks=[Callback()]):
        with pytest.raises(RuntimeError, match="within an optimizer compile callback"):
            emit_optimizer_event(optimizer, event)
