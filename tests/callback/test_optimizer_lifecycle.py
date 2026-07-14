import asyncio
import functools
import inspect

import pytest

import dspy
from dspy.utils.callback import (
    ACTIVE_CALL_ID,
    BaseCallback,
    OptimizerEvent,
    emit_optimizer_event,
    optimizer_has_event_listeners,
    with_callbacks,
)


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


class OptimizerAndModuleCallback(OptimizerCallback):
    def on_module_start(self, call_id, instance, inputs):
        self.calls.append({"handler": "on_module_start", "call_id": call_id})

    def on_module_end(self, call_id, outputs, exception):
        self.calls.append({"handler": "on_module_end", "call_id": call_id})


def test_optimizer_callbacks_preserve_inputs_output_and_signature():
    class Optimizer(dspy.Teleprompter):
        def compile(self, student, *, trainset, scale=1):
            return student

    callback = OptimizerAndModuleCallback()
    dspy.configure(callbacks=[callback])
    optimizer = Optimizer()
    student = dspy.Predict("question -> answer")

    result = optimizer.compile(student, trainset=[], scale=2)

    assert inspect.signature(optimizer.compile) == inspect.Signature(
        parameters=[
            inspect.Parameter("student", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("trainset", inspect.Parameter.KEYWORD_ONLY),
            inspect.Parameter("scale", inspect.Parameter.KEYWORD_ONLY, default=1),
        ]
    )
    assert result is student
    assert [call["handler"] for call in callback.calls] == ["on_optimizer_start", "on_optimizer_end"]
    start, end = callback.calls
    assert start["instance"] is optimizer
    assert start["inputs"] == {"student": student, "trainset": [], "scale": 2}
    assert start["parent_call_id"] is None
    assert end["call_id"] == start["call_id"]
    assert end["outputs"] is student
    assert end["exception"] is None


@pytest.mark.parametrize("kind", ["inherited", "super", "decorated"])
def test_optimizer_compile_is_wrapped_once(kind):
    class Parent(dspy.Teleprompter):
        def compile(self, student, *, trainset):
            return student

    if kind == "inherited":

        class Optimizer(Parent):
            pass

    elif kind == "super":

        class Optimizer(Parent):
            def compile(self, student, *, trainset):
                return super().compile(student, trainset=trainset)

    else:

        class Optimizer(dspy.Teleprompter):
            @with_callbacks
            def compile(self, student, *, trainset):
                return student

    callback = OptimizerAndModuleCallback()
    dspy.configure(callbacks=[callback])

    Optimizer().compile(dspy.Predict("question -> answer"), trainset=[])

    assert [call["handler"] for call in callback.calls] == ["on_optimizer_start", "on_optimizer_end"]


def test_distinct_nested_optimizers_emit_nested_callbacks():
    class Inner(dspy.Teleprompter):
        def compile(self, student, *, trainset):
            return student

    inner = Inner()

    class Outer(dspy.Teleprompter):
        def compile(self, student, *, trainset):
            return inner.compile(student, trainset=trainset)

    outer = Outer()
    callback = OptimizerCallback()
    dspy.configure(callbacks=[callback])

    outer.compile(dspy.Predict("question -> answer"), trainset=[])

    outer_start, inner_start, inner_end, outer_end = callback.calls
    assert outer_start["instance"] is outer
    assert outer_start["parent_call_id"] is None
    assert inner_start["instance"] is inner
    assert inner_start["parent_call_id"] == outer_start["call_id"]
    assert inner_end["call_id"] == inner_start["call_id"]
    assert outer_end["call_id"] == outer_start["call_id"]


def test_optimizer_callback_receives_compile_exception():
    error = ValueError("compile failed")

    class Optimizer(dspy.Teleprompter):
        def compile(self, student, *, trainset):
            raise error

    callback = OptimizerCallback()
    dspy.configure(callbacks=[callback])

    with pytest.raises(ValueError, match="compile failed") as raised:
        Optimizer().compile(dspy.Predict("question -> answer"), trainset=[])

    assert raised.value is error
    assert callback.calls[1]["outputs"] is None
    assert callback.calls[1]["exception"] is error


def test_optimizer_instance_callbacks():
    callback = OptimizerCallback()

    class Optimizer(dspy.Teleprompter):
        def __init__(self):
            self.callbacks = [callback]

        def compile(self, student, *, trainset):
            return student

    Optimizer().compile(dspy.Predict("question -> answer"), trainset=[])

    assert [call["handler"] for call in callback.calls] == ["on_optimizer_start", "on_optimizer_end"]


@pytest.mark.asyncio
async def test_async_optimizer_callbacks():
    class Optimizer(dspy.Teleprompter):
        async def compile(self, student, *, trainset):
            return student

    callback = OptimizerCallback()
    student = dspy.Predict("question -> answer")

    with dspy.context(callbacks=[callback]):
        result = await Optimizer().compile(student, trainset=[])

    assert result is student
    assert [call["handler"] for call in callback.calls] == ["on_optimizer_start", "on_optimizer_end"]


def test_optimizer_wrapping_preserves_user_decorators():
    body_calls = []

    def custom_decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            body_calls.append("custom")
            return fn(*args, **kwargs)

        return wrapper

    class Optimizer(dspy.Teleprompter):
        @custom_decorator
        @with_callbacks
        def compile(self, student, *, trainset):
            body_calls.append("body")
            return student

    callback = OptimizerAndModuleCallback()
    dspy.configure(callbacks=[callback])
    student = dspy.Predict("question -> answer")

    Optimizer().compile(student, trainset=[])

    assert body_calls == ["custom", "body"]
    assert [call["handler"] for call in callback.calls] == ["on_optimizer_start", "on_optimizer_end"]
    assert callback.calls[0]["inputs"] == {"student": student, "trainset": []}


def test_optimizer_super_compile_suppresses_parent_callback_wrapper():
    class Parent(dspy.Teleprompter):
        @with_callbacks
        def compile(self, student, *, trainset):
            return student

    class Optimizer(Parent):
        def compile(self, student, *, trainset):
            return super().compile(student, trainset=trainset)

    callback = OptimizerAndModuleCallback()
    dspy.configure(callbacks=[callback])

    Optimizer().compile(dspy.Predict("question -> answer"), trainset=[])

    assert [call["handler"] for call in callback.calls] == ["on_optimizer_start", "on_optimizer_end"]


def test_compile_callback_suppression_is_invocation_local():
    @with_callbacks
    def shared_method(self, student=None, *, trainset=None):
        return student

    class CompileOwner(dspy.Teleprompter):
        compile = shared_method

    class HelperOwner(dspy.Teleprompter):
        helper = shared_method

        def compile(self, student, *, trainset):
            self.helper()
            return student

    callback = OptimizerAndModuleCallback()
    dspy.configure(callbacks=[callback])

    CompileOwner().compile(dspy.Predict("question -> answer"), trainset=[])
    assert [call["handler"] for call in callback.calls] == ["on_optimizer_start", "on_optimizer_end"]

    callback.calls.clear()
    HelperOwner().compile(dspy.Predict("question -> answer"), trainset=[])
    assert [call["handler"] for call in callback.calls] == [
        "on_optimizer_start",
        "on_module_start",
        "on_module_end",
        "on_optimizer_end",
    ]


def test_optimizer_classification_does_not_depend_on_compile_function_name():
    body_calls = []

    def opaque_decorator(fn):
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    class Optimizer(dspy.Teleprompter):
        @opaque_decorator
        def compile(self, student, *, trainset):
            body_calls.append("body")
            return student

    callback = OptimizerCallback()
    dspy.configure(callbacks=[callback])

    Optimizer().compile(dspy.Predict("question -> answer"), trainset=[])

    assert body_calls == ["body"]
    assert [call["handler"] for call in callback.calls] == ["on_optimizer_start", "on_optimizer_end"]


def test_optimizer_accepts_none_for_instance_callbacks():
    class Optimizer(dspy.Teleprompter):
        def __init__(self):
            self.callbacks = None

        def compile(self, student, *, trainset):
            return student

    student = dspy.Predict("question -> answer")

    assert Optimizer().compile(student, trainset=[]) is student


def test_mixin_first_optimizer_compile_is_instrumented():
    class CompileMixin:
        def compile(self, student, *, trainset):
            return student

    class Optimizer(CompileMixin, dspy.Teleprompter):
        pass

    callback = OptimizerCallback()
    dspy.configure(callbacks=[callback])

    Optimizer().compile(dspy.Predict("question -> answer"), trainset=[])

    assert [call["handler"] for call in callback.calls] == ["on_optimizer_start", "on_optimizer_end"]


def test_callback_decorated_optimizer_helper_keeps_module_semantics():
    class Callback(OptimizerCallback):
        def on_module_start(self, call_id, instance, inputs):
            self.calls.append({"handler": "on_module_start", "call_id": call_id})

        def on_module_end(self, call_id, outputs, exception):
            self.calls.append({"handler": "on_module_end", "call_id": call_id})

    class Optimizer(dspy.Teleprompter):
        @with_callbacks
        def helper(self):
            return "helper"

        def compile(self, student, *, trainset):
            self.helper()
            return student

    callback = Callback()
    dspy.configure(callbacks=[callback])
    optimizer = Optimizer()

    assert optimizer.helper() == "helper"
    assert [call["handler"] for call in callback.calls] == ["on_module_start", "on_module_end"]

    callback.calls.clear()
    optimizer.compile(dspy.Predict("question -> answer"), trainset=[])
    assert [call["handler"] for call in callback.calls] == [
        "on_optimizer_start",
        "on_module_start",
        "on_module_end",
        "on_optimizer_end",
    ]


def test_recursive_optimizer_compile_uses_one_lifecycle():
    depths = []

    class Optimizer(dspy.Teleprompter):
        def compile(self, student, *, trainset, depth):
            depths.append(depth)
            if depth:
                return self.compile(student, trainset=trainset, depth=depth - 1)
            return student

    callback = OptimizerCallback()
    dspy.configure(callbacks=[callback])

    Optimizer().compile(dspy.Predict("question -> answer"), trainset=[], depth=1)

    assert depths == [1, 0]
    assert [call["handler"] for call in callback.calls] == ["on_optimizer_start", "on_optimizer_end"]


@pytest.mark.asyncio
async def test_async_optimizer_cancellation_is_reported():
    started = asyncio.Event()

    class Optimizer(dspy.Teleprompter):
        async def compile(self, student, *, trainset):
            started.set()
            await asyncio.Event().wait()

    callback = OptimizerCallback()
    with dspy.context(callbacks=[callback]):
        task = asyncio.create_task(Optimizer().compile(dspy.Predict("question -> answer"), trainset=[]))
        await started.wait()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert [call["handler"] for call in callback.calls] == ["on_optimizer_start", "on_optimizer_end"]
    assert isinstance(callback.calls[-1]["exception"], asyncio.CancelledError)


@pytest.mark.asyncio
async def test_async_optimizer_behind_async_decorator_keeps_lifecycle_open():
    order = []

    def async_decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            return await fn(*args, **kwargs)

        return wrapper

    class Callback(OptimizerCallback):
        def on_optimizer_start(self, call_id, instance, inputs):
            order.append("start")
            super().on_optimizer_start(call_id, instance, inputs)

        def on_optimizer_end(self, call_id, outputs, exception):
            order.append("end")
            super().on_optimizer_end(call_id, outputs, exception)

    class Optimizer(dspy.Teleprompter):
        @async_decorator
        async def compile(self, student, *, trainset):
            await asyncio.sleep(0)
            order.append("body")
            return student

    callback = Callback()
    student = dspy.Predict("question -> answer")
    with dspy.context(callbacks=[callback]):
        result = await Optimizer().compile(student, trainset=[])

    assert result is student
    assert order == ["start", "body", "end"]
    assert [call["handler"] for call in callback.calls] == ["on_optimizer_start", "on_optimizer_end"]
    assert callback.calls[-1]["outputs"] is student


def test_sync_decorator_around_async_optimizer_is_rejected():
    body_calls = []

    def sync_decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    class Optimizer(dspy.Teleprompter):
        @sync_decorator
        async def compile(self, student, *, trainset):
            body_calls.append("body")
            return student

    optimizer = Optimizer()
    student = dspy.Predict("question -> answer")
    message = "decorators around async compile must also use an async def wrapper"

    with dspy.context(callbacks=[]):
        with pytest.raises(TypeError, match=message):
            optimizer.compile(student, trainset=[])

    callback = OptimizerCallback()
    with dspy.context(callbacks=[callback]):
        with pytest.raises(TypeError, match=message) as raised:
            optimizer.compile(student, trainset=[])

    assert body_calls == []
    assert [call["handler"] for call in callback.calls] == ["on_optimizer_start", "on_optimizer_end"]
    assert callback.calls[-1]["outputs"] is None
    assert callback.calls[-1]["exception"] is raised.value
    assert ACTIVE_CALL_ID.get() is None


def test_syncifying_decorator_preserves_sync_optimizer_contract():
    def syncify(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return asyncio.run(fn(*args, **kwargs))

        return wrapper

    class Optimizer(dspy.Teleprompter):
        @syncify
        async def compile(self, student, *, trainset):
            return student

    callback = OptimizerCallback()
    dspy.configure(callbacks=[callback])
    student = dspy.Predict("question -> answer")

    result = Optimizer().compile(student, trainset=[])

    assert result is student
    assert not inspect.isawaitable(result)
    assert [call["handler"] for call in callback.calls] == ["on_optimizer_start", "on_optimizer_end"]


@pytest.mark.asyncio
async def test_sync_optimizer_preserves_returned_task_identity():
    class Optimizer(dspy.Teleprompter):
        def compile(self, student, *, trainset):
            return asyncio.create_task(asyncio.sleep(0))

    callback = OptimizerCallback()
    with dspy.context(callbacks=[callback]):
        task = Optimizer().compile(dspy.Predict("question -> answer"), trainset=[])

    assert isinstance(task, asyncio.Task)
    assert callback.calls[-1]["outputs"] is task
    await task


def test_sync_optimizer_interruption_is_reported():
    error = KeyboardInterrupt("stop")

    class Optimizer(dspy.Teleprompter):
        def compile(self, student, *, trainset):
            raise error

    callback = OptimizerCallback()
    dspy.configure(callbacks=[callback])

    with pytest.raises(KeyboardInterrupt) as raised:
        Optimizer().compile(dspy.Predict("question -> answer"), trainset=[])

    assert raised.value is error
    assert callback.calls[-1]["exception"] is error


def test_optimizer_base_exception_restores_callback_context():
    class Stop(BaseException):
        pass

    error = Stop("stop")

    class Optimizer(dspy.Teleprompter):
        def compile(self, student, *, trainset):
            raise error

    callback = OptimizerCallback()
    dspy.configure(callbacks=[callback])
    optimizer = Optimizer()

    with pytest.raises(Stop) as raised:
        optimizer.compile(dspy.Predict("question -> answer"), trainset=[])

    assert raised.value is error
    assert callback.calls[-1]["exception"] is error
    assert ACTIVE_CALL_ID.get() is None


@pytest.mark.asyncio
async def test_child_task_can_start_new_lifecycle_after_parent_ends():
    gate = asyncio.Event()

    class Optimizer(dspy.Teleprompter):
        async def compile(self, student, *, trainset, mode):
            if mode == "inner":
                return "inner-result"

            async def compile_later():
                await gate.wait()
                return await self.compile(student, trainset=trainset, mode="inner")

            return asyncio.create_task(compile_later())

    callback = OptimizerCallback()
    with dspy.context(callbacks=[callback]):
        task = await Optimizer().compile(
            dspy.Predict("question -> answer"),
            trainset=[],
            mode="outer",
        )
        gate.set()
        assert await task == "inner-result"

    assert [call["handler"] for call in callback.calls] == [
        "on_optimizer_start",
        "on_optimizer_end",
        "on_optimizer_start",
        "on_optimizer_end",
    ]


@pytest.mark.asyncio
async def test_child_task_starts_independent_lifecycle_while_parent_is_active():
    inner_started = asyncio.Event()
    release_inner = asyncio.Event()

    class Optimizer(dspy.Teleprompter):
        async def compile(self, student, *, trainset, mode):
            if mode == "inner":
                inner_started.set()
                await release_inner.wait()
                return "inner-result"

            task = asyncio.create_task(self.compile(student, trainset=trainset, mode="inner"))
            await inner_started.wait()
            return task

    callback = OptimizerCallback()
    with dspy.context(callbacks=[callback]):
        task = await Optimizer().compile(
            dspy.Predict("question -> answer"),
            trainset=[],
            mode="outer",
        )

        outer_start, inner_start, outer_end = callback.calls
        assert [call["handler"] for call in callback.calls] == [
            "on_optimizer_start",
            "on_optimizer_start",
            "on_optimizer_end",
        ]
        assert inner_start["parent_call_id"] == outer_start["call_id"]
        assert outer_end["call_id"] == outer_start["call_id"]

        release_inner.set()
        assert await task == "inner-result"

    inner_end = callback.calls[-1]
    assert inner_end["handler"] == "on_optimizer_end"
    assert inner_end["call_id"] == inner_start["call_id"]


@pytest.mark.asyncio
async def test_child_task_cannot_emit_event_for_parent_optimizer_invocation():
    child_has_listeners = []

    class Callback(OptimizerCallback):
        def on_optimizer_event(self, call_id, instance, event):
            self.calls.append({"handler": "on_optimizer_event", "call_id": call_id})

    class Optimizer(dspy.Teleprompter):
        async def compile(self, student, *, trainset):
            assert optimizer_has_event_listeners(self)
            emit_optimizer_event(self, OptimizerEvent())

            async def emit_from_child():
                child_has_listeners.append(optimizer_has_event_listeners(self))
                with pytest.raises(RuntimeError, match="owning execution context"):
                    emit_optimizer_event(self, OptimizerEvent())

            await asyncio.create_task(emit_from_child())
            return student

    callback = Callback()
    student = dspy.Predict("question -> answer")
    with dspy.context(callbacks=[callback]):
        result = await Optimizer().compile(student, trainset=[])

    assert result is student
    assert child_has_listeners == [False]
    optimizer_start, optimizer_event, optimizer_end = callback.calls
    assert [call["handler"] for call in callback.calls] == [
        "on_optimizer_start",
        "on_optimizer_event",
        "on_optimizer_end",
    ]
    assert optimizer_event["call_id"] == optimizer_start["call_id"]
    assert optimizer_end["call_id"] == optimizer_start["call_id"]
