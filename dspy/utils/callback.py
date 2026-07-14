import functools
import inspect
import logging
import threading
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable

import anyio

import dspy
from dspy.utils.callback_context import ACTIVE_CALL_ID

logger = logging.getLogger(__name__)


class OptimizerEvent:
    """Base class for typed events emitted during an optimizer invocation."""

    __slots__ = ()


@dataclass
class _OptimizerInvocation:
    instance: Any
    call_id: str
    callbacks: tuple[Any, ...]
    closed: threading.Event
    owner: tuple[threading.Thread, object | None]
    suppressed_callback_wrappers: set[Callable]


_ACTIVE_OPTIMIZER_INVOCATIONS = ContextVar("active_optimizer_invocations", default=())


def _get_optimizer_invocation(instance: Any) -> _OptimizerInvocation | None:
    for invocation in reversed(_ACTIVE_OPTIMIZER_INVOCATIONS.get()):
        if invocation.instance is instance:
            return invocation
    return None


def _current_execution_owner() -> tuple[threading.Thread, object | None]:
    try:
        task = anyio.get_current_task()
    except RuntimeError:
        task = None

    return threading.current_thread(), task


def _get_owned_optimizer_invocation(
    instance: Any,
    owner: tuple[threading.Thread, object | None],
) -> _OptimizerInvocation | None:
    for invocation in reversed(_ACTIVE_OPTIMIZER_INVOCATIONS.get()):
        if invocation.instance is instance and invocation.owner == owner:
            return invocation
    return None


def _validate_sync_optimizer_result(result: Any) -> Any:
    if inspect.iscoroutine(result):
        result.close()
        raise TypeError(
            "Synchronous optimizer compile returned a coroutine. Declare compile with async def; "
            "decorators around async compile must also use an async def wrapper."
        )
    return result


def _callback_wrappers_in_chain(fn: Callable) -> set[Callable]:
    """Return exact DSPy callback wrappers exposed by ``fn.__wrapped__``."""
    wrappers = set()
    seen = set()
    current = fn
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if (
            getattr(current, "_dspy_callbacks_applied", False)
            and getattr(current, "_dspy_callback_kind", None) != "optimizer"
            and getattr(current, "__wrapped__", None) is getattr(current, "_dspy_callback_original", None)
        ):
            wrappers.add(current)
        current = getattr(current, "__wrapped__", None)
    return wrappers


def _is_callback_wrapper_suppressed(instance: Any, wrapper: Callable) -> bool:
    if not _ACTIVE_OPTIMIZER_INVOCATIONS.get():
        return False

    invocation = _get_owned_optimizer_invocation(instance, _current_execution_owner())
    return (
        invocation is not None
        and not invocation.closed.is_set()
        and wrapper in invocation.suppressed_callback_wrappers
    )


def _get_active_callbacks(instance: Any) -> list[Any]:
    global_callbacks = dspy.settings.get("callbacks", []) or []
    instance_callbacks = getattr(instance, "callbacks", None) or []
    return list(global_callbacks) + list(instance_callbacks)


class BaseCallback:
    """A base class for defining callback handlers for DSPy components.

    To use a callback, subclass this class and implement the desired handlers. Each handler
    will be called at the appropriate time before/after the execution of the corresponding component.  For example, if
    you want to print a message before and after an LM is called, implement `the on_llm_start` and `on_lm_end` handler.
    Users can set the callback globally using `dspy.configure` or locally by passing it to the component
    constructor.


    Example 1: Set a global callback using `dspy.configure`.

    ```
    import dspy
    from dspy.utils.callback import BaseCallback

    class LoggingCallback(BaseCallback):

        def on_lm_start(self, call_id, instance, inputs):
            print(f"LM is called with inputs: {inputs}")

        def on_lm_end(self, call_id, outputs, exception):
            print(f"LM is finished with outputs: {outputs}")

    dspy.configure(
        callbacks=[LoggingCallback()]
    )

    cot = dspy.ChainOfThought("question -> answer")
    cot(question="What is the meaning of life?")

    # > LM is called with inputs: {'question': 'What is the meaning of life?'}
    # > LM is finished with outputs: {'answer': '42'}
    ```

    Example 2: Set a local callback by passing it to the component constructor.

    ```
    lm_1 = dspy.LM("gpt-3.5-turbo", callbacks=[LoggingCallback()])
    lm_1(question="What is the meaning of life?")

    # > LM is called with inputs: {'question': 'What is the meaning of life?'}
    # > LM is finished with outputs: {'answer': '42'}

    lm_2 = dspy.LM("gpt-3.5-turbo")
    lm_2(question="What is the meaning of life?")
    # No logging here because only `lm_1` has the callback set.
    ```
    """

    def on_module_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when forward() method of a module (subclass of dspy.Module) is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Module instance.
            inputs: The inputs to the module's forward() method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        pass

    def on_module_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ):
        """A handler triggered after forward() method of a module (subclass of dspy.Module) is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the module's forward() method. If the method is interrupted by
                an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass

    def on_lm_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when __call__ method of dspy.LM instance is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The LM instance.
            inputs: The inputs to the LM's __call__ method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        pass

    def on_lm_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ):
        """A handler triggered after __call__ method of dspy.LM instance is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the LM's __call__ method. If the method is interrupted by
                an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass

    def on_adapter_format_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when format() method of an adapter (subclass of dspy.Adapter) is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Adapter instance.
            inputs: The inputs to the Adapter's format() method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        pass

    def on_adapter_format_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ):
        """A handler triggered after format() method of an adapter (subclass of dspy.Adapter) is called..

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the Adapter's format() method. If the method is interrupted
                by an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass

    def on_adapter_parse_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when parse() method of an adapter (subclass of dspy.Adapter) is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Adapter instance.
            inputs: The inputs to the Adapter's parse() method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        pass

    def on_adapter_parse_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ):
        """A handler triggered after parse() method of an adapter (subclass of dspy.Adapter) is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the Adapter's parse() method. If the method is interrupted
                by an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass

    def on_tool_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when a tool is called.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Tool instance.
            inputs: The inputs to the Tool's __call__ method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        pass

    def on_tool_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ):
        """A handler triggered after a tool is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the Tool's __call__ method. If the method is interrupted by
                an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass

    def on_evaluate_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when evaluation is started.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            instance: The Evaluate instance.
            inputs: The inputs to the Evaluate's __call__ method. Each arguments is stored as
                a key-value pair in a dictionary.
        """
        pass

    def on_evaluate_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: Exception | None = None,
    ):
        """A handler triggered after evaluation is executed.

        Args:
            call_id: A unique identifier for the call. Can be used to connect start/end handlers.
            outputs: The outputs of the Evaluate's __call__ method. If the method is interrupted by
                an exception, this will be None.
            exception: If an exception is raised during the execution, it will be stored here.
        """
        pass

    def on_optimizer_start(
        self,
        call_id: str,
        instance: Any,
        inputs: dict[str, Any],
    ):
        """A handler triggered when an optimizer starts compiling a program.

        Args:
            call_id: A unique identifier for the optimizer invocation.
            instance: The Teleprompter instance.
            inputs: The arguments passed to the optimizer's ``compile`` method.
        """
        pass

    def on_optimizer_end(
        self,
        call_id: str,
        outputs: Any | None,
        exception: BaseException | None = None,
    ):
        """A handler triggered when an optimizer finishes compiling a program.

        Args:
            call_id: The identifier shared with the corresponding start handler.
            outputs: The compiled program, or None if compilation failed.
            exception: The exception raised by compilation, if any.
        """
        pass

    def on_optimizer_event(
        self,
        call_id: str,
        instance: Any,
        event: OptimizerEvent,
    ):
        """A handler triggered for a typed event within an optimizer invocation.

        Args:
            call_id: The identifier of the optimizer invocation emitting the event.
            instance: The Teleprompter instance.
            event: A typed optimizer event.
        """
        pass


def _get_optimizer_event_handlers(callbacks: list[Any] | tuple[Any, ...]) -> list[tuple[Any, Callable]]:
    handlers = []
    for callback in callbacks:
        handler = getattr(callback, "on_optimizer_event", None)
        if handler is None or getattr(handler, "__func__", None) is BaseCallback.on_optimizer_event:
            continue
        handlers.append((callback, handler))
    return handlers


def optimizer_has_event_listeners(instance: Any) -> bool:
    """Return whether the current optimizer invocation has an event listener."""
    invocation = _get_owned_optimizer_invocation(instance, _current_execution_owner())
    return (
        invocation is not None
        and not invocation.closed.is_set()
        and bool(_get_optimizer_event_handlers(invocation.callbacks))
    )


def emit_optimizer_event(instance: Any, event: OptimizerEvent) -> None:
    """Emit an event from optimizer coordinator code.

    Optimizer invocation context is task-local and is not propagated to parallel
    workers. Events from detached tasks are ignored after the invocation ends. If
    no event listeners are configured, emission is a no-op.
    """
    invocation = _get_owned_optimizer_invocation(instance, _current_execution_owner())
    if invocation is None:
        inherited_invocation = _get_optimizer_invocation(instance)
        if inherited_invocation is not None and inherited_invocation.closed.is_set():
            return
        callbacks = inherited_invocation.callbacks if inherited_invocation is not None else _get_active_callbacks(instance)
        if not _get_optimizer_event_handlers(callbacks):
            return
        raise RuntimeError(
            "Optimizer events must be emitted from within an optimizer compile callback's owning execution context."
        )
    if invocation.closed.is_set():
        return

    for callback, handler in _get_optimizer_event_handlers(invocation.callbacks):
        try:
            handler(call_id=invocation.call_id, instance=instance, event=event)
        except Exception as e:
            logger.warning(f"Error when calling callback {callback}: {e}")


def with_callbacks(fn, *, _callback_kind: str | None = None):
    """Decorator to add callback functionality to instance methods."""
    is_optimizer_callback = _callback_kind == "optimizer"
    callback_wrappers_to_suppress = _callback_wrappers_in_chain(fn) if is_optimizer_callback else set()

    def _execute_start_callbacks(instance, fn, call_id, callbacks, args, kwargs):
        """Execute all start callbacks for a function call."""
        if is_optimizer_callback:
            bound_args = inspect.signature(fn).bind(instance, *args, **kwargs)
            bound_args.apply_defaults()
            inputs = dict(bound_args.arguments)
        else:
            inputs = inspect.getcallargs(fn, instance, *args, **kwargs)
        if "self" in inputs:
            inputs.pop("self")
        elif "instance" in inputs:
            inputs.pop("instance")
        for callback in callbacks:
            try:
                _get_on_start_handler(callback, instance, fn, is_optimizer_callback)(
                    call_id=call_id,
                    instance=instance,
                    inputs=inputs,
                )
            except Exception as e:
                logger.warning(f"Error when calling callback {callback}: {e}")

    def _execute_end_callbacks(instance, fn, call_id, results, exception, callbacks):
        """Execute all end callbacks for a function call."""
        for callback in callbacks:
            try:
                _get_on_end_handler(callback, instance, fn, is_optimizer_callback)(
                    call_id=call_id,
                    outputs=results,
                    exception=exception,
                )
            except Exception as e:
                logger.warning(f"Error when applying callback {callback}'s end handler on function {fn.__name__}: {e}.")

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(instance, *args, **kwargs):
            if not is_optimizer_callback and _is_callback_wrapper_suppressed(instance, async_wrapper):
                return await fn(instance, *args, **kwargs)

            callbacks = _get_active_callbacks(instance)
            if not callbacks:
                return await fn(instance, *args, **kwargs)

            is_optimizer = is_optimizer_callback
            owner = _current_execution_owner()
            optimizer_invocation = _get_owned_optimizer_invocation(instance, owner) if is_optimizer else None
            if optimizer_invocation is not None and not optimizer_invocation.closed.is_set():
                optimizer_invocation.suppressed_callback_wrappers.update(callback_wrappers_to_suppress)
                return await fn(instance, *args, **kwargs)

            call_id = uuid.uuid4().hex

            _execute_start_callbacks(instance, fn, call_id, callbacks, args, kwargs)

            # Active ID must be set right before the function is called, not before calling the callbacks.
            call_id_token = ACTIVE_CALL_ID.set(call_id)
            optimizer_token = None
            if is_optimizer:
                invocation = _OptimizerInvocation(
                    instance=instance,
                    call_id=call_id,
                    callbacks=tuple(callbacks),
                    closed=threading.Event(),
                    owner=owner,
                    suppressed_callback_wrappers=set(callback_wrappers_to_suppress),
                )
                optimizer_token = _ACTIVE_OPTIMIZER_INVOCATIONS.set(
                    _ACTIVE_OPTIMIZER_INVOCATIONS.get() + (invocation,)
                )

            results = None
            exception = None
            try:
                results = await fn(instance, *args, **kwargs)
                return results
            except BaseException as e:
                if is_optimizer or isinstance(e, Exception):
                    exception = e
                raise
            finally:
                if is_optimizer:
                    invocation.closed.set()
                if optimizer_token is not None:
                    _ACTIVE_OPTIMIZER_INVOCATIONS.reset(optimizer_token)
                ACTIVE_CALL_ID.reset(call_id_token)
                _execute_end_callbacks(instance, fn, call_id, results, exception, callbacks)

        async_wrapper._dspy_callbacks_applied = True
        async_wrapper._dspy_callback_original = fn
        async_wrapper._dspy_callback_kind = _callback_kind
        return async_wrapper

    else:

        @functools.wraps(fn)
        def sync_wrapper(instance, *args, **kwargs):
            if not is_optimizer_callback and _is_callback_wrapper_suppressed(instance, sync_wrapper):
                return fn(instance, *args, **kwargs)

            callbacks = _get_active_callbacks(instance)
            if not callbacks:
                result = fn(instance, *args, **kwargs)
                return _validate_sync_optimizer_result(result) if is_optimizer_callback else result

            is_optimizer = is_optimizer_callback
            owner = _current_execution_owner()
            optimizer_invocation = _get_owned_optimizer_invocation(instance, owner) if is_optimizer else None
            if optimizer_invocation is not None and not optimizer_invocation.closed.is_set():
                optimizer_invocation.suppressed_callback_wrappers.update(callback_wrappers_to_suppress)
                result = fn(instance, *args, **kwargs)
                return _validate_sync_optimizer_result(result)

            call_id = uuid.uuid4().hex

            _execute_start_callbacks(instance, fn, call_id, callbacks, args, kwargs)

            # Active ID must be set right before the function is called, not before calling the callbacks.
            call_id_token = ACTIVE_CALL_ID.set(call_id)
            optimizer_token = None
            if is_optimizer:
                invocation = _OptimizerInvocation(
                    instance=instance,
                    call_id=call_id,
                    callbacks=tuple(callbacks),
                    closed=threading.Event(),
                    owner=owner,
                    suppressed_callback_wrappers=set(callback_wrappers_to_suppress),
                )
                optimizer_token = _ACTIVE_OPTIMIZER_INVOCATIONS.set(
                    _ACTIVE_OPTIMIZER_INVOCATIONS.get() + (invocation,)
                )

            results = None
            exception = None
            try:
                result = fn(instance, *args, **kwargs)
                results = _validate_sync_optimizer_result(result) if is_optimizer else result
                return results
            except BaseException as e:
                if is_optimizer or isinstance(e, Exception):
                    exception = e
                raise
            finally:
                if is_optimizer:
                    invocation.closed.set()
                if optimizer_token is not None:
                    _ACTIVE_OPTIMIZER_INVOCATIONS.reset(optimizer_token)
                ACTIVE_CALL_ID.reset(call_id_token)
                _execute_end_callbacks(instance, fn, call_id, results, exception, callbacks)

        sync_wrapper._dspy_callbacks_applied = True
        sync_wrapper._dspy_callback_original = fn
        sync_wrapper._dspy_callback_kind = _callback_kind
        return sync_wrapper


def _get_on_start_handler(
    callback: BaseCallback,
    instance: Any,
    fn: Callable,
    is_optimizer_callback: bool,
) -> Callable:
    """Selects the appropriate on_start handler of the callback based on the instance and function name."""
    if isinstance(instance, dspy.BaseLM):
        return callback.on_lm_start
    elif isinstance(instance, dspy.Evaluate):
        return callback.on_evaluate_start
    elif is_optimizer_callback:
        return callback.on_optimizer_start

    if isinstance(instance, dspy.Adapter):
        if fn.__name__ == "format":
            return callback.on_adapter_format_start
        elif fn.__name__ == "parse":
            return callback.on_adapter_parse_start
        else:
            raise ValueError(f"Unsupported adapter method for using callback: {fn.__name__}.")

    if isinstance(instance, dspy.Tool):
        return callback.on_tool_start

    # We treat everything else as a module.
    return callback.on_module_start


def _get_on_end_handler(
    callback: BaseCallback,
    instance: Any,
    fn: Callable,
    is_optimizer_callback: bool,
) -> Callable:
    """Selects the appropriate on_end handler of the callback based on the instance and function name."""
    if isinstance(instance, dspy.BaseLM):
        return callback.on_lm_end
    elif isinstance(instance, dspy.Evaluate):
        return callback.on_evaluate_end
    elif is_optimizer_callback:
        return callback.on_optimizer_end

    if isinstance(instance, (dspy.Adapter)):
        if fn.__name__ == "format":
            return callback.on_adapter_format_end
        elif fn.__name__ == "parse":
            return callback.on_adapter_parse_end
        else:
            raise ValueError(f"Unsupported adapter method for using callback: {fn.__name__}.")

    if isinstance(instance, dspy.Tool):
        return callback.on_tool_end

    # We treat everything else as a module.
    return callback.on_module_end
