from typing import Any

from dspy.primitives import Example, Module
from dspy.utils.callback import with_callbacks


def _is_optimizer_callback_wrapper(fn) -> bool:
    """Return whether ``fn`` is the exact wrapper installed by this module."""
    return (
        getattr(fn, "_dspy_optimizer_callbacks_applied", False)
        and getattr(fn, "__wrapped__", None) is getattr(fn, "_dspy_optimizer_callback_original", None)
    )


def _with_optimizer_callbacks(fn):
    wrapper = with_callbacks(fn, _callback_kind="optimizer")
    wrapper._dspy_optimizer_callbacks_applied = True
    wrapper._dspy_optimizer_callback_original = fn
    return wrapper


class Teleprompter:
    def __init__(self):
        pass

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        compile_method = cls.__dict__.get("compile")
        if compile_method is None:
            compile_method = cls.compile
        if not _is_optimizer_callback_wrapper(compile_method):
            cls.compile = _with_optimizer_callbacks(compile_method)

    def compile(self, student: Module, *, trainset: list[Example], teacher: Module | None = None, valset: list[Example] | None = None, **kwargs) -> Module:
        """
        Optimize the student program.

        Args:
            student: The student program to optimize.
            trainset: The training set to use for optimization.
            teacher: The teacher program to use for optimization.
            valset: The validation set to use for optimization.

        Returns:
            The optimized student program.
        """
        raise NotImplementedError

    def get_params(self) -> dict[str, Any]:
        """
        Get the parameters of the teleprompter.

        Returns:
            The parameters of the teleprompter.
        """
        return self.__dict__
