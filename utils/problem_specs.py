"""
Optional per-problem input override registry.

Use this for irregular signatures that need richer synthetic data than the
default `create_input_tensor` logic.
"""

from __future__ import annotations

from typing import Callable


InputOverride = Callable[..., tuple]
_INPUT_OVERRIDES: dict[int, InputOverride] = {}


def register_input_override(problem_id: int):
    def decorator(fn: InputOverride) -> InputOverride:
        _INPUT_OVERRIDES[problem_id] = fn
        return fn

    return decorator


def get_problem_input_override(problem_id: int) -> InputOverride | None:
    return _INPUT_OVERRIDES.get(problem_id)


# Add overrides below as needed:
#
# @register_input_override(50)
# def make_inputs_problem_50(**kwargs) -> tuple:
#     ...
