# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import ast

import fastforward as ff
import pytest
import syrupy


def _autoquantize_str(input: str, as_module: bool = False, use_type_inference: bool = False) -> str:
    autoquant_result = ff.testing.autoquant.autoquantize_str(
        input, as_module=as_module, as_cst=False, use_type_inference=use_type_inference
    )
    assert isinstance(autoquant_result, str)
    return autoquant_result


@pytest.mark.slow
def test_raise(snapshot: syrupy.assertion.SnapshotAssertion) -> None:
    # Autoquant for `raise` behaves similarly to `return`. I.e., no new
    # quantizer should be introduced after 'resetting the quantization status'.
    # In the case below, the assignment to `x` in the if branch is not 'live'
    # after the raise statement. Hence, no quantization calls must be
    # introduced due to the multiplication in the final return statement.
    input = """
    def my_function(self, x: torch.Tensor) -> torch.Tensor:
        if True:
            x = some_function(x) # reset quantization status
            raise ValueError("msg")
        return x * x
    """
    result = _autoquantize_str(input)
    assert snapshot == result


@pytest.mark.slow
def test_autoquant_ignore_annotations(snapshot: syrupy.assertion.SnapshotAssertion) -> None:
    # Autoquant must ignore all annotations and leave them unchanged.
    input = """
    def my_func_with_annotations(self, x: int | float, y: float | int) -> str | bool:
        z: str | bool = y | x
        return z
    """
    result = _autoquantize_str(input, use_type_inference=True)
    assert snapshot == result


@pytest.mark.slow
def test_autoquant_skip_isolation_for_if_expr(snapshot: syrupy.assertion.SnapshotAssertion) -> None:
    # Autoquant must not isolate expressions in the `body` or `orelse` of an
    # if-expression since isolation would lead to unnecessary compute (i.e.,
    # both branches are evaluated).
    input = """
    def my_func_with_annotations(self, x: Tensor, y: Tensor, flag: bool) -> Tensor:
        out = x + y if flag else y - x
        return out
    """
    result = _autoquantize_str(input)
    assert snapshot == result


@pytest.mark.slow
@pytest.mark.parametrize(
    ("input", "use_type_inference"),
    [
        pytest.param(
            """
            def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
                y = x + 1
                z = y[idx] * 0 + y
                return z
            """,
            False,
            id="subscript_load_before_assignment",
        ),
        pytest.param(
            """
            def forward(self, image_flags: list[int], lengths: torch.Tensor):
                valid_mask = []
                for flag, length in zip(image_flags, lengths):
                    valid_mask.extend([flag] * length)
                return valid_mask
            """,
            False,
            id="loop_list_load_before_assignment",
        ),
        pytest.param(
            """
            def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
                selected_W = self.W[cat_ids]
                selected_b = self.b[cat_ids]
                return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)
            """,
            True,
            id="selected_b_unsqueeze_load_before_assignment_with_type_inference",
        ),
    ],
)
def test_autoquant_load_before_assignment(
    snapshot: syrupy.assertion.SnapshotAssertion,
    input: str,
    use_type_inference: bool,
) -> None:
    """Verifies autoquant can emit quantized expressions that read local variables before assignment."""
    result = _autoquantize_str(input, use_type_inference=use_type_inference)

    ast.parse(result)
    assert snapshot == result


@pytest.mark.slow
@pytest.mark.parametrize(
    "input",
    [
        pytest.param(
            """
            def forward(self, values: list[tuple[int, int]], b: torch.Tensor):
                hw_list = []
                for x, y in values:
                    hw_list.extend([(x, y)] * b + [(x, y)] * b)
                return hw_list
            """,
            id="loop_tuple_shared_quantized_value",
        ),
        pytest.param(
            """
            def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
                y = x + 1
                a = y[idx] * 2
                b = y[idx] * 3
                return a + b
            """,
            id="repeated_subscript_shared_quantized_value",
        ),
    ],
)
def test_autoquant_shared_quantized_value(
    snapshot: syrupy.assertion.SnapshotAssertion,
    input: str,
) -> None:
    result = _autoquantize_str(input)

    ast.parse(result)
    assert snapshot == result


@pytest.mark.slow
@pytest.mark.parametrize(
    "input",
    [
        pytest.param(
            """
            def forward(self, xs: list[torch.Tensor], ys: list[torch.Tensor]) -> list[torch.Tensor]:
                return [x + y for x, y in zip(xs, ys)]
            """,
            id="comprehension_fallback",
        ),
        pytest.param(
            """
            def forward(self, x: torch.Tensor, y: torch.Tensor, flag: bool):
                return x if flag and y.sum() > 0 else y
            """,
            id="boolean_and_fallback",
        ),
    ],
)
def test_autoquant_unsupported_contexts_fallback_inline(
    snapshot: syrupy.assertion.SnapshotAssertion, input: str
) -> None:
    # These expression contexts cannot always be safely rewritten via explicit
    # hoisted temporaries while preserving evaluation semantics.
    result = _autoquantize_str(input)
    ast.parse(result)
    assert snapshot == result
