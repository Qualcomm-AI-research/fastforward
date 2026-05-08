# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for scalar arithmetic type propagation in autoquant."""

import inspect
import io

from typing import Any, Generator
from unittest import mock

import fastforward as ff
import libcst
import pytest
import syrupy
import torch
import torch.nn.functional as F

from fastforward._autoquant.pybuilder import TextIOWriter


def _generated_forward(generated_code: str) -> str:
    """Extract the forward method source from autoquant-generated code.

    The full generated output contains helper functions (e.g.
    ``quantized__in_projection_packed``) that embed ``_tmp_NNN`` variable names
    and verbatim comments from called library functions — both torch-version
    sensitive.  The ``forward`` method of the generated ``Quantized*`` class
    contains only the user's own code flow and is stable across torch versions.
    """
    cst = libcst.parse_module(generated_code)
    for node in cst.body:
        if isinstance(node, libcst.ClassDef):
            for item in node.body.body:
                if isinstance(item, libcst.FunctionDef) and item.name.value == "forward":
                    return cst.code_for_node(item)
    return ""


# ---------------------------------------------------------------------------
# Fixture: execution context
# ---------------------------------------------------------------------------


def _strip_relative_imports(code: str) -> str:
    """Replace relative import lines with blanks so mypy can parse standalone.

    Older torch versions (e.g. 2.4) use relative imports inside
    torch.nn.functional (``from .._jit_internal import ...``).  When that
    source is prepended as a binder-corruption preamble, mypy rejects it with
    "No parent module -- cannot perform relative import".  Stripping those
    lines (replaced with blank lines to preserve line numbers) allows mypy to
    parse the rest of the preamble and still trigger the binder corruption we
    need for the test.
    """
    result = []
    in_relative = False
    paren_depth = 0
    for line in code.splitlines(keepends=True):
        stripped = line.lstrip()
        if not in_relative and stripped.startswith(("from .", "from ..")):
            in_relative = True
            paren_depth = line.count("(") - line.count(")")
            result.append("\n")
            if paren_depth <= 0:
                in_relative = False
        elif in_relative:
            paren_depth += line.count("(") - line.count(")")
            result.append("\n")
            if paren_depth <= 0:
                in_relative = False
        else:
            result.append(line)
    return "".join(result)


@pytest.fixture(
    params=[
        pytest.param("standalone", id="standalone"),
        pytest.param("with_nn_functional", id="with_nn_functional"),
    ]
)
def _autoquant_context(request: pytest.FixtureRequest) -> Generator[str, None, None]:
    """Execution context for autoquant type-inference tests.

    standalone
        Preamble is ``import torch``. mypy binder is fresh; type annotations
        resolve cleanly. Autoquant works correctly with or without the fix.

    with_nn_functional
        Preamble is the full torch.nn.functional source (~7000 lines).
        inspect.getsource is patched so autoquant's internal pipeline sees
        ``preamble + real_class_source``. This triggers the same 16 SystemExit
        binder corruptions that caused spurious quantization of scalar math.
        With the fix, autoquant output must be identical to standalone.
    """
    preamble = (
        "import torch\n"
        if request.param == "standalone"
        else inspect.getsource(F) + "\nimport torch\n"
    )
    _real = inspect.getsource

    def _patched(obj: Any) -> str:
        return _strip_relative_imports(preamble + _real(obj))

    with mock.patch(
        "fastforward._autoquant.pysource.source_context.inspect.getsource",
        side_effect=_patched,
    ):
        yield preamble


# ---------------------------------------------------------------------------
# Test-case data: torch.nn.Module subclasses
# ---------------------------------------------------------------------------

# --- typed-variable cases ---


class _NameExprIntModule(torch.nn.Module):
    def forward(self, q: torch.Tensor) -> int:
        E: int = q.size(-1)
        x = E * 2
        _y = E + 1
        return x


class _TypedLocalsModule(torch.nn.Module):
    def forward(self, q: torch.Tensor) -> torch.Tensor:
        num_heads: int = 8
        t = q + q
        _r = num_heads * 2
        return t


class _NumHeadsAndScaleModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads: int = 8
        self.scale: float = 0.125

    def forward(self, q: torch.Tensor, _k: torch.Tensor) -> torch.Tensor:
        _r = self.num_heads % 2
        s = self.scale * 2.0
        return q * s


class _NumHeadsOnlyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads: int = 8

    def forward(self, q: torch.Tensor, _k: torch.Tensor) -> torch.Tensor:
        _r = self.num_heads % 2
        return q


# --- unary literal negation regression ---


class _UnaryNegLiteralModule(torch.nn.Module):
    def forward(self, q: torch.Tensor) -> torch.Tensor:
        # -1 and -2 as dimension indices: the is_simple_literal guard must
        # prevent these unary negations from being wrapped in quantizer calls.
        a = q + q
        return a.unsqueeze(-1)


# --- shape/size unpack cases ---


class _ShapeTupleUnpackModule(torch.nn.Module):
    def forward(self, t: torch.Tensor) -> int:
        x, y, z = t.shape
        w = x * y + z
        return w


class _SizeNoArgsUnpackModule(torch.nn.Module):
    def forward(self, t: torch.Tensor) -> int:
        a, b = t.size()
        w = a * b
        return w


class _SizeDimScalarModule(torch.nn.Module):
    def forward(self, t: torch.Tensor) -> int:
        n = t.size(0)
        w = n * 2
        return w


class _TorchSizeVariableModule(torch.nn.Module):
    def forward(self, t: torch.Tensor) -> int:
        s = t.shape
        x, y = s
        w = x + y
        return w


# --- real-world: wraps F._in_projection_packed ---


class _InProjectionPackedModule(torch.nn.Module):
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.in_proj_weight = torch.nn.Parameter(torch.zeros(3 * embed_dim, embed_dim))
        self.in_proj_bias = torch.nn.Parameter(torch.zeros(3 * embed_dim))

    def forward(self, query: torch.Tensor) -> list[torch.Tensor]:
        return F._in_projection_packed(  # type: ignore[attr-defined, no-any-return]
            query,
            query,
            query,
            self.in_proj_weight,
            self.in_proj_bias,
        )


_AUTOQUANT_TEST_CASES = [
    pytest.param(_NameExprIntModule(), id="nameexpr_int_annotated"),
    pytest.param(_TypedLocalsModule(), id="typed_local_int"),
    pytest.param(_NumHeadsAndScaleModule(), id="num_heads_and_scale"),
    pytest.param(_NumHeadsOnlyModule(), id="num_heads_only"),
    pytest.param(_UnaryNegLiteralModule(), id="unary_neg_literal"),
    pytest.param(_ShapeTupleUnpackModule(), id="shape_tuple_unpack"),
    pytest.param(_SizeNoArgsUnpackModule(), id="size_no_args_unpack"),
    pytest.param(_SizeDimScalarModule(), id="size_dim_scalar"),
    pytest.param(_TorchSizeVariableModule(), id="torch_size_variable"),
    pytest.param(_InProjectionPackedModule(), id="in_projection_packed"),
]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("module", _AUTOQUANT_TEST_CASES)
def test_autoquant_snapshot(
    _autoquant_context: str,
    module: torch.nn.Module,
    snapshot: syrupy.assertion.SnapshotAssertion,
) -> None:
    """Snapshot the ff.autoquantize output for each typed-arithmetic module.

    autoquant_context patches inspect.getsource so both standalone and
    with_nn_functional runs receive correctly typed autoquantized output —
    proving the fix holds under binder corruption.
    """
    buffer = io.StringIO()
    ff.autoquantize(
        module,
        code_writer=TextIOWriter(type(module).__name__, writer=buffer),
        auto_import=False,
        use_type_inference=True,
    )
    assert snapshot == _generated_forward(buffer.getvalue())
