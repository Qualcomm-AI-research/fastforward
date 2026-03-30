# FastForward Autoquant Tutorial

`ff.autoquantize` (Autoquant) automatically rewrites a PyTorch module into a
*quantized* counterpart by analysing the module's `forward` method and any
helper functions it calls. The generated code inserts `QuantizerStub`
placeholders at every point where a tensor value is produced or consumed, so
you can later attach real quantizers and calibrate them without touching the
generated file by hand.

This tutorial walks through the most important features of Autoquant with
self-contained examples.

---

## Basic Usage

The simplest way to use Autoquant is to call `ff.autoquantize` on any
`torch.nn.Module`. Because FastForward already knows how to quantize the
standard `torch.nn.Linear` layer, the example below subclasses it and
overrides `forward` so that Autoquant has something to analyse.

```python
import torch
import torch.nn.functional as F
import fastforward as ff


# Creating a custom linear layer because
# autoquant ignores known modules such
# as torch.nn.Linear
class MyLinear(torch.nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)


# Initialize model and call autoquant
model = MyLinear(10, 10)
ff.autoquantize(
    model,
    output_path="quantized.py",  # Write the output to quantized.py
    use_type_inference=False,  # Do not use type inference
    force_overwrite=True,
)
```

Autoquant generates a new class that inherits from both
`fastforward.nn.QuantizedModule` and the original module. The
`__init_quantization__` method is where all `QuantizerStub` instances are
created - one for each tensor-valued input, weight, bias, and output. In
`forward`, every tensor is passed through its corresponding stub before being
used in the computation.

```python
import fastforward
import torch
from __main__ import MyLinear


class QuantizedMyLinear(fastforward.nn.QuantizedModule, MyLinear):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_linear: ... = fastforward.nn.QuantizerStub()
        self.quantizer_self_bias: ... = fastforward.nn.QuantizerStub()
        self.quantizer_self_weight: ... = fastforward.nn.QuantizerStub()
        self.quantizer_input: ... = fastforward.nn.QuantizerStub()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self.quantizer_input(input)
        self_bias = self.quantizer_self_bias(self.bias)
        self_weight = self.quantizer_self_weight(self.weight)
        return fastforward.nn.functional.linear(
            input, self_weight, self_bias, output_quantizer=self.quantizer_linear
        )
```

---

## Helper functions

Modules often delegate work to standalone helper functions. Autoquant
follows those call chains and generates a quantized version of every helper
it encounters, threading the required quantizer arguments through each call
site automatically.

```python
import torch
import fastforward as ff


# helper function that uses a helper
# itself
def helper(x, y):
    return inner_helper(x, y) + inner_helper(y, x)


def inner_helper(x, y):
    return x - y


class MyModuleWithHelper(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return helper(x, y)


model = MyModuleWithHelper()
ff.autoquantize(model, output_path="quantized.py", use_type_inference=False, force_overwrite=True)
```

Each helper gets its own quantized variant. Notice that `quantized_helper`
receives *all* quantizers for both of its `quantized_inner_helper` calls as
keyword-only arguments. Autoquant names them to avoid collisions (e.g.
`quantizer_inner_helper_sub_1` vs `quantizer_inner_helper_sub_2`).

```python
def quantized_helper(
    x, y, *,
    quantizer_add: fastforward.nn.Quantizer,
    quantizer__tmp_1: fastforward.nn.Quantizer,
    quantizer__tmp_2: fastforward.nn.Quantizer,
    [...]
    quantizer_inner_helper_y_2: fastforward.nn.Quantizer,
):
    _tmp_1 = quantized_inner_helper(x, y,
        quantizer_sub=quantizer_inner_helper_sub_1,
        quantizer_x=quantizer_inner_helper_x_1,
        quantizer_y=quantizer_inner_helper_y_1,
    )
    _tmp_1 = quantizer__tmp_1(_tmp_1)
    _tmp_2 = quantized_inner_helper(y, x,
        quantizer_sub=quantizer_inner_helper_sub_2,
        quantizer_x=quantizer_inner_helper_x_2,
        quantizer_y=quantizer_inner_helper_y_2,
    )
    _tmp_2 = quantizer__tmp_2(_tmp_2)
    return fastforward.nn.functional.add(_tmp_1, _tmp_2,
        output_quantizer=quantizer_add
    )

def quantized_inner_helper(
    x,
    y,
    *,
    quantizer_sub: ...,
    quantizer_x: ...,
    quantizer_y: ...,
):
    x = quantizer_x(x)
    y = quantizer_y(y)
    return fastforward.nn.functional.sub(x, y, output_quantizer=quantizer_sub)
```

## Annotations

Type annotations on helper functions give Autoquant the information it needs
to decide which arguments are tensors (and therefore need quantizers) and
which are plain Python values (and can be left alone). Without annotations,
Autoquant conservatively treats every argument as a potential tensor.

```python
import torch
import torch.nn.functional as F
import fastforward as ff


# helper function called from module
def helper(x, y):
    y = y + y
    for y_ in y:
        x = x + y_
    return x


class MyModuleWithHelper(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return helper(x, [2])


model = MyModuleWithHelper()
ff.autoquantize(model, output_path="quantized.py", use_type_inference=False, force_overwrite=True)
```

Without annotations and type inference enabled, Autoquant will assume
everything can be a tensor. This leads to unnecessary quantizers being
inserted for the `list[int]` argument `y` and the loop variable `y_`.

```python
def quantized_helper(
    x: torch.Tensor,
    y: list[int],
    *,
    quantizer_add_1: fastforward.nn.Quantizer,
    quantizer_add_2: fastforward.nn.Quantizer,
    quantizer_y_: fastforward.nn.Quantizer,
    quantizer_x: fastforward.nn.Quantizer,
    quantizer_y: fastforward.nn.Quantizer,
):
    x = quantizer_x(x)
    y = quantizer_y(y)
    y = fastforward.nn.functional.add(y, y, output_quantizer=quantizer_add_1)
    for y_ in y:
        y_ = quantizer_y_(y_)
        x = fastforward.nn.functional.add(x, y_, output_quantizer=quantizer_add_2)
    return x
```

Adding a type annotation `y: list[int]` tells Autoquant that `y` is not a
tensor. Enabling `use_type_inference=True` lets Autoquant propagate that
information through the function body.

```python
import torch
import torch.nn.functional as F
import fastforward as ff


# helper function called from module
def helper(x: torch.Tensor, y: list[int]):
    y = y + y
    for y_ in y:
        x = x + y_
    return x


class MyModuleWithHelper(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return helper(x, [2])


model = MyModuleWithHelper()
ff.autoquantize(model, output_path="quantized.py", use_type_inference=True, force_overwrite=True)
```

With the annotation in place, Autoquant correctly infers that `y` and `y_`
are integers. The redundant quantizers for `y` and `y_` disappear, and the
`y + y` list concatenation is left as plain Python. Annotations can
dramatically improve the result of Autoquant.

```python
def quantized_helper(
    x: torch.Tensor,
    y: list[int],
    *,
    quantizer_add: fastforward.nn.Quantizer,
    quantizer_y_: fastforward.nn.Quantizer,
    quantizer_x: fastforward.nn.Quantizer,
):
    x = quantizer_x(x)
    y = y + y
    for y_ in y:
        y_ = quantizer_y_(y_)
        x = fastforward.nn.functional.add(x, y_, output_quantizer=quantizer_add)
    return x
```

## Operator Table

Autoquant relies on an internal *operator table* to know which PyTorch
operators have quantized equivalents in FastForward. When an operator is
missing from the table, Autoquant cannot quantize it and falls back to the
original call.

```python
import torch
import torch.nn.functional as F
import fastforward as ff


# This example seems similar to
# the first Linear example
class MyBilinear(torch.nn.Bilinear):
    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        return F.bilinear(input1, input2, self.weight, self.bias)


model = MyBilinear(10, 10, 10)
ff.autoquantize(model, output_path="quantized.py", use_type_inference=False, force_overwrite=True)
```

Because `F.bilinear` is not yet in FastForward's operator table, Autoquant
cannot produce a quantized version of the `forward` method. The generated
class is essentially a no-op wrapper that calls the original implementation
unchanged.

```python
import fastforward
import torch
import torch.nn.functional as F

from __main__ import MyBilinear


class QuantizedMyBilinear(fastforward.nn.QuantizedModule, MyBilinear):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        return F.bilinear(input1, input2, self.weight, self.bias)
```

You can extend the operator table at runtime by creating a custom
`OperatorTable`, defining the quantization schema for the missing operator,
and passing the table to `ff.autoquantize`. The schema string uses a
mini-language that describes which arguments are `Quantized`,
`MaybeQuantized`, or plain Python values.

```python
import fastforward as ff
from fastforward.autoquant import OperatorTable, optable


def my_quantized_bilinear(
    input1: torch.Tensor,
    input2: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor: ...


custom_optable = OperatorTable.from_yaml(alias_extensions=optable.STR_ALIASES_EXTENSIONS)
custom_schema = (
    "bilinear(input1: Quantized, input2: Quantized, "
    "weight: Quantized, bias: Optional[MaybeQuantized] = None) -> Quantized"
)
custom_optable.add(custom_schema, torch.nn.functional.bilinear, dispatch_op=my_quantized_bilinear)

ff.autoquantize(
    model,
    output_path="quantized.py",
    use_type_inference=False,
    force_overwrite=True,
    operator_table=custom_optable,
)
```

With the custom operator registered, Autoquant can now fully quantize
`MyBilinear`. The generated `forward` method quantizes all inputs and
weights before dispatching to `my_quantized_bilinear`.

```python
import fastforward
import torch
from __main__ import MyBilinear


class QuantizedMyBilinear(fastforward.nn.QuantizedModule, MyBilinear):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()
        self.quantizer_bilinear: ... = fastforward.nn.QuantizerStub()
        [...]
        self.quantizer_input2: ... = fastforward.nn.QuantizerStub()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        input1 = self.quantizer_input1(input1)
        input2 = self.quantizer_input2(input2)
        self_bias = self.quantizer_self_bias(self.bias)
        self_weight = self.quantizer_self_weight(self.weight)
        return __main__.my_quantized_bilinear(
            input1, input2, self_weight, self_bias, output_quantizer=self.quantizer_bilinear
        )
```

## Replacement Patterns

Sometimes you want Autoquant to route calls through your own operator
implementations rather than FastForward's built-ins. `PatternRule` lets you
express a simple string-template rewrite: any call matching the *pattern* is
rewritten to the *replacement* before the rest of Autoquant runs.


```python
import syrupy
import fastforward as ff
import fastforward.autoquant as autoquant
from tests.autoquant.helpers import autoquant_with_defaults, codeformat_with_defaults


class ExampleModule1B(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.conv2d(x, x)
        return torch.nn.functional.linear(y, y)


def test_pattern_based_replacement(snapshot: syrupy.assertion.SnapshotAssertion) -> None:
    module = ExampleModule1B()
    rule = ff.autoquant.PatternRule.from_str(
        pattern="torch.nn.functional.{func}({a}, {b})",
        replacement="my_own.{func}({a}, {b})",
    )
    quantized = autoquant_with_defaults(
        module, use_type_inference=False, replacement_patterns=[rule]
    )
    quantized = codeformat_with_defaults(quantized)
    assert snapshot == quantized
```

The pattern `torch.nn.functional.{func}({a}, {b})` matches both `conv2d` and
`linear` calls. Autoquant rewrites them to `my_own.conv2d` and
`my_own.linear` respectively before generating the quantized module. Note
that because the replacement targets are unknown to Autoquant's operator
table, no quantizer stubs are inserted — the rewritten calls are emitted
verbatim.

```python
import fastforward
import torch
from tests.autoquant.test_autoquant import ExampleModule1B


class QuantizedExampleModule1B(fastforward.nn.QuantizedModule, ExampleModule1B):
    def __init_quantization__(self) -> None:
        super().__init_quantization__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = my_own.conv2d(x, x)
        return my_own.linear(y, y)
```

## Conclusion

Autoquant removes the bulk of the manual work involved in writing quantized
PyTorch modules. The key ideas to take away are:

- **Subclass, don't replace.** The generated class inherits from both
  `fastforward.nn.QuantizedModule` and the original module, so it can be
  dropped in wherever the original was used.

- **Helpers are handled automatically.** Any standalone function called from
  `forward` is traced and rewritten alongside the module. Quantizer
  arguments are threaded through the call chain for you.

- **Annotations pay off.** Adding type annotations to helper functions —
  especially for non-tensor arguments — significantly reduces the number of
  unnecessary quantizers in the output. Pair annotations with
  `use_type_inference=True` for the best results.

- **Extend the operator table when needed.** If Autoquant silently skips an
  operator, it is almost certainly missing from FastForward's operator table.
  You can register a custom schema and dispatch function at runtime without
  waiting for an upstream change.

- **Replacement patterns for custom dispatch.** `PatternRule` lets you
  redirect whole families of calls to your own implementations before
  Autoquant runs, giving you fine-grained control over the generated code.

With these tools combined, Autoquant can handle everything from simple linear
layers to complex modules with loops, conditionals, and third-party operators.
