# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Autoquant is a feature of FastForward that simplifies the
process of integrating new models.

!!! experimental
    Please be aware that autoquant is an experimental feature. Use it with caution and
    expect changes as we continue the development of the feature.

    We encourage you to report any issues or feature requests.

Autoquant takes a PyTorch module (`torch.nn.Module`) instance as input and
generates Python code that implements quantized modules for the input module
and all of its submodules. The newly generated modules can either be used
directly or saved as a file -- after which they can be modified and integrated
into existing projects.

## Example
The following example will generate subclasses of `QuantizedModule`
needed<sup>1</sup> to quantize `my_model` and write them to
`generated_file.py`. Note that `fastforward.autoquantize` does not change `my_model`.

```python
my_model = get_model() # instantiate a PyTorch module
fastforward.autoquantize(my_model, output_path="generated_file.py")
```

In order to start using the newly generated `QuantizedModule`s we still
need to quantize `my_model` using `fastforward.quantize_model`:

```python
my_model = get_model() # instantiate a PyTorch module
fastforward.autoquantize(my_model, output_path="generated_file.py", auto_import=True)
fastforward.quantize_model(my_model)
```

<small>1: Autoquant is an experimental feature under active development. This means
 not all PyTorch modules are currently supported. Please reach out to us if
a particular module is important to you.</small>

---

"""  # noqa: D205, D212

import dataclasses
import pathlib
import types

from typing import overload

import torch

from fastforward._autoquant import pybuilder
from fastforward._autoquant.autoquant import (
    autoquant_with_defaults,
    codeformat_with_defaults,
    emit_code_of_module,
)
from fastforward._autoquant.pybuilder.importing import import_code
from fastforward._quantops import optable


@overload
def autoquantize(
    module: torch.nn.Module,
    *,
    operator_table: optable.OperatorTable | None = None,
    code_formatter: pybuilder.CodeFormatter | None = None,
    output_path: pathlib.Path | str | None = None,
    force_overwrite: bool = False,
    auto_import: bool = False,
    use_type_inference: bool = True,
) -> "AutoQuantizedCode": ...


@overload
def autoquantize(
    module: torch.nn.Module,
    *,
    operator_table: optable.OperatorTable | None = None,
    code_formatter: pybuilder.CodeFormatter | None = None,
    code_writer: pybuilder.BasicCodeWriter | None = None,
    auto_import: bool = False,
    use_type_inference: bool = True,
) -> "AutoQuantizedCode": ...


def autoquantize(
    module: torch.nn.Module,
    *,
    operator_table: optable.OperatorTable | None = None,
    code_formatter: pybuilder.CodeFormatter | None = None,
    output_path: pathlib.Path | str | None = None,
    force_overwrite: bool = False,
    code_writer: pybuilder.BasicCodeWriter | None = None,
    auto_import: bool = False,
    use_type_inference: bool = True,
) -> "AutoQuantizedCode":
    """Create Python source code for quantized version of `module`.

    Args:
        module: The module to quantize.
        operator_table: The operator table that defines the non-quantized to
            quantized operator mapping.
        code_formatter: The code formatter to use for formatting the generated
            code. If not provided, the default formatter is used.
        output_path: The path to write the generated code to. Mutually exclusive with code_writer.
        force_overwrite: If True, overwrite the output file if it already exists.
        code_writer: The code writer to use for writing the generated code. Mutually exclusive with
            output_path.
        auto_import: If True, automatically import the written module.
        use_type_inference: If True, use type inference to reduce false
            positive function rewrites in autoquant at the cost of a longer
            runtime.
    """
    autoquant_code = autoquant_with_defaults(
        module, operator_table, use_type_inference=use_type_inference
    )
    formatted_code = codeformat_with_defaults(autoquant_code, code_formatter=code_formatter)
    pymodule_name = emit_code_of_module(
        formatted_code,
        output_path=output_path,
        force_overwrite=force_overwrite,
        code_writer=code_writer,
    )

    if auto_import:
        pymodule = import_code(code=formatted_code, pymodule_name=pymodule_name)
    else:
        pymodule = None
    return AutoQuantizedCode(
        code=formatted_code,
        pymodule=pymodule,
        pymodule_name=pymodule_name,
    )


@dataclasses.dataclass
class AutoQuantizedCode:
    """Contains the generated code and the corresponding Python module."""

    code: str
    pymodule: types.ModuleType | None
    pymodule_name: str
