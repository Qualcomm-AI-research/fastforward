# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Functionalities related to import of generated code."""

import sys
import types

from types import ModuleType

from fastforward._autoquant.pybuilder.writer import logger


def import_code(code: str, pymodule_name: str) -> ModuleType:
    """Imports `code` dynamically and makes it importable from `module_name`."""
    mod = types.ModuleType(pymodule_name)
    exec(code, mod.__dict__)
    if pymodule_name in sys.modules:
        logger.warning(
            f"Module `{pymodule_name}` already existed in `sys.modules`,"
            " overwriting existing module with new definition."
        )
    sys.modules[pymodule_name] = mod
    return mod
