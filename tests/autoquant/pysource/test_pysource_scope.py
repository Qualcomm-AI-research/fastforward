# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from fastforward._autoquant.pysource.scope import _resolve_relative_module_name


def test_resolve_relative_module_name() -> None:
    relative_root = "path.tocurrent.module"
    module_name = "other.submodule"

    resolved = _resolve_relative_module_name(relative_root, 0, module_name)
    assert resolved == "other.submodule"

    resolved = _resolve_relative_module_name(relative_root, 1, module_name)
    assert resolved == "path.tocurrent.other.submodule"

    resolved = _resolve_relative_module_name(relative_root, 2, module_name)
    assert resolved == "path.other.submodule"

    resolved = _resolve_relative_module_name(relative_root, 3, module_name)
    assert resolved == "other.submodule"
