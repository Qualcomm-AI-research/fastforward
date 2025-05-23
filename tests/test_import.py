# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from collections import defaultdict

import pytest

from fastforward._import import QualifiedNameReference


def test_qualified_name_reference() -> None:
    ref = QualifiedNameReference("collections.defaultdict")
    assert defaultdict is ref.import_()

    ref = QualifiedNameReference("collections.defaultdict.copy")
    assert defaultdict.copy is ref.import_()

    ref = QualifiedNameReference("collections.defaultdict.does_not_exist")
    with pytest.raises(ImportError):
        ref.import_()

    ref = QualifiedNameReference("collections.does_not_exist")
    with pytest.raises(ImportError):
        ref.import_()

    ref = QualifiedNameReference("does_not_exist.attribute")
    with pytest.raises(ImportError):
        ref.import_()
