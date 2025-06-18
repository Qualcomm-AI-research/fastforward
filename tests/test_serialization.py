# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


from fastforward.serialization import _has_custom_method


def test_has_custom_method() -> None:
    """Test detection of default methods."""

    # GIVEN: A class with default `__new__` and custom __init__ methods
    class DefaultClass:
        def __init__(self) -> None:
            pass

    # WHEN: Checking if the class has custom methods
    # THEN: Should return False for default methods
    assert not _has_custom_method(DefaultClass, "__new__")
    assert _has_custom_method(DefaultClass, "__init__")
