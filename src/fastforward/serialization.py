# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


def _has_custom_method(cls: type, method_name: str) -> bool:
    """Check if a class has a custom method implementation.

    Args:
        cls: The class to check.
        method_name: The name of the method to check (e.g., '__new__', '__init__').

    Returns:
        True if the class has a custom method implementation, False if it uses the default.
        If there is no such method, the AttributeError will be raised.
    """
    class_method = getattr(cls, method_name)
    super_method = getattr(super(cls, cls), method_name)
    return class_method is not super_method
