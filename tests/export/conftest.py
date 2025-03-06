# Copyright (c) 2025 Qualcomm Technologies, Inc.
# All Rights Reserved.

import pytest
import torch


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Called after collection has been performed. May filter or re-order the items in-place."""
    for item in items:
        if "xfail_due_to_too_new_torch" in item.keywords:
            item.add_marker(
                pytest.mark.xfail(
                    torch.__version__ >= "2.6",
                    reason="torch_onnx doesn't work with torch >= 2.6 (issue #66)",
                )
            )
