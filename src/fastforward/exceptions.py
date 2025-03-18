# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


class QuantizationError(RuntimeError):
    """General quantization error."""


class ExportError(RuntimeError):
    """Export related error."""
