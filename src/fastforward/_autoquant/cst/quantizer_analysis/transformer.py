# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


import libcst

from .annotator import QuantizationAnnotationProvider


class QuantizerFunctionTransformer(libcst.CSTTransformer):
    METADATA_DEPENDENCIES = (QuantizationAnnotationProvider,)
