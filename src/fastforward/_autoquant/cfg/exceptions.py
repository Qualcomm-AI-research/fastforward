# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


class CFGError(Exception):
    """General CFG related Exception."""


class CFGConstructionError(CFGError):
    """CFG Construction Exception."""


class CFGReconstructionError(CFGError):
    """CFG Reconstruction Exception."""
