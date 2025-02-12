# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


class CFGError(Exception):
    """General CFG related Exception."""


class CFGConstructionError(CFGError):
    """CFG Construction Exception."""


class CFGReconstructionError(CFGError):
    """CFG Reconstruction Exception."""
