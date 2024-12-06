# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.


class CFGError(Exception):
    pass


class CFGConstructionError(CFGError):
    pass


class CFGReconstructionError(CFGError):
    pass
