# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear


try:
    from fastforward._autoquant.mypy.type_provider_impl import MypyTypeProvider as MypyTypeProvider
    from fastforward._autoquant.mypy.type_provider_impl import TypeInfo as TypeInfo
except (ImportError, ModuleNotFoundError):
    from fastforward._autoquant.mypy.type_provider_fallback import (  # type: ignore[assignment]
        MypyTypeProvider as MypyTypeProvider,
    )
    from fastforward._autoquant.mypy.type_provider_fallback import (  # type: ignore[assignment]
        TypeInfo as TypeInfo,
    )
