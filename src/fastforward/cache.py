# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import os

from pathlib import Path


def get_assets_path(
    namespace: str,
    tag: str,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Generate (not create) a path for caching assets locally.

    Args:
        namespace (str): Namespace to which the data belongs to.
        tag (str): A tag is used as a subfolder of namespace.
        cache_dir (str | Path, optional): The directory where assets are cached.
            Defaults to the value of the `FF_CACHE` environment variable, then
            `XDG_CACHE_HOME/fastforward/` if XDG_CACHE_HOME is set, or finally
            `~/.cache/fastforward/` if neither variable is set.

    Returns:
        The local path where the asset is cached.

    Notes:
    - The asset path will be appended to the cache directory to form the final cache path.
    """
    if cache_dir is None:
        cache_dir = os.getenv("FF_CACHE", None)
    if cache_dir is None and (xdg_cache := os.getenv("XDG_CACHE_HOME", None)) is not None:
        cache_dir = Path(xdg_cache) / "fastforward"
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "fastforward"
    cache_dir = Path(cache_dir).expanduser().resolve()

    for part in (" ", "/", "\\", ":", "<", ">", "|"):
        namespace = namespace.replace(part, "--")
        tag = tag.replace(part, "--")

    path = cache_dir / namespace / tag
    if path.exists() and path.is_file():
        msg = f"The asset path '{path}' points to an existing file (not a directory)"
        raise ValueError(msg)
    return path
