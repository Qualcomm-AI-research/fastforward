# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from pathlib import Path
from unittest.mock import patch

import pytest

from fastforward.cache import get_assets_path


def test_get_assets_path_default_cache_dir(tmp_home: Path) -> None:
    """Test get_assets_path with default cache directory."""
    # GIVEN: A namespace and tag
    namespace = "test_namespace"
    tag = "test_tag"

    # WHEN: Getting assets path without specifying cache directory
    cache_path = get_assets_path(namespace, tag)

    # THEN: Should use default cache directory under home
    assert cache_path == tmp_home / ".cache" / "fastforward" / namespace / tag


def test_get_assets_path_custom_cache_dir(tmp_path: Path) -> None:
    """Test get_assets_path with custom cache directory."""
    # GIVEN: A namespace, tag, and custom cache directory
    namespace = "test_namespace"
    tag = "test_tag"

    # WHEN: Getting assets path with custom cache directory
    cache_path = get_assets_path(namespace, tag, cache_dir=tmp_path)

    # THEN: Should use the custom cache directory
    assert cache_path == tmp_path / namespace / tag


def test_get_assets_path_ff_cache_environment_variable(tmp_path: Path) -> None:
    """Test get_assets_path with FF_CACHE environment variable."""
    # GIVEN: A namespace, tag, and FF_CACHE environment variable set
    namespace = "test_namespace"
    tag = "test_tag"

    # WHEN: Getting assets path with FF_CACHE environment variable
    with patch.dict("os.environ", {"FF_CACHE": str(tmp_path)}):
        cache_path = get_assets_path(namespace, tag)

    # THEN: Should use the FF_CACHE directory
    assert cache_path == tmp_path / namespace / tag


def test_get_assets_path_xdg_cache_home_environment_variable(tmp_path: Path) -> None:
    """Test get_assets_path with XDG_CACHE_HOME environment variable."""
    # GIVEN: A namespace, tag, and XDG_CACHE_HOME environment variable set
    namespace = "test_namespace"
    tag = "test_tag"
    xdg_cache_dir = tmp_path / "xdg_cache"

    # WHEN: Getting assets path with XDG_CACHE_HOME environment variable
    with patch.dict("os.environ", {"XDG_CACHE_HOME": str(xdg_cache_dir)}, clear=True):
        cache_path = get_assets_path(namespace, tag)

    # THEN: Should use the XDG_CACHE_HOME directory with fastforward subdirectory
    assert cache_path == xdg_cache_dir / "fastforward" / namespace / tag


def test_get_assets_path_ff_cache_priority_over_xdg(tmp_path: Path) -> None:
    """Test that FF_CACHE takes priority over XDG_CACHE_HOME."""
    # GIVEN: A namespace, tag, and both FF_CACHE and XDG_CACHE_HOME environment variables set
    namespace = "test_namespace"
    tag = "test_tag"
    ff_cache_dir = tmp_path / "ff_cache"
    xdg_cache_dir = tmp_path / "xdg_cache"

    # WHEN: Getting assets path with both environment variables set
    with patch.dict(
        "os.environ", {"FF_CACHE": str(ff_cache_dir), "XDG_CACHE_HOME": str(xdg_cache_dir)}
    ):
        cache_path = get_assets_path(namespace, tag)

    # THEN: Should prioritize FF_CACHE over XDG_CACHE_HOME
    assert cache_path == ff_cache_dir / namespace / tag


def test_get_assets_path_fallback_to_home_cache(tmp_home: Path) -> None:
    """Test fallback to ~/.cache/fastforward when no environment variables are set."""
    # GIVEN: A namespace and tag with no environment variables set
    namespace = "test_namespace"
    tag = "test_tag"

    # WHEN: Getting assets path without environment variables
    cache_path = get_assets_path(namespace, tag)

    # THEN: Should fallback to default home cache directory
    assert cache_path == tmp_home / ".cache" / "fastforward" / namespace / tag


@pytest.mark.parametrize("invalid_char", [" ", "/", "\\", ":", "<", ">", "|"])
def test_get_assets_path_invalid_namespace(invalid_char: str, tmp_path: Path) -> None:
    """Test get_assets_path with invalid characters."""
    # GIVEN: A namespace and tag containing invalid characters
    namespace = "test-namespace".replace("-", invalid_char)
    tag = "test-tag".replace("-", invalid_char)

    # WHEN: Getting assets path with invalid characters
    cache_path = get_assets_path(namespace, tag, cache_dir=tmp_path)

    # THEN: Should replace invalid characters with double dashes
    assert cache_path.name == "test--tag"
    assert cache_path.parent.name == "test--namespace"


def test_get_assets_path_file_exists(tmp_path: Path) -> None:
    """Test get_assets_path with existing file."""
    # GIVEN: A namespace, tag, and existing file at the expected path
    namespace = "test_namespace"
    tag = "test_tag"
    cache_dir = tmp_path / "test_cache"
    cache_path = cache_dir / namespace / tag
    cache_path.parent.mkdir(parents=True)
    cache_path.touch()

    # WHEN: Getting assets path where a file already exists at that location
    # THEN: Should raise ValueError
    with pytest.raises(ValueError):
        get_assets_path(namespace, tag, cache_dir=cache_dir)
