# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Find the local installation directory of a Python package.

Given a model_id such as "facebook/sam3", derives a candidate package name
("sam3"), then tries in order:

  1. pip show <pkg>          — works for pip-installed packages
  2. importlib.util.find_spec — covers editable installs and sys.path entries
  3. PYTHONPATH scan          — checks each directory in PYTHONPATH directly

Prints a JSON object:
  {
    "package_name": "sam3",
    "package_path": "/path/to/sam3",   # null if not found
    "source": "pip" | "importlib" | "pythonpath" | null,
    "error": null | "<message>"
  }

Exit codes:
  0  package found
  1  package not found or unexpected error
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys

from pathlib import Path


def _package_name_from_model_id(model_id: str) -> str:
    """Derive a candidate package name from a HuggingFace model ID.

    "facebook/sam3" -> "sam3"
    "bert-base-uncased" -> "bert-base-uncased"
    """
    return model_id.split("/")[-1]


def _try_pip(package_name: str) -> str | None:
    """Return the package Location from pip show, or None."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            if line.startswith("Location:"):
                location = line.split(":", 1)[1].strip()
                candidate = Path(location) / package_name
                if candidate.exists():
                    return str(candidate)
                # editable installs may put the package directly at location
                if Path(location).exists():
                    return str(location)
    except Exception:  # noqa: BLE001
        pass
    return None


def _try_importlib(package_name: str) -> str | None:
    """Return the package root directory via importlib, or None."""
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            return None
        if spec.submodule_search_locations:
            locs = list(spec.submodule_search_locations)
            if locs:
                return str(Path(locs[0]).resolve())
        if spec.origin:
            return str(Path(spec.origin).parent.resolve())
    except Exception:  # noqa: BLE001
        pass
    return None


def _try_pythonpath(package_name: str) -> str | None:
    """Scan each PYTHONPATH entry for a directory named package_name."""
    pythonpath = os.environ.get("PYTHONPATH", "")
    for entry in pythonpath.split(os.pathsep):
        if not entry:
            continue
        candidate = Path(entry) / package_name
        if candidate.is_dir():
            return str(candidate.resolve())
    return None


def find_local_package(model_id: str) -> dict:
    """Locate the installed package directory for a given HuggingFace model ID."""
    package_name = _package_name_from_model_id(model_id)

    path = _try_pip(package_name)
    if path:
        return {"package_name": package_name, "package_path": path, "source": "pip", "error": None}

    path = _try_importlib(package_name)
    if path:
        return {
            "package_name": package_name,
            "package_path": path,
            "source": "importlib",
            "error": None,
        }

    path = _try_pythonpath(package_name)
    if path:
        return {
            "package_name": package_name,
            "package_path": path,
            "source": "pythonpath",
            "error": None,
        }

    return {
        "package_name": package_name,
        "package_path": None,
        "source": None,
        "error": (
            f"Could not locate package '{package_name}' via pip, importlib, or PYTHONPATH. "
            "Please provide the path manually via the model_path field."
        ),
    }


def main() -> None:
    """CLI entry point for locating a package directory by model ID."""
    parser = argparse.ArgumentParser(
        description="Find the local installation directory of a Python package"
    )
    parser.add_argument("model_id", help="HuggingFace model ID, e.g. facebook/sam3")
    args = parser.parse_args()

    result = find_local_package(args.model_id)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["package_path"] is not None else 1)


if __name__ == "__main__":
    main()
