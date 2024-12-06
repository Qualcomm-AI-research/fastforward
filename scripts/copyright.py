#! /usr/bin/env python


# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse
import os
import pathlib

from typing import Callable, Iterator, Optional

HEADER = "# Copyright (c) 2024 Qualcomm Technologies, Inc.\n# All Rights Reserved."


class HeaderViolation(Exception):
    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=pathlib.Path, default=os.getcwd())
    parser.add_argument("--header", type=str, default=HEADER)
    args = parser.parse_args()
    run(args.src, args.header)


def run(root: pathlib.Path, header: str) -> None:
    """
    Add headers to all python files in root and subdirectories.

    Reports on:
    - Files that where changed
    - Files that already included the header and where not changed
    - Files that includee the header but voiolate the 'top-of-file' placement.
    """
    violations: list[pathlib.Path] = []
    for file in _src_files(root, filter=_py_filter):
        try:
            process_file(file, header=header)
        except HeaderViolation:
            violations.append(file)

    for violation in violations:
        print(f"⛔ {violation}: includes incorrectly placed copyright header")


def process_file(filepath: pathlib.Path, header: str) -> None:
    """
    Add header to file at filepath if it doesn't contain it yet

    The header may only be preceded by shebang and empty lines,
    if not a HeaderViolation is raised.
    """
    with filepath.open() as f:
        contents = f.read()
    if _has_header(contents, header):
        if not _check_header(contents, header):
            raise HeaderViolation(filepath)
        print(f"✅ {filepath}: already included a copyright header")
        return

    with filepath.open("w") as f:
        print(f"⏳ {filepath}: adding copyright header")
        head, tail = _split_for_insertion(contents)
        f.write(head)
        if len(head) > 0:
            f.write("\n")
        f.write(header)
        f.write("\n\n")
        f.write(tail)


def _has_header(contents: str, header: str) -> bool:
    return header in contents


def _check_header(contents: str, header: str) -> bool:
    _, tail = _split_for_insertion(contents)
    header_lines = header.split("\n")
    tail_lines = tail.split("\n")
    if len(header_lines) > len(tail_lines):
        return False

    for l1, l2 in zip(header_lines, tail_lines):
        if l1 != l2:
            return False

    return True


def _split_for_insertion(contents: str) -> tuple[str, str]:
    head: list[str] = []

    lines = contents.split("\n")
    i = 0
    for i, line in enumerate(lines):
        if not (line.startswith("#!") or len(line.strip()) == 0):
            break
        head.append(line)
    tail = lines[i:]
    head_str = "\n".join(head)
    head_str = "" if len(head_str.strip()) == 0 else head_str
    return head_str, "\n".join(tail)


def _src_files(
    root: pathlib.Path, filter: Optional[Callable[[pathlib.Path], bool]] = None
) -> Iterator[pathlib.Path]:
    filter = filter or (lambda _: True)
    for dirpath, _, files in os.walk(root):
        path = pathlib.Path(dirpath)
        for file in files:
            filepath = path / file
            if filter(filepath):
                yield filepath


def _py_filter(filepath: pathlib.Path) -> bool:
    return filepath.suffix == ".py"


if __name__ == "__main__":
    main()
