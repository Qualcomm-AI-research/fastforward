# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Check Conventional Commit titles in a Git revision range using only the standard library.

For exact specification, see: https://www.conventionalcommits.org/en/v1.0.0/.
"""

import argparse
import re
import subprocess
import sys

_ALLOWED_TYPES = frozenset({
    "feat",
    "fix",
    "docs",
    "style",
    "refactor",
    "test",
    "perf",
    "ci",
    "build",
    "chore",
})

# A conventional commit title has a type, optional scope and breaking-change marker, and a description.
_TITLE_PATTERN = re.compile(
    r"""
    (?P<type>[a-z]+)                   # Commit type (should be one of _ALLOWED_TYPES)
    (?: \( (?P<scope>[^()\r\n]+) \) )? # Optional scope
    (?P<breaking>!)?                   # Optional breaking-change marker
    : [ \t]+                           # Colon and whitespace
    (?P<description>\S[^\r\n]*)        # Nonempty description
    """,
    re.VERBOSE,
)


def title_error(title: str) -> str | None:
    """Return why the title is not a valid Conventional Commit title, or None if it is."""
    match = _TITLE_PATTERN.fullmatch(title)
    if match is None:
        return "expected 'type: description' or 'type(scope): description'"

    if match["type"] not in _ALLOWED_TYPES:
        return f"type {match['type']!r} is not one of {', '.join(sorted(_ALLOWED_TYPES))}"

    return None


def main(argv: list[str] | None = None) -> int:
    """Check every commit in a revision range and report all invalid titles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision_range", help="Git revision range, e.g. origin/main..HEAD")
    args = parser.parse_args(argv)

    # Specify a format that we can easily split on later:
    # Abbreviate commit hash (%h), then the message (%B) separated by null chars (%x00).
    result = subprocess.run(
        ["git", "log", "--format=%h%x00%B%x00", args.revision_range, "--"],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        return result.returncode

    # Splitting on the null char creates [hash_c1, msg_c1, hash_c2, msg_c2, ...].
    fields = result.stdout.split("\0")
    invalid = 0
    for sha, message in zip(fields[::2], fields[1::2]):
        title = message.partition("\n")[0].removesuffix("\r")
        if (error := title_error(title)) is not None:
            sys.stderr.write(f"{sha.strip()}: {error}: {title!r}\n")
            invalid += 1

    if invalid:
        sys.stderr.write(f"{invalid} commit(s) failed.\n")

    return int(invalid > 0)


if __name__ == "__main__":
    sys.exit(main())
