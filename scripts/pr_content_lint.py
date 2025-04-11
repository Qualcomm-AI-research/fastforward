# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Lints our PR descriptions.

To pass the linting:

- No checkboxes from the PR template may be removed.
- All checkboxes must be either checked or waived.

A checkbox is waived if it has the string `(waive)` next to it, e.g.

- [ ] (waive) This is a an example checkbox.

Note: Additional checkboxes can be added to the PR and will be linted, too.
"""

import re
import sys

_CHECKBOX_MARKER = r"^\s*- \[[ xX]\]\s?(\(waive\))?"
_CHECKED_CHECKBOX = r"^\s*- \[[xX]\]"
_WAIVED_CHECKBOX = r"^\s*- \[ \]\s?\(waive\)"


def _is_checkbox(s: str) -> bool:
    """Checks if the string forms a markdown-style checkbox.

    An optional `(waive)` is considered part of the checkbox.
    """
    return re.match(_CHECKBOX_MARKER, s) is not None


class Checkbox:
    """A line with a checkbox."""

    def __init__(self, line: str) -> None:
        self.line = line
        self.content = self._get_content()

    def _get_content(self) -> str:
        """The remaining string after the markdown checkbox syntax."""
        return re.sub(_CHECKBOX_MARKER, "", self.line).strip()

    @property
    def is_checked(self) -> bool:
        """True if checkbox is checked, False otherwise."""
        return re.match(_CHECKED_CHECKBOX, self.line) is not None

    @property
    def is_waived(self) -> bool:
        """True if checkbox is waived, False otherwise."""
        return re.match(_WAIVED_CHECKBOX, self.line) is not None

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>({self.line})"


class PullRequestContent:
    """The content of a pull request."""

    def __init__(self, body: str) -> None:
        self.body = body
        self.lines = body.splitlines()
        self.checkboxes = self._get_checkboxes()

    def _get_checkboxes(self) -> tuple[Checkbox, ...]:
        """Parses the content of the pull request into checkbox objects."""
        return tuple(Checkbox(line) for line in self.lines if _is_checkbox(line))

    def assert_no_checkboxes_left_unchecked(self) -> None:
        """Tests if some checkboxes in the PR body were not checked."""
        for checkbox in self.checkboxes:
            if not (checkbox.is_checked or checkbox.is_waived):
                msg = f"Neither checked nor waived: the checkbox `{checkbox.content}`."
                raise ValueError(msg)


def _main() -> None:
    """Verify PR JSON data provided on stdin matches expectations.

    Expects the PR description via stdin.
    """
    pr_body = sys.stdin.read()

    pr = PullRequestContent(pr_body)
    pr.assert_no_checkboxes_left_unchecked()


if __name__ == "__main__":
    _main()
