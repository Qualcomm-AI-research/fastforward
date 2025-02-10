# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import os
import re

from pathlib import Path

import pytest


def pytest_addoption(parser):
    """Enable logging benchmark results."""
    group = parser.getgroup("benchmark")
    group.addoption(
        "--benchmark",
        action="store_true",
        help="Run benchmark tests",
    )
    group.addoption(
        "--benchmark-report",
        type=Path,
        help="Path to store a report in an open metrics format",
    )


def pytest_configure(config):
    if config.getoption("--benchmark-report") is not None:
        config.option.benchmark = True

    if re.match(r"\s*benchmark\s*", config.option.markexpr) is None:
        config.option.markexpr = " and ".join([
            f"({config.option.markexpr})",
            f"({'' if config.option.benchmark else 'not'} benchmark)",
        ])


@pytest.hookimpl(hookwrapper=True)
def pytest_report_teststatus(report, config):
    yield
    if report.when != "teardown":
        return
    text = os.linesep.join(f"{val}" for name, val in report.user_properties if name == "benchmark")
    report.sections.append(("benchmark", text))


@pytest.hookimpl(hookwrapper=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    yield

    reports = terminalreporter.getreports("")
    content = os.linesep.join(
        text for report in reports for secname, text in report.sections if secname == "benchmark"
    )
    if content:
        terminalreporter.ensure_newline()
        terminalreporter.section("Benchmarks", sep="-", blue=True, bold=True)
        terminalreporter.line(
            os.linesep.join(
                line for line in content.splitlines() if line and not line.startswith("#")
            )
        )
        report = config.getoption("--benchmark-report")
        if report is not None:
            report.write_text(content)
