# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging as logging
import textwrap

from io import StringIO

import fastforward as ff


def test_duplicate_log_filter():
    # Given: a logger with a DuplicateLogFilter filter that filters duplicated
    # message at the WARNING level
    logger = logging.Logger("test_logger")
    logger.addFilter(ff.logging.DuplicateLogFilter(levels=(logging.WARNING,)))

    log_output = StringIO()
    handler = logging.StreamHandler(log_output)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)

    # When: duplicated messages are logged at different levels
    logger.critical("critical message")
    logger.warning("warning message")
    logger.warning("another warning message")
    logger.warning("warning message")
    logger.critical("critical message")

    # Then: duplicated messages at the WARNING level are only
    # logged once to the output buffer while duplicated messages at other
    # levels are logged every time.
    expected_logs = textwrap.dedent("""
        CRITICAL: critical message
        WARNING: warning message
        WARNING: another warning message
        CRITICAL: critical message
    """).strip()
    logs = log_output.getvalue().strip()

    assert logs == expected_logs
