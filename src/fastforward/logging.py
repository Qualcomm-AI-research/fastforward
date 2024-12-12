import logging

from typing import Sequence

CRITICAL = logging.CRITICAL
FATAL = logging.FATAL
ERROR = logging.ERROR
WARNING = logging.WARNING
WARN = logging.WARN
INFO = logging.INFO
DEBUG = logging.DEBUG
NOTSET = logging.NOTSET


class DuplicateLogFilter(logging.Filter):
    """
    Filter for duplicated log messages.

    Filters log messages that are logged more than once on a specific loglevel
    included in `levels` are filtered. This reduces noise on either stdout or
    logfiles. Note that log messages are only filtered if their loglevel is a
    member of `levels`. This means that logs of a lower level are not filtered
    if that lower level is not a member of levels.

    Args:
        name: Name of the filter
        levels: Sequence of loglevels (e.g., logging.FATAL, logging.WARNING, etc.)
    """

    def __init__(self, name: str = "", levels: Sequence[int] = ()) -> None:
        super().__init__(name)
        self._supressed_levels = tuple(levels)
        self._logged_warnings: set[tuple[int, str]] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno not in self._supressed_levels:
            return True

        log = (record.levelno, record.msg)
        if log in self._logged_warnings:
            return False

        self._logged_warnings.add(log)
        return True
