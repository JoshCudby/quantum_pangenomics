"""Logging configuration for the qiskit_qaoa package.

Provides a factory function ``get_logger`` that returns a named Python logger
with two handlers:

- A ``stdout`` handler at ``INFO`` level for informational messages.
- A ``stderr`` handler at ``WARN`` level for warnings and errors.

Both handlers use the same timestamp format.  Duplicate handlers are removed
before new ones are attached so that the logger can be safely called multiple
times with the same module name.
"""

import logging
import sys


class InfoFilter(logging.Filter):
    """Logging filter that passes only INFO-level records."""

    def filter(self, record):
        """Return True only for INFO-level log records.

        Args:
            record: The ``logging.LogRecord`` to evaluate.

        Returns:
            True if the record level is INFO, False otherwise.
        """
        return record.levelno in [logging.INFO]


class WarnErrorFilter(logging.Filter):
    """Logging filter that passes only WARNING and ERROR-level records."""

    def filter(self, record):
        """Return True only for WARNING or ERROR-level log records.

        Args:
            record: The ``logging.LogRecord`` to evaluate.

        Returns:
            True if the record level is WARN or ERROR, False otherwise.
        """
        return record.levelno in [logging.WARN, logging.ERROR]


def get_logger(name: str):
    """Create or retrieve a configured logger for the given module name.

    Args:
        name: The logger name, typically ``__name__`` of the calling module.

    Returns:
        A ``logging.Logger`` instance with INFO-level stdout and WARN-level
        stderr handlers attached.
    """
    root = logging.getLogger(name)
    root.setLevel(logging.INFO)
    if (root.hasHandlers()):
        root.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s","%H:%M:%S")

    out_handler = logging.StreamHandler(sys.stdout)
    out_handler.setLevel(logging.INFO)
    out_handler.addFilter(InfoFilter())
    out_handler.setFormatter(formatter)

    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setLevel(logging.WARN)
    err_handler.addFilter(WarnErrorFilter())
    err_handler.setFormatter(formatter)

    root.addHandler(out_handler)
    root.addHandler(err_handler)
    return root