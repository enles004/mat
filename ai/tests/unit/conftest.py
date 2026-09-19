import logging

import pytest

from src.core.logging import stop_audit_listener

_CHANNEL_NAMES = ("src", "mat.audit", "mat.monitor")


@pytest.fixture
def reset_log_channels():
    """Detach every APP/AUDIT/MON handler so tests see a fresh channel setup."""

    def _reset() -> None:
        stop_audit_listener()
        for name in _CHANNEL_NAMES:
            logger = logging.getLogger(name)
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
            logger.setLevel(logging.NOTSET)
            logger.propagate = True

    _reset()
    yield _reset
    _reset()
