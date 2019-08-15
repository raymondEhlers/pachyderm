#!/usr/bin/env python

""" Logging related fixtures to aid testing.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@yale.edu>, Yale University
"""

import logging
from typing import Any

import pytest

# Set logging level as a global variable to simplify configuration.
# This is not ideal, but fine for simple tests.
logging_level = logging.DEBUG

@pytest.fixture  # type: ignore
def logging_mixin(caplog: Any) -> None:
    """ Logging mixin to capture logging messages from modules.

    It logs at the debug level, which is probably most useful for when a test fails.
    """
    caplog.set_level(logging_level)
