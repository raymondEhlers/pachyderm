#!/usr/bin/env python

""" Physics Analysis Core for Heavy-Ions package, known as Pachyderm.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

__all__ = [
    "generic_class",
    "generic_config",
    "histogram",
    "projectors",
    "utils",
]

# Provide easy access to the version
# __version__ is the version string, while version_info is a tuple with an entry per point in the verion
from pachyderm.version import __version__   # noqa
from pachyderm.version import version_info  # noqa
