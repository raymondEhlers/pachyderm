#!/usr/bin/env python

""" Typing helpers for package.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from typing import Any, Union, Type

try:
    import ROOT
    Hist = Union[ROOT.TH1, ROOT.THnBase]
    Axis = Type[ROOT.TAxis]
except ImportError:
    # It doesn't like the possibility of redefining this, so we need to tell ``mypy`` to ignore it.
    Hist = Any  # type: ignore
    Axis = Any  # type: ignore
