#!/usr/bin/env python

""" Typing helpers for package.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from typing import Any, Union

try:
    import ROOT
    Hist = Union[ROOT.TH1, ROOT.THnBase]
    Axis = ROOT.TAxis
    TFile = ROOT.TFile
except ImportError:
    # It doesn't like the possibility of redefining this, so we need to tell ``mypy`` to ignore it.
    Hist = Any  # type: ignore
    Axis = Any
    TFile = Any
