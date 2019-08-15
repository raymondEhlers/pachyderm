#!/usr/bin/env python

""" Typing helpers for package.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from typing import Any, Union

try:
    import ROOT
    # Tell ROOT to ignore command line options so args are passed to python
    # NOTE: Must be immediately after import ROOT and it must be called the first time ROOT is imported!
    #       We do this here (even though it is unrelated to typing helpers) because it is the most common
    #       first import which requires ROOT. So by putting it here, it should (hopefully) cover all executables.
    #       However, this means that it needs to be imported before some pachyderm modules.
    ROOT.PyConfig.IgnoreCommandLineOptions = True

    Hist = Union[ROOT.TH1, ROOT.THnBase]
    Axis = ROOT.TAxis
    Canvas = ROOT.TCanvas
    TFile = ROOT.TFile
except ImportError:
    # It doesn't like the possibility of redefining this, so we need to tell ``mypy`` to ignore it.
    Hist = Any  # type: ignore
    Axis = Any
    Canvas = Any
    TFile = Any
