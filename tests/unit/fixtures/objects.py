#!/usr/bin/env python

""" Objects related fixtures to aid testing.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@yale.edu>, Yale University
"""

from dataclasses import dataclass
from typing import Any, Callable, Tuple

import numpy as np
import pytest

@pytest.fixture  # type: ignore
def test_root_hists() -> Any:
    """ Create minimal TH*F hists in 1D, 2D, and 3D. Each has been filled once.

    Args:
        None
    Returns:
        tuple: (TH1F, TH2F, TH3F) for testing
    """
    import ROOT

    @dataclass
    class RootHists:
        """ ROOT histograms for testing.

        Just for convenience.
        """
        hist1D: ROOT.TH1
        hist2D: ROOT.TH2
        hist3D: ROOT.TH3

    # Define the hist to use for testing
    hist = ROOT.TH1F("test", "test", 10, 0, 1)
    hist.Fill(.1)
    hist2D = ROOT.TH2F("test2", "test2", 10, 0, 1, 10, 0, 20)
    hist2D.Fill(.1, 1)
    hist3D = ROOT.TH3F("test3", "test3", 10, 0, 1, 10, 0, 20, 10, 0, 100)
    hist3D.Fill(.1, 1, 10)

    return RootHists(hist1D = hist, hist2D = hist2D, hist3D = hist3D)

@pytest.fixture  # type: ignore
def setup_non_uniform_binning() -> Any:
    """ Test a ROOT histogram with non-uniform binning.

    Args:
        None
    Returns:
        1D histogram with non-uniform binning.
    """
    import ROOT

    binning = np.array([0, 1, 2, 4, 5, 6], dtype = np.float64)
    hist = ROOT.TH1F("test", "test", 5, binning)
    hist.Fill(1.5)

    return hist

@pytest.fixture  # type: ignore
def test_sparse() -> Any:
    """ Create a THnSparseF for testing.

    Fills in a set of values for testing.

    Args:
        None.
    Returns:
        tuple: (THnSparseF, fill_value) for testing.
    """
    import ROOT

    @dataclass
    class SparseAxis:
        """ THnSparse axis information.

        Just for convenience.
        """
        n_bins: int
        min: float
        max: float

    ignored_axis   = SparseAxis(n_bins =  1, min =   0.0, max =  1.0)  # noqa: E221, E222
    selected_axis1 = SparseAxis(n_bins = 10, min =   0.0, max = 20.0)  # noqa: E222
    selected_axis2 = SparseAxis(n_bins = 20, min = -10.0, max = 10.0)
    selected_axis3 = SparseAxis(n_bins = 30, min =   0.0, max = 30.0)  # noqa: E222
    # We want to select axes 2, 4, 5
    axes = [ignored_axis, ignored_axis, selected_axis1, ignored_axis, selected_axis2, selected_axis3, ignored_axis]

    # Create the actual sparse
    # NOTE: dtype is required here for the sparse to be created successfully.
    bins = np.array([el.n_bins for el in axes], dtype=np.int32)
    mins = np.array([el.min for el in axes])
    maxes = np.array([el.max for el in axes])
    #logger.debug("bins: {}, mins: {}, maxs: {}".format(bins, mins, maxes))
    sparse = ROOT.THnSparseF("testSparse", "testSparse", len(axes), bins, mins, maxes)

    # Fill in some strategic values.
    # Wrapper function is for convenience.
    def fill_sparse(one: float, two: float, three: float) -> None:
        # NOTE: For whatever reason, this _has_ to be float64 even though this is a
        #       SparseF. Apparently switching to a SparseD also works with float64,
        #       so something strange seems to be happening internally. But since
        #       float64 works, we stick with it.
        sparse.Fill(np.array([0., 0., one, 0., two, three, 0.], dtype = np.float64))
    fill_values = [
        (4., -2., 10.),
        (4., 2., 10.)
    ]
    for values in fill_values:
        fill_sparse(*values)

    return (sparse, fill_values)

@pytest.fixture  # type: ignore
def simple_test_functions() -> Tuple[Callable[[float, float, float], float],
                                     Callable[[float, float, float], float]]:
    """ Define simple test functions for use in tests.

    Args:
        None.
    Returns:
        func 1 (args: x, a, b), func 2 (args: x, c, d)
    """
    def func_1(x: float, a: float, b: float) -> float:
        """ Test function. """
        return x + a + b

    def func_2(x: float, c: float, d: float) -> float:
        """ Test function 2. """
        return x + c + d
    return func_1, func_2
