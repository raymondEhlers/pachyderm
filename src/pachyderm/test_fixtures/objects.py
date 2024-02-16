""" Objects related fixtures to aid testing.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@yale.edu>, Yale University
"""
from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest  # pylint: disable=import-error


@pytest.fixture()
def test_root_hists() -> Any:
    """Create minimal TH*F hists in 1D, 2D, and 3D. Each has been filled once.

    Args:
        None
    Returns:
        tuple: (TH1F, TH2F, TH3F) for testing
    """
    ROOT = pytest.importorskip("ROOT")

    @dataclass
    class RootHists:
        """ROOT histograms for testing.

        Just for convenience.
        """

        hist1D: ROOT.TH1  # type: ignore[name-defined]
        hist2D: ROOT.TH2  # type: ignore[name-defined]
        hist3D: ROOT.TH3  # type: ignore[name-defined]

    # Define the hist to use for testing

    tag = uuid.uuid4()
    hist = ROOT.TH1F(f"test_{tag}", f"test_{tag}", 10, 0, 1)
    hist.Fill(0.1)

    tag = uuid.uuid4()
    hist2D = ROOT.TH2F(f"test2_{tag}", f"test2_{tag}", 10, 0, 1, 10, 0, 20)
    hist2D.Fill(0.1, 1)

    tag = uuid.uuid4()
    hist3D = ROOT.TH3F(f"test3_{tag}", f"test3_{tag}", 10, 0, 1, 10, 0, 20, 10, 0, 100)
    hist3D.Fill(0.1, 1, 10)

    return RootHists(hist1D=hist, hist2D=hist2D, hist3D=hist3D)


@pytest.fixture()
def setup_non_uniform_binning() -> Any:
    """Test a ROOT histogram with non-uniform binning.

    Args:
        None
    Returns:
        1D histogram with non-uniform binning.
    """
    ROOT = pytest.importorskip("ROOT")

    binning = np.array([0, 1, 2, 4, 5, 6], dtype=np.float64)
    hist = ROOT.TH1F("test", "test", 5, binning)
    hist.Fill(1.5)

    return hist


@pytest.fixture()
def test_sparse() -> Any:
    """Create a THnSparseF for testing.

    Fills in a set of values for testing.

    Args:
        None.
    Returns:
        tuple: (THnSparseF, fill_value) for testing.
    """
    ROOT = pytest.importorskip("ROOT")

    @dataclass
    class SparseAxis:
        """THnSparse axis information.

        Just for convenience.
        """

        n_bins: int
        min: float
        max: float

    ignored_axis = SparseAxis(n_bins=1, min=0.0, max=1.0)
    selected_axis1 = SparseAxis(n_bins=10, min=0.0, max=20.0)
    selected_axis2 = SparseAxis(n_bins=20, min=-10.0, max=10.0)
    selected_axis3 = SparseAxis(n_bins=30, min=0.0, max=30.0)
    # We want to select axes 2, 4, 5
    axes = [ignored_axis, ignored_axis, selected_axis1, ignored_axis, selected_axis2, selected_axis3, ignored_axis]

    # Create the actual sparse
    # NOTE: dtype is required here for the sparse to be created successfully.
    bins = np.array([el.n_bins for el in axes], dtype=np.int32)
    mins = np.array([el.min for el in axes])
    maxes = np.array([el.max for el in axes])
    # logger.debug("bins: {}, mins: {}, maxs: {}".format(bins, mins, maxes))
    sparse = ROOT.THnSparseF("testSparse", "testSparse", len(axes), bins, mins, maxes)

    # Fill in some strategic values.
    # Wrapper function is for convenience.
    def fill_sparse(one: float, two: float, three: float) -> None:
        # NOTE: For whatever reason, this _has_ to be float64 even though this is a
        #       SparseF. Apparently switching to a SparseD also works with float64,
        #       so something strange seems to be happening internally. But since
        #       float64 works, we stick with it.
        sparse.Fill(np.array([0.0, 0.0, one, 0.0, two, three, 0.0], dtype=np.float64))

    fill_values = [(4.0, -2.0, 10.0), (4.0, 2.0, 10.0)]
    for values in fill_values:
        fill_sparse(*values)

    return (sparse, fill_values)


@pytest.fixture()
def simple_test_functions() -> tuple[Callable[[float, float, float], float], Callable[[float, float, float], float]]:
    """Define simple test functions for use in tests.

    Args:
        None.
    Returns:
        func 1 (args: x, a, b), func 2 (args: x, c, d)
    """

    def func_1(x: float, a: float, b: float) -> float:
        """Test function."""
        return x + a + b

    def func_2(x: float, c: float, d: float) -> float:
        """Test function 2."""
        return x + c + d

    return func_1, func_2
