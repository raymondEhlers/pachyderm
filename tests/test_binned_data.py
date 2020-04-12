#!/usr/bin/env python3

""" Tests for binned_data

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from typing import Any

import numpy as np
import pytest

from pachyderm import binned_data

@pytest.mark.parametrize("h", [  # type: ignore
    binned_data.BinnedData(
        axes = [np.array(range(11))],
        values = np.array(range(10)),
        variances = np.array(range(10)),
    ),
    binned_data.BinnedData(
        axes = [np.array(range(11)), np.array(range(6))],
        values = np.array([list(range(10)), np.zeros(10), np.ones(10), np.zeros(10), np.array(range(10))]).T,
        variances = np.array([list(range(10)), np.zeros(10), np.ones(10), np.zeros(10), np.array(range(10))]).T,
    ),
], ids = ["1D", "2D"])
def test_conversion_and_projection_with_boost_histogram(logging_mixin: Any, h: binned_data.BinnedData) -> None:
    """ Test conversion of binned_data to boost-histogram (and back).

    """
    bh = pytest.importorskip("boost_histogram")
    # First, convert and check that it was successful.
    bh_hist = h.to_boost_histogram()
    for i, a in enumerate(bh_hist.axes):
        np.testing.assert_allclose(h.axes[i].bin_edges, a.edges)
    np.testing.assert_allclose(h.values, bh_hist.view().value)
    np.testing.assert_allclose(h.variances, bh_hist.view().variance)
    # Next, check that conversion to ROOT and back still works.
    loop = binned_data.BinnedData.from_existing_data(bh_hist)
    for i, a in enumerate(loop.axes):
        np.testing.assert_allclose(h.axes[i].bin_edges, a.bin_edges)
    np.testing.assert_allclose(h.values, loop.values)
    np.testing.assert_allclose(h.variances, loop.variances)

    if h.values.ndim > 1:
        # Check projection
        bh_proj = bh_hist[:, ::bh.sum]
        h_proj = binned_data.BinnedData(
            axes=h.axes[0],
            values=np.sum(h.values, axis=1),
            variances=np.sum(h.variances, axis=1),
        )
        np.testing.assert_allclose(h_proj.values, bh_proj.view().value)
        np.testing.assert_allclose(h_proj.variances, bh_proj.view().variance)

    print(h)

@pytest.mark.parametrize("h", [  # type: ignore
    binned_data.BinnedData(
        axes = [np.array(range(11))],
        values = np.array(range(10)),
        variances = np.array(range(10)),
    ),
    binned_data.BinnedData(
        axes = [np.array(range(11)), np.array(range(6))],
        # NOTE: The transpose is important here! We want the shape to be (10, 5), but this has shape (5, 10).
        #       The class will detect this reversal and transpose it.
        values = np.array([list(range(10)), np.zeros(10), np.ones(10), np.zeros(10), np.array(range(10))]).T,
        variances = np.array([list(range(10)), np.zeros(10), np.ones(10), np.zeros(10), np.array(range(10))]).T,
    ),
], ids = ["1D", "2D"])
def test_conversion_and_projection_with_root_hist(logging_mixin: Any, h: binned_data.BinnedData) -> None:
    """ Test conversion of binned_data to boost-histogram (and back).

    """
    ROOT = pytest.importorskip("ROOT")  # noqa: F841
    root_hist = h.to_ROOT()
    # First, check the ROOT conversion by hand.
    # Values
    # We need to transpose and our arrays to put our arrays into the same format as ROOT.
    np.testing.assert_allclose(
        h.values.T.reshape(-1), [root_hist.GetBinContent(i) for i in range(0, root_hist.GetNcells()) if not(root_hist.IsBinUnderflow(i) or root_hist.IsBinOverflow(i))]
    )
    # Sanity check that the flow bins are empty.
    flow_values = [root_hist.GetBinContent(i) for i in range(0, root_hist.GetNcells()) if root_hist.IsBinUnderflow(i) or root_hist.IsBinOverflow(i)]
    np.testing.assert_allclose(
        np.zeros(len(flow_values)), flow_values
    )
    # Errors
    # We need to transpose and our arrays to put our arrays into the same format as ROOT.
    np.testing.assert_allclose(
        h.errors.T.reshape(-1), [root_hist.GetBinError(i) for i in range(0, root_hist.GetNcells()) if not(root_hist.IsBinUnderflow(i) or root_hist.IsBinOverflow(i))]
    )
    # Sanity check that the flow bins are empty.
    flow_errors = [root_hist.GetBinError(i) for i in range(0, root_hist.GetNcells()) if root_hist.IsBinUnderflow(i) or root_hist.IsBinOverflow(i)]
    np.testing.assert_allclose(
        np.zeros(len(flow_errors)), flow_errors
    )
    # Next, check that conversion to ROOT and back still works.
    loop = binned_data.BinnedData.from_existing_data(root_hist)
    for i, a in enumerate(loop.axes):
        np.testing.assert_allclose(h.axes[i].bin_edges, a.bin_edges)
    np.testing.assert_allclose(h.values, loop.values)
    np.testing.assert_allclose(h.variances, loop.variances)

    # Check projection
    if h.values.ndim > 1:
        root_proj = root_hist.ProjectionX()
        proj_loop = binned_data.BinnedData.from_existing_data(root_proj)
        h_proj = binned_data.BinnedData(
            axes=h.axes[0],
            values=np.sum(h.values, axis=1),
            variances=np.sum(h.variances, axis=1),
        )

        # Check axes, values
        np.testing.assert_allclose(h_proj.axes[0].bin_edges, proj_loop.axes[0].bin_edges)
        np.testing.assert_allclose(h_proj.values, np.array([root_proj.GetBinContent(i) for i in range(1, root_proj.GetNcells()) if not (root_proj.IsBinUnderflow(i) or root_proj.IsBinOverflow(i))]))
        np.testing.assert_allclose(h_proj.values, proj_loop.values)


