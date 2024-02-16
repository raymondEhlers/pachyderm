""" Tests for binned_data

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pachyderm import binned_data


def test_axis_slice_copy() -> None:
    axis = binned_data.Axis(np.arange(1, 11))
    sliced_axis = axis[:]

    assert sliced_axis.bin_edges is not axis.bin_edges
    assert not np.may_share_memory(sliced_axis.bin_edges, axis.bin_edges)
    assert sliced_axis == axis


@pytest.mark.parametrize(
    ("start", "stop", "step", "expected_bin_edges"),
    [
        (2, None, None, np.arange(3, 12)),
        (2j, None, None, np.arange(2, 12)),
        (2.1j, None, None, np.arange(2, 12)),
        # +1 because we need to include the upper bin edge
        (None, 6, None, np.arange(1, 7 + 1)),
        # +1 because we need to include the upper bin edge
        (None, 6j, None, np.arange(1, 6 + 1)),
        (4j, 8, None, np.arange(4, 10)),
        (None, None, 2, np.arange(1, 12, 2)),
    ],
    ids=[
        "set start by bin",
        "set start by value",
        "set start by float value",
        "set stop by bin",
        "set stop by value",
        "start value, stop bin",
        "rebin by int",
    ],
)
def test_axis_slice(
    start: int | None,
    stop: int | None,
    step: int | None,
    expected_bin_edges: npt.NDArray[np.int64],
) -> None:
    axis = binned_data.Axis(bin_edges=np.arange(1, 12))
    s = slice(start, stop, step)
    sliced_axis = axis[s]

    assert sliced_axis != axis
    np.testing.assert_allclose(sliced_axis.bin_edges, expected_bin_edges)

    # Cross check with hist
    # NOTE: We use hist here rather than boost-histogram because it allows us to use complex numbers
    #       (ie. uhi) for the indexing comparison.
    # NOTE: We need to create a histogram because we can't slice directly on an axis.
    hist = pytest.importorskip("hist")
    hist_s = slice(s.start, s.stop, hist.rebin(s.step) if s.step else s.step)
    hist_h = hist.Hist(hist.axis.Regular(10, 1, 11))
    hist_sliced_axis = hist_h[hist_s].axes[0]

    np.testing.assert_allclose(sliced_axis.bin_edges, hist_sliced_axis.edges)


@pytest.mark.parametrize(
    ("start", "stop", "step", "expected_bin_edges"),
    [
        (None, None, binned_data.Rebin(2), np.arange(1, 12, 2)),
        (3j, None, binned_data.Rebin(2), np.arange(3, 12, 2)),
        (None, 9j, binned_data.Rebin(2), np.arange(1, 10, 2)),
        (3j, 9j, binned_data.Rebin(2), np.arange(3, 10, 2)),
        (3j, None, binned_data.Rebin(np.arange(3, 12, 2)), np.arange(3, 12, 2)),
        (None, 9j, binned_data.Rebin(np.arange(1, 10, 2)), np.arange(1, 10, 2)),
    ],
    ids=[
        "rebin by int",
        "rebin by int with start",
        "rebin by int with stop",
        "rebin by int with start+stop",
        "rebin by array with start",
        "rebin by array with stop",
    ],
)
def test_axis_slice_rebin(
    start: int | None,
    stop: int | None,
    step: int | None,
    expected_bin_edges: npt.NDArray[np.int64],
) -> None:
    axis = binned_data.Axis(bin_edges=np.arange(1, 12))
    s = slice(start, stop, step)
    sliced_axis = axis[s]

    assert sliced_axis != axis
    np.testing.assert_allclose(sliced_axis.bin_edges, expected_bin_edges)


@pytest.fixture()
def hists_for_rebinning() -> tuple[npt.NDArray[np.float64], binned_data.BinnedData]:
    _values = []
    for i in range(2, 12):
        # repeat i number of times
        for _ in range((i - 1) * 2):
            # offset by 0.5 to put it in the center of the bin
            _values.append(i - 0.5)
    values = np.array(_values, dtype=np.float64)

    hist = pytest.importorskip("hist")
    h_hist = hist.Hist(hist.axis.Regular(10, 1, 11), storage=hist.storage.Weight())
    h_hist.fill(values)

    h = binned_data.BinnedData.from_existing_data(h_hist)

    return values, h


@pytest.mark.parametrize(
    ("start", "stop", "step", "expected_bin_edges"),
    [
        (2, None, None, np.arange(3, 12)),
        (2j, None, None, np.arange(2, 12)),
        (2.1j, None, None, np.arange(2, 12)),
        # +1 because we need to include the upper bin edge
        (None, 6, None, np.arange(1, 7 + 1)),
        # +1 because we need to include the upper bin edge
        (None, 6j, None, np.arange(1, 6 + 1)),
        (4j, 8, None, np.arange(4, 10)),
        (None, None, 2, np.arange(1, 12, 2)),
    ],
    ids=[
        "set start by bin",
        "set start by value",
        "set start by float value",
        "set stop by bin",
        "set stop by value",
        "start value, stop bin",
        "rebin by int",
    ],
)
def test_hist_slice(
    hists_for_rebinning: Any,
    start: int | None,
    stop: int | None,
    step: int | None,
    expected_bin_edges: npt.NDArray[np.int64],
) -> None:
    values, h = hists_for_rebinning
    s = slice(start, stop, step)
    sliced_h = h[s]

    # Cross check on bin edges
    np.testing.assert_allclose(sliced_h.axes[0].bin_edges, expected_bin_edges)

    # Cross check with hist
    hist = pytest.importorskip("hist")
    hist_h = hist.Hist(hist.axis.Variable(sliced_h.axes[0].bin_edges))
    hist_h.fill(values)
    print(f"hist values: {hist_h.values()}")

    np.testing.assert_allclose(sliced_h.axes[0].bin_edges, hist_h.axes[0].edges)
    np.testing.assert_allclose(sliced_h.values, hist_h.values())
    np.testing.assert_allclose(sliced_h.variances, hist_h.variances())


@pytest.mark.parametrize(
    ("start", "stop", "step", "expected_bin_edges"),
    [
        (None, None, binned_data.Rebin(2), np.arange(1, 12, 2)),
        (3j, None, binned_data.Rebin(2), np.arange(3, 12, 2)),
        (None, 9j, binned_data.Rebin(2), np.arange(1, 10, 2)),
        (3j, 9j, binned_data.Rebin(2), np.arange(3, 10, 2)),
        (3j, None, binned_data.Rebin(np.arange(3, 12, 2)), np.arange(3, 12, 2)),
        (None, 9j, binned_data.Rebin(np.arange(1, 10, 2)), np.arange(1, 10, 2)),
        (None, None, binned_data.Rebin(np.array([2, 5, 9])), np.array([2, 5, 9])),
        (None, None, binned_data.Rebin(np.array([1.0, 2, 3, 8])), np.array([1, 2, 3, 8])),
        (None, None, binned_data.Rebin(np.array([2, 11])), np.array([2, 11])),
    ],
    ids=[
        "rebin by int",
        "rebin by int with start",
        "rebin by int with stop",
        "rebin by int with start+stop",
        "rebin by array with start",
        "rebin by array with stop",
        "rebin with varied bin width, variation 1",
        "rebin with varied bin width, variation 2",
        "rebin with varied bin width, variation 3",
    ],
)
def test_hist_slice_rebin(
    hists_for_rebinning: Any,
    start: int | None,
    stop: int | None,
    step: int | None,
    expected_bin_edges: npt.NDArray[np.int64],
) -> None:
    values, h = hists_for_rebinning
    s = slice(start, stop, step)
    sliced_h = h[s]

    # Cross check on bin edges
    np.testing.assert_allclose(sliced_h.axes[0].bin_edges, expected_bin_edges)

    # Cross check with hist
    hist = pytest.importorskip("hist")
    hist_h = hist.Hist(hist.axis.Variable(sliced_h.axes[0].bin_edges))
    hist_h.fill(values)
    print(f"hist values: {hist_h.values()}")

    np.testing.assert_allclose(sliced_h.axes[0].bin_edges, hist_h.axes[0].edges)
    np.testing.assert_allclose(sliced_h.values, hist_h.values())
    np.testing.assert_allclose(sliced_h.variances, hist_h.variances())


@pytest.mark.parametrize(
    "h",
    [
        binned_data.BinnedData(
            axes=[np.array(range(11))],
            values=np.array(range(10)),
            variances=np.array(range(10)),
        ),
        binned_data.BinnedData(
            axes=[np.array(range(11)), np.array(range(6))],
            values=np.array([list(range(10)), np.zeros(10), np.ones(10), np.zeros(10), np.array(range(10))]).T,
            variances=np.array([list(range(10)), np.zeros(10), np.ones(10), np.zeros(10), np.array(range(10))]).T,
        ),
    ],
    ids=["1D", "2D"],
)
def test_conversion_and_projection_with_boost_histogram(h: binned_data.BinnedData) -> None:
    """Test conversion of binned_data to boost-histogram (and back)."""
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
        bh_proj = bh_hist[:, :: bh.sum]
        h_proj = binned_data.BinnedData(
            axes=h.axes[0],
            values=np.sum(h.values, axis=1),
            variances=np.sum(h.variances, axis=1),
        )
        np.testing.assert_allclose(h_proj.values, bh_proj.view().value)
        np.testing.assert_allclose(h_proj.variances, bh_proj.view().variance)

    print(h)


@pytest.mark.parametrize(
    "h",
    [
        binned_data.BinnedData(
            axes=[np.array(range(11))],
            values=np.array(range(10)),
            variances=np.array(range(10)),
        ),
        binned_data.BinnedData(
            axes=[np.array(range(11)), np.array(range(6))],
            # NOTE: The transpose is important here! We want the shape to be (10, 5), but this has shape (5, 10).
            #       The class will detect this reversal and transpose it.
            values=np.array([list(range(10)), np.zeros(10), np.ones(10), np.zeros(10), np.array(range(10))]).T,
            variances=np.array([list(range(10)), np.zeros(10), np.ones(10), np.zeros(10), np.array(range(10))]).T,
        ),
    ],
    ids=["1D", "2D"],
)
def test_conversion_and_projection_with_root_hist(h: binned_data.BinnedData) -> None:
    """Test conversion of binned_data to boost-histogram (and back)."""
    ROOT = pytest.importorskip("ROOT")  # noqa: F841
    root_hist = h.to_ROOT()
    # First, check the ROOT conversion by hand.
    # Values
    # We need to transpose and our arrays to put our arrays into the same format as ROOT.
    np.testing.assert_allclose(
        h.values.T.reshape(-1),
        [
            root_hist.GetBinContent(i)
            for i in range(root_hist.GetNcells())
            if not (root_hist.IsBinUnderflow(i) or root_hist.IsBinOverflow(i))
        ],
    )
    # Sanity check that the flow bins are empty.
    flow_values = [
        root_hist.GetBinContent(i)
        for i in range(root_hist.GetNcells())
        if root_hist.IsBinUnderflow(i) or root_hist.IsBinOverflow(i)
    ]
    np.testing.assert_allclose(np.zeros(len(flow_values)), flow_values)
    # Errors
    # We need to transpose and our arrays to put our arrays into the same format as ROOT.
    np.testing.assert_allclose(
        h.errors.T.reshape(-1),
        [
            root_hist.GetBinError(i)
            for i in range(root_hist.GetNcells())
            if not (root_hist.IsBinUnderflow(i) or root_hist.IsBinOverflow(i))
        ],
    )
    # Sanity check that the flow bins are empty.
    flow_errors = [
        root_hist.GetBinError(i)
        for i in range(root_hist.GetNcells())
        if root_hist.IsBinUnderflow(i) or root_hist.IsBinOverflow(i)
    ]
    np.testing.assert_allclose(np.zeros(len(flow_errors)), flow_errors)
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
        np.testing.assert_allclose(
            h_proj.values,
            np.array(
                [
                    root_proj.GetBinContent(i)
                    for i in range(1, root_proj.GetNcells())
                    if not (root_proj.IsBinUnderflow(i) or root_proj.IsBinOverflow(i))
                ]
            ),
        )
        np.testing.assert_allclose(h_proj.values, proj_loop.values)
