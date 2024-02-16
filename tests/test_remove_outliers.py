""" Tests for the outliers removal module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""
from __future__ import annotations

import ctypes
import logging
import math

import numpy as np
import pytest

from pachyderm import projectors, remove_outliers

# Setup logger
logger = logging.getLogger(__name__)


def test_mean_and_median(test_root_hists):
    """Test calculating the mean and median of a histogram."""
    ROOT = pytest.importorskip("ROOT")  # noqa: F841
    hist = test_root_hists.hist1D
    for i in range(1, 11):
        hist.SetBinContent(i, i)

    # Some helpful debug information for good measure.
    # The expected histogram should have [i for i in range(1, 11)] entries (ie [1, 2, 3, ..., 10])
    for i in range(1, 11):
        logger.debug(f"{i}: {hist.GetBinContent(i)}")

    mean, median = remove_outliers._get_mean_and_median(hist)
    # These values are weighted by the bin content
    expected_mean = 0.65
    expected_median = 0.692857142857143

    assert np.isclose(mean, expected_mean)
    assert np.isclose(median, expected_median)


@pytest.mark.parametrize(
    ("moving_average", "expected_cut_index"),
    [
        # The expected cut axis here is where the array changes to 0, and then shifted
        # to the index that corresponds to the moving average being calculated from the middle
        # as opposed to only looking forward. See the function docs.
        # The moving average drops below at 6, so the cut index is: 6 + 4 // 2 = 8
        (np.array([2, 2, 2, 2, 2, 2, 0, 0, 0, 0]), 8),
        # Have a non-zero entry early on, then no entries, then back below.
        # It should ignore the early drop, and the moving average drops below
        # at 12, so the cut index is: 12 + 4 // 2 = 14
        (np.array([2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0]), 14),
    ],
    ids=["Constant until below", "Early bump above threshold"],
)
def test_outliers_determination_from_moving_average(moving_average, expected_cut_index):
    """Test outliers determination for a pathological moving average.

    Note:
        Remember that this test uses values that are the moving averages directly - not
        the values from which the moving average would be calculated!
    """
    # Setup
    limit_of_number_of_values_below_threshold = 4

    # Determine the outliers.
    cut_index = remove_outliers._determine_outliers_for_moving_average(
        moving_average=moving_average,
        moving_average_threshold=1.0,
        number_of_values_to_search_ahead=5,
        limit_of_number_of_values_below_threshold=limit_of_number_of_values_below_threshold,
    )

    # Check the final result.
    assert cut_index == expected_cut_index


@pytest.fixture(params=["2D", "3D"])
def setup_outliers_hist(request):
    ROOT = pytest.importorskip("ROOT")

    # Setup
    hist3D = False
    # Additional scale down factor by number of the bins of the other axes because it will
    # project down over those bins, which will increase the projected value by a factor
    # of number of bins (so for a 2D hist with 10 bins on the x axis, it's a factor of 10).
    function_scale_factor = 10.0
    if request.param == "3D":
        hist3D = True
        # Need to scale down by another factor of 10 to account for the z axis bins.
        function_scale_factor *= 10.0

    # Create the hist
    if hist3D:
        hist = ROOT.TH3F("test3D", "test2D", 10, 0, 10, 100, 0, 100, 10, 0, 10)
    else:
        hist = ROOT.TH2F("test2D", "test2D", 10, 0, 10, 100, 0, 100)

    # Fill function as an power law.
    def f(x: int | float, y: int | float) -> int | float:  # noqa: ARG001
        """Power function to the power of -1.0

        Normalization selected such that x = 50 is equal to 1.

        Additional division by number of the bins of the other axes because it will
        project down over those bins, which will increase the projected value by a factor
        of number of bins (so for a 2D hist with 10 bins on the x axis, it's a factor of 10).
        """
        return 50.0 * math.pow(y, -1.0) / function_scale_factor

    # Create the histogram.
    for i in range(1, hist.GetNcells()):
        if hist.IsBinUnderflow(i) or hist.IsBinOverflow(i):
            continue

        # Get fill location
        x = ctypes.c_int(0)
        y = ctypes.c_int(0)
        z = ctypes.c_int(0)
        hist.GetBinXYZ(i, x, y, z)

        # logger.debug(f"y: {y.value}, f: {f(x.value, y.value)}")

        # Determine the appropriate values for filling.
        base_args = [x.value, y.value]
        if hist3D:
            base_args.append(z.value)
        bin_content_args = [*base_args, f(x.value, y.value)]
        # Fake the errors. Don't use sqrt because it will create errors larger than filled values.
        # Since we're faking the input data, we don't really care about the errors. We just want
        # them to be non-zero.
        bin_error_args = [*base_args, f(x.value, y.value) / 2]

        # Fill in the determined values
        hist.SetBinContent(*bin_content_args)
        hist.SetBinError(*bin_error_args)

    yield hist

    # Cleanup
    del hist


class TestOutliersRemovalIntegration:
    ROOT = pytest.importorskip("ROOT")

    @pytest.mark.parametrize(
        "remove_entries",
        [
            False,
            True,
        ],
        ids=["Do not remove entry", "Remove early entries"],
    )
    def test_remove_outliers(self, setup_outliers_hist, remove_entries):
        """Integration test for removing outliers.

        We check a 2D and a 3D hist, both projecting to the y axis (arbitrarily selected).
        """
        # Setup
        input_hist = setup_outliers_hist
        if remove_entries:
            x = ctypes.c_int(0)
            z = ctypes.c_int(0)
            for i in range(10, 13):
                y = ctypes.c_int(i)
                # Remove some entries. This should have no impact.
                input_hist.SetBinContent(x.value, y.value, z.value, 0)
                input_hist.SetBinContent(x.value, y.value, z.value, 0)

        # Keep a reference for the original hist.
        initial_hist = input_hist.Clone("InitialHist")

        # Setup and run the manager.
        outliers_manager = remove_outliers.OutliersRemovalManager()
        outliers_start_index = outliers_manager.run(
            outliers_removal_axis=projectors.TH1AxisType.y_axis, hist=input_hist
        )

        # Now, check if the input_hist has been modified in place.
        # Since the function that we use is power law that isn't symmetric around a bin, it's
        # easier to just determine this empirically. For more targeted tests, see the
        # pathological tests of the moving average below.
        assert outliers_start_index == 51

        # Check bins of input_hist. They should be different than initial hist.
        x = ctypes.c_int(0)
        y = ctypes.c_int(0)
        z = ctypes.c_int(0)
        for index in range(initial_hist.GetNcells()):
            # Get the bin x, y, z from the global bin
            initial_hist.GetBinXYZ(index, x, y, z)

            # Ensure that the values are actually removed.
            if y.value >= outliers_start_index:
                # They should be 0.
                assert np.isclose(input_hist.GetBinContent(index), 0.0)
                assert np.isclose(input_hist.GetBinError(index), 0.0)
                # Additional check for good measure.
                # NOTE: The underflow and overflow bins could still be 0, so we don't check those.
                if not input_hist.IsBinUnderflow(index) and not input_hist.IsBinOverflow(index):
                    assert input_hist.GetBinContent(index) != initial_hist.GetBinContent(index)
                    assert input_hist.GetBinError(index) != initial_hist.GetBinError(index)
            else:
                # These should just be the standard filled values.
                assert input_hist.GetBinContent(index) == initial_hist.GetBinContent(index)
                assert input_hist.GetBinError(index) == initial_hist.GetBinError(index)
