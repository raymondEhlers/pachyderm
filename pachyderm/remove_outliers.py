#!/usr/bin/env python

""" Provides outliers removal methods.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@yale.edu>, Yale University
"""

import ctypes
from dataclasses import dataclass, field
import enum
import logging
import numpy as np
from typing import Any, Dict, List, Tuple, Union

from pachyderm import histogram
from pachyderm import projectors
from pachyderm.typing_helpers import Hist
from pachyderm import utils

logger = logging.getLogger(__name__)

# Typing helper
T_ParticleLevelAxis = Union[projectors.TH1AxisType, enum.Enum]

def _get_mean_and_median(hist: Hist) -> Tuple[float, float]:
    """ Retrieve the mean and median from a ROOT histogram.

    Note:
        These values are not so trivial to calculate without ROOT, as they are the bin values
        weighted by the bin content.

    Args:
        hist: Histogram from which the values will be extract.
    Returns:
        mean, median of the histogram.
    """
    # Median
    # See: https://root-forum.cern.ch/t/median-of-histogram/7626/5
    x = ctypes.c_double(0)
    q = ctypes.c_double(0.5)
    # Apparently needed to be safe(?)
    hist.ComputeIntegral()
    hist.GetQuantiles(1, x, q)

    mean = hist.GetMean()

    return (mean, x.value)

@dataclass
class _OutputObject:
    """ Helper object to retrieve the result of a projector. """
    output: Hist

def _project_to_part_level(hist: Hist, particle_level_axis: T_ParticleLevelAxis) -> Hist:
    """ Project the input histogram to the particle level axis.

    Args:
        hist: Histogram to check for outliers.
        particle_level_axis: Identifies the particle level axis.
    Returns:
        The histogram to check for outliers.
    """
    # Setup the projector
    import ROOT
    projection_information: Dict[str, Any] = {}
    output_object = _OutputObject(None)
    projector = projectors.HistProjector(
        observable_to_project_from = hist,
        output_observable = output_object,
        output_attribute_name = "output",
        projection_name_format = "particle_level_hist",
        projection_information = projection_information,
    )
    # No additional_axis_cuts or projection_dependent_cut_axes
    # Projection axis
    projector.projection_axes.append(
        projectors.HistAxisRange(
            axis_type = particle_level_axis,
            axis_range_name = "particle_level_axis",
            min_val = projectors.HistAxisRange.apply_func_to_find_bin(None, 1),
            max_val = projectors.HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.GetNbins),
        )
    )

    # Perform the actual projection and return the output.
    projector.project()
    return output_object.output

def _determine_outliers_index(hist: Hist,
                              moving_average_threshold: float = 2.0,
                              number_of_values_to_search_ahead: int = 5,
                              limit_of_number_of_values_below_threshold: int = None) -> int:
    """ Determine the location of where outliers begin in a 1D histogram.

    When the moving average falls below the limit, we consider the outliers to have begun.

    To determine the location of outliers:

    - Calculate the moving average for number_of_values_to_search_ahead values.
    - First, the moving average must go above the limit at least once to guard against a random cut
      in a low pt bin causing most of the data to be cut out.
    - Next, we look for a consecutive number of entries below limit_of_number_of_values_below_threshold.
    - If we meet that condition, we have found the index where the outliers begin. We then return the ROOT
      bin index of the value.
    - If not, we return -1.

    Note:
        The index returned is when the moving average first drops below the threshold for a moving average
        calculated with that bin at the center. This is somewhat different from a standard moving average
        calculation which would only look forward in the array.

    Args:
        hist: Histogram to be checked for outliers.
        moving_average_threshold: Value of moving average under which we consider the moving average
            to be 0. Default: 2.
        number_of_values_to_search_ahead: Number of values to search ahead in the array when calculating
            the moving average. Default: 5.
        limit_of_number_of_values_below_threshold: Number of consecutive bins below the threshold to be considered
            the beginning of outliers. Default: None, which will correspond to number_of_values_to_search_ahead - 1.
    Returns:
        Index of the histogram axes where the outliers begin.
    """
    # Validation
    import ROOT
    if isinstance(hist, (ROOT.TH2, ROOT.TH3, ROOT.THnBase)):
        raise ValueError(
            f"Given histogram '{hist.GetName()}' of type {type(hist)}, but can only"
            " determine the outlier location of a 1D histogram. Please project to"
            " the particle level axis first."
        )

    if limit_of_number_of_values_below_threshold is None:
        # In principle, this could be another value. However, this is what was used in the previous outliers
        # removal implementation.
        limit_of_number_of_values_below_threshold = number_of_values_to_search_ahead - 1

    # It is much more convenient to work with a numpy array.
    hist_to_check = histogram.Histogram1D.from_existing_hist(hist)

    logger.debug(f"y: {hist_to_check.y}")

    # Must have at least one bin above the specified threshold.
    found_at_least_one_bin_above_threshold = False
    # Index we will search for from which outliers will be cut.
    cut_index = -1

    # Calculate the moving average for the entire axis, looking ahead including the current bin + 4 = 5 ahead.
    number_of_values_to_search_ahead = 5
    moving_average = utils.moving_average(hist_to_check.y, n = number_of_values_to_search_ahead)
    below_threshold = moving_average < moving_average_threshold
    #below_threshold = moving_average[moving_average < moving_average_threshold]

    logger.debug(f"moving_average: {moving_average}")
    #logger.debug(f"below_threshold: {below_threshold}")

    # Build up a list of values to check if they are below threshold. This list allows us to easily look
    # forward in the below_threshold array.
    values_to_check = []
    for i in range(limit_of_number_of_values_below_threshold):
        # Basically, this gives us (for limit_of_number_of_values_below_threshold = 4):
        # below_threshold[0:-3], below_threshold[1:-2], below_threshold[2:-1], below_threshold[3:None]
        values_to_check.append(
            below_threshold[i:-(limit_of_number_of_values_below_threshold - 1 - i) or None]
        )

    logger.debug(f"values_to_check: {values_to_check}")
    logger.debug(f"hist length: {len(hist_to_check.x)}, moving avg length: {len(moving_average)}, l: {[len(v) for v in values_to_check]}")

    # Determine the index where the limit_of_number_of_values_below_threshold bins are consequentially below the threshold.
    for i, values in enumerate(zip(*values_to_check)):
        # True if below threshold, so check if not True.
        above_threshold = [not value for value in values]
        # We require the values to go above the moving average threshold at least once.
        if any(above_threshold):
            logger.debug(f"Found bin i {i} above threshold with moving average: {moving_average[i]}, hist value: {hist_to_check.y[i]}")
            found_at_least_one_bin_above_threshold = True

        # All values from which we are looking ahead must be below the threshold to consider the index
        # as below threshold.
        if found_at_least_one_bin_above_threshold and all(np.invert(above_threshold)):
            # The previous outlier removal implementation used a moving average centered on a value
            # (ie. checked arr[-2 + current_index: current_index + 3]). Thus, we need to shift the
            # cut_index that we assign by limit_of_number_of_values_below_threshold / 2 for the index where
            # we have found all values below the threshold.
            logger.debug(f"i at founding cut_index: {i} with moving_average: {moving_average[i]}, hist value: {hist_to_check.y[i]}")
            cut_index = i + limit_of_number_of_values_below_threshold // 2
            # NOTE: ROOT histograms are 1 indexed, so we add another 1.
            # TODO: I don't think this value is quite right because values isn't the full length of the hist.
            cut_index += 1
            logger.debug(f"hist at cut_index: {hist_to_check.y[cut_index]}, cut_index minus slice: {hist_to_check.y[cut_index - 4: cut_index]}")
            break
    else:
        # We never hit the break statement.
        cut_index = -1

    return cut_index

#def _determine_outliers_for_moving_avreage(moving_average: Sequence[float], ) -> int:

def _remove_outliers_from_hist(hist: Hist, outliers_start_index: int, particle_level_axis: T_ParticleLevelAxis) -> None:
    """ Remove outliers from a given histogram.

    Args:
        hist: Histogram to check for outliers.
        outliers_start_index: Index in the truth axis where outliers begin.
        particle_level_axis: Identifies the particle level axis.
    Returns:
        None. The histogram is modified in place.
    """
    # Use on TH1, TH2, and TH3 since we don't start removing immediately, but instead only after the limit
    if outliers_start_index > 0:
        #logger.debug("Removing outliers")
        # Check for values above which they should be removed by translating the global index
        x = ctypes.c_int(0)
        y = ctypes.c_int(0)
        z = ctypes.c_int(0)
        # Maps axis to valaues
        # This is kind of dumb, but it works.
        outliers_removal_axis_values: Dict[T_ParticleLevelAxis, ctypes.c_int] = {
            projectors.TH1AxisType.x_axis: x,
            projectors.TH1AxisType.y_axis: y,
            projectors.TH1AxisType.z_axis: z,
        }
        for index in range(0, hist.GetNcells()):
            # Get the bin x, y, z from the global bin
            hist.GetBinXYZ(index, x, y, z)
            # Watch out for any problems
            if hist.GetBinContent(index) < hist.GetBinError(index):
                logger.warning(f"Bin content < error. Name: {hist.GetName()}, Bin content: {hist.GetBinContent(index)}, Bin error: {hist.GetBinError(index)}, index: {index}, ({x.value}, {y.value})")
            if outliers_removal_axis_values[particle_level_axis].value >= outliers_start_index:
                #logger.debug("Cutting for index {}. x bin {}. Cut index: {}".format(index, x, cutIndex))
                hist.SetBinContent(index, 0)
                hist.SetBinError(index, 0)
    else:
        logger.info(f"Hist {hist.GetName()} did not have any outliers to cut")

@dataclass
class OutliersRemovalManager:
    particle_level_axis: T_ParticleLevelAxis
    moving_average_threshold: float = field(default = 1.0)

    def run(self, hist: Hist = None, hists: Dict[str, Hist] = None) -> int:
        """ Remove outliers from the given histogram(s).

        Args:
            hist: Histogram to check for outliers. Either this or ``hists`` must be specified.
            hists: Histograms to check for outliers. Either this or ``hist`` must be specified.
            limit: Cut off under which we consider the moving average to be 0.
            reference_index: External index noting where outliers were removed for other hists
                (and potentially where they should be removed for this hist.)
        Return:
            Bin index value from which the outliers were removed. The histogram(s) is modified in place.
        """
        # Validation
        if hist is None and hists is None:
            raise ValueError("Must specify either a single hist or a sequence of hists.")
        if hist and hists:
            raise ValueError("Cannot specify both a single hist and a sequence of hists.")
        # Convert the hist into a temporary list so that we can use the same code below.
        if hist is not None:
            hists = {"hist": hist}
        # To help mypy typing
        assert hists is not None
        # Final validation
        for h in hists:
            if hasattr(h, "ProjectionND") and hasattr(h, "Projection"):
                raise ValueError("Cannot remove outliers from THn hists. Project to TH3 or lower first.")

        # Keep track of the outliers index for each hist to determine the maximum of the hists
        # that are passed in.
        outliers_indices: List[int] = []
        # Keep track of pre/post median values
        pre_removal_mean = {}
        post_removal_mean = {}
        pre_removal_median = {}
        post_removal_median = {}

        for hist_name, hist in hists.items():
            # Setup
            hist_to_check = _project_to_part_level(hist = hist, particle_level_axis = self.particle_level_axis)

            # Check these values before and after outlier removal.
            (pre_removal_mean[hist_name], pre_removal_median[hist_name]) = _get_mean_and_median(hist_to_check)

            # Determine the index where the outliers begin and then remove them.
            outliers_indices.append(
                _determine_outliers_index(
                    hist = hist_to_check,
                    moving_average_threshold = self.moving_average_threshold,
                )
            )

        outliers_start_index = np.max(outliers_indices)
        logger.debug(f"outliers_start_index: {outliers_start_index}")

        for hist_name, hist in hists.items():
            # Then do the actual outliers removal.
            _remove_outliers_from_hist(
                hist = hist,
                outliers_start_index = outliers_start_index,
                particle_level_axis = self.particle_level_axis
            )

            # Now check the mean and median to see how much they've changed.
            hist_to_check = _project_to_part_level(hist = hist, particle_level_axis = self.particle_level_axis)
            (post_removal_mean[hist_name], post_removal_median[hist_name]) = _get_mean_and_median(hist_to_check)
            mean_percentage_difference = (post_removal_mean[hist_name] - pre_removal_mean[hist_name]) / post_removal_mean[hist_name]
            median_percentage_difference = (post_removal_median[hist_name] - pre_removal_median[hist_name]) / post_removal_median[hist_name]
            # No need to do more than report
            logger.info(f"Hist {hist_name}: pre- vs post- outliers removal mean percentage difference: {mean_percentage_difference}, median percentage difference: {median_percentage_difference}")

        return outliers_start_index
