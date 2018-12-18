#!/usr/bin/env python

""" Histogram related classes and functionality.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from dataclasses import dataclass
import logging
import numpy as np
from typing import Any, Dict, Tuple

# Setup logger
logger = logging.getLogger(__name__)

def get_histograms_in_list(filename: str, list_name: str = "AliAnalysisTaskJetH_tracks_caloClusters_clusbias5R2GA") -> Dict[str, Any]:
    """ Get histograms from the file and make them available in a dict.

    Lists are recursively explored, with all lists converted to dictionaries, such that the return
    dictionaries which only contains hists and dictionaries of hists (ie there are no ROOT ``TCollection``
    derived objects).

    Args:
        filename: Filename of the ROOT file containing the list.
        list_name: Name of the list to retrieve.
    Returns:
        Contains hists with keys as their names. Lists are recursively added, mirroring
            the structure under which the hists were stored.
    Raises:
        ValueError: If the list could not be found in the given file.
    """
    import ROOT

    hists: dict = {}
    fIn = ROOT.TFile(filename, "READ")
    hist_list = fIn.Get(list_name)
    if not hist_list:
        fIn.ls()
        raise ValueError(f"Could not find list with name \"{list_name}\". Possible names are listed above.")

    # Retrieve objects in the hist list
    for obj in hist_list:
        _retrieve_object(hists, obj)

    # Cleanup
    fIn.Close()

    return hists

def _retrieve_object(output_dict: Dict[str, Any], obj: Any) -> None:
    """ Function to recursively retrieve histograms from a list in a ROOT file.

    ``SetDirectory(True)`` is applied to TH1 derived hists and python is explicitly given
    ownership of the retrieved objects.

    Args:
        output_dict (dict): Dict under which hists should be stored.
        obj (ROOT.TObject derived): Object(s) to be stored. If it is a collection,
            it will be recursed through.
    Returns:
        None: Changes in the dict are reflected in the output_dict which was passed.
    """
    import ROOT

    # Store TH1 or THn
    if isinstance(obj, ROOT.TH1) or isinstance(obj, ROOT.THnBase):
        # Ensure that it is not lost after the file is closed
        # Only works for TH1
        if isinstance(obj, ROOT.TH1):
            obj.SetDirectory(0)

        # Explicitly note that python owns the object
        # From more on memory management with ROOT and python, see:
        # https://root.cern.ch/root/html/guides/users-guide/PythonRuby.html#memory-handling
        ROOT.SetOwnership(obj, True)

        # Store the objects
        output_dict[obj.GetName()] = obj

    # Recurse over lists
    if isinstance(obj, ROOT.TCollection):
        # Keeping it in order simply makes it easier to follow
        output_dict[obj.GetName()] = {}
        for obj_temp in list(obj):
            _retrieve_object(output_dict[obj.GetName()], obj_temp)

@dataclass
class Histogram1D:
    """ Contains histogram data.

    Attributes:
        x (np.ndarray): The bin centers.
        y (np.ndarray): The bin value.
        errors (np.ndarray): The bin errors.
        errors_squared (np.ndarray): The bin sum weight squared errors.
    """
    x: np.ndarray
    y: np.ndarray
    errors_squared: np.ndarray

    @property
    def errors(self) -> np.ndarray:
        return np.sqrt(self.errors_squared)

    @staticmethod
    def _from_uproot(hist) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Convert a uproot histogram to a set of array for creating a Histogram.

        Note:
            Underflow and overflow bins are excluded! Bins are assumed to be fixed
            size.

        Args:
            hist (uproot.hist.TH1*): Input histogram.
        Returns:
            tuple: (x, y, errors) where x is the bin centers, y is the bin values, and
                errors are the sumw2 bin errors.
        """
        # This excluces underflow and overflow
        (y, edges) = hist.numpy()

        # Assume uniform bin size
        bin_size = (hist.high - hist.low) / hist.numbins
        # Shift all of the edges to the center of the bins
        # (and drop the last value, which is now invalid)
        x = edges[:-1] + bin_size / 2.0

        # Also retrieve errors from sumw2.
        # If more sophistication is needed, we can modify this to follow the approach to
        # calculating bin errors from TH1::GetBinError()
        errors = hist.variances

        return (x, y, errors)

    @staticmethod
    def _from_th1(hist) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Convert a TH1 histogram to a Histogram.

        Note:
            Underflow and overflow bins are excluded! Bins are assumed to be fixed
            size.

        Args:
            hist (ROOT.TH1): Input histogram.
        Returns:
            tuple: (x, y, errors) where x is the bin centers, y is the bin values, and
                errors are the sumw2 bin errors.
        """
        # Enable sumw2 if it's not already calculated
        if hist.GetSumw2N() == 0:
            hist.Sumw2(True)

        x_axis = hist.GetXaxis()
        # Don't include overflow
        x_bins = range(1, x_axis.GetNbins() + 1)
        x = np.array([x_axis.GetBinCenter(i) for i in x_bins])
        # NOTE: The y value and bin error are stored with the hist, not the axis.
        y = np.array([hist.GetBinContent(i) for i in x_bins])
        errors = np.array(hist.GetSumw2())
        # Exclude the under/overflow binsov
        errors = errors[1:-1]

        return (x, y, errors)

    @classmethod
    def from_existing_hist(cls, hist: Any):
        """ Convert an existing histogram.

        Note:
            Underflow and overflow bins are excluded! Bins are assumed to be fixed
            size.

        Args:
            hist (uproot.rootio.TH1* or ROOT.TH1): Histogram to be converted.
        Returns:
            Histogram: Dataclass with x, y, and errors
        """
        try:
            # Convert jet_hadron.base.analysis_objects.HistogramContainer -> TH1 or uproot hist.
            # It goes HistogramContainer.hist -> TH1 or uproot hist
            logger.debug("Converting HistogramContainer to standard hist")
            hist = hist.hist
        except AttributeError:
            # Just use the existing histogram
            pass
        # "values" is a proxy for if we have an uproot hist.
        logger.debug(f"{hist}, {type(hist)}")
        if hasattr(hist, "values"):
            (x, y, errors_squared) = cls._from_uproot(hist)
        else:
            # Handle traditional ROOT hists
            (x, y, errors_squared) = cls._from_th1(hist)

        return cls(x = x, y = y, errors_squared = errors_squared)

def get_array_from_hist2D(hist: Any, set_zero_to_NaN: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Extract the necessary data from the hist.

    Converts the histogram into a numpy array, and suitably processes it for a surface plot
    by removing 0s (which can cause problems when taking logs), and returning the bin centers
    for (X,Y).

    Note:
        This is a different format than the 1D version!

    Args:
        hist (ROOT.TH2): Histogram to be converted.
        set_zero_to_NaN (bool): If true, set 0 in the array to NaN. Useful with matplotlib so that
            it will ignore the values when plotting. See comments in this function for more
            details. Default: True.
    Returns:
        tuple: Contains (x bin centers, y bin centers, numpy array of hist data) where X,Y
            are values on a grid (from np.meshgrid)
    """
    # Process the hist into a suitable state
    shape = (hist.GetXaxis().GetNbins(), hist.GetYaxis().GetNbins())
    # To keep consistency with the root_numpy 2D hist format, we transpose the final result
    # This format has x values as columns.
    hist_array = np.array([hist.GetBinContent(x) for x in range(1, hist.GetNcells()) if not hist.IsBinUnderflow(x) and not hist.IsBinOverflow(x)]).reshape(shape).T
    # Set all 0s to nan to get similar behavior to ROOT. In ROOT, it will basically ignore 0s. This is
    # especially important for log plots. Matplotlib doesn't handle 0s as well, since it attempts to
    # plot them and then will throw exceptions when the log is taken.
    # By setting to nan, matplotlib basically ignores them similar to ROOT
    # NOTE: This requires a few special functions later which ignore nan when calculating min and max.
    if set_zero_to_NaN:
        hist_array[hist_array == 0] = np.nan

    # We want an array of bin centers
    x_range = np.array([hist.GetXaxis().GetBinCenter(i) for i in range(1, hist.GetXaxis().GetNbins() + 1)])
    y_range = np.array([hist.GetYaxis().GetBinCenter(i) for i in range(1, hist.GetYaxis().GetNbins() + 1)])
    X, Y = np.meshgrid(x_range, y_range)

    return (X, Y, hist_array)

