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

class RootOpen:
    """ Very simple helper to open root files. """
    def __init__(self, filename: str, mode: str = "read"):
        import ROOT
        self.filename = filename
        self.mode = mode
        self.f = ROOT.TFile.Open(self.filename, self.mode)

    def __enter__(self):
        if not self.f or self.f.IsZombie():
            raise IOError(f"Failed to open ROOT file '{self.filename}'.")
        return self.f

    def __exit__(self, type, value, traceback):
        # Pass on all of the exceptions, but make sure that the file is closed.
        # NOTE: The file isn't always valid because one has to deal with ROOT,
        #       so we have to explicitly check that is is valid before continuing.
        if self.f:
            self.f.Close()

        # We don't return anything because we always want the exceptions to continue
        # to be raised.

def get_histograms_in_file(filename: str) -> Dict[str, Any]:
    """ Helper function which gets all histograms in a file.

    Args:
        filename: Filename of the ROOT file containing the list.
    Returns:
        Contains hists with keys as their names. Lists are recursively added, mirroring
            the structure under which the hists were stored.
    """
    return get_histograms_in_list(filename = filename)

def get_histograms_in_list(filename: str, list_name: str = None) -> Dict[str, Any]:
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
    hists: dict = {}
    with RootOpen(filename = filename, mode = "READ") as fIn:
        if list_name is not None:
            hist_list = fIn.Get(list_name)
        else:
            hist_list = [obj.ReadObj() for obj in fIn.GetListOfKeys()]

        if not hist_list:
            fIn.ls()
            # Closing this file appears (but is not entirely confirmed) to be extremely important! Otherwise,
            # the memory will leak, leading to ROOT memory issues!
            fIn.Close()
            raise ValueError(f"Could not find list with name \"{list_name}\". Possible names are listed above.")

        # Retrieve objects in the hist list
        for obj in hist_list:
            _retrieve_object(hists, obj)

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
        ROOT.SetOwnership(obj, False)

        # Store the object
        output_dict[obj.GetName()] = obj

    # Recurse over lists
    if isinstance(obj, ROOT.TCollection):
        # Keeping it in order simply makes it easier to follow
        output_dict[obj.GetName()] = {}
        # Iterate over the objects in the collection and recursively store them
        for obj_temp in list(obj):
            _retrieve_object(output_dict[obj.GetName()], obj_temp)

@dataclass
class Histogram1D:
    """ Contains histogram data.

    Note:
        Underflow and overflow bins are excluded!

    Args:
        bin_edges (np.ndarray): The histogram bin edges.
        y (np.ndarray): The histogram bin values.
        errors_squared (np.ndarray): The bin sum weight squared errors.

    Attributes:
        x (np.ndarray): The bin centers.
        y (np.ndarray): The bin values.
        bin_edges (np.ndarray): The bin edges.
        errors (np.ndarray): The bin errors.
        errors_squared (np.ndarray): The bin sum weight squared errors.
    """
    bin_edges: np.ndarray
    y: np.ndarray
    errors_squared: np.ndarray

    @property
    def errors(self) -> np.ndarray:
        return np.sqrt(self.errors_squared)

    @property
    def x(self) -> np.ndarray:
        """ The histogram bin centers (``x``).

        This property caches the x value so we don't have to calculate it every time.
        """
        try:
            return self._x
        except AttributeError:
            bin_widths = (self.bin_edges[1:] - self.bin_edges[:-1]) / 2
            x = self.bin_edges[:-1] + bin_widths
            self._x: np.ndarray = x

        return self._x

    @staticmethod
    def _from_uproot(hist) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Convert a uproot histogram to a set of array for creating a Histogram.

        Note:
            Underflow and overflow bins are excluded!

        Args:
            hist (uproot.hist.TH1*): Input histogram.
        Returns:
            tuple: (x, y, errors) where x is the bin centers, y is the bin values, and
                errors are the sumw2 bin errors.
        """
        # This excluces underflow and overflow
        (y, bin_edges) = hist.numpy()

        # Also retrieve errors from sumw2.
        # If more sophistication is needed, we can modify this to follow the approach to
        # calculating bin errors from TH1::GetBinError()
        errors = hist.variances

        return (bin_edges, y, errors)

    @staticmethod
    def _from_th1(hist) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Convert a TH1 histogram to a Histogram.

        Note:
            Underflow and overflow bins are excluded!

        Args:
            hist (ROOT.TH1): Input histogram.
        Returns:
            tuple: (x, y, errors) where x is the bin centers, y is the bin values, and
                errors are the sumw2 bin errors.
        """
        # Enable sumw2 if it's not already calculated
        if hist.GetSumw2N() == 0:
            hist.Sumw2(True)

        # Don't include overflow
        bin_edges = get_bin_edges_from_axis(hist.GetXaxis())
        # NOTE: The y value and bin error are stored with the hist, not the axis.
        y = np.array([hist.GetBinContent(i) for i in range(1, hist.GetXaxis().GetNbins() + 1)])
        errors = np.array(hist.GetSumw2())
        # Exclude the under/overflow binsov
        errors = errors[1:-1]

        return (bin_edges, y, errors)

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
            (bin_edges, y, errors_squared) = cls._from_uproot(hist)
        else:
            # Handle traditional ROOT hists
            (bin_edges, y, errors_squared) = cls._from_th1(hist)

        return cls(bin_edges = bin_edges, y = y, errors_squared = errors_squared)

def get_array_from_hist2D(hist: Any, set_zero_to_NaN: bool = True, return_bin_edges: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Extract x, y, and bin values from a 2D ROOT histogram.

    Converts the histogram into a numpy array, and suitably processes it for a surface plot
    by removing 0s (which can cause problems when taking logs), and returning a set of (x, y) mesh
    values utilziing either the bin edges or bin centers.

    Note:
        This is a different format than the 1D version!

    Args:
        hist (ROOT.TH2): Histogram to be converted.
        set_zero_to_NaN: If true, set 0 in the array to NaN. Useful with matplotlib so that it will
            ignore the values when plotting. See comments in this function for more details. Default: True.
        return_bin_edges: Return x and y using bin edges instead of bin centers.
    Returns:
        Contains (x values, y values, numpy array of hist data) where (x, y) are values on a
            grid (from np.meshgrid) using the selected bin values.
    """
    # Process the hist into a suitable state
    # NOTE: The shape specific can be somewhat confusing (ie. I would naviely expected to specify the x first.)
    # This says that the ``GetYaxis().GetNbins()`` number of rows and ``GetXaxis().GetNbins()`` number of columns.
    shape = (hist.GetYaxis().GetNbins(), hist.GetXaxis().GetNbins())
    # To keep consistency with the root_numpy 2D hist format, we transpose the final result
    # This format has x values as columns.
    hist_array = np.array([hist.GetBinContent(x) for x in range(1, hist.GetNcells()) if not hist.IsBinUnderflow(x) and not hist.IsBinOverflow(x)])
    # The hist_array was linear, so we need to shape it into our expected 2D values.
    hist_array = hist_array.reshape(shape)
    # Transpose the array to better match expectations
    # In particular, by transposing the array, it means that ``thist_array[1][0]`` gives the 2nd x
    # value (x_index = 1) and the 1st y value (y_index = 1). This is as we would expect. This is also
    # the same convention as used by root_numpy
    hist_array = hist_array.T
    # Set all 0s to nan to get similar behavior to ROOT. In ROOT, it will basically ignore 0s. This is
    # especially important for log plots. Matplotlib doesn't handle 0s as well, since it attempts to
    # plot them and then will throw exceptions when the log is taken.
    # By setting to nan, matplotlib basically ignores them similar to ROOT
    # NOTE: This requires a few special functions later which ignore nan when calculating min and max.
    if set_zero_to_NaN:
        hist_array[hist_array == 0] = np.nan

    if return_bin_edges:
        # Bin edges
        x_bin_edges = get_bin_edges_from_axis(hist.GetXaxis())
        y_bin_edges = get_bin_edges_from_axis(hist.GetYaxis())

        # NOTE: The addition of epsilon to the max is extremely important! Otherwise, the x and y
        #       ranges will be one bin short since ``arange`` is not inclusive. This could also be resolved
        #       by using ``linspace``, but I think this approach is perfectly fine.
        # NOTE: This epsilon is smaller than the one in ``utils`` because we are sometimes dealing
        #       with small times (~ns). The other value is larger because (I seem to recall) that
        #       smaller values didn't always place nice with ROOT, but it is fine here, since we're
        #       working with numpy.
        # NOTE: This should be identical to taking the min and max of the axis using
        #       ``TAxis.GetXmin()`` and ``TAxis.GetXmax()``, but I prefer this approach.
        epsilon = 1e-9
        x_range = np.arange(
            np.amin(x_bin_edges),
            np.amax(x_bin_edges) + epsilon,
            hist.GetXaxis().GetBinWidth(1)
        )
        y_range = np.arange(
            np.amin(y_bin_edges),
            np.amax(y_bin_edges) + epsilon,
            hist.GetYaxis().GetBinWidth(1)
        )
    else:
        # We want an array of bin centers
        x_range = np.array([hist.GetXaxis().GetBinCenter(i) for i in range(1, hist.GetXaxis().GetNbins() + 1)])
        y_range = np.array([hist.GetYaxis().GetBinCenter(i) for i in range(1, hist.GetYaxis().GetNbins() + 1)])

    X, Y = np.meshgrid(x_range, y_range)

    return (X, Y, hist_array)

def get_bin_edges_from_axis(axis) -> np.ndarray:
    """ Get bin edges from a ROOT hist axis.

    Note:
        Doesn't include over- or underflow bins!

    Args:
        axis (ROOT.TAxis): Axis from which the bin edges should be extracted.
    Returns:
        Array containing the bin edges.
    """
    # Don't include over- or underflow bins
    bins = range(1, axis.GetNbins() + 1)
    # Bin edges
    bin_edges = np.empty(len(bins) + 1)
    bin_edges[:-1] = [axis.GetBinLowEdge(i) for i in bins]
    bin_edges[-1] = axis.GetBinUpEdge(axis.GetNbins())

    return bin_edges

