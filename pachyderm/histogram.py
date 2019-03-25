#!/usr/bin/env python

""" Histogram related classes and functionality.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from dataclasses import dataclass
import logging
import numpy as np
from typing import Any, Dict, Tuple, TypeVar, Union

from pachyderm.typing_helpers import Hist

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

# Typing helpers
_T = TypeVar("_T", bound = "Histogram1D")

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
    def bin_widths(self) -> np.ndarray:
        """ Bin widths calculated from the bin edges.

        Returns:
            Array of the bin widths.
        """
        return self.bin_edges[1:] - self.bin_edges[:-1]

    @property
    def x(self) -> np.ndarray:
        """ The histogram bin centers (``x``).

        This property caches the x value so we don't have to calculate it every time.

        Args:
            None
        Returns:
            Array of center of bins.
        """
        try:
            return self._x
        except AttributeError:
            half_bin_widths = self.bin_widths / 2
            x = self.bin_edges[:-1] + half_bin_widths
            self._x: np.ndarray = x

        return self._x

    def find_bin(self, value: float) -> int:
        """ Find the bin corresponding to the specified value.

        Note:
            Bins are 0-indexed here, while in ROOT they are 1-indexed.

        Args:
            value: Value for which we want want the corresponding bin.
        Returns:
            Bin corresponding to the value.
        """
        # This will return the index position where the value should be inserted.
        # This means that if we have the bin edges [0, 1, 2], and we pass value 1.5, it will return
        # index 2, but we want to return bin 1, so we subtract one from the result.
        # NOTE: By specifying that ``side = "right"``, it find values as arr[i] <= value < arr[i - 1],
        #       which matches the ROOT convention.
        return np.searchsorted(self.bin_edges, value, side = "right") - 1

    def copy(self):
        """ Copies the object.

        In principle, this should be the same as ``copy.deepcopy(...)``, at least when this was written in
        Feb 2019. But ``deepcopy(...)`` often seems to have very bad performance (and perhaps does additional
        implicit copying), so we copy these numpy arrays by hand.
        """
        # We want to copy bin_edges, y, and errors_squared, but not anything else.
        # Namely, we skip _x here. In principle, it wouldn't really be a problem to
        # copy, but there may be other "_" fields that we want to skip later, so we
        # do the right thing now.
        kwargs = {k: np.array(v, copy = True) for k, v in vars(self).items() if not k.startswith("_")}
        return type(self)(**kwargs)

    def counts_in_interval(self,
                           min_value: float = None, max_value: float = None,
                           min_bin: int = None, max_bin: int = None) -> Tuple[float, float]:
        """ Count the number of counts within bins in an interval.

        Note:
            The integration limits could be described as inclusive. This matches the ROOT convention.
            See ``histogram1D._integral(...)`` for further details on how these limits are determined.

        Note:
            The arguments can be mixed (ie. a min bin and a max value), so be careful!

        Args:
            min_value: Minimum value for the integral (we will find the bin which contains this value).
            max_value: Maximum value for the integral (we will find the bin which contains this value).
            min_bin: Minimum bin for the integral.
            max_bin: Maximum bin for the integral.
        Returns:
            (value, error): Integral value, error
        """
        return self._integral(
            min_value = min_value, max_value = max_value,
            min_bin = min_bin, max_bin = max_bin,
            multiply_by_bin_width = False,
        )

    def integral(self,
                 min_value: float = None, max_value: float = None,
                 min_bin: int = None, max_bin: int = None) -> Tuple[float, float]:
        """ Integrate the histogram over the given range.

        Note:
            The integration limits could be described as inclusive. This matches the ROOT convention.
            See ``histogram1D._integral(...)`` for further details on how these limits are determined.

        Note:
            The arguments can be mixed (ie. a min bin and a max value), so be careful!

        Args:
            min_value: Minimum value for the integral (we will find the bin which contains this value).
            max_value: Maximum value for the integral (we will find the bin which contains this value).
            min_bin: Minimum bin for the integral.
            max_bin: Maximum bin for the integral.
        Returns:
            (value, error): Integral value, error
        """
        return self._integral(
            min_value = min_value, max_value = max_value,
            min_bin = min_bin, max_bin = max_bin,
            multiply_by_bin_width = True,
        )

    def _integral(self,
                  min_value: float = None, max_value: float = None,
                  min_bin: int = None, max_bin: int = None,
                  multiply_by_bin_width: bool = False) -> Tuple[float, float]:
        """ Integrate the histogram over the specified range.

        This function provides the underlying implementation of the integral, giving the option to multiply
        by the bin with (in which case, one gets the integral), or not (in which case, one gets the number
        of counts in the range).

        Note:
            Limits of the integral could be described as inclusive. To understand this, consider an example
            where the bin edges are ``[0, 1, 2, 5]``, and we request value limits of ``(1.2, 3.6)``. The limits
            correspond to bins ``(1, 2)``, and therefore the integral will include the values from both bins 1 and 2.
            This matches the ROOT convention, and means that if a user wants the counts in only one bin, they
            should set the upper min and max bins to the same bin.

        Args:
            min_value: Minimum value for the integral (we will find the bin which contains this value).
            max_value: Maximum value for the integral (we will find the bin which contains this value).
            min_bin: Minimum bin for the integral.
            max_bin: Maximum bin for the integral.
            multiply_by_bin_width: If true, we will multiply each value by the bin width. The should be done
                for integrals, but not for counting values in an interval.
        Returns:
            (value, error): Integral value, error
        """
        # Validate arguments
        # Specified both values and bins, which is invalid.
        if min_value is not None and min_bin is not None:
            raise ValueError("Specified both min value and min bin. Only specify one.")
        if max_value is not None and max_bin is not None:
            raise ValueError("Specified both max value and max bin. Only specify one.")

        # Determine the bins from the values
        if min_value is not None:
            min_bin = self.find_bin(min_value)
        if max_value is not None:
            max_bin = self.find_bin(max_value)

        # Help out mypy.
        assert min_bin is not None
        assert max_bin is not None

        # Final validation to ensure that the bins properly ordered, with the min <= max.
        # NOTE: It is valid for the bins to be equal. In that case, we only take values from that single bin.
        if min_bin > max_bin:
            raise ValueError(
                f"Passed min_bin {min_bin} which is greater than the max_bin {max_bin}. The min bin must be smaller."
            )

        # Provide the opportunity to scale by bin width
        widths = np.ones(len(self.y))
        if multiply_by_bin_width:
            widths = self.bin_widths

        # Integrate by summing up all of the bins and the errors.
        # Perform the integral.
        # NOTE: We set the upper limits to + 1 from the found value because we want to include the bin
        #       where the upper limit resides. This matches the ROOT convention. Practically, this means
        #       that if the user wants to integrate over 1 bin, then the min bin and max bin should be the same.
        logger.debug(f"Integrating from {min_bin} - {max_bin + 1}")
        value = np.sum(self.y[min_bin:max_bin + 1] * widths[min_bin:max_bin + 1])
        error_squared = np.sum(self.errors_squared[min_bin:max_bin + 1] * widths[min_bin:max_bin + 1] ** 2)

        return value, np.sqrt(error_squared)

    def __add__(self: _T, other: _T) -> _T:
        """ Handles ``a = b + c.`` """
        new = self.copy()
        new += other
        return new

    def __radd__(self: _T, other: _T) -> _T:
        """ For use with sum(...). """
        if other == 0:
            return self
        else:
            return self + other

    def __iadd__(self: _T, other: _T) -> _T:
        """ Handles ``a += b``. """
        if not np.allclose(self.bin_edges, other.bin_edges):
            raise TypeError(
                f"Binning is different for given histograms."
                f"len(self): {len(self.bin_edges)}, len(other): {len(other.bin_edges)}."
                f"Cannot add!"
            )
        self.y += other.y
        self.errors_squared += other.errors_squared
        return self

    def __sub__(self: _T, other: _T) -> _T:
        """ Handles ``a = b - c``. """
        new = self.copy()
        new -= other
        return new

    def __isub__(self: _T, other: _T) -> _T:
        """ Handles ``a += b``. """
        if not np.allclose(self.bin_edges, other.bin_edges):
            raise TypeError(
                f"Binning is different for given histograms."
                f"len(self): {len(self.bin_edges)}, len(other): {len(other.bin_edges)}."
                f"Cannot subtract!"
            )
        self.y -= other.y
        self.errors_squared += other.errors_squared
        return self

    def __mul__(self: _T, other: _T) -> _T:
        """ Handles ``a = b * c``. """
        new = self.copy()
        new *= other
        return new

    def __imul__(self: _T, other: _T) -> _T:
        """ Handles ``a *= b``. """
        if not np.allclose(self.bin_edges, other.bin_edges):
            raise TypeError(
                f"Binning is different for given histograms."
                f"len(self): {len(self.bin_edges)}, len(other): {len(other.bin_edges)}."
                f"Cannot multiply!"
            )
        # NOTE: We need to calculate the errors_squared first because the depend on the existing y values
        # Errors are from ROOT::TH1::Multiply(const TH1 *h1)
        # NOTE: This is just error propagation, simplified with a = b * c!
        self.errors_squared = self.errors_squared * other.y ** 2 + other.errors_squared * self.y ** 2
        self.y *= other.y
        return self

    def __truediv__(self: _T, other: _T) -> _T:
        """ Handles ``a = b / c``. """
        new = self.copy()
        new /= other
        return new

    def __itruediv__(self: _T, other: _T) -> _T:
        """ Handles ``a /= b``. """
        if not np.allclose(self.bin_edges, other.bin_edges):
            raise TypeError(
                f"Binning is different for given histograms."
                f"len(self): {len(self.bin_edges)}, len(other): {len(other.bin_edges)}."
                f"Cannot divide!"
            )
        # Errors are from ROOT::TH1::Divide(const TH1 *h1)
        # NOTE: This is just error propagation, simplified with the a = b / c!
        # NOTE: We need to calculate the errors_squared first before setting y because the errors depend on
        #       the existing y values
        errors_squared_numerator = self.errors_squared * other.y ** 2 + other.errors_squared * self.y ** 2
        errors_squared_denominator = other.y ** 4
        # NOTE: We have to be a bit clever when we divide to avoid dividing by bins with 0 entries. The
        #       approach taken here basically replaces any divide by 0s with a 0 in the output hist.
        #       For more info, see: https://stackoverflow.com/a/37977222
        self.errors_squared = np.divide(
            errors_squared_numerator, errors_squared_denominator,
            out = np.zeros_like(errors_squared_numerator), where = errors_squared_denominator != 0,
        )
        self.y = np.divide(self.y, other.y, out = np.zeros_like(self.y), where = other.y != 0)
        return self

    def __eq__(self, other):
        """ Check for equality. """
        attributes = [k for k in vars(self) if not k.startswith("_")]
        other_attributes = [k for k in vars(other) if not k.startswith("_")]

        # As a beginning check, they must have the same attributes available.
        if attributes != other_attributes:
            return False

        # All attributes are np arrays, so we compare the arrays using ``np.allclose``
        agreement = [np.allclose(getattr(self, a), getattr(other, a)) for a in attributes]
        # All arrays must agree.
        return all(agreement)

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
        # This excludes underflow and overflow
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
        # Exclude the under/overflow bins
        errors = errors[1:-1]

        return (bin_edges, y, errors)

    @classmethod
    def from_existing_hist(cls, hist: Union[Hist, Any]):
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
            # Convert a histogram containing object -> TH1 or uproot hist.
            # It goes "HistogramContainer".hist -> TH1 or uproot hist
            #logger.debug("Converting HistogramContainer to standard hist")
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

def get_array_from_hist2D(hist: Hist, set_zero_to_NaN: bool = True, return_bin_edges: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

