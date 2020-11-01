#!/usr/bin/env python

""" Histogram related classes and functionality.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import collections
import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, ContextManager, Dict, List, Mapping, Optional, Tuple, Type, TypeVar, Union, cast

import numpy as np

from pachyderm.typing_helpers import Axis, Hist, TFile

# Setup logger
logger = logging.getLogger(__name__)

_T_ContextManager = TypeVar("_T_ContextManager")
T_Extraction_Function = Tuple[Union[List[float], np.ndarray], Union[List[float], np.ndarray], Dict[str, Any]]

class RootOpen(ContextManager[_T_ContextManager]):
    """ Very simple helper to open root files. """
    def __init__(self, filename: Union[Path, str], mode: str = "read"):
        import ROOT
        # Valdiate as a path
        self.filename = Path(filename)
        self.mode = mode
        self.f = ROOT.TFile.Open(str(self.filename), self.mode)

    def __enter__(self) -> TFile:
        if not self.f or self.f.IsZombie():
            raise IOError(f"Failed to open ROOT file '{self.filename}'.")
        return self.f

    def __exit__(self, execption_type: Optional[Type[BaseException]],
                 exception_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        """ We want to pass on all raised exceptions, but ensure that the file is always closed. """
        # The typing information is from here:
        # https://github.com/python/mypy/blob/master/docs/source/protocols.rst#context-manager-protocols

        # Pass on all of the exceptions, but make sure that the file is closed.
        # NOTE: The file isn't always valid because one has to deal with ROOT,
        #       so we have to explicitly check that is is valid before continuing.
        if self.f:
            self.f.Close()

        # We don't return anything because we always want the exceptions to continue
        # to be raised.

def get_histograms_in_file(filename: Union[Path, str]) -> Dict[str, Any]:
    """ Helper function which gets all histograms in a file.

    Args:
        filename: Filename of the ROOT file containing the list.
    Returns:
        Contains hists with keys as their names. Lists are recursively added, mirroring
            the structure under which the hists were stored.
    """
    # Validation
    filename = Path(filename)
    return get_histograms_in_list(filename = filename)

def get_histograms_in_list(filename: Union[Path, str], list_name: Optional[str] = None) -> Dict[str, Any]:
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
    # Validation
    filename = Path(filename)

    hists: Dict[str, Any] = {}
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

    if isinstance(obj, ROOT.TDirectory):
        # Keeping it in order simply makes it easier to follow
        output_dict[obj.GetName()] = {}
        # Iterate over the objects in the collection and recursively store them
        for obj_temp in obj.GetListOfKeys():
            _retrieve_object(output_dict[obj.GetName()], obj_temp.ReadObj())

def _extract_values_from_hepdata_dependent_variable(var: Mapping[str, Any]) -> T_Extraction_Function:
    """ Extract values from a HEPdata dependent variable.

    As the simplest useful HEPdata extraction function possible, it retrieves y values, symmetric
    statical errors. Symmetric systematic errors are stored in the metadata.

    Args:
        var: HEPdata dependent variable.
    Returns:
        y values, errors squared, metadata containing the systematic errors.
    """
    values = var["values"]
    hist_values = [val["value"] for val in values]
    # For now, only support symmetric errors.
    hist_stat_errors = []
    hist_sys_errors = []

    for val in values:
        for error in val["errors"]:
            if error["label"] == "stat":
                hist_stat_errors.append(error["symerror"])
            elif "sys" in error["label"]:
                hist_sys_errors.append(error["symerror"])

    # Validate the collected values.
    if len(hist_stat_errors) == 0:
        raise ValueError(
            f"Could not retrieve statistical errors for dependent var {var}.\n"
            f" hist_stat_errors: {hist_stat_errors}"
        )
    if len(hist_values) != len(hist_stat_errors):
        raise ValueError(
            f"Could not retrieve the same number of values and statistical errors for dependent var {var}.\n"
            f" hist_values: {hist_values}\n"
            f" hist_stat_errors: {hist_stat_errors}"
        )
    if len(hist_sys_errors) != 0 and len(hist_sys_errors) != len(hist_stat_errors):
        raise ValueError(
            f"Could not extract the same number of statistical and systematic errors for dependent var {var}.\n"
            f" hist_stat_errors: {hist_stat_errors}\n"
            f" hist_sys_errors: {hist_sys_errors}"
        )

    # Create the histogram
    metadata: Dict[str, Any] = {
        "sys_error": np.array(hist_sys_errors)
    }

    return hist_values, hist_stat_errors, metadata

# Typing helpers
_T = TypeVar("_T", bound = "Histogram1D")

@dataclass
class Histogram1D:
    """ Contains histogram data.

    Note:
        Underflow and overflow bins are excluded!

    When converting from a TH1 (either from ROOT or uproot), additional statistical information will be extracted
    from the hist to enable the calculation of additional properties. The information available is:

    - Total sum of weights (equal to np.sum(self.y), which we store)
    - Total sum of weights squared (equal to np.sum(self.errors_squared), which we store)
    - Total sum of weights * x
    - Total sum of weights * x * x

    Each is a single float value. Since the later two values are unique, they are stored in the metadata.

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
        metadata (dict): Any additional metadata that should be stored with the histogram. Keys are expected to be
            strings, while the values can be anything. For example, could contain systematic errors, etc.
    """
    bin_edges: np.ndarray
    y: np.ndarray
    errors_squared: np.ndarray
    metadata: Dict[str, Any] = field(default_factory = dict)

    def __post_init__(self) -> None:
        """ Perform validation on the inputs. """
        # Define this array for convenience in accessing the members.
        arrays = {k: v for k, v in vars(self).items() if not k.startswith("_") and k != "metadata"}

        # Ensure that they're numpy arrays.
        for name, arr in arrays.items():
            try:
                setattr(self, name, np.array(arr))
            except TypeError as e:
                raise ValueError(
                    f"Arrays must be numpy arrays, but could not convert object {name} of"
                    f" type {type(arr)} to numpy array."
                ) from e

        # Ensure that they're the appropriate length
        if not (len(self.bin_edges) - 1 == len(self.y) == len(self.errors_squared)):
            logger.debug("mis matched")
            raise ValueError(
                f"Length of input arrays doesn't match! Bin edges should be one longer than"
                f" y and errors_squared. Lengths: bin_edges: {len(self.bin_edges)},"
                f" y: {len(self.y)}, errors_squared: {len(self.errors_squared)}"
            )

        # Ensure they don't point to one another (which can cause issues when performing
        # operations in place).
        for (a_name, a), (b_name, b) in itertools.combinations(arrays.items(), 2):
            if np.may_share_memory(a, b):
                logger.warning(f"Object '{b_name}' shares memory with object '{a_name}'. Copying object '{b_name}'!")
                setattr(self, b_name, b.copy())

        # Create stats based on the stored data.
        # Only recalculate if they haven't already been passed in via the metadata.
        calculate_stats = False
        for key in _stats_keys:
            if key not in self.metadata:
                calculate_stats = True
                break
        if calculate_stats:
            self._recalculate_stats()

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

    @property
    def mean(self) -> float:
        """ Mean of values filled into the histogram.

        Calculated in the same way as ROOT and physt.

        Args:
            None.
        Returns:
            Mean of the histogram.
        """
        return binned_mean(self.metadata)

    @property
    def std_dev(self) -> float:
        """ Standard deviation of the values filled into the histogram.

        Calculated in the same way as ROOT and physt.

        Args:
            None.
        Returns:
            Standard deviation of the histogram.
        """
        return binned_standard_deviation(self.metadata)

    @property
    def variance(self) -> float:
        """ Variance of the values filled into the histogram.

        Calculated in the same way as ROOT and physt.

        Args:
            None.
        Returns:
            Variance of the histogram.
        """
        return binned_variance(self.metadata)

    @property
    def n_entries(self) -> float:
        """ The number of entries in the hist.

        Note:
            This value is dependent on the weight. We don't have a weight independent measure like a ROOT hist,
            so this value won't agree with the number of entries from a weighted ROOT hist.
        """
        return cast(float, np.sum(self.y))

    def find_bin(self, value: float) -> int:
        """ Find the bin corresponding to the specified value.

        For further information, see ``find_bin(...)`` in this module.

        Note:
            Bins are 0-indexed here, while in ROOT they are 1-indexed.

        Args:
            value: Value for which we want want the corresponding bin.
        Returns:
            Bin corresponding to the value.
        """
        return find_bin(self.bin_edges, value)

    def copy(self: _T) -> _T:
        """ Copies the object.

        In principle, this should be the same as ``copy.deepcopy(...)``, at least when this was written in
        Feb 2019. But ``deepcopy(...)`` often seems to have very bad performance (and perhaps does additional
        implicit copying), so we copy these numpy arrays by hand.
        """
        # We want to copy bin_edges, y, and errors_squared, but not anything else. Namely, we skip _x here.
        # In principle, it wouldn't really be a problem to copy, but there may be other "_" fields that we
        # want to skip later, so we do the right thing now.
        kwargs = {k: np.array(v, copy = True) for k, v in vars(self).items() if not k.startswith("_") and k != "metadata"}
        # We also want to make an explicit copy of the metadata
        kwargs["metadata"] = self.metadata.copy()
        return type(self)(**kwargs)

    def counts_in_interval(self,
                           min_value: Optional[float] = None, max_value: Optional[float] = None,
                           min_bin: Optional[int] = None, max_bin: Optional[int] = None) -> Tuple[float, float]:
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
                 min_value: Optional[float] = None, max_value: Optional[float] = None,
                 min_bin: Optional[int] = None, max_bin: Optional[int] = None) -> Tuple[float, float]:
        """ Integrate the histogram over the given range.

        Note:
            Be very careful here! The equivalent of `TH1::Integral(...)` is `counts_in_interval(..)`.
            That's because when we multiply by the bin width, we implicitly should be resetting the stats.
            We will still get the right answer in terms of y and errors_squared, but if this result is used
            to normalize the hist, the stats will be wrong. We can't just reset them here because the integral
            doesn't modify the histogram.

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
                  min_value: Optional[float] = None, max_value: Optional[float] = None,
                  min_bin: Optional[int] = None, max_bin: Optional[int] = None,
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

        # We explicitly cast the final result to float to ensure that it doesn't cause any problems
        # with saving the final values to YAML.
        return float(value), float(np.sqrt(error_squared))

    def _recalculate_stats(self: _T) -> None:
        """ Recalculate the hist stats. """
        self.metadata.update(calculate_binned_stats(
            bin_edges=self.bin_edges, y = self.y, weights_squared = self.errors_squared
        ))

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
        # Update stats.
        for key in _stats_keys:
            if key not in self.metadata:
                logger.warning(f"Add: Missing stats {key} in existing hist. Can not update stored stats.")
                continue
            if key not in other.metadata:
                logger.warning(f"Add: Missing stats {key} in other hist. Can not update stored stats.")
                continue
            self.metadata[key] += other.metadata[key]

        return self

    def __sub__(self: _T, other: _T) -> _T:
        """ Handles ``a = b - c``. """
        new = self.copy()
        new -= other
        return new

    def __isub__(self: _T, other: _T) -> _T:
        """ Handles ``a -= b``. """
        if not np.allclose(self.bin_edges, other.bin_edges):
            raise TypeError(
                f"Binning is different for given histograms."
                f"len(self): {len(self.bin_edges)}, len(other): {len(other.bin_edges)}."
                f"Cannot subtract!"
            )
        self.y -= other.y
        self.errors_squared += other.errors_squared
        # According to ROOT, we need to reset stats because we are subtracting. Otherwise, one
        # can get negative variances
        self._recalculate_stats()
        return self

    def _scale_stats(self: _T, scale_factor: float) -> None:
        for key in _stats_keys:
            if key not in self.metadata:
                logger.warning(f"Scaling: Missing stats {key}. Can not update stored stats.")
                continue
            factor = scale_factor
            if key == "_total_sum_w2":
                factor = scale_factor * scale_factor
            self.metadata[key] = self.metadata[key] * factor

    def __mul__(self: _T, other: Union[_T, float]) -> _T:
        """ Handles ``a = b * c``. """
        new = self.copy()
        new *= other
        return new

    def __imul__(self: _T, other: Union[_T, float]) -> _T:
        """ Handles ``a *= b``. """
        if np.isscalar(other) or isinstance(other, np.ndarray):
            # Help out mypy...
            assert isinstance(other, (float, int, np.number, np.ndarray))
            # Scale histogram by a scalar
            self.y *= other
            self.errors_squared *= (other ** 2)
            # Scale stats accordingly. We can only preserve the stats if using a scalar (according to ROOT).
            if np.isscalar(other):
                self._scale_stats(scale_factor = other)
            else:
                self._recalculate_stats()
        else:
            # Help out mypy...
            assert isinstance(other, Histogram1D)
            # Validation
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

            # Recalculate the stats (same as ROOT)
            self._recalculate_stats()

        return self

    def __truediv__(self: _T, other: Union[_T, float]) -> _T:
        """ Handles ``a = b / c``. """
        new = self.copy()
        new /= other
        return new

    def __itruediv__(self: _T, other: Union[_T, float]) -> _T:
        """ Handles ``a /= b``. """
        if np.isscalar(other) or isinstance(other, np.ndarray):
            # Help out mypy...
            assert isinstance(other, (float, int, np.number, np.ndarray))
            # Scale histogram by a scalar
            self *= 1. / other
            # Scale stats accordingly. We can only preserve the stats if using a scalar (according to ROOT).
            if np.isscalar(other):
                self._scale_stats(scale_factor = 1. / other)
            else:
                self._recalculate_stats()
        else:
            # Help out mypy...
            assert isinstance(other, Histogram1D)
            # Validation
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

            # Recalculate the stats (same as ROOT)
            self._recalculate_stats()

        return self

    def __eq__(self, other: Any) -> bool:
        """ Check for equality. """
        attributes = [k for k in vars(self) if not k.startswith("_")]
        other_attributes = [k for k in vars(other) if not k.startswith("_")]

        # As a beginning check, they must have the same attributes available.
        if attributes != other_attributes:
            return False

        # All attributes are np arrays, so we compare the arrays using ``np.allclose``
        # NOTE: allclose can't handle the metadata dict, so we skip it here and check
        #       it explicitly below.
        agreement = [np.allclose(getattr(self, a), getattr(other, a)) for a in attributes if a != "metadata"]
        # Check metadata
        metadata = self.metadata.copy()
        other_metadata = other.metadata.copy()
        metadata_agree = True
        for key in _stats_keys:
            if not np.isclose(metadata.pop(key, None), other_metadata.pop(key, None)):
                metadata_agree = False
                break
        metadata_agree = metadata_agree and (metadata == other_metadata)
        # All arrays and the metadata must agree.
        return all(agreement) and metadata_agree

    @staticmethod
    def _from_uproot(hist: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """ Convert a uproot histogram to a set of array for creating a Histogram.

        Note:
            Underflow and overflow bins are excluded!

        Args:
            hist (uproot.hist.TH1*): Input histogram.
        Returns:
            tuple: (bin_edges, y, errors, metadata) where bin_edges are the bin edges, y is the bin values, and
                errors are the sumw2 bin errors, and metadata is the extracted metadata.
        """
        # This excludes underflow and overflow
        # Also retrieve errors from sumw2.
        # If more sophistication is needed for the errors, we can modify this to follow the approach to
        # calculating bin errors from TH1::GetBinError()
        (y, errors), (bin_edges,) = hist.to_numpy(flow=False, errors=True, dd=True)

        print(dir(hist))
        print(hist.member("fTsumw"))

        # Extract stats information. It will be stored in the metadata.
        metadata = {}
        ## We extract the values directly from the members.
        metadata.update(_create_stats_dict_from_values(
            hist.member("fTsumw"), hist.member("fTsumw2"), hist.member("fTsumwx"), hist.member("fTsumwx2")
        ))

        return (bin_edges, y, errors ** 2, metadata)

    @classmethod
    def from_hepdata(cls: Type[_T], hist: Mapping[str, Any],
                     extraction_function: Callable[[Mapping[str, Any]], T_Extraction_Function] = _extract_values_from_hepdata_dependent_variable
                     ) -> List[_T]:
        """ Convert (a set) of HEPdata histogram(s) to a Histogram1D.

        Will include any information that the extraction function extracts and returns.

        Note:
            This is not included in the ``from_existing_hist(...)`` function because HEPdata files are oriented
            towards potentially containing multiple histograms in a single object. So we just return all of them
            and let the user sort it out.

        Note:
            It only grabs the first independent variable to determining the x axis.

        Args:
            hist: HEPdata input histogram(s).
            extraction_function: Extract values from HEPdata dict to be used to construct a histogram. Default:
                Retrieves y values, symmetric statical errors. Symmetric systematic errors are stored in the metadata.
        Returns:
            List of Histogram1D constructed from the input HEPdata.
        """
        # HEP Data is just a map containing the data.
        if not isinstance(hist, collections.abc.Mapping):
            raise TypeError(
                f"Does not appear to be valid HEPdata. Must pass a map with the HEPdata information. Passed: {hist}"
            )
        histograms = []
        try:
            # We only support one independent variable, so we take the first entry.
            independent_variable = hist["independent_variables"][0]
            # Assumes that the bin edges are continuous.
            bin_edges = [v["low"] for v in independent_variable["values"]]
            # Grab the last upper bin edge.
            bin_edges.append(independent_variable["values"][-1]["high"])

            # We only take the first dependent variable.
            # Loop over the dependent variables. We will create one Histogram1D for each.
            dependent_variables = hist["dependent_variables"]
            for var in dependent_variables:
                y, errors_squared, metadata = extraction_function(var)
                histograms.append(
                    cls(
                        bin_edges = bin_edges,
                        y = y,
                        errors_squared = [err ** 2 for err in errors_squared],
                        metadata = metadata,
                    )
                )

        except IndexError as e:
            raise TypeError(f"Invalid HEPdata histogram {hist}") from e

        for h in histograms:
            # Calculate stats, which we won't have stored in HEPdata.
            h._recalculate_stats()

        return histograms

    @staticmethod
    def _from_th1(hist: Hist) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """ Convert a TH1 histogram to a Histogram.

        Note:
            Underflow and overflow bins are excluded!

        Args:
            hist (ROOT.TH1): Input histogram.
        Returns:
            tuple: (bin_edges, y, errors, metadata) where bin_edges are the bin edges, y is the bin values, and
                errors are the sumw2 bin errors, and metadata is the extracted metadata.
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
        metadata = {}

        # Check for a TProfile.
        # In that case we need to retrieve the errors manually because the Sumw2() errors are
        # not the anticipated errors.
        if hasattr(hist, "BuildOptions"):
            errors = np.array([hist.GetBinError(i) for i in range(1, hist.GetXaxis().GetNbins() + 1)])
            # We expected errors squared
            errors = errors ** 2
        else:
            # Sanity check. If they don't match, something odd has almost certainly occurred.
            if not np.isclose(errors[0], hist.GetBinError(1) ** 2):
                raise ValueError("Sumw2 errors don't seem to represent bin errors!")

            # Retrieve the stats and store them in the metadata.
            # They are useful for calculating histogram properties (mean, variance, etc).
            stats = np.array([0, 0, 0, 0], dtype = np.float64)
            hist.GetStats(np.ctypeslib.as_ctypes(stats))
            # Return values are (each one is a single float):
            # [1], [2], [3], [4]
            # [1]: total_sum_w: Sum of weights (equal to np.sum(y) if unscaled)
            # [2]: total_sum_w2: Sum of weights squared (equal to np.sum(errors_squared) if unscaled)
            # [3]: total_sum_wx: Sum of w*x
            # [4}: total_sum_wx2: Sum of w*x*x
            metadata.update(_create_stats_dict_from_values(*stats))

        return (bin_edges, y, errors, metadata)

    @classmethod
    def from_existing_hist(cls: Type[_T], hist: Union[Hist, Any]) -> _T:
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
        logger.debug(f"{hist}, {type(hist)}")

        # If it's already a histogram, just return a copy
        if isinstance(hist, cls):
            logger.warning(f"Passed hist is already a {cls.__name__}. Returning the existing object.")
            return hist

        # Now actually deal with conversion from other types.
        # "values" is a proxy for if we have an uproot hist.
        if hasattr(hist, "values"):
            (bin_edges, y, errors_squared, metadata) = cls._from_uproot(hist)
        else:
            # Handle traditional ROOT hists
            (bin_edges, y, errors_squared, metadata) = cls._from_th1(hist)

        return cls(bin_edges = bin_edges, y = y, errors_squared = errors_squared, metadata = metadata)

def get_array_from_hist2D(hist: Hist, set_zero_to_NaN: bool = True, return_bin_edges: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Extract x, y, and bin values from a 2D ROOT histogram.

    Converts the histogram into a numpy array, and suitably processes it for a surface plot
    by removing 0s (which can cause problems when taking logs), and returning a set of (x, y) mesh
    values utilizing either the bin edges or bin centers.

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
    # NOTE: The shape specific can be somewhat confusing (ie. I would naively expected to specify the x first.)
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

def get_bin_edges_from_axis(axis: Axis) -> np.ndarray:
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

def find_bin(bin_edges: np.ndarray, value: float) -> int:
    """ Determine the index position where the value should be inserted.

    This is basically ``ROOT.TH1.FindBin(value)``, but it can used for any set of bin_edges.

    Note:
        Bins are 0-indexed here, while in ROOT they are 1-indexed.

    Args:
        bin_edges: Bin edges of the histogram.
        value: Value to find within those bin edges.
    Returns:
        Index of the bin where that value would reside in the histogram.
    """
    # This will return the index position where the value should be inserted.
    # This means that if we have the bin edges [0, 1, 2], and we pass value 1.5, it will return
    # index 2, but we want to return bin 1, so we subtract one from the result.
    # NOTE: By specifying that ``side = "right"``, it find values as arr[i] <= value < arr[i - 1],
    #       which matches the ROOT convention.
    return cast(
        int,
        np.searchsorted(bin_edges, value, side = "right") - 1
    )

_stats_keys = [
    "_total_sum_w", "_total_sum_w2", "_total_sum_wx", "_total_sum_wx2",
]

def _create_stats_dict_from_values(total_sum_w: float, total_sum_w2: float,
                                   total_sum_wx: float, total_sum_wx2: float) -> Dict[str, float]:
    """ Create a statistics dictionary from the provided set of values.

    This is particularly useful for ensuring that the dictionary values are created uniformly.

    Args:
        total_sum_w: Total sum of the weights (ie. the frequencies).
        total_sum_w2: Total sum of the weights squared (ie. sum of Sumw2 array).
        total_sum_wx: Total sum of weights * x.
        total_sum_wx2: Total sum of weights * x * x.
    Returns:
        Statistics dict suitable for storing in the metadata.
    """
    return {
        "_total_sum_w": total_sum_w, "_total_sum_w2": total_sum_w2,
        "_total_sum_wx": total_sum_wx, "_total_sum_wx2": total_sum_wx2,
    }

def calculate_binned_stats(bin_edges: np.array, y: np.array, weights_squared: np.array) -> Dict[str, float]:
    """ Calculate the stats needed to fully determine histogram properties.

    The values are calculated the same way as in ``ROOT.TH1.GetStats(...)``. Recalculating the statistics
    is not ideal because information is lost compared to the information available when filling the histogram.
    In particular, we actual passed x value is used to calculate these values when filling, but we can
    only approximate this with the bin center when calculating these values later. Calculating them here is
    equivalent to calling ``hist.ResetStats()`` before retrieving the stats.

    These results are accessible from the ROOT hist using ``ctypes`` via:

        >>> stats = np.array([0, 0, 0, 0], dtype = np.float64)
        >>> hist.GetStats(np.ctypeslib.as_ctypes(stats))

    Note:
        ``sum_w`` and ``sum_w2`` calculated here are _not_ equal to the ROOT values if the histogram
        has been scaled. This is because the weights don't change even if the histogram has been scaled.
        If the hist stats are reset, it loses this piece of information and has to reconstruct the
        stats from the current frequencies, such that it will then agree with this function.

    Args:
        bin_edges: Histogram bin edges.
        y: Histogram bin frequencies.
        weights_squared: Filling weights squared (ie. this is the Sumw2 array).
    Returns:
        Stats dict containing the newly calculated statistics.
    """
    x = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2
    total_sum_w = np.sum(y)
    total_sum_w2 = np.sum(weights_squared)
    total_sum_wx = np.sum(y * x)
    total_sum_wx2 = np.sum(y * x ** 2)
    return _create_stats_dict_from_values(total_sum_w, total_sum_w2, total_sum_wx, total_sum_wx2)

def binned_mean(stats: Dict[str, float]) -> float:
    """ Mean of values stored in the histogram.

    Calculated in the same way as ROOT and physt.

    Args:
        stats: The histogram statistical properties.
    Returns:
        Mean of the histogram.
    """
    # Validation
    if "_total_sum_wx" not in stats:
        raise ValueError("Sum of weights * x is not available, so we cannot calculate the mean.")
    if "_total_sum_w" not in stats:
        raise ValueError("Sum of weights is not available, so we cannot calculate the mean.")
    # Calculate the mean.
    total_sum_w = stats["_total_sum_w"]
    if total_sum_w > 0:
        return stats["_total_sum_wx"] / total_sum_w
    # Can't divide, so return NaN
    return np.nan  # type: ignore

def binned_standard_deviation(stats: Dict[str, float]) -> float:
    """ Standard deviation of the values filled into the histogram.

    Calculated in the same way as ROOT and physt.

    Args:
        stats: The histogram statistical properties.
    Returns:
        Standard deviation of the histogram.
    """
    return cast(float, np.sqrt(binned_variance(stats)))

def binned_variance(stats: Dict[str, float]) -> float:
    """ Variance of the values filled into the histogram.

    Calculated in the same way as ROOT and physt.

    Args:
        stats: The histogram statistical properties.
    Returns:
        Variance of the histogram.
    """
    # Validation
    if "_total_sum_wx" not in stats:
        raise ValueError("Sum of weights * x is not available, so we cannot calculate the variance.")
    if "_total_sum_wx2" not in stats:
        raise ValueError("Sum of weights * x * x is not available, so we cannot calculate the variance.")
    if "_total_sum_w" not in stats:
        raise ValueError("Sum of weights is not available, so we cannot calculate the mean.")

    # Calculate the variance.
    total_sum_w = stats["_total_sum_w"]
    if total_sum_w > 0:
        return (stats["_total_sum_wx2"] - (stats["_total_sum_wx"] ** 2 / total_sum_w)) / total_sum_w
    # Can't divide, so return NaN
    return np.nan  # type: ignore
