#!/usr/bin/env python3

""" Functionality related to binned data.

.. codeauthor:: Ramyond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import collections
import logging
import operator
from functools import reduce
from typing import Any, Dict, Sequence, Tuple, Type, TypeVar, Union, cast

import attr
import numpy as np

logger = logging.getLogger(__name__)


def _axis_bin_edges_converter(value: Any) -> np.ndarray:
    """ Convert the bin edges input to a numpy array.

    If an `Axis` is passed, we grab its bin edges.

    Args:
        value: Value to be converted to a numpy array.
    Returns:
        The converted numpy array.
    """
    # Check for self
    if isinstance(value, Axis):
        value = value.bin_edges
    # Ravel to ensure that we have a standard 1D array.
    return np.ravel(np.array(value))

def _np_array_converter(value: Any) -> np.ndarray:
    """ Convert the given value to a numpy array.

    Normally, we would just use np.array directly as the converter function. However, mypy will complain if
    the converter is untyped. So we add (trivial) typing here.  See: https://github.com/python/mypy/issues/6172.

    Args:
        value: Value to be converted to a numpy array.
    Returns:
        The converted numpy array.
    """
    return np.array(value)

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

_T_Axis = TypeVar("_T_Axis", bound="Axis")

@attr.s(eq=False)
class Axis:
    bin_edges: np.ndarray = attr.ib(converter=_axis_bin_edges_converter)

    def __len__(self) -> int:
        """ The number of bins. """
        return len(self.bin_edges) - 1

    @property
    def bin_widths(self) -> np.ndarray:
        """ Bin widths calculated from the bin edges.

        Returns:
            Array of the bin widths.
        """
        return self.bin_edges[1:] - self.bin_edges[:-1]

    @property
    def bin_centers(self) -> np.ndarray:
        """ The axis bin centers (``x`` for 1D).

        This property caches the values so we don't have to calculate it every time.

        Args:
            None
        Returns:
            Array of center of bins.
        """
        try:
            return self._bin_centers
        except AttributeError:
            half_bin_widths = self.bin_widths / 2
            bin_centers = self.bin_edges[:-1] + half_bin_widths
            self._bin_centers: np.ndarray = bin_centers

        return self._bin_centers

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

    def copy(self: _T_Axis) -> _T_Axis:
        """ Copies the object.

        In principle, this should be the same as ``copy.deepcopy(...)``, at least when this was written in
        Feb 2020. But ``deepcopy(...)`` often seems to have very bad performance (and perhaps does additional
        implicit copying), so we copy these numpy arrays by hand.
        """
        return type(self)(
            bin_edges = np.array(self.bin_edges, copy=True)
        )

    def __eq__(self, other: Any) -> bool:
        """ Check for equality. """
        return cast(bool, np.allclose(self.bin_edges, other.bin_edges))


T_AxesTuple = TypeVar("T_AxesTuple", bound="AxesTuple")

class AxesTuple(Tuple[Axis, ...]):
    @property
    def bin_edges(self) -> Tuple[np.ndarray, ...]:
        return tuple(a.bin_edges for a in self)

    @property
    def bin_widths(self) -> Tuple[np.ndarray, ...]:
        return tuple(a.bin_widths for a in self)

    @property
    def bin_centers(self) -> Tuple[np.ndarray, ...]:
        return tuple(a.bin_centers for a in self)

    @classmethod
    def from_axes(cls: Type[T_AxesTuple], axes: Union[Axis, Sequence[Axis], np.ndarray, Sequence[np.ndarray]]) -> T_AxesTuple:
        values = axes
        # Convert to a list if necessary
        # Ideally, we want to check for anything that isn't a collection, and convert it to one if it's not.
        # However, this is not entirely straightforward because a numpy array is a collection. So in the case of
        # a numpy array, we we to wrap it in a list if it's one dimensional. This check is as general as possible,
        # but if it becomes problematic, we can instead use the more specific:
        # if isinstance(axes, (Axis, np.ndarray)):
        if not isinstance(values, collections.abc.Iterable) or (isinstance(values, np.ndarray) and values.ndim == 1):
            values = [axes]
        # Help out mypy
        assert isinstance(values, collections.abc.Iterable)
        return cls([Axis(a) for a in values])

    def __eq__(self, other: Any) -> bool:
        """ Check for equality. """
        return all(a == b for a, b in zip(self, other))

def _axes_tuple_from_axes_sequence(axes: Union[Axis, Sequence[Axis], np.ndarray, Sequence[np.ndarray]]) -> AxesTuple:
    """ Workaround for mypy issue in creating an AxesTuple from axes.

    Converter class methods are currently not supported by mypy, so we ignore the typing here.
    See: https://github.com/python/mypy/issues/7912. So instead we wrap the call here.

    Args:
        axes: Axes to be stored in the AxesTuple.
    Returns:
        AxesTuple containing the axes.
    """
    return AxesTuple.from_axes(axes)

T_BinnedData = TypeVar("T_BinnedData", bound = "BinnedData")

def _array_length_from_axes(axes: AxesTuple) -> int:
    return reduce(operator.mul, (len(a) for a in axes))

def _validate_axes(instance: T_BinnedData, attribute: attr.Attribute[AxesTuple], value: AxesTuple) -> None:
    array_length = _array_length_from_axes(value)
    for other_name, other_value in [("values", instance.values), ("variances", instance.variances)]:
        if array_length != other_value.size:
            raise ValueError(
                f"Length of {attribute.name} does not match expected length of the {other_name}."
                f" len({attribute.name}) = {array_length}, expected length from '{other_name}': {len(other_value)}."
            )

def _validate_arrays(instance: T_BinnedData, attribute: attr.Attribute[np.ndarray], value: np.ndarray) -> None:
    expected_length = _array_length_from_axes(instance.axes)
    if value.size != expected_length:
        raise ValueError(
            f"Length of {attribute} does not match expected length."
            f" len({attribute}) = {len(value)}, expected length: {expected_length}."
        )

def _shared_memory_check(instance: T_BinnedData, attribute: attr.Attribute[np.ndarray], value: np.ndarray) -> None:
    # TODO: This trivially fails for axes.
    # Define this array for convenience in accessing the members. This way, we're less likely to miss
    # newly added members.
    arrays = {k: v for k, v in vars(instance).items() if not k.startswith("_") and k != "metadata" and k != attribute.name}
    # Ensure the members don't point to one another (which can cause issues when performing operations in place).
    # Check the other values.
    for other_name, other_value in arrays.items():
        #logger.debug(f"{attribute.name}: Checking {other_name} for shared memory.")
        if np.may_share_memory(value, other_value):
            logger.warning(f"Object '{other_name}' shares memory with object '{attribute.name}'. Copying '{attribute}'!")
            setattr(instance, attribute.name, value.copy())

_T_BinnedData = TypeVar("_T_BinnedData", bound="BinnedData")

@attr.s(eq=False)
class BinnedData:
    axes: AxesTuple = attr.ib(converter=_axes_tuple_from_axes_sequence, validator=[_shared_memory_check, _validate_axes])
    values: np.ndarray = attr.ib(converter=_np_array_converter, validator=[_shared_memory_check, _validate_arrays])
    variances: np.ndarray = attr.ib(converter=_np_array_converter, validator=[_shared_memory_check, _validate_arrays])
    metadata: Dict[str, Any] = attr.ib(factory = dict)

    def axis(self) -> Axis:
        """ Returns the single axis when the binned data is 1D.

        This is just a helper function, but can be nice for one dimensional data.

        Returns:
            The axis.
        """
        if len(self.axes) != 1:
            raise ValueError("Calling axis is only valid for one axis. There are {len(self.axes)} axes.")
        return self.axes[0]

    @property
    def errors(self) -> np.ndarray:
        return np.sqrt(self.variances)

    def copy(self: _T_BinnedData) -> _T_BinnedData:
        """ Copies the object.

        In principle, this should be the same as ``copy.deepcopy(...)``, at least when this was written in
        Feb 2020. But ``deepcopy(...)`` often seems to have very bad performance (and perhaps does additional
        implicit copying), so we copy these numpy arrays by hand.
        """
        return type(self)(
            axes = AxesTuple(axis.copy() for axis in self.axes),
            values = np.array(self.values, copy = True),
            variances = np.array(self.variances, copy = True),
            metadata = self.metadata.copy()
        )

    # TODO: Add integral: Need to devise how best to pass axis limits.
    # TODO: Stats
    # TODO: Add conversion from other types.

    def __add__(self: _T_BinnedData, other: _T_BinnedData) -> _T_BinnedData:
        """ Handles ``a = b + c.`` """
        new = self.copy()
        new += other
        return new

    def __radd__(self: _T_BinnedData, other: _T_BinnedData) -> _T_BinnedData:
        """ For use with sum(...). """
        if other == 0:
            return self
        else:
            return self + other

    def __iadd__(self: _T_BinnedData, other: _T_BinnedData) -> _T_BinnedData:
        """ Handles ``a += b``. """
        if self.axes != other.axes:
            raise TypeError(
                f"Binning is different for given binned data, so cannot be added!"
                f" len(self.axes): {len(self.axes)}, len(other.axes): {len(other.axes)}."
                f" axes: {self.axes}, other axes: {other.axes}."
            )
        self.values += other.values
        self.variances += other.variances
        return self

    def __sub__(self: _T_BinnedData, other: _T_BinnedData) -> _T_BinnedData:
        """ Handles ``a = b - c``. """
        new = self.copy()
        new -= other
        return new

    def __isub__(self: _T_BinnedData, other: _T_BinnedData) -> _T_BinnedData:
        """ Handles ``a -= b``. """
        if self.axes != other.axes:
            raise TypeError(
                f"Binning is different for given binned data, so cannot be subtracted!"
                f" len(self.axes): {len(self.axes)}, len(other.axes): {len(other.axes)}."
                f" axes: {self.axes}, other axes: {other.axes}."
            )
        self.values -= other.values
        self.variances += other.variances
        return self

    def __mul__(self: _T_BinnedData, other: Union[_T_BinnedData, float]) -> _T_BinnedData:
        """ Handles ``a = b * c``. """
        new = self.copy()
        new *= other
        return new

    def __imul__(self: _T_BinnedData, other: Union[_T_BinnedData, float]) -> _T_BinnedData:
        """ Handles ``a *= b``. """
        if np.isscalar(other) or isinstance(other, np.ndarray):
            # Help out mypy...
            assert isinstance(other, (float, int, np.number, np.ndarray))
            # Scale data by a scalar
            self.values *= other
            self.variances *= (other ** 2)
        else:
            # Help out mypy...
            assert isinstance(other, type(self))
            # Validation
            if self.axes != other.axes:
                raise TypeError(
                    f"Binning is different for given binned data, so cannot be multiplied!"
                    f" len(self.axes): {len(self.axes)}, len(other.axes): {len(other.axes)}."
                    f" axes: {self.axes}, other axes: {other.axes}."
                )
            # NOTE: We need to calculate the errors_squared first because the depend on the existing y values
            # Errors are from ROOT::TH1::Multiply(const TH1 *h1)
            # NOTE: This is just error propagation, simplified with a = b * c!
            self.variances = self.variances * other.values ** 2 + other.variances * self.values ** 2
            self.values *= other.values
        return self

    def __truediv__(self: _T_BinnedData, other: Union[_T_BinnedData, float]) -> _T_BinnedData:
        """ Handles ``a = b / c``. """
        new = self.copy()
        new /= other
        return new

    def __itruediv__(self: _T_BinnedData, other: Union[_T_BinnedData, float]) -> _T_BinnedData:
        """ Handles ``a /= b``. """
        if np.isscalar(other) or isinstance(other, np.ndarray):
            # Help out mypy...
            assert isinstance(other, (float, int, np.number, np.ndarray))
            # Scale data by a scalar
            self *= 1. / other
        else:
            # Help out mypy...
            assert isinstance(other, type(self))
            # Validation
            if self.axes != other.axes:
                raise TypeError(
                    f"Binning is different for given binned data, so cannot be divided!"
                    f" len(self.axes): {len(self.axes)}, len(other.axes): {len(other.axes)}."
                    f" axes: {self.axes}, other axes: {other.axes}."
                )
            # Errors are from ROOT::TH1::Divide(const TH1 *h1)
            # NOTE: This is just error propagation, simplified with a = b / c!
            # NOTE: We need to calculate the variances first before setting values because the variances depend on
            #       the existing values
            variances_numerator = self.variances * other.values ** 2 + other.variances * self.values ** 2
            variances_denominator = other.values ** 4
            # NOTE: We have to be a bit clever when we divide to avoid dividing by bins with 0 entries. The
            #       approach taken here basically replaces any divide by 0s with a 0 in the output hist.
            #       For more info, see: https://stackoverflow.com/a/37977222
            self.variances = np.divide(
                variances_numerator, variances_denominator,
                out = np.zeros_like(variances_numerator), where = variances_denominator != 0,
            )
            self.values = np.divide(self.values, other.values, out = np.zeros_like(self.values), where = other.values != 0)
        return self

    def __eq__(self, other: Any) -> bool:
        """ Check for equality. """
        attributes = [k for k in vars(self) if not k.startswith("_")]
        other_attributes = [k for k in vars(other) if not k.startswith("_")]

        # As a beginning check, they must have the same attributes available.
        if attributes != other_attributes:
            return False

        # The values and variances are numpy arrays, so we compare the arrays using ``np.allclose``
        # NOTE: allclose can't handle the axes or the metadata dictionary, so we skip it here
        #       and check it explicitly below.
        keys_to_exclude = ["axes", "metadata"]
        agreement = [np.allclose(getattr(self, a), getattr(other, a)) for a in attributes if a not in keys_to_exclude]
        # Check axes
        axes_agree = (self.axes == other.axes)
        # Check metadata
        metadata_agree = (self.metadata == other.metadata)
        # All arrays and the metadata must agree.
        return all(agreement) and axes_agree and metadata_agree
