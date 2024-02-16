""" Functionality related to binned data.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import collections
import itertools
import logging
import operator
import typing
import uuid
from collections.abc import Mapping, Sequence
from functools import reduce
from typing import TYPE_CHECKING, Any

import attrs
import numpy as np
import numpy.typing as npt
import ruamel.yaml

logger = logging.getLogger(__name__)

# Work around typing issues in python 3.6
# If only supporting 3.7+, we can add `from __future__ import annotations` and just use the more detailed definition
if TYPE_CHECKING:
    AxesTupleAttribute = attrs.Attribute["AxesTuple"]
    NPAttribute = attrs.Attribute[npt.NDArray[Any]]
else:
    AxesTupleAttribute = attrs.Attribute
    NPAttribute = attrs.Attribute


@attrs.frozen
class Rebin:
    value: int | npt.NDArray[Any] = attrs.field()


def _axis_bin_edges_converter(value: Any) -> npt.NDArray[Any]:
    """Convert the bin edges input to a numpy array.

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
    # We specify the dtype here just to be safe.
    return np.ravel(np.array(value, dtype=np.float64))


@typing.overload
def find_bin(bin_edges: npt.NDArray[Any], value: float) -> int:
    ...


@typing.overload
def find_bin(bin_edges: npt.NDArray[Any], value: npt.NDArray[Any]) -> npt.NDArray[np.int64]:
    ...


def find_bin(bin_edges: npt.NDArray[Any], value: float | npt.NDArray[Any]) -> int | npt.NDArray[np.int64]:
    """Determine the index position where the value should be inserted.

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
    return np.searchsorted(bin_edges, value, side="right") - 1


def _expand_slice_start_and_stop(axis: Axis, selection: slice) -> tuple[int | None, int | None]:
    """Expand out the start and stop values for a slice.

    Args:
        axis: Axis to apply the slice to.
        selection: Slice to apply to the axis.
    Returns:
        (start, stop). Note that they may be None.
    """
    # Evaluate the selections, expanding the axis values if passed via complex numbers
    start = selection.start
    stop = selection.stop
    if isinstance(start, complex):
        start = int(start.real) + axis.find_bin(start.imag)
    if isinstance(stop, complex):
        stop = int(stop.real) + axis.find_bin(stop.imag)

    return start, stop


@attrs.define(eq=False)
class Axis:
    bin_edges: npt.NDArray[Any] = attrs.field(converter=_axis_bin_edges_converter)
    # Only for caching
    _bin_centers: npt.NDArray[Any] = attrs.field(init=False)

    def __len__(self) -> int:
        """The number of bins."""
        return len(self.bin_edges) - 1

    @property
    def bin_widths(self) -> npt.NDArray[Any]:
        """Bin widths calculated from the bin edges.

        Returns:
            Array of the bin widths.
        """
        res: npt.NDArray[Any] = self.bin_edges[1:] - self.bin_edges[:-1]
        return res

    @property
    def bin_centers(self) -> npt.NDArray[Any]:
        """The axis bin centers (``x`` for 1D).

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
            self._bin_centers: npt.NDArray[Any] = bin_centers

        return self._bin_centers

    def find_bin(self, value: float) -> int:
        """Find the bin corresponding to the specified value.

        For further information, see ``find_bin(...)`` in this module.

        Note:
            Bins are 0-indexed here, while in ROOT they are 1-indexed.

        Args:
            value: Value for which we want want the corresponding bin.
        Returns:
            Bin corresponding to the value.
        """
        return find_bin(self.bin_edges, value)

    def copy(self: Axis) -> Axis:
        """Copies the object.

        In principle, this should be the same as ``copy.deepcopy(...)``, at least when this was written in
        Feb 2020. But ``deepcopy(...)`` often seems to have very bad performance (and perhaps does additional
        implicit copying), so we copy these numpy arrays by hand.
        """
        return type(self)(bin_edges=np.array(self.bin_edges, copy=True))

    def __eq__(self, other: Any) -> bool:
        """Check for equality."""
        if self.bin_edges.shape == other.bin_edges.shape:
            return np.allclose(self.bin_edges, other.bin_edges)
        return False

    def __getitem__(self, selection: int | slice) -> Axis:
        """Select a subset of the axis.

        Args:
            selection: Selection of the axis.

        Returns:
            new axis with the new binning
        """
        # Basic validation
        # If it's just an int, it was probably an accident. Let the user know.
        if isinstance(selection, int):
            _msg = "Passed an integer to getitem. This is a bit ambiguous, so if you want single values (edges or centers), access the bin edges directly."
            raise ValueError(_msg)

        # Evaluate the selections, expanding the axis values if passed via complex numbers
        start, stop = _expand_slice_start_and_stop(self, selection)

        # Handle the step
        step: int | npt.NDArray[np.float64] | Rebin = selection.step
        if isinstance(step, Rebin):
            # Extract the value if it's stored in the object.
            step = step.value

        if isinstance(step, np.ndarray):
            # We have an array. The new axis will be the array, but we need to check that the rebin bin
            # edges match up to the start and stop.
            bin_edges = np.array(step, copy=True)
            # Validation
            if start is not None and bin_edges[0] != self.bin_edges[start]:
                _msg = f"Lower edge doesn't match rebin. index: {start}, value: {self.bin_edges[start]}, rebin: {bin_edges}"
                raise ValueError(_msg)
            if stop is not None and bin_edges[-1] != self.bin_edges[stop]:
                _msg = (
                    f"Upper edge doesn't match rebin. index: {stop}, value: {self.bin_edges[stop]}, rebin: {bin_edges}"
                )
                raise ValueError(_msg)

            # Validate that the new binning lies within the old binning
            # (ie. each new bin edge is contained in the previous binning)
            # NOTE: I first tried this with sets: `not set(bin_edges).issubset(self.bin_edges)`.
            #       However, it fails due to rounding issues (ie. isclose). So we take the approach
            #       described here: https://stackoverflow.com/a/58623261/12907985 . It's less efficient,
            #       but should be good enough.
            # Require that all of the values are close to one value
            if not np.all(
                # Require that at least one value is close for each value in bin_edges
                np.any(
                    # Check isclose for each value of bin_edges
                    np.isclose(bin_edges[:, np.newaxis], self.bin_edges),
                    axis=1,
                )
            ):
                _msg = f"New bin edges ({bin_edges}) aren't a subset of the old binning ({self.bin_edges}). We can't/don't want to handle this..."
                raise ValueError(_msg)

            # From here, we're good, so pass on the bin edges
        else:
            # The step is either an integer step or None. In either case, we can treat it the same.

            # First, if we're not rebinning with an array (where we ignore the passed start and stop and just
            # check for consistency), then we need to ensure that the stop value is consistent.
            if stop is not None:
                # pylint: disable-next=pointless-string-statement
                """
                If we've set an upper limit, we want to add +1 to ensure that the upper edge of the bin that contains
                the value is included. If we do find_bin on a bin edge, it returns the correct value, but in that case,
                we need the +1 because otherwise the slice will end one value too early (due to slicing rules).
                As an example,

                >>> a = binned_data.Axis(np.arange(1, 12))
                >>> a
                Axis(bin_edges=array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.]))
                >>> a.find_bin(11)
                10
                >>> a.bin_edges[:10]
                array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])

                So if we wanted the stop to be at 11, we need the +1 to get the slice to contain 11
                """
                stop = stop + 1

            # Pass in the values into the evaluated slice.
            evaluated_slice = slice(start, stop, step)
            bin_edges = np.array(self.bin_edges[evaluated_slice], copy=True)

        # Finally, we have the bin edges and we can construct the axis
        return type(self)(bin_edges=bin_edges)

    @classmethod
    def to_yaml(
        cls: type[Axis], representer: ruamel.yaml.representer.BaseRepresenter, obj: Axis
    ) -> ruamel.yaml.nodes.MappingNode:
        """Encode YAML representation.

        For some reason, YAML doesn't encode this object properly, so we have to tell it how to do so.

        Args:
            representer: Representation from YAML.
            obj: Axis to be converted to YAML.
        Returns:
            YAML representation of the Axis object.
        """
        # We want to include a serialization version so we can do a schema evolution later if necessary
        representation = representer.represent_mapping(
            f"!{cls.__name__}", {"bin_edges": obj.bin_edges, "serialization_version": 1}
        )

        # Finally, return the represented object.
        return representation  # noqa: RET504

    @classmethod
    def from_yaml(
        cls: type[Axis],
        constructor: ruamel.yaml.constructor.BaseConstructor,
        data: ruamel.yaml.nodes.MappingNode,
    ) -> Axis:
        """Decode YAML representation.

        For some reason, YAML doesn't encode this object properly, so we have to tell it how to do so.

        Args:
            constructor: Constructor from the YAML object.
            data: YAML mapping node representing the Axis object.
        Returns:
            The Axis object constructed from the YAML specified values.
        """
        kwargs = {constructor.construct_object(k): constructor.construct_object(v) for k, v in data.value}
        serialization_version = kwargs.pop("serialization_version", None)
        # Specialize for each version
        if serialization_version == 1:
            return cls(**kwargs)

        _msg = f"Unknown serialization version {serialization_version} for {cls.__name__}"
        raise ValueError(_msg)


class AxesTuple(tuple[Axis, ...]):
    @property
    def bin_edges(self) -> tuple[npt.NDArray[Any], ...]:
        return tuple(a.bin_edges for a in self)

    @property
    def bin_widths(self) -> tuple[npt.NDArray[Any], ...]:
        return tuple(a.bin_widths for a in self)

    @property
    def bin_centers(self) -> tuple[npt.NDArray[Any], ...]:
        return tuple(a.bin_centers for a in self)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(a) for a in self)

    @classmethod
    def from_axes(
        cls: type[AxesTuple], axes: Axis | Sequence[Axis] | npt.NDArray[Any] | Sequence[npt.NDArray[Any]]
    ) -> AxesTuple:
        values = axes
        # Convert to a list if necessary
        # Ideally, we want to check for anything that isn't a collection, and convert it to one if it's not.
        # However, this is not entirely straightforward because a numpy array is a collection. So in the case of
        # a numpy array, we we to wrap it in a list if it's one dimensional. This check is as general as possible,
        # but if it becomes problematic, we can instead use the more specific:
        # if isinstance(axes, (Axis, np.ndarray)):
        if not isinstance(values, collections.abc.Iterable) or (isinstance(values, np.ndarray) and values.ndim == 1):
            values = [axes]  # type: ignore[assignment]
        # Help out mypy
        assert isinstance(values, collections.abc.Iterable)
        return cls([Axis(a) for a in values])

    def __eq__(self, other: Any) -> bool:
        """Check for equality."""
        if other:
            return all(a == b for a, b in itertools.zip_longest(self, other))
        return False

    @classmethod
    def to_yaml(
        cls: type[AxesTuple], representer: ruamel.yaml.representer.BaseRepresenter, obj: AxesTuple
    ) -> ruamel.yaml.nodes.MappingNode:
        """Encode YAML representation.

        For some reason, YAML doesn't encode this object properly, so we have to tell it how to do so.

        We encode a mapping with the tuple stored in a sequence, as well as the serialization version.

        Args:
            representer: Representation from YAML.
            data: AxesTuple to be converted to YAML.
        Returns:
            YAML representation of the AxesTuple object.
        """
        # We want to include a serialization version so we can do a schema evolution later if necessary
        representation = representer.represent_mapping(
            f"!{cls.__name__}", {"obj": list(obj), "serialization_version": 1}
        )

        # Finally, return the represented object.
        return representation  # noqa: RET504

    @classmethod
    def from_yaml(
        cls: type[AxesTuple],
        constructor: ruamel.yaml.constructor.BaseConstructor,
        data: ruamel.yaml.nodes.MappingNode,
    ) -> AxesTuple:
        """Decode YAML representation.

        For some reason, YAML doesn't encode this object properly, so we have to tell it how to do so.

        Args:
            constructor: Constructor from the YAML object.
            node: YAML mapping node representing the AxesTuple object.
        Returns:
            The AxesTuple object constructed from the YAML specified values.
        """
        # Setup
        stored_mapping = {constructor.construct_object(k): v for k, v in data.value}
        serialization_version_node = stored_mapping.pop("serialization_version", None)

        # Validation + construction
        if serialization_version_node:
            serialization_version = constructor.construct_object(serialization_version_node)
        else:
            _msg = f"Unable to retrieve serialization version for {cls.__name__}"
            raise ValueError(_msg)

        # Specialize for each version
        axes = []
        if serialization_version == 1:
            # Extract relevant values
            stored_obj_data = stored_mapping["obj"]
            for axis_data in stored_obj_data.value:
                axes.append(constructor.construct_object(axis_data))
            return cls(axes)

        _msg = f"Unknown serialization version {serialization_version} for {cls.__name__}"
        raise ValueError(_msg)


def _axes_tuple_from_axes_sequence(
    axes: Axis | Sequence[Axis] | npt.NDArray[Any] | Sequence[npt.NDArray[Any]],
) -> AxesTuple:
    """Workaround for mypy issue in creating an AxesTuple from axes.

    Converter class methods are currently not supported by mypy, so we ignore the typing here.
    See: https://github.com/python/mypy/issues/7912. So instead we wrap the call here.

    Args:
        axes: Axes to be stored in the AxesTuple.
    Returns:
        AxesTuple containing the axes.
    """
    return AxesTuple.from_axes(axes)


def _axes_shared_memory_check(instance: BinnedData, attribute_name: str, value: AxesTuple) -> None:
    """Ensure none of the axes share memory for the numpy arrays."""
    # Ensure the axes don't point to one another (which can cause issues when performing operations in place).
    found_shared_memory = False
    for a_i, b_i in itertools.combinations(range(len(value)), 2):
        if np.may_share_memory(value[a_i].bin_edges, value[b_i].bin_edges):
            found_shared_memory = True
            logger.warning(f"Axis at index {a_i} shares memory with axis at index {b_i}. Copying axis {a_i}!")
            value[a_i] = value[a_i].copy()  # type: ignore[index]

    # If we found some shared memory, be certain that we save the modified object
    if found_shared_memory:
        setattr(instance, attribute_name, value)


def _array_length_from_axes(axes: AxesTuple) -> int:
    return reduce(operator.mul, (len(a) for a in axes))


def _validate_axes(instance: BinnedData, attribute: AxesTupleAttribute, value: AxesTuple) -> None:
    array_length = _array_length_from_axes(value)
    for other_name, other_value in [("values", instance.values), ("variances", instance.variances)]:
        if array_length != other_value.size:
            _msg = (
                f"Length of {attribute.name} does not match expected length of the {other_name}."
                f" len({attribute.name}) = {array_length}, expected length from '{other_name}': {len(other_value)}."
            )
            raise ValueError(_msg)


def _validate_arrays(instance: BinnedData, attribute: NPAttribute, value: npt.NDArray[Any]) -> None:
    expected_length = _array_length_from_axes(instance.axes)
    if value.size != expected_length:
        _msg = (
            f"Length of {attribute} does not match expected length."
            f" len({attribute}) = {len(value)}, expected length: {expected_length}."
        )
        raise ValueError(_msg)


def _shared_memory_check(instance: BinnedData, attribute_name: str, value: npt.NDArray[Any]) -> None:
    # Define this array for convenience in accessing the members. This way, we're less likely to miss
    # newly added members.
    arrays = {
        k: v
        for k, v in attrs.asdict(instance, recurse=False).items()
        if not k.startswith("_") and k != "metadata" and k != "axes" and k != attribute_name
    }
    # Extract the axes to check those arrays too
    arrays.update({f"axis_{i}": v.bin_edges for i, v in enumerate(instance.axes)})
    # Ensure the members don't point to one another (which can cause issues when performing operations in place).
    # Check the other values.
    for other_name, other_value in arrays.items():
        # logger.debug(f"{attribute.name}: Checking {other_name} for shared memory.")
        if np.may_share_memory(value, other_value):
            logger.warning(
                f"Object '{other_name}' shares memory with object '{attribute_name}'. Copying '{attribute_name}'!"
            )
            setattr(instance, attribute_name, value.copy())


def _shape_array_check(instance: BinnedData, attribute_name: str, value: npt.NDArray[Any]) -> None:
    """Ensure that the arrays are shaped the same as the shape expected from the axes."""
    # If we're passed a flattened array, reshape it to follow the shape of the axes.
    # NOTE: One must be a bit careful with this to ensure that the it is formatted as expected.
    #       Especially when converting between ROOT and numpy.
    if value.ndim == 1:
        setattr(instance, attribute_name, value.reshape(instance.axes.shape))
    if instance.axes.shape != value.shape:
        # Protection for if the shapes are reversed.
        if instance.axes.shape == tuple(reversed(value.shape)):
            logger.info(f"Shape of {attribute_name} appears to be reversed. Transposing the array.")
            setattr(instance, attribute_name, value.T)
        else:
            # Otherwise, something is entirely wrong. Just let the user know.
            _msg = f"Shape of {attribute_name} mismatches axes. {attribute_name:}.shape: {value.shape}, axes.shape: {instance.axes.shape}"
            raise ValueError(_msg)


@attrs.define(eq=False)
class BinnedData:
    axes: AxesTuple = attrs.field(
        converter=_axes_tuple_from_axes_sequence,
        validator=[_validate_axes],
    )
    values: npt.NDArray[Any] = attrs.field(
        converter=np.asarray,
        validator=[_validate_arrays],
    )
    variances: npt.NDArray[Any] = attrs.field(
        converter=np.asarray,
        validator=[_validate_arrays],
    )
    metadata: dict[str, Any] = attrs.field(factory=dict)

    def __attrs_post_init__(self) -> None:
        # Validation
        # NOTE: We can't do this in the attrs validators because we can't set the instance value in
        #       the validator anymore
        # Axes
        _axes_shared_memory_check(self, "axes", self.axes)
        # Values and variances
        _shared_memory_check(self, "values", self.values)
        _shape_array_check(self, "values", self.values)
        _shared_memory_check(self, "variances", self.variances)
        _shape_array_check(self, "variances", self.variances)

    @classmethod
    def to_yaml(
        cls: type[BinnedData], representer: ruamel.yaml.representer.BaseRepresenter, obj: BinnedData
    ) -> ruamel.yaml.nodes.MappingNode:
        """Encode YAML representation.

        For some reason, YAML doesn't encode this object properly, so we have to tell it how to do so.

        We encode a mapping with the tuple stored in a sequence, as well as the serialization version.

        Args:
            representer: Representation from YAML.
            data: AxesTuple to be converted to YAML.
        Returns:
            YAML representation of the AxesTuple object.
        """
        # We want to include a serialization version so we can do a schema evolution later if necessary
        representation = representer.represent_mapping(
            f"!{cls.__name__}",
            {
                "axes": obj.axes,
                "values": obj.values,
                "variances": obj.variances,
                "metadata": obj.metadata,
                "serialization_version": 1,
            },
        )

        # Finally, return the represented object.
        return representation  # noqa: RET504

    @classmethod
    def from_yaml(
        cls: type[BinnedData],
        constructor: ruamel.yaml.constructor.BaseConstructor,
        data: ruamel.yaml.nodes.MappingNode,
    ) -> BinnedData:
        """Decode YAML representation.

        For some reason, YAML doesn't encode this object properly, so we have to tell it how to do so.

        Args:
            constructor: Constructor from the YAML object.
            node: YAML mapping node representing the BinnedData object.
        Returns:
            The BinnedData object constructed from the YAML specified values.
        """
        # Setup
        stored_mapping = {constructor.construct_object(k): v for k, v in data.value}
        serialization_version_node = stored_mapping.pop("serialization_version", None)

        # Validation + construction
        if serialization_version_node:
            serialization_version = constructor.construct_object(serialization_version_node)
        else:
            _msg = f"Unable to retrieve serialization version for {cls.__name__}"
            raise ValueError(_msg)

        # Specialize for each version
        if serialization_version == 1:
            # Convert the mapping into the relevant objects
            # Extract relevant values
            stored_data = {
                k: constructor.construct_object(stored_mapping[k])
                # We can blindly loop because we already popped the version
                for k in list(stored_mapping.keys())
            }
            return cls(**stored_data)

        _msg = f"Unknown serialization version {serialization_version} for {cls.__name__}"
        raise ValueError(_msg)

    @property
    def axis(self) -> Axis:
        """Returns the single axis when the binned data is 1D.

        This is just a helper function, but can be nice for one dimensional data.

        Returns:
            The axis.
        """
        if len(self.axes) != 1:
            _msg = f"Calling axis is only valid for one axis. There are {len(self.axes)} axes."
            raise ValueError(_msg)
        return self.axes[0]

    @property
    def errors(self) -> npt.NDArray[Any]:
        res: npt.NDArray[Any] = np.sqrt(self.variances)
        return res

    def copy(self: BinnedData) -> BinnedData:
        """Copies the object.

        In principle, this should be the same as ``copy.deepcopy(...)``, at least when this was written in
        Feb 2020. But ``deepcopy(...)`` often seems to have very bad performance (and perhaps does additional
        implicit copying), so we copy these numpy arrays by hand.
        """
        return type(self)(
            axes=AxesTuple(axis.copy() for axis in self.axes),
            values=np.array(self.values, copy=True),
            variances=np.array(self.variances, copy=True),
            metadata=self.metadata.copy(),
        )

    # TODO: Add integral: Need to devise how best to pass axis limits.
    # TODO: Stats

    def __add__(self: BinnedData, other: BinnedData) -> BinnedData:
        """Handles ``a = b + c.``"""
        new = self.copy()
        new += other
        return new

    def __radd__(self: BinnedData, other: int | BinnedData) -> BinnedData:
        """For use with sum(...)."""
        if other == 0:
            return self
        # Help out mypy
        assert not isinstance(other, int)
        return self + other

    def __iadd__(self: BinnedData, other: BinnedData) -> BinnedData:
        """Handles ``a += b``."""
        if self.axes != other.axes:
            _msg = (
                f"Binning is different for given binned data, so cannot be added!"
                f" len(self.axes): {len(self.axes)}, len(other.axes): {len(other.axes)}."
                f" axes: {self.axes}, other axes: {other.axes}."
            )
            raise TypeError(_msg)
        self.values += other.values
        self.variances += other.variances
        return self

    def __sub__(self: BinnedData, other: BinnedData) -> BinnedData:
        """Handles ``a = b - c``."""
        new = self.copy()
        new -= other
        return new

    def __isub__(self: BinnedData, other: BinnedData) -> BinnedData:
        """Handles ``a -= b``."""
        if self.axes != other.axes:
            _msg = (
                f"Binning is different for given binned data, so cannot be subtracted!"
                f" len(self.axes): {len(self.axes)}, len(other.axes): {len(other.axes)}."
                f" axes: {self.axes}, other axes: {other.axes}."
            )
            raise TypeError(_msg)
        self.values -= other.values
        self.variances += other.variances
        return self

    def __mul__(self: BinnedData, other: BinnedData | npt.NDArray[Any] | float) -> BinnedData:
        """Handles ``a = b * c``."""
        new = self.copy()
        new *= other
        return new

    def __imul__(self: BinnedData, other: BinnedData | npt.NDArray[Any] | float) -> BinnedData:
        """Handles ``a *= b``."""
        if np.isscalar(other) or isinstance(other, np.ndarray):
            # Help out mypy...
            assert isinstance(other, float | int | np.number | np.ndarray)
            # Scale data by a scalar
            self.values *= other
            self.variances *= other**2
        else:
            # Help out mypy...
            assert isinstance(other, type(self))
            # Validation
            if self.axes != other.axes:
                _msg = (
                    f"Binning is different for given binned data, so cannot be multiplied!"
                    f" len(self.axes): {len(self.axes)}, len(other.axes): {len(other.axes)}."
                    f" axes: {self.axes}, other axes: {other.axes}."
                )
                raise TypeError(_msg)
            # NOTE: We need to calculate the errors_squared first because the depend on the existing y values
            # Errors are from ROOT::TH1::Multiply(const TH1 *h1)
            # NOTE: This is just error propagation, simplified with a = b * c!
            self.variances = self.variances * other.values**2 + other.variances * self.values**2
            self.values *= other.values
        return self

    def __truediv__(self: BinnedData, other: BinnedData | npt.NDArray[Any] | float) -> BinnedData:
        """Handles ``a = b / c``."""
        new = self.copy()
        new /= other
        return new

    def __itruediv__(self: BinnedData, other: BinnedData | npt.NDArray[Any] | float) -> BinnedData:
        """Handles ``a /= b``."""
        if np.isscalar(other) or isinstance(other, np.ndarray):
            # Help out mypy...
            assert isinstance(other, float | int | np.number | np.ndarray)
            # Scale data by a scalar
            self *= 1.0 / other
        else:
            # Help out mypy...
            assert isinstance(other, type(self))
            # Validation
            if self.axes != other.axes:
                _msg = (
                    f"Binning is different for given binned data, so cannot be divided!"
                    f" len(self.axes): {len(self.axes)}, len(other.axes): {len(other.axes)}."
                    f" axes: {self.axes}, other axes: {other.axes}."
                )
                raise TypeError(_msg)
            # Errors are from ROOT::TH1::Divide(const TH1 *h1)
            # NOTE: This is just error propagation, simplified with a = b / c!
            # NOTE: We need to calculate the variances first before setting values because the variances depend on
            #       the existing values
            variances_numerator = self.variances * other.values**2 + other.variances * self.values**2
            variances_denominator = other.values**4
            # NOTE: We have to be a bit clever when we divide to avoid dividing by bins with 0 entries. The
            #       approach taken here basically replaces any divide by 0s with a 0 in the output hist.
            #       For more info, see: https://stackoverflow.com/a/37977222
            self.variances = np.divide(
                variances_numerator,
                variances_denominator,
                out=np.zeros_like(variances_numerator),
                where=variances_denominator != 0,
            )
            self.values = np.divide(self.values, other.values, out=np.zeros_like(self.values), where=other.values != 0)
        return self

    def __eq__(self, other: Any) -> bool:
        """Check for equality."""
        attributes = [k for k in attrs.asdict(self, recurse=False) if not k.startswith("_")]
        other_attributes = [k for k in attrs.asdict(other, recurse=False) if not k.startswith("_")]

        # As a beginning check, they must have the same attributes available.
        if attributes != other_attributes:
            return False

        # The values and variances are numpy arrays, so we compare the arrays using ``np.allclose``
        # NOTE: allclose can't handle the axes or the metadata dictionary, so we skip it here
        #       and check it explicitly below.
        keys_to_exclude = ["axes", "metadata"]
        agreement = [
            np.allclose(getattr(self, a), getattr(other, a), equal_nan=True)
            for a in attributes
            if a not in keys_to_exclude
        ]
        # Check axes
        axes_agree = self.axes == other.axes
        # Check metadata
        metadata_agree = self.metadata == other.metadata
        # All arrays and the metadata must agree.
        return all(agreement) and axes_agree and metadata_agree

    @classmethod
    def from_hepdata(cls: type[BinnedData], hist: Mapping[str, Any]) -> list[BinnedData]:  # noqa: ARG003 # pylint: disable=unused-argument
        """Convert (a set) of HEPdata histogram(s) to BinnedData objects.

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
            List of BinnedData constructed from the input HEPdata.
        """
        _msg = "Not yet implemented."
        raise NotImplementedError(_msg)

    @classmethod
    def _from_uproot3(cls: type[BinnedData], hist: Any) -> BinnedData:
        """Convert from uproot read histogram to BinnedData."""
        # All of these methods should excludes underflow and overflow bins
        bin_edges = hist.edges
        values = hist.values
        variances = hist.variances

        metadata: dict[str, Any] = {}

        return cls(axes=bin_edges, values=values, variances=variances, metadata=metadata)

    @classmethod
    def _from_uproot4(cls: type[BinnedData], hist: Any) -> BinnedData:
        """Convert from uproot4 to BinnedData.

        Cannot just use the boost_histogram conversion because it includes flow bins.

        """
        # We explicitly decide to exclude flow bins.
        values = hist.values(flow=False)
        variances = hist.variances(flow=False)
        bin_edges = [axis.edges(flow=False) for axis in hist.axes]

        metadata: dict[str, Any] = {}

        return cls(
            axes=bin_edges,
            values=values,
            variances=variances,
            metadata=metadata,
        )

    @classmethod
    def _from_tgraph(cls: type[BinnedData], hist: Any) -> BinnedData:
        """Convert from uproot4 TGraphAsymmetricErrors to BinnedData.

        We have to make a number of assumptions here, but it seems that it should work
        for well behaved cases.
        """
        bin_centers, values = hist.values(axis="both")
        x_errors_low, y_errors_low = hist.errors(which="low", axis="both")
        x_errors_high, y_errors_high = hist.errors(which="high", axis="both")

        # Aim to reconstruct the bin widths from the x_errors.
        possible_low_bin_edges = bin_centers - x_errors_low
        possible_high_bin_edges = bin_centers + x_errors_high
        if not np.allclose(possible_low_bin_edges[1:], possible_high_bin_edges[:-1]):
            _msg = (
                "Bin edges in graph are inconsistent. Please fix this and try again."
                f"\n\tLow: {possible_low_bin_edges}"
                f"\n\tHigh: {possible_high_bin_edges}"
                f"\n\tValues: {values}"
            )
            raise ValueError(_msg)
        # x errors are consistent, so we can create bin edges from them.
        bin_edges = np.append(possible_low_bin_edges, possible_high_bin_edges[-1])

        # If the errors agree, we can just store them in a standard binned data.
        # Otherwise, we have to use the metadata.
        metadata = {}
        if np.allclose(y_errors_low, y_errors_high):
            variances = y_errors_low**2
        else:
            variances = np.ones_like(y_errors_low)
            metadata["y_errors"] = {"low": y_errors_low, "high": y_errors_high}

        return cls(
            axes=bin_edges,
            values=values,
            variances=variances,
            metadata=metadata,
        )

    @classmethod
    def _from_boost_histogram(cls: type[BinnedData], hist: Any) -> BinnedData:
        """Convert from boost histogram to BinnedData."""
        view = hist.view()
        metadata: dict[str, Any] = {}

        return cls(
            axes=hist.axes.edges,
            values=view.value,
            variances=np.copy(view.variance),
            metadata=metadata,
        )

    @classmethod
    def _from_ROOT(cls: type[BinnedData], hist: Any) -> BinnedData:
        """Convert TH1, TH2, or TH3 histogram to BinnedData.

        Note:
            Under/Overflow bins are excluded.

        """
        # Setup
        # Enable sumw2 if it's not already calculated
        if hist.GetSumw2N() == 0:
            hist.Sumw2(True)
        class_name = hist.ClassName()
        # Determine the number of dimensions
        # TProfile
        if "TProfile" in class_name:
            n_dim = 1 if class_name == "TProfile" else int(class_name[-1])
        else:
            # TH*D
            n_dim = int(class_name[2])
        # If it doesn't match these, then let it throw a ValueError so we know what's going on.

        # Then determine the axes based on the dimensions
        axis_methods = [hist.GetXaxis, hist.GetYaxis, hist.GetZaxis]
        root_axes = axis_methods[:n_dim]

        def get_bin_edges_from_axis(axis: Any) -> npt.NDArray[Any]:
            """Get bin edges from a ROOT hist axis.

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

        # Determine the main values
        # Exclude overflow
        # Axes
        axes = [Axis(get_bin_edges_from_axis(axis())) for axis in root_axes]
        # Values and variances
        # ROOT stores the values in a flat array including underflow and overflow bins,
        # so we need to remove the flow bins, and then appropriately shape the arrays.
        # Specifically, to get the appropriate shape for the arrays, we need to reshape in the opposite
        # order of the axes, and then transpose.
        # NOTE: These operations _do not_ commute.
        shape = tuple(len(a) for a in reversed(axes))
        bins_without_flow_mask = np.array(
            [not (hist.IsBinUnderflow(i) or hist.IsBinOverflow(i)) for i in range(hist.GetNcells())]
        )
        values = np.array([hist.GetBinContent(i) for i in range(hist.GetNcells())])
        values = values[bins_without_flow_mask].reshape(shape).T
        variances = np.array(hist.GetSumw2())
        variances = variances[bins_without_flow_mask].reshape(shape).T

        # Check for a TProfile.
        # In that case we need to retrieve the errors manually because the Sumw2() errors are
        # not the anticipated errors.
        if hasattr(hist, "BuildOptions"):
            errors = np.array([hist.GetBinError(i) for i in range(1, hist.GetXaxis().GetNbins() + 1)])
            # We expected variances (errors squared)
            variances = errors**2
        else:
            # Cross check. If they don't match, something odd has almost certainly occurred.
            # We use lambdas so we don't go beyond the length of the axis unless we're certain
            # that we have that many dimensions.
            first_non_overflow_bin_map = {
                # Using 10 ((by 10) by 10) as an example, to derive the specific values below, and then generalizing.
                # 1
                1: lambda axes: 1,  # noqa: ARG005
                # 12 + 1 = 13
                2: lambda axes: (len(axes[0]) + 2) + 1,
                # 12 * 12 + 12 + 1 = 157
                3: lambda axes: (len(axes[0]) + 2) * (len(axes[1]) + 2) + (len(axes[0]) + 2) + 1,
            }
            first_non_overflow_bin = first_non_overflow_bin_map[len(axes)](axes)  # type: ignore[no-untyped-call]
            if not np.isclose(variances.flatten()[0], hist.GetBinError(first_non_overflow_bin) ** 2):
                _msg = "Sumw2 errors don't seem to represent bin errors!"
                raise ValueError(_msg)

        metadata: dict[str, Any] = {}

        return cls(
            axes=axes,
            values=values,
            variances=variances,
            metadata=metadata,
        )

    @classmethod
    def from_existing_data(
        cls: type[BinnedData], binned_data: Any, return_copy_if_already_converted: bool = True
    ) -> BinnedData:
        """Convert an existing histogram.

        Note:
            Underflow and overflow bins are excluded!

        Args:
            hist (uproot.rootio.TH1* or ROOT.TH1): Histogram to be converted.
        Returns:
            Histogram: Dataclass with x, y, and errors
        """
        # If it's already BinnedData, just return it
        if isinstance(binned_data, cls):
            if return_copy_if_already_converted:
                logger.debug(f"Passed binned data is already a {cls.__name__}. Returning a copy of the object.")
                return binned_data.copy()
            else:  # noqa: RET505
                logger.warning(f"Passed binned data is already a {cls.__name__}. Returning the existing object.")
                return binned_data

        # Now actually deal with conversion from other types.
        # Need to deal with boost histogram first because it now (Feb 2021) has values and variances.
        if hasattr(binned_data, "view"):
            return cls._from_boost_histogram(binned_data)
        # Uproot4: has "_values_variances" for hists, but doesn't appear to for profiles, so we
        # also consider `to_pyroot`, which profiles do have (as should all(?) uproot objects)
        # NOTE: `to_pyroot` should be specific to uproot4+ since it wasn't available in uproot3.
        #       It also shouldn't overlap with ROOT itself.
        if hasattr(binned_data, "_values_variances") or hasattr(binned_data, "to_pyroot"):
            return cls._from_uproot4(binned_data)
        # Uproot3: "values" and "variances" is a proxy for an uproot3 hist. uproot4 hists also have these,
        # so we need to check for uproot4 first
        if hasattr(binned_data, "values") and hasattr(binned_data, "variances"):
            return cls._from_uproot3(binned_data)
        # Next, look for TGraphs
        if hasattr(binned_data, "values") and hasattr(binned_data, "errors"):
            return cls._from_tgraph(binned_data)

        # Fall back to handling a traditional ROOT hist.
        return cls._from_ROOT(binned_data)

    # Convert to other formats.
    def to_ROOT(self, copy: bool = True) -> Any:
        """Convert into a ROOT histogram.

        NOTE:
            This is a lossy operation because there is nowhere to store metadata is in the ROOT hist.

        Args:
            copy: Copy the arrays before assigning them. The ROOT hist may be able to view the array memory,
                such that modifications in one would affect the other. Be extremely careful, as that can have
                unexpected side effects! So only disable with a very good reason. Default: True.
        Returns:
            ROOT histogram containing the data.
        """
        try:
            import ROOT  # pyright: ignore [reportMissingImports]

            # NOTE: This is really important to avoid a deadlock (appears to be on taking the gil according to lldb).
            #       In principle, it's redundant after the first import, but calling anything on the ROOT module deadlocks
            #       it's really annoying for debugging! So we just always call it.
            ROOT.gROOT.SetBatch(True)
        except ImportError as e:
            _msg = "Unable to import ROOT. Please ensure that ROOT is installed and in your $PYTHONPATH."
            raise RuntimeError(_msg) from e

        # Setup
        # We usually want to be entirely certain that the ROOT arrays are not pointing at the same memory
        # as the current hist, so we make a copy. We basically always want to copy.
        h = self.from_existing_data(self) if copy else self

        unique_name = str(uuid.uuid4())
        name = h.metadata.get("name", unique_name)
        title = h.metadata.get("title", unique_name)
        # Axes need to be of the form: n_bins, bin_edges
        axes = list(itertools.chain.from_iterable((len(axis), axis.bin_edges) for axis in h.axes))

        args = [name, title, *axes]
        if len(h.axes) <= 3:
            h_ROOT = getattr(ROOT, f"TH{len(h.axes)}D")(*args)
        else:
            _msg = f"Asking to create hist with {len(h.axes)} > 3 dimensions."
            raise RuntimeError(_msg)

        # We have to keep track on the bin index by hand, because ROOT.
        # NOTE: The transpose is extremely import! Without it, the arrays aren't in the order
        #       that ROOT expects! ROOT expects for the arrays to increment first through x bins,
        #       then increment the y bin, and iterate over x again, etc. We cast the arrays this via
        #       via a transpose.
        i = 1
        for value, error in zip(h.values.T.flatten(), h.errors.T.flatten(), strict=True):
            # Sanity check.
            if i >= h_ROOT.GetNcells():
                _msg = "Indexing is wrong..."
                raise ValueError(_msg)

            # Need to advance to the next bin that we care about.
            # We don't want to naively increment and continue because then we should histogram values.
            while h_ROOT.IsBinUnderflow(i) or h_ROOT.IsBinOverflow(i):
                h_ROOT.SetBinContent(i, 0)
                h_ROOT.SetBinError(i, 0)
                i += 1

            # Set the content
            h_ROOT.SetBinContent(i, value)
            h_ROOT.SetBinError(i, error)
            i += 1

        return h_ROOT

    def to_boost_histogram(self) -> Any:
        """Convert into a boost-histogram.

        NOTE:
            This is a lossy operation. The metadata is not preserved.

        Returns:
            Boost histogram containing the data.
        """
        try:
            import boost_histogram as bh
        except ImportError as e:
            _msg = "Unable to import boost histogram. Please install it to export to a boost histogram."
            raise RuntimeError(_msg) from e

        # It seems to copy by default, so we don't need to do it ourselves.

        axes = []
        for axis in self.axes:
            # NOTE: We use Variable instead of Regular even if the bin edges are Regular because it allows us to
            #       construct the axes just from the bin edges.
            axes.append(bh.axis.Variable(axis.bin_edges, underflow=False, overflow=False))
        h = bh.Histogram(*axes, storage=bh.storage.Weight())
        # Need to shape the array properly so that it will actually be able to assign to the boost histogram.
        arr = np.zeros(shape=h.view().shape, dtype=h.view().dtype)
        arr["value"] = self.values
        arr["variance"] = self.variances
        h[...] = arr

        return h

    def to_histogram1D(self) -> Any:
        """Convert to a Histogram 1D.

        This is entirely a convenience function. Generally, it's best to stay with BinnedData, but
        a Histogram1D is required in some cases, such as for fitting.

        Returns:
            Histogram1D containing the data.
        """
        # Validation
        if len(self.axes) > 1:
            _msg = f"Can only convert to 1D histogram. Given {len(self.axes)} axes"
            raise ValueError(_msg)

        from pachyderm import histogram

        return histogram.Histogram1D(
            bin_edges=self.axes[0].bin_edges,
            y=self.values,
            errors_squared=self.variances,
        )

    def to_numpy(self) -> tuple[npt.NDArray[Any], ...]:
        """Convert to a numpy histogram.

        Returns:
            Tuple of values, and then axes bin edges.
        """
        # TODO: Check that the values don't need to be transposed or similar.
        return (self.values, *self.axes.bin_edges)

    def __getitem__(self, selection: int | slice) -> BinnedData:
        """Select a subset of data, including rebinning.

        Args:
            selection: Selection of the data.
        Returns:
            Binned data corresponding to the data selection. Note that the user is responsible
            for applying the selections to anything stored in the metadata.
        """
        # Basic validation
        # If it's just an int, it was probably an accident. Let the user know.
        if isinstance(selection, int):
            _msg = "Passed an integer to getitem. This is a bit ambiguous, so if you want single values, access the values directly."
            raise ValueError(_msg)
        if len(self.axes) > 1:
            _msg = "Not yet implemented for more than 1D"
            raise NotImplementedError(_msg)

        # First, determine the new axis. We can just defer that to axes implementation
        new_axis = self.axes[0][selection]

        # Build up map from old binning to new binning
        old_to_new_index = find_bin(new_axis.bin_edges, self.axes[0].bin_centers)
        # Unneeded, but it can be helpful to make this into a true map for debugging purposes.
        # old_to_new_index_helper = dict(zip(range(len(self.axes[0].bin_edges)), find_bin(new_axis.bin_edges, self.axes[0].bin_centers)))

        # Select and rebin the values and variances.
        new_values = _apply_rebin(old_to_new_index=old_to_new_index, values=self.values, n_bins_new_axis=len(new_axis))
        new_variances = _apply_rebin(
            old_to_new_index=old_to_new_index, values=self.variances, n_bins_new_axis=len(new_axis)
        )

        return type(self)(
            axes=[new_axis],
            values=new_values,
            variances=new_variances,
            metadata={},
        )


def _apply_rebin(
    old_to_new_index: npt.NDArray[np.int64], values: npt.NDArray[np.float64], n_bins_new_axis: int
) -> npt.NDArray[np.float64]:
    """Apply rebinning to a set of values based on how the indices should be mapped.

    Note:
        This optionally uses numba if it's available. numba isn't a dependency otherwise, so it seems
        like a pretty heavily addition just for this, especially when performance should be _so_ bad.

    Args:
        old_to_new_index: Array containing the mapping from the old bins to the new bins.
            Note that the map is implicit - it's from the index of the position in old_to_new_index
            (which is the same length as the old binning) to the index in the new binning
            (ie. the values are the bins where the values are to be inserted in the new binning)
        values: Values in the old binning to rebin into the new binning.
        n_bins_new_axis: Number of bins in the output array (ie. in the new binning).
    Returns:
        The values mapping into the new binning.
    """
    # Our strategy here is to sum run_length values from run_start. This makes indexing
    # and keeping track of each current sum much easier.

    # First, find run lengths
    # From: https://stackoverflow.com/a/58540073/12907985
    loc_run_start: npt.NDArray[np.bool_] = np.empty(len(old_to_new_index), dtype=np.bool_)
    loc_run_start[0] = True
    np.not_equal(old_to_new_index[:-1], old_to_new_index[1:], out=loc_run_start[1:])
    run_starts = np.nonzero(loc_run_start)[0]
    run_values = old_to_new_index[loc_run_start]
    run_lengths = np.diff(np.append(run_starts, len(old_to_new_index)))

    # Only use numba if available
    f = _sum_values_for_rebin_numba if _sum_values_for_rebin_numba is not None else _sum_values_for_rebin
    return f(  # type: ignore[no-any-return]
        n_bins_new_axis=n_bins_new_axis,
        values=values,
        run_starts=run_starts,
        run_values=run_values,
        run_lengths=run_lengths,
    )


def _sum_values_for_rebin(
    n_bins_new_axis: int,
    values: npt.NDArray[np.float64],
    run_starts: npt.NDArray[np.float64],
    run_values: npt.NDArray[np.float64],
    run_lengths: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Implementation of summing up values for the rebinning

    This translates the values from the old binning to the new binning.

    Note:
        This is kept as a separate function so we can potentially use it with numba. See interface
        function `_apply_rebin`.

    Args:
        n_bins_new_axis: Number of bins in the output array (ie. in the new binning).
        values: Values in the old binning to rebin into the new binning.
        run_starts: The start of each run in the old binning.
        run_values: The index in the new binning where the run starts.
        run_lengths: The length of each run in the old binning.

    Returns:
        The values mapped into the new binning.
    """

    # Setup
    output = np.zeros(n_bins_new_axis, dtype=values.dtype)

    # For each run length that is a valid
    # NOTE: We skip the `strict` keyword for zip because numba doesn't support it, and we
    #       conditionally compile this function. However, once numba supports it, it should
    #       be added in because these are all expected to be the same length.
    for run_start, v_index, run_length in zip(run_starts, run_values, run_lengths):  # noqa: B905
        # Only sum up values which are valid indices for the output
        # If it's below 0, it's in the underflow. If it's >= to the number of new bins,
        # it's in the overflow. In either case, we ignore them.
        if v_index < 0 or v_index >= n_bins_new_axis:
            continue
        # Sum up all of the values where the new binning index is the same. All
        # of those values are supposed to go into the same bin
        output[v_index] = np.sum(values[run_start : run_start + run_length])

    return output


# Attempt to compile this function using numba if numba is available
_sum_values_for_rebin_numba = None
try:
    import numba  # pyright: ignore [reportMissingImports]
except ImportError:
    numba = None

if numba is not None:
    _sum_values_for_rebin_numba = numba.njit(_sum_values_for_rebin)
