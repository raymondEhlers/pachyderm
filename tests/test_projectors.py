""" Test projector functionality.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""
from __future__ import annotations

import copy
import dataclasses
import enum
import logging
from typing import Any

import pytest

from pachyderm import projectors, typing_helpers, utils

logger = logging.getLogger(__name__)


class SparseAxisLabels(enum.Enum):
    """Defines the relevant axis values for testing the sparse hist."""

    axis_two = 2
    axis_four = 4
    axis_five = 5


@pytest.fixture()
def create_hist_axis_range():
    """Create a HistAxisRange object to use for testing."""
    object_args: dict[str, Any] = {
        "axis_range_name": "z_axis_test_projector",
        "axis_type": projectors.TH1AxisType.y_axis,
        "min_val": lambda x: x,
        "max_val": lambda y: y,
    }
    obj = projectors.HistAxisRange(**object_args)
    # axis_range_name is referred to as name internally, so we rename to that
    object_args["name"] = object_args.pop("axis_range_name")

    return (obj, object_args)


def test_hist_axis_range(create_hist_axis_range):
    """Tests for creating a HistAxisRange object."""
    obj, object_args = create_hist_axis_range

    assert obj.name == object_args["name"]
    assert obj.axis_type == object_args["axis_type"]
    assert obj.min_val == object_args["min_val"]
    assert obj.max_val == object_args["max_val"]

    # Test repr and str to ensure that they are up to date.
    assert repr(
        obj
    ) == "HistAxisRange(name = {name!r}, axis_type = {axis_type}, min_val = {min_val!r}, max_val = {max_val!r})".format(
        **object_args
    )
    assert str(
        obj
    ) == "HistAxisRange: name: {name}, axis_type: {axis_type}, min_val: {min_val}, max_val: {max_val}".format(
        **object_args
    )
    # Assert that the dict is equal so we don't miss anything in the repr or str representations.
    assert obj.__dict__ == object_args


@pytest.mark.parametrize(
    ("axis_type", "axis"),
    [
        (projectors.TH1AxisType.x_axis, "x_axis"),
        (projectors.TH1AxisType.y_axis, "y_axis"),
        (projectors.TH1AxisType.z_axis, "z_axis"),
        (0, "x_axis"),
        (1, "y_axis"),
        (2, "z_axis"),
    ],
    ids=["xAxis", "yAxis", "zAxis", "number for x axis", "number for y axis", "number for z axis"],
)
@pytest.mark.parametrize("hist_to_test", range(3), ids=["1D", "2D", "3D"])
def test_TH1Axis_determination(create_hist_axis_range, axis_type, axis, hist_to_test, test_root_hists):
    """Test TH1 axis determination in the HistAxisRange object."""
    ROOT = pytest.importorskip("ROOT")
    axis_map = {
        "x_axis": ROOT.TH1.GetXaxis,
        "y_axis": ROOT.TH1.GetYaxis,
        "z_axis": ROOT.TH1.GetZaxis,
    }
    axis = axis_map[axis]
    # Get the HistAxisRange object
    obj, object_args = create_hist_axis_range
    # Insert the proper axis type
    obj.axis_type = axis_type
    # Determine the test hist
    hist = dataclasses.astuple(test_root_hists)[hist_to_test]

    # Check that the axis retrieved by the specified function is the same
    # as that retrieved by the HistAxisRange object.
    # NOTE: GetZaxis() (for example) is still valid for a TH1. It is a minimal axis
    #       object with 1 bin. So it is fine to check for equivalence for axes that
    #       don't really make sense in terms of a hist's dimensions.
    assert axis(hist) == obj.axis(hist)


@pytest.mark.parametrize(
    "axis_selection",
    [SparseAxisLabels.axis_two, SparseAxisLabels.axis_four, SparseAxisLabels.axis_five, 2, 4, 5],
    ids=["axis_two", "axis_four", "axis_five", "number for axis one", "number for axis two", "number for axis three"],
)
def test_THn_axis_determination(axis_selection, create_hist_axis_range, test_sparse):
    """Test THn axis determination in the HistAxisRange object."""
    ROOT = pytest.importorskip("ROOT")  # noqa: F841
    # Retrieve sparse.
    sparse, _ = test_sparse
    # Retrieve object and setup.
    obj, object_args = create_hist_axis_range
    obj.axis_type = axis_selection

    axis_value = axis_selection.value if isinstance(axis_selection, enum.Enum) else axis_selection
    assert sparse.GetAxis(axis_value) == obj.axis(sparse)


class TestsForHistAxisRange:
    """Tests for HistAxisRange which require ROOT."""

    @pytest.mark.parametrize(
        ("min_val", "max_val", "min_val_func", "max_val_func", "expected_func"),
        [
            (0, 10, "find_bin_min", "find_bin_max", lambda axis, x, y: axis.SetRangeUser(x, y)),
            (1, 9, "find_bin_min", "find_bin_max", lambda axis, x, y: axis.SetRangeUser(x, y)),
            (
                1,
                None,
                None,
                "n_bins",
                lambda axis, x, y: True,  # noqa: ARG005
            ),  # This is just a no-op. We don't want to restrict the range.
            (0, 7, None, None, lambda axis, x, y: axis.SetRange(x, y)),
        ],
        ids=[
            "0 - 10 with apply_func_to_find_bin with FindBin",
            "1 - 9 (mid bin) with apply_func_to_find_bin with FindBin",
            "1 - Nbins with apply_func_to_find_bin (no under/overflow)",
            "0 - 10 with raw bin value passed apply_func_to_find_bin",
        ],
    )
    def test_apply_range_set(self, min_val, max_val, min_val_func, max_val_func, expected_func, test_sparse):
        """Test apply a range set to an axis via a HistAxisRange object.

        This is intentionally tested against SetRangeUser, so we can be certain that it reproduces
        that selection as expected.

        Note:
            It doesn't matter whether we operate on TH1 or THn, since they both set ranges on TAxis.

        Note:
            This implicitly tests apply_func_to_find_bin, which is fine given how often the two are used
            together (almost always).
        """
        ROOT = pytest.importorskip("ROOT")

        # Setup functions
        function_map = {
            None: lambda x: projectors.HistAxisRange.apply_func_to_find_bin(None, x),
            "find_bin_min": lambda x: projectors.HistAxisRange.apply_func_to_find_bin(
                ROOT.TAxis.FindBin, x + utils.EPSILON
            ),
            "find_bin_max": lambda x: projectors.HistAxisRange.apply_func_to_find_bin(
                ROOT.TAxis.FindBin, x - utils.EPSILON
            ),
            "n_bins": lambda x: projectors.HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.GetNbins),  # noqa: ARG005
        }
        min_val_func = function_map[min_val_func]
        max_val_func = function_map[max_val_func]

        selected_axis = SparseAxisLabels.axis_two
        sparse, _ = test_sparse
        expected_axis = sparse.GetAxis(selected_axis.value).Clone("axis2")
        expected_func(expected_axis, min_val, max_val)

        obj = projectors.HistAxisRange(
            axis_range_name="axis_two_test",
            axis_type=selected_axis,
            min_val=min_val_func(min_val),
            max_val=max_val_func(max_val),
        )
        # Apply the restriction to the sparse.
        obj.apply_range_set(sparse)
        ax = sparse.GetAxis(selected_axis.value)

        # Unfortunately, equality comparison doesn't work for TAxis...
        # GetXmin() and GetXmax() aren't restricted by SetRange(), so instead use GetFirst() and GetLast()
        assert ax.GetFirst() == expected_axis.GetFirst()
        assert ax.GetLast() == expected_axis.GetLast()
        # Sanity check that the overall axis still agrees
        assert ax.GetNbins() == expected_axis.GetNbins()
        assert ax.GetName() == expected_axis.GetName()

    def test_disagreement_with_set_range_user(self, test_sparse):
        """Test the disagreement between SetRange and SetRangeUser when the epsilon shift is not included."""
        ROOT = pytest.importorskip("ROOT")

        # Setup values
        selected_axis = SparseAxisLabels.axis_two
        min_val = 2
        max_val = 8
        sparse, _ = test_sparse
        # Determine expected value (must be first to avoid interfering with applying the axis range)
        expected_axis = sparse.GetAxis(selected_axis.value).Clone("axis2")
        expected_axis.SetRangeUser(min_val, max_val)

        obj = projectors.HistAxisRange(
            axis_range_name="axis_two_test",
            axis_type=selected_axis,
            min_val=projectors.HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.FindBin, min_val),
            max_val=projectors.HistAxisRange.apply_func_to_find_bin(ROOT.TAxis.FindBin, max_val),
        )
        # Apply the restriction to the sparse.
        obj.apply_range_set(sparse)
        ax = sparse.GetAxis(selected_axis.value)

        # Unfortunately, equality comparison doesn't work for TAxis...
        # GetXmin() and GetXmax() aren't restricted by SetRange(), so instead use GetFirst() and GetLast()
        # The lower bin will still agree.
        assert ax.GetFirst() == expected_axis.GetFirst()
        # The upper bin will not.
        assert ax.GetLast() != expected_axis.GetLast()
        # If we subtract a bin (equivalent to including - epsilon), it will agree.
        assert ax.GetLast() - 1 == expected_axis.GetLast()
        # Sanity check that the overall axis still agrees
        assert ax.GetNbins() == expected_axis.GetNbins()
        assert ax.GetName() == expected_axis.GetName()

    @pytest.mark.parametrize(
        ("func", "value", "expected"),
        [(None, 3, 3), ("n_bins", None, 10), ("find_bin", 10 - utils.EPSILON, 5)],
        ids=["Only value", "Func only", "Func with value"],
    )
    def test_retrieve_axis_value(self, func, value, expected, test_sparse):
        """Test retrieving axis values using apply_func_to_find_bin()."""
        ROOT = pytest.importorskip("ROOT")

        # Setup functions
        function_map = {
            "n_bins": ROOT.TAxis.GetNbins,
            "find_bin": ROOT.TAxis.FindBin,
        }
        if func:
            func = function_map[func]
        # Setup objects
        selected_axis = SparseAxisLabels.axis_two
        sparse, _ = test_sparse
        expected_axis = sparse.GetAxis(selected_axis.value)

        assert projectors.HistAxisRange.apply_func_to_find_bin(func, value)(expected_axis) == expected


def find_non_zero_bins(hist) -> list[int]:
    """Helper function to find the non-zero non-overflow bins.

    Args:
        hist (ROOT.TH1): Histogram to check for non-zero bins.
    Returns:
        List of the indices of non-zero bins.
    """
    non_zero_bins = []
    for x in range(1, hist.GetNcells()):
        if hist.GetBinContent(x) != 0 and not hist.IsBinUnderflow(x) and not hist.IsBinOverflow(x):
            logger.debug(f"non-zero bin at {x}")
            non_zero_bins.append(x)

    return non_zero_bins


def setup_hist_axis_range(hist_range: projectors.HistAxisRange) -> projectors.HistAxisRange:
    """Helper function to setup HistAxisRange min and max values.

    This exists so we can avoid explicit ROOT dependence.

    Args:
        hist_range (projectors.HistAxisRange): Range which includes single min and max values which will
            be used in ``apply_func_to_find_bin`` function calls that will replace them.
    Return:
        Updated hist axis range with the initial value passed to ``ROOT.TAxis.FindBin``.
    """
    # Often, ROOT should be imported before this function is called, but we call it here just in case.
    # Plus, this allows us to be lazy in importing ROOT in the calling functions.
    ROOT = pytest.importorskip("ROOT")

    # We don't want to modify the original objects, since we need them to be preserved for other tests.
    hist_range = copy.copy(hist_range)
    hist_range.min_val = projectors.HistAxisRange.apply_func_to_find_bin(
        ROOT.TAxis.FindBin,
        hist_range.min_val + utils.EPSILON,  # type: ignore[operator]
    )
    hist_range.max_val = projectors.HistAxisRange.apply_func_to_find_bin(
        ROOT.TAxis.FindBin,
        hist_range.max_val - utils.EPSILON,  # type: ignore[operator]
    )
    return hist_range


# Convenient access to hist axis ranges.
HistAxisRanges = dataclasses.make_dataclass("HistAxisRanges", ["x_axis", "y_axis", "z_axis"])

# Hist axis ranges
# NOTE: We don't define these in the class because we won't be able to access these variables when
# defining the tests restricted ranges, but with entries.
hist_axis_ranges = HistAxisRanges(
    projectors.HistAxisRange(
        axis_type=projectors.TH1AxisType.x_axis, axis_range_name="an_x_axis", min_val=0.1, max_val=0.8
    ),
    projectors.HistAxisRange(
        axis_type=projectors.TH1AxisType.y_axis, axis_range_name="an_y_axis", min_val=0, max_val=12
    ),
    projectors.HistAxisRange(
        axis_type=projectors.TH1AxisType.z_axis, axis_range_name="an_z_axis", min_val=10, max_val=60
    ),
)
# Restricted ranges with no counts
hist_axis_ranges_without_entries = HistAxisRanges(
    projectors.HistAxisRange(
        axis_type=projectors.TH1AxisType.x_axis, axis_range_name="an_x_axis_no_entries", min_val=0.2, max_val=0.8
    ),
    projectors.HistAxisRange(
        axis_type=projectors.TH1AxisType.y_axis, axis_range_name="an_y_axis_no_entries", min_val=4, max_val=12
    ),
    projectors.HistAxisRange(
        axis_type=projectors.TH1AxisType.z_axis, axis_range_name="an_z_axis_no_entries", min_val=20, max_val=60
    ),
)
# Restricted ranges with counts in some entries
# Focuses only the on y axis.
# This abuses the names of the axes within the named tuple, but it is rather convenient, so we keep it.
hist_axis_ranges_restricted = (
    projectors.HistAxisRange(
        axis_type=projectors.TH1AxisType.y_axis, axis_range_name="an_y_axis_lower", min_val=0, max_val=4
    ),
    projectors.HistAxisRange(
        axis_type=projectors.TH1AxisType.y_axis, axis_range_name="an_y_axis_middle", min_val=4, max_val=8
    ),
    projectors.HistAxisRange(
        axis_type=projectors.TH1AxisType.y_axis, axis_range_name="an_y_axis_upper", min_val=8, max_val=12
    ),
)


@dataclasses.dataclass
class SingleObservable:
    """Test class for single observable projections."""

    hist: Any


def determine_projector_input_args(
    single_observable: bool, hist: typing_helpers.Hist, hist_label: str
) -> tuple[dict[str, Any], SingleObservable, dict[str, typing_helpers.Hist]]:
    """Determine some projector input arguments.

    Note:
        This doesn't cover all arguments. Some additional ones must be specified during the test.

    Args:
        single_observable: True if we are testing with a signal observable.
        hist: Histogram to be projected.
        hist_label: Label for the histogram to be projected.
    Returns:
        Keyword arguments, single_observable, output_observable
    """
    kwdargs: dict[str, Any] = {}
    # These observables have to be defined here so we don't lose reference to them.
    observable = SingleObservable(hist=None)
    output_observable: dict[str, typing_helpers.Hist] = {}

    # The arguments depend on the observable type.
    if single_observable:
        kwdargs["output_observable"] = observable
        kwdargs["output_attribute_name"] = "hist"
        kwdargs["observable_to_project_from"] = hist
    else:
        kwdargs["output_observable"] = output_observable
        kwdargs["observable_to_project_from"] = {hist_label: hist}

    return kwdargs, observable, output_observable


def check_and_get_projection(
    single_observable: bool, observable: SingleObservable, output_observable: dict[str, typing_helpers.Hist]
) -> typing_helpers.Hist:
    """Run basic checks and get the projection.

    Args:
        single_observable: True if we are testing with a signal observable.
        observable: Single observable object which may contain the projection.
        output_observable: Dict which many contain the projection.
    Returns:
        The projected histogram.
    """
    if single_observable:
        assert len(output_observable) == 0
        assert observable.hist is not None
        proj = observable.hist
    else:
        assert len(output_observable) == 1
        assert observable.hist is None
        proj = next(iter(output_observable.values()))

    return proj


class TestProjectorsWithRoot:
    """Tests for projectors for TH1 derived histograms."""

    @pytest.mark.parametrize(
        "single_observable",
        [
            False,
            True,
        ],
        ids=["Dict observable input", "Single observable input"],
    )
    def test_projectors(self, single_observable, test_root_hists):
        """Test creation and basic methods of the projection class."""
        ROOT = pytest.importorskip("ROOT")  # noqa: F841

        # Args
        projection_name_format = "{test} world"
        kwdargs, observable, output_observable = determine_projector_input_args(
            single_observable=single_observable,
            hist=None,
            hist_label="histogram",
        )
        kwdargs["projection_name_format"] = projection_name_format
        kwdargs["projection_information"] = {"test": "Hello"}
        # Create object
        obj = projectors.HistProjector(**kwdargs)

        assert obj.output_attribute_name is ("hist" if single_observable else None)

        # These objects should be overridden so they aren't super meaningful, but we can still
        # test to ensure that they provide the basic functionality that is expected.
        assert obj.projection_name(test="Hello") == projection_name_format.format(test="Hello")  # type: ignore[arg-type]
        assert obj.get_hist(observable=test_root_hists.hist2D) == test_root_hists.hist2D
        assert obj.output_key_name(
            input_key="input_key",
            output_hist=test_root_hists.hist2D,
            projection_name=projection_name_format.format(test="Hello"),
        ) == projection_name_format.format(test="Hello")
        assert (
            obj.output_hist(output_hist=test_root_hists.hist1D, input_observable=test_root_hists.hist2D)
            == test_root_hists.hist1D
        )

        # Checking printing of the projector settings.
        # Add one additional per selection entry so we have something to print.
        obj.additional_axis_cuts.append("my_axis")  # type: ignore[arg-type]
        obj.projection_dependent_cut_axes.append([hist_axis_ranges_without_entries.x_axis])
        obj.projection_axes.append("projection_axis")  # type: ignore[arg-type]
        # It is rather minimal, but that is fine since it is only printed information.
        expected_str = "HistProjector: Projection Information:\n"
        expected_str += f'\tprojection_name_format: "{projection_name_format}"\n'
        expected_str += "\tprojection_information:\n"
        expected_str += "\n".join(["\t\t- " + "Arg: " + str(val) for arg, val in {"test": "Hello"}.items()])
        expected_str += "\n\tadditional_axis_cuts:\n"
        expected_str += "\t\t- my_axis"
        expected_str += "\n\tprojection_dependent_cut_axes:\n"
        expected_str += "\t\t- ['an_x_axis_no_entries']"
        expected_str += "\n\tprojection_axes:\n"
        expected_str += "\t\t- projection_axis"

        assert str(obj) == expected_str

    @pytest.mark.parametrize(
        "single_observable",
        [
            False,
            True,
        ],
        ids=["Dict observable input", "Single observable input"],
    )
    # Other axes:
    # AAC = Additional Axis Cuts
    # PDCA = Projection Dependent Cut Axes
    @pytest.mark.parametrize(
        ("use_PDCA", "additional_cuts", "expected_additional_cuts"),
        [
            (False, None, True),
            (False, hist_axis_ranges.y_axis, True),
            (False, hist_axis_ranges_without_entries.y_axis, False),
            (True, None, True),
            (True, [], True),
            (True, [hist_axis_ranges.y_axis], True),
            (True, [hist_axis_ranges_without_entries.y_axis], False),
            (True, [hist_axis_ranges_restricted[0], hist_axis_ranges_restricted[1]], True),
            (True, [hist_axis_ranges_restricted[1], hist_axis_ranges_restricted[0]], True),
            (True, [hist_axis_ranges_restricted[1], hist_axis_ranges_restricted[2]], False),
        ],
        ids=[
            "No AAC selection",
            "AAC with entries",
            "AAC with no entries",
            "None PDCA",
            "Empty PDCA",
            "PDCA",
            "PDCA with no entries",
            "Disconnected PDCA with entries",
            "Reversed and disconnected PDCA with entries",
            "Disconnected PDCA with no entries",
        ],
    )
    # PA = Projection Axes
    @pytest.mark.parametrize(
        ("projection_axes", "expected_projection_axes"),
        [
            (hist_axis_ranges.x_axis, True),
            (hist_axis_ranges_without_entries.x_axis, False),
        ],
        ids=["PA with entries", "PA without entries"],
    )
    def test_TH2_projection(
        self,
        test_root_hists,
        single_observable,
        use_PDCA,
        additional_cuts,
        expected_additional_cuts,
        projection_axes,
        expected_projection_axes,
    ):
        """Test projection of a TH2 to a TH1."""
        ROOT = pytest.importorskip("ROOT")  # noqa: F841

        # Setup hist ranges
        if additional_cuts:
            if use_PDCA:
                additional_cuts = [setup_hist_axis_range(cut) for cut in additional_cuts]
            else:
                additional_cuts = setup_hist_axis_range(additional_cuts)
        projection_axes = setup_hist_axis_range(projection_axes)
        # Setup projector
        kwdargs: dict[str, Any] = {}
        # These observables have to be defined here so we don't lose reference to them.
        kwdargs, observable, output_observable = determine_projector_input_args(
            single_observable=single_observable,
            hist=test_root_hists.hist2D,
            hist_label="hist2D",
        )
        kwdargs["projection_name_format"] = "hist"
        kwdargs["projection_information"] = {}
        obj = projectors.HistProjector(**kwdargs)

        # Set the projection axes.
        # Using additional cut axes or PDCA is mutually exclusive because we only have one
        # non-projection axis to work with.
        if use_PDCA:
            if additional_cuts is not None:
                # We need to iterate here separately so that we can separate out the cuts
                # for the disconnected PDCAs.
                for axis_set in additional_cuts:
                    obj.projection_dependent_cut_axes.append([axis_set])
        elif additional_cuts is not None:
            obj.additional_axis_cuts.append(additional_cuts)
        obj.projection_axes.append(projection_axes)

        # Perform the projection.
        obj.project()

        # Check the output and get the projection.
        proj = check_and_get_projection(
            single_observable=single_observable,
            observable=observable,
            output_observable=output_observable,
        )
        assert proj.GetName() == "hist"

        logger.debug(f"output_observable: {output_observable}, proj.GetEntries(): {proj.GetEntries()}")

        # Check the axes (they should be in the same order that they are defined above).
        # Use the axis max as a proxy (this function name sux).
        assert proj.GetXaxis().GetXmax() == 0.8

        # Find the non-zero bin content so that it can be checked below.
        non_zero_bins = find_non_zero_bins(hist=proj)

        expected_count = 0
        # It will only be non-zero if all of the expected values are true.
        expected_non_zero_counts = all([expected_additional_cuts, expected_projection_axes])
        if expected_non_zero_counts:
            expected_count = 1
        assert len(non_zero_bins) == expected_count
        # Check the precise bin which was found and the bin value.
        if expected_count != 0:
            # Only check if we actually expected a count
            non_zero_bin_location = next(iter(non_zero_bins))
            # I determined the expected value empirically by looking at the projection.
            assert non_zero_bin_location == 1
            assert proj.GetBinContent(non_zero_bin_location) == 1

    @pytest.mark.parametrize(
        "single_observable",
        [
            False,
            True,
        ],
        ids=["Dict observable input", "Single observable input"],
    )
    # AAC = Additional Axis Cuts
    @pytest.mark.parametrize(
        ("additional_axis_cuts", "expected_additional_axis_cuts"),
        [(None, True), (hist_axis_ranges.x_axis, True), (hist_axis_ranges_without_entries.x_axis, False)],
        ids=["No AAC selection", "AAC with entries", "AAC with no entries"],
    )
    # PDCA = Projection Dependent Cut Axes
    @pytest.mark.parametrize(
        ("projection_dependent_cut_axes", "expected_projection_dependent_cut_axes"),
        [
            (None, True),
            ([], True),
            ([hist_axis_ranges.y_axis], True),
            ([hist_axis_ranges_without_entries.y_axis], False),
            ([hist_axis_ranges_restricted[0], hist_axis_ranges_restricted[1]], True),
            ([hist_axis_ranges_restricted[1], hist_axis_ranges_restricted[0]], True),
            ([hist_axis_ranges_restricted[1], hist_axis_ranges_restricted[2]], False),
        ],
        ids=[
            "None PDCA",
            "Empty PDCA",
            "PDCA",
            "PDCA with no entries",
            "Disconnected PDCA with entries",
            "Reversed and disconnected PDCA with entries",
            "Disconnected PDCA with no entries",
        ],
    )
    # PA = Projection Axes
    @pytest.mark.parametrize(
        ("projection_axes", "expected_projection_axes"),
        [(hist_axis_ranges.z_axis, True), (hist_axis_ranges_without_entries.z_axis, False)],
        ids=["PA with entries", "PA without entries"],
    )
    def test_TH3_to_TH1_projection(
        self,
        test_root_hists,
        single_observable,
        additional_axis_cuts,
        expected_additional_axis_cuts,
        projection_dependent_cut_axes,
        expected_projection_dependent_cut_axes,
        projection_axes,
        expected_projection_axes,
    ):
        """Test projection from a TH3 to a TH1 derived class."""
        ROOT = pytest.importorskip("ROOT")  # noqa: F841

        # Setup hist ranges
        if additional_axis_cuts:
            additional_axis_cuts = setup_hist_axis_range(additional_axis_cuts)
        if projection_dependent_cut_axes:
            projection_dependent_cut_axes = [setup_hist_axis_range(cut) for cut in projection_dependent_cut_axes]
        projection_axes = setup_hist_axis_range(projection_axes)
        # Setup projector
        kwdargs, observable, output_observable = determine_projector_input_args(
            single_observable=single_observable,
            hist=test_root_hists.hist3D,
            hist_label="hist3D",
        )
        kwdargs["projection_name_format"] = "hist"
        kwdargs["projection_information"] = {}
        obj = projectors.HistProjector(**kwdargs)

        # Set the projection axes.
        if additional_axis_cuts is not None:
            obj.additional_axis_cuts.append(additional_axis_cuts)
        if projection_dependent_cut_axes is not None:
            # We need to iterate here separately so that we can separate out the cuts
            # for the disconnected PDCAs.
            for axis_set in projection_dependent_cut_axes:
                obj.projection_dependent_cut_axes.append([axis_set])
        obj.projection_axes.append(projection_axes)

        # Perform the projection.
        obj.project()

        # Check the output and get the projection.
        proj = check_and_get_projection(
            single_observable=single_observable,
            observable=observable,
            output_observable=output_observable,
        )
        assert proj.GetName() == "hist"

        logger.debug(f"output_observable: {output_observable}, proj.GetEntries(): {proj.GetEntries()}")

        expected_bins = 5
        # If we don't expect a count, we've restricted the range further, so we need to reflect this in our check.
        if expected_projection_axes is False:
            expected_bins = 4
        assert proj.GetXaxis().GetNbins() == expected_bins

        # Find the non-zero bin content so that it can be checked below.
        non_zero_bins = find_non_zero_bins(hist=proj)

        expected_count = 0
        # It will only be non-zero if all of the expected values are true.
        expected_non_zero_counts = all(
            [expected_additional_axis_cuts, expected_projection_dependent_cut_axes, expected_projection_axes]
        )
        if expected_non_zero_counts:
            expected_count = 1
        assert len(non_zero_bins) == expected_count
        # Check the precise bin which was found and the bin value.
        if expected_count != 0:
            # Only check if we actually expected a count
            non_zero_bin_location = next(iter(non_zero_bins))
            # I determined the expected value empirically by looking at the projection.
            assert non_zero_bin_location == 1
            assert proj.GetBinContent(non_zero_bin_location) == 1

    @pytest.mark.parametrize(
        "single_observable",
        [
            False,
            True,
        ],
        ids=["Dict observable input", "Single observable input"],
    )
    # Other axes:
    # AAC = Additional Axis Cuts
    # PDCA = Projection Dependent Cut Axes
    @pytest.mark.parametrize(
        ("use_PDCA", "additional_cuts", "expected_additional_cuts"),
        [
            (False, None, True),
            (False, hist_axis_ranges.y_axis, True),
            (False, hist_axis_ranges_without_entries.y_axis, False),
            (True, None, True),
            (True, [], True),
            (True, [hist_axis_ranges.y_axis], True),
            (True, [hist_axis_ranges_without_entries.y_axis], False),
            (True, [hist_axis_ranges_restricted[0], hist_axis_ranges_restricted[1]], True),
            (True, [hist_axis_ranges_restricted[1], hist_axis_ranges_restricted[0]], True),
            (True, [hist_axis_ranges_restricted[1], hist_axis_ranges_restricted[2]], False),
        ],
        ids=[
            "No AAC selection",
            "AAC with entries",
            "AAC with no entries",
            "None PDCA",
            "Empty PDCA",
            "PDCA",
            "PDCA with no entries",
            "Disconnected PDCA with entries",
            "Reversed and disconnected PDCA with entries",
            "Disconnected PDCA with no entries",
        ],
    )
    # PA = Projection Axes
    @pytest.mark.parametrize(
        ("projection_axes", "expected_projection_axes"),
        [
            ([hist_axis_ranges.z_axis, hist_axis_ranges.x_axis], True),
            ([hist_axis_ranges.z_axis, hist_axis_ranges_without_entries.x_axis], False),
            ([hist_axis_ranges_without_entries.z_axis, hist_axis_ranges.x_axis], False),
            ([hist_axis_ranges_without_entries.z_axis, hist_axis_ranges_without_entries.x_axis], False),
        ],
        ids=["PA with entries", "PA without entries due to x", "PA without entries due to z", "PA without entries"],
    )
    def test_TH3_to_TH2_projection(
        self,
        test_root_hists,
        single_observable,
        use_PDCA,
        additional_cuts,
        expected_additional_cuts,
        projection_axes,
        expected_projection_axes,
    ):
        """Test projection of a TH3 into a TH2."""
        ROOT = pytest.importorskip("ROOT")  # noqa: F841

        # Setup hist ranges
        if additional_cuts:
            if use_PDCA:
                additional_cuts = [setup_hist_axis_range(cut) for cut in additional_cuts]
            else:
                additional_cuts = setup_hist_axis_range(additional_cuts)
        projection_axes = [setup_hist_axis_range(cut) for cut in projection_axes]
        # Setup projector
        kwdargs, observable, output_observable = determine_projector_input_args(
            single_observable=single_observable,
            hist=test_root_hists.hist3D,
            hist_label="hist3D",
        )
        kwdargs["projection_name_format"] = "hist"
        kwdargs["projection_information"] = {}
        obj = projectors.HistProjector(**kwdargs)

        # Set the projection axes.
        # Using additional cut axes or PDCA is mutually exclusive because we only have one
        # non-projection axis to work with.
        if use_PDCA:
            if additional_cuts is not None:
                # We need to iterate here separately so that we can separate out the cuts
                # for the disconnected PDCAs.
                for axis_set in additional_cuts:
                    obj.projection_dependent_cut_axes.append([axis_set])
        elif additional_cuts is not None:
            obj.additional_axis_cuts.append(additional_cuts)
        for ax in projection_axes:
            obj.projection_axes.append(ax)

        # Perform the projection.
        obj.project()

        # Check the output and get the projection.
        proj = check_and_get_projection(
            single_observable=single_observable,
            observable=observable,
            output_observable=output_observable,
        )
        assert proj.GetName() == "hist"

        logger.debug(f"output_observable: {output_observable}, proj.GetEntries(): {proj.GetEntries()}")

        # Check the axes (they should be in the same order that they are defined above).
        # Use the axis max as a proxy (this function name sux).
        assert proj.GetXaxis().GetXmax() == 60.0
        assert proj.GetYaxis().GetXmax() == 0.8
        logger.debug(f"x axis min: {proj.GetXaxis().GetXmin()}, y axis min: {proj.GetYaxis().GetXmin()}")

        # Find the non-zero bin content so that it can be checked below.
        non_zero_bins = find_non_zero_bins(hist=proj)

        expected_count = 0
        # It will only be non-zero if all of the expected values are true.
        expected_non_zero_counts = all([expected_additional_cuts, expected_projection_axes])
        if expected_non_zero_counts:
            expected_count = 1
        assert len(non_zero_bins) == expected_count
        # Check the precise bin which was found and the bin value.
        if expected_count != 0:
            # Only check if we actually expected a count
            non_zero_bin_location = next(iter(non_zero_bins))
            # I determined the expected value empirically by looking at the projection.
            assert non_zero_bin_location == 8
            assert proj.GetBinContent(non_zero_bin_location) == 1

    @pytest.mark.parametrize(
        "PDCA_axis",
        [
            hist_axis_ranges.x_axis,
            hist_axis_ranges_without_entries.x_axis,
        ],
        ids=["Same range PDCA", "Different range PDCA"],
    )
    def test_invalid_PDCA_axis(self, test_root_hists, PDCA_axis):
        """Test catching a PDCA on the same axis as the projection axis."""
        ROOT = pytest.importorskip("ROOT")  # noqa: F841

        # Setup projector
        output_observable: dict[str, typing_helpers.Hist] = {}
        observable_to_project_from = {"hist3D": test_root_hists.hist3D}
        projection_name_format = "hist"
        obj = projectors.HistProjector(
            output_observable=output_observable,
            observable_to_project_from=observable_to_project_from,
            projection_name_format=projection_name_format,
            projection_information={},
        )

        # Set the projection axes.
        # It is invalid even if the ranges are different
        obj.projection_dependent_cut_axes.append([PDCA_axis])
        obj.projection_axes.append(hist_axis_ranges.x_axis)

        # Perform the projection.
        with pytest.raises(ValueError, match="configuration is not allowed") as exception_info:
            obj.project()

        assert "This configuration is not allowed" in exception_info.value.args[0]


# Define similar axis and axis selection structures for the THnSparse.
# We use some subset of nearly all of these options in the various THn tests.
sparse_hist_axis_ranges = HistAxisRanges(
    projectors.HistAxisRange(axis_type=SparseAxisLabels.axis_two, axis_range_name="axis_two", min_val=2, max_val=18),
    projectors.HistAxisRange(axis_type=SparseAxisLabels.axis_four, axis_range_name="axis_four", min_val=-8, max_val=8),
    projectors.HistAxisRange(axis_type=SparseAxisLabels.axis_five, axis_range_name="axis_five", min_val=2, max_val=20),
)
sparse_hist_axis_ranges_with_no_entries = HistAxisRanges(
    projectors.HistAxisRange(
        axis_type=SparseAxisLabels.axis_two, axis_range_name="axis_two_no_entries", min_val=10, max_val=18
    ),
    projectors.HistAxisRange(
        axis_type=SparseAxisLabels.axis_four, axis_range_name="axis_four_no_entries", min_val=4, max_val=8
    ),
    projectors.HistAxisRange(
        axis_type=SparseAxisLabels.axis_five, axis_range_name="axis_five_no_entires", min_val=12, max_val=20
    ),
)
# This abuses the names of the axes within the named tuple, but it is rather convenient, so we keep it.
sparse_hist_axis_ranges_restricted = [
    projectors.HistAxisRange(
        axis_type=SparseAxisLabels.axis_four, axis_range_name="axis_four_lower", min_val=-8, max_val=-4
    ),
    projectors.HistAxisRange(
        axis_type=SparseAxisLabels.axis_four, axis_range_name="axis_four_lower_middle", min_val=-4, max_val=0
    ),
    projectors.HistAxisRange(
        axis_type=SparseAxisLabels.axis_four, axis_range_name="axis_four_upper_middle", min_val=0, max_val=4
    ),
    projectors.HistAxisRange(
        axis_type=SparseAxisLabels.axis_four, axis_range_name="axis_four_upper", min_val=4, max_val=8
    ),
]


class TestsForTHnSparseProjection:
    """Tests for projectors for THnSparse derived histograms."""

    @pytest.mark.parametrize(
        "single_observable",
        [
            False,
            True,
        ],
        ids=["Dict observable input", "Single observable input"],
    )
    # AAC = Additional Axis Cuts
    @pytest.mark.parametrize(
        ("additional_axis_cuts", "expected_additional_axis_cuts_counts"),
        [(None, 1), (sparse_hist_axis_ranges.x_axis, 1), (sparse_hist_axis_ranges_with_no_entries.x_axis, 0)],
        ids=["No AAC selection", "AAC with entries", "AAC with no entries"],
    )
    # PDCA = Projection Dependent Cut Axes
    @pytest.mark.parametrize(
        ("projection_dependent_cut_axes", "expected_projection_dependent_cut_axes_counts"),
        [
            (None, 2),
            ([], 2),
            ([sparse_hist_axis_ranges.y_axis], 2),
            ([sparse_hist_axis_ranges_with_no_entries.y_axis], 0),
            ([sparse_hist_axis_ranges_restricted[1], sparse_hist_axis_ranges_restricted[3]], 1),
            ([sparse_hist_axis_ranges_restricted[2], sparse_hist_axis_ranges_restricted[0]], 1),
            ([sparse_hist_axis_ranges_restricted[0], sparse_hist_axis_ranges_restricted[3]], 0),
        ],
        ids=[
            "None PDCA",
            "Empty PDCA",
            "PDCA",
            "PDCA with no entries",
            "Disconnected PDCA with entries",
            "Reversed and disconnected PDCA with entries",
            "Disconnected PDCA with no entries",
        ],
    )
    # PA = Projection Axes
    @pytest.mark.parametrize(
        ("projection_axes", "expected_projection_axes_counts"),
        [(sparse_hist_axis_ranges.z_axis, 1), (sparse_hist_axis_ranges_with_no_entries.z_axis, 0)],
        ids=["PA with entries", "PA without entries"],
    )
    def test_THn_projection(
        logging_mixin,
        test_sparse,
        single_observable,
        additional_axis_cuts,
        expected_additional_axis_cuts_counts,
        projection_dependent_cut_axes,
        expected_projection_dependent_cut_axes_counts,
        projection_axes,
        expected_projection_axes_counts,
    ):
        """Test projection of a THnSparse into a TH1."""
        ROOT = pytest.importorskip("ROOT")  # noqa: F841

        # Setup hist ranges
        if additional_axis_cuts:
            additional_axis_cuts = setup_hist_axis_range(additional_axis_cuts)
        if projection_dependent_cut_axes:
            projection_dependent_cut_axes = [setup_hist_axis_range(cut) for cut in projection_dependent_cut_axes]
        projection_axes = setup_hist_axis_range(projection_axes)
        # Setup objects
        sparse, _ = test_sparse
        # Setup projector
        kwdargs, observable, output_observable = determine_projector_input_args(
            single_observable=single_observable,
            hist=sparse,
            hist_label="hist_sparse",
        )
        kwdargs["projection_name_format"] = "hist"
        kwdargs["projection_information"] = {}
        obj = projectors.HistProjector(**kwdargs)

        # Set the projection axes.
        if additional_axis_cuts is not None:
            obj.additional_axis_cuts.append(additional_axis_cuts)
        if projection_dependent_cut_axes is not None:
            # We need to iterate here separately so that we can separate out the cuts
            # for the disconnected PDCAs.
            for axis_set in projection_dependent_cut_axes:
                obj.projection_dependent_cut_axes.append([axis_set])
        obj.projection_axes.append(projection_axes)

        # Perform the projection.
        obj.project()

        # Check the output and get the projection.
        proj = check_and_get_projection(
            single_observable=single_observable,
            observable=observable,
            output_observable=output_observable,
        )
        assert proj.GetName() == "hist"

        logger.debug(f"output_observable: {output_observable}, proj.GetEntries(): {proj.GetEntries()}")

        # Find the non-zero bin content so that it can be checked below.
        non_zero_bins = []
        for x in range(1, proj.GetNcells()):
            if proj.GetBinContent(x) != 0 and not proj.IsBinUnderflow(x) and not proj.IsBinOverflow(x):
                logger.debug(f"non-zero bin at {x}")
                non_zero_bins.append(x)

        # The expected value can be more than one. We find it by multiply the expected values. We can get away with
        # this because the largest value will be a single 2.
        expected_count = (
            expected_additional_axis_cuts_counts
            * expected_projection_dependent_cut_axes_counts
            * expected_projection_axes_counts
        )
        # However, we will only find one non-zero bin regardless of the value, so we just check for the one bin if
        # we have a non-zero value.
        assert len(non_zero_bins) == (1 if expected_count else 0)
        # Check the precise bin which was found and the bin value.
        if expected_count != 0:
            # Only check if we actually expected a count
            non_zero_bin_location = next(iter(non_zero_bins))
            # I determined the expected value empirically by looking at the projection.
            assert non_zero_bin_location == 9
            assert proj.GetBinContent(non_zero_bin_location) == expected_count
