#!/usr/bin/env python

""" Test projector functionality.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import copy
import enum
import dataclasses
import logging
import pytest
from typing import List

from pachyderm import projectors
from pachyderm import utils

logger = logging.getLogger(__name__)

class SparseAxisLabels(enum.Enum):
    """ Defines the relevant axis values for testing the sparse hist. """
    axis_two = 2
    axis_four = 4
    axis_five = 5

@pytest.fixture
def createHistAxisRange():
    """ Create a HistAxisRange object to use for testing. """
    #axisType, axis = request.param
    objectArgs = {
        "axisRangeName": "zAxisTestProjector",
        "axisType": projectors.TH1AxisType.yAxis,
        "minVal": lambda x: x,
        "maxVal": lambda y: y
    }
    obj = projectors.HistAxisRange(**objectArgs)
    # axisRangeName is referred to as name internally, so we rename to that
    objectArgs["name"] = objectArgs.pop("axisRangeName")

    return (obj, objectArgs)

def testHistAxisRange(loggingMixin, createHistAxisRange):
    """ Tests for creating a HistAxisRange object. """
    obj, objectArgs = createHistAxisRange

    assert obj.name == objectArgs["name"]
    assert obj.axisType == objectArgs["axisType"]
    assert obj.minVal == objectArgs["minVal"]
    assert obj.maxVal == objectArgs["maxVal"]

    # Test repr and str to esnure that they are up to date.
    assert repr(obj) == "HistAxisRange(name = {name!r}, axisType = {axisType}, minVal = {minVal!r}, maxVal = {maxVal!r})".format(**objectArgs)
    assert str(obj) == "HistAxisRange: name: {name}, axisType: {axisType}, minVal: {minVal}, maxVal: {maxVal}".format(**objectArgs)
    # Assert that the dict is equal so we don't miss anything in the repr or str representations.
    assert obj.__dict__ == objectArgs

@pytest.mark.ROOT
@pytest.mark.parametrize("axisType, axis", [
    (projectors.TH1AxisType.xAxis, "x_axis"),
    (projectors.TH1AxisType.yAxis, "y_axis"),
    (projectors.TH1AxisType.zAxis, "z_axis"),
    (0, "x_axis"),
    (1, "y_axis"),
    (2, "z_axis"),
], ids = ["xAxis", "yAxis", "zAxis", "number for x axis", "number for y axis", "number for z axis"])
@pytest.mark.parametrize("histToTest", range(0, 3), ids = ["1D", "2D", "3D"])
def testTH1AxisDetermination(loggingMixin, createHistAxisRange, axisType, axis, histToTest, testRootHists):
    """ Test TH1 axis determination in the HistAxisRange object. """
    import ROOT
    axis_map = {
        "x_axis": ROOT.TH1.GetXaxis,
        "y_axis": ROOT.TH1.GetYaxis,
        "z_axis": ROOT.TH1.GetZaxis,
    }
    axis = axis_map[axis]
    # Get the HistAxisRange object
    obj, objectArgs = createHistAxisRange
    # Insert the proepr axis type
    obj.axisType = axisType
    # Determine the test hist
    hist = dataclasses.astuple(testRootHists)[histToTest]

    # Check that the axis retrieved by the specified function is the same
    # as that retrieved by the HistAxisRange object.
    # NOTE: GetZaxis() (for example) is still valid for a TH1. It is a minimal axis
    #       object with 1 bin. So it is fine to check for equivalnce for axes that
    #       don't really make sense in terms of a hist's dimensions.
    assert axis(hist) == obj.axis(hist)

@pytest.mark.ROOT
@pytest.mark.parametrize("axisSelection", [
    SparseAxisLabels.axis_two,
    SparseAxisLabels.axis_four,
    SparseAxisLabels.axis_five,
    2, 4, 5
], ids = ["axis_two", "axis_four", "axis_five", "number for axis one", "number for axis two", "number for axis three"])
def testTHnAxisDetermination(loggingMixin, axisSelection, createHistAxisRange, testSparse):
    """ Test THn axis determination in the HistAxisRange object. """
    # Retrieve sparse.
    sparse, _ = testSparse
    # Retrieve object and setup.
    obj, objectArgs = createHistAxisRange
    obj.axisType = axisSelection

    axisValue = axisSelection.value if isinstance(axisSelection, enum.Enum) else axisSelection
    assert sparse.GetAxis(axisValue) == obj.axis(sparse)

@pytest.mark.ROOT
class TestsForHistAxisRange():
    """ Tests for HistAxisRange which require ROOT. """
    @pytest.mark.parametrize("minVal, maxVal, minValFunc, maxValFunc, expectedFunc", [
        (0, 10,
            "find_bin_min",
            "find_bin_max",
            lambda axis, x, y: axis.SetRangeUser(x, y)),
        (1, 9,
            "find_bin_min",
            "find_bin_max",
            lambda axis, x, y: axis.SetRangeUser(x, y)),
        (1, None,
            None,
            "n_bins",
            lambda axis, x, y: True),  # This is just a no-op. We don't want to restrict the range.
        (0, 7,
            None,
            None,
            lambda axis, x, y: axis.SetRange(x, y))
    ], ids = ["0 - 10 with ApplyFuncToFindBin with FindBin", "1 - 9 (mid bin) with ApplyFuncToFindBin with FindBin", "1 - Nbins with ApplyFuncToFindBin (no under/overflow)", "0 - 10 with raw bin value passed ApplyFuncToFindBin"])
    def testApplyRangeSet(self, loggingMixin, minVal, maxVal, minValFunc, maxValFunc, expectedFunc, testSparse):
        """ Test apply a range set to an axis via a HistAxisRange object.

        This is intentionally tested against SetRangeUser, so we can be certain that it reproduces
        that selection as expected.

        Note:
            It doens't matter whether we operate on TH1 or THn, since they both set ranges on TAxis.

        Note:
            This implicity tests ApplyFuncToFindBin, which is fine given how often the two are used
            together (almost always).
        """
        import ROOT

        # Setup functions
        function_map = {
            None: lambda x: projectors.HistAxisRange.ApplyFuncToFindBin(None, x),
            "find_bin_min": lambda x: projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, x + utils.epsilon),
            "find_bin_max": lambda x: projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, x - utils.epsilon),
            "n_bins": lambda x: projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.GetNbins),
        }
        minValFunc = function_map[minValFunc]
        maxValFunc = function_map[maxValFunc]

        selectedAxis = SparseAxisLabels.axis_two
        sparse, _ = testSparse
        expectedAxis = sparse.GetAxis(selectedAxis.value).Clone("axis2")
        expectedFunc(expectedAxis, minVal, maxVal)

        obj = projectors.HistAxisRange(
            axisRangeName = "axis_twoTest",
            axisType = selectedAxis,
            minVal = minValFunc(minVal),
            maxVal = maxValFunc(maxVal))
        # Applys the restriction to the sparse.
        obj.ApplyRangeSet(sparse)
        ax = sparse.GetAxis(selectedAxis.value)

        # Unfortunately, equality comparison doesn't work for TAxis...
        # GetXmin() and GetXmax() aren't restircted by SetRange(), so instead use GetFirst() and GetLast()
        assert ax.GetFirst() == expectedAxis.GetFirst()
        assert ax.GetLast() == expectedAxis.GetLast()
        # Sanity check that the overall axis still agrees
        assert ax.GetNbins() == expectedAxis.GetNbins()
        assert ax.GetName() == expectedAxis.GetName()

    def testDisagreementWithSetRangeUser(self, loggingMixin, testSparse):
        """ Test the disagreement between SetRange and SetRangeUser when the epsilon shift is not included. """
        import ROOT

        # Setup values
        selectedAxis = SparseAxisLabels.axis_two
        minVal = 2
        maxVal = 8
        sparse, _ = testSparse
        # Detemine expected value (must be first to avoid interfering with applying the axis range)
        expectedAxis = sparse.GetAxis(selectedAxis.value).Clone("axis2")
        expectedAxis.SetRangeUser(minVal, maxVal)

        obj = projectors.HistAxisRange(
            axisRangeName = "axis_two_test",
            axisType = selectedAxis,
            minVal = projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, minVal),
            maxVal = projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, maxVal))
        # Applys the restriction to the sparse.
        obj.ApplyRangeSet(sparse)
        ax = sparse.GetAxis(selectedAxis.value)

        # Unfortunately, equality comparison doesn't work for TAxis...
        # GetXmin() and GetXmax() aren't restircted by SetRange(), so instead use GetFirst() and GetLast()
        # The lower bin will still agree.
        assert ax.GetFirst() == expectedAxis.GetFirst()
        # The upper bin will not.
        assert ax.GetLast() != expectedAxis.GetLast()
        # If we subtract a bin (equivalent to including - epsilon), it will agree.
        assert ax.GetLast() - 1 == expectedAxis.GetLast()
        # Sanity check that the overall axis still agrees
        assert ax.GetNbins() == expectedAxis.GetNbins()
        assert ax.GetName() == expectedAxis.GetName()

    @pytest.mark.parametrize("func, value, expected", [
        (None, 3, 3),
        ("n_bins", None, 10),
        ("find_bin", 10 - utils.epsilon, 5)
    ], ids = ["Only value", "Func only", "Func with value"])
    def testRetrieveAxisValue(self, loggingMixin, func, value, expected, testSparse):
        """ Test retrieving axis values using ApplyFuncToFindBin(). """
        import ROOT

        # Setup functions
        function_map = {
            "n_bins": ROOT.TAxis.GetNbins,
            "find_bin": ROOT.TAxis.FindBin,
        }
        if func:
            func = function_map[func]
        # Setup objects
        selectedAxis = SparseAxisLabels.axis_two
        sparse, _ = testSparse
        expectedAxis = sparse.GetAxis(selectedAxis.value)

        assert projectors.HistAxisRange.ApplyFuncToFindBin(func, value)(expectedAxis) == expected

def find_non_zero_bins(hist) -> List[int]:
    """ Helper function to find the non-zero non-overflow bins.

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
    """ Helper function to setup HistAxisRange min and max values.

    This exists so we can avoid explicit ROOT dependence.

    Args:
        hist_range (projectors.HistAxisRange): Range which includes single min and max values which will
            be used in ``ApplyFuncToFindBin`` function calls that will replace them.
    Return:
        Updated hist axis range with the initial value passed to ``ROOT.TAxis.FindBin``.
    """
    # Often, ROOT should be imported before this function is called, but we call it here just in case.
    # Plus, this allows us to be lazy in importing ROOT in the calling functions.
    import ROOT

    # We don't want to modify the original objects, since we need them to be preserved for other tests.
    hist_range = copy.copy(hist_range)
    hist_range.minVal = projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, hist_range.minVal + utils.epsilon)
    hist_range.maxVal = projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, hist_range.maxVal - utils.epsilon)
    return hist_range

# Convenient access to hist axis ranges.
HistAxisRanges = dataclasses.make_dataclass("histAxisRanges", ["xAxis", "yAxis", "zAxis"])

# Hist axis ranges
# NOTE: We don't define these in the class because we won't be able to access these variables when defining the tests
# Restricted ranges, but with entries
hist_axis_ranges = HistAxisRanges(
    projectors.HistAxisRange(
        axisType = projectors.TH1AxisType.xAxis,
        axisRangeName = "xAxis",
        minVal = 0.1,
        maxVal = 0.8),
    projectors.HistAxisRange(
        axisType = projectors.TH1AxisType.yAxis,
        axisRangeName = "yAxis",
        minVal = 0,
        maxVal = 12),
    projectors.HistAxisRange(
        axisType = projectors.TH1AxisType.zAxis,
        axisRangeName = "zAxis",
        minVal = 10,
        maxVal = 60)
)
# Restricted ranges with no counts
hist_axis_ranges_without_entries = HistAxisRanges(
    projectors.HistAxisRange(
        axisType = projectors.TH1AxisType.xAxis,
        axisRangeName = "xAxisNoEntries",
        minVal = 0.2,
        maxVal = 0.8),
    projectors.HistAxisRange(
        axisType = projectors.TH1AxisType.yAxis,
        axisRangeName = "yAxisNoEntries",
        minVal = 4,
        maxVal = 12),
    projectors.HistAxisRange(
        axisType = projectors.TH1AxisType.zAxis,
        axisRangeName = "zAxisNoEntries",
        minVal = 20,
        maxVal = 60)
)
# Restricted ranges with counts in some entries
# Focuses only the on y axis.
# This abuses the names of the axes within the named tuple, but it is rather convenient, so we keep it.
hist_axis_ranges_restricted = (
    projectors.HistAxisRange(
        axisType = projectors.TH1AxisType.yAxis,
        axisRangeName = "yAxisLower",
        minVal = 0,
        maxVal = 4),
    projectors.HistAxisRange(
        axisType = projectors.TH1AxisType.yAxis,
        axisRangeName = "yAxisMiddle",
        minVal = 4,
        maxVal = 8),
    projectors.HistAxisRange(
        axisType = projectors.TH1AxisType.yAxis,
        axisRangeName = "yAxisUpper",
        minVal = 8,
        maxVal = 12)
)

@pytest.mark.ROOT
class TestProjectorsWithRoot():
    """ Tests for projectors for TH1 derived histograms. """
    def testProjectors(self, loggingMixin, testRootHists):
        """ Test creation and basic methods of the projection class. """
        import ROOT  # noqa: F401

        # Args
        projectionNameFormat = "{test} world"
        # Create object
        obj = projectors.HistProjector(observableList = {},
                                       observableToProjectFrom = {},
                                       projectionNameFormat = projectionNameFormat,
                                       projectionInformation = {})

        # These objects should be overridden so they aren't super meaningful, but we can still
        # test to ensure that they provide the basic functionality that is expected.
        assert obj.ProjectionName(test = "Hello") == projectionNameFormat.format(test = "Hello")
        assert obj.GetHist(observable = testRootHists.hist2D) == testRootHists.hist2D
        assert obj.OutputKeyName(inputKey = "inputKey",
                                 outputHist = testRootHists.hist2D,
                                 projectionName = projectionNameFormat.format(test = "Hello")) == projectionNameFormat.format(test = "Hello")
        assert obj.OutputHist(outputHist = testRootHists.hist1D,
                              inputObservable = testRootHists.hist2D) == testRootHists.hist1D

    # Other axes:
    # AAC = Additional Axis Cuts
    # PDCA = Projection Dependent Cut Axes
    @pytest.mark.parametrize("use_PDCA, additionalCuts, expectedAdditionalCuts", [
        (False, None, True),
        (False, hist_axis_ranges.yAxis, True),
        (False, hist_axis_ranges_without_entries.yAxis, False),
        (True, None, True),
        (True, [], True),
        (True, [hist_axis_ranges.yAxis], True),
        (True, [hist_axis_ranges_without_entries.yAxis], False),
        (True, [hist_axis_ranges_restricted[0], hist_axis_ranges_restricted[1]], True),
        (True, [hist_axis_ranges_restricted[1], hist_axis_ranges_restricted[0]], True),
        (True, [hist_axis_ranges_restricted[1], hist_axis_ranges_restricted[2]], False)
    ], ids = [
        "No AAC selection", "AAC with entries", "AAC with no entries",
        "None PDCA", "Empty PDCA", "PDCA",
        "PDCA with no entries", "Disconnected PDCA with entries", "Reversed and disconnected PDCA with entries", "Disconnected PDCA with no entries",
    ])
    # PA = Projection Axes
    @pytest.mark.parametrize("projectionAxes, expectedProjectionAxes", [
        (hist_axis_ranges.xAxis, True),
        (hist_axis_ranges_without_entries.xAxis, False),
    ], ids = ["PA with entries", "PA without entries"])
    def testTH2Projection(self, loggingMixin, testRootHists,
                          use_PDCA, additionalCuts, expectedAdditionalCuts,
                          projectionAxes, expectedProjectionAxes):
        """ Test projection of a TH2 to a TH1. """
        import ROOT   # noqa: F401

        # Setup hist ranges
        if additionalCuts:
            if use_PDCA:
                additionalCuts = [setup_hist_axis_range(cut) for cut in additionalCuts]
            else:
                additionalCuts = setup_hist_axis_range(additionalCuts)
        projectionAxes = setup_hist_axis_range(projectionAxes)
        # Setup projector
        observableList = {}
        observableToProjectFrom = {"hist2D": testRootHists.hist2D}
        projectionNameFormat = "hist"
        obj = projectors.HistProjector(observableList = observableList,
                                       observableToProjectFrom = observableToProjectFrom,
                                       projectionNameFormat = projectionNameFormat,
                                       projectionInformation = {})

        # Set the projection axes.
        # Using additional cut axes or PDCA is mutually exclusive because we only have one
        # non-projection axis to work with.
        if use_PDCA:
            if additionalCuts is not None:
                # We need to iterate here separately so that we can separate out the cuts
                # for the disconnected PDCAs.
                for axisSet in additionalCuts:
                    obj.projectionDependentCutAxes.append([axisSet])
        else:
            if additionalCuts is not None:
                obj.additionalAxisCuts.append(additionalCuts)
        obj.projectionAxes.append(projectionAxes)

        # Perform the projection.
        obj.Project()

        # Check the output.
        assert len(observableList) == 1
        proj = next(iter(observableList.values()))
        assert proj.GetName() == "hist"

        logger.debug("observableList: {}, proj.GetEntries(): {}".format(observableList, proj.GetEntries()))

        # Check the axes (they should be in the same order that they are defined above).
        # Use the axis max as a proxy (this function name sux).
        assert proj.GetXaxis().GetXmax() == 0.8

        # Find the non-zero bin content so that it can be checked below.
        non_zero_bins = find_non_zero_bins(hist = proj)

        expectedCount = 0
        # It will only be non-zero if all of the expected values are true.
        expectedNonZeroCounts = all([expectedAdditionalCuts, expectedProjectionAxes])
        if expectedNonZeroCounts:
            expectedCount = 1
        assert len(non_zero_bins) == expectedCount
        # Check the precise bin which was found and the bin value.
        if expectedCount != 0:
            # Only check if we actually expected a count
            nonZeroBinLocation = next(iter(non_zero_bins))
            # I determined the expected value empirically by looking at the projection.
            assert nonZeroBinLocation == 1
            assert proj.GetBinContent(nonZeroBinLocation) == 1

    # AAC = Additional Axis Cuts
    @pytest.mark.parametrize("additionalAxisCuts, expectedAdditionalAxisCuts", [
        (None, True),
        (hist_axis_ranges.xAxis, True),
        (hist_axis_ranges_without_entries.xAxis, False)
    ], ids = ["No AAC selection", "AAC with entries", "AAC with no entries"])
    # PDCA = Projection Dependent Cut Axes
    @pytest.mark.parametrize("projectionDependentCutAxes, expectedProjectionDependentCutAxes", [
        (None, True),
        ([], True),
        ([hist_axis_ranges.yAxis], True),
        ([hist_axis_ranges_without_entries.yAxis], False),
        ([hist_axis_ranges_restricted[0], hist_axis_ranges_restricted[1]], True),
        ([hist_axis_ranges_restricted[1], hist_axis_ranges_restricted[0]], True),
        ([hist_axis_ranges_restricted[1], hist_axis_ranges_restricted[2]], False)
    ], ids = ["None PDCA", "Empty PDCA", "PDCA", "PDCA with no entries", "Disconnected PDCA with entries", "Reversed and disconnected PDCA with entries", "Disconnected PDCA with no entries"])
    # PA = Projection Axes
    @pytest.mark.parametrize("projectionAxes, expectedProjectionAxes", [
        (hist_axis_ranges.zAxis, True),
        (hist_axis_ranges_without_entries.zAxis, False)
    ], ids = ["PA with entries", "PA without entries"])
    def testTH3ToTH1Projection(self, loggingMixin, testRootHists,
                               additionalAxisCuts, expectedAdditionalAxisCuts,
                               projectionDependentCutAxes, expectedProjectionDependentCutAxes,
                               projectionAxes, expectedProjectionAxes):
        """ Test projection from a TH3 to a TH1 derived class. """
        import ROOT  # noqa: F401

        # Setup hist ranges
        if additionalAxisCuts:
            additionalAxisCuts = setup_hist_axis_range(additionalAxisCuts)
        if projectionDependentCutAxes:
            projectionDependentCutAxes = [setup_hist_axis_range(cut) for cut in projectionDependentCutAxes]
        projectionAxes = setup_hist_axis_range(projectionAxes)
        # Setup projector
        observableList = {}
        observableToProjectFrom = {"hist3D": testRootHists.hist3D}
        projectionNameFormat = "hist"
        obj = projectors.HistProjector(observableList = observableList,
                                       observableToProjectFrom = observableToProjectFrom,
                                       projectionNameFormat = projectionNameFormat,
                                       projectionInformation = {})

        # Set the projection axes.
        if additionalAxisCuts is not None:
            obj.additionalAxisCuts.append(additionalAxisCuts)
        if projectionDependentCutAxes is not None:
            # We need to iterate here separately so that we can separate out the cuts
            # for the disconnected PDCAs.
            for axisSet in projectionDependentCutAxes:
                obj.projectionDependentCutAxes.append([axisSet])
        obj.projectionAxes.append(projectionAxes)

        # Perform the projection.
        obj.Project()

        # Check the basic output.
        assert len(observableList) == 1
        proj = next(iter(observableList.values()))
        assert proj.GetName() == "hist"

        logger.debug("observableList: {}, proj.GetEntries(): {}".format(observableList, proj.GetEntries()))

        expectedBins = 5
        # If we don't expect a count, we've restricted the range further, so we need to reflect this in our check.
        if expectedProjectionAxes is False:
            expectedBins = 4
        assert proj.GetXaxis().GetNbins() == expectedBins

        # Find the non-zero bin content so that it can be checked below.
        non_zero_bins = find_non_zero_bins(hist = proj)

        expectedCount = 0
        # It will only be non-zero if all of the expected values are true.
        expectedNonZeroCounts = all([expectedAdditionalAxisCuts, expectedProjectionDependentCutAxes, expectedProjectionAxes])
        if expectedNonZeroCounts:
            expectedCount = 1
        assert len(non_zero_bins) == expectedCount
        # Check the precise bin which was found and the bin value.
        if expectedCount != 0:
            # Only check if we actually expected a count
            nonZeroBinLocation = next(iter(non_zero_bins))
            # I determined the expected value empirically by looking at the projection.
            assert nonZeroBinLocation == 1
            assert proj.GetBinContent(nonZeroBinLocation) == 1

    # Other axes:
    # AAC = Additional Axis Cuts
    # PDCA = Projection Dependent Cut Axes
    @pytest.mark.parametrize("use_PDCA, additionalCuts, expectedAdditionalCuts", [
        (False, None, True),
        (False, hist_axis_ranges.yAxis, True),
        (False, hist_axis_ranges_without_entries.yAxis, False),
        (True, None, True),
        (True, [], True),
        (True, [hist_axis_ranges.yAxis], True),
        (True, [hist_axis_ranges_without_entries.yAxis], False),
        (True, [hist_axis_ranges_restricted[0], hist_axis_ranges_restricted[1]], True),
        (True, [hist_axis_ranges_restricted[1], hist_axis_ranges_restricted[0]], True),
        (True, [hist_axis_ranges_restricted[1], hist_axis_ranges_restricted[2]], False)
    ], ids = [
        "No AAC selection", "AAC with entries", "AAC with no entries",
        "None PDCA", "Empty PDCA", "PDCA",
        "PDCA with no entries", "Disconnected PDCA with entries", "Reversed and disconnected PDCA with entries", "Disconnected PDCA with no entries",
    ])
    # PA = Projection Axes
    @pytest.mark.parametrize("projectionAxes, expectedProjectionAxes", [
        ([hist_axis_ranges.zAxis, hist_axis_ranges.xAxis], True),
        ([hist_axis_ranges.zAxis, hist_axis_ranges_without_entries.xAxis], False),
        ([hist_axis_ranges_without_entries.zAxis, hist_axis_ranges.xAxis], False),
        ([hist_axis_ranges_without_entries.zAxis, hist_axis_ranges_without_entries.xAxis], False),
    ], ids = ["PA with entries", "PA without entries due to x", "PA without entires due to z", "PA without entries"])
    def testTH3ToTH2Projection(self, loggingMixin, testRootHists,
                               use_PDCA, additionalCuts, expectedAdditionalCuts,
                               projectionAxes, expectedProjectionAxes):
        """ Test projection of a TH3 into a TH2. """
        import ROOT  # noqa: F401

        # Setup hist ranges
        if additionalCuts:
            if use_PDCA:
                additionalCuts = [setup_hist_axis_range(cut) for cut in additionalCuts]
            else:
                additionalCuts = setup_hist_axis_range(additionalCuts)
        projectionAxes = [setup_hist_axis_range(cut) for cut in projectionAxes]
        # Setup projector
        observableList = {}
        observableToProjectFrom = {"hist3D": testRootHists.hist3D}
        projectionNameFormat = "hist"
        obj = projectors.HistProjector(observableList = observableList,
                                       observableToProjectFrom = observableToProjectFrom,
                                       projectionNameFormat = projectionNameFormat,
                                       projectionInformation = {})

        # Set the projection axes.
        # Using additional cut axes or PDCA is mutually exclusive because we only have one
        # non-projection axis to work with.
        if use_PDCA:
            if additionalCuts is not None:
                # We need to iterate here separately so that we can separate out the cuts
                # for the disconnected PDCAs.
                for axisSet in additionalCuts:
                    obj.projectionDependentCutAxes.append([axisSet])
        else:
            if additionalCuts is not None:
                obj.additionalAxisCuts.append(additionalCuts)
        for ax in projectionAxes:
            obj.projectionAxes.append(ax)

        # Perform the projection.
        obj.Project()

        # Check the basic output.
        assert len(observableList) == 1
        proj = next(iter(observableList.values()))
        assert proj.GetName() == "hist"

        logger.debug("observableList: {}, proj.GetEntries(): {}".format(observableList, proj.GetEntries()))

        # Check the axes (they should be in the same order that they are defined above).
        # Use the axis max as a proxy (this function name sux).
        assert proj.GetXaxis().GetXmax() == 60.0
        assert proj.GetYaxis().GetXmax() == 0.8
        logger.debug(f"x axis min: {proj.GetXaxis().GetXmin()}, y axis min: {proj.GetYaxis().GetXmin()}")

        # Find the non-zero bin content so that it can be checked below.
        non_zero_bins = find_non_zero_bins(hist = proj)

        expectedCount = 0
        # It will only be non-zero if all of the expected values are true.
        expectedNonZeroCounts = all([expectedAdditionalCuts, expectedProjectionAxes])
        if expectedNonZeroCounts:
            expectedCount = 1
        assert len(non_zero_bins) == expectedCount
        # Check the precise bin which was found and the bin value.
        if expectedCount != 0:
            # Only check if we actually expected a count
            nonZeroBinLocation = next(iter(non_zero_bins))
            # I determined the expected value empirically by looking at the projection.
            assert nonZeroBinLocation == 8
            assert proj.GetBinContent(nonZeroBinLocation) == 1

    @pytest.mark.parametrize("PDCA_axis", [
        hist_axis_ranges.xAxis,
        hist_axis_ranges_without_entries.xAxis,
    ], ids = ["Same range PDCA", "Different range PDCA"])
    def test_invalid_PDCA_axis(loggingMixin, testRootHists, PDCA_axis):
        """ Test catching a PDCA on the same axis as the projection axis. """
        import ROOT  # noqa: F401

        # Setup projector
        observableList = {}
        observableToProjectFrom = {"hist3D": testRootHists.hist3D}
        projectionNameFormat = "hist"
        obj = projectors.HistProjector(observableList = observableList,
                                       observableToProjectFrom = observableToProjectFrom,
                                       projectionNameFormat = projectionNameFormat,
                                       projectionInformation = {})

        # Set the projection axes.
        # It is invalid even if the ranges are different
        obj.projectionDependentCutAxes.append([PDCA_axis])
        obj.projectionAxes.append(hist_axis_ranges.xAxis)

        # Perform the projection.
        with pytest.raises(ValueError) as exception_info:
            obj.Project()

        assert "This configuration is not allowed" in exception_info.value.args[0]

# Define similar axis and axis seletion structures for the THnSparse.
# We use some subset of nearly all of these options in the various THn tests.
sparse_hist_axis_ranges = HistAxisRanges(
    projectors.HistAxisRange(
        axisType = SparseAxisLabels.axis_two,
        axisRangeName = "axis_two",
        minVal = 2,
        maxVal = 18),
    projectors.HistAxisRange(
        axisType = SparseAxisLabels.axis_four,
        axisRangeName = "axis_four",
        minVal = -8,
        maxVal = 8),
    projectors.HistAxisRange(
        axisType = SparseAxisLabels.axis_five,
        axisRangeName = "axis_five",
        minVal = 2,
        maxVal = 20)
)
sparse_hist_axis_ranges_with_no_entries = HistAxisRanges(
    projectors.HistAxisRange(
        axisType = SparseAxisLabels.axis_two,
        axisRangeName = "axis_two_no_entries",
        minVal = 10,
        maxVal = 18),
    projectors.HistAxisRange(
        axisType = SparseAxisLabels.axis_four,
        axisRangeName = "axis_four_no_entries",
        minVal = 4,
        maxVal = 8),
    projectors.HistAxisRange(
        axisType = SparseAxisLabels.axis_five,
        axisRangeName = "axis_five_no_entires",
        minVal = 12,
        maxVal = 20)
)
# This abuses the names of the axes within the named tuple, but it is rather convenient, so we keep it.
sparse_hist_axis_ranges_restricted = [
    projectors.HistAxisRange(
        axisType = SparseAxisLabels.axis_four,
        axisRangeName = "axis_four_lower",
        minVal = -8,
        maxVal = -4),
    projectors.HistAxisRange(
        axisType = SparseAxisLabels.axis_four,
        axisRangeName = "axis_four_lower_middle",
        minVal = -4,
        maxVal = 0),
    projectors.HistAxisRange(
        axisType = SparseAxisLabels.axis_four,
        axisRangeName = "axis_four_upper_middle",
        minVal = 0,
        maxVal = 4),
    projectors.HistAxisRange(
        axisType = SparseAxisLabels.axis_four,
        axisRangeName = "axis_four_upper",
        minVal = 4,
        maxVal = 8),
]

@pytest.mark.ROOT
class TestsForTHnSparseProjection():
    """ Tests for projectors for THnSparse derived histograms. """
    # AAC = Additional Axis Cuts
    @pytest.mark.parametrize("additional_axis_cuts, expected_additional_axis_cuts_counts", [
        (None, 1),
        (sparse_hist_axis_ranges.xAxis, 1),
        (sparse_hist_axis_ranges_with_no_entries.xAxis, 0)
    ], ids = ["No AAC selection", "AAC with entries", "AAC with no entries"])
    # PDCA = Projection Dependent Cut Axes
    @pytest.mark.parametrize("projection_dependent_cut_axes, expected_projection_dependent_cut_axes_counts", [
        (None, 2),
        ([], 2),
        ([sparse_hist_axis_ranges.yAxis], 2),
        ([sparse_hist_axis_ranges_with_no_entries.yAxis], 0),
        ([sparse_hist_axis_ranges_restricted[1], sparse_hist_axis_ranges_restricted[3]], 1),
        ([sparse_hist_axis_ranges_restricted[2], sparse_hist_axis_ranges_restricted[0]], 1),
        ([sparse_hist_axis_ranges_restricted[0], sparse_hist_axis_ranges_restricted[3]], 0)
    ], ids = ["None PDCA", "Empty PDCA", "PDCA", "PDCA with no entries", "Disconnected PDCA with entries", "Reversed and disconnected PDCA with entries", "Disconnected PDCA with no entries"])
    # PA = Projection Axes
    @pytest.mark.parametrize("projection_axes, expected_projection_axes_counts", [
        (sparse_hist_axis_ranges.zAxis, 1),
        (sparse_hist_axis_ranges_with_no_entries.zAxis, 0)
    ], ids = ["PA with entries", "PA without entries"])
    def test_THn_projection(loggingMixin, testSparse,
                            additional_axis_cuts, expected_additional_axis_cuts_counts,
                            projection_dependent_cut_axes, expected_projection_dependent_cut_axes_counts,
                            projection_axes, expected_projection_axes_counts):
        """ Test projection of a THnSparse into a TH1. """
        import ROOT  # noqa: F401

        # Setup hist ranges
        if additional_axis_cuts:
            additional_axis_cuts = setup_hist_axis_range(additional_axis_cuts)
        if projection_dependent_cut_axes:
            projection_dependent_cut_axes = [setup_hist_axis_range(cut) for cut in projection_dependent_cut_axes]
        projection_axes = setup_hist_axis_range(projection_axes)
        # Setup objects
        sparse, _ = testSparse
        # Setup projector
        observableList = {}
        observableToProjectFrom = {"histSparse": sparse}
        projectionNameFormat = "hist"
        obj = projectors.HistProjector(observableList = observableList,
                                       observableToProjectFrom = observableToProjectFrom,
                                       projectionNameFormat = projectionNameFormat,
                                       projectionInformation = {})

        # Set the projection axes.
        if additional_axis_cuts is not None:
            obj.additionalAxisCuts.append(additional_axis_cuts)
        if projection_dependent_cut_axes is not None:
            # We need to iterate here separately so that we can separate out the cuts
            # for the disconnected PDCAs.
            for axisSet in projection_dependent_cut_axes:
                obj.projectionDependentCutAxes.append([axisSet])
        obj.projectionAxes.append(projection_axes)

        # Perform the projection.
        obj.Project()

        # Basic output checks.
        assert len(observableList) == 1
        proj = next(iter(observableList.values()))
        assert proj.GetName() == "hist"
        logger.debug("observableList: {}, proj.GetEntries(): {}".format(observableList, proj.GetEntries()))

        # Find the non-zero bin content so that it can be checked below.
        non_zero_bins = []
        for x in range(1, proj.GetNcells()):
            if proj.GetBinContent(x) != 0 and not proj.IsBinUnderflow(x) and not proj.IsBinOverflow(x):
                logger.debug(f"non-zero bin at {x}")
                non_zero_bins.append(x)

        # The expected value can be more than one. We find it by multiply the expected values. We can get away with
        # this because the largest value will be a single 2.
        expected_count = expected_additional_axis_cuts_counts * expected_projection_dependent_cut_axes_counts * expected_projection_axes_counts
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

