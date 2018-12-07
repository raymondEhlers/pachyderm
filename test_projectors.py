#!/usr/bin/env python

# Test projector functionality
#
# author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
# date: 6 June 2018

import aenum
import collections
import dataclasses
import logging
import pytest

import rootpy.ROOT as ROOT

from jet_hadron.base import projectors
from jet_hadron.base import utils

logger = logging.getLogger(__name__)

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

@pytest.mark.parametrize("axisType, axis", [
    (projectors.TH1AxisType.xAxis, ROOT.TH1.GetXaxis),
    (projectors.TH1AxisType.yAxis, ROOT.TH1.GetYaxis),
    (projectors.TH1AxisType.zAxis, ROOT.TH1.GetZaxis),
    (0, ROOT.TH1.GetXaxis),
    (1, ROOT.TH1.GetYaxis),
    (2, ROOT.TH1.GetZaxis),
], ids = ["xAxis", "yAxis", "zAxis", "number for x axis", "number for y axis", "number for z axis"])
@pytest.mark.parametrize("histToTest", range(0, 3), ids = ["1D", "2D", "3D"])
def testTH1AxisDetermination(loggingMixin, createHistAxisRange, axisType, axis, histToTest, testRootHists):
    """ Test TH1 axis determination in the HistAxisRange object. """
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

class selectedTestAxis(aenum.Enum):
    """ Enum to map from our selected axes to their axis values. Goes along with the sparse created in testSparse. """
    axisOne = 2
    axisTwo = 4
    axisThree = 5

@pytest.mark.parametrize("axisSelection", [
    selectedTestAxis.axisOne,
    selectedTestAxis.axisTwo,
    selectedTestAxis.axisThree,
    2, 4, 5
], ids = ["axisOne", "axisTwo", "axisThree", "number for axis one", "number for axis two", "number for axis three"])
def testTHnAxisDetermination(loggingMixin, axisSelection, createHistAxisRange, testSparse):
    """ Test THn axis determination in the HistAxisRange object. """
    # Retrieve sparse.
    sparse, _ = testSparse
    # Retrieve object and setup.
    obj, objectArgs = createHistAxisRange
    obj.axisType = axisSelection

    axisValue = axisSelection.value if isinstance(axisSelection, aenum.Enum) else axisSelection
    assert sparse.GetAxis(axisValue) == obj.axis(sparse)

@pytest.mark.parametrize("minVal, maxVal, minValFunc, maxValFunc, expectedFunc", [
    (0, 10,
        lambda x: projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, x + utils.epsilon),
        lambda x: projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, x - utils.epsilon),
        lambda axis, x, y: axis.SetRangeUser(x, y)),
    (1, 9,
        lambda x: projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, x + utils.epsilon),
        lambda x: projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, x - utils.epsilon),
        lambda axis, x, y: axis.SetRangeUser(x, y)),
    (1, None,
        lambda x: projectors.HistAxisRange.ApplyFuncToFindBin(None, x),
        lambda x: projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.GetNbins),
        lambda axis, x, y: True),  # This is just a no-op. We don't want to restrict the range.
    (0, 7,
        lambda x: projectors.HistAxisRange.ApplyFuncToFindBin(None, x),
        lambda x: projectors.HistAxisRange.ApplyFuncToFindBin(None, x),
        lambda axis, x, y: axis.SetRange(x, y))
], ids = ["0 - 10 with ApplyFuncToFindBin with FindBin", "1 - 9 (mid bin) with ApplyFuncToFindBin with FindBin", "1 - Nbins with ApplyFuncToFindBin (no under/overflow)", "0 - 10 with raw bin value passed ApplyFuncToFindBin"])
def testApplyRangeSet(loggingMixin, minVal, maxVal, minValFunc, maxValFunc, expectedFunc, testSparse):
    """ Test apply a range set to an axis via a HistAxisRange object.

    This is intentionally tested against SetRangeUser, so we can be certain that it reproduces
    that selection as expected.

    Note:
        It doens't matter whether we operate on TH1 or THn, since they both set ranges on TAxis.

    Note:
        This implicity tests ApplyFuncToFindBin, which is fine given how often the two are used
        together (almost always).
    """
    selectedAxis = selectedTestAxis.axisOne
    sparse, _ = testSparse
    expectedAxis = sparse.GetAxis(selectedAxis.value).Clone("axis2")
    expectedFunc(expectedAxis, minVal, maxVal)

    obj = projectors.HistAxisRange(
        axisRangeName = "axisOneTest",
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

def testDisagreementWithSetRangeUser(loggingMixin, testSparse):
    """ Test the disagreement between SetRange and SetRangeUser when the epsilon shift is not included. """
    # Setup values
    selectedAxis = selectedTestAxis.axisOne
    minVal = 2
    maxVal = 8
    sparse, _ = testSparse
    # Detemine expected value (must be first to avoid interfering with applying the axis range)
    expectedAxis = sparse.GetAxis(selectedAxis.value).Clone("axis2")
    expectedAxis.SetRangeUser(minVal, maxVal)

    obj = projectors.HistAxisRange(
        axisRangeName = "axisOneTest",
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
    (ROOT.TAxis.GetNbins, None, 10),
    (ROOT.TAxis.FindBin, 10 - utils.epsilon, 5)
], ids = ["Only value", "Func only", "Func with value"])
def testRetrieveAxisValue(loggingMixin, func, value, expected, testSparse):
    """ Test retrieving axis values using ApplyFuncToFindBin(). """
    selectedAxis = selectedTestAxis.axisOne
    sparse, _ = testSparse
    expectedAxis = sparse.GetAxis(selectedAxis.value)

    assert projectors.HistAxisRange.ApplyFuncToFindBin(func, value)(expectedAxis) == expected

def testProjectors(loggingMixin, testRootHists):
    """ Test creation and basic methods of the projection class. """
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

# Global to allow easier definition of the parametrization
histAxisRangesNamedTuple = collections.namedtuple("histAxisRanges", ["xAxis", "yAxis", "zAxis"])

histAxisRanges = histAxisRangesNamedTuple(
    projectors.HistAxisRange(
        axisType = projectors.TH1AxisType.xAxis,
        axisRangeName = "xAxis",
        minVal = projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, 0.1 + utils.epsilon),
        maxVal = projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, 0.8 - utils.epsilon)),
    projectors.HistAxisRange(
        axisType = projectors.TH1AxisType.yAxis,
        axisRangeName = "yAxis",
        minVal = projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, 0 + utils.epsilon),
        maxVal = projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, 12 - utils.epsilon)),
    projectors.HistAxisRange(
        axisType = projectors.TH1AxisType.zAxis,
        axisRangeName = "zAxis",
        minVal = projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, 10 + utils.epsilon),
        maxVal = projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, 60 - utils.epsilon))
)

histAxisRangesWithNoEntries = histAxisRangesNamedTuple(
    projectors.HistAxisRange(
        axisType = projectors.TH1AxisType.xAxis,
        axisRangeName = "xAxisNoEntries",
        minVal = projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, 0.2 + utils.epsilon),
        maxVal = projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, 0.8 - utils.epsilon)),
    projectors.HistAxisRange(
        axisType = projectors.TH1AxisType.yAxis,
        axisRangeName = "yAxisNoEntries",
        minVal = projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, 4 + utils.epsilon),
        maxVal = projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, 12 - utils.epsilon)),
    projectors.HistAxisRange(
        axisType = projectors.TH1AxisType.zAxis,
        axisRangeName = "zAxisNoEntries",
        minVal = projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, 20 + utils.epsilon),
        maxVal = projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, 60 - utils.epsilon))
)

histAxisRangesRestricted = (
    projectors.HistAxisRange(
        axisType = projectors.TH1AxisType.yAxis,
        axisRangeName = "yAxisLower",
        minVal = projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, 0 + utils.epsilon),
        maxVal = projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, 4 - utils.epsilon)),
    projectors.HistAxisRange(
        axisType = projectors.TH1AxisType.yAxis,
        axisRangeName = "yAxisMiddle",
        minVal = projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, 4 + utils.epsilon),
        maxVal = projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, 8 - utils.epsilon)),
    projectors.HistAxisRange(
        axisType = projectors.TH1AxisType.yAxis,
        axisRangeName = "yAxisUpper",
        minVal = projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, 8 + utils.epsilon),
        maxVal = projectors.HistAxisRange.ApplyFuncToFindBin(ROOT.TAxis.FindBin, 12 - utils.epsilon))
)

# Other axes:
# AAC = Additional Axis Cuts
# PDCA = Projection Dependent Cut Axes
@pytest.mark.parametrize("use_PDCA, additionalCuts, expectedAdditionalCuts", [
    (False, None, True),
    (False, histAxisRanges.yAxis, True),
    (False, histAxisRangesWithNoEntries.yAxis, False),
    (True, None, True),
    (True, [], True),
    (True, [histAxisRanges.yAxis], True),
    (True, [histAxisRangesWithNoEntries.yAxis], False),
    (True, [histAxisRangesRestricted[0], histAxisRangesRestricted[1]], True),
    (True, [histAxisRangesRestricted[1], histAxisRangesRestricted[2]], False)
], ids = [
    "No AAC selection", "AAC with entries", "AAC with no entries",
    "None PDCA", "Empty PDCA", "PDCA",
    "PDCA with no entries", "Disconnected PDCA with entries", "Disconnected PDCA with no entries"
])
# PA = Projection Axes
@pytest.mark.parametrize("projectionAxes, expectedProjectionAxes", [
    (histAxisRanges.xAxis, True),
    (histAxisRangesWithNoEntries.xAxis, False),
], ids = ["PA with entries", "PA without entries"])
def testTH2Projection(loggingMixin, testRootHists,
                      use_PDCA, additionalCuts, expectedAdditionalCuts,
                      projectionAxes, expectedProjectionAxes):
    """ Test projection of a TH2 to a TH1. """
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

    # Check the bin content. There should be one entry at 10, which translates to
    # the bin 1
    nonZeroBins = []
    for x in range(1, proj.GetNcells()):
        if proj.GetBinContent(x) != 0 and not proj.IsBinUnderflow(x) and not proj.IsBinOverflow(x):
            logger.debug(f"non-zero bin at {x}")
            nonZeroBins.append(x)

    expectedCount = 0
    # It will only be non-zero if all of the expected values are true.
    expectedNonZeroCounts = all([expectedAdditionalCuts, expectedProjectionAxes])
    if expectedNonZeroCounts:
        expectedCount = 1
    assert len(nonZeroBins) == expectedCount
    # Check the precise bin which was found.
    if expectedCount != 0:
        # Only check if we actually expected a count
        assert next(iter(nonZeroBins)) == 1

# AAC = Additional Axis Cuts
@pytest.mark.parametrize("additionalAxisCuts, expectedAdditionalAxisCuts", [
    (None, True),
    (histAxisRanges.xAxis, True),
    (histAxisRangesWithNoEntries.xAxis, False)
], ids = ["No AAC selection", "AAC with entries", "AAC with no entries"])
# PDCA = Projection Dependent Cut Axes
@pytest.mark.parametrize("projectionDependentCutAxes, expectedProjectionDependentCutAxes", [
    (None, True),
    ([], True),
    ([histAxisRanges.yAxis], True),
    ([histAxisRangesWithNoEntries.yAxis], False),
    ([histAxisRangesRestricted[0], histAxisRangesRestricted[1]], True),
    ([histAxisRangesRestricted[1], histAxisRangesRestricted[2]], False)
], ids = ["None PDCA", "Empty PDCA", "PDCA", "PDCA with no entries", "Disconnected PDCA with entries", "Disconnected PDCA with no entries"])
# PA = Projection Axes
@pytest.mark.parametrize("projectionAxes, expectedProjectionAxes", [
    (histAxisRanges.zAxis, True),
    (histAxisRangesWithNoEntries.zAxis, False)
], ids = ["PA with entries", "PA without entries"])
def testTH3ToTH1Projection(loggingMixin, testRootHists,
                           additionalAxisCuts, expectedAdditionalAxisCuts,
                           projectionDependentCutAxes, expectedProjectionDependentCutAxes,
                           projectionAxes, expectedProjectionAxes):
    """ Test projection from a TH3 to a TH1 derived class. """
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

    # Check the output.
    assert len(observableList) == 1
    proj = next(iter(observableList.values()))
    assert proj.GetName() == "hist"

    logger.debug("observableList: {}, proj.GetEntries(): {}".format(observableList, proj.GetEntries()))

    expectedBins = 5
    # If we don't expect a count, we've restricted the range further, so we need to reflect this in our check.
    if expectedProjectionAxes is False:
        expectedBins = 4
    assert proj.GetXaxis().GetNbins() == expectedBins

    # Check the bin content. There should be one entry at 10, which translates to
    # the bin 1
    nonZeroBins = []
    for x in range(1, proj.GetXaxis().GetNbins() + 1):
        if proj.GetBinContent(x) != 0:
            nonZeroBins.append(x)

    expectedCount = 0
    # It will only be non-zero if all of the expected values are true.
    expectedNonZeroCounts = all([expectedAdditionalAxisCuts, expectedProjectionDependentCutAxes, expectedProjectionAxes])
    if expectedNonZeroCounts:
        expectedCount = 1
    assert len(nonZeroBins) == expectedCount
    if expectedCount != 0:
        # Only check if we actually expected a count
        assert next(iter(nonZeroBins)) == 1

# Other axes:
# AAC = Additional Axis Cuts
# PDCA = Projection Dependent Cut Axes
@pytest.mark.parametrize("use_PDCA, additionalCuts, expectedAdditionalCuts", [
    (False, None, True),
    (False, histAxisRanges.yAxis, True),
    (False, histAxisRangesWithNoEntries.yAxis, False),
    (True, None, True),
    (True, [], True),
    (True, [histAxisRanges.yAxis], True),
    (True, [histAxisRangesWithNoEntries.yAxis], False),
    (True, [histAxisRangesRestricted[0], histAxisRangesRestricted[1]], True),
    (True, [histAxisRangesRestricted[1], histAxisRangesRestricted[2]], False)
], ids = [
    "No AAC selection", "AAC with entries", "AAC with no entries",
    "None PDCA", "Empty PDCA", "PDCA",
    "PDCA with no entries", "Disconnected PDCA with entries", "Disconnected PDCA with no entries"
])
# PA = Projection Axes
@pytest.mark.parametrize("projectionAxes, expectedProjectionAxes", [
    ([histAxisRanges.zAxis, histAxisRanges.xAxis], True),
    ([histAxisRanges.zAxis, histAxisRangesWithNoEntries.xAxis], False),
    ([histAxisRangesWithNoEntries.zAxis, histAxisRanges.xAxis], False),
    ([histAxisRangesWithNoEntries.zAxis, histAxisRangesWithNoEntries.xAxis], False),
], ids = ["PA with entries", "PA without entries due to x", "PA without entires due to z", "PA without entries"])
def testTH3ToTH2Projection(loggingMixin, testRootHists,
                           use_PDCA, additionalCuts, expectedAdditionalCuts,
                           projectionAxes, expectedProjectionAxes):
    """ Test projection of a TH3 into a TH2. """
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

    # Check the output.
    assert len(observableList) == 1
    proj = next(iter(observableList.values()))
    assert proj.GetName() == "hist"

    logger.debug("observableList: {}, proj.GetEntries(): {}".format(observableList, proj.GetEntries()))

    # Check the axes (they should be in the same order that they are defined above).
    # Use the axis max as a proxy (this function name sux).
    assert proj.GetXaxis().GetXmax() == 60.0
    assert proj.GetYaxis().GetXmax() == 0.8
    logger.debug(f"x axis min: {proj.GetXaxis().GetXmin()}, y axis min: {proj.GetYaxis().GetXmin()}")

    # Check the bin content. There should be one entry at 10, which translates to
    # the bin 1
    nonZeroBins = []
    for x in range(1, proj.GetNcells()):
        if proj.GetBinContent(x) != 0 and not proj.IsBinUnderflow(x) and not proj.IsBinOverflow(x):
            logger.debug(f"non-zero bin at {x}")
            nonZeroBins.append(x)

    expectedCount = 0
    # It will only be non-zero if all of the expected values are true.
    expectedNonZeroCounts = all([expectedAdditionalCuts, expectedProjectionAxes])
    if expectedNonZeroCounts:
        expectedCount = 1
    assert len(nonZeroBins) == expectedCount
    # Check the precise bin which was found.
    if expectedCount != 0:
        # Only check if we actually expected a count
        assert next(iter(nonZeroBins)) == 8

def testTHnProjection(loggingMixin, testSparse):
    """ Test projection of a THnSparse. """
    assert False

@pytest.mark.parametrize("PDCA_axis", [
    histAxisRanges.xAxis,
    histAxisRangesWithNoEntries.xAxis,
], ids = ["Same range PDCA", "Different range PDCA"])
def test_invalid_PDCA_axis(loggingMixin, testRootHists, PDCA_axis):
    """ Test catching a PDCA on the same axis as the projection axis. """
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
    obj.projectionAxes.append(histAxisRanges.xAxis)

    # Perform the projection.
    with pytest.raises(ValueError) as exception_info:
        obj.Project()

    assert "This configuration is not allowed" in exception_info.value.args[0]

