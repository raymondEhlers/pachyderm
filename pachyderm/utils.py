#!/usr/bin/env python

""" (Mainly histogram) utilities.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import collections
import logging
import numpy as np
import ruamel.yaml

# Setup logger
logger = logging.getLogger(__name__)

# Small value - epsilon
# For use to offset from bin edges when finding bins for use with SetRange()
# NOTE: sys.float_info.epsilon is too small in some cases and thus should be avoided
epsilon = 1e-5

###################
# Utility functions
###################
def getHistogramsInList(filename, listName = "AliAnalysisTaskJetH_tracks_caloClusters_clusbias5R2GA"):
    """ Get histograms from the file and make them available in a dict.

    Lists are recursively explored, with all lists converted to dictionaries, such that the return
    dictionaries which only contains hists and dictionaries of hists (ie there are no ROOT ``TCollection``
    derived objects).

    Args:
        filename (str): Filename of the ROOT file containing the list.
        listName (str): Name of the list to retrieve.
    Returns:
        dict: Contains hists with keys as their names. Lists are recursively added, mirroring
            the structure under which the hists were stored.
    """
    import ROOT

    hists = collections.OrderedDict()
    fIn = ROOT.TFile(filename, "READ")
    hist_list = fIn.Get(listName)
    if not hist_list:
        logger.critical("Could not find list with name \"{}\". Possible names include:".format(listName))
        fIn.ls()
        return None

    # Retrieve objects in the hist list
    for obj in hist_list:
        retrieve_object(hists, obj)

    # Cleanup
    fIn.Close()

    return hists

def retrieve_object(output_dict, obj):
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
        output_dict[obj.GetName()] = collections.OrderedDict()
        for obj_temp in list(obj):
            retrieve_object(output_dict[obj.GetName()], obj_temp)

def readYAML(filename: str, fileAccessMode: str = "r") -> dict:
    """ Read the YAML file at filename.

    Uses the round trip mode.

    Args:
        filename (str): Filename of the YAML file to be read.
        fileAccessMode (str): Mode under which the file should be opened
    Returns:
        dict-like: Dict containing the parameters read from the YAML file.
    """
    parameters = None
    with open(filename, fileAccessMode) as f:
        yaml = ruamel.yaml.YAML(typ = "rt")
        yaml.default_flow_style = False
        parameters = yaml.load(f)
    return parameters

def writeYAML(parameters: dict, filename: str, fileAccessMode: str = "w"):
    """ Write the given output dict to file using YAML.

    Uses the round trip mode.

    Args:
        parameters (dict): Output parameters to be written to the YAML file.
        filename (str): Filename of the YAML file to write.
        fileAccessMode (str): Mode under which the file should be opened
    Returns:
        None.
    """
    with open(filename, fileAccessMode) as f:
        yaml = ruamel.yaml.YAML(typ = "rt")
        yaml.default_flow_style = False
        yaml.dump(parameters, f)

def movingAverage(arr, n=3):
    """ Calculate the moving overage over an array.

    Algorithm from: https://stackoverflow.com/a/14314054

    Args:
        arr (np.ndarray): Array over which to calculate the moving average.
        n (int): Number of elements over which to calculate the moving average. Default: 3
    Returns:
        np.ndarray: Moving average calculated over n.
    """
    ret = np.cumsum(arr, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def getArrayFromHist(observable):
    """ Return array of data from a histogram.

    Args:
        observable (JetHUtils.Observable or ROOT.TH1): Histogram from which the array should be extracted.
    Returns:
        dict: "y": hist data, "errors" : y errors, "binCenters" : x bin centers
    """
    try:
        hist = observable.hist.hist
    except AttributeError:
        hist = observable
    #logger.debug("hist: {}".format(hist))
    xAxis = hist.GetXaxis()
    # Don't include overflow
    xBins = range(1, xAxis.GetNbins() + 1)
    array_from_hist = np.array([hist.GetBinContent(i) for i in xBins])
    # NOTE: The bin error is stored with the hist, not the axis.
    errors = np.array([hist.GetBinError(i) for i in xBins])
    bin_centers = np.array([xAxis.GetBinCenter(i) for i in xBins])
    return {"y": array_from_hist, "errors": errors, "binCenters": bin_centers}

def getArrayFromHist2D(hist, setZeroToNaN = True):
    """ Extract the necessary data from the hist.

    Converts the histogram into a numpy array, and suitably processes it for a surface plot
    by removing 0s (which can cause problems when taking logs), and returning the bin centers
    for (X,Y).

    Note:
        This is a different format than the 1D version!

    Args:
        hist (ROOT.TH2): Histogram to be converted.
        setZeroToNaN (bool): If true, set 0 in the array to NaN. Useful with matplotlib so that
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
    if setZeroToNaN:
        hist_array[hist_array == 0] = np.nan

    # We want an array of bin centers
    xRange = np.array([hist.GetXaxis().GetBinCenter(i) for i in range(1, hist.GetXaxis().GetNbins() + 1)])
    yRange = np.array([hist.GetYaxis().GetBinCenter(i) for i in range(1, hist.GetYaxis().GetNbins() + 1)])
    X, Y = np.meshgrid(xRange, yRange)

    return (X, Y, hist_array)

def getArrayForFit(observables, trackPtBin, jetPtBin):
    """ Get an hist data as a np.ndarray based on selected bins. This is often used
    to retrieve data for fitting.

    Args:
        observables (dict): The observables from which the hist should be retrieved.
        trackPtBin (int): Track pt bin of the desired hist.
        jetPtbin (int): Jet pt bin of the desired hist.
    Returns:
        dict: "y": hist data, "errors" : y errors, "binCenters" : x bin centers (Values from `getArrayFromHist()`).
    """
    for name, observable in observables.items():
        if observable.trackPtBin == trackPtBin and observable.jetPtBin == jetPtBin:
            return getArrayFromHist(observable)

