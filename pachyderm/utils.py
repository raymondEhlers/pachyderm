#!/usr/bin/env python

""" Broad collection of utility functions and constants.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import numpy as np
import ruamel.yaml

from pachyderm import histogram

# Setup logger
logger = logging.getLogger(__name__)

# Small value - epsilon
# For use to offset from bin edges when finding bins for use with SetRange()
# NOTE: sys.float_info.epsilon is too small in some cases and thus should be avoided
epsilon = 1e-5

###################
# Utility functions
###################
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

def writeYAML(parameters: dict, filename: str, fileAccessMode: str = "w") -> None:
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

def movingAverage(arr: np.ndarray, n: int = 3) -> np.ndarray:
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

def getArrayForFit(observables: dict, trackPtBin: int, jetPtBin: int) -> histogram.Histogram1D:
    """ Get a Histogram1D associated with the selected jet and track pt bins.

    This is often used to retrieve data for fitting.

    Args:
        observables (dict): The observables from which the hist should be retrieved.
        trackPtBin (int): Track pt bin of the desired hist.
        jetPtbin (int): Jet pt bin of the desired hist.
    Returns:
        Histogram1D: Converted TH1 or uproot histogram.
    Raises:
        ValueError: If the requested observable couldn't be found.
    """
    for name, observable in observables.items():
        if observable.trackPtBin == trackPtBin and observable.jetPtBin == jetPtBin:
            return histogram.Histogram1D.from_existing_hist(observable.hist)

    raise ValueError("Cannot find fit with jet pt bin {jetPtBin} and track pt bin {trackPtBin}")
