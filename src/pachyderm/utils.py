""" Broad collection of utility functions and constants.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""
from __future__ import annotations

import functools
import logging
import operator
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

# Setup logger
logger = logging.getLogger(__name__)

# Small value - epsilon
# For use to offset from bin edges when finding bins for use with SetRange()
# NOTE: sys.float_info.epsilon is too small in some cases and thus should be avoided
EPSILON = 1e-5


###################
# Utility functions
###################
def moving_average(arr: npt.NDArray[Any], n: int = 3) -> npt.NDArray[Any]:
    """Calculate the moving overage over an array.

    Algorithm from: https://stackoverflow.com/a/14314054

    Args:
        arr (np.ndarray): Array over which to calculate the moving average.
        n (int): Number of elements over which to calculate the moving average. Default: 3
    Returns:
        np.ndarray: Moving average calculated over n.
    """
    ret = np.cumsum(arr, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    res: npt.NDArray[Any] = ret[n - 1 :] / n
    return res


def recursive_getattr(obj: Any, attr: str, *args: Any) -> Any:
    """Recursive ``getattr``.

    This can be used as a drop in for the standard ``getattr(...)``. Credit to:
    https://stackoverflow.com/a/31174427

    Args:
        obj: Object to retrieve the attribute from.
        attr: Name of the attribute, with each successive attribute separated by a ".".
    Returns:
        The requested attribute. (Same as ``getattr``).
    Raises:
        AttributeError: If the attribute was not found and no default was provided. (Same as ``getattr``).
    """

    def _getattr(obj_recurse: Any, attr_recurse: str) -> Any:
        return getattr(obj_recurse, attr_recurse, *args)

    return functools.reduce(_getattr, [obj, *attr.split(".")])


def recursive_setattr(obj: Any, attr: str, val: Any) -> Any:
    """Recursive ``setattr``.

    This can be used as a drop in for the standard ``setattr(...)``. Credit to:
    https://stackoverflow.com/a/31174427

    Args:
        obj: Object to retrieve the attribute from.
        attr: Name of the attribute, with each successive attribute separated by a ".".
        value: Value to set the attribute to.
    Returns:
        The requested attribute. (Same as ``getattr``).
    Raises:
        AttributeError: If the attribute was not found and no default was provided. (Same as ``getattr``).
    """
    pre, _, post = attr.rpartition(".")
    return setattr(recursive_getattr(obj, pre) if pre else obj, post, val)


def recursive_getitem(d: Mapping[str, Any], keys: str | Sequence[str]) -> Any:
    """Recursively retrieve an item from a nested dict.

    Credit to: https://stackoverflow.com/a/52260663

    Args:
        d: Mapping of strings to objects.
        keys: Names of the keys under which the object is stored. Can also just be a single string.
    Returns:
        The object stored under the keys.
    Raises:
        KeyError: If one of the keys isn't found.
    """
    # If only a string, then just just return the item
    if isinstance(keys, str):
        return d[keys]
    return functools.reduce(operator.getitem, keys, d)
