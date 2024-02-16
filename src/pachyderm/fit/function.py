""" Functions for use with fitting.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""
from __future__ import annotations

import abc
import logging
import operator
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from pachyderm import generic_class
from pachyderm.fit import base as fit_base

logger = logging.getLogger(__name__)


class CombinePDF(generic_class.EqualityMixin, abc.ABC):
    """Combine functions (PDFs) together.

    Args:
        functions: Functions to be added.
        prefixes: Prefix for arguments of each function. Default: None. If specified, there must
            be one prefix for each function.
        skip_prefixes: Prefixes to skip when assigning prefixes. As noted in probfit, this can be
            useful to mix prefixed and non-prefixed arguments. Default: None.

    Attributes:
        functions: List of functions that are combined in the PDF.
        func_code: Function arguments derived from the fit functions. They need to be separately
            specified to allow iminuit to determine the proper arguments.
        argument_positions: Map of merged arguments to the arguments for each individual function.
    """

    # Don't specify the function arguments to work around a mypy bug.
    # For an unclear reason, it won't properly detect the number of arguments.
    _operation: Callable[..., float]
    _call_function: Callable[..., float]

    def __init__(
        self,
        *functions: Callable[..., float],
        prefixes: Sequence[str] | None = None,
        skip_prefixes: Sequence[str] | None = None,
    ) -> None:
        # Store the functions
        self.functions = list(functions)

        # Determine the arguments for the functions.
        merged_args, argument_positions = fit_base.merge_func_codes(
            self.functions, prefixes=prefixes, skip_prefixes=skip_prefixes
        )
        logger.debug(f"merged_args: {merged_args}")
        self.func_code = fit_base.FuncCode(merged_args)
        self.argument_positions = argument_positions

    def __call__(self, x: npt.NDArray[Any], *merged_args: float) -> float:
        """Call the added PDF.

        Args:
            x: Value(s) where the functions should be evaluated.
            merged_args: Merged arguments for the functions. Must contain all of the arguments
                need to call the functions.
        Returns:
            Value(s) of the functions when evaluated with the given input values.
        """
        # We add in the x values into the function arguments here so we don't have to play tricks later
        # to get the function argument indices correct.
        return fit_base.call_list_of_callables_with_operation(
            self._operation,
            self.functions,
            self.argument_positions,
            *[x, *merged_args],  # type: ignore[arg-type]
        )


class AddPDF(CombinePDF):
    """Add functions (PDFs) together.

    Args:
        functions: Functions to be added.
        prefixes: Prefix for arguments of each function. Default: None. If specified, there must
            be one prefix for each function.
        skip_prefixes: Prefixes to skip when assigning prefixes. As noted in probfit, this can be
            useful to mix prefixed and non-prefixed arguments. Default: None.

    Attributes:
        functions: List of functions that are combined in the PDF.
        func_code: Function arguments derived from the fit functions. They need to be separately
            specified to allow iminuit to determine the proper arguments.
        argument_positions: Map of merged arguments to the arguments for each individual function.
    """

    _operation = operator.add


class SubtractPDF(CombinePDF):
    """Subtract functions (PDFs) together.

    Args:
        functions: Functions to be added.
        prefixes: Prefix for arguments of each function. Default: None. If specified, there must
            be one prefix for each function.
        skip_prefixes: Prefixes to skip when assigning prefixes. As noted in probfit, this can be
            useful to mix prefixed and non-prefixed arguments. Default: None.

    Attributes:
        functions: List of functions that are combined in the PDF.
        func_code: Function arguments derived from the fit functions. They need to be separately
            specified to allow iminuit to determine the proper arguments.
        argument_positions: Map of merged arguments to the arguments for each individual function.
    """

    _operation = operator.sub


class MultiplyPDF(CombinePDF):
    """Multiply functions (PDFs) together.

    Args:
        functions: Functions to be added.
        prefixes: Prefix for arguments of each function. Default: None. If specified, there must
            be one prefix for each function.
        skip_prefixes: Prefixes to skip when assigning prefixes. As noted in probfit, this can be
            useful to mix prefixed and non-prefixed arguments. Default: None.

    Attributes:
        functions: List of functions that are combined in the PDF.
        func_code: Function arguments derived from the fit functions. They need to be separately
            specified to allow iminuit to determine the proper arguments.
        argument_positions: Map of merged arguments to the arguments for each individual function.
    """

    _operation = operator.mul


class DividePDF(CombinePDF):
    """Divide functions (PDFs) together.

    Args:
        functions: Functions to be added.
        prefixes: Prefix for arguments of each function. Default: None. If specified, there must
            be one prefix for each function.
        skip_prefixes: Prefixes to skip when assigning prefixes. As noted in probfit, this can be
            useful to mix prefixed and non-prefixed arguments. Default: None.

    Attributes:
        functions: List of functions that are combined in the PDF.
        func_code: Function arguments derived from the fit functions. They need to be separately
            specified to allow iminuit to determine the proper arguments.
        argument_positions: Map of merged arguments to the arguments for each individual function.
    """

    _operation = operator.truediv


def gaussian(x: npt.NDArray[np.float64] | float, mean: float, sigma: float) -> npt.NDArray[np.float64] | float:
    r"""Normalized gaussian.

    .. math::

        f = 1 / \sqrt{2 * \pi * \sigma^{2}} * \exp{-\frac{(x - \mu)^{2}}{(2 * \sigma^{2}}}

    Args:
        x: Value(s) where the gaussian should be evaluated.
        mean: Mean of the gaussian distribution.
        sigma: Width of the gaussian distribution.
    Returns:
        Calculated gaussian value(s).
    """
    return 1.0 / np.sqrt(2 * np.pi * np.square(sigma)) * np.exp(-1.0 / 2.0 * np.square((x - mean) / sigma))  # type: ignore[no-any-return]


def extended_gaussian(
    x: npt.NDArray[np.float64] | float, mean: float, sigma: float, amplitude: float
) -> npt.NDArray[np.float64] | float:
    r"""Extended gaussian.

    .. math::

        f = A / \sqrt{2 * \pi * \sigma^{2}} * \exp{-\frac{(x - \mu)^{2}}{(2 * \sigma^{2}}}

    Args:
        x: Value(s) where the gaussian should be evaluated.
        mean: Mean of the gaussian distribution.
        sigma: Width of the gaussian distribution.
        amplitude: Amplitude of the gaussian.
    Returns:
        Calculated gaussian value(s).
    """
    return amplitude / np.sqrt(2 * np.pi * np.square(sigma)) * np.exp(-1.0 / 2.0 * np.square((x - mean) / sigma))  # type: ignore[no-any-return]
