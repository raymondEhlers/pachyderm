#!/usr/bin/env python3

""" Models for fitting.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import abc
import logging
import operator
from typing import Any, Callable, Dict, Iterable, Iterator, TypeVar, Union

import iminuit
import numpy as np
import scipy.integrate

from pachyderm import generic_class, histogram
from pachyderm.fit import base as fit_base

logger = logging.getLogger(__name__)

T_CostFunction = TypeVar("T_CostFunction", bound = "CostFunctionBase")

def _quad(f: Callable[..., float], bin_edges: np.ndarray, *args: Union[float, np.ndarray]) -> np.ndarray:
    """ Integrate over the given function using QUADPACK.

    Unfortunately, this option is fairly slow because we can't take advantage of vectorized numpy operations.
    Something like numba could speed this up if all functions and classes could be supported.

    Args:
        f: Function to integrate.
        bin_edges: Bin edges of the data histogram.
        args: Arguments for evaluating the function.
    Returns:
        Integral over each bin.
    """
    values = []
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        res, _ = scipy.integrate.quad(func = f, a = lower, b = upper, args = tuple(args))
        values.append(res)
    return np.array(values)

def _simpson_38(f: Callable[..., float], bin_edges: np.ndarray, *args: Union[float, np.ndarray]) -> np.ndarray:
    """ Integrate over each histogram bin with the Simpson 3/8 rule.

    Implemented via the expression at https://en.wikipedia.org/wiki/Simpson%27s_rule#Simpson's_3/8_rule .

    Could also be implemented via ``scipy.integrate.simps``, but it implements the composite
    rule which, as far as I can tell, doesn't map onto our histogram data quite as nicely
    as our own implementation.

    Args:
        f: Function to integrate.
        bin_edges: Bin edges of the data histogram.
        args: Arguments for evaluating the function.
    Returns:
        Integral over each bin.
    """
    a = bin_edges[:-1]
    b = bin_edges[1:]
    # Recall that bin_edges[1:] - bin_edges[:-1] is the bin widths
    return (b - a) / 8 * (f(a, *args) + 3 * f((2 * a + b) / 3, *args) + 3 * f((a + 2 * b) / 3, *args) + f(b, *args))

def _integrate_1D(f: Callable[..., float], bin_edges: np.ndarray, *args: Union[float, np.ndarray]) -> np.ndarray:
    """ Integrate the given function over each bin in 1D.

    A number of options are available for integration, including a simple method evaluated on
    the bin edges, Simpson's 3/8 rule, and using QUADPACK.

    Note:
        Integration depends on the bin edges, which implicitly undoes the bin width scaling that
        is applied to a histogram. To compensate, we divide the integrated values by the bin width.

    Args:
        f: Function to integrate.
        bin_edges: Bin edges of the data histogram.
        args: Arguments for evaluating the function.
    Returns:
        Integral over each bin.
    """
    # Simplest case where we just evaluate on the edges.
    #return (f(bin_edges[1:], *args) + f(bin_edges[:-1], *args)) / 2 * (bin_edges[1:] - bin_edges[:-1])
    # QUADPACK is another option, but it's slow.
    #return _quad(f, bin_edges, *args) / (bin_edges[1:] - bin_edges[:-1])
    # Simpson's 3/8 rule is better than the simple case, but faster than QUADPACK.
    return _simpson_38(f, bin_edges, *args) / (bin_edges[1:] - bin_edges[:-1])

def unravel_simultaneous_fits(functions: Iterable[Union["CostFunctionBase", "SimultaneousFit"]]
                              ) -> Iterator["CostFunctionBase"]:
    """ Unravel the cost functions from possible simultaneous fit objects.

    The functions are unravel by recursively retrieving the functions from existing ``SimultaneousFit`` objects
    that may be in the list of passed functions. The cost functions store their fit data, so they are fully
    self contained. Consequently, we are okay to fully unravel the functions without worrying about the
    intermediate ``SimultaneousFit`` objects.

    Args:
        functions: Functions to unravel.
    Returns:
        Iterator of the base cost functions.
    """
    for f in functions:
        if isinstance(f, SimultaneousFit):
            yield from unravel_simultaneous_fits(f.cost_functions)
        else:
            yield f

class SimultaneousFit(generic_class.EqualityMixin):
    """ Cost function for the simultaneous fit of the given cost functions.

    Args:
        cost_functions: The cost functions.

    Attributes:
        functions: The cost functions.
        func_code: Function arguments derived from the fit functions. They need to be separately
            specified to allow iminuit to determine the proper arguments.
        argument_positions: Map of merged arguments to the arguments for each individual function.
    """
    def __init__(self, *cost_functions: Union[T_CostFunction, "SimultaneousFit"]):
        # Validation
        # Ensure that we unravel any SimultaneousFit objects to their base cost functions.
        funcs = list(unravel_simultaneous_fits(list(cost_functions)))

        self.cost_functions = funcs
        logger.debug("Simultaneous Fit")
        merged_args, argument_positions = fit_base.merge_func_codes(self.cost_functions)
        # We don't drop any of the arguments here because the cost functions already did it.
        self.func_code = fit_base.FuncCode(merged_args)
        self.argument_positions = argument_positions

    def __add__(self, other: Union[T_CostFunction, "SimultaneousFit"]) -> "SimultaneousFit":
        """ Add a new function to the simultaneous fit. """
        return type(self)(self, other)

    def __radd__(self, other: Union[T_CostFunction, "SimultaneousFit"]) -> "SimultaneousFit":
        """ For use with ``sum(...)``. """
        if other == 0:
            return self
        else:
            return self + other

    def __call__(self, *args: float) -> float:
        """ Calculate the cost function for all x values in the data. """
        return fit_base.call_list_of_callables_with_operation(
            operator.add, self.cost_functions, self.argument_positions, *args
        )

class CostFunctionBase(abc.ABC):
    """ Base cost function.

    Args:
        f: The fit function.
        data: Data to be used for fitting.
        additional_call_options: Additional keyword options to be passed when calling the cost function.
    Attributes:
        f: The fit function.
        func_code: Function arguments derived from the fit function. They need to be separately specified
            to allow iminuit to determine the proper arguments.
        data: Data to be used for fitting.
        additional_call_options: Additional keyword options to be passed when calling the cost function.
        _cost_function: Function to be used to calculate the actual cost function.
    """
    _cost_function: Callable[..., float]

    def __init__(self, f: Callable[..., float], data: Union[np.ndarray, histogram.Histogram1D],
                 **additional_call_options: Any):
        # If using numba, we would need to JIT the function to be able to pass it to the cost function.
        self.f = f
        # We need to drop the leading x argument
        self.func_code = fit_base.FuncCode(iminuit.util.describe(self.f)[1:])
        self.data = data
        self._additional_call_options: Dict[str, Any] = additional_call_options

    def __add__(self: T_CostFunction, other: T_CostFunction) -> SimultaneousFit:
        """ Creates a simultaneous fit when added with another cost function. """
        return SimultaneousFit(self, other)

    def __radd__(self: T_CostFunction, other: T_CostFunction) -> Union[T_CostFunction, SimultaneousFit]:
        """ For use with ``sum(...)``. """
        if other == 0:
            return self
        else:
            return self + other

    def __call__(self, *args: float) -> float:
        """ Calculate the cost function for all x values in the data. """
        return self._call_cost_function(self.data, self.f, *args, **self._additional_call_options)

    @classmethod
    @abc.abstractmethod
    def _call_cost_function(cls, data: Union[np.ndarray, histogram.Histogram1D],
                            f: Callable[..., float], *args: Union[float, np.ndarray], **kwargs: Any) -> float:
        """ Wrapper to allow access to the method as if it's unbound.

        This is needed for use with numba.

        Args:
            data: The input data.
            f: Fit function.
            args: Other arguments for the fit function (not including where it will be evaluated (ie. x)).
            kwargs: Additional arguments to pass to the cost function.
        Returns:
            The cost function evaluated at all of the corresponding data points.
        """
        ...

class StandaloneCostFunction(CostFunctionBase):
    """ Cost function which only needs a list of input data.

    This is in contrast to those which need data to compare against at each point. One example of
    a cost function which only needs the input data is the unbinned log likelihood.

    Attributes:
        f: The fit function.
        func_code: Function arguments derived from the fit function. They need to be separately specified
            to allow iminuit to determine the proper arguments.
        data: Numpy array of all input values (not binned in any way). It's just a list of the values.
        _cost_function: Function to be used to calculate the actual cost function.
    """
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Check that we have the proper input data. This isn't very pythnoic, but it's
        # important that have the data in the proper format.
        assert isinstance(self.data, np.ndarray)

    @classmethod
    def _call_cost_function(cls, data: np.ndarray,
                            f: Callable[..., float], *args: Union[float, np.ndarray], **kwargs: Any) -> float:
        """ Wrapper to allow access to the method as if it's unbound.

        This is needed for use with numba.

        Args:
            data: The input histogram. This should simply be an array of the values.
        Returns:
            The cost function evaluated at all of the corresponding data points (ie. ``self.data``).
        """
        return cls._cost_function(data, f, *args, **kwargs)

class DataComparisonCostFunction(CostFunctionBase):
    """ Cost function which needs comparison data, the points where it was evaluated, and the errors.

    This is in contrast to those which only need the input data. Examples of cost functions needing
    input data included the chi squared (both unbinned and binned), as well as the binned log likelihood.

    Attributes:
        f: The fit function.
        func_code: Function arguments derived from the fit function. They need to be separately specified
            to allow iminuit to determine the proper arguments.
        data: Numpy array of all input values (not binned in any way). It's just a list of the values.
        _cost_function: Function to be used to calculate the actual cost function.
    """
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # Check that we have the proper input data. This isn't very pythnoic, but it's
        # important that have the data in the proper format.
        assert isinstance(self.data, histogram.Histogram1D)

    @classmethod
    def _call_cost_function(cls, data: Any,
                            f: Callable[..., float], *args: Union[float, np.ndarray], **kwargs: Any) -> float:
        """ Wrapper to allow access to the method as if it's unbound.

        This is needed for use with numba.

        Args:
            data: Input data in the form of a histogram. Note that this must be a ``Histogram1D``
                (but it's specified as Any above to avoid needing isinstance calls in performance
                sensitive code).
            f: Fit function.
            args: Arguments for the function.
        Returns:
            The cost function evaluated at all the corresponding data points (ie. data.x).
        """
        return cls._cost_function(data.x, data.y, data.errors, data.bin_edges, f, *args, **kwargs)

def _chi_squared(x: np.ndarray, y: np.ndarray,
                 errors: np.ndarray, _: np.ndarray,
                 f: Callable[..., float], *args: float) -> Any:
    r""" Actual implementation of the chi squared.

    Implemented with some help from the iminuit advanced tutorial. The chi squared is defined as:

    .. math::

        \Chi^{2} = \sum_{i} (\frac{(y_{i} - f(x, *args)}{error_{i})})^{2}

    Note:
        It returns a float, but numba can't handle cast. So we return ``Any`` and then cast the result.

    Args:
        x: x values for calculation.
        y: y values for calculation.
        errors: Errors for calculation.
        _: Ignored here.
        f: Fit function.
        args: Additional arguments for the fit function.
    Returns:
        (Unbinned) chi squared for the given arguments.
    """
    return np.sum(np.square((y - f(x, *args)) / errors))

class ChiSquared(DataComparisonCostFunction):
    """ chi^2 cost function.

    Attributes:
        f: The fit function.
        func_code: Function arguments derived from the fit function. They need to be separately specified
            to allow iminuit to determine the proper arguments.
        data: Data to be used for fitting.
        _cost_function: Function to be used to calculate the actual cost function.
    """
    _cost_function = _chi_squared

def _binned_chi_squared(x: np.ndarray, y: np.ndarray,
                        errors: np.ndarray, bin_edges: np.ndarray,
                        f: Callable[..., float], *args: float) -> Any:
    r""" Actual implementation of the binned chi squared.

    The binned chi squared is defined as:

    .. math::

        \Chi^{2} = \sum_{i} (\frac{(y_{i} - \int_{i \mathrm{lower edge}}^{i \mathrm{upper edge}} f(x, *args)}{error_{i})})^{2}

    where the function is integrated over each bin.

    Note:
        It returns a float, but numba can't handle cast. So we return ``Any`` and then cast the result.

    Args:
        x: x values where the function should be evaluated.
        y: Histogram values at each x.
        errors: Histogram errors at each x.
        bin_edges: Histogram bin edges.
        f: Fit function.
        args: Arguments for the fit function.
    Returns:
        Binned chi squared calculated for each x value.
    """
    # ROOT appears to use the unbinned chi squared, despite working with binned data
    #return np.sum(np.square((y - f(x, *args)) / errors))
    # Evaluate the function value over the entire bin.
    expected_values = _integrate_1D(f, bin_edges, *args)
    return np.sum(np.square((y - expected_values) / errors))

def binned_chi_squared_safe_for_zeros(x: np.ndarray, y: np.ndarray,
                                      errors: np.ndarray, bin_edges: np.ndarray,
                                      f: Callable[..., float], *args: float) -> Any:
    """ Actual implementation of the binned chi squared.

    See `_binned_chi_squared` for further information. This function is just the standard binned chi squared,
    but the division is protected from divide by 0. This allows safe use when calculating a binned chi squared.

    Args:
        x: x values where the function should be evaluated.
        y: Histogram values at each x.
        errors: Histogram errors at each x.
        bin_edges: Histogram bin edges.
        f: Fit function.
        args: Arguments for the fit function.
    Returns:
        Binned chi squared calculated for each x value.
    """
    expected_values = _integrate_1D(f, bin_edges, *args)
    return np.sum(np.square(
        np.divide((y - expected_values), errors, out = np.zeros_like(errors), where = errors != 0)
    ))

class BinnedChiSquared(DataComparisonCostFunction):
    """ Binned chi^2 cost function.

    Calling this class will calculate the chi squared. Implemented with some help from ...

    Attributes:
        f: The fit function.
        func_code: Function arguments derived from the fit function. They need to be separately specified
            to allow iminuit to determine the proper arguments.
        data: Data to be used for fitting.
        _cost_function: Function to be used to calculate the actual cost function.
    """
    _cost_function = _binned_chi_squared

def _log_likelihood(data: np.ndarray, f: Callable[..., float], *args: float) -> Any:
    r""" Actual implementation of the log likelihood cost function.

    The unbinned log likelihood is defined as:

    .. math::

        \mathrm{likelihood} = - \sum_{i \in data} \log{f(x, *args)}

    Note:
        It returns a float, but numba can't handle cast. So we return ``Any`` and then cast the result.

    Args:
        data: Data points (raw data, not histogramed).
        f: Fit function.
        args: Fit function arguments.
    Returns:
        Unbinned log likelihood for the given data points.
    """
    return np.log(f(data, *args))

class LogLikelihood(StandaloneCostFunction):
    """ Log likelihood cost function.

    Calling this class will calculate the chi squared. Implemented with some help from ...

    Attributes:
        f: The fit function.
        func_code: Function arguments derived from the fit function. They need to be separately specified
            to allow iminuit to determine the proper arguments.
        data: Data to be used for fitting.
        _cost_function: Function to be used to calculate the actual cost function.
    """
    _cost_function = _log_likelihood

def _extended_binned_log_likelihood(x: np.ndarray, y: np.ndarray,
                                    errors: np.ndarray, bin_edges: np.ndarray,
                                    f: Callable[..., float], *args: float, use_weights: bool = False) -> Any:
    r""" Actual implementation of the extended binned log likelihood (cost function).

    Based on Probfit's binned log likelihood implementation. I also looked at
    ROOT's ``FitUtil::EvaluatePoissonLogL(...)`` for evaluating the binned log likelihood,
    and they appear to be consistent.

    The actual expression that is implemented is:

    .. math::

        \textrm{Likelihood} = -\sum_{i \in bins} s_i \times  \left(  h_i \times \log (\frac{E_i}{h_i}) + (h_i-E_i) \right)

    where E_i is the expected value (from the fit function), and h_i is the histogram. To make a proper
    comparison between E_i and h_i, we need to evaluate the value of the entire bin. To do so, we use:

    .. math::

        E_i = \frac{f(l_i, arg\ldots )+f(r_i, arg \ldots )}{2} \times b_i

    and s_i is unity when performing a standard fit, and:

    .. math::

        s_i = h_i / error_i^2

    when performing a weighted fit.

    Note:
        It returns a float, but numba can't handle cast. So we return ``Any`` and then cast the result.

    Args:
        x: x values for calculation.
        y: y values for calculation.
        errors: Errors for calculation.
        bin_edges: Bin widths of the histogram.
        f: Fit function.
        args: Additional arguments for the fit function.
        use_weights: Use data weights in calculating the log likelihood.
    Returns:
        Binned log likelihood.
    """
    # Need to normalize the contributions.
    scale = y / errors ** 2 if use_weights else np.ones(len(y))
    expected_values = _integrate_1D(f, bin_edges, *args)
    # We don't use log rules to combine the log1p expressions (ie. log(expected_values / y)) because it appears
    # to create numerical issues (throwing NaN).
    # It sounds like the absolute value of FCN doesn't necessarily mean much for the log likelihood ratio
    # In principle, I should be able to get it to match ROOT, but that doesn't seem so trivial in practice.
    return -1 * np.sum(scale * (y * (np.log1p(expected_values) - np.log1p(y)) + (y - expected_values)))

class BinnedLogLikelihood(DataComparisonCostFunction):
    """ Binned log likelihood cost function.

    Calling this class will calculate the chi squared. Implemented with some help from ...

    Args:
        f: The fit function.
        data: Data to be used for fitting.
        use_weights: Whether to use the data weights when calculating the cost function.
            This is equivalent to the "WL" option in ROOT. Default: False.
        additional_call_options: Additional keyword options to be passed when calling the cost function.
    Attributes:
        f: The fit function.
        func_code: Function arguments derived from the fit function. They need to be separately specified
            to allow iminuit to determine the proper arguments.
        data: Data to be used for fitting.
        _cost_function: Function to be used to calculate the actual cost function.
    """
    def __init__(self, use_weights: bool = False, *args: Any, **kwargs: Any):
        kwargs["use_weights"] = use_weights
        super().__init__(*args, **kwargs)

    _cost_function = _extended_binned_log_likelihood
