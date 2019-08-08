#1/usr/bin/env python3

""" Base module for performing fits with Minuit.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from dataclasses import dataclass
import iminuit
import itertools
import logging
import numdifftools as nd
import numpy as np
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type, TYPE_CHECKING, TypeVar, Union

from pachyderm import generic_class

if TYPE_CHECKING:
    from pachyderm.fit import cost_function

logger = logging.getLogger(__name__)

# Typing
T_FuncCode = TypeVar("T_FuncCode", bound = "FuncCode")
T_ArgumentPositions = List[List[int]]
_T_FitResult = TypeVar("_T_FitResult", bound = "FitResult")

class FitFailed(Exception):
    """ Raised if the fit failed. The message will include further details. """
    pass

@dataclass
class FitResult:
    """ Fit result base class.

    Note:
        free_parameters + fixed_parameters = parameters

    Attributes:
        parameters: Names of the parameters used in the fit.
        free_parameters: Names of the free parameters used in the fit.
        fixed_parameters: Names of the fixed parameters used in the fit.
        values_at_minimum: Contains the values of the full RP fit function at the minimum. Keys are the
            names of parameters, while values are the numerical values at convergence.
        errors_on_parameters: Contains the values of the errors associated with the parameters
            determined via the fit.
        covariance_matrix: Contains the values of the covariance matrix. Keys are tuples
            with (param_name_a, param_name_b), and the values are covariance between the specified parameters.
            Note that fixed parameters are _not_ included in this matrix.
        x: x values where the fit result should be evaluated.
        n_fit_data_points: Number of data points used in the fit.
        minimum_val: Minimum value of the fit when it coverages. This is the chi squared value for a
            chi squared minimization fit.
        errors: Store the errors associated with the component fit function.
    """
    parameters: List[str]
    free_parameters: List[str]
    fixed_parameters: List[str]
    values_at_minimum: Dict[str, float]
    errors_on_parameters: Dict[str, float]
    covariance_matrix: Dict[Tuple[str, str], float]
    x: np.ndarray
    n_fit_data_points: int
    minimum_val: float
    errors: np.ndarray

    @property
    def nDOF(self) -> int:
        """ Number of degrees of freedom. """
        return self.n_fit_data_points - len(self.free_parameters)

    @property
    def correlation_matrix(self) -> Dict[Tuple[str, str], float]:
        """ The correlation matrix of the free parameters.

        These values are derived from the covariance matrix values stored in the fit.

        Note:
            This property caches the correlation matrix value so we don't have to calculate it every time.

        Args:
            None
        Returns:
            The correlation matrix of the fit result.
        """
        try:
            # We attempt to cache the covaraince matrix, so first try to return that.
            return self._correlation_matrix
        except AttributeError:
            def corr(i_name: str, j_name: str) -> float:
                """ Calculate the correlation matrix (definition from iminuit) from the covariance matrix. """
                # The + 1e-100 is just to ensure that we don't divide by 0.
                value = (self.covariance_matrix[(i_name, j_name)]
                         / (np.sqrt(self.covariance_matrix[(i_name, i_name)]
                            * self.covariance_matrix[(j_name, j_name)]) + 1e-100)
                         )
                # Need to explicitly cast to float. Otherwise, it will return a np.float64, which will cause problems
                # for YAML...
                return float(value)

            matrix: Dict[Tuple[str, str], float] = {}
            for i_name in self.free_parameters:
                for j_name in self.free_parameters:
                    matrix[(i_name, j_name)] = corr(i_name, j_name)

            self._correlation_matrix = matrix

        return self._correlation_matrix

    def effective_chi_squared(self, cost_func: "cost_function.DataComparisonCostFunction") -> float:
        """ Calculate the effective chi squared value.

        If the fit was performed using a chi squared cost function, it's just equal to
        the ``minimal_val``. If it's log likelihood, one must calculate the effective
        chi squared.

        Note:
            We attempt to cache this value so we don't have to calculate it every time.

        Args:
            cost_function: Cost function used to create the fit function.
            data: Data to be used to calculate the chi squared.
        Returns:
            The effective chi squared value.
        """
        try:
            # We attempt to cache the chi squared, so first try to return that.
            return self._chi_squared
        except AttributeError:
            # Setup
            from pachyderm.fit import cost_function
            from pachyderm import histogram

            # Calculate the chi_squared
            self._chi_squared: float
            if isinstance(cost_func, (cost_function.ChiSquared, cost_function.BinnedChiSquared)):
                self._chi_squared = self.minimum_val
            elif isinstance(cost_func, cost_function.BinnedLogLikelihood):
                data = cost_func.data
                # Help out mypy...
                assert isinstance(data, histogram.Histogram1D)
                # Calculate using the binned chi squared
                self._chi_squared = cost_function._binned_chi_squared(
                    data.x, data.y, data.errors, data.bin_edges, cost_func.f, *self.values_at_minimum.values()
                )
            else:
                raise NotImplementedError("Needs to be implement for unbinned data.")
                #data = cost_func.data
                ## Help out mypy...
                #assert isinstance(data, histogram.Histogram1D)
                #self._chi_squared = cost_function._chi_squared(
                #    data.x, data.y, data.errors, data.bin_edges, cost_func.f, *self.values_at_minimum.values()
                #)

        return self._chi_squared

    @classmethod
    def from_minuit(cls: Type[_T_FitResult], minuit: iminuit.Minuit,
                    cost_func: Callable[..., float], x: np.ndarray) -> _T_FitResult:
        """ Create a fit result form the Minuit fit object.

        Args:
            minuit: Minuit fit object after performing the fit.
            cost_func: Cost function used to perform the fit.
        """
        # Validation
        if not minuit.migrad_ok():
            raise RuntimeError("The fit is invalid - unable to extract result!")

        # Determine the relevant fit parameters.
        fixed_parameters: List[str] = [k for k, v in minuit.fixed.items() if v is True]
        # We use the cost function because we want intentionally want to skip "x"
        parameters: List[str] = iminuit.util.describe(cost_func)
        # Can't just use set(parameters) - set(fixed_parameters) because set() is unordered!
        free_parameters = [p for p in parameters if p not in set(fixed_parameters)]
        # Store the result
        return cls(
            parameters = parameters, free_parameters = free_parameters, fixed_parameters = fixed_parameters,
            values_at_minimum = dict(minuit.values), errors_on_parameters = dict(minuit.errors),
            covariance_matrix = minuit.covariance,
            x = x,
            n_fit_data_points = len(x), minimum_val = minuit.fval,
            errors = [],
        )

def fit_with_minuit(cost_func: Callable[..., float], minuit_args: Dict[str, float],
                    log_likelihood: bool, x: np.ndarray) -> Tuple[FitResult, iminuit.Minuit]:
    """ Perform a fit using the given cost function with Minuit.

    Args:
        cost_func: Cost function to be used with Minuit.
        minuit_args: Arguments for minuit. Need to set the initial value, limits, and error (step)
            of each parameter.
        log_likelihood: True if the cost function is a log likelihood (such that we need to modify
            the errordef of Minuit).
        x: x value(s) where the fit is evaluated, which will be stored in the fit result.
    Returns:
        (fit_result, Minuit object): The fit result extracts values from the Minuit object, but
            the Minuit object is also returned for good measure.
    """
    # Perform the fit
    minuit = iminuit.Minuit(cost_func, **minuit_args, errordef = 0.5 if log_likelihood else 1)
    minuit.migrad()
    # Just in case (doesn't hurt anything, but may help in a few cases).
    minuit.hesse()

    # Check that the fit is actually good
    if not minuit.migrad_ok():
        raise FitFailed("Minimization failed! The fit is invalid!")

    # Create the fit result and caluclate the errors.
    fit_result = FitResult.from_minuit(minuit, cost_func, x)
    # We can calculate the fit errors if the cost function has a single function.
    # If it's a simultaneous fit, it's unclear how best this should be handled. Perhaps it could
    # be unraveled and summed, but it's not obvious that that's the best approach. More likely,
    # one only wants the errors for an individual cost function.
    # We use getattr instead of hasattr to help out mypy
    f = getattr(cost_func, "f", None)
    if f:
        errors = calculate_function_errors(f, fit_result, x)
    else:
        errors = []
    fit_result.errors = errors

    return fit_result, minuit

def calculate_function_errors(func: Callable[..., float], fit_result: FitResult, x: np.ndarray) -> np.array:
    """ Calculate the errors of the given function based on values from the fit.

    Note:
        We don't take the x values for the fit_result as it may be desirable to calculate the errors for
        only a subset of x values. Plus, the component fit result doesn't store the x values, so it would
        complicate the validation. It's much easier to just require the user to pass the x values (and it takes
        little effort to do so).

    Args:
        func: Function to use in calculating the errors.
        fit_result: Fit result for which the errors will be calculated.
        x: x values where the errors will be evaluated.
    Returns:
        The calculated error values.
    """
    # Setup the paramaters needed to execute the function.
    # Determine relevant parameters for the given function
    func_parameters = iminuit.util.describe(func)
    # Determine the arguments for the fit function
    # NOTE: The fit result may have more arguments at minimum and free parameters than the fit function that we've
    #       passed (for example, if we've calculating the background parameters for the inclusive signal fit), so
    #       we need to determine the free parameters here.
    args_at_minimum = {k: v for k, v in fit_result.values_at_minimum.items() if k in func_parameters}
    # Retrieve the parameters to use in calculating the fit errors.
    free_parameters = [p for p in fit_result.free_parameters if p in func_parameters]
    # To calculate the error, we need to match up the parameter names to their index in the arguments list
    args_at_minimum_keys = list(args_at_minimum)
    name_to_index = {name: args_at_minimum_keys.index(name) for name in free_parameters}
    logger.debug(f"args_at_minimum: {args_at_minimum}")
    logger.debug(f"free_parameters: {free_parameters}")
    logger.debug(f"name_to_index: {name_to_index}")

    # To take the gradient, ``numdifftools`` requires a particular function signature. The first argument
    # must contain a list of values that it will vary when taking the gradient. The the rest of the args are
    # passed on to the function via *args and **kwargs, but they won't be varied.
    # To ensure the proper signature, we wrap the function and route the arguments.
    def func_wrap(args_to_vary: Sequence[float], x: np.array) -> float:
        """ Wrap the given function to ensure that the arguments are routed properly for ``numdifftools``.

        To take the gradient, ``numdifftools`` requires a particular function signature. The first argument
        must contain a list of values that it will vary when taking the gradient. The the rest of the args are
        passed on to the function via *args and **kwargs, but they won't be varied (we don't event use those
        generic arguments here). To ensure the proper signature given any function, we wrap the function and
        route the arguments.

        Args:
            args_to_vary: List of arguments to vary when taking the gradient. This should correspond
                to the value of the free parameters.
            x: x value(s) where the function will be evaluated.
        Returns:
            Function evaluated at the given values.
        """
        # Need to expand the arguments
        return func(x, *args_to_vary)
    # Setup to compute the derivative
    partial_derivative_func = nd.Gradient(func_wrap)

    logger.debug("Calculating the gradient")
    # We time it to keep track of how long it takes to evaluate. Sometimes it can be a bit slow.
    start = time.time()
    # Actually evaluate the gradient.
    # It returns an array with dimensions (len(x), len(args_to_vary)) containing the derivatives with
    # respect to each parameter at each point.
    #
    # NOTE: In principle, we've doing some unnecessary work because we also calculate the gradient with
    #       respect to fixed parameters. But due to the argument requirements of ``numdifftools``, it would be
    #       quite difficult to tell it to only take the gradient with respect to a non-continuous selection of
    #       parameters. So we just accept the inefficiency.
    partial_derivative_result = partial_derivative_func(list(args_at_minimum.values()), x)
    end = time.time()
    logger.debug(f"Finished calculating the gradient in {end-start} seconds.")

    # If we are only in 1D, we need to promote to a 2D (shape of 1D, 1) to get the indexing correct below.
    # This only occurs if we only have one parameter that is varied.
    if partial_derivative_result.ndim == 1:
        logger.debug(f"shape before adding axis: {partial_derivative_result.shape}")
        partial_derivative_result = partial_derivative_result[:, np.newaxis]
        logger.debug(f"shape after adding axis : {partial_derivative_result.shape}")

    # Finally, calculate the error by multiplying the matrix of gradient values by the covariance matrix values.
    error_vals = np.zeros(len(x))
    for i_name in free_parameters:
        for j_name in free_parameters:
            # Determine the error value
            #logger.debug(f"Calculating error for i_name: {i_name}, j_name: {j_name}")
            # Add error to overall error value
            # NOTE: This is a vector operation for the partial_derivative_result values.
            error_vals += (
                partial_derivative_result[:, name_to_index[i_name]]
                * partial_derivative_result[:, name_to_index[j_name]]
                * fit_result.covariance_matrix[(i_name, j_name)]
            )
    #logger.debug("error_val: shape: {error_val.shape}, error_val: {error_val}")

    # We want the error itself, so we take the square root.
    return np.sqrt(error_vals)

class FuncCode(generic_class.EqualityMixin):
    """ Minimal class to describe function arguments.

    Same approach as is taken in ``iminuit``. Note that the precise name of the parameters is
    extremely important.

    Args:
        args: List of function arguments.
    Attributes:
        co_varnames: Name of the function arguments.
        co_argcount: Number of function arguments.
    """
    __slots__ = ("co_varnames", "co_argcount")

    def __init__(self, args: List[str]):
        self.co_varnames = args
        self.co_argcount = len(args)

    def __repr__(self) -> str:
        return f"FuncCode({self.co_varnames})"

    @classmethod
    def from_function(cls: Type[T_FuncCode], func: Callable[..., float],
                      leading_parameters_to_remove: int = 1) -> T_FuncCode:
        """ Create a func code from a function.

        Args:
            func: Function for which we want a func code.
            leading_parameters_to_remove: Number of leading parameters to remove in the func code. Default: 1,
                which corresponds to ``x`` as the first argument.
        """
        return cls(iminuit.util.describe(func)[leading_parameters_to_remove:])

def merge_func_codes(functions: Iterable[Callable[..., float]], prefixes: Optional[Sequence[str]] = None,
                     skip_prefixes: Optional[Sequence[str]] = None) -> Tuple[List[str], List[List[int]]]:
    """ Merge the arguments of the given functions into one func code.

    Note:
        This has very similar functionality and is heavily inspired by ``Probfit.merge_func_code...)``.

    Args:
        functions: Functions whose arguments are to be merged.
        prefixes: Prefix for arguments of each function. Default: None. If specified, there must
            be one prefix for each function.
        skip_prefixes: Prefixes to skip when assigning prefixes. As noted in probfit, this can be
            useful to mix prefixed and non-prefixed arguments. Default: None.
    Returns:
        Merged list of arguments, map from merged arguments to arguments for each individual function.
    """
    # Validation
    # Ensure that we don't exhaust the iterator during validation.
    funcs = list(functions)
    # Ensure that we have the proper number of prefixes.
    if prefixes:
        if len(funcs) != len(prefixes):
            raise ValueError("Number of prefixes ({len(prefixes)} doesn't match the number of functions: {len(funcs)}")
    else:
        # Create an empty prefix array so we can zip with it.
        prefixes = ["" for _ in funcs]
    skip_prefix = set(skip_prefixes if skip_prefixes else [])

    # Retrieve all of the args.
    args = []
    for f, pre in zip(funcs, prefixes):
        temp = []
        for arg in iminuit.util.describe(f):
            value = f"{pre}_{arg}" if pre and arg not in skip_prefix else arg
            temp.append(value)
        args.append(temp)

    # Determine the unique arugments.
    # We want to ensure that we maintain the oder, so we use dict.fromkeys to do so.
    merged_args = list(dict.fromkeys(itertools.chain.from_iterable(args)))

    # Determine the map from merged arguments to arguments for each individual function.
    argument_positions = []
    for func_args in args:
        positions = []
        for arg in func_args:
            positions.append(merged_args.index(arg))
        argument_positions.append(positions)

    logger.debug(f"funcs: {funcs}, args: {args}")
    logger.debug(f"merged args: {merged_args}")
    logger.debug(f"argument_positions: {argument_positions}")

    return merged_args, argument_positions

#@jit(nopython = True, )  # type: ignore
def call_list_of_callables(functions: Iterable[Callable[..., float]], argument_positions: T_ArgumentPositions,
                           *args: Union[float, np.ndarray]) -> float:
    """ Call a list of callables with the given args.

    Args:
        functions: Functions to be evaluated.
        argument_positions: Map from merged arguments to arguments for each function.
        args: Arguments for the functions. Must include the x argument!
    Returns:
        Sum of the values of the functions.
    """
    value = 0.
    for func, arg_positions in zip(functions, argument_positions):
        # Determine the arguments for the given function using the arg positions map.
        function_args = []
        for v in arg_positions:
            # We don't skip over the x argument because it's supplied in the args.
            function_args.append(args[v])
        #logger.debug(f"full args: {args}")
        #logger.debug(f"arg_positions: {arg_positions}, function_args: {function_args}")
        #logger.debug(f"describe args: {iminuit.util.describe(func)}")
        value += func(*function_args)
    return value

