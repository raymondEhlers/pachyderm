#!/usr/bin/env python3

""" Integration of functionality in the fit modules.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import abc
import iminuit
import logging
import numpy as np
import ruamel.yaml
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Type, TypeVar, Union, cast

from pachyderm import generic_class
from pachyderm.fit import base, cost_function


if TYPE_CHECKING:
    from pachyderm import histogram

logger = logging.getLogger(__name__)

# Typing
T_FitArguments = Dict[str, Union[bool, float, Tuple[float, float]]]

_T_Fit = TypeVar("_T_Fit", bound="Fit")


class Fit(abc.ABC, generic_class.EqualityMixin):
    """ Class to direct fitting a histogram to a fit function.

    This allows us to easily store the fit function right alongside the minimization.

    Args
        use_log_likelihood: True if the log likelihood cost function should be used.
        fit_options: Options to use for the fit, such as the range in which the data will be fit.
            It's up to the derived class to determine how to use this information.
        user_arguments: User arguments for the fit. Default: None.

    Attributes:
        use_log_likelihood: True if the log likelihood cost function should be used.
        fit_options: Options to use for the fit, such as the range in which the data will be fit.
            It's up to the derived class to determine how to use this information.
        user_arguments: User arguments for the fit. Default: None.
        fit_function: Function to be fit.
        fit_result: Result of the fit. Only valid after the fit has been performed.
    """

    def __init__(
        self,
        use_log_likelihood: bool,
        fit_options: Optional[Dict[str, Any]] = None,
        user_arguments: Optional[T_FitArguments] = None,
    ):
        # Validation
        if user_arguments is None:
            user_arguments = {}
        if fit_options is None:
            fit_options = {}

        # Store main parameters
        self.use_log_likelihood = use_log_likelihood
        self.fit_options = fit_options
        self.user_arguments: T_FitArguments = user_arguments
        self.fit_function: Callable[..., float]
        self.fit_result: base.FitResult

        # Create the cost function based on the fit parameters.
        self._cost_func: Type[cost_function.DataComparisonCostFunction]
        if use_log_likelihood:
            self._cost_func = cost_function.BinnedLogLikelihood
        else:
            self._cost_func = cost_function.BinnedChiSquared

        # Check fit initialization.
        self._post_init_validation()

    @abc.abstractmethod
    def _post_init_validation(self) -> None:
        """ Validate that the fit object was setup properly.

        This can be any method that the user devises to ensure that
        all of the information needed for the fit is available.

        Args:
            None.
        Returns:
            None.
        """
        ...

    @abc.abstractmethod
    def _setup(self, h: "histogram.Histogram1D") -> Tuple["histogram.Histogram1D", T_FitArguments]:
        """ Setup the histogram and arguments for the fit.

        Args:
            h: Background subtracted histogram to be fit.
        Returns:
            Histogram to use for the fit, default arguments for the fit. Note that the histogram may be range
                restricted or otherwise modified here.
        """
        ...

    def _create_cost_function(self, h: "histogram.Histogram1D") -> cost_function.DataComparisonCostFunction:
        """ Create the cost function from the data and stored parameters.

        Args:
            h: Data to be used for the fit.
        Returns:
            The created cost function.
        """
        return self._cost_func(f=self.fit_function, data=h)

    def __call__(self, *args: float, **kwargs: float) -> float:
        """ Call the fit function.

        This is provided for convenience. This way, we can easily evaluate the function while
        still storing the information necessary to perform the entire fit.

        Args:
            args: Arguments to pass to the fit function.
            kwargs: Arguments to pass to the fit function.
        Returns:
            The fit function called with these arguments.
        """
        return self.fit_function(*args, **kwargs)

    def calculate_errors(self, x: Optional[np.ndarray] = None) -> np.ndarray:
        """ Calculate the errors on the fit function for the given x values.

        Args:
            x: x values where the fit function error should be evaluated. If not specified,
                the x values over which the fit was performed will be used.
        Returns:
            The fit function error calculated at each x value.
        """
        if x is None:
            x = self.fit_result.x
        return base.calculate_function_errors(func=self.fit_function, fit_result=self.fit_result, x=x,)

    def fit(self, h: "histogram.Histogram1D", user_arguments: Optional[T_FitArguments] = None) -> base.FitResult:
        """ Fit the given histogram to the stored fit function using iminuit.

        The fit errors will be automatically calculated if possible. It is possible if the fit function
        accessible through the cost function is not an added PDF.

        Args:
            h: Histogram to be fit.
            user_arguments: Additional user arguments (beyond those already specified when the object
                was created). They will override the already specified options. Default: None.
        Returns:
            Result of the fit. The user is responsible for storing it in the fit.
        """
        # Validation
        if user_arguments is None:
            user_arguments = {}

        # Setup the fit and the cost function
        hist_for_fit, user_fit_arguments = self._setup(h=h)
        # Update the default user arguments provided in setup by...
        # ... The stored user arguments when the object was created.
        user_fit_arguments.update(self.user_arguments)
        # ... The user arguments passed to the fit function.
        user_fit_arguments.update(user_arguments)
        # Then create the cost function according to the parameters and (potentially restricted) data.
        cost_func = self._create_cost_function(h=hist_for_fit)

        # Perform the fit by minimizing the chi squared
        fit_result, _ = fit_with_minuit(
            cost_func=cost_func,
            minuit_args=user_fit_arguments,
            x=hist_for_fit.x,
            use_minos=self.fit_options.get("minos", False),
        )

        # Calculate the fit result only if requested.
        if self.fit_options.get("calculate_errors", False):
            fit_result.errors = self.calculate_errors(hist_for_fit.x)

        return fit_result

    @classmethod
    def to_yaml(
        cls: Type[_T_Fit], representer: ruamel.yaml.representer.BaseRepresenter, obj: _T_Fit
    ) -> ruamel.yaml.nodes.SequenceNode:
        """ Encode YAML representation.

        Since YAML won't handle function very nicely, we convert them to strings and then check them
        on conversion from YAML as a cross check that recreating the object hasn't gone wrong. This
        also requires all functions to be recreatable on object initialization.

        Args:
            representer: Representation from YAML.
            data: Fit function to be converted to YAML.
        Returns:
            YAML representation of the fit object.
        """
        # ``RoundTripRepresenter`` doesn't represent objects directly, so we grab a dict of the members to
        # store them in YAML.
        # NOTE: We must make a copy of the vars. Otherwise, the original fit object will be modified.
        members = dict(vars(obj))
        # We can't store unbound functions, so we instead set it to the function name (we won't really
        # use this name to try to directly recreate the function when the object is being loaded from YAML,
        # but it's useful to store what was used, and we can at least warn if it changed).
        members["fit_function"] = obj.fit_function.__name__
        members["_cost_func"] = obj._cost_func.__name__
        representation = representer.represent_mapping(f"!{cls.__name__}", members)

        # Finally, return the represented object.
        return cast(ruamel.yaml.nodes.SequenceNode, representation)

    @classmethod
    def from_yaml(
        cls: Type[_T_Fit], constructor: ruamel.yaml.constructor.BaseConstructor, data: ruamel.yaml.nodes.MappingNode
    ) -> _T_Fit:
        """ Decode YAML representation.

        Args:
            constructor: Constructor from the YAML object.
            node: YAML Mapping node representing the fit object.
        Returns:
            The fit object constructed from the YAML specified values.
        """
        # First, we construct the class member objects.
        members = {
            constructor.construct_object(key_node): constructor.construct_object(value_node)
            for key_node, value_node in data.value
        }
        # Then we deal with members which require special handling:
        # The fit result isn't set through the constructor, so we grab it and then
        # set it after creating the object.
        fit_result = members.pop("fit_result", None)
        # The fit function will be set in the fit constructor, so we don't need to use this
        # value to setup the object. However, since this contains the name of the function,
        # we can use it to check if the name of the function that is set in the constructor
        # is the same as the one that we stored. (If they are different, this isn't necessarily
        # a problem, as we sometimes rename functions, but regardless it's good to be notified
        # if that's the case).
        function_names = {}
        for attr_name in ["fit_function", "_cost_func"]:
            function_names[attr_name] = members.pop(attr_name)

        # Finally, create the object and set the properties as needed.
        obj = cls(**members)
        # The fit result may not have been defined yet, so only set it if it has.
        if fit_result is not None:
            obj.fit_result = fit_result
        # Sanity checks on fit and cost function names (see above).
        for attr_name, function_name in function_names.items():
            if function_name != getattr(obj, attr_name).__name__:
                logger.warning(
                    "The stored '{attr_name}' function name from YAML doesn't match the name of the function"
                    " created in the fit object."
                    f" Stored name: {function_name}, object created fit function: {getattr(obj, attr_name).__name__}."
                    " This may indicate a problem (but is fine if the same function was just renamed)."
                )

        # Now that the object is fully constructed, we can return it.
        return obj


def _validate_minuit_args(
    cost_func: Union[cost_function.CostFunctionBase, cost_function.SimultaneousFit], minuit_args: T_FitArguments
) -> None:
    """ Validate the arguments provided for Minuit.

    Checks that there are sufficient and valid arguments for each parameter in the fit function.

    Args:
        cost_func: Cost function to be used with Minuit.
        minuit_args: Arguments for minuit. Need to set the initial value, limits, and error (step)
            of each parameter.
    Returns:
        None. An exception is raised if there's a problem with any of the arguments.
    """
    # Need the parameters in the function to check the arguments. Recall that "x" is skipped because it's a cost
    # function (which is useful, because we want to skip x since it's not a parameter).
    parameters = iminuit.describe(cost_func)
    # Loop over the available parameters because each one needs to be specified in the Minuit args.
    for p in parameters:
        if p in minuit_args:
            if f"limit_{p}" not in minuit_args:
                raise ValueError(f"Limits on parameter '{p}' must be specified.")
            if f"error_{p}" not in minuit_args:
                raise ValueError(f"Initial error on parameter '{p}' must be specified.")
            # If p and limit_p and error_p were specified, then the parameter is fully specified.
        elif p.replace("fix", "") in minuit_args:
            # The parameter is fixed and therefore needs no further specification.
            pass
        else:
            # The parameter wasn't specified.
            raise ValueError(f"Parameter '{p}' must be specified in the fit arguments.")


def fit_with_minuit(
    cost_func: Union[cost_function.CostFunctionBase, cost_function.SimultaneousFit],
    minuit_args: T_FitArguments,
    x: np.ndarray,
    use_minos: Optional[bool] = False,
) -> Tuple[base.FitResult, iminuit.Minuit]:
    """ Perform a fit using the given cost function with Minuit.

    Args:
        cost_func: Cost function to be used with Minuit.
        minuit_args: Arguments for minuit. Need to set the initial value, limits, and error (step)
            of each parameter.
        x: x value(s) where the fit is evaluated, which will be stored in the fit result.
        use_minos: Calculate MINOS errors. They have to be accessed through the Minuit object. Default: False.
    Returns:
        (fit_result, Minuit object): The fit result extracts values from the Minuit object, but
            the Minuit object is also returned for good measure.
    """
    # Validation
    # Will raise an exception if there are invalid arguments.
    _validate_minuit_args(cost_func=cost_func, minuit_args=minuit_args)
    # Set the error definition.
    # We check if it's set to the allow the user to override if they are so inclined.
    # (Overriding it should be pretty rare).
    if "errordef" not in minuit_args:
        # Log likelihood cost functions needs an errordef of 0.5 to scale the errors properly, while 1 should
        # be used for chi squared cost functions.
        error_def = 1.0
        if isinstance(cost_func, (cost_function.LogLikelihood, cost_function.BinnedLogLikelihood)):
            error_def = 0.5
        # Store the value.
        minuit_args["errordef"] = error_def

    # Perform the fit
    minuit = iminuit.Minuit(cost_func, **minuit_args)
    # Improve minimization reliability.
    minuit.strategy = 2
    minuit.migrad()
    # Just in case (doesn't hurt anything, but may help in a few cases).
    minuit.hesse()
    if use_minos:
        minuit.minos()

    # Check that the fit is actually good
    if not minuit.valid:
        raise base.FitFailed("Minimization failed! The fit is invalid!")
    # Check covariance matrix accuracy. We need to check it explicitly because It appears that it is not
    # included in the migrad_ok status check.
    if not minuit.accurate:
        raise base.FitFailed("Covariance matrix is inaccurate! The fit is invalid!")

    # Create the fit result and calculate the errors.
    fit_result = base.FitResult.from_minuit(minuit, cost_func, x)
    # We can calculate the fit errors if the cost function has a single function.
    # If it's a simultaneous fit, it's unclear how best this should be handled. Perhaps it could
    # be unraveled and summed, but it's not obvious that that's the best approach. More likely,
    # one only wants the errors for an individual cost function, so we leave that to the user.
    # We use getattr instead of hasattr to help out mypy
    if isinstance(cost_func, cost_function.CostFunctionBase):
        errors = base.calculate_function_errors(cost_func.f, fit_result, x)
    else:
        errors = []
    fit_result.errors = errors

    return fit_result, minuit
