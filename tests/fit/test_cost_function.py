#!/usr/bin/env python3

""" Tests for the cost functions module.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import numpy as np
import pytest
import scipy.integrate
from typing import Any, Tuple

from pachyderm.typing_helpers import Hist

import pachyderm.fit.base as fit_base
from pachyderm.fit import cost_function

from pachyderm import histogram

logger = logging.getLogger(__name__)

def func_1(x: float, a: float, b: float) -> float:
    """ Test function. """
    return x + a + b

def func_2(x: float, c: float, d: float) -> float:
    """ Test function 2. """
    return x + c + d

def test_integration(logging_mixin: Any) -> None:
    """ Test our implementation of the Simpson 3/8 rule, along with some other integration methods. """
    # Setup
    f = func_1
    h = histogram.Histogram1D(
        bin_edges = np.array([0, 1, 2]), y = np.array([0, 1]), errors_squared = np.array([1, 2])
    )
    args = [0, 0]

    integral = cost_function._simpson_38(f, h.bin_edges, *args)
    # Evaluate at the bin center
    expected = np.array([f(i, *args) for i in h.x])
    np.testing.assert_allclose(integral, expected)

    # Compare against our quad implementation
    integral_quad = cost_function._quad(f, h.bin_edges, *args)
    np.testing.assert_allclose(integral, integral_quad)

    # Also compare against probfit and scipy for good measure
    probfit = pytest.importorskip("probfit")
    expected_probfit = []
    expected_scipy = []
    expected_quad = []
    for i in h.bin_edges[1:]:
        # Assumes uniform bin width
        expected_probfit.append(probfit.integrate1d(f, (i - 1, i), 1, tuple(args)))
        scipy_x = np.linspace(i - 1, i, 5)
        expected_scipy.append(scipy.integrate.simps(y = f(scipy_x, *args), x = scipy_x))
        res, _ = scipy.integrate.quad(f, i - 1, i, args = tuple(args))
        expected_quad.append(res)

    np.testing.assert_allclose(integral, expected_probfit)
    np.testing.assert_allclose(integral, expected_scipy)
    np.testing.assert_allclose(integral, expected_quad)

def test_chi_squared(logging_mixin: Any) -> None:
    """ Test the chi squared calculation. """
    # Setup
    h = histogram.Histogram1D(
        bin_edges = np.array(np.arange(-0.5, 5.5)), y = np.array(np.ones(5)), errors_squared = np.ones(5),
    )
    chi_squared = cost_function.ChiSquared(f = func_1, data = h)

    # Check that it's set up properly
    assert chi_squared.func_code.co_varnames == ["a", "b"]

    # Calculate the chi_squared for the given parameters.
    result = chi_squared(np.array(range(-1, -6, -1)), np.zeros(5))
    # Each term is (1 - -1)^2 / 1^2 = 4
    assert result == 4 * 5

#####################################
# Testing cost functions against ROOT
#####################################
def parabola(x: float, scale: float) -> float:
    """ Parabolic function.

    Note:
        It returns a float, but numba can't handle cast. So we return ``Any`` and then cast the result.

    Args:
        x: Where the parabola will be evaluated.
        scale: Scale factor for the parabola.
    Returns:
        Value of parabola for given parameters.
    """
    return scale * np.square(x)  # type: ignore

@pytest.fixture  # type: ignore
def setup_parabola(logging_mixin: Any) -> Tuple[histogram.Histogram1D, Hist]:
    """ Setup a parabola for tests of fitting procedures. """
    ROOT = pytest.importorskip("ROOT")

    # Specify a seed so the test is reproducible.
    np.random.seed(12345)

    h_ROOT = ROOT.TH1F("test", "test", 42, -10.5, 10.5)
    h_ROOT.Sumw2()
    for x in np.linspace(-10.25, 10.25, 42):
        # Ensure that the bin at 0 is not precisely 0
        if x == 0.0:
            h_ROOT.Fill(x, 2)

        # Adds a gaussian noise term with a width of 3. It's offset from 0 to ensure that we don't get 0.
        #for _ in np.arange(int(parabola(np.abs(x), 1) + np.random.normal(5, 4))):
        for _ in np.arange(int(np.ceil(parabola(np.abs(x), 1) + np.random.normal(3, 3)))):
            #logger.debug(f"Filling for x: {x}")
            h_ROOT.Fill(x)

    # Scale by bin width
    h_ROOT.Scale(1.0 / h_ROOT.GetBinWidth(1))

    # Convert
    h = histogram.Histogram1D.from_existing_hist(h_ROOT)
    logger.debug(f"h: {h}")

    #c = ROOT.TCanvas("c", "c")
    #h_ROOT.Draw()
    #c.SaveAs("test_parabola.pdf")

    return h, h_ROOT

@pytest.mark.parametrize("cost_func, fit_option", [  # type: ignore
    (cost_function.BinnedChiSquared, "SV"),
    (cost_function.BinnedLogLikelihood, "SLV"),
    ("probfit", "SV"),
], ids = ["Binned chi squared", "Binned log likelihood", "Probfit Chi2"])
def test_binned_cost_functions_against_ROOT(logging_mixin: Any, cost_func: Any, fit_option: Any,
                                            setup_parabola: Any) -> None:
    """ Test the binned cost function implementations against ROOT. """
    # Setup
    h, h_ROOT = setup_parabola
    ROOT = pytest.importorskip("ROOT")
    minuit_args = {
        "scale": 1, "error_scale": 0.1,
    }
    log_likelihood = "L" in fit_option
    if cost_func == "probfit":
        probfit = pytest.importorskip("probfit")
        cost_func = probfit.Chi2Regression

    # Fit with ROOT
    fit_ROOT = ROOT.TF1("parabola", "[0] * TMath::Power(x, 2)", -10.5, 10.5)
    # Expect it to be around 1.
    fit_ROOT.SetParameter(0, minuit_args["scale"])
    fit_result_ROOT = h_ROOT.Fit(fit_ROOT, fit_option + "0")
    logger.debug(f"ROOT: chi_2: {fit_result_ROOT.Chi2()}, ndf: {fit_result_ROOT.Ndf()}")

    # Fit with the defined cost function
    args = {"f": parabola}
    if issubclass(cost_func, cost_function.CostFunctionBase):
        args.update({"data": h})
    else:
        args.update({"x": h.x, "y": h.y, "error": h.errors})
    cost = cost_func(**args)
    fit_result, _ = fit_base.fit_with_minuit(cost, minuit_args, log_likelihood, h.x)

    # Check the minimized value.
    # It doesn't appear that it will agree for log likelihood
    if not log_likelihood:
        assert np.isclose(fit_result.minimum_val, fit_result_ROOT.MinFcnValue(), rtol = 0.03)

    if cost_func is cost_function.BinnedLogLikelihood:
        # Calculate the chi squared equivalent and set that to be the minimum value for comparison.
        binned_chi_squared = cost_function._binned_chi_squared(
            h.x, h.y, h.errors, h.bin_edges, parabola, *list(fit_result.values_at_minimum.values())
        )
        unbinned_chi_squared = cost_function._chi_squared(
            h.x, h.y, h.errors, h.bin_edges, parabola, *list(fit_result.values_at_minimum.values())
        )
        logger.debug(
            f"minimual_val before changing: {fit_result.minimum_val}, ROOT func min: {fit_result_ROOT.MinFcnValue()}"
        )
        logger.debug(f"binned chi_squared: {binned_chi_squared}, unbinned chi_squared: {unbinned_chi_squared}")
        fit_result.minimum_val = binned_chi_squared

    # Calculate errors.
    fit_result.errors = fit_base.calculate_function_errors(
        func = parabola,
        fit_result = fit_result,
        x = fit_result.x
    )

    # Check the result
    logger.debug(f"Fit chi_2: {fit_result.minimum_val}, ndf: {fit_result.nDOF}")
    # It won't agree exactly because ROOT appears to use the unbinned chi squared to calculate this value.
    # This can be seen because probfit agress with ROOT.
    assert np.isclose(fit_result.minimum_val, fit_result_ROOT.Chi2(), rtol = 0.035)
    assert np.isclose(fit_result.nDOF, fit_result_ROOT.Ndf())
    # Check the parameters
    # Value
    assert np.isclose(
        fit_result.values_at_minimum["scale"], fit_result_ROOT.Parameter(0), rtol = 0.05,
    )
    # Error
    # TODO: For some reason, there error is substantially larger in the log likelihood cost function comapred to ROOT
    # This requires more investigation, but shouldn't totally derail progress at the moment.
    if not log_likelihood:
        assert np.isclose(fit_result.errors_on_parameters["scale"], fit_result_ROOT.ParError(0), rtol = 0.005)
    # Check the effective chi squared. This won't work in the probfit case because we don't recognize
    # the type properly (and it's not worth the effort).
    if issubclass(cost_func, cost_function.CostFunctionBase):
        assert fit_result.effective_chi_squared(cost) == (
            cost_function._binned_chi_squared(
                cost.data.x, cost.data.y, cost.data.errors, cost.data.bin_edges,
                cost.f, *fit_result.values_at_minimum.values()
            ) if log_likelihood else fit_result.minimum_val
        )

##################
# Simultaneous Fit
##################

@pytest.fixture  # type: ignore
def setup_simultaneous_fit_data(logging_mixin: Any, setup_parabola: Any) -> Tuple[histogram.Histogram1D, histogram.Histogram1D, Hist, Hist]:
    """ Setup the data for tests of a simultaneous fit. """
    h, h_ROOT = setup_parabola

    # Create a new parabola that's shifted up by two.
    h_shifted_ROOT = h_ROOT.Clone("shifted_parabola")
    h_shifted_ROOT.Add(h_ROOT)

    h_shifted = histogram.Histogram1D.from_existing_hist(h_shifted_ROOT)

    return h, h_shifted, h_ROOT, h_shifted_ROOT

def test_simultaneous_fit_basic(logging_mixin: Any, setup_simultaneous_fit_data: Any) -> None:
    """ Test basic Simultaneous fit functionality. """
    # Setup
    h, h_shifted, _, _ = setup_simultaneous_fit_data

    # Check with cost functions
    cost_func1 = cost_function.ChiSquared(func_1, data = h)
    cost_func2 = cost_function.ChiSquared(func_2, data = h_shifted)
    s2 = cost_function.SimultaneousFit(cost_func1, cost_func2)
    assert s2.func_code == fit_base.FuncCode(["a", "b", "c", "d"])

    # Check with manually added functions
    s3 = cost_func1 + cost_func2
    assert s3.func_code == fit_base.FuncCode(["a", "b", "c", "d"])
    assert s3 == s2

def test_nested_simultaneous_fit_objects(logging_mixin: Any, setup_simultaneous_fit_data: Any) -> None:
    """ Test for unraveling nested simultaneous fit objects. """
    # Setup
    h, h_shifted, _, _ = setup_simultaneous_fit_data

    # Check with cost functions
    cost_func1 = cost_function.ChiSquared(func_1, data = h)
    cost_func2 = cost_function.ChiSquared(func_2, data = h_shifted)
    cost_func3 = cost_function.ChiSquared(lambda x, e, f: x + e + f, data = h)
    s = cost_func1 + cost_func2
    s2 = s + cost_func3
    assert s2.func_code == fit_base.FuncCode(["a", "b", "c", "d", "e", "f"])

    # Test out using sum
    s3 = sum([cost_func1, cost_func2, cost_func3])
    # Help out mypy...
    assert isinstance(s3, cost_function.SimultaneousFit)
    assert s3.func_code == fit_base.FuncCode(["a", "b", "c", "d", "e", "f"])

def test_simultaneous_fit(logging_mixin: Any, setup_simultaneous_fit_data: Any) -> None:
    """ Test Simultaneous Fit functionality vs probfit with an integration test. """
    # Setup
    h, h_shifted, _, _ = setup_simultaneous_fit_data
    cost_func1 = cost_function.ChiSquared(parabola, data = h)
    cost_func2 = cost_function.ChiSquared(parabola, data = h_shifted)
    minuit_args = {
        "scale": 1.5, "error_scale": 0.15,
    }

    # Setup the probfit version
    probfit = pytest.importorskip("probfit")
    s_probfit = probfit.SimultaneousFit(*[cost_func1, cost_func2])

    # Setup the comparison version
    s = cost_func1 + cost_func2

    # First, basic checks
    logger.debug(f"func_code: {s.func_code}, co_varnames: {s.func_code.co_varnames}")
    assert s.func_code == fit_base.FuncCode(["scale"])
    assert s.func_code.co_varnames == list(s_probfit.func_code.co_varnames)

    # Now perform the fits
    fit_result, _ = fit_base.fit_with_minuit(
        cost_func = s, minuit_args = minuit_args, log_likelihood = False, x = h.x
    )
    fit_result_probfit, _ = fit_base.fit_with_minuit(
        cost_func = s_probfit, minuit_args = minuit_args, log_likelihood = False, x = h.x
    )
    # And check that the fit results agree
    logger.debug(f"scale: {fit_result.values_at_minimum['scale']} +/- {fit_result.errors_on_parameters['scale']}")
    assert fit_result == fit_result_probfit

