#!/usr/bin/env python

""" Tests for the base of the fit module.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import iminuit
import numpy as np
import pytest
from typing import Any, List

import pachyderm.fit.base as fit_base

def test_func_code(logging_mixin: Any, simple_test_functions: Any) -> None:
    """ Test creating function codes with FuncCode. """
    # Setup
    func_1, func_2 = simple_test_functions

    # Define the func code and check the properties
    func_code = fit_base.FuncCode(iminuit.util.describe(func_1))
    assert func_code.co_varnames == ["x", "a", "b"]
    assert func_code.co_argcount == 3

    # Alternatively create via static method
    func_code_2 = fit_base.FuncCode.from_function(func_1)
    assert func_code == func_code_2

@pytest.mark.parametrize("function_list_names, expected_result, expected_argument_positions", [  # type: ignore
    ([1, 2], ["x", "a", "b", "c", "d"], [[0, 1, 2], [0, 3, 4]]),
    ([2, 1], ["x", "c", "d", "a", "b"], [[0, 1, 2], [0, 3, 4]]),
], ids = ["1, 2", "2, 1"])
def test_merge_func_code(logging_mixin: Any, simple_test_functions: Any,
                         function_list_names: List[int], expected_result: List[str],
                         expected_argument_positions: List[List[int]]) -> None:
    """ Test merging function codes for a list of functions. """
    funcs = simple_test_functions
    function_list = [funcs[f_label - 1] for f_label in function_list_names]
    result, argument_positions = fit_base.merge_func_codes(function_list)
    assert result == expected_result
    assert argument_positions == expected_argument_positions

@pytest.mark.parametrize("function_list_names", [  # type: ignore
    (1, 2),
    (2, 1),
], ids = ["1, 2", "2, 1"])
def test_merge_func_code_against_probfit(logging_mixin: Any, simple_test_functions: Any, function_list_names: List[int]) -> None:
    """ Test merging function codes against probfit. """
    # Setup
    funcs = simple_test_functions
    function_list = tuple([funcs[f_label - 1] for f_label in function_list_names])
    probfit = pytest.importorskip("probfit")

    # Run the test and check
    result, argument_positions = fit_base.merge_func_codes(function_list)
    probfit_result, probfit_argument_positions = probfit.merge_func_code(*function_list)
    assert result == list(probfit_result.co_varnames)
    np.testing.assert_allclose(argument_positions, probfit_argument_positions)

