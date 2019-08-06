#!/usr/bin/env python3

""" Tests for functions related to fitting.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import numpy as np
import pytest
from typing import Any, List

from pachyderm.fit import function

logger = logging.getLogger(__name__)

def test_gaussian(logging_mixin: Any) -> None:
    """ Test the gaussian function. """
    # Setup
    x = np.arange(-10, 10, 0.1)
    mean = 0.5
    sigma = 4.2

    # Calculate independently outside of the tests for mean = 0.5, sigma = 4.2.
    expected = np.array([
        0.0041734 , 0.00442811, 0.00469569, 0.00497663, 0.00527138,  # noqa: E203
        0.00558043, 0.00590424, 0.00624331, 0.0065981 , 0.00696911,  # noqa: E203
        0.0073568 , 0.00776167, 0.00818417, 0.00862478, 0.00908396,  # noqa: E203
        0.00956216, 0.01005984, 0.01057741, 0.01111532, 0.01167396,  # noqa: E203
        0.01225372, 0.01285499, 0.01347812, 0.01412345, 0.01479129,  # noqa: E203
        0.01548193, 0.01619563, 0.01693263, 0.01769313, 0.01847732,  # noqa: E203
        0.01928533, 0.02011726, 0.02097318, 0.02185314, 0.0227571 ,  # noqa: E203
        0.02368503, 0.02463683, 0.02561235, 0.02661141, 0.02763376,  # noqa: E203
        0.02867914, 0.02974718, 0.03083752, 0.03194971, 0.03308325,  # noqa: E203
        0.03423759, 0.03541212, 0.03660619, 0.03781908, 0.03905002,  # noqa: E203
        0.04029816, 0.04156264, 0.04284249, 0.04413673, 0.0454443 ,  # noqa: E203
        0.04676408, 0.04809493, 0.04943561, 0.05078487, 0.05214139,  # noqa: E203
        0.0535038 , 0.05487069, 0.05624062, 0.05761208, 0.05898353,  # noqa: E203
        0.06035341, 0.06172011, 0.06308198, 0.06443737, 0.06578457,  # noqa: E203
        0.06712188, 0.06844755, 0.06975986, 0.07105703, 0.0723373 ,  # noqa: E203
        0.07359891, 0.07484008, 0.07605905, 0.07725407, 0.07842339,  # noqa: E203
        0.0795653 , 0.08067808, 0.08176006, 0.0828096 , 0.08382508,  # noqa: E203
        0.08480492, 0.08574759, 0.0866516 , 0.08751552, 0.08833796,  # noqa: E203
        0.08911759, 0.08985315, 0.09054344, 0.09118732, 0.09178374,  # noqa: E203
        0.0923317 , 0.09283029, 0.09327869, 0.09367612, 0.09402194,  # noqa: E203
        0.09431555, 0.09455646, 0.09474425, 0.09487862, 0.09495934,  # noqa: E203
        0.09498626, 0.09495934, 0.09487862, 0.09474425, 0.09455646,  # noqa: E203
        0.09431555, 0.09402194, 0.09367612, 0.09327869, 0.09283029,  # noqa: E203
        0.0923317 , 0.09178374, 0.09118732, 0.09054344, 0.08985315,  # noqa: E203
        0.08911759, 0.08833796, 0.08751552, 0.0866516 , 0.08574759,  # noqa: E203
        0.08480492, 0.08382508, 0.0828096 , 0.08176006, 0.08067808,  # noqa: E203
        0.0795653 , 0.07842339, 0.07725407, 0.07605905, 0.07484008,  # noqa: E203
        0.07359891, 0.0723373 , 0.07105703, 0.06975986, 0.06844755,  # noqa: E203
        0.06712188, 0.06578457, 0.06443737, 0.06308198, 0.06172011,  # noqa: E203
        0.06035341, 0.05898353, 0.05761208, 0.05624062, 0.05487069,  # noqa: E203
        0.0535038 , 0.05214139, 0.05078487, 0.04943561, 0.04809493,  # noqa: E203
        0.04676408, 0.0454443 , 0.04413673, 0.04284249, 0.04156264,  # noqa: E203
        0.04029816, 0.03905002, 0.03781908, 0.03660619, 0.03541212,  # noqa: E203
        0.03423759, 0.03308325, 0.03194971, 0.03083752, 0.02974718,  # noqa: E203
        0.02867914, 0.02763376, 0.02661141, 0.02561235, 0.02463683,  # noqa: E203
        0.02368503, 0.0227571 , 0.02185314, 0.02097318, 0.02011726,  # noqa: E203
        0.01928533, 0.01847732, 0.01769313, 0.01693263, 0.01619563,  # noqa: E203
        0.01548193, 0.01479129, 0.01412345, 0.01347812, 0.01285499,  # noqa: E203
        0.01225372, 0.01167396, 0.01111532, 0.01057741, 0.01005984,  # noqa: E203
        0.00956216, 0.00908396, 0.00862478, 0.00818417, 0.00776167   # noqa: E203
    ])

    # Calculate using our defined gaussian
    results = function.gaussian(x = x, mean = mean, sigma = sigma)

    # Check the result.
    # We need a bit of extra tolerance because they are calculated at different times to float precision.
    np.testing.assert_allclose(results, expected, atol = 1e-6)

@pytest.mark.parametrize("skip_prefixes, expected_co_varnames", [  # type:ignore
    (["x"], ["x", "f_a", "f_b", "g_c", "g_d"]),
    (["x", "a"], ["x", "a", "f_b", "g_c", "g_d"]),
], ids = ["No skipped prefixes", "Skipped x prefix"])
def test_AddPDF(logging_mixin: Any, simple_test_functions: Any, skip_prefixes: List[str], expected_co_varnames: List[str]) -> None:
    """ Test for adding multiple functions with ``AddPDF``.

    Note:
        This is effectively an integration test for most of the ``AddPDF`` features.
    """
    # Setup
    func_1, func_2 = simple_test_functions

    # Create added functions, prepending prefixes, and skipping a subset of the prefixes.
    # This is effectively an integration test.
    added = function.AddPDF(func_1, func_2, prefixes = ["f", "g"], skip_prefixes = skip_prefixes)

    # Check properties
    assert added.func_code.co_varnames == expected_co_varnames
    np.testing.assert_allclose(added(np.array([1, 2]), *[1., 3, 3, 4]), [5 + 8, 6 + 9])

