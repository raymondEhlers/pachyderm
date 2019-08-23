#!/usr/bin/env python3

""" Tests for integration of functionality in the fit modules.

.. code-author: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
from io import StringIO
from typing import Any, Tuple, Union

import numpy as np
import pytest  # noqa: F401

import pachyderm.fit
from pachyderm import histogram, yaml

logger = logging.getLogger(__name__)

def dump_to_string_and_retrieve(input_object: Any, y: yaml.ruamel.yaml.YAML = None) -> Any:
    """ Dump the given input object via YAML and then retrieve it for comparison.

    Args:
        input_object: Object to be dumped and retrieved.
        y: YAML object to use for the dumping. If not specified, one will be created.
    Returns:
        The dumped and then retrieved object.
    """
    # Create a YAML object if necessary
    if y is None:
        y = yaml.yaml()

    # Dump to a string
    s = StringIO()
    y.dump([input_object], s)
    s.seek(0)
    # And then load from the string. Note the implicit unpacking
    output_object, = y.load(s)

    return output_object

def pedestal(x: Union[np.ndarray, float], pedestal: float) -> Union[np.ndarray]:
    return pedestal

class PedestalFit(pachyderm.fit.Fit):
    """ Minimal pedestal fit object to test reading and writing to YAML. """
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.fit_function = pedestal

    def _post_init_validation(self) -> None:
        """ Validate that the fit object was setup properly. """
        ...

    def _setup(self, h: histogram.Histogram1D) -> Tuple[histogram.Histogram1D, pachyderm.fit.T_FitArguments]:
        """ Setup the histogram and arguments for the fit. """
        return h, {"pedestal": 0}

def test_round_trip_of_fit_object_to_YAML(logging_mixin: Any) -> None:
    """ Test a YAML round trip for a fit object. """
    # Setup
    input_fit_object = PedestalFit(use_log_likelihood = False)
    # YAML object
    y = yaml.yaml(classes_to_register = [PedestalFit])

    # Dump and load (ie round trip)
    output_fit_object = dump_to_string_and_retrieve(input_fit_object, y = y)

    # Check the result
    assert output_fit_object == input_fit_object
