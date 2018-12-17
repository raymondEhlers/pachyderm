#!/usr/bin/env python

""" Tests for the utilities module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
import numpy as np
import pytest
import ruamel.yaml
import tempfile

from pachyderm import histogram
from pachyderm import utils

# Setup logger
logger = logging.getLogger(__name__)

def test_YAML_functionality(logging_mixin, mocker):
    """ Tests for reading and writing YAML.

    These are performed together because they are inverses.
    """
    # Get input data to use for testing:
    # The dictionary of parameters
    input_data = {
        "hello": "world",
        2: 3,
    }
    # The serialized string
    with tempfile.TemporaryFile() as f:
        yaml = ruamel.yaml.YAML(typ = "rt")
        yaml.default_flow_style = False
        yaml.dump(input_data, f)

        # Extract expected data
        f.seek(0)
        input_data_stream = f.read()

    # Test reading
    m_read = mocker.mock_open(read_data = input_data_stream)
    mocker.patch("pachyderm.utils.open", m_read)
    parameters = utils.read_YAML(filename = "tempFilename.yaml")

    # Check the expected read call.
    m_read.assert_called_once_with("tempFilename.yaml", "r")
    assert parameters == input_data

    # Test writing
    m_write = mocker.mock_open()
    mocker.patch("pachyderm.utils.open", m_write)
    m_yaml = mocker.MagicMock()
    mocker.patch("pachyderm.utils.ruamel.yaml.YAML.dump", m_yaml)
    utils.write_YAML(parameters = input_data, filename = "tempFilename.yaml")
    m_write.assert_called_once_with("tempFilename.yaml", "w")
    m_yaml.assert_called_once_with(input_data, m_write())

@pytest.mark.parametrize("inputs, expected", [
    ((3, np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])),
        np.array([6, 9, 12, 13, 12, 9, 6])),
    ((4, np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])),
        np.array([10, 14, 16, 16, 14, 10])),
    ((3, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])),
        np.array([6, 9, 12, 15, 18, 21, 24, 27])),
    ((3, np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])),
        np.array([27, 24, 21, 18, 15, 12, 9, 6]))
], ids = ["n = 3 trianglur values", "n = 4 triangular values", "n = 3 increasing values", "n = 3 decreasing values"])
def test_moving_average(logging_mixin, inputs, expected):
    """ Test the moving average calculation. """
    (n, arr) = inputs
    expected = expected / n
    assert np.array_equal(utils.moving_average(arr = arr, n = n), expected)

@pytest.mark.ROOT
class TestWithRootHists():
    def test_get_array_for_fit(self, logging_mixin, mocker, test_root_hists):
        """ Test getting an array from a hist in a dict of observables. """
        observables = {}
        for i in range(5):
            observables[i] = mocker.MagicMock(spec = ["jet_pt_bin", "track_pt_bin", "hist"],
                                              jet_pt_bin = i, track_pt_bin = i + 2,
                                              hist = None)
        # We mock the Observable containing a HistogramContainer, which then contains a normal histogram.
        # We only want one Observable to work. All others shouldn't have a hist to ensure that the test
        # will fail if something has gone awry.
        observables[3].hist = mocker.MagicMock(spec = ["hist"], hist = test_root_hists.hist1D)
        hist_array = utils.get_array_for_fit(observables, jet_pt_bin = 3, track_pt_bin = 5)

        # Expected values
        expected_hist_array = histogram.Histogram1D.from_existing_hist(hist = test_root_hists.hist1D)

        # This is basically a copy of test_histogram.check_hist, but since it is brief and convenient
        # to have it here, we just leave it.
        assert len(hist_array.x) > 0
        assert np.array_equal(hist_array.x, expected_hist_array.x)
        assert len(hist_array.y) > 0
        assert np.array_equal(hist_array.y, expected_hist_array.y)
        assert len(hist_array.errors) > 0
        assert np.array_equal(hist_array.errors, expected_hist_array.errors)

