""" Test the integration between various classes.

"""
from __future__ import annotations

import logging
from io import StringIO
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pachyderm import binned_data, histogram, yaml

logger = logging.getLogger(__name__)


def dump_to_string_and_retrieve(input_object: Any, y: yaml.ruamel.yaml.YAML = None) -> Any:  # type: ignore[name-defined]
    """Dump the given input object via YAML and then retrieve it for comparison.

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
    (output_object,) = y.load(s)

    return output_object


def test_Histogram1D_with_yaml() -> None:
    """Test writing and then reading a Histogram1D via YAML.

    This ensures that writing a histogram1D can be done successfully.
    """
    # Setup
    # YAML object
    y = yaml.yaml(classes_to_register=[histogram.Histogram1D])
    # Test hist
    input_hist = histogram.Histogram1D(
        bin_edges=np.linspace(0, 10, 11), y=np.linspace(2, 20, 10), errors_squared=np.linspace(2, 20, 10)
    )
    # Access "x" since it is generated but then stored in the class. This could disrupt YAML, so
    # we should explicitly test it.
    _ = input_hist.x

    # Dump and load (ie round trip)
    output_hist = dump_to_string_and_retrieve(input_hist, y=y)

    # Check the result
    assert input_hist == output_hist


@pytest.mark.parametrize(
    "axes",
    [
        ([np.linspace(0, 10, 11)]),
        ([np.linspace(0, 10, 11), np.linspace(0, 20, 21)]),
    ],
    ids=["1D", "2D"],
)
def test_binned_data_with_yaml(
    axes: list[npt.NDArray[np.float64]],
) -> None:
    """Test writing and then reading BinnedData via YAML.

    This ensures that writing BinnedData can be done successfully.
    """
    # Setup
    # YAML object
    y = yaml.yaml(modules_to_register=[binned_data])
    # y = yaml.yaml(classes_to_register = [binned_data.Axis, binned_data.AxesTuple, binned_data.BinnedData])

    # Determine the values based on the input axes for convenience.
    # By doing this, we can derive the test values and variances with the right shape, and we don't have to worry
    # overly much about the exact values, nor how we would generate them correctly.
    # NOTE: Remember that this needs to be calculated with bin centers, not bin edges! In the spirit of integration tests,
    #       we can use AxesTuples to help
    bin_centers = binned_data.AxesTuple([binned_data.Axis(v) for v in axes]).bin_centers
    # Specifies an array with a value across each point. As of September 2922, I don't fully understand the return values here,
    # but they give me the desired shape and the values vary, which is enough.
    mesh_grids = np.meshgrid(*bin_centers)

    # Test hist
    logger.info((mesh_grids[0].T * 2).size)
    logger.info(mesh_grids[0].T * 2)
    input_hist = binned_data.BinnedData(
        # Trick with the axes just to test directly passing a numpy array as well as a list
        axes=axes if len(axes) > 0 else axes[0],
        # We'll just multiply them be some factors (picking 2 arbitrarily) to ensure that they're not identical
        # (which might cause us to miss a serialization issue if it was uniform).
        values=mesh_grids[0].T * 2,
        variances=(mesh_grids[0].T * 2) ** 2,
    )
    # Access "bin_centers" since it is generated but then stored in the Axis class. This could disrupt YAML, so
    # we should explicitly test it.
    _ = input_hist.axes.bin_centers

    # Dump and load (ie round trip)
    output_hist = dump_to_string_and_retrieve(input_hist, y=y)

    # Check the result
    assert input_hist == output_hist
