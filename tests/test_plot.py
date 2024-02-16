""" Tests for the plot module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""
from __future__ import annotations

from typing import Any

import matplotlib as mpl
import pytest

import pachyderm.plot as pplot


@pytest.fixture()
def reset_matplotlib_options() -> None:  # noqa: PT004
    """Setup for matplotlib options testing by resetting the options."""
    mpl.rcParams.update(mpl.rcParamsDefault)


def test_restore_default_configuration(reset_matplotlib_options: Any) -> None:  # noqa: ARG001
    """Test for resetting the plotting configuration."""
    # Modify the parameters
    mpl.rcParams["text.usetex"] = True
    # Check that it was set correctly (so that our restore actually does something)

    assert mpl.rcParams["text.usetex"] is True
    # Then restore the defaults
    pplot.restore_defaults()

    # Check that they've changed.
    # Of course, this is just a proxy for the rest of the values
    assert mpl.rcParams["text.usetex"] is False


def test_plot_configuration(reset_matplotlib_options: Any) -> None:  # noqa: ARG001
    """Test for updating the plot configuration."""
    # Check that we're starting from default settings.
    # Of course, this is just a proxy for the rest of the values
    assert mpl.rcParams["text.usetex"] is False

    # Configure the settings.
    pplot.configure()

    # Check that they've changed.
    # Of course, these are just proxies for the rest of the values
    assert mpl.rcParams["text.usetex"] is True
    assert mpl.rcParams["legend.fontsize"] == 18.0  # type: ignore[unreachable]

    # NOTE: Unfortunately, we cannot actually plot with the current settings because:
    #        - Plotting requires LaTeX, which is not available in testing environments (travis, etc).
    #        - pytest-mpl will reset the style to the default before the comparison, and we don't set
    #          the style in a way that plays nice with their method for setting the style.
    #       So skip plotting and rely on our assertions above.
