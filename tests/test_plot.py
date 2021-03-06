#!/usr/bin/env python

""" Tests for the plot module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from typing import Any

import matplotlib
import pytest

import pachyderm.plot as pplot

@pytest.fixture  # type: ignore
def reset_matplotlib_options(logging_mixin: Any) -> None:
    """ Setup for matplotlib options testing by resetting the options. """
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

def test_restore_default_configuration(reset_matplotlib_options: Any) -> None:
    """ Test for reseting the plotting configuration. """
    # Modify the parameters
    matplotlib.rcParams["text.usetex"] = True
    # Check that it was set correctly (so that our restore actually does something)

    assert matplotlib.rcParams["text.usetex"] is True
    # Then restore the defaults
    pplot.restore_defaults()

    # Check that they've changed.
    # Of course, this is just a proxy for the rest of the values
    assert matplotlib.rcParams["text.usetex"] is False

def test_plot_configuration(reset_matplotlib_options: Any) -> None:
    """ Test for updating the plot configuration. """
    # Check that we're starting from default settings.
    # Of course, this is just a proxy for the rest of the values
    assert matplotlib.rcParams["text.usetex"] is False

    # Configure the settings.
    pplot.configure()

    # Check that they've changed.
    # Of course, these are just proxies for the rest of the values
    assert matplotlib.rcParams["text.usetex"] is True
    assert matplotlib.rcParams["legend.fontsize"] == 18.0

    # NOTE: Unfortunately, we cannot actually plot with the current settings because:
    #        - Plotting requires LaTeX, which is not available in testing envrionments (travis, etc).
    #        - pytest-mpl will reset the style to the default before the comparison, and we don't set
    #          the style in a way that plays nice with their method for setting the style.
    #       So skip plotting and rely on our assertions above.
