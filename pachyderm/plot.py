#!/usr/bin/env python

""" Plotting styling and utilities.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import matplotlib

def restore_defaults() -> None:
    """ Restore the default matplotlib settings. """
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

def configure() -> None:
    """ Configure matplotlib according to my (biased) specification.

    As a high level summary, this is a combination of a number of seaborn settings, along with
    my own tweaks. By calling this function, the matplotlilb ``rcParams`` will be modified according
    to these settings.

    Up to this point, the settings have been configured by importing the `jet_hadron.plot.base`
    module, which set a variety of parameters on import. This included some options which were set
    by seaborn. Additional modifications were made to the fonts to ensure that they are the same in
    labels and latex. Lastly, it tweaked smaller visual settings. The differences between the default
    matplotlib and these settings are:

    .. code-block:: python
        >>> pprint.pprint(diff)
        {'axes.axisbelow': 'original: line, new: True',
         'axes.edgecolor': 'original: black, new: .15',
         'axes.labelcolor': 'original: black, new: .15',
         'axes.labelsize': 'original: medium, new: 12.0',
         'axes.linewidth': 'original: 0.8, new: 1.25',
         'axes.prop_cycle': "original: cycler('color', ['#1f77b4', '#ff7f0e', "
                            "'#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', "
                            "'#7f7f7f', '#bcbd22', '#17becf']), new: cycler('color', "
                            '[(0.2980392156862745, 0.4470588235294118, '
                            '0.6901960784313725), (0.8666666666666667, '
                            '0.5176470588235295, 0.3215686274509804), '
                            '(0.3333333333333333, 0.6588235294117647, '
                            '0.40784313725490196), (0.7686274509803922, '
                            '0.3058823529411765, 0.3215686274509804), '
                            '(0.5058823529411764, 0.4470588235294118, '
                            '0.7019607843137254), (0.5764705882352941, '
                            '0.47058823529411764, 0.3764705882352941), '
                            '(0.8549019607843137, 0.5450980392156862, '
                            '0.7647058823529411), (0.5490196078431373, '
                            '0.5490196078431373, 0.5490196078431373), (0.8, '
                            '0.7254901960784313, 0.4549019607843137), '
                            '(0.39215686274509803, 0.7098039215686275, '
                            '0.803921568627451)])',
         'axes.titlesize': 'original: large, new: 12.0',
         'font.sans-serif': "original: ['DejaVu Sans', 'Bitstream Vera Sans', "
                            "'Computer Modern Sans Serif', 'Lucida Grande', 'Verdana', "
                            "'Geneva', 'Lucid', 'Arial', 'Helvetica', 'Avant Garde', "
                            "'sans-serif'], new: ['Arial', 'DejaVu Sans', 'Liberation "
                            "Sans', 'Bitstream Vera Sans', 'sans-serif']",
         'font.size': 'original: 10.0, new: 12.0',
         'grid.color': 'original: #b0b0b0, new: .8',
         'grid.linewidth': 'original: 0.8, new: 1.0',
         'image.cmap': 'original: viridis, new: rocket',
         'legend.fontsize': 'original: medium, new: 11.0',
         'lines.solid_capstyle': 'original: projecting, new: round',
         'mathtext.bf': 'original: sans:bold, new: Bitstream Vera Sans:bold',
         'mathtext.fontset': 'original: dejavusans, new: custom',
         'mathtext.it': 'original: sans:italic, new: Bitstream Vera Sans:italic',
         'mathtext.rm': 'original: sans, new: Bitstream Vera Sans',
         'patch.edgecolor': 'original: black, new: w',
         'patch.facecolor': 'original: C0, new: (0.2980392156862745, '
                            '0.4470588235294118, 0.6901960784313725)',
         'patch.force_edgecolor': 'original: False, new: True',
         'text.color': 'original: black, new: .15',
         'text.usetex': 'original: False, new: True',
         'xtick.color': 'original: black, new: .15',
         'xtick.direction': 'original: out, new: in',
         'xtick.labelsize': 'original: medium, new: 11.0',
         'xtick.major.size': 'original: 3.5, new: 6.0',
         'xtick.major.width': 'original: 0.8, new: 1.25',
         'xtick.minor.size': 'original: 2.0, new: 4.0',
         'xtick.minor.top': 'original: True, new: False',
         'xtick.minor.visible': 'original: False, new: True',
         'xtick.minor.width': 'original: 0.6, new: 1.0',
         'ytick.color': 'original: black, new: .15',
         'ytick.direction': 'original: out, new: in',
         'ytick.labelsize': 'original: medium, new: 11.0',
         'ytick.major.size': 'original: 3.5, new: 6.0',
         'ytick.major.width': 'original: 0.8, new: 1.25',
         'ytick.minor.right': 'original: True, new: False',
         'ytick.minor.size': 'original: 2.0, new: 4.0',
         'ytick.minor.visible': 'original: False, new: True',
         'ytick.minor.width': 'original: 0.6, new: 1.0'}

    I implemented most of these below (although I left out a few color options).

    Args:
        None.
    Returns:
        None. The current matplotlib ``rcParams`` are modified.
    """
    # Color definitions from seaborn
    # NOTE: They need to be strings rather than raw floats.
    light_grey = ".8"
    # NOTE: I elect not to label with dark grey instead of black. It's not clear to me
    #       why that might be preferable here.

    # Setup the LaTeX preamble
    # Enable AMS math package (for among other things, "\text")
    matplotlib.rcParams["text.latex.preamble"].append(r"\usepackage{amsmath}")
    # Add fonts that will be used below. See the `mathtext` fonts set below for further info.
    matplotlib.rcParams["text.latex.preamble"].append(r"\usepackage{sfmath}")
    params = {
        # Enable latex
        "text.usetex": True,
        # Enable axis ticks (after they can be disabled by seaborn)
        "xtick.bottom": True,
        "ytick.left": True,
        # Make minor axis ticks visible (but only on left and bottom)
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.top": False,
        "ytick.minor.right": False,
        # Ensure that axis ticks go inward instead of outward
        "xtick.direction": "in",
        "ytick.direction": "in",
        # Below, we set the LaTeX fonts to be the same fonts as those used in matplotlib.
        # For sans serif fonts in LaTeX (required for setting the fonts below), see: https://stackoverflow.com/a/11612347
        # To set the latex fonts to be the same as the normal matplotlib fonts, see: https://stackoverflow.com/a/27697390
        "mathtext.fontset": "custom",
        "mathtext.rm": "Bitstream Vera Sans",
        "mathtext.it": "Bitstream Vera Sans:italic",
        "mathtext.bf": "Bitstream Vera Sans:bold",
        ##### Extracted from seaborn
        # Plot axis underneath points
        "axes.axisbelow": True,
        # Modify label sizes.
        "axes.labelsize": 12.0,
        "axes.linewidth": 1.25,
        "axes.titlesize": 12.0,
        "font.size": 12.0,
        "legend.fontsize": 11.0,
        # Set the possible sans serif fonts. These are the ones made available in seaborn.
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation " "Sans",
                            "Bitstream Vera Sans", "sans-serif"],
        # Make the grid lines light grey and slightly larger.
        "grid.color": light_grey,
        "grid.linewidth": 1.0,
        # End a line in a rounded style.
        "lines.solid_capstyle": "round",
        # This will disable lines connecting data points.
        # NOTE: This is disabled because if you forget to set the marker, then nothing will show up,
        #       which is a very frustrating user experience. Better to instead just disable it for a
        #       given plot.
        #"lines.linestyle": "none",
        # Set the edge color to white.
        "patch.edgecolor": "none",
        # Apparently this has to be enabled just for setting the edge color to be possible.
        "patch.force_edgecolor": True,
        # Tick label size.
        "xtick.labelsize": 11.0,
        "ytick.labelsize": 11.0,
        # Major tick settings
        "xtick.major.size": 6.0,
        "ytick.major.size": 6.0,
        "xtick.major.width": 1.25,
        "ytick.major.width": 1.25,
        # Minor tick settings
        "xtick.minor.size": 4.0,
        "ytick.minor.size": 4.0,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
    }

    # Apply the updated settings.
    matplotlib.rcParams.update(params)

