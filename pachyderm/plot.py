#!/usr/bin/env python

""" Plotting styling and utilities.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import logging
from typing import Optional, Union

import matplotlib
import matplotlib.axes
import matplotlib.colors
import numpy as np

# Setup logger
logger = logging.getLogger(__name__)

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
    matplotlib.rcParams["text.latex.preamble"] = "\n".join([
        # Enable AMS math package (for among other things, "\text")
        r"\usepackage{amsmath}",
        # Add fonts that will be used below. See the `mathtext` fonts set below for further info.
        r"\usepackage{sfmath}",
        # Ensure that existing values are included.
        matplotlib.rcParams["text.latex.preamble"],
    ])
    params = {
        # Enable latex
        "text.usetex": True,
        # Enable tex preview, which improves the alignment of the baseline
        # Not necessary anymore for matplotlib ^3.3
        #"text.latex.preview": True,
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
        "axes.labelsize": 20.0,
        "axes.linewidth": 1.25,
        "axes.titlesize": 12.0,
        "font.size": 18.0,
        "legend.fontsize": 18.0,
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
        "xtick.labelsize": 20.0,
        "ytick.labelsize": 20.0,
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

def error_boxes(ax: matplotlib.axes.Axes,
                x_data: np.ndarray, y_data: np.ndarray,
                y_errors: np.ndarray, x_errors: np.ndarray = None,
                **kwargs: Union[str, float]) -> matplotlib.collections.PatchCollection:
    """ Plot error boxes for the given data.

    Inpsired by: https://matplotlib.org/gallery/statistics/errorbars_and_boxes.html and
    https://github.com/HDembinski/pyik/blob/217ae25bbc316c7a209a1a4a1ce084f6ca34276b/pyik/mplext.py#L138

    Args:
        ax: Axis onto which the rectangles will be drawn.
        x_data: x location of the data.
        y_data: y location of the data.
        y_errors: y errors of the data. The array can either be of length n, or of length (n, 2)
            for asymmetric errors.
        x_errors: x errors of the data. The array can either be of length n, or of length (n, 2)
            for asymmetric errors. Default: None. This corresponds to boxes that are 10% of the
            distance between the two given point and the previous one.
    """
    # Validation
    if x_errors is None:
        # Default to 10% of the distance between the two points.
        x_errors = (x_data[1:] - x_data[:-1]) * 0.1
        # Use the last width for the final point. (This is a bit of a hack).
        x_errors = np.append(x_errors, x_errors[-1])
        logger.debug(f"x_errors: {x_errors}")

    # Validate input data.
    if len(x_data) != len(y_data):
        raise ValueError(f"Length of x_data and y_data doesn't match! x_data: {len(x_data)}, y_data: {len(y_data)}")
    if len(x_errors.T) != len(x_data):
        raise ValueError(f"Length of x_data and x_errors doesn't match! x_data: {len(x_data)}, x_errors: {len(x_errors)}")
    if len(y_errors.T) != len(y_data):
        raise ValueError(f"Length of y_data and y_errors doesn't match! y_data: {y_data.shape}, y_errors: {y_errors.shape}")

    # Default arguments
    if "alpha" not in kwargs:
        kwargs["alpha"] = 0.5

    # Create the rectangles
    error_boxes = []
    # We need to transpose the errors, because they are expected to be of the shape (n, 2).
    # NOTE: It will still work as expected if they are only of length n.
    for x, y, xerr, yerr in zip(x_data, y_data, x_errors.T, y_errors.T):
        # For the errors, we want to support symmetric and asymmetric errors.
        # Thus, for asymmetric errors, we sum up the distance, but for symmetric
        # errors, we want to take * 2 of the error.
        xerr = np.atleast_1d(xerr)
        yerr = np.atleast_1d(yerr)
        #logger.debug(f"yerr: {yerr}")
        r = matplotlib.patches.Rectangle(
            (x - xerr[0], y - yerr[0]),
            xerr.sum() if len(xerr) == 2 else xerr * 2,
            yerr.sum() if len(yerr) == 2 else yerr * 2,
        )
        error_boxes.append(r)

    # Create the patch collection and add it to the given axis.
    patch_collection = matplotlib.collections.PatchCollection(
        error_boxes, **kwargs,
    )
    ax.add_collection(patch_collection)

    return patch_collection


def convert_mpl_color_scheme_to_ROOT(name: Optional[str] = None,
                                     cmap: Optional[Union[matplotlib.colors.ListedColormap, matplotlib.colors.LinearSegmentedColormap]] = None,
                                     reversed: bool = False, n_values_to_cut_from_top: int = 0) -> str:
    """ Convert matplotlib color scheme to ROOT.

    Args:
        name: Name of the matplotlib color scheme.
        reversed: True if the color scheme should be reversed.
        n_values_to_cut_from_top: Number of values to cut from the top of the color scheme.
    Returns:
        Snippet to add the color scheme to ROOT.
    """
    # Setup
    import numpy as np

    # Validation
    if name is None and cmap is None:
        raise ValueError("Must pass the name of a colormap to retrieve from MPL or an existing colormap.")
    # We select on the passed cmap rather than the name because we might use the name later.
    if cmap is None:
        color_scheme = matplotlib.cm.get_cmap(name)
    else:
        if not isinstance(cmap, matplotlib.colors.ListedColormap):
            color_scheme = matplotlib.colors.ListedColormap(cmap)
        else:
            color_scheme = cmap

    # Reverse if requested.
    if reversed:
        color_scheme = color_scheme.reversed()

    # Extract the colors
    if isinstance(color_scheme, matplotlib.colors.ListedColormap):
        arrs = np.array(color_scheme.colors)
        n_colors = arrs.shape[0]
    else:
        # Need to query for the colors, so we have to decide how many we want beforehand.
        n_colors = 255
        arrs = color_scheme(range(n_colors))
    reds = arrs[:, 0]
    greens = arrs[:, 1]
    blues = arrs[:, 2]

    if n_values_to_cut_from_top:
        n_colors -= n_values_to_cut_from_top
        reds = reds[:-n_values_to_cut_from_top]
        greens = greens[:-n_values_to_cut_from_top]
        blues = blues[:-n_values_to_cut_from_top]

    stops: np.ndarray = np.linspace(0, 1, n_colors + 1)

    def listing_array(arr: np.ndaray) -> str:
        return ", ".join(str(v) for v in arr)

    s = f"""
const Int_t NRGBs = {n_colors};
const Int_t NCont = 99;
Int_t colors[NRGBs] = {{0}};
Double_t stops[{len(stops)}] = {{ {listing_array(stops)} }};
Double_t red[NRGBs]   = {{ {listing_array(reds)} }};
Double_t green[NRGBs] = {{ {listing_array(greens)} }};
Double_t blue[NRGBs]  = {{ {listing_array(blues)} }};
/*Int_t res = TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
gStyle->SetNumberContours(NCont);*/

/*for (unsigned int i = 0; i < NRGBs; i++) {{
    colors[i] = res + i;
}}
gStyle->SetPalette(NCont + 1, colors);*/
"""
    return s
