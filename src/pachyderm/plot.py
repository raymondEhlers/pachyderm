""" Plotting styling and utilities.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University + Oak Ridge National Lab
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import attrs
import matplotlib.axes
import matplotlib.colors
import numpy as np
import numpy.typing as npt

# Setup logger
logger = logging.getLogger(__name__)


def restore_defaults() -> None:
    """Restore the default matplotlib settings."""
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def configure(disable_interactive_backend: bool = False) -> None:
    """Configure matplotlib according to my (biased) specification.

    As a high level summary, this is a combination of a number of seaborn settings, along with
    my own tweaks. By calling this function, the matplotlib ``rcParams`` will be modified according
    to these settings.

    Up to this point, the settings have been configured by importing the `jet_hadron.plot.base`
    module, which set a variety of parameters on import. This included some options which were set
    by seaborn. Additional modifications were made to the fonts to ensure that they are the same in
    labels and latex. Lastly, it tweaked smaller visual settings. The differences between the default
    matplotlib and these settings are:

    ```pycon
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
     #'xtick.minor.top': 'original: True, new: False',
     'xtick.minor.visible': 'original: False, new: True',
     'xtick.minor.width': 'original: 0.6, new: 1.0',
     'ytick.color': 'original: black, new: .15',
     'ytick.direction': 'original: out, new: in',
     'ytick.labelsize': 'original: medium, new: 11.0',
     'ytick.major.size': 'original: 3.5, new: 6.0',
     'ytick.major.width': 'original: 0.8, new: 1.25',
     'ytick.minor.right': 'original: True, new: False',
     'ytick.minor.size': 'original: 2.0, new: 4.0',
     #'ytick.minor.visible': 'original: False, new: True',
     'ytick.minor.width': 'original: 0.6, new: 1.0'}
    ```

    I implemented most of these below (although I left out a few color options).

    For more on the non-interactive mode,
    see: https://gist.github.com/matthewfeickert/84245837f09673b2e7afea929c016904

    Args:
        disable_interactive_backend: If True, configure the MPL backend to be non-interactive.
            This should make loading a bit more efficient, since I rarely use the GUI.
    Returns:
        None. The current matplotlib ``rcParams`` are modified.
    """
    # Color definitions from seaborn
    # NOTE: They need to be strings rather than raw floats.
    light_grey = ".8"
    # NOTE: I elect not to label with dark grey instead of black. It's not clear to me
    #       why that might be preferable here.

    if disable_interactive_backend:
        matplotlib.use("agg")

    # Setup the LaTeX preamble
    matplotlib.rcParams["text.latex.preamble"] = "\n".join(
        [
            # Enable AMS math package (for among other things, "\text")
            r"\usepackage{amsmath}",
            # Add fonts that will be used below. See the `mathtext` fonts set below for further info.
            r"\usepackage{sfmath}",
            # Ensure that existing values are included.
            matplotlib.rcParams["text.latex.preamble"],
        ]
    )
    params = {
        # Enable latex
        "text.usetex": True,
        # Enable tex preview, which improves the alignment of the baseline
        # Not necessary anymore for matplotlib ^3.3
        # "text.latex.preview": True,
        # Enable axis ticks (after they can be disabled by seaborn)
        "xtick.bottom": True,
        "xtick.top": True,
        "ytick.left": True,
        "ytick.right": True,
        # Make minor axis ticks visible
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
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
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Bitstream Vera Sans", "sans-serif"],
        # Make the grid lines light grey and slightly larger.
        "grid.color": light_grey,
        "grid.linewidth": 1.0,
        # End a line in a rounded style.
        "lines.solid_capstyle": "round",
        # This will disable lines connecting data points.
        # NOTE: This is disabled because if you forget to set the marker, then nothing will show up,
        #       which is a very frustrating user experience. Better to instead just disable it for a
        #       given plot.
        # "lines.linestyle": "none",
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


def error_boxes(
    ax: matplotlib.axes.Axes,
    x_data: npt.NDArray[Any],
    y_data: npt.NDArray[Any],
    y_errors: npt.NDArray[Any],
    x_errors: npt.NDArray[Any] | None = None,
    **kwargs: str | float,
) -> matplotlib.collections.PatchCollection:
    """Plot error boxes for the given data.

    Inspired by: https://matplotlib.org/gallery/statistics/errorbars_and_boxes.html and
    https://github.com/HDembinski/pyik/blob/217ae25bbc316c7a209a1a4a1ce084f6ca34276b/pyik/mplext.py#L138

    Note:
        The errors are distances from the central value. ie. for 10% error on 1, the two entry version
        should be [0.1, 0.1].

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
        # Help out mypy...
        assert x_errors is not None

    # Validate input data.
    if len(x_data) != len(y_data):
        _msg = f"Length of x_data and y_data doesn't match! x_data: {len(x_data)}, y_data: {len(y_data)}"
        raise ValueError(_msg)
    if len(x_errors.T) != len(x_data):
        _msg = f"Length of x_data and x_errors doesn't match! x_data: {len(x_data)}, x_errors: {len(x_errors)}"
        raise ValueError(_msg)
    if len(y_errors.T) != len(y_data):
        _msg = f"Length of y_data and y_errors doesn't match! y_data: {y_data.shape}, y_errors: {y_errors.shape}"
        raise ValueError(_msg)

    # Default arguments
    passed_kwargs: dict[str, Any] = kwargs.copy()
    if "alpha" not in kwargs:
        passed_kwargs["alpha"] = 0.5

    # Create the rectangles
    output_error_boxes = []
    # We need to transpose the errors, because they are expected to be of the shape (n, 2).
    # NOTE: It will still work as expected if they are only of length n.
    for x, y, xerr, yerr in zip(x_data, y_data, x_errors.T, y_errors.T, strict=True):
        # For the errors, we want to support symmetric and asymmetric errors.
        # Thus, for asymmetric errors, we sum up the distance, but for symmetric
        # errors, we want to take * 2 of the error.
        xerr = np.atleast_1d(xerr)  # noqa: PLW2901
        yerr = np.atleast_1d(yerr)  # noqa: PLW2901
        # NOTE: All of the calls to float are necessary to avoid the points being incorrectly
        #       interpreted as something like np arrays, which will cause the creation of the
        #       PatchCollection to fail. It's clunky, but this works, so good enough.
        r = matplotlib.patches.Rectangle(
            (float(x - xerr[0]), float(y - yerr[0])),
            float(xerr.sum() if len(xerr) == 2 else xerr * 2),
            float(yerr.sum() if len(yerr) == 2 else yerr * 2),
        )
        output_error_boxes.append(r)

    # Create the patch collection and add it to the given axis.
    patch_collection = matplotlib.collections.PatchCollection(
        output_error_boxes,
        **passed_kwargs,
    )
    ax.add_collection(patch_collection)

    return patch_collection


def convert_mpl_color_scheme_to_ROOT(
    name: str | None = None,
    cmap: matplotlib.colors.ListedColormap | matplotlib.colors.LinearSegmentedColormap | None = None,
    reverse_cmap: bool = False,
    n_values_to_cut_from_top: int = 0,
) -> str:
    """Convert matplotlib color scheme to ROOT.

    Args:
        name: Name of the matplotlib color scheme.
        reversed: True if the color scheme should be reversed.
        n_values_to_cut_from_top: Number of values to cut from the top of the color scheme.
    Returns:
        Snippet to add the color scheme to ROOT.
    """
    # Setup
    # import numpy as np

    # Validation
    if name is None and cmap is None:
        _msg = "Must pass the name of a colormap to retrieve from MPL or an existing colormap."
        raise ValueError(_msg)
    # We select on the passed cmap rather than the name because we might use the name later.
    if cmap is None:
        color_scheme = matplotlib.cm.get_cmap(name)
    elif not isinstance(cmap, matplotlib.colors.ListedColormap):
        color_scheme = matplotlib.colors.ListedColormap(cmap)  # type: ignore[arg-type]
    else:
        color_scheme = cmap

    # Reverse if requested.
    if reverse_cmap:
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

    stops: npt.NDArray[Any] = np.linspace(0, 1, n_colors + 1)

    def listing_array(arr: npt.NDArray[Any]) -> str:
        return ", ".join(str(v) for v in arr)

    return f"""
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


def _validate_axis_name(instance: AxisConfig, attribute: attrs.Attribute[str], value: str) -> None:  # noqa: ARG001
    if value not in ["x", "y", "z"]:
        _msg = f"Invalid axis name: {value}"
        raise ValueError(_msg)


def _convert_major_axis_multiple_locator_with_base(value: float | bool | None) -> float | None:
    if isinstance(value, bool) and value:
        return 1.0
    return value


@attrs.define
class AxisConfig:
    """Configuration for an axis."""

    axis: str = attrs.field(validator=[_validate_axis_name])
    label: str = attrs.field(default="")
    log: bool = attrs.field(default=False)
    range: tuple[float | None, float | None] | None = attrs.field(default=None)
    font_size: float | None = attrs.field(default=None)
    tick_font_size: float | None = attrs.field(default=None)
    # This is basically a shortcut. We can always apply this manually, but we do it
    # often enough that it's worth having a shortcut.
    use_major_axis_multiple_locator_with_base: float | None = attrs.field(
        default=None, converter=_convert_major_axis_multiple_locator_with_base
    )

    def apply(self, ax: matplotlib.axes.Axes) -> None:
        """Apply the axis configuration to the given axis."""
        # Validation
        if self.log and self.use_major_axis_multiple_locator_with_base is not None:
            logger.warning("Set both log and major axis multiple locator! Not sure what will happen...")

        # And apply the settings
        if self.label:
            getattr(ax, f"set_{self.axis}label")(self.label, fontsize=self.font_size)
        # Set the tick font size too. Here, we want to consider two cases:
        # 1) If we've provided a label, we want to increase the size as appropriate to match the label.
        # 2) If we're provided an explicit tick size, we definitely want to increase it.
        if self.label or self.tick_font_size:
            # The font size that we want is dictated by the label if unspecified. That way,
            # it will match the size by default
            tick_font_size = self.tick_font_size if self.tick_font_size else self.font_size
            ax.tick_params(axis=self.axis, which="major", labelsize=tick_font_size)  # type: ignore[arg-type]
        if self.log:
            getattr(ax, f"set_{self.axis}scale")("log")
            # Probably need to increase the number of ticks for a log axis. We just assume that's the case.
            # I really wish it handled this better by default...
            # See: https://stackoverflow.com/a/44079725/12907985
            major_locator = matplotlib.ticker.LogLocator(base=10, numticks=12)
            getattr(ax, f"{self.axis}axis").set_major_locator(major_locator)
            minor_locator = matplotlib.ticker.LogLocator(base=10.0, subs=np.linspace(0.2, 0.9, 8), numticks=12)  # type: ignore[arg-type]
            getattr(ax, f"{self.axis}axis").set_minor_locator(minor_locator)
            # But we don't want to label these ticks.
            getattr(ax, f"{self.axis}axis").set_minor_formatter(matplotlib.ticker.NullFormatter())
        if self.use_major_axis_multiple_locator_with_base is not None:
            getattr(ax, f"{self.axis}axis").set_major_locator(
                matplotlib.ticker.MultipleLocator(base=self.use_major_axis_multiple_locator_with_base)
            )
        if self.range:
            min_range, max_range = self.range
            min_current_range, max_current_range = getattr(ax, f"get_{self.axis}lim")()
            if min_range is None:
                min_range = min_current_range
            if max_range is None:
                max_range = max_current_range
            getattr(ax, f"set_{self.axis}lim")([min_range, max_range])


@attrs.define
class TextConfig:
    """Configuration for text on a plot."""

    text: str = attrs.field()
    x: float = attrs.field()
    y: float = attrs.field()
    alignment: str | None = attrs.field(default=None)
    color: str | None = attrs.field(default="black")
    font_size: float | None = attrs.field(default=None)
    text_kwargs: dict[str, Any] = attrs.field(factory=dict)

    def apply(self, ax_or_fig: matplotlib.axes.Axes | matplotlib.figure.Figure) -> None:
        """Apply the text configuration to the given axis or figure."""
        # Some reasonable defaults
        if self.alignment is None:
            ud = "upper" if self.y >= 0.5 else "lower"
            lr = "right" if self.x >= 0.5 else "left"
            self.alignment = f"{ud} {lr}"

        alignments = {
            "upper right": {
                "horizontalalignment": "right",
                "verticalalignment": "top",
                "multialignment": "right",
            },
            "upper left": {
                "horizontalalignment": "left",
                "verticalalignment": "top",
                "multialignment": "left",
            },
            "lower right": {
                "horizontalalignment": "right",
                "verticalalignment": "bottom",
                "multialignment": "right",
            },
            "lower left": {
                "horizontalalignment": "left",
                "verticalalignment": "bottom",
                "multialignment": "left",
            },
        }
        kwargs: dict[str, Any] = alignments[self.alignment]
        if not isinstance(ax_or_fig, matplotlib.figure.Figure):
            # We always want to place using normalized coordinates.
            # In the rare case that we don't want to, we can place by hand.
            kwargs["transform"] = ax_or_fig.transAxes
        # finally, merge in the rest of the text kwargs
        kwargs = {**kwargs, **self.text_kwargs}

        # Finally, draw the text.
        ax_or_fig.text(
            self.x,
            self.y,
            self.text,
            color=self.color,
            fontsize=self.font_size,
            **kwargs,
        )


@attrs.define
class LegendConfig:
    """Configuration for a legend on a plot."""

    location: str = attrs.field(default=None)
    # Takes advantage of the fact that None will use the default.
    anchor: tuple[float, float] | None = attrs.field(default=None)
    font_size: float | None = attrs.field(default=None)
    ncol: float | None = attrs.field(default=1)
    marker_label_spacing: float | None = attrs.field(default=None)
    # NOTE: Default in mpl is 0.5
    label_spacing: float | None = attrs.field(default=None)
    # NOTE: Default in mpl is 2.0
    column_spacing: float | None = attrs.field(default=None)
    handle_height: float | None = attrs.field(default=None)
    handler_map: dict[str, Any] = attrs.field(factory=dict)

    def apply(
        self,
        ax: matplotlib.axes.Axes,
        legend_handles: Sequence[matplotlib.container.ErrorbarContainer] | None = None,
        legend_labels: Sequence[str] | None = None,
    ) -> matplotlib.legend.Legend | None:
        """Apply the legend configuration to the given axis.

        Note:
            If provided, we'll use the given legend_handles and legend_labels to create the legend
            rather than those already associated with the legend.
        """
        if not self.location:
            return None
        kwargs: dict[str, Any] = {}
        if legend_handles:
            kwargs["handles"] = legend_handles
        if legend_labels:
            kwargs["labels"] = legend_labels

        return ax.legend(
            loc=self.location,
            bbox_to_anchor=self.anchor,
            # If we specify an anchor, we want to reduce an additional padding
            # to ensure that we have accurate placement.
            borderaxespad=(0 if self.anchor else None),
            borderpad=(0 if self.anchor else None),
            frameon=False,
            fontsize=self.font_size,
            ncol=self.ncol,
            handletextpad=self.marker_label_spacing,
            labelspacing=self.label_spacing,
            columnspacing=self.column_spacing,
            handleheight=self.handle_height,
            handler_map=(self.handler_map if self.handler_map else None),
            **kwargs,
        )


@attrs.define
class TitleConfig:
    """Configuration for a title of a plot."""

    text: str = attrs.field()
    size: float | None = attrs.field(default=None)

    def apply(
        self,
        ax: matplotlib.axes.Axes,
    ) -> None:
        """Apply the title configuration to the given axis."""
        ax.set_title(self.text, fontsize=self.size)


def _ensure_sequence_of_axis_config(value: AxisConfig | Sequence[AxisConfig]) -> Sequence[AxisConfig]:
    if isinstance(value, AxisConfig):
        value = [value]
    return value


def _ensure_sequence_of_text_config(value: TextConfig | Sequence[TextConfig]) -> Sequence[TextConfig]:
    if isinstance(value, TextConfig):
        value = [value]
    return value


@attrs.define
class Panel:
    """Configuration for a panel within a plot.

    The `Panel` is a configuration for an `ax` object.

    Attributes:
        axes: Configuration of the MPL axis. We allow for multiple AxisConfig because each config specifies
            a single axis (ie. x or y). Careful not to confuse with the actual `ax` object provided by MPL.
    """

    axes: Sequence[AxisConfig] = attrs.field(converter=_ensure_sequence_of_axis_config)
    text: Sequence[TextConfig] = attrs.field(converter=_ensure_sequence_of_text_config, factory=list)
    legend: LegendConfig | None = attrs.field(default=None)
    title: TitleConfig | None = attrs.field(default=None)

    def apply(
        self,
        ax: matplotlib.axes.Axes,
        legend_handles: Sequence[matplotlib.container.ErrorbarContainer] | None = None,
        legend_labels: Sequence[str] | None = None,
    ) -> None:
        """Apply the panel configuration to the given axis."""
        # Axes
        for axis in self.axes:
            axis.apply(ax)
        # Text
        for text in self.text:
            text.apply(ax)
        # Legend
        if self.legend is not None:
            self.legend.apply(ax, legend_handles=legend_handles, legend_labels=legend_labels)
        # Title
        if self.title is not None:
            self.title.apply(ax=ax)


@attrs.define
class Figure:
    """Configuration for a MPL figure."""

    edge_padding: Mapping[str, float] = attrs.field(factory=dict)
    text: Sequence[TextConfig] = attrs.field(converter=_ensure_sequence_of_text_config, factory=list)

    def apply(self, fig: matplotlib.figure.Figure) -> None:
        """Apply the figure configuration to the given figure."""
        # Add text
        for text in self.text:
            text.apply(fig)

        # It shouldn't hurt to align the labels if there's only one.
        fig.align_ylabels()

        # Adjust the layout.
        fig.tight_layout()
        adjust_default_args = {
            # Reduce spacing between subplots
            "hspace": 0,
            "wspace": 0,
            # Reduce external spacing
            "left": 0.10,
            "bottom": 0.105,
            "right": 0.98,
            "top": 0.98,
        }
        adjust_default_args.update(self.edge_padding)
        fig.subplots_adjust(**adjust_default_args)


def _ensure_sequence_of_panels(value: Panel | Sequence[Panel]) -> Sequence[Panel]:
    if isinstance(value, Panel):
        value = [value]
    return value


@attrs.define
class PlotConfig:
    """Configuration for an overall plot.

    A plot consists of some number of panels, which are each configured with their own axes, text, etc.
    These axes are on a figure.

    Attributes:
        name: Name of the plot. Usually used for the filename.
        panels: Configuration for the panels of the plot.
        figure: Configuration for the figure of the plot.
    """

    name: str = attrs.field()
    panels: Sequence[Panel] = attrs.field(converter=_ensure_sequence_of_panels)
    figure: Figure = attrs.field(factory=Figure)

    def apply(
        self,
        fig: matplotlib.figure.Figure,
        ax: matplotlib.axes.Axes | None = None,
        axes: Sequence[matplotlib.axes.Axes] | None = None,
        legend_handles: Sequence[matplotlib.container.ErrorbarContainer] | None = None,
        legend_labels: Sequence[str] | None = None,
    ) -> None:
        """Apply the plot configuration to the given figure and axes."""
        # Validation
        if ax is None and axes is None:
            _msg = "Must pass the axis or axes of the figure."
            raise TypeError(_msg)
        if ax is not None and axes is not None:
            _msg = "Cannot pass both a single axis and multiple axes."
            raise TypeError(_msg)
        # If we just have a single axis, wrap it up into a list so we can process it along with our panels.
        if ax is not None:
            axes = [ax]
        # Help out mypy...
        assert axes is not None
        if len(axes) != len(self.panels):
            _msg = f"Must have the same number of axes and panels. Passed axes: {axes}, panels: {self.panels}"
            raise ValueError(_msg)

        # Finally, we can actually apply the stored properties.
        # Apply panels to the axes.
        for _ax, panel in zip(axes, self.panels, strict=True):
            panel.apply(_ax, legend_handles=legend_handles, legend_labels=legend_labels)
        # Figure
        self.figure.apply(fig)
