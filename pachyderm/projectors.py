#!/usr/bin/env python

""" Handle generic TH1 and THn projections.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import copy
import enum
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from pachyderm import generic_class
from pachyderm.typing_helpers import Hist, Axis

# Setup logger
logger = logging.getLogger(__name__)

class TH1AxisType(enum.Enum):
    """ Map from (x,y,z) axis to the axis number.

    Other enumerations that refer to this enum should refer to the _values_ to ensure
    consistency in `.value` pointing to the axis value.
    """
    x_axis = 0
    y_axis = 1
    z_axis = 2

def hist_axis_func(axis_type: enum.Enum) -> Callable[[Hist], Axis]:
    """ Wrapper to retrieve the axis of a given histogram.

    This can be convenient outside of just projections, so it's made available in the API.

    Args:
        axis_type: The type of axis to retrieve.
    Returns:
        Callable to retrieve the specified axis when given a hist.
    """
    def axis_func(hist: Hist) -> Axis:
        """ Retrieve the axis associated with the ``HistAxisRange`` object for a given hist.

        Args:
            hist: Histogram from which the selected axis should be retrieved.
            axis_type: Enumeration corresponding to the axis to be restricted. The numerical
                value of the enum should be axis number (for a THnBase).
        Returns:
            ROOT.TAxis: The axis associated with the ``HistAxisRange`` object.
        """
        # Determine the axis_type value
        # Use try here instead of checking for a particular type to protect against type changes
        # (say in the enum)
        try:
            # Try to extract the value from an enum
            hist_axis_type = axis_type.value
        except AttributeError:
            # Seems that we received an int, so just use that value
            hist_axis_type = axis_type

        if hasattr(hist, "ProjectionND") and hasattr(hist, "Projection"):
            # THnBase defines ProjectionND and Projection, so we will use those as proxies.
            # Return the proper THn access
            #logger.debug(f"From hist: {hist}, hist_axis_type: {hist_axis_type}, axis: {hist.GetAxis(hist_axis_type.value)}")
            return hist.GetAxis(hist_axis_type)
        else:
            # If it's not a THn, then it must be a TH1 derived
            axis_function_map = {
                TH1AxisType.x_axis.value: hist.GetXaxis,
                TH1AxisType.y_axis.value: hist.GetYaxis,
                TH1AxisType.z_axis.value: hist.GetZaxis
            }

            # Retrieve the axis function and execute it. It is done separately to
            # clarify any possible errors.
            return_func = axis_function_map[hist_axis_type]
            return return_func()

    return axis_func

RangeMinMaxType = Union[float, Callable[[Any], float]]

class HistAxisRange(generic_class.EqualityMixin):
    """ Represents the restriction of a range of an axis of a histogram.

    An axis can be restricted by multiple ``HistAxisRange`` elements (although separate projections are
    needed to apply more than one. This would be accomplished with separate entries to the
    HistProjector.projection_dependent_cut_axes).

    NOTE:
        A single axis which has multiple ranges could be represented by multiple ``HistAxisRange`` objects!

    Args:
        axis_range_name (str): Name of the axis range. Usually some combination of the axis name and
            some sort of description of the range.
        axis_type (enum.Enum): Enumeration corresponding to the axis to be restricted. The numerical
            value of the enum should be axis number (for a THnBase).
        min_val (function): Minimum range value for the axis. Usually set via ``apply_func_to_find_bin()``.
        min_val (function): Maximum range value for the axis. Usually set via ``apply_func_to_find_bin()``.
    """
    def __init__(self, axis_range_name: str, axis_type: enum.Enum, min_val: RangeMinMaxType, max_val: RangeMinMaxType):
        self.name = axis_range_name
        self.axis_type = axis_type
        self.min_val = min_val
        self.max_val = max_val

    def __repr__(self) -> str:
        """ Representation of the object. """
        # The axis type is an enumeration of some type. In such a case, we want the repr to represent
        # it using the str method instead
        return "{}(name = {name!r}, axis_type = {axis_type}, min_val = {min_val!r}, max_val = {max_val!r})".format(self.__class__.__name__, **self.__dict__)

    def __str__(self) -> str:
        """ Print the elements of the object. """
        return "{}: name: {name}, axis_type: {axis_type}, min_val: {min_val}, max_val: {max_val}".format(self.__class__.__name__, **self.__dict__)

    @property
    def axis(self) -> Callable[[Any], Any]:
        """ Determine the axis to return based on the hist type. """
        axis_func = hist_axis_func(
            axis_type = self.axis_type
        )
        return axis_func

    def apply_range_set(self, hist: Hist) -> None:
        """ Apply the associated range set to the axis of a given hist.

        Note:
            The min and max values should be bins, not user ranges! For more, see the binning
            explanation in ``apply_func_to_find_bin(...)``.

        Args:
            hist: Histogram to which the axis range restriction should be applied.
        Returns:
            None. The range is set on the axis.
        """
        # Do individual assignments to clarify which particular value is causing an error here.
        axis = self.axis(hist)
        #logger.debug(f"axis: {axis}, axis(): {axis.GetName()}")
        # Help out mypy
        assert not isinstance(self.min_val, float)
        assert not isinstance(self.max_val, float)
        # Evaluate the functions to determine the values.
        min_val = self.min_val(axis)
        max_val = self.max_val(axis)
        # NOTE: Using SetRangeUser() here was a bug, since I've been passing bin values! In general,
        #       passing bin values is more flexible, but requires the values to be passed to
        #       ``apply_func_to_find_bin()`` to be shifted by some small epsilon to get the desired bin.
        self.axis(hist).SetRange(min_val, max_val)

    @staticmethod
    def apply_func_to_find_bin(
        func: Union[None, Callable[..., Union[float, int, Any]]],
        values: Optional[float] = None
    ) -> Callable[[Any], Union[float, int]]:
        """ Closure to determine the bin associated with a value on an axis.

        It can apply a function to an axis if necessary to determine the proper bin.  Otherwise,
        it can just return a stored value.

        Note:
            To properly determine the value, carefully note the information below. In many cases,
            such as when we want values [2, 5), the values need to be shifted by a small epsilon
            to retrieve the proper bin. This is done automatically in ``SetRangeUser()``.

            >>> hist = ROOT.TH1D("test", "test", 10, 0, 10)
            >>> x = 2, y = 5
            >>> hist.FindBin(x)
            2
            >>> hist.FindBin(x+epsilon)
            2
            >>> hist.FindBin(y)
            6
            >>> hist.FindBin(y-epsilon)
            5

            Note that the bin + epsilon on the lower bin is not strictly necessary, but it is
            used for consistency with the upper bound.

        Args:
            func (Callable): Function to apply to the histogram axis. If it is None, the value
                will be returned.
            values (int or float): Value to pass to the function. Default: None (in which case,
                it won't be passed).
        Returns:
            Function to be called with an axis to determine the desired bin on that axis.
        """
        def return_func(axis) -> Any:
            """ Apply the stored function and value to a given axis.

            Args:
                axis (TAxis or similar): Axis to which the function should be applied.
            Returns:
                any: The value returned by the function. Often a float or int, but not necessarily.
            """
            #logger.debug(f"func: {func}, values: {values}")
            if func:
                if values is not None:
                    return func(axis, values)
                else:
                    return func(axis)
            else:
                return values

        return return_func

class HistProjector:
    """ Handles generic ROOT ``THn`` and ``TH1`` projections.

    There are three types of cuts which can be specified:

    - ``additional_axis_cuts``: Axis cuts which do not change based on the projection axis.
    - ``projection_dependent_cut_axes``: Axis cuts which change based on the projection axis.
    - ``projection_axes``: Axes onto which the projection will be performed.

    For a full description of each type of cut and the necessary details, see their descriptions
    in the attributes.

    Note:
        The TH1 projections have not been tested as extensively as the ``THn`` projections.

    Note:
        ``input_key``, ``input_hist``, ``input_observable``, ``projection_name``, and ``output_hist`` are
        all reserved keys, such they will be overwritten by predefined information when passed to the
        various functions. Thus, they should be avoided by the user when storing projection information

    Args:
        observable_to_project_from: The observables which should be used to project from. The dict
            key is passed to ``projection_name(...)`` as ``input_key``.
        output_observable: Object or dict where the projected hist will be stored.
        projection_name_format: Format string to determine the projected hist name.
        output_attribute_name: Name of the attribute where which the single observable projection will
            be stored in the output_observable object. Must not be specified if projecting with multiple
            objects. Default: None.
        projection_information: Keyword arguments to be passed to ``projection_name(...)`` to determine
            the name of the projected histogram. Default: None.

    Attributes:
        single_observable_projection: True if the projector is only performing a single observable projection.
        output_attribute_name: Name of the attribute under which the single observable projection will be stored
            in the output_observable object.
        observable_to_project_from: The observable(s) which should be used to project from. The dict
            key is passed to ``projection_name(...)`` as ``input_key``.
        output_observable: Where the projected hist(s) will be stored. They will be stored under the dict
            key determined by ``output_key_name(...)``.
        projection_name_format: Format string to determine the projected hist name.
        projection_information: Keyword arguments to be passed to ``projection_name(...)`` to determine
            the name of the projected histogram.
        additional_axis_cuts (list): List of axis cuts which are neither projected nor depend on the axis
            being projected.
        projection_dependent_cut_axes (list): List of list of axis cuts which depend on the projected axis.
            For example, if we want to project non-continuous ranges of a non-projection axis (say, dEta
            when projecting dPhi). It is a list of list to allow for groups of cuts to be specified
            together if necessary.
        projection_axes (list): List of axes which should be projected.
    """
    def __init__(self,
                 observable_to_project_from: Union[Dict[str, Any], Hist, Any],
                 output_observable: Union[Dict[str, Any], Hist, Any],
                 projection_name_format: str,
                 output_attribute_name: str = None,
                 projection_information: Optional[Dict[str, Any]] = None):
        # Determine whether we projecting multiple objects.
        # If not, store the attribute under which we are going to store the output.
        single_observable_projection = False
        if output_attribute_name is not None:
            single_observable_projection = True
        self.single_observable_projection = single_observable_projection
        self.output_attribute_name = output_attribute_name

        # Validate and assign input and output objects.
        if self.single_observable_projection and (isinstance(observable_to_project_from, dict) or isinstance(output_observable, dict)):
            raise ValueError("A dict of observables is not compatible with a single observable projection. Check your arguments.")
        self.observable_to_project_from = observable_to_project_from
        self.output_observable = output_observable

        # Output hist name format
        self.projection_name_format = projection_name_format

        # Additional projection information to help create names, input/output objects, etc
        # NOTE: See reserved keys enumerated above.
        if projection_information is None:
            projection_information = {}
        # Ensure that the dict is copied successfully
        self.projection_information = copy.deepcopy(projection_information)

        # Axes
        # Cuts for axes which are not projected
        self.additional_axis_cuts: List[HistAxisRange] = []
        # Axes cuts which depend on the projection axes
        # ie. If we want to change the range of the axis that we are projecting
        # For example, we may want to project an axis non-continuously (say, -1 - 0.5, 0.5 - 1)
        self.projection_dependent_cut_axes: List[List[HistAxisRange]] = []
        # Axes to actually project
        self.projection_axes: List[HistAxisRange] = []

    # Printing functions
    def __str__(self) -> str:
        """ Prints the properties of the projector.

        This will only show up properly when printed - otherwise the tabs and newlines won't be printed.
        """
        ret_val = f"{self.__class__.__name__}: Projection Information:\n"
        ret_val += f"\tprojection_name_format: \"{self.projection_name_format}\""
        ret_val += "\n\tprojection_information:\n"
        ret_val += "\n".join(["\t\t- " + str("Arg: ") + str(val) for arg, val in self.projection_information.items()])
        ret_val += "\n\tadditional_axis_cuts:\n"
        ret_val += "\n".join(["\t\t- " + str(axis) for axis in self.additional_axis_cuts])
        ret_val += "\n\tprojection_dependent_cut_axes:\n"
        ret_val += "\n".join(["\t\t- " + str([",".join(axis.name for axis in axis_list)]) for axis_list in self.projection_dependent_cut_axes])
        ret_val += "\n\tprojection_axes:\n"
        ret_val += "\n".join(["\t\t- " + str(axis) for axis in self.projection_axes])

        return ret_val

    def call_projection_function(self, hist: Hist) -> Hist:
        """ Calls the actual projection function for the hist.

        Args:
            hist: Histogram from which the projections should be performed.
        Returns:
            The projected histogram.
        """
        # Restrict projection axis ranges
        for axis in self.projection_axes:
            logger.debug(f"Apply projection axes hist range: {axis.name}")
            axis.apply_range_set(hist)

        projected_hist = None
        if hasattr(hist, "ProjectionND") and hasattr(hist, "Projection"):
            # THnBase defines ProjectionND and Projection, so we will use those as proxies.
            projected_hist = self._project_THn(hist = hist)
        elif hasattr(hist, "ProjectionZ") and hasattr(hist, "Project3D"):
            # TH3 defines ProjectionZ and Project3D, so we will use those as proxies.
            projected_hist = self._project_TH3(hist = hist)
        elif hasattr(hist, "ProjectionX") and hasattr(hist, "ProjectionY"):
            # TH2 defines ProjectionX and ProjectionY, so we will use those as proxies.
            projected_hist = self._project_TH2(hist = hist)
        else:
            raise TypeError(type(hist), f"Could not recognize hist {hist} of type {type(hist)}")

        # Cleanup restricted axes
        self.cleanup_cuts(hist, cut_axes = self.projection_axes)

        return projected_hist

    def _project_THn(self, hist: Hist) -> Any:
        """ Perform the actual THn -> THn or TH1 projection.

        This projection could be to 1D, 2D, 3D, or ND.

        Args:
            hist (ROOT.THnBase): Histogram from which the projections should be performed.
        Returns:
            ROOT.THnBase or ROOT.TH1: The projected histogram.
        """
        # THnBase projections args are given as a list of axes, followed by any possible options.
        projection_axes = [axis.axis_type.value for axis in self.projection_axes]

        # Handle ROOT THnBase quirk...
        # 2D projection are called as (y, x, options), so we should reverse the order so it performs
        # as expected
        if len(projection_axes) == 2:
            # Reverses in place
            projection_axes.reverse()

        # Test calculating errors
        # Add "E" to ensure that errors will be calculated
        args = projection_axes + ["E"]
        # Do the actual projection
        logger.debug(f"hist: {hist.GetName()} args: {args}")

        if len(projection_axes) > 3:
            # Project into a THnBase object.
            projected_hist = hist.ProjectionND(*args)
        else:
            # Project a TH1 derived object.
            projected_hist = hist.Projection(*args)

        return projected_hist

    def _project_TH3(self, hist: Hist) -> Any:
        """ Perform the actual TH3 -> TH1 projection.

        This projection could be to 1D or 2D.

        Args:
            hist (ROOT.TH3): Histogram from which the projections should be performed.
        Returns:
            ROOT.TH1: The projected histogram.
        """
        # Axis length validation
        if len(self.projection_axes) < 1 or len(self.projection_axes) > 2:
            raise ValueError(len(self.projection_axes), "Invalid number of axes")

        # Need to concatenate the names of the axes together
        projection_axis_name = ""
        for axis in self.projection_axes:
            # Determine the axis name based on the name of the axis type.
            # [:1] returns just the first letter. For example, we could get "xy" if the first axis as
            # x_axis and the second was y_axis.
            # NOTE: Careful. This depends on the name of the enumerated values!!! Since this isn't terribly
            #       safe, we then perform additional validation on the same to ensure that it is one of the
            #       expected axis names.
            proj_axis_name = axis.axis_type.name[:1]
            if proj_axis_name not in ["x", "y", "z"]:
                raise ValueError(f"Projection axis name {proj_axis_name} is not 'x', 'y', or 'z'. Please check your configuration.")
            projection_axis_name += proj_axis_name

        # Handle ROOT Project3D quirk...
        # 2D projection are called as (y, x, options), so we should reverse the order so it performs
        # as expected.
        # NOTE: This isn't well documented in TH3. It is instead described in THnBase.Projection(...)
        if len(self.projection_axes) == 2:
            # Reverse the axes
            projection_axis_name = projection_axis_name[::-1]

        # Do the actual projection
        logger.info(f"Projecting onto axes \"{projection_axis_name}\" from hist {hist.GetName()}")
        projected_hist = hist.Project3D(projection_axis_name)

        return projected_hist

    def _project_TH2(self, hist: Hist) -> Any:
        """ Perform the actual TH2 -> TH1 projection.

        This projection can only be to 1D.

        Args:
            hist (ROOT.TH2): Histogram from which the projections should be performed.
        Returns:
            ROOT.TH1: The projected histogram.
        """
        if len(self.projection_axes) != 1:
            raise ValueError(len(self.projection_axes), "Invalid number of axes")

        #logger.debug(f"self.projection_axes[0].axis: {self.projection_axes[0].axis}, axis range name: {self.projection_axes[0].name}, axis_type: {self.projection_axes[0].axis_type}")
        # NOTE: We cannot use TH3.ProjectionZ(...) because it has different semantics than ProjectionX
        #       and ProjectionY. In particular, it doesn't respect the axis limits of axis onto which it
        #       is projected.  So we have to separate the projection by histogram type as opposed to axis
        #       length.
        projection_func_map = {
            TH1AxisType.x_axis.value: hist.ProjectionX,
            TH1AxisType.y_axis.value: hist.ProjectionY
        }

        # Determine the axis_type value
        # Use try here instead of checking for a particular type to protect against type changes (say
        # in the enum)
        try:
            # Try to extract the value from an enum
            axis_type = self.projection_axes[0].axis_type.value
        except ValueError:
            # Seems that we received an int, so just use that value
            axis_type = self.axis_type  # type: ignore

        projection_func = projection_func_map[axis_type]

        # Do the actual projection
        logger.info(f"Projecting onto axis range {self.projection_axes[0].name} from hist {hist.GetName()}")
        projected_hist = projection_func()

        return projected_hist

    def _project_observable(self, input_key: str,
                            input_observable: Any,
                            get_hist_args: Dict[str, Any] = None,
                            projection_name_args: Dict[str, Any] = None,
                            **kwargs) -> Hist:
        """ Perform a projection for a single observable.

        Note:
            All cuts on the original histograms will be reset when this function is completed.

        Args:
            input_key: Key to describe the input observable.
            input_observable: Observable to project from.
            get_hist_args: Arguments to pass to ``get_hist(...)``. Made available so the args can be cached
                to avoid a ``deepcopy`` when looping. Default: None. In this case, they will be retrieved
                automatically.
            projection_name_args: Arguments to pass to ``projection_name(...)``. Made available so the args
                can be cached to avoid a ``deepcopy`` when looping. Default: None. In this case, they will be
                retrieved automatically.
            kwargs: Additional named args to be passed to projection_name(...) and output_key_name(...).
        Returns:
            The projected histogram.
        """
        # Validation of other optional arguments.
        if get_hist_args is None:
            get_hist_args = copy.deepcopy(kwargs)
        if projection_name_args is None:
            projection_name_args = copy.deepcopy(kwargs)

        # Retrieve histogram
        # We update ``input_observable`` in ``get_hist_args`` every loop, so we don't have to worry
        # about passing the wrong observable.
        get_hist_args.update({"observable": input_observable})
        hist = self.get_hist(**get_hist_args)

        # Define projection name
        projection_name_args.update(self.projection_information)
        # In principle, we could have overwritten one of the kwargs, so we ensure with one of the other
        # updates, so we update it again to be certain.
        projection_name_args.update(kwargs)
        # Put the values included by default last to ensure nothing overwrites these values
        projection_name_args.update({  # type: ignore
            "input_key": input_key,
            "input_observable": input_observable,
            "input_hist": hist
        })
        projection_name = self.projection_name(**projection_name_args)

        # First apply the cuts
        # Restricting the range with SetRange(User) works properly for both THn and TH1.
        logger.debug(f"hist: {hist}")
        for axis in self.additional_axis_cuts:
            logger.debug(f"Apply additional axis hist range: {axis.name}")
            axis.apply_range_set(hist)

        # We need to ensure that it isn't empty so at least one project occurs
        if self.projection_dependent_cut_axes == []:
            self.projection_dependent_cut_axes.append([])

        # Validate the projection dependent cut axes
        # It is invalid to have PDCA on the same axes as the projection axes.
        duplicated_axes = [
            PDCA
            for PA in self.projection_axes
            for PDCA_group in self.projection_dependent_cut_axes
            for PDCA in PDCA_group
            if PDCA.axis_type == PA.axis_type
        ]
        if duplicated_axes:
            raise ValueError(
                f"Axis {duplicated_axes} is in the projection axes and the projection dependent cut axes."
                " This configuration is not allowed, as the range in the PDCA will be overwritten by the projection axes!"
                " Please revise your configuration."
            )

        # Perform the projections
        hists = []
        for i, axes in enumerate(self.projection_dependent_cut_axes):
            # Projection dependent range set
            for axis in axes:
                logger.debug(f"Apply projection dependent hist range: {axis.name}")
                axis.apply_range_set(hist)

            # Do the projection
            projected_hist = self.call_projection_function(hist)
            projected_hist.SetName(f"{projection_name}_{i}")

            hists.append(projected_hist)

            # Cleanup projection dependent cuts (although they should be set again on the next
            # iteration of the loop)
            self.cleanup_cuts(hist, cut_axes = axes)

        # Cleanup the rest of the cuts
        self.cleanup_cuts(hist, cut_axes = self.additional_axis_cuts)

        # Combine all of the projections together
        output_hist = hists[0]
        for temp_hist in hists[1:]:
            output_hist.Add(temp_hist)

        # Final settings
        output_hist.SetName(projection_name)
        # Ensure that the hist doesn't get deleted by ROOT
        # A reference to the histogram within python may not be enough
        output_hist.SetDirectory(0)

        return output_hist, projection_name, projection_name_args

    def _project_single_observable(self, **kwargs: Dict[str, Any]) -> Hist:
        """ Driver function for projecting and storing a single observable.

        Args:
            kwargs (dict): Additional named args to be passed to projection_name(...) and output_key_name(...)
        Returns:
            The projected histogram. The histogram is also stored in the output specified by ``output_observable``.
        """
        # Help out mypy
        assert isinstance(self.output_attribute_name, str)

        # Run the actual projection.
        output_hist, projection_name, projection_name_args, = self._project_observable(
            input_key = "single_observable",
            input_observable = self.observable_to_project_from,
            **kwargs,
        )
        # Store the output.
        output_hist_args = projection_name_args
        output_hist_args.update({  # type: ignore
            "output_hist": output_hist,
            "projection_name": projection_name
        })

        # Store the final histogram.
        output_hist = self.output_hist(**output_hist_args)  # type: ignore

        # Store the final output hist
        if not hasattr(self.output_observable, self.output_attribute_name):
            raise ValueError(f"Attempted to assign hist to non-existent attribute {self.output_attribute_name} of object {self.output_observable}. Check the attribute name!")
        # Actually store the histogram.
        setattr(self.output_observable, self.output_attribute_name, output_hist)

        # Return the observable
        return output_hist

    def _project_dict(self, **kwargs: Dict[str, Any]) -> Dict[str, Hist]:
        """ Driver function for projecting and storing a dictionary of observables.

        Args:
            kwargs (dict): Additional named args to be passed to projection_name(...) and output_key_name(...)
        Returns:
            The projected histograms. The projected histograms are also stored in ``output_observable``.
        """
        # Setup function arguments with values which don't change per loop.
        get_hist_args = copy.deepcopy(kwargs)
        projection_name_args = copy.deepcopy(kwargs)
        for key, input_observable in self.observable_to_project_from.items():
            output_hist, projection_name, projection_name_args, = self._project_observable(
                input_key = key,
                input_observable = input_observable,
                get_hist_args = get_hist_args,
                projection_name_args = projection_name_args,
                **kwargs,
            )

            # Store the output observable
            output_hist_args = projection_name_args
            output_hist_args.update({  # type: ignore
                "output_hist": output_hist,
                "projection_name": projection_name
            })
            output_key_name = self.output_key_name(**output_hist_args)  # type: ignore
            self.output_observable[output_key_name] = self.output_hist(**output_hist_args)  # type: ignore

        return self.output_observable

    def project(self, **kwargs: Dict[str, Any]) -> Union[Hist, Dict[str, Hist]]:
        """ Perform the requested projection(s).

        Note:
            All cuts on the original histograms will be reset when this function is completed.

        Args:
            kwargs (dict): Additional named args to be passed to projection_name(...) and output_key_name(...)
        Returns:
            The projected histogram(s). The projected histograms are also stored in ``output_observable``.
        """
        if self.single_observable_projection:
            return self._project_single_observable(**kwargs)
        else:
            return self._project_dict(**kwargs)

    def cleanup_cuts(self, hist: Hist, cut_axes: Iterable[HistAxisRange]) -> None:
        """ Cleanup applied cuts by resetting the axis to the full range.

        Inspired by: https://github.com/matplo/rootutils/blob/master/python/2.7/THnSparseWrapper.py

        Args:
            hist: Histogram for which the axes should be reset.
            cut_axes: List of axis cuts, which correspond to axes that should be reset.
        """
        for axis in cut_axes:
            # According to the function TAxis::SetRange(first, last), the widest possible range is
            # (1, Nbins). Anything beyond that will be reset to (1, Nbins)
            axis.axis(hist).SetRange(1, axis.axis(hist).GetNbins())

    #############################
    # Functions to be overridden!
    #############################
    def projection_name(self, **kwargs: Dict[str, Any]) -> str:
        """ Define the projection name for this projector.

        Note:
            This function is just a basic placeholder and likely should be overridden.

        Args:
            kwargs: Projection information dict combined with additional arguments passed to the
                projection function.
        Returns:
            Projection name string formatted with the passed options. By default, it returns
                ``projection_name_format`` formatted with the arguments to this function.
        """
        return self.projection_name_format.format(**kwargs)

    def get_hist(self, observable: Any, **kwargs: Dict[str, Any]) -> Any:
        """ Get the histogram that may be stored in some object.

        This histogram is used to project from.

        Note:
            The output object could just be the raw ROOT histogram.

        Note:
            This function is just a basic placeholder and likely should be overridden.

        Args:
            observable (object): The input object. It could be a histogram or something more complex
            kwargs: Additional arguments passed to the projection function
        Return:
            ROOT.TH1 or ROOT.THnBase histogram which should be projected. By default, it returns the
                observable (input object).
        """
        return observable

    def output_key_name(self, input_key: str, output_hist: Hist, projection_name: str, **kwargs) -> str:
        """ Returns the key under which the output object should be stored.

        Note:
            This function is just a basic placeholder which returns the projection name
            and likely should be overridden.

        Args:
            input_key: Key of the input hist in the input dict
            output_hist: The output histogram
            projection_name: Projection name for the output histogram
            kwargs: Projection information dict combined with additional arguments passed to
                the projection function.
        Returns:
            Key under which the output object should be stored. By default, it returns the
                projection name.
        """
        return projection_name

    def output_hist(self, output_hist: Hist, input_observable: Any, **kwargs: Dict[str, Any]) -> Union[Hist, Any]:
        """ Return an output object. It should store the ``output_hist``.

        Note:
            The output object could just be the raw histogram.

        Note:
            This function is just a basic placeholder which returns the given output object (a histogram)
            and likely should be overridden.

        Args:
            output_hist: The output histogram
            input_observable (object): The corresponding input object. It could be a histogram or something
                more complex.
            kwargs: Projection information dict combined with additional arguments passed to the
                projection function
        Return:
            The output object which should be stored in the output dict. By default, it returns the
                output hist.
        """
        return output_hist

