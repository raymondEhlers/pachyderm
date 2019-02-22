#!/usr/bin/env python

""" Analysis configuration base module.

For usage information, see ``jet_hadron.base.analysis_config``.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import copy
import dataclasses
import enum
import itertools
import logging
import string
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple, Type, Union

from pachyderm import yaml
from pachyderm.yaml import DictLike

logger = logging.getLogger(__name__)

def load_configuration(yaml: yaml.ruamel.yaml.YAML, filename: str) -> DictLike:
    """ Load an analysis configuration from a file.

    Args:
        yaml: YAML object to use in loading the configuration.
        filename: Filename of the YAML configuration file.
    Returns:
        dict-like object containing the loaded configuration
    """
    with open(filename, "r") as f:
        config = yaml.load(f)

    return config

def override_options(config: DictLike, selected_options: Tuple[Any, ...], set_of_possible_options: Tuple[enum.Enum, ...], config_containing_override: DictLike = None) -> DictLike:
    """ Determine override options for a particular configuration.

    The options are determined by searching following the order specified in selected_options.

    For the example config,

    .. code-block:: yaml

        config:
            value: 3
            override:
                2.76:
                    track:
                        value: 5

    value will be assigned the value 5 if we are at 2.76 TeV with a track bias, regardless of the event
    activity or leading hadron bias. The order of this configuration is specified by the order of the
    selected_options passed. The above example configuration is from the jet-hadron analysis.

    Since anchors aren't kept for scalar values, if you want to override an anchored value, you need to
    specify it as a single value in a list (or dict, but list is easier). After the anchor values propagate,
    single element lists can be converted into scalar values using ``simplify_data_representations()``.

    Args:
        config: The dict-like configuration from ruamel.yaml which should be overridden.
        selected_options: The selected analysis options. They will be checked in the order with which
            they are passed, so make certain that it matches the order in the configuration file!
        set_of_possible_options (tuple of enums): Possible options for the override value categories.
        config_containing_override: The dict-like config containing the override options in a map called
            "override". If it is not specified, it will look for it in the main config.
    Returns:
        dict-like object: The updated configuration
    """
    if config_containing_override is None:
        config_containing_override = config
    override_opts = config_containing_override.pop("override")
    override_dict = determine_override_options(selected_options, override_opts, set_of_possible_options)
    logger.debug(f"override_dict: {override_dict}")

    # Set the configuration values to those specified in the override options
    # Cannot just use update() on config because we need to maintain the anchors.
    for k, v in override_dict.items():
        # Check if key is there and if it is not None! (The second part is important)
        if k in config:
            try:
                # If it has an anchor, we know that we want to preserve the type. So we check for the anchor
                # by trying to access it (Note that we don't actually care what the value is - just that it
                # exists). If it fails with an AttributeError, then we know we can just assign the value. If it
                # has an anchor, then we want to preserve the anchor information.
                config[k].anchor
                logger.debug(f"type: {type(config[k])}, k: {k}")
                if isinstance(config[k], list):
                    # Clear out the existing list entries
                    del config[k][:]
                    if isinstance(override_dict[k], (str, int, float, bool)):
                        # We have to treat str carefully because it is an iterable, but it will be expanded as
                        # individual characters if it's treated the same as a list, which is not the desired
                        # behavior! If we wrap it in [], then it will be treated as the only entry in the list
                        # NOTE: We also treat the basic types this way because they will be passed this way if
                        #       overriding indirectly with anchors (since the basic scalar types don't yet support
                        #       reassignment while maintaining their anchors).
                        config[k].append(override_dict[k])
                    else:
                        # Here we just assign all entries of the list to all entries of override_dict[k]
                        config[k].extend(override_dict[k])
                elif isinstance(config[k], dict):
                    # Clear out the existing entries because we are trying to replace everything
                    # Then we can simply update the dict with our new values
                    config[k].clear()
                    config[k].update(override_dict[k])
                elif isinstance(config[k], (int, float, bool)):
                    # This isn't really very good (since we lose information), but there's nothing that can be done
                    # about it at the moment (Dec 2018)
                    logger.debug("Overwriting YAML anchor object. It is currently unclear how to reassign this value.")
                    config[k] = v
                else:
                    # Raise a value error on all of the cases that we aren't already aware of.
                    raise ValueError(f"Object {k} (type {type(config[k])}) somehow has an anchor, but is something other than a list or dict. Attempting to assign directly to it.")
            except AttributeError:
                # If no anchor, just overwrite the value at this key
                config[k] = v
        else:
            raise KeyError(k, f"Trying to override key \"{k}\" that it is not in the config.")

    return config

def simplify_data_representations(config: DictLike) -> DictLike:
    """ Convert one entry lists to the scalar value

    This step is necessary because anchors are not kept for scalar values - just for lists and dictionaries.
    Now that we are done with all of our anchor references, we can convert these single entry lists to
    just the scalar entry, which is more usable.

    Some notes on anchors in ruamel.yaml are here: https://stackoverflow.com/a/48559644

    Args:
        config: The dict-like configuration from ruamel.yaml which should be simplified.
    Returns:
        The updated configuration.
    """
    for k, v in config.items():
        if v and isinstance(v, list) and len(v) == 1:
            logger.debug("v: {}".format(v))
            config[k] = v[0]

    return config

def determine_override_options(selected_options: tuple, override_opts: DictLike, set_of_possible_options: tuple = ()) -> Dict[str, Any]:
    """ Recursively extract the dict described in override_options().

    In particular, this searches for selected options in the override_opts dict. It stores only
    the override options that are selected.

    Args:
        selected_options: The options selected for this analysis, in the order defined used
            with ``override_options()`` and in the configuration file.
        override_opts: dict-like object returned by ruamel.yaml which contains the options that
            should be used to override the configuration options.
        set_of_possible_options (tuple of enums): Possible options for the override value categories.
    """
    override_dict: Dict[str, Any] = {}
    for option in override_opts:
        # We need to cast the option to a string to effectively compare to the selected option,
        # since only some of the options will already be strings
        if str(option) in list(map(lambda opt: str(opt), selected_options)):
            override_dict.update(determine_override_options(selected_options, override_opts[option], set_of_possible_options))
        else:
            logger.debug(f"override_opts: {override_opts}")
            # Look for whether the key is one of the possible but unselected options.
            # If so, we haven't selected it for this analysis, and therefore they should be ignored.
            # NOTE: We compare both the names and value because sometimes the name is not sufficient,
            #       such as in the case of the energy (because a number is not allowed to be a field name.)
            found_as_possible_option = False
            for possible_options in set_of_possible_options:
                # Same type of comparison as above, but for all possible options instead of the selected
                # options.
                if str(option) in list(map(lambda opt: str(opt), possible_options)):
                    found_as_possible_option = True
                # Below is more or less equivalent to the above (although .str() hides the details or
                # whether we should compare to the name or the value in the enum and only compares against
                # the designated value).
                #for possible_opt in possible_options:
                    #if possible_opt.name == option or possible_opt.value == option:
                    #    found_as_possible_option = True

            if not found_as_possible_option:
                # Store the override value, since it doesn't correspond with a selected option or a possible
                # option and therefore must be an option that we want to override.
                logger.debug(f"Storing override option \"{option}\", with value \"{override_opts[option]}\"")
                override_dict[option] = override_opts[option]
            else:
                logger.debug(f"Found option \"{option}\" as possible option, so skipping!")

    return override_dict

def determine_selection_of_iterable_values_from_config(config: DictLike, possible_iterables: Mapping[str, Type[enum.Enum]]) -> Dict[str, List[Any]]:
    """ Determine iterable values to use to create objects for a given configuration.

    All values of an iterable can be included be setting the value to ``True`` (Not as a single value list,
    but as the only value.). Alternatively, an iterator can be disabled by setting the value to ``False``.

    Args:
        config: The dict-like configuration from ruamel.yaml which should be overridden.
        possible_iterables: Key value pairs of names of enumerations and their values.
    Returns:
        dict: Iterables values that were requested in the config.
    """
    iterables = {}
    requested_iterables = config["iterables"]
    for k, v in requested_iterables.items():
        if k not in possible_iterables:
            raise KeyError(k, f"Cannot find requested iterable in possible_iterables: {possible_iterables}")
        logger.debug(f"k: {k}, v: {v}")
        additional_iterable: List[Any] = []
        enum_values = possible_iterables[k]
        # Check for a string. This is wrong, and the user should be notified.
        if isinstance(v, str):
            raise TypeError(type(v), f"Passed string {v} when must be either bool or list")
        # Allow the possibility to skip
        if v is False:
            continue
        # Allow the possibility to including all possible values in the enum.
        elif v is True:
            additional_iterable = list(enum_values)
        else:
            if enum_values is None:
                # The enumeration values are none, which means that we want to take
                # all of the values defined in the config.
                additional_iterable = list(v)
            else:
                # Otherwise, only take the requested values.
                for el in v:
                    additional_iterable.append(enum_values[el])
        # Store for later
        iterables[k] = additional_iterable

    return iterables

def _key_index_iter(self) -> Iterator[Tuple[str, Any]]:
    """ Allows for iteration over the ``KeyIndex`` values.

    This function is intended to be assigned to a newly created KeyIndex class. It enables iteration
    over the ``KeyIndex`` names and values. We don't use a mixin to avoid issues with YAML.

    Note:
        This isn't recursive like ``dataclasses.asdict(...)``. Generally, we don't want those recursive
        conversion properties. Plus, this approach is much faster.
    """
    for k, v in vars(self).items():
        yield k, v

def create_key_index_object(key_index_name: str, iterables: Dict[str, Any]) -> Any:
    """ Create a ``KeyIndex`` class based on the passed attributes.

    This is wrapped into a helper function to allow for the ``__itter__`` to be specified for the object.
    Further, this allows it to be called outside the package when it is needed in analysis tasks..

    Args:
        key_index_name: Name of the iterable key index.
        iterables: Iterables which will be specified by this ``KeyIndex``. The keys should be the names of
            the values, while the values should be the iterables themselves.
    Returns:
        A ``KeyIndex`` class which can be used to specify an object. The keys and values will be iterable.
    Raises:
        TypeError: If one of the iterables which is passed is an iterator that can be exhausted. The iterables
            must all be passed within containers which can recreate the iterator each time it is called to
            iterate.
    """
    # Validation
    # We are going to use the iterators when determining the fields, so we need to notify if an iterator was
    # passed, as this will cause a problem later. Instead of passing an iterator, a iterable should be passed,
    # which can recreate the iter.
    # See: https://effectivepython.com/2015/01/03/be-defensive-when-iterating-over-arguments/
    for name, iterable in iterables.items():
        if iter(iterable) == iter(iterable):
            raise TypeError(
                f"Iterable {name} is in iterator which can be exhausted. Please pass the iterable"
                f" in a container that can recreate the iterable. See the comments here for more info."
            )

    # We need the types of the fields to create the dataclass. However, we are provided with iterables
    # in the values of the iterables dict. Thus, we need to look at one value of each iterable, and use
    # that to determine the type of that particular iterable. This is safe to do because the iterables
    # must always have at least one entry (or else they wouldn't be one of the iterables).
    # NOTE: The order here matters when we create the ``KeyIndex`` later, so we cannot just take all
    #       objects from the iterables and blindly use set because set won't preserve the order.
    fields = [(name, type(next(iter(iterable)))) for name, iterable in iterables.items()]
    KeyIndex = dataclasses.make_dataclass(
        key_index_name,
        fields,
        frozen = True
    )
    # Allow for iteration over the key index values
    KeyIndex.__iter__ = _key_index_iter

    return KeyIndex

def create_objects_from_iterables(obj, args: dict, iterables: Dict[str, Any], formatting_options: Dict[str, Any], key_index_name: str = "KeyIndex") -> Tuple[Any, Dict[str, Any], dict]:
    """ Create objects for each set of values based on the given arguments.

    The iterable values are available under a key index ``dataclass`` which is used to index the returned
    dictionary. The names of the fields are determined by the keys of iterables dictionary. The values are
    the newly created object. Note that the iterable values must be convertible to a str() so they can be
    included in the formatting dictionary.

    Each set of values is also included in the object args.

    As a basic example,

    .. code-block:: python

        >>> create_objects_from_iterables(
        ...     obj = obj,
        ...     args = {},
        ...     iterables = {"a" : ["a1","a2"], "b" : ["b1", "b2"]},
        ...     formatting_options = {}
        ... )
        (
            KeyIndex,
            {"a": ["a1", "a2"], "b": ["b1", "b2"]}
            {
                KeyIndex(a = "a1", b = "b1"): obj(a = "a1", b = "b1"),
                KeyIndex(a = "a1", b = "b2"): obj(a = "a1", b = "b2"),
                KeyIndex(a = "a2", b = "b1"): obj(a = "a2", b = "b1"),
                KeyIndex(a = "a2", b = "b2"): obj(a = "a2", b = "b2"),
            }
        )

    Args:
        obj (object): The object to be constructed.
        args: Arguments to be passed to the object to create it.
        iterables: Iterables to be used to create the objects, with entries of the form
            ``"name_of_iterable": iterable``.
        formatting_options: Values to be used in formatting strings in the arguments.
        key_index_name: Name of the iterable key index.
    Returns:
        (object, list, dict, dict): Roughly, (KeyIndex, iterables, objects). Specifically, the
            key_index is a new dataclass which defines the parameters used to create the object, iterables
            are the iterables used to create the objects, which names as keys and the iterables as values.
            The objects dictionary keys are KeyIndex objects which describe the iterable arguments passed to the
            object, while the values are the newly constructed arguments. See the example above.
    """
    # Setup
    objects = {}
    names = list(iterables)
    logger.debug(f"iterables: {iterables}")
    # Create the key index object, where the name of each field is the name of each iterable.
    KeyIndex = create_key_index_object(
        key_index_name = key_index_name,
        iterables = iterables,
    )
    # ``itertools.product`` produces all possible permutations of the iterables values.
    # NOTE: Product preserves the order of the iterables values, which is important for properly
    #       assigning the values to the ``KeyIndex``.
    for values in itertools.product(*iterables.values()):
        logger.debug(f"Values: {values}")
        # Skip if we don't have a sufficient set of values to create an object.
        if not values:
            continue

        # Add in the values into the arguments and formatting options.
        # NOTE: We don't need a deep copy for the iterable values in the args and formatting options
        #       because the values will be overwritten for each object.
        for name, val in zip(names, values):
            # We want to keep the original value for the arguments.
            args[name] = val
            # Here, we convert the value, regardless of type, into a string that can be displayed.
            formatting_options[name] = str(val)

        # Apply formatting options
        # If we formatted in place, we would need to deepcopy the args to ensure that the iterable dependent
        # values in the formatted values are properly set for each iterable object individually.
        # However, by formatting into new variables, we can avoid a deepcopy, which greatly improves performance!
        # NOTE: We don't need a deep copy do this for iterable value names themselves because they will be overwritten
        #       for each object. They are set in the block above.
        object_args = copy.copy(args)
        logger.debug(f"object_args pre format: {object_args}")
        object_args = apply_formatting_dict(object_args, formatting_options)
        # Print our results for debugging purposes. However, we skip printing the full
        # config because it is quite long
        print_args = {k: v for k, v in object_args.items() if k != "config"}
        print_args["config"] = "..."
        logger.debug(f"Constructing obj \"{obj}\" with args: \"{print_args}\"")

        # Finally create the object.
        objects[KeyIndex(*values)] = obj(**object_args)

    # If nothing has been created at this point, then we are didn't iterating over anything and something
    # has gone wrong.
    if not objects:
        raise ValueError(iterables, "There appear to be no iterables to use in creating objects.")

    return (KeyIndex, iterables, objects)

class formatting_dict(dict):
    """ Dict to handle missing keys when formatting a string.

    It returns the missing key for later use in formatting. See: https://stackoverflow.com/a/17215533
    """
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"

def apply_formatting_dict(obj: Any, formatting: Dict[str, Any]) -> Any:
    """ Recursively apply a formatting dict to all strings in a configuration.

    Note that it skips applying the formatting if the string appears to contain latex (specifically,
    if it contains an "$"), since the formatting fails on nested brackets.

    Args:
        obj: Some configuration object to recursively applying the formatting to.
        formatting (dict): String formatting options to apply to each configuration field.
    Returns:
        dict: Configuration with formatting applied to every field.
    """
    #logger.debug("Processing object of type {}".format(type(obj)))
    new_obj = obj

    if isinstance(obj, str):
        # Apply the formatting options to the string.
        # We explicitly allow for missing keys. They will be kept so they can be filled later.
        # see: https://stackoverflow.com/a/17215533
        # If a more sophisticated solution is needed,
        # see: https://ashwch.github.io/handling-missing-keys-in-str-format-map.html
        # Note that we can't use format_map because it is python 3.2+ only.
        # The solution below works in py 2/3
        if "$" not in obj:
            new_obj = string.Formatter().vformat(obj, (), formatting_dict(**formatting))
        #else:
        #    logger.debug("Skipping str {} since it appears to be a latex string, which may break the formatting.".format(obj))
    elif isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            # Using indirect access to ensure that the original object is updated.
            new_obj[k] = apply_formatting_dict(v, formatting)
    elif isinstance(obj, list):
        new_obj = []
        for i, el in enumerate(obj):
            # Using indirect access to ensure that the original object is updated.
            new_obj.append(apply_formatting_dict(el, formatting))
    elif isinstance(obj, int) or isinstance(obj, float) or obj is None:
        # Skip over this, as there is nothing to be done - we just keep the value.
        pass
    elif isinstance(obj, enum.Enum):
        # Skip over this, as there is nothing to be done - we just keep the value.
        # This only occurs when a formatting value has already been transformed
        # into an enumeration.
        pass
    else:
        # This may or may not be expected, depending on the particular value.
        logger.debug(f"Unrecognized obj '{obj}' of type '{type(obj)}'")

    return new_obj

def iterate_with_selected_objects(analysis_objects: Mapping[Any, Any], **selections: Mapping[str, Any]) -> Iterator[Tuple[Any, Any]]:
    """ Iterate over an analysis dictionary with selected attributes.

    Args:
        analysis_objects: Analysis objects dictionary.
        selections: Keyword arguments used to select attributes from the analysis dictionary.
    Yields:
        object: Matching analysis object.
    """
    for key_index, obj in analysis_objects.items():
        # If selections is empty, we return every object. If it's not empty, then we only want to return
        # objects which are selected in through the selections.
        selected_obj = not selections or all([getattr(key_index, selector) == selected_value for selector, selected_value in selections.items()])

        if selected_obj:
            yield key_index, obj

def iterate_with_selected_objects_in_order(analysis_objects: Mapping[Any, Any],
                                           analysis_iterables: Dict[str, Sequence[Any]],
                                           selection: Union[str, Sequence[str]]) -> Iterator[List[Tuple[Any, Any]]]:
    """ Iterate over an analysis dictionary, yielding the selected attributes in order.

    So if there are three iterables, a, b, and c, if we selected c, then we iterate over a and b,
    and return c in the same order each time for each set of values of a and b. As an example, consider
    the set of iterables:

    .. code-block:: python

        >>> a = ["a1", "a2"]
        >>> b = ["b1", "b2"]
        >>> c = ["c1", "c2"]

    then it will effectively return:

    .. code-block:: python

        >>> for a_val in a:
        ...     for b_val in b:
        ...         for c_val in c:
        ...             obj(a_val, b_val, c_val)

    This will yield:

    .. code-block:: python

        >>> output = list(iterate_with_selected_objects_in_order(..., selection = ["a"]))
        [[("a1", "b1", "c1"), ("a2", "b1", "c1")], [("a1", "b2", "c1"), ("a2", "b2", "c1")], ...]

    This is particularly nice because we can then select on a set of iterables to be returned without
    having to specify the rest of the iterables that we don't really care about.

    Args:
        analysis_objects: Analysis objects dictionary.
        analysis_iterables: Iterables used in constructing the analysis objects.
        selection: Selection of analysis selections to return. Can be either a string or a sequence of
            selections.
    Yields:
        object: Matching analysis object.
    """
    # Validation
    if isinstance(selection, str):
        selection = [selection]
    # Help out mypy. We don't check if it is a list to allow for other sequences.
    assert not isinstance(selection, str)
    # We don't want to impact the original analysis iterables when we pop some values below.
    analysis_iterables = copy.copy(analysis_iterables)

    # Extract the selected iterators from the possible iterators so we can select on them later.
    # First, we want want each set of iterators to be of the form:
    # {"selection1": [value1, value2, ...], "selection2": [value3, value4, ...]}
    selected_iterators = {}
    for s in selection:
        selected_iterators[s] = analysis_iterables.pop(s)

    logger.debug(f"Initial analysis_iterables: {analysis_iterables}")
    logger.debug(f"Initial selected_iterators: {selected_iterators}")

    # Now, we convert them to the form:
    # [[("selection1", value1), ("selection1", value2)], [("selection2", value3), ("selection2", value4)]]
    # This allows them to iterated over conveniently via itertools.product(...)
    selected_iterators = [[(k, v) for v in values] for k, values in selected_iterators.items()]  # type: ignore
    analysis_iterables = [[(k, v) for v in values] for k, values in analysis_iterables.items()]  # type: ignore

    logger.debug(f"Final analysis_iterables: {analysis_iterables}")
    logger.debug(f"Final selected_iterators: {selected_iterators}")
    # Useful debug information, but too verbose for standard usage.
    #logger.debug(f"analysis_iterables product: {list(itertools.product(*analysis_iterables))}")
    #logger.debug(f"selected_iterators product: {list(itertools.product(*selected_iterators))}")

    for values in itertools.product(*analysis_iterables):
        selected_analysis_objects = []
        for selected_values in itertools.product(*selected_iterators):
            for key_index, obj in analysis_objects.items():
                selected_via_analysis_iterables = all(
                    getattr(key_index, k) == v for k, v in values
                )
                selected_via_selected_iterators = all(
                    getattr(key_index, k) == v for k, v in selected_values
                )
                selected_obj = selected_via_analysis_iterables and selected_via_selected_iterators

                if selected_obj:
                    selected_analysis_objects.append((key_index, obj))

        logger.debug(f"Yielding: {selected_analysis_objects}")
        yield selected_analysis_objects

