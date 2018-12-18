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
import ruamel.yaml
from typing import Any, Dict, List, Tuple, Type

logger = logging.getLogger(__name__)
# Make it a bit easier to specify the CommentedMap type.
DictLike = Type[ruamel.yaml.comments.CommentedMap]

def load_configuration(filename: str) -> DictLike:
    """ Load an analysis configuration from a file.

    Args:
        filename: Filename of the YAML configuration file.
    Returns:
        dict-like object containing the loaded configuration
    """
    # Initialize the YAML object in the round trip mode
    # NOTE: "typ" is a not a typo. It stands for "type"
    yaml = ruamel.yaml.YAML(typ = "rt")

    with open(filename, "r") as f:
        config = yaml.load(f)

    return config

def override_options(config: DictLike, selected_options: tuple, set_of_possible_options: tuple, config_containing_override: DictLike = None) -> DictLike:
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
                # We can't check for the anchor - we just have to try to access it.
                # However, we don't actually care about the value. We just want to
                # preserve it if it is exists.
                config[k].anchor
                logger.debug("type: {}, k: {}".format(type(config[k]), k))
                if isinstance(config[k], list):
                    # Clear out the existing list entries
                    del config[k][:]
                    if isinstance(override_dict[k], str):
                        # We have to treat str carefully because it is an iterable, but it will be expanded as
                        # individual characters if it's treated the same as a list, which is not the desired
                        # behavior! If we wrap it in [], then it will be treated as the only entry in the list
                        config[k].append(override_dict[k])
                    else:
                        # Here we just assign all entries of the list to all entries of override_dict[k]
                        config[k].extend(override_dict[k])
                elif isinstance(config[k], dict):
                    # Clear out the existing entries because we are trying to replace everything
                    # Then we can simply update the dict with our new values
                    config[k].clear()
                    config[k].update(override_dict[k])
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

def determine_override_options(selected_options: tuple, override_opts: DictLike, set_of_possible_options: tuple = ()):
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

def determine_selection_of_iterable_values_from_config(config: DictLike, possible_iterables: Dict[str, Type[enum.Enum]]) -> Dict[str, List[Any]]:
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
            raise TypeError(type(v), "Passed string {v} when must be either bool or list".format(v = v))
        # Allow the possibility to skip
        if v is False:
            continue
        # Allow the possibility to including all possible values in the enum.
        elif v is True:
            additional_iterable = list(enum_values)
        else:
            # Otherwise, only take the requested values.
            for el in v:
                additional_iterable.append(enum_values[el])
        # Store for later
        iterables[k] = additional_iterable

    return iterables

def create_objects_from_iterables(obj, args: dict, iterables: dict, formatting_options: dict, key_index_name: str = "KeyIndex") -> Tuple[Any, List[str], dict]:
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
            ["a", "b"],
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
        formatting_options (dict): Values to be used in formatting strings in the arguments.
        key_obj_name (str): Name of the iterable key object.
    Returns:
        (object, list, dict): Roughly, (KeyIndex, names, objects). Specifically, the key_index is a
            new dataclass which defines the parameters used to create the object, names is the names
            of the iterables used. The dictionary keys are KeyIndex objects which describe the iterable
            arguments passed to the object, while the values are the newly constructed arguments. See the
            example above.
    """
    # Setup
    objects = {}
    names = list(iterables)
    logger.debug("iterables: {iterables}".format(iterables = iterables))
    # Create the key index object, where the name of each field is the name of each iterable.
    KeyIndex = dataclasses.make_dataclass(
        key_index_name,
        [(name, type(iterable)) for name, iterable in iterables.items()],
        frozen = True
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
        # Need a deep copy to ensure that the iterable dependent values in the formatting are
        # properly set for each object individually.
        # NOTE: We don't need a deep copy do this for iterable value names because they will be overwritten
        #       for each object. See above.
        object_args = copy.deepcopy(args)
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

    return (KeyIndex, names, objects)

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

    if isinstance(obj, str):
        # Apply the formatting options to the string.
        # We explicitly allow for missing keys. They will be kept so they can be filled later.
        # see: https://stackoverflow.com/a/17215533
        # If a more sophisticated solution is needed,
        # see: https://ashwch.github.io/handling-missing-keys-in-str-format-map.html
        # Note that we can't use format_map because it is python 3.2+ only.
        # The solution below works in py 2/3
        if "$" not in obj:
            obj = string.Formatter().vformat(obj, (), formatting_dict(**formatting))
        #else:
        #    logger.debug("Skipping str {} since it appears to be a latex string, which may break the formatting.".format(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            # Using indirect access to ensure that the original object is updated.
            obj[k] = apply_formatting_dict(v, formatting)
    elif isinstance(obj, list):
        for i, el in enumerate(obj):
            # Using indirect access to ensure that the original object is updated.
            obj[i] = apply_formatting_dict(el, formatting)
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
        logger.info("NOTE: Unrecognized type {} of obj {}".format(type(obj), obj))

    return obj

def iterate_with_selected_objects(analysis_objects: Dict[Any, Any], **selections: Dict[str, Any]) -> Any:
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

