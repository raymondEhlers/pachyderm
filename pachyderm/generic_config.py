#!/usr/bin/env python

""" Analysis configuration base module.

For usage information, see ``jet_hadron.base.analysisConfig``.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import collections
import copy
import dataclasses
import enum
import itertools
import logging
import string
import ruamel.yaml
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

def loadConfiguration(filename):
    """ Load an analysis configuration from a file.

    Args:
        filename (str): Filename of the YAML configuration file.
    Returns:
        dict-like: dict-like object containing the loaded configuration
    """
    # Initialize the YAML object in the round trip mode
    # NOTE: "typ" is a not a typo. It stands for "type"
    yaml = ruamel.yaml.YAML(typ = "rt")

    with open(filename, "r") as f:
        config = yaml.load(f)

    return config

def overrideOptions(config, selectedOptions, setOfPossibleOptions, configContainingOverride = None):
    """ Determine override options for a particular configuration.

    The options are determined by searching following the order specified in selectedOptions.

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
    selectedOptions passed. The above example configuration is from the jet-hadron analysis.

    Since anchors aren't kept for scalar values, if you want to override an anchored value, you need to
    specify it as a single value in a list (or dict, but list is easier). After the anchor values propagate,
    single element lists can be converted into scalar values using ``simplifyDataRepresentations()``.

    Args:
        config (CommentedMap): The dict-like configuration from ruamel.yaml which should be overridden.
        selectedOptions (tuple): The selected analysis options. They will be checked in the order with which
            they are passed, so make certain that it matches the order in the configuration file!
        setOfPossibleOptions (tuple of enums): Possible options for the override value categories.
        configContainingOverride (CommentedMap): The dict-like config containing the override options in a map called
            "override". If it is not specified, it will look for it in the main config.
    Returns:
        dict-like object: The updated configuration
    """
    if configContainingOverride is None:
        configContainingOverride = config
    override_opts = configContainingOverride.pop("override")
    overrideDict = determineOverrideOptions(selectedOptions, override_opts, setOfPossibleOptions)
    logger.debug(f"overrideDict: {overrideDict}")

    # Set the configuration values to those specified in the override options
    # Cannot just use update() on config because we need to maintain the anchors.
    for k, v in overrideDict.items():
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
                    if isinstance(overrideDict[k], str):
                        # We have to treat str carefully because it is an iterable, but it will be expanded as
                        # individual characters if it's treated the same as a list, which is not the desired
                        # behavior! If we wrap it in [], then it will be treated as the only entry in the list
                        config[k].append(overrideDict[k])
                    else:
                        # Here we just assign all entries of the list to all entries of overrideDict[k]
                        config[k].extend(overrideDict[k])
                elif isinstance(config[k], dict):
                    # Clear out the existing entries because we are trying to replace everything
                    # Then we can simply update the dict with our new values
                    config[k].clear()
                    config[k].update(overrideDict[k])
            except AttributeError:
                # If no anchor, just overwrite the value at this key
                config[k] = v
        else:
            raise KeyError(k, f"Trying to override key \"{k}\" that it is not in the config.")

    return config

def simplifyDataRepresentations(config):
    """ Convert one entry lists to the scalar value

    This step is necessary because anchors are not kept for scalar values - just for lists and dictionaries.
    Now that we are done with all of our anchor references, we can convert these single entry lists to
    just the scalar entry, which is more usable.

    Some notes on anchors in ruamel.yaml are here: https://stackoverflow.com/a/48559644

    Args:
        config (CommentedMap): The dict-like configuration from ruamel.yaml which should be simplified.
    Returns:
        dict-like object: The updated configuration
    """
    for k, v in config.items():
        if v and isinstance(v, list) and len(v) == 1:
            logger.debug("v: {}".format(v))
            config[k] = v[0]

    return config

def determineOverrideOptions(selectedOptions, override_opts, setOfPossibleOptions = ()):
    """ Recursively extract the dict described in overrideOptions().

    In particular, this searches for selected options in the override_opts dict.
    It stores only the override options that are selected.

    Args:
        selectedOptions (tuple): The options selected for this analysis, in the order defined used
            with overrideOptions() and in the configuration file.
        override_opts (CommentedMap): dict-like object returned by ruamel.yaml which contains the options that
            should be used to override the configuration options.
        setOfPossibleOptions (tuple of enums): Possible options for the override value categories.
    """
    overrideDict = {}
    for option in override_opts:
        # We need to cast the option to a string to effectively compare to the selected option,
        # since only some of the options will already be strings
        if str(option) in list(map(lambda opt: opt.str(), selectedOptions)):
            overrideDict.update(determineOverrideOptions(selectedOptions, override_opts[option], setOfPossibleOptions))
        else:
            logger.debug(f"override_opts: {override_opts}")
            # Look for whether the key is one of the possible but unselected options.
            # If so, we haven't selected it for this analysis, and therefore they should be ignored.
            # NOTE: We compare both the names and value because sometimes the name is not sufficient,
            #       such as in the case of the energy (because a number is not allowed to be a field name.)
            foundAsPossibleOption = False
            for possibleOptions in setOfPossibleOptions:
                # Same type of comparison as above, but for all possible options instead of the selected options.
                if str(option) in list(map(lambda opt: opt.str(), possibleOptions)):
                    foundAsPossibleOption = True
                # Below is more or less equivalent to the above (although .str() hides the details or whether
                # we should compare to the name or the value in the enum and only compares against the designated value).
                #for possibleOpt in possibleOptions:
                    #if possibleOpt.name == option or possibleOpt.value == option:
                    #    foundAsPossibleOption = True

            if not foundAsPossibleOption:
                # Store the override value, since it doesn't correspond with a selected option or a possible option
                # and therefore must be an option that we want to override.
                logger.debug(f"Storing override option \"{option}\", with value \"{override_opts[option]}\"")
                overrideDict[option] = override_opts[option]
            else:
                logger.debug(f"Found option \"{option}\" as possible option, so skipping!")

    return overrideDict

def determineSelectionOfIterableValuesFromConfig(config, possibleIterables):
    """ Determine iterable values to use to create objects for a given configuration.

    All values of an iterable can be included be setting the value to ``True`` (Not as a single value list,
    but as the only value.). Alternatively, an iterator can be disabled by setting the value to ``False``.

    Args:
        config (CommentedMap): The dict-like configuration from ruamel.yaml which should be overridden.
        possibleIterables (collections.OrderedDict): Key value pairs of names of enumerations and their values.
    Returns:
        collections.OrderedDict: Iterables values that were requested in the config.
    """
    iterables = collections.OrderedDict()
    requestedIterables = config["iterables"]
    for k, v in requestedIterables.items():
        if k not in possibleIterables:
            raise KeyError(k, "Cannot find requested iterable in possibleIterables: {possibleIterables}".format(possibleIterables = possibleIterables))
        logger.debug("k: {}, v: {}".format(k, v))
        additionalIterable = []
        enum_values = possibleIterables[k]
        # Check for a string. This is wrong, and the user should be notified.
        if isinstance(v, str):
            raise TypeError(type(v), "Passed string {v} when must be either bool or list".format(v = v))
        # Allow the possibility to skip
        if v is False:
            continue
        # Allow the possibility to including all possible values in the enum.
        elif v is True:
            additionalIterable = list(enum_values)
        else:
            # Otherwise, only take the requested values.
            for el in v:
                additionalIterable.append(enum_values[el])
        # Store for later
        iterables[k] = additionalIterable

    return iterables

def create_objects_from_iterables(obj, args: dict, iterables: dict, formatting_options: dict, key_index_name: str = "KeyIndex") -> Tuple[object, List[str], dict]:
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
            ``"nameOfIterable": iterable``.
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
        object_args = applyFormattingDict(object_args, formatting_options)
        # Print our results for debugging purposes. However, we skip printing the full
        # config because it is quite long
        printArgs = {k: v for k, v in object_args.items() if k != "config"}
        printArgs["config"] = "..."
        logger.debug(f"Constructing obj \"{obj}\" with args: \"{printArgs}\"")

        # Finally create the object.
        objects[KeyIndex(*values)] = obj(**object_args)

    # If nothing has been created at this point, then we are didn't iterating over anything and something
    # has gone wrong.
    if not objects:
        raise ValueError(iterables, "There appear to be no iterables to use in creating objects.")

    return (KeyIndex, names, objects)

class formattingDict(dict):
    """ Dict to handle missing keys when formatting a string. It returns the missing key
    for later use in formatting. See: https://stackoverflow.com/a/17215533 """
    def __missing__(self, key):
        return "{" + key + "}"

def applyFormattingDict(obj, formatting):
    """ Recursively apply a formatting dict to all strings in a configuration.

    Note that it skips applying the formatting if the string appears to contain latex (specifically,
    if it contains an "$"), since the formatting fails on nested brackets.

    Args:
        obj (dict): Some configuration object to recursively applying the formatting to.
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
            obj = string.Formatter().vformat(obj, (), formattingDict(**formatting))
        #else:
        #    logger.debug("Skipping str {} since it appears to be a latex string, which may break the formatting.".format(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            # Using indirect access to ensure that the original object is updated.
            obj[k] = applyFormattingDict(v, formatting)
    elif isinstance(obj, list):
        for i, el in enumerate(obj):
            # Using indirect access to ensure that the original object is updated.
            obj[i] = applyFormattingDict(el, formatting)
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

