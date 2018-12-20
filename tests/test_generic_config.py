#!/usr/bin/env python

""" Tests for generic analysis configuration.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import dataclasses
import enum
import logging
import pytest
from io import StringIO
import ruamel.yaml

from pachyderm import generic_config

logger = logging.getLogger(__name__)

def log_yaml_dump(yaml, config):
    """ Helper function to log the YAML config. """
    s = StringIO()
    yaml.dump(config, s)
    s.seek(0)
    logger.debug(s)

@pytest.fixture
def basic_config():
    """ Basic YAML configuration to test overriding the configuration.

    See the config for which selected options are implemented.

    Args:
        None
    Returns:
        tuple: (dict-like CommentedMap object from ruamel.yaml containing the configuration, str containing
            a string representation of the YAML configuration)
    """
    test_yaml = """
responseTasks: &responseTasks
    responseMaker: &responseMakerTaskName "AliJetResponseMaker_{cent}histos"
    jetHPerformance: &jetHPerformanceTaskName ""
responseTaskName: &responseTaskName [""]
pythiaInfoAfterEventSelectionTaskName: *responseTaskName
# Demonstrate that anchors are preserved
test1: &test1
- val1
- val2
test2: *test1
# Test overrid values
test3: &test3 ["test3"]
test4: *test3
testList: [1, 2]
testDict:
    1: 2
override:
    responseTaskName: *responseMakerTaskName
    test3: "test6"
    testList: [3, 4]
    testDict:
        3: 4
    """

    yaml = ruamel.yaml.YAML()
    data = yaml.load(test_yaml)

    return (data, test_yaml)

def basic_config_exception(data):
    """ Add an unmatched key (ie does not exist in the main config) to the override
    map to cause an exception.

    Note that this assumes that "test_exception" does not exist in the main configuration!

    Args:
        data (CommentedMap): dict-like object containing the configuration
    Returns:
        CommentedMap: dict-like object containing an unmatched entry in the override map.
    """
    data["override"]["test_exception"] = "value"
    return data

def override_data(config):
    """ Helper function to override the configuration.

    It can print the configuration before and after overridding the options if enabled.

    Args:
        config (CommentedMap): dict-like object containing the configuration to be overridden.
    Returns:
        CommentedMap: dict-like object containing the overridden configuration
    """
    yaml = ruamel.yaml.YAML()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Before override:")
        log_yaml_dump(yaml, config)

    # Override and simplify the values
    config = generic_config.override_options(config, (), ())
    config = generic_config.simplify_data_representations(config)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("After override:")
        log_yaml_dump(yaml, config)

    return config

def test_override_retrieve_unrelated_value(logging_mixin, basic_config):
    """ Test retrieving a basic value unrelated to the overridden data. """
    (basic_config, yaml_string) = basic_config

    value_name = "test1"
    value_before_override = basic_config[value_name]
    basic_config = override_data(basic_config)

    assert basic_config[value_name] == value_before_override

def test_override_with_basic_config(logging_mixin, basic_config):
    """ Test override with the basic config.  """
    (basic_config, yaml_string) = basic_config
    basic_config = override_data(basic_config)

    # This value is overridden directly
    assert basic_config["test3"] == "test6"

def test_basic_anchor_override(logging_mixin, basic_config):
    """ Test overriding with an anchor.

    When an anchor refernce is overridden, we expect that the anchor value is updated.
    """
    (basic_config, yaml_string) = basic_config
    basic_config = override_data(basic_config)

    # The two conditions below are redundant, but each are useful for visualizing
    # different configuration circumstances, so both are kept.
    assert basic_config["responseTaskName"] == "AliJetResponseMaker_{cent}histos"
    assert basic_config["test4"] == "test6"

def test_advanced_anchor_override(logging_mixin, basic_config):
    """ Test overriding a anchored value with another anchor.

    When an override value is using an anchor value, we expect that value to propagate fully.
    """
    (basic_config, yaml_string) = basic_config
    basic_config = override_data(basic_config)

    # This value is overridden indirectly, from another referenced value.
    assert basic_config["responseTaskName"] == basic_config["pythiaInfoAfterEventSelectionTaskName"]

def test_for_unmatched_keys(logging_mixin, basic_config):
    """ Test for an unmatched key in the override field (ie without a match in the config).

    Such an unmatched key should cause a `KeyError` exception, which we catch.
    """
    (basic_config, yaml_string) = basic_config
    # Add entry that will cause the exception.
    basic_config = basic_config_exception(basic_config)

    # Test fails if it _doesn't_ throw an exception.
    with pytest.raises(KeyError) as exception_info:
        basic_config = override_data(basic_config)
    # This is the value that we expected to fail.
    assert exception_info.value.args[0] == "test_exception"

def test_complex_object_override(logging_mixin, basic_config):
    """ Test override with complex objects.

    In particular, test with lists, dicts.
    """
    (basic_config, yaml_string) = basic_config
    basic_config = override_data(basic_config)

    assert basic_config["testList"] == [3, 4]
    assert basic_config["testDict"] == {3: 4}

def test_load_configuration(logging_mixin, basic_config):
    """ Test that loading yaml goes according to expectations. This may be somewhat trivial, but it
    is still important to check in case ruamel.yaml changes APIs or defaults.

    NOTE: We can only compare at the YAML level because the dumped string does not preserve anchors that
          are not actually referenced, as well as some trivial variation in quote types and other similarly
          trivial formatting issues.
    """
    (basic_config, yaml_string) = basic_config
    classes_to_register = [dataclasses.make_dataclass("TestClass", ["hello", "world"])]
    yaml_string += """
hello:
    - !TestClass
      hello: "str"
      world: "str2" """
    basic_config["hello"] = [classes_to_register[0](hello = "str", world = "str2")]

    import tempfile
    with tempfile.NamedTemporaryFile() as f:
        # Write and move back to the start of the file
        f.write(yaml_string.encode())
        f.seek(0)
        # Then get the config from the file
        retrieved_config = generic_config.load_configuration(f.name, classes_to_register = classes_to_register)

    assert retrieved_config == basic_config

    # NOTE: Not utilized due to the note above
    # Use yaml.dump() to dump the configuration to a string.
    #yaml = ruamel.yaml.YAML(typ = "rt")
    #with tempfile.NamedTemporaryFile() as f:
    #    yaml.dump(retrieved_config, f)
    #    f.seek(0)
    #    # Save as a standard string. Need to decode from bytes
    #    retrieved_string = f.read().decode()
    #assert retrieved_string == yaml_string

@pytest.fixture
def data_simplification_config():
    """ Simple YAML config to test the data simplification functionality of the generic_config module.

    It povides example configurations entries for numbers, str, list, and dict.

    Args:
        None
    Returns:
        CommentedMap: dict-like object from ruamel.yaml containing the configuration.
    """

    test_yaml = """
int: 3
float: 3.14
str: "hello"
singleEntryList: [ "hello" ]
multiEntryList: [ "hello", "world" ]
singleEntryDict:
    hello: "world"
multiEntryDict:
    hello: "world"
    foo: "bar"
"""
    yaml = ruamel.yaml.YAML()
    data = yaml.load(test_yaml)

    return data

def test_data_simplification_on_base_types(logging_mixin, data_simplification_config):
    """ Test the data simplification function on base types.

    Here we tests int, float, and str.  They should always stay the same.
    """
    config = generic_config.simplify_data_representations(data_simplification_config)

    assert config["int"] == 3
    assert config["float"] == 3.14
    assert config["str"] == "hello"

def test_data_simplification_on_lists(logging_mixin, data_simplification_config):
    """ Test the data simplification function on lists.

    A single entry list should be returned as a string, while a multiple entry list should be
    preserved as is.
    """
    config = generic_config.simplify_data_representations(data_simplification_config)

    assert config["singleEntryList"] == "hello"
    assert config["multiEntryList"] == ["hello", "world"]

def test_dict_data_simplification(logging_mixin, data_simplification_config):
    """ Test the data simplification function on dicts.

    Dicts should always maintain their structure.
    """
    config = generic_config.simplify_data_representations(data_simplification_config)

    assert config["singleEntryDict"] == {"hello": "world"}
    assert config["multiEntryDict"] == {"hello": "world", "foo": "bar"}

class reaction_plane_orientation(enum.Enum):
    """ Example enumeration for testing. This represents RP orientation. """
    inPlane = 0
    midPlane = 1
    outOfPlane = 2
    all = 3

class qvector(enum.Enum):
    """ Example enumeration for testing. This represents the q vector. """
    all = 0
    bottom10 = 1
    top10 = 2

class collision_energy(enum.Enum):
    """ Example enumeration for testing. This represents collision system energies. """
    twoSevenSix = 2.76
    fiveZeroTwo = 5.02

@pytest.fixture
def object_creation_config():
    """ Configuration to test creating objects based on the stored values. """
    config = """
iterables:
    reaction_plane_orientation:
        - inPlane
        - midPlane
    qVector: True
    collisionEnergy: False
"""
    yaml = ruamel.yaml.YAML()
    config = yaml.load(config)

    possible_iterables = {}
    possible_iterables["reaction_plane_orientation"] = reaction_plane_orientation
    possible_iterables["qVector"] = qvector
    possible_iterables["collisionEnergy"] = collision_energy

    return (config, possible_iterables, ([reaction_plane_orientation.inPlane, reaction_plane_orientation.midPlane], list(qvector)))

def test_determine_selection_of_iterable_values_from_config(logging_mixin, object_creation_config):
    """ Test determining which values of an iterable to use. """
    (config, possible_iterables, (reaction_plane_orientations, qvectors)) = object_creation_config
    iterables = generic_config.determine_selection_of_iterable_values_from_config(
        config = config,
        possible_iterables = possible_iterables
    )

    assert iterables["reaction_plane_orientation"] == reaction_plane_orientations
    assert iterables["qVector"] == qvectors
    # Collision Energy should _not_ be included! It was only a possible iterator.
    # Check in two ways.
    assert "collisionEnergy" not in iterables
    assert len(iterables) == 2

def test_determine_selection_of_iterable_values_with_undefined_iterable(logging_mixin, object_creation_config):
    """ Test determining which values of an iterable to use when an iterable is not defined. """
    (config, possible_iterables, (reaction_plane_orientations, qvectors)) = object_creation_config

    del possible_iterables["qVector"]
    with pytest.raises(KeyError) as exception_info:
        generic_config.determine_selection_of_iterable_values_from_config(
            config = config,
            possible_iterables = possible_iterables
        )
    assert exception_info.value.args[0] == "qVector"

def test_determine_selection_of_iterable_values_with_string_selection(logging_mixin, object_creation_config):
    """ Test trying to determine values with a string.

    This is not allowed, so it should raise an exception.
    """
    (config, possible_iterables, (reaction_plane_orientations, qvectors)) = object_creation_config

    config["iterables"]["qVector"] = "True"
    with pytest.raises(TypeError) as exception_info:
        generic_config.determine_selection_of_iterable_values_from_config(
            config = config,
            possible_iterables = possible_iterables
        )
    assert exception_info.value.args[0] is str

@pytest.fixture
def object_and_creation_args():
    """ Create the object and args for object creation. """
    # Define fake object. We don't use a mock because we need to instantiate the object
    # in the function that is being tested. This is not super straightforward with mock,
    # so instead we create a test object by hand.
    obj = dataclasses.make_dataclass("TestObj", ["reaction_plane_orientation", "qVector", "a", "b", "options_fmt"])
    # Include args that depend on the iterable values to ensure that they are varied properly!
    args = {"a": 1, "b": "{fmt}", "options_fmt": "{reaction_plane_orientation}_{qVector}"}
    formatting_options = {"fmt": "formatted", "options_fmt": "{reaction_plane_orientation}_{qVector}"}

    return (obj, args, formatting_options)

def test_create_objects_from_iterables(logging_mixin, object_creation_config, object_and_creation_args):
    """ Test object creation from a set of iterables. """
    # Collect variables
    (config, possible_iterables, (reaction_plane_orientations, qvectors)) = object_creation_config
    (obj, args, formatting_options) = object_and_creation_args

    # Get iterables
    iterables = generic_config.determine_selection_of_iterable_values_from_config(
        config = config,
        possible_iterables = possible_iterables
    )

    # Create the objects.
    (key_index, names, objects) = generic_config.create_objects_from_iterables(
        obj = obj,
        args = args,
        iterables = iterables,
        formatting_options = formatting_options,
        key_index_name = "KeyIndex",
    )

    # Check the names of the iterables.
    assert names == list(iterables)
    # Check the precise values passed to the object.
    for rp_angle in reaction_plane_orientations:
        for qVector in qvectors:
            created_object = objects[key_index(reaction_plane_orientation = rp_angle, qVector = qVector)]
            assert created_object.reaction_plane_orientation == rp_angle
            assert created_object.qVector == qVector
            assert created_object.a == args["a"]
            assert created_object.b == formatting_options["fmt"]
            assert created_object.options_fmt == formatting_options["options_fmt"].format(reaction_plane_orientation = rp_angle, qVector = qVector)

def test_missing_iterable_for_object_creation(logging_mixin, object_and_creation_args):
    """ Test object creation when the iterables are missing. """
    (obj, args, formatting_options) = object_and_creation_args
    # Create empty iterables for this test.
    iterables = {}

    # Create the objects.
    with pytest.raises(ValueError) as exception_info:
        (names, objects) = generic_config.create_objects_from_iterables(
            obj = obj,
            args = args,
            iterables = iterables,
            formatting_options = formatting_options
        )
    assert exception_info.value.args[0] == iterables

@pytest.fixture
def formatting_config():
    """ Config for testing the formatting of strings after loading them.

    Returns:
        tuple: (Config with formatting applied, formatting dict)
    """
    config = r"""
int: 3
float: 3.14
noFormat: "test"
format: "{a}"
noFormatBecauseNoFormatter: "{noFormatHere}"
list:
    - "noFormat"
    - 2
    - "{a}{c}"
dict:
    noFormat: "hello"
    format: "{a}{c}"
dict2:
    dict:
        str: "do nothing"
        format: "{c}"
latexLike: $latex_{like \mathrm{x}}$
noneExample: null
"""
    yaml = ruamel.yaml.YAML()
    config = yaml.load(config)

    formatting = {"a": "b", "c": 1}

    return (generic_config.apply_formatting_dict(config, formatting), formatting)

def test_apply_formatting_to_basic_types(logging_mixin, formatting_config):
    """ Test applying formatting to basic types. """
    config, formatting_dict = formatting_config

    assert config["int"] == 3
    assert config["float"] == 3.14
    assert config["noFormat"] == "test"
    assert config["format"] == formatting_dict["a"]
    assert config["noFormatBecauseNoFormatter"] == "{noFormatHere}"

def test_apply_formatting_to_iterable_types(logging_mixin, formatting_config):
    """ Test applying formatting to iterable types. """
    config, formatting_dict = formatting_config

    assert config["list"] == ["noFormat", 2, "b1"]
    assert config["dict"] == {"noFormat": "hello", "format": "{}{}".format(formatting_dict["a"], formatting_dict["c"])}
    # NOTE: The extra str() call is because the formated string needs to be compared against a str.
    assert config["dict2"]["dict"] == {"str": "do nothing", "format": str(formatting_dict["c"])}

def test_apply_formatting_skip_latex(logging_mixin, formatting_config):
    """ Test skipping the application of the formatting to strings which look like latex. """
    config, formatting_dict = formatting_config

    assert config["latexLike"] == r"$latex_{like \mathrm{x}}$"

@pytest.fixture
def setup_analysis_iterator(logging_mixin):
    """ Setup for testing iteration over analysis objects. """
    KeyIndex = dataclasses.make_dataclass("KeyIndex", ["a", "b", "c"], frozen = True)
    test_dict = {
        KeyIndex(a = "a1", b = "b1", c = "c"): "obj1",
        KeyIndex(a = "a1", b = "b2", c = "c"): "obj2",
        KeyIndex(a = "a2", b = "b1", c = "c"): "obj3",
        KeyIndex(a = "a2", b = "b2", c = "c"): "obj4",
    }

    return KeyIndex, test_dict

def test_iterate_with_no_selected_items(setup_analysis_iterator):
    """ Test iterating over analysis objects without any selection. """
    KeyIndex, test_dict = setup_analysis_iterator

    # Create the iterator
    object_iter = generic_config.iterate_with_selected_objects(
        analysis_objects = test_dict,
    )

    # Iterate over it.
    assert next(object_iter) == (KeyIndex(a = "a1", b = "b1", c = "c"), "obj1")
    assert next(object_iter) == (KeyIndex(a = "a1", b = "b2", c = "c"), "obj2")
    assert next(object_iter) == (KeyIndex(a = "a2", b = "b1", c = "c"), "obj3")
    assert next(object_iter) == (KeyIndex(a = "a2", b = "b2", c = "c"), "obj4")
    # It should be exhausted now.
    with pytest.raises(StopIteration):
        next(object_iter)

def test_iterate_with_selected_items(setup_analysis_iterator):
    """ Test iterating over analysis objects with a selection. """
    # Setup
    KeyIndex, test_dict = setup_analysis_iterator

    # Create the iterator
    object_iter = generic_config.iterate_with_selected_objects(
        analysis_objects = test_dict,
        a = "a1",
    )

    # Iterate over it.
    assert next(object_iter) == (KeyIndex(a = "a1", b = "b1", c = "c"), "obj1")
    assert next(object_iter) == (KeyIndex(a = "a1", b = "b2", c = "c"), "obj2")
    # It should be exhausted now.
    with pytest.raises(StopIteration):
        next(object_iter)

def test_iterate_with_multiple_selected_items(setup_analysis_iterator):
    """ Test iterating over analysis objects with multiple selections. """
    # Setup
    KeyIndex, test_dict = setup_analysis_iterator

    # Create the iterator
    object_iter = generic_config.iterate_with_selected_objects(
        analysis_objects = test_dict,
        a = "a1",
        b = "b2",
    )

    # Iterate over it.
    assert next(object_iter) == (KeyIndex(a = "a1", b = "b2", c = "c"), "obj2")
    # It should be exhausted now.
    with pytest.raises(StopIteration):
        next(object_iter)

