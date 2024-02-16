""" Tests for generic analysis configuration.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""
from __future__ import annotations

import copy
import dataclasses
import enum
import itertools
import logging
from io import StringIO
from typing import Any

import pytest
import ruamel.yaml

from pachyderm import generic_config, yaml

logger = logging.getLogger(__name__)


def log_yaml_dump(yml: Any, config: dict[str, Any]) -> None:
    """Helper function to log the YAML config."""
    s = StringIO()
    yml.dump(config, s)
    s.seek(0)
    logger.debug(s.read())


@pytest.fixture()
def basic_config():
    """Basic YAML configuration to test overriding the configuration.

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
# Test override values
test3: &test3 ["test3"]
test4: *test3
# Basic int override
testInt: 1
# Basic test for when int has anchors
testIntAnchor: &testInt 10
testIntAnchorValue: *testInt
# Test for int override indirection
testIntRef: &testIntIndirection [3]
testIntAnchorIndirection: *testIntIndirection
# Basic float override (will apparently use ScalarFloat)
testFloat: 1.234
# Basic test for when float has anchors
testFloatAnchor: &testFloat 2.3456
testFloatAnchorValue: *testFloat
# Test for float override indirection
testFloatRef: &testFloatAnchor [3.1]
testFloatAnchorIndirection: *testFloatAnchor
# bool
testBool: false
testList: [1, 2]
testDict:
    1: 2
override:
    responseTaskName: *responseMakerTaskName
    test3: "test6"
    testInt: 2
    #testIntAnchor: 12
    testIntRef: 4
    testFloat: 2.71
    testFloatRef: 3.14
    testBool: true
    testList: [3, 4]
    testDict:
        3: 4
    """

    yaml = ruamel.yaml.YAML()
    data = yaml.load(test_yaml)

    return (data, test_yaml)


def basic_config_exception(data: dict[str, Any]) -> dict[str, Any]:
    """Add an unmatched key (ie does not exist in the main config) to the override
    map to cause an exception.

    Note that this assumes that "test_exception" does not exist in the main configuration!

    Args:
        data (CommentedMap): dict-like object containing the configuration
    Returns:
        CommentedMap: dict-like object containing an unmatched entry in the override map.
    """
    data["override"]["test_exception"] = "value"
    return data


def override_data(config: dict[str, Any]) -> dict[str, Any]:
    """Helper function to override the configuration.

    It can print the configuration before and after overriding the options if enabled.

    Args:
        config (CommentedMap): dict-like object containing the configuration to be overridden.
    Returns:
        CommentedMap: dict-like object containing the overridden configuration
    """
    yml = ruamel.yaml.YAML()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Before override:")
        log_yaml_dump(yml, config)

    # Override and simplify the values
    config = generic_config.override_options(config, (), ())  # type: ignore[arg-type]
    config = generic_config.simplify_data_representations(config)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("After override:")
        log_yaml_dump(yml, config)

    return config


def test_override_retrieve_unrelated_value(basic_config):
    """Test retrieving a basic value unrelated to the overridden data."""
    (basic_config, yaml_string) = basic_config

    value_name = "test1"
    value_before_override = basic_config[value_name]
    basic_config = override_data(basic_config)

    assert basic_config[value_name] == value_before_override


def test_override_with_basic_config(basic_config):
    """Test override with the basic config."""
    (basic_config, yaml_string) = basic_config
    basic_config = override_data(basic_config)

    # This value is overridden directly
    assert basic_config["test3"] == "test6"


def test_basic_anchor_override(basic_config):
    """Test overriding with an anchor.

    When an anchor reference is overridden, we expect that the anchor value is updated.
    """
    (basic_config, yaml_string) = basic_config
    basic_config = override_data(basic_config)

    # Test all of the basic types. It is important to tests str, float, int, bool, etc
    # because they are handled specially by ruamel.yaml to preserve anchors, etc.
    # The two conditions below are redundant, but each are useful for visualizing
    # different configuration circumstances, so both are kept.
    assert basic_config["responseTaskName"] == "AliJetResponseMaker_{cent}histos"
    assert basic_config["test4"] == "test6"
    # Test basic types
    assert basic_config["testInt"] == 2
    assert basic_config["testFloat"] == 2.71
    assert basic_config["testBool"] is True
    # Test anchor indirection for basic types
    assert basic_config["testIntAnchorValue"] == 10
    assert basic_config["testIntRef"] == 4
    assert basic_config["testIntAnchorIndirection"] == 4
    assert basic_config["testFloatAnchorValue"] == 2.3456
    assert basic_config["testFloatRef"] == 3.14
    assert basic_config["testFloatAnchorIndirection"] == 3.14


def test_advanced_anchor_override(basic_config):
    """Test overriding a anchored value with another anchor.

    When an override value is using an anchor value, we expect that value to propagate fully.
    """
    (basic_config, yaml_string) = basic_config
    basic_config = override_data(basic_config)

    # This value is overridden indirectly, from another referenced value.
    assert basic_config["responseTaskName"] == basic_config["pythiaInfoAfterEventSelectionTaskName"]


def test_for_unmatched_keys(basic_config):
    """Test for an unmatched key in the override field (ie without a match in the config).

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


def test_complex_object_override(basic_config):
    """Test override with complex objects.

    In particular, test with lists, dicts.
    """
    (basic_config, yaml_string) = basic_config
    basic_config = override_data(basic_config)

    assert basic_config["testList"] == [3, 4]
    assert basic_config["testDict"] == {3: 4}


def test_load_configuration(basic_config):
    """Test that loading yaml goes according to expectations. This may be somewhat trivial, but it
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
    basic_config["hello"] = [classes_to_register[0](hello="str", world="str2")]
    yml = yaml.yaml(classes_to_register=classes_to_register)

    import tempfile

    with tempfile.NamedTemporaryFile() as f:
        # Write and move back to the start of the file
        f.write(yaml_string.encode())
        f.seek(0)
        # Then get the config from the file
        retrieved_config = generic_config.load_configuration(yaml=yml, filename=f.name)

    assert retrieved_config == basic_config

    # NOTE: Not utilized due to the note above
    # Use yaml.dump() to dump the configuration to a string.
    # yaml = ruamel.yaml.YAML(typ = "rt")
    # with tempfile.NamedTemporaryFile() as f:
    #    yaml.dump(retrieved_config, f)
    #    f.seek(0)
    #    # Save as a standard string. Need to decode from bytes
    #    retrieved_string = f.read().decode()
    # assert retrieved_string == yaml_string


@pytest.fixture()
def data_simplification_config():
    """Simple YAML config to test the data simplification functionality of the generic_config module.

    It provides example configurations entries for numbers, str, list, and dict.

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
    yaml = ruamel.yaml.YAML(typ="rt")
    return yaml.load(test_yaml)


def test_data_simplification_on_base_types(data_simplification_config):
    """Test the data simplification function on base types.

    Here we tests int, float, and str.  They should always stay the same.
    """
    config = generic_config.simplify_data_representations(data_simplification_config)

    assert config["int"] == 3
    assert config["float"] == 3.14
    assert config["str"] == "hello"


def test_data_simplification_on_lists(data_simplification_config):
    """Test the data simplification function on lists.

    A single entry list should be returned as a string, while a multiple entry list should be
    preserved as is.
    """
    config = generic_config.simplify_data_representations(data_simplification_config)

    assert config["singleEntryList"] == "hello"
    assert config["multiEntryList"] == ["hello", "world"]


def test_dict_data_simplification(data_simplification_config):
    """Test the data simplification function on dicts.

    Dicts should always maintain their structure.
    """
    config = generic_config.simplify_data_representations(data_simplification_config)

    assert config["singleEntryDict"] == {"hello": "world"}
    assert config["multiEntryDict"] == {"hello": "world", "foo": "bar"}


class reaction_plane_orientation(enum.Enum):
    """Example enumeration for testing. This represents RP orientation."""

    inPlane = 0
    midPlane = 1
    outOfPlane = 2
    all = 3


class qvector(enum.Enum):
    """Example enumeration for testing. This represents the q vector."""

    all = 0
    bottom10 = 1
    top10 = 2


class collision_energy(enum.Enum):
    """Example enumeration for testing. This represents collision system energies."""

    twoSevenSix = 2.76
    fiveZeroTwo = 5.02


@pytest.fixture()
def object_creation_config():
    """Configuration to test creating objects based on the stored values."""
    config = """
iterables:
    reaction_plane_orientation:
        - inPlane
        - midPlane
    qVector: True
    collisionEnergy: False
"""
    yaml = ruamel.yaml.YAML(typ="rt")
    config = yaml.load(config)

    possible_iterables: dict[str, Any] = {}
    possible_iterables["reaction_plane_orientation"] = reaction_plane_orientation
    possible_iterables["qVector"] = qvector
    possible_iterables["collisionEnergy"] = collision_energy

    return (
        config,
        possible_iterables,
        ([reaction_plane_orientation.inPlane, reaction_plane_orientation.midPlane], list(qvector)),
    )


def test_determine_selection_of_iterable_values_from_config(object_creation_config):
    """Test determining which values of an iterable to use."""
    (config, possible_iterables, (reaction_plane_orientations, qvectors)) = object_creation_config
    iterables = generic_config.determine_selection_of_iterable_values_from_config(
        config=config, possible_iterables=possible_iterables
    )

    assert iterables["reaction_plane_orientation"] == reaction_plane_orientations
    assert iterables["qVector"] == qvectors
    # Collision Energy should _not_ be included! It was only a possible iterator.
    # Check in two ways.
    assert "collisionEnergy" not in iterables
    assert len(iterables) == 2


def test_determine_selection_of_iterable_values_from_config_list_of_values(object_creation_config):
    """Test creating objects from lists of values in a configuration file."""

    # Create fake object needed for using lists of objects.
    @dataclasses.dataclass
    class TestIterable:
        a: int
        b: int

    # Setup
    (config, possible_iterables, (reaction_plane_orientations, qvectors)) = object_creation_config
    list_of_values = [
        TestIterable(a=1, b=2),
        TestIterable(a=2, b=3),
    ]
    # Create set of values and allow it as a possible iterable
    config["iterables"]["test_iterable_values"] = copy.copy(list_of_values)
    possible_iterables["test_iterable_values"] = None
    # Actually create the objects
    iterables = generic_config.determine_selection_of_iterable_values_from_config(
        config=config, possible_iterables=possible_iterables
    )

    # Check the iterables
    assert iterables["reaction_plane_orientation"] == reaction_plane_orientations
    assert iterables["qVector"] == qvectors
    assert iterables["test_iterable_values"] == list_of_values
    assert len(iterables) == 3


def test_determine_selection_of_iterable_values_with_undefined_iterable(object_creation_config):
    """Test determining which values of an iterable to use when an iterable is not defined."""
    (config, possible_iterables, (reaction_plane_orientations, qvectors)) = object_creation_config

    del possible_iterables["qVector"]
    with pytest.raises(KeyError) as exception_info:
        generic_config.determine_selection_of_iterable_values_from_config(
            config=config, possible_iterables=possible_iterables
        )
    assert exception_info.value.args[0] == "qVector"


def test_determine_selection_of_iterable_values_with_string_selection(object_creation_config):
    """Test trying to determine values with a string.

    This is not allowed, so it should raise an exception.
    """
    (config, possible_iterables, (reaction_plane_orientations, qvectors)) = object_creation_config

    config["iterables"]["qVector"] = "True"
    with pytest.raises(TypeError) as exception_info:
        generic_config.determine_selection_of_iterable_values_from_config(
            config=config, possible_iterables=possible_iterables
        )
    assert exception_info.value.args[0] is str


@pytest.fixture()
def object_and_creation_args():
    """Create the object and args for object creation."""
    # Define fake object. We don't use a mock because we need to instantiate the object
    # in the function that is being tested. This is not super straightforward with mock,
    # so instead we create a test object by hand.
    obj = dataclasses.make_dataclass(
        "TestObj", ["reaction_plane_orientation", "qVector", "a", "b", "options_fmt", "nested_fmt"]
    )
    # Include args that depend on the iterable values to ensure that they are varied properly!
    # "nested_fmt" is needed to show when deepcopy is necessary.
    args = {
        "a": 1,
        "b": "{fmt}",
        "options_fmt": "{reaction_plane_orientation}_{qVector}",
        "nested_fmt": [[{"val": "{qVector}_{reaction_plane_orientation}"}]],
    }
    formatting_options = {"fmt": "formatted", "options_fmt": "{reaction_plane_orientation}_{qVector}"}

    return (obj, args, formatting_options)


def test_create_objects_from_iterables(object_creation_config, object_and_creation_args):
    """Test object creation from a set of iterables."""
    # Collect variables
    (config, possible_iterables, (reaction_plane_orientations, qvectors)) = object_creation_config
    (obj, args, formatting_options) = object_and_creation_args

    # Get iterables
    iterables = generic_config.determine_selection_of_iterable_values_from_config(
        config=config, possible_iterables=possible_iterables
    )

    # Create the objects.
    (key_index, returned_iterables, objects) = generic_config.create_objects_from_iterables(
        obj=obj,
        args=args,
        iterables=iterables,
        formatting_options=formatting_options,
        key_index_name="KeyIndex",
    )

    # Check the names of the iterables.
    assert list(returned_iterables) == list(iterables)
    # Check the precise values passed to the object.
    for rp_angle in reaction_plane_orientations:
        for qVector in qvectors:
            expected_key_index = key_index(reaction_plane_orientation=rp_angle, qVector=qVector)
            created_object = objects[expected_key_index]
            assert created_object.reaction_plane_orientation == rp_angle
            assert created_object.qVector == qVector
            assert created_object.a == args["a"]
            assert created_object.b == formatting_options["fmt"]
            logger.debug(f"options_fmt: {created_object.options_fmt}")
            assert created_object.options_fmt == formatting_options["options_fmt"].format(
                reaction_plane_orientation=rp_angle, qVector=qVector
            )
            assert created_object.nested_fmt == [[{"val": f"{qVector}_{rp_angle}"}]]

            # Check that the KeyIndex iterators work
            found_key_index = False
            for obj_key_index in objects:
                if obj_key_index == expected_key_index:
                    found_key_index = True
                    expected_key_index_values = {
                        "reaction_plane_orientation": rp_angle,
                        "qVector": qVector,
                    }
                    for (k, v), (expected_k, expected_v) in itertools.zip_longest(
                        obj_key_index, expected_key_index_values.items()
                    ):
                        assert k == expected_k
                        assert v == expected_v

            assert found_key_index is True


def test_missing_iterable_for_object_creation(object_and_creation_args):
    """Test object creation when the iterables are missing."""
    (obj, args, formatting_options) = object_and_creation_args
    # Create empty iterables for this test.
    iterables: dict[str, Any] = {}

    # Create the objects.
    with pytest.raises(ValueError, match="no iterables") as exception_info:
        generic_config.create_objects_from_iterables(
            obj=obj, args=args, iterables=iterables, formatting_options=formatting_options
        )
    assert exception_info.value.args[0] == iterables


@pytest.fixture()
def formatting_config():
    """Config for testing the formatting of strings after loading them.

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
    yaml = ruamel.yaml.YAML(typ="rt")
    config = yaml.load(config)

    formatting = {"a": "b", "c": 1}

    return (generic_config.apply_formatting_dict(config, formatting), formatting)


def test_apply_formatting_to_basic_types(formatting_config):
    """Test applying formatting to basic types."""
    config, formatting_dict = formatting_config

    assert config["int"] == 3
    assert config["float"] == 3.14
    assert config["noFormat"] == "test"
    assert config["format"] == formatting_dict["a"]
    assert config["noFormatBecauseNoFormatter"] == "{noFormatHere}"


def test_apply_formatting_to_iterable_types(formatting_config):
    """Test applying formatting to iterable types."""
    config, formatting_dict = formatting_config

    assert config["list"] == ["noFormat", 2, "b1"]
    assert config["dict"] == {"noFormat": "hello", "format": "{}{}".format(formatting_dict["a"], formatting_dict["c"])}
    # NOTE: The extra str() call is because the formatted string needs to be compared against a str.
    assert config["dict2"]["dict"] == {"str": "do nothing", "format": str(formatting_dict["c"])}


def test_apply_formatting_skip_latex(formatting_config):
    """Test skipping the application of the formatting to strings which look like latex."""
    config, formatting_dict = formatting_config

    assert config["latexLike"] == r"$latex_{like \mathrm{x}}$"


@pytest.fixture()
def setup_analysis_iterator():
    """Setup for testing iteration over analysis objects."""
    KeyIndex = dataclasses.make_dataclass("KeyIndex", ["a", "b", "c"], frozen=True)
    KeyIndex.__iter__ = generic_config._key_index_iter  # type: ignore[attr-defined]
    analysis_iterables = {"a": ["a1", "a2"], "b": ["b1", "b2"], "c": ["c"]}
    test_dict = {
        KeyIndex(a="a1", b="b1", c="c"): "obj1",
        KeyIndex(a="a1", b="b2", c="c"): "obj2",
        KeyIndex(a="a2", b="b1", c="c"): "obj3",
        KeyIndex(a="a2", b="b2", c="c"): "obj4",
    }

    return KeyIndex, analysis_iterables, test_dict


def test_key_index_iterator(setup_analysis_iterator):
    """Test the iterator over the KeyIndex values."""
    KeyIndex, _, _ = setup_analysis_iterator

    # The values are randomly selected.
    kwargs = {"a": "12", "b": 3, "c": "Hello"}
    key_index = KeyIndex(**kwargs)

    # Check the entire dict. It should be in order, so this should be fine, and it helps
    # ensure that we check every key of the dict.
    iter_values = dict(key_index)
    assert kwargs == iter_values

    # We also check explicit iteration just for good measure.
    for (k, v), (k_expected, v_expected) in itertools.zip_longest(key_index, kwargs.items()):
        assert k == k_expected
        assert v == v_expected


def test_iterate_with_no_selected_items(setup_analysis_iterator):
    """Test iterating over analysis objects without any selection."""
    KeyIndex, _, test_dict = setup_analysis_iterator

    # Create the iterator
    object_iter = generic_config.iterate_with_selected_objects(
        analysis_objects=test_dict,
    )

    # Iterate over it.
    assert next(object_iter) == (KeyIndex(a="a1", b="b1", c="c"), "obj1")
    assert next(object_iter) == (KeyIndex(a="a1", b="b2", c="c"), "obj2")
    assert next(object_iter) == (KeyIndex(a="a2", b="b1", c="c"), "obj3")
    assert next(object_iter) == (KeyIndex(a="a2", b="b2", c="c"), "obj4")
    # It should be exhausted now.
    with pytest.raises(StopIteration):
        next(object_iter)


def test_iterate_with_selected_items(setup_analysis_iterator):
    """Test iterating over analysis objects with a selection."""
    # Setup
    KeyIndex, _, test_dict = setup_analysis_iterator

    # Create the iterator
    object_iter = generic_config.iterate_with_selected_objects(
        analysis_objects=test_dict,
        a="a1",
    )

    # Iterate over it.
    assert next(object_iter) == (KeyIndex(a="a1", b="b1", c="c"), "obj1")
    assert next(object_iter) == (KeyIndex(a="a1", b="b2", c="c"), "obj2")
    # It should be exhausted now.
    with pytest.raises(StopIteration):
        next(object_iter)


def test_iterate_with_multiple_selected_items(setup_analysis_iterator):
    """Test iterating over analysis objects with multiple selections."""
    # Setup
    KeyIndex, _, test_dict = setup_analysis_iterator

    # Create the iterator
    object_iter = generic_config.iterate_with_selected_objects(
        analysis_objects=test_dict,
        a="a1",
        b="b2",
    )

    # Iterate over it.
    assert next(object_iter) == (KeyIndex(a="a1", b="b2", c="c"), "obj2")
    # It should be exhausted now.
    with pytest.raises(StopIteration):
        next(object_iter)


@pytest.mark.parametrize(
    "selection",
    [
        "a",
        ["a"],
    ],
    ids=["Single selection", "List of selections"],
)
def test_iterate_with_selected_objects_in_order(setup_analysis_iterator, selection):
    """Test iterating over analysis objects with non-selected attributes."""
    # Setup
    KeyIndex, analysis_iterables, test_dict = setup_analysis_iterator

    object_iter = generic_config.iterate_with_selected_objects_in_order(
        analysis_objects=test_dict,
        analysis_iterables=analysis_iterables,
        selection=selection,
    )

    # Expected output should be in two groups. Both are ordered in "a".
    expected_output = [
        [(KeyIndex(a="a1", b="b1", c="c"), "obj1"), (KeyIndex(a="a2", b="b1", c="c"), "obj3")],
        [(KeyIndex(a="a1", b="b2", c="c"), "obj2"), (KeyIndex(a="a2", b="b2", c="c"), "obj4")],
    ]

    # Iterate over it.
    for value, expected in zip(object_iter, expected_output, strict=False):
        assert value == expected

    # It should be exhausted now.
    with pytest.raises(StopIteration):
        next(object_iter)
