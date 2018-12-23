#!/usr/bin/env python

""" Tests for generic class properties.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import dataclasses
import enum
import logging
import pytest
import ruamel.yaml
import tempfile

from pachyderm import generic_class

logger = logging.getLogger(__name__)

@pytest.fixture
def setup_equality_mixin():
    """ Create a basic class for tests of the equality mixin. """

    class EqualityMixinTestClass(generic_class.EqualityMixin):
        def __init__(self, aNumber, aString, aList, aDict):
            self.aNumber = aNumber
            self.aString = aString
            self.aList = aList
            self.aDict = aDict

    # Define some test values. We want them to be complciated enough
    # that we can test comparion of all of the relevant types.
    aNumber = 10.3
    aString = "hello world"
    aList = [1, 2, 3, {"hello": "world"}],
    aDict = {"string": "string", "list": [1, 2, 3], "dict": {"hello": "world"}}

    test_class = EqualityMixinTestClass(aNumber, aString, aList, aDict)
    expected_class = EqualityMixinTestClass(aNumber, aString, aList, aDict)

    return (test_class, expected_class)

def test_equality_mixin(logging_mixin, setup_equality_mixin):
    """ Test the equality mixin with the same classes. """
    test_class, expected_class = setup_equality_mixin

    # Check basic assertions
    assert test_class is test_class
    assert test_class == test_class
    # Check against an identical instance of the same class.
    assert test_class == expected_class
    assert not test_class != expected_class

    # Modify the test class to make the classes unequal.
    # (We will work through a simple shift of the elements one member forward).
    # (I would do this with a paramterization, but I don't see any straightforward
    # way to do it, so this will be fine)
    test_class.aNumber = expected_class.aDict
    assert test_class != expected_class
    assert not test_class == expected_class

    test_class.aString = expected_class.aNumber
    assert test_class != expected_class
    assert not test_class == expected_class

    test_class.aList = expected_class.aString
    assert test_class != expected_class
    assert not test_class == expected_class

    test_class.aDict = expected_class.aList
    assert test_class != expected_class
    assert not test_class == expected_class

    # Restore the changes so they can be used later (just to be certain)
    test_class.aNumber = expected_class.aNumber
    test_class.aString = expected_class.aString
    test_class.aList = expected_class.aList
    test_class.aDict = expected_class.aDict

def test_equality_mixin_against_other_classes(logging_mixin, setup_equality_mixin):
    """ Test the quality mixin against other classes, for which comparions are not implemented. """
    test_class, expected_class = setup_equality_mixin

    # Create a named tuple object to compare against.
    TestClass = dataclasses.make_dataclass("TestClass", ["hello", "world"])
    another_object = TestClass(hello = "hello", world = "world")

    # Can't catch NotImplemented, as it's a special type of raised value
    # that isn't handled the same way as other raised exceptions.
    # Instead, we just perform the assertions to cover tests against different objects.
    assert not test_class == another_object
    assert test_class != another_object

@pytest.fixture
def setup_enum_with_yaml(logging_mixin):
    """ Setup for testing reading and writing enum values to YAML. """
    # Test enumeration
    class TestEnum(enum.Enum):
        a = 1
        b = 2

        def __str__(self):
            return str(self.name)

        to_yaml = classmethod(generic_class.enum_to_yaml)
        from_yaml = classmethod(generic_class.enum_from_yaml)

    yaml = ruamel.yaml.YAML(typ = "rt")
    yaml.register_class(TestEnum)

    return TestEnum, yaml

def test_enum_with_yaml(setup_enum_with_yaml, logging_mixin):
    """ Test closure of reading and writing enum values to YAML. """
    # Setup
    TestEnum, yaml = setup_enum_with_yaml
    input_value = TestEnum.a

    # Read and write to a temp file for convenience.
    with tempfile.TemporaryFile() as f:
        # Dump the value to the file
        yaml.dump([input_value], f)
        # Then load it back.
        f.seek(0)
        result = yaml.load(f)

    assert result == [input_value]

