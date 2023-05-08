#!/usr/bin/env python

""" Tests for generic class properties.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

import dataclasses
import logging
from typing import Any, Dict, List, Tuple

import pytest

from pachyderm import generic_class

logger = logging.getLogger(__name__)

@pytest.fixture  # type: ignore
def setup_equality_mixin() -> Tuple[Any, Any]:
    """ Create a basic class for tests of the equality mixin. """

    class EqualityMixinTestClass(generic_class.EqualityMixin):
        def __init__(self, aNumber: float, aString: str, aList: List[Any], aDict: Dict[str, Any]):
            self.aNumber = aNumber
            self.aString = aString
            self.aList = aList
            self.aDict = aDict

    # Define some test values. We want them to be complicated enough
    # that we can test comparison of all of the relevant types.
    aNumber = 10.3
    aString = "hello world"
    aList = [1, 2, 3, {"hello": "world"}]
    aDict = {"string": "string", "list": [1, 2, 3], "dict": {"hello": "world"}}

    test_class = EqualityMixinTestClass(aNumber, aString, aList, aDict)
    expected_class = EqualityMixinTestClass(aNumber, aString, aList, aDict)

    return (test_class, expected_class)

def test_equality_mixin(logging_mixin: Any, setup_equality_mixin: Any) -> None:
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
    # (I would do this with a parameterization, but I don't see any straightforward
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

def test_equality_mixin_against_other_classes(logging_mixin: Any, setup_equality_mixin: Any) -> None:
    """ Test the quality mixin against other classes, for which comparisons are not implemented. """
    test_class, expected_class = setup_equality_mixin

    # Create a dataclass object to compare against.
    TestClass = dataclasses.make_dataclass("TestClass", ["hello", "world"])
    another_object = TestClass(hello = "hello", world = "world")

    # Can't catch NotImplemented, as it's a special type of raised value
    # that isn't handled the same way as other raised exceptions.
    # Instead, we just perform the assertions to cover tests against different objects.
    assert not test_class == another_object
    assert test_class != another_object
