""" Tests for generic class properties.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any

import pytest

from pachyderm import generic_class

logger = logging.getLogger(__name__)


@pytest.fixture()
def setup_equality_mixin() -> tuple[Any, Any]:
    """Create a basic class for tests of the equality mixin."""

    class EqualityMixinTestClass(generic_class.EqualityMixin):
        def __init__(self, a_number: float, a_string: str, a_list: list[Any], a_dict: dict[str, Any]):
            self.a_number = a_number
            self.a_string = a_string
            self.a_list = a_list
            self.a_dict = a_dict

    # Define some test values. We want them to be complicated enough
    # that we can test comparison of all of the relevant types.
    a_number = 10.3
    a_string = "hello world"
    a_list = [1, 2, 3, {"hello": "world"}]
    a_dict = {"string": "string", "list": [1, 2, 3], "dict": {"hello": "world"}}

    test_class = EqualityMixinTestClass(a_number, a_string, a_list, a_dict)
    expected_class = EqualityMixinTestClass(a_number, a_string, a_list, a_dict)

    return (test_class, expected_class)


def test_equality_mixin(setup_equality_mixin: Any) -> None:
    """Test the equality mixin with the same classes."""
    test_class, expected_class = setup_equality_mixin

    # Check basic assertions
    assert test_class is test_class  # noqa: PLR0124
    assert test_class == test_class  # noqa: PLR0124
    # Check against an identical instance of the same class.
    assert test_class == expected_class
    assert not test_class != expected_class  # noqa: SIM202

    # Modify the test class to make the classes unequal.
    # (We will work through a simple shift of the elements one member forward).
    # (I would do this with a parameterization, but I don't see any straightforward
    # way to do it, so this will be fine)
    test_class.a_number = expected_class.a_dict
    assert test_class != expected_class
    assert not test_class == expected_class  # noqa: SIM201

    test_class.a_string = expected_class.a_number
    assert test_class != expected_class
    assert not test_class == expected_class  # noqa: SIM201

    test_class.a_list = expected_class.a_string
    assert test_class != expected_class
    assert not test_class == expected_class  # noqa: SIM201

    test_class.a_dict = expected_class.a_list
    assert test_class != expected_class
    assert not test_class == expected_class  # noqa: SIM201

    # Restore the changes so they can be used later (just to be certain)
    test_class.a_number = expected_class.a_number
    test_class.a_string = expected_class.a_string
    test_class.a_list = expected_class.a_list
    test_class.a_dict = expected_class.a_dict


def test_equality_mixin_against_other_classes(setup_equality_mixin: Any) -> None:
    """Test the quality mixin against other classes, for which comparisons are not implemented."""
    test_class, expected_class = setup_equality_mixin

    # Create a dataclass object to compare against.
    TestClass = dataclasses.make_dataclass("TestClass", ["hello", "world"])
    another_object = TestClass(hello="hello", world="world")

    # Can't catch NotImplemented, as it's a special type of raised value
    # that isn't handled the same way as other raised exceptions.
    # Instead, we just perform the assertions to cover tests against different objects.
    assert not test_class == another_object  # noqa: SIM201
    assert test_class != another_object
