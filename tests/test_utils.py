""" Tests for the utilities module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from pachyderm import utils

# Setup logger
logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        (
            (3, np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])),
            np.array([6, 9, 12, 13, 12, 9, 6]),
        ),
        ((4, np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])), np.array([10, 14, 16, 16, 14, 10])),
        ((3, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])), np.array([6, 9, 12, 15, 18, 21, 24, 27])),
        ((3, np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])), np.array([27, 24, 21, 18, 15, 12, 9, 6])),
    ],
    ids=["n = 3 triangular values", "n = 4 triangular values", "n = 3 increasing values", "n = 3 decreasing values"],
)
def test_moving_average(inputs, expected):
    """Test the moving average calculation."""
    (n, arr) = inputs
    expected = expected / n
    assert np.array_equal(utils.moving_average(arr=arr, n=n), expected)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("standard_attr", "standard_attr_value"),
        ("attr1.attr3.my_attr", "recursive_attr_value"),
    ],
    ids=["Standard attribute", "Recursive attribute"],
)
def test_recursive_getattr(mocker, path, expected):
    """Tests for recursive getattr."""
    # Setup mock objects from which we will recursively grab attributes
    mock_obj1 = mocker.MagicMock(spec=["standard_attr", "attr1"])
    mock_obj2 = mocker.MagicMock(spec=["attr3"])
    mock_obj3 = mocker.MagicMock(spec=["my_attr"])
    mock_obj1.standard_attr = "standard_attr_value"
    mock_obj1.attr1 = mock_obj2
    mock_obj2.attr3 = mock_obj3
    mock_obj3.my_attr = "recursive_attr_value"

    # For convenience
    obj = mock_obj1
    # Check the returned value
    assert expected == utils.recursive_getattr(obj, path)


def test_recursive_getattr_default_value(mocker: Any) -> None:
    """Test for retrieving a default value with getattr."""
    obj = mocker.MagicMock(spec=["sole_attr"])
    assert utils.recursive_getattr(obj, "nonexistent_attr", "default_value") == "default_value"


def test_recursive_getattr_fail(mocker: Any) -> None:
    """Test for failure of recursive getattr.

    It will fail the same was as the standard getattr.
    """
    obj = mocker.MagicMock(spec=["sole_attr"])

    with pytest.raises(AttributeError) as exception_info:
        utils.recursive_getattr(obj, "nonexistent_attr")
    assert "nonexistent_attr" in exception_info.value.args[0]


@pytest.fixture()
def setup_recursive_setattr() -> tuple[Any, str]:
    """Setup an object for testing the recursive setattr."""

    # We don't mock the objects because I'm not entirely sure how mock will interact with setattr.
    @dataclass
    class SecondLevel:
        attribute2: Any

    @dataclass
    class FirstLevel:
        attribute1: SecondLevel

    # We want to return both the object and the relevant information to set the value
    obj = FirstLevel(SecondLevel("initial value"))
    path = "attribute1.attribute2"

    return obj, path


def test_recursive_setattr(setup_recursive_setattr: Any) -> None:
    """Test setting an attribute with recursive setattr."""
    # Setup
    obj, path = setup_recursive_setattr

    # Set the attribute and check the result
    new_value = "new value"
    utils.recursive_setattr(obj, path, new_value)
    assert utils.recursive_getattr(obj, path) == new_value


def test_recursive_setattr_fail(setup_recursive_setattr: Any) -> None:
    """Test failing to set an attribute with recursive setattr."""
    # Setup
    obj, path = setup_recursive_setattr

    # Use a random path, which should fail.
    with pytest.raises(AttributeError) as exception_info:
        utils.recursive_setattr(obj, "random.fake.path", "fake value")
    # It will only return that "random" was not found, as it can't look further into the path.
    assert "random" in exception_info.value.args[0]


@pytest.fixture()
def setup_recursive_getitem() -> tuple[dict[str, dict[str, str]], list[str], str]:
    """Setup a test dict for use with recursive_getitem."""
    expected = "hello"
    keys = ["a", "b"]
    d = {"a": {"b": expected}}

    return d, keys, expected


def test_recursive_getitem(setup_recursive_getitem: Any) -> None:
    """Tests for recursive getitem."""
    d, keys, expected = setup_recursive_getitem

    assert utils.recursive_getitem(d, keys) == expected


def test_recursive_getitem_single_key(setup_recursive_getitem: Any) -> None:
    """Tests for recursive getitem with a single key."""
    d, keys, expected = setup_recursive_getitem

    assert utils.recursive_getitem(d["a"], "b") == expected


def test_recursive_getitem_fail(setup_recursive_getitem: Any) -> None:
    """Tests failing for recursive getitem."""
    d, keys, expected = setup_recursive_getitem

    # Use a random set of keys, which will fail.
    with pytest.raises(KeyError) as exception_info:
        utils.recursive_getitem(d, ["fake", "keys"])
    # It will only return that "random" was not found, as it can't look further into the path.
    assert "fake" in exception_info.value.args[0]
