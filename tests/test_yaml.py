""" Tests for the YAML module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""
from __future__ import annotations

import enum
import logging
import tempfile
from dataclasses import dataclass
from io import StringIO
from typing import Any

import numpy as np
import pytest  # noqa: F401

from pachyderm import yaml

logger = logging.getLogger(__name__)


def dump_and_load_yaml(yml: yaml.ruamel.yaml.YAML, input_value: list[Any]) -> Any:  # type: ignore[name-defined]
    """Perform a dump and load YAML loop."""
    # Read and write to a temp file for convenience.
    with tempfile.TemporaryFile() as f:
        # Dump the value to the file
        yml.dump(input_value, f)
        # Then load it back.
        f.seek(0)
        result = yml.load(f)

    return result  # noqa: RET504


def test_enum_with_yaml() -> None:
    """Test closure of reading and writing enum values to YAML."""

    # Setup
    class TestEnum(enum.Enum):
        a = 1
        b = 2

        def __str__(self) -> str:
            return str(self.name)

        to_yaml = classmethod(yaml.enum_to_yaml)
        from_yaml: Any = classmethod(yaml.enum_from_yaml)

    yml = yaml.yaml(classes_to_register=[TestEnum])
    input_value = TestEnum.a

    # Perform a round-trip of dumping and loading
    result = dump_and_load_yaml(yml=yml, input_value=[input_value])

    assert result == [input_value]


def test_numpy() -> None:
    """Test reading and writing numpy to YAML."""
    # Setup
    yml = yaml.yaml()

    test_array = np.array([1, 2, 3, 4, 5])

    # Perform a round-trip of dumping and loading
    result = dump_and_load_yaml(yml=yml, input_value=[test_array])

    assert np.allclose(test_array, result)


def test_hand_written_numpy() -> None:
    """Test constructing a numpy array from a hand written array.

    This is in contrast to machine written arrays, which will be encoded
    in the numpy binary format.
    """
    # Setup
    yml = yaml.yaml()
    test_array = np.array([1, 2, 3, 4, 5])

    # Create the YAML and load it.
    input_yaml = f"""---
x: !numpy_array {test_array.tolist()}
"""
    s = StringIO()
    s.write(input_yaml)
    s.seek(0)
    result = yml.load(s)

    logger.debug(f"result['x']: {result['x']}")

    # Check that it was loaded properly.
    assert np.allclose(result["x"], test_array)


def test_numpy_float() -> None:
    """Test reading and writing numpy floats to YAML."""
    # Setup
    yml = yaml.yaml()

    test_value = np.float64(123.456)

    # Perform a round-trip of dumping and loading
    result = dump_and_load_yaml(yml=yml, input_value=[test_value])

    assert np.isclose(test_value, result)


def test_hand_written_numpy_float() -> None:
    """Test constructing a numpy array from a hand written array.

    This is in contrast to machine written arrays, which will be encoded
    in the numpy binary format.
    """
    # Setup
    yml = yaml.yaml()
    test_value = np.float64(1234.567)

    # Create the YAML and load it.
    input_yaml = f"""---
x: !numpy_float64 {test_value}
"""
    s = StringIO()
    s.write(input_yaml)
    s.seek(0)
    result = yml.load(s)

    logger.debug(f"result['x']: {result['x']}")

    # Check that it was loaded properly.
    assert np.isclose(result["x"], test_value)


def test_module_registration(mocker: Any) -> None:
    """Test registering the classes in a module."""

    # Setup
    @dataclass
    class TestClass:
        a: int
        b: int

    # Mock inspect so we don't have to actually depend on another module.
    m_inspect_getmembers = mocker.MagicMock(return_value=[(None, TestClass)])
    mocker.patch("pachyderm.yaml.inspect.getmembers", m_inspect_getmembers)

    yml = yaml.yaml(modules_to_register=["Fake module"])

    # Perform a round-trip of dumping and loading
    input_value = TestClass(a=1, b=2)
    result = dump_and_load_yaml(yml=yml, input_value=[input_value])

    assert result == [input_value]
