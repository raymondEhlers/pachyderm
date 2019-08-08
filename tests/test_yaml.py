#!/usr/bin/env python

""" Tests for the YAML module.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from dataclasses import dataclass
import enum
import numpy as np
import pytest  # noqa: F401
import tempfile
from typing import Any, List

from pachyderm import yaml

def dump_and_load_yaml(yml: yaml.ruamel.yaml.YAML, input_value: List[Any]) -> Any:
    """ Perform a dump and load YAML loop. """
    # Read and write to a temp file for convenience.
    with tempfile.TemporaryFile() as f:
        # Dump the value to the file
        yml.dump(input_value, f)
        # Then load it back.
        f.seek(0)
        result = yml.load(f)

    return result

def test_enum_with_yaml(logging_mixin: Any) -> None:
    """ Test closure of reading and writing enum values to YAML. """
    # Setup
    class TestEnum(enum.Enum):
        a = 1
        b = 2

        def __str__(self) -> str:
            return str(self.name)

        to_yaml = classmethod(yaml.enum_to_yaml)
        from_yaml = classmethod(yaml.enum_from_yaml)

    yml = yaml.yaml(classes_to_register = [TestEnum])
    input_value = TestEnum.a

    # Perform a round-trip of dumping and loading
    result = dump_and_load_yaml(yml = yml, input_value = [input_value])

    assert result == [input_value]

def test_numpy(logging_mixin: Any) -> None:
    """ Test reading and writing numpy to YAML. """
    # Setup
    yml = yaml.yaml()

    test_array = np.array([1, 2, 3, 4, 5])

    # Perform a round-trip of dumping and loading
    result = dump_and_load_yaml(yml = yml, input_value = [test_array])

    assert np.allclose(test_array, result)

def test_module_registration(logging_mixin: Any, mocker: Any) -> None:
    """ Test registering the classes in a module. """
    # Setup
    @dataclass
    class TestClass:
        a: int
        b: int
    # Mock inspect so we don't have to actually depend on another module.
    m_inspect_getmembers = mocker.MagicMock(return_value = [(None, TestClass)])
    mocker.patch("pachyderm.yaml.inspect.getmembers", m_inspect_getmembers)

    yml = yaml.yaml(modules_to_register = ["Fake module"])

    # Perform a round-trip of dumping and loading
    input_value = TestClass(a = 1, b = 2)
    result = dump_and_load_yaml(yml = yml, input_value = [input_value])

    assert result == [input_value]
