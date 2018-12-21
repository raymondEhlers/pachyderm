#!/usr/bin/env python

""" Contains generic classes

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

# Both are for typing information
import ruamel.yaml
from typing import Type, TypeVar

class EqualityMixin(object):
    """ Mixin generic comparison operations using `__dict__`.

    Can then be mixed into any other class using multiple inheritance.

    Inspired by: https://stackoverflow.com/a/390511.
    """
    def __eq__(self, other) -> bool:
        """ Check for equality of members. """
        # Check identity to avoid needing to perform the (potentially costly) dict comparison.
        if self is other:
            return True
        # Compare via the member values.
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return NotImplemented

T_EnumToYAML = TypeVar("T_EnumToYAML", bound = "EnumToYAML")

class EnumToYAML:
    """ Mixin method for writing enum classes to YAML.

    This method writes whatever is used in the string representation of the YAML value.
    Usually, this will be the unique name of the enumeration value. If the name is used,
    the corresponding ``EnumFromYAML`` mixin can be used to recreate the value. If the name
    isn't used, more care may be necessary, so a ``from_yaml`` method for that particular
    enumeration may be necessary.
    """
    @classmethod
    def to_yaml(cls, representer: Type[ruamel.yaml.representer.BaseRepresenter], node: Type[T_EnumToYAML]) -> ruamel.yaml.nodes.ScalarNode:
        """ Encodes YAML representation.

        Args:
            representer: Representation from YAML.
            node: Enumeration value to be encoded.
        Returns:
            Scalar representation of the name of the enumeration value.
        """
        return representer.represent_scalar(
            f"!{cls.__name__}",
            f"{str(node)}"
        )

T_EnumFromYAML = TypeVar("T_EnumFromYAML", bound = "EnumFromYAML")

class EnumFromYAML:
    """ Mixin method for reading enum values from YAML.

    This method assumes that the name of the enumeration value was stored as a scalar node.
    """
    @classmethod
    def from_yaml(cls: Type[T_EnumFromYAML], constructor: ruamel.yaml.constructor.BaseConstructor, node: ruamel.yaml.nodes.ScalarNode) -> Type[T_EnumFromYAML]:
        """ Decode YAML representation.

        Args:
            constructor: Constructor from the YAML object.
            node: Scalar node extracted from the YAML being read.
        Returns:
            The constructed YAML value from the name of the enumerated value.
        """
        # mypy doesn't like indexing to construct the enumeration.
        return cls[node.value]  # type: ignore

class EnumWithYAML(EnumToYAML, EnumFromYAML):
    """ Mixin methods for reading and writing the names of enumeration values to YAML.

    Just a convenience class for combining the read and write to YAML methods that will be used
    for most enumeration methods.
    """
    pass

