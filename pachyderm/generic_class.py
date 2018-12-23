#!/usr/bin/env python

""" Contains generic classes

Note:
    The YAML to/from enum values would be much better as a mixin. However, such an approach causes substantial issues.
    In particular, although we don't explicitly pickle the values, calling ``copy.copy`` implicitly calls pickle, so
    we must maintain compatibility. However, enum mixins preclude pickling the enum value
    (see `cpython/enum.py line 177 <https://github.com/python/cpython/blob/master/Lib/enum.py#L177>`__). The problem
    basically comes down to the fact that we are assigning a bound staticmethod to the class when we mix
    it in, and it doesn't seem to be able to resolving pickling the object (perhaps due to name resolution issues).
    For a bit more, see the comments `on this stackoverflow post <https://stackoverflow.com/q/46230799>`__.
    Piratically, I believe that we could also resolve this by implementing ``__reduce_ex``, but that appears as if
    it will be more work than our implemented workaround. Our workaround can be implemented as:

    .. code-block:: python

    >>> class TestEnum(enum.Enum):
    ...   a = 1
    ...   b = 2
    ...
    ...   def __str__(self):
    ...     return self.name
    ...
    ...   to_yaml = staticmethod(generic_class.enum_to_yaml)
    ...   from_yaml = staticmethod(generic_class.enum_from_yaml)

    This enum object will pickle properly. Note that rather strangely, this issue showed up during tests on Debian
    Stretch, but not the exact same version of python on macOS. I don't know why that's the case, but the workaround
    seems to be fine on both systems, so we'll just continue to use it.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

# All are for typing information
import enum
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

T_EnumToYAML = TypeVar("T_EnumToYAML", bound = enum.Enum)

def enum_to_yaml(cls: Type[T_EnumToYAML], representer: Type[ruamel.yaml.representer.BaseRepresenter], node: Type[T_EnumToYAML]) -> ruamel.yaml.nodes.ScalarNode:
    """ Encodes YAML representation.

    This is a mixin method for writing enum values to YAML. It needs to be added to the enum
    as a classmethod. See the module docstring for further information on this approach and how
    to implement it.

    This method writes whatever is used in the string representation of the YAML value.
    Usually, this will be the unique name of the enumeration value. If the name is used,
    the corresponding ``EnumFromYAML`` mixin can be used to recreate the value. If the name
    isn't used, more care may be necessary, so a ``from_yaml`` method for that particular
    enumeration may be necessary.

    Note:
        This method assumes that the name of the enumeration value was stored as a scalar node.

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

T_EnumFromYAML = TypeVar("T_EnumFromYAML", bound = enum.Enum)

def enum_from_yaml(cls: Type[T_EnumFromYAML], constructor: ruamel.yaml.constructor.BaseConstructor, node: ruamel.yaml.nodes.ScalarNode) -> Type[T_EnumFromYAML]:
    """ Decode YAML representation.

    This is a mixin method for reading enum values from YAML. It needs to be added to the enum
    as a classmethod. See the module docstring for further information on this approach and how
    to implement it.

    Note:
        This method assumes that the name of the enumeration value was stored as a scalar node.

    Args:
        constructor: Constructor from the YAML object.
        node: Scalar node extracted from the YAML being read.
    Returns:
        The constructed YAML value from the name of the enumerated value.
    """
    # mypy doesn't like indexing to construct the enumeration.
    return cls[node.value]  # type: ignore

