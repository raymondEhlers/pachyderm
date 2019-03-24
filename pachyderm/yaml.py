#!/usr/bin/env python

""" Module related to YAML.

Contains a way to construct the main YAML object, as well as relevant mixins and classes.

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

import enum
import inspect
import logging
import numpy as np
import ruamel.yaml
from typing import Any, Iterable, Optional, Sequence, Type, TypeVar

logger = logging.getLogger(__name__)

# Typing helpers
DictLike = ruamel.yaml.comments.CommentedMap
Representer = Type[ruamel.yaml.representer.BaseRepresenter]
Constructor = Type[ruamel.yaml.constructor.BaseConstructor]
T_EnumToYAML = TypeVar("T_EnumToYAML", bound = enum.Enum)
T_EnumFromYAML = TypeVar("T_EnumFromYAML", bound = enum.Enum)

def yaml(modules_to_register: Iterable[Any] = None, classes_to_register: Iterable[Any] = None) -> ruamel.yaml.YAML:
    """ Create a YAML object for loading a YAML configuration.

    Args:
        modules_to_register: Modules containing classes to be registered with the YAML object. Default: None.
        classes_to_register: Classes to be registered with the YAML object. Default: None.
    Returns:
        A newly creating YAML object, configured as apporpirate.
    """
    # Defein a round-trip yaml object for us to work with. This object should be imported by other modules
    # NOTE: "typ" is a not a typo. It stands for "type"
    yaml = ruamel.yaml.YAML(typ = "rt")

    # Register representers and constructors
    # Numpy
    yaml.representer.add_representer(np.ndarray, numpy_to_yaml)
    yaml.constructor.add_constructor("!numpy_array", numpy_from_yaml)
    # Register external classes
    yaml = register_module_classes(yaml = yaml, modules = modules_to_register)
    yaml = register_classes(yaml = yaml, classes = classes_to_register)

    return yaml

def register_classes(yaml: ruamel.yaml.YAML, classes: Optional[Iterable[Any]] = None) -> ruamel.yaml.YAML:
    """ Register externally defined classes. """
    # Validation
    if classes is None:
        classes = []

    # Register the classes
    for cls in classes:
        logger.debug(f"Registering class {cls} with YAML")
        yaml.register_class(cls)

    return yaml

def register_module_classes(yaml: ruamel.yaml.YAML, modules: Optional[Iterable[Any]] = None) -> ruamel.yaml.YAML:
    """ Register all classes in the given modules with the YAML object.

    This is a simple helper function.
    """
    # Validation
    if modules is None:
        modules = []

    # Extract the classes from the modules
    classes_to_register = set()
    for module in modules:
        module_classes = [member[1] for member in inspect.getmembers(module, inspect.isclass)]
        classes_to_register.update(module_classes)

    # Register the extracted classes
    return register_classes(yaml = yaml, classes = classes_to_register)

#
# Representers and constructors for individual classes.
#

def numpy_to_yaml(representer: Representer, data: np.ndarray) -> Sequence[Any]:
    """ Write a numpy array to YAML.

    It registers the array under the tag ``!numpy_array``.

    Use with:

    .. code-block:: python

        >>> yaml = ruamel.yaml.YAML()
        >>> yaml.representer.add_representer(np.ndarray, yaml.numpy_to_yaml)

    Note:
        We cannot use ``yaml.register_class`` because it won't register the proper type.
        (It would register the type of the class, rather than of `numpy.ndarray`). Instead,
        we use the above approach to register this method explicitly with the representer.
    """
    return representer.represent_sequence(
        "!numpy_array",
        data.tolist()
    )

def numpy_from_yaml(constructor: Constructor, data: ruamel.yaml.nodes.SequenceNode) -> np.ndarray:
    """ Read an array from YAML to numpy.

    It reads arrays registered under the tag ``!numpy_array``.

    Use with:

    .. code-block:: python

        >>> yaml = ruamel.yaml.YAML()
        >>> yaml.constructor.add_constructor("!numpy_array", yaml.numpy_from_yaml)

    Note:
        We cannot use ``yaml.register_class`` because it won't register the proper type.
        (It would register the type of the class, rather than of `numpy.ndarray`). Instead,
        we use the above approach to register this method explicitly with the representer.
    """
    # Construct the contained values so that we properly construct int, float, etc.
    # We just leave this to YAML because it already stores this information.
    values = [constructor.construct_object(n) for n in data.value]
    logger.debug(f"{data}, {values}")
    return np.array(values)

def enum_to_yaml(cls: Type[T_EnumToYAML], representer: Representer, data: T_EnumToYAML) -> ruamel.yaml.nodes.ScalarNode:
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
        This method assumes that the name of the enumeration value should be stored as a scalar node.

    Args:
        representer: Representation from YAML.
        data: Enumeration value to be encoded.
    Returns:
        Scalar representation of the name of the enumeration value.
    """
    return representer.represent_scalar(
        f"!{cls.__name__}",
        f"{str(data)}"
    )

def enum_from_yaml(cls: Type[T_EnumFromYAML], constructor: Constructor, node: ruamel.yaml.nodes.ScalarNode) -> T_EnumFromYAML:
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

