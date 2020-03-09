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
    Practically, I believe that we could also resolve this by implementing ``__reduce_ex``, but that appears as if
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

import base64
import enum
import inspect
import logging
from io import BytesIO
from typing import Any, Iterable, Optional, Type, TypeVar, cast

import numpy as np
import ruamel.yaml

logger = logging.getLogger(__name__)

# Typing helpers
# Careful: Importing these in other modules can cause mypy to give nonsense typing information!
DictLike = ruamel.yaml.comments.CommentedMap
Representer = ruamel.yaml.representer.BaseRepresenter
Constructor = ruamel.yaml.constructor.BaseConstructor
T_EnumToYAML = TypeVar("T_EnumToYAML", bound = enum.Enum)
T_EnumFromYAML = TypeVar("T_EnumFromYAML", bound = enum.Enum)

def yaml(modules_to_register: Optional[Iterable[Any]] = None,
         classes_to_register: Optional[Iterable[Any]] = None) -> ruamel.yaml.YAML:
    """ Create a YAML object for loading a YAML configuration.

    Args:
        modules_to_register: Modules containing classes to be registered with the YAML object. Default: None.
        classes_to_register: Classes to be registered with the YAML object. Default: None.
    Returns:
        A newly creating YAML object, configured as appropriate.
    """
    # Define a round-trip YAML object for us to work with. This object should be imported by other modules
    # NOTE: "typ" is a not a typo. It stands for "type"
    yaml = ruamel.yaml.YAML(typ = "rt")

    # Register representers and constructors
    # Numpy array
    yaml.representer.add_representer(np.ndarray, numpy_array_to_yaml)
    yaml.constructor.add_constructor("!numpy_array", numpy_array_from_yaml)
    # Numpy float64
    yaml.representer.add_representer(np.float64, numpy_float64_to_yaml)
    yaml.constructor.add_constructor("!numpy_float64", numpy_float64_from_yaml)
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

def numpy_array_to_yaml(representer: ruamel.yaml.representer.BaseRepresenter, data: np.ndarray) -> str:
    """ Write a numpy array to YAML.

    It registers the array under the tag ``!numpy_array``.

    Use with:

    .. code-block:: python

        >>> yaml = ruamel.yaml.YAML()
        >>> yaml.representer.add_representer(np.ndarray, yaml.numpy_array_to_yaml)

    Note:
        We cannot use ``yaml.register_class`` because it won't register the proper type.
        (It would register the type of the class, rather than of `numpy.ndarray`). Instead,
        we use the above approach to register this method explicitly with the representer.
    """
    # Create a bytes object, dump to it, encode the bytes to a str, and then write them.
    # It's less transparent when physically reading it, but it should avoid encoding issues.
    b = BytesIO()
    np.save(b, data)
    b.seek(0)
    # The representer is seen by mypy as Any, so we need to explicitly note that it's a str.
    return cast(
        str,
        representer.represent_scalar(
            "!numpy_array", base64.encodebytes(b.read()).decode("utf-8"),
        )
    )

def numpy_array_from_yaml(constructor: ruamel.yaml.constructor.BaseConstructor,
                          data: ruamel.yaml.nodes.SequenceNode) -> np.ndarray:
    """ Read an array from YAML to numpy.

    It reads arrays registered under the tag ``!numpy_array``.

    Use with:

    .. code-block:: python

        >>> yaml = ruamel.yaml.YAML()
        >>> yaml.constructor.add_constructor("!numpy_array", yaml.numpy_array_from_yaml)

    Note:
        We cannot use ``yaml.register_class`` because it won't register the proper type.
        (It would register the type of the class, rather than of `numpy.ndarray`). Instead,
        we use the above approach to register this method explicitly with the representer.

    Note:
        In order to allow users to write an array by hand, we check the data given. If it's
        a list, we convert the values and put them into an array. If it's binary encoded,
        we decode and load it.

    Args:
        constructor: YAML constructor being used to read and create the objects specified in the YAML.
        data: Data stored in the YAML node currently being processed.
    Returns:
        Numpy array containing the data in the current YAML node.
    """
    return_value: np.ndarray
    if isinstance(data.value, list):
        # These are probably from a hand encoded file. We will convert them into an array.
        # Construct the contained values so that we properly construct int, float, etc.
        # We just leave this to YAML because it already stores this information.
        values = [constructor.construct_object(n) for n in data.value]
        return_value = np.array(values)
    else:
        # Binary encoded numpy. Decode and load it.
        b = data.value.encode("utf-8")
        # Requires explicitly allowing pickle to load arrays. This used to be default True,
        # so our risk hasn't changed.
        return_value = np.load(BytesIO(base64.decodebytes(b)), allow_pickle = True)
    return return_value

def numpy_float64_to_yaml(representer: ruamel.yaml.representer.BaseRepresenter, data: np.float64) -> str:
    """ Write a numpy float64 to YAML.

    It registers the float under the tag ``!numpy_float64``.

    Use with:

    .. code-block:: python

        >>> yaml = ruamel.yaml.YAML()
        >>> yaml.representer.add_representer(np.float64, yaml.numpy_float64_to_yaml)

    Note:
        We cannot use ``yaml.register_class`` because it won't register the proper type.
        (It would register the type of the class, rather than of `numpy.float64`). Instead,
        we use the above approach to register this method explicitly with the representer.
    """
    # Create a bytes object, dump to it, encode the bytes to a str, and then write them.
    # It's less transparent when physically reading it, but it should avoid encoding issues.
    b = BytesIO()
    np.save(b, data)
    b.seek(0)
    # The representer is seen by mypy as Any, so we need to explicitly note that it's a str.
    return cast(
        str,
        representer.represent_scalar(
            "!numpy_float64", base64.encodebytes(b.read()).decode("utf-8"),
        )
    )

def numpy_float64_from_yaml(constructor: ruamel.yaml.constructor.BaseConstructor,
                            data: ruamel.yaml.nodes.ScalarNode) -> np.float64:
    """ Read an float64 from YAML to numpy.

    It reads the float64 registered under the tag ``!numpy_float64``.

    Use with:

    .. code-block:: python

        >>> yaml = ruamel.yaml.YAML()
        >>> yaml.constructor.add_constructor("!numpy_float64", yaml.numpy_float64_from_yaml)

    Note:
        We cannot use ``yaml.register_class`` because it won't register the proper type.
        (It would register the type of the class, rather than of `numpy.float64`). Instead,
        we use the above approach to register this method explicitly with the representer.

    Note:
        In order to allow users to write an float by hand, we check the data given. If it's
        a raw float, we put it into an float64. If it's binary encoded, we decode and load it.

    Args:
        constructor: YAML constructor being used to read and create the objects specified in the YAML.
        data: Data stored in the YAML node currently being processed.
    Returns:
        Numpy float64 containing the data in the current YAML node.
    """
    return_value: np.float64
    try:
        # First guess that it's a hand encoded file. We can't detect the type because the value
        # is just a str. We will convert it into a numpy float.
        return_value = np.float64(data.value)
    except ValueError:
        # It can't convert to a float, so it's probably binary encoded. Decode and load it.
        b = data.value.encode("utf-8")
        # Requires explicitly allowing pickle to load arrays. This used to be default True,
        # so our risk hasn't changed.
        return_value = np.load(BytesIO(base64.decodebytes(b)), allow_pickle = True)
    return return_value

def enum_to_yaml(cls: Type[T_EnumToYAML],
                 representer: ruamel.yaml.representer.BaseRepresenter,
                 data: T_EnumToYAML) -> ruamel.yaml.nodes.ScalarNode:
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
    return cast(
        ruamel.yaml.nodes.ScalarNode,
        representer.represent_scalar(
            f"!{cls.__name__}",
            f"{str(data)}"
        )
    )

def enum_from_yaml(cls: Type[T_EnumFromYAML],
                   constructor: ruamel.yaml.constructor.BaseConstructor,
                   node: ruamel.yaml.nodes.ScalarNode) -> T_EnumFromYAML:
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
    return cls[node.value]
