#!/usr/bin/env python

""" Contains generic classes

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

from typing import Any, cast, TypeVar

_T = TypeVar("_T", bound = "EqualityMixin")

class EqualityMixin:
    """ Mixin generic comparison operations using `__dict__`.

    Can then be mixed into any other class using multiple inheritance.

    Inspired by: https://stackoverflow.com/a/390511.
    """
    # Typing is only ignore in the function definition line because it conflicts with the base object,
    # which wants to return Any. This is apparently the preferred approach.
    # See: https://github.com/python/mypy/issues/2783
    def __eq__(self: _T, other: Any) -> bool:
        """ Check for equality of members. """
        # Check identity to avoid needing to perform the (potentially costly) dict comparison.
        if self is other:
            return True
        # Compare via the member values.
        if type(other) is type(self):
            return cast(bool, self.__dict__ == other.__dict__)
        return NotImplemented

