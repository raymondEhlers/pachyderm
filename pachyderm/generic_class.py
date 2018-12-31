#!/usr/bin/env python

""" Contains generic classes

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

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

