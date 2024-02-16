from __future__ import annotations

import importlib.metadata

import pachyderm as m


def test_version():
    assert importlib.metadata.version("pachyderm") == m.__version__
