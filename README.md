# Pachyderm

[![Documentation Status](https://readthedocs.org/projects/pachyderm-heavy-ion/badge/?version=latest)](https://pachyderm-heavy-ion.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/raymondEhlers/pachyderm.svg?branch=master)](https://travis-ci.com/raymondEhlers/pachyderm)
[![codecov](https://codecov.io/gh/raymondEhlers/pachyderm/branch/master/graph/badge.svg)](https://codecov.io/gh/raymondEhlers/pachyderm)

Pachyderm[\*](#name-meaning) provides core functionality for heavy-ion physics analyses. The main
functionality includes a generic histogram projection interface, a recursive configuration determination
module (including overriding (merging) capabilities), and general utilities (especially for histograms). It
provides base functionality to the [ALICE jet-hadron
analysis](https://github.com/raymondEhlers/alice-jet-hadron) package. This package provides many examples of
how pachyderm can be used in various analysis tasks.

For further information on the capabilities, see the
[docuemntation](https://readthedocs.org/projects/pachyderm-heavy-ion/badge/?version=latest).

## Installation

Pachyderm requires python 3.6 or above. It is available on [PyPI](https://pypi.org/project/pachyderm/) and can
be installed via pip:

```bash
$ pip install pachyderm
```

## Dependencies

All dependencies are specified in the `setup.py` (and will be handled automatically when installed via pip)
except for ROOT. The package can be installed without ROOT with limited functionality, but for full
functionality, ROOT must be available.

### Dockerfile

There is a Dockerfile which is used for testing pachyderm with ROOT. It is based on the
[Overwatch](https://github.com/raymondEhlers/OVERWATCH) [base docker
image](https://hub.docker.com/r/rehlers/overwatch-base/) to allow us to avoid redeveloping another container
just to have ROOT available. It may also be used to run pachyderm if so desired, although such a use case
doesn't seem tremendously useful (which is why the image isn't pushed to docker hub).

## Development

I recommend setting up the development environment as follows:

```bash
# Setup
$ pip install -e .[dev,tests,docs]
# Setup git pre-commit hooks to reduce errors
$ pre-commit install
# develop develop develop...
```

## Documentation

All classes, functions, etc, should be documented, including with typing information. [The
docs](https://pachyderm-heavy-ion.readthedocs.io/en/latest/) are built on each new successful commit. They can
also be built locally using:

```bash
# Setup
$ pip install -e .[dev,tests,docs]
# Create the docs
$ pushd doc && make html && popd
# Open the created docs
$ open docs/_build/html/index.html
```

## Name Meaning

**PACHYDERM**: **P**hysics **A**nalysis **C**ore for **H**eav**Y**-ions with **D**etermination of (analysis)
**E**lements via **R**ecursion and **M**erging.

