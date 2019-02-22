#/usr/bin/env python

""" Setup pachyderm.

Derived from the setup.py in aliBuild and Overwatch
and based on: https://python-packaging.readthedocs.io/en/latest/index.html

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
import os

def get_version():
    versionModule = {}
    with open(os.path.join("pachyderm", "version.py")) as f:
        exec(f.read(), versionModule)
    return versionModule["__version__"]

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = "pachyderm",
    version = get_version(),

    description = "Physics Analysis Core for Heavy-Ions",
    long_description = long_description,
    long_description_content_type = "text/markdown",

    author = "Raymond Ehlers",
    author_email = "raymond.ehlers@cern.ch",

    url = "https://github.com/raymondEhlers/pachyderm",
    license = "BSD 3-Clause",

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers = [
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],

    # What does your project relate to?
    keywords = 'HEP ALICE',

    packages = find_packages(exclude=(".git", "tests")),

    # Rename scripts to the desired executable names
    # See: https://stackoverflow.com/a/8506532
    entry_points = {
        "console_scripts": [
        ],
    },

    # This is usually the minimal set of the required packages.
    install_requires = [
        "dataclasses;python_version<'3.7'",
        "ruamel.yaml",
        "numpy",
        "uproot",
        # Depends on ROOT, but that can't be installed through pip.
        # The dependence is only implicit, so it can actually be installed without it,
        # but some functionality depends on it being available.
        #"ROOT"
    ],

    # Include additional files
    include_package_data = True,
    package_data = {
        "pachyderm": ["py.typed"],
    },

    extras_require = {
        "tests": [
            "pytest",
            "pytest-cov",
            "pytest-mock",
            "codecov",
        ],
        "docs": [
            "sphinx",
            # Allow markdown files to be used
            "recommonmark",
            # Allow markdown tables to be used
            "sphinx_markdown_tables",
        ],
        "dev": [
            "pre-commit",
            "flake8",
            # Makes flake8 easier to parse
            "flake8-colors",
            # Type checking
            "mypy",
        ]
    }
)
