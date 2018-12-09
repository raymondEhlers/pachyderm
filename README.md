# Pachyderm

[![Documentation Status](https://readthedocs.org/projects/pachyderm-heavy-ion/badge/?version=latest)](https://pachyderm-heavy-ion.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/raymondEhlers/pachyderm.svg?branch=master)](https://travis-ci.com/raymondEhlers/pachyderm)
[![codecov](https://codecov.io/gh/raymondEhlers/pachyderm/branch/master/graph/badge.svg)](https://codecov.io/gh/raymondEhlers/pachyderm)

Physics analysis core for heavy-ions.

## Dockerfile

There is a Dockerfile which is used for testing pachyderm with ROOT. It is based on the overwatch base image
to allow usage of ROOT. It may also be used to run pachyderm if so desired, although such a use case doesn't
seem tremendously useful (which is why the image isn't pushed to docker hub).
