# Dockerfile for testing pachyderm
# NOTE: This could also be used for running pachyderm, but it wouldn't be terribly useful.
# We use the Overwatch base image so we don't have to deal with setting up ROOT.
# All we need to know is that the user is named "overwatch".
FROM rehlers/overwatch-base:py3.6.6
LABEL maintainer="Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University"

# Setup environment
ENV ROOTSYS="/opt/root"
ENV PATH="${ROOTSYS}/bin:/home/overwatch/.local/bin:${PATH}"
ENV LD_LIBRARY_PATH="${ROOTSYS}/lib:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="${ROOTSYS}/lib:${PYTHONPATH}"

# Setup pachyderm
ENV PACHYDERM_ROOT /opt/pachyderm
# We intentionally make the directory before setting it as the workdir so the directory is made with user permissions
# (workdir always creates the directory with root permissions)
RUN mkdir -p ${PACHYDERM_ROOT}
WORKDIR ${PACHYDERM_ROOT}

# Copy pachyderm into the image.
COPY --chown=overwatch:overwatch . ${PACHYDERM_ROOT}

# TEMP: As of 12 Dec 2018, lz4 will fail to install because there is an empty "pkgconfig" directory in the $ROOTSYS/lib directory
#       Posted an issuen to lz4: https://github.com/python-lz4/python-lz4/issues/158.
RUN pip install --user --upgrade --no-cache-dir pkgconfig
# Install pachyderm.
RUN pip install --user --upgrade --no-cache-dir -e .[tests,dev,docs]
