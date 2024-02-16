""" Copy some set of files from AliEn.

Uses some code from Markus' download train run-by-run output script.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""
from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def valid_alien_token() -> bool:
    """Check if there is a valid AliEn token.

    The function must be robust enough to fetch all possible xrd error states which
    it usually gets from the stdout of the query process.

    Args:
        None.
    Returns:
        True if there is a valid AliEn token, or false otherwise.
    """
    # With JAliEn, this information is no longer available, so this is a no-op
    # that always just returns True.
    return True


def local_md5(fname: Path | str) -> str:
    """Calculate a chunked md5 sum for the file at a given filename.

    Args:
        fname: Path to the file.
    Returns:
        md5 sum of the file.
    """
    # Validation
    fname = Path(fname)

    # Calculate md5 sum
    hash_md5 = hashlib.md5()
    with fname.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def grid_md5(gridfile: Path | str) -> str:
    """Calculate md5 sum of a file on the grid.

    The function must be robust enough to fetch all possible xrd error states which
    it usually gets from the stdout of the query process.

    Args:
        gridfile: Path to the file on the grid.
    Returns:
        md5 sum of the file.
    """
    # Validation
    gridfile = Path(gridfile)

    # Calculate the md5 sum via ghbox. If there is an error, we retry.
    errorstate = True
    while errorstate:
        errorstate = False
        result = subprocess.run(["alien.py", "md5sum", str(gridfile)], capture_output=True, check=True)
        gb_out = result.stdout.decode()
        # Check that the command ran successfully
        if gb_out.startswith(("Error", "Warning")) or "CheckErrorStatus" in gb_out:
            errorstate = True

    return gb_out.split("\t")[0]


def check_output_file(inputfile: Path | str, outputfile: Path | str, verbose: bool = True) -> bool:
    """Check if the output file exists and matches the input file.

    Args:
        inputfile: Path to the file to be copied.
        outputfile: Path to where the file should be copied.
    Returns:
        True if the output file was copied successfully (exists and matches MD5 sum).
    """
    # Validation
    inputfile = Path(inputfile)
    outputfile = Path(outputfile)

    if outputfile.exists():
        localmd5 = local_md5(outputfile)
        gridmd5 = grid_md5(inputfile)
        # logger.debug(f"MD5local: {localmd5}, MD5grid {gridmd5}")
        if localmd5 != gridmd5:
            logger.error(f"Mismatch in MD5 sum for file {outputfile}. Need to recopy.")
            # Incorrect MD5 sum - probably a corrupted output file.
            outputfile.unlink()
            return False
        # Success!
        if verbose:
            logger.info(f"Output file {outputfile} (was) copied correctly")
        return True

    # Fall through if the output file doesn't exist.
    if verbose:
        logger.error(f"output file {outputfile} not found")
    return False


def copy_from_alien(inputfile: Path | str, outputfile: Path | str) -> bool:
    """Copy a file using alien_cp.

    The function must be robust enough to fetch all possible xrd error states which
    it usually gets from the stdout of the query process.

    Args:
        inputfile: Path to the file to be copied.
        outputfile: Path to where the file should be copied.
    Returns:
        True if the file was copied successfully.
    """
    # Validation
    inputfile = Path(inputfile)
    outputfile = Path(outputfile)

    # Create the output location
    logger.info(f"Copying {inputfile} to {outputfile}")
    outputfile.parent.mkdir(mode=0o755, exist_ok=True, parents=True)
    process = subprocess.run(["alien_cp", str(inputfile), f"file://{outputfile}"], capture_output=True, check=False)
    if process.returncode:
        # Process failed. Return false so we can try again.
        return False

    # Check that the file was copied successfully.
    return check_output_file(inputfile=inputfile, outputfile=outputfile)


def list_alien_dir(input_dir: Path | str) -> list[str]:
    """List the files in a directory on AliEn.

    The function must be robust against error states which it can only get from the stdout. As
    long as the request ends in error state it should retry.

    Args:
        input_dir: Path to the directory on AliEn.
    Returns:
        List of files on AliEn in the given directory.
    """
    # Validation
    input_dir = Path(input_dir)

    # Search for the files.
    errorstate = True
    while errorstate:
        # Grab the list of files from alien via alien_ls
        logger.debug("Searching for files on AliEn...")
        process = subprocess.run(["alien_ls", str(input_dir)], capture_output=True, check=False)

        # Extract the files from the output.
        errorstate = False
        result: list[str] = []
        if process.returncode == 0:
            # Only process if the command itself didn't report an issue via the return code.
            for d in process.stdout.decode().split("\n"):
                if d.startswith(("Error", "Warning")):
                    errorstate = True
                    break
                # Remove leading and trailing whitespace.
                # jalien now returns directories with trailing slashes (different from legacy AliEn).
                # We depended on this behavior, so we strip out the trailing slashes (via os.sep).
                mydir = d.rstrip().lstrip().rstrip(os.sep)

                # if len(mydir) and mydir.isdigit():
                if len(mydir):
                    result.append(mydir)
        else:
            # The directory doesn't exist (most likely), or some other issue reported directly from
            # alien_ls via return code. In this case, we just want to return an empty list.
            ...

        # If we haven't succeeded, let's try again.
        if errorstate:
            continue

    return result
