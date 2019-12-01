#!/usr/bin/env python3

""" Copy some set of files from AliEn.

Uses some code from Markus' download train run-by-run output script.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import hashlib
import logging
import subprocess
from pathlib import Path
from typing import List, Union

logger = logging.getLogger(__name__)

def valid_alien_token() -> bool:
    """ Check if there is a valid AliEn token.

    The function must be robust enough to fetch all possible xrd error states which
    it usually gets from the stdout of the query process.

    Args:
        None.
    Returns:
        True if there is a valid AliEn token, or false otherwise.
    """
    errorstate = True
    while errorstate:
        errorstate = False
        result = subprocess.run(["alien-token-info"], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        output = result.stdout.decode()
        # Check that the command ran successfully
        if (
            output.startswith("Error")
            or output.startswith("Warning")
            or "CheckErrorStatus" in output
        ):
            errorstate = True

    # We've successfully queried, so now we look at the result.
    # One possible approach is:
    # If there's no token, we get "No Token found!"
    # If there's a valid token, we get "Token is still valid!" at the end.
    #return ("Token is still valid!" in output)
    # However, we can do better:
    # `alien-token-init` returns a status code of 1 if there is no token or a problem
    # and 0 if there is a valid token.
    return result.returncode == 0

def local_md5(fname: Union[Path, str]) -> str:
    """ Calculate a chunked md5 sum for the file at a given filename.

    Args:
        fname: Path to the file.
    Returns:
        md5 sum of the file.
    """
    # Validation
    fname = Path(fname)

    # Calculate md5 sum
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def grid_md5(gridfile: Union[Path, str]) -> str:
    """ Calculate md5 sum of a file on the grid.

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
        result = subprocess.run(
            ["gbbox", "md5sum", str(gridfile)], stdout = subprocess.PIPE, stderr = subprocess.PIPE, check = True
        )
        gb_out = result.stdout.decode()
        # Check that the command ran successfully
        if (
            gb_out.startswith("Error")
            or gb_out.startswith("Warning")
            or "CheckErrorStatus" in gb_out
        ):
            errorstate = True

    return gb_out.split("\t")[0]

def copy_from_alien(inputfile: Union[Path, str], outputfile: Union[Path, str]) -> bool:
    """ Copy a file using alien_cp.

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
    outputfile.parent.mkdir(mode = 0o755, exist_ok = True, parents = True)
    subprocess.run(
        ["alien_cp", f"alien://{inputfile}", str(outputfile)],
        stdout = subprocess.PIPE, stderr = subprocess.PIPE, check = True
    )

    # Check that the file was copied successfully.
    if outputfile.exists():
        localmd5 = local_md5(outputfile)
        gridmd5 = grid_md5(inputfile)
        #logger.debug(f"MD5local: {localmd5}, MD5grid {gridmd5}")
        if localmd5 != gridmd5:
            logger.error(f"Mismatch in MD5 sum for file {outputfile}")
            # Incorrect MD5 sum - probably a corrupted output file.
            outputfile.unlink()
            return False
        else:
            logger.info(f"Output file {outputfile} copied correctly")
            return True
    else:
        logger.error(f"output file {outputfile} not found")
        return False

def list_alien_dir(input_dir: Union[Path, str]) -> List[str]:
    """ List the files in a directory on AliEn.

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
        process = subprocess.run(["alien_ls", str(input_dir)], stdout = subprocess.PIPE, stderr = subprocess.PIPE)

        # Extract the files from the output.
        errorstate = False
        result: List[str] = []
        if process.returncode == 0:
            # Only process if the command itself didn't report an issue via the return code.
            for d in process.stdout.decode().split("\n"):
                if d.startswith("Error") or d.startswith("Warning"):
                    errorstate = True
                    break
                mydir = d.rstrip().lstrip()
                #if len(mydir) and mydir.isdigit():
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
