#!/usr/bin/env python3

""" Copy some set of files from AliEn.

Uses some code from Markus' download train run-by-run output script.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import abc
#import argparse
import hashlib
import itertools
import logging
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Type, TypeVar, Union

from pachyderm import yaml

logger = logging.getLogger(__name__)

def local_md5(fname: Union[Path, str]) -> str:
    """ Calculate md5 sum for the file at a given filename.

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
        #_, gb_out = subprocess.getstatusoutput(f"gbbox md5sum {gridfile}")
        result = subprocess.run(["gbbox", "md5sum", str(gridfile)], capture_output = True, check = True)
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
    #if not os.path.exists(os.path.dirname(outputfile)):
    #    os.makedirs(os.path.dirname(outputfile), 0o755)
    outputfile.parent.mkdir(mode = 0o755, exist_ok = True, parents = True)
    #subprocess.call(["alien_cp", f"alien://{inputfile}", outputfile])
    subprocess.run(["alien_cp", f"alien://{inputfile}", str(outputfile)], capture_output = True, check = True)

    # Check that the file was copied successfully.
    if outputfile.exists():
        localmd5 = local_md5(outputfile)
        gridmd5 = grid_md5(inputfile)
        logger.debug(f"MD5local: {localmd5}, MD5grid {gridmd5}")
        if localmd5 != gridmd5:
            logger.error(f"Mismatch in MD5 sum for file {outputfile}")
            # incorrect MD5sum, outputfile probably corrupted
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
        #dirs = subprocess.getstatusoutput(f"alien_ls {input_dir}")
        process = subprocess.run(["alien_ls", str(input_dir)], capture_output = True, check = True)
        logger.debug("Found files in AliEn.")
        #logger.debug(f"process: {process}")
        #logger.debug(f"process.stdout: {process.stdout}")
        #logger.debug(f"process.stderr: {process.stderr}")

        # Extract the files from the output.
        errorstate = False
        result: List[str] = []
        for d in process.stdout.decode().split("\n"):
            if d.startswith("Error") or d.startswith("Warning"):
                errorstate = True
                break
            mydir = d.rstrip().lstrip()
            if len(mydir) and mydir.isdigit():
                result.append(mydir)

        # If we haven't successes, let's try again.
        if errorstate:
            continue

    return result

@dataclass
class FilePair:
    source: Union[Path, str]
    target: Union[Path, str]
    n_tries: int = 0

    def __post_init__(self) -> None:
        """ Ensure that we actually receive Paths. """
        self.source = Path(self.source)
        self.target = Path(self.target)

def does_period_contain_data(period: str) -> bool:
    """ Does the given period contain data or simulation?

    If the period is 6 characters long, then it's data.

    Args:
        period: Run period to be checked.
    Returns:
        True if the period contains data.
    """
    if len(period) == 6:
        return True
    return False

_T = TypeVar("_T", bound = "DataSet")

@dataclass
class DataSet:
    """ Contains dataset information necessary to download the associated files.

    """
    period: str
    system: str
    data_type: str
    year: int
    file_type: str
    search_path: Path
    filename: str
    selections: Dict[str, Any] = field(default_factory = dict)

    @property
    def is_data(self) -> bool:
        return does_period_contain_data(self.period)

    @classmethod
    def from_specification(cls: Type[_T], period: str, system: str, data_type: str, year: int,
                           file_types: Dict[str, Dict[str, str]], **selections: Any) -> _T:
        # Validation
        # Ensure that "LHC" is in all caps.
        period = period[:3].upper() + period[3:]
        if not does_period_contain_data(period):
            # Need to prepend "000"
            #selections["run"] = ["000" + str(r) for r in selections["run"]]
            ...
        # Extract the rest of the information from the file type options
        file_type = selections["file_type"]
        options = file_types[file_type]
        search_path = options.get("search_path", "/alice/{data_type}/{year}/{period}/{pt_hard_bin}/{run}/")
        path = Path(search_path)
        filename = options.get("filename", "root_archive.zip")

        # Create the object
        return cls(
            period = period, system = system, data_type = data_type, year = year, file_type = file_type,
            search_path = path, filename = filename,
            selections = selections
        )

def _extract_datasets_from_yaml(filename: Union[Path, str]) -> yaml.DictLike:
    """ Extract the datasets from YAML.

    Args:
        filename: Path to the YAML file.
    Returns:
        The datasets extract from the YAML file.
    """
    # Validation
    filename = Path(filename)

    y = yaml.yaml()
    with open(filename, "r") as f:
        datasets = y.load(f)

    return datasets

#def _validate_input(run: Optional[int] = None, runs: Optional[List[int]] = None) -> bool:
#    # Validation
#    if run is None and runs is None:
#        raise ValueError("Must pass either a single run or list of runs.")
#    # We want to proceed with runs regardless of what was passed. So if a single
#    # run was passed, we convert it into runs list of length 1.
#    if runs is None:
#        # Help out mypy...
#        assert run is not None
#        runs = [run]

def _combinations(selections: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    """ Combine all permutations of selections.

    Args:
        selections: Selections from the dataset.
    Returns:
        Iterable whose elements contain dicts containing the arguments for each combination.
    """
    # Validation
    sels = {k: [v] if not isinstance(v, list) else v for k, v in selections.items()}
    # Return all combinations in a dict, such that they can be used as kwargs
    # See: https://stackoverflow.com/a/15211805
    return (dict(zip(sels, v)) for v in itertools.product(*sels.values()))

def download_dataset(dataset_config_filename: str, dataset_name: str, output_path: str) -> str:
    """ Download files from the given dataset with the provided selections.

    """
    data_pool: queue.Queue[Union[FilePair, None]] = queue.Queue()
    pool_filler = DatasetDownloadFiller(
        config_filename = dataset_config_filename, dataset = dataset_name, output_path = output_path,
        data_pool = data_pool,
    )
    pool_filler.start()

    workers = []
    for i in range(0, 1):
        #worker = DummyHandler(data_pool=data_pool)
        worker = CopyHandler(data_pool=data_pool)
        worker.start()
        workers.append(worker)

    pool_filler.join()
    # Finish up.
    data_pool.join()
    # Tell the workers to exit.
    data_pool.put(None)
    for worker in workers:
        worker.join()

    # TODO: Generate file list
    return ""

class PoolFiller(threading.Thread, abc.ABC):
    def __init__(self, data_pool: queue.Queue[Union[FilePair, None]]) -> None:
        super().__init__()
        self._queue = data_pool

    def run(self) -> None:
        """ Main entry point called when joining a thread. """
        self._process()

    def _wait(self) -> None:
        """ If the pool is full, wait until it starts to empty before filling further. """
        # If less than the max pool size, no need to wait.
        if self._queue.qsize() < self._queue.maxsize:
            return None
        # Pool full, wait until half empty
        empty_limit = self._queue.maxsize / 2
        while self._queue.qsize() > empty_limit:
            time.sleep(5)

    def _process(self) -> None:
        """ Find and fill files into the queue.

        To be implemented by the daughter classes.
        """
        ...

class DummyHandler(threading.Thread):
    def __init__(self, data_pool: queue.Queue[Union[FilePair, None]]):
        threading.Thread.__init__(self)
        self._queue = data_pool
        self.max_tries = 5

    def run(self) -> None:
        """ Dummy to test copying the files stored into the data pool. """
        import random
        while True:
            # This blocks waiting for the next file.
            next_file = self._queue.get()
            # We're all done - time to stop.
            logger.debug(f"next_file: {next_file}")
            if next_file is None:
                # Ensure that it propagates to the other tasks.
                self._queue.put(None)
                break

            # Attempt to copy the file from AliEn
            copy_status = random.choice([False, True])
            logger.debug(f"Success: {copy_status}. Sleep for a second.")
            time.sleep(2)
            # Deal with failures.
            if not copy_status:
                # Put file back in the queue in case of copy failure.
                # Only allow for a maximum amount of copy tries
                n_tries = next_file.n_tries
                n_tries += 1
                if n_tries >= self.max_tries:
                    logger.error(f"File {next_file.source} failed copying in {self.max_tries} tries - giving up")
                else:
                    logger.error(f"File {next_file.source} failed copying ({n_tries}/{self.max_tries}) "
                                 "- re-inserting into the pool ...")
                    self._queue.task_done()
                    next_file.n_tries = n_tries
                    self._queue.put(next_file)
            else:
                # Notify that the file was copied successfully.
                logger.debug(f"Successfully copied {next_file.source} to {next_file.target}")
                time.sleep(5)
                self._queue.task_done()

class CopyHandler(threading.Thread):
    def __init__(self, data_pool: queue.Queue[Union[FilePair, None]]):
        threading.Thread.__init__(self)
        self._queue = data_pool
        self.max_tries = 5

    def run(self) -> None:
        """ Copy the files stored into the data pool. """
        while True:
            # This blocks waiting for the next file.
            next_file = self._queue.get()
            # We're all done - time to stop.
            if next_file is None:
                # Ensure that it propagates to the other handlers.
                self._queue.put(None)
                break

            # Attempt to copy the file from AliEn
            copy_status = copy_from_alien(next_file.source, next_file.target)
            # TEMP
            logger.debug(f"Copy success: {copy_status}. Sleep for a second.")
            time.sleep(2)
            # ENDTEMP
            # Deal with failures.
            if not copy_status:
                # Put file back in the queue in case of copy failure.
                # Only allow for a maximum amount of copy tries
                n_tries = next_file.n_tries
                n_tries += 1
                if n_tries >= self.max_tries:
                    logger.error(f"File {next_file.source} failed copying in {self.max_tries} tries - giving up")
                else:
                    logger.error(f"File {next_file.source} failed copying ({n_tries}/{self.max_tries}) "
                                 "- re-inserting into the pool ...")
                    # Although this copying failed and we're going to reinsert this into queue, from
                    # the perspective of the queue, the task was "completed", so we have to note that
                    # the task is done in order for us to be able to join the queue.
                    self._queue.task_done()
                    next_file.n_tries = n_tries
                    self._queue.put(next_file)
            else:
                # TEMP
                logger.debug(f"Successfully copied {next_file.source} to {next_file.target}")
                time.sleep(2)
                # ENDTEMP
                # Notify that the file was copied successfully.
                self._queue.task_done()

class RunByRunTrainOutputFiller(PoolFiller):
    def __init__(self, output_dir: Union[Path, str], train_run: int, legotrain: str, dataset: str,
                 recpass: str, aodprod: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._output_dir = Path(output_dir)
        self._train_run = train_run
        self._lego_train = legotrain
        self._dataset = dataset
        self._reconstruction_pass = recpass
        self._aod_production = aodprod

    @property
    def _year(self) -> int:
        if not self._dataset.startswith("LHC"):
            return 0
        return int(self._dataset[3:5]) + 2000

    @property
    def _is_data(self) -> bool:
        return does_period_contain_data(self._dataset)

    def _extract_train_ID(self, idstring: str) -> int:
        trainid = idstring.split("_")[0]
        if trainid.isdigit():
            return int(trainid)
        return 0

    def _process(self) -> None:
        datatag = "data" if self._is_data else "sim"
        grid_base: Path = Path("/alice") / datatag / str(self._year) / self._dataset
        logger.info(f"Searching output files in train directory {grid_base}")
        for r in list_alien_dir(grid_base):
            logger.info(f"Checking run {r}")
            if not r.isdigit():
                continue

            run_dir = grid_base / r
            if self._is_data:
                run_dir = run_dir / self._reconstruction_pass
            if self._aod_production:
                run_dir = run_dir / self._aod_production
            run_output_dir = self._output_dir / r
            logger.info(f"run_dir: {run_dir}")

            data = list_alien_dir(run_dir)
            if not self._lego_train.split("/")[0] in data:
                logger.error(f"PWG dir not found four run {r}")
                continue

            lego_trains_dir = run_dir / self._lego_train.split("/")[0]
            lego_trains = list_alien_dir(lego_trains_dir)
            if not self._lego_train.split("/")[1] in lego_trains:
                logger.error(
                    f"Train {self._lego_train.split('/')[1]} not found in pwg dir for run {r}"
                )
                continue

            train_base = run_dir / self._lego_train
            train_runs = list_alien_dir(train_base)
            train_dir = [
                x for x in train_runs if self._extract_train_ID(x) == self._train_run
            ]
            if not len(train_dir):
                logger.error(f"Train run {self._train_run} not found for run {r}")
                continue

            full_train_dir = train_base / train_dir[0]
            train_files = list_alien_dir(full_train_dir)
            if "AnalysisResults.root" not in train_files:
                logger.info(
                    f"Train directory {full_train_dir} doesn't contain AnalysisResults.root. Skipping run {r}..."
                )
            else:
                inputfile = full_train_dir / "AnalysisResults.root"
                outputfile = run_output_dir / "AnalysisResults.root"
                if outputfile.exists():
                    logger.info(f"Output file {outputfile} already found - not copying again")
                else:
                    logger.info(f"Copying {inputfile} to {outputfile}")
                    self._wait()
                    self._queue.put(FilePair(inputfile, outputfile))

class DatasetDownloadFiller(PoolFiller):
    def __init__(self, config_filename: Union[Path, str], dataset: str, output_path: Union[Path, str],
                 *args: queue.Queue[Union[FilePair, None]], **kwargs: queue.Queue[Union[FilePair, None]]) -> None:
        super().__init__(*args, **kwargs)
        self.config_filename = Path(config_filename)
        self.dataset = dataset
        self.output_path = Path(output_path)

    def _process(self) -> None:
        # Setup dataset
        datasets = _extract_datasets_from_yaml(filename = self.config_filename)
        try:
            dataset_information = datasets[self.dataset]
        except KeyError as e:
            raise KeyError(f"Dataset {self.dataset} not found. Must specify it in the configuration!") from e
        logger.debug(f"parameters: {list(dataset_information['parameters'].keys())}")
        logger.debug(f"selections: {list(dataset_information['selections'].keys())}")
        dataset = DataSet.from_specification(
            period = self.dataset, **dataset_information["parameters"], **dataset_information["selections"]
        )

        # Setup downloads
        for j, properties in enumerate(_combinations(dataset.selections)):
            # TEMP
            if j > 2:
                break
            # Determine search and output paths
            logger.debug(f"dataset.__dict__: {dataset.__dict__}")
            logger.debug(f"properties: {properties}")
            kwargs = dataset.__dict__.copy()
            kwargs.update(properties)
            #search_path = Path(str(dataset.search_path).format(**dataset.__dict__, **properties))
            #output_path = Path(str(self.output_path).format(**dataset.__dict__, **properties))
            search_path = Path(str(dataset.search_path).format(**kwargs))
            output_path = Path(str(self.output_path).format(**kwargs))
            logger.debug(f"search_path: {search_path}, output_path: {output_path}")

            grid_files = list_alien_dir(search_path)
            # We are only looking for numbered directories, so we can easy grab these by requiring them to be digits.
            grid_files = [v for v in grid_files if v.isdigit()]

            for i, directory in enumerate(grid_files):
                # TEMP!
                if i > 0:
                    break
                # Determine output directory. It will be created if necessary when copying.
                output = output_path / directory / dataset.filename
                logger.debug(f"Adding input: {search_path / directory / dataset.filename}, output: {output}")
                # Add to the queue
                self._queue.put(FilePair(
                    search_path / directory / dataset.filename, output
                ))

def fetchtrainparallel(outputpath: Union[Path, str], trainrun: int, legotrain: str, dataset: str,
                       recpass: str, aodprod: str) -> None:
    data_pool: queue.Queue[Union[FilePair, None]] = queue.Queue(maxsize = 1000)
    logger.info(f"Checking dataset {dataset} for train with ID {trainrun} ({legotrain})")

    pool_filler = RunByRunTrainOutputFiller(
        outputpath, trainrun,
        legotrain, dataset,
        recpass, aodprod if len(aodprod) > 0 else "",
        data_pool = data_pool,
    )
    pool_filler.start()

    workers = []
    # use 4 threads in order to keep number of network request at an acceptable level
    for i in range(0, 4):
        worker = CopyHandler(data_pool=data_pool)
        worker.start()
        workers.append(worker)

    pool_filler.join()
    for worker in workers:
        worker.join()

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(
    #    prog="fetchTrainRunByRunParallel",
    #    description="Tool to get runwise train output",
    #)
    #parser.add_argument(
    #    "outputpath",
    #    metavar="OUTPUTPATH",
    #    help="Path where to store the output files run-by-run",
    #)
    #parser.add_argument(
    #    "trainrun",
    #    metavar="TRAINRUN",
    #    type=int,
    #    help="ID of the train run (number is sufficient, time stamp not necessary)",
    #)
    #parser.add_argument(
    #    "legotrain",
    #    metavar="LEGOTRAIN",
    #    help="Name of the lego train (i.e. PWGJE/Jets_EMC_pPb)",
    #)
    #parser.add_argument("dataset", metavar="DATASET", help="Name of the dataset")
    #parser.add_argument(
    #    "-p",
    #    "--recpass",
    #    type=str,
    #    default="pass1",
    #    help="Reconstruction pass (only meaningful in case of data) [default: pass1]",
    #)
    #parser.add_argument(
    #    "-a",
    #    "--aod",
    #    type=str,
    #    default="",
    #    help="Dedicated AOD production (if requested) [default: not set]",
    #)
    #args = parser.parse_args()
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.DEBUG)
    #fetchtrainparallel(
    #    args.outputpath,
    #    args.trainrun, args.legotrain,
    #    args.dataset, args.recpass, args.aod,
    #)

    download_dataset(
        dataset_config_filename = "pachyderm/alien/dataset.yaml",
        dataset_name = "lhc16j5",
        output_path = "alice/{data_type}/{year}/{period}/{pt_hard_bin}/{run}/AOD{production_number}",
    )
