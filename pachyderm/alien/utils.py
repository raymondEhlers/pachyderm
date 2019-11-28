#!/usr/bin/env python3

""" Copy some set of files from AliEn.

Uses some code from Markus' download train run-by-run output script.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import abc
import argparse
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
        result = subprocess.run(f"gbbox md5sum {gridfile}", capture_output = True)
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
    outputfile.parent.mkdir(mode = 0o755, exist_ok = True)
    #subprocess.call(["alien_cp", f"alien://{inputfile}", outputfile])
    subprocess.run(["alien_cp", f"alien://{inputfile}", outputfile])

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

def list_alien_dir(inputdir: Union[Path, str]) -> List[str]:
    """ List the files in a directory on AliEn.

    The function must be robust against error states which it can only get from the stdout. As
    long as the request ends in error state it should retry.

    Args:
        inputdir: Path to the directory on AliEn.
    Returns:
        List of files on AliEn in the given directory.
    """
    # Validation
    inputdir = Path(inputdir)

    # Search for the files.
    errorstate = True
    while errorstate:
        # Grab the list of files from alien via alien_ls
        logger.debug("Searching for files on AliEn...")
        #dirs = subprocess.getstatusoutput(f"alien_ls {inputdir}")
        process = subprocess.run(f"alien_ls {inputdir}")
        logger.debug("Found files in AliEn.")

        # Extract the files from the output.
        errorstate = False
        result: List[str] = []
        for d in process.stdout.decode().split("\n"):
            if d.startswith("Error") or d.startswith("Warning"):
                errorstate = True
                break
            mydir = d.rstrip().lstrip()
            if len(mydir):
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

_T = TypeVar("_T", bound = "DataSet")

@dataclass
class DataSet:
    """ Contains dataset information necessary to download the associated files.

    """
    system: str
    data_type: str
    year: int
    file_type: str
    search_path: Path
    filename: str
    selections: Dict[str, Any] = field(default_factory = dict)

    @classmethod
    def from_specification(cls: Type[_T], system: str, data_type: str, year: int, file_type: str, file_type_options: Dict[str, Dict[str, str]], **selections: Any) -> _T:
        # Extract the rest of the information from the file type options
        options = file_type_options[file_type]
        search_path = options.get("search_path", "/alice/${data}/${year}/${period}/${ptHardBin}/${run}/")
        path = Path(search_path)
        filename = options.get("filename", "root_archive.zip")

        # Create the object
        return cls(
            system = system, data_type = data_type, year = year, file_type = file_type,
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

def download_dataset(dataset_config_filename: str, dataset_name: str) -> str:
    """ Download files from the given dataset with the provided selections.

    """
    # Setup dataset
    datasets = _extract_datasets_from_yaml(filename = dataset_config_filename)
    try:
        dataset_information = datasets[dataset_name]
    except KeyError as e:
        raise KeyError(f"Dataset {dataset_name} not found. Must specify it in the configuration!") from e
    dataset = DataSet.from_specification(**dataset_information["parameters"], **dataset_information["selections"])

    data_pool: queue.Queue[FilePair] = queue.Queue()
    pool_filler = PoolFiller(data_pool = data_pool)

    workers = []

    # Setup downloads
    for properties in _combinations(dataset.selections):
        worker = CopyHandler(data_pool=data_pool, pool_filler = pool_filler)
        #worker.download(dataset, properties)
        workers.append(worker)

    # Perform the actual downloads.
    for worker in workers:
        worker.join()

    # Generate file list
    return ""

class PoolFiller(threading.Thread, abc.ABC):
    def __init__(self, data_pool: queue.Queue[FilePair], max_pool_size: int = 1000) -> None:
        super().__init__()
        self._queue = data_pool
        self._max_pool_size = max_pool_size
        self.active = False

    def run(self) -> None:
        """ Main entry point called when joining a thread. """
        self.active = True
        self._process()
        self.active = False

    def _wait(self) -> None:
        """ If the pool is full, wait until it starts to empty before filling further. """
        # If less than the max pool size, no need to wait.
        #if self.__datapool.getpoolsize() < self.__maxpoolsize:
        if self._queue.qsize() < self._max_pool_size:
            return None
        # Pool full, wait until half empty
        empty_limit = self._max_pool_size / 2
        while self._queue.qsize() > empty_limit:
            time.sleep(5)

    def _process(self) -> None:
        """ Find and fill files into the queue.

        To be implemented by the daughter classes.
        """
        ...

class CopyHandler(threading.Thread):
    def __init__(self, data_pool: queue.Queue[FilePair], pool_filler: PoolFiller):
        threading.Thread.__init__(self)
        self._data_pool = data_pool
        self._pool_filler = pool_filler
        self.max_trials = 5

    def wait_for_work(self) -> None:
        """ Wait for work to be available. """
        # If there's something in the pool, then return immediately and start working.
        if self._data_pool.qsize():
            return None
        # If the Pool filler isn't active, then go to work so we can finish up.
        if not self._pool_filler.active:
            return None
        # If nothing in the pool, there's nothing to do.
        while not self._data_pool.qsize():
            # If the pool filler becomes inactive, then break so we can finish up.
            if not self._pool_filler.active:
                break
            # Wait for new files to be added into the pool.
            time.sleep(5)
        return None

    def run(self) -> None:
        """ Copy the files stored into the data pool. """
        has_work = True
        while has_work:
            self.wait_for_work()
            next_file = self._data_pool.get()
            # If the pool is empty, then next_file will be empty.
            if next_file:
                copy_status = copy_from_alien(next_file.source, next_file.target)
                if not copy_status:
                    # put file back on the pool in case of copy failure
                    # only allow for a maximum amount of copy trials
                    trials = next_file.n_tries
                    trials += 1
                    if trials >= self.max_trials:
                        logger.error(f"File {next_file.source} failed copying in {self.max_trials} trials - giving up")
                    else:
                        logger.error(f"File {next_file.source} failed copying ({trials}/{self.max_trials}) "
                                     "- re-inserting into the pool ...")
                        next_file.n_tries = trials
                        self._data_pool.put(next_file)
            if not self._pool_filler.active:
                # if pool is empty exit, else keep thread alive for remaining files
                if not self._data_pool.qsize():
                    has_work = False

class RunByRunTrainOutputFiller(PoolFiller):
    def __init__(
        self, output_dir: Union[Path, str], train_run: int, legotrain: str, dataset: str, recpass: str, aodprod: str, *args: Any, **kwargs: Any
    ):
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
        if len(self._dataset) == 6:
            return True
        return False

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
                logger.error(
                    f"Train run {self._train_run} not found for run {r}"
                )
                continue
            full_train_dir = train_base / train_dir[0]
            trainfiles = list_alien_dir(full_train_dir)
            if "AnalysisResults.root" not in trainfiles:
                logger.info(
                    "Train directory %s doesn't contain AnalysisResults.root",
                    full_train_dir,
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
    def __init__(self, data_pool: queue.Queue[FilePair], dataset: str, ) -> None:
        ...

def fetchtrainparallel(outputpath: Union[Path, str], trainrun: int, legotrain: str, dataset: str,
                       recpass: str, aodprod: str) -> None:
    data_pool: queue.Queue[FilePair] = queue.Queue()
    logger.info(f"Checking dataset {dataset} for train with ID {trainrun} ({legotrain})")

    pool_filler = RunByRunTrainOutputFiller(
        outputpath, trainrun,
        legotrain, dataset,
        recpass, aodprod if len(aodprod) > 0 else "",
        data_pool = data_pool, max_pool_size = 1000,
    )
    #poolfiller.setdatapool(datapool)
    #poolfiller.setalientool(alienhelper)
    pool_filler.start()

    workers = []
    # use 4 threads in order to keep number of network request at an acceptable level
    for i in range(0, 4):
        worker = CopyHandler(data_pool=data_pool, pool_filler=pool_filler)
        #worker.setdatapool(datapool)
        #worker.setpoolfiller(poolfiller)
        #worker.setalienhelper(alienhelper)
        worker.start()
        workers.append(worker)

    pool_filler.join()
    for worker in workers:
        worker.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="fetchTrainRunByRunParallel",
        description="Tool to get runwise train output",
    )
    parser.add_argument(
        "outputpath",
        metavar="OUTPUTPATH",
        help="Path where to store the output files run-by-run",
    )
    parser.add_argument(
        "trainrun",
        metavar="TRAINRUN",
        type=int,
        help="ID of the train run (number is sufficient, time stamp not necessary)",
    )
    parser.add_argument(
        "legotrain",
        metavar="LEGOTRAIN",
        help="Name of the lego train (i.e. PWGJE/Jets_EMC_pPb)",
    )
    parser.add_argument("dataset", metavar="DATASET", help="Name of the dataset")
    parser.add_argument(
        "-p",
        "--recpass",
        type=str,
        default="pass1",
        help="Reconstruction pass (only meaningful in case of data) [default: pass1]",
    )
    parser.add_argument(
        "-a",
        "--aod",
        type=str,
        default="",
        help="Dedicated AOD production (if requested) [default: not set]",
    )
    args = parser.parse_args()
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    fetchtrainparallel(
        args.outputpath,
        args.trainrun, args.legotrain,
        args.dataset, args.recpass, args.aod,
    )
