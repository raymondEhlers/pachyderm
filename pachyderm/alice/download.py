#!/usr/bin/env python3

""" Download data related to ALICE.

Based on code from Markus Fasel's download train run-by-run output in parallel script.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import abc
import argparse
import itertools
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Type, TypeVar, Union

from pachyderm import yaml
from pachyderm.alice import utils

try:
    import importlib.resources as resources
except ImportError:
    # Try the back ported package.
    import importlib_resources as resources  # type: ignore

logger = logging.getLogger(__name__)

# Work around typing issues in python 3.6
# If only supporting 3.7+, we can add `from __future__ import annotations` and just use the more detailed definition
if TYPE_CHECKING:
    FilePairQueue = queue.Queue[Union["FilePair", None]]
else:
    FilePairQueue = queue.Queue

# Use 4 threads by default in order to keep number of network request at an acceptable level
NTHREADS: int = 4

@dataclass
class FilePair:
    """ Pair for file paths to copy from source to target.

    This also a counter for the number of times that we've tried to copy this file.

    Attributes:
        source: Path to the source file.
        target: Path to the target file.
        n_tries: Number of attempts to copy the source file.
    """
    source: Union[Path, str]
    target: Union[Path, str]
    n_tries: int = 0

    def __post_init__(self) -> None:
        """ Ensure that we actually receive Paths. """
        self.source = Path(self.source)
        self.target = Path(self.target)

class QueueFiller(threading.Thread, abc.ABC):
    """ Fill file pairs into the queue.

    Args:
        q: Queue where the file pairs will be stored.
    """
    def __init__(self, q: FilePairQueue) -> None:
        super().__init__()
        self._queue = q

    def run(self) -> None:
        """ Main entry point called when joining a thread.

        The daughter class should implemented ``_process()`` instead of ``run()``
        so that we can control calls to the processing if necessary.

        Args:
            None.
        Returns:
            None.
        """
        self._process()

    def _wait(self) -> None:
        """ Determine whether to wait before filling into the queue.

        If the pool is full, it will wait until it starts to empty before filling further.

        Note:
            Since the queue blocks when it is full, I'm not sure this is really necessary.
            It is perhaps nice that it backs off, but I don't think that's required.

        Args:
            None.
        Returns:
            None.
        """
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

        Args:
            None.
        Returns:
            None.
        """
        ...

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

    Attributes:
        period: The run period.
        system: The collision system.
        year: Year the data was collected or simulated.
        file_type: Data type - either "ESD" or "AOD".
        search_path: Search path to be utilized via ``alien_ls``.
        filename: Name of the file to download from AliEn.
        selections: User selections provided in the YAML file.
        is_data: True if the DataSet corresponds to real data or simulation.
        data_type: Data type - either "data" or "sim".
    """
    period: str
    system: str
    year: int
    file_type: str
    search_path: Path
    filename: str
    selections: Dict[str, Any] = field(default_factory = dict)

    @property
    def is_data(self) -> bool:
        """ Check whether the DataSet provides real data or simulated data.

        Args:
            None.
        Returns:
            True if the DataSet contains real data.
        """
        return does_period_contain_data(self.period)

    @property
    def data_type(self) -> str:
        """ The data type of either "data" or "sim".

        Args:
            None.
        Returns:
            "data" if the DataSet contains real data, and "sim" if it is simulation.
        """
        return "data" if self.is_data else "sim"

    @classmethod
    def from_specification(cls: Type[_T], period: str, system: str, year: int,
                           file_types: Dict[str, Dict[str, str]], **selections: Any) -> _T:
        """ Initializes the DataSet from a specification stored in a YAML file.

        Args:
            period: The run period.
            system: The collision system.
            year: Year the data was collected or simulated.
            file_types: Specifications that vary for the ESD or AOD files, including
                the search path and the filename.
            selections: User selections provided in the YAML file.
        Returns:
            A DataSet constructed using this information.
        """
        # Validation
        # Ensure that "LHC" is in all caps.
        period = period[:3].upper() + period[3:]
        # Rename variables so that they better fit the expected arguments.
        # Piratically, this means removing the trailing "s"
        for var_name in ["runs", "pt_hard_bins"]:
            # They may not always be defined, so skip if they're not.
            val = selections.pop(var_name, None)
            if val is not None:
                selections[var_name[:-1]] = val
        # Rename "runs" -> "run"
        if does_period_contain_data(period):
            # Need to prepend "000"
            selections["run"] = ["000" + str(r) for r in selections["run"]]

        # Extract the rest of the information from the file type options
        file_type = selections["file_type"]
        options = file_types[file_type]
        search_path = options.get("search_path", "/alice/{data_type}/{year}/{period}/{pt_hard_bin}/{run}/")
        path = Path(search_path)
        filename = options.get("filename", "root_archive.zip")

        # Create the object
        return cls(
            period = period, system = system, year = year, file_type = file_type,
            search_path = path, filename = filename,
            selections = selections
        )

def _extract_dataset_from_yaml(period: str, datasets_path: Optional[Union[Path, str]] = None) -> DataSet:
    """ Extract the datasets from YAML.

    Args:
        period: Run period which we should load.
        datatsets_path: Filename of the YAML configuration file. Default: None,
            in which case, the files will be taken from those defined in the package.
    Returns:
        The dataset extract from the YAML file.
    """
    # Validation
    # This will always be named according to the run period.
    filename = f"{period}.yaml"
    if datasets_path:
        datasets_path = Path(datasets_path) / filename
        with open(datasets_path, "r") as f:
            file_contents = f.read()
    else:
        file_contents = resources.read_text("pachyderm.alice.datasets", filename)
    # Read the YAML
    y = yaml.yaml()
    dataset_information = y.load(file_contents)

    # Setup dataset
    logger.debug(f"parameters: {list(dataset_information['parameters'].keys())}")
    logger.debug(f"selections: {list(dataset_information['selections'].keys())}")
    dataset = DataSet.from_specification(
        period = period, **dataset_information["parameters"], **dataset_information["selections"]
    )

    return dataset

def _combinations_of_selections(selections: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    """ Find all permutations of combinations of selections.

    This is useful for passing the selections as kwargs to a function (perhaps for formatting).
    As a concrete example,

    ```python
    >>> selections = {"a": [1], "b": [2, 3]}
    >>> list(_combinations_of_selections(selections))
    [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}]

    ```

    Note:
        The arguments are validated such that if there is only a single value for a given selection,
        it is converted to a list of length 1. Of course, it will be returned as a single value in
        the arguments.

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

class CopyHandler(threading.Thread):
    def __init__(self, q: FilePairQueue):
        threading.Thread.__init__(self)
        self._queue = q
        self.max_tries = 5
        self._copy_function = utils.copy_from_alien
        # If we want to test, make below the copy function.
        #def random_choice(source: Path, target: Path) -> bool:
        #    import random
        #    logging.debug("Random choice for debugging!")
        #    return random.choice([False, True])
        #self._copy_function = random_choice

    def run(self) -> None:
        """ Copy the files stored into the data pool. """
        while True:
            # This blocks waiting for the next file.
            next_file = self._queue.get()
            logger.debug(f"next_file: {next_file}")
            # We're all done - time to stop.
            if next_file is None:
                # Ensure that it propagates to the other handlers.
                self._queue.put(None)
                break

            # Attempt to copy the file from AliEn
            copy_status = self._copy_function(next_file.source, next_file.target)
            logger.debug(f"Copy success: {copy_status}.")
            # Help out with debugging if needed.
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Sleep for 2 second for debugging purposes.")
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
                    # Although this copying failed and we're going to reinsert this into queue, from
                    # the perspective of the queue, the task was "completed", so we have to note that
                    # the task is done in order for us to be able to join the queue.
                    self._queue.task_done()
                    next_file.n_tries = n_tries
                    self._queue.put(next_file)
            else:
                logger.debug(f"Successfully copied {next_file.source} to {next_file.target}")
                # Help out with debugging if needed.
                if logger.isEnabledFor(logging.DEBUG):
                    time.sleep(2)
                # Notify that the file was copied successfully.
                self._queue.task_done()

def download(queue_filler: QueueFiller, q: FilePairQueue, fewer_threads: bool = False) -> bool:
    """ Actually utilize the queue filler and copy the files.

    Args:
        queue_filler: Class to handle filling the queue.
        q: The queue to be filled.
        fewer_threads: If True, reduce the number of threads by half.
    Returns:
        True if the tasks were successful.
    """
    # Check that we have a valid alien-token.
    # Otherwise, the other calls to the grid will silently fail.
    if not utils.valid_alien_token():
        raise RuntimeError("AliEn token doesn't appear to be valid. Please check!")

    # Start filling the queue
    queue_filler.start()

    workers = []
    n_threads = int(NTHREADS / 2) if fewer_threads else NTHREADS
    logger.info(f"Using {n_threads} threads to download files.")
    for i in range(0, n_threads):
        worker = CopyHandler(q = q)
        worker.start()
        workers.append(worker)

    # Finish up.
    # First, ensure that all of the filers are added to the queue.
    queue_filler.join()
    # Next, we join the queue to ensure that all of the file pairs in it are processed.
    q.join()
    # Next, tell the workers to exit.
    q.put(None)
    # And then wait for them to process the exit signal.
    for worker in workers:
        worker.join()

    return True

class FileListDownloadFiller(QueueFiller):
    """ Download list of file pairs already provided to the task.

    This is a basic task for simple cases when it's easier to just enumerate the list of files to download by hand.

    Note:
        We don't provide an entry point for this class. Instead, we expect others to use it.

    The class can be used via:

    ```python
    # Create and somehow fill in the file_pair_list
    file_pair_list = [
        # ...
    ]
    # Setup the queue and filler, and then start downloading.
    q: download.FilePairQueue = queue.Queue()
    queue_filler = download.FileListDownloadFiller(pairs = file_pair_list, q = q)
    download.download(queue_filler = queue_filler, q = q)
    ```

    Args:
        pairs: File pairs that are externally generated. It's up to the user to determine the input and output paths.
    """
    def __init__(self, pairs: Sequence[FilePair], *args: FilePairQueue, **kwargs: FilePairQueue) -> None:
        super().__init__(*args, **kwargs)
        self._file_pairs = pairs

    def _process(self) -> None:
        for file_pair in self._file_pairs:
            # Only download the file if it doesn't exist or the checksum is incorrect.
            if utils.check_output_file(inputfile=file_pair.source, outputfile=file_pair.target, verbose=False):
                logger.info(f"Output file {file_pair.target} already found - not copying again")
            else:
                logger.info(f"Adding input: {file_pair.source}, output: {file_pair.target}")
                # Add to the queue
                self._queue.put(file_pair)

class DatasetDownloadFiller(QueueFiller):
    """ Fill in files to download from a given DataSet.

    Args:
        dataset: DataSet which provides the properties of the dataset.
        output_dir: Base output directory where the files will be copied. This doesn't specify the entire path.
            Rather, the grid path (except for "alice") is appended to this directory. Maintaining this directory
            structure yields better compatibility with the analysis manager.
        q: Queue where the file pairs should be stored.

    Attributes:
        dataset: DataSet which provides the properties of the dataset.
        output_dir: Base output directory where the files will be copied. This doesn't specify the entire path.
            Rather, the grid path (except for "alice") is appended to this directory. Maintaining this directory
            structure yields better compatibility with the analysis manager.
    """
    def __init__(self, dataset: DataSet, output_dir: Union[Path, str],
                 *args: FilePairQueue, **kwargs: FilePairQueue) -> None:
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.output_dir = Path(output_dir)

    def _process(self) -> None:
        """ Determine and fill in the filenames of the dataset.

        Args:
            None.
        Returns:
            None.
        """
        for properties in _combinations_of_selections(self.dataset.selections):
            # Determine search and output paths
            logger.debug(f"dataset.__dict__: {self.dataset.__dict__}")
            logger.debug(f"properties: {properties}")
            kwargs = self.dataset.__dict__.copy()
            kwargs["data_type"] = self.dataset.data_type
            kwargs.update(properties)
            #search_path = Path(str(self.dataset.search_path).format(**self.dataset.__dict__, **properties))
            #output_dir = Path(str(self.output_dir).format(**self.dataset.__dict__, **properties))
            search_path = Path(str(self.dataset.search_path).format(**kwargs))
            #output_dir = Path(str(self.output_dir).format(**kwargs))
            # We want the output_dir to emulate the path on the grid. We just want to change where it's stored.
            output_dir = self.output_dir / str(search_path).replace("/alice/", "")
            logger.debug(f"search_path: {search_path}, output_dir: {output_dir}")

            max_files_per_selection = properties["n_files_per_selection"]

            grid_files = utils.list_alien_dir(search_path)
            # We are only looking for numbered directories, so we can easy grab these by requiring them to be digits.
            grid_files = [v for v in grid_files if v.isdigit()]
            logger.debug(f"grid_files: {grid_files}")

            for i, directory in enumerate(grid_files):
                # We could receive a ton of files. Only copy as many as the max.
                if i >= max_files_per_selection:
                    break

                # Determine output directory. It will be created if necessary when copying.
                output = output_dir / directory / self.dataset.filename
                if output.exists():
                    logger.info(f"Output file {output} already found - not copying again")
                else:
                    logger.debug(f"Adding input: {search_path / directory / self.dataset.filename}, output: {output}")
                    # Add to the queue
                    self._queue.put(FilePair(
                        search_path / directory / self.dataset.filename, output
                    ))

def download_dataset(period: str, output_dir: Union[Path, str], fewer_threads: bool,
                     datasets_path: Optional[Union[Path, str]] = None) -> List[Path]:
    """ Download files from the given dataset with the provided selections.

    Args:
        period: Name of the period to be downloaded.
        output_dir: Path to where the data should be stored.
        fewer_threads: If True, reduce the number of threads by half.
        dataset_config_filename: Filename of the configuration file. Default: None,
            in which case, the files will be taken from those defined in the package.
    Returns:
        None.
    """
    # Validation
    output_dir = Path(output_dir)
    if datasets_path:
        datasets_path = Path(datasets_path)

    # Setup the dataset
    dataset = _extract_dataset_from_yaml(period = period, datasets_path = datasets_path)

    # Setup
    q: FilePairQueue = queue.Queue()
    queue_filler = DatasetDownloadFiller(
        dataset = dataset, output_dir = output_dir,
        q = q,
    )
    download(queue_filler = queue_filler, q = q, fewer_threads = fewer_threads)

    # Return the files that are stored corresponding to this period.
    period_specific_dir = output_dir / dataset.data_type / str(dataset.year) / dataset.period
    period_files = sorted(Path(period_specific_dir).glob(f"**/{dataset.filename}"))
    logger.info(f"period_specific_dir: {period_specific_dir}, number of files: {len(period_files)}")
    # Write out the file list
    filelist = Path(output_dir) / "filelists" / f"{dataset.period}{dataset.file_type}.txt"
    filelist.parent.mkdir(exist_ok=True, parents=True)
    # Add the suffix to access the ROOT file if it's contained in a zip archive.
    suffix = ""
    if ".zip" in dataset.filename:
        suffix = "#AliAOD.root" if dataset.file_type == "AOD" else "#AliESDs.root"
    with open(filelist, "w") as f:
        # One file per line.
        f.write("\n".join([f"{p}{suffix}" for p in period_files]))
    return period_files

def run_dataset_download() -> None:
    """ Entry point for download a dataset.

    Example invocation:

    ```bash
    $ downloadALICEDataset -p lhc16j5 -o /alice/
    ```
    """
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog="downloadALICEDataset",
        description="Download an ALICE dataset in parallel",
    )
    parser.add_argument(
        "-p", "--period", type=str, default="",
        help="Run period (i.e. dataset) to download. For example, lhc16j5.",
    )
    parser.add_argument(
        "-o", "--outputdir", type=str, default="alice",
        help="Base output directory. [default: 'alice']",
    )
    parser.add_argument(
        "-f", "--fewerThreads", action="store_true", default=False,
        help="Decrease the number of threads by half. Default: using 4 threads."
    )
    parser.add_argument(
        "-d", "--datasets", type=str, default=None, metavar="PATH",
        help="Path to the datasets directory. Default: unset."
    )
    args = parser.parse_args()
    output = download_dataset(
        period = args.period,
        output_dir = args.outputdir,
        fewer_threads = args.fewerThreads,
        datasets_path = args.datasets,
    )
    print(output)

class RunByRunTrainOutputFiller(QueueFiller):
    """ Fill in files to download run-by-run train output.

    Args:
        output_dir: Path to where the train output should be stored.
        train_run: Train number.
        legotrain: Name of the LEGO train, such as "PWGJE/Jets_EMC_pp".
        dataset: Name of the dataset, such as "LHC17p".
        recpass: Name of the reconstruction pass, such as "pass1" or "pass1_FAST".
        aodprod: Name of the AOD production, such as "AOD208".
    """
    def __init__(self, output_dir: Union[Path, str], train_run: int, legotrain: str, dataset: str,
                 pt_hard_bin: int, recpass: str, aodprod: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # We want the output to be stored inside a directory labeled by the train number.
        self._output_dir = Path(output_dir) / str(train_run)
        self._train_run = train_run
        self._lego_train = legotrain
        self._dataset = dataset
        self._pt_hard_bin = pt_hard_bin
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

    @property
    def _grid_base(self) -> Path:
        datatag = "data" if self._is_data else "sim"
        grid_base: Path = Path("/alice") / datatag / str(self._year) / self._dataset
        if self._pt_hard_bin:
            grid_base = grid_base / str(self._pt_hard_bin)

        return grid_base

    def _process(self) -> None:
        logger.info(f"Searching output files in train directory {self._grid_base}")
        for r in utils.list_alien_dir(self._grid_base):
            logger.info(f"Checking run {r}")
            if not r.isdigit():
                continue

            run_dir = self._grid_base / r
            if self._is_data:
                run_dir = run_dir / self._reconstruction_pass
            if self._aod_production:
                run_dir = run_dir / self._aod_production
            run_output_dir = self._output_dir / r
            logger.info(f"run_dir: {run_dir}")

            data = utils.list_alien_dir(run_dir)
            if not self._lego_train.split("/")[0] in data:
                logger.error(f"PWG directory {self._lego_train.split('/')[0]} was not found for run {r}")
                continue

            lego_trains_dir = run_dir / self._lego_train.split("/")[0]
            lego_trains = utils.list_alien_dir(lego_trains_dir)
            if not self._lego_train.split("/")[1] in lego_trains:
                logger.error(
                    f"Train {self._lego_train.split('/')[1]} not found in PWG directory for run {r}"
                )
                continue

            train_base = run_dir / self._lego_train
            train_runs = utils.list_alien_dir(train_base)
            train_dir = [
                x for x in train_runs if self._extract_train_ID(x) == self._train_run
            ]
            if not len(train_dir):
                logger.error(f"Train run {self._train_run} not found for run {r}")
                continue

            full_train_dir = train_base / train_dir[0]
            train_files = utils.list_alien_dir(full_train_dir)
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

def download_run_by_run_train_output(outputpath: Union[Path, str],
                                     trainrun: int, legotrain: str, dataset: str, pt_hard_bin: int,
                                     recpass: str, aodprod: str, fewer_threads: bool) -> None:
    """ Download run-by-run train output for the given arguments.

    Args:
        output_dir: Path to where the train output should be stored.
        train_run: Train number.
        legotrain: Name of the LEGO train, such as "PWGJE/Jets_EMC_pp".
        dataset: Name of the dataset, such as "LHC17p".
        pt_hard_bin: Number of the pt hard bin to download. Only meaningful for MC.
        recpass: Name of the reconstruction pass, such as "pass1" or "pass1_FAST".
        aodprod: Name of the AOD production, such as "AOD208".
        fewer_threads: If True, reduce the number of threads by half.
    Returns:
        None.
    """
    q: FilePairQueue = queue.Queue(maxsize = 1000)
    logger.info(f"Checking dataset {dataset} for train with ID {trainrun} ({legotrain})")

    queue_filler = RunByRunTrainOutputFiller(
        outputpath, trainrun,
        legotrain, dataset, pt_hard_bin,
        recpass, aodprod if len(aodprod) > 0 else "",
        q = q,
    )
    download(queue_filler = queue_filler, q = q, fewer_threads = fewer_threads)

def run_download_run_by_run_train_output() -> None:
    """ Entry point for download run-by-run train output.

    Example invocation:

    ```bash
    $ downloadALICERunByRun -o /alice/trains/pp/ -t 1744 -l PWGJE/Jets_EMC_pp -d LHC17p -p pass1_CENT_woSDD -a AOD208
    ```
    """
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog="downloadAliceRunByRun",
        description="Download run-by-run LEGO train outputs",
    )
    parser.add_argument(
        "-o", "--outputdir", metavar="OUTPUTDIR",
        help="Path where to store the output files run-by-run",
    )
    parser.add_argument(
        "-t", "--trainrun", metavar="TRAINRUN", type=int,
        help="ID of the train run (number is sufficient, time stamp not necessary)",
    )
    parser.add_argument(
        "-l", "--legotrain", metavar="LEGOTRAIN",
        help="Name of the lego train (i.e. PWGJE/Jets_EMC_pPb)",
    )
    parser.add_argument("-d", "--dataset", metavar="DATASET", help="Name of the dataset")
    parser.add_argument(
        "--ptHardBin", metavar="BIN",
        help="Pt hard bin (only meaningful in the case of pt-hard binned MC)"
    )
    parser.add_argument(
        "-p", "--recpass", type=str, default="pass1",
        help="Reconstruction pass (only meaningful in case of data) [default: pass1]",
    )
    parser.add_argument(
        "-a", "--aod", type=str, default="",
        help="Dedicated AOD production (if requested) [default: not set]",
    )
    parser.add_argument(
        "-f", "--fewerThreads", action="store_true", default=False,
        help="Decrease the number of threads by half."
    )
    args = parser.parse_args()
    download_run_by_run_train_output(
        args.outputdir,
        args.trainrun, args.legotrain,
        args.dataset, args.ptHardBin, args.recpass, args.aod,
        args.fewerThreads,
    )
