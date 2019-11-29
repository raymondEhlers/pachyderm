#!/usr/bin/env python3

""" Download data related to ALICE.

Based on code from Markus Fasel's download train run-by-run output in parallel script.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import abc
import argparse
import itertools
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Type, TypeVar, Union

from pachyderm import yaml
from pachyderm.alice import utils

try:
    import importlib.resources as resources
except ImportError:
    # Try the back ported package.
    import importlib_resources as resources  # type: ignore

logger = logging.getLogger(__name__)

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
    def __init__(self, q: queue.Queue[Union[FilePair, None]]) -> None:
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
            selections[var_name[:-1]] = selections.pop(var_name)
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

class DummyHandler(threading.Thread):
    def __init__(self, q: queue.Queue[Union[FilePair, None]]):
        threading.Thread.__init__(self)
        self._queue = q
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
    def __init__(self, q: queue.Queue[Union[FilePair, None]]):
        threading.Thread.__init__(self)
        self._queue = q
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
            copy_status = utils.copy_from_alien(next_file.source, next_file.target)
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
                 *args: queue.Queue[Union[FilePair, None]], **kwargs: queue.Queue[Union[FilePair, None]]) -> None:
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

def download_dataset(period: str, output_dir: Union[Path, str], datasets_path: Optional[Union[Path, str]] = None) -> List[Path]:
    """ Download files from the given dataset with the provided selections.

    Args:
        period: Name of the period to be downloaded.
        output_dir: Path to where the data should be stored.
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
    q: queue.Queue[Union[FilePair, None]] = queue.Queue()
    queue_filler = DatasetDownloadFiller(
        dataset = dataset, output_dir = output_dir,
        q = q,
    )
    queue_filler.start()

    workers = []
    for i in range(0, 1):
        #worker = DummyHandler(q = q)
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

    # Return the files that are stored corresponding to this period.
    period_specific_dir = output_dir / dataset.data_type / str(dataset.year) / dataset.period.upper()
    # TODO: Store this filelist.
    return sorted(Path(period_specific_dir).glob("**/*.root"))

def run_dataset_download() -> None:
    """ Entry point for download a dataset. """
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog="downloadALICEDataset",
        description="Download an ALICE dataset in parallel",
    )
    parser.add_argument(
        "-p", "--period", type=str, default="",
        help="Run period (i.e. dataset) to download.",
    )
    parser.add_argument(
        "-o", "--outputdir", type=str, default="alice",
        help="Base output directory. [default: 'alice']",
    )
    parser.add_argument(
        "-d", "--datasets", type=str, default=None, metavar="PATH",
        help="Path to the datasets directory."
    )
    args = parser.parse_args()
    output = download_dataset(
        period = args.period,
        output_dir = args.outputdir,
        datasets_path = args.datasets,
    )
    print(output)

class RunByRunTrainOutputFiller(QueueFiller):
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
        for r in utils.list_alien_dir(grid_base):
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

            data = utils.list_alien_dir(run_dir)
            if not self._lego_train.split("/")[0] in data:
                logger.error(f"PWG dir not found four run {r}")
                continue

            lego_trains_dir = run_dir / self._lego_train.split("/")[0]
            lego_trains = utils.list_alien_dir(lego_trains_dir)
            if not self._lego_train.split("/")[1] in lego_trains:
                logger.error(
                    f"Train {self._lego_train.split('/')[1]} not found in pwg dir for run {r}"
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
                                     trainrun: int, legotrain: str, dataset: str,
                                     recpass: str, aodprod: str) -> None:
    q: queue.Queue[Union[FilePair, None]] = queue.Queue(maxsize = 1000)
    logger.info(f"Checking dataset {dataset} for train with ID {trainrun} ({legotrain})")

    queue_filler = RunByRunTrainOutputFiller(
        outputpath, trainrun,
        legotrain, dataset,
        recpass, aodprod if len(aodprod) > 0 else "",
        q = q,
    )
    queue_filler.start()

    workers = []
    # use 4 threads in order to keep number of network request at an acceptable level
    for i in range(0, 4):
        worker = CopyHandler(q=q)
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

def run_download_run_by_run_train_output() -> None:
    """ Entry point for download run-by-run train output. """
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog="downloadAliceRunByRun",
        description="Download run-by-run LEGO train outputs",
    )
    parser.add_argument(
        "outputpath", metavar="OUTPUTPATH",
        help="Path where to store the output files run-by-run",
    )
    parser.add_argument(
        "trainrun", metavar="TRAINRUN", type=int,
        help="ID of the train run (number is sufficient, time stamp not necessary)",
    )
    parser.add_argument(
        "legotrain", metavar="LEGOTRAIN",
        help="Name of the lego train (i.e. PWGJE/Jets_EMC_pPb)",
    )
    parser.add_argument("dataset", metavar="DATASET", help="Name of the dataset")
    parser.add_argument(
        "-p", "--recpass", type=str, default="pass1",
        help="Reconstruction pass (only meaningful in case of data) [default: pass1]",
    )
    parser.add_argument(
        "-a", "--aod", type=str, default="",
        help="Dedicated AOD production (if requested) [default: not set]",
    )
    args = parser.parse_args()
    download_run_by_run_train_output(
        args.outputpath,
        args.trainrun, args.legotrain,
        args.dataset, args.recpass, args.aod,
    )
