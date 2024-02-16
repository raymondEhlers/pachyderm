# Pachyderm Changelog

Changelog based on the [format here](https://keepachangelog.com/en/1.0.0/).

## [3.0] - 1 November 2020

### Added

- `BinnedData` class to generalize the `Histogram1D` class to general sets of binned data. It supports
  arbitrary number of axes, a substantially improved interface, and better maintainability. Note that
  this class doesn't yet have every feature from `Histogram1D`, but many of them are not necessary with
  `boost-histogram` available (some `BinnedData` features are inspired by bh). For example, we can convert to
  a bh hist to project, and then convert back (or do the summing directly - it's usually not difficult, except
  for profiles). As a notable new features, it has conversion to and from a wide variety of formats. This has
  been the basis for most of my analysis work this year. Written and improved over a variety of commits,
  starting with `d7e5bf8`.
- Helper function for plotting error boxes (refactor from `alice-jet-hadron`). See: `77ae89e`.
- A variety of ALICE datasets for download.

### Changed

- Move from `setup.py` to poetry. Starting from: `e7a9391`, with a ton of bug fixes.
- Support JAlien for ALICE download scripts. This deprecates support for the older alien. See: `fba152a`.
- General maintenance tasks for typing, TravisCI, etc.

### Fixed

- A ton of small bugs.

## [2.5] - 5 December 2019

### Added

- Support for multiplying or dividing by a numpy array. See: `c4187228b5f942c2daf36d39c487d984f9772a61`.
- Extracting `TDirectory` objects from a ROOT file. See: `d698fb40d18bcacf630f42fa5a7b9fb4b8fddb29`.
- Support downloading pt hard binned train output. See: `f41606d78c8f871a584069a27c2af8ea38dcfa95`.
- Some additional documentation.

### Changed

- Improved support for failed copying from AliEn. See: `76c7aa7082e5533441e464d1355d398996288686`.
- Improved tests output. See: `b4f599f8d355790eac5a662376ede6bea0d5d81d`.
- Improved `LHC16j5` dataset definition. See: `9e3b5bd61fd3e22a8795991d277a30d479655820`.

### Fixed

- Filelist generation didn't always include the right files. See: `684c42c8388f9bf765d6f49f205780e345da4291`.

## [2.4] - 4 December 2019

### Added

- Tools for downloading ALICE datasets and train outputs in parallel. It can download real or simulated
  datasets, run-by-run LEGO train output, or any list of input and output files (for simpler cases, such as
  meta-dataset trains). Derived and generalized from code from M. Fasel. See:
  `76ce114ccff10e59330b541e38d685fc4aa7fb2d` to `51c469b078e8a9fe25a980bc7c8b61fb9633f83a`.
- Convert HEPdata files directly into Histogram1D objects. Call via `Histogram1D.from_hepdata(...)`. Note that
  it doesn't fit nicely into the from_existing_histogram function because additional information is needed
  when reading HEPdata files. See: `c79f40b5a9c6cb5a8b31bd35275004196a5d8228`.
- Ability to convert matplotlib color schemes to ROOT. See: `292106009c9e9d6d5d47bf5648a9481c89b1d20c`.

### Changed

- Support passing `Path` objects for opening ROOT files. See: `a2d644b6cab7c35fa15ba3614fac027599484b24`.
- Update pre-commit hook versions. See: `2ad77745008ce81a62ba37282a35177e74e868fb`.

## [2.3.2] - 21 October 2019

### Fixed

- Improved typing for `generic_config` iterators of possible values. See: `797848212b03e8c36174cbb1a6b9bc1fda7b7739`.

## [2.3.1] - 20 October 2019

### Fixed

- Fixed typing for `Histogram1D` scalar division. See: `7cc7a7e9c2b16546e6dadacbb92cb0da10c032bb`.

## [2.3] - 16 October 2019

### Added

- Scalar division for `Histogram1D`. See: `c4230c6cfed7334cb0da44490d67cd5c351ae1fb`.
- All arithmetic operations for PDFs when fitting. This expands from just adding PDFs, which was all that was
  available up to now. See: `1799c9e68a2e8b10b6762e03850d125d9f29e1d1`.

### Changed

- Make gradient calculation available through the package API. It can be useful for more complex error
  propagation. See: `6454c8627972156e0526a31a280ee629a55f99ca` and `3891fedffc138112c85a01b870a25260b9f31d0f`.

### Fixed

- Binned log likelihood fit values, error matrix, covariance, etc, now agrees with ROOT. This is important for
  error calculation! See: `dc3e1ff02157bf3cc8005f07985acfbf46bba026`.
- Explicit weighted histogram option for binned log likelihood. See:
  `dc3e1ff02157bf3cc8005f07985acfbf46bba026`.
- Calculation of chi squared for goodness of fit when there are empty bins (i.e. for calculating the effective
  chi squared when performing a log likelihood fit). The empty bins are ignored. See:
  `0e66393b745c00fd925ef6670512eb7f86647c6f`.
- Typing for a variety of functions.
- Typos in documentation.
- Check for `Histogram1D` when attempting to create a new one. See: `62d8613fcfa0b1a4117a13f4537e99d00b556bfd`.

## [2.2.1] - 7 September 2019

### Added

- Added effective chi squared for simultaneous fit. See: `90a98d916dc6c5fa29a48b98c9ccafec9e5b4769`. Bug fix:
  `92ac7b01e77cec033f6bcf41a95b6a6bb53525ab`.
- Write numpy floats to YAML. This helps to avoid a bunch of difficult to debug errors. See:
  `474a24203f87dd09595df9144ead0793fc3647e0`.

### Changed

- Improve minimization settings. See: `8bb1602b0bd22fa1dd1eb6c49b3b3768522230bc`.

### Fixed

- Don't convert the histogram metadata into a numpy array. See: `2fafe779c2be2b254fbc7b957ffbb5f464940db2`.
- Ensure that metadata is copied when copying a histogram. See: `f1d1fbf8a7de0e5a2d81b58b68d44d6447fd0749`.

## [2.2] - 23 August 2019

### Added

- Generic fit base class. It can direct fits with minimal work for the user. It's generalized and ported over
  from the jet-hadron analysis. See: `83f1a93350c988d3d7267e2f99d3b2c63464d3ec`.
- Histogram statistics calculation for mean, standard deviation, and variance. Can be extracted from ROOT
  hists as well as recalculated manually. See: `c5b23a7b079dc0cbf75962b5bfd6e29ee6679bcb`,
  `6ee9f7b4fdb5b71016296c18ade0472816777949`, and `aa4cb32252418035adbb6968783f1258da325410`.
- Extended gaussian fit function. See: `af69d29ee931c50b6398a3f536e54bc9341646da`.
- Wrapper for calculating chi squared probability. See: `2277444a4c37e995143459841f1e386a3512462a`.

### Changed

- Improved Minuit wrapper fit function argument validation. See: `f09ace01b19e285d1b229a3f0bcde819c59fce34`.
- Refactor Minuit wrapper fit function into the new module with the generic fit class. The interface of the
  `pacyhderm.fit` package defined in `__init__.py` did not change. See:
  `83f1a93350c988d3d7267e2f99d3b2c63464d3ec`
- Allow for numpy arrays to be written by hand in YAML inputs. See: `235f41f900f40713663fe999718aea35a4201dc8`.

### Fixed

- Fix numpy writing to YAML with newer versions of numpy. See: `1fedadc9f3365fd292ba168c47ce4e1d72179a16`.
- Fix typos in docs and comments.

## [2.1] - 17 August 2019

### Added

- `isort` to the pre-commit checks. See: `9930ca1ef9dc13b603e21dea40bae832d377d6d9`.
- Scalar multiplication for histograms. See: `522cd73c5d08225edd7f1f4785ba243dc7e4a079`.
- `Histogram1D` validation and sanity checks. The class should be much more robust. See:
  `b69ad4ba8481ec89813bef5fc20f671c25ac5552`.
- Support for converting a `TProfile` to `Histogram1D`. A `TProfile` doesn't strictly conceptually convert
  quite so nicely, but `Histogram1D` is very useful as a data container, so it's better to support it. See:
  `d3395f8ced9d4642d28a18c23daf81e63a51b571`.

### Changed

- We're not going to use `numba` for now, so possible code related to it was removed. See:
  `a0b65374b20ad08192132e7f7e7c3b61b7a78d32`.
- Split out a `BaseFitResult` class. It's really convenient to have such a class for individual fit
  components. See: `6d638b9020e2334feff0bff64357aa94b4caa40f`.
- Imports are now sorted by `isort`. This ensures consistency. See: `0a8c771f8785bc620e6ec5240fba005fdad77254`.
- Improve numpy encoding when writing to YAML files. It will write in the binary format, which won't be as
  convenient to edit by hand, but I believe it will avoid a bunch of floating point issues where later digits
  fluctuate each time they are calculated, thereby changing the YAML file. See:
  `8bf22118d8fb953a5ff1691e12d95edcdf8f8f6a`.

## Fixed

- Prevent ROOT from intercepting args when importing type helpers. See:
  `9894c3f526712f6eb72df1c0d454444992d58dcb`.
- Improve typing. See: `20d7a06f7879c9d4df869ecb33c5f5e952834824`, `47199d86a1b468cfa2881b4130947872fe92d623`,
  and `8bedd17fbc5e5c9481b7bdadfd525e6d96accae4`.

## [2.0] - 7 August 2019

## Added

- Fit package utilizing `iminuit`. Includes base fitting functionality (function to direct fitting, fit result
  class, etc), cost functions, useful functions (AddPDF, gaussian, etc). All of this functionality is based on
  `numpy` and therefore can execute in a vectorized manner (including the error calculation). Some
  functionality of the `reaction_plane_fit` package was ported into Pachyderm. Much of the development itself
  was done privately in the `reaction_plane_fit` package. These developments include all of the features of
  `probfit` that I've used, such that dependence on that package isn't necessary. See:
  `b0620e0dc610566565dacc5b2ab899707d275b76`, `2626c20fc0864619d10c10ca3f5bf24a218e76dc`,
  `ebb1dd19ab28d0db25723488e847e1bcee59be3a`, and `3e5441f514a104cd1696add779d192c8c2b3a862`.
- Effective chi squared calculation for fit results. See: `4ad189cbc6add79538dce0274f6dda76a0675e7d`.

## Fixed

- Typing improvements, fixed tests and import issues, etc. See: `ebac51f782d35cc1ee1b87b66d2409121dc9569e` and
  `3994877c8addcc005ddcd610d3b794d7d1d9078e`, amongst others.

## [1.9.6] - 19 July 2019

### Added

- `metadata` field to `Histogram1D` for storing additional information, such as additional error arrays (for
  example, systematics). See: `58d35b341f3b4890c59c9cb9ec70b72836793fc2`.

### Changed

- Fixed some typing and pinned `ruamel.yaml` to the pre `py.typed` version until the typing information
  stabilizes. See: `933a1fb85f3ab7bf1daadeb2d260dacd37732a1a`.

## [1.9.5] - 17 June 2019

### Changed

- Increased default plot font size. This generally makes things much more readable (but of course can be
  adjusted for each plot later if necessary). See: `56a6bef789b1329bb810139918cbabdf74336709`.

## [1.9.4] - 31 May 2019

### Changed

- Allow configuration of max fractional difference limits of mean and median in outliers removal. See:
  `f94d9d691a0e7e74426a38bc6cba8c35e1e2a820`.
- Fix for latex preamble changes in MPL 3.1. See: `168f548d32ae7fd8f7c39db09ee0864ac1b1a158`.

## [1.9.3] - 16 May 2019

### Changed

- Refactor `histogram.find_bin(...)` so that it can be used externally. See:
  `a2a9838a0e11d950ba5680656df9271e88c4c8b3`.

## [1.9.2] - 4 May 2019

### Added

- Enabled strict `mypy` checks. See: `6ac6bf50b6787dea547516196c043a989e9cf70d`.

### Changed

- Changed the return typing of `selected_iterables` in `generic_config` from `Dict[str, Any]` to `Mapping[str,
Any]` to accept more generic mappings. See: `f9bf3a173e5a448bb174357d73950e024220da48`.

### Fixed

- Many minor typing fixes. See: `86669f2047de24ca5ef9694a9bd29e2f2ff6f58d` and
  `9882e9950d958eb70fb85537f773043a7934e220`.

### Removed

- Obsolete code related to extracting analysis objects when fitting. See:
  `4e1144baae727f93c52c8cad882983e1b07875e0`.

## [1.9.1] - 3 May 2019

### Changed

- Don't plot lines around patches. See: `aaa1d8374fd871246290ce76f1796f2f7582b01d`.

### Fixed

- Histogram integral values now write properly to YAML. See: `c88dabceedcd0e4576802ab322b96592dcaf5233`.
- Fix `mypy` typing in third party packages by adding `zip_safe = False` in the `setup.py`. See:
  `d3d212b8569a2b994a33dc5d817d7bd7395dcc1a`.

## [1.9] - 27 March 2019

### Added

- Plot module to handle tasks related to plotting. Currently, it just configures `matplotlib` styles which
  were extracted from the `alice-jet-hadron` package. See: `82f4be7a344c124a3a51faf34f521fa125c21bbc`.
- Integration tests to ensure that classes can be dumped by YAML. See:
  `d03aa5a0d2a7ce0979c87086ccb5641b270a2e59`.

### Changed

- Updated pre-commit hooks with newer versions of the packages. See:
  `b7b7f570b27412492ae24c65e956865943bc7fc1`.

## [1.8] - 24 March 2019

### Added

- Equality operator for `Histogram1D`. See: `d769eec644dc91de17bcb3cc92435346cd305196`.
- Integrate and count over bins in a specified range of a `Histogram1D`. See:
  `06fcb96076da89677bef5afc772b462db82ab39b`, `fd00a741e4f62e44cbfed6fa1eb3fe50b5aba8bb` (initial
  implementation), `5f8853d6a7db8bdebe1464bf1cf34273a95059fc`, `794054b41d76fc92bf326743d97c71e431fbf20b`
  (tests improved), and `e16613bdb4ef139d6a5564e68b46fd78a57124dd` (improve docs).
- Implemented `find_bin(...)` method for `Histogram1D` to find the bin that corresponds to the given value.
  See: `5ca286e3e4f8ffa2f0ab05a3679250bbe9166aa9`.
- Refactor `bin_widths` into property for `Histogram1D` to ease calculating the bin widths. See:
  `01578f75ec8203e176075005408b166845fb0725`.

### Changed

- Improved `yaml` module typing information. See: `effd1d5193891c5bd268e033cf18eb2be71db6f1`.

## [1.7.3] - 26 February 2019

### Added

- `recursive_getitem` to retrieve an item from a recursive dict. In principle, this should also work for other
  objects which implement `__get_item__(...)`, but it is not tested for those other objects. See:
  `5982c6bfd5a13a4b8f09464ca27ac24ba8477f6f` and `a3653859c94bc892082842ecc7997e4facda1e60`.

## [1.7.2] - 22 February 2019

### Changed

- Relaxed some `Dict` -> `Mapping` typing information where possible for greater compatibility with inputs.
  See: `53ae9f762f344978161ff662226f7fa9202c8b84`.

## [1.7.1] - 22 February 2019

### Fixed

- Don't modify the passed analysis iterables when iterating with
  `iterate_with_selected_objects_in_order(...)`. See: `7daf01ea590e9ccbdc6511d6a5379757409537fa`.
- Indicate that the package is typed (it already was, but the typing wasn't being checked in downstream
  packages). See: `409922e80541155d414c2c101eb913b4ab9fd9ac`.
- Improved typing information in the `histogram` module. See: `dded82643a8c7c7acc684d64c958eed63c84b3e6`.
- Improved typing information in the `generic_config` module. See: `c9a8f57fea5f72e62841f264d768e785cd6da3c5`,
  and `03ad1faf0cb295c7387badd1e59082c5740fbdfb`.

## [1.7] - 20 February 2019

### Added

- `Histogram1D` operators (add, subtract, multiply, divide) with other hists, including error propagation. These
  operations were confirmed by comparing against the same operations in ROOT histograms (it turns out that the
  error propagation is exactly as expected, but the ROOT code was sometimes difficult to interpret). See:
  `052ea365e4e6a8a4d6bce38779ed9fe2edcfc089` and `7df8ceaa445d5c7aea4c62f6b32bc198eb59ccc3`.

## [1.6.2] - 18 February 2019

### Changed

- Refactor `KeyIndex` creation into a separate function so it can be accessed by other packages. In
  particular, this allows for their creation (with the iterable field access, etc) from analysis packages.
  See: `d821908c80e5e81df43986cb0831f112b706b91b`.

## [1.6.1] - 5 February 2019

### Added

- Added iterator (`__iter__`) over names and values to the `KeyIndex` object. See:
  `a7dfa745aefbc381a245017dbde722f368ad39de`.

### Fixed

- Added `__init__.py` to the `tests` directory. Otherwise, test fixture discovery sometimes fails. See:
  `c1a1f059e2c921f5e1afb3dfc9dc5624a2754596`.

## [1.6] - 29 January 2019

### Added

- Iteration over analysis objects grouped by selections. See: `0164157982147142561f33d4a790156513cbba34`.
- Recursive `getattr(...)`. Function from [Stack Overflow](https://stackoverflow.com/a/31174427). See:
  `381d6ec2be0e7482b92ff3d3afd9847a9aa84375`.
- Recursive `setattr(...)`. Function from [Stack Overflow](https://stackoverflow.com/a/31174427). See:
  `22ba4c6df527dc67d10c729ab85192ba8ef173b5`.
- Raise exception if mean or median after outliers removal changes by more than 1%. Such a condition almost
  certainly indicates that something went wrong with the outliers removal. See:
  `c7d5d0c449a826a4cc7a5c7f83608751da37bd6d`.

### Changed

- Quiet down some logging information.
- Naming of `particle_level_axis` -> `outliers_removal_axis` in the `remove_outliers` module. While it is
  common to the particle level axis, it isn't always. Further, sometimes we use the axis for projections,
  and sometimes it is just for determining which axis to use when removing outliers. So it's better to use a
  more general name. See: `12bde27f421eac59e310029c10f2ca750bcd6da6`.
- Specify the outliers removal axis when calling `run(...)` instead of at object creation. This way, we can
  use the manager over for different histograms which many have different axes. See:
  `12bde27f421eac59e310029c10f2ca750bcd6da6`.
- Ignore the first bin of the outliers removal. Some older embedding trains had some erroneous entries in the
  1st bin of the pt hard spectra, regardless of the pt hard bin. In any case, we would never expect to have
  outliers in the first bin. See: `354013e1713d2c4d84d48f0a1d21a6bc06afea2e`.

### Fixed

- Included new modules in the docs. See: `3c2d30f363e04eca181f7c05d40306bfbb0d0470`.
- Typo: `_determine_outliers_for_moving_avreage` -> `_determine_outliers_for_moving_average`. See:
  `11ce7d01905004dfcd3104549548b58ca17e3808`.

## [1.5] - 20 January 2019

### Added

- Outliers removal module, `remove_outliers`. It removes outliers from a given histogram. It was developed
  based on previous code. See: `dd472b3dff0715f015d27d083258555bd9674028`,
  `12268cb696daa9d80ca680fc219e5d5ab01f67d2`, and `6b62e751c773c05be11019980c67d9ef69d1e487`.
- Centralized shared typing. See: `4e00fd66994971b67d40031627cd6b536c0439d0`.

### Changed

- Improved YAML dependencies and typing. It is now only handled in the `yaml` module. See:
  `ba364f65d152617c5cd1ee88de88a25c69048203`.
- Provide access to wrapper to get a `TAxis` object given a histogram and a selected axis number. See:
  `9569bda571fea44142bf3d46bb2d11b4f979562b`.

### Fixed

- Fixed 3D projectors axis determination. The axis name was being taken rather than the name of the axis type.
  See: `c32ebea1321054eea71bed1c7c6d68c0b3794a9a`.

### Removed

- Obsolete YAML read and write code. We can now directly write classes, which is much easier. See:
  `f87ede530519761be35b098a66b86f0cb4e26aab`.

## [1.4] - 14 January 2019

### Added

- Added very simple context manager for opening ROOT files. Basically a simpler version of
  `rootpy.io.root_open`. See: `0d7a923f19d714125ec5f1eaed55ad9203ae718e`.

### Changed

- Heavily modified the `projectors` module to simplify projecting single objects. See:
  `d578b97fc2b97f0888170ca678014d8807faf876`.
- Modify the `HistProjector` initialization names and objects for improved clarity. See:
  `7eb6f18bb72d0c12cbd47e4bf05178a6d934895c`
- Clarify typing information. See: `eac9586a7d7c0bf1e2be1f0ef2235fb75bf0e676` and
  `4de3493092b413dbaaeccebc3b3a10d70b3ad0df`.

### Fixed

- Improved typing information in the `yaml` module. See: `eac9586a7d7c0bf1e2be1f0ef2235fb75bf0e676`.

## [1.3.3] - 11 January 2019

### Added

- A few more git pre-commit hooks. See: `dcdde76d223b93ec12d1f43cb5f9a94894efa6c8`.

### Fixed

- ROOT memory issue that sometimes occurred during testing. Fixed by ensuring that the open ROOT file is
  always closed when an exception is raised. Also included ensuring that only python has ownership of the
  objects. See: `c21612967376fb15aa1cffa2e2c8c7b663dae90b` (many of the preceding commits are related to
  trying to solve this issue).

## [1.3.2] - 5 January 2019

### Changed

- `Histogram1D` now utilizes bin edges to define the axes instead of the bin centers. This allows for variable
  sized bins. Further, the bin centers (ie `x`) are easy to calculate from the bin edges, but the edges are
  a bit more painful. See: `b053fe8214591da6754e55d5f9dabf79a3cd5f56`.
- `Histogram1D` supports non-uniform binning. See: `32d2ed53f5743e0b8cab353130e045c733677255`.
- `get_array_from_hist2D(...)` can now return an (x, y) mesh using bin edges by enabling the
  `return_bin_edges` option. By default, it will use bin centers. See:
  `44a760a655eca0ec416042ec0a629cf88eb2ed7b`.

## [1.3.1] - 2 January 2019

### Added

- Improved test coverage to demonstrate why `deepcopy` is necessary when constructing objects. See:
  `369da022`.
- `histogram.get_histograms_in_file(filename)` to return all histograms stored in a ROOT file. See:
  `5b333884`.

### Changed

- Avoided a very expensive `deepcopy` by formatting strings into new objects instead of writing into existing
  objects (which had to be copied first). See: `085f8f18`.
- Return the iterables that were actually utilized in `generic_config.create_objects_from_iterables`. It
  replaces the iterables names return values with a dict, where the keys are the iterables names and the
  values are lists of the utilized iterables. See: `973b85ae`.

### Fixed

- `KeyIndex` already could be written to YAML - it didn't need `to_yaml` and `from_yaml` methods.

## [1.3] - 30 December 2018

### Added

- New YAML module to create YAML objects, register classes + modules for reading and writing, and additional
  helper functions. It generally encapsulate much of the basic YAML functionality. See: `e8009517`.
- Pre-commit hooks configuration to the repository based on `pre-commit`. This should generally improve the
  quality of commits. Currently includes `flake8`, `mypy`, `yamllint`, and some additional minor python
  checks. See: `586c1549` and `0510a0ba`.
- YAML linting via `yamllint`. Little is done with this yet, but more files should filter in, so it's useful
  to have . See: `a6bc250f`.

### Fixed

- Fixed breaking change made by `ruamel.yaml` upstream related to base python types receiving anchor support.
  It is a good start, but still needs some time to mature for our purposes. Also expanded the tests to check
  for such issues. See: `9aa7f4b7` (the commit message has far more information).
- Minor test improvements for `generic_config`. See: `d723bbcb`.

## [1.2.3] - 23 December 2018

### Fixed

- Mixin approach for enum values to/from YAML causes problems with pickling (which is implicitly called while
  copying). Revised the approach to be unbound functions, which doesn't have this same problem. See:
  `89159dc9`.

## [1.2.2] - 22 December 2018

### Added

- Enumeration YAML read and write mixins. They utilize the `__name__` of the enumeration (not the value!).
  See: `3f987afb`.

### Changed

- Allow for classes to be registered and constructed via YAML. See: `b877a09e`
- Allows for set of iterable values to be specified in a configuration file. Often used in conjunction
  with creating objects in YAML. See: `7cab0476`.

## [1.2.1] - 18 December 2018

### Changed

- Convert each argument via `str()` in the generic_config instead of `obj.str()` which is less portable and
  pythonic. Precipitated by changes in the definition of enumeration classes in the jet-hadron package. See:
  `595ef2a8`.

## [1.2] - 17 December 2018

### Added

- Include `--ignore-missing-includes` when running `mypy` automatically. See: `0ac7d0bd`.
- Full type annotations for the package. See: `a83372b1`.

### Changed

- Fully updated API naming scheme to follow python conventions. See: `cd097f3b`, `c7bedc6b`, `68048970`, and
  `7d8b1ca9`.
- Remove `pkgconfig` workaround introduced in `4a3c6216` since `python-lz4` merged in an alternative
  [approach](https://github.com/python-lz4/python-lz4/pull/160) to workaround the issue. See: `755c276a`.

## [1.1] - 15 December 2018

### Added

- Added `mypy` to Travis CI checks. See: `2c6f7dc0`.

### Changed

- Reworked iteration over analysis dictionaries. See: `0ead6db8`.
- Updated overwatch-base python version. See: `39fede7e`.

### Fixed

- Existing typing issues identified by `mypy`.

## [1.0] - 10 December 2018

### Added

- `Histogram1D` class moved from the `reaction_plane_fit` package to `pachyderm`. See: `89d2eaa3.`
- Added `uproot` dependency and add fix for `lz4` issue that is caused by ROOT. See: `89d2eaa3` and
  `4a3c6216`.
- version info is now available in the package by simply typing `import pachyderm; pachyderm.version_info`.
  See: `32ff4f96`.

### Changed

- Moved histogram functionality from `utils` to a new `histogram` module. This module also contains the added
  `Histogram1D` class (see above). See: `89d2eaa3`.

## [0.9] - 10 December 2018

- Initial release, with most of the development performed in
  [alice-yale-dev](https://github.com/ALICEYale/alice-yale-dev), and a bit in
  [alice-jet-hadron](https://github.com/raymondEhlers/alice-jet-hadron).
