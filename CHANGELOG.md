# Pachyderm Changelog

Changelog based on the [format here](https://keepachangelog.com/en/1.0.0/).

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
