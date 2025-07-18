# Changelog

## [v1.0.0] - Work in progress
This release merge several fork and add new functionality to Biom3d.

**New funtionalities**
- 2D images can be taken as input
- OMERO training data (same way as the prediction does)
- Easier to install with our brand new installers (Windows and macOS)
- Remote mode on macOS
- Official Docker images
- Automated deployment pipeline

**Changes**
- Documentation :
  - Documentation structure has changed
  - Documentation now have a Developper part where the API and the whole deployment scripts are describes
  - Installation documentation updated
  - New Docker documentation
  - New Server/Client documentation
  - New Config documentation
- Bug fix
  - Fixed, added a new safegard in `auto_config.py` that prevent `None` element in `spacings` list, preventing a `numpy.median(spacings)` error
  - Added `roll_stride` option to models and utils to make retrocompatibility with model trained before commit `f2ac9ee` (August 2023)
  - Fixed error in Google Collab
- Code 
  - Simples refactorings and code cleaning
  - Added condition `if not REMOTE` to display start locally button in GUI
- Miscellaneous
  - Added `CHANGELOG.md` file  
  - Now using semantic versionning

**Still in progress**
- Integration in Bacmman
- HDF5 integration
- Bioimage Model Zoo model exportation
- Docker Desktop documentation

## [v0.0.30] - 2023-Sep-26
All notable changes to this project will be documented in this file.
New version of Biom3d!

Most of the bugs have been solved:

    Preprocessing has been improved and takes multi channel images into account.
    Added Postprocessor module for ensemble prediction.
    Logs and metadata are now stored in the log folder.
    Finetuning works.
    Bug associated with the graphical user interface have been solved (tkinter, multiprocessing, matplotlib etc.)

New simplified GUI ! (cf. documentation).

## [v0.0.6] - 2022-Dec-6
This is the official and first version of the Windows Graphical User Interface of biom3d.

Careful! The current version works only in remote mode. You must also install biom3d on a remote server to make it work. Please install biom3d, with Pypi (pip install biom3d), on a Linux remote server with a Nvidia GPU and then use “Start remotely” button in the local GUI to start. Refer to the documentation to get more information about how to install biom3d.

To use the local version of the GUI you must install via Pypi and then run python -m biom3d.gui -L command.