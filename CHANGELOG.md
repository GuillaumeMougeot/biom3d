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
- Metal (Applie Silicon) compatibility, due to us not having Macs with Apple Silicon, it may still be unstable, contact us if you encounter a problem
- New DataHandler system that abstract image loading/saving and allow to easily implement new image format
- HDF5 files are now supported

**Changes**
- Documentation :
  - Documentation structure has changed
  - Documentation now have a Developper part where the API and the whole deployment scripts are described
  - Installation documentation updated
  - New Docker documentation
  - New Server/Client documentation
  - New Config documentation
  - New Dataset documentation
  - New DataHandler documentation
  - Almost all classes and functions are now documented, exception are:
    - FileHandler and HDF5Handler, refer to DataHandler doc
    - gui.py, because it's a mess
- Bug fix
  - Added a new safegard in `auto_config.py` that prevent `None` element in `spacings` list and empty `spacings`, preventing a `numpy.median(spacings)` error
  - Fixed `semseg_torchio` to ensure extracted foreground are casted to `numpy.ndarray`
  - Added `roll_stride` option to models and utils to make retrocompatibility with model trained before commit `f2ac9ee` (August 2023)
  - Fixed error in Google Collab
  - Added missing doc dependencies in pyproject.toml
- Code 
  - Simples refactorings and code cleaning
  - `versus_one()` doesn't load images by himself, you need to provide it with the images to compare, you can use `eval()` from `biom3d.eval` to do that automatically
  - Everywhere an image was loaded/saved, we now use a `Datahandler` that abstract the logic and allow to easily implement new formats. Sole exceptions are old predictions function.
  - Almost every variable refering to directory for image have been renamed to emphasize the new `DataHandler` system (except in GUI).
  - `Biom3d.utils` has been shattered to multiple files to ease reading and modification. The namespace hasn't changed.
  - Methods to read and save images has been moved to `utils.data_handler.file_handler` in the class `ImageManager`. The old path `biom3d.utils` still work but has been marked as deprecated.
  - Added condition `if not REMOTE` to display start locally button in GUI
  - Put constant initialization before imports in `gui.py`
- Miscellaneous
  - Added `CHANGELOG.md` file  
  - Now using semantic versionning
  - Biom3d version and date time (UTC) of training are stored in model config.yaml

**Still in progress**
- DockerHub link to include

**Knwown issues**
- On Windows, the `omero_preprocess_train` will crash trying to export model. You will need to manually export it to your Project


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