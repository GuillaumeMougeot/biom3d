# Changelog

## [v1.0.0] - 2025-July-9
I am a debug version introducing new deployment.

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