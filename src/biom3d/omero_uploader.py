#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2018 University of Dundee & Open Microscopy Environment.
# All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Sample code to import files and directories.

The OMERO session used for expimportort is created using the standard OMERO CLI.
If a CLI session already exists it will be reused.

Remember to logout of any existing sessions if you want to export to a
different server!

This utility has several optional arguments, see --help for details.

Each of the remaining arguments will form a fileset:

- Files will be imported into single-file filesets
- Directories will be imported into a fileset containing all files in that
  directory

See http://www.openmicroscopy.org/community/viewtopic.php?f=6&t=8407
Uses code from
https://gitlab.com/openmicroscopy/incubator/omero-python-importer/-/blob/master/import.py
"""


# TODO please do some code cleaning, some function are never used and can't work due to missing import

import argparse
import locale
import os
import platform
import sys
from typing import Any, Optional
import zipfile

import omero
from omero.model import ChecksumAlgorithmI
from omero.model import NamedValue
from omero.model.enums import ChecksumAlgorithmSHA1160
from omero.rtypes import rstring, rbool
from omero_version import omero_version
from omero.callbacks import CmdCallbackI
from omero.gateway import BlitzGateway
from omero.clients import BaseClient
from ezomero import post_dataset

def get_files_for_fileset(fs_path: str) -> list[str]:
    """
    Retrieve the list of files from a given file or directory path.

    If the input path is a file, return it as a single-item list.
    If the input path is a directory, return a list of all non-hidden files in it.

    Parameters
    ----------
    fs_path : str
        Path to a file or directory.

    Returns
    -------
    list of str
        List of file paths. If `fs_path` is a file, returns `[fs_path]`.
        If `fs_path` is a directory, returns the paths of all non-hidden files inside.

    Notes
    -----
    - Hidden files (starting with `.`) are ignored when `fs_path` is a directory.
    - Does not recurse into subdirectories.
    """
    if os.path.isfile(fs_path):
        files = [fs_path]
    else:
        files = [os.path.join(fs_path, f)
                 for f in os.listdir(fs_path) if not f.startswith('.')]
    return files


def create_fileset(files: list[str]) -> 'omero.model.FilesetI':
    """
    Create a new OMERO Fileset object from a list of local file paths.

    This function constructs a Fileset with entries corresponding to the given local files.
    It sets only the filename (not the full path) as the client path and attaches system
    version information as metadata for traceability.

    Parameters
    ----------
    files : list of str
        List of local file paths to include in the fileset.

    Returns
    -------
    omero.model.FilesetI
        An OMERO Fileset object populated with the given files and linked to an UploadJob.

    Notes
    -----
    - The `clientPath` is set to the basename of each file.
    - System information (OS, architecture, OMERO version) is added to the `UploadJob`.
    - If locale detection fails, it is silently skipped.
    """
    fileset = omero.model.FilesetI()
    for f in files:
        entry = omero.model.FilesetEntryI()
        entry.setClientPath(rstring(os.path.basename(f)))  # Set only the filename
        fileset.addFilesetEntry(entry)

    # Fill version info
    system, _, release, _, machine, _ = platform.uname()

    client_version_info = [
        NamedValue('omero.version', omero_version),
        NamedValue('os.name', system),
        NamedValue('os.version', release),
        NamedValue('os.architecture', machine)
    ]
    try:
        client_version_info.append(
            NamedValue('locale', locale.getdefaultlocale()[0]))
    except:
        pass

    upload = omero.model.UploadJobI()
    upload.setVersionInfo(client_version_info)
    fileset.linkJob(upload)
    return fileset


def create_settings() -> 'omero.grid.ImportSettings':
    """
    Create and configure an OMERO ImportSettings object.

    This function initializes a new `ImportSettings` object and sets several default import parameters, 
    including thumbnail generation and checksum algorithm.

    Returns
    -------
    omero.grid.ImportSettings
        Configured OMERO import settings object.

    Notes
    -----
    - Enables thumbnail generation (`doThumbnails=True`).
    - Disables skipping of stats info (`noStatsInfo=False`).
    - Sets the checksum algorithm to SHA-1 160-bit.
    - User-specified import options are left unset (`None`).
    """
    settings = omero.grid.ImportSettings()
    settings.doThumbnails = rbool(True)
    settings.noStatsInfo = rbool(False)
    settings.userSpecifiedTarget = None
    settings.userSpecifiedName = None
    settings.userSpecifiedDescription = None
    settings.userSpecifiedAnnotationList = None
    settings.userSpecifiedPixels = None
    settings.checksumAlgorithm = ChecksumAlgorithmI()
    s = rstring(ChecksumAlgorithmSHA1160)
    settings.checksumAlgorithm.value = s
    return settings


def upload_files(proc:'omero.gateway.UploadProcess', files:list[str], client:'omero.gateway.BlitzGateway'):
    """
    Upload multiple files to OMERO using the provided upload process.

    The function reads each file in chunks (1 MB) and streams the content to OMERO
    via the upload handler obtained from the process. It also computes and returns
    the SHA1 hashes of the uploaded files using the OMERO client.

    Parameters
    ----------
    proc : omero.gateway.UploadProcess
        The OMERO upload process object that manages file uploads.
    files : list of str
        List of local file paths to upload.
    client : omero.gateway.BlitzGateway or similar
        OMERO client instance used to compute SHA1 hashes of files.

    Returns
    -------
    list of str
        List of SHA1 hash strings corresponding to each uploaded file.

    Notes
    -----
    - Files are read in 1MB blocks to efficiently handle large files.
    - The uploader resource for each file is properly closed after upload.
    - Prints upload progress per file.
    """
    ret_val = []
    for i, fobj in enumerate(files):
        rfs = proc.getUploader(i)
        try:
            with open(fobj, 'rb') as f:
                print ('Uploading: %s' % fobj)
                offset = 0
                block = []
                rfs.write(block, offset, len(block))  # Touch
                while True:
                    block = f.read(1000 * 1000)
                    if not block:
                        break
                    rfs.write(block, offset, len(block))
                    offset += len(block)
                ret_val.append(client.sha1(fobj))
        finally:
            rfs.close()
    return ret_val


def assert_import(client: 'omero.client', proc: Any, files: list[str], wait: int) -> Optional['omero.cmd.Response']:
    """
    Upload files to OMERO and assert that they are correctly imported.

    This function uploads files using the provided processor, waits for the import job to complete,
    and verifies that at least one image (pixel set) has been imported successfully.

    Parameters
    ----------
    client : omero.client
        Active OMERO client used to communicate with the server.
    proc : object
        Upload processor, must implement `verifyUpload()` method compatible with OMERO.
    files : list of str
        List of file paths to upload and import.
    wait : int
        Behavior for waiting on the job:
        - `0`: Do not wait.
        - `<0`: Wait indefinitely until import finishes.
        - `>0`: Wait for the specified number of milliseconds.

    Returns
    -------
    omero.cmd.Response or None
        Response object from the OMERO server if waiting, otherwise None.

    Raises
    ------
    Exception
        If an OMERO error response is returned (`omero.cmd.ERR`).
    AssertionError
        If no images (pixels) were imported.
    
    Notes
    -----
    Prints the hash of each uploaded file.
    """
    hashes = upload_files(proc, files, client)
    print ('Hashes:\n  %s' % '\n  '.join(hashes))
    handle = proc.verifyUpload(hashes)
    cb = CmdCallbackI(client, handle)

    # https://github.com/openmicroscopy/openmicroscopy/blob/v5.4.9/components/blitz/src/ome/formats/importer/ImportLibrary.java#L631
    if wait == 0:
        cb.close(False)
        return None
    if wait < 0:
        while not cb.block(2000):
            sys.stdout.write('.')
            sys.stdout.flush()
        sys.stdout.write('\n')
    else:
        cb.loop(wait, 1000)
    rsp = cb.getResponse()
    if isinstance(rsp, omero.cmd.ERR):
        #TODO more specific exception
        raise Exception(rsp)
    assert len(rsp.pixels) > 0
    return rsp


def full_import(client: 'omero.client', fs_path: str, wait: int = -1) -> Optional['omero.cmd.Response']:
    """
    Perform a full import of a local file or directory into OMERO.

    This function handles all steps required to import data into OMERO:
    - Retrieves the managed repository.
    - Gathers files from the given path.
    - Creates a Fileset and ImportSettings.
    - Starts the import process.
    - Optionally waits for the import job to complete and validates it.

    Parameters
    ----------
    client : omero.client
        Active OMERO client session.
    fs_path : str
        Path to a single file or a directory containing files to import.
    wait : int, default=-1
        Determines how long to wait for the import to complete:
        - `0` : Do not wait.
        - `<0`: Wait indefinitely (default).
        - `>0`: Wait up to the specified number of milliseconds.

    Returns
    -------
    omero.cmd.Response or None
        OMERO command response if waiting for completion, otherwise None.

    Raises
    ------
    AssertionError
        If no files are found in the specified `fs_path`.
    Exception
        If OMERO returns an error during the import process.

    Notes
    -----
    Ensures `proc` is closed properly after the import, even if an error occurs.
    """
    mrepo = client.getManagedRepository()
    files = get_files_for_fileset(fs_path)
    assert files, 'No files found: %s' % fs_path

    fileset = create_fileset(files)
    settings = create_settings()

    proc = mrepo.importFileset(fileset, settings)
    try:
        return assert_import(client, proc, files, wait)
    finally:
        proc.close()
        
def run(
    username: str,
    password: str,
    host: str,
    project: int,
    attachment: Optional[str] = None,
    dataset_name: Optional[str] = None,
    path: Optional[str] = None,
    is_pred: bool = False,
    wait: int = -1,
    session_id: Optional[str] = None
) -> None:
    """
    Upload a new dataset or prediction result to OMERO and optionally attach log files.

    This function handles several OMERO tasks:
    - Connects to OMERO (via credentials or session ID).
    - Optionally creates a new dataset and imports image files from a given path.
    - If specified, attaches a zipped log directory (excluding image outputs) to the dataset.

    Parameters
    ----------
    username : str
        OMERO username (ignored if `session_id` is provided).
    password : str
        OMERO password (ignored if `session_id` is provided).
    host : str
        Hostname or IP of the OMERO server.
    project : int
        ID of the existing dataset or project to attach data to.
    attachment : str, optional
        Folder name used for zipping logs and attaching them to the dataset.
    dataset_name : str, optional
        Name for the new dataset (if `path` is provided).
    path : str, optional
        Local directory containing files to import into OMERO.
    is_pred : bool, default=False
        Indicates if the uploaded data is a prediction result.
    wait : int, default=-1
        Time (in ms) to wait for the OMERO import process:
        - `0`: do not wait
        - `<0`: wait indefinitely (default)
        - `>0`: wait up to specified time
    session_id : str, optional
        Existing OMERO session ID (used instead of logging in with username/password).

    Returns
    -------
    None

    Notes
    -----
    - The function can be used both for training logs and prediction results.
    - When `path` is provided, a new dataset is created and populated with the files.
    - When `attachment` is specified, the corresponding log folder is zipped (excluding images) and attached as a `FileAnnotation`.
    - The function automatically links imported images to the dataset.

    Raises
    ------
    AssertionError
        If no files are found in the specified import path.
    Exception
        If OMERO returns an error during import or file upload.
    """
    dataset_id = project
    if session_id is not None:
        client = BaseClient(host=host, port=4064)
        client.joinSession(session_id)
        conn = BlitzGateway(client_obj=client)
    else:
        conn = BlitzGateway(username=username, passwd=password, host=host, port=4064)
        conn.connect()

    if project:
        # Get the dataset by ID
        dataset = conn.getObject("Dataset", project)

        if not is_pred:
            dataset_name = dataset.getName() + "_trained"

        parent_project = dataset.listParents()
        if parent_project:
            project = parent_project[0].getId()
        else:
            print("No project found, Dataset will be orphaned")
            project = None


    if path is not None :
        # create a new Omero Dataset
        dataset = post_dataset(conn,dataset_name, project)
        directory_path =str(path)    
        filees = get_files_for_fileset(directory_path)
        for fs_path in filees:
                print ('Importing: %s' % fs_path)
                rsp = full_import(conn.c, fs_path, wait)
                if rsp:
                    links = []
                    for p in rsp.pixels:
                        print ('Imported Image ID: %d' % p.image.id.val)
                        if dataset:
                            link = omero.model.DatasetImageLinkI()
                            link.parent = omero.model.DatasetI(dataset, False)
                            link.child = omero.model.ImageI(p.image.id.val, False)
                            links.append(link)
                    conn.getUpdateService().saveArray(links, conn.SERVICE_OPTS) 
        dataset_id = dataset


    if attachment is not None:
        if path is not None:
            logs_path = "./logs"
            last_folder_path = os.path.join(logs_path, "{}".format(attachment))
            zip_file_path = os.path.join(logs_path, "{}.zip".format(attachment))
            # Create a zip file excluding the "image" folder
            with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(last_folder_path):
                    # Exclude the "image" directory
                    if 'image' in dirs:
                        dirs.remove('image')
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, start=last_folder_path)
                        zipf.write(file_path, arcname)

            print(f"Zipped folder (excluding 'image' folder): {zip_file_path}")    

        dataset = conn.getObject("Dataset", dataset_id)
        # Specify a local file e.g. could be result of some analysis
        file_to_upload = zip_file_path  if path is not None else attachment # This file should already exist

        # create the original file and file annotation (uploads the file etc.)

        print("\nCreating an OriginalFile and FileAnnotation")
        file_ann = conn.createFileAnnfromLocalFile(
            file_to_upload, mimetype="text/plain", desc=None)
        print("Attaching FileAnnotation to Dataset: ", "File ID:", file_ann.getId(), \
            ",", file_ann.getFile().getName(), "Size:", file_ann.getFile().getSize())
        dataset.linkAnnotation(file_ann)     # link it to dataset.
    conn.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=int, help=(
        'Add imported files to this Project ID (not valid when wait=-1)'))
    parser.add_argument('--wait', type=int, default=-1, help=(
        'Wait for this number of seconds for each import to complete. '
        '0: return immediately, -1: wait indefinitely (default)'))
    parser.add_argument('--dataset_name', default="Biom3d_pred",
        help='Name of the Omero dataset.')
    parser.add_argument('--path', 
        help='Files or directories')
    parser.add_argument('--username',
        help="User name")
    parser.add_argument('--password',
        help="Password")
    parser.add_argument('--hostname',
        help="Host name")
    parser.add_argument('--attachment', default=None,
        help="Attachment file")
    parser.add_argument('--is_pred', default=False,
        help="Whether it's a prediction or a training dataset.")
    parser.add_argument('--session_id', default=None,
        help="Omero Session id")
    args = parser.parse_args()
    
    run(args.username, args.password, host=args.hostname,
        project=args.project,
        dataset_name=args.dataset_name,
        path=args.path,
        wait=args.wait,
        attachment=args.attachment,
        is_pred=args.is_pred,
        session_id=args.session_id,
    )
    
