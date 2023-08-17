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

The OMERO session used for import is created using the standard OMERO CLI.
If a CLI session already exists it will be reused.

Remember to logout of any existing sessions if you want to import to a
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

import argparse
import locale
import os
import platform
import sys

import omero.clients
from omero.model import ChecksumAlgorithmI
from omero.model import NamedValue
from omero.model.enums import ChecksumAlgorithmSHA1160
from omero.rtypes import rstring, rbool
from omero_version import omero_version
from omero.callbacks import CmdCallbackI
from omero.gateway import BlitzGateway
from ezomero import post_dataset

def get_files_for_fileset(fs_path):
    if os.path.isfile(fs_path):
        files = [fs_path]
    else:
        files = [os.path.join(fs_path, f)
                 for f in os.listdir(fs_path) if not f.startswith('.')]
    return files


def create_fileset(files):
    """Create a new Fileset from local files."""
    fileset = omero.model.FilesetI()
    for f in files:
        entry = omero.model.FilesetEntryI()
        entry.setClientPath(rstring(f))
        fileset.addFilesetEntry(entry)

    # Fill version info
    system, node, release, version, machine, processor = platform.uname()

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


def create_settings():
    """Create ImportSettings and set some values."""
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


def upload_files(proc, files, client):
    """Upload files to OMERO from local filesystem."""
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


def assert_import(client, proc, files, wait):
    """Wait and check that we imported an image."""
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
        raise Exception(rsp)
    assert len(rsp.pixels) > 0
    return rsp


def full_import(client, fs_path, wait=-1):
    """Re-usable method for a basic import."""
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
        
def run(username, password, hostname, project, dataset_name, path, wait=-1):
    conn = BlitzGateway(username=username, passwd=password, host=hostname, port=4064)
    conn.connect()

    if project and not conn.getObject('Project', project):
        print ('Project id not found: %s' % project)
        sys.exit(1)

    # create a new Omero Dataset
    dataset = post_dataset(conn,dataset_name,project)

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
    conn.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=int, help=(
        'Add imported files to this Project ID (not valid when wait=-1)'))
    parser.add_argument('--wait', type=int, default=-1, help=(
        'Wait for this number of seconds for each import to complete. '
        '0: return immediately, -1: wait indefinitely (default)'))
    parser.add_argument('--dataset_name', 
        help='Name of the Omero dataset.')
    parser.add_argument('--path', 
        help='Files or directories')
    parser.add_argument('--username',
        help="User name")
    parser.add_argument('--password',
        help="Password")
    parser.add_argument('--hostname',
        help="Host name")
    args = parser.parse_args()

    
    run(args.username, args.password, args.hostname,
        project=args.project,
        dataset_name=args.dataset_name,
        path=args.path,
        wait=args.wait
    )
    
