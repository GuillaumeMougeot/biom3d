"""
Use to download an OMERO dataset to a folder.

Adtapted from https://gist.github.com/will-moore/a9f90c97b5b6f1a0da277a5179d62c5a 
Documentation: https://downloads.openmicroscopy.org/omero/5.3.1/api/python/omero/omero.plugins.html 
"""

import argparse
import sys
import os

from omero.gateway import BlitzGateway
from omero.cli import cli_login
from omero.clients import BaseClient

from omero.plugins.download import DownloadControl

OBJ_INFO = "obj should be 'Project:ID' or 'Dataset:ID'"

"""
Usage:
python download_pdi.py Project:123 my_project_directory
"""

from typing import Optional
from omero.cli import CLI

def download_datasets_cli(datasets: list, target_dir:str)->None:
    """
    Download datasets using OMERO CLI interface.

    Parameters
    ----------
    datasets : list
        A list of OMERO Dataset objects to download.
    target_dir : str
        Path to the directory where the datasets will be downloaded.

    Returns
    -------
    None
    """
    with cli_login() as cli:
        cli.register("download", DownloadControl, "omero_downloader.py")

        for dataset in datasets:
            print("Downloading Dataset", dataset.id, dataset.name)
            dataset_dir = os.path.join(target_dir, dataset.name)
            os.makedirs(dataset_dir, exist_ok=True)

            for image in dataset.listChildren():
                if image.getFileset() is None:
                    print("No files to download for Image", image.id)
                    continue

                # If each image is a single file, or are guaranteed not to clash
                # then we don't need image_dir. Could use dataset_dir instead

                cli.invoke(["download", f'Image:{image.id}', dataset_dir])


def download_object_cli(cli:CLI, obj:str, target_dir:str):
    """
    Download a dataset or project using OMERO CLI and its ID.

    Parameters
    ----------
    cli : omero.cli.CLI
        Authenticated CLI session.
    obj : str
        OMERO object string, e.g. "Dataset:123" or "Project:456".
    target_dir : str
        Directory where data will be downloaded.

    Returns
    -------
    datasets: list
        Represent a list of OMERO datasets
    target_dir: str
        Final target folder path.

    Examples
    --------
    .. code-block:: python

        with cli_login() as cli:
            download_object_cli(cli, args.obj, args.target)
    """
    conn = BlitzGateway(client_obj=cli._client)
    conn.SERVICE_OPTS.setOmeroGroup(-1)

    try:
        obj_id = int(obj.split(":")[1])
        obj_type = obj.split(":")[0]
    except:
        print(OBJ_INFO)

    parent = conn.getObject(obj_type, obj_id)
    if parent is None:
        print("Not Found:", obj)

    datasets = []

    if obj_type == "Dataset":
        datasets.append(parent)
    elif obj_type == "Project":
        datasets = list(parent.listChildren())
        target_dir = os.path.join(target_dir, parent.getName())
    else:
        print(OBJ_INFO)

    print("Downloading to ", target_dir)

    #TODO fix the missing argument
    download_datasets(datasets, target_dir)

    return datasets, target_dir

def download_datasets(conn: BlitzGateway, datasets: list, target_dir: str) -> None:
    """
    Download datasets using OMERO BlitzGateway connection.

    Parameters
    ----------
    conn : BlitzGateway
        Active OMERO connection.
    datasets : list
        List of OMERO dataset objects to download.
    target_dir : str
        Path to the directory where the datasets will be downloaded.

    Returns
    -------
    None
    """
    for dataset in datasets:
        print("Downloading Dataset", dataset.id, dataset.name)
        dc = DownloadControl()
        dataset_dir = os.path.join(target_dir, dataset.name)
        os.makedirs(dataset_dir, exist_ok=True)

        for image in dataset.listChildren():
            if image.getFileset() is None:
                print("No files to download for Image", image.id)
                continue
            
            # If each image is a single file, or are guaranteed not to clash
            # then we don't need image_dir. Can use dataset_dir instead
            
            fileset = image.getFileset()
            if fileset is None:
                print('Image has no Fileset')
                continue
            dc.download_fileset(conn, fileset, dataset_dir)

def download_object(username: str,
                    password: str,
                    hostname: str,
                    obj: str,
                    target_dir: str,
                    session_id: Optional[str] = None,
                    ) -> tuple[list, str]:
    """
    Connect to OMERO and download a dataset or project via BlitzGateway.

    Parameters
    ----------
    username : str
        OMERO username.
    password : str
        OMERO password.
    hostname : str
        OMERO server hostname.
    obj : str
        Object identifier, e.g. "Dataset:123" or "Project:456".
    target_dir : str
        Target directory for download.
    session_id : str, optional
        Existing OMERO session ID to reuse.

    Returns
    -------
    datasets: list
        Represent a list of OMERO datasets
    target_dir: str
        Final target folder path.
    """
    if session_id is not None:
        client = BaseClient(host=hostname, port=4064)
        client.joinSession(session_id)
        conn = BlitzGateway(client_obj=client)
    else :
        conn = BlitzGateway(username=username, passwd=password, host=hostname, port=4064)
        conn.connect()
    try:
        obj_id = int(obj.split(":")[1])
        obj_type = obj.split(":")[0]
    except:
        print(OBJ_INFO)

    parent = conn.getObject(obj_type, obj_id)
    if parent is None:
        print("Not Found:", obj)

    datasets = []

    if obj_type == "Dataset":
        datasets.append(parent)
    elif obj_type == "Project":
        datasets = list(parent.listChildren())
        target_dir = os.path.join(target_dir, parent.getName())
    else:
        print(OBJ_INFO)

    print("Downloading to ", target_dir)

    download_datasets(conn, datasets, target_dir)

    # conn.close()

    return datasets, target_dir

def download_attachment(hostname: str,
                        username: str,
                        password: str,
                        session_id: Optional[str],
                        attachment_id: int,
                        config: bool = True,
                        ) -> Optional[str]:
    """
    Download an attachment (OMERO FileAnnotation) to the local file system.

    Parameters
    ----------
    hostname : str
        Hostname of the OMERO server.
    username : str
        OMERO username.
    password : str
        OMERO password.
    session_id : str, optional
        Optional OMERO session ID to reuse an existing session.
    attachment_id : int
        ID of the FileAnnotation to download.
    config : bool, default=True
        Whether to save to "configs/" directory. If False, saves to "logs/".

    Returns
    -------
    str or None
        Local path of the downloaded file, or None if not found.
    """
    # Connect to the OMERO server using session ID or username/password
    if session_id is not None:
        client = BaseClient(host=hostname, port=4064)
        client.joinSession(session_id)
        conn = BlitzGateway(client_obj=client)
    else:
        conn = BlitzGateway(username=username, passwd=password, host=hostname, port=4064)
        conn.connect()

    try:
        # Get the FileAnnotation object by ID
        annotation = conn.getObject("FileAnnotation", attachment_id)
        if not annotation:
            print(f"FileAnnotation with ID {attachment_id} not found.")
            return

        # Get the linked OriginalFile object
        original_file = annotation.getFile()
        if original_file is None:
            print(f'No OriginalFile linked to annotation ID {attachment_id}')
            return

        file_id = original_file.id
        file_name = original_file.name
        file_size = original_file.size

        print(f"File ID: {file_id}, Name: {file_name}, Size: {file_size}")

        if config : file_path = os.path.join("configs", file_name)
        else : file_path = os.path.join("logs", file_name)

        # Download the file data in chunks
        print(f"\nDownloading file to {file_path}...")
        with open(file_path, 'wb') as f:
            for chunk in annotation.getFileInChunks():
                f.write(chunk)
        return file_path

    finally:
        # Close the connection
        print("Downloaded!")

# Why not directly in __main__ ?
def main(argv: list[str]) -> None:
    """
    Entry point for downloading OMERO datasets or projects from command-line arguments.

    Parses command-line arguments for object identifier, destination directory, and connection credentials.
    Then triggers the download using `download_object`.

    Parameters
    ----------
    argv : list of str
        List of command-line arguments (excluding the script name). Typically `sys.argv[1:]`.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj',
        help="Download object: 'Project:ID' or 'Dataset:ID'")
    parser.add_argument('--target',
        help="Directory name to download into")
    parser.add_argument('--username', default=None,
        help="User name")
    parser.add_argument('--password', default=None,
        help="Password")
    parser.add_argument('--hostname', default=None,
        help="Host name")
    parser.add_argument('--session_id', default=None,
        help="Session ID")
    args = parser.parse_args(argv)

    # TODO: Add safeguards

    download_object(args.username, args.password, args.hostname, args.obj, args.target, args.session_id)

if __name__ == '__main__':
    main(sys.argv[1:])