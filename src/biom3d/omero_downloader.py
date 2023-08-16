# adtapted from https://gist.github.com/will-moore/a9f90c97b5b6f1a0da277a5179d62c5a 
# documentation: https://downloads.openmicroscopy.org/omero/5.3.1/api/python/omero/omero.plugins.html 

import argparse
import sys
import os

from omero.gateway import BlitzGateway
from omero.cli import cli_login, CLI


from omero.plugins.download import DownloadControl

OBJ_INFO = "obj should be 'Project:ID' or 'Dataset:ID'"

"""
Usage:
python download_pdi.py Project:123 my_project_directory
"""

def download_datasets_cli(datasets, target_dir):
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
                # image_dir = os.path.join(dataset_dir, image.name)
                # If each image is a single file, or are guaranteed not to clash
                # then we don't need image_dir. Could use dataset_dir instead
                # cli.invoke(["download", f'Image:{image.id}', image_dir])
                cli.invoke(["download", f'Image:{image.id}', dataset_dir])


def download_object_cli(cli, obj, target_dir):
    """
    usage:
    with cli_login() as cli:
        download_object(cli, args.obj, args.target)
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

    download_datasets(datasets, target_dir)

    return datasets, target_dir

def download_datasets(conn, datasets, target_dir):

    for dataset in datasets:
        print("Downloading Dataset", dataset.id, dataset.name)
        dc = DownloadControl()
        dataset_dir = os.path.join(target_dir, dataset.name)
        os.makedirs(dataset_dir, exist_ok=True)

        for image in dataset.listChildren():
            if image.getFileset() is None:
                print("No files to download for Image", image.id)
                continue
            # image_dir = os.path.join(dataset_dir, image.name)
            # If each image is a single file, or are guaranteed not to clash
            # then we don't need image_dir. Can use dataset_dir instead
            
            fileset = image.getFileset()
            if fileset is None:
                print('Image has no Fileset')
                continue
            dc.download_fileset(conn, fileset, dataset_dir)

def download_object(username, password, hostname, obj, target_dir):
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

    conn.close()

    return datasets, target_dir


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj',
        help="Download object: 'Project:ID' or 'Dataset:ID'")
    parser.add_argument('--target',
        help="Directory name to download into")
    parser.add_argument('--username',
        help="User name")
    parser.add_argument('--password',
        help="Password")
    parser.add_argument('--hostname',
        help="Host name")
    args = parser.parse_args(argv)

    download_object(args.username, args.password, args.hostname, args.obj, args.target)

if __name__ == '__main__':
    main(sys.argv[1:])