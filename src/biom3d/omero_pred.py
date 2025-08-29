"""
Predictions with Omero.

This script can download data from Omero, compute predictions, and upload back into Omero.
"""

import argparse
import os
import shutil
from typing import Optional
from omero.cli import cli_login


from biom3d import omero_downloader 
try:
    from biom3d import omero_uploader
except:
    print("[WARNING] Couldn't import omero uploader.")
    pass
from biom3d import pred  

def run(
    obj: str,
    target: str,
    log:str,
    dir_out: str,
    host: Optional[str] = None,
    user: Optional[str] = None,
    pwd: Optional[str] = None,
    upload_id: Optional[str] = None,
    ext: str = "_predictions",
    attachment: Optional[str] = None,
    session_id: Optional[str] = None,
    skip_preprocessing: bool = False
) -> Optional[str]:
    """
    Download a dataset or project from Omero, perform predictions, and optionally upload the results back.

    Depending on whether the object is a "Dataset" or "Project", the function handles:
    - downloading the data (either via Omero API or CLI),
    - running inference,
    - optionally uploading the predicted results back to Omero,
    - cleaning up temporary files if upload is done.

    Parameters
    ----------
    obj : str
        Type and ID of the object to process (e.g., "Dataset:123" or "Project:456").
    target : str
        Target location for downloading.
    log : str
        Path to the model folder.
    dir_out : str
        Path to the directory where predictions should be saved.
    host : str, optional
        Hostname of the Omero server, if using API authentication.
    user : str, optional
        Username for Omero authentication.
    pwd : str, optional
        Password for Omero authentication.
    upload_id : str, optional
        ID of the project to upload predictions back to. If None, uploading is skipped.
    ext : str, default="_predictions"
        Suffix to append to prediction output directories.
    attachment : str, optional
        Path to an optional attachment file to include in the upload (e.g., logs or configs).
    session_id : str, optional
        Session ID for Omero (used for authenticated operations).
    skip_preprocessing : bool, default=False
        Whether to skip preprocessing steps before prediction.

    Returns
    -------
    str or None
        Path to the output directory containing predictions, or None if an error occurred.

    Notes
    -----
    - The function prints messages that can be parsed remotely with the format "REMOTE:key:value".
    - Uploading back to Omero is deprecated but still supported.
    """
    print("Start dataset/project downloading...")
    if host is not None:
        datasets, dir_in = omero_downloader.download_object(user, pwd, host, obj, target, session_id)
    else:
        with cli_login() as cli:
            datasets, dir_in = omero_downloader.download_object_cli(cli, obj, target)

    print("Done downloading dataset/project!")

    print("Start prediction...")
    if 'Dataset' in obj:
        dir_in = os.path.join(dir_in, datasets[0].name)
        dir_out = os.path.join(dir_out, datasets[0].name + ext)
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out, exist_ok=True)
        dir_out = pred.pred(log, dir_in, dir_out,skip_preprocessing=skip_preprocessing)


        # eventually upload the dataset back into Omero [DEPRECATED]
        if upload_id is not None and host is not None:

            # create a new Omero Dataset
            dataset_name = os.path.basename(dir_in)
            if len(dataset_name)==0: # this might happen if pred_dir=='path/to/folder/'
                dataset_name = os.path.basename(os.path.dirname(dir_in))
            dataset_name += ext

            omero_uploader.run(
                username=user,
                password=pwd,
                host=host,
                project=upload_id, 
                attachment=attachment, 
                is_pred=True, 
                dataset_name=dataset_name,
                path=dir_out,
                session_id=session_id)
            
            # Remove all folders (pred, to_pred, attachment File)
            try :
                shutil.rmtree(dir_in)
                shutil.rmtree(dir_out)
                os.remove(attachment+".zip")
            except:
                pass

        print("Done prediction!")

        # print for remote. Format TAG:key:value
        print("REMOTE:dir_out:{}".format(dir_out))
        return dir_out

    elif 'Project' in obj:
        dir_out = os.path.join(dir_out, os.path.split(dir_in)[-1])
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out, exist_ok=True)
        pred.pred_multiple(log, dir_in, dir_out)
        print("Done prediction!")

        # print for remote. Format TAG:key:value
        print("REMOTE:dir_out:{}".format(dir_out))
        return dir_out
    else:
        print("[Error] Type of object unknown {}. It should be 'Dataset' or 'Project'".format(obj)) #TODO raise error, or exit with error code
    
if __name__=='__main__':

    # parser
    parser = argparse.ArgumentParser(description="Prediction with Omero.")
    parser.add_argument('--obj', type=str,
        help="Download object: 'Project:ID' or 'Dataset:ID'")
    parser.add_argument('--target', type=str, default="data/to_pred/",
        help="Directory name to download into")
    parser.add_argument("--log", type=str, default="logs/unet_nucleus",
        help="Path of the builder directory")
    parser.add_argument("--dir_out", type=str, default="data/pred/",
        help="Path to the output prediction directory")
    parser.add_argument('--hostname', type=str, default=None,
        help="(optional) Host name for Omero server. If not mentioned use the CLI.")
    parser.add_argument('--username', type=str, default=None,
        help="(optional) User name for Omero server")
    parser.add_argument('--password', type=str, default=None,
        help="(optional) Password for Omero server")
    parser.add_argument('--upload_id', type=int, default=None,
        help="(optional) Id of Omero Project in which to upload the dataset. Only works with Omero Project Id and folder of images.")
    parser.add_argument('--attachment', type=str, default=None,
        help="(optional) Attachment file")
    parser.add_argument('--session_id', default=None,
        help="(optional) Session ID for Omero client.")
    parser.add_argument('--ext', type=str, default='_predictions',
        help='(optional) Name of the extension added to the future uploaded Omero dataset.')
    parser.add_argument("--skip_preprocessing", default=False, action='store_true',dest="skip_prepprocessing",
        help="(default=False) Skip preprocessing")
    args = parser.parse_args()

    run(
        obj=args.obj,
        target=args.target,
        log=args.log,
        dir_out=args.dir_out,
        host=args.hostname,
        user=args.username,
        pwd=args.password,
        upload_id=args.upload_id,
        ext=args.ext,
        attachment=args.attachment,
        session_id=args.session_id,
        skip_preprocessing=args.skip_preprocessing,
    )