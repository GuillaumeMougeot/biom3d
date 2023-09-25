#---------------------------------------------------------------------------
# Predictions with Omero
# This script can download data from Omero, compute predictions,
# and upload back into Omero.
#---------------------------------------------------------------------------

import argparse
import os
from omero.cli import cli_login
from omero.gateway import BlitzGateway

from biom3d import omero_downloader 
try:
    from biom3d import omero_uploader
except:
    pass
from biom3d import pred  

def run(obj, target, log, dir_out, host=None, user=None, pwd=None, upload_id=None, ext="_predictions"):
    print("Start dataset/project downloading...")
    if host is not None:
        datasets, dir_in = omero_downloader.download_object(user, pwd, host, obj, target)
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
        dir_out = pred.pred(log, dir_in, dir_out)

        # eventually upload the dataset back into Omero [DEPRECATED]
        if upload_id is not None and host is not None:

            # create a new Omero Dataset
            dataset_name = os.path.basename(dir_in)
            if len(dataset_name)==0: # this might happen if pred_dir=='path/to/folder/'
                dataset_name = os.path.basename(os.path.dirname(dir_in))
            dataset_name += ext

            omero_uploader.run(user, pwd, host,upload_id,dataset_name,dir_out)
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
        print("[Error] Type of object unknown {}. It should be 'Dataset' or 'Project'".format(obj))
    
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
    # parser.add_argument("-e", "--eval_only", default=False,  action='store_true', dest='eval_only',
    #     help="Do only the evaluation and skip the prediction (predictions must have been done already.)") 
    parser.add_argument('--ext', type=str, default='_predictions',
        help='Name of the extension added to the future uploaded Omero dataset.')
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
    )