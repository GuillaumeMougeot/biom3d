import argparse
import os
from omero.cli import cli_login
from omero.gateway import BlitzGateway

from biom3d import omero_downloader 
from biom3d import pred 


def run(obj, target, bui_dir, dir_out, host=None, user=None, pwd=None):
    print("Start dataset/project downloading...")
    if host:
        conn = BlitzGateway(user, pwd, host=host, port=4064)
        conn.connect()
        datasets, dir_in = omero_downloader.download_object(conn, obj, target)
        conn.close()
    else:
        with cli_login() as cli:
            datasets, dir_in = omero_downloader.download_object_cli(cli, obj, target)
    print("Done downloading dataset/project!")

    print("Start prediction...")
    if 'Dataset' in obj:
        dir_in = os.path.join(dir_in, datasets[0].name)
        dir_out = os.path.join(dir_out, datasets[0].name)
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out, exist_ok=True)
        pred.pred(bui_dir, dir_in, dir_out)
    elif 'Project' in obj:
        dir_out = os.path.join(dir_out, os.path.split(dir_in)[-1])
        if not os.path.isdir(dir_out):
            os.makedirs(dir_out, exist_ok=True)
        pred.pred_multiple(bui_dir, dir_in, dir_out)
    else:
        print("[Error] Type of object unknown {}. It should be 'Dataset' or 'Project'".format(obj))
    print("Done prediction!")


if __name__=='__main__':

    # parser
    parser = argparse.ArgumentParser(description="Main training file.")
    parser.add_argument('--obj', type=str,
        help="Download object: 'Project:ID' or 'Dataset:ID'")
    parser.add_argument('--target', type=str, default="data/to_pred/",
        help="Directory name to download into")
    parser.add_argument("--bui_dir", type=str, default="logs/unet_nucleus",
        help="Path of the builder directory")
    parser.add_argument("--dir_out", type=str, default="data/pred/",
        help="Path to the output prediction directory")
    parser.add_argument('--hostname', type=str, 
        help="(optional) Host name for Omero server. If not mentioned use the CLI.")
    parser.add_argument('--username', type=str, 
        help="(optional) User name for Omero server")
    parser.add_argument('--password', type=str, 
        help="(optional) Password for Omero server")
    # parser.add_argument("-e", "--eval_only", default=False,  action='store_true', dest='eval_only',
    #     help="Do only the evaluation and skip the prediction (predictions must have been done already.)") 
    args = parser.parse_args()

    run(
        obj=args.obj,
        target=args.target,
        bui_dir=args.bui_dir,
        dir_out=args.dir_out,
        host=args.hostname,
        user=args.username,
        pwd=args.password,
    )