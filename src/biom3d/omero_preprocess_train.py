#---------------------------------------------------------------------------
# Predictions with Omero
# This script can download data from Omero, compute predictions,
# and upload back into Omero.
#---------------------------------------------------------------------------

import argparse
import os
import shutil
from omero.cli import cli_login
from biom3d import omero_downloader 
from biom3d import omero_uploader
from biom3d import preprocess_train
from biom3d import preprocess
from biom3d import train 

def run(obj_raw, obj_mask, num_classes, config_dir, base_config, ct_norm, desc, max_dim, num_epochs,  target , action, host=None, user=None, pwd=None, upload_id=None ,dir_out =None, omero_session_id=None):

    if not action == "train" :
        print("Start dataset/project downloading...")
        if host is not None and omero_session_id is None:
            datasets, dir_in = omero_downloader.download_object(user, pwd, host, obj_raw, target, omero_session_id)
            datasets_mask, dir_in_mask = omero_downloader.download_object(user, pwd, host, obj_mask, target, omero_session_id)
        elif omero_session_id is not None and host is not None:
            datasets, dir_in = omero_downloader.download_object(user, pwd, host, obj_raw, target, omero_session_id)
            datasets_mask, dir_in_mask = omero_downloader.download_object(user, pwd, host, obj_mask, target,omero_session_id)        
        else:
            with cli_login() as cli:
                datasets, dir_in = omero_downloader.download_object_cli(cli, obj_raw, target)
                datasets_mask, dir_in_mask = omero_downloader.download_object_cli(cli, obj_mask, target)

        print("Done downloading dataset!")

 
        if 'Dataset' in obj_raw:
            dir_in = os.path.join(dir_in, datasets[0].name)
            dir_in_mask = os.path.join(dir_in_mask, datasets_mask[0].name)
            
    print("Start Training with Omero...")     
    if action == "preprocess_train" :
        preprocess_train.preprocess_train(
            img_dir=dir_in,
            msk_dir=dir_in_mask,
            num_classes=num_classes,
            config_dir=config_dir,
            base_config=base_config,
            ct_norm=ct_norm,
            desc=desc,
            max_dim=max_dim,
            num_epochs=num_epochs
            )
    elif action == "preprocess" :
        config_path = preprocess.auto_config_preprocess(
            img_dir=dir_in,
            msk_dir=dir_in_mask,
            num_classes=num_classes,
            config_dir=config_dir,
            base_config=base_config,
            ct_norm=ct_norm,
            desc=desc,
            max_dim=max_dim,
            num_epochs=num_epochs
        )
    elif action == "train" :
        conf_dir =omero_downloader.download_attachment(hostname=host, username=user, password=pwd, session_id=omero_session_id, attachment_id=config_dir)
        print("Running training with current configuration file :",conf_dir)
        train.train(config=conf_dir)

    # eventually upload the dataset back into Omero [DEPRECATED]
    if upload_id is not None and host is not None:
        
        logs_path = "./logs"  # Use relative path

        if not os.path.exists(logs_path):
            print(f"Directory '{logs_path}' does not exist.")
        else:
            directories = [d for d in os.listdir(logs_path) if os.path.isdir(os.path.join(logs_path, d))]
            if not directories:
                print("No directories found in the logs path.")
            else:
                directories.sort(key=lambda d: os.path.getmtime(os.path.join(logs_path, d)), reverse=True)
                last_folder =  config_path if action == "preprocess" else directories[0]
                image_folder = None if action == "preprocess" else os.path.join(logs_path, last_folder, "image")
                if not action=="preprocess" : plot_learning_curve(os.path.join(logs_path, last_folder))
                
                print("last folder: ",last_folder)
                print("image_folder : ",image_folder)
                omero_uploader.run(username=user, password=pwd, hostname=host, project=upload_id, path = image_folder ,is_pred=False, attachment=last_folder, session_id =omero_session_id)
                try :
                    os.remove(os.path.join(logs_path, last_folder+".zip"))
                except: 
                    pass
                
        if not action =="preprocess" : shutil.rmtree(target)
        print("Done Training!")

        # print for remote. Format TAG:key:value
        print("REMOTE:dir_out:{}".format(dir_out))
        return dir_out

    else:
        print("[Error] Type of object unknown {}. It should be 'Dataset' or 'Project'".format(obj_raw))


def load_csv(filename):
    from csv import reader
    # Open file in read mode
    file = open(filename,"r")
    # Reading file 
    lines = reader(file)
    
    # Converting into a list 
    data = list(lines)
    
    return data

def plot_learning_curve(last_folder):
        import matplotlib.pyplot as plt
        # CSV file path
        print("this is it : ",last_folder)
        csv_file = os.path.join(last_folder+"/log/log.csv")

        # PLOT
        data = load_csv(csv_file)
        # Extract epoch and train_loss, val_loss values
        epochs = [int(row[0]) for row in data[1:]]  # Skip the header row
        train_losses = [float(row[1]) for row in data[1:]]  # Skip the header row
        val_losses = [float(row[2]) for row in data[1:]]  # Skip the header row
        
        plt.clf()  # Clear the current plot
        plt.plot(epochs, train_losses ,label='Train loss')
        plt.plot(epochs, val_losses , label ='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curves')
        plt.grid(True)
        plt.legend()
        plt.pause(0.1)  # Pause for a short duration to allow for updating               
        # save figure locally
        plt.savefig(last_folder+'/image/Learning_curves_plot.png')  
if __name__=='__main__':

    # parser
    parser = argparse.ArgumentParser(description="Training with Omero.")
    parser.add_argument('--raw', type=str,
        help="Download Raw Dataset ")
    parser.add_argument('--mask', type=str,
    help="Download Masks Dataset ")
    parser.add_argument('--target', type=str, default="data/to_train/",
        help="Directory name to download into")
    parser.add_argument('--action', type=str, default="preprocess_train",
    help="Action : preprocess | train | preprocess_train ")
    parser.add_argument("--num_classes", type=int, default=1,
        help="(default=1) Number of classes (types of objects) in the dataset. The background is not included.")
    parser.add_argument("--max_dim", type=int, default=128,
        help="(default=128) max_dim^3 determines the maximum size of patch for auto-config.")
    parser.add_argument("--num_epochs", type=int, default=1000,
        help="(default=1000) Number of epochs for the training.")
    parser.add_argument("--config_dir", type=str, default='configs/',
        help="(default=\'configs/\') Configuration folder to save the auto-configuration.")
    parser.add_argument("--base_config", type=str, default=None,
        help="(default=None) Optional. Path to an existing configuration file which will be updated with the preprocessed values.")
    parser.add_argument("--desc", type=str, default='unet_default',
        help="(default=unet_default) Optional. A name used to describe the model.")
    parser.add_argument("--ct_norm", default=False,  action='store_true', dest='ct_norm',
        help="(default=False) Whether to use CT-Scan normalization routine (cf. nnUNet).")      
    parser.add_argument('--hostname', type=str, default=None,
        help="(optional) Host name for Omero server. If not mentioned use the CLI.")
    parser.add_argument('--username', type=str, default=None,
        help="(optional) User name for Omero server")
    parser.add_argument('--password', type=str, default=None,
        help="(optional) Password for Omero server")
    parser.add_argument('--session_id', default=None,
        help="(optional) Session ID for Omero client")
    args = parser.parse_args()
    
    raw = "Dataset:"+args.raw
    if not args.action=="train":
        mask = "Dataset:"+args.mask
    else :
        mask=None

    run(
        obj_raw=raw,
        obj_mask=mask,
        num_classes=args.num_classes,
        config_dir=args.config_dir,
        base_config=args.base_config,
        ct_norm=args.ct_norm,
        desc=args.desc,
        max_dim=args.max_dim,
        num_epochs=args.num_epochs,       
        target=args.target,
        action=args.action,
        host=args.hostname,
        user=args.username,
        pwd=args.password,
        upload_id=args.raw,
        omero_session_id=args.session_id
    )
    
    # python -m biom3d.omero_preprocess_train --raw  --mask  --num_epochs  --desc  --hostname  --username --password