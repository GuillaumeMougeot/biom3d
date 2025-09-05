"""Group preprocessing, training and prediction with OMERO data."""
#TODO rename this file and raise errors when needed
import argparse
import os
import shutil
from typing import Literal, Optional
import zipfile
from omero.cli import cli_login
from biom3d import omero_downloader 
from biom3d import omero_uploader
from biom3d import omero_pred
from biom3d import preprocess_train
from biom3d import preprocess
from biom3d import train 

def run(
    obj_raw: str,
    obj_mask: Optional[str],
    num_classes: int,
    config_dir: str,
    base_config: str,
    ct_norm: bool,
    desc: str,
    max_dim: int,
    num_epochs: int,
    target: Literal["preprocess","preprocess_train","train","pred"],
    action: str,
    host: Optional[str] = None,
    user: Optional[str] = None,
    pwd: Optional[str] = None,
    upload_id: Optional[int] = None,
    dir_out: Optional[str] = None,
    omero_session_id: Optional[str] = None
) -> Optional[str]:
    """
    Execute the pipeline for preprocessing, training, or prediction using OMERO data.

    Depending on the specified action (`preprocess`, `preprocess_train`, `train`, or `pred`), this function:
    - Downloads raw and optionally mask datasets from OMERO (via API or CLI).
    - Performs preprocessing and/or training.
    - Downloads model configurations and runs inference.
    - Optionally uploads resulting images/logs back to OMERO.
    - Generates learning curve plots after training.

    Parameters
    ----------
    obj_raw : str
        Identifier of the raw OMERO object (e.g., "Dataset:123").
    obj_mask : str, optional
        Identifier of the corresponding mask object, if available.
    num_classes : int
        Number of segmentation classes for the training.
    config_dir : str
        Target folder for auto-configuration result.
    base_config : str
        Path to an existing configuration file which will be updated with the preprocessed values.
    ct_norm : bool
        Whether to apply CT normalization during preprocessing.
    desc : str
        Model name.
    max_dim : int
        Maximum dimension of a patch.
    num_epochs : int
        Number of epochs for model training.
    target : str
        Output directory for data to download into.
    action : str literal
        Action to perform. One of: `"preprocess"`, `"preprocess_train"`, `"train"`, `"pred"`.
    host : str, optional
        OMERO server host (for API-based downloads).
    user : str, optional
        OMERO username.
    pwd : str, optional
        OMERO password.
    upload_id : int, optional
        OMERO project ID where outputs should be uploaded.
    dir_out : str, optional
        Directory to store prediction outputs.
    omero_session_id : str, optional
        Session ID for authenticated OMERO access.

    Returns
    -------
    str or None
        Path to the output directory if applicable (e.g., after training), otherwise None.

    Notes
    -----
    - If `upload_id` is provided, results are uploaded back to OMERO after training or preprocessing.
    - Model logs are extracted and plotted to visualize learning curves.
    - Model zip files and config attachments are managed using `download_attachment`.
    - On Windows, the model upload at the end will fail due to lock preventing zipping.

    Raises
    ------
    RuntimeError
        If object type is unrecognized or missing required information.
    """
    if action == "preprocess" or action=="preprocess_train" :
        print("Start dataset/project downloading...")
        if host is not None and omero_session_id is None:
            datasets, dir_in = omero_downloader.download_object(user, pwd, host, obj_raw, target, omero_session_id)
            if obj_mask is not None :
                datasets_mask, dir_in_mask = omero_downloader.download_object(user, pwd, host, obj_mask, target, omero_session_id)
        elif omero_session_id is not None and host is not None:
            datasets, dir_in = omero_downloader.download_object(user, pwd, host, obj_raw, target, omero_session_id)
            if obj_mask is not None :
                datasets_mask, dir_in_mask = omero_downloader.download_object(user, pwd, host, obj_mask, target,omero_session_id)        
        else:
            with cli_login() as cli:
                datasets, dir_in = omero_downloader.download_object_cli(cli, obj_raw, target)
                if obj_mask is not None :
                    datasets_mask, dir_in_mask = omero_downloader.download_object_cli(cli, obj_mask, target)

        print("Done downloading dataset!")


        if 'Dataset' in obj_raw:
            dir_in = os.path.join(dir_in, datasets[0].name)
            dir_in_mask = os.path.join(dir_in_mask, datasets_mask[0].name)

    print("Start Training with Omero...")     
    if action == "preprocess_train" :
        preprocess_train.preprocess_train(
            img_path=dir_in,
            msk_path=dir_in_mask,
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
            img_path=dir_in,
            msk_path=dir_in_mask,
            num_classes=num_classes,
            config_dir=config_dir,
            base_config=base_config,
            ct_norm=ct_norm,
            desc=desc,
            max_dim=max_dim,
            num_epochs=num_epochs
        )

    elif action == "train" :
        conf_dir =omero_downloader.download_attachment(
            hostname=host, 
            username=user, 
            password=pwd, 
            session_id=omero_session_id, 
            attachment_id=config_dir,
            config=True)

        print("Running training with current configuration file :",conf_dir)

        train.train(config=conf_dir)
        try :
            shutil.rmtree(conf_dir)
        except:
            pass 
    elif action == "pred" :
        #Download the model 
        model =omero_downloader.download_attachment(
            hostname=host, 
            username=user, 
            password=pwd, 
            session_id=omero_session_id, 
            attachment_id=config_dir,
            config=False)
        # extract the model
        log_folder = unzip_file(model, os.path.join("logs"))

        target = "data/to_pred"
        if not os.path.isdir(target):
            os.makedirs(target, exist_ok=True)

        attachment_file, _ = os.path.splitext(os.path.basename(log_folder))
        upload_id = int(obj_raw.split(":")[1])
        omero_pred.run(
            obj=obj_raw,
            log=log_folder, 
            dir_out=os.path.join("data","pred"), 
            host = host,
            session_id=omero_session_id, 
            attachment=attachment_file, 
            upload_id=upload_id, 
            target=target)

        try :
            shutil.rmtree(log_folder)
            os.remove(model)
        except:
            pass
    # eventually upload the dataset back into Omero [DEPRECATED]
    if upload_id is not None and host is not None:

        if action == "train" or action == "preprocess_train" :
            # For Training
            logs_path = "./logs" 
            if not os.path.exists(logs_path)  :
                print(f"Directory '{logs_path}' does not exist.")
            else:
                directories = [d for d in os.listdir(logs_path) if os.path.isdir(os.path.join(logs_path, d))]
                if not directories:
                    print("No directories found in the logs path.")
                else:
                    directories.sort(key=lambda d: os.path.getmtime(os.path.join(logs_path, d)), reverse=True)
                    last_folder = directories[0]
                    image_folder = os.path.join(logs_path, last_folder, "image")
                    plot_learning_curve(os.path.join(logs_path, last_folder))
                    omero_uploader.run(username=user, password=pwd, host=host, project=upload_id, path = image_folder ,is_pred=False, attachment=last_folder, session_id =omero_session_id)
                    try :
                        os.remove(os.path.join(logs_path, last_folder+".zip"))
                        shutil.rmtree(os.path.join(logs_path, last_folder))
                    except: 
                        pass
                    shutil.rmtree(target)

            print("Done Training!")
            # print for remote. Format TAG:key:value
            print("REMOTE:dir_out:{}".format(dir_out))
            return dir_out
        elif action == "preprocess" :
            # For Preprocessing
            last_folder =  config_path 
            image_folder = None 
            print("last folder: ",last_folder)
            print("image_folder : ",image_folder)
            omero_uploader.run(username=user, password=pwd, host=host, project=upload_id, path = image_folder ,is_pred=False, attachment=last_folder, session_id =omero_session_id)          

    else:
        print("[Error] Type of object unknown {}. It should be 'Dataset' or 'Project'".format(obj_raw))


def load_csv(filename: str) -> list[list[str]]:
    """
    Load a CSV file and return its content as a list of rows.

    Parameters
    ----------
    filename : str
        Path to the CSV file to load.

    Returns
    -------
    list of list of str
        Data extracted from the CSV file, where each row is a list of string values.

    Notes
    -----
    - Assumes the file is comma-delimited.
    - The file is read entirely into memory.
    """
    from csv import reader
    # Open file in read mode
    file = open(filename,"r")
    # Reading file 
    lines = reader(file)

    # Converting into a list 
    data = list(lines)
    file.close()

    return data

def plot_learning_curve(last_folder: str) -> None:
    """
    Plot training and validation loss curves from a CSV log file.

    The CSV file is expected at `<last_folder>/log/log.csv` and must contain:
    - Epoch numbers in the first column,
    - Training loss in the second column,
    - Validation loss in the third column.

    Parameters
    ----------
    last_folder : str
        Path to the folder containing the training logs.

    Returns
    -------
    None

    Notes
    -----
    - The resulting plot is saved as `<last_folder>/image/Learning_curves_plot.png`.
    """
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

def unzip_file(zip_path: str, extract_to: str) -> str:
    """
    Extract a zip file to a specific directory and return the extraction path.

    Parameters
    ----------
    zip_path : str
        Path to the zip archive.
    extract_to : str
        Directory where contents should be extracted.

    Returns
    -------
    str
        Full path of the directory where the archive was extracted.

    Notes
    -----
    - Creates a subdirectory named after the zip file (without extension) inside `extract_to`.
    - The extracted directory is created if it doesn't exist.
    """
    # Get the base name of the zip file without extension
    zip_base_name = os.path.splitext(os.path.basename(zip_path))[0]

    # Create the full extraction path
    full_extract_path = os.path.join(extract_to, zip_base_name)

    # Ensure the extraction directory exists
    if not os.path.exists(full_extract_path):
        os.makedirs(full_extract_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(full_extract_path)

    print(f"Extracted all files to {full_extract_path}")
    return full_extract_path

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
    if args.action=="preprocess" or args.action=="preprocess_train":
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