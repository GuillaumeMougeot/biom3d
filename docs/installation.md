# Installation 

As biom3d integrates a Remote Graphical User Interface (GUI), a Local GUI, a package or Application Programming Interface (API) and a source code, this installation tutorial has been divided into four sections. The first section is addressed to remote GUI-users, non programmers with limited computation means. The second section is addressed to non programmer who would like to use Biom3d on their own computer by using an installer/executable. The third section is for programmer that want the API of biom3d. The fourth section is addressed to programmers who would like to configure a Python environment to install the source code of Biom3d and eventually modify it.

## Remote graphical interface
Biom3d has been deployed for Windows and macOS only. Use the following link to download the latest executable file (end with Remote): [link](https://github.com/GuillaumeMougeot/biom3d/releases/). 

To use Biom3d, double-click on the downloaded file. Then you can follow the [remote documentation](remote).

> This downloadable version is remote only which means that you must connect to a Linux server where biom3d is installed. To install biom3d on a server, follow the [server documentation](server).
> You won't need a powerful computer as every computation will be done one server, however, if you use a dataset that is on your computer, a good internet connexion with the server will speed up the dataset transfert.

> You may encounter a security warning as it is not recongnized by Microsoft and Apple, you can ignore it (only for Biom3d, don't ignore security warning)  

## Hardware requirement
**OS**
- Windows 10 and 11
- Ubuntu >=18
- macOS 14

Other OS or version have not been tested.

**Architecture**

Biom3d can run on `x86_64` processor (most common for Windows and Linux) and on `arm64` processor (for macOS). 

**Memory** 

Will mostly depends on your dataset. At least 12Go but we recommend 16Go or higher (lower than 12Go can still work).

**GPU**

A GPU is not necessary but it is recommended  as it is significantly slower without. The GPU must be a Nvidia GPU with CUDA 11 or 12 compatibility (other version have not been tested). 
AMD with RocM or Apple with Metal are not implemented yet. 
At least 10Go of VRAM is advised (Predictions can be done with less but not training), smaller Graphic card might also work but some reconfiguration might be required (such as the patch size). 
We tested the following graphic cards: T4, RTX 2080Ti, RTX 3090, P100, V100, A100 and Quadro T1000 (with patch size reduction).

## Local graphical interface
To use the local version of Biom3d, meaning that computations will be executed on local computer, the simpliest way is to use the given zip for Windows or macOS. You can still use remote mode with this version.

> Warning: The local mode means that all the computation will be executed on the local computer, where Biom3d is installed. In this case, check the [requirement](#hardware-requirement).

**Installation** 
- Download the zip for your OS [here](https://github.com/GuillaumeMougeot/biom3d/releases/)
- Extract the contained folder in your chosen destination
- It is installed, you can now use it, on first use it will scan your computer for GPU and auto-update its environment so it may take several minutes
  - On macOS, the extracted folder is a `.app` so you just have to double clic on it
  - On Windows, open the folder and double click on `Biom3d.bat`

> You may encounter a security warning as it is not recongnized by Microsoft and Apple, you can ignore it (only for Biom3d, don't ignore security warning)  

## Installing the package
Check the [requirement](#hardware-requirement) first.

There are two types of installations:
* **Conda installation** is easier.
* **Python environment** will be lighter and more versatile.

Python >=3.9 is needed, it is recommened to use python 3.11 as it has been tested and is completly compatible with all dependencies.

To know the CUDA version of your GPU, simply use `nvidia-smi`
```text
> nvidia-smi
Thu Jul 17 11:43:05 2025
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 457.51       Driver Version: 457.51       CUDA Version: 11.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce GTX 106... WDDM  | 00000000:01:00.0  On |                  N/A |
| 43%   39C    P2    22W / 120W |   1054MiB /  6144MiB |      3%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```
If your card use CUDA 11.x or 12.x you can use PyTorch with CUDA 11.8 or 12.8. Recent version of PyTorch (the ones we uses) comes with CUDA and cuDNN so you don't have to manually install it.

### Conda installation
* Install [Anaconda](https://www.anaconda.com/download) or [Miniforge](https://github.com/conda-forge/miniforge) (which is a lighter and free version of Anaconda).
* Start an Anaconda (or Miniforge) prompt. For Windows user, look for "Anaconda prompt" (or "Miniforge prompt") in your Windows search bar.
* Create a new environment (here named `b3d` but it is up to you): (type in the prompt and press Enter)

```bash
# If you plan to use the GUI
conda create --name b3d python 3.11 tk

# If you plan to use only command lines (or just the API)
conda create --name b3d python 3.11 
```

* Activate your new environment: (type + Enter)

```bash
conda activate b3d
```

* Install Pytorch. The best is to look on [Pytorch website](https://pytorch.org/get-started/locally/). Conda is not available anymore so we have to pass through pip. We don't use `torchaudio` or `torchvision` so you can remove them.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
# Or
pip install torch --index-url https://download.pytorch.org/whl/cu128
# Or (cpu only)
pip install torch
```

* Install OMERO (optionnal)
```bash
conda install -c ome omero-py
# It is important to use --no-deps here to avoid breaking other dependencies
pip install --no-deps ezomero
```
Here you should have a warning/non blocking error about a conflict between `ezomero` and `numpy` as `ezomero` doesn't support `numpy 2.x`, however, using a lower version of `numpy` would break `torch` and we confirmed that `ezomero` work with `numpy 2.2.6`.  

* Install biom3d:

```bash
pip install biom3d
```

### Python virtual environment
* Install Python 3.11 [link for Windows](https://www.python.org/downloads/release/python-3110/) or `sudo apt install python3.11` (if it says it doesn't exist you may have to use `deadsnake` repository `sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update`)
* If you want to use the GUI, you must install `tkinter`, on Windows it should be included in python installation (or restart the installtaion > custom installation > tick tcl/tk and IDLE), on Linux `sudo apt install python3.11 tk`.
* Setup a new Python environment before installing Biom3d. You can do that by opening a new terminal or the Command Prompt in Windows (type 'cmd' in the Windows search bar) and type:

```bash
python -m venv b3d
```
* If you have several version of python you can make a specific call, you can replace every `python` with `python3.11` and every `pip`by `python3.11 -m pip` to be sure it use the good version.
```bash
python3.11 -m venv b3d
```

> This command will create a folder where the command prompt/terminal is opened (on Windows by default it will be `C:\Users\your_username`). If you want to change this path either you can type `cd path/to/your/folder` or, in Windows, open a Windows folder explorer in the appropriate location and type in the path location bar `cmd`.

> If you have an error indicating that Python cannot be found on your system, please make sure that you have tick the box "Add Python to environment variable" during installation, or, add Python to your environment variables [How to install Python on Windows](https://www.digitalocean.com/community/tutorials/install-python-windows-10).

* Activate your new environment:

(for Windows users)
```batch
b3d\Scripts\activate
```

(for Linux users)
```bash
source b3d/bin/activate
```

* Install Pytorch. Go to Pytorch [website](https://pytorch.org/get-started/locally/) and install Pytorch with Pip and the right OS and CUDA version. Once installed in your virtual environment, please make sure that you have access to your NVidia GPU with the following command: `python -c "import torch; print(torch.cuda.is_available())"`.
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
# Or
pip install torch --index-url https://download.pytorch.org/whl/cu128
# Or (cpu only)
pip install torch
```

* If the previous requirements are fulfilled, install the API of biom3d:

```bash
pip install biom3d
```

If would like to use OMERO as well, type the following:
On Windows
> Warning: On Windows, `omero-py` always tries to recompile manually `zeroc-ice`. So we preinstall `omero-py` dependencies mannually before installing it without dependencies. `ezomero` also need to be installed without dependencies or it would break other packages.

> For Linux users, to fasten the installation of zeroc-ice, dependency of biom3d[gui]. We strongly advise to install it indenpendently from prebuilt wheel [here](https://github.com/orgs/ome/repositories?q=zeroc) or the [one](https://github.com/glencoesoftware/zeroc-ice-py-linux-x86_64/releases/) used in our docker images. 

On Windows :
```bash
pip install zeroc-ice==3.6.5 
pip install pillow future portalocker pywin32 requests "urllib3<2"
#Forced to do --no-deps because it would try to reinstall zeroc-ice by compiling it
pip install omero-py --no-deps
pip install ezomero --no-deps
```

On Linux :
```bash
pip install https://github.com/glencoesoftware/zeroc-ice-py-linux-x86_64/releases/download/20240202/zeroc_ice-3.6.5-cp311-cp311-manylinux_2_28_x86_64.whl 
pip install omero-py
pip install ezomero --no-deps
```

* (Optionally) You can try to start the GUI:

```bash
python -m biom3d.gui
# Or 
Biom3d
```


## Source code

Please first follow the requirements detailed in the [installating as a package](#installing-the-package).

Once PyTorch installed (eventually tkinter and omero-py too), you can download the source code via GitHub by cloning the Biom3d repository:

```bash
git clone https://github.com/GuillaumeMougeot/biom3d.git
```

Or by forking the repository with your GitHub account and cloning it after.

`cd` to your newly downloaded biom3d repository and the installation can be completed with the following command (after activating your pip environment):

```bash
pip install -e .
```

*Optional*: If planning to edit the documentation, please use the following command:

```
pip install -e .[docs]
```
