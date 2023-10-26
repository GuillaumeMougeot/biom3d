# Installation 

As biom3d integrates a Graphical User Interface (GUI), an Application Programming Interface (API) and a source code, this installation tutorial has been divided into three sections. The first section is addressed to GUI-users, where you will be able to install the Graphical User Interface of biom3d with a simple link. The second section is addressed to programmers who would like to use the API of biom3d.The third section is addressed to programmers who would like to configure a Python environment to install the source code of biom3d.

## Graphical User Interface 

Biom3d has been deployed for Windows only. Use the following link to download the latest executable file: [link](https://github.com/GuillaumeMougeot/biom3d/releases/). To use biom3d, double-click on the downloaded file. 

> This downloadable version is remote only which means that you must connect to a Linux server where biom3d is installed. To install biom3d on a server, follow the API instructions below.

To use the local version of Biom3d, meaning that computations will be executed on local computer, you must follow the API installation instructions below.

> Warning: The local mode means that all the computation will be executed on the local computer, where Biom3d is installed. In this case, a good graphic card will have to be available on the local computer, where "good" means a NVidia GPUs with at least 8Go of VRAM such as: T4, Geforce GTX 1080, RTX 2080Ti, RTX 3090, P100, V100 or A100. Other NVidia graphic cards with smaller capacity (such as Quadro T1000 or Geforce GTX 1060) might work but will require to reduce the patch size to fit in the GPU memory. The remote mode will avoid the need of a good graphic card on a local computer by allowing you to use a remote server where a good graphic card is installed. In this case, no special requirement are needed on the local computer and the API of biom3d will have to be installed on the remote server (follow the next section to learn how to install the API version).

## Application Programming Interface

**Requirements**:
* A NVidia GPUs with at least 10Go of VRAM (at least a Geforce GTX 1080), smaller Graphic card might also work but some reconfiguration might be required (such as the patch size). We tested the following graphic cards: T4, RTX 2080Ti, RTX 3090, P100, V100, A100 and Quadro T1000 (with patch size reduction). 
* Windows 10 or Ubuntu 18.04 (other OS have not been tested)

There are two types of installations:
* **Conda installation** includes Python installation and most of heavy library installation such as CUDA, CuDNN and Pytorch. Conda installation also ease the installation of Omero (notably for Windows users).
* **Python environment** installation requires to manually install Python, CUDA, CuDNN and Pytorch but might be more adapted to all GPU types.

### Conda installation

* Install [Anaconda](https://www.anaconda.com/download).
* Start an Anaconda prompt. For Windows user, look for "Anaconda prompt" in your Windows search bar.
* Create a new environment (here named `b3d` but it is up to you): (type in the prompt and press Enter)

```
conda create --name b3d
```

* Activate your new environment: (type + Enter)

```
conda activate b3d
```

* Install Pytorch. The best is to look on [Pytorch website](https://pytorch.org/get-started/locally/). Make sure Conda is ticked. You can removed torchvision and torchaudio which are not used by Biom3d. An example of installation command is:

```
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

* Install biom3d:

```
python -m pip install biom3d
```

* (Optional) If you intended to use Omero or the remote version of Biom3d: (two command lines)

```
conda install -c ome omero-py
python -m pip install biom3d[gui]
```

### Python virtual environment

* Install CUDA and CuDNN (cf [Nvidia doc](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)). We tested CUDA 11.6 and 11.7. **Before installing CUDA and CuDNN, please make sure that Pytorch is compatible (we tested Pytorch 1.10, 1.11, 1.12, 1.13 and 2.0)**. You can find archived version 11.7 of CUDA [here](https://developer.nvidia.com/cuda-11-7-0-download-archive) and version 8.6.0 of CuDNN [here](https://developer.nvidia.com/rdp/cudnn-archive).
* Install Python 3.8 or 3.9 or 3.10 (newer or older version have not been tested). For Windows users, you can install Python from [here](https://www.python.org/downloads/windows/) and please make sure to "Add Python to environment variable" during installation, or, if already installed, please add it.
* Setup a new Python environment before installing biom3d. You can do that by opening a new terminal or the Command Prompt in Windows (type 'cmd' in the Windows search bar) and type:

```
python -m venv b3d
```

> This command will create a folder where the command prompt/terminal is opened (on Windows by default it will be `C:\Users\your_username`). If you want to change this path either you can type `cd path/to/your/folder` or, in Windows, open a Windows folder explorer in the appropriate location and type in the path location bar `cmd`.

> If you have an error indicating that Python cannot be found on your system, please make sure that you have tick the box "Add Python to environment variable" during installation, or, add Python to your environment variables [How to install Python on Windows](https://www.digitalocean.com/community/tutorials/install-python-windows-10).

* Activate your new environment:

(for Windows users)
```
b3d\Scripts\activate
```

(for Linux users)
```
source b3d/bin/activate
```

* Install Pytorch. Go to Pytorch [website](https://pytorch.org/get-started/locally/) and install Pytorch with Pip and the right OS and CUDA version. Once installed in your virtual environment, please make sure that you have access to your NVidia GPU with the following command: `python -c "import torch; print(torch.cuda.is_available())"`.


* If the previous requirements are fulfilled, install the API of biom3d:

```
pip install biom3d
```

If would like to install the GUI as well, type the following:

> Warning: For Windows users: you might need to install [Visual Studio C++ 14.0](https://stackoverflow.com/questions/29846087/error-microsoft-visual-c-14-0-is-required-unable-to-find-vcvarsall-bat) to install `zeroc` dependency. If it does not work, please refers to the Anaconda installation. 

> For Linux users, to fasten the installation of zeroc-ice, dependency of biom3d[gui]. We strongly advise to install it indenpendently from prebuilt wheel [here](https://github.com/orgs/ome/repositories?q=zeroc). 

```
pip install biom3d[gui]
```

* (Optionally) You can try to start the GUI:

```
python -m biom3d.gui
```

## Source code

Please first follow the requirements detailed in the API section.

Once pytorch installed, you can download the source code via GitHub by cloning the biom3d repository:

```
git clone https://github.com/GuillaumeMougeot/biom3d.git
```

Or by forking the repository with your GitHub account and cloning it after.

`cd` to your newly downloaded biom3d repository and the installation can be completed with the following command (after activating your pip environment):

```
pip install -e .
```

*Optional*: If planning to edit the GUI source code or to use omero_dowloader to download datasets/projects from omero, please use the following command:

```
pip install -e .[gui]
```

*Optional*: If planning to edit the documentation, please use the following command:

```
pip install -e .[docs]
```
