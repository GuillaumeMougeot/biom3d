# Installation 

As biom3d integrates a Graphical User Interface (GUI), an Application Programming Interface (API) and a source code, this installation tutorial has been divided into three sections. The first section is addressed to GUI-users, where you will be able to install the Graphical User Interface of biom3d with a simple link. The second section is addressed to programmers who would like to use the API of biom3d.The third section is addressed to programmers who would like to configure a Python environment to install the source code of biom3d.

## Graphical User Interface 

[DEPRECATION!] The graphical user interface is currently being updated!

Biom3d has been deployed for Windows. Use the following link to download the latest executable file: [link](https://github.com/GuillaumeMougeot/biom3d/releases/). You will be able to use biom3d directly by clicking on the downloading file.

Careful! Biom3d has two modes: a local mode and a remote mode. The local mode means that all the computation will be executed on the local computer, where biom3d is installed. In this case, a good graphic card will have to be available on the local computer, where "good" means a NVidia GPUs with at least 12Go of VRAM such as: Geforce GTX 1080, RTX 2080Ti, RTX 3090, P100, V100 or A100. The remote mode will avoid the need of a good graphic card on your computer by allowing you to use a remote server where a good graphic card is installed. In this case, no special requirement are needed on the local computer and the API of biom3d will have to be installed on the remote server (follow the next section to learn how to install the API version).

> Note: for API users who followed the next tutorial, the GUI is included in the API.

> Warning: the version of the GUI on GitHub is remote only, which means that you must also install the API on a server. If you would like to train your deep learning model locally and use the GUI, please follow the next section. 

## Application Programming Interface

Requirements:
* A NVidia GPUs with at least 12Go of VRAM (at least a Geforce GTX 1080). We tried with the following graphic cards: RTX 2080Ti, RTX 3090, P100, V100 or A100. 
* Windows 10 or Ubuntu 18.04 (other OS have not been tested)
* Python 3.8 or 3.9 or 3.10 (newer or older version have not been tested)
* CUDA & CuDNN (cf [Nvidia doc](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)). We used CUDA 11.6 and 11.7. Before installing CUDA and CuDNN, please be sure that Pytorch is compatible (we tested Pytorch 1.10, 1.11, 1.12 and 1.13). You can find archived version 11.7 of CUDA [here](https://developer.nvidia.com/cuda-11-7-0-download-archive) and version 8.6.0 of CuDNN [here](https://developer.nvidia.com/rdp/cudnn-archive).

We also recommend to first setup a new Python environment before installing biom3d, with:

```
python -m venv b3d
```

and to activate your new environment use:

(for Windows users)
```
b3d\Scripts\activate
```

(for Linux users)
```
source b3d/bin/activate
```

> Note: for some reasons, the Pytorch installation may cause some problems. We thus recommend to install Pytorch independently before installing biom3d. Please go to Pytorch [website](https://pytorch.org/get-started/locally/) and install Pytorch with Pip and the right OS and CUDA version. Once installed in your virtual environment, please be sure that you have access to your NVidia GPU with the following command: `python -c "import torch; print(torch.cuda.is_available())"`.


If the previous requirements are fulfilled, installing the API of biom3d is as simple as:

```
pip install biom3d
```

If you would like to start the GUI, you can run the following command:

```
python -m biom3d.gui -L
```

## Source code

Please follow first the requirements detailed in the API section before the biom3d installation.

Once pytorch installed, you can download the source code via GitHub by cloning the biom3d repository:

```
git clone https://github.com/GuillaumeMougeot/biom3d.git
```

or by forking the repository with your GitHub account and cloning it after.

`cd` to your newly downloaded biom3d repository and the installation can be completed with the following command (after activating your pip environment):

```
pip install -e .
```

Optional: If planning to edit the GUI source code or to use omero_dowloader to download datasets/projects from omero, please use the following command:

```
pip install -e .[gui]
```

For Windows users, careful with the previous package: you might need to install [Visual Studio C++ 14.0](https://stackoverflow.com/questions/29846087/error-microsoft-visual-c-14-0-is-required-unable-to-find-vcvarsall-bat) to install `zeroc` dependency.

Optional: If planning to edit the documentation, please use the following command:

```
pip install -e .[docs]
```
