# Installation 

As this tool has been designed for both programmers and non-programmers, this installation section has been divided into two sections. The first section is addressed to non-programmers, where you will be able to install the Graphical User Interface of Biom3d with a simple link. The second section is addressed to programmers who would like to configure a Python environment to install the source code of Biom3d.

## Graphical User Interface 

Biom3d has been deployed for Windows. Use the following link to download the executable file: [link]. You will be able to use Biom3d directly by clicking on the downloading file.

Careful! Biom3d has two modes: a local mode and a remote mode. The local mode means that all the computation will be executed on the local computer, where Biom3d is installed. In this case, a good graphic card will have to be available on the local computer, where "good" means a NVidia GPUs with at least 12Go of VRAM such as: Geforce GTX 1080, RTX 2080Ti, RTX 3090, P100, V100 or A100. The remote mode will avoid the need of a good graphic card on your computer by allowing you to use a remote server where a good graphic card is installed. In this case, no special requirement are needed on the local computer and the source code version of Biom3d will have to be installed on the remote server (follow the next section to learn how to install the source code version).

## Source code

Requirements:
* A NVidia GPUs (at least a Geforce GTX 1080). We tried with the following graphic cards: RTX 2080Ti, RTX 3090, P100, V100 or A100.
* Windows 10 or Ubuntu 18.04 (other OS have not been tested)
* Python 3.8 or 3.9 or 3.10 (newer or older version have not been tested)
* CUDA & CuDNN (cf [Nvidia doc](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)). We used CUDA 11.6.
* (optional but recommended) Use Conda or Pip-env. The later is recommended: use `python3 -m venv env` to create a virtual environment and `source env/bin/activate` to activate it. 
* pytorch==1.12.1. Please be sure that you have access to your NVidia GPU with the following command: `python -c "import torch; print(torch.cuda.is_available())"`

Once pytorch installed, the installation can be completed with the following command:

```
pip3 install -r requirements.txt
```

Or with the following:

```
pip3 install SimpleITK==2.1.1 pandas==1.4.0 scikit-image==0.19.0 tensorboard==2.8.0 tqdm==4.62.3 numpy==1.21.2 matplotlib==3.5.3 PyYAML==6.0 torchio==0.18.83 protobuf==3.19.3
```

Optional: If planning to use omero_dowloader to download datasets/projects from omero, please install omero-py with the following command:

```
pip3 install omero-py
```

For Windows users, careful with the previous package: you might need to install [Visual Studio C++ 14.0](https://stackoverflow.com/questions/29846087/error-microsoft-visual-c-14-0-is-required-unable-to-find-vcvarsall-bat) to install `zeroc` dependency.