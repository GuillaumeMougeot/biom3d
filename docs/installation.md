# Installation 

Requirements:
* A NVidia GPUs (at least a Geforce GTX 1080)
* Windows 10 or Ubuntu 18.04 (other OS have not been tested)
* Python 3.8 or 3.9 or 3.10 (newer or older version have not been tested)
* CUDA & CuDNN (cf [Nvidia doc](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html))
* (optional but recommended) Use Conda or Pip-env. The later is recommended: use `python3 -m venv env` to create a virtual environment and `source env/bin/activate` to activate it. 
* pytorch==1.12.1 (please be sure that you have access to your NVidia GPU)

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