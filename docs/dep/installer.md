# Installer
The Biom3d installer is a self-contained package intended to simplify deployment, it is aimed to Biologist so it is very simple to use. It includes a full Python environment and automatically updates itself on first run to match the host system’s CUDA configuration.

There are currently two version of those :
- **Windows 10/11** in `x86_64` architecture
- **macOS** on `arm64` architecture (hasn't be fully tested yet)

> Linux hasn't an installer as it is hard to make something universal and we consider Linux users able to do a [classic installation](../installation).

Use :
- Download it on a release
- Unzip it
- Execute the `.bat` (Windows) or `.app` (MacOS)

## Windows
It follow this structure :
```
Biom3d/
├── bin/
│   ├── Scripts/
│   │   ├── conda-unpack.exe
│   │   └── ...
│   ├── ...
│   ├── python.exe
│   ├── auto_update.py
│   ├── env.bat
│   └── ...
├── Biom3d.bat
└── Biom3d.ico
```

## MacOS
It follow the MacOS application structure :
```
Biom3d.app/
└── Contents/
    ├── Resources/
    │   └── Biom3d.icns
    ├── MacOS/
    │   ├── bin/
    │   │   ├── bin/
    │   │   │   ├── conda-unpack
    │   │   │   ├── python3.11
    │   │   │   └── ...
    │   │   └── ...
    │   └── Biom3d.sh
    └── Info.plist
```

## Logic
This section details the logic behind the executables and packaging scripts used to build and launch the Biom3d installer.

### Launcher scripts
First are the "executables", `Biom3d.bat` (Windows) and `Biom3d.sh` (MacOS, and potentially Linux).

They are very simple, here is their algorithm :
1. Get variables from `bin\env.bat` (Windows) or `bin/env.sh` (macOS/Linux)
2. If it is first launch (given by `FIRST_LAUNCH` variable):
   1. Execute `conda-unpack`, it will make the virtual environment usable. 
   2. Execute `bin\auto_update.py` (described [here](#auto-updating)). 
      > For the moment, macOS build doesn't use `auto_update.py` as Mac use Metal GPU drivers instead of CUDA and so the script is irrelevant there.
3. Execute `biom3d.gui` with the included `python` to launch Biom3d.

### Packaging
Then are the packing script, `pack.bat` and `pack.sh`.
1. **Detect system architecure** :
    We retrive the building machine processor architecture, on Windows it was hard to obtain `x86_64` instead of `AMD64` that was unclear for non programmer so we decided that it should be passed as an argument (with a default value of `x86_64`).
2. **Create the conda environment** :
    We create a `conda` environment with `tkinter` and `python 3.11` (it has the most compatibilities with Biom3d dependencies). 
    We are assuming two point :
     - `conda` is installed and in `PATH`
     - If there already is an environment with the same name, it has the same purpose and is reused.
3. **Install dependencies** : 
   - We activate our environment.
   - Install `conda-pack`, it will allow us to export our environment.
   - We install `pip 23.1` with `conda` as it is a stable version of `pip` and that if not reinstalled we will have a `conda/pip` conflict at packing step and it will crash.
   - We then install all that is necessary for omero : 
     - `zeroc-ica 3.6.5` with `conda`. 
     - Then the others `omero-py` dependencies with `pip` to finally install `omero-py`. On Windows `omero-py` would try to recompile `zeroc-ice` so we use `--no-deps`
     - `ezomero` with `pip` and `--no-deps`. We use `--no-deps` as `ezomero` would reinstall `numpy` and break other packages (they're should be a incompatibility warning with `numpy 2.x` and `ezomero` but we tested with `numpy 2.2.6` and it worked fine).
4. **Install Biom3d** :
   With a simple `pip install .` with source code. The script are made to be used in the CI/CD so we assume we are in the repository.
5. **Creating the folder** 
   We create the folder with the folder structure describe earlier.
6. **Packing**
   - Then we pack in the `bin/` subfolder with the command `conda pack --format=no-archive -o %DIR%\bin`.
   - Copy the `auto_update.py` and `env.bat` or `env.sh` in `bin/`.
   - We copy the `logo.ico` and `Biom3d.bat` or `logo.icns` and `Biom3d.sh`.
7. **Zipping**
  We zip it with the following name convention : `Biom3d_$OS_$ARCHITECURE.zip`.

Note that the hardest part of this scipt is dependencies for two reasons :
- You must find the correct versions so all is compatible (hence `python 3.11`)
- `conda-pack` doesn't like when `pip` and `conda` touch the same files, so you must find an installation order that avoid those case. The current installation order has been empirically tested to respect that.

## Auto updating
Here is the `auto_update.py` script :
dockerfile` here :
```{literalinclude} ../../deployment/exe/auto_update.py
:language: python
:linenos:
```

It detect the major version of CUDA with either `nvcc` or `nvidia-smi` and install the `x.8` version that should be retrocompatible with lower `x.y` version. If no CUDA is found, it keep `torch cpu`. It can be easily augmented to other drivers if Biom3d implement them.

## Other
The executable script actually just launch the GUI meaning that all limitation of the GUI are kept, however it is possible to use the environment by using the python executable in `bin/` (eg: `bin/bin/python3.11 -m biom3d.pred ...` on macOS).