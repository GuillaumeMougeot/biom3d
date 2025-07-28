# Docker deployment
Biom3d provides Docker images, and this section will describe their maintenance and usage in more detail. 
They are created by `template.dockerfile` here :
```{literalinclude} ../../deployment/dockerfiles/template.dockerfile
:language: Dockerfile
:linenos:
```
> Available images are Linux only, Windows ones aren't on our backlog.

## Build arguments 
The following build arguments are supported :
- `BASE_IMAGE` : The base image used to build Biom3d images. By default, it's ubuntu:22.04. Our images aiming for GPU (Nvidia) uses official `PyTorch` images.
- `TARGET` : Either cpu or gpu, it ils only used in our [CI/CD](cicd.md) that use it to automatically create the tag and create a CUDA symlink in the image for usage sake.
- `PYTHON_VERSION` : Python version used in the image, we recommend the `3.11` as some Biom3d dependencies aren't all compatible with all version.
- `OMERO_VERSION` : The `omero-py` package version. For now only the `5.21.0` has been tested.
- `TESTED` : Indicate the image stability
  - `1` (default), tested and stable
  - `0`, the entry point will display a warning message. It is used for images that should work (theoretically) but couldn't be tested extensively.

## Installing dependencies
Biom3d automatically install its dependencies with :
```bash
pip install biom3d
```

But some optionnal dependencies require additional steps :
- **`omero-py`** : It require `zeroc-ice<3.7` difficult to find for Linux, prebuild found [here](https://github.com/glencoesoftware/zeroc-ice-py-linux-x86_64/releases). Then we can install `omero-py`. Always install `zeroc-ice` before `omero-py` or it will try to compile it from C++.
- **`ezomero`** must be installed with `--no-deps` or it will downgrade `numpy` to an incompatible version and break everything. Other dependencies use `numpy 2.x` that is marked as not compatible with `ezomero` but the `2.2.6` hasn't created a problem. Expect a incompatibility warning that you can ignore. As it is installed with `--no-deps`, always install it after `omero-py`.
- **`tkinter`** comes with system dependencies but is easily installed with `apt install python${PYTHON_VERSION}-tk`.  

## Entrypoint
The default entrypoint is 
dockerfile` here :
```{literalinclude} ../../deployment/dockerfiles/entrypoint.sh
:language: bash
:linenos:
```

It displays a warning if needed and launches Biom3d, waiting for a submodule and its arguments.
This script is intentionally simple — feel free to replace it with a custom entrypoint suited to your use case.

## Other specificities
The `WORKDIR` is set on `/workspace` which mean that dataset volume should be attached there. If you want a more granular approach (for example that `preprocess_train` doesn't create preprocessed data folder there) you can do it by modifying the `ENTRYPOINT` script or `WORKDIR` (or both).

The line `ln -s /opt/conda/lib/libnvrtc.so.11.2 /opt/conda/lib/libnvrtc.so || true` is here to do a symlink so that `torch` find the cuda drivers.