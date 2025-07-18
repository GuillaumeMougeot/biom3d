# Continuous Integration / Deployment
Biom3d now has a continuous deployment pipeline that activate when you push a tag.

## How to trigger it
It trigger when a tag is pushed, typically :
```bash
    git tag vx.y.z
    git push origin tag vx.y.z
```
The tag name must be in the specific format above, starting with a 'v' and with [semantic versionning](https://semver.org/) (optional but better).
You can also remove a tag by doing :
```bash
    git tag - vx.y.z # Remove local tag
    git push origin --delete vx.y.z # Remove remote tag
```
So if the CI/CD crash or you make a mistake, you can safely delete the tag and create a new one.

## Working
Here is the script :
```{literalinclude} ../../.github/workflows/build.yml
:language: yaml
:linenos:
```

The pipeline is composed of 5 jobs :
- Retrieve Docker information from a JSON file
- Create and push Docker images
- Building on macOS
- Building on Windows
- Create a GitHub release

### Retrieving Docker info
Reading Json files is natively supported in GitHub Actions, so we can just read it and store it in a GitHub variable. A YAML format was considered to make the file more editable, but it would have introduced unnecessary complexity.

### Creating Docker Images
It is run on `ubuntu-latest` as it has Docker pre-installed. We use the JSON parsed by the [previous step](#retrieving-docker-info) to determine build argument and other parameters. 

Here is the list of supported keys:
- `architecture`: Processor architecture
- `torch_version`: PyTorch version to install and use. If using a PyTorch base image, it should match this version. It is used in the name of the image.
- `base_image` : Base Docker Image used to build from.
- `python_version` : Python version used in the image (`3.11` is recommended)
- `omero_version` : OMERO version used in the image (`5.21.0` is recommended)
- `target` : Must be either `"cpu"` or `"gpu"`, it is mandatory only if `"cpu"` value to determine the name and prevent creating a symlink for non existing GPU library.
- `cuda_version` : CUDA version to use. If using a PyTorch base image, it should match this version. It is used in the name of the image. Don't use if `target:"cpu"`.
- `cudnn_version` : cuDNN version to use. If using a PyTorch base image, it should match this version. It is used in the name of the image. Don't use if `target:"cpu"`.
- `tested` : Either `1` (image has been fully tested and validated) or `0` (not tested, or only partially).

Here is an example Json :
```json
{
  "configs": [
    {
      "architecture": "x86_64",
      "torch_version": "2.7.1",
      "base_image": "ubuntu:22.04",
      "python_version": "3.11",
      "omero_version": "5.21.0",
      "target": "cpu",
      "tested":1
    },
    {
      "architecture": "x86_64",
      "torch_version": "2.3.1",
      "cuda_version": "11.8",
      "cudnn_version": "8",
      "base_image": "pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime",
      "python_version": "3.11",
      "omero_version": "5.21.0",
      "tested":1
    }]
}
```

As shown above, the file consists of a "configs" array, where each object defines one image to build using the parameters described.

Each image defined in the JSON file is built the pushed on **DockerHub** using `DOCKER_USERNAME` and `DOCKER_PASSWORD` (or more precisely access toker) stored in GitHub Secrets. 

For security reason, we recommend :
- Restricting access to those variable to trusted collaborators 
- Change the access token on a regular basis.

### Building
Here we will describe the differents steps of building. The overall logic the same accross plateforms, but syntaxe vary depending on the OS. We will go into syntax details only where necessary. 

Keep in mind that the build is done in the `GITHUB_WORKSPACE`, that is a the root of the GitHub repository. 

**Steps**
- **Versioning on macOS :**
  macOS has a additional step, it is to retrieve the release version and modify the `CFBundleVersion` in `Info.plist`, so the application metadata reflects the correct version.
- **Conda installation :**
  We install Conda and initialize it. 
  On Windows we use `Invoke-WebRequest` instead of `curl` because :
  -  In PowerShell, `curl` is an alias of `Invoke-WebRequest`.
  -  Using a Bash terminal to run `curl` causes permission issues.
  We add Conda to the path, on MacOS, we also add it to the `GITHUB_ENV` variable so it is correctly transfered between steps.
- **Architecture detection :**
  We retrive the runner architecture and store it in a `GITHUB_OUTPUT` variable. 
  On Windows, we added a switch that replace `"AMD64"` with `"x86_64"` to avoid ambiguity.
- **Packaging :**
  We move to `deployment/exe/os` directory and use the packing script described in the [installer documentation](installer.md). 
  On macOS we have to reactivate Conda.
- **Setting up remote :** 
  Once the installer is created, we prepare the remote version by switching the `REMOTE=False` to `REMOTE=True` in `gui.py`. 
  This is intended to let `pyinstaller` statically determine which imports are required (although it doesn't fully work), but more importantly, it hides the **"Start locally"** option in the GUI.
- **Creating a minimal pyvenv :**
  Instead of using the existing Conda environment, we create a minimal `venv`.  
  This significantly reduces the final build size — from around **200MB to 15MB**.  
  We then use `pyinstaller` to build `gui.py`.  
  Note: For unknown reasons, `pyinstaller` places the executable **outside** the `GITHUB_WORKSPACE`, so we manually copy it to the root for access in later steps.
- **Creating the build artifact :**
  We create a folder containing both the installer and the remote executable, and place it at the repository root (it simplify the path for future use). This folder is then exported as a GitHub artifact.

This concludes the build process.

### Release
This step is straightforward and only runs after all other jobs have completed successfully. 
- It will download MacOS and Windows artifacts. 
- It extracts the corresponding changelog section from `CHANGELOG.md`, based on the pushed tag. 
  > Important: `CHANGELOG.md` **must include a section** matching the pushed tag, for example:  
  > `## [v1.0.0] - 2025-July-9`
  If this part fail, it will create a release with the following body `Version $VERSION not found in CHANGELOG.md`. 
- The CI/CD contain a step to zip the source code however it was commented as GitHub automatically does it. 
- It create a release associated to the tag, it includes :
  - The changelog as body.
  - The two installers.
  - The two remote executable.
  - Source code archived (in `.zip` and `.tar.gz`).