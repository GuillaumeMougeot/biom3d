# How to use Biom3d in Docker
Biom3d now has Docker images that you can find [here](). We will explore here some simples use-cases of those images.

## What is Docker ?
Docker is a tool that lets you run software in **isolated environments**, called **containers**.
These containers are based on **images**, prebuilt environments that contain everything the software needs to run (OS, libraries, Python, etc.).
For example, the command `docker run hello-world` creates and runs a container from the official hello-world image. It download the image, create the container and execute the default command of the image.

## Installing Docker
Installing Docker is a complex step and we strongly recommend asking your IT support. 

For Windows you can use [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/).
> Biom3d use Linux-based images, you must enable WSL2 Back-end or Hyper-V backend to run them. 

> Docker Desktop is also available on MacOs but Biom3d doesn't have arm64 images yet.

For Linux, the installation process depends on the distribution.

## With Docker desktop
*Not documented yet*

## With command lines
### Basic Docker commands
This section introduces the essential Docker commands. If you are already used to Docker you can skip to [Running a Biom3d container](#running-a-biom3d-container).

Docker is separated in several submodules and three of them will be used in this tutorial : `image`, `container` and `build`.

#### Image 
The image submodule is here to manipulate images. An image is a blueprint, it contains the whole environment, pre-built, ready-to-use. 

It is defined by two part : the image name (or repository) and the tag. For example let's take `ubuntu:22.04` where `ubuntu` is the image name and `22.04`, is the tag. The tag is here to differentiate the differents versions of the same image. 

Those images are stored in repository such as [DockerHub](https://hub.docker.com/) or directly on your machine. To download an image on your computer you use `docker pull` such as :
```shell
  docker pull helloworld
  docker pull ubuntu:22.04
  docker pull pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
```
It will download it from distant repository, default being `DockerHub`.

Once you have the image on your machine you can see it by doing `docker image ls` :
```shell
  docker image ls
  REPOSITORY        TAG                             IMAGE ID       CREATED              SIZE
  ubuntu            22.04                           1b668a2d748d   2 weeks ago          77.9MB
  pytorch/pytorch   2.7.1-cuda11.8-cudnn9-runtime   cc0fe24aee5e   4 weeks ago          6.48GB
```

If you want to remove an image (to liberate disk space) you can do `docker image rm` with an identifier:
```shell
  docker image rm ubuntu # The images from ubuntu repository
  docker image rm ubuntu:22.04 # A specific image by full name
  docker image rm 1b668a2d748d # A specific image by ID
```
With that we covered the basics on image managment. We can now see how to use them.

#### Container
The `container` submodule is the one used by default when you don't specify one, which mean that those two commands are equivalent :
```shell
  docker run helloworld
  docker container run helloworld
```

##### Running a container
Container are instances of an image, you can run one of them with `docker run` :
```shell
  docker run docker_arguments image_name image_argument
```

With som examples :
```bash
  docker run --rm helloworld
  docker run biom3d pred -i foo -o bar
  docker run ubuntu
  docker run ubuntu:22.04
```
Let's go break down our `docker run docker_arguments image_name image_argument` :
1. **Docker arguments**, they describe how the container should run. Here are the most commons :
   - `--rm` destroy the container once its job is finished.
   - `-n foo` or `--name foo` give a name to the container (here foo).
   - `-e FOO=bar` set an environment variable in the container
   - `-v absolute_path_on_machine:absolute_path_in_container` attach a volume to the container. 
      For example `-v /home/me:/home/me` or `-v C:\users\me:/home/me` will link your user folder to the user folder in the container, which mean that any modification done in this folder in the container is also done outside and reverse. Another example is `-v $(pwd):/data` will link the folder you are in to the `/data`folder. Be careful to not link a folder you don't want to modify or be sure that the things you do in the container doesn't modify it.

    Those are the basics in Docker argument, you can find a complete list [here](https://docs.docker.com/reference/cli/docker/container/run/)

2. **Image name**, seen earlier. The image is pulled if not on your machine and by default the tag used is `latest` if not specified. Biom3d doesn't have a latest tag, so a precise tag must be given, which means :
  ```shell
    docker run biom3d # Will not work
    docker run biom3d:0.0.30-x86_64-torch2.7.1-cpu # Will work
  ```
3. **Image arguments**, they depends on the image entrypoint that is by default not defined. This entrypoint describe what script is launch when you do a run on an image. Biom3d containers have Biom3d as entry point which mean they want to know the module you want to use.
  ```bash
    docker run biom3d:0.0.30-x86_64-torch2.7.1-cpu pred -i foo -o bar # Will run prediction on the foo folder and send it to bar
    docker run biom3d:0.0.30-x86_64-torch2.7.1-cpu # Will crash because the entry point want you to say which submodule you want to use and then its parameters
  ```
  Keep in mind that everything that is written **after** the image name will be treated as a command to execute in the container.

##### Listing containers
Now your container is running but how to monitor it ? Well you can do `docker ps` :
```shell
  docker container ls 
  # Or
  docker ps
  # Or better
  docker ps -a # Will show you even non running container
  CONTAINER ID   IMAGE          COMMAND                  CREATED             STATUS                      PORTS     NAMES
  00e85ccba457   ubuntu:22.04   "/bin/bash"              12 seconds ago      Exited (0) 8 seconds ago              cool_kowalevski
  73ec202bb921   ubuntu:22.04   "/bin/bash"              18 seconds ago      Exited (0) 14 seconds ago             stupefied_banach
```
It will show you the name, id and status (running, stopped,...) of every container. 

##### Cleaning containers
Biom3d container stop once they finish their job (Status Exited like above) so they must be removed after by doing one of the following:
```shell
  docker rm my_biom3d_container_name
  # Or
  docker rm my_biom3d_container_id
  # Or run it with --rm arguments
```
If you want to remove a container while it is running you can do :
```shell
  # Proper way
  docker stop my_biom3d_container
  docker rm my_biom3d_container
  # More violent way
  docker rm my_biom3d_container -f
```
Here is a bonus : to remove all existing container you can do
```shell
  docker rm $(docker ps -aq) # Eventually -f
```
But it is irreversible so use it wisely.

Now you see the basics of running container. If you just want to use Biom3d you can directly go to [running a Biom3d container](#running-a-biom3d-container), if you want to contribute on developement you may be interested on [building](#building)

(building)=
#### Building
If you want to contribute to Biom3d or make your custom images tailored to your needs, you'll have to build an image. To do that you use a `DockerFile`, you can see some examples [here](https://github.com/GuillaumeMougeot/biom3d/tree/deployment/deployment/dockerfiles/examples). 

A DockerFile is a plain text that describe how to build an image. It follow this structure :
```Dockerfile
  FROM baseimage # It is always the first line
  ENV FOO=Bar
  COPY relative_path_from_dockerfile absolute_path_in_image
  ADD source destination
  WORKDIR folder # Set the folder in which you locate the rest of the dockerfile or execution
  RUN command
  ENTRYPOINT my_entrypoint_script # Define the default command
```
Let's break down :
- `FROM` is always the first (or one of the first) command. It is the address of an existing Docker image, and you will add new thing to it. 
- `ENV` set an environment variable.
- `COPY` copy the element at givent location on your drive to the given location in the image. 
- `ADD` work the same way as copy in a sense it copy the source to the destination in the image. But it can also take an URL as a source and it automically exract archives (`.tar`,`tar.gz`,...)
- `WORKDIR path` allow you to go to the given path, it is equivalent to a `cd path` command. For example in the Biom3d DockerFiles, there is always at the end `WORKDIR /workspace`, it means that when you use the container it will save files in `/workspace` and that's why we always want to attach a volume to `/workspace`
- `RUN command` will run the command, following the base image syntaxe (shell for MacOs/Linux, batch or powershell on Windows), it is quite simple but there are some good pratices. One of those are to group `RUN` statement :
  ```
    # Don't do
    RUN mdir /foo
    RUN pip install biom3d
    # But
    RUN mkdir /foo \
    && pip install torch
    # Or on Windows images
    RUN mkdir /foo ^
    && pip install torch
  ``` 
  This will make your image smaller. But while you're creating a new image, keep separated `RUN` statement, it will be easier to debug.
- `ENTRYPOINT` is a script or program (shell, batch, python,..) that will be used each time you access the container.

You can also add arguments with the `ARG FOO=default_value`, and access the value with `${FOO}` (or `%FOO%` if you're using it in a Windows command).

Then you build the image with `docker build`
```shell
  docker build -f path_to_dockerfile .
  docker build . # Only build the file named Dockerfile
  docker build --build_arg FOO=not_bar -f path . # Will change the value of the given ARG
  docker build -f path ./subforlder # The ADD or COPY will be relative to path/subfolder instead of path 
```
With that you should be able to build your image if you need it, the hard part of Biom3d build is the managment of python dependencies (there are examples of DockerFiles on the [git repository](https://github.com/GuillaumeMougeot/biom3d) and a specific guide on Biom3d images [here](../dep/docker)).

### Running a Biom3d container 
Now you should know some basics on how to use Docker, so here we will see how to use Biom3d container themselves. As explained earlier, Biom3d has a specific entrypoint that make containers not (easily) reusable, they are throwable container. 

You run a command -> it executes -> it exits -> it gets removed.

Here is the basic command for using Biom3d container :
```shell
  docker run --rm \
  -v folder_with_data_set:/workpace \
  biom3d:tag_relevant_with_hardware \
  module \
  module_argument \

  # Example
  docker run --rm\
  -v /home/me/dataset1:/workspace \
  biom3d:0.0.30-x86_64-torch2.7.1-cpu \
  preprocess_train \
  --img_dir raw \ # Is a subfolder of /home/me/dataset1
  --img_dir pred  # Will use or create subfolder of /home/me/dataset1
```
Or on windows :
```batch
  docker run --rm ^
  -v folder_with_data_set:/workpace ^
  biom3d:tag_relevant_with_hardware ^
  module ^
  module_argument ^

  :: Example
  docker run --rm^
  -v C:\users\me\dataset1:/workspace ^
  biom3d:0.0.30-x86_64-torch2.7.1-cpu ^
  preprocess_train ^
  --img_dir raw ^ :: Is a subfolder of C:\users\me\dataset1
  --img_dir pred  :: Will use or create subfolder of C:\users\me\dataset1
```
Here is a description. 
1. As said earlier, Biom3d container are one use only so we add `--rm` that allow us to automatically destroy the container at the end of execution. 
2. We want to transmit our dataset to the container so Biom3d can use it, the simpler solution is by mounting a volume with the folder containing the dataset with `-v`. 
3. We select the tag that is the most pertinent for your use case (CUDA drivers, Biom3d version, architecture,...). 
4. We pass the submodule and it's argument (that are described in [CLI Documentation](../tuto/cli.md)).

However there is a twist, if you want to use the `GUI` module, you have another thing to do, and it is giving the container access to you screen.

On Linux with :
```shell
  -e DISPLAY=$DISPLAY\ # Tell to use your screen
  # Transmit permission to container
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v $HOME/.Xauthority:/root/.Xauthority:ro \

  # That give 
  docker run --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v $HOME/.Xauthority:/root/.Xauthority:ro \
  -v /home/me/dataset1:/workspace \
  biom3d:0.0.30-torch2.3.1-cpu gui
```
On Windows : 
*Not documented yet*




