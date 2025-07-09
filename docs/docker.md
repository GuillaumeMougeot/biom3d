# How to use Biom3d in Docker
Biom3d now has Docker images that you can find [here](). We will explore here some simples use-cases of those images.

## What is Docker ?
Docker is an isolator, which mean that you can create an entire environment and use it anywhere. Docker work with images, that are already built environment. When you do `docker run helloworld` you create an instance of the image `helloworld`, those instances are named containers.

## Installing Docker
Installing Docker is a complex step and we strongly recommend asking your IT service. 

For Windows you can use [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/), since Biom3d use Linux images, we recommend setting up WSL2 Back-end or Hyper-V backend. Docker Desktop is also available on MacOs but Biom3d doesn't have Arm images yet.

For Linux, the installation process depends on the distribution.

## With Docker desktop
*Not documented yet*

## With command lines
### Basic Docker commands
Here we will see some simple Docker command that will be used later. If you are already used to Docker you can skip to [Running a Biom3d container](#running-a-biom3d-container) part.

Docker is separated in several submodules and three of them will be used in this tutorial : `image`, `container` and `build`.

#### Image 
The image submodule is here to manipulate images. An image is defined by two part : the image name (or repository) and the tag. For example let's take `ubuntu:22.04` where `ubuntu` is the image name and `22.04`, is the tag. The tag is here to differentiate the differents versions of the same image. Those images are stored in repository such as [DockerHub](https://hub.docker.com/) or directly on your machine. To download an image on your computer you use `docker pull` such as :
```shell
  docker pull helloworld
  docker pull ubuntu:22.04
  docker pull pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
```
It will download it from distant repository, default being `DockerHub`.
Once you have the image on your machine you can see it by doing this :
```shell
  docker image ls
  REPOSITORY        TAG                             IMAGE ID       CREATED              SIZE
  ubuntu            22.04                           1b668a2d748d   2 weeks ago          77.9MB
  pytorch/pytorch   2.7.1-cuda11.8-cudnn9-runtime   cc0fe24aee5e   4 weeks ago          6.48GB
```
As you can see, there are several information,

If you want to remove an image (To liberate disk space) you can do one of the following:
```shell
  docker image rm ubuntu 
  docker image rm ubuntu:22.04 
  docker image rm 1b668a2d748d
```
With that we covered the basics on image managment. We can now see how to use them.

#### Container
This is the basic Docker submodule, in fact it is the implicite submodule, which mean that those two commands are equivalent :
```shell
  docker run helloworld
  docker container run helloworld
```
Container are instances of an image, you can run one of them with the `docker run` :
```shell
  docker run docker_arguments image_name image_argument

  # With som examples :
  docker run --rm helloworld
  docker run biom3d pred -i foo -o bar
  docker run ubuntu
  docker run ubuntu:22.04
```
Let's go in reading order, first we have the Docker arguments, that describe how the container should run. Here are the most commons :
- `--rm` destroy the container once its job is finished.
- `-n foo` or `--name foo` give a name to the container (here foo).
- `-e FOO=bar` set an environment variable in the container
- `-v absolute_path_on_machine:absolute_path_in_container` attach a volume to the container. For example `-v /home/me:/home/me` or `-v C:\users\me:/home/me` will link your user folder to the user folder in the container, which mean that any modification done in this folder in the container is also done outside and reverse. Another example is `-v $(pwd):/data` will link the folder you are in to the `/data`folder. Be careful to not link a folder you don't want to modify or be sure that the things you do in the container doesn't modify it.

Those are the basics in Docker argument, you can find a complete list [here](https://docs.docker.com/reference/cli/docker/container/run/)
Then the image name, the image is pulled if not on your machine and by default the tag used is `latest` if not specified. Biom3d doesn't have a latest tag, so a precise tag must be given, which means :
```shell
  docker run biom3d # Will not work
  docker run biom3d:0.0.30-x86_64-torch2.7.1-cpu # Will work
```
Then are the image arguments, they depends on the image entrypoint that is by default not defined. This entrypoint describe what script is launch when you do a run on an image. Biom3d containers have Biom3d as entry point which mean they want to know the module you want to use.
```shell
  docker run biom3d:0.0.30-x86_64-torch2.7.1-cpu pred -i foo -o bar # Will run prediction on the foo folder and send it to bar
  docker run biom3d:0.0.30-x86_64-torch2.7.1-cpu # Will crash because the entry point want you to say which submodule you want to use and then its parameters
```
Now your container is running but how to monitor it ? Well you can do this :
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
It will show you the name, id and status (running, stopped,...) of every container. Biom3d container stop once they finish their job (Status Exited) so they must be removed after by doing :
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
Now you see the basics of running container. If you just want to use Biom3d you can directly go to [running a Biom3d container](#running-a-biom3d-container), if you want to contribute on developement you may be interested on [building](#building)

#### Building
If you want to contribute to Biom3d or make your custom images to better answer your problematics, you will need to build an image. To do that you use a `DockerFile`, you can see some examples in `/deployment/dockerfiles/examples`. A DockerFile follow the following structure :
```Dockerfile
  FROM baseimage # It is always the first line
  ENV FOO=Bar
  COPY relative_path_from_dockerfile absolute_path_in_image
  ADD source destination
  WORKDIR folder # Set the folder in which you locate the rest of the dockerfile or execution
  RUN command
  ENTRYPOINT my_entrypoint_script
```
The DockerFile always start with `FROM` but after, you're free to do whatever you want. Let's add some details statements.
- `ENV` set an environment variable in the container
- `COPY` copy the element at givent location on your drive to the given location in the image. 
- `ADD` work the same way as copy in a sense it copy the source to the destination in the image. But it can also take an URL as a source and it automically exract archives (`.tar`,`tar.gz`,...)
- `WORKDIR path` allow you to go to the given path, it is equivalent to a `cd path` command. For example in the Biom3d DockerFiles, there is always at the end `WORKDIR /workspace`, it means that when you use the container it will save files in `/workspace` and that's why we always want to attach a volume to `/workspace`
- `RUN command` will run the command, following the base image syntaxe (shell for MacOs/Linux, batch or powershell on Windows), it is quite simple but there are some good pratices. One of those are to group `RUN` statement :
```Dockerfile
  # Don't do
  RUN mdir /foo
  RUN pip install biom3d
  # But
  RUN mkdir /foo \
  && pip install torch
  # Or on Windows
  RUN mkdir /foo ^
  && pip install torch
``` 
This will make your image smaller. But while you're creating a new image, keep separated `RUN` statement, it will be easier to debug.

You can also add arguments with the `ARG FOO=default_value`, and access the value with `${FOO}` (or `%FOO%` if you're using it in a Windows command).

Then you build the image with 
```shell
  docker build -f path_to_dockerfile .
  docker build . # Only build the file named Dockerfile
  docker build --build_arg FOO=not_bar -f path . # Will change the value of the given ARG
  docker build -f path ./subforlder # The ADD or COPY will be relative to path/subfolder instead of path 
```
With that you should be able to build your image if you need it, the hard part of Biom3d build is the managment of python dependencies (there are examples of DockerFiles on the [git repository](https://github.com/GuillaumeMougeot/biom3d)).

### Running a Biom3d container 
Now you should know some basics on how to use Docker, so here we will see how to use Biom3d container themselves. As explained earlier, Biom3d has a specific entrypoint that make containers not (easily) reusable, they are throwable container.

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
Here is a description. First as said earlier, Biom3d container are one use only so we add `--rm` that allow us to automatically destroy the container at the end of execution. Then we want to transmit our dataset to the container so Biom3d can use it, the simpler solution is by mounting a volume with the folder containing the dataset with `-v`. Then we select the tag that is the most pertinent for your use case (CUDA drivers, Biom3d version, architecture,...). Lastly we pass the submodule and it's argument (that are described in [CLI Documentation](tuto_cli.md)).

However there is a twist, if you want to use the `GUI` module, you have another thing to do, and it is giving the container access to you screen with :
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
Or on Windows : 
*Not documented yet*




