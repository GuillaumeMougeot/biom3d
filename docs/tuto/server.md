# Server and client
Biom3d can be used in a server/client mode, allowing usage from low spec computer by delegating computation to another computer.

(remote)=
## Using the client (or remote)
It is a light version of Biom3d specifically made to interact with a server.

To use the client, simply download it from a [release](https://github.com/GuillaumeMougeot/biom3d/releases). It is the file ending with a "Remote". Once you have dowloaded it, start it (you may have a warning that require you to confirm the execution).

Then you just have to enter the IP adress of the server, the username, password and folder to work in. Every of those parameters are to be given by the person that installed the server instance (most likely your IT department). 

Once you've enter those parameter and clicked on `start remotely` and then the same way as the [GUI](gui.md). Here we will describe the differences. In the [GUI](gui.md) documentation, there is a full walktrough.

### Training
In the training tab, the sole difference is the dataset sending. Dataset are stored on the server and you need to send them to the server.
For that you select the folder where are the raw et the one where are the label and name your dataset with a unique identifier. Then click on `Send Dataset` and it will be send to the server.
You can then select that dataset to train in the training configuration (if you just sended it, you should click on `update`).
![Screenshot of training tab on client](../_static/image/gui_remote_preprocess&train.png)

### Prediction
The predictions tab have the same functionnality of sending data. 
You can select existing data folder or create a new one. However you can only select models that are present on the server and not export one (this feature is on our backlog). Then, once the prediction is finished, you can select the folder on the server and download it to the given folder on your computer (or send it to OMERO).
![Screenshot of prediction tab on client](../_static/image/gui_remote_predict.png)

(server)=
## Setting up a server
### File architecture on server
For the moment, server only work on Linux. You need to create a dedicated ²user and define a work folder. Then you must have Biom3d installed (see [here](../installation.md)).
- You can install it on the machine directly, however we advise using a virtual environment.
- If you install it in a virtual environment, it should be a `pyvenv` (it will be activated with `source ./bin/activate`), that environment must be at the root of the work folder.

Biom3d will be called with `python -m biom3d.submodule` so ensure that `python` refer to the version with Biom3d.

Biom3d will store its datasets and trained models in the folder `data` and `logs`, that have to be created in the working folder.

Structure of working folder.
```
├── data
├── logs
└── env (optionnal)
```
A good practice would be to use `/home/user` as working folder, and never a folder at root.

### Connection
You have what Biom3d want on your server, now lets see the connection.
Biom3d connect in `SSH` and transfert file with `SFTP` so your server must have a SSH server (like `open-ssh`) running and listening on port 22. There is also the possibility to use a proxy for connexion.

### Security
As we use SSH, it is important to set some security. 
Biom3d only authentification method is that password of the user where Biom3d is installed must be used when initializing a connection (you can also not use one). 

We **strongly advise** to set up a classical key authentification and give the public key to your end user (it will require a setup step that they may struggle with so document it). 

Another security measure would be to filter the SSH commands. You can find the command used by Biom3d by searching for `REMOTE.exec_command` in the [gui code](../../src/biom3d/gui.py).

### Docker 
Official Biom3d images are made to be one use, however it is possible to use them as a base fo building a server by either :
- Doing the basic setup and creating a SSH entrypoint that will modify and delegate call to Biom3d to the Docker image
- Building a new Docker image that act as a standalone server by either creating a whole new [dockerfile](../../deployment/dockerfiles/template.dockerfile) or using existing image as base image.

### What your user need 
Your user will need to know :
- The IP adress of the server
- The user
- The user's password
- The working folder (absolute path)
- The environment name (if you use one)

And also if you use a proxy, the IP address, user and password.

That's the minimum, if you use key authentification, you will have to transmit the public key and tell them how to make it accessible for the SSH client.
