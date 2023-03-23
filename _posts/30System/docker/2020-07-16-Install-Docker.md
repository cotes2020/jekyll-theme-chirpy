---
title: Docker - Install and Use Docker
date: 2020-07-16 11:11:11 -0400
categories: [30System, Docker]
tags: [Docker]
math: true
image:
---


# Install and Use Docker

- [Install and Use Docker](#install-and-use-docker)
- [Install and Use Docker on CentOS 7](#install-and-use-docker-on-centos-7)
  - [Uninstall old versions](#uninstall-old-versions)
  - [Step 1 — Installing Docker](#step-1--installing-docker)
  - [Step 2 — Executing Docker Command Without Sudo (Optional)](#step-2--executing-docker-command-without-sudo-optional)
    - [Step 3 — Using the Docker Command](#step-3--using-the-docker-command)
  - [Step 4 — Working with Docker Images](#step-4--working-with-docker-images)
  - [Step 5 — Running a Docker Container](#step-5--running-a-docker-container)
  - [Step 6 — Committing Changes in a Container to a Docker Image](#step-6--committing-changes-in-a-container-to-a-docker-image)
  - [Step 7 — Listing Docker Containers](#step-7--listing-docker-containers)
  - [Step 8 — Pushing Docker Images to a Docker Repository](#step-8--pushing-docker-images-to-a-docker-repository)


---

# Install and Use Docker on CentOS 7

## Uninstall old versions

```bash
$ sudo yum remove docker \
                  docker-client \
                  docker-client-latest \
                  docker-common \
                  docker-latest \
                  docker-latest-logrotate \
                  docker-logrotate \
                  docker-engine
```

---

## Step 1 — Installing Docker

```py
sudo yum check-update
sudo yum install -y yum-utils

# Install required packages:
sudo yum install -y device-mapper-persistent-data lvm2

# Install the Docker CE packages and containerd.io:
sudo yum yum install -y https://download.docker.com/linux/centos/7/x86_64/stable/Packages/containerd.io-1.2.6-3.3.el7.x86_64.rpm

sudo yum install docker-ce docker-ce-cli containerd.io
--skip-broken

# add the official Docker repository, download the latest version of Docker, and install it:
curl -fsSL https://get.docker.com/ | sh

# Add the Docker CE repo:
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

# Optional: Enable the nightly or test repositories.
$ sudo yum-config-manager --enable docker-ce-nightly
$ sudo yum-config-manager --enable docker-ce-test
$ sudo yum-config-manager --disable docker-ce-nightly
$ sudo yum-config-manager --disable docker-ce-test

# Verify that it’s running:
sudo systemctl start docker
sudo systemctl status docker
```


## Step 2 — Executing Docker Command Without Sudo (Optional)

```py
sudo usermod -aG docker $(whoami)
sudo usermod -aG docker username

# restar
```


### Step 3 — Using the Docker Command


docker [option] [command] [arguments]

```py
# To view all available subcommands,
docker

Output

    attach    Attach to a running container
    build     Build an image from a Dockerfile
    commit    Create a new image from a container's changes
    cp        Copy files/folders between a container and the local filesystem
    create    Create a new container
    diff      Inspect changes on a container's filesystem
    events    Get real time events from the server
    exec      Run a command in a running container
    export    Export a container's filesystem as a tar archive
    history   Show the history of an image
    images    List images
    import    Import the contents from a tarball to create a filesystem image
    info      Display system-wide information
    inspect   Return low-level information on a container or image
    kill      Kill a running container
    load      Load an image from a tar archive or STDIN
    login     Log in to a Docker registry
    logout    Log out from a Docker registry
    logs      Fetch the logs of a container
    network   Manage Docker networks
    pause     Pause all processes within a container
    port      List port mappings or a specific mapping for the CONTAINER
    ps        List containers
    pull      Pull an image or a repository from a registry
    push      Push an image or a repository to a registry
    rename    Rename a container
    restart   Restart a container
    rm        Remove one or more containers
    rmi       Remove one or more images
    run       Run a command in a new container
    save      Save one or more images to a tar archive
    search    Search the Docker Hub for images
    start     Start one or more stopped containers
    stats     Display a live stream of container(s) resource usage statistics
    stop      Stop a running container
    tag       Tag an image into a repository
    top       Display the running processes of a container
    unpause   Unpause all processes within a container
    update    Update configuration of one or more containers
    version   Show the Docker version information
    volume    Manage Docker volumes
    wait      Block until a container stops, then print its exit code


# To view the switches available to a specific command, type:
docker docker-subcommand --help


# To view system-wide information, use:

docker info
```

## Step 4 — Working with Docker Images

```py
# check whether you can access and download images from Docker Hub, type:
docker run hello-world

# search for images available on Docker Hub
# search for the CentOS image
docker search centos
# The script will crawl Docker Hub and return a listing of all images whose name match the search string. In this case, the output will be similar to this:
Output
NAME                            DESCRIPTION                                     STARS     OFFICIAL   AUTOMATED
centos                          The official build of CentOS.                   2224      [OK]
jdeathe/centos-ssh              CentOS-6 6.7 x86_64 / CentOS-7 7.2.1511 x8...   22                   [OK]
jdeathe/centos-ssh-apache-php   CentOS-6 6.7 x86_64 / Apache / PHP / PHP M...   17                   [OK]
million12/centos-supervisor     Base CentOS-7 with supervisord launcher, h...   11                   [OK]
nimmis/java-centos              This is docker images of CentOS 7 with dif...   10                   [OK]
torusware/speedus-centos        Always updated official CentOS docker imag...   8                    [OK]
nickistre/centos-lamp           LAMP on centos setup                            3                    [OK]

# download it to computer using the pull subcommand, like so:
docker pull centos

# After an image has been downloaded, run a container using the downloaded image with run subcommand.
# If an image has not been downloaded when docker is executed with the run subcommand, the Docker client will first download the image, then run a container using it:
docker run centos

# To see the images that have been downloaded to your computer, type:
docker images
# The output should look similar to the following:
[secondary_lable Output]
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
centos              latest              778a53015523        5 weeks ago         196.7 MB
hello-world         latest              94df4f0ce8a4        2 weeks ago         967 B
```


## Step 5 — Running a Docker Container

```py
# run a container using the latest image of CentOS.
# The combination of the -i and -t switches gives interactive shell access into the container:
docker run -it centos

# Your command prompt should change
# reflect that you’re now working inside the container
Output
[root@59839a1b7de2 /]#
# Important: Note the container id in the command prompt.


# Now may run any command inside the container.
# For example, install MariaDB server in the running container.
# No need to prefix any command with sudo, because you’re operating inside the container with root privileges:
yum install mariadb-server
```

## Step 6 — Committing Changes in a Container to a Docker Image
When you start up a Docker image, you can create, modify, and delete files just like you can with a virtual machine.
The changes that you make will only apply to that container.
You can start and stop it, but once you destroy it with the docker rm command, the changes will be lost for good.

to save the state of a container as a new Docker image.

```py
# After installing MariaDB server inside the CentOS container, you now have a container running off an image, but the container is different from the image you used to create it.

# To save the state of the container as a new image
# first exit from it:
exit

# Then commit the changes to a new Docker image instance using the following command.
# -m : the commit message that helps you and others know what changes you made,
# -a : specify the author.
# container-id
# Unless you created additional repositories on Docker Hub, the repository is usually your Docker Hub username:
docker commit -m "What did you do to the image" -a "Author Name" container-id repository/new_image_name
docker commit -m "added mariadb-server" -a "Sunday Ogwu-Chinuwa" cadcb16c158e centos-mariadb

docker images

Output
REPOSITORY             TAG                 IMAGE ID            CREATED             SIZE
finid/centos-mariadb   latest              23390430ec73        6 seconds ago       424.6 MB
centos                 latest              778a53015523        5 weeks ago         196.7 MB
hello-world            latest              94df4f0ce8a4        2 weeks ago         967 B
```


## Step 7 — Listing Docker Containers
After using Docker for a while, you’ll have many active (running) and inactive containers on your computer.

```py
# To view the active ones, use:
docker ps
# You will see output similar to the following:
Output
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
f7c79cc556dd        centos              "/bin/bash"         3 hours ago         Up 3 hours                              silly_spence

# -a switch: view all containers
docker ps -a

# -l switch: view the latest container you created
docker ps -l

# Stopping a running or active container
docker stop container-id
```


## Step 8 — Pushing Docker Images to a Docker Repository
to share it, the whole world on Docker Hub, or other Docker registry that you have access to.

To push an image to Docker Hub or any other Docker registry, you must have an account there

Afterwards, to push your image

```py
# log into Docker Hub.
docker login -u docker-registry-username

# push your own image using:
~ $ docker tag appimage ocho4l/demo-app:latest
~ $ docker push ocho4l/demo-app

docker push docker-registry-username/docker-image-name


# It will take sometime to complete, and when completed, the output will be of this sort:
Output
The push refers to a repository [docker.io/finid/centos-mariadb]
670194edfaf5: Pushed
5f70bf18a086: Mounted from library/centos
6a6c96337be1: Mounted from library/centos

# If a push attempt results in an error of this sort, then you likely did not log in:
Output
The push refers to a repository [docker.io/finid/centos-mariadb]
e3fbbfb44187: Preparing
5f70bf18a086: Preparing
a3b5c80a4eba: Preparing
7f18b442972b: Preparing
3ce512daaf78: Preparing
7aae4540b42d: Waiting
unauthorized: authentication required
```
