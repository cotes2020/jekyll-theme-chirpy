<!-- ---
title: Virtulization - Docker Note
date: 2020-11-26 11:11:11 -0400
categories: [00Basic, VMandContainer, Containers]
tags: [Linux, VMs, Docker]
math: true
image:
--- -->

[toc]

---

# Docker Note

---

## basic

- developed with GO launched by Google,

- based on `cgroup` and `namespace` of Linux Kernel and Union FS like AUFS

- a software platform that <font color=red> packages software (such as applications) into containers </font>
  - to <font color=blue> package and isolate the processes </font> which belong to <font color=blue> Operating system level virtualization technology </font>

  - Docker is further packaged on a container basis,
    - from ile system, network interconnection to process isolation, etc,
    - isolated processes are independent of the host and other isolated processes.
  - <font color=blue> greatly simplifying container creation and maintenance </font>.

- <font color=red> Docker is installed on each server that will host containers </font>,
  - provides simple commands to build, start, or stop containers.
  - quickly deploy and scale applications into any environment.


---

### architecture in linux

The initial implementation is based on LXC.
- It removed LXC and use libcontainer instead which is developed by themself since 0.7. Starting with 1.11, it uses runC and containerd further.

![docker-on-linux](https://i.imgur.com/U1jdplX.png)

> `runc` is a Linux command-line tool for creating and running containers according to the OCI container runtime specification.

> `containerd` is a daemon that manages container life cycle from downloading and unpacking the container image to container execution and supervision.

---

## Why use Docker

best used as a solution to:
- <font color=red> Standardize environments </font>
- <font color=red> Reduce conflicts between language stacks and versions </font>
- <font color=red> Use containers as a service </font>
- <font color=red> Run microservices using standardized code deployments </font>
- <font color=red> Require portability for data processing </font>

platform for developers and sysadmins to <font color=red> build, run, and share applications with containers </font>
- The use of containers to deploy applications is called `containerization`.

containerization
- **Flexible**: Even the most complex applications can be containerized.
- **Lightweight**: Containers leverage and share the host kernel, making them much more efficient in terms of system resources than virtual machines.
- **Portable**: build locally, deploy to the cloud, and run anywhere.
- **Loosely coupled**: Containers are highly self sufficient and encapsulated, can replace or upgrade one without disrupting others.
- **Scalable**: You can increase and automatically distribute container replicas across a datacenter.
- **Secure**: Containers apply aggressive constraints and isolations to processes without any configuration required on the part of the user.

---


### Images and containers

![containers](https://i.imgur.com/ubkaVhF.png)

- Fundamentally, a **container** is nothing but a running process, with some added encapsulation features applied to it in order to keep it isolated from the host and from other containers.

- container isolation: <font color=red> each container interacts with its own private filesystem </font>

- this filesystem is provided by a **Docker image**.
  - An image includes everything needed to run an application
  - the code or binary, runtimes, dependencies, and any other filesystem objects required.


---

## Containers/Docker vs Traditional Virtualization.

differences between Docker and Traditional Virtualization.
- The **traditional Virtual Machine technology** :
  - <font color=red> virtualize a set of hardwares to </font>
    - run a complete operation system
    - and run the required application process on the system.
    - <font color=red> runs a full-blown “guest” operating system </font> with virtual access to host resources through a hypervisor.
    - VMs incur a lot of overhead beyond what is being consumed by your application logic.


- **Container/Docker**
  - The application process in the container <font color=red> runs directly on the host kernel </font>
  - the container does not have its own Kernel and hardware virtualiztion.
  - Therefore much lighter than traditional virtual machines.


![virtualization](https://i.imgur.com/XdauvJG.png)

![docker](https://i.imgur.com/ULoJiCf.png)


### 优势

1. 更高效的利用系统资源
   - a method of operating system virtualization
     - 容器不需要进行硬件虚拟以及运行完整操作系统等额外开销，
     - Docker 对系统资源的利用率更高。
     - runs natively on Linux and <font color=red> shares the kernel of the host machine with other containers </font>
     - runs a discrete process, taking no more memory than any other executable, making it lightweight.
   - <font color=red> run an application and its dependencies </font> in <font color=red> resource-isolated processes </font>
     - 无论是 应用执行速度、内存损耗或者文件存储速度，都要比传统虚拟机技术更高效。
     - 因此，一个相同配置的主机，往往可以运行更多数量的 Docker 应用。

2. 更快速的启动时间 Faster Startup Time
   - 传统的虚拟机技术启动应用服务往往需要数分钟，
   - Docker 容器应用，由于直接运行于宿主内核，无需启动完整的操作系统，因此可以做到秒级、甚至毫秒级的启动时间。
   - 节约了开发、测试、部署的时间。

3. 一致的运行环境 <font color=red> Consistent Operating Environment </font>
   - because everything are packaged into a single object.
   - 开发过程中一个常见的问题是环境一致性问题。开发环境、测试环境、生产环境不一致，导致bug未在开发过程中被发现。
   - Docker 的镜像提供了除内核外完整的运行时环境，确保了应用运行环境一致性
     - 不会再出现 「这段代码在我机器上没问题啊」 这类问题。
   - Containers hold everything that the software needs to run,
     - such as libraries, system tools, application’s code, configurations, dependencies, and the runtime.
   - ensure <font color=blue> quick, reliable, and consistent deployments </font>. regardless of deployment environment.

4. 持续交付和部署 CI/CD
   - 对开发和运维(DevOps)人员来说，最希望的就是一次创建或配置，可以在任意地方正常运行。
   - create and configure once to run anywhere.
   - 使用 Docker 可以通过定制`应用镜像 application mirrors with Docker`来实现持续集成、持续交付、部署。
   - **Developers**: **build images with Dockerfile** and use `Continuous Integration for integration testing`.
   - **Operation teams**: **deploy production environments quickly with the images**, and even **make automatic deployments** possible by using `Continuous Delivery/Deployment techniques`.
   - 而且使用 Dockerfile 使镜像构建透明化，不仅仅开发团队可以理解应用运行环境，也方便运维团队理解应 用运行所需条件，帮助更好的生产环境中部署该镜像。

5. 更轻松的迁移 Easier migration
   - 由于 Docker 确保了执行环境的一致性，使得应用的迁移更加容易。
   - Docker 可以在很多平台上运行，无 论是物理机、虚拟机、公有云、私有云，甚至是笔记本，其运行结果是一致的。
   - 因此用户可以很轻易的将在一个 平台上运行的应用，迁移到另一个平台上，而不用担心运行环境的变化导致应用无法正常运行的情况。

6. 更轻松的维护和扩展
   - Docker 使用的分层存储以及镜像的技术，使得应用重复部分的复用更为容易，也使得应用的维护更新更加简单，基于基础镜像进一步扩展镜像也变得非常简单。
   - Docker 团队同各个开源项目团队一起维护了高质量的 官方镜像，既可以直接在生产环境使用，又可以作为基础进一步定制，大大降低了应用服务的镜像 制作成本。


![Screen Shot 2020-11-28 at 13.18.53](https://i.imgur.com/eZtGmjR.png)

![Screen Shot 2020-11-25 at 01.57.58](https://i.imgur.com/NOjFPJV.png)

![Screen Shot 2020-05-06 at 20.38.20](https://i.imgur.com/hC6oigX.png)

![Screen Shot 2020-07-27 at 09.38.41](https://i.imgur.com/4LDPQK7.png)


---

## 使用

基本概念 Docker 包括三个基本概念
- 镜像( Image )
- 容器( Container )
- 仓库( Repository )


---


### `Docker Image` 镜像

操作系统分为 <font color=blue> 内核 </font> 和 <font color=blue> 用户空间 </font>
- 对于 Linux 而言，内核启动后，会挂载 root 文件系统为其提 供用户空间支持。
  - Docker Image，就相当于是一个 root 文件系统。
  - 比如官方镜像 ubuntu:18.04 就包含了完整的一套 Ubuntu 18.04 最小系统的 root 文件系统。
- Docker Image
  - 是一个特殊的文件系统
  - 除了提供容器运行时所需的程序、库、资源、配置等文件外
  - 还包含了一 些为运行时准备的一些配置参数(如匿名卷、环境变量、用户等)。
- 镜像不包含任何动态数据，其内容在构建之 后也不会被改变。


#### 分层存储 Advanced Multi-layered Unification Filesystem (AUFS)
因为镜像包含操作系统完整的 root 文件系统，其体积往往是庞大的
- 因此在 Docker 设计时，就充分利用 `Union FS` 的技术，将其设计为分层存储的架构。
- 所以严格来说，镜像并非是像一个 ISO 那样的打包文件，镜像只是一个虚拟的概念，其实际并非由一个文件组成，而是由一组文件系统组成，或者说，由多层文件系统联 合组成。
- 镜像构建时，会一层层构建，前一层是后一层的基础。每一层构建完就不会再发生改变，后一层上的任何改变只 发生在自己这一层。
  - 比如，
  - 删除前一层文件的操作，
  - 实际不是真的删除前一层的文件，
  - 而是仅在当前层标记为该 文件已删除。
  - 在最终容器运行的时候，虽然不会看到这个文件，但是实际上该文件会一直跟随镜像。
- 因此，在构 建镜像的时候，需要额外小心，每一层尽量只包含该层需要添加的东西，任何额外的东西应该在该层构建结束前 清理掉。
- 分层存储的特征还使得镜像的复用、定制变的更为容易。
- 甚至可以用之前构建好的镜像作为基础层，然后进一步 添加新的层，以定制自己所需的内容，构建新的镜像。


---

### `Docker Container` 容器
- 镜像( Image )和容器( Container )的关系，就像是面向对象程序设计中的 类 和 实例 一样
- 镜像是静态的定义，
- 容器是镜像运行时的实体。
  - 容器可以被创建、启动、停止、删除、暂停等。
  - 容器的实质是进程，但与直接在宿主执行的进程不同，容器进程运行于属于自己的独立的 命名空间。
- 因此容器可以拥有自己的 root 文件系统、自己的网络配置、自己的进程空间，甚至自己的用户 ID 空间。
- 容器内的进程 是运行在一个隔离的环境里，使用起来，就好像是在一个独立于宿主的系统下操作一样。
- 这种特性使得容器封装 的应用比直接在宿主运行更加安全。
  - 因为这种隔离的特性，很多人初学 Docker 时常常会混淆容器和虚拟机。
- `multi-layered filesystem` is applied to **images**, and so as the **containers**.
  - When a container is running, it is based on its image, with a writable layer created on top of it.
  - We call this layer prepared for R/W at runtime `Container Layer`
  - `容器存储层`的生存周期和容器一样
    - 容器消亡时，容器存储层也随之消亡。
    - 因此任何保存于容器存储层的信息 都会随容器删除而丢失。

---

#### `Docker Container` 的读写

> recommended by the Docker Development Best Practices

**should not write any data to the `container layer`**
- make it stateless.
- All file write operations should adhere to `Volume` or `bind mounts`.
- `Writing to volume or bind mounts` skips the container layer and `R/W to host storage(or network storage) directly`
  - achieves better performance and stability.
- 数据卷的生存周期独立于容器
  - 容器消亡，数据卷不会消亡。
  - 因此使用数据卷后，容器删除或者重新运行之 后，数据却不会丢失。

> 容器 = 镜像 + 读写层。并且容器的定义并没有提及是否要运行容器。
> running container: 一个可读写的 统一文件系统 加上隔离的进程空间 和包含其中的进程

![dacd660134e97cca60e04ba5fdcfa79e](https://i.imgur.com/7SQmtvm.png)

![3b9079593f15903497a37c64a23b9c39](https://i.imgur.com/YqqmTqZ.png)


正是文件系统隔离技术使得Docker成为了一个前途无量的技术。
- 一个容器中的进程可能会对文件进行修改、删除、创建，
- 这些改变都将作用于可读写层（read-write layer）。

![1ccc2aa9e11e25e5ed2efd18cf1052c4](https://i.imgur.com/IG7bHa3.png)

![a20b70e3e4ca61faa2c3436e1bb2d93a](https://i.imgur.com/J9MUw7z.png)


```bash
# 可以通过运行以下命令来验证:
docker run ubuntu touch happiness.txt

# 即便是这个ubuntu容器不再运行，我们依旧能够在主机的文件系统上找到这个新文件。
find / -name happiness.txt
/var/lib/docker/aufs/diff/860a7b...889/happiness.txt
```


#### Image Layer Definition

为了将零星的数据整合起来，提出了镜像层（image layer）这个概念。
- 一个层并不仅仅包含文件系统的改变，它还能包含了其他重要信息。
- 元数据（metadata）就是关于这个层的额外信息，它不仅能够让Docker获取运行和构建时的信息，还包括父层的层次信息。
- 需要注意，只读层和读写层都包含元数据。

![f3218a8fcdd1fd8bcd5a719ca3c64f59](https://i.imgur.com/EDcKDa7.png)

![39fb0f8e630b338bcca7a29da3acabb7](https://i.imgur.com/wxnZ8cb.png)

![53e377deeeb5f30bf939ed0836f851c9](https://i.imgur.com/QCmf8xH.png)


### docker commands

cmd | pic
---|---
`docker pull ubuntu:18.04` | download
`docker run <image-id>` | ![01aedf55bd21abbe607b3864d76f0ec0](https://i.imgur.com/XKic8PI.jpg)
`docker create <image-id>` | ![062e7af0929dd205b2ac6efdd937d6f4](https://i.imgur.com/J82Sli2.jpg)
`docker start <container-id>` | ![1c38d4735e9760bdca025ff50a1b5386](https://i.imgur.com/AGAzgEq.jpg) <br> ![275cc486d4ce7ecbdecf0ecc1de0a34b](https://i.imgur.com/wq40mMI.jpg)
`docker ps` | ![f05a2a8dfc6641d8237306ff575aa283](https://i.imgur.com/4U1HNNz.jpg)
`docker ps –a` | ![d7f7b0ada7fc1c641c90745a959f9c05](https://i.imgur.com/7uFxaWP.jpg)
`docker images` | ![d5b0b3e2e7acdcf35e7577d7670a46f7](https://i.imgur.com/Ede9USJ.jpg)
`docker images -a` |![ce083575e95a0e46b105c3596c12ca71](https://i.imgur.com/86wPwgT.jpg)
`docker stop <container-id>` |![a41de8fe542efe25e3620691ad9238df](https://i.imgur.com/H0GEqJf.jpg)
`docker kill <container-id>` |![ac1cbc31d4f191e26e05fb1a11f04d26](https://i.imgur.com/F9zAAgG.jpg)
`docker pause <container-id>` |![b701a3c51e7d1915da3bea0bc43efcd1](https://i.imgur.com/12MQZ0E.jpg)
`docker rm <container-id>` | ![8fa2d6e19c29f18548624efd64eb6dfa](https://i.imgur.com/VvbXmHc.jpg) <br> docker rm命令会移除构成容器的可读写层。注意，这个命令只能对非运行态容器执行。
`docker rmi <image-id>`| ![32a7f413a5f8ff936dd0f4a31d25fdcc](https://i.imgur.com/5DgiDu6.jpg) <br> docker rmi 命令会移除构成镜像的一个只读层。<br> 只能够使用docker rmi来移除最顶层（top level layer）（也可以说是镜像）<br> 也可以使用-f参数来强制删除中间的只读层。
`docker commit <container-id> [REPOSITORY[:TAG]]` | ![3c02ccf4e7a2a353af065d93b26ae89e](https://i.imgur.com/NsvFAtt.jpg) <br> ![28059b3a499faba896263c0ff077fe3a](https://i.imgur.com/DYct48G.jpg) <br> 将容器的可读写层转换为一个只读层，把一个容器转换成了不可变的镜像。
`docker build -t makali:v1` | ![b22cd304f28c715ae3ae6812476b222d](https://i.imgur.com/zOlePPX.jpg) <br>![100712263ecf4544dd11602adc39ee3e](https://i.imgur.com/fBRTJCc.png) <br> build命令根据Dockerfile文件中的FROM指令获取到镜像，然后重复地 <br> 1）run（create和start）<br> 2）修改 <br> 3）commit。<br> 在循环中的每一步都会生成一个新的层，因此许多新的层会被创建。
`docker exec <running-container-id>` | ![db64b3b38aff136d42e9ffeb81675bda](https://i.imgur.com/ac6qMxt.jpg)
`docker inspect <container-id> or <image-id>` | ![184f9d55770ca036cb5b1e6d96ce4a12](https://i.imgur.com/5AWZAer.jpg) <br> docker inspect命令会提取出容器或者镜像最顶层的元数据。
`docker save <image-id>` | ![70cdbaf975c88bc83423d88be85476b5](https://i.imgur.com/MNiSruI.jpg) <br> 创建一个镜像的压缩文件，这个文件能够在另外一个主机的Docker上使用。和export命令不同，这个命令为每一个层都保存了它们的元数据。这个命令只能对镜像生效。
`docker export <container-id>` |![1714c3dd524c807bf9c9b4d0fbe4d056](https://i.imgur.com/lsayIKA.jpg) <br> 创建一个tar文件，并且移除了元数据和不必要的层，将多个层整合成了一个层，只保存了当前统一视角看到的内容 <br> expoxt后的容器再import到Docker中，通过docker images –tree命令只能看到一个镜像；<br> save后的镜像能够看到这个镜像的历史镜像
`docker history <image-id>` | ![b513dd5467f23fdd23523f60242d5dcb](https://i.imgur.com/Tc2gUar.jpg) <br> docker history命令递归地输出指定镜像的历史镜像。




---

### `Docker Registry` 仓库

After the construction of an image, we can easily run it on a host.
- but, to use the image on other servers, need a centralized image storage and distribution service, **Docker Registry**.
- 一个 Docker Registry 中可以包含多个 `仓库( Repository )`;
- 每个仓库可以包含多个 `标签( Tag )`;
- 每个标签对应一个`镜像`。
  - 通常，一个仓库会包含同一个软件不同版本的镜像，而标签就常用于对应该软件的各个版本。
  - 通过 `<仓库名>:<标签>` 的格式来指定具体是这个软件哪个版本的镜像。
  - 如果不给出标签，将以 `latest` 作为 默认标签。


以 Ubuntu 镜像 为例，
- ubuntu 是仓库的名字，其内包含有不同的版本标签，如， 16.04 , 18.04 。
- 我们可以通过 ubuntu:16.04 ，或者 ubuntu:18.04 来具体指定所需哪个版本的镜像。
- 如果忽略了标 签，比如 ubuntu ，那将视为 ubuntu:latest 。


仓库名经常以 两段式路径 形式出现，
- 比如 jwilder/nginx-proxy ，前者往往意味着 Docker Registry 多 用户环境下的用户名，后者则往往是对应的软件名。但这并非绝对，取决于所使用的具体 Docker Registry 的软 件或服务。

#### Docker Registry 公开服务
- 开放给用户使用、允许用户管理镜像的 Registry 服务。
- 一般这类公开服务允许用户 免费上传、下载公开的镜像，并可能提供收费服务供用户管理私有镜像。
- 最常使用的 Registry 公开服务是官方的 Docker Hub，这也是默认的 Registry，并拥有大量的高质量的官方镜 像。
- 除此以外，还有 Red Hat 的 Quay.io;Google 的 Google Container Registry，Kubernetes 的镜像使用 的就是这个服务。


#### 私有 Docker Registry
- 除了使用公开服务外，用户还可以在本地搭建私有 Docker Registry。
- Docker 官方提供了 Docker Registry 镜 像，可以直接使用做为私有 Registry 服务。


---


## 安装 Docker

Docker 分为 stable test 和 nightly 三个更新频道。

---



### macOS 安装

```bash
Docker 系统要求
# Docker Desktop for Mac 要求系统最低为 macOS Catalina 10.13。

安装
# 使用 Homebrew 安装
$ brew cask install docker

# 手动下载安装
# 如果需要手动下载，请点击以下链接下载 Stable 或 Edge 版本的 Docker Desktop for Mac。
# 如同 macOS 其它软件一样，安装也非常简单，双击下载的 .dmg 文件，然后将那只叫 Moby 的鲸鱼图标拖 拽到 Application 文件夹即可(其间需要输入用户密码)。


# 启动终端后，通过命令可以检查安装后的 Docker 版本。
$ docker --version
Docker version 19.03.8, build afacb8b

$ docker-compose --version
docker-compose version 1.25.5, build 8a1c60f6

# The docker setup does not work as in a normal Linux machine, on a Mac it is much more complicated. But it can be done!

brew cask install docker virtualbox
brew install docker-machine
docker-machine create --driver virtualbox default
docker-machine restart
eval "$(docker-machine env default)" # This might throw an TSI connection error. In that case run docker-machine regenerate-certs default
(docker-machine restart) # maybe needed
docker run hello-world
```

test:

```bash
# 1. 运行一个 Nginx 服务器:
$ docker run -d -p 80:80 --name webserver nginx
# 服务运行后，可以访问 http://localhost，如果看到了 "Welcome to nginx!"，就说明 Docker Desktop for Mac 安装成功了。
# 要停止 Nginx 服务器并删除执行下面的命令:
$ docker stop webserver
$ docker rm webserver



# 2. running the hello-world Docker image:

$ docker run hello-world

    Unable to find image 'hello-world:latest' locally
    latest: Pulling from library/hello-world
    ca4f61b1923c: Pull complete
    Digest: sha256:ca0eeb6fb05351dfc8759c20733c91def84cb8007aa89a5bf606bc8b315b9fc7
    Status: Downloaded newer image for hello-world:latest

    Hello from Docker!
    This message shows that your installation appears to be working correctly.
    ...

# Run docker image ls to list the hello-world image that you downloaded to your machine.
$ docker image ls

# List the hello-world container (spawned by the image) which exits after displaying its message. If it is still running, you do not need the --all option:
    $ docker ps --all

    CONTAINER ID     IMAGE           COMMAND      CREATED            STATUS
    54f4984ed6a8     hello-world     "/hello"     20 seconds ago     Exited (0) 19 seconds ago

```


---

## `docker image` command

- 镜像是 Docker 的三大组件之一。
- Docker 运行容器前需要本地存在对应的镜像
- 如果本地不存在该镜像，Docker 会从镜像仓库下载该镜像。

---

### `docker pull ubuntu:18.04`获取镜像

`docker pull`
- Docker Hub 上有大量的高质量的镜像可以用，
- 从 Docker Image 仓库获取镜像

```bash
docker pull [选项] [Docker Registry 地址[:端口号]/]仓库名[:标签]
# - Docker Image 仓库地址: 一般是 <域名/IP>[:端口号] 。默认地址是 Docker Hub(docker.io)。
# - 仓库名: 两段式名称，即 <用户名>/<软件名> 。
# - 对于 Docker Hub，如果 不给出用户名，则默认为 library 官方镜像。

# 比如:

$ docker pull kalilinux/kali-rolling

$ docker pull ubuntu:18.04
# 18.04: Pulling from library/ubuntu
# bf5d46315322: Pull complete
# 9f13e0ac480c: Pull complete
# e8988b5b3097: Pull complete
# 40af181810e7: Pull complete
# e6f7c7e5c03e: Pull complete
# Digest: sha256:147913621d9cdea08853f6ba9116c2e27a3ceffecf3b492983ae97c3d643fbbe Status: Downloaded newer image for ubuntu:18.04

# 上面的命令中没有给出 Docker Image 仓库地址，因此将会从 Docker Hub 获取镜像。
# 镜像名称是 ubuntu:18.04 ，因此将会获取官方镜像 library/ubuntu 仓库中标签为 18.04 的镜像。

# 从下载过程中可以分层存储，镜像是由多层存储所构成。
# 下载也是一层层的去下载， 并非单一文件。
# 下载过程中给出了每一层的 ID 的前 12 位。
# 并且下载结束后，给出该镜像完整的 sha256 的 摘要，以确保下载一致性。
```

---

### `docker run -it --rm ubuntu:18.04 bash` 运行

- 以镜像为基础启动并运行一个容器。
  - -i :交互式操作
  - -t : 进入 bash 执行一些命令并查看返回结果，需要交互式终端。
  - --rm: 容器退出后随之将其删除。
    - 默认情况下，为了排障需求，退出的容器并不会立即删除，除非手动 docker rm 。
    - 不需要排障和保留结果，因此使用 --rm 可以避免浪费空间。

```bash
# 以上面的 ubuntu:18.04 为例，启动里面的 bash 并且进行交互式操作:
$ docker run -it --rm ubuntu:18.04 bash

root@e7009c6ce357:/# cat /etc/os-release NAME="Ubuntu"
VERSION="18.04.1 LTS (Bionic Beaver)" ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 18.04.1 LTS"
VERSION_ID="18.04"
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/" BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/" PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-pol icy"
VERSION_CODENAME=bionic
UBUNTU_CODENAME=bionic

# docker run: 运行容器的命令
# -i :交互式操作
# -t 终端。进入 bash 执 行一些命令并查看返回结果，因此我们需要交互式终端。
# --rm: 容器退出后随之将其删除。默认情况下，为了排障需求，退出的容器并不会立即删除，除非手动 docker rm 。不需要排障和保留结果，因此使用 --rm 可以避免浪费空间。
# ubuntu:18.04 :这是指用 ubuntu:18.04 镜像为基础来启动容器。
# bash: 放在镜像名后的是 命令，这里我们希望有个交互式 Shell，因此用的是 bash 。
# 进入容器后，我们可以在 Shell 下操作，执行任何所需的命令。
# 通过 exit 退出了这个容器。
```

---


### `docker image ls` 列出镜像

```bash
# 列出已经下载下来的镜像
$ docker image ls
# 仓库名              标签、               镜像 ID、           创建时间              所占用的空间
REPOSITORY           TAG                 IMAGE ID            CREATED             SIZE
redis                latest              5f515359c7f8        5 days ago          183 MB
nginx                latest              05a60462f8ba        5 days ago          181 MB
mongo                3.2                 fe9198c04d62        5 days ago          342 MB
<none>               <none>              00285df0df87        5 days ago          342 MB  # 虚悬镜像 dangling image
ubuntu               18.04               f753707788c5        4 weeks ago         127 MB
ubuntu               latest              f753707788c5        4 weeks ago         127 MB

# 镜像 ID 则是镜像的唯一标识
# 一个镜像可以对应多个 标签。
```


### `docker system df` 镜像体积

```bash
# 镜像体积所占用空间和在 Docker Hub 上看到的镜像大小不同。
# - ubuntu:18.04 镜像大小，在这里是 127 MB ，
# - 但是在 Docker Hub 显示的却是 50 MB 。
# - 这是因为 Docker Hub 中显示的体积是压缩后的体积。
# - 在镜像下载和上传过程中镜像是保持着压缩状态的，因此 Docker Hub 所显示的大小是网络传输中更关心的流量大小。
# - 而 docker image ls 显示的是镜像下载到本地后，展开的大小(展开后的各层所占空间的总和)，因为镜像到本地后，查看空间的时候，更关心的是本地磁 盘空间占用的大小。

# docker image ls 列表中的镜像体积总和并非是所有镜像实际硬盘消耗。
# - 由 于 Docker Image 是多层存储结构，并且可以继承、复用，因此不同镜像可能会因为使用相同的基础镜像，从而拥 有共同的层。由于 Docker 使用 Union FS，相同的层只需要保存一份即可，因此实际镜像硬盘占用空间很可能要 比这个列表镜像大小的总和要小的多。

# 便捷的查看镜像、容器、数据卷所占用的空间。
$ docker system df
TYPE                TOTAL               ACTIVE              SIZE                RECLAIMABLE
Images              24                  0                   1.992GB             1.992GB (100%)
Containers          1                   0                   62.82MB             62.82MB (100%)
Local Volumes       9                   0                   652.2MB             652.2MB (100%)
Build Cache                                                 0B                  0B
```


### `docker image ls -f dangling=true` 虚悬镜像

虚悬镜像(dangling image)
- 由于新旧镜像同名，旧镜像名称被取消，从而出现仓库名、标签均为 <none> 的镜像。
- 无标签镜像

```bash
# 既没有仓库名，也没有标签，均为 <none>
# 仓库名              标签、               镜像 ID、           创建时间              所占用的空间
REPOSITORY           TAG                 IMAGE ID            CREATED             SIZE
redis                latest              5f515359c7f8        5 days ago          183 MB
<none>               <none>              00285df0df87        5 days ago          342 MB  # 虚悬镜像 dangling image

# docker pull
# 镜像 mongo:3.2, 随着新版本后，重新 docker pull mongo:3.2 时， mongo:3.2 这个镜像名被转移到了新下载的镜像身上
# 旧的镜像上的这个名称则被取消，从而成为了 <none> 。
# docker build


# 专门显示 这类镜像:
$ docker image ls -f dangling=true
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
<none>              <none>              00285df0df87        5 days ago          342 MB

# 一般来说，虚悬镜像已经失去了存在的价值，是可以随意删除的:
$ docker image prune
```


#### `docker image ls -a` 中间层镜像

- 为了加速镜像构建、重复利用资源，Docker 会利用 中间层镜像。
- 在使用一段时间后，可能会看到一些依赖 的中间层镜像。
- 默认的 docker image ls 只会显示顶层镜像

```bash
# 显示包括中间层镜像在内的列出镜像所有镜像的话，加 -a 参数。
$ docker image ls -a
# 这样会看到很多无标签的镜像
# 与之前的虚悬镜像不同，这些无标签的镜像很多都是中间层镜像，是其它镜像所依赖的镜像。

# 这些无标签镜像不应该删除，否则会导致上层镜像因为依赖丢失而出错。
# 也没必要删除，因为相同的层只会存一遍，而这些镜像是别的镜像的依赖，因此并不会因为它们被列出来而多存了一份，无论如何你也会需要它们。
# 只要删除那些依赖它们的镜像后，这些依赖的中间层镜像也会被连带删除。
```


### `docker image ls xxx ` 列出部分镜像


```bash
# 不加任何参数的情况下， docker image ls 会列出所有顶层镜像，
# 只希望列出部分镜像。 docker image ls 有好几个参数可以帮助做到这个事情。

# 根据仓库名列出镜像
$ docker image ls ubuntu
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
ubuntu              18.04               f753707788c5        4 weeks ago         127 MB
ubuntu              latest              f753707788c5        4 weeks ago         127 MB


# 列出特定的某个镜像，也就是说指定仓库名和标签
$ docker image ls ubuntu:18.04
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
ubuntu              18.04               f753707788c5        4 weeks ago         127 MB


# 除此以外， docker image ls 还支持强大的过滤器参数 --filter ，或者简写 -f 。
# 使用过滤器来列出虚悬镜像
$ docker image ls -f dangling=true
# 希望看到在 mongo:3.2 之后建 立的镜像，可以用下面的命令:
$ docker image ls -f since=mongo:3.2
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
redis               latest              5f515359c7f8        5 days ago          183 MB
nginx               latest              05a60462f8ba        5 days ago          181 MB
# 想查看某个位置之前的镜像也可以，只需要把 since 换成 before 即可。
$ docker image ls -f before=mongo:3.2
# 此外，如果镜像构建时，定义了 LABEL ，还可以通过 LABEL 来过滤。
$ docker image ls -f label=com.example.version=0.1
```

#### 以特定格式显示


```bash
# 默认情况下， docker image ls 会输出一个完整的表格，但是我们并非所有时候都会需要这些内容。
# 比 如，刚才删除虚悬镜像的时候，需要利用 docker image ls 把虚悬镜像的 ID 列来，然后交给 docker image rm 命令作为参数来删除指定的这些镜像，这个时候就用到了 -q 参数。
$ docker image ls -q
5f515359c7f8
05a60462f8ba
fe9198c04d62
00285df0df87
f753707788c5
f753707788c5
1e0c3dd64ccd

# --filter 配合 -q 产生出指定范围的 ID 列表，然后送给另一个 docker 命令作为参数，从而针对这 组实体成批的进行某种操作的做法在 Docker 命令行使用过程中非常常见，不仅仅是镜像，将来我们会在各个命 令中看到这类搭配以完成很强大的功能。


# 只是对表格的结构不满意，希望自己组织列; 或者不希望有标题, 这就用到了 Go 的模板语法。

# 直接列出镜像结果，并且只包含镜像ID和仓库名:
$ docker image ls --format "{{.ID}}: {{.Repository}}"
5f515359c7f8: redis
05a60462f8ba: nginx
fe9198c04d62: mongo
00285df0df87: <none>
f753707788c5: ubuntu
f753707788c5: ubuntu
1e0c3dd64ccd: ubuntu


# 或者打算以表格等距显示，并且有标题行，和默认一样，不过自己定义列:
$ docker image ls --format "table {{.ID}}\t{{.Repository}}\t{{.Tag}}"
IMAGE ID            REPOSITORY          TAG
5f515359c7f8        redis               latest
05a60462f8ba        nginx               latest
fe9198c04d62        mongo               3.2
00285df0df87        <none>              <none>
f753707788c5        ubuntu              18.04
f753707788c5        ubuntu              latest
```

---


### `docker image rm centos:latest` 删除

删除本地的镜
- 用 ID、镜像名、摘要删除镜像
- 可以是 镜像短ID 、 镜像长ID 、 镜像名 或者 镜像摘要 。

```bash
$ docker image rm [选项] <镜像1> [<镜像2> ...]

$ docker image ls
REPOSITORY                  TAG                 IMAGE ID            CREATED             SIZE
centos                      latest              0584b3d2cf6d        3 weeks ago         196.5 MB
redis                       alpine              501ad78535f0        3 weeks ago         21.03 MB
docker                      latest              cf693ec9b5c7        3 weeks ago         105.1 MB
nginx                       latest              e43d811ce2f4        5 weeks ago         181.5 MB

# 用短 ID 来删除镜像。
# docker image ls 默认已经是短 ID 了
# 一 般取前3个字符以上就可以了。
$ docker image rm 501

# 用 镜像名 ，<仓库名>:<标签>
$ docker image rm centos:latest

# 用 镜像摘要 删除镜像。
$ docker image ls --digests
REPOSITORY   TAG       DIGEST                                                                    IMAGE ID            CREATED             SIZE
node         slim      sha256:b4f0e0bdeb578043c1ea6862f0d40cc4afe32a4a582f3be235a3b164422be228   6e0c4c8e3913        3 weeks ago         214 MB

$ docker image rm node@sha256:b4f0e0bdeb578043c1ea6862f0d40cc4afe32a4a582f3be235a3b164422be228
Untagged: node@sha256:b4f0e0bdeb578043c1ea6862f0d40cc4afe32a4a582f3be235a3b164422be228
```



#### Untagged 和 Deleted

删除行为分为两类，一类是 `Untagged` ，另一类 是 `Deleted` 。
- 镜像的唯一标识是其 `ID` 和 `摘要`
- 一个镜像可以有多个标签。
- 因此当我们使用上面命令删除镜像的时候，实际上是在要求删除某个标签的镜像。
  - 将满足我 们要求的所有镜像标签都取消，这就是我们看到的 Untagged 的信息。
  - 因为一个镜像可以对应多个标签，当还有别的标签指向了这个镜像，那么 Delete 行为 就不会发生。
- 所以并非所有的 docker image rm 都会产生删除镜像的行为，有可能仅仅是取消了某个标签而已。
- 当该镜像所有的标签都被取消了，该镜像很可能会失去了存在的意义，因此会触发删除行为。
- 镜像是多层存储结 构，因此在删除的时候也是从上层向基础层方向依次进行判断删除。
- 镜像的多层结构让镜像复用变得非常容易， 因此很有可能某个其它镜像正依赖于当前镜像的某一层。
- 这种情况，依旧不会触发删除该层的行为。直到没有任何层依赖当前层时，才会真实的删除当前层。
- 这就是为什么，有时候会奇怪，为什么明明没有别的标签指向这个 镜像，但是它还是存在的原因，也是为什么有时候会发现所删除的层数和自己 docker pull 看到的层数不 一样的原因。
- 除了镜像依赖以外，还需要注意的是容器对镜像的依赖。如果有用这个镜像启动的容器存在(即使容器没有运 行)，那么同样不可以删除这个镜像。
- 容器是以镜像为基础，再加一层容器存储层，组成这样的多层 存储结构去运行的。
- 因此该镜像如果被这个容器所依赖的，那么删除必然会导致故障。
- 如果这些容器是不需要 的，应该先将它们删除，然后再来删除镜像。

```bash
# 用 docker image ls 命令来配合像其它可以承接多个实体的命令一样，可以使用 docker image ls -q 来配合使用docker image rm ，这样可以成批的删除希望删除的镜像。
# 删除所有仓库名为 redis 的镜像:
$ docker image rm $(docker image ls -q redis)
# 或者删除所有在 mongo:3.2 之前的镜像:
$ docker image rm $(docker image ls -q -f before=mongo:3.2)
```


---

## commit 理解镜像构成

```bash

首先使用 docker ps -l命令获得安装完ping命令之后容器的id。然后把这个镜像保存为learn/ping。

提示：
1. 运行docker commit，可以查看该命令的参数列表。

2. 你需要指定要提交保存容器的ID。(译者按：通过docker ps -l 命令获得)

3. 无需拷贝完整的id，通常来讲最开始的三至四个字母即可区分。（译者按：非常类似git里面的版本号)

# 正确的命令：
$ docker commit 698 learn/ping
```


---


## `docker` command

- 容器是独立运行的`一个或一组应用`，以及它们的`运行态环境`。
- 对应的，虚拟机可以理解为`模拟运行的一整套操作系统`(提供了运行态环境和其他系统环境)和`跑在上面的应用`。


### 启动容器 `docker run ubuntu:18.04`
启动容器有两种方式
- 基于镜像新建一个容器并启动，
- 将在终止状态( stopped )的容器重新启动。


当利用 docker run 来创建容器时，Docker 在后台运行的标准操作包括:
- 检查本地是否存在指定的镜像，不存在就从公有仓库下载
- 利用镜像创建并启动一个容器
- 分配一个文件系统，并在只读的镜像层外面挂载一层可读写层
- 从宿主主机配置的网桥接口中桥接一个虚拟接口到容器中去
- 从地址池配置一个 ip 地址给容器
- 执行用户指定的应用程序
- 执行完毕后容器被终止


> 因为 Docker 的容器实在太轻量级了，很多时候都是随时删除和新创建容器。


```bash
# 新建并启动
# 输出一个 “Hello World”，之后终止容器。
$ docker run ubuntu:18.04 /bin/echo 'Hello world'
Hello world
# 这跟在本地直接执行 /bin/echo 'hello world' 几乎感觉不出任何区别。


# 启动一个 bash 终端，允许用户进行交互。
$ docker run -it ubuntu:18.04 /bin/bash
root@af8bae53bdd3:/#
root@af8bae53bdd3:/# ls
bin boot dev etc home lib lib64 media mnt opt proc root run sbin srv sys tmp usr var
# -t 选项让Docker分配一个伪终端(pseudo-tty)并绑定到容器的标准输入上
# -i 则让容器的标准输入保持打开。
# 在交互模式下，用户可以通过所创建的终端来输入命令


# 列出本机正在运行的容器
$ docker container ls

# 列出本机所有容器，包括终止运行的容器
$ docker container ls --all


```


---

### 启动已终止容器 `docker container start`

将一个已经终止的容器启动运行。
- 容器的核心为所执行的应用程序，所需要的资源都是应用程序运行所必需的。
- 除此之外，并没有其它的资源。
- 可 以在伪终端中利用 ps 或 top 来查看进程信息。


```bash
root@ba267838cc1b:/# ps
  PID TTY          TIME CMD
    1 ?        00:00:00 bash
   11 ?        00:00:00 ps


可见，容器中仅运行了指定的 bash 应用。
这种特点使得 Docker 对资源的利用率极高，是货真价实的轻量级虚 拟化。
```

---

### Daemon 运行 `docker run -d ubuntu:18.04 /bin/bash whoami`


```bash
# 1. 后台运行
# 让 Docker 在后台运行而不是直接把执行命令的结果输出在当前宿主机下。
# 通过添 加 -d 参数来实现。

# 不使用 -d 参数运行容器。
# 容器会把输出的结果 (STDOUT) 打印到宿主机上面
$ docker run ubuntu:18.04 /bin/sh -c "while true; do echo hello world; sleep 1; done"
hello world
hello world
hello world
hello world

# 使用了 -d 参数运行容器。
# 此时容器会在后台运行并不会把输出的结果 (STDOUT) 打印到宿主机上面
$ docker run -d ubuntu:18.04 /bin/sh -c "while true; do echo hello world; sleep 1; done"
77b2dc01fe0f3f1265df143181e7b9af5e05279a884f4776ee75350ea9d8017a
# 输出结果可以用 docker logs 查 看
$ docker container logs [container ID or NAMES]
hello world
hello world
hello world
. . .

# 注: 容器是否会长久运行，是和 docker run 指定的命令有关，和 -d 参数无关。
# 使用 -d 参数启动后会返回一个唯一的 id，
# 可以通过 docker container ls 命令来查看容器信息。
$ docker container ls
CONTAINER ID  IMAGE         COMMAND                  CREATED        STATUS             PORTS NAMES
77b2dc01fe0f  ubuntu:18.04  /bin/sh -c 'while tr..'  2 minutes ago  Up 1 minute        agitated_wright
```

---

### 终止容器 `docker container stop`

- 当 Docker 容器中指定的应用终结时，容器也自动终止。
- 例如对于上一章节中只启动了一个终端的容器，用户通过 exit 命令或 Ctrl+d 来退出终端时，所创建的 容器立刻终止。

```bash
# 终止状态的容器: docker container ls -a
docker container ls -a
CONTAINER ID   IMAGE                    COMMAND                CREATED             STATUS                          PORTS               NAMES
ba267838cc1b   ubuntu:18.04             "/bin/bash"            30 minutes ago      Exited (0) About a minute ago                       trusting_newton
98e5efa7d997   training/webapp:latest   "python app.py"        About an hour ago   Exited (0) 34 minutes ago                           backstabbing_pike


# 处于终止状态的容器，通过 docker container start 命令来重新启动。
# docker container restart 命令会将一个运行态的容器终止，然后再重新启动它。
```

---

### 进入容器 `docker run -dit ubuntu`

使用 `-d` 参数，容器启动后会进入后台。
- 某些时候需要进入容器进行操作，包括使用 `docker attach` 命令或 `docker exec` 命令，
- 推荐使用 docker exec 命令

> 只用 -i 参数时，由于没有分配伪终端，界面没有熟悉的 Linux 命令提示符，但命令执行结果仍然可以返回。
> -i -t 参数一起使用时，则可以看到Linux 命令提示符。


```bash
# docker attach 命令
$ docker run -dit ubuntu
243c32535da7d142fb0e6df616a3c3ada0b8ab417937c853a9e1c251f499f550

$ docker container ls
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
243c32535da7        ubuntu:latest       "/bin/bash"         18 seconds ago      Up 17 seconds                           nostalgic_hypatia

$ docker attach 243c
root@243c32535da7:/#
# 注意: 如果从这个 stdin 中 exit，会导致容器的停止。



# exec 命令
$ docker run -dit ubuntu
69d137adef7a8a689cbcb059e94da5489d3cddd240ff675c640c8d96e84fe1f6

$ docker container ls
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
69d137adef7a        ubuntu:latest       "/bin/bash"         18 seconds ago      Up 17 seconds                           zealous_swirles

$ docker exec -i 69d1 bash
ls
bin
boot
dev
...

$ docker exec -it 69d1 bash
root@69d137adef7a:/#

# 如果从这个 stdin 中 exit，不会导致容器的停止。
# 这就是为什么推荐大家使用docker exec
```


---

### 导出容器 `$ docker export 7691a814370e > ubuntu.tar`

```bash
# 导出容器 docker export
$ docker container ls -a
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS                    PORTS               NAMES
7691a814370e        ubuntu:18.04        "/bin/bash"         36 hours ago        Exited (0) 21 hours ago                       test

$ docker export 7691a814370e > ubuntu.tar
```

---


### 导入容器 `$ cat ubuntu.tar | docker import - test/ubuntu:v1.0`

```bash
导入 docker import

# 从容器快照文件中再导入为镜像
$ cat ubuntu.tar | docker import - test/ubuntu:v1.0

$ docker image ls
REPOSITORY          TAG                 IMAGE ID            CREATED              VIRTUAL SIZE
test/ubuntu         v1.0                9d37a6082e97        About a minute ago   171.3 MB


# 通过指定 URL 或者某个目录来导入
$ docker import http://example.com/exampleimage.tgz example/imagerepo



# docker load 导入 镜像存储文件 到本地镜像库，
# docker import 导入一个 容器快照 到本地镜像库。
# 两者的区别在于
# 容器快照文件将丢弃所有的历史记录和元数据信息(即仅 保存容器当时的快照状态)，
# 而镜像存储文件将保存完整记录，体积也要大。
# 此外，从容器快照文件导入时可以 重新指定标签等元数据信息。
```


---


### `docker container rm ubuntu` 删除容器


```bash
# 删除终止状态的容器
$ docker container rm trusting_newton
trusting_newton

# 删除一个运行中的容器，
# 添加 -f 参数。Docker 会发送 SIGKILL 信号给容器。

# 对于那些不会自动终止的容器，必须使用docker container kill 命令手动终止。
$ docker container kill [containID]

# 清理所有处于终止状态的容器
# 用 docker container ls -a 命令可以查看所有已经创建的包括终止状态的容器，如果数量太多，用下面的命令可以清理掉所有处于终止状态的容器。
$ docker container prune
```


---

## 访问仓库


`仓库( Repository )`是集中存放镜像的地方。
- 一个容易混淆的概念是`注册服务器( Registry )`。
- 实际上 Registry 是管理 Repository 的具体服务器
- 每个 Registry 上可以有多个Repository，而每个Repository下面有多个镜像。
- 仓库可以被认为是一个具体的项目或目录。
- 例 如对于仓库地址 docker.io/ubuntu 来说， docker.io 是注册服务器地址， ubuntu 是仓库名。
- 大部分时候，并不需要严格区分这两者的概念。

### Docker Hub
- Docker 官方维护一个公共仓库 Docker Hub，其中已经包括了数量超过 2,650,000 的镜像。
- 大部分需求 都可以通过在 Docker Hub 中直接下载镜像来实现。
  - 注册 https://hub.docker.com
  - 执行 docker login 命令交互式的输入用户名及密码来完成在命令行界面登录 Docker Hub。
  - 通过 docker logout 退出登录。
  - 通过 docker search 命令来查找官方仓库中的镜像，
    - `$ docker search centos`
    - ![Screen Shot 2020-11-27 at 23.47.30](https://i.imgur.com/tsm4uv5.png)
  - 利用 docker pull 命令来将它下载到 本地。
    - `$ docker pull centos`
    - ![Screen Shot 2020-11-27 at 23.46.59](https://i.imgur.com/T5Dlbur.png)
  - 通过 docker push 将自己的镜像推送到 Docker Hub。
    - `$ docker push username/ubuntu:18.04`
    - ![Screen Shot 2020-11-27 at 23.50.11](https://i.imgur.com/6wlM31P.png)


### 自动构建 Automated Builds

对于需要经常升级镜像内程序来说十分方便。
- 有时候，构建了镜像，安装了某个软件，当软件发布新版本则需要手动更新镜像。
- 而自动构建允许用户通过 Docker Hub 指定跟踪一个目标网站(支持 GitHub 或 BitBucket)上的项目，一旦项 目发生新的提交 (commit)或者创建了新的标签(tag)，Docker Hub 会自动构建镜像并推送到 Docker Hub 中。

```bash
# 要配置自动构建，包括如下的步骤:

登录 Docker Hub;
在 Docker Hub 点击右上角头像，在账号设置(Account Settings)中关联(Linked Accounts)目标网 站;
在 Docker Hub 中新建或选择已有的仓库，在 Builds 选项卡中选择 Configure Automated Builds ;
选取一个目标网站中的项目(需要含 Dockerfile )和分支;
指定 Dockerfile 的位置，并保存。
之后，可以在 Docker Hub 的仓库页面的 Timeline 选项卡中查看每次构建的状态。
```



---


### 私有仓库

用户可以创建一个本地仓库供私人使用。
- `docker-registry` 是官方提供的工具，用于构建私有的镜像仓库。

```bash
# 安装运行 docker-registry 容器运行
# 通过获取官方 registry 镜像来运行。
$ docker run -d -p 5000:5000 --restart=always --name registry registry
# 这将使用官方的 registry 镜像来启动私有仓库。


# 默认情况下，仓库会被创建在容器的 /var/lib/registry 目录下。
# 通过 -v 参数来将镜像文件存放在本地的指定路径。
# 例如下面的例子将上传的镜像放到本地的 /opt/data/registry 目录。
$ docker run -d -p 5000:5000 -v /opt/data/registry:/var/lib/registry registry



# 在私有仓库上传、搜索、下载镜像
# 使用 docker tag 来标记一个镜像，然后推送它到仓库。
# 例如私有仓库地址为 127.0.0.1:5000 。

# 在本机查看已有的镜像。
$ docker image ls
REPOSITORY                        TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
ubuntu                            latest              ba5877dc9bec        6 weeks ago         192.7 MB

# 使用 docker tag 将 ubuntu:latest 这个镜像标记为 127.0.0.1:5000/ubuntu:latest 。
# 格式为 docker tag IMAGE[:TAG] [REGISTRY_HOST[:REGISTRY_PORT]/]REPOSITORY[:TAG] 。
$ docker tag ubuntu:latest 127.0.0.1:5000/ubuntu:latest
$ docker image ls
REPOSITORY                        TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
ubuntu                            latest              ba5877dc9bec        6 weeks ago         192.7 MB
127.0.0.1:5000/ubuntu:latest      latest              ba5877dc9bec        6 weeks ago         192.7 MB



# 用 curl 查看仓库中的镜像。
$ curl 127.0.0.1:5000/v2/_catalog
{"repositories":["ubuntu"]}
# 这里可以看到 {"repositories":["ubuntu"]} ，表明镜像已经被成功上传了。


# 先删除已有镜像，再尝试从私有仓库中下载这个镜像。
$ docker image rm 127.0.0.1:5000/ubuntu:latest

$ docker pull 127.0.0.1:5000/ubuntu:latest
Pulling repository 127.0.0.1:5000/ubuntu:latest ba5877dc9bec: Download complete
511136ea3c5a: Download complete
9bad880da3d2: Download complete
25f11f5fb0cb: Download complete
ebc34468f71d: Download complete
2318d26665ef: Download complete

$ docker image ls
REPOSITORY                         TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
127.0.0.1:5000/ubuntu:latest       latest              ba5877dc9bec        6 weeks ago         192.7 MB


注意事项
如果你不想使用 127.0.0.1:5000 作为仓库地址，比如想让本网段的其他主机也能把镜像推送到私有仓库。
就得把例如 192.168.199.100:5000 这样的内网地址作为私有仓库地址，这时你会发现无法成功推 送镜像。
这是因为 Docker 默认不允许非 HTTPS 方式推送镜像。我们可以通过 Docker 的配置选项来取消这个限制， 或者查看下一节配置能够通过 HTTPS 访问的私有仓库。
```


---


### Ubuntu 16.04+, Debian 8+, centos 7



```bash
# 对于使用 systemd 的系统，请在 /etc/docker/daemon.json 中写入如下内容(如果文件不存在请新 建该文件)
{
  "registry-mirror": [
    "https://registry.docker-cn.com"
  ],
  "insecure-registries": [
    "192.168.199.100:5000"
  ]
}


注意:该文件必须符合 json 规范，否则 Docker 将不能启动。
```

---

### 其他

```
对于 Docker Desktop for Windows 、 Docker Desktop for Mac 在设置中的 辑 ，增加和上边一样的字符串即可。
```


---



## 数据管理

![types-of-mounts](https://i.imgur.com/7WJZSlC.png)

在容器中管理数据主要有两种方式:
- 数据卷(Volumes)
- 挂载主机目录 (Bind mounts)

---

### 数据卷(Volumes)

- 一个可供一个或多个容器使用的特殊目录
- 它绕过 UFS，可以提供很多有用的特性:
- 可以在容器之间共享和重用
- 对 数据卷 的修改会立马生效
- 对 数据卷 的更新，不会影响镜像
- 数据卷 默认会一直存在，即使容器被删除
注意: 数据卷 的使用，类似于 Linux 下对目录或文件进行 mount，镜像中的被指定为挂载点的目录中的文件会复制到数据卷中(仅数据卷为空时会复制)。


```bash
# 创建一个数据卷
$ docker volume create my-vol

# 查看所有的 数据卷
$ docker volume ls
DRIVER VOLUME NAME
local my-vol
```

---



#### 删除数据卷 `docker volume rm my-vol`


数据卷 是被设计用来持久化数据的
- 它的生命周期独立于容器，Docker 不会在容器被删除后自动删除
- 也不存在垃圾回收这样的机制来处理没有任何容器引用的 数据卷 。
- 删除数据卷 `docker volume rm my-vol`
- 如果需要在 删除容器的同时移除数据卷 `docker rm -v`
- 无主的数据卷会占据很多空间，要清理请使用以下命令 `docker volume prune`


---


#### 启动一个挂载数据卷的容器

用 docker run 命令的时候
- 使用 `--mount` 标记来将 数据卷 挂载到容器里。
- 在一次 docker run 中可以挂载多个 数据卷 。

```bash
# 创建一个名为 web 的容器，
# 并加载一个 数据卷 到容器的 /usr/share/nginx/html 目录。
$ docker run -d -P --name web \
    # -v my-vol:/webapp \
    --mount source=my-vol, target=/webapp \
    training/webapp \
    python app.py
```


---

### 启动一个挂载主机目录的容器

- 挂载一个主机目录作为数据卷
- 使用 --mount 标记可以指定挂载一个本地主机的目录到容器中去。


```bash
$ docker run -d -P --name web \
    # -v /src/webapp:/opt/webapp \
    --mount type=bind, source=/src/webapp, target=/opt/webapp \
    training/webapp \
    python app.py

# 上面的命令加载主机的 /src/webapp 目录到容器的 /usr/share/nginx/html 目录。
# 这个功能在进行 测试的时候十分方便，比如用户可以放置一些程序到本地目录中，来查看容器是否正常工作。
# 本地目录的路径必 须是绝对路径，
# 以前使用 -v 参数时如果本地目录不存在 Docker 会自动为你创建一个文件夹，
# 现在使用--mount 参数时如果本地目录不存在，Docker 会报错。

# Docker挂载主机目录的默认权限是 读写 ，用户也可以通过增加 readonly 指定为 只读 。
$ docker run -d -P \
    --name web \
    # -v /src/webapp:/opt/webapp:ro \
    --mount type=bind,source=/src/webapp,target=/opt/webapp,readonly \
    training/webapp \
    python app.py



# 加了 readonly 之后，就挂载为 只读 了。
# 如果你在容器内 /usr/share/nginx/html 目录新建文件，会显示如下错误
/opt/webapp # touch new.txt
touch: new.txt: Read-only file system
```

---


#### 查看数据卷的具体信息 `$ docker inspect web`

```bash
# 在主机里使用以下命令可以查看 web 容器的信息
$ docker inspect web
# 挂载主机目录 的配置信息在 "Mounts" Key 下面
"Mounts": [
    {
        "Type": "bind",
        "Source": "/src/webapp",
        "Destination": "/opt/webapp",
        "Mode": "",
        "RW": true,
        "Propagation": "rprivate"
    }
],
```


---


#### 挂载一个本地主机文件作为数据卷

```bash
# 从主机挂载单个文件到容器中
$ docker run --rm -it \
   # -v $HOME/.bash_history:/root/.bash_history \
   --mount type=bind, source=$HOME/.bash_history, target=/root/.bash_history \
   ubuntu:18.04 bash


$ docker run --rm -it --mount type=bind, source=$HOME/.bash_history, target=/root/.bash_history ubuntu:18.04 bash

root@2affd44b4667:/# history
1  ls
2  diskutil list

# 这样就可以记录在容器输入过的命令了。
```

---


## 网络

> Docker 中的网络功能介绍 Docker 允许通过外部访问容器或容器互联的方式来提供网络服务。


---

### 外部访问容器

- 容器中可以运行一些网络应用，要让外部也可以访问这些应用，通过 -P 或 -p 参数来指定端口映 射。
- -P: Docker 会随机映射一个端口到内部容器开放的网络端口。
- -p: 可以指定要映射的端口，
  - 并且，在一个指定端口上只可以绑定一个容器。
  - 支持的格式有
  - `ip:hostPort:containerPort`
  - `ip::containerPort`
  - `hostPort:containerPort`


```bash

# 使用 docker container ls 可以看到，本地主机的 32768 被映射到了容器的 80 端口。
# 此时访问本机的 32768 端口即可访问容器内 NGINX 默认页面。
$ docker run -d -P training/webapp python app.py

$ docker container ls -l
CONTAINER ID  IMAGE                   COMMAND       CREATED        STATUS        PORTS                    NAMES
bc533791f3f5  training/webapp:latest  python app.py 5 seconds ago  Up 2 seconds  0.0.0.0:49155->5000/tcp  nostalgic_morse



# 可以通过 docker logs 命令来查看应用的信息。
$ docker logs -f nostalgic_morse
* Running on http://0.0.0.0:5000/
10.0.2.2 - - [23/May/2014 20:16:31] "GET / HTTP/1.1" 200 -
10.0.2.2 - - [23/May/2014 20:16:31] "GET /favicon.ico HTTP/1.1" 404 -
```




> 注意：
> 容器有自己的内部网络和 ip 地址（使用 docker inspect 可以获取所有的变量，Docker 还可以有一个可变的网络配置。）

-p 标记可以多次使用来绑定多个端口

例如

```bash
$ docker run -d -p 5000:5000 -p 3000:80 \
    training/webapp \
    python app.py
```


---

#### 映射所有接口地址 `$ docker run -d -p 5000:5000 training/webapp python app.py`

```bash
# 使用 hostPort:containerPort 格式本地的 80 端口映射到容器的 80 端口:
$ docker run -d -p 5000:5000 training/webapp python app.py
此时默认会绑定本地所有接口上的所有地址。
```



---

#### 映射到指定地址的指定端口 `$ docker run -d -p 127.0.0.1:5000:5000 training/webapp python app.py`

```bash
# 可以使用 ip:hostPort:containerPort 格式指定映射使用一个特定地址，比如 localhost 地址 127.0.0.1
$ docker run -d -p 127.0.0.1:5000:5000 training/webapp python app.py
```


---

#### 映射到指定地址的任意端口

```bash
# 使用 ip::containerPort 绑定 localhost 的任意端口到容器的 80 端口，本地主机会自动分配一个端口。
$ docker run -d -p 127.0.0.1::5000 training/webapp python app.py


# 还可以使用 udp 标记来指定 udp 端口
$ docker run -d -p 127.0.0.1:5000:5000/udp training/webapp python app.py
```

---


#### 查看映射端口配置 `$ docker port container_name 5000`

```bash
# 使用 docker port 来查看当前映射的端口配置，也可以查看到绑定的地址
$ docker port container_name 5000
127.0.0.1:49155.
```


---

### 容器互联

容器互联
- 使用 --link 参数来使容器互联。
- 将容器加入自定义的 Docker 网络来连接多个容器



---

### 新建网络 `$ docker network create -d bridge my-net`

```bash
# 创建一个新的 Docker 网络。
$ docker network create -d bridge my-net
# -d 参数指定 Docker 网络类型: bridge, overlay(用于 Swarm mode)
```

---


### 连接容器

```bash
# 运行一个容器并连接到新建的 my-net 网络
$ docker run -it --rm --name busybox1 --network my-net busybox sh
# 打开新的终端，再运行一个容器并加入到 my-net 网络
$ docker run -it --rm --name busybox2 --network my-net busybox sh
# 再打开一个新的终端查看容器信息
$ docker container ls

CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
b47060aca56b        busybox             "sh"                11 minutes ago      Up 11 minutes                           busybox2
8720575823ec        busybox             "sh"                16 minutes ago      Up 16 minutes                           busybox1


# 通过 ping 来证明 busybox1 容器和 busybox2 容器建立了互联关系。
# 在 busybox1 容器输入以下命令
/ # ping busybox2
PING busybox2 (172.19.0.3): 56 data bytes
64 bytes from 172.19.0.3: seq=0 ttl=64 time=0.072 ms
64 bytes from 172.19.0.3: seq=1 ttl=64 time=0.118 ms
# 用 ping 来测试连接 busybox2 容器，它会解析成 172.19.0.3。


# 同理在 busybox2 容器执行 ping busybox1，也会成功连接到。
/ # ping busybox1
PING busybox1 (172.19.0.2): 56 data bytes
64 bytes from 172.19.0.2: seq=0 ttl=64 time=0.064 ms
64 bytes from 172.19.0.2: seq=1 ttl=64 time=0.143 ms


# 这样， busybox1 容器和 busybox2 容器建立了互联关系。


Docker Compose
如果你有多个容器之间需要互相连接，推荐使用 Docker Compose。
```


---


### 配置 DNS

自定义配置容器的主机名和 DNS
- Docker 利用虚拟文件来挂载容器的 3 个相关配置文件。


```bash
# 在容器中使用 mount 命令可以看到挂载信息：
# 这种机制可以让宿主主机 DNS 信息发生更新后，所有 Docker 容器的 DNS 配置通过 /etc/resolv.conf 文件立刻得到更新。
$ mount
/dev/disk/by-uuid/1fec...ebdf on /etc/hostname type ext4 ...
/dev/disk/by-uuid/1fec...ebdf on /etc/hosts type ext4 ...
tmpfs on /etc/resolv.conf type tmpfs ...



# 也可以在 /etc/docker/daemon.json 文件中增加以下内容来设置。配置全部容器的 DNS.
# 这样每次启动的容器 DNS 自动配置为 114.114.114.114 和 8.8.8.8。
{
  "dns" : [
    "114.114.114.114",
    "8.8.8.8"
  ]
}


# 使用以下命令来证明其已经生效。
$ docker run -it --rm ubuntu:18.04 cat etc/resolv.conf
nameserver 114.114.114.114
nameserver 8.8.8.8



# 手动指定容器的配置，在使用 docker run 命令启动容器时加入如下参数:
# -h HOSTNAME, --hostname=HOSTNAME
    # 设定容器的主机名，它会被写到容器内的 /etc/hostname 和 /etc/hosts 。
    # 但它在容器外部看不到，既不会在 docker container ls 中 显示，也不会在其他的容器的 /etc/hosts 看到。

# --dns=IP_ADDRESS
    # 添加 DNS 服务器到容器的 /etc/resolv.conf 中
    # 让容器用这个服务器来解析所有不在 /etc/hosts 中的主机名。

# --dns-search=DOMAIN
    # 设定容器的搜索域
    # 当设定搜索域为 .example.com 时，在搜索一个名为 host 的主机时，DNS 不仅搜索 host，还会搜索 host.example.com

# 注意:如果在容器启动时没有指定最后两个参数，Docker 会默认用主机上的 /etc/resolv.conf 来配 置容器。
```


---


## 高级网络配置





---


## Docker Compose

Docker Compose
- Docker 官方编排(Orchestration)开源项目之一
- 负责实现对 Docker 容器集群的快速编排。
- 从功能上看，跟OpenStack 中的 Heat 十分类似。
- 其代码目前在 https://github.com/docker/compose 上开源。
- Compose 定位是 「定义和运行多个 Docker 容器的应用(Defining and running multi-container Docker applications)」，其前身是开源项目 Fig。



使用一个 Dockerfile 模板文件，可以让用户很方便的定义一个单独的应用容器。
- 然而，在日常工作中，经常会碰到需要多个容器相互配合来完成某项任务的情况。
- 例如要实现一个 Web 项目，除了 Web 服务容器本身，往往还需要再加上后端的数据库服务容器，甚至还包括负载均衡容器等。


Compose 恰好满足了这样的需求。
- 它允许用户通过一个单独的 docker-compose.yml 模板文件 (YAML 格式)来定义一组相关联的应用容器为一个项目(project)。


Compose 中有两个重要的概念:
- `服务 ( service )`:
  - 一个应用的容器，
  - 实际上可以包括若干运行相同镜像的容器实例。
- `项目 ( project )`:
  - 由一组关联的应用容器组成的一个完整业务单元，在 docker-compose.yml 文 件中定义。
- 一个 project 可以由多个 service（容器）关联而成
- Compose 面向 project 进行管理。
- Compose 的默认管理对象是项目，通过子命令对项目中的一组容器进行便捷地生命周期管理。


Compose 项目由 Python 编写，实现上调用了 Docker 服务提供的 API 来对容器进行管理。
- 因此，只要所操 作的平台支持 Docker API，就可以在其上利用 Compose 来进行编排管理。


---

### 安装与卸载


`Compose` 支持 Linux、macOS、Windows 10 三大平台。

`Compose` 可以通过 Python 的包管理工具 `pip` 进行安装，也可以直接下载编译好的二进制文件使用，甚至能够直接在 Docker 容器中运行。

`Docker Desktop for Mac/Windows` 自带 `docker-compose` 二进制文件，安装 Docker 之后可以直接使用。

    $ docker-compose --version

    docker-compose version 1.24.1, build 4667896b


```bash
# Linux安装。

# 二进制包
# 从 [官方 GitHub Release](https://github.com/docker/compose/releases) 处直接下载编译好的二进制文件即可。
# 例如，在 Linux 64 位系统上直接下载对应的二进制包。

$ sudo curl -L https://github.com/docker/compose/releases/download/1.24.1/docker-compose-`uname -s`-`uname -m` > /usr/local/bin/docker-compose

$ sudo chmod +x /usr/local/bin/docker-compose


# PIP 安装
# _注：_ `x86_64` 架构的 Linux 建议按照上边的方法下载二进制包进行安装，如果您计算机的架构是 `ARM` (例如，树莓派)，再使用 `pip` 安装。
# 这种方式是将 Compose 当作一个 Python 应用来从 pip 源中安装。

# 执行安装命令：
$ sudo pip install -U docker-compose
# 可以看到类似如下输出，说明安装成功。
Collecting docker-compose
Downloading docker-compose-1.24.1.tar.gz (149kB): 149kB downloaded
...
Successfully installed docker-compose cached-property requests texttable websocket-client docker-py dockerpty sienum34 backports.ssl-match-hostname ipaddress


# bash 补全命令
$ curl -L https://raw.githubusercontent.com/docker/compose/1.24.1/contrib/completion/bash/docker-compose > /etc/bash_completion.d/docker-compose


# 卸载

# 如果是二进制包方式安装的，删除二进制文件即可。

    $ sudo rm /usr/local/bin/docker-compose


# 如果是通过 `pip` 安装的，则执行如下命令即可删除。

    $ sudo pip uninstall docker-compose
```
























---

ref
- [Visualizing Docker Containers and Images](http://merrigrove.blogspot.com/2015/10/visualizing-docker-containers-and-images.html)



.
