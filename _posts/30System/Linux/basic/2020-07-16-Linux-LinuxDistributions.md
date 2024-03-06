---
title: Linux - Linux Distributions
date: 2020-07-16 11:11:11 -0400
categories: [30System, Basic]
tags: [Linux]
math: true
image:
---

# Linux distributions

[toc]


## which one

phones: Android
routers: DDWRT
security: Kali Linux
Internet: Nest, Raspberry Pi
privacy,: Tails


## intro

Linux distributions use a *package management system* to install, update, and remove software.
- Set of tools/utilities to make it easier to install, upgrade, remove and configure software packages.
- Typically connect to the internet and download packages from remote servers
- Package managers that do not recursively get dependencies are more difficult to use.


Some package formats:


Debian, foundation for Ubuntu

Ubuntu – deb/dpkg
- Linux Mint is an Ubuntu-based distro

Red Hat – RPM and YUM
- Fedora is the free version of Red Hat
- centos
- Oracle Linux, which is a derivative of Red Hat Enterprise Linux
- Scientific Linux, a distribution derived from the same sources used by Red Hat
- Mandriva Linux was a Red Hat Linux derivative popular in several European countries and Brazil



Solaris – pkgadd


**Ubuntu**
uses APT (Advanced Packaging Tool)
- Synaptic and aptitude are Front End interfaces
- Synaptic has GUI
- aptitude has text based GUI

APT uses `/etc/apt/sources.list`, list which servers to download the packages from
- Can add more servers and more packages
- Can setup your own repositories (repos) for package distribution - a must in high security environments


to installing software on Ubuntu

```c
First step, update your sources:

    apt-get update

Second step, apply upgrades

    apt-get upgrade

Third step, install whatever you want:

    apt-get install "package name"
    apt-get install openssh-server
```


**Red Hat**
uses YUM

YUM uses `/etc/yum.repos.d/` to list which servers to download the packages from
- Can add more servers and more packages
- Can setup your own repositories (repos) for package distribution - this is a must in high security environments


to installing software on Red Hat

```c
First step, always update your sources:

    yum update

Second step, install whatever you want:

    yum install "package name"
    yum install openssh-server

```


## CentOS

`Community Enterprise Operating System`
社区企业操作系统, 是Linux操作系统的一个发行版本。

CentOS并不是全新的Linux发行版，倘若一说到Red Hat这个大名，大家似乎都听过。在Red Hat家族中有企业版的产品，它是Red Hat Enterprise Linux（以下称之为RHEL），CentOS正是这个RHEL的克隆版本。

RHEL是很多企业采用的Linux发行版本，需要向Red Hat付费才可以使用，并能得到付过费用的服务和技术支持和版本升级。CentOS可以像RHEL一样的构筑Linux系统环境，但不需要向Red Hat付任何的产品和服务费用，同时也得不到任何有偿技术支持和升级服务。

Red Hat公司的产品中，有Red Hat Linux（如Redhat8,9）和针对企业发行的版本Red Hat Enterprise Linux，都能够通过网络FTP免费的获得并使用，但是在2003年的时候，Red Hat Linux停止了发布，它的项目由Fedora Project这个项目所取代，并以Fedora Core这个名字发行并提供给普通用户免费使用。

Fedora Core这个Linux发行版更新很快，大约半年左右就有新的版本发布。目前的版本是Fedora Core 6，这个Fedora Core试验的韵味比较浓厚，每次发行都有新的功能被加入到其中，得到的成功结果将被采用道RHEL的发布中。虽说这样，频繁的被改进更新的不安定产品对于企业来说并不是最好的选择，大多数企业还是会选择有偿的RHEL产品（这里面有很深的含义，比如说企业用Linux赚钱，赚到的钱回报给企业，资金在企业间流通，回报社会，提高服务水准等）。

在构成RHEL的大多数软件包中，都是基于GPL协议发布的，也就是我们常说的开源软件。正因为是这样，Red Hat公司也遵循这个协议，将构成RHEL的软件包公开发布，只要是遵循GPL协议，任何人都可以在原有的软件构成的基础上再开发和发布。

CentOS就是这样在RHEL发布的基础上将RHEL的构成克隆再现的一个Linux发行版本。

RHEL的克隆版本不只CentOS一个，还有White Box Enterprise Linux和TAO Linux 和Scientific Linux（其他的这些都没听说过，是吧？）。

CentOS的特点
Enterprise OS，企业系统并不是企业级别的系统，而是它可以提供`企业级应用`所需要的要素。
例如：
- 稳定的环境
- 长期的升级更新支持
- 保守性强
- 大规模的系统也能够发挥很好的性能

CentOS满足以上的要素，满足上面要素的发行版还有Fedora 。Fedora和CentOS非常的相像，但是对CentOS来说，Fedora提供更多的新的功能和软件，发布更新快等特点，这样在稳定性和管理方面就增加了很多工作。

企业所需要的系统环境应该是，高效稳定的系统环境，一次构建后能够长期使用的系统环境，所以Fedora那样的频繁更新发布的系统环境并不对应企业的应用。

另一方面，CentOS却能够满足以上企业的需要，在众多的RHEL的克隆版本中，CentOS是很出众很优秀的。


CentOS 与 RHEL 的区别

RHEL 在发行的时候，有两种方式。一种是二进制的发行方式，另外一种是源代码的发行方式。哪一种都可以免费获得（例如从网上下载），并再次发布。但如果你使用了他们的在线升级（包括补丁）或咨询服务，就必须要付费。

RHEL 一直都提供源代码的发行方式，CentOS 就是将 RHEL 发行的源代码从新编译一次，形成一个可使用的二进制版本。由于 LINUX 的源代码是 GNU，所以从获得 RHEL 的源代码到编译成新的二进制，都是合法。只是 REDHAT 是商标，所以必须在新的发行版里将 REDHAT 的商标去掉。

REDHAT 对这种发行版的态度是：“我们其实并不反对这种发行版，真正向我们付费的用户，他们重视的并不是系统本身，而是我们所提供的商业服务。”

所以，CentOS 可以得到 RHEL 的所有功能，甚至是更好的软件。但 CentOS 并不向用户提供商业支持，当然也不负上任何商业责任。



1.RedHat.Enterprise.Linux.5 与 redhat linux 9.0 还有redhat fedora core 三者之间的具体关联和区别是什么？ centos又是从哪冒出来的？


redhat成名的原因：历史悠久，1993年就开始做linux；公司运营，提供完整的解决方案，更专业，而不像debian是社区形式的；独创rpm包，使linux安装软件变得非常简单，免去编译的麻烦。

redhat在发行的9.03版之后，就不再延续以前的开发代号，而是以RedHat.Enterprise.Linux命名（简称rhel）即redhat企业版，现在已  经开发到5，rhel好像是从3开始，需要客户购买license，即想获得系统的后续更新与服务是需要付费的（可以免费更新60天，而且如果不想享受更新，系统也是可以免费使用的。），而其个人桌面免费版交给redhat社区在做，这个社区是可以获得redhat公司支持的，这个社区发布的版本就是fedora（直译也是一种男士帽子），一年两个版本，现在已经发行到10，fedora一直是rhel的一个实验场，每个版本所采用的软件，内核与库版本几乎都是最新的，因而配置起来有些困难，不过基于redhat的基础，使用fedora的人仍然占很大的比例。centos是将rhel再次编译，去掉redhat标志，并有社区发布的linux版本，所以，centos与rhel几乎是没有区别的，主要的区别就是不用付费即可使用，从rhel的角度来说，centos是非常适合企业使用的。


linux盈利方式：linux个人桌面版是可以免费获得并使用的，但像redhat企业版，redflag红旗企业版，是需要购买服务的，企业版主要针对的是银行，政府，或者大型企业这种对于稳定性和安全行要求较高的行业，比起昂贵的unix，linux还是有销路的。个人桌面版也并不是无利可图的，至少很多linux社区不会赔钱，因为为社区工作的人都是分布在世界各地的，开发linux也是利用业余时间来做的，不为了获得报酬，只为了一份执着而工作。

## 哪几种比较好

`Redhat企业版（rhel）`，适合企业使用，出色的稳定性和兼容性表现在每个版本都使用了比较成熟的库与内核，并且对一些大型的EDA软件都预先进行了测试安装，比如cadence，所以比较适合做服务器和工作站，但不适合当个人桌面，因为不购买license，就不能享受到丰富的更新，而且由于内核与库都比较保守，有点跟不上linux的发展速度，以至于很多娱乐软件安装起来非常困难。`centos`与rhel类似。

`fedora`，发行都比较冒进，以至于很多驱动程序都不能很好的配置，但最新的fedora10还是很保守和稳定的。yu软件源基于rpm包管理，安装软件很方便。

`ubuntu`，基于debian，桌面环境以gnome为主，是目前最流行的linux个人桌面，它的优点是配置起来非常简单，安装完系统之后，只要硬件不是太新，基本不用进行其他配置，硬件都可以识别并安装好驱动。而且其apt更新源服务器中的软件非常丰富，只要打一条命令，就可以自动从网络下载安装所需软件。安装方便，甚至于可以使用wubi将linux安装在windows分区。
ubuntu还有很多衍生版本，包括
- Kubuntu（桌面采用KDE，较为华丽），
- xubuntu（采用xfce，要求配置较低），
- eubuntu（面向儿童和教育），
用户可以根据需求，偏好，和硬件配置进行选择。

`suse` 本质和其他版本都是一样的，只是在窗口美工上开发者下了一定功夫，付出更高的系统资源占用。其他的linux版本通过一些改造，完全是可以实现suse的效果的。

`redflag`，中科院开发的linux版本，主要面向政府用户，其个人桌面版免费，美工与windows非常接近，是使用者的入门难度降低，但实际上桌面也是基于KDE的，很平常。

`puppy`，一个非常小巧的linux版本，安装镜像90多M，却包括了图形桌面，浏览器，办公等常用的软件，系统运行时都存在与内存中，据说安装在U盘中的puppy，在系统启动后，可以将U盘拿掉，系统依然可以运行。


linux的内核目前还在飞速的发展，现在常见的是2.X版本，X, 奇数为不稳定版，偶数为稳定版，比如rhel采用的2.4和目前最新的，很多个人桌面采用的2.6。不同的linux发行版本采用的内核不尽相同，比如fedora一般都是采用最新的内核。
