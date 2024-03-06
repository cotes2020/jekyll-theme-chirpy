---
title: VirtualBox Setups
date: 2020-07-16 11:11:11 -0400
categories: [30System]
tags: [Install, Setup, OS, VirtualBox]
math: true
image:
---


# VirtualBox Setups


- [VirtualBox Setups](#virtualbox-setups)
- [share clipboard](#share-clipboard)
- [设置共享文件夹](#设置共享文件夹)
  - [安装增强功能](#安装增强功能)
  - [配置增强插件](#配置增强插件)
  - [设置共享文件夹](#设置共享文件夹-1)
  - [映射网络驱动器](#映射网络驱动器)
  - [File sharing between host Windows and Virtual Machine Ubuntu](#file-sharing-between-host-windows-and-virtual-machine-ubuntu)



---

```bash

# 1. install

# 2. install and update

sudo yum update

yum -y install epel-release

yum install gcc gcc-c++ bison flex libpcap-devel qt-devel gtk3-devel rpm-build libtool c-ares-devel qt5-qtbase-devel qt5-qtmultimedia-devel qt5-linguist desktop-file-utils

yum install dpkg-devel dpkg-dev

yum install tcpdump wireshark wireshark-gnome
yum install wireshark wireshark-qt

sudo groupadd wireshark

sudo usermod -a -G wireshark $USER

sudo usermod -a -G wireshark server

sudo chgrp wireshark /usr/bin/dumpcap
sudo chmod o-rx /usr/bin/dumpcap
sudo setcap 'CAP_NET_RAW+eip CAP_NET_ADMIN+eip' /usr/bin/dumpcap
sudo getcap /usr/bin/dumpcap

yum install firewalld


3. share clipboard

4. hostfile, network, dns, restart




```



# share clipboard

Unable to insert the virtual optical disk /Applications/VirtualBox.app/Contents/MacOS/VBoxGuestAdditions.iso into the machine CentOS 7.

1. Install VirtualBox Guest Additions in CentOS

```c

1. enabling the EPEL repository on your CentOS to install some required packages needed for the installation process.

yum -y install epel-release


2. update each package on your guest system including the kernel to the latest version.
   Once the upgrade process is done, reboot system to complete the upgrade process and start using the new kernel.

yum -y update


3. Once update process completes, install all kernel headers, developer tools and other related packages that are required for installing the guest additions from source as shown.

yum install make gcc kernel-headers kernel-devel perl dkms bzip2


4. Next, set the KERN_DIR environment variable to kernel source code directory (/usr/src/kernels/$(uname -r)) and export it at the same time as shown.

export KERN_DIR=/usr/src/kernels/$(uname -r)

5. Now enable the shared clipboard and drag’n’drop functionality for your guest operating system. select an option.

5. Now, mount the Guest Additions ISO and run the installer in two ways:

    1. Install Guest Additions via a GUI

    install "Devices": VBoxGuestAdditions.iso
    // if its fail
    // follow the step:
        Shut down all your VMs.

        menu File » Virtual Media Manager » Optical Disks » find the VBoxGuestAdditions.iso » select it and from the toolbar click (if enabled) the "Release" button, and then the "Remove" button.

        open the VM
    reboot


    2. Install Guest Additions via a Terminal

    mount the Guest Additions ISO file, move into the directory where the guest additions ISO has been mounted, inside there you will find VirtualBosx guest addition installers for various platform, run the one for Linux, as follows.

    mount -r /dev/cdrom /media
    cd /media/
    ./VBoxLinuxAdditions.run
    reboot


6. power off and enable shared clipboard
```


---




# 设置共享文件夹

## 安装增强功能
* VirtualBox自带增强工具`Sun VirtualBox Guest Additions`
* 启动虚拟Win7后，点击控制菜单“设备（Devices）” > “安装增强功能(Insert Guest Additions CD images)”，可以看到程序安装界面。双击安装VBoxGuestAdditions.iso，需重启。

* 【Devices】- 【Insert Guest Additions CD Image…】
* 然后打开我的电脑，可以看到CD驱动器已经加载
* 打开光盘，看到如下目录，双击【VBoxWindowsAdditions.exe】
* 所有操作都默认即可，一路点击【Next】，然后点击【Install】
* 系统设备安装提示，点击【安装】
* 安装完成，要求重启，点击【Finish】重启即可

## 配置增强插件
* 左上角找到【Devices】- 【Shared Clipboard】>【Bidirectional】，实现剪贴板共享，现在你可以在主机上复制，在虚拟机里粘贴 （反之亦然，注意windows下是ctrl+c和ctrl+v，在mac下是commond+c和commond+v）
* 【Devices】 - 【Drag and Drop】，选择【Bidirectional】，实现文件拖拽，现在你可以拖拽文件到虚拟机，但是反过来不行，会报错，暂且没有找到方法，如果要从虚拟机拷贝文件到主机，参考下面【文件夹共享】


## 设置共享文件夹
* 切换到“Oracle VM VirtualBox管理器”主页面 > 选择相应的win7系统 > 设置 > 共享文件夹
* 【Devices】 - 【Shared Folders】 - 【Shared Folders Settings…】
* 点击右侧“+”号 > 添加共享文件夹，编辑共享文件夹路径和名称 > 勾选 `auto-mount` `make permanet` > 确定

## 映射网络驱动器
* open 虚拟机Win7 > 打开“计算机” > 选择“映射网络驱动”
* 为要连接到的连接和文件夹指定驱动器号。驱动器最好选择除“Z: ”以外的驱动器， “Z: ”作为来安装其他镜像文件时预留的驱动器，否则可能无法安装其他镜像文件，如图所示：
* 映射完成后，在“计算机”中就可以看到映射成功的网络驱动器，快速访问主机中的文件夹。




## File sharing between host Windows and Virtual Machine Ubuntu
- Create folder on Window
- Start Virtual Machine Ubuntu
- select `Insert Guest Additions CD image 1/2`
- Select `Device`, and `Shared Folders`,
    - click the `+` button,
    - select the above folder on Windows, remember the Folder Name.
    - check `Auto-mount` and `Make permanent`.
    - Click `OK`.
- Open Terminal, change to your home directory, make a Share directory, and then mount the above folder to this new directory
- `sudo mount –t vboxsf` **Project1** **/home/sean/share**

遇到了mount: wrong fs type, bad option, bad superblock的问题。
网上查找解决方法
* `sudo apt install nfs-common`
* `sudo apt install cifs-utils`
* 在Visual Box上 前两种 我没能解决 `sudo apt-get install virtualbox-guest-utils` 这条命令解决了我的问题



























.
