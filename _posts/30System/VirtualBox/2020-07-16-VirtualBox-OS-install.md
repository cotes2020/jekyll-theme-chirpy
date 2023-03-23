---
title: VirtualBox OS Install
date: 2020-07-16 11:11:11 -0400
categories: [30System]
tags: [Install, Setup, OS, VirtualBox]
math: true
image:
---


# VirtualBox OS Install

- [VirtualBox OS Install](#virtualbox-os-install)
- [win10](#win10)
  - [下载并安装VirtualBox](#下载并安装virtualbox)
  - [下载win10镜像文件(.ios文件)](#下载win10镜像文件ios文件)
  - [新建虚拟机](#新建虚拟机)
  - [安装系统](#安装系统)
- [Fedora](#fedora)
  - [在VirtualBox上安装](#在virtualbox上安装)
- [CentOS install](#centos-install)
  - [on Mac](#on-mac)
  - [on windows](#on-windows)


---

# win10

## 下载并安装VirtualBox
进入VirtualBox官网，选择OS X hosts版本，下载好后双击打开安装即可。

## 下载win10镜像文件(.ios文件)
到官网下载安装包。

## 新建虚拟机
* VirtualBox
* 新建 > `Windows 10(64-bit)`
* 后面连续点击两次继续，到虚拟硬盘界面，
* 选择 > 默认项现在创建虚拟硬盘
* 选择默认项 > `VDI(VirtualBox磁盘映像)`
* 选择默认项动态分配
* 默认 32G的硬盘

## 安装系统
* 创建的虚拟机win10，双击打开
* 选择系统.png
* 选择刚刚下载的win10系统，点击Start，下面就等系统自动安装。


# Fedora
- Fedora 30正式版本发布了，提供64位/32位ISO下载
- Fedora 30 Workstation（专注于提供最新的开源桌面工具，采用GNOME 3.32桌面环境）
- Fedora 30 Server（可用于服务器部署，新功能是Linux系统角色：Ansible执行的角色和模块的集合，用于协助Linux管理员配置常见的GNU/Linux子系统）及各种定制版本，服务器版本还提供aarch64镜像下载。
不管你使用哪种Fedora 30版本，都搭载有GCC 9、Bash 5.0和PHP 7.3等重要的软件包。
同时Fedora Silverblue有Workstation的所有功能，结合了Fedora Atomic的rpm-ostree功能。

## 在VirtualBox上安装
1. 创建一个新VM - Name, Type:`Linux`, Version:`Fedora`
2. memory size: `1024 MB`
3. hard disk: `create a virtual hard disk now`
4. hard disk file type: `VDI`
5. storage on physical hard disk: `dynamically`
5. File location and size: `20 GB`
6. 单击“开始（Start）”以选择安装ISO映像：
7. choose a virtual optical disk file: `Fedora 30 ISO文件` > `Start`
8. 开始Fedora 30安装过程
9. 第一个屏幕将要求你启动Fedora 30 Live安装：`enter`
10. welcome to fedora: `Install to Hard Drive`
11. 选择安装语言>键盘和时区
12. 选择 Installation Destination: 在VM上进行安装，storage configuration: `Automatic`
13. 如果你正在进行高级存储配置，例如swap、/var、/tmp或RAID配置的单独分区，请选择“存储配置”下的“自定义”
14. 完成后，单击屏幕顶部的 完成（Done）
15. 开始安装并等待它完成，这需要一些安装时间，计算机配置好的几分钟就可以完成了
16. 删除安装介质，对于Virtual Environment，分离ISO文件并重新引导系统。
17. 同时设置你的用户信息和用户名：grace0w0, 114114LjyL201*

---


# CentOS install

## on Mac

1. Download CentOS ISO Installation Image

![1-8](https://i.imgur.com/sIGWiPC.png)

2. Create VirtualBox Virtual Machine for CentOS
    - Open VirtualBox, click New.
    - name, Type to Linux, Version to Red Hat (64-bit).
    - *Memory (RAM)*
      - headless servers, 1 GB or 1024 MB is enough.
      - server with graphical user interface, at least 2GB or 2048 MB.
    - *hard disk* : `Create a virtual hard disk`
    - *hard disk file type* : `VDI`
    - *storage* : `Dynamically`
    - *virtual hard disk size* : `20 GB` is enough for more task.
    - A new VM should be created. Now, select the VM and click on Settings.
    - the `Storage` section --> `Empty` in Controller: IDE --> Optical Drive: Choose Virtual Optical Disk File --> select the CentOS 8 ISO installation image and click on Open, OK.
    - The VM should start
3. run the Settings
    - select `Install CentOS Linux 8.0.1905` from the GRUB menu and press <Enter>.
    - Select language.
    - **software selection**: `Server with GUI`.
    - **Installation Destination**: select the virtual hard drive, select `Custom` from `Storage Configuration`.
      - Manual partitioning: clink `to create automatically`.
      - `/boot`: where kernel images are stored.
      - `/`: OS
      - `swap`
    - **Network & Host Name**: Type in a host name and click on Apply. Then, click on Done.
    - to install `CentOS 8 server with graphical user interface`, then you don’t have to do anything else.
    - Begin Installation.
    - *create user*: create `Root Pwd` and `User`, make it as admin.
4. VirtualBox VM may boot from the CentOS 8 Installation DVD again.
    - To avoid that, open the VM, click on `Devices` > `Optical Drives` > `Remove disk from virtual drive`.
    - Click on Force Unmount.
    - Now, click on Machine > Reset to reset the VM.
    - Click on `Reset` to confirm the action.
    - Now, the VM should boot from the virtual hard drive.
    - Once CentOS 8 boots, you can login using the username and password that you’ve set during the installation.
4. reboot
5. license.
    - `I accept` --> done
    - login, lenguage, keyboard, skip online account.
6. reboot.




**network**

open network.




---

## on windows

https://blog.csdn.net/rockage/article/details/90374771?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522158610562119195162518197%2522%252C%2522scm%2522%253A%252220140713.130056874..%2522%257D&request_id=158610562119195162518197&biz_id=0&utm_source=distribute.pc_search_result.none-task-blog-blog_SOOPENSEARCH-7

https://blog.csdn.net/qq_39723600/article/details/83028819?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522158610562119195162518197%2522%252C%2522scm%2522%253A%252220140713.130056874..%2522%257D&request_id=158610562119195162518197&biz_id=0&utm_source=distribute.pc_search_result.none-task-blog-blog_SOOPENSEARCH-11

https://www.bilibili.com/read/cv2480151/

Win10、CentOS 7双系统的经历
https://www.cnblogs.com/xiaoyao-lxy/p/5561728.html
