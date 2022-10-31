---
title: Virtulization - Running Kali on Docker
date: 2020-11-26 11:11:11 -0400
categories: [00Basic, VMandContainer, Containers]
tags: [Linux, VMs, Docker]
math: true
image:
---

[toc]

---

# Running Kali on Docker


```bash
# Installing Docker on Linux
$ sudo apt install docker.io


# create folder
# make a folder on our host filesystem that will allow us to pull data out of the running image.
# One of the benefits of docker images, the fact that things are not persistent, can be a drawback if you need to keep data or logs from the work you do within the image.
# to sync a folder in and out of a docker image, create a folder to shuttle information in and out of the image.
$ mkdir ~/Pentest
$ docker run -v ~/Pentest:/Pentest -t -i kalilinux/kali-linux-docker /bin/bash
# -v: does the folder sync.
# -v <host direcoty>:<container directory> : in the container we can copy files in and out of “/Pentest/”.



# Install Kali tools
# it doesn’t come with most of the kali tools pre-installed.
$ apt update
$ apt install metasploit-framework
$ msfconsole


# exit
$ exit
```

![Screen Shot 2020-11-26 at 23.20.59](https://i.imgur.com/llgydDC.png)


.
