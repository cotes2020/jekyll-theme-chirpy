---
title: Linux File System Hierachy
author: dyl4n
date: 2023-03-05 11:35:23 +0700
categories: [Operating System]
tags: [linux]
render_with_liquid: false
comments: true
image:
  path: /thumbnails/linux-file-system.png
---

The Linux File Hierarchy Structure or the Filesystem Hierarchy Standard (FHS) defines the directory structure and directory contents in Unix-like operating systems. It is maintained by the Linux Foundation.

- In the FHS, all files and directories appear under the root directory /, even if they are stored on different physical or virtual devices.
- Some of these directories only exist on a particular system if certain subsystems, such as the X Window System, are installed.
- Most of these directories exist in all UNIX operating systems and are generally used in much the same way; however, the descriptions here are those used specifically for the FHS and are not considered authoritative for platforms other than Linux.

![Linux Filesystem Hierarchy](https://user-images.githubusercontent.com/98354414/222957251-fcf371e6-acd0-4335-b918-0ddb10f31386.png)

## /root

- Every single file and directory starts from the root directory.
- Only root user has write privilege under this directory.
- Please note that /root is root user’s home directory, which is not same as /.
  ![root](https://user-images.githubusercontent.com/98354414/222957530-526d2e2b-1597-48cb-b634-ff62dce69838.png)

## /bin

- Contains binary executables.
- Common linux commands you need to use in single-user modes are located under this directory.
- Commands used by all the users of the system are located here.
- For example: ps, ls, ping, grep, cp.
  ![/bin](https://user-images.githubusercontent.com/98354414/222957553-624ef594-cf2a-4331-9697-149f88043e6d.png)

## /sbin

- Just like /bin, /sbin also contains binary executables.
- But, the linux commands located under this directory are used typically by system aministrator, for system maintenance purpose.
- For example: iptables, reboot, fdisk, ifconfig, swapon
  ![/sbin](https://user-images.githubusercontent.com/98354414/222957578-7d1fcfe7-1d69-4632-b05b-9dd4837e89eb.png)

## /etc

- Contains configuration files required by all programs.
- This also contains startup and shutdown shell scripts used to start/stop individual programs.
- For example: /etc/resolv.conf, /etc/logrotate.conf
  ![/etc](https://user-images.githubusercontent.com/98354414/222956984-a4b3400e-24f3-4652-bf35-68be0e665b6a.png)

## /dev

- Contains device files.
- These include terminal devices, usb, or any device attached to the system.
- For example: /dev/tty1, /dev/usbmon0
  ![/dev](https://user-images.githubusercontent.com/98354414/222957038-7e688ad1-44bf-4587-8da2-8db37d6940c2.png)

## /proc

- Contains information about system process.
- This is a pseudo filesystem contains information about running process. For example: /proc/{pid} directory contains information about the process with that particular pid.
- This is a virtual filesystem with text information about system resources. For example: /proc/uptime
  ![/proc](https://user-images.githubusercontent.com/98354414/222957605-508355f8-1aab-4dc1-9b82-9df7b4e298db.png)

## /var

- Content of the files that are expected to grow can be found under this directory.
- This includes — system log files (/var/log); packages and database files (/var/lib); emails (/var/mail); print queues (/var/spool); lock files (/var/lock); temp files needed across reboots (/var/tmp);
  ![/var](https://user-images.githubusercontent.com/98354414/222957733-88121ed2-5562-4e77-9505-e86ac1a421a7.png)

## /tmp

- Directory that contains temporary files created by system and users.
- Files under this directory are deleted when system is rebooted.
  ![/tmp](https://user-images.githubusercontent.com/98354414/222957760-1cc809d7-319e-49f7-931f-0fbfaf1f2c8d.png)

## /usr

- Contains binaries, libraries, documentation, and source-code for second level programs.
- `/usr/bin` contains binary files for user programs. If you can’t find a user binary under **/bin**, look under **/usr/bin**. For example: at, awk, cc, less, scp
- ` /usr/sbin` contains binary files for system administrators. If you can’t find a system binary under **/sbin**, look under **/usr/sbin**. For example: atd, cron, sshd, useradd, userdel
- `/usr/lib` contains libraries for /usr/bin and /usr/sbin
- `/usr/local` contains users programs that you install from source. For example, when you install apache from source, it goes under /usr/local/apache2
- `/usr/src` holds the Linux kernel sources, header-files and documentation.
  ![/usr](https://user-images.githubusercontent.com/98354414/222958023-79a7c87b-4c5e-4b04-a5bb-6d5d2b859c9c.png)

## /home

- Home directories for all users to store their personal files
- For examples: /home/john, /home/dylan.
  ![/home](https://user-images.githubusercontent.com/98354414/222958101-bac84ee1-3fc0-4a59-9699-390103b13ceb.png)

## /boot

- Contains boot loader related files
- **Kernel initrd, vmlinux, grub files** are located under **/boot**
- Kernel initrd, vmlinux, grub files are located under /boot
  ![/boot](https://user-images.githubusercontent.com/98354414/222958200-34315881-b1df-45ea-822e-bd8072743ea2.png)

## /lib

- Contains library files that supports the binaries located under /bin and /sbin
- Library filenames are either ld* or lib*.so.\*
- For example: ld-2.11.1.so, libncurses.so.5.7
  ![/lib](https://user-images.githubusercontent.com/98354414/222958282-646c16bd-e496-4fa6-83fc-fbafe70ae77a.png)

## /opt

- Contains add-on applications from individual vendors.
- add-on applications should be installed under either /opt/ or /opt/ sub-directory.
  ![/opt](https://user-images.githubusercontent.com/98354414/222958331-3b23f566-3352-4f95-85d3-bb30737ce60f.png)

## /mnt

- Temporary mount directory where sysadmins can mount filesystems.
  ![/mnt](https://user-images.githubusercontent.com/98354414/222958375-dca47f69-9f44-4a78-9662-81f84aea1897.png)

## /media

- Temporary mount directory for removable devices.
- For examples, /media/cdrom for CD-ROM; /media/floppy for floppy drives; /media/cdrecorder for CD writer
  ![/media](https://user-images.githubusercontent.com/98354414/222958423-7fb606c3-fcc2-469d-8ef1-0cbc9e89a6bd.png)

## /srv

- Contains server specific services related data.
- For example, /srv/cvs contains CVS related data
  ![/srv](https://user-images.githubusercontent.com/98354414/222958484-1bb51b15-5d5f-4f1e-9d71-c3c3a6723232.png)
