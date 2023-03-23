---
title: Linux - File Systems - Detail
date: 2020-07-16 11:11:11 -0400
categories: [30System, Basic]
tags: [Linux, Sysadmin]
math: true
image:
---

# file systems

[toc]


## Basics

file system?
- Responsible for managing the files on a device.

What does a file system need to do?
- Keep track of where the file is
- File Length
- Ownership
- File name
- File path
- File type (block, character, directory, etc.)

---

**File System Types**

use df -T to determine mounted file systems and their types.

![filesystem-distribution](https://i.imgur.com/H8UlBAu.png)

**ext2**
**ext3 – ext2 w/ Journaling**
**ext4**
different generations of the standard Linux filesystem.

**reiserfs**
a filesystem developed by Microsoft for their Windows 8 Server.

**FAT/FAT32/NTFS** Windows supports

*NTFS, NT File System*, the most modern file system. Windows uses NTFS for its system drive and, by default, for most non-removable drives.
- NTFS is packed with modern features which are not available on FAT32 and exFAT: file permissions for security, a change journal that can help quickly recover errors if your computer crashes, shadow copies for backups, encryption, disk quota limits, hard links, and various other features.
- Compatibility : Works with all versions of Windows, but read-only with Mac by default, and may be read-only by default with some Linux distributions.
- Limitations : No realistic file-size or partition size limits.
- Ideal Usage : Windows system drive and other internal drives that will just be used with Windows.

*FAT32, File Allocation Table*, is an older file system that’s not as efficient as NTFS and doesn’t support as big a feature set, but does offer greater compatibility with other operating systems.
- Compatibility : Works with all versions of Windows, Mac, Linux, game consoles, and practically anything with a USB port.
- Limitations : 4 GB maximum file size, 8 TB maximum partition size.
- Ideal Usage : Use it on removable drives where you need maximum compatibility with the widest range of devices,assuming you don’t have any files 4 GB or larger in size.


*exFAT,Extended File Allocation Table*,  modern replacement for FAT32 and more devices and operating systems support it than NTFS but it’s not nearly as widespread as FAT32.
- Compatibility : Works with all versions of Windows and modern versions of Mac OS X, but requires additional software on Linux. More devices support exFAT than support NTFS.
- Limitations : No realistic file-size or partition-size limits.
- Ideal Usage : Use it when you need bigger file size and partition limits than FAT32 offers and when you need more compatibility than NTFS offers.

**ISO 9660**

**HFS**
- Mac OS Standard filesystem, used on Apple computers. HFS is obsolete and it is being phased out by Apple. Starting with Mac OS X 10.6 (Snow Leopard), Mac OS does not format disks to HFS, and existing HFS disks are read only.
**HFSPlus**
- Mac OS Extended filesystem, the default filesystem on Apple computers and other complex devices like iPod.


---

## `fdisk [必要参数][选择参数]`

```py

必要参数：
-l 列出素所有分区表
-u 与"-l"搭配使用，显示分区数目


选择参数：
-s<分区编号> 指定分区
-v 版本信息


菜单操作说明
m ：显示菜单和帮助信息
a ：活动分区标记/引导分区
d ：删除分区
l ：显示分区类型
n ：新建分区
p ：显示分区信息
q ：退出不保存
t ：设置分区号
v ：进行分区检查
w ：保存修改
x ：扩展应用，高级功能
```


```c
# fdisk -l

Disk /dev/sda: 10.7 GB, 10737418240 bytes
255 heads, 63 sectors/track, 1305 cylinders
Units = cylinders of 16065 * 512 = 8225280 bytes

Device Boot   Start     End   Blocks  Id System
/dev/sda1  *      1     13   104391   83 Linux
/dev/sda2       14    1305  10377990  8e Linux LVM

Disk /dev/sdb: 5368 MB, 5368709120 bytes
255 heads, 63 sectors/track, 652 cylinders
Units = cylinders of 16065 * 512 = 8225280 bytes

Disk /dev/sdb doesnt contain a valid partition table
```

---

## `mkfs [-V] [-t fstype] [fs-options] filesys [blocks]`

在特定的分区上建立 linux 文件系统

```c
参数 ：

device ： 预备检查的硬盘分区，例如：/dev/sda1
-V : 详细显示模式
-t fstype : 给定 档案系统的型式(ext3, reiserfs, ext2, fat32, msdos)，预设值为 ext2
-c partition: 在制做档案系统前，检查该partition 是否有坏轨
-l bad_blocks_file : 将有坏轨的block资料加到 bad_blocks_file 里面
block : 给定 block 的大小


在 /dev/hda5 上建一个 msdos 的档案系统，同时检查是否有坏轨存在，并且将过程详细列出来 :

    mkfs -V -t msdos -c /dev/hda5


将sda6分区格式化为ext3格式

    mfks -t ext3 /dev/sda6



(first login in as a root user)
create new filesystem with following command
    mkfs.ext3 /dev/sda5

Create mount point directory for the file system
    mkdir /datadisk1

Mount the new file system
    mount /dev/sda5 /datadisk1


make sure file system /dev/hda5 automatically mounted at /datadisk1 mount point after system reboots.

add partition to /etc/fstab file
vi to add following entry

vi /etc/fstab
Add/append following entry to file:
/dev/sda5 /datadisk1 ext3 defaults 0 2

- /dev/sda5 : File system or partition name
- /datadisk1 : Mount point
- ext3 : File system type
- defaults : Mount options (Read man page of mount command for all options)
- 0 : Indicates whether you need to include or exclude this filesystem from dump command backup. Zero means this filesystem does not required dump.
- 2 : It is used by the fsck program to determine the order in which filesystem checks are done at reboot time. The root (/) filesystem should be specified with a #1, and otherfilesystems should have a # 2 value.


```


---

## File System

**Linux 系统目录结构**

```
root@kali:~# ls /
bin         initrd.img.old  media  sbin  vmlinuz
boot        lib             mnt    srv   vmlinuz.old
dev         lib32           opt    sys
etc         lib64           proc   tmp
home        libx32          root   usr
initrd.img  lost+found      run    var
```

- When you first log on to a Linux system, the working directory is set to your home directory. On most systems: `/home/your_user_name`.

directories
1. `/` : root
    - In most cases the root directory only contains subdirectories.

2. `/boot` :
    - where the `Linux kernel` and `boot loader` files are kept.
    - contains the files necessary to get the computer booted up
    - The kernel is a file called `vmlinuz`.
    - 存放的是 *启动Linux* 时使用的一些核心文件，包括一些连接文件以及镜像文件。

- `/bin, /usr/bin` : contain programs that regular users on the system can run.
  - `/bin` Binary: the *essential programs that the system requires* to operate, 存放着最常用的命令。This is where your ls, cd, pwd, echo, and other commands would reside.
  - `/usr/bin` :  *applications* for the system's users. in which most programs are installed. 系统用户使用的应用程序。

- `/sbin, /usr/sbin`:
  - `/sbin`: Super User，存放系统管理员使用的系统管理程序。contain *programs for system administration*, mostly for use by the *superuser*.
  - `/usr/sbin`: 超级用户使用的比较高级的管理程序和系统守护程序。

- `/usr` : contains a variety of things that support user applications. Some highlights:
  - `/usr/share/X11` : Support files for the X Window system
  - `/usr/share/dict` : Dictionaries for the spelling checker.
  - `/usr/share/doc` : Various documentation files in a variety of formats.
  - `/usr/share/man` : The man pages are kept here.
  - `/usr/src` : 内核源代码默认的放置目录。 *Source code files*. If you installed the kernel source code package, you will find the entire Linux kernel source code here.
  - `/usr/local`
    - /usr/local and its subdirectories are used for the *installation of software and other files* for use on the local machine.
    - software that is not part of the official distribution (which usually goes in /usr/bin) goes here.
    - When you find interesting programs to install on your system, they should be installed in one of the /usr/local directories. Most often, the directory of choice is /usr/local/bin.

- `/root` : the root user's home directory.


3. `/dev` : 存放的是Linux的外部设备，在Linux中访问设备的方式和访问文件的方式是相同的。
    - In Linux devices are treated like files. read and write devices as were files.
    - location where all devices are reference outlined_flag from.
        - This includes hard drives, keyboards, USB devices, sound cards and anything else that connects to the computer
    - For example
    - `/dev/fd0` is the first floppy disk drive,
    - `/dev/sda` (/dev/hda on older systems) is the first hard drive.
    - All the devices that the kernel understands are represented here.

4. `/etc` : the *configuration files* for the system. 所有的系统管理所需要的配置文件和子目录。
    - All of the files in /etc should be text files. Points of interest:
    - `/etc/passwd` : contains the essential information for each user. It is here that users are defined.
    - `/etc/fstab` : contains a table of devices that get mounted when your system boots. This file defines your disk drives.
    - `/etc/hosts` : lists the network host names and IP addresses that are intrinsically known to the system.
    - `/etc/init.d` : contains the scripts that start various system services typically at boot time.

5. `/home` : where users keep their personal work.
    - In general the only place users are allowed to write files. This keeps things nice and clean
    - 用户的主目录，在Linux中，每个用户都有一个自己的目录，一般该目录名是以用户的账号命名的。


6. `/lib` : 存放着系统最基本的 *动态连接共享库* ，类似于Win的DLL文件。
    - 几乎所有的应用程序都需要用到这些共享库。
    - The shared libraries (similar to DLLs in that other operating system) are kept here.
    - contain code that is shared out to many of the applications on our system.
        - broken down into two directories.
        - The live directory for 32 bit library files
        - and the live 64 directory for 64 bit library files.

7. `/media,/mnt` : used in a special way. for mount points.

    - `/mnt`: 系统提供该目录是为了让用户临时挂载别的文件系统的，我们可以将光驱挂载在/mnt/上，然后进入该目录就可以查看光驱里的内容了。
        - where you would access other hard drives that you would connect to your system.
        - the different physical storage devices (like hard disk drives) are attached to the file system tree in various places. This process of attaching a device to the tree is called mounting. For a device to be available, it must first be mounted.
        - When your system boots, it reads a list of mounting instructions in the file `/etc/fstab`, describes which device is mounted at which mount point in the directory tree.
        - This takes care of the hard drives, but you may also have devices that are considered temporary, such as CD-ROMs, thumb drives, and floppy disks. Since these are removable, they do not stay mounted all the time.

    - The `/media`:
        - linux系统会自动识别一些设备，例如U盘、光驱等等，当识别后，linux会把识别的设备挂载到这个目录下。
        - devices uch a CD drives can be mounted to, along with USB drives and others
        - used by the automatic device mounting mechanisms found in modern desktop oriented Linux distributions.
        - On systems that require manual mounting of removable devices, the `/mnt` directory provides a convenient place for mounting these temporary devices.
        - You will often see the directories /`mnt/floppy` and `/mnt/cdrom`.

8. `/opt`：: 给主机额外安装软件所摆放的目录。比如你安装一个ORACLE数据库则就可以放到这个目录下。默认是空的。
    - an optional location for applications to be stored if they're not located in the bin directory.


9. `/proc` :
    - entirely virtual. 虚拟目录，*系统内存的映射* ，可以通过直接访问这个目录来获取系统信息。
    - provides information about a running Linux system. The Linux kernel outputs information that many applications use to this directory.
    - contains little peep holes into the kernel itself. There are a group of numbered entries in this directory that correspond to all the processes running on the system.
    - In addition, there are a number of named entries that permit access to the current configuration of the system. Many of these entries can be viewed.
    - Try viewing `/proc/cpuinfo`. This entry will tell you what the kernel thinks of your CPU.
    - 这个目录的内容不在硬盘上而是在内存里，可以直接修改里面的某些文件
      - 比如
      - 屏蔽主机的ping命令，使别人无法ping你的机器：
      - `echo 1 > /proc/sys/net/ipv4/icmp_echo_ignore_all`


10. `/srv`: 该目录存放一些服务启动之后需要提取的数据。
    - typically used for server applications, such as web servers.


11. `/sys`:
    - contains information about hardware that is on the system.
    - 这是linux2.6内核的一个很大的变化。该目录下安装了2.6内核中新出现的一个文件系统 sysfs 。
    - sysfs文件系统集成了下面3种文件系统的信息：
        - 针对进程信息的proc文件系统、
        - 针对设备的devfs文件系统
        - 针对伪终端的devpts文件系统。
        - 该文件系统是内核设备树的一个直观反映。
        - 当一个内核对象被创建的时候，对应的文件和目录也在内核对象子系统中被创建。

12. `/tmp` :
    - used by applications to store temporary data.
    - 存放一些临时文件。

13. `/var` : 这个目录中存放不断扩充着的东西，习惯将那些经常被修改的目录放在这个目录下。
    - files that change as the system is running.
    - contains files that tend to vary in size
        - such as log files, printer files, and local system email files.
    - So that is the high level overview of the Linux file system.
    - `/var/log` : contains log files. These are updated as the system runs.
        - *view the files in this directory from time to time, to monitor the health of system*.
        - `/var/spool` : used to hold files that are queued for some process, such as mail messages and print jobs. When a user's mail first arrives on the local system (assuming you have local mail), the messages are first stored in /var/spool/mail

- `/lost+found`: 这个目录一般情况下是空的，当系统非法关机后，这里就存放了一些文件。


- `/selinux`: 这个目录是Redhat/CentOS所特有的目录，Selinux是一个安全机制，类似于windows的防火墙，但是这套机制比较复杂，这个目录就是存放selinux相关的文件的。

---

在linux系统中，有几个目录是比较重要的，平时需要注意不要误删除或者随意更改内部文件。
- `/etc`：系统配置文件，如果你更改了该目录下的某个文件可能会导致系统不能启动。
- `/bin, /sbin, /usr/bin, /usr/sbin`: 这是系统预设的执行文件的放置目录: ls 就是在/bin/ls 目录下的。
  - /bin, /usr/bin 是给系统用户使用的指令（除root外的通用户），
  - /sbin, /usr/sbin 则是给root使用的指令。
- `/var`: 非常重要的目录，系统跑程序，每个程序都会有相应日志产生，而这些日志就被记录到这个目录下，具体在/var/log 目录下，另外mail的预设放置也是在这里。

remainds:
- File names in Linux are case sensitive. "File1" and "file1" refer to different files.
- Linux has no concept of a "file extension" like legacy operating systems. You may name files any way you like. However, while Linux itself does not care about file extensions, many application programs do.
- Though Linux supports long file names which may contain embedded spaces and punctuation characters, limit the punctuation characters to period, dash, and underscore. Most importantly, do not embed spaces in file names. If you want to represent spaces between words in a file name, use underscore characters. You will thank yourself later.

--

## File System Organization
- Linux的目录结构为 树状结构 `hierarchical directory structure` ，最顶级的目录为根目录 /。
- they are organized in a tree-like pattern of directories (called folders in other systems), which may contain files and other directories. The first directory in the file system is called the root directory. The root directory contains files and subdirectories, which contain more files and subdirectories and so on and so on.

- 其他目录通过挂载可以将它们添加到树中，通过解除挂载可以移除它们。

- 绝对路径：由根目录 / 写起，例如： /usr/share/doc 这个目录。
- 相对路径：不是由 / 写起，例如由 /usr/share/doc 要到 /usr/share/man 底下时，可以写成: `cd ../man`

- 处理目录的常用命令
  - `ls`: 列出目录
  - `cd`：切换目录
  - `pwd`：显示目前的目录
  - `mkdir`：创建一个新的目录
  - `rmdir`：删除一个空的目录
  - `cp`: 复制文件或目录
  - `rm`: 移除文件或目录
  - `mv`: 移动文件与目录、文件重命名
  - `man [命令]`: 来查看各个命令的使用文档，如 ：man cp。

---

### `cd` (切换目录) Change Directory
- The "." notation refers to the working directory itself
- the ".." notation refers to the working directory's parent directory.

type `cd nothing`, cd will change the working directory to your home directory.
type `cd ~user_name`, cd will change the working directory to the home directory of the specified user.

```c
cd [相对路径或绝对路径]
//使用 mkdir 命令创建w3cschool.cn目录
[root@www ~]# mkdir w3cschool.cn

//使用绝对路径切换到w3cschool.cn目录
[root@www ~]# cd /root/w3cschool.cn/

//使用相对路径切换到w3cschool.cn目录
[root@www ~]# cd ./w3cschool.cn/

//回到自己的家目录, /root 这个目录
[root@www w3cschool.cn]# cd ~

//表示去到目前的上一级目录，亦即是 /root 的上一级目录的意思；
[root@www ~]# cd ..
```
---

### `cp` 复制文件或目录
```py
$ cp [-adfilprsu] source destination
$ cp [options] source1 source2 source3 .... directory

$ cp file1 file2
# Copies the contents of file1 into file2.
# If file2 does not exist, it is created;
# otherwise, file2 is silently overwritten with the contents of file1.

$ cp file1 dir1
# Copy the contents of file1 (into a file named file1) inside of directory dir1.

-a ：相当於 -pdr 的意思 (常用)
-d ：若来源档为连结档的属性(link file)，则复制连结档属性而非文件本身；
-f ：为强制(force)的意思，若目标文件已经存在且无法开启，则移除后再尝试一次；

-i (interactive)：若目标档(destination)已经存在时，在覆盖时会先询问动作的进行
$ cp -i ~/.bashrc /tmp/bashrc
cp: overwrite `/tmp/bashrc`? n
# n不覆盖，y为覆盖

-l ：进行硬式连结(hard link)的连结档创建，而非复制文件本身；
-p ：连同文件的属性一起复制过去，而非使用默认属性(备份常用)；
-s ：复制成为符号连结档 (symbolic link)，亦即『捷径』文件；
-u ：若 destination 比 source 旧才升级 destination ！

-r ：递回持续复制，用於目录的复制行为；(常用)
-R dir1 dir2 : Copy the contents of the directory dir1. If directory dir2 does not exist, it is created. Otherwise, it creates a directory named dir1 within directory dir2.
```
---

### `file` to determine data a file contains before view it.
`file` file.txt : examine a file and tell you what kind of file it is.

The file program can recognize most types of files

File Type	| Description	| Viewable as text?
---|---|---
ASCII text                      | The name says it all                                             | yes
Bourne-Again shell script text  | A bash script                                                    | yes
ELF 32-bit LSB core file        | A core dump file (a program will create this when it crashes)    | no
ELF 32-bit LSB executable       | An executable binary program                                     | no
ELF 32-bit LSB shared object    | A shared library                                                 | no
GNU tar archive                 | A tape archive file. A common way of storing groups of files.    | no, use tar tvf to view listing.
gzip compressed data            | An archive compressed with gzip                                  | no
HTML document text              | A web page                                                       | yes
JPEG image data                 | A compressed JPEG image                                          | no
PostScript document text        | A PostScript file                                                | yes
RPM                             | A Red Hat Package Manager archive                                | no, use rpm -q to examine contents.
Zip archive data                | An archive compressed with zip                                   | no

---

### `less` view text files.
This is very handy since many of the files used to control and configure Linux are human readable.

`$ less text_file` : will display the file.

The gzip package includes a special version of less called `zless` that will display the contents of gzip-compressed text files.

Command	| Action
---|---
Page Up or b       | Scroll back one page
Page Down or space | Scroll forward one page
G             | Go to the end of the text file
1G            | Go to the beginning of the text file
/characters   | Search forward in the text file for an occurrence of the specified characters
n             | Repeat the previous search
h             | Display a complete list less commands and options
q             | Quit

---

### `ls` 列出目录 list
list files and directories


```py
$ ls [-aAdfFhilnrRSt] 目录名称
$ ls [--color={never,auto,always}] 目录名称
$ ls [--full-time] 目录名称
$ ls -la ..
# List all files (even hidden) in the parent of the working directory in long format

-a ：全部的文件，连同隐藏档( 开头为 . 的文件) 一起列出来(常用)
-d ：仅列出目录本身，而不是列出目录内的文件数据(常用)
-l ：长数据串列出，包含文件的属性与权限等等数据 List the files in the working directory in long format
-t: 按照顺序字母排列

$ ls -la ..
-rw-------   1 bshotts  bshotts       576 Apr 17  1998 weather.txt
drwxr-xr-x   6 bshotts  bshotts      1024 Oct  9  1999 web_page
-rw-rw-r--   1 bshotts  bshotts    276480 Feb 11 20:41 web_site.tar
-rw-------   1 bshotts  bshotts      5743 Dec 16  1998 xmas_file.txt
----------     -------  -------  -------- ------------ -------------
    |             |        |         |         |             |
    |             |        |         |         |         File Name
    |             |        |         |         Modification Time
    |             |        |        Size (in bytes)
    |             |       Group
    |            Owner
File Permissions


$ ls -l
lrwxrwxrwx     25 Jul  3 16:42 System.map -> /boot/System.map-2.0.36-3
-rw-r--r-- 105911 Oct 13  1998 System.map-2.0.36-0.7
-rw-r--r-- 105935 Dec 29  1998 System.map-2.0.36-3
-rw-r--r-- 181986 Dec 11  1999 initrd-2.0.36-0.7.img
-rw-r--r-- 182001 Dec 11  1999 initrd-2.0.36.img

lrwxrwxrwx     26 Jul  3 16:42 module-info -> /boot/module-info-2.0.36-3
-rw-r--r--  11773 Oct 13  1998 module-info-2.0.36-0.7
-rw-r--r--  11773 Dec 29  1998 module-info-2.0.36-3

lrwxrwxrwx     16 Dec 11  1999 vmlinuz -> vmlinuz-2.0.36-3
-rw-r--r-- 454325 Oct 13  1998 vmlinuz-2.0.36-0.7
-rw-r--r-- 454434 Dec 29  1998 vmlinuz-2.0.36-3

the strange notation after the file names
These three files are called "symbolic links". special type of file that points to another file.
With symbolic links, it is possible for a single file to have multiple names.

Heres how it works: Whenever the system is given a file name that is a symbolic link, it transparently maps it to the file it is pointing to.

This system has had multiple versions of the Linux kernel installed. "vmlinuz-2.0.36-0.7 and vmlinuz-2.0.36-3". both version 2.0.36-0.7 and 2.0.36-3 are installed. B
ecause the file names contain the version it is easy to see the differences in the directory listing. However, this would be confusing to programs that rely on a fixed name for the kernel file.
These programs might expect the kernel to simply be called "vmlinuz". Here is where the beauty of the symbolic link comes in. By creating a symbolic link called vmlinuz that points to vmlinuz-2.0.36-3, we have solved the problem.
To create symbolic links, use the ln command.
```
---

### `mkdir` 创建新目录 make directory
```py
mkdir [-mp] 目录名称
------------------------------------------------------------------------
-m ：直接配置文件的权限！不需要看默认权限 (umask) 的脸色～
# 创建权限为rwx--x--x的目录
# 如果没有加上 -m 来强制配置属性，系统会使用默认属性。
$ mkdir -m 711 test2
$ ls -l
drwxr-xr-x  3 root  root 4096 Jul 18 12:50 test
drwxr-xr-x  3 root  root 4096 Jul 18 12:53 test1
drwx--x--x  2 root  root 4096 Jul 18 12:54 test2
------------------------------------------------------------------------
-p ：帮助你直接将所需要的目录(包含上一级目录)递回创建起来
$ mkdir test1/test2/test3/test4
mkdir: cannot create directory `test1/test2/test3/test4`:
No such file or directory
# 没法直接创建此目录啊

# -p 选项，可以自行帮你创建多层目录
$ mkdir -p test1/test2/test3/test4
```
---

### `mv` 移动或修改名称

```py
$ mv [-fiu] source destination
$ mv [options] file1 file2 file3 dir1
# The files file1, file2, file3 are moved to directory dir1.
# If dir1 does not exist, mv will exit with an error.

$ mv file1 file2
# If file2 does not exist, then file1 is renamed file2. If file2 exists, its contents are silently replaced with the contents of file1.

-f ：force 强制的意思，如果目标文件已经存在，不会询问而直接覆盖；
-i (interactive) ：若目标文件 (destination) 已经存在时，就会询问是否覆盖！
-u ：若目标文件已经存在，且 source 比较新，才会升级 (update)

//复制一文件，创建一目录，将文件移动到目录中
[root@www ~]# cd /tmp
[root@www tmp]# cp ~/.bashrc bashrc
[root@www tmp]# mkdir mvtest
//移动
[root@www tmp]# mv bashrc mvtest
//将刚刚的目录名称更名为 mvtest2
[root@www tmp]# mv mvtest mvtest2
```
---

### `pwd` 显示目前所在目录 Print Working Directory
- The directory you are standing in is called the working directory. To find the name of the working directory, use the pwd command.

```c
//单纯显示出目前的工作目录：
[root@www ~]# pwd
/root

[root@www ~]# pwd [-P]
-P: 显示出确实的路径，而非使用连结 (link) 路径。

//显示出实际的工作目录，而非连结档本身的目录名而已
[root@www ~]# cd /var/mail   //注意，/var/mail是一个连结档
[root@www mail]# pwd
/var/mail         //列出目前的工作目录
[root@www mail]# pwd -P
/var/spool/mail   //加 -P 差很多

[root@www mail]# ls -ld /var/mail
lrwxrwxrwx 1 root root 10 Sep  4 17:54 /var/mail -> spool/mail
//因为 /var/mail 是连结档，连结到 /var/spool/mail
//所以，加上 pwd -P 的选项后，会不以连结档的数据显示，而是显示正确的完整路径啊！
```
---

### `rmdir` (删除空的目录)

```c
rmdir [-p] 目录名称
-p ：连同上一级『空的』目录也一起删除

//删除 w3cschool.cn 目录
[root@www tmp]# rmdir w3cschool.cn/

//将於mkdir范例中创建的目录(/tmp底下)删除掉！

[root@www tmp]# ls -l
drwxr-xr-x  3 root  root 4096 Jul 18 12:50 test
drwxr-xr-x  3 root  root 4096 Jul 18 12:53 test1
drwx--x--x  2 root  root 4096 Jul 18 12:54 test2
[root@www tmp]# rmdir test  //可直接删除掉，没问题
[root@www tmp]# rmdir test1 //因为尚有内容，所以无法删除
rmdir: `test1`: Directory not empty

//利用 -p 这个选项，立刻就可以将 test1/test2/test3/test4 一次删除。
[root@www tmp]# rmdir -p test1/test2/test3/test4
[root@www tmp]# ls -l
drwx--x--x  2 root  root 4096 Jul 18 12:54 test2
//要注意的是，这个 rmdir 仅能删除空的目录，你可以使用 rm 命令来删除非空目录。
```
---

### `rm` 移除文件或目录
```py
rm [-fir] 文件或目录
-f ：就是 force 的意思，忽略不存在的文件，不会出现警告信息；
-i (interactive)：互动模式，在删除前会询问使用者是否动作, 避免删除到错误的档名！

-r ：递回删除啊！最常用在目录的删除了！非常危险的选项！
$ rm -r dir1 dir2
# dir1 and dir2 are deleted along with all of their contents.
```

---


## Linux 文件内容查看
- 查看文件的内容：
  - `cat ` 由第一行开始显示文件内容
  - `tac ` 从最后一行开始显示，可以看出 tac 是 cat 的倒着写！
  - `nl  ` 显示的时候，顺道输出行号！
  - `more` 一页一页的显示文件内容
  - `less` 与 more 类似，但是比 more 更好的是，他可以往前翻页！
  - `head` 只看头几行
  - `tail` 只看尾巴几行

---

### 1. `cat` 由第一行开始显示文件内容

```c
cat [-AbEnTv]
-A ：相当於 -vET 的整合选项，可列出一些特殊字符而不是空白而已；
-b ：列出行号，仅针对非空白行做行号显示，空白行不标行号！   //= nl
-E ：将结尾的断行字节 $ 显示出来；
-n ：列印出行号，连同空白行也会有行号，与 -b 的选项不同；
-T ：将 [tab] 按键以 ^I 显示出来；
-v ：列出一些看不出来的特殊字符

//检看 /etc/issue 这个文件的内容：
[root@www ~]# cat /etc/issue
CentOS release 6.4 (Final)
Kernel \r on an \m
```

---

### 2. `tac` 与cat相反，文件内容从最后一行开始显示
- tac 是 cat 的倒着写！如：

```
[root@www ~]# tac /etc/issue
Kernel \r on an \m
CentOS release 6.4 (Final)
```

### 3. `nl` 显示行号

```c
nl [-bnw] 文件
-b ：指定行号指定的方式，主要有两种：
-b a ：表示不论是否为空行，也同样列出行号(类似 cat -n)；
-b t ：如果有空行，空的那一行不要列出行号(默认值)；
-n ：列出行号表示的方法，主要有三种：
-n ln ：行号在萤幕的最左方显示；
-n rn ：行号在自己栏位的最右方显示，且不加 0 ；
-n rz ：行号在自己栏位的最右方显示，且加 0 ；
-w ：行号栏位的占用的位数。

//用 nl 列出 /etc/issue 的内容
[root@www ~]# nl /etc/issue
     1  CentOS release 6.4 (Final)
     2  Kernel \r on an \m

//left number
root@kali:~# nl -n ln abd
1     	abddd
2     	d

//right number
root@kali:~# nl -n rn abd
     1	abddd
     2	d
root@kali:~# nl -n rz abd
000001	abddd
000002	d
000003	d
```
---

### 4. `more` 一页一页翻动

```c
[root@www ~]# more /etc/man.config
#
# Generated automatically from man.conf.in by the
# configure script.
#
# man.conf from man-1.6d
--More--(28%)  //重点在这一行, 光标会在这里等待你的命令

//有几个按键可以按的：
空白键 (space)：代表向下翻一页；
Enter        ：代表向下翻『一行』；
/字串         ：代表在这个显示的内容当中，向下搜寻『字串』这个关键字；
:f           ：立刻显示出档名以及目前显示的行数；
q            ：代表立刻离开 more ，不再显示该文件内容。
b 或 [ctrl]-b ：代表往回翻页，不过这动作只对文件有用，对管线无用。
```

---

### 5. `less` 一页一页翻动

```
//输出/etc/man.config文件的内容：
[root@www ~]# less /etc/man.config
#
# Generated automatically from man.conf.in by the
# configure script.
#
# man.conf from man-1.6d
....(中间省略)....
:
//这里可以等待你输入命令！
//less运行时可以输入的命令有：

空白键    ：向下翻动一页；
[pagedown]：向下翻动一页；
[pageup]  ：向上翻动一页；
/字串      ：向下搜寻『字串』的功能；
?字串      ：向上搜寻『字串』的功能；
n         ：重复前一个搜寻 (与 / 或 ? 有关)
N         ：反向的重复前一个搜寻 (与 / 或 ? 有关)
q         ：离开 less 这个程序；
```
---

### 5. `head` 取出文件前面几行

```c
head [-n number] 文件
-n ：后面接数字，代表显示几行的意思

[root@www ~]# head /etc/man.config
//默认显示前面 10 行
//若要显示前 20 行:
[root@www ~]# head -n 20 /etc/man.config
```
---

### 6. `tail` 取出文件后面几行

```c
tail [-n number] 文件
-n ：后面接数字，代表显示几行的意思
-f ：表示持续侦测后面所接的档名，要等到按下[ctrl]-c才会结束tail的侦测

[root@www ~]# tail /etc/man.config
//默认显示前面 10 行
//若要显示前 20 行:
[root@www ~]# tail -n 20 /etc/man.config
```

## Linux文件筛选

### `sort` filter the text
- sort是在Linux里非常常用的一个命令，管排序的

1. sort的工作原理
    - sort将文件的每一行作为一个单位，相互比较
    - 比较原则: 从首字符向后，依次按ASCII码值进行比较，最后按升序输出。

```py
------------------------------------------------
$ cat data.txt
banana
apple
pear
orange
------------------------------------------------
1. sort # 将文件的每一行作为一个单位，相互比较
    - 比较原则: 从首字符向后，依次按ASCII码值进行比较，最后按升序输出。

$ sort data.txt
apple
banana
orange
pear
pear
------------------------------------------------
2. sort -u # 在输出行中去除重复行。

$ sort -u data.txt
apple
banana
orange
pear
# pear由于重复被 -u 删除了。
------------------------------------------------
3. sort -r # sort默认的排序方式是升序，-r 改成降序

$ cat number.txt
1
3
2
$ sort number.txt
1
2
3
$ sort -r number.txt
3
2
1
------------------------------------------------
4. sort -o # 将结果写入原文件
- sort默认是把结果输出到标准输出
- 所以需要用重定向才能将结果写入文件: sort filename > newfile。
- 但如果你想把排序结果输出到原文件中，用重定向不行。
- -o: 成功解决了这个问题，让你放心的将结果写入原文件。
- 这或许也是-o比重定向的唯一优势所在。

$ sort -r number.txt > number.txt
$ cat number.txt
$                      # 将number清空了。

$ cat number.txt
1
3
5
2
4
$ sort -r number.txt -o number.txt
$ cat number.txt
5
4
3
2
1
------------------------------------------------
5. sort -n # 以数值来排序
- 10比2小的情况。由于排序程序将这些数字按字符来排序了，1和2，1小，所以就将10放在2前面喽。
- sort -n 要以数值来排序

$ cat number.txt
1
10
2
$ sort number.txt
1
10
2
$ sort -n number.txt
1
2
10
------------------------------------------------
6. sort -t <间隔符号> -k <指定列数>

$ cat facebook.txt
banana:30:5.5
apple:10:2.5
pear:90:2.3
orange:20:3.4

# 这个文件有三列，列与列之间用冒号隔开了
# 第一列表示水果类型，第二列表示水果数量，第三列表示水果价格。
# 以水果数量来排序(以第二列来排序)，如何利用sort实现？
# sort -t: 后面可以设定<间隔符号>（是不是想起了cut和paste的-d选项，共鸣～～）
# sort -k: 指定列数。

$ sort -n -k 2 -t : facebook.txt
apple:10:2.5
orange:20:3.4
banana:30:5.5
pear:90:2.3
# 使用冒号作为间隔符，并针对第二列来进行数值升序排序
------------------------------------------------
7. -k 选项的具体语法格式
- -k选项的语法格式：
[ FStart [ .CStart ] ] [ Modifier ] [ , [ FEnd [ .CEnd ] ][ Modifier ] ]

- 这个语法格式可以被其中的逗号（“，”）分为两大部分，Start部分和End部分。

- 先给你灌输一个思想，那就是“如果不设定End部分，那么就认为End被设定为行尾”。这个概念很重要的，但往往你不会重视它。

- Start部分也由三部分组成，其中的Modifier部分就是我们之前说过的类似n和r的选项部分。我们重点说说Start部分的FStart和C.Start。

- C.Start也是可以省略的，省略的话就表示从本域的开头部分开始。之前例子中的-k 2和-k 3就是省略了C.Start的例子喽。

- FStart.CStart，其中FStart就是表示使用的域，而CStart则表示在FStart域中从第几个字符开始算“排序首字符”。

- 同理，在End部分中，你可以设定FEnd.CEnd，如果你省略.CEnd，则表示结尾到“域尾”，即本域的最后一个字符。或者，如果你将CEnd设定为0(零)，也是表示结尾到“域尾”。
------------------------------------------------
8. 其他的sort常用选项
- `-f`: 会将小写字母都转换为大写字母来进行比较，亦即忽略大小写
- `-c`: 会检查文件是否已排好序，如果乱序，则输出第一个乱序的行的相关信息，最后返回1
- `-C`: 会检查文件是否已排好序，如果乱序，不输出内容，仅返回1
- `-M`: 会以月份来排序，比如JAN小于FEB等等
- `-b`: 会忽略每一行前面的所有空白部分，从第一个可见字符开始比较。

# 第一个域是公司名称，第二个域是公司人数，第三个域是员工平均工资。
$ cat facebook.txt
google 110 5000
baidu 100 5000
guge 50 3000
sohu 100 4500


# 让这个文件按公司(第一个域)的字母顺序排序
$ sort -t ‘ ‘ -k 1 facebook.txt
baidu 100 5000
google 110 5000
guge 50 3000
sohu 100 4500
# 其实此处并不严格，稍后你就会知道 !!!!!!


# 让facebook.txt按照公司人数排序
$ sort -n -t ‘ ‘ -k 2 facebook.txt
guge 50 3000
baidu 100 5000
sohu 100 4500
google 110 5000
# 但是，baidu和sohu的公司人数相同，都是100人
# 按照默认规矩，是从第一个域开始进行升序排序，baidu在sohu前面。


# 想让facebook.txt按照公司人数排序
# 人数相同的按照员工平均工资升序排序：
$ sort -n -t ‘ ‘ -k 2 -k 3 facebook.txt
guge 50 3000
sohu 100 4500
baidu 100 5000
google 110 5000
# 加了一个-k2 -k3就解决了问题。
# sort支持这种设定，就是说设定域排序的优先级，先以第2个域进行排序，如果相同，再以第3个域进行排序。
#（如果你愿意，可以一直这么写下去，设定很多个排序优先级）


# 让facebook.txt按照员工工资降序排序
# 如果员工人数相同的，则按照公司人数升序排序
$ sort -n -t ‘ ‘ -k 3r -k 2 facebook.txt
baidu 100 5000
google 110 5000
sohu 100 4500
guge 50 3000
# 在-k 3后面加上了一个小写字母r。
# r和-r选项的作用是一样的，就是表示逆序
# 因为sort默认按照升序排序，所以加上r表示第三个域按照降序排序。
# 还可以加上n，就表示对这个域进行排序时，要按照数值大小进行排序
$ sort -t ‘ ‘ -k 3nr -k 2n facebook.txt
baidu 100 5000
google 110 5000
sohu 100 4500
guge 50 3000
# 去掉了最前面的-n选项，而是将它加入到了每一个-k选项中了。

# 从公司英文名称的第二个字母开始进行排序：
$ sort -t ‘ ‘ -k 1.2 facebook.txt
baidu 100 5000
sohu 100 4500
google 110 5000
guge 50 3000
# 使用-k 1.2，这就表示对第一个域的第二个字符开始到本域的最后一个字符为止的字符串进行排序。
# baidu因为第二个字母是a而名列榜首
# sohu和 google第二个字符都是o，但sohu的h在google的o前面，所以两者分别排在第二和第三。
# guge只能屈居第四了。


# 只针对公司英文名称的第二个字母进行排序，如果相同的按照员工工资进行降序排序：
$ sort -t ‘ ‘ -k 1.2,1.2 -k 3,3nr facebook.txt
baidu 100 5000
google 110 5000
sohu 100 4500
guge 50 3000
# 由于只对第二个字母进行排序，所以用 -k 1.2,1.2, 表示“只”对第二个字母进行排序。（用 -k 1.2 不行，因为你省略了End部分，这就意味着你将对从第二个字母起到本域最后一个字符为止的字符串进行排序）。
# 对于员工工资进行排序，也使用了-k 3,3，这是最准确的表述，表示“只”对本域进行排序，因为如果省略了后面的3，就变成了“对第3个域开始到最后一个域位置的内容进行排序” 了。
------------------------------------------------
9. 在modifier部分还可以用到哪些选项？
- 可以用到b、d、f、i、n 或 r。
- n: according to the Number
- r: revers
- b: 忽略本域的签到空白符号。
- d: 对本域按照字典顺序排序（即，只考虑空白和字母）。
- f: 对本域忽略大小写进行排序。
- i: 忽略“不可打印字符”，只针对可打印字符进行排序。（有些ASCII就是不可打印字符，比如\a是报警，\b是退格，\n是换行，\r是回车等等）
------------------------------------------------
10. -k和-u联合使用的例子：

$ cat facebook.txt
google 110 5000
baidu 100 5000
guge 50 3000
sohu 100 4500

$ sort -n -k 2 facebook.txt
guge 50 3000
baidu 100 5000
sohu 100 4500
google 110 5000

$ sort -n -k 2 -u facebook.txt
guge 50 3000
baidu 100 5000
google 110 5000

# 当设定以公司员工域进行数值排序，然后加-u后，sohu一行就被删除了！
# 原来-u只识别用-k设定的域，发现相同，就将后续相同的行都删除。
$ sort  -k 1 -u facebook.txt
baidu 100 5000
google 110 5000
guge 50 3000
sohu 100 4500

$ sort  -k 1.1,1.1 -u facebook.txt
baidu 100 5000
google 110 5000
sohu 100 4500
# 开头字符是g的guge就没有幸免于难。

$ sort -n -k 2 -k 3 -u facebook.txt
guge 50 3000
sohu 100 4500
baidu 100 5000
google 110 5000
# 设置了两层排序优先级的情况下，使用-u就没有删除任何行。
# -u是会权衡所有-k选项，将都相同的才会删除，只要其中有一级不同都不会轻易删除的:)（不信，你可以自己加一行sina 100 4500试试看）
------------------------------------------------
11 最诡异的排序：

$ sort -n -k 2.2,3.1 facebook.txt
guge 50 3000
baidu 100 5000
sohu 100 4500
google 110 5000
# 以 <第二个域的第二个字符>到<第三个域的第一个字符>部分进行排序。
# 第一行，会提取0 3，第二行提取00 5，第三行提取00 4，第四行提取10 5
# 又因为sort认为0小于00小于000小于0000….
# 因此0 3肯定是在第一个。10 5肯定是在最后一个。但为什么00 5却在00 4前面呢？（你可以自己做实验思考一下。）
答案揭晓：原来“跨域的设定是个假象”
# sort只会比较第二个域的第二个字符到第二个域的最后一个字符的部分，而不会把第三个域的开头字符纳入比较范围。
# 当发现00和00相同时，sort就会自动比较第一个域去了。当然baidu在sohu前面了。用一个范例即可证实：
$ sort -n -k 2.2,3.1 -k 1,1r facebook.txt
guge 50 3000
sohu 100 4500
baidu 100 5000
google 110 5000
------------------------------------------------
12. sort命令后会看到+1 -2这些符号

On older systems, `sort’ supports an obsolete origin-zero syntax `+POS1 [-POS2]‘ for specifying sort keys.  POSIX 1003.1-2001 (*note Standards conformance::) does not allow this; use `-k’ instead.

原来，这种古老的表示方式已经被淘汰了，以后可以理直气壮的鄙视使用这种表示方法的脚本喽！

（为了防止古老脚本的存在，在这再说一下这种表示方法，加号表示Start部分，减号表示End部分。最最重要的一点是，这种方式方法是从0开始计数的，以前所说的第一个域，在此被表示为第0个域。以前的第2个字符，在此表示为第1个字符。明白？）
------------------------------------------------
- sort command: like `cat`, displays the contents of the file however it sorts the file lexicographically 词典上 by lines (it reorders them alphabetically so that matching ones are together).
- |: is a pipe that redirects the output from one command into another.
- uniq command: reports or omits 省略 repeated lines and by passing it the `-u` argument to report only unique lines.

$ cat data.txt
NN4e37KW2tkIb3dC9ZHyOPdq1FqZwq9h
jpEYciZvDIs6MLPhYoOGWQHNIoQZzE5q
3rpovhi1CyT7RUTunW30goGek5Q5Fu66
JOaWd4uAPii4Jc19AP2McmBNRzBYDAkO
JOaWd4uAPii4Jc19AP2McmBNRzBYDAkO
...
tx7tQ6kgeJnC446CHbiJY7fyRwrwuhrs

$ sort data.txt | uniq -u
```

---
