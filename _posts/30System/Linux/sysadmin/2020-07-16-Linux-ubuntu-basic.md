---
title: Linux - ubuntu command
date: 2020-07-16 11:11:11 -0400
categories: [30System, Sysadmin]
tags: [Linux, Sysadmin, OnePage]
math: true
image:
---

# Linux - ubuntu command

[TOC]

# network related

##  checked network with lshw

```
sudo lshw -C network

*-network
   description:   Wireless interface
   product:       BCM4313 802.11bgn Wireless Network Adapter
   vendor:        Broadcom Corporation
   physical id:   0
   bus info:      pci@0000:03:00.0
   logical name:  eth2
   version:       01
   serial:        08:3e:8e:a2:91:9f
   width:         64 bits
   clock:         33MHz
   capabilities:  pm msi pciexpress bus_master cap_list ethernet physical wireless
   configuration: broadcast=yes driver=wl0 driverversion=6.20.155.1 (r326264)
                  latency=0 multicast=yes wireless=IEEE 802.11abg
   resources:     irq:17 memory:f2d00000-f2d03fff
```



## firewall
```
~$ sudo ufw status
password:
status: inactive

~$ sudo ufw enable
//激活fw

~$ sudo ufw status
status: active
```



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

### 1.3.1. 目录和文件系统
Linux 和 Unix 文件系统被组织成一个有层次的树形结构。
文件系统的最上层是 /，或称为 根目录。
在根 (/) 目录下，有一组重要的系统目录，在大部分 Linux 发行版里都通用。
直接位于根 (/) 目录下的常见目录列表如下：

* /bin - 重要的二进制 (binary) 应用程序
* /boot - 启动 (boot) 配置文件
* /dev - 设备 (device) 文件
* /etc - 配置文件、启动脚本等 (etc)
* /home - 本地用户主 (home) 目录
* /lib - 系统库 (libraries) 文件
* /lost+found - 在根 (/) 目录下提供一个遗失+查找(lost+found) 系统
* /media - 挂载可移动介质 (media)，诸如 CD、数码相机等
* /mnt - 挂载 (mounted) 文件系统
* /opt - 提供一个供可选的 (optional) 应用程序安装目录
* /proc - 特殊的动态目录，用以维护系统信息和状态，包括当前运行中进程 (processes) 信息。
* /root - root (root) 用户主文件夹，读作“slash-root”
* /sbin - 重要的系统二进制 (system binaries) 文件
* /sys - 系统 (system) 文件
* /tmp - 临时(temporary)文件
* /usr - 包含绝大部分所有用户(users)都能访问的应用程序和文件
* /var - 经常变化的(variable)文件，诸如日志或数据库等

### 1.3.2. 权限
所有文件皆有如下三组权限加以保护，按重要性依次排列：

* 用户(user)：适用于该文件的所有者
* 组(group)：适用于该文件所属的组
* 其他(other)：适用于所有其他用户

上述三组权限信息的每一组都包含了实际权限。这些权限信息，连同它们对文件和目录具有的不同含义，概述如下：

* 读(read)：允许显示/打开该文件 可以显示目录内容
* 写(write)：可以编辑或删除该文件 可以更改目录内容
* 可执行(execute)：可执行文件可以作为程序运行 可以进入该目录

要查看和编辑文件或目录的权限，请打开 位置 → 主文件夹 并在文件或者目录上单击右键。然后选择 属性。授权信息就在 权限 标签页中，如果您是该文件的所有者，您可以编辑所有的授权等级。

## 2 命令行指南
### 2.1 命令模式的基本结构和概念

dud@shadowplay:~ $
or
[dud@shadowplay:~]$

* **dud**：登录的用户，
* **shadowplay**：这台计算机的主机名，
* **~**：表示当前目录

---

2. Explore the *File System* and *Directory Structures* on Linux
* Use the *File tool*.
* Use *commands*:
    * `/bin`- Common binaries.
    * `/sbin` - Binaries used for system administration are placed here.
    * `/boot` - Static files of the boot loader. Usually it contains the Linux kernel, Grub boot loader files and so on.
    * `/dev` - Device files such as your CD drive, hard disk, and any other physical device.
    * `/home` - User **HOME directories** are found here. In UNIX like FreeBSD, the HOME directories are found in `/usr/home`. And in Solaris it is in `/export`.
    * `lib` - Essential shared libraries and kernel modules.
    * `/mnt` - Temporary mount point useful for when you insert your USB stick and it gets mounted under `/mnt`. Though in Ubuntu and the likes, it is usually mounted under `/media`.
    * `/var` - **Variable data**, like logs, news, mail spool files ... which is constantly being modified by various programs running on your system.
    * `/tmp` - **Temporary files** are placed here by default.
    * `/usr` - The secondary hierarchy which contain its own bin and sbin sub-directories.
    * `/etc` - Usually contain the **configuration files for all the programs** that run on your Linux/Unix system.
    * `/opt` - Third party application packages which does not conform to the standard Linux file hierarchy can be installed here.
    * `/srv`- Contains data for **services provided** by the system

#### `/etc/group` user group file
one entry per line, following format:
`ITterm`:`X`:`1008`:`john,david`
group_name:password:GID:user_list

1. **group_name**: the name of the group.
2. **password**: the (encrypted) group password. If this field is empty, no password is needed.
3. **GID**: the numeric group ID.
4. **user_list**: a list of the usernames that are members of this group, separated by commas.

create a user account:
* directly create a user account through editing `/etc/passwd`, `/etc/groups` and `/etc/shadow` files.
    * For security reason, not recommended.
* Add a new user by command: `adduser`
    * `sudo` `adduser` test
    * `sudo` `usermod` –aG `sudo` test (add the test user to the sudo group)
    * `su` – test (switch to test)

* change password: `passwd`
* change other user’s password: `sudo` `passwd` username
* Root password on Ubuntu: by default it has no password and is locked.
    * To create a password for root: `sudo` `passwd` root
    * Then you can login as root: `su` `- root`
    * Enabling the root account is not secure for the system. To dele
    * delete the root password and lock it: `sudo` `passwd` –l root

* Change file permissions
    * `Chmod` nnn file_name,
    * `chmod` u+rw file_name (directory name)

#### `/etc/shadow` shadowed password file

* `/etc/passwd`: the password file, User account information.
* `/etc/passwd-`: Backup file for /etc/passwd.
* `/etc/shadow`: optional encrypted password file
* `/etc/shadow-`: Backup file for /etc/shadow.


`shadow file`: contains the *password information* for the system's accounts and *optional aging information*.
must not be readable by regular users if password security is to be maintained.
Each line 9 fields, separated by colons (“:”):

`sean`:`$6$W2iaMd9m$qEpxyIcb6poeWkSd9Bxy7Wq8OffX8ow#5`...:`17742`:0:99999:7:::`

1. **login name**: valid account name exist on system.
2. **encrypted password**: Refer to crypt(3) for details on how this string is interpreted.
    1. Usually password format: `$`id`$`salt`$`hashed, The `$id` is the algorithm used On GNU/Linux as follows:
        1. $1$ is MD5
        2. $2a$ is Blowfish
        3. $2y$ is Blowfish
        4. $5$ is SHA-256
        5. $6$ is SHA-512
    2. If the password field contains some string that is not a valid result of crypt(3), for instance ! or *, the user will not be able to use a unix password to log in (but the user may log in the system by other means).
    3. empty: then no passwords are required to authenticate the specified login name. However, some applications which read the `/etc/shadow` file may decide not to permit any access at all if the password field is empty.
    4. A password field which starts with a exclamation (!) mark means that the password is locked. The remaining characters on the line represent the password field before the password was locked.
    5. a (*) entry: the account has been disabled.
3. **date of last password change**: since Jan 1,1970.
    1. The value 0 has a special meaning: the user should change her password the next time she will log in the system.
    2. An empty field means that password aging features are disabled.
4. **minimum password age**: the number of days the user have to wait before change her password again.
    1. Empty field / value 0: no minimum password age.
5. **maximum password age**: the number of days after which the user will have to change her password.
    1. After this number of days is elapsed, the password may still be valid. The user should be asked to change her password the next time she will log in.
    2. Empty field: no maximum password age, no password warning period, and no password inactivity period (see below).
    3. If the maximum password age is lower than the minimum password age, the user cannot change her password.
6. **password warning period**: The number of days before a password is going to expire (see the maximum password age above) during which the user should be warned.
    1. Empty field / value 0: no password warning period.
7. **password inactivity period**: The number of days after a password has expired (see the maximum password age above) during which the password should still be accepted (user should update password during the next login).
    1. After expiration of the password and this expiration period is elapsed, no login is possible using the current user's password. The user should contact her administrator.
    2. Empty field: no enforcement of an inactivity period.
8. **account expiration date**: days since Jan 1, 1970.
    1. account expiration vs password expiration:
        1. account expiration: the user shall not be allowed to login.
        2. password expiration: the user is not allowed to login using her password.
    2. Empty field: the account will never expire.
    3. The value 0 should not be used as it is interpreted as either an account with no expiration, or as an expiration on Jan 1, 1970.
9. **reserved field**: reserved for future use.


#### `/etc/passwd`

* `/etc/passwd`: the password file, User account information.
* `/etc/passwd-`: Backup file for /etc/passwd.

Note that this file is used by the tools of the shadow toolsuite, but not by all user and password management tools.

`passwd file` entry looks as follows:

```
sean:x:1000:1008:Sean,,,:/home/sean:/bin/bash
john:x:1001:1001:John,,,:/home/john:/bin/bash

//one entry per line for each user account of the system.
//All fields are separated by a colon (:) symbol.
//Total seven fields.
```

1. **Username**: used when user logs in. between 1 and 32 characters in length.
2. **Password**: An `x` character indicates that encrypted password is stored in `/etc/shadow` file. Please note that you need to use the `passwd` command to *computes the hash of a password typed at the CLI* or *to store/update the hash of the password* in `/etc/shadow` file.
3. **User ID (UID)**: Each user must be assigned a user ID.
    1. UID 0: for root.
    2. UIDs 1-99: for other predefined accounts.
    3. UID 100-999: for system administrative and system accounts/groups.
4. **Group ID (GID)**: The primary group ID (stored in `/etc/group` file)
5. **User ID Info**: The comment field. It allows you to add extra information about the users such as user’s full name, phone number etc. This field use by finger command.
6. **Home directory**: The absolute path to the directory the user will be in when they log in. If this directory does not exists then users directory becomes /
7. **Command/shell**: The absolute path of a command or shell (/bin/bash). Typically, this is a shell. Please note that it does not have to be a shell.

### 2.2 基本用法

在命令操作时系统基本上不会给你什么提示，

`-v` :绝大多数的命令可以加,要求系统给出执行命令的反馈信息；

dud@shadowplay:~ $ mv `-v` file1.txt new_file.txt
**`file1.txt' -> 'new_file.txt'**

`-p`:让系统显示某一项的类型，比如是文件/文件夹/快捷链接等
`-i`:让系统在执行删除操作前输出一条确认提示；i(interactive)也就是交互性的意思；
特别提示：在使用命令操作时，系统假设你很明确自己在做什么，它不会给你太多的提示，比如你执行rm -Rf /，它将会删除你硬盘上所有的东西，并且不会给你任何提示，所以，尽量在使用命令时加上-i的参数，以让系统在执行前进行一次确认，防止你干一些蠢事。如果你觉得每次都要输入-i太麻烦，你可以执行以下的命令，让－i成为默认参数：
alias rm='rm -i'

#### 2.2.1 常用命令

##### `cat` password
`cat shadow`:

##### `cat` 显示文件内容：

dud@shadowplay:~ $ `cat` *file1.txt*
*Roses are red.*
*Violets are blue,*
*and you have the bird-flue!*

//可以用来在终端上显示txt文本文件的内容。
---

##### `cd` 切换目录 (change directory)
`cd -` 进入上次访问的目录 (相当于 back)
`cd..`: 进入上级目录 back to previous folder
`cd folderA`: change path


dud@shadowplay:~ $ pwd
/home/dud
dud@shadowplay:~ $ `cd` *[root]*
*dud@shadowplay:~ /root*$
*dud@shadowplay:~ /root*$ pwd
/home/dud/root

---

##### `chmod` Change access permissions, change mode
Syntax:
       chmod [Options]... Mode [,Mode]... file...
       chmod [Options]... Numeric_Mode file...
       chmod [Options]... --reference=RFile file...
Options
  `-f`, --silent, --quiet   suppress most error messages
  `-v`, --verbose           output a diagnostic for every file processed
  `-c`, --changes           like verbose but report only when a change is made
        --reference=RFile   use RFile's mode instead of MODE values
  `-R`, --recursive         change files and directories recursively
        --help              display help and exit
        --version           output version information and exit

**Mode** can be specified with octal numbers or with letters.

1. Numeric mode: -rwxrwxrwx 421
    * **read** = list files in the directory
    * **write** = add new files to the directory
    * **execute** = access files in the directory

`chmod` *777* filename
`chmod` *777* `*` : any project under the directory.

2. Symbolic Mode:

* letters `ugoa`:  which users' access to the file will be changed.

    * `u`: The user who owns it.
    * `g`: Other users in the file's Group.
    * `0`: Other users not in the file's group.
    * `a`: All users + none of these. = user + group + others.

* The operator:
    * `+`: causes the permissions selected to be added to the existing permissions of each file;
    * `-`: causes them to be removed;
    * `=`: causes them to be the only permissions that the file has.

* The letters `rwxXstugo`: the new permissions for the affected users:
    * `r`: Read
    * `w`: Write
    * `x`: Execute (or access for directories)
    * `X`: Execute only if the file is a directory (or already has execute permission for some user)
    * `s`: Set user or group ID on execution
    * `t`: Restricted deletion flag or sticky bit
    * `u`: The permissions that *the User who owns the file* currently has for it.
    * `g`: The permissions that *other users in the file's Group* have for it.
    * `o`: Permissions that *Other users not in the file's group* have for it.

`chmod` **a-x** file
//Deny execute permission to everyone:
`chmod` **a+r** file
//Allow read permission to everyone:
`chmod` **go+rw** file
//Make a file readable and writable by the group and others:
`chmod` **u+x** myscript.sh
//Make a shell script executable by the user/owner
`chmod` **=rwx,g+s** file
//Allow everyone to read, write, and execute the file and turn on the set group-ID:

---

##### `cp` copy 复制文件/目录:
`cp` *A* **newA**: 拷贝
`cp` */home/xxx/A.txt* **.**: 复制到目前目录
`cp -r`: 可以拷贝您指定的任意目录（注：包括该目录里的文件和子目录）。

dud@shadowplay:~ $ `cp` file1.txt **file1_copy.txt**
dud@shadowplay:~ $ cat **file1_copy.txt**
Roses are red.
Violets are blue,
and you have the bird-flue!

---

##### `df` 显示文件系统空间信息
df -h  用 M 和 G 做单位显示文件系统空间信息 -h 意思是 human-readable

---

##### `du` 显示目录的空间使用信息
`du -sh` directory
-s 意思 summary
-h 意思 human-readable

```
user@users-desktop:~$ du /media/floppy
1032    /media/floppy/files
1036    /media/floppy/

user@users-desktop:~$ du -sh /media/floppy
1.1M    /media/floppy/
```

---

##### `echo` 在屏幕上输出字符

dud@shadowplay:~ $ **echo** *"Hello World"*
*Hello World*


---

##### `gedit` 创建txt

$ `gedit` **c.tet**

---

##### `reinstalled`
sudo apt-get install --reinstall *wicd*


---


##### `ls` 查看目录 (list)

`ls /` 将列出根目录`/`下的文件清单.如果给定一个参数，则命令行会把该参数当作命令行的工作目录。换句话说，命令行不再以当前目录为工作目录。
`ls ~`: will show the files that are in your home directory.
`ls -l`: show file detail
`ls -a`: 将列出包括隐藏文件(以.开头的文件)在内的所有文件.
`ls -h`: 将以KB/MB/GB的形式给出文件大小,而不是以纯粹的Bytes.

```
owner-group-public | owner | group | time | username
---|---|---|---|---
drwx rwx rwx | sean | sean | 10022018 | sean
(755)|
```

##### `locate` 查找文件
查找文件/目录： locate （文件或目录名）


##### `dir`
`mkdir`创建目录：make folder
`rmdir`: remove folder

dud@shadowplay:~ $ ls
...
dud@shadowplay:~ $ `mkdir` **test_dir**
dud@shadowplay:~ $ ls
...
**test_dir**



##### change permission
`chmod` **755** **userA**

```
$ls -l
drwxr-xr-x sean sena 10022018 sena

$ chmod 700 sean

$ls -l
drwx------ sean sena 10022018 sena

```

##### `man` 显示某个命令 (manual)
`mkdir` music: create directories called "music".



##### `mv` 重命名文件/目录 (move)
可以重命名/移动您指定的任意文件或目录。

dud@shadowplay:~ $ ls
file1.txt
file2.txt
dud@shadowplay:~ $ `mv` file1.txt *new_file.txt*
dud@shadowplay:~ $ ls
file2.txt
new_file.txt

##### `passwd` 重命名文件/目录 (move)


##### `ps` 查询当前进程：
`ps`:例出你所启动的所有进程；
`ps -a`:例出系统当前运行的所有进程，包括由其他用户启动的进程；
`ps auxww`是一条相当人性化的命令，它会例出除一些很特殊进程以外的所有进程，并会以一个高可读的形式显示结果，每一个进程都会有较为详细的解释；

dud@shadowplay:~ $ `ps`
PID TTY          TIME CMD
11278 pts/1    00:00:00 bash
24448 pts/1    00:00:00 ps

##### `pwd` 示当前目录 (print working directory)
dud@shadowplay:~ $ `pwd`
/home/dud


##### `rm` 删除文件/目录 (remove)
`rm` * .odt: * all related to .oodt
`rmdir`: delete an empty directory.
`rm -r`:delete a directory and all of its contents recursively.

dud@shadowplay:~ $ `ls -p`
...
tempfile.txt
test_dir/

删除文件:
dud@shadowplay:~ $ `rm` `-i` **tempfile.txt**
rm: remove regular empty file 'tempfile.txt'? **y**
dud@shadowplay:~ $ `ls -p`
...
test_dir/

删除一个文件夹:
dud@shadowplay:~ $ `rm` test_dir
rm: cannot remove 'test_dir': Is a directory
dud@shadowplay:~ $ `rm` `-R` test_dir
dud@shadowplay:~ $ ls -p
...


##### `touch` 建立一个空文本文件：
dud@shadowplay:~ $ `touch` *file1.txt* *file2.txt* *file2.txt*
dud@shadowplay:~ $ ls
```
file1.txt
file2.txt
file3.txt
```

#### `tree` find the directory accoreding to the name

`tree -f  / | grep` *filename*

### 2.2.2 `sudo`

#### Add a new user
$ sudo `adduser` **test**
$ sudo `usermod` `–aG` sudo **test**
(add the test user to the sudo group)
$ su `– test` (switch to test)

#### Add a new group
$ sudo `groupadd` `-g` **Group_ID** **Group_Name**

*$ sudo groupadd -g 11600 CSIS536*
(Example, adds a new group CSIS536 with ID as 11600)

#### Change the primary group of a user
$ sudo `usermod` `-g` **Group_Name** **User_Name**

*$ sudo usermod -g CSIS536 sean*
(Example, change sean’s primary group to CSIS536)

#### change name or id
1.
usermod -u <NEWUID> <LOGIN>
groupmod -g <NEWGID> <GROUP>
find / -user <OLDUID> -exec chown -h <NEWUID> {} \;
find / -group <OLDGID> -exec chgrp -h <NEWGID> {} \;
usermod -g <NEWGID> <LOGIN>

2.
To assign a new UID/GID to user/group called foo
usermod -u 2005 foo
usermod -u 2005 foo
