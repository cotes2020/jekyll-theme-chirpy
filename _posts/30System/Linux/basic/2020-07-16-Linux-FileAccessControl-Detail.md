---
title: Linux - File Access Control - Detail
date: 2020-07-16 11:11:11 -0400
categories: [30System, Basic]
tags: [Linux, Sysadmin]
math: true
image:
---


# file access control

[toc]

## Permissions 更改文件属性
The Unix-like operating systems, differ from other computing systems in that they are not only multitasking but also `multi-user`.
- more than one user can be operating the computer at the same time.

example
computer is attached to a network/Internet
- remote users can log in via `ssh (secure shell)` and operate the computer.
- In fact, remote users can execute graphical applications and have the output displayed on a remote computer. The X Window system supports this.

The `multi-user` capability of Unix-like systems is a feature that is deeply ingrained into the design of the operating system. If you remember the environment in which Unix was created, this makes perfect sense. Years ago before computers were "personal," they were large, expensive, and centralized. A typical university computer system consisted of a large mainframe computer located in some building on campus and terminals were located throughout the campus, each connected to the large central computer. The computer would support many users at the same time.

In order to make this practical, a method had to be devised to protect the users from each other. After all, you could not allow the actions of one user to crash the computer, nor could you allow one user to interfere with the files belonging to another user.

## File Permissions
![file_permissions](https://i.imgur.com/vawBx6h.png)

## Directory Permissions
The chmod command can also be used to control the access permissions for directories. Again, we can use the octal notation to set permissions, but the meaning of the r, w, and x attributes is different:

r - Allows the contents of the directory to be listed if the x attribute is also set.
w - Allows files within the directory to be created, deleted, or renamed if the x attribute is also set.
x - Allows a directory to be entered (i.e. cd dir).

---

## change commands

### `chmod` 数字/符号。
It is easy to think of the permission settings as a series of bits

```
rwx rwx rwx = 111 111 111
rw- rw- rw- = 110 110 110
rwx --- --- = 111 000 000

rwx = 111 in binary = 7
rw- = 110 in binary = 6
r-x = 101 in binary = 5
r-- = 100 in binary = 4
```

```py

One advantage using the symbolic mode
it let you add or subtract permissions.

-----------------------------------------------
- 数字基本权 Numeric mode
  - 分别是 owner/group/others
  - 三种身份各有自己的read/write/execute权限。
  - r:4
  - w:2
  - x:1
  - rwx = 4+2+1 = 7

- chmod [-R] xyz 文件或目录
  - xyz : 数字类型的权限属性，为 rwx 属性数值的相加。
  - -R : 进行递归(recursive)的持续变更，亦即连同此目录下的所有文件都会变更

# 将.bashrc这个文件所有的权限都设定启用
$ chmod 777 .bashrc
# 将权限变成 -rwxr-xr--
# 权限: [4+2+1][4+0+1][4+0+0]=754。
-----------------------------------------------

- 符号权限 symbolic mode
  - (1)user (2)group (3)others
  - u, g, o 代表三种身份的权限
  - a 代表 all 亦即全部的身份
  - 权限: r, w, x
  - +/-/=

$ chmod	u=rwx g+rwx o=rwx / a=rwx 文件或目录

#
$ chmod u=rwx g+rw o= file.txt

$ chmod g-w file.txt
$ chmod a+w file.txt

$ chmod ugo-x file.txt

```

---

### `chgrp` 更改文件属组 Changing Group Ownership

The group ownership of a file or directory may be changed with `chgrp`.
You must be the owner of the file or directory to perform a chgrp.

```py
changed the group ownership of some_file from its previous group to "new_group".
$ chgrp new_group some_file

- chgrp [-R] 属组名文件名
  - -R：递归更改文件属组, 在更改某个目录文件的属组时，如果加上-R的参数，那么该目录下的所有文件的属组都会更改
```

---

### `chown` 更改文件属主

也可同时更改文件属组 Changing File Ownership

`chown [options] <user>:<group> <file>`

```py
#change the owner of a file

- chown [–R] 属主名 文件名
- chown [-R] 属主名：属组名 文件名
  - -R: ensure all files in the folder will change.

# 将install.log的owner改为 bin：
$ sudo chown bin install.log

# 将install.log的拥有者与群组改为root：
$ sudo chown root:root install.log
```

to change the owner of a file, you must be the root or with sudo.

chown works the same way on directories as it does on files.


---

## Becoming the Superuser for a Short While `su`
- It is often necessary to become the superuser to perform important system administration tasks, but as you have been warned, `you should not stay logged in as the superuser`.
- `su` (substitute user) give you temporary access to the superuser's privileges. can be used in those cases when you need to be the superuser for a small number of tasks.

```py
# To become the superuser
$ su
Password:
[root@linuxbox me]$
# type exit and you will return to your previous session.
```

In some distributions, most notably Ubuntu, an alternate method is used. Rather than using su, these systems employ the `sudo` command instead. With sudo, one or more users are granted superuser privileges on an as needed basis.

To execute a command as the superuser, the desired command is simply preceded with the sudo command. After the command is entered, the user is prompted for the user's password rather than the superuser's:

```py
    [me@linuxbox me]$ sudo some_command
    Password:
    [me@linuxbox me]$
```


---

## Default permissions using umask

When files are created, default permissions are applied automatically. These permissions are calculated based on a `bitmask` called `umask`.


  777 rwxrwxrwx : maximum initial permissions for directory
  666 rw-rw-rw- : maximum initial permissions for file
- 002             umask
  775             default diractory Permissions
  664             default file permissions

```Shell

# To see umask
$ umask
000002

$ unmask -S
u=rwx g=rwx o=rx


# CentOS has different umasks for root and regular users
[bob] $ su - root
[root]$ umask
0022
[root]$ exit
logout
[bob] $


# to change unmask

1. change unmask only works for our current login session.
    $ umask 022


2. user wants to change their umask, add it to their Bash startup file.
    # use VI
    $ vi ~/.basherc
    # Go into insert mode by pressing the Insert key,
    umask 0022,
    # save this by pressing Escape, :wq


3. administrator change a system-wide umask,, add a file to /etc/profile.d.
    $ sudo vi /etc/profile.d/umask.sh
    # for all user
    umask 022
    # have a different umask for root and regular users
    # add a condition.
    if [ "$UID" -ge 1000 ] ; then
            umask 022
    fi
    # :wq

    # check the user ID of the currently logged in user.
    # if greater than or equal to 1,000, change the umask to 022.
    # This wont take effect until you log in again.

```

---

## SUID, SGID, sticky

SUID and SGID
- special bits for `privilege escalation` on executable files.
- `SUID`:  allows non-user owners to execute commands with the privileges of the user owner.

---

### `SUID`

`-rwsr-xr-x. 1 root root 32072 Aug 2 2016 /usr/bin/su`

s: means the SUID bit set.
  - lowercase, then the execute is also set.
  - uppercase, then the execute permissions are not set.
  - The case of the "S" is the only way to tell if execute permissions are set or not.

The permissions for the user owner: "rws", no longer see the owners execute position.

When the SUID bit is set
- regular users such as Bob executes it, their privileges get elevated to that of the user owner's.
- Bob would be executing the "su" command as root user.
- his command would have the power of the root user.

```py

To set the SUID bit

- utilize an extra column of numeric permissions.
- First, underlying permission on the file: "rwx", "rx", "rx", or 755 permissions.
- then decide what extra bits we want. SUID=4.

# Setting these permissions via numeric mode
$ chmod 4755 /usr/bin/su
# use symbolic method as well,
$ chmod space u+s /usr/bin/su
```

---

### `SGID`

The SGID bit is very similar to SUID.
- s in the execute position of the group owner's permissions.
- When a regular user executes a command with SGID bit set, it runs with the privileges of the group owner of the file.
- In this case, the "screen" group.

Setting a directory's permissions to 2755 in numeric mode would enable inheritance of the group owner of this directory by any file created inside of it .

-rwxr-sr-x. 1 root root 475168 Aug 2 2016 /usr/bin/screen


```py

To add SGID bits to a file

prefixed the permissions with 2, the permissions would 2755

# Setting these permissions via numeric mode
$ chmod 2755 /usr/bin/screen
# set this with symbolic mode
$ chmod g+s /usr/bin/screen
```

allow privilege escalation without prompting for a password, it may be advantageous to know where in the file system they are for security reasons.

```py
$ sudo mkdir accounting
# drwxr-xr-x. 2 root root 6 Feb 12 05:22 accounting
$ sudo groupadd accounting
$ sudo chown :accounting accounting
# drwxr-xr-x. 2 root accounting 6 Feb 12 05:22 accounting

$ sudo chmod 2770 accounting/
# drwxrws---. 2 root accounting 6 Feb 12 05:22 accounting

$ sudo useradd -G accounting bob  # create user
$ sudo passwd bob                 # change password
$ su - bob                        # change to user bob
[bob]$ cd /home/accounting/
[bob]$ touch file.txt    # create a file, owner: bob and accounting group
[bob]$ exit

if bob create a file in accounting directory. it will be owner by bob user and the accounting group.
if sally create a file in accounting directory. it will be owner by sally user and the accounting group.


```

---

use the find command to locate SUID and SGID files.

```py

# searching for permissions that are greater than 4,000
# will find all of the files with the SUID bit set.
$ sudo find / -perm -4000

# find files with the SGID bit set
$ sudo find / -perm -2000
```

---


### sticky

keep user from deleting or moving other's file.

```py

# set sticky bit by add ` to the left of the pwemissions.`
$ sudo mkdir stickydir
$ sudo chmod 1770 stickydir/
# drwxrwxrwt. 2 root root 6 Feb 12 05:22 stickydir
$ cd /home/stickydir/
$ touch file.txt
$ chmod 777 file.txt
# -rwxrwxrwx. 1 grant grant 0 Feb 12 05:22 file.txt


$ su - bob                        # change to user bob
password
[bob]$ cd /home/stickydir/
[bob]$ rm file.txt
# error: file cannot be deleted.


# delete the sticky bits
$ chmod -s /usr/sbin/locate
```

---

# Access Control Lists(ACL) in Linux

Access control list (ACL) provides an additional, more flexible permission mechanism for file systems.

It is designed to assist with `UNIX file permissions`.

ACL allows you to give permissions for any user or group to any disc resource.


when a particular user is not a member of group created by you but still you want to give some read or write access
- ACLs are used to make a flexible permission mechanism in Linux.

From Linux man pages, ACLs are used to define more fine-grained `discretionary access rights` for files and directories.


`setfacl` and `getfacl` are used for setting up ACL and showing ACL respectively.

---


### `getfacl` View ACL

```shell

1. check acl

    $ getfacl test/declarations.h
    #
    # file: test/declarations.h
    # owner: mandeep
    # group: mandeep
    user::rw-
    group::rw-
    other::r--


2. view the data in tabular format

    $ getfacl -t test/declarations.h
    #
    # file: test/declarations.h
    USER  grant    rw-
    GROUP grant    rw-
    OTHER          r--



3. check all file in dir

    $ getfacl -R /home > home-perms.txt
    $ cat home-perms.txt
    # file: home/grant/.ssh
    ...
    # file: home/grant/aclfile
    ...
    # file: home/grant/aclexerciese/home-perms.txt
    ...


```

---

### `setfacl` setting up ACL

![acl1](https://i.imgur.com/YFjYnlw.png)

Observe the difference between output of `getfacl` command before and after setting up ACL permissions using `setfacl` command.
- one extra line added for user mandeep which is highlighted in image above.

set acl of a user to a folder
- drwxr-xr-x
- drwxr-xr-x`+`: extra permissions on that folder.


```py

-m: modify
-R: works for all files in directory
-d Default

1. To add permission

    for user: $ setfacl -m user:username:permissions /filepath

    $ setfacl -m u:mandeep:rwx test/declarations.h
    $ setfacl -m u:mandeep:r-x test/declarations.h
    # set rwx permissions for bob on /home/file.txt
    $ setfacl -m user:bob:rwx /home/file.txt

    # sets the standard permissions for the user owner
    $ setfacl -m u::rwx file.txt
    # equal to
    $ sudo chmod u=rwx file.txt


    for a group: setfacl -m g:group:permissions /filepath

    $ setfacl -m user:username:permissions, g:group:permissions /filepath



2. To allow all files or directories to inherit ACL entries from the directory it is within

    setfacl -dm "entry" /dirpath


3. To remove a specific entry

    setfacl -x "entry" /filepath

```


---

#### `setfacl -b` Remove ACL
If you want to remove the set ACL permissions, use setfacl -b

```py

To remove all entries, remove set permissions

$ setfacl -b /filepath

# result: no particular entry for user mandeep in later output.
```

---

### `ls -ltr` check any extra permissions

check if there are any extra permissions set through ACL

![acl4](https://i.imgur.com/3kRye6s.png)

- extra “+” sign after the permissions like `-rw-rwxr–+`: there are extra ACL permissions, check by `getfacl` command.

---

### `setfacl -d` Using Default ACL :
The default ACL
- a specific type of permission assigned to only *directory*,
- doesn’t change the permissions of the directory itself
- but makes that specified ACLs set by default on all the files created inside.

to create a directory and assign default ACL to it by using the `-d` option:

```py

$ mkdir test && setfacl -d -m u:dummy:rw dir1

$ sudo setfacl -d -m u:dummy:rw dir1
# all file in dir1 in the future will have it

```

---

## delete acl

```py

-x remove specific ACLs
-k remove all default ACLs
-b remove all ACLs


# specific acl
$ setfacl -x u:root acldeldir
$ setfacl -x root acldeldir

# setfacl -x default:acltype:username file
$ setfacl -x default:user:root acldeldir




```













.
