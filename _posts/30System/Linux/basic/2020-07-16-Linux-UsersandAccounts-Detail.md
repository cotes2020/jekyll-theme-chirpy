---
title: Linux - Users and Accounts - Detail
date: 2020-07-16 11:11:11 -0400
categories: [30System, Basic]
tags: [Linux, Sysadmin]
math: true
image:
---

# users and accounts

[toc]

## Linux 用户和用户组管理
- Linux系统是一个多用户多任务的分时操作系统，任何一个要使用系统资源的用户，都必须首先向系统管理员申请一个账号，然后以这个账号的身份进入系统。
- 用户的账号
  - 一方面可以帮助系统管理员对使用系统的用户进行跟踪，并控制他们对系统资源的访问；
  - 另一方面也可以帮助用户组织文件，并为用户提供安全性保护。
- 每个用户账号都拥有一个惟一的用户名和各自的口令。
  - 用户在登录时键入正确的用户名和口令后，就能够进入系统和自己的主目录。

- 用户账号管理：
  - 用户账号的添加、删除与修改。
  - 用户口令的管理。
  - 用户组的管理。

---

## user file

### passwd

```c
// local account
$ cat /etc/passwd
apache:x:48:48:Apache:/usr/share/httpd:/sbin/nologin
server0:x:1000:1000:server0:/home/server0:/bin/bash


$ getent passwd // extra user
$ getent group  // local group and listing domain-based groups

// name services switch file
$ grep passwd /etc/nsswitch.conf
#     passwd: sss files
#     passwd: files
#     passwd: sss files # from profile
passwd:     sss files systemd
# passwd:    db files

```

---

## secure user

### creating a user
- useradd greg
- passwd greg
- mkdir /home/greg
- chown greg:greg /home/greg
- vipw – modify default shell to what ever
- /etc/adduser.conf – modify default parameters … or … specify different flags on useradd


```py

$ sudo cat /etc/passwd

root:x:0:0:root:/root:/bin/bash
ubuntu:x:1000:1000:Ubuntu:/home/ubuntu:/bin/bash
bob:x:1001:1001::/home/bob:
lisa:x:1002:1002::/home/lisa:


# edit the etc/passwd file
$ sudo vipw

```

---

## delete user

```py

1. delete user

sudo usrdel lisa

# remove
/etc/passwd


2. Then
remove their access for many local databases,
remove any alias files that they have, remove any cron jobs
remove the users home directory or their mail queue
remove any other files that they may have had on the system.
# check other file

$ ll /home
# user is still over there, no UID, GID but with secure ID
drwxr-xr-x  5 root   root   4096 Apr 11 21:10 ./
drwxr-xr-x 23 root   root   4096 Apr 12 20:04 ../
drwxr-xr-x 17 bob    bob    4096 Apr 11 22:12 bob/
drwxr-xr-x  2               4096 Apr 11 21:10 lisa/
drwxr-xr-x  7 ubuntu ubuntu 4096 Apr 12 20:38 ubuntu/

# delete it
$ sudo rm -rf lisa
-f: force


3. check lingering files for the user.

$ sudo find / -xdev -nouser

looks for any files out there without any user associated with it.


```





---

## 一、Linux系统用户账号的管理
- 用户账号的添加、修改和删除。
- 添加用户账号就是在系统中创建一个新账号，然后为新账号分配`用户号、用户组、主目录和登录Shell等资源`。刚添加的账号是被锁定的，无法使用。

### `useradd urname` 添加新的用户账号

- `useradd 选项 用户名`
  - `-c comment`: 指定一段注释性描述。
  - `-d 目录`: 指定用户主目录，如果此目录不存在，则同时使用`-m`选项, 创建主目录。
  - `-s Shell文件`: 指定用户的登录Shell。
  - `-g 用户组`: 指定用户所属的用户组。
  - `-G 用户组，用户组`: 指定用户所属的附加组。
  - `-u 用户号`: 指定用户的用户号，如果同时有-o选项，则可以重复使用其他用户的标识号。

```shell

$ sudo useradd testuser
$ cat /etc/passwd     # add one line at the end
# 增加用户账号就是在/etc/passwd 文件中为新用户增加一条记录
# 同时更新其他系统文件如/etc/shadow, /etc/group等。



$ useradd –d /usr/sam -m sam
# 创建用户sam
# -d和-m选项用来为登录名sam产生一个主目录/usr/sam（/usr为默认的用户主目录所在的父目录）

$ useradd -s /bin/sh -g group –G adm,root gem
# 新建用户gem
# 该用户的登录Shell是 /bin/sh
# 它属于group用户组，同时又属于adm和root用户组，其中group用户组是其主组。

# 这里可能新建组：#groupadd group及groupadd adm


Linux提供了集成的系统管理工具userconf，它可以用来对用户账号进行统一管理。
```

useradd functions:


```py
$ cat /etc/default/useradd

# Default values for useradd(8)
#
# The SHELL variable specifies the default login shell on your
# system.
# Similar to DHSELL in adduser. However, we use "sh" here because
# useradd is a low level utility and should be as general
# as possible
SHELL=/bin/sh
#
# The default group for users
# 100=users on Debian systems
# Same as USERS_GID in adduser
# This argument is used when the -n flag is specified.
# The default behavior (when -n and -g are not specified) is to create a
# primary user group with the same name as the user being added to the system.
# GROUP=100
#
# The default home directory. Same as DHOME for adduser
# HOME=/home
#
# The number of days after a password expires until the account
# is permanently disabled
# INACTIVE=-1
#
# The default expire date
# EXPIRE=
#
# The SKEL variable specifies the directory containing "skeletal" user
# files; in other words, files such as a sample .profile that will be
# copied to the new user's home directory when it is created.
# SKEL=/etc/skel
#
# Defines whether the mail spool should be created while
# creating the account
# CREATE_MAIL_SPOOL=yes

```

---

### `userdel 用户名` 删除帐号
- 删除用户账号就是要将/etc/passwd等系统文件中的该用户记录删除，必要时还删除用户的主目录。

- `userdel 选项 用户名`
  - `-r`: 把用户的主目录一起删除。

```c
# userdel sam
//删除用户sam在系统文件中（主要是/etc/passwd, /etc/shadow, /etc/group等）的记录
//同时删除用户的主目录。
```

---

### `usermod 选项 用户名` 修改帐号

更改用户的有关属性，如用户号、主目录、用户组、登录Shell等。

![Screen Shot 2020-04-05 at 21.01.47](https://i.imgur.com/07tJztg.png)

```shell
- `usermod 选项 用户名`
  - 常用的选项包括-c, -d, -m, -g, -G, -s, -u以及-o等，
  - 有些系统可以使用选项：-l 新用户名, 指定一个新的账号，即将原来的用户名改为新的用户名。

$ usermod -s /bin/ksh -d /home/z –g developer sam
# 将用户sam的登录Shell修改为ksh，主目录改为/home/z，用户组改为developer。

```

---

4. 用户口令的管理
- 指定和修改用户口令的Shell命令是passwd。
- 超级用户可以为自己和其他用户指定口令，普通用户只能用它修改自己的口令。

- `passwd 选项 用户名`
  - `-l` 锁定口令，即禁用账号。
  - `-u` 口令解锁。
  - `-d` 使账号无口令。
  - `-f` 强迫用户下次登录时修改口令。
  - 如果默认用户名，则修改当前用户的口令。

```c
//假设当前用户是sam，修改该用户自己的口令：
$ passwd
Old password:******
New password:*******
Re-enter new password:*******

//如果是超级用户，可指定任何用户的口令：
# passwd sam
New password:*******
Re-enter new password:*******

//普通用户修改自己的口令时，passwd命令会先询问原口令，验证后再要求用户输入两遍新口令，如果两次//输入的口令一致，则将这个口令指定给用户
//超级用户为用户指定口令时，就不需要知道原口令。


//为用户指定空口令：
# passwd -d sam
passwd: password expiry information changed.
//将用户sam的口令删除
//用户sam下一次登录时，系统就不再询问口令。

//用-l(lock)选项锁定某一用户，使其不能登录，例如：
# passwd -l sam
```







---

## 二、用户组的管理
- 每个用户都有一个用户组，系统可以对一个用户组中的所有用户进行集中管理。
- 不同Linux 系统对用户组的规定有所不同，如Linux下的用户属于与它同名的用户组，这个用户组在创建用户时同时创建。

- 用户组的管理涉及用户组的添加、删除和修改。
- 组的增加、删除和修改实际上就是对`/etc/group`文件的更新。

### `groupadd testgroup` 增加用户组:

```shell
groupadd 选项 用户组
- -g GID 指定新用户组的组标识号（GID）。
- -o 一般与-g选项同时使用，表示新用户组的GID可以与系统已有用户组的GID相同。

$ groupadd group1
# 增加新组group1
# 新组的组标识号是在当前已有的最大组标识号的基础上加1。

$ groupadd -g 101 group2
# 增加新组group2，同时指定新组的组标识号是101。

$ cat /etc/group    # verify the add
```

---

### `groupdel 用户组` 删除用户组

```c
groupdel 用户组

# groupdel group1
//删除组group1。
```

---

3. 修改用户组的属性: groupmod

```c
groupmod 选项 用户组
-g: GID 为用户组指定新的组标识号。
-o: 与-g选项同时使用，用户组的新GID可以与系统已有用户组的GID相同。
-n 新用户组: 将用户组的名字改为新名字

# groupmod -g 102 group2
此命令将组group2的组标识号修改为102。

# groupmod –g 10000 -n group3 group2
//将组group2的标识号改为10000，组名修改为group3。
```

---

4. 如果一个用户同时属于多个用户组，那么用户可以在用户组之间切换，以便具有其他用户组的权限。
用户可以在登录后，使用命令newgrp切换到其他用户组，这个命令的参数就是目的用户组。

```c
$ newgrp root
//将当前用户切换到root用户组
//前提条件是root用户组确实是该用户的主组或附加组。
```

类似于用户账号的管理，用户组的管理也可以通过集成的系统管理工具来完成。

---

## 三、与用户账号有关的系统文件
- 与用户和用户组相关的信息都存放在一些系统文件中
- 文件包括/etc/passwd, /etc/shadow, /etc/group等。

1. /etc/passwd: 最重要的一个文件。
- 每个用户都在/etc/passwd文件中有一个对应的记录行
- 记录了这个用户的一些基本属性。
- 这个文件对所有用户都是可读的。

```c
＃ cat /etc/passwd

root:x:0:0:Superuser:/:
daemon:x:1:1:System daemons:/etc:
bin:x:2:2:Owner of system commands:/bin:
sys:x:3:3:Owner of system files:/usr/sys:
adm:x:4:4:System accounting:/usr/adm:
uucp:x:5:5:UUCP administrator:/usr/lib/uucp:
auth:x:7:21:Authentication administrator:/tcb/files/auth:
cron:x:9:16:Cron daemon:/usr/spool/cron:
listen:x:37:4:Network daemon:/usr/net/nls:
lp:x:71:18:Printer administrator:/usr/spool/lp:
sam:x:200:50:Sam san:/usr/sam:/bin/sh
```

用户名:口令:用户标识号:组标识号:注释性描述:主目录:登录Shell

  1. `用户名`: 代表用户账号的字符串。不能有冒号(:)，因为冒号在这里是分隔符。最好不要包含点字符(.)，并且不使用连字符(-)和加号(+)打头。

  2. `口令`: 一些系统中，存放着加密后的用户口令字。
    - 存放的只是用户口令的加密串，不是明文，但是由于/etc/passwd文件对所有用户都可读，所以这仍是一个安全隐患。
    - 因此，现在许多Linux 系统（如SVR4）都使用了shadow技术，把真正的加密后的用户口令字存放到/etc/shadow文件中
    - 而在/etc/passwd文件的口令字段中只存放一个特殊的字符，例如`x`或者`*`。

  3. `用户标识号`: 是一个整数，系统内部用它来标识用户。
    - 一般情况下它与用户名是一一对应的。
    - 如果几个用户名对应的用户标识号是一样的，系统内部将把它们视为同一个用户，但是它们可以有不同的口令、不同的主目录以及不同的登录Shell等。
    - 通常用户标识号的取值范围是0～65 535。
    - 0是超级用户root的标识号，1～99由系统保留，作为管理账号
    - 普通用户的标识号从100开始。
    - 在Linux系统中，这个界限是500。

  4. `组标识号`: 用户所属的用户组。对应着/etc/group文件中的一条记录。

  5. `注释性描述`: 记录着用户的一些个人情况。
    - 例如用户的真实姓名、电话、地址等，这个字段并没有什么实际的用途。在不同的Linux 系统中，这个字段的格式并没有统一。
    - 在许多Linux系统中，这个字段存放的是一段任意的注释性描述文字，用做finger命令的输出。

  6. `主目录`: ，也就是用户的起始工作目录。
    - 它是用户在登录到系统之后所处的目录。
    - 在大多数系统中，各用户的主目录都被组织在同一个特定的目录下，而用户主目录的名称就是该用户的登录名。
    - 各用户对自己的主目录有读、写、执行（搜索）权限，其他用户对此目录的访问权限则根据具体情况设置。

  7. 用户登录后，要启动一个进程，负责将用户的操作传给内核，这个进程是用户登录到系统后运行的命令解释器或某个特定的程序，即`Shell`。
    - Shell是用户与Linux系统之间的接口。
    - Linux的Shell有许多种，每种都有不同的特点。
    - 常用的有sh(Bourne Shell), csh(C Shell), ksh(Korn Shell), tcsh(TENEX/TOPS-20 type C Shell), bash(Bourne Again Shell)等。

    - 系统管理员可以根据系统情况和用户习惯为用户指定某个Shell。
    - 不指定Shell，使用sh为默认的登录Shell，即这个字段的值为/bin/sh。

    - 用户的登录Shell也可以指定为某个特定的程序（此程序不是一个命令解释器）。
      - 利用这一特点，我们可以限制用户只能运行指定的应用程序，在该应用程序运行结束后，用户就自动退出了系统。有些Linux 系统要求只有那些在系统中登记了的程序才能出现在这个字段中。

  8. 系统中有一类用户称为伪用户（`psuedo users`）。
    - 用户在/etc/passwd文件中也占有一条记录，但是不能登录，因为它们的登录Shell为空。
    - 它们的存在主要是方便系统管理，满足相应的系统进程对文件属主的要求。

    - 常见的伪用户如下所示：
      - `bin` 拥有可执行的用户命令文件
      - `sys` 拥有系统文件
      - `adm` 拥有帐户文件
      - `uucp` UUCP使用
      - `lp` lp或lpd子系统使用
      - `nobody` NFS使用

---

### 拥有帐户文件

1. 除了上面列出的伪用户外，还有许多标准的伪用户，例如：audit, cron, mail, usenet等，它们也都各自为相关的进程和文件所需要。
- 由于`/etc/passwd`文件是所有用户都可读的，如果用户的密码太简单或规律比较明显的话，一台普通的计算机就能够很容易地将它破解，因此对安全性要求较高的Linux系统都把加密后的口令字分离出来，单独存放在 `/etc/shadow`文件。
- 有超级用户才拥有该文件读权限，保证了用户密码的安全性。

2. /etc/shadow中的记录行与/etc/passwd中的一一对应
  - 它由pwconv命令根据/etc/passwd中的数据自动产生
  - 它的文件格式与/etc/passwd类似:
  `登录名:加密口令:最后一次修改时间:最小时间间隔:最大时间间隔:警告时间:不活动时间:失效时间:标志`

`登录名`: 是与/etc/passwd文件中的登录名相一致的用户账号
`口令`: 字段存放的是加密后的用户口令字，长度为13个字符。如果为空，则用户没有口令，登录时不需要口令；如果含有不属于集合 { ./0-9A-Za-z }中的字符，则对应的用户不能登录。
`最后一次修改时间`: 表示的是从某个时刻起，到用户最后一次修改口令时的天数。时间起点对不同的系统可能不一样。例如在SCO Linux 中，这个时间起点是1970年1月1日。
`最小时间间隔`: 指的是两次修改口令之间所需的最小天数。
`最大时间间隔`: 指的是口令保持有效的最大天数。
`警告时间 expire period`: 从系统开始警告用户到用户密码正式失效之间的天数。
`不活动时间`: 表示的是用户没有登录活动但账号仍能保持有效的最大天数。
`失效时间`: 字段给出的是一个绝对的天数，如果使用了这个字段，那么就给出相应账号的生存期。期满后，该账号就不再是一个合法的账号，也就不能再用来登录了。

```c
// /etc/shadow的一个例子：
＃ cat /etc/shadow

root:Dnakfw28zf38w:8764:0:168:7:::
daemon:*::0:0::::
bin:*::0:0::::
sys:*::0:0::::
adm:*::0:0::::
uucp:*::0:0::::
nuucp:*::0:0::::
auth:*::0:0::::
cron:*::0:0::::
listen:*::0:0::::
lp:*::0:0::::
sam:EkdiSECLWPdSa:9740:0:0::::
```

3. 用户组的所有信息都存放在/etc/group文件中。
- 将用户分组是Linux 系统中对用户进行管理及控制访问权限的一种手段。
- 每个用户都属于某个用户组；一个组中可以有多个用户，一个用户也可以属于不同的组。
- 当一个用户同时是多个组中的成员时，在/etc/passwd文件中记录的是用户所属的`主组`，登录时所属的默认组，其他组称为附加组。
- 用户要访问属于附加组的文件时，必须先使用`newgrp`命令使自己成为所要访问的组中的成员。

用户组的所有信息都存放在/etc/group文件中:`组名:口令:组标识号:组内用户列表`
`组名`: 是用户组的名称，由字母或数字构成。与/etc/passwd中的登录名一样，组名不应重复。
`口令`: 字段存放的是用户组加密后的口令字。一般Linux 系统的用户组都没有口令，即这个字段一般为空，或者是*。
`组标识号`: 与用户标识号类似，也是一个整数，被系统内部用来标识组。
`组内用户列表`: 是属于这个组的所有用户的列表/b]，不同用户之间用逗号(,)分隔。这个用户组可能是用户的主组，也可能是附加组。

```c
// /etc/group文件的一个例子如下：

root::0:root
bin::2:root,bin
sys::3:root,uucp
adm::4:root,adm
daemon::5:root,daemon
lp::7:root,lp
users::20:root,sam
```

## 四. 批量添加用户

1. 编辑一个文本用户文件。
  - 每一列按照/etc/passwd密码文件的格式书写
  - 注意每个用户的用户名、UID、宿主目录都不可以相同
  - 密码栏可以留白或输入x号。

```
// 范例文件user.txt内容如下：
user001::600:100:user:/home/user001:/bin/bash
user002::601:100:user:/home/user002:/bin/bash
user003::602:100:user:/home/user003:/bin/bash
user004::603:100:user:/home/user004:/bin/bash
user005::604:100:user:/home/user005:/bin/bash
user006::605:100:user:/home/user006:/bin/bash
```

2. 以root身份执行命令 `/usr/sbin/newusers`
  - 从刚创建的用户文件user.txt中导入数据, 创建用户：
  - `# newusers user.txt `
  - 然后执行命令 `vipw` 或 `vi /etc/passwd`: 检查 /etc/passwd 文件是否已经出现这些用户的数据，并且用户的宿主目录是否已经创建。

3. 执行命令`/usr/sbin/pwunconv`。
  - 将 /etc/shadow 产生的 shadow 密码解码，然后回写到 /etc/passwd 中，并将/etc/shadow的shadow密码栏删掉。
  - 为了方便下一步的密码转换工作，先取消 shadow password 功能。

`# pwunconv`


4. 编辑每个用户的密码对照文件。

```
//范例文件 passwd.txt 内容如下：
user001:passwordis101
user002:passwordis102
user003:passwordis103
user004:passwordis104
user005:passwordis105
user006:passwordis106
```

5. 以root身份执行命令 `/usr/sbin/chpasswd`。
  - 创建用户密码，`chpasswd` 会将经过 /usr/bin/passwd 命令编码过的密码写入 /etc/passwd 的密码栏。

`# chpasswd < passwd.txt`

6. 确定密码经编码写入/etc/passwd的密码栏后。
  - 执行命令 `/usr/sbin/pwconv`
  - 将密码编码为 shadow password，并将结果写入 /etc/shadow。

`# pwconv`

这样就完成了大量用户的创建了，之后您可以到/home下检查这些用户宿主目录的权限设置是否都正确，并登录验证用户密码是否正确。
