---
title: Linux - Sysfile
date: 2020-07-16 11:11:11 -0400
categories: [30System, Sysadmin]
tags: [Linux, Sysadmin]
math: true
image:
---

# Linux - Sysfile

[toc]

## /etc/passwd

```bash

$ sudo cat /etc/passwd

----------------------------------------------
Username, password, uid, gid, comment, home directory, default shell
----------------------------------------------

root:x:0:0:root:/root:/bin/bash
daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
bin:x:2:2:bin:/bin:/usr/sbin/nologin
sys:x:3:3:sys:/dev:/usr/sbin/nologin
# nologin
pollinate:x:111:1::/var/cache/pollinate:/bin/false
ubuntu:x:1000:1000:Ubuntu:/home/ubuntu:/bin/bash
bob:x:1001:1001::/home/bob:
lisa:x:1002:1002::/home/lisa:
```


### nologin

nologin的作用就是限制某些用户通过ssh登陆到shell上。有时候为了进行系统维护工作，临时禁止其他用户登录
可以使用 nologin 文件, 在/etc/目录下创建一个名称为 nologin 的文件

touch /etc/nologin

禁止用户登录时，/etc/nologin 文件中的内容将会显示给用户，会一闪而过。
例如，按如下方式创建 nologin 文件：

disable login by admin temporarily!

当用户试图登陆时，将会给用户显示"disable login by admin temporarily!"，
当系统维护结束以后，再删除/etc/nologin文件，其他用户就又可以恢复登陆了，

这只是限于能登陆shell的用户来说的，
对于登陆shell为/sbin/nologin的用户来说没有影响，因为他们本身就无法登陆shell。

当/etc/nologin档案存在时，则任何一个一般身份帐号在尝试登入时，都仅会获得/etc/nologin内容的资讯，而无法 登入主机。 举例来说，当我建立/etc/nologin ，并且内容设定为『This Linux server is maintaining....』， 那么任何人尝试登入时，都仅会看到上面提到的这段讯息，而且无法登入喔！ 一直要到/etc/nologin 被移除后，一般身份使用者才能够再次的登入啊


---

## /etc/shadow

```py
user:$6$BUT3hIub$JjoOhlK0:14478:0:99999:7:::

Username
Version of cipher - separated by a "$"
Encrypted password* - separated by a "$"
Salt* - separated by a "$"
Days since epoch of last password change
Days until change allowed
Days before change required
Days warning for expiration
Days before account inactive
Days since Epoch when account expires
Reserved
```

---

## /etc/group

```py
safes:*:500:williams, jones

Groupname:Password:Group ID:Group users
```


















.
