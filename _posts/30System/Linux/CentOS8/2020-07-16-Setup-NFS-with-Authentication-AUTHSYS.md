---
title: Linux - Setup NFS with Authentication AUTHSYS
date: 2020-07-16 11:11:11 -0400
categories: [30System, CentOS8]
tags: [Linux, Setup, NFS, Authentication]
math: true
image:
---


# NFS

[toc]

Assignment under:

LFCE: Advanced Network and System Administration / Configuring and Managing NFS - [LFCEbyPluralsight](https://app.pluralsight.com/library/courses/advanced-network-system-administration-lfce/table-of-contents)


# NFS file permission

UID and GID overlap:
- configuration file specific the user machine, lack scalability.
- central authentication server

`AUTH_SYS`:
- dafult for NFS.
- UID/GID model

`AUTH_GSS`:
- base on kerberos.
- authorize both the user and the system.
- requirement: configuration need to be setup
  - kerneros key distribution center (KDC) installed
  - host and service principals added for client and server.
  - create a add key-tabs on client and server.
  - change the authentication mechanisms on NFS client and server to use.
    - sec=krb5, krb5i, krb5p

---

# AUTH_SYS (default security mechanism)

## for user

no authentication, but passing same UID.

```c
server1:
$ touch /mnt/file1.test
touch: cannot touch '/mnt/file1.test': Permission denied
$ ll /mnt
drwxr-xr-x.   2 root root       6 May 10  2019 mnt

server0:
$ chown server1:server1 /share1

server1 not able to access /mnt
================================================
$ cat /etc/passwd | grep server0
server0:x:1000:1000:server0:/home/server0:/bin/bash
$ cat /etc/passwd | grep server1
server1:x:1000:1000:server1:/home/server1:/bin/bash
```

---

## for group

server0:
1. setup share directory
2. setup usergroup
3. add exports, update configure file.

```c
groupadd -g 2000 marketing
$ sudo mkdir /marketing
$ sudo chown nobody:marketing /marketing/

$ ll /
drwxrwxr-x.   2 nobody marketing       6 May  1 22:23 marketing

$ sudo vi /etc/exports
// /marketing server?.psdemo.local(rw)

$ sudo exportfs -arv
```


server1:

1. add group with same group id
2. add user for that group
3. add mount directory
4. mount it.
5. user demo cannot modify the dir. but anthony from the group can.

```c
$ sudo groupadd -g 2000 marketing

$ sudo useradd -G marketing anthony

$ groups anthony
anthony : anthony marketing

$ sudo mkdir /marketing

$ sudo mount -t nfs server0.psdemo.local:/marketing /marketing

[demo]$ touch marketing.txt
touch: cannot touch 'marketing.txt': Permission denied

[anthony]$ touch marketing.txt
$ ls
marketing.txt

```















.
