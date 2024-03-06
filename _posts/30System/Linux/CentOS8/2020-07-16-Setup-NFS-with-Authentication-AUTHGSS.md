---
title: Linux - Install NFS in CentOS8
date: 2020-07-16 11:11:11 -0400
categories: [30System, CentOS8]
tags: [Linux, Install, NFS, CentOS8]
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

content:

[Install Kerberos Authentication in Centos 8](https://github.com/ocholuo/system/blob/master/linux/step/InstallKerberosAuthenticationinCentos8.md)


.
