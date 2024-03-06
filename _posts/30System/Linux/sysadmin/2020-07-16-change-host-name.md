---
title: Linux - Change Server’s Hstname
date: 2020-07-16 11:11:11 -0400
categories: [30System, Sysadmin]
tags: [Linux, Sysadmin]
math: true
image:
---

# Change a server’s hostname

```py
vi /etc/sysconfig/network
#add
HOSTNAME=server0.demo.local

vi /etc/hosts
127.0.0.1   localhost localhost.localdomain localhost4 localhost4.localdomain4
::1         localhost localhost.localdomain localhost6 localhost6.localdomain6
192.168.1.1	server0.demo.local
192.168.2.1     server0.demo.local
192.168.1.100     server1.demo.local
192.168.2.200     server2.demo.local


hostnamectl set-hostname server0.demo.local

systemctl restart network
```
