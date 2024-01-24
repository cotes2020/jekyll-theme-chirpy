---
title: Linux - Install NTP in CentOS8
date: 2020-07-16 11:11:11 -0400
categories: [30System, CentOS8]
tags: [Linux, Install, NTP, CentOS8]
math: true
image:
---


# Install NTP in CentOS8

[toc]

Assignment under:

LFCE: Advanced Network and System Administration / Configuring and Managing NFS - [LFCEbyPluralsight](https://app.pluralsight.com/library/courses/advanced-network-system-administration-lfce/table-of-contents)

## NTP

an accurate system time on a Linux server is very important for system components such as Cron and Anacron, backup scripts...

Accurate timekeeping need Network Time Protocol (NTP) protocol.

```c
1. setup ntp server.

systemctl enable ntpd
systemctl start ntpd
systemctl status ntpsd

firewall-cmd --permanent --add-service=ntp
firewall-cmd --reload

vi /etc/ntp.conf
//add
restrict 192.168.1.0 mask 255.255.255.0 nomodify notrap
server0.psdemo.local
```

## chrony

In RHEL Linux 8, the `ntp` package is no longer supported and it is implemented by the `chronyd` (a daemon that runs in user-space)

The `chrony` suite consists of `chronyd`, and `chronyc`
- a command line utility which is used to change various operating parameters and to monitor its performance whilst it is running.
- `chrony` works both as an NTP server and as an NTP client, which is used to synchronize the system clock with NTP servers, and can be used to synchronize the system clock with a reference clock (e.g a GPS receiver).

---

## How to Install Chrony in Centos 8


```c
$ yum install chrony
$ systemctl start chronyd
$ systemctl enable chronyd
$ systemctl status chronyd
```

---

## How to Configure NTP Server Using Chrony in Centos 8

```c

# vi /etc/hosts

127.0.0.1   localhost localhost.localdomain localhost4 localhost4.localdomain4
::1         localhost localhost.localdomain localhost6 localhost6.localdomain6
192.168.1.1     server0.psdemo.local
192.168.1.100   server1.psdemo.local
192.168.2.100   server2.psdemo.local

192.168.1.1 server0 server0.lab.com
192.168.1.100 server1 server1.lab.com
192.168.2.100 server2 server2.lab.com


$ vi /etc/ntp.conf

$ vi /etc/chrony.conf
// change
// set its value to the network or subnet address from which the clients are allowed to connect.
# Allow NTP client access from local network.
allow 192.168.1.0/24
allow 192.168.2.0/25

$ systemctl restart chronyd

$ firewall-cmd --permanent --add-service=ntp
$ firewall-cmd --reload


// after client setup
$ sudo chronyc clients
Hostname                      NTP   Drop Int IntL Last     Cmd   Drop Int  Last
===============================================================================
localhost                       0      0   -   -     -       1      0   -   142
server1.psdemo.local            2      0   6   -    27       0      0   -     -
[server0@server0 ~]$

```

---

## How to Configure NTP Client Using Chrony in CentOS 8

To configure a system as an NTP client, it needs to know which NTP servers it should ask for the current time.
- specify the servers using the server/pool directive.


```c
$ vi /etc/chrony.conf
// change
# Use public servers from the pool.ntp.org project.
# Please consider joining the pool (https://www.pool.ntp.org/join.html).
# pool 2.centos.pool.ntp.org iburst
server 192.168.1.1 iburst prefer

$ systemctl restart chronyd


// show the current time sources (NTP server) that chronyd is accessing,
# chronyc sources
MS Name/IP address         Stratum Poll Reach LastRx Last sample
===============================================================================
^* server0.psdemo.local          3   6     1    19  -1066us[-1066us] +/-   68ms


$ firewall-cmd --permanent --add-service=ntp
$ firewall-cmd --reload
```

---

## Issues with chronyc

### error 501 Not authorised

```c
$ chronyc clients
Hostname                      NTP   Drop Int IntL Last     Cmd   Drop Int  Last
===============================================================================
501 Not authorised
[server0@server0 ~]$ sudo chronyc clients
Hostname                      NTP   Drop Int IntL Last     Cmd   Drop Int  Last
===============================================================================
localhost                       0      0   -   -     -       1      0   -   142
server1.psdemo.local            2      0   6   -    27       0      0   -     -
```



Since version 2.2, the password command doesn’t do anything and chronyc needs to run locally under the root or chrony user, which are allowed to access the chronyd's Unix domain command socket.

With older versions, you need to authenticate with the password command first or use the -a option to authenticate automatically on start.

The configuration file needs to specify a file which contains keys (keyfile directive) and which key in the key file should be used for chronyc authentication (commandkey directive).










.
