---
title: Linux - Display Routing Table
date: 2020-07-16 11:11:11 -0400
categories: [30System, Sysadmin]
tags: [Linux, Sysadmin]
math: true
image:
---

# Display Routing Table

- route
- netstat
- ip

```bash
# route
$ sudo route -n
Kernel IP routing table
Destination     Gateway         Genmask         Flags Metric Ref    Use Iface
192.168.0.0     0.0.0.0         255.255.255.0   U     0      0        0 eth0
0.0.0.0         192.168.0.1     0.0.0.0         UG    0      0        0 eth0


# -n option means that you want numerical IP addresses displayed, instead of the corresponding host names.



# netstat
$ netstat -rn
Kernel IP routing table
Destination     Gateway         Genmask         Flags   MSS Window  irtt Iface
192.168.0.0     0.0.0.0         255.255.255.0   U         0 0          0 eth0
0.0.0.0         192.168.0.1     0.0.0.0         UG        0 0          0 eth0

# -r option specifies that you want the routing table. The -n option is similar to that of the route command.



# ip
$ ip route list
192.168.0.0/24 dev eth0  proto kernel  scope link  src 192.168.0.103
default via 192.168.0.1 dev eth0
```
