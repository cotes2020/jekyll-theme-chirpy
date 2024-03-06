---
title: Linux - Network Error
date: 2020-07-16 11:11:11 -0400
categories: [30System, error]
tags: [error]
math: true
image:
---

# Linux - Network Error

[toc]

---

## ping fail

`/etc/resolv.conf` gets regenerated on boot, prevent `resolv.conf` to be update at boot time,

```c
# resolv.conf
nameserver 10.0.2.3
nameserver 192.168.1.1
nameserver 8.8.8.8
nameserver 8.8.4.4
search localhost

# cat /etc/sysconfig/network-scripts/ifcfg-eth0
Change PEERDNS=yes to PEERDNS=no
```


## ping LAN but no internet

do have a connection, but you cannot reach your DNS; you can diagnose this by

```
ping -c1 8.8.4.4
```

1. can reach Google, then you have a connection, and you only need to update your DNS servers. Edit (as sudo) your /etc/resolv.conf file and add these two lines:

    ```
    nameserver 8.8.8.8
    nameserver 8.8.4.4
    ```

    and now you are good to go.

2. cannot ping Google, but you can ping your router, or any other pc in your LAN. In this case case, it is possible that you also have problem 1, so you will have to check for that, but first, you need to check your routing table. Print it with

    ```c
    ip route show default

    // if reply, like this one:
    default via 192.168.11.1 dev wlan0 proto dhcp metric 600
    ```

    (this is for my laptop). What is important is that the correct IP address of your router is shown exactly where mine (192.168.11.1) is shown. If an incorrect IP address is shown, or, worse, if the ip route show default command receives no reply, then your routing table has been corrupted. You may simply restore it by means of:
    ```
    sudo ip route del default (only if the wrong IP address appears)
    sudo ip route add default via IP.address.OfYour.Router
    ```

    and now we may go step 1.

3. If you cannot ping any pc on your LAN, then there is another kind of problem, and more questions will need to be asked. But we'll cross that bridge when we get there.


---

## configure ip for Interface


```c
$ route -n
$ ip r
```


### dhcp

```c
// Lease information on the host:
$ cat /var/lib/dhclient/dhclient.leases

// Lookup DHCP host:
$ sudo grep “DHCPOFFER” /var/log/message

// check DHCP client
$ ss -luntp | grep dhclinet
```

capture the DHCP package

```c
$ sudo dhclient -r
$ dhclient
```

![Screen Shot 2020-07-06 at 15.34.33](https://i.imgur.com/BZNbU20.png)

---

### static

1. add ip

```c
// Add an entry for the 1.1.1.1 to use 10.0.1.10 as the gateway.
$ ip route ad 1.1.1.1 via 10.0.1.10 dev eth0

// static ip
$ nmcli connection modify interface01 ipv4.method manual ipv4.address 10.0.1.15/24 ipv4.gateway 192.168.51.2

// configure DNS
$ nmcli con mod interface01 ipv4.dns 192.168.51.1

$ nmcli con down interface01
$ nmcli con up interface01

// change it to DHCP
$ nmcli con mod interface01 ipv4.method auto ipv4.address "" ipv4.gateway "" ipv4.dns ""
```

2. prohibit ip

```c
$ yum install bind-utils
$ host google.com

$ ip route add prohibit 1.1.1.1   // runtime
$ ip route flush 1.1.1.1

```

---


### Multiple IPs on the Same Interface


```c

// show interface
$ nmcli c  // connection
$ nmcli d  // device

$ nmcli d show eth0

// add ip addresses:
$ nmcli con mod System\eth0 ipv4.method manual ipv4.addresses 10.0.1.10/24 ipv4.gateway 10.0.1.1 ipv4.dns 10.0.0.2 ipv4.dns-search ec2.internal

// add 2 additional addresses:
$ sudo nmcli c mod System\eth0 ipv4.addresses 10.0.1.15/24, 10.0.1.20/24

//check where things stand now, run:
$ nmcli c show System\eth0 | grep ipv4
$ nmcli con show System\eth0 | grep ipv4

// Restart Networking
$ sudo systemctl restart network
```

---

### NIC Bond and Teamin
- for fault tolerant, redundant connection
- Bond
- Team


#### Bond

1. create master bond
```c
$ nmcli con add type bond con-name bond0 ifname bond0 mode actice-backup ip4 192.168.51.170/24
```
![Screen Shot 2020-07-06 at 15.57.23](https://i.imgur.com/aOJrNUk.png)


2. add bond-slave
```c
$ nmcli con add type bond-slave ifname ens33 master bond0
$ nmcli con add type bond-slave ifname ens38 master bond0

$ nmcli con up bond
$ nmcli con up bond-slave33
$ nmcli con up bond-slave38
```

![Screen Shot 2020-07-06 at 16.02.00](https://i.imgur.com/CDydebD.png)

3. add gateway for master bond
```c
$ nmcli con mod bond0 ipv4.gateway 192.168.51.2

$ nmcli con down bond0
$ nmcli con up bond0
```


#### Team

1. clear the interface

![Screen Shot 2020-07-06 at 16.11.05](https://i.imgur.com/N5Ac4uG.png)

```c
$ nmcli con deletet Wired\connection\1
$ nmcli con deletet Wired\connection\2
```

![Screen Shot 2020-07-06 at 16.11.25](https://i.imgur.com/DZWQTQC.png)


2. Configure the Team Interface

```c
$ yum install -y teamd

// example configure
$ cat /usr/share/doc/teamd-1.27/example_configs/activebackup_ethtool_3.conf

$ nmcli con add type team con-name Team0 ifname team0 team.config '{"runner":{"name":"activebackup"}, "link_watch":{"name"::ethtool}}'

// for roundrobin
$ nmcli con add type team con-name Team0 ifname team0

$ nmcli con mod Team0 ipve.address 10.0.1.15/24 ipve.gateway 10.0.1.1 ipv4.method namual
```

![Screen Shot 2020-07-06 at 16.19.49](https://i.imgur.com/qs7JTt7.png)

![Screen Shot 2020-07-06 at 16.33.04](https://i.imgur.com/GwpLaC6.png)

3. Add Slave Interfaces

```c
$ nmcli con add type team-slave con-name slave1 ifname eth1 master team0
$ nmcli con add type team-slave con-name slave2 ifname eth2 master team0

$ nmcli con up slave1
$ nmcli con up slave2
$ nmcli con up Team0

$ teamdctl team0 State
```

![Screen Shot 2020-07-06 at 16.22.11](https://i.imgur.com/3IM98X4.png)


---




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










.
