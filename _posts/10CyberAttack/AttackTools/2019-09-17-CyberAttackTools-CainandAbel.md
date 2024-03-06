---
title: Meow's Testing Tools - Cain and Abel
date: 2019-09-17 11:11:11 -0400
categories: [10CyberAttack, CyberAttackTools]
tags: [CyberAttack, CyberAttackTools]
toc: true
image:
---

# Cain and Abel

[toc]

---

## basic

Cain and Abel
- a Windows-based password recovery tool.
- have many features that can allow Network Sniffing and Hijacking of IP traffic between hosts.
- Many other features include network sn, Hash calculator, Certificate collector, Record VoIP conversations, and ARP poisoning.

---

## ARP Poisoning

1. select which network adapter to target
   - installed on a Windows host connected to a network.
   - specify the network adapter that Cain and Abel will sniff.

![0*NMe8lLaJjBV5cGbd](https://i.imgur.com/CFkJdNt.png)

2. MAC Address scanner window.
   - perform a scan to identify a list of hosts connected to the network.
   - Selece entire subnet or only a specific range within your subnet.

![0*Nx-jyYEarYz3uZz0](https://i.imgur.com/QTdtRwu.png)

3. Screenshot of both Virtual Machines.
   - Left:
     - list of hosts connected to a network after MAC address scan.
   - Right:
     - the target VM with a simple IPv4 address lookup with `ipconfig`.
     - The IPv4 address on a Windows host
     - Displayed: ipconfig command.

![0*OS5CjHO8NTkb5_nl](https://i.imgur.com/bwGB90I.png)

4. Address Poison Routing “pool” window on Cain and Abel.
   - to intercept the traffic flowing in between these two devices.
   - selecting the `network gateway (172.20.10.1)` and the target `device (172.20.10.13)`
    - adding both devices to the `ARP Poison Routing “pool”`
   - now perform ARP spoofing.

![0*izEZvkRDPIEhh0iH](https://i.imgur.com/96wzBsE.png)
￼
5. `ARP Address Poison` Routing window:
   - shows traffic intercepted in between the target device and router.
   - The IP address column in the lower half of the window on towards the right side shows the different outbound connections made to the Internet.

![0*W_FqcaLv3j6AZvMm](https://i.imgur.com/mrNrg3f.png)

6. The window above shows a captured password intercepted on a HTTP website.
   - Since Cain and Abel is sniffing the network, including the target device
   - any usernames or passwords entered into unencrypted HTTP websites can be intercepted as cleartext.
   - You can directly launch the website by clicking on the URL.

![0*C8izOG9iNXBK35QY](https://i.imgur.com/UfhJzuY.png)
