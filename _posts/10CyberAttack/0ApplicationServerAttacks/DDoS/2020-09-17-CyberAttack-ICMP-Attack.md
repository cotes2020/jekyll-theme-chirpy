---
title: Meow's CyberAttack - Application/Server Attacks - DDos Dos - ICMP Attacks
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack, DDos]
tags: [CyberAttack, DDos, ICMP]
toc: true
image:
---

- [Meow's CyberAttack - Application/Server Attacks - DDos Dos - ICMP Attacks](#meows-cyberattack---applicationserver-attacks---ddos-dos---icmp-attacks)
  - [ICMP Attacks](#icmp-attacks)
  - [ping of death](#ping-of-death)
  - [Smurf attack can use ICMP traffic,](#smurf-attack-can-use-icmp-traffic)

---

# Meow's CyberAttack - Application/Server Attacks - DDos Dos - ICMP Attacks

---

## ICMP Attacks
- Availability Attacks
- Many networks permit the use of ICMP traffic (like ping…), because pings can be useful for network troubleshooting.
- attackers can use ICMP for DoS attacks.
- ICMP DoS attack variant:

---

## ping of death

- uses ICMP packets that are too **big**.
- sends ICMP traffic as a series of fragments, to `overflow the fragment reassembly buffers` on the target device.
- network attacks, sending an abnormally large packet size that exceeds TCP/IP specifications.

- This is when an IP datagram is received with the "protocol" field in the IP header set to 1 (ICMP), the Last Fragment bit is set, and (IP offset ` 8) + (IP data length) >65535.
	- the IP offset (which represents the starting position of this fragment in the original packet, and which is in 8-byte units) plus the rest of the packet is greater than the maximum size for an IP packet.

---

## Smurf attack can use ICMP traffic,
- <font color=LightSlateBlue> Ping a broadcast address using a spoofed source address directed to a subnet, to flood a target system with ping replies. </font>
- attacker sends a ping to the broadcast address of subnet 172.16.0.0/16.
- This collection of pings instruct devices on that subnet to send their ping replies to the target system at IP address 10.2.2.2
- thus flooding the target system’s bandwidth and processing resources.
- in the subnet being used for the Smurf attack, thousands of systems could potentially be involved and send ping replies to the target system.

Prevented smurf
- Command for cisco routers: No ip directed-broadcasts

![Screenshot 2024-03-05 at 17.58.54](/assets/img/Screenshot%202024-03-05%20at%2017.58.54.png)
