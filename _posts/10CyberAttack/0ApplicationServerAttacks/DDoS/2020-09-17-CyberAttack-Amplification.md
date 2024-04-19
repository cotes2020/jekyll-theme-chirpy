---
title: Meow's CyberAttack - Application/Server Attacks - DDos Dos - Amplification / Reflected DoS
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack, DDos]
tags: [CyberAttack, DDos]
toc: true
image:
---

- [Meow's CyberAttack - Application/Server Attacks - DDos Dos - Amplification / Reflected DoS](#meows-cyberattack---applicationserver-attacks---ddos-dos---amplification--reflected-dos)
  - [Amplification / Reflected DoS](#amplification--reflected-dos)
    - [Amplification Example](#amplification-example)

---

# Meow's CyberAttack - Application/Server Attacks - DDos Dos - Amplification / Reflected DoS

book: Security+ 7th ch9

---

## Amplification / Reflected DoS

- usually employed as a part of a DDoS attack.

- <font color=LightSlateBlue>get a response to their request in a greater than 1:1 ratio</font>, the additional bandwidth traffic works to <font color=LightSlateBlue>congest and slow the responding server down. </font>

- significantly increases the amount of traffic sent to, or requested from, a victim.

- The ratio achieved: <font color=LightSlateBlue> amplification factor</font>, and high numbers are possible with UDP-based protocols like `NTP, CharGen, and DNS`.

Example,
- the command `monlist` can be used with an NTP amplification attack
- to send details of the last 600 people who have requested the time from that computer back to the requester,
- resulting in more than 550 times the amount of data that was requested to be sent back to a spoofed victim.
- Bots can be used to send requests with the same spoofed IP source address from lots of different zombies and <font color=OrangeRed>cause the servers to send a massive amount of data back to the victim</font>;
- for this reason, it is also referred to as <font color=LightSlateBlue>reflected DoS</font>.


### Amplification Example

<font color=LightSlateBlue>spoofs the source address</font> of a directed broadcast ping packet to <font color=OrangeRed>flood a victim with ping replies</font>.
- <font color=OrangeRed>A ping is normally unicast—one computer to one computer</font>.
  - A ping sends `ICMP echo requests` to one computer, and the receiving computer responds with `ICMP echo responses`.
- <font color=OrangeRed>The smurf attack: sends the ping out as a broadcast</font>.
  - broadcast, one computer sends the packet to all other computers in the subnet.
- <font color=OrangeRed>The smurf attack spoofs the source IP</font>.
  - If the source IP address isn’t changed, the computer sending out the broadcast ping will get flooded with the ICMP replies.
  - the smurf attack substitutes the source IP with the IP address of the victim, and the victim gets flooded with these ICMP replies.

<font color=OrangeRed>DNS amplification attacks</font>
- send DNS requests to DNS servers spoofing the IP address of the victim.
- Instead of just asking for a single record, these attacks tell the DNS servers to send as much zone data as possible, amplifying the data sent to the victim.
- Repeating this process from multiple attackers can overload the victim system.

<font color=OrangeRed>Network Time Protocol (NTP) amplification attack</font>
- uses the `monlist` command: sends a list of the last 600 hosts that connected to the NTP server.
- the attacker spoofs the source IP address when sending the command.
- The NTP server then floods the victim with details of the last 600 systems that requested the time from the NTP server.
