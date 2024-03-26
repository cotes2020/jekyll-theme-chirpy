---
title: Meow's CyberAttack - ICMP Attacks 
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

---

## Amplification / Reflected DoS

- usually employed as a part of a DDoS attack.

- `get a response to their request in a greater than 1:1 ratio`, the additional bandwidth traffic works to `congest and slow the responding server down`. 

- significantly increases the amount of traffic sent to, or requested from, a victim.

- The ratio achieved: amplification factor, and high numbers are possible with UDP-based protocols like `NTP, CharGen, and DNS`. 

Example, 
- the command `monlist` can be used with an NTP amplification attack 
- to send details of the last 600 people who have requested the time from that computer back to the requester, 
- resulting in more than 550 times the amount of data that was requested to be sent back to a spoofed victim. 
- Bots can be used to send requests with the same spoofed IP source address from lots of different zombies and `cause the servers to send a massive amount of data back to the victim`;
- for this reason, it is also referred to as `reflected DoS`. 


### Amplification Example

`spoofs the source address` of a directed broadcast ping packet to `flood a victim with ping replies`. 
- A ping is normally unicast—one computer to one computer.
  - A ping sends `ICMP echo requests` to one computer, and the receiving computer responds with `ICMP echo responses`.
- The smurf attack: sends the ping out as a broadcast. 
  - broadcast, one computer sends the packet to all other computers in the subnet.
- The smurf attack spoofs the source IP.
  - If the source IP address isn’t changed, the computer sending out the broadcast ping will get flooded with the ICMP replies. 
  - Instead, the smurf attack substitutes the source IP with the IP address of the victim, and the victim gets flooded with these ICMP replies.

DNS amplification attacks 
- send DNS requests to DNS servers spoofing the IP address of the victim. 
- Instead of just asking for a single record, these attacks tell the DNS servers to send as much zone data as possible, amplifying the data sent to the victim. 
- Repeating this process from multiple attackers can overload the victim system.

Network Time Protocol (NTP) amplification attack 
- uses the monlist command: sends a list of the last 600 hosts that connected to the NTP server. 
- the attacker spoofs the source IP address when sending the command. 
- The NTP server then floods the victim with details of the last 600 systems that requested the time from the NTP server.
