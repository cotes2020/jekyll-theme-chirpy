---
title: Meow's CyberAttack - Dos DDos
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack, DDos]
tags: [CyberAttack, DDos]
toc: true
image:
---

[toc]

---

# Meow's CyberAttack - Dos DDos

---

![images](https://i.imgur.com/k7bvIZy.png)

---

## Denial-of-Service

DoS的目標不是打服務，而是打掉對方的網路與資源，造成社交攻擊的機會

DoS攻擊手法：
（1）Bandwidth Attack：發送大量ICMP ECHO封包。
（2）Service Request Floods：要求對方回應自己快速發出的所有requests。
（3）SYN Flooding：送出大量SYN封包，讓對方一直等待到記憶體用盡。
（4）ICMP Flood Attack：送出大量假地址的ICMP，使其回應給假地址，造成放大效果。
（5）Peer-to-Peer Attack：利用DC++（Direct Connect）埠打DoS。
（6）Permanent Denial-of-Service Attack：針對硬體做攻擊。
（7）Application-Level Flood Attack：針對應用層發起的攻擊。
（8）Distributed Reflection DoS（DRDoS）：假冒對方身份向其他主機發出某種類型封包，讓其他主機回應給對方。


- denial-of-service (DoS) attack: attack from one attacker against one target.
- distributed denial-of-service (DDoS) attack:  attack from two or more computers against a single target.

Denial-of-Service (DoS):
- Most simple DoS attacks occur from a single system, and a specific server or organization is the target.
Several types of attacks can occur:
- Deny access to information, applications, systems, or communications.Bring down a website while the communications and systems continue to operate.
- Crash the operating system (a simple reboot may restore the server to normal operation).
- Fill the communications channel of a network and prevent access by authorized users.
- Open as many TCP sessions as possible. This type of attack is called a TCP SYN flood DoS attack.
Two of the most common types of DoS attacks:
- ping of death: crashes system by sending Internet Control Message Protocol (ICMP) packets (think echoes) that are larger than the system can handle.
  - sPing.
- buffer overflow: put more data (usually long input strings) into the buffer than it can hold.
  - Code Red, Slapper, and Slammer: took advantage of buffer overflows

Type of DoS:
- Smurf attacks
- SYN floods
- Local area network denial (LAND)
- fraggle.

---

## distributed denial-of-service (DDoS) attack
- similar to a DoS attack.
- amplifies the concepts of a DoS attack by using multiple computer systems (often through botnets) to conduct the attack against a single organization.
- attacks often include sustained, abnormally high network traffic on the network interface card of the attacked computer.
- Other system resource usage (processor, memory usage…) will also be abnormally high.
- The goal: perform a service attack and prevent legitimate authorized users from accessing services on the target computer.
- The servers can be physically busy, or consumes all of the available bandwidth.
  - overload the resources (such as the processor and memory) and lead to resource exhaustion.
  - preventing legitimate users from viewing web pages.
  - In extreme cases of resource exhaustion, the attacked computer might crash.

- exploit the inherent weaknesses of dedicated networks like DSL and cable.
  - These permanently attached systems usually have little, if any, protection.
- Load attack program onto dozens, hundreds of computer that use DSL/cable modems.
- The attack program lies dormant on these computers until they get an attack signal from a master computer. The signal triggers the systems, which launch an attack simultaneously on the target network or system.
  - The systems infacted: zombies or nodes, carry out the instruction by the master computer.
- common on the Internet, hit large companies, often widely publicized in the media.
- DDoS: far more common and effective than DoS.
- The nasty part: the machines operate attack belong to normal computer users and the attack gives no special warning to those users. When the attack is complete, the attack program may remove itself or infect the unsuspecting user’s computer with a virus that destroys the hard drive, thereby wiping out the evidence.
- example of DDoS programs: Tribal Flood Network (TFN), Shaft and Trinoo.

---

## Prevent

![DDoS-Mitigation-Stages](https://i.imgur.com/I6SLQ4O.jpg)

- In general, little that you can do to prevent DoS or DDoS attacks.
- Many operating systems are particularly susceptible to these types of attacks.
- Make sure that your operating system and the applications you use are up-to-date.
- Use traceback to identify the source of the flooded pakcets.
  - The downside: the source is most likely innocent but compromised machines.
  - IP spoofing
- Implement filter to remove flooded packets. Before it reach the host.
  - The downside: will be filtering out some legitimate packets as well.
- 透過異常行為檢測DoS的方法：
  - 定行為的正常值
  - 分時、分段
  - 改變點的偵測
