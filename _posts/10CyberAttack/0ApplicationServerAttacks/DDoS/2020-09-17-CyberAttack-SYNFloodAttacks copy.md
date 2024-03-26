---
title: Meow's CyberAttack - Application/Server Attacks - DDos Dos - SYN Flood Attacks
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack, DDos]
tags: [CyberAttack, DDos]
toc: true
image:
---

- [Meow's CyberAttack - Application/Server Attacks - DDos Dos - SYN Flood Attacks](#meows-cyberattack---applicationserver-attacks---ddos-dos---syn-flood-attacks)
  - [SYN Flood Attacks](#syn-flood-attacks)
  - [Protect the network](#protect-the-network)

---

# Meow's CyberAttack - Application/Server Attacks - DDos Dos - SYN Flood Attacks

book: (S+ 7th)

---

## SYN Flood Attacks

- a common attack <font color=LightSlateBlue>used against servers on the Internet.</font>

- Easy to launch, difficult to stop, can cause significant problems.

- disrupts the TCP handshake process and can prevent legitimate clients from connecting.

- TCP sessions use a <font color=OrangeRed>three-way handshake</font> when establishing a session.

  - 2 systems normally start a TCP session by exchanging <font color=LightSlateBlue>three packets in a TCP handshake.</font>

  - Example:

  - when a client establishes a session with a server, it takes the following steps:
  	1. The client sends a `SYN (synchronize) packet` to the server.
  	2. The server responds with a `SYN/ACK (synchronize/acknowledge) packet`.
  	3. The client completes the handshake by sending an `ACK (acknowledge) packet`.
  	4. After establishing the session, the two systems exchange data.

![Image5](/assets/img/Image5.jpg)
￼
- the attacker <font color=OrangeRed>never completes the handshake by sending the ACK packet</font>.

- the attacker <font color=OrangeRed>sends a barrage of SYN packets</font>, leaving the server with multiple half-open connections.

  - server一旦接收到SYN包就需要为即将建立的TCP连接分配<font color=OrangeRed>TCB（Transmission Control Block）</font>，并进入<font color=OrangeRed>half-open状态</font>

  - 由于最多可开启的半开连接个数是一定的，受内存限制，当半开连接的个数过多，就会消耗掉可用的内存，使得新的正常的连接请求不能被处理。此时victim对server进行访问，建立TCP连接的请求就不能被正常处理。

  - half-open connections can consume resources and can actually crash.

- More often, the server <font color=OrangeRed>limits the number of these half-open connections.</font>
  - Once the limit is reached, the server won’t accept any new connections, blocking connections from legitimate users.

- example:

  - Linux: support `iptables` command: <font color=LightSlateBlue>set a threshold for SYN packets</font>, blocking them after the threshold is set.

  - prevents the SYN flood attack from crashing the system, but also denies service to legitimate clients.

- Attackers can launch SYN flood attacks from a single system in a DoS attack.
  - often spoof the source IP address when doing so.

- Attackers can also coordinate an attack from multiple systems using a DDoS attack.

## Protect the network

- <font color=OrangeRed>SYN cookies</font>
  - 在ACK到达之前不分配任何资源。
  - Instead of allocating a record, send a SYN-ACK with a carefully constructed `sequence number generated as a hash of the clients IP address, port number, and other information`.
  - When the client responds with a normal ACK, that special sequence number will be included, which the server then verifies.
  - Thus, the server first allocates memory on the third packet of the handshake, not the first.

- <font color=OrangeRed>RST cookies</font>
  - The server `sends a wrong SYN/ACK back` to the client. The client should then generate a RST packet telling the server that something is wrong.
  - At this point, the server knows the client is valid and will now accept incoming connections from that client normally

- <font color=OrangeRed>Stack Tweaking 拧</font>
  - TCP stacks can be tweaked in order to <font color=LightSlateBlue>reduce the effect of SYN floods.</font>
  - <font color=LightSlateBlue>Reduce the timeout before a stack frees up the memory</font> allocated for a connection

- <font color=OrangeRed>Micro Blocks</font>
  - Instead of allocating a complete connection,
  - simply <font color=LightSlateBlue>allocate a micro record</font> of 16-bytes for the incoming SYN object

校验和
- `TCP的校验和计算` 和 `IP头部的校验和计算` 方法是一致的，但是覆盖的数据范围不一样。
  - TCP校验和覆盖TCP首部和TCP数据，
  - IP首部中的校验和只覆盖IP的头部。
  - TCP的校验和是必需的，而UDP的校验和是可选的。
  - TCP和UDP计算校验和时，都要加上一个12字节的伪首部。
  - 伪首部包含：源IP地址、目的IP地址、保留字节(置0)、传输层协议号(TCP是6)、TCP报文长度(报头+数据)。
  - 伪首部是为了增加TCP校验和的检错能力：如检查TCP的源和目的IP地址、传输层协议等。
