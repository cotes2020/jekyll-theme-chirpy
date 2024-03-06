---
title: NetworkSec - Network basic
# author: Grace JyL
date: 2018-05-18 11:11:11 -0400
description:
excerpt_separator:
categories: [15NetworkSec, NetworkBasic]
tags: [NetworkSec, TCP]
math: true
# pin: true
toc: true
image: /assets/img/note/tls-ssl-handshake.png
---

- [网络协议](#网络协议)
- [The OSI Model](#the-osi-model)
  - [Layers in OSI](#layers-in-osi)
- [TCP & UDP, TCP/IP](#tcp--udp-tcpip)
  - [layers in TCP/IP](#layers-in-tcpip)
    - [4. **Application Layer**:](#4-application-layer)
    - [3. **Host-to-Host**](#3-host-to-host)
    - [2. **Internet**](#2-internet)
    - [1. **Network Access**](#1-network-access)
  - [Common Application Protocols in the TCP/IP Stack](#common-application-protocols-in-the-tcpip-stack)
  - [封装与解封装](#封装与解封装)
  - [TCP vs UDP](#tcp-vs-udp)
  - [Transmission Control Protocol (TCP)](#transmission-control-protocol-tcp)
    - [TCP three-way handshake](#tcp-three-way-handshake)
    - [TCP Features](#tcp-features)
    - [Sliding window protocol:](#sliding-window-protocol)
    - [Buffering:](#buffering)
    - [TCP Packet Format](#tcp-packet-format)
  - [User Datagram Protocol (UDP)](#user-datagram-protocol-udp)
- [实际数据传输举例](#实际数据传输举例)
- [网络构成](#网络构成)
  - [网络构成要素](#网络构成要素)
  - [通信介质与数据链路](#通信介质与数据链路)

- ref
  - [36张图详解网络基础知识](https://mp.weixin.qq.com/s/XGSpPNf4IYCXNjd63ERZ0w)
  - [plur](https://www.pluralsight.com/blog/it-ops/networking-basics-tcp-udp-tcpip-osi-models)

---


![Screen Shot 2021-05-24 at 05.54.21](https://i.imgur.com/8T20cQ0.png)

The **Transmission Control Protocol/Internet Protocol (TCP/IP)** suite
- created by the `U.S. Department of Defense (DoD)`
- ensure that communications could survive any conditions and that data integrity wouldn't be compromised under malicious attacks.

The **Open Systems Interconnection Basic Reference Model (OSI Model)**
- an abstract description for network protocol design
- developed as an effort to standardize networking.


---

# 网络协议
我们用手机连接上网的时候，会用到许多网络协议。
- 从手机连接 WiFi 开始，使用的是 802.11 （WLAN）协议，通过 WLAN 接入网络；
- 手机自动获取网络配置，使用的是 DHCP 协议，获取配置后手机才能正常通信。
  - 这时手机已经连入局域网，可以访问局域网内的设备和资源，但还不能使用互联网应用。
- 想要访问互联网，还需要在手机的上联网络设备上实现相关协议，即在无线路由器上配置 NAT、 PPPOE 等功能，再通过运营商提供的互联网线路把局域网接入到互联网中

![Screen Shot 2021-05-24 at 06.01.28](https://i.imgur.com/Z4RHark.png)

> 局域网 ：小范围内的私有网络，一个家庭内的网络、一个公司内的网络、一个校园内的网络都属于局域网。

> 广域网：把不同地域的局域网互相连接起来的网络。运营商搭建广域网实现跨区域的网络互连。

> 互联网：互联全世界的网络。互联网是一个开放、互联的网络，不属于任何个人和任何机构，接入互联网后可以和互联网的任何一台主机进行通信。


手机、无线路由器等设备通过多种**网络协议**实现通信。

**网络协议**就是为了通信各方能够互相交流而定义的标准或规则，设备只要遵循相同的网络协议就能够实现通信。那网络协议又是谁规定的呢？
- ISO 制定了一个国际标准 **OSI** ， 其中的 OSI 参考模型常被用于网络协议的制定。

---


# The OSI Model

OSI 参考模型将网络协议提供的服务分成 7 层，并定义每一层的服务内容
- 实现每一层服务的是**协议**，协议的具体内容是**规则**。
- 上下层之间通过**接口**进行交互，同一层之间通过**协议**进行交互。
- OSI 参考模型只对各层的服务做了粗略的界定，并没有对协议进行详细的定义，但是许多协议都对应了 7 个分层的某一层。


**All People Seem To Need Data Processing**
- Layer 1: The physical layer
- Layer 2: The data link layer
- Layer 3: The network layer
- Layer 4: The transport layer
- Layer 5: The session layer
- Layer 6: The presentation layer
- Layer 7: The application layer

![Screen Shot 2020-04-14 at 22.53.56](https://i.imgur.com/qj4yNvP.png)

![Screen Shot 2021-05-24 at 06.06.08](https://i.imgur.com/VFN2xBA.png)

## Layers in OSI

应用层:
- 应用程序和网络之间的接口，直接向用户提供服务。
- 应用层协议有电子邮件、远程登录等协议。

![Screen Shot 2021-05-24 at 06.07.39](https://i.imgur.com/DpSAsQh.png)


表示层:
- 负责数据格式的互相转换，如编码、数据格式转换和加密解密等。
- 保证一个系统应用层发出的信息可被另一系统的应用层读出。

![Screen Shot 2021-05-24 at 06.08.23](https://i.imgur.com/vX6yaEs.png)



会话层:
- 主要是管理和协调不同主机上各种进程之间的通信（对话）
- 负责建立、管理和终止应用程序之间的会话。

![Screen Shot 2021-05-24 at 06.08.41](https://i.imgur.com/fO8mtaL.png)


传输层:
- 为上层协议提供通信主机间的可靠和透明的数据传输服务，包括处理差错控制和流量控制等问题。
- 只在通信主机上处理，不需要在路由器上处理。

![Screen Shot 2021-05-24 at 06.09.09](https://i.imgur.com/EiIJ1A4.png)


网络层:
- 在网络上将数据传输到目的地址，主要负责寻址和路由选择。

![Screen Shot 2021-05-24 at 06.09.17](https://i.imgur.com/HhNHhPr.png)


数据链路层
- 负责物理层面上两个互连主机间的通信传输，将由 0 、 1 组成的比特流划分成数据帧传输给对端，即数据帧的生成与接收。
- 通信传输实际上是通过物理的传输介质实现的。
- 数据链路层的作用就是在这些通过传输介质互连的设备之间进行数据处理。

![Screen Shot 2021-05-24 at 06.09.25](https://i.imgur.com/cg9f2Yq.png)


> 网络层与数据链路层都是基于目标地址将数据发送给接收端的，
> 但是网络层负责将整个数据发送给最终目标地址，
> 而数据链路层则只负责发送一个分段内的数据。


物理层:
- 负责逻辑信号（比特流）与物理信号（电信号、光信号）之间的互相转换，
- 通过传输介质为数据链路层提供物理连接。

![Screen Shot 2021-05-24 at 06.09.32](https://i.imgur.com/Jup85j5.png)


---

# TCP & UDP, TCP/IP

由于 OSI 参考模型把服务划得过于琐碎，先定义参考模型再定义协议，有点理想化。
- TCP/IP 模型则正好相反，通过已有的协议归纳总结出来的模型，成为业界的实际网络协议标准。
- TCP/IP 是由 IETF 建议、推进其标准化的一种协议，
- 是 IP 、 TCP 、 HTTP 等协议的集合。
- TCP/IP 是为使用互联网而开发制定的协议族，所以**互联网的协议**就是 TCP/IP 。
- shorter version of the OSI model.

![TCP1](https://i.imgur.com/wPGfs9K.jpg)

- The ISO developed the OSI reference model to be generic, in terms of what protocols and technologies could be categorized by the model.

- DoD model  / TCP/IP stack:
  - Most traffic on the networks based on the TCP/IP protocol suite.
  - a more relevant model is developed by the United States Department of Defense (DoD). The DoD model / TCP/IP stack.

- Network Control Protocol (NCP):
  - An older protocol, similar to the TCP/IP protocol suite,
  - a protocol used on ARPANET (the predecessor to the Internet),
  - provided features similar to (not as robust) to TCP/IP suite of protocols.

- TCP/IP suit:
  - `a suite of protocols` that controls the way information travels from location to location
  - 6 protocols generally serve as the foundation of the TCP/IP suite:
    - `IP, DNS, TCP,UDP, ICMP and ARP`.
￼

## layers in TCP/IP

![Screen Shot 2021-05-24 at 06.18.52](https://i.imgur.com/i9Exmr6.png)

![Screen Shot 2021-05-24 at 10.12.53](https://i.imgur.com/UVegMDU.png)

应用层设备有电脑、手机、服务器等。应用层设备不转发数据，它们是数据的源或目的，拥有应用层以下的各层功能。
- 发送数据时，从上而下的顺序，逐层对数据进行封装，再通过以太网将数据发送出去。
- 接收数据时，从下而上的顺序，逐层对数据进行解封装，最终恢复成原始数据。

数据链路层设备有二层交换机、网桥等。
- 二层网络设备只转发数据，通过识别数据的 MAC 地址进行转发。
- 二层交换机接收数据后，对数据最外层封装的以太网头部信息进行查看，看到数据的目的 MAC 地址后，把数据帧从对应端口发送出去。
- 交换机并不会对数据帧进行解封装，只要知道 MAC 地址信息就可以正确地将数据转发出去。

网络层设备有路由器、三层交换机等。
- 三层网络设备只转发数据，通过识别数据的 IP 地址进行转发。
- 路由器接收数据后，首先查看最外层封装的以太网头部信息，当目的 MAC 地址是自己时，就会将以太网头部解封装，查看数据的 IP 地址。
- 根据 IP 路由表做出转发决定时，路由器会把下一跳设备的 MAC 地址作为以太网头部的目的 MAC 地址，重新封装以太网头部并将数据转发出去。
- 转发数据的网络设备和应用层的数据，就像快递员和包裹一样。快递员根据目的地址运送包裹，不必了解包裹里的具体内容。


### 4. **Application Layer**:
- `representation, encoding and dialog control issues`.
- TCP/IP stack vs OSI model: biggest difference is in application layer.
  - Maps to Layers 5, 6, and 7 (the session, presentation, and application layers) of the OSI model.
- 相当于 OSI 模型中的第 5 - 7 层的集合，不仅要实现 OSI 模型应用层的功能，还要实现会话层和表示层的功能。
- TCP/IP 应用的架构绝大多数属于客户端/服务端模型。
  - 提供服务的程序叫服务端， 接受服务的程序叫客户端。
  - 客户端可以随时发送请求给服务端。
- HTTP 、 POP3 、 TELNET 、 SSH 、 FTP 、 SNMP都是应用层协议。
  - HTTP: WWW 浏览器和服务器之间的应用层通信协议，所传输数据的主要格式是 HTML 。HTTP 定义高级命令或者方法供浏览器用来与Web服务器通信。
  - POP3: 简单邮件传输协议，邮件客户端和邮件服务器使用。
  - TELNET 和 SSH: 远程终端协议，用于远程管理网络设备。TELNET 是明文传输， SSH 是加密传输。
  - SNMP: 简单网络管理协议，用于网管软件进行网络设备的监控和管理。
  - ![Screen Shot 2021-05-24 at 06.33.42](https://i.imgur.com/XrzJWOi.png)

### 3. **Host-to-Host**
- `application data segmentation, transmission reliability, flow and error control`.
- Host-to-Host protocol in the TCP/IP model provides more or less the same services with its equivalent Transport protocol in the OSI model.
- 相当于 OSI 模型中的第 4 层传输层，主要功能就是让应用程序之间互相通信，通过端口号识别应用程序，使用的协议有面向连接的 TCP 协议和面向无连接的 UDP 协议。

**Host-to-Host Layer Protocols**
- TCP 面向连接
  - **Connection-oriented protocol**
  - 是在发送数据之前， 在收发主机之间连接一条逻辑通信链路, 能够对自己提供的连接实施控制。
  - 适用于要求可靠传输的应用，例如文件传输。
  - 好比平常打电话，输入完对方电话号码拨出之后，只有对方接通电话才能真正通话，通话结束后将电话机扣上就如同切断电源。
  - reliable connection stream between two nodes
    - Protocols operate by acknowledging / confirming ever connection request or transmission.
    - Overhead is more and the performance is less.
  - Consists of set up, transmission, and tear down phases
  - Creates virtual circuit-switched network

- UDP 面向无连接
  - **Connectionless protocol**
  - 不要求建立和断开连接。发送端可于任何时候自由发送数据, 不会对自己提供的连接实施控制。
  - 适用于实时应用，例如：IP电话、视频会议、直播等。
  - 如同去寄信，不需要确认收件人信息是否真实存在，也不需要确认收件人是否能收到信件，只要有个寄件地址就可以寄信了。
  - Do not require an acknowledge:
    - Faster, lack of requirement.
  - Sends data out as soon as there is enough data to be transmitted


### 2. **Internet**
- `route packets to their destination independent of the path taken`
- Internet layer in TCP/IP model provides the same services as the OSIs Network layer.
- 相当于 OSI 模型中的第 3 层网络层，使用 IP 协议。
  - IP 协议基于 IP 地址转发分包数据，作用是将数据包从源地址发送到目的地址。
- TCP/IP 分层中的网络层与传输层的功能通常由操作系统提供。 路由器就是通过网络层实现转发数据包的功能。
- 网络传输中，每个节点会根据数据的地址信息，来判断该报文应该由哪个网卡发送出去。
  - 各个地址会参考一个发出接口列表， MAC 寻址中所参考的这张表叫做 MAC 地址转发表，而 IP 寻址中所参考的叫做路由控制表。
  - MAC 地址转发表根据自学自动生成。
  - 路由控制表则根据路由协议自动生成。
  - MAC 地址转发表中所记录的是实际的 MAC 地址本身，而路由表中记录的 IP 地址则是集中了之后的网络号（即网络号与子网掩码)
    - IP
      - IP 是跨越网络传送数据包，使用 IP 地址作为主机的标识，使整个互联网都能收到数据的协议。
      - IP 协议独立于底层介质，实现从源到目的的数据转发。
      - IP 协议不具有重发机制，属于非可靠性传输协议。
    - ICMP
      - 用于在 IP 主机、路由器之间传递控制消息，用来诊断网络的健康状况。
    - ARP
      - 从数据包的 IP 地址中解析出 MAC 地址的一种协议。



### 1. **Network Access**
- `physical issues concerning data termination on network media`
- includes all the concepts of the data link and physical layers of the OSI model for both LAN and WAN media.
- TCP/IP 是以 OSI 参考模型的物理层和数据链路层的功能是透明的为前提制定的，并未对这两层进行定义，所以可以把物理层和数据链路层合并称为网络接入层。
- 网络接入层是对网络介质的管理，定义如何使用网络来传送数据。
- 但是在通信过程中这两层起到的作用不一样，所以也有把物理层和数据链路层分别称为**硬件、网络接口层**。
- TCP/IP 分为四层或者五层都可以，只要能理解其中的原理即可。
- 设备之间通过物理的传输介质互连， 而互连的设备之间使用 MAC 地址实现数据传输。
- 采用 MAC 地址，目的是为了识别连接到同一个传输介质上的设备。

---

## Common Application Protocols in the TCP/IP Stack

![TCP2](https://i.imgur.com/TM6p21I.jpg)

- Application layer protocols in the TCP/IP stack: identifiable by port numbers.
- example:
    - when enter web address in browser, by default communicating with web address, uses TCP port 80. HTTP
    - data send to remote web server has `destination port 80`.
    - **That data is then encapsulated into a TCP segment** at the transport layer.
    - **That segment is then encapsulated into a packet** at the Internet layer,
    - **the packet is sent out on the network** using an underlying network interface layer technology (like Ethernet).
      - the packet:
      - destination IP address of the web server, destination port 80
      - IP address of your computer, your computer selects a port number greater than 1023. (Because your computer is not acting as a web server, its port is not 80.)
    - when the web server sends content back to the PC, the data is destined for the PC’s IP address and for the port number PC selected.
    - With both source and destination port numbers, source and destination IP addresses, two-way communication becomes possible.
    - well-known ports: ports numbered below 1023
    - ephemeral ports: ports numbered above 1023
    - The maximum value of a port is 65,535.

IP Vulnerabilities
- Unencrypted transmission
  - Eavesdropping possible at any intermediate host during routing
- No source authentication
  - Sender can spoof source address, making it difficult to trace packet back to attacker
- No integrity checking
  - Entire packet, header and payload, can be modified while en route to destination, enabling content forgeries, redirections, and man-in-the-middle attacks
- No bandwidth constraints
  - Large number of packets can be injected into network to launch a denial-of-service attack
  - Broadcast addresses provide additional leverage

---

## 封装与解封装

通常，为协议提供的信息为包头部，所要发送的内容为数据。

每个分层中，都会对所发送的数据附加一个头部
- 在这个头部中包含了该层必要的信息， 如发送的目标地址以及协议相关信息。
- 在下一层的角度看，从上一分层收到的包全部都被认为是本层的数据。
- 数据发送前，按照参考模型从上到下，在数据经过每一层时，添加协议报文头部信息，这个过程叫封装。

![Screen Shot 2021-05-24 at 06.45.10](https://i.imgur.com/tkqG495.png)

- 数据接收后，按照参考模型从下到上，在数据经过每一层时，去掉协议头部信息，这个过程叫解封装。

![Screen Shot 2021-05-24 at 06.47.07](https://i.imgur.com/vRdP7Ai.png)

1. 经过传输层协议封装后的数据称为**段**
2. 经过网络层协议封装后的数据称为**包**
3. 经过数据链路层协议封装后的数据称为**帧**
4. 物理层传输的数据为**比特**。

TCP/IP 通信中使用 MAC 地址、 IP 地址、端口号等信息作为地址标识。甚至在应用层中，可以将电子邮件地址作为网络通信的地址。

---

## TCP vs UDP

Two protocols: `Transmission Control Protocol (TCP)` and `User Datagram Protocol (UDP)` are defined for transmitting datagrams.
1. UDP can be much faster than TCP, which often requires retransmissions and delaying of packets.
2. **UDP is often used**
   - `time-sensitive applications`
     - where data integrity is not as important as speed
     - Short client-server request like DNS, single message request
     - Voice over IP (VoIP).
     - High-perfomece networking
     - Application handles reliable transmission
   - Primary use: `send small packets of information`.
3. **TCP is used for**
   -  applications where `data order and data integrity is important`
     - like HTTP, SSH, and FTP.

![TCP7](https://i.imgur.com/pwrzleA.jpg)

---

## Transmission Control Protocol (TCP)
- transport layer protocol
- distinguish data for multiple concurrent applications on the same host.
- Most popular application protocols, like WWW, FTP and SSH are built on top of TCP

If a process needs to send a complete file to another computer
1. chopping it into IP packets
2. sending them to the other machine
3. Double-checking that all the packets made it intact 完整的
4. resending any that were lost

### TCP three-way handshake
1. TCP session: starts by establishing a communication connection between sender and receiver.
2. the client sends a `SYN (synchronize) packet`.
3. The server responds with a `SYN/ACK (synchronize/acknowledge) packet`
4. the client completes the handshake with an `ACK packet` to establish the connection.
5. Once connection created, the parties communicate over the established channel.

![Image2](https://i.imgur.com/n2fQFBa.jpg)


### TCP Features
- **connection-oriented traffic**
- **delivery messages in order**
  - guaranteeing reliable data transfer
  - requiring positive `acknowledgment numbers` of delivery
  - ensures reliable transmission by using `sequence number` (three-way handshake)
  - each party is aware when packets arrive out of order or not at all.
- **Congestion 拥挤 Control**:
  - prevent overwhelming a network with traffic
    - cause poor transmission rates and dropped packets
  - not implemented into TCP packets specifically
    - but based on information gathered by keeping track of `acknowledgments` for previously sent data and the `time required `for certain operations.
  - TCP adjusts
    - `data transmission rates` using this information to prevent network congestion.
    - incorporates a cumulative 累积的 acknowledgment scheme.
      - sender and receiver, communicating via established TCP connection.
      - sender sends the receiver a specified amount of data,
      - the receiver confirms that it has received the data: sending a response packet to the sender with the acknowledgment field set to the next sequence number it expects to receive.
      - If any information has been lost, then the sender will retransmit it.
- **provide flow control**
  - distinguish data for multiple concurrent applications on the same host.
    - like WWW, FTP and SSH are built on top of TCP
  - handle startups and shutdowns
  - manages the amount of data that can be sent by one party, avoiding overwhelming the processing resources of the other / the bandwidth of the network itself.
  - 2 common flow control approaches: `Sliding window` and `Buffering`.
- **Error checking**
  - provides error checking with `checksums`

  - not be cryptographically secure, but to detect inconsistencies in data due to network errors rather than malicious tampering.
  - This checksum is typically supplemented by an additional checksum at the link layer
    - such as Ethernet, which uses the CRC-32 checksum.
  - ensure correctness of data.
    - comparing a checksum of the data with a checksum encoded in the packet

### Sliding window protocol:
- TCP uses it to manage flow control.
  - Sliding window used by the receiver to slow down the sender
- Congestion control: TCP can react to changing network conditions (slow down/speed up dens rates)
  - Does it by adjust the congestion widonw size of the sender.
  - congestion widonw size used by the sender to reduce the network congestion
  - Various techniques exist: back off, then add load
  - Unacknowledged semgment sent send at a point of time,
￼￼
- Serves 3 different roles
  - Reliable
  - Preserve the order 保留順序(過程亂序，但回傳給上層是照順序的)
  - Each frame has a sequence number
  - Delivery order is maintained by sequence number
  - The receiver does not pass a frame up to the next higher-level protocol until it has already passed up all frames with a smaller sequence number
  - Frame control
  - Receiver throttle 节流 the sender, 用RWS來控制
  - Keeps the sender from overrunning the receiver
  - transmitting more data than the receiver is able to process

### Buffering:
- device (like router) allocates a chunk of memory (buffer/queue) to store segments if bandwidth is not currently available to transmit those segments.
  - A queue 队列 has finite capacity, will overflow (drop segments) in the event of sustained network congestion.


### TCP Packet Format
```
Frame 1: Physical layer
Ethernet II, Src: Data layer
Internet Protocol Version 4, Src: 192.168.1.100, Dst: 192.168.1.1 Network layer
Internet Control Message Protocol Transport layer
```

![TCP4](https://i.imgur.com/yxqDZTd.jpg)

- Includes source and destination ports, define the communication connection for this packet and others like it.
- In TCP, connection sessions are maintained beyond the life of a single packet,
  - so TCP connections have a state, defines the status of the connection.
  - TCP communication session, goes from states used to open a connection, to those used to exchange data and acknowledgments, to those used to close a connection.
- TCP is a byte-oriented protocol
  - the sender writes bytes into a TCP connection
  - the receiver reads bytes out of the TCP connection.

- Byte stream:
  - but TCP does not transmit individual bytes,
  - The source host buffers enough bytes from the sender to fill a reasonably sized packet, and then sends this packet to its peer on the destination host.
  - The destination host then empties the contents of the packet into a receiver buffer, and the receiving process reads from this buffer at its leisure.
- The packets exchanged between TCP peers are segments.
- Buffer: make sure every is in order.
- 2 incarnations of the same connection: closed and open a same TCP connection.

![Screen Shot 2019-02-09 at 09.47.12](https://i.imgur.com/me6koF4.png)

TCP Header

![TCP3](https://i.imgur.com/83zsF4g.jpg)

- TCP packet header: (4x6)24 bytes, 64 bits.
- `SrcPor / DstPort`:
  - the source / destination ports.
  - to which upper-layer protocol data should be forwarded,
  - from which upper-layer protocol the data is being sent.
- `SequenceNum`:
  - Identify the position in the segmented data stream
  - contains the sequence number for the first byte of data carried in that segment.
  - if segments arrive out of order, the recipient can put them back in the appropriate order.
- `AcknowledgmentNum`:
  - the next sequence number expected to receive.
  - The number of the next octet to be received in the data stream
  - This is a way for the receiver to let the sender know that all segments up to and including that point have been received.
- `HeaderLength`:
  - The TCP header is of variable length, HdrLen gives the length of the header in 32-bit words. Also known as Offset field.
- `AdvertisedWindow`:
  - how many bytes a device can receive before expecting an acknowledgment.
  - offers flow control.
- `Flags`:
  - relay control information between TCP peers. Used to determine the conditions and status of the TCP connection.
  - Possible flags: SYN, FIN, RESET, PUSH, URG, and ACK.
  - URG flag: contains urgent data. The urgent data is contained at the front of the segment body, and including a UrgPtr bytes into the segment. Indicates that the data contained in the packet is urgent and should process immediately.
  - UrgPtr field: Urgent pointer, indicates where the non-urgent data contained in this segment begins.
  - ACK flag: set any time the Acknowledgment field is valid, implying that the receiver should pay attention to it. Acknowledge the receipt of a packet.
  - PSH / PUSH flag: the sender invoked the push operation, which indicates to the receiver that it should notify the receiving process of this fact. Instructs the sending system to send all buffered data immediately.
  - RST / RESET flag: the receiver has become confused, it received a segment it did not expect to receive, wants to abort the connection. Reset a connection.
  - SYN / FIN flags:
  - SYN: Initiates a connection between two hosts to facilitate communication.
  - FIN: Tells the remote system about the end of the communication. In essence, this gracefully closes a connection.
- `Checksum`:
  - computed over the TCP header, data, and pseudo-header, which is made up of the source address, destination address, and length fields from the IP header.
- The `Checksum` field is used
  - provide extra reliability and security to the TCP segment.
- The `actual user data`
  - included after the end of the header.

TCP options must be in multiples of four (4) bytes. If 10 bytes of options are added to the TCP header, two single byte No-Operation will be added to the options header.

---

## User Datagram Protocol (UDP)

![TCP6](https://i.imgur.com/uCCDhzx.jpg)

UDP is a good for applications need maximize bandwidth and do not require acknowledgments (example, audio or video streams).
- UDP is considered to be a connectionless, unreliable protocol,
1. no initial handshake to establish a connection
   - no handshake, harder to scan and enumerate.
   - allows parties to send messages, known as datagrams, immediately.
   - If a sender wants to communicate via UDP, it need only use a socket (defined with respect to a port on a receiver) and start sending datagrams, no elaborate setup needed.
2. features a 16-bit `checksum`
   - to verify the integrity of each individual packet
3. no sequence number scheme, window size, and acknowledgment numbering present in the header of a TCP segment.
   - transmissions can arrive out of order or may not arrive at all.
   - does not make guarantee about the order or correctness of packet delivery.
   - UDP header is so much smaller than a TCP header,
- It is assumed that checking for missing packets in a sequence of datagrams is left to applications processing these packets.

![TCP5](https://i.imgur.com/Guu1ilc.jpg)

---

# 实际数据传输举例

互联网是使用的 TCP/IP 协议进行网络连接的。

![Screen Shot 2021-05-24 at 06.50.21](https://i.imgur.com/NGTJOUj.png)

以访问网站为例:

1. 发送数据包
   - 访问 HTTP 网站页面时，打开浏览器，输入网址，敲下回车键就开始进行 TCP/IP 通信了。
   - 应用程序处理
     - ![Screen Shot 2021-05-24 at 06.51.30](https://i.imgur.com/qgVepAa.png)
     - 首先，应用程序中会进行 HTML 格式编码处理，相当于 OSI 的表示层功能。
     - 编码转化后，不一定会马上发送出去，相当于会话层的功能。
     - 在请求发送的那一刻，建立 TCP 连接，然后在 TCP 连接上发送数据。
     - 接下来就是将数据发送给下一层的 TCP 进行处理。
   - TCP 模块处理
     - ![Screen Shot 2021-05-24 at 06.52.55](https://i.imgur.com/HiiOoQe.png)
     - TCP 会将应用层发来的数据顺利的发送至目的地。
     - 实现可靠传输的功能，需要给数据封装 TCP 头部信息。
     - TCP 头部信息包括源端口号和目的端口号（识别主机上应用）、序号（确认哪部分是数据）以及校验和（判断数据是否被损坏）。
     - 随后封装了 TCP 头部信息的段再发送给 IP 。
   - IP 模块处理
     - ![Screen Shot 2021-05-24 at 06.55.39](https://i.imgur.com/FUGC9cT.png)
     - IP 将 TCP 传过来的数据段当做自己的数据，并封装 IP 头部信息。
     - IP 头部信息中包含目的 IP 地址和源 IP 地址，以及上层协议类型信息。
     - IP 包生成后，根据主机路由表进行数据发送。
   - 网络接口处理
     - ![Screen Shot 2021-05-24 at 06.56.22](https://i.imgur.com/QsR8lST.png)
     - 网络接口对传过来的 IP 包封装上以太网头部信息并进行发送处理。
     - 以太网头部信息包含目的 MAC 地址、源 MAC 地址，以及上层协议类型信息。
     - 然后将以太网数据帧通过物理层传输给接收端。
     - 发送处理中的 FCS 由硬件计算， 添加到包的最后。
     - 设置 FCS 的目的是为了判断数据包是否由于噪声而被破坏。
2. <font color=red> 接收数据包 </font>
   - 包的接收流程是发送流程的反向过程。
   - <font color=blue> 网络接口处理 </font>
     - 收到以太网包后，首先查看头部信息的目的 <font color=blue> MAC 地址是否是发给自己的包 </font>
       - 如果不是发送给自己的包就丢弃。
       - 如果是发送给自己的包
         - 查看上层协议类型是 IP 包，以太网帧解封装成 IP 包，传给 IP 模块进行处理。
         - 如果是无法识别的协议类型，则丢弃数据。
   - <font color=blue> IP  模块处理 </font>
     - 收到 IP 包后，进行类似处理。
     - 根据头部信息的目的 <font color=blue> IP 地址判断是否是发送给自己包 </font>
       - 如果是发送给自己的包，则查看上一层的协议类型。
         - 上一层协议是 TCP ，就把 IP 包解封装发送给 TCP 协议处理。
       - 假如有路由器，且接收端不是自己的地址，那么根据路由控制表转发数据。
   - <font color=blue> TCP 模块处理 </font>
     - 收到 TCP 段后
       - 首先查看**校验和**，<font color=blue> 判断数据是否被破坏 </font>
       - 然后检查是否按照**序号**接收数据。
       - 最后检查**端口号**，确定具体的应用程序。
     - 数据接收完毕后，发送一个 “ 确认回执 ” 给发送端。
       - 如果这个回执信息未能达到发送端，那么发送端会认为接收端没有接收到数据而一直反复发送。
     - 数据被完整接收后，会把 TCP 段解封装发送给由端口号识别的应用程序。
   - 应用程序处理
     - 应用程序收到数据后，通过解析数据内容获知发送端请求的网页内容
     - 然后按照 HTTP 协议进行后续数据交互。

---

# 网络构成

## 网络构成要素
搭建一套网络涉及各种线缆和网络设备。下面介绍一些常见的硬件设备。
- 硬件设备所说的层数是参照的 OSI 参考模型，而不是 TCP/IP 模型。

![Screen Shot 2021-05-24 at 09.59.39](https://i.imgur.com/SsNIr7O.png)

![Screen Shot 2021-05-24 at 10.00.24](https://i.imgur.com/krITx4L.png)

## 通信介质与数据链路
设备之间通过线缆进行连接。
- 有线线缆有双绞线、光纤、串口线等。根据数据链路不同选择对应的线缆。
- 传输介质还可以被分为电波、微波等不同类型的电磁波。

传输速率
- 单位为 bps
- 是指单位时间内传输的数据量有多少。
- 又称作带宽，带宽越大网络传输能力就越强。

吞吐量
- 单位为 bps
- 主机之间实际的传输速率。
- 吞吐量这个词不仅衡量带宽， 同时也衡量主机的 CPU 处理能力、 网络的拥堵程度、 报文中数据字段的占有份额等信息。


网卡
- 任一主机连接网络时，必须要使用网卡。
- 可以是有线网卡，用来连接有线网络
- 也可以是无线网卡连接 WiFi 网络。
- 每块网卡都有一个唯一的 MAC 地址，也叫做硬件地址或物理地址。

![Screen Shot 2021-05-24 at 10.03.54](https://i.imgur.com/ZYFOnxW.png)


二层交换机
- 二层交换机位于 OSI 模型的第 2 层（数据链路层）。
- 它能够识别数据链路层中的数据帧，并将帧转发给相连的另一个数据链路。
- 数据帧中有一个数据位叫做 FCS ，用以校验数据是否正确送达目的地。
-   二层交换机通过检查这个值，将损坏的数据丢弃。
- 二层交换机根据 MAC 地址自学机制判断是否需要转发数据帧。

![Screen Shot 2021-05-24 at 10.04.33](https://i.imgur.com/02Sxedv.png)


路由器 / 三层交换机
- 路由器是在 OSI 模型的第 3 层（网络层）上连接两个网络、并对报文进行转发的设备。
  - 二层交换机是根据 MAC 地址进行处理，
  - 而路由器 / 三层交换机则是根据 IP 地址进行处理的。
- 因此 TCP/IP 中网络层的地址就成为了 IP 地址。
- 路由器可以连接不同的数据链路。
  - 比如连接两个以太网，或者连接一个以太网与一个无线网。
  - 家庭里面常见的无线路由器也是路由器的一种。

![Screen Shot 2021-05-24 at 10.11.07](https://i.imgur.com/j23NUG5.png)


负载均衡设备 / 四至七层交换机
- 四至七层交换机负责处理 OSI 模型中从传输层至应用层的数据。
- 以 TCP 等协议的传输层及其上面的应用层为基础，分析收发数据，并对其进行特定的处理。
  - 例如
  - 视频网站的一台服务器不能满足访问需求，通过**负载均衡设备**将访问分发到后台多个服务器上，就是四至七层交换机的一种。
  - 还有带宽控制、广域网加速器、防火墙等应用场景。

![Screen Shot 2021-05-24 at 10.12.08](https://i.imgur.com/kHz27MI.png)
