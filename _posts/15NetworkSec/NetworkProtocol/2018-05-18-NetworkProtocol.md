---
title: Network Protocol Port Number
# author: Grace JyL
date: 2018-05-18 11:11:11 -0400
description:
excerpt_separator:
categories: [15NetworkSec, NetworkProtocol]
tags: [NetworkSec, NetworkProtocol, PortNumber]
math: true
# pin: true
toc: true
image: /assets/img/note/tls-ssl-handshake.png
---



# Network Protocol

- [Network Protocol](#network-protocol)
  - [Protocol List](#protocol-list)


---

## Protocol List

* ICMP has no ports and is neither TCP nor UDP.

number| TCP/UDP | protocol | use
------|---------|----------|---
`0`   | TCP,UDP | 保留端口；不使用（若发送过程不准备接受回复消息，则可以作为源端口）	官方
------|---------|
`20`  | TCP,UDP | *FTP* - File Transfer Protocol **默认数据端口 DATA**	官方
`21`  | TCP,UDP | *FTP* - File Transfer Protocol **控制端口 Control**	官方
`22`  | TCP,UDP | *SSH* - Secure Shell	远程登录协议，用于安全登录文件传输（SCP，SFTP）及端口重新定向	官方
------|---------| secure replacement for Telnet	官方
------|---------| SFTP runs over an SSH session on TCP port 22.
------|---------| Secure copy protocol (SCP) uses port 22.
------|---------| a secure shell (ssh) utility for securely copying files between hosts.
`26`  | TCP,UDP | RSFTP - 一个简单的类似FTP的协议	非官方
`989` | TCP,UDP | FTP Protocol (data) over TLS | SSL	官方
`990` | TCP,UDP | FTPS (FTP over SSL)	FTP Protocol (control) over TLS | SSL	官方
`5004`| UDP     | RTP, SRTP - Secure Real-time Transport Protocol	实时传输协议实时传输协议
------|---------| encryption, message authentication, and integrity for streaming media
`69`  | UDP	    | TFTP Trivial File Transfer Protocol, 小型文件传输协议（小型文件传输协议）
------|---------| port 69 is open, risk: Unauthenticated access
------|---------|
`23`  | TCP,UDP | *Telnet* 终端仿真协议 - 未加密文本通信	官方
`107` | TCP     | 远程Telnet协议 官方
`992` | TCP,UDP | 基于TLS/SSL的Telnet协议
------|---------|
`25`  | TCP,UDP | *SMTP* - Simple Mail Transfer Protocol （简单邮件传输协议） 用于邮件服务器间的电子邮件传递	官方
------|---------| a communication protocol for electronic mail transmission. Mail servers, message transfer agents use SMTP to send and receive mail messages. Proprietary systems (Microsoft Exchange, IBM Notes) and webmail systems (Outlook, Gmail) may use non-standard protocols internally, but all use SMTP when sending to or receiving email from outside their own systems.
`57`  | TCP     | MTP - Mail Transfer Protocol	MTP，邮件传输协议	官方
`465` | TCP     | SMTP with SSL	传输层安全性协议加密的简单邮件传输协议	非官方
`587` | TCP     | SMTP with TLS	邮件消息提交（简单邮件传输协议，RFC 2476）	官方
`109` | TCP     | POP2 - Post Office Protocol 	POP (Post Office Protocol)，“邮局协议”，第2版	官方
`110` | TCP     | *POP3* - Post Office Protocol - 用于接收电子邮件	官方
------|---------| POP3 client contacts the POP3 server to receive mail.
`995` | TCP     | POP3S - Post Office Protocol over TLS | SSL	基于 传输层安全性协议的邮局协议 (加密传输)
`143` | TCP,UDP | IMAP4 - Internet Message Access Protocol 因特网信息访问协议 4 - 用于检索 电子邮件s	官方
------|---------| used by email clients to retrieve email messages from a mail server over a TCP/IP connection.
`993` | TCP     | IMAPS - IMAP over SSL | TLS	基于 传输层安全性协议的因特网信息访问协议 (加密传输)	官方
------|---------|
37 | TCP,UDP | TIME	TIME时间协议	官方
39 | TCP,UDP | Resource Location Protocol（资源定位协议）	官方
41 | TCP,UDP | 图形	官方
42 | TCP,UDP | WINS（WINS主机名服务）	非官方
43 | TCP     | WhoIs	WHOIS协议	官方
50 | - | IPSec: Encapsulating Security Payload (ESP)
51 | _ | IPSec: Authentication Header (AH)
------|---------|
`389` | TCP,UDP | *LDAP* - Lightweight Directory Access 	轻型目录访问协议 LDAP	官方
`636` | TCP,UDP | LDAPS - LDAP over SSL	LDAP over SSL（加密传输）	官方
`3868`| TCP,UDP | Diameter	Diameter base protocol	官方
`49`  | TCP,UDP | TACACS+ Login - Login Host Protocol	TACACS登录主机协议	官方
`88`  | TCP	    | *Kerberos*	Kerberos 认证代理	官方
`464` | TCP,UDP | Kerberos	Kerberos 更改, 设定密码	官方
------|---------|
42 | TCP,UDP | 	Nameserv - Host Name Server	Host Name Server ARPA主机名服务器协议	官方
------|---------| 已过时的用于转换主机名到互联网地址的网络协议。
53 | TCP,UDP | 	DNS - Domain Name System	DNS（域名服务系统）	官方
------|---------|
56 | TCP,UDP | 		远程访问协议	官方
67 | UDP	| DHCP - Dynamic Host Configuration Protocol	BOOTP（BootStrap协议）服务；同时用于动态主机设置协议	官方
68 | UDP	| DHCP - Dynamic Host Configuration Protocol	BOOTP客户端；同时用于动态主机设定协议	官方
546 | TCP,UDP | 	DHCP Client	DHCPv6客户端	官方
547 | TCP,UDP | 	DHCP Server	DHCPv6服务器	官方
------|---------|
`80`  | TCP   	| HTTP -  HyperText Transfer Protocol (HTTP)	超文本传输协议（超文本传输协议）- 用于传输网页	官方
`443` | TCP	    | HTTPS - HTTP over TLS | SSL	超文本传输安全协议 - 超文本传输协议 over TLS | SSL（加密传输）	官方
------|---------|
`118` | TCP,UDP | *SQL Services*	SQL服务	官方
`156` | TCP,UDP | *SQL Server*	SQL服务	官方
`1433`| TCP,UDP | *Microsoft SQL* *MSSQL* 数据库系统	官方
`1434`| TCP,UDP | *Microsoft SQL* 活动监视器	官方
`1521`| TCP     | *Oracle* 数据库 default listener, in future releases official port 2483	非官方
`1526`| TCP     | Oracle数据库 common alternative for listener	非官方
`2483`| TCP,UDP | Oracle listening port for insecure client connections to the listener, replaces port 1521
`2484`| TCP,UDP | Oracle listening port for SSL client connections to the listener	官方
`3306`| TCP,UDP | *MySQL* 数据库系统	官方
------|---------|
------|---------|
70 | TCP |	Gopher Services	Gopher	官方
`79`  | TCP     |	*Finger*	手指协议	官方
------|---------|
81 | TCP     | XB Browser - Tor	非官方
82 | UDP | XB Browser - 控制端口	非官方
---|---|---|---
101 | TCP     | 主机名	官方
102 | TCP     | ISO-TSAP协议	官方
111 | TCP,UDP | 		Sun远程过程调用协议	官方
113 | TCP     | Ident - 旧的服务器身份识别系统，仍然被IRC服务器用来认证它的用户	官方
------|---------|
117 | TCP     | UNIX间复制协议（Unix to Unix Copy Protocol，UUCP）的路径确定服务	官方
------|---------|
119 | TCP	| NNTP - Network News Transport Protocol 	网络新闻传输协议 - 用来收取新闻组的消息	官方
563 | TCP,UDP | 	NNTPS	网络新闻传输协议 通过安全套接字层的网络新闻传输协议（NNTPS）	官方
------|---------|
`123` | UDP	    | *NTP* - Network Time Protocol	用于时间同步	官方
------|---------|
135 | TCP,UDP | 		分布式运算环境（Distributed Computing Environment，DCE）终端解决方案及定位服务	官方
------|---------|
`135` | TCP,UDP | *NetBIOS*	微软终端映射器（End Point Mapper，EPMAP）	官方
`137` | TCP,UDP | *NetBIOS* Name Service	NetBIOS NetBIOS 名称服务	官方
`138` | TCP,UDP | *NetBIOS*	NetBIOS NetBIOS 数据报文服务	官方
`139` | TCP,UDP | *NetBIOS* Datagram Service	NetBIOS NetBIOS 会话服务	官方
`150` | ------- | *NetBIOS* Session Service
`445` | TCP	    | *Microsoft-DS*	Microsoft-DS (Active Directory，Windows 共享, 震荡波蠕虫，Agobot, Zobotworm)	官方
`445` | UDP     | *Microsoft-DS* 服务器消息块 SMB文件共享	官方
------|---------|
152 | TCP,UDP | 		BFTP, 后台文件传输程序	官方
153 | TCP,UDP | 		简单网关监控协议（Simple Gateway Monitoring Protocol，SGMP）	官方
158 | TCP,UDP | 		DMSP, 分布式邮件服务协议	非官方
------|---------|
`161` | TCP,UDP | *SNMP* - Simple Network Management Protocol	简单网络管理协议 官方
`162` | TCP,UDP | *SNMP* - Simple Network Management Protocol	SNMP协议的TRAP操作	官方
------|---------| SNMP is a connectionless protocol that uses UDP instead of TCP packets
------|---------|
170 | TCP     | 打印服务	官方
179 | TCP	| BGP - Border Gateway Protocol	边界网关协议 (边界网关协议)	官方
190	| - | Gateway Access Control Protocol (GACP)
194 | TCP	| Internet Relay Chat (IRC)	IRC（互联网中继聊天）	官方
197	| - | Directory Location Service (DLS)
201 | TCP,UDP | 		AppleTalk 路由维护	官方
209 | TCP,UDP | 		Quick Mail 传输协议	官方
213 | TCP,UDP | 		IPX，互联网分组交换协议	官方
218 | TCP,UDP | 		MPP，信息发布协议	官方
220 | TCP,UDP | 		因特网信息访问协议，交互邮件访问协议第3版	官方
259 | TCP,UDP | 		ESRO, Efficient Short Remote Operations	官方
264 | TCP,UDP | 		BGMP，边界网关多播协议	官方
308 | TCP     | Novastor 在线备份	官方
311 | TCP     | Apple 服务器管理员工具、工作组管理	官方
318 | TCP,UDP | 		TSP，时间戳协议	官方
323 | TCP,UDP | 		IMMP, Internet消息映射协议	官方
383 | TCP,UDP | 		HP OpenView HTTPs 代理程序	官方
366 | TCP,UDP | 		SMTP, 简单邮件传送协议. ODMR,按需邮件传递	官方
369 | TCP,UDP | 		Rpc2 端口映射	官方
371 | TCP,UDP | 		ClearCase 负载平衡	官方
384 | TCP,UDP | 		一个远程网络服务器系统	官方
387 | TCP,UDP | 		AURP, AppleTalk 升级用路由协议	官方
---|---|---|---
396	| - | Novell Netware over IP
401 | TCP,UDP | 		不间断电源，不间断电源供应系统	官方
411 | TCP     | Direct Connect Hub 端口	非官方
412 | TCP     | Direct Connect 客户端—客户端 端口	非官方
427 | TCP,UDP | 		服务定位协议 (位置服务协议)	官方
444 | TCP,UDP | 	Simple Network Paging Protocol (SNPP)	SNPP，简单网络分页协议	官方
458	| - | Apple QuickTime
465 | TCP     | Cisco 专用协议	官方
475 | TCP     | tcpnethaspsrv（Hasp 服务, TCP | IP 版本）	官方
497 | TCP     | dantz 备份服务	官方
`500` | TCP,UDP | *IKE* - Internet 密钥交换	网络安全关系与密钥管理协议， 官方
502 | TCP,UDP | 		Modbus 协议	官方
512 | TCP     | exec, 远程进程执行	官方
512 | UDP | comsat 和 biff 命令：用于电子邮件系统	官方
513 | TCP     | login，登录命令	官方
513 | UDP | Who命令，查看当前登录计算机的用户	官方
------|---------|
`514` | TCP     | 远程外壳 protocol - 用于在远程计算机上执行非交互式命令，并查看返回结果	官方
`514` | UDP     | *remote shell / Syslog* 协议 - 用于系统登录	官方
`1080`| tcp     |	*Socks*	SOCKS代理	官方
`3389`| tcp     | *RDP - Remote Desktop Protocol*	远程桌面协议（RDP）	官方
`6665`| TCP     | *IRC*	官方
`6666`| TCP     |
`6667`| TCP     |
`6668`| TCP     |
`6669`| TCP     |
`6679`| TCP     | IRC SSL （安全互联网中继聊天） - 通常使用的端口	非官方
`6697`| TCP     | IRC SSL （安全互联网中继聊天） - 通常使用的端口	非官方
------|---------|
515 | TCP     | Line Printer Daemon protocol - 用于 LPD 打印机服务器	官方
517 | UDP | Talk	官方
518 | UDP | NTalk	官方
520 | TCP     | efs	官方
520 | UDP | Routing - 路由信息协议	官方
513 | UDP | 路由器	官方
524 | TCP,UDP | 		NetWare核心协议（NetWare 核心协议）用于一系列功能，例如访问NetWare主服务器资源、同步时间等	官方
525 | UDP | Timed，时间服务	官方
530 | TCP,UDP | 		远程过程调用	官方
531 | TCP,UDP | 		AOL 即时通信软件, IRC	非官方
532 | TCP     | netnews	官方
533 | UDP | netwall,用于紧急广播	官方
540 | TCP     | UUCP (Unix-to-Unix 复制协议)	官方
542 | TCP,UDP | 		商业 (Commerce Applications)	官方
543 | TCP     | klogin, Kerberos登陆	官方
544 | TCP     | kshell, Kerberos 远程shell	官方
548 | TCP     | 通过传输控制协议（TCP）的 Appletalk 文件编制协议（AFP(苹果归档协议))	官方
550 | UDP | new-rwho, new-who	官方
554 | TCP,UDP | 		即时流协议 即时流协定	官方
556 | TCP     | Brunhoff 的远程文件系统（RFS）	官方
560 | UDP | rmonitor, Remote Monitor	官方
561 | UDP | monitor	官方
569	| - | MSN
591 | TCP     | FileMaker 6.0（及之后版本）网络共享（HTTP的替代，见80端口）	官方
593 | TCP,UDP | 		HTTP RPC Ep Map（RPC over HTTP, often used by Distributed COM services and Microsoft Exchange Server）	官方
604 | TCP     | TUNNEL	官方
`631` | TCP,UDP | *Internet Printing Protocol (IPP)* 互联网打印协议
639 | TCP,UDP | 		MSDP, 组播源发现协议	官方
646 | TCP,UDP | 		LDP, 标签分发协议	官方
647 | TCP     | DHCP故障转移协议	官方
648 | TCP     | RRP( Registry Registrar Protocol)，注册表注册协议	官方
652 | TCP     | DTCP(Dynamic Tunnel Configuration Protocol)，动态主机设置协议	官方
654 | UDP | AODV(Ad hoc On-Demand Distance Vector)，无线自组网按需平面距离向量路由协议	官方
665 | TCP     | sun-dr, Remote Dynamic Reconfiguration	非官方
666 | UDP | 毁灭战士，计算机平台上的一系列第一人称射击游戏。	官方
674 | TCP     | ACAP(Application Configuration Access Protocol)，应用配置访问协议	官方
691 | TCP     | MS Exchange Routing	官方
692 | TCP     | Hyperwave-ISP
694 | UDP | 用于带有高可用性的聚类的心跳服务	非官方
695 | TCP     | IEEE-MMS-SSL
698 | UDP | OLSR(Optimized Link State Routing)，优化链路状态路由协议
699 | TCP     | Access Network
700 | TCP     | EPP, 可扩展供应协议
701 | TCP     | LMP,链路管理协议
702 | TCP     | IRIS over BEEP
706 | TCP     | SILC, Secure Internet Live Conferencing
711 | TCP     | TDP, 标签分发协议
712 | TCP     | TBRPF, Topology Broadcast based on Reverse-Path Forwarding
720 | TCP     | SMQP, Simple Message Queue Protocol
749 | TCP, UDP | kerberos-adm, Kerberos administration
750 | UDP | Kerberos version IV
782 | TCP     | Conserver serial-console management server
829 | TCP     | 证书管理协议（CMP）[2]
860 | TCP     | ISCSI，Internet小型计算机系统接口
873 | TCP     | Rsync ，文件同步协议	官方
901 | TCP     | Samba 网络管理工具(SWAT)	非官方
902 | --- | VMware服务器控制台[3]	非官方
904 | --- | VMware服务器替代（如果902端口已被占用）	非官方
911 | TCP     | Network Console on Acid（NCA） - local tty redirection over OpenSSH
981 | TCP     | Check Point Remote HTTPS management for firewall devices running embedded Checkpoint Firewall-1software	非官方
991 | TCP,UDP | 		NAS (Netnews Admin System)
1025 | TCP     | NFS-or-IIS	非官方
1026 | TCP     | 通常用于微软Distributed COM服务器	非官方
1029 | TCP     | 通常用于微软Distributed COM服务器	非官方
1058 | TCP     | nim IBM AIX	官方
1059 | TCP     | nimreg	官方
1099 | TCP,UDP  | Java远程方法调用 Registry	官方
1109 | TCP     | Kerberos POP
1140 | TCP     | AutoNOC	官方
1167 | UDP     | phone, conference calling
1176 | TCP     | Perceptive Automation Indigo home control server	官方
1182 | TCP,UDP  | AcceleNet	官方
1194 | UDP     | OpenVPN	官方
1198 | TCP,UDP  | The cajo project Free dynamic transparent distributed computing in Java	官方
1200 | UDP     | Steam	官方
1214 | TCP     | Kazaa	官方
1223 | TCP,UDP  | TGP: "TrulyGlobal Protocol" aka "The Gur Protocol"	官方
1241 | TCP,UDP  | Nessus Security Scanner	官方
1248 | TCP     | NSClient/NSClient++/NC_Net (Nagios)	非官方
1270 | TCP,UDP  | Microsoft Operations Manager 2005 agent (MOM 2005)	官方
1311 | TCP     | Dell Open Manage Https Port	非官方
1313 | TCP     | Xbiim (Canvii server) Port	非官方
1337 | TCP     | WASTE Encrypted File Sharing Program	非官方
1352 | TCP     | IBM IBM Lotus Notes/Domino RPC	官方
1387 | TCP,UDP  | Computer Aided Design Software Inc LM (cadsi-lm )	官方
1414 | TCP     | IBM MQSeries	官方
1431 | TCP     | RGTP	官方
------|---------|
1494 | TCP     | 思杰系统 ICA Client	官方
1512 | TCP,UDP  | WINS
1521 | TCP     | nCube License Manager	官方
1524 | TCP,UDP  | ingreslock, ingres	官方
1533 | TCP     | IBM Lotus Sametime IM - Virtual Places Chat	官方
1547 | TCP,UDP  | Laplink	官方
1550 | - |		Gadu-Gadu (Direct Client-to-Client)	非官方
1581 | UDP     | MIL STD 2045-47001 VMF	官方
1589 | UDP     | Cisco VQP (VLAN Query Protocol) / VMPS	非官方
1627 | - |		iSketch	非官方
1677 | TCP     | Novell GroupWise clients in client/server access mode
1701 | UDP     | 第二层隧道协议, Layer 2 Tunnelling protocol
1716 | TCP     | 美国陆军系列 MMORPG Default Game Port	官方
1723 | TCP,UDP  | Microsoft 点对点隧道协议 VPN	官方
1725 | UDP     | Valve Steam Client	非官方
1755 | TCP,UDP  | MMS (协议) (MMS, ms-streaming)	官方
1761 | TCP,UDP  | cft-0	官方
1761 | TCP     | Novell Zenworks Remote Control utility	非官方
1762-1768 | TCP,UDP  | cft-1 to cft-7	官方
1812 | UDP     | radius, 远端用户拨入验证服务 authentication protocol
1813 | UDP     | radacct, 远端用户拨入验证服务 accounting protocol
1863 | TCP     | Windows Live Messenger, MSN	官方
1900 | UDP     | Microsoft 简单服务发现协议 Enables discovery of UPnP devices	官方
1935 | TCP     | 实时消息协议 (RTMP) raw protocol	官方
1970 | TCP,UDP  | Danware Data NetOp Remote Control	官方
1971 | TCP,UDP  | Danware Data NetOp School	官方
1972 | TCP,UDP  | InterSystems Caché	官方
1975-77 | UDP     | Cisco TCO (Documentation)	官方
1984 | TCP     | Big Brother - network monitoring tool	官方
1985 | UDP     | 热备份路由器协议	官方
1994 | TCP     | STUN-SDLC protocol for tunneling
1998 | TCP     | Cisco X.25 service (XOT)
2000 | TCP,UDP  | Cisco SCCP (Skinny)	官方
2002 | TCP     | Cisco Secure Access Control Server (ACS) for Windows	非官方
2030 | - |	甲骨文公司 Services for Microsoft Transaction Server	非官方
2031 | TCP,UDP  | mobrien-chat - Mike O'Brien <mike@mobrien.com> November 2004	官方
2049 | UDP     | nfs, NFS Server	官方
2049 | UDP     | shilp	官方
2053 | TCP     | knetd, Kerberos de-multiplexor
2056 | UDP     | 文明IV multiplayer	非官方
2073 | TCP,UDP  | DataReel Database	官方
2074 | TCP,UDP  | Vertel VMF SA (i.e. App.. SpeakFreely)	官方
2082 | TCP     | Infowave Mobility Server	官方
2082 | TCP     | CPanel, default port	非官方
2083 | TCP     | Secure Radius Service (radsec)	官方
2083 | TCP     | CPanel default SSL port	非官方
2086 | TCP     | GNUnet	官方
2086 | TCP     | CPanel default port	非官方
2087 | TCP     | CPanel default SSL port	非官方
2095 | TCP     | CPanel default webmail port	非官方
2096 | TCP     | CPanel default SSL webmail port	非官方
2161 | TCP     | 问号-APC Agent	官方
2181 | TCP,UDP  | EForward-document transport system	官方
2200 | TCP     | Tuxanci game server	非官方
2219 | TCP,UDP  | NetIQ NCAP Protocol	官方
2220 | TCP,UDP  | NetIQ End2End	官方
2222 | TCP     | DirectAdmin's default port	非官方
2222 | UDP     | Microsoft Office OS X antipiracy network monitor [1]	非官方
2301 | TCP     | HP System Management Redirect to port 2381	非官方
2302 | UDP     | 武装突袭 multiplayer (default for game)	非官方
2302 | UDP     | 最后一战：战斗进化 multiplayer	非官方
2303 | UDP     | 武装突袭 multiplayer (default for server reporting) (default port for game +1)	非官方
2305 | UDP     | 武装突袭 multiplayer (default for VoN) (default port for game +3)	非官方
2369 | TCP     | Default port for BMC软件公司 CONTROL-M/Server - Configuration Agent port number - though often changed during installation	非官方
2370 | TCP     | Default port for BMC软件公司 CONTROL-M/Server - Port utilized to allow the CONTROL-M/Enterprise Manager to connect to the CONTROL-M/Server - though often changed during installation	非官方
2381 | TCP     | HP Insight Manager default port for webserver	非官方
2404 | TCP     | IEC 60870-5-104	官方
2427 | UDP     | Cisco MGCP	官方
2447 | TCP,UDP  | ovwdb - OpenView Network Node Manager (NNM) daemon	官方
2546 | TCP,UDP  | Vytal Vault - Data Protection Services	非官方
2593 | TCP,UDP  | RunUO - 网络创世纪 server	非官方
2598 | TCP     | new ICA - when Session Reliability is enabled, TCP port 2598 replaces port 1494	非官方
2612 | TCP,UDP  | QPasa from MQSoftware	官方
2710 | TCP     | XBT Bittorrent Tracker	非官方
2710 | UDP     | XBT Bittorrent Tracker experimental UDP tracker extension	非官方
2710 | TCP     | Knuddels.de	非官方
2735 | TCP,UDP  | NetIQ Monitor Console	官方
2809 | TCP     | "corbaloc:iiop URL, per the CORBA 3.0.3 specification.
---- | - | Also used by IBM IBM WebSphere Application Server Node Agent"	官方
2809 | UDP     | corbaloc:iiop URL, per the CORBA 3.0.3 specification.	官方
2944 | UDP     | Megaco Text H.248	非官方
2945 | UDP     | Megaco Binary (ASN.1) H.248	非官方
2948 | TCP,UDP  | 无线应用协议-push 彩信 (MMS)	官方
2949 | TCP,UDP  | 无线应用协议-pushsecure 彩信 (MMS)	官方
2967 | TCP     | Symantec AntiVirus Corporate Edition	非官方
3000 | TCP     | Miralix License server	非官方
3000 | UDP     | Distributed Interactive Simulation (DIS), modifiable default port	非官方
3000 | TCP     | Puma Web Server	非官方
3001 | TCP     | Miralix Phone Monitor	非官方
3002 | TCP     | Miralix CSTA	非官方
3003 | TCP     | Miralix GreenBox API	非官方
3004 | TCP     | Miralix InfoLink	非官方
3006 | TCP     | Miralix SMS Client Connector	非官方
3007 | TCP     | Miralix OM Server	非官方
3025 | TCP     | netpd.org	非官方
3050 | TCP,UDP  | gds_db (Interbase/Firebird)	官方
3074 | TCP,UDP  | Xbox Live	官方
3128 | TCP     | 超文本传输协议 used by Web缓存s and the default port for the Squid (软件)	官方
3260 | TCP     | ISCSI target	官方
3268 | TCP     | msft-gc, Microsoft Global Catalog (轻型目录访问协议 service which contains data from Active Directoryforests)	官方
3269 | TCP     | msft-gc-ssl, Microsoft Global Catalog over SSL (similar to port 3268, 轻型目录访问协议 over 传输层安全性协议 version)	官方
3300 | TCP     | TripleA game server	非官方
3305 | TCP,UDP  | ODETTE-FTP	官方
3333 | TCP     | Network Caller ID server	非官方
3386 | TCP,UDP  | GTP' 3GPP GSM/通用移动通讯系统 CDR logging protocol	官方
------|---------|
3396 | TCP     | Novell NDPS Printer Agent	官方
3689 | TCP     | DAAP Digital Audio Access Protocol used by 苹果公司 ITunes	官方
3690 | TCP     | Subversion version control system	官方
3702 | TCP,UDP  | Web Services Dynamic Discovery (WS-Discovery), used by various components of Windows Vista	官方
3724 | TCP     | 魔兽世界 Online gaming MMORPG	官方
3784 | TCP,UDP  | Ventrilo VoIP program used by Ventrilo	官方
3785 | UDP     | Ventrilo VoIP program used by Ventrilo	官方
3872 | TCP     | Oracle Management Remote Agent	非官方
3899 | TCP     | Remote Administrator	非官方
3900 | TCP     | Unidata UDT OS udt_os	官方
3945 | TCP     | Emcads server service port, a Giritech product used by G/On	官方
4000 | TCP     | "暗黑破坏神II game NoMachine Network Server (nxd)"	非官方
4007 | TCP     | PrintBuzzer printer monitoring socket server	非官方
4089 | TCP,UDP  | OpenCORE Remote Control Service	官方
4093 | TCP,UDP  | PxPlus Client server interface ProvideX	官方
4096 | UDP     | Bridge-Relay Element ASCOM	官方
4100 | - |		WatchGuard Authentication Applet - default port	非官方
4111 | TCP,UDP  | Xgrid	官方
4111 | TCP     | SharePoint - 默认管理端口	非官方
4226 | TCP,UDP  | Aleph One (computer game)	非官方
4224 | TCP     | 思科系统 CDP Cisco discovery Protocol	非官方
4569 | UDP     | Inter-Asterisk eXchange	非官方
4662 | TCP     | OrbitNet Message Service	官方
4662 | TCP     | 通常用于EMule	非官方
4664 | TCP     | Google桌面搜索	非官方
4672 | UDP     | EMule - 常用端口	非官方
4894 | TCP     | LysKOM Protocol A	官方
4899 | TCP     | Radmin 远程控制工具	官方
5000 | TCP     | commplex-main	官方
5000 | TCP     | UPnP - Windows network device interoperability	非官方
5000 | TCP,UDP  | VTun - 虚拟专用网 软件	非官方
5001 | TCP,UDP  | Iperf (Tool for measuring TCP and UDP bandwidth performance)	非官方
5001 | TCP     | Slingbox及Slingplayer	非官方
5003 | TCP     | FileMaker Filemaker Pro	官方
------|---------|官方
------|---------|enables secure, real-time delivery of audio and video over an IP network
5005 | UDP     | 实时传输协议实时传输协议	官方
5031 | TCP,UDP  | AVM CAPI-over-TCP (综合业务数字网 over 以太网 tunneling)	非官方
5050 | TCP     | Yahoo! Messenger	官方
5051 | TCP     | ita-agent Symantec Intruder Alert	官方
5060 | TCP,UDP  | 会话发起协议 (SIP)	官方
5061 | TCP     | 会话发起协议 (SIP) over 传输层安全性协议 (TLS)	官方
5093 | UDP     | SPSS License Administrator (SPSS)	官方
5104 | TCP     | IBM NetCOOL / IMPACT HTTP Service	非官方
5106 | TCP     | A-Talk Common connection	非官方
5107 | TCP     | A-Talk 远程服务器连接	非官方
5110 | TCP     | ProRat Server	非官方
5121 | TCP     | 无冬之夜	官方
5176 | TCP     | ConsoleWorks default UI interface	非官方
5190 | TCP     | ICQ and AIM (应用程序)	官方
5222 | TCP     | XMPP/Jabber - client connection	官方
5223 | TCP     | XMPP/Jabber - default port for SSL Client Connection	非官方
5269 | TCP     | XMPP/Jabber - server connection	官方
5351 | TCP,UDP  | NAT Port Mapping Protocol - client-requested configuration for inbound connections through 网络地址转换	官方
5353 | UDP     | mDNS - multicastDNS
5402 | TCP,UDP  | StarBurst AutoCast MFTP	官方
5405 | TCP,UDP  | NetSupport	官方
5421 | TCP,UDP  | Net Support 2	官方
5432 | TCP     | PostgreSQL database system	官方
5445 | UDP     | 思科系统 Vidéo VT Advantage	非官方
5495 | TCP     | Applix TM1 Admin server	非官方
5498 | TCP     | Hotline tracker server connection	非官方
5499 | UDP     | Hotline tracker server discovery	非官方
5500 | TCP     | VNC remote desktop protocol - for incoming listening viewer, Hotline control connection	非官方
5501 | TCP     | Hotline file transfer connection	非官方
5517 | TCP     | Setiqueue Proxy server client for SETI@home project	非官方
5555 | TCP     | Freeciv multiplay port for versions up to 2.0, 惠普 Data Protector, 会话通告协议	非官方
5556 | TCP     | Freeciv multiplay port	官方
5631 | TCP     | 赛门铁克 pcAnywhere	官方
5632 | UDP     | 赛门铁克 pcAnywhere	官方
5666 | TCP     | NRPE (Nagios)	非官方
5667 | TCP     | NSCA (Nagios)	非官方
5800 | TCP     | VNC remote desktop protocol - for use over 超文本传输协议	非官方
5814 | TCP,UDP  | 惠普 Support Automation (HP OpenView Self-Healing Services)	官方
5900 | TCP     | VNC remote desktop protocol (used by ARD)	官方
6000 | TCP     | X窗口系统 - used between an X client and server over the network	官方
6001 | UDP     | X窗口系统 - used between an X client and server over the network	官方
6005 | TCP     | Default port for BMC软件公司 CONTROL-M/Server - Socket Port number used for communication between CONTROL-M processes - though often changed during installation	非官方
6050 | TCP     | Brightstor Arcserve Backup	非官方
6051 | TCP     | Brightstor Arcserve Backup	非官方
6100 | TCP     | Vizrt System	非官方
6110 | TCP,UDP  | softcm HP SoftBench CM	官方
6111 | TCP,UDP  | spc HP SoftBench Sub-Process Control	官方
6112 | TCP     | "dtspcd" - a network daemon that accepts requests from clients to execute commands and launch applications remotely	官方
6112 | TCP     | 暴雪娱乐's 暴雪战网 gaming service, ArenaNet gaming service	官方
6129 | TCP     | Dameware Remote Control	非官方
6257 | UDP     | WinMX （参见6699端口）	非官方
6346 | TCP,UDP  | gnutella-svc (FrostWire, LimeWire, Bearshare, etc.)	官方
6347 | TCP,UDP  | gnutella-rtr	官方
6379 | TCP     | Redis - Redis	非官方
6444 | TCP,UDP  | Oracle Grid Engine - Qmaster Service	官方
6445 | TCP,UDP  | Oracle Grid Engine - Execution Service	官方
6502 | TCP,UDP  | Danware Data NetOp Remote Control	非官方
6522 | TCP     | Gobby (and other libobby-based software)	非官方
6543 | UDP     | Jetnet - default port that the Paradigm Research & Development Jetnet protocol communicates on	非官方
6566 | TCP     | SANE (Scanner Access Now Easy) - SANE network scanner daemon	非官方
6600 | TCP     | Music Playing Daemon (MPD)	非官方
6619 | TCP,UDP  | ODETTE-FTP over TLS/SSL	官方
6699 | TCP     | WinMX （参见6257端口）	非官方
6881-6999 | TCP,UDP  | BitTorrent 通常使用的端口	非官方
6891-6900 | TCP,UDP  | Windows Live Messenger （文件传输）	官方
6901 | TCP,UDP  | Windows Live Messenger （语音）	官方
6969 | TCP     | acmsoda	官方
6969 | TCP     | BitTorrent tracker port	非官方
7000 | TCP     | Default port for Azureus's built in 超文本传输安全协议 BitTorrent tracker	非官方
7001 | TCP     | Default port for BEA WebLogic Server's 超文本传输协议 server - though often changed during installation	非官方
7002 | TCP     | Default port for BEA WebLogic Server's 超文本传输安全协议 server - though often changed during installation	非官方
7005 | TCP,UDP  | Default port for BMC软件公司 CONTROL-M/Server and CONTROL-M/Agent's - Agent to Server port though often changed during installation	非官方
7006 | TCP,UDP  | Default port for BMC软件公司 CONTROL-M/Server and CONTROL-M/Agent's - Server to Agent port though often changed during installation	非官方
7010 | TCP     | Default port for Cisco AON AMC (AON Management Console) [2]	非官方
7025 | TCP     | Zimbra - lmtp [mailbox] - local mail delivery	非官方
7047 | TCP     | Zimbra - conversion server	非官方
7171 | TCP     | Tibia
7306 | TCP     | Zimbra - mysql [mailbox]	非官方
7307 | TCP     | Zimbra - mysql [logger] - logger	非官方
7312 | UDP     | Sibelius License Server port	非官方
7670 | TCP     | BrettSpielWelt BSW Boardgame Portal	非官方
7777 | TCP     | Default port used by Windows backdoor program tini.exe	非官方
8000 | TCP     | iRDMI - often mistakenly used instead of port 8080 (The Internet Assigned Numbers Authority (iana.org) officially lists this port for iRDMI protocol)	官方
8000 | TCP     | Common port used for internet radio streams such as those using SHOUTcast	非官方
8002 | TCP     | Cisco Systems Unified Call Manager Intercluster Port
8008 | TCP     | 超文本传输协议 替代端口	官方
8008 | TCP     | IBM HTTP Server 默认管理端口	非官方
8009 | TCP     | 阿帕契族 JServ 协议 v13 (ajp13) 例如：Apache mod_jk Tomcat会使用。	非官方
8010 | TCP     | XMPP/Jabber 文件传输	非官方
8074 | TCP     | Gadu-Gadu	非官方
8080 | TCP     | 超文本传输协议 替代端口 （http_alt） - commonly used for 代理服务器 and caching server, or for running a web server as a non-Root user	官方
8080 | TCP     | Apache Tomcat	非官方
8086 | TCP     | HELM Web Host Automation Windows Control Panel	非官方
8086 | TCP     | Kaspersky AV Control Center TCP Port	非官方
8087 | TCP     | Hosting Accelerator Control Panel	非官方
8087 | UDP     | Kaspersky AV Control Center UDP Port	非官方
8087 | TCP     | 英迈 控制面板	非官方
8090 | TCP     | Another 超文本传输协议 Alternate (http_alt_alt) - used as an alternative to port 8080	非官方
8118 | TCP     | Privoxy web proxy - advertisements-filtering web proxy	官方
8123 | TCP     | Dynmap[4]默认网页端口号(Minecraft在线地图)	非官方
8200 | TCP     | GoToMyPC	非官方
8220 | TCP     | Bloomberg	非官方
8222 | - |	VMware服务器管理用户界面(不安全网络界面)[5]。参见 8333端口	非官方
8291 | TCP     | Winbox - Default port on a MikroTik RouterOS for a Windows application used to administer MikroTik RouterOS	非官方
8294 | TCP     | Bloomberg	非官方
8333 | - |		VMware 服务器管理用户界面（安全网络界面）[6]。参见8222端口	非官方
8400 | - |		Commvault Unified Data Management	官方
8443 | TCP     | 英迈 Control Panel	非官方
8500 | TCP     | Adobe ColdFusion Macromedia/Adobe ColdFusion default Webserver port	非官方
8501 | UDP     | 毁灭公爵3D - Default Online Port	官方
8767 | UDP     | TeamSpeak - Default UDP Port	非官方
8880 | - |		IBM WebSphere Application Server 简单对象访问协议 Connector port
8881 | TCP     | Atlasz Informatics Research Ltd Secure Application Server	非官方
8882 | TCP     | Atlasz Informatics Research Ltd Secure Application Server	非官方
8888 | TCP,UDP  | NewsEDGE server	官方
8888 | TCP     | Sun Answerbook 网页服务器 server (deprecated by docs.sun.com)	非官方
8888 | TCP     | GNUmp3d HTTP music streaming and web interface port	非官方
8888 | TCP     | 英雄大作战 Network Game Server	非官方
9000 | TCP     | Buffalo LinkSystem web access	非官方
9000 | TCP     | DBGp	非官方
9000 | UDP     | UDPCast	非官方
9000 | - |		PHP - php-fpm	非官方
9001 | - |		cisco-xremote router configuration	非官方
9001 | - |		Tor network default port	非官方
9001 | TCP     | DBGp Proxy	非官方
9002 | - |		Default ElasticSearch port
9009 | TCP,UDP  | Pichat Server - Peer to peer chat software	官方
9043 | TCP     | IBM WebSphere Application Server Administration Console secure port
9060 | TCP     | IBM WebSphere Application Server Administration Console
9100 | TCP     | Jetdirect HP Print Services	官方
9110 | UDP     | SSMP Message protocol	非官方
------|---------|
9101	|	- | Bacula Director	官方
9102	|	- | Bacula File Daemon	官方
9103	| - | Bacula Storage Daemon	官方
9119 | TCP,UDP  | Mxit Instant Messenger	官方
9535 | TCP     | man, Remote Man Server
9535 | - |		mngsuite - Management Suite Remote Control	官方
9800 | TCP,UDP  | 基于Web的分布式编写和版本控制 Source Port	官方
9800 | - |		WebCT e-learning portal	非官方
9999 | - |		Hydranode - edonkey2000 telnet control port	非官方
9999 | - |		Urchin Web Analytics	非官方
10000 | - |		Webmin - web based Linux admin tool	非官方
10000 | - |		BackupExec	非官方
10008 | - |		Octopus Multiplexer - CROMP protocol primary port, hoople.org	官方
10017 | - |		AIX,NeXT, HPUX - rexd daemon control port	非官方
10024 | TCP     | Zimbra - smtp [mta] - to amavis from postfix	非官方
10025 | TCP     | Ximbra - smtp [mta] - back to postfix from amavis	非官方
10050 | TCP     | Zabbix-Agent
10051 | TCP     | Zabbix-Server
10113 | TCP,UDP  | NetIQ Endpoint	官方
10114 | TCP,UDP  | NetIQ Qcheck	官方
10115 | TCP,UDP  | NetIQ Endpoint	官方
10116 | TCP,UDP  | NetIQ VoIP Assessor	官方
10480	| - | SWAT 4 Dedicated Server	非官方
11211	| - |	Memcached	非官方
11235	| - |	Savage:Battle for Newerth Server Hosting	非官方
11294	| - |	Blood Quest Online Server	非官方
11371	| - |	PGP HTTP Keyserver	官方
11576	| - |	IPStor Server management communication	非官方
12035 | UDP     | Linden Lab viewer to sim	非官方
12345	| - | 	NetBus - remote administration tool (often 特洛伊木马 (计算机)). Also used by NetBuster. Little Fighter 2 (TCP).	非官方
12975 | TCP     | LogMeIn Hamachi (VPN tunnel software;also port 32976)
13000-13050 | UDP     | Linden Lab viewer to sim	非官方
13720 | TCP     | 赛门铁克 NetBackup - bprd (formerly VERITAS)
13721 | TCP     | 赛门铁克 NetBackup - bpdbm (formerly VERITAS)
13724 | TCP     | 赛门铁克 Network Utility - vnet (formerly VERITAS)
13782 | TCP     | 赛门铁克 NetBackup - bpcd (formerly VERITAS)
13783 | TCP     | 赛门铁克 VOPIED protocol (formerly VERITAS)
14567 | UDP     | 战地风云1942 and mods	非官方
15000 | TCP     | Bounce (网络)	非官方
15000 | TCP     | 韦诺之战
15567 | UDP     | 战地风云：越南 and mods	非官方
15345 | UDP     | XPilot	官方
16000 | TCP     | Bounce (网络)	非官方
16080 | TCP     | MacOS Server performance cache for 超文本传输协议[7]	非官方
16384 | UDP     | Iron Mountain Digital - online backup	非官方
16567 | UDP     | 战地2 and mods	非官方
17788 | UDP     | PPS网络电视	非官方
19132 | TCP     | Minecraft基岩版默认服务器端口号	非官方
19226 | TCP     | 熊猫 (消歧义) AdminSecure Communication Agent	非官方
19638 | TCP     | Ensim Control Panel	非官方
19813 | TCP     | 4D database Client Server Communication	非官方
20000	| - |	Usermin - 基于网络的用户工具	官方
20720 | TCP     | Symantec i3 Web GUI server	非官方
22347 | TCP,UDP  | WibuKey - default port for WibuKey Network Server of WIBU-SYSTEMS AG	官方
22350 | TCP,UDP  | CodeMeter - default port for CodeMeter Server of WIBU-SYSTEMS AG	官方
24554 | TCP,UDP  | binkp - FidoNet mail transfers over TCP/IP协议族	官方
24800	| - |	Synergy：keyboard/mouse sharing software	非官方
24842	| - |	StepMania：Online: 劲爆热舞 Simulator	非官方
25565 | TCP     | Minecraft默认服务器端口号	非官方
25999 | TCP     | Xfire	非官方
26000 | TCP,UDP  | Id Software's Quake server,	官方
26000 | TCP     | CCP Games's 星战前夜 Online gaming MMORPG,	非官方
27000 | UDP     | (through 27006) Id Software's 雷神世界 master server	非官方
27010 | UDP     | Half-Life及其修改版，如反恐精英系列	非官方
27015 | UDP     | Half-Life及其修改版，如反恐精英系列	非官方
27374 | - |		Sub7's default port. Most 脚本小子s do not change the default port.	非官方
27500 | UDP     | (through 27900) Id Software's 雷神世界	非官方
27888 | UDP     | Kaillera server	非官方
27900 | - |		(through 27901) 任天堂 任天堂Wi-Fi连接	非官方
27901 | UDP     | (through 27910) Id Software's 雷神之锤II master server	非官方
27960 | UDP     | (through 27969) 动视's Enemy Territory and Id Software's 雷神之锤III竞技场 and Quake III and some ioquake3 derived games	非官方
28910	| - |	任天堂 任天堂Wi-Fi连接	非官方
28960	| - |	决胜时刻2 Common Call of Duty 2 port - (PC Version)	非官方
29900	| - |	(through 29901) 任天堂 任天堂Wi-Fi连接	非官方
29920	| - |	任天堂 任天堂Wi-Fi连接	非官方
30000	| - |	Pokemon Netbattle	非官方
30564 | TCP     | Multiplicity：keyboard/mouse/clipboard sharing software	非官方
31337 | TCP     | Back Orifice - remote administration tool（often 特洛伊木马 (计算机)）	非官方
31337 | TCP     | xc0r3 - xc0r3 security antivir port	非官方
31415	| - |	ThoughtSignal - Server Communication Service（often Informational）	非官方
31456-31458 | TCP     | TetriNET ports (in order: IRC, game, and spectating)	非官方
32245 | TCP     | MMTSG-mutualed over MMT (encrypted transmission)	非官方
33434	| - |	Traceroute	官方
37777 | TCP     | Digital Video Recorder hardware	非官方
36963	| - |	 Counter Strike 2D multiplayer port (2D clone of popular CounterStrike computer game)	非官方
40000	| - |	SafetyNET p	官方
40523	| - |	data packets	不确定 6
43594-43595 | TCP     | RuneScape	非官方
47808	| - |	BACnet Building Automation and Control Networks	官方
