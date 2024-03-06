---
title: Meow's Testing Tools - nmap
date: 2019-09-17 11:11:11 -0400
categories: [10CyberAttack, CyberAttackTools]
tags: [CyberAttack, CyberAttackTools]
toc: true
image:
---


[toc]

---

![Screen Shot 2020-10-30 at 19.45.35](https://i.imgur.com/a0IeE5F.png)

# Nmap

- free download, port scan machines.
- can reveal what services are running as well as info about the target machine’s operating system.
- 17-20 sec
- can scan a range of IP addresses as well as a single IP.
- set a number of flags (either with the command-line version of nmap or the Windows version) that customize your scan.

- can perform many different types of scans (from simply identifying active machines to port scanning and enumeration)
- can be configured to control the speed at which a scan operates. In general, the slower the scan, the less likely you are to be discovered.
- It comes in both a command-line version and a GUI version (Zenmap), works on multiple OS platforms, and can even scan over TCP and UDP.
- The target can besingle IP address, multiple individual IPs separated by spaces, or an entire subnet range (using CIDR notation).


![Screen Shot 2020-09-22 at 21.20.11](https://i.imgur.com/6j7WGaA.png)

![Nmap-Cheat-Sheet-1](https://i.imgur.com/HgcDLxE.jpg)

![page88image152923104](https://i.imgur.com/6mO8lhp.jpg)


---

```bash
nmap 0.0.0.0
nmap 0.0.0.0 0.0.0.1 0.0.0.2 scanme.nmap.org
nmap 0.0.0.1-125
nmap 0.0.0.0/24
nmap 0.0.0.0/24 --exclude 0.0.0.3
nmap 0.0.0.0/24 --excludefile list.txt
nmap scanme.nmap.org
nmap nmap.org/29
nmap -O scanme.nmap.org
nmap -iL targetlist.txt
nmap -iR 20  # random select 20

Nmap                              Port Selection
Scan a single Port	              nmap -p 22 192.168.1.1
Scan a range of ports	            nmap -p 1-100 192.168.1.1
Scan 100 most common ports (Fast)	nmap -F 192.168.1.1
Scan all 65535 ports            	nmap -p- 192.168.1.1


# Nmap Output Formats
nmap 0.0.0.1 -oN output.txt # normal output
nmap 0.0.0.1 -oX output.xml # xml
nmap 0.0.0.1 -oG output.txt
nmap 0.0.0.1 -oA output     # 3 mainfile type, gnmap, nmap, xml
nmap -oA output.txt 0.0.0.1 -O


nmap -sO # IP protocol scan


nmap -O -v 0.0.0.1 -oA output.txt   # detail, verbositive level
nmap -O -vv 0.0.0.1 -oA output.txt

nmap -F 0.0.0.1-50      # fast mode. 100 most popular port
nmap -T4 -A -v 0.0.0.0  # advanced

nmap -sn 0.0.0.0 # fast ping swipe
nmap -sS # TCP SYN scan
nmap -sS -p 80 -T2 -iR 50
nmap -sA # TCP ACK scan
nmap -sT # TCP connection scan
nmap -sY # SCTP INIT scan
nmap -sW # TCP Window scan
nmap -sM # TCP Maimon scan

nmap -sU # UDP scan
nmap -sU -p 53,88,123,137,138 0.0.0.0

nmap -sN # Null scan
nmap -sF # FIN scan REST=livehost
nmap -sX 0.0.0.0 # Xmas scan push urgent=1
nmap -b # FTP Bounce scan

-Pn # don't ping before scanning
nmap -sn -PE # ICMP ping
-PP # ICMP timestamp request
-PM # ICMP netmask request
-PA # ACK ping
-PS # SYN ping
-PU # UDP probes
-PY # SCTP INIT ping probes

-d             # debugging level
--packet-trace # Packet trace
-r             # disable randomizing scanned ports
--traceroute   # traceroute to targets
--max-retries  # Max Retries


nmap -v --script whois-ip emailheaderip
nmap -v --script whois-domain header.from(trophtdepot.com)
nmap -T4 -O --script smb-os-discovery, whois-domain header.from(0.0.0.0)
# A scan to search for DDOS reflection UDP services
Scan for UDP DDOS reflectors	nmap –sU –A –PN –n –pU:19,53,123,161 –script=ntp-monlist,dns-recursion,snmp-sysdescr 192.168.1.0/24
# HTTP Service Information
Gather page titles from HTTP services	nmap --script=http-title 192.168.1.0/24
Get HTTP headers of web services     	nmap --script=http-headers 192.168.1.0/24
Find web apps from known paths       	nmap --script=http-enum 192.168.1.0/24
# Detect Heartbleed SSL Vulnerability
nmap -sV -p 443 --script=ssl-heartbleed 192.168.1.0/24
# Find Information about IP address
nmap --script=asn-query,whois,ip-geolocation-maxmind 192.168.1.0/24

nmap -6 --hop-limit
nmap -6 -S source target
nmap -6 0.0.0.0
nmap --ipv6 0.0.0.0


# Service and OS Detection
Detect OS and Services	          nmap -A 192.168.1.1
Standard service detection       	nmap -sV 192.168.1.1
More aggressive Service Detection	nmap -sV --version-intensity 5 192.168.1.1
Lighter banner grabbing detection	nmap -sV --version-intensity 0 192.168.1.1

```



---

- `s`: determine the type of scan to perform,
  - run a SYN port scan on a target as quietly as possible. ￼
    - `namap 111.111.11.1/24 -sS T0`
  - wanted an aggressive XMAS scan. ￼
    - `namap 111.111.11.1/24 -sX T4`
- `P`: set up ping sweep options,
- `o`: deal with output.
- `T`: deal with speed and stealth, with the serial methods taking the longest amount of time.
  - Parallel methods are much faster because they run multiple scans simultaneously.
  - Again, the slower you run scans, the less likely you are to be discovered. The choice of which one to run is yours.
  - slower being better, paranoid and sneaky scans can take exceedingly long times to complete.
  - Nmap at very fast (-T5) speeds, you’ll overwhelm your NIC and start getting some really weird results.
  - default: -T3, “normal.”

- Upon successful response from the targeted host,
  - If the command successfully finds a live host, it returns a message indicating that the IP address of the targeted host is up, along with the media access control (MAC) address and the network card vendor.
  - `nmap –sn –v <target IP address>`




- Nmap in a nutshell, offers Host discovery, Port discovery, Service discovery. Operating system version information. Hardware (MAC) address information, Service version detection, Vulnerability & exploit detection using Nmap scripts (NGE).
  - `nmap –sn –PE –PA <port numbers>  <Starting IP/ending IP>`
- output to a file:
  - `nmap 1.1.1.1 > result.txt`

Nmap handles all scan types we discussed in the previous section, using switches identified earlier. In addition to those listed, Nmap offers a “Windows” scan.
- It works like the ACK scan but is intended for use on Windows networks
- provides all sorts of information on open ports.
- Many more switches and options are available for the tool.

NOTE Port sweeping and enumeration on a machine is also known as fingerprinting, although the term is normally associated with examining the OS itself. You can fingerprint operating systems with several tools we’ve discussed already, along with goodies such as SolarWinds, Netcraft, and HTTrack.


---

## install

```bash
# for mac
install homebrew
brew update
brew upgrade
brew install wget
brew install nmap
brew install Zenmap

```

---

## os detect [nmap –O]

`nmap –O [目標IP]`

- 作業系統偵測的方式
- 使用TTL來大致判斷作業系統的類型
- NMAP也可使用`TCP/IP stack fingerprinting`進行遠端作業系統識別
  - NMAP會送出一系列的 TCP 和 UDP 封包到目標主機
  - 檢驗回傳(response)的每一個bit。
  - 在一連串的測試結果回傳後，NMAP會把這些資料拿去 nmap-os-db 的資料庫比對，他的資料庫中有超過2600種以上類型的作業系統資料，
  - NMAP若在資料庫中找到相符資訊，就會顯示出來。


每一個fingerprint都包含了一些作業系統的文字描述，並可以用來分類供應商(Sun/Microsoft), 作業系統(Solaris/Microsoft Windows), 版本 (10/2016), 及設備類型(route/switch..) 最常見的包含Common Platform Enumeration (CPE) representation例如: cpe:/o:linux:linux_kernel:2.6.

作業系統偵測會採用其他蒐集資料的方式
- 例如`TCP Sequence Predictability Classification`，這個測量標準取決於TCP的連線有多難建立。
- 另外一個作業系統探測的方式，目標主機的開機的時間，使用了TCP的時間戳記來判斷機器上次是何時重新啟動。雖然這種方式有可能因為計數器沒有歸零而不準確，但仍舊是判斷的標準之一。

![20119885QKelRVIEdE](https://i.imgur.com/UVlEJMN.jpg)

Windows 上面開啟的 WireShark也可以看到不同的封包送過來

![201198859tIIAzVaG3](https://i.imgur.com/V2XpawD.jpg)

掃描 RedHat 的結果

![201198850drjWEiMmG](https://i.imgur.com/gRls3fj.jpg)

結果會用「猜的」並配上百分比,非常準確。



![20119885iaQWJN3ocm](https://i.imgur.com/ZWbXAiQ.jpg)

---

## Nmap output

- The GUI version of the tool, `Zenmap`, makes reading this output easy
- the output is available via several methods.
- The default is called interactive, and it is sent to standard output (text sent to the terminal). Normal output displays less run-time information and fewer warnings because it is expected to be analyzed after the scan completes rather than interactively.
- You can also send output as XML (which can be parsed by graphical user interfaces or imported into databases) or in a “greppable” format (for easy searching).

```bash

$ nmap scanme.nmap.org
Starting Nmap 7.80 ( https://nmap.org ) at 2020-10-30 19:39 EDT
Nmap scan report for scanme.nmap.org (45.33.32.156)
Host is up (0.11s latency).
Not shown: 992 closed ports
PORT      STATE    SERVICE
22/tcp    open     ssh
25/tcp    filtered smtp
Nmap done: 1 IP address (1 host up) scanned in 23.08 seconds

nmap -sS 127.0.0.1
Starting Nmap 4.01 at 2006-07-06 17:23 BST
Interesting ports on chaos (127.0.0.1):
(The 1668 ports scanned but not shown below are in state: closed)
PORT     STATE SERVICE
21/tcp   open  ftp
22/tcp   open  ssh
Nmap finished: 1 IP address (1 host up) scanned in 0.207 seconds

```

![page89image152295312](https://i.imgur.com/ZaGR0fp.jpg)

![Screen Shot 2020-09-22 at 21.22.01](https://i.imgur.com/Gk9c2A8.png)

![Screen Shot 2020-09-22 at 21.23.12](https://i.imgur.com/p0btjP5.png)

![Screen Shot 2020-09-22 at 21.23.26](https://i.imgur.com/5yeUsUI.png)

---

## Basic Scan Types [-sT, -sS]

The two basic scan types used most in Nmap are
1. TCP connect() scanning [-sT]
2. SYN scanning (half-open / stealth scanning) [-sS].

The sample below shows a SYN scan and a FIN scan, performed against a Linux system.

```bash
nmap -sS 127.0.0.1
Starting Nmap 4.01 at 2006-07-06 17:23 BST
Interesting ports on chaos (127.0.0.1):
(The 1668 ports scanned but not shown below are in state: closed)
PORT     STATE SERVICE
21/tcp   open  ftp
22/tcp   open  ssh
Nmap finished: 1 IP address (1 host up) scanned in 0.207 seconds


nmap -sF 127.0.0.1
Starting Nmap 4.01 at 2006-07-06 17:23 BST
Interesting ports on chaos (127.0.0.1):
(The 1668 ports scanned but not shown below are in state:closed)
PORT     STATE         SERVICE
21/tcp   open|filtered ftp
22/tcp   open|filtered ssh
Nmap finished: 1 IP address (1 host up) scanned in 1.284
```

---

### TCP connect() Scan [-sT]
- These scans are so called because UNIX sockets programming uses a system call named `connect()` to begin a TCP connection to a remote site.
  - If `connect()` succeeds, connection made.
  - If it fails, the connection could not be made
    - the remote system is offline
    - the port is closed
    - some other error occurred along the way
- This allows a basic type of port scan
  - attempts to connect to every port in turn
  - notes whether or not the connection succeeded.
  - Once the scan is completed
    - ports to which a connection could be established are listed as `open`
    - the rest are said to be `closed`.

pros:
1. very effective, provides a clear picture of the ports you can and cannot access.
   - If a connect() scan lists a port as open, you can definitely connect to it

major drawback
1. very easy to detect on the system being scanned.
   - If a firewall or intrusion detection system is running on the victim
   - attempts to `connect()` to every port on the system will almost always trigger a warning.
   - modern firewalls, attempt to connect to a single port (which has been blocked or has not been specifically "opened") will usually result in the `connection attempt being logged`.
   - most servers will log connections and their source IP
   - it would be easy to detect the source of a `TCP connect() scan`.

---

### SYN Stealth Scan [-sS]
For this reason, the TCP Stealth Scan was developed.
- When a TCP connection is made between two systems, a process known as a "three way handshake" occurs. This involves the exchange of three packets, and synchronises the systems with each other (necessary for the error correction built into TCP.
- The system initiating the connection sends a packet to the system it wants to connect to. TCP packets have a header section with a flags field. Flags tell the receiving end something about the type of packet, and thus what the correct response is.
- four possible flags.
  - SYN (Synchronise), ACK (Acknowledge), FIN (Finished) and RST (Reset).
  - `SYN` packets include a `TCP sequence number`: lets the remote system know what sequence numbers to expect in subsequent communication.
  - `ACK` acknowledges receipt of a packet or set of packets,
  - `FIN`: when a communication is finished, requesting that the connection be closed
  - `RST`: when the connection is to be reset (closed immediately)

**SYN / Stealth scanning** makes use of this procedure by sending a SYN packet and looking at the response.
1. the port is open
   - `SYN/ACK` is sent back
   - the remote end is trying to open a TCP connection.
   - The scanner then sends an `RST` to tear down the connection before it can be established fully; often preventing the connection attempt appearing in application logs.
2. the port is closed
   - `RST` will be sent back
3. it is filtered
   - `no response` will be sent back
   - the SYN packet will have been dropped and no response will be sent.

In this way, Nmap can detect three port states
- open, closed and filtered.
- Filtered ports may require further probing since they could be subject to firewall rules which render them open to some IPs or conditions, and closed to others.

Modern firewalls and Intrusion Detection Systems can detect SYN scans, but in combination with other features of Nmap, it is possible to create a virtually undetectable SYN scan by altering timing and other options (explained later).

---

### FIN, Null and Xmas Tree Scans [-sF, -sN, -sX]

Each scan type refers to the flags set in the TCP header.
- a **closed port** should respond with an `RST` upon receiving packets
- an **open port** should `just drop them` (it’s listening for packets with SYN set).
- This way, you never make even part of a connection, and never send a SYN packet (what most IDS’ look out for).

1. The FIN scan sends a packet with only the `FIN` flag set,
2. the Xmas Tree scan sets the `FIN, URG and PUSH` flags
3. the Null scan sends a packet `with no flags` switched on.

These scan types will work against any system where the TCP/IP implementation follows RFC 793.
- Microsoft Windows does not follow the RFC, and will ignore these packets even on closed ports.
- to detect an MS Windows system
  - running SYN along with one of these scans.
  - If the SYN scan shows open ports, and the FIN/NUL/XMAS does not, chances are you’re looking at a Windows box
  - (OS Fingerprinting is much more reliable way)

---

## Ping Scan [-sP]
- lists the hosts within the specified range that responded to a ping.
- to detect which computers are online, rather than which ports are open.

Four methods exist within Nmap for ping sweeping.
1. sends an `ICMP ECHO REQUEST (ping request) packet` to the destination system.
   - `ICMP ECHO REPLY` received, the system is up, and ICMP packets are not blocked.
   - `no response` to the ICMP ping, Nmap will try a "TCP Ping", to determine whether ICMP is blocked, or if the host is really not online.

2. sends either a `SYN or an ACK packet` to any port (80 default) on the remote system.
   - `RST or a SYN/ACK` is returned, then the remote system is online.
   - If the remote system `does not respond`, either it is offline, or the chosen port is filtered, and thus not responding to anything.

3. run an Nmap ping scan as root, the default is to use the `ICMP and ACK methods`.
4. Non-root users will use the `connect() method`, which attempts to connect to a machine, waiting for a response, and tearing down the connection as soon as it has been established (similar to the SYN/ACK method for root users, but this one establishes a full TCP connection!)

The ICMP scan type can be disabled by setting `-P0`

---

## UDP Scan [-sU]
Nmap sends 0-byte UDP packets to each target port on the victim.
- Receipt of an `ICMP Port Unreachable` message signifies the port is closed
- otherwise it is assumed open.

disadvantage
1. when a firewall blocks outgoing ICMP Port Unreachable messages, the port will appear open.
   - false-positives are hard to distinguish from real open ports.
2. the speed at which it can be performed.
   - Most operating systems limit the number of `ICMP Port Unreachable messages` which can be generated in a certain time period
     - thus slowing the speed of a UDP scan.
   - Nmap adjusts its scan speed accordingly to avoid flooding a network with useless packets.
   - **Microsoft do not limit the `Port Unreachable error` generation frequency**
     - thus it is easy to scan a Windows machine’s 65,535 UDP Ports in very little time!!

UDP Scanning is not usually useful for most types of attack, but it can reveal information about services or trojans which rely on UDP, for example SNMP, NFS, the Back Orifice trojan backdoor and many other exploitable services.

Most modern services utilise TCP, and thus UDP scanning is not usually included in a pre-attack information gathering exercise unless a TCP scan or other sources indicate that it would be worth the time taken to perform a UDP scan.




---

ref:
- [nmap](https://nmap.org/bennieston-tutorial/)
