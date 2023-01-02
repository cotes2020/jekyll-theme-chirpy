---
title: Lab - HTB - Esay - ScriptKiddie
date: 2021-03-29 11:11:11 -0400
description: HackTheBox
categories: [Lab, HackTheBox]
# img: /assets/img/sample/rabbit.png
tags: [Lab, HackTheBox]
---

- [ScriptKiddie](#scriptkiddie)
	- [Initial](#initial)
		- [Recon: Nmap&nikto](#recon-nmapnikto)
		- [Recon: Brup](#recon-brup)
		- [CVE](#cve)
			- [漏洞概述](#漏洞概述)
			- [漏洞复现](#漏洞复现)
	- [Gain access to shell: Brup](#gain-access-to-shell-brup)
		- [user.txt: Brupsuite](#usertxt-brupsuite)
		- [user.txt: ReverseShell](#usertxt-reverseshell)
	- [Privilege escalation](#privilege-escalation)
		- [Root.txt: Execute ruby scripts](#roottxt-execute-ruby-scripts)
		- [Root.txt: ssh](#roottxt-ssh)


- ref:
  - [Knife — Hack The Box](https://zhuanlan.zhihu.com/p/374971092)


---

# ScriptKiddie

![v2-2ce6215a74c7ab7a72fe29537d5f95af_r](https://i.imgur.com/gcDa3jA.jpg)

---

## Initial

### Recon: Nmap&nikto

```bash
$ nmap -A -sS -sV -sC 10.10.10.226
# Starting Nmap 7.91 ( https://nmap.org ) at 2021-05-31 19:27 EDT
Nmap scan report for 10.10.10.226
Host is up (0.047s latency).
Not shown: 942 closed ports, 56 filtered ports
PORT     STATE SERVICE VERSION
22/tcp   open  ssh     OpenSSH 8.2p1 Ubuntu 4ubuntu0.1 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey:
|   3072 3c:65:6b:c2:df:b9:9d:62:74:27:a7:b8:a9:d3:25:2c (RSA)
|   256 b9:a1:78:5d:3c:1b:25:e0:3c:ef:67:8d:71:d3:a3:ec (ECDSA)
|_  256 8b:cf:41:82:c6:ac:ef:91:80:37:7c:c9:45:11:e8:43 (ED25519)
5000/tcp open  http    Werkzeug httpd 0.16.1 (Python 3.8.5)
|_http-server-header: Werkzeug/0.16.1 Python/3.8.5
|_http-title: k1d5 h4ck3r t00l5
No exact OS matches for host (If you know what OS is running on it, see https://nmap.org/submit/ ).
TCP/IP fingerprint:
OS:SCAN(V=7.91%E=4%D=5/31%OT=22%CT=1%CU=31801%PV=Y%DS=2%DC=T%G=Y%TM=60B5710
OS:6%P=x86_64-apple-darwin19.6.0)SEQ(SP=105%GCD=1%ISR=10C%TI=Z%CI=Z%II=I%TS
OS:=A)SEQ(SP=100%GCD=1%ISR=10E%TI=Z%CI=Z%TS=A)OPS(O1=M550ST11NW7%O2=M550ST1
OS:1NW7%O3=M550NNT11NW7%O4=M550ST11NW7%O5=M550ST11NW7%O6=M550ST11)WIN(W1=FE
OS:88%W2=FE88%W3=FE88%W4=FE88%W5=FE88%W6=FE88)ECN(R=Y%DF=Y%T=40%W=FAF0%O=M5
OS:50NNSNW7%CC=Y%Q=)T1(R=Y%DF=Y%T=40%S=O%A=S+%F=AS%RD=0%Q=)T2(R=N)T3(R=N)T4
OS:(R=Y%DF=Y%T=40%W=0%S=A%A=Z%F=R%O=%RD=0%Q=)T5(R=Y%DF=Y%T=40%W=0%S=Z%A=S+%
OS:F=AR%O=%RD=0%Q=)T6(R=Y%DF=Y%T=40%W=0%S=A%A=Z%F=R%O=%RD=0%Q=)T7(R=Y%DF=Y%
OS:T=40%W=0%S=Z%A=S+%F=AR%O=%RD=0%Q=)U1(R=Y%DF=N%T=40%IPL=164%UN=0%RIPL=G%R
OS:ID=G%RIPCK=G%RUCK=G%RUD=G)IE(R=Y%DFI=N%T=40%CD=S)

Network Distance: 2 hops
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

TRACEROUTE (using port 53/tcp)
HOP RTT      ADDRESS
1   76.29 ms 10.10.14.1
2   76.28 ms 10.10.10.226

OS and Service detection performed. Please report any incorrect results at https://nmap.org/submit/ .
Nmap done: 1 IP address (1 host up) scanned in 25.53 seconds
```

Open port 22 and 5000

![Screen Shot 2021-05-31 at 19.37.08](https://i.imgur.com/HdB8Sy8.png)

5000端口:
- 运行了Werkzeug，
- Werkzeug存在Werkzeg debug shell漏洞，但对比了下版本不适用，那就只能先访问5000端口看看具体情况了
- go to `10.10.10.226:5000`
- 继承了nmap、msfvenom、searchsploit工具的一个小黑客工具


try nmap
- nmap verzion is 7.80
![Screen Shot 2021-05-31 at 19.39.41](https://i.imgur.com/KSPYNVF.png)


try msfvenom
- 搜索venom发现了一个APK模板命令注入漏洞
- 将`exploit-db`上的exp给下载下来并进行修改
- 然后执行该EXP
- 获得了evil.apk文件，在本地监听443端口后将该文件上传




### Recon: Brup

---


### CVE

#### 漏洞概述

**漏洞影响**
- `PHP 8.1.0-dev`


#### 漏洞复现

---

## Gain access to shell: Brup

### user.txt: Brupsuite

### user.txt: ReverseShell

---

## Privilege escalation


### Root.txt: Execute ruby scripts


### Root.txt: ssh


Thanks for reading this far. We will meet in my next article.

**Happy Hacking** ☺






.
