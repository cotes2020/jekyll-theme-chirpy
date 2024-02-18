---
title : Root-me Backup file (풀이 봄)
categories : [Wargame, Root-me]
tags: [nmap, nmap NSE, nmap http backup finder, 풀이 봄]
---

## Backup file (풀이 봄)
<hr style="border-top: 1px solid;"><br>

```
Author
g0uZ,  27 February 2011

Statement
No clue.
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Solution
<hr style="border-top: 1px solid;"><br>

풀이를 찾아보니.. nmap의 NSE로 ```http-backup-finder```라는게 있다고 함.

<br>

```
nmap -p 80 --script=http-backup-finder --script-args http-backup-finder.url=/web-serveur/ch11/index.php challenge01.root-me.org

Starting Nmap 7.60 ( https://nmap.org ) at 2020-06-12 00:46 KST
Nmap scan report for challenge01.root-me.org (212.129.38.224)
Host is up (0.32s latency).
Other addresses for challenge01.root-me.org (not scanned): 2001:bc8:35b0:c166::151

PORT   STATE SERVICE
80/tcp open  http
| http-backup-finder:
| Spidering limited to: maxdepth=3; maxpagecount=20; withinhost=challenge01.root-me.org
|_  http://challenge01.root-me.org:80/web-serveur/ch11/index.php~

Nmap done: 1 IP address (1 host up) scanned in 12.76 seconds
```

<br><br>
<hr style="border: 2px solid;">
<br><br>
