---------------------

### CTF: Tryhackme
### LAB: Greenhorn

---------------------

![image](https://github.com/user-attachments/assets/6bd55144-4a5f-490d-935c-0c478386ce71)

---------------------

### RECONNAISSANCE:

- Rustscan's output
      
      ❯ rustscan -a greenhorn.htb -- -sC -sV -Pn
      .----. .-. .-. .----..---.  .----. .---.   .--.  .-. .-.
      | {}  }| { } |{ {__ {_   _}{ {__  /  ___} / {} \ |  `| |
      | .-. \| {_} |.-._} } | |  .-._} }\     }/  /\  \| |\  |
      `-' `-'`-----'`----'  `-'  `----'  `---' `-'  `-'`-' `-'
      The Modern Day Port Scanner.
      ________________________________________
      : https://discord.gg/GFrQsGy           :
      : https://github.com/RustScan/RustScan :
       --------------------------------------
      😵 https://admin.tryhackme.com
      
      [~] The config file is expected to be at "/home/sensei/.rustscan.toml"
      [!] File limit is lower than default batch size. Consider upping with --ulimit. May cause harm to sensitive servers
      [!] Your file limit is very small, which negatively impacts RustScan's speed. Use the Docker image, or up the Ulimit with '--ulimit 5000'. 
      Open 10.10.11.25:22
      Open 10.10.11.25:80
      Open 10.10.11.25:3000
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-09-14 08:50 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 08:50
      Completed NSE at 08:50, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 08:50
      Completed NSE at 08:50, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 08:50
      Completed NSE at 08:50, 0.01s elapsed
      Initiating Connect Scan at 08:50
      Scanning greenhorn.htb (10.10.11.25) [3 ports]
      Discovered open port 80/tcp on 10.10.11.25
      Discovered open port 22/tcp on 10.10.11.25
      Discovered open port 3000/tcp on 10.10.11.25
      Completed Connect Scan at 08:50, 0.25s elapsed (3 total ports)
      Initiating Service scan at 08:50
      Scanning 3 services on greenhorn.htb (10.10.11.25)
      Completed Service scan at 08:52, 98.59s elapsed (3 services on 1 host)
      NSE: Script scanning 10.10.11.25.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 08:52
      Completed NSE at 08:52, 23.05s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 08:52
      Completed NSE at 08:52, 2.10s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 08:52
      Completed NSE at 08:52, 0.00s elapsed
      Nmap scan report for greenhorn.htb (10.10.11.25)
      Host is up, received user-set (0.25s latency).
      Scanned at 2024-09-14 08:50:29 EDT for 124s
      
      PORT     STATE SERVICE REASON  VERSION
      22/tcp   open  ssh     syn-ack OpenSSH 8.9p1 Ubuntu 3ubuntu0.10 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   256 57:d6:92:8a:72:44:84:17:29:eb:5c:c9:63:6a:fe:fd (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBOp+cK9ugCW282Gw6Rqe+Yz+5fOGcZzYi8cmlGmFdFAjI1347tnkKumDGK1qJnJ1hj68bmzOONz/x1CMeZjnKMw=
      |   256 40:ea:17:b1:b6:c5:3f:42:56:67:4a:3c:ee:75:23:2f (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEZQbCc8u6r2CVboxEesTZTMmZnMuEidK9zNjkD2RGEv
      80/tcp   open  http    syn-ack nginx 1.18.0 (Ubuntu)
      |_http-server-header: nginx/1.18.0 (Ubuntu)
      | http-title: Welcome to GreenHorn ! - GreenHorn
      |_Requested resource was http://greenhorn.htb/?file=welcome-to-greenhorn
      |_http-generator: pluck 4.7.18
      | http-methods: 
      |_  Supported Methods: GET POST
      |_http-trane-info: Problem with XML parsing of /evox/about
      | http-cookie-flags: 
      |   /: 
      |     PHPSESSID: 
      |_      httponly flag not set
      | http-robots.txt: 2 disallowed entries 
      |_/data/ /docs/
      3000/tcp open  ppp?    syn-ack
      | fingerprint-strings: 
      |   GenericLines, Help, RTSPRequest: 
      |     HTTP/1.1 400 Bad Request
      |     Content-Type: text/plain; charset=utf-8
      |     Connection: close
      |     Request
      |   GetRequest: 
      |     HTTP/1.0 200 OK
      |     Cache-Control: max-age=0, private, must-revalidate, no-transform
      |     Content-Type: text/html; charset=utf-8
      |     Set-Cookie: i_like_gitea=98c7984fd88212c6; Path=/; HttpOnly; SameSite=Lax
      |     Set-Cookie: _csrf=tgJrdjjVc4E5bW9ek-p31ZrbXSc6MTcyNjMxODIzOTc2MDY5ODQ5MA; Path=/; Max-Age=86400; HttpOnly; SameSite=Lax
      |     X-Frame-Options: SAMEORIGIN
      |     Date: Sat, 14 Sep 2024 12:50:39 GMT
      |     <!DOCTYPE html>
      |     <html lang="en-US" class="theme-auto">
      |     <head>
      |     <meta name="viewport" content="width=device-width, initial-scale=1">
      |     <title>GreenHorn</title>
      |     <link rel="manifest" href="data:application/json;base64,eyJuYW1lIjoiR3JlZW5Ib3JuIiwic2hvcnRfbmFtZSI6IkdyZWVuSG9ybiIsInN0YXJ0X3VybCI6Imh0dHA6Ly9ncmVlbmhvcm4uaHRiOjMwMDAvIiwiaWNvbnMiOlt7InNyYyI6Imh0dHA6Ly9ncmVlbmhvcm4uaHRiOjMwMDAvYXNzZXRzL2ltZy9sb2dvLnBuZyIsInR5cGUiOiJpbWFnZS9wbmciLCJzaXplcyI6IjUxMng1MTIifSx7InNyYyI6Imh0dHA6Ly9ncmVlbmhvcm4uaHRiOjMwMDAvYX
      |   HTTPOptions: 
      |     HTTP/1.0 405 Method Not Allowed
      |     Allow: HEAD
      |     Allow: HEAD
      |     Allow: GET
      |     Cache-Control: max-age=0, private, must-revalidate, no-transform
      |     Set-Cookie: i_like_gitea=d7d4f84a96cc0455; Path=/; HttpOnly; SameSite=Lax
      |     Set-Cookie: _csrf=W3IufYSw6N9E6wfnc7vpXsS9G2Q6MTcyNjMxODI0NTkyMTc1MTM2Nw; Path=/; Max-Age=86400; HttpOnly; SameSite=Lax
      |     X-Frame-Options: SAMEORIGIN
      |     Date: Sat, 14 Sep 2024 12:50:45 GMT
      |_    Content-Length: 0
      1 service unrecognized despite returning data. If you know the service/version, please submit the following fingerprint at https://nmap.org/cgi-bin/submit.cgi?new-service :
      SF-Port3000-TCP:V=7.94SVN%I=7%D=9/14%Time=66E5869C%P=x86_64-pc-linux-gnu%r
      SF:(GenericLines,67,"HTTP/1\.1\x20400\x20Bad\x20Request\r\nContent-Type:\x
      SF:20text/plain;\x20charset=utf-8\r\nConnection:\x20close\r\n\r\n400\x20Ba
      SF:d\x20Request")%r(GetRequest,2A60,"HTTP/1\.0\x20200\x20OK\r\nCache-Contr
      SF:ol:\x20max-age=0,\x20private,\x20must-revalidate,\x20no-transform\r\nCo
      SF:ntent-Type:\x20text/html;\x20charset=utf-8\r\nSet-Cookie:\x20i_like_git
      SF:ea=98c7984fd88212c6;\x20Path=/;\x20HttpOnly;\x20SameSite=Lax\r\nSet-Coo
      SF:kie:\x20_csrf=tgJrdjjVc4E5bW9ek-p31ZrbXSc6MTcyNjMxODIzOTc2MDY5ODQ5MA;\x
      SF:20Path=/;\x20Max-Age=86400;\x20HttpOnly;\x20SameSite=Lax\r\nX-Frame-Opt
      SF:ions:\x20SAMEORIGIN\r\nDate:\x20Sat,\x2014\x20Sep\x202024\x2012:50:39\x
      SF:20GMT\r\n\r\n<!DOCTYPE\x20html>\n<html\x20lang=\"en-US\"\x20class=\"the
      SF:me-auto\">\n<head>\n\t<meta\x20name=\"viewport\"\x20content=\"width=dev
      SF:ice-width,\x20initial-scale=1\">\n\t<title>GreenHorn</title>\n\t<link\x
      SF:20rel=\"manifest\"\x20href=\"data:application/json;base64,eyJuYW1lIjoiR
      SF:3JlZW5Ib3JuIiwic2hvcnRfbmFtZSI6IkdyZWVuSG9ybiIsInN0YXJ0X3VybCI6Imh0dHA6
      SF:Ly9ncmVlbmhvcm4uaHRiOjMwMDAvIiwiaWNvbnMiOlt7InNyYyI6Imh0dHA6Ly9ncmVlbmh
      SF:vcm4uaHRiOjMwMDAvYXNzZXRzL2ltZy9sb2dvLnBuZyIsInR5cGUiOiJpbWFnZS9wbmciLC
      SF:JzaXplcyI6IjUxMng1MTIifSx7InNyYyI6Imh0dHA6Ly9ncmVlbmhvcm4uaHRiOjMwMDAvY
      SF:X")%r(Help,67,"HTTP/1\.1\x20400\x20Bad\x20Request\r\nContent-Type:\x20t
      SF:ext/plain;\x20charset=utf-8\r\nConnection:\x20close\r\n\r\n400\x20Bad\x
      SF:20Request")%r(HTTPOptions,1A4,"HTTP/1\.0\x20405\x20Method\x20Not\x20All
      SF:owed\r\nAllow:\x20HEAD\r\nAllow:\x20HEAD\r\nAllow:\x20GET\r\nCache-Cont
      SF:rol:\x20max-age=0,\x20private,\x20must-revalidate,\x20no-transform\r\nS
      SF:et-Cookie:\x20i_like_gitea=d7d4f84a96cc0455;\x20Path=/;\x20HttpOnly;\x2
      SF:0SameSite=Lax\r\nSet-Cookie:\x20_csrf=W3IufYSw6N9E6wfnc7vpXsS9G2Q6MTcyN
      SF:jMxODI0NTkyMTc1MTM2Nw;\x20Path=/;\x20Max-Age=86400;\x20HttpOnly;\x20Sam
      SF:eSite=Lax\r\nX-Frame-Options:\x20SAMEORIGIN\r\nDate:\x20Sat,\x2014\x20S
      SF:ep\x202024\x2012:50:45\x20GMT\r\nContent-Length:\x200\r\n\r\n")%r(RTSPR
      SF:equest,67,"HTTP/1\.1\x20400\x20Bad\x20Request\r\nContent-Type:\x20text/
      SF:plain;\x20charset=utf-8\r\nConnection:\x20close\r\n\r\n400\x20Bad\x20Re
      SF:quest");
      Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

- An `http` server is running on both port `80` and `3000`.
- Port `80` is hosting a `pluckCMS` page.

![image](https://github.com/user-attachments/assets/2277bffe-774e-4f81-b4cb-fe90553c7603)

- Port `3000` is hosting a `gitea` page.

![image](https://github.com/user-attachments/assets/545d9a07-b6ab-4dc2-b380-3614251c4d7c)

- FFUF's output for port `80`

![image](https://github.com/user-attachments/assets/b1b584b3-0424-4fb5-b876-e44a739148ba)

- FFUF's output for port `3000`

![image](https://github.com/user-attachments/assets/c9ef7fd1-8dd8-4bc1-9c9b-8c6b5d6e70dc)
  
- The `sitemap.xml` reveals a sitemap for repositories.

![image](https://github.com/user-attachments/assets/0221918b-98c7-4cd0-86d2-31f71292e083)

- Greenhorn source code repository

![image](https://github.com/user-attachments/assets/e49857fd-5e92-4292-808a-9d29bf546a34)

### FInding Sensitive info

- I cloned the repository with git.

![image](https://github.com/user-attachments/assets/16c5ac85-4259-4504-85dc-321a8033a56d)

- I decided to grep for admin credentials using the keyword `admin`.I got an admin email.

![image](https://github.com/user-attachments/assets/72c7e9d9-a1e7-41df-abc8-02b560194cfb)

- I traced the directory to get other sensitive info and discovered a `pass.php` file.

![image](https://github.com/user-attachments/assets/f75a999c-a157-46c6-9d14-cb602376015f)

- The file contains an hash.

![image](https://github.com/user-attachments/assets/53415884-a1ae-4b4b-a432-875d8b73ce0c)

- I was able to get the password with `crackstation.net`.

![image](https://github.com/user-attachments/assets/8eb573af-3c65-4926-98a0-f2185006563d)

- `PLUCK CMS` is vulnerable to  authenticated Remote Code Execution. I wrote an exploit for [it](https://github.com/SENSEiXENUS/senseixenus.github.io/tree/main/posts/ctf/HTB/scripts/CVE-2023-50564_PluckCMS).

![image](https://github.com/user-attachments/assets/05aa7d6c-ba46-4e9f-9c80-14c69617fc87)

- RCE

![image](https://github.com/user-attachments/assets/5d184525-5dca-415b-917a-5a48e2f61baa)

- Shell as `www-data`

![image](https://github.com/user-attachments/assets/3399abc0-8215-4aaa-82e1-f9b73387e328)

### User junior

- I logged into the user `junior` with the `PLUCKCMS` creds

![image](https://github.com/user-attachments/assets/4654fb46-01ad-4158-afd7-abd3cffe61cd)

### Depix

- User `junior`'s home directory contains a pdf file named `Using Openvas.pdf`.

![image](https://github.com/user-attachments/assets/27911991-b07b-4c75-a8dd-d5f186374ebc)

- I transferred it to my machine,the pdf is not from the `root` user to user `junior`.I noticed a pixelized image in it.

![image](https://github.com/user-attachments/assets/435f51b4-c7ea-4284-a771-8941ee9e9b34)

- In a bid to extract the file, I used a tool `pdftohtml` to extract the image content.

![image](https://github.com/user-attachments/assets/69ff25c3-8414-4c55-8cc0-21965ea639a1)

- Then, I used the tool `Depix` to depixelize the image.

![image](https://github.com/user-attachments/assets/c2896a73-df00-44ee-93ad-46ee4bf1a0fb)

- Output image

![image](https://github.com/user-attachments/assets/cde0d769-bdb3-41d2-8ecf-4523bf29e1dd)

- At first, I got a weird text, I tried to log into the `root` account with the password and it worked surprisingly.

![image](https://github.com/user-attachments/assets/94195088-93ef-4064-9381-eeb18a7b97f7)

- Root

![image](https://github.com/user-attachments/assets/bb5a4aee-ed77-4649-96e0-9ddc020d3644)

-------------------

### THANKS FOR READING!!!

--------------------

### REFERENCE:

- [My Exploit](https://github.com/SENSEiXENUS/senseixenus.github.io/tree/main/posts/ctf/HTB/scripts/CVE-2023-50564_PluckCMS)

---------------------
















