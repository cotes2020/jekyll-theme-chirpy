---------------

### CTF-: TRYHACKME
### Lab-: Silver Platter

---------------

![image](https://github.com/user-attachments/assets/ce041ebd-a32b-4418-9494-c24ffc490059)

----------------

- Rustscan's output

```bash
❯ rustscan -a 10.10.45.81 -- -Pn -sC -sV
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
Open 10.10.45.81:22
Open 10.10.45.81:80
Open 10.10.45.81:8080
[~] Starting Script(s)
[>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")

Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
[~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-03-06 00:06 WAT
NSE: Loaded 156 scripts for scanning.
NSE: Script Pre-scanning.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 00:06
Completed NSE at 00:06, 0.00s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 00:06
Completed NSE at 00:06, 0.00s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 00:06
Completed NSE at 00:06, 0.00s elapsed
Initiating Parallel DNS resolution of 1 host. at 00:06
Completed Parallel DNS resolution of 1 host. at 00:06, 0.00s elapsed
DNS resolution of 1 IPs took 0.01s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
Initiating Connect Scan at 00:06
Scanning 10.10.45.81 [3 ports]
Discovered open port 22/tcp on 10.10.45.81
Discovered open port 8080/tcp on 10.10.45.81
Discovered open port 80/tcp on 10.10.45.81
Completed Connect Scan at 00:06, 0.17s elapsed (3 total ports)
Initiating Service scan at 00:06
Scanning 3 services on 10.10.45.81
Completed Service scan at 00:07, 84.40s elapsed (3 services on 1 host)
NSE: Script scanning 10.10.45.81.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 00:07
Completed NSE at 00:07, 5.29s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 00:07
Completed NSE at 00:07, 1.53s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 00:07
Completed NSE at 00:07, 0.00s elapsed
Nmap scan report for 10.10.45.81
Host is up, received user-set (0.17s latency).
Scanned at 2025-03-06 00:06:25 WAT for 92s

PORT     STATE SERVICE    REASON  VERSION
22/tcp   open  ssh        syn-ack OpenSSH 8.9p1 Ubuntu 3ubuntu0.4 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey: 
|   256 1b:1c:87:8a:fe:34:16:c9:f7:82:37:2b:10:8f:8b:f1 (ECDSA)
| ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBJ0ia1tcuNvK0lfuy3Ep2dsElFfxouO3VghX5Rltu77M33pFvTeCn9t5A8NReq3felAqPi+p+/0eRRfYuaeHRT4=
|   256 26:6d:17:ed:83:9e:4f:2d:f6:cd:53:17:c8:80:3d:09 (ED25519)
|_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKecigNtiy6tW5ojXM3xQkbtTOwK+vqvMoJZnIxVowju
80/tcp   open  http       syn-ack nginx 1.18.0 (Ubuntu)
|_http-server-header: nginx/1.18.0 (Ubuntu)
| http-methods: 
|_  Supported Methods: GET HEAD
|_http-title: Hack Smarter Security
8080/tcp open  http-proxy syn-ack
|_http-title: Error
| fingerprint-strings: 
|   FourOhFourRequest, HTTPOptions: 
|     HTTP/1.1 404 Not Found
|     Connection: close
|     Content-Length: 74
|     Content-Type: text/html
|     Date: Wed, 05 Mar 2025 23:06:32 GMT
|     <html><head><title>Error</title></head><body>404 - Not Found</body></html>
|   GenericLines, Help, Kerberos, LDAPSearchReq, LPDString, RTSPRequest, SMBProgNeg, SSLSessionReq, Socks5, TLSSessionReq, TerminalServerCookie: 
|     HTTP/1.1 400 Bad Request
|     Content-Length: 0
|     Connection: close
|   GetRequest: 
|     HTTP/1.1 404 Not Found
|     Connection: close
|     Content-Length: 74
|     Content-Type: text/html
|     Date: Wed, 05 Mar 2025 23:06:31 GMT
|_    <html><head><title>Error</title></head><body>404 - Not Found</body></html>
1 service unrecognized despite returning data. If you know the service/version, please submit the following fingerprint at https://nmap.org/cgi-bin/submit.cgi?new-service :
SF-Port8080-TCP:V=7.94SVN%I=7%D=3/6%Time=67C8D8F8%P=x86_64-pc-linux-gnu%r(
SF:GetRequest,C9,"HTTP/1\.1\x20404\x20Not\x20Found\r\nConnection:\x20close
SF:\r\nContent-Length:\x2074\r\nContent-Type:\x20text/html\r\nDate:\x20Wed
SF:,\x2005\x20Mar\x202025\x2023:06:31\x20GMT\r\n\r\n<html><head><title>Err
SF:or</title></head><body>404\x20-\x20Not\x20Found</body></html>")%r(HTTPO
SF:ptions,C9,"HTTP/1\.1\x20404\x20Not\x20Found\r\nConnection:\x20close\r\n
SF:Content-Length:\x2074\r\nContent-Type:\x20text/html\r\nDate:\x20Wed,\x2
SF:005\x20Mar\x202025\x2023:06:32\x20GMT\r\n\r\n<html><head><title>Error</
SF:title></head><body>404\x20-\x20Not\x20Found</body></html>")%r(RTSPReque
SF:st,42,"HTTP/1\.1\x20400\x20Bad\x20Request\r\nContent-Length:\x200\r\nCo
SF:nnection:\x20close\r\n\r\n")%r(FourOhFourRequest,C9,"HTTP/1\.1\x20404\x
SF:20Not\x20Found\r\nConnection:\x20close\r\nContent-Length:\x2074\r\nCont
SF:ent-Type:\x20text/html\r\nDate:\x20Wed,\x2005\x20Mar\x202025\x2023:06:3
SF:2\x20GMT\r\n\r\n<html><head><title>Error</title></head><body>404\x20-\x
SF:20Not\x20Found</body></html>")%r(Socks5,42,"HTTP/1\.1\x20400\x20Bad\x20
SF:Request\r\nContent-Length:\x200\r\nConnection:\x20close\r\n\r\n")%r(Gen
SF:ericLines,42,"HTTP/1\.1\x20400\x20Bad\x20Request\r\nContent-Length:\x20
SF:0\r\nConnection:\x20close\r\n\r\n")%r(Help,42,"HTTP/1\.1\x20400\x20Bad\
SF:x20Request\r\nContent-Length:\x200\r\nConnection:\x20close\r\n\r\n")%r(
SF:SSLSessionReq,42,"HTTP/1\.1\x20400\x20Bad\x20Request\r\nContent-Length:
SF:\x200\r\nConnection:\x20close\r\n\r\n")%r(TerminalServerCookie,42,"HTTP
SF:/1\.1\x20400\x20Bad\x20Request\r\nContent-Length:\x200\r\nConnection:\x
SF:20close\r\n\r\n")%r(TLSSessionReq,42,"HTTP/1\.1\x20400\x20Bad\x20Reques
SF:t\r\nContent-Length:\x200\r\nConnection:\x20close\r\n\r\n")%r(Kerberos,
SF:42,"HTTP/1\.1\x20400\x20Bad\x20Request\r\nContent-Length:\x200\r\nConne
SF:ction:\x20close\r\n\r\n")%r(SMBProgNeg,42,"HTTP/1\.1\x20400\x20Bad\x20R
SF:equest\r\nContent-Length:\x200\r\nConnection:\x20close\r\n\r\n")%r(LPDS
SF:tring,42,"HTTP/1\.1\x20400\x20Bad\x20Request\r\nContent-Length:\x200\r\
SF:nConnection:\x20close\r\n\r\n")%r(LDAPSearchReq,42,"HTTP/1\.1\x20400\x2
SF:0Bad\x20Request\r\nContent-Length:\x200\r\nConnection:\x20close\r\n\r\n
SF:");
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel
```

- On port 80, I discovered a username for one silverpeas management portal.

![image](https://github.com/user-attachments/assets/72095802-6ad4-4269-b984-0657d7fb7a4d)

- On port `8080`,I discovered `silverpeas` on directory `silverpeas` with a default login page.

![image](https://github.com/user-attachments/assets/9844d9f3-81a4-4412-b0d0-10669a63a2b1)

- This version of silverpeas is vulnerable to `Authentication bypass` as explained in this [Github Repo](https://gist.github.com/ChrisPritchard/4b6d5c70d9329ef116266a6c238dcb2d).I was able to access user `scr1ptkiddy` account.

![image](https://github.com/user-attachments/assets/12e8b6fa-8d77-4031-8a68-50993cdb114e)


- Request made-:

![image](https://github.com/user-attachments/assets/d6cb6795-f773-489c-8911-635a30d9b821)

- I was able to get another user `Manager` in the recent mails sent to the current user.

![image](https://github.com/user-attachments/assets/907b5225-ebc2-4167-a0b7-5e7bea2f0bd2)

- I discovered ssh credentials for the current user's mail.

![image](https://github.com/user-attachments/assets/4d42140e-2f59-4e48-9730-5ebe76164abc)

- User `tim` ssh access-:

![image](https://github.com/user-attachments/assets/c51b11e3-af74-4030-8d0a-f01e0567d517)

-----------------

### Privesc with exposed password

------------------

- User `tim` is part of the `adm` groups which means we can read logs.I checked for password with grep and discovered `postgresql` logs.

![image](https://github.com/user-attachments/assets/9e26b668-3709-4ac6-a0e7-be77dcb7fe34)

- There are 3 users on the server `Root,Tim and Tyler`.

![image](https://github.com/user-attachments/assets/6a0aa78e-184e-434d-a8fb-c567a401118d)

- The password worked for user `Tyler`.Ssh access-:

![image](https://github.com/user-attachments/assets/2739375b-c2b6-4057-8f43-ab4313dd699d)

- Since, user `tyler` is part of the group `sudo`,we can `sudo` with no constraints.I escalated to root witht he steps showing in the image.

![image](https://github.com/user-attachments/assets/d104712f-5cc2-4886-912c-19a69445bf22)

- Root-:

![image](https://github.com/user-attachments/assets/bcc99705-9c59-4643-9b2b-6afabc2482c2)


----------------

### Reference-:

----------------

- [Silverpeas Auth Bypass](https://gist.github.com/ChrisPritchard/4b6d5c70d9329ef116266a6c238dcb2d)

-----------------
