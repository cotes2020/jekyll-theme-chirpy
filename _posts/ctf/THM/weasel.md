--------------

### LAB-: Weasel
### CTF-: THM

--------------

![image](https://github.com/user-attachments/assets/e58f49c6-0760-41a1-8432-934306df5eb7)

--------------

- Rustscan's result-:

```bash
❯ rustscan -a 10.10.53.32 -- -Pn -sC -sV
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
Open 10.10.53.32:22
Open 10.10.53.32:135
Open 10.10.53.32:139
Open 10.10.53.32:445
Open 10.10.53.32:3389
Open 10.10.53.32:5985
Open 10.10.53.32:8888
Open 10.10.53.32:47001
Open 10.10.53.32:49664
Open 10.10.53.32:49665
Open 10.10.53.32:49667
Open 10.10.53.32:49668
Open 10.10.53.32:49669
Open 10.10.53.32:49670
Open 10.10.53.32:49672
[~] Starting Script(s)
[>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")

Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
[~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-12-12 05:32 WAT
NSE: Loaded 156 scripts for scanning.
NSE: Script Pre-scanning.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 05:32
Completed NSE at 05:32, 0.00s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 05:32
Completed NSE at 05:32, 0.00s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 05:32
Completed NSE at 05:32, 0.00s elapsed
Initiating Parallel DNS resolution of 1 host. at 05:32
Completed Parallel DNS resolution of 1 host. at 05:32, 0.04s elapsed
DNS resolution of 1 IPs took 0.04s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
Initiating Connect Scan at 05:32
Scanning 10.10.53.32 [15 ports]
Discovered open port 135/tcp on 10.10.53.32
Discovered open port 3389/tcp on 10.10.53.32
Discovered open port 139/tcp on 10.10.53.32
Discovered open port 8888/tcp on 10.10.53.32
Discovered open port 445/tcp on 10.10.53.32
Discovered open port 22/tcp on 10.10.53.32
Discovered open port 49669/tcp on 10.10.53.32
Discovered open port 49672/tcp on 10.10.53.32
Discovered open port 49667/tcp on 10.10.53.32
Discovered open port 49665/tcp on 10.10.53.32
Discovered open port 49668/tcp on 10.10.53.32
Discovered open port 5985/tcp on 10.10.53.32
Discovered open port 49664/tcp on 10.10.53.32
Discovered open port 49670/tcp on 10.10.53.32
Discovered open port 47001/tcp on 10.10.53.32
Completed Connect Scan at 05:32, 0.34s elapsed (15 total ports)
Initiating Service scan at 05:32
Scanning 15 services on 10.10.53.32
Service scan Timing: About 60.00% done; ETC: 05:33 (0:00:37 remaining)
Completed Service scan at 05:33, 57.22s elapsed (15 services on 1 host)
NSE: Script scanning 10.10.53.32.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 05:33
Completed NSE at 05:33, 11.09s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 05:33
Completed NSE at 05:33, 0.88s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 05:33
Completed NSE at 05:33, 0.01s elapsed
Nmap scan report for 10.10.53.32
Host is up, received user-set (0.17s latency).
Scanned at 2024-12-12 05:32:21 WAT for 70s

PORT      STATE SERVICE       REASON  VERSION
22/tcp    open  ssh           syn-ack OpenSSH for_Windows_7.7 (protocol 2.0)
| ssh-hostkey: 
|   2048 2b:17:d8:8a:1e:8c:99:bc:5b:f5:3d:0a:5e:ff:5e:5e (RSA)
| ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDBae1NsdsMcZJNQQ2wjF2sxXK2ZF3c7qqW3TN/q91pWiDee3nghS1J1FZrUXaEj0wnAAAbYRg5vbRZRP9oEagBwfWG3QJ9AO6s5UC+iTjX+YKH6phKNmsY5N/LKY4+2EDcwa5R4uznAC/2Cy5EG6s7izvABLcRh3h/w4rVHduiwrueAZF9UjzlHBOxHDOPPVtg+0dniGhcXRuEU5FYRA8/IPL8P97djscu23btk/hH3iqdQWlC9b0CnOkD8kuyDybq9nFaebAxDW4XFj7KjCRuuu0dyn5Sr62FwRXO4wu08ePUEmJF1Gl3/fdYe3vj+iE2yewOFAhzbmFWEWtztjJb
|   256 3c:c0:fd:b5:c1:57:ab:75:ac:81:10:ae:e2:98:12:0d (ECDSA)
| ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBOGl51l9Z4Mg4hFDcQz8v6XRlABMyVPWlkEXrJIg53piZhZ9WKYn0Gi4fKkzo3blDAsdqpGFQ11wwocBCSJGjQU=
|   256 e9:f0:30:be:e6:cf:ef:fe:2d:14:21:a0:ac:45:7b:70 (ED25519)
|_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOHw9uTZkIMEgcZPW9Z28Mm+FX66+hkxk+8rOu7oI6J9
135/tcp   open  msrpc         syn-ack Microsoft Windows RPC
139/tcp   open  netbios-ssn   syn-ack Microsoft Windows netbios-ssn
445/tcp   open  microsoft-ds? syn-ack
3389/tcp  open  ms-wbt-server syn-ack Microsoft Terminal Services
|_ssl-date: 2024-12-12T04:33:29+00:00; -2s from scanner time.
| ssl-cert: Subject: commonName=DEV-DATASCI-JUP
| Issuer: commonName=DEV-DATASCI-JUP
| Public Key type: rsa
| Public Key bits: 2048
| Signature Algorithm: sha256WithRSAEncryption
| Not valid before: 2024-12-11T04:29:13
| Not valid after:  2025-06-12T04:29:13
| MD5:   b6a6:bafa:59ae:c6de:c699:10af:384a:cf0e
| SHA-1: 2eab:b3d3:21d6:7307:b437:417a:6382:c108:54e1:936a
| -----BEGIN CERTIFICATE-----
| MIIC4jCCAcqgAwIBAgIQFLA3amGZEIRNZvZ9UhtBVjANBgkqhkiG9w0BAQsFADAa
| MRgwFgYDVQQDEw9ERVYtREFUQVNDSS1KVVAwHhcNMjQxMjExMDQyOTEzWhcNMjUw
| NjEyMDQyOTEzWjAaMRgwFgYDVQQDEw9ERVYtREFUQVNDSS1KVVAwggEiMA0GCSqG
| SIb3DQEBAQUAA4IBDwAwggEKAoIBAQDTpPOevodQzvAwn0ZwNk1/FuaZzbpwrygS
| zzf5pdw6RIVJ7/2iO1GYXMuCYtnIwWG5Qi5buVj4pt721zePVlhR5XUubbrwXfUr
| dkaXymc8+iFTqznND6l6KXFHzAGjRs5WUNo2NioGAkAfVWDDeo4XncOb/s9hdLXv
| EmqwBd4QTpa+Jy8w6sfACqdwXZ6Da+MfflYNTBtOKp9ZubCi7or8khV2iwtCsV1N
| nsPSgcuEV/aiyIzsGVvqPGtf/LzYM0/pqdbcY1aYTCkJM8QmpVOd+cdTt4oczQpJ
| Xg/kG9C8FHp+3aZSt+m4FVJGAnbxuQsEae2xWItw1ED3i9IDg4xVAgMBAAGjJDAi
| MBMGA1UdJQQMMAoGCCsGAQUFBwMBMAsGA1UdDwQEAwIEMDANBgkqhkiG9w0BAQsF
| AAOCAQEAlG+2JZjAAkWQUE1OCjo1X1NymcRm1/XuuCxIAo2AwlqZt3EENf6eKbC8
| 9PK7oMQj6c2fEPGEDwiw10mP4QPOk0FAUqpUn/eodgSheJpzmFk7HRciTwCNaT+2
| rMDuKyrl8YRZ5+zvPKa3ooG09SLI74DmTxXatHpqcPOrB5yOcimC9iQPjkFUDGGw
| +t3NvzGoFYqF7mn1ek5XqWYYxY2et5wGMwhJjoeSGKzeXDFME8TtCxUbtc05NAqh
| 6ZYMBkRfCs+6q+fQD9wfH2+04jjeYB0d3Xwt1jlREztsQFbWsp1fXyuBdQL81F8D
| PDFtdAywKx4o/DFA6LYHuy6MoS9OXw==
|_-----END CERTIFICATE-----
| rdp-ntlm-info: 
|   Target_Name: DEV-DATASCI-JUP
|   NetBIOS_Domain_Name: DEV-DATASCI-JUP
|   NetBIOS_Computer_Name: DEV-DATASCI-JUP
|   DNS_Domain_Name: DEV-DATASCI-JUP
|   DNS_Computer_Name: DEV-DATASCI-JUP
|   Product_Version: 10.0.17763
|_  System_Time: 2024-12-12T04:33:19+00:00
5985/tcp  open  http          syn-ack Microsoft HTTPAPI httpd 2.0 (SSDP/UPnP)
|_http-server-header: Microsoft-HTTPAPI/2.0
|_http-title: Not Found
8888/tcp  open  http          syn-ack Tornado httpd 6.0.3
| http-title: Jupyter Notebook
|_Requested resource was /login?next=%2Ftree%3F
|_http-server-header: TornadoServer/6.0.3
|_http-favicon: Unknown favicon MD5: 97C6417ED01BDC0AE3EF32AE4894FD03
| http-methods: 
|_  Supported Methods: GET POST
| http-robots.txt: 1 disallowed entry 
|_/ 
47001/tcp open  http          syn-ack Microsoft HTTPAPI httpd 2.0 (SSDP/UPnP)
|_http-title: Not Found
|_http-server-header: Microsoft-HTTPAPI/2.0
49664/tcp open  msrpc         syn-ack Microsoft Windows RPC
49665/tcp open  msrpc         syn-ack Microsoft Windows RPC
49667/tcp open  msrpc         syn-ack Microsoft Windows RPC
49668/tcp open  msrpc         syn-ack Microsoft Windows RPC
49669/tcp open  msrpc         syn-ack Microsoft Windows RPC
49670/tcp open  msrpc         syn-ack Microsoft Windows RPC
49672/tcp open  msrpc         syn-ack Microsoft Windows RPC
Service Info: OS: Windows; CPE: cpe:/o:microsoft:windows

Host script results:
| smb2-security-mode: 
|   3:1:1: 
|_    Message signing enabled but not required
| smb2-time: 
|   date: 2024-12-12T04:33:23
|_  start_date: N/A
|_clock-skew: mean: -1s, deviation: 0s, median: -1s
| p2p-conficker: 
|   Checking for Conficker.C or higher...
|   Check 1 (port 58923/tcp): CLEAN (Couldn't connect)
|   Check 2 (port 56118/tcp): CLEAN (Couldn't connect)
|   Check 3 (port 35399/udp): CLEAN (Failed to receive data)
|   Check 4 (port 23016/udp): CLEAN (Timeout)
|_  0/4 checks are positive: Host is CLEAN or ports are blocked
```

- Smb enumeration with netexec,we have anonymous access-:

![image](https://github.com/user-attachments/assets/d7e9c787-2edd-4147-97a7-9ad0c0b0c530)

- Dumping the shares

```bash
❯ nxc smb "10.10.77.27" -u 'guest' -p '' -M spider_plus -o DOWNLOAD_FLAG=True
SMB         10.10.77.27     445    DEV-DATASCI-JUP  [*] Windows 10 / Server 2019 Build 17763 x64 (name:DEV-DATASCI-JUP) (domain:DEV-DATASCI-JUP) (signing:False) (SMBv1:False)
SMB         10.10.77.27     445    DEV-DATASCI-JUP  [+] DEV-DATASCI-JUP\guest: 
SPIDER_PLUS 10.10.77.27     445    DEV-DATASCI-JUP  [*] Started module spidering_plus with the following options:
SPIDER_PLUS 10.10.77.27     445    DEV-DATASCI-JUP  [*]  DOWNLOAD_FLAG: True
SPIDER_PLUS 10.10.77.27     445    DEV-DATASCI-JUP  [*]     STATS_FLAG: True
SPIDER_PLUS 10.10.77.27     445    DEV-DATASCI-JUP  [*] EXCLUDE_FILTER: ['print$', 'ipc$']
SPIDER_PLUS 10.10.77.27     445    DEV-DATASCI-JUP  [*]   EXCLUDE_EXTS: ['ico', 'lnk']
SPIDER_PLUS 10.10.77.27     445    DEV-DATASCI-JUP  [*]  MAX_FILE_SIZE: 50 KB
SPIDER_PLUS 10.10.77.27     445    DEV-DATASCI-JUP  [*]  OUTPUT_FOLDER: /home/sensei/.nxc/modules/nxc_spider_plus
SMB         10.10.77.27     445    DEV-DATASCI-JUP  [*] Enumerated shares
SMB         10.10.77.27     445    DEV-DATASCI-JUP  Share           Permissions     Remark
SMB         10.10.77.27     445    DEV-DATASCI-JUP  -----           -----------     ------
SMB         10.10.77.27     445    DEV-DATASCI-JUP  ADMIN$                          Remote Admin
SMB         10.10.77.27     445    DEV-DATASCI-JUP  C$                              Default share
SMB         10.10.77.27     445    DEV-DATASCI-JUP  datasci-team    READ,WRITE      
SMB         10.10.77.27     445    DEV-DATASCI-JUP  IPC$            READ            Remote IPC
SPIDER_PLUS 10.10.77.27     445    DEV-DATASCI-JUP  [+] Saved share-file metadata to "/home/sensei/.nxc/modules/nxc_spider_plus/10.10.77.27.json".
SPIDER_PLUS 10.10.77.27     445    DEV-DATASCI-JUP  [*] SMB Shares:           4 (ADMIN$, C$, datasci-team, IPC$)
SPIDER_PLUS 10.10.77.27     445    DEV-DATASCI-JUP  [*] SMB Readable Shares:  2 (datasci-team, IPC$)
SPIDER_PLUS 10.10.77.27     445    DEV-DATASCI-JUP  [*] SMB Writable Shares:  1 (datasci-team)
SPIDER_PLUS 10.10.77.27     445    DEV-DATASCI-JUP  [*] SMB Filtered Shares:  1
SPIDER_PLUS 10.10.77.27     445    DEV-DATASCI-JUP  [*] Total folders found:  4
SPIDER_PLUS 10.10.77.27     445    DEV-DATASCI-JUP  [*] Total files found:    13
SPIDER_PLUS 10.10.77.27     445    DEV-DATASCI-JUP  [*] Files filtered:       5
SPIDER_PLUS 10.10.77.27     445    DEV-DATASCI-JUP  [*] File size average:    356.34 KB
SPIDER_PLUS 10.10.77.27     445    DEV-DATASCI-JUP  [*] File size min:        12 B
SPIDER_PLUS 10.10.77.27     445    DEV-DATASCI-JUP  [*] File size max:        3.33 MB
SPIDER_PLUS 10.10.77.27     445    DEV-DATASCI-JUP  [*] File unique exts:     5 (csv, pdf, ipynb, txt, html)
SPIDER_PLUS 10.10.77.27     445    DEV-DATASCI-JUP  [*] Unmodified files:     8
SPIDER_PLUS 10.10.77.27     445    DEV-DATASCI-JUP  [*] All files were not changed.
SPIDER_PLUS 10.10.77.27     445    DEV-DATASCI-JUP  [+] All files processed successfully.
```
- Files-:

![image](https://github.com/user-attachments/assets/f67b8b3d-966c-42fc-a36b-b01d3ef55455)

- On port `8888`,we have python data framework installed on it named `tornado` which requires a token to login.

![image](https://github.com/user-attachments/assets/52196f9d-b2f3-4065-9b50-7fff5ce99d83)

- Jupyter token discovered amongst the dumped smb shares' files.

![image](https://github.com/user-attachments/assets/64702946-f9d7-41b5-922a-9d58b0f42cfe)

- Tornado accessed-:

![image](https://github.com/user-attachments/assets/84ccde2a-6353-4292-8249-82aa24fc6c6f)

------------------

### RCE with Tornado

-----------------

- You can access `Tornado` terminal which is also a python3 interactive shell.

![image](https://github.com/user-attachments/assets/f347dee4-3b46-4fa1-b541-83ec4a1dd38e)

- I spawned a bash reverse shell.

![image](https://github.com/user-attachments/assets/4f658f9a-864e-4dd7-b37e-81f488014f48)

- I discovered a ssh private key in the home directory.

![image](https://github.com/user-attachments/assets/83209472-3b79-4420-a714-5390e30598a7)

- SSH access to window host-:

![image](https://github.com/user-attachments/assets/d459e221-1c3e-4040-b160-6fc6cd479e89)

- I ran winpeas.exe to get more info of the host.I got 2 privilege escalation vectors `AlwaysInstallElevated` and `Auto-Logon creds`.

![image](https://github.com/user-attachments/assets/b70cfbdd-ffe8-467f-b20d-adf83ba3046c)

![image](https://github.com/user-attachments/assets/fcfeaccd-bcdf-49ce-9d06-4c4c7fcdb4ad)

- I created an msi reverse shell payload with msfvenom.

![image](https://github.com/user-attachments/assets/9a6cfd62-c300-4bd9-a3f4-6fbd511e85b5)

- I used the runas command to execute the msi payload.`AlwaysInstallElevated` allows us to install a msi package with administrative privileges.It requires the user's password although we've gotten that from the autologon creds.

![image](https://github.com/user-attachments/assets/a05b7d24-04be-4807-a704-e2b1ccc1936c)

- Reverse shell as `nt authority/system`-:

![image](https://github.com/user-attachments/assets/cda0d289-b4d2-4b02-9720-ef67924ba80a)


--------------------------



