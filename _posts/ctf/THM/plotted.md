------------------

### Platform: THM
### Lab: Plotted

------------------

![image](https://github.com/user-attachments/assets/2c4dc886-18bc-4d3b-bac3-ddd7cc605a7d)

------------------

- Rustscan's output

```bash
❯ rustscan -a 10.10.51.85  -- -Pn -sC -sV
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
Open 10.10.51.85:22
Open 10.10.51.85:80
Open 10.10.51.85:445
[~] Starting Script(s)
[>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")

Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
[~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-11-19 17:40 WAT
NSE: Loaded 156 scripts for scanning.
NSE: Script Pre-scanning.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 17:40
Completed NSE at 17:40, 0.00s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 17:40
Completed NSE at 17:40, 0.00s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 17:40
Completed NSE at 17:40, 0.00s elapsed
Initiating Parallel DNS resolution of 1 host. at 17:40
Completed Parallel DNS resolution of 1 host. at 17:40, 0.13s elapsed
DNS resolution of 1 IPs took 0.13s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
Initiating Connect Scan at 17:40
Scanning 10.10.51.85 [3 ports]
Discovered open port 22/tcp on 10.10.51.85
Discovered open port 445/tcp on 10.10.51.85
Discovered open port 80/tcp on 10.10.51.85
Completed Connect Scan at 17:40, 0.25s elapsed (3 total ports)
Initiating Service scan at 17:40
Scanning 3 services on 10.10.51.85
Completed Service scan at 17:40, 19.27s elapsed (3 services on 1 host)
NSE: Script scanning 10.10.51.85.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 17:40
NSE Timing: About 98.35% done; ETC: 17:41 (0:00:01 remaining)
Completed NSE at 17:41, 41.15s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 17:41
Completed NSE at 17:41, 0.96s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 17:41
Completed NSE at 17:41, 0.01s elapsed
Nmap scan report for 10.10.51.85
Host is up, received user-set (0.25s latency).
Scanned at 2024-11-19 17:40:19 WAT for 62s

PORT    STATE SERVICE REASON  VERSION
22/tcp  open  ssh     syn-ack OpenSSH 8.2p1 Ubuntu 4ubuntu0.3 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey: 
|   3072 a3:6a:9c:b1:12:60:b2:72:13:09:84:cc:38:73:44:4f (RSA)
| ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDSk+lD9baengtZg1zPPR4SVHS2JWnI2fkH90VgBhh7iRQKND35/SOP13L/a3oDA3qub2FgT1ejvHA3D7wcY5ZCgq17mLXCw6WW0IDRWbH7kyPUBolc9h6ZI+Zpiyr7sUitywYRW5WCrEHpUs6ol92pR46UnXfwmsuvY6RVWaviUT95xmUZPgVUpw8PJjDU3TJpCYEtnW6AoEO0/7OSx7LkbrvMCnIitZi2mcBvfc/WbCmvtiOLsKBwh21VCXUhLAzVGZ5xOdD4rAcD3OACM/gJVGe5wJJJL1Abt/1flGBJyvYZUoz/JQxoa+HpjcRXmSa+nprBxPdvmQDjsf+UPmpegVPME9iNfkmoEWDgN/lWWZnyPC8kBzhxkM8/rQkfmJlK1F9Lq60BoF6ipj6/W1O94yzaFL7+mNRFrV86zgZhbr1l9MQyUcJoDnlCMygYo1HhkYsfGBR1Tu5M031sZpVNIEUSSfXwrlUX4k4ThaCPDsEMB941K/OUbAuhmQo2MGE=
|   256 b9:3f:84:00:f4:d1:fd:c8:e7:8d:98:03:38:74:a1:4d (ECDSA)
| ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBMLlGKfQy13XGzOkqSgnrB7thrs/Bh+kpzchoHn6PCCBDOZ0j3uFzQWvl5uimdLDXombozAcFHlzDjGL50hKarQ=
|   256 d0:86:51:60:69:46:b2:e1:39:43:90:97:a6:af:96:93 (ED25519)
|_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHs4NezobK71HOHpkwVK5b5LS0MgCghx1Oj4eld8ONa1
80/tcp  open  http    syn-ack Apache httpd 2.4.41 ((Ubuntu))
|_http-title: Apache2 Ubuntu Default Page: It works
| http-methods: 
|_  Supported Methods: GET HEAD POST OPTIONS
|_http-server-header: Apache/2.4.41 (Ubuntu)
445/tcp open  http    syn-ack Apache httpd 2.4.41 ((Ubuntu))
| http-methods: 
|_  Supported Methods: GET HEAD POST
|_http-server-header: Apache/2.4.41 (Ubuntu)
|_http-title: Apache2 Ubuntu Default Page: It works
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

Host script results:
| p2p-conficker: 
|   Checking for Conficker.C or higher...
|   Check 1 (port 34709/tcp): CLEAN (Couldn't connect)
|   Check 2 (port 20262/tcp): CLEAN (Couldn't connect)
|   Check 3 (port 32770/udp): CLEAN (Failed to receive data)
|   Check 4 (port 10108/udp): CLEAN (Failed to receive data)
|_  0/4 checks are positive: Host is CLEAN or ports are blocked
|_smb2-time: Protocol negotiation failed (SMB2)
|_smb2-security-mode: Couldn't establish a SMBv2 connection.
```

- I scanned port `445` which hosts an apache http server for directories and spotted a direcotry `/management/`.

![image](https://github.com/user-attachments/assets/f3db9e70-2bc5-4283-94d6-a60272e5793b)

- The directory hosts a traffic management system.

![image](https://github.com/user-attachments/assets/d7d5e659-8c04-49a5-bd35-12add382ef1f)

- The system is vulnerable to sql injection to bypass the admin login page and remote code execution through unrestricted file upload.I got a script to exploit it from the github [site](https://github.com/hunkaracar/Online-Traffic-Offense-Management-System-1.0---Remote-Code-Execution-RCE-Unauthenticated-).

![image](https://github.com/user-attachments/assets/b83aab87-ae20-439d-b75e-806578a4cd05)

- Shell access-:

![image](https://github.com/user-attachments/assets/44b8d9be-a31f-4f7e-9b1a-d7b31ba72acb)

### Privesc to user `plot_admin` with crontab

- I read the crontab file and noticed that user `plot_admin` has a crontab in `/var/www/scripts`

![image](https://github.com/user-attachments/assets/433de64e-b195-4580-8e1e-11cbc8fa5e83)

- I was able to delete `backup.sh` and write a code to pop a reverse shell

![image](https://github.com/user-attachments/assets/9301ec37-9428-4851-99a6-ce20a0615431)

- Reverse shell as user `plot_admin`

![image](https://github.com/user-attachments/assets/dbd1220e-5905-4029-9113-1d11728d832f)

### Exploiting `doas` to run `openssl` and gain root

- I read `/etc/doas.conf` and discovered that I can run openssl as user `root`.

![image](https://github.com/user-attachments/assets/0182c704-2a2f-49e2-9ae8-f52b49e92192)

- I was able  to read the `/etc/shadow` file

![image](https://github.com/user-attachments/assets/e6b926e4-63f6-4e6f-8193-a2bc4ac11b5f)

- I escalated privilege by adding an hash to the /etc/passwd file and creating a new user

![image](https://github.com/user-attachments/assets/d8685edc-ccda-4259-9ea1-19f7f8507e2d)

- I wrote the file to `/etc/passwd`.

![image](https://github.com/user-attachments/assets/9c9781f0-016c-47ee-928a-3a2612ab3e0c)

- Root

![image](https://github.com/user-attachments/assets/21c57ab4-cfd9-42d6-aae7-5163cd0c2a85)

---------------------

### THANKS FOR READING...!!!!

---------------------










