-------------------

### CTF: Boardlight
### Platform: HTB

-------------------

![image](https://github.com/user-attachments/assets/a13c0195-d4a6-458d-8023-19b8d82594c2)

------------------

### RECONNAISSANCE

------------------

- Rustscan's output

      ❯ rustscan -a 10.10.11.11 -- -sC -sV
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
      Open 10.10.11.11:22
      Open 10.10.11.11:80
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-09-02 18:38 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 18:38
      Completed NSE at 18:38, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 18:38
      Completed NSE at 18:38, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 18:38
      Completed NSE at 18:38, 0.00s elapsed
      Initiating Ping Scan at 18:38
      Scanning 10.10.11.11 [2 ports]
      Completed Ping Scan at 18:38, 0.15s elapsed (1 total hosts)
      Initiating Connect Scan at 18:38
      Scanning boardlight.htb (10.10.11.11) [2 ports]
      Discovered open port 80/tcp on 10.10.11.11
      Discovered open port 22/tcp on 10.10.11.11
      Completed Connect Scan at 18:38, 0.69s elapsed (2 total ports)
      Initiating Service scan at 18:38
      Scanning 2 services on boardlight.htb (10.10.11.11)
      Completed Service scan at 18:38, 6.60s elapsed (2 services on 1 host)
      NSE: Script scanning 10.10.11.11.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 18:38
      Completed NSE at 18:38, 7.45s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 18:38
      Completed NSE at 18:38, 0.76s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 18:38
      Completed NSE at 18:38, 0.00s elapsed
      Nmap scan report for boardlight.htb (10.10.11.11)
      Host is up, received syn-ack (0.27s latency).
      Scanned at 2024-09-02 18:38:08 EDT for 16s
      
      PORT   STATE SERVICE REASON  VERSION
      22/tcp open  ssh     syn-ack OpenSSH 8.2p1 Ubuntu 4ubuntu0.11 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   3072 06:2d:3b:85:10:59:ff:73:66:27:7f:0e:ae:03:ea:f4 (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDH0dV4gtJNo8ixEEBDxhUId6Pc/8iNLX16+zpUCIgmxxl5TivDMLg2JvXorp4F2r8ci44CESUlnMHRSYNtlLttiIZHpTML7ktFHbNexvOAJqE1lIlQlGjWBU1hWq6Y6n1tuUANOd5U+Yc0/h53gKu5nXTQTy1c9CLbQfaYvFjnzrR3NQ6Hw7ih5u3mEjJngP+Sq+dpzUcnFe1BekvBPrxdAJwN6w+MSpGFyQSAkUthrOE4JRnpa6jSsTjXODDjioNkp2NLkKa73Yc2DHk3evNUXfa+P8oWFBk8ZXSHFyeOoNkcqkPCrkevB71NdFtn3Fd/Ar07co0ygw90Vb2q34cu1Jo/1oPV1UFsvcwaKJuxBKozH+VA0F9hyriPKjsvTRCbkFjweLxCib5phagHu6K5KEYC+VmWbCUnWyvYZauJ1/t5xQqqi9UWssRjbE1mI0Krq2Zb97qnONhzcclAPVpvEVdCCcl0rYZjQt6VI1PzHha56JepZCFCNvX3FVxYzEk=
      |   256 59:03:dc:52:87:3a:35:99:34:44:74:33:78:31:35:fb (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBK7G5PgPkbp1awVqM5uOpMJ/xVrNirmwIT21bMG/+jihUY8rOXxSbidRfC9KgvSDC4flMsPZUrWziSuBDJAra5g=
      |   256 ab:13:38:e4:3e:e0:24:b4:69:38:a9:63:82:38:dd:f4 (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILHj/lr3X40pR3k9+uYJk4oSjdULCK0DlOxbiL66ZRWg
      80/tcp open  http    syn-ack Apache httpd 2.4.41 ((Ubuntu))
      |_http-server-header: Apache/2.4.41 (Ubuntu)
      | http-methods: 
      |_  Supported Methods: GET HEAD POST OPTIONS
      |_http-title: Site doesn't have a title (text/html; charset=UTF-8).
      Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel
      
      NSE: Script Post-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 18:38
      Completed NSE at 18:38, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 18:38
      Completed NSE at 18:38, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 18:38
      Completed NSE at 18:38, 0.00s elapsed
      Read data files from: /usr/bin/../share/nmap
      Service detection performed. Please report any incorrect results at https://nmap.org/submit/ .
      Nmap done: 1 IP address (1 host up) scanned in 17.19 seconds

- Subdomain enumeration spots subdomain `crm`, don't forget to add to `/etc/hosts`.

  ![image](https://github.com/user-attachments/assets/70d829a4-900b-41ef-aa74-7f4e93d06218)

- Subdomain `crm` presents this cms `Dolibarr.17.01`

  ![image](https://github.com/user-attachments/assets/ee35bce0-5e91-4344-b0c0-184b8c494a7f)

- This version of dolibarr is vulnerable to `php code injection` as explained [here](https://github.com/nikn0laty/Exploit-for-Dolibarr-17.0.0-CVE-2023-30253).
I was able to get an exploit from the same site.The credentials `admin:admin` will grant us login access.

![image](https://github.com/user-attachments/assets/6d67faac-c5bc-49cc-9e6c-2ff655780a62)

- Shell access as `www-data`

![image](https://github.com/user-attachments/assets/9eb20c26-83b2-41fe-86af-d41d312a2646)

----------------------

### User Larissa

- I was able to get user `Larissa'spassword` from `/var/www/html/crm.board.htb/htdocs/conf/conf.php`

![image](https://github.com/user-attachments/assets/dff6d0f3-aac0-48c3-8f6d-8b88633cfe13)

- SSH access

![image](https://github.com/user-attachments/assets/3bf8adef-d358-4b31-bdcc-0a3df3e42c48)

-------------------------

### Privesc with suid binary

- I ran `linpeas.sh` and discovered a suid binary `enlightment` and disovered a local privilege exploit for it.

  ![image](https://github.com/user-attachments/assets/94956d3b-0469-49c7-8dee-dcf060cc3cb8)

- The Enlightenment Version: 0.25.3 is vulnerable to local privilege escalation.Enlightenment_sys in Enlightenment before 0.25.3 allows local users to gain privileges because it is setuid root.
I found an exploit for it on this [github repo](https://github.com/MaherAzzouzi/CVE-2022-37706-LPE-exploit).

![image](https://github.com/user-attachments/assets/22bb4863-fd83-45f3-8a43-5d306a12f059)

- Root

![image](https://github.com/user-attachments/assets/b0bcdd8b-e551-48ab-9e22-f17718a7903d)

-----------------

### THANKS FOR READING!!!

-----------------

### REFERENCES:

- [Dolibarr 17.00](https://github.com/nikn0laty/Exploit-for-Dolibarr-17.0.0-CVE-2023-30253)
- [Enlightment_SYS](https://github.com/MaherAzzouzi/CVE-2022-37706-LPE-exploit)

-----------------


  




