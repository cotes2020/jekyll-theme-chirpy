-------------

### Lab: Planning
### CTF: Hackthebox

-------------

![image](https://github.com/user-attachments/assets/f242393d-de6a-42d8-bfe2-9e9f8b0369d0)

--------------

### Reconnaissance

--------------

- The ctf creator provided creds to access the admin account.

![image](https://github.com/user-attachments/assets/1e5e09dc-9ac6-4f45-a733-fec96fda4756)

- Rustscan's output-:


```bash
❯ rustscan -a planning.htb -- -Pn -sC -sV
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
Open 10.10.11.68:22
Open 10.10.11.68:80
[~] Starting Script(s)
[>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")

Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
[~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-05-12 13:36 WAT
NSE: Loaded 156 scripts for scanning.
NSE: Script Pre-scanning.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 13:36
Completed NSE at 13:36, 0.00s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 13:36
Completed NSE at 13:36, 0.00s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 13:36
Completed NSE at 13:36, 0.00s elapsed
Initiating Connect Scan at 13:36
Scanning planning.htb (10.10.11.68) [2 ports]
Discovered open port 22/tcp on 10.10.11.68
Discovered open port 80/tcp on 10.10.11.68
Completed Connect Scan at 13:36, 0.23s elapsed (2 total ports)
Initiating Service scan at 13:36
Scanning 2 services on planning.htb (10.10.11.68)
Completed Service scan at 13:37, 6.55s elapsed (2 services on 1 host)
NSE: Script scanning 10.10.11.68.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 13:37
Completed NSE at 13:37, 6.44s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 13:37
Completed NSE at 13:37, 1.04s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 13:37
Completed NSE at 13:37, 0.00s elapsed
Nmap scan report for planning.htb (10.10.11.68)
Host is up, received user-set (0.23s latency).
Scanned at 2025-05-12 13:36:53 WAT for 15s

PORT   STATE SERVICE REASON  VERSION
22/tcp open  ssh     syn-ack OpenSSH 9.6p1 Ubuntu 3ubuntu13.11 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey: 
|   256 62:ff:f6:d4:57:88:05:ad:f4:d3:de:5b:9b:f8:50:f1 (ECDSA)
| ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBMv/TbRhuPIAz+BOq4x+61TDVtlp0CfnTA2y6mk03/g2CffQmx8EL/uYKHNYNdnkO7MO3DXpUbQGq1k2H6mP6Fg=
|   256 4c:ce:7d:5c:fb:2d:a0:9e:9f:bd:f5:5c:5e:61:50:8a (ED25519)
|_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKpJkWOBF3N5HVlTJhPDWhOeW+p9G7f2E9JnYIhKs6R0
80/tcp open  http    syn-ack nginx 1.24.0 (Ubuntu)
| http-methods: 
|_  Supported Methods: GET HEAD POST
|_http-title: Edukate - Online Education Website
|_http-server-header: nginx/1.24.0 (Ubuntu)
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel
```

- I didn't get any useful information from the main site.I decided to fuzz for subdomains and discovered the `grafana` sudomain.

![image](https://github.com/user-attachments/assets/17bd36f8-2bf0-45b3-9832-eb541f53d303)

- I accessed the admin's account using the credentials provided.

![image](https://github.com/user-attachments/assets/27851f88-474a-4446-af7c-29ed8ac15b2c)

- I discovered that the Grafana version is vulnerable to command injection.I checked for the version with endpoint `/api/health`.

![image](https://github.com/user-attachments/assets/0af0b815-9385-4da2-a5ed-78c2683dd14f)

- I got an exploit from this github [repo](https://github.com/nollium/CVE-2024-9264)

![image](https://github.com/user-attachments/assets/9170c45a-9139-4ebd-8085-c0452be8aa67)

- Command injection achieved-:

![image](https://github.com/user-attachments/assets/e2b78a3f-c11a-4fc7-bf4d-18f32069af26)

- After spawning a revshell,I discovered that it is a docker container.I ran `env` to check for environmental variables and discovered creds for user `enzo` on the main server.

![image](https://github.com/user-attachments/assets/b5e8a529-a8e7-487a-be93-8d38c22dcd7c)

- User Enzo accessed via ssh-:

![image](https://github.com/user-attachments/assets/8099b2ab-8442-4a96-8d84-039cfcd4bdf9)

---------------

### Privesc with node module `crontab-ui`

--------------

- I noticed a service runnning on port 8000.

![image](https://github.com/user-attachments/assets/c326ee3e-721a-42bf-aa91-12b2cf91f40c)

- I noticed this directory `crontab-ui` and I decided to look into it.

![image](https://github.com/user-attachments/assets/a5f51032-ba89-463f-bbc4-d7e3f49c2429)

- [Crontab-ui](https://github.com/alseambusher/crontab-ui) is a nodejs web application that allows you manage your crontabs.

![image](https://github.com/user-attachments/assets/0bc0ba95-c24f-47de-a307-f23251b2f3b1)

- I portforwarded the port with ssh to access it on my local machine.

![image](https://github.com/user-attachments/assets/ee29ad8a-cb8f-4ed6-8db7-dffcc83c279d)

- When I tried to access it, I met an obstacle "a nginx login prompt".After checking the docs,I discovered that crontab-ui can be used with nginx as an intermediary.

![image](https://github.com/user-attachments/assets/bec82aab-ab9c-446b-8c32-d6c7d89e2aa3)

- I checked the crontab-ui's installation directory for creds and discovered some db files.

![image](https://github.com/user-attachments/assets/d7ff6b39-7e56-40b3-97e8-0a6fc607f045)

- I saw a password in one of the files.

![image](https://github.com/user-attachments/assets/e8d0bc2d-371e-4ca5-8078-eca924138411)

- The `root_grafana:password` combo didn't work.Then, I tried `root:password` and the combo worked.

![image](https://github.com/user-attachments/assets/04d20c84-7ab5-4b32-b944-3875c6a64894)

- To escalate privileges,create a crontab that spawns a reverse shell.

![image](https://github.com/user-attachments/assets/4772b7a4-1800-4ba6-8080-13a04b3bb773)

- Run it-:

![image](https://github.com/user-attachments/assets/a3452f6b-2930-4e83-9705-936031d360dd)

- Reverse shell and roottt!!!-:

![image](https://github.com/user-attachments/assets/56669d79-5092-444e-b4ee-e2197759de14)

--------------------

### THANKS FOR READING

--------------------

### REFERENCES

--------------------

- [Grafana's exploit](https://github.com/nollium/CVE-2024-9264)
- [Crontab-ui](https://github.com/alseambusher/crontab-ui)

---------------------



