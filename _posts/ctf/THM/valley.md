--------------

### CTF-: Valley
### LAB-: THM

--------------

![image](https://github.com/user-attachments/assets/23beca8c-b484-4a08-bb72-6a4a7e450e70)

--------------

- Rustscan's output-:

```bash
❯ rustscan -a 10.10.57.116 -- -Pn -sC -sV
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
Open 10.10.57.116:22
Open 10.10.57.116:80
Open 10.10.57.116:37370
[~] Starting Script(s)
[>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")

Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
[~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-12-31 14:29 WAT
NSE: Loaded 156 scripts for scanning.
NSE: Script Pre-scanning.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 14:30
Completed NSE at 14:30, 0.00s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 14:30
Completed NSE at 14:30, 0.00s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 14:30
Completed NSE at 14:30, 0.00s elapsed
Initiating Parallel DNS resolution of 1 host. at 14:30
Completed Parallel DNS resolution of 1 host. at 14:30, 0.05s elapsed
DNS resolution of 1 IPs took 0.05s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
Initiating Connect Scan at 14:30
Scanning 10.10.57.116 [3 ports]
Discovered open port 37370/tcp on 10.10.57.116
Discovered open port 80/tcp on 10.10.57.116
Discovered open port 22/tcp on 10.10.57.116
Completed Connect Scan at 14:30, 0.24s elapsed (3 total ports)
Initiating Service scan at 14:30
Scanning 3 services on 10.10.57.116
Completed Service scan at 14:30, 6.40s elapsed (3 services on 1 host)
NSE: Script scanning 10.10.57.116.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 14:30
Completed NSE at 14:30, 13.06s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 14:30
Completed NSE at 14:30, 1.56s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 14:30
Completed NSE at 14:30, 0.00s elapsed
Nmap scan report for 10.10.57.116
Host is up, received user-set (0.24s latency).
Scanned at 2024-12-31 14:30:05 WAT for 21s

PORT      STATE SERVICE REASON  VERSION
22/tcp    open  ssh     syn-ack OpenSSH 8.2p1 Ubuntu 4ubuntu0.5 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey: 
|   3072 c2:84:2a:c1:22:5a:10:f1:66:16:dd:a0:f6:04:62:95 (RSA)
| ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCf7Zvn7fOyAWUwEI2aH/k8AyPehxzzuNC1v4AAlhDa4Off4085gRIH/EXpjOoZSBvo8magsCH32JaKMMc59FSK4canP2I0VrXwkEX0F8PjA1TV4qgqXJI0zNVwFrfBORDdlCPNYiqRNFp1vaxTqLOFuHt5r34134yRwczxTsD4Uf9Z6c7Yzr0GV6NL3baGHDeSZ/msTiFKFzLTTKbFkbU4SQYc7jIWjl0ylQ6qtWivBiavEWTwkHHKWGg9WEdFpU2zjeYTrDNnaEfouD67dXznI+FiiTiFf4KC9/1C+msppC0o77nxTGI0352wtBV9KjTU/Aja+zSTMDxoGVvo/BabczvRCTwhXxzVpWNe3YTGeoNESyUGLKA6kUBfFNICrJD2JR7pXYKuZVwpJUUCpy5n6MetnonUo0SoMg/fzqMWw2nCZOpKzVo9OdD8R/ZTnX/iQKGNNvgD7RkbxxFK5OA9TlvfvuRUQQaQP7+UctsaqG2F9gUfWorSdizFwfdKvRU=
|   256 42:9e:2f:f6:3e:5a:db:51:99:62:71:c4:8c:22:3e:bb (ECDSA)
| ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBNIiJc4hdfcu/HtdZN1fyz/hU1SgSas1Lk/ncNc9UkfSDG2SQziJ/5SEj1AQhK0T4NdVeaMSDEunQnrmD1tJ9hg=
|   256 2e:a0:a5:6c:d9:83:e0:01:6c:b9:8a:60:9b:63:86:72 (ED25519)
|_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEZhkboYdSkdR3n1G4sQtN4uO3hy89JxYkizKi6Sd/Ky
80/tcp    open  http    syn-ack Apache httpd 2.4.41 ((Ubuntu))
| http-methods: 
|_  Supported Methods: GET POST OPTIONS HEAD
|_http-server-header: Apache/2.4.41 (Ubuntu)
|_http-title: Site doesn't have a title (text/html).
37370/tcp open  ftp     syn-ack vsftpd 3.0.3
Service Info: OSs: Linux, Unix; CPE: cpe:/o:linux:linux_kernel
```
- Ftp is on port `37370`.FFUF's output

![image](https://github.com/user-attachments/assets/fe13f394-519e-4776-9a0f-ab500b9b79ff)

- I scanned for directories under `/static` and discovered this file `00`.

![image](https://github.com/user-attachments/assets/c6722238-72b9-48b2-ae27-b0dbdd9de8c2)

- The file `00` contains a note from the dev and also points to the existence of an hidden admin page.

![image](https://github.com/user-attachments/assets/76cfbda5-4dc0-4782-a7c2-cb5a09f12cc3)

- After checking the source js code of the page, I discovered a credential which worked for the ftp server.

![image](https://github.com/user-attachments/assets/9f74b00c-a4aa-49cf-b2d8-5190ef4f8db8)

- Ftp Access

![image](https://github.com/user-attachments/assets/d3ad268b-0968-40af-824c-1dec6470a047)

- The ftp server contains 3 files as seen above.
- I examined the `SIEMHTTP2.pcapng` file with wireshark and discovered credentials for a user `valleyDev`.

![image](https://github.com/user-attachments/assets/51edb8d1-77da-4b48-bad6-4993672fe89f)

- SSH access as user `valleyDev`

![image](https://github.com/user-attachments/assets/d7a544d8-d3c5-42d8-9774-5a00d0b4058a)

### Privesc to user `valley`

- I noticed a binary in the directory named `valleyAuthenticator`.

![image](https://github.com/user-attachments/assets/d7c577f5-eef8-42c6-8822-51b539640100)

- I used strings to search for keywords and noticed this hash.

![image](https://github.com/user-attachments/assets/a0a68621-2631-4390-b0e3-f87300ec6d22)

- I cracked it with crackstation and got this password.

![image](https://github.com/user-attachments/assets/deb1fa39-e41a-4939-a228-93e5b9a8d373)

- SSH access as user `valley`.

![image](https://github.com/user-attachments/assets/c19451fb-a6d5-4e44-959e-4014488ead67)

### Root privesc with python library hijacking

- Based on `pspy`'s results,I noticed that user `root` runs a script in the background.

![image](https://github.com/user-attachments/assets/4020427d-1b9b-480d-acc6-79b0ac72e793)

- I checked the file and noticed that I have no write access to the file as user `valley`.

![image](https://github.com/user-attachments/assets/c6be419c-53f6-4ce6-adde-042b71f00e22)

- But it imports module `base64` which we have write access to,we can hijack the library to spawn a root reverse shell.Users in group `valleyAdmin` which user `valley` is a member, can acces and edit the `base64.py` module.

![image](https://github.com/user-attachments/assets/56330754-b8f1-458e-a35d-954d0f668b82)

- I added the reverse shell code as seen below.

![image](https://github.com/user-attachments/assets/97358ad3-579d-4ea2-9390-d77fce69b4fb)

- Root shell-:

![image](https://github.com/user-attachments/assets/18bd035d-5713-42c7-8d4d-b592e78925ae)

--------------

### THANKS FOR READING!!!!

---------------









