-----------------

### CTF: HTB
### LAB: Editorial

------------------

![image](https://github.com/user-attachments/assets/bbe2080b-72ec-4d90-a92b-031b32cbad62)

------------------

### Reconnaissance

- Rustscan's output

      ❯ rustscan -a editorial.htb -- -sC -sV -Pn
      .----. .-. .-. .----..---.  .----. .---.   .--.  .-. .-.
      | {}  }| { } |{ {__ {_   _}{ {__  /  ___} / {} \ |  `| |
      | .-. \| {_} |.-._} } | |  .-._} }\     }/  /\  \| |\  |
      `-' `-'`-----'`----'  `-'  `----'  `---' `-'  `-'`-' `-'
      The Modern Day Port Scanner.
      ________________________________________
      : https://discord.gg/GFrQsGy           :
      : https://github.com/RustScan/RustScan :
       --------------------------------------
      Please contribute more quotes to our GitHub https://github.com/rustscan/rustscan
      
      [~] The config file is expected to be at "/home/sensei/.rustscan.toml"
      [!] File limit is lower than default batch size. Consider upping with --ulimit. May cause harm to sensitive servers
      [!] Your file limit is very small, which negatively impacts RustScan's speed. Use the Docker image, or up the Ulimit with '--ulimit 5000'. 
      Open 10.10.11.20:22
      Open 10.10.11.20:80
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-09-13 18:06 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 18:07
      Completed NSE at 18:07, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 18:07
      Completed NSE at 18:07, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 18:07
      Completed NSE at 18:07, 0.00s elapsed
      Initiating Connect Scan at 18:07
      Scanning editorial.htb (10.10.11.20) [2 ports]
      Discovered open port 80/tcp on 10.10.11.20
      Discovered open port 22/tcp on 10.10.11.20
      Completed Connect Scan at 18:07, 0.30s elapsed (2 total ports)
      Initiating Service scan at 18:07
      Scanning 2 services on editorial.htb (10.10.11.20)
      Completed Service scan at 18:07, 6.52s elapsed (2 services on 1 host)
      NSE: Script scanning 10.10.11.20.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 18:07
      Completed NSE at 18:07, 15.92s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 18:07
      Completed NSE at 18:07, 3.28s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 18:07
      Completed NSE at 18:07, 0.00s elapsed
      Nmap scan report for editorial.htb (10.10.11.20)
      Host is up, received user-set (0.30s latency).
      Scanned at 2024-09-13 18:07:08 EDT for 26s
      
      PORT   STATE SERVICE REASON  VERSION
      22/tcp open  ssh     syn-ack OpenSSH 8.9p1 Ubuntu 3ubuntu0.7 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   256 0d:ed:b2:9c:e2:53:fb:d4:c8:c1:19:6e:75:80:d8:64 (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBMApl7gtas1JLYVJ1BwP3Kpc6oXk6sp2JyCHM37ULGN+DRZ4kw2BBqO/yozkui+j1Yma1wnYsxv0oVYhjGeJavM=
      |   256 0f:b9:a7:51:0e:00:d5:7b:5b:7c:5f:bf:2b:ed:53:a0 (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMXtxiT4ZZTGZX4222Zer7f/kAWwdCWM/rGzRrGVZhYx
      80/tcp open  http    syn-ack nginx 1.18.0 (Ubuntu)
      |_http-server-header: nginx/1.18.0 (Ubuntu)
      |_http-title: Editorial Tiempo Arriba
      | http-methods: 
      |_  Supported Methods: OPTIONS HEAD GET
      Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel
  
- FFUF's output

![image](https://github.com/user-attachments/assets/9419995b-d652-4a7d-8112-077ca8cec274)

- The `upload` route allows publishers to upload books

![image](https://github.com/user-attachments/assets/a670f5ae-020d-4e36-88cf-ac4d935fcc31)

- I noticed a field accept url, I decided to test it for ssrf

![image](https://github.com/user-attachments/assets/123f56ec-2da0-465b-bb93-338952ef50d8)

- I set up a netcat to see if the server will make a request to our url and it did.

![image](https://github.com/user-attachments/assets/e62e3c9e-a761-4229-b5c5-3f89f8903a35)

- I fuzzed with burp intruder and got a different result when a request was made to port 5000.The usual result was an image stored in `/static/image` directory

![image](https://github.com/user-attachments/assets/d8f1d581-4bab-45a7-a45f-ef817f83e4a2)

- The file exposed api endpoints

![image](https://github.com/user-attachments/assets/803878e5-9b4f-4a8a-bb9b-a491295d3c72)

- I made a request to endpoint `api/latest/metadata/messages/authors` and got a file containing ssh creds for a user `dev`.

![image](https://github.com/user-attachments/assets/2b68614f-d35d-40c5-b8d7-4a4a0092a1cd)

- SSH access as user `dev`

![image](https://github.com/user-attachments/assets/5c0080f5-968e-46c2-be95-a304060e5864)

--------------------

### Pivoting to user `prod`

- I discovered a git file in dev's directory `app/`

![image](https://github.com/user-attachments/assets/ee4dab44-cc65-44f9-9379-4b5182d1f064)

- I transferred  the directories to my machine and extracted the necessary objects with this [tool](https://github.com/internetwache/GitTools)

![image](https://github.com/user-attachments/assets/5dc9a265-762c-449c-b713-f648bfabf277)

- I switched to the directory the contents where extracted,grepped for passwords and got user `prod` password.

![image](https://github.com/user-attachments/assets/2de0cbc0-bbb6-4510-8961-3af0ed533d93)

- User `prod`

![image](https://github.com/user-attachments/assets/1d76aca2-4e2f-487d-974b-c50d1f93cfac)


------------------------

### Privesc with gitpython `clone_from()` function

- I ran `sudo -l` and discovered I can run a python script as root and it has a wildcard character that allows us to add other characters to it.

![image](https://github.com/user-attachments/assets/39b9a030-bacc-42c9-91d4-0e769b33b67d)

- I decided to dig into the script and  a spotted `clone_from` and decided to read more on it. i stumbled on a report by [snyk](https://security.snyk.io/vuln/SNYK-PYTHON-GITPYTHON-3113858)

![image](https://github.com/user-attachments/assets/4de0429e-4f2d-4705-9c10-107ed72a8d6a)

- According to snyk,this function is exploitable if the `-c protocol.ext.allow=always` is set to `always` which allows use of the `ext` transport protocol.The payload below is an example of a crafted url to gain RCE.

  Payload-:```ext::bash -c [path to binary]% ls```

![image](https://github.com/user-attachments/assets/26257844-c1a0-4e26-b6d6-d0a9d7601cac)

- Creating a python script that execute a shell

![image](https://github.com/user-attachments/assets/2d00ab97-c7d0-489d-a87e-73cb1d219397)

- Root shell

![image](https://github.com/user-attachments/assets/8724b5c5-2504-432d-9f0f-f32edca94cf3)

- Root..!!!!

![image](https://github.com/user-attachments/assets/de210a1d-9241-424f-8fe2-e2b46dc62fb7)

---------------------

### THANKS FOR READING

---------------------

### REFERENCES:

- [GitTools](https://github.com/internetwache/GitTools)
- [GitpythonRCE](https://security.snyk.io/vuln/SNYK-PYTHON-GITPYTHON-3113858)

----------------------

















