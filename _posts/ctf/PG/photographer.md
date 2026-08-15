------------------

### CTF: Proving Grounds
### LAB: Photographer

------------------

![image](https://github.com/user-attachments/assets/0254c755-4248-46dc-b247-6b566dad1491)

------------------

### Reconnaissance

- Rustscan's network scan output

      ❯ rustscan -a 192.168.215.76 -- -Pn -sC -sV
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
      Open 192.168.215.76:22
      Open 192.168.215.76:80
      Open 192.168.215.76:139
      Open 192.168.215.76:445
      Open 192.168.215.76:8000
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-09-25 05:47 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 05:47
      Completed NSE at 05:47, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 05:47
      Completed NSE at 05:47, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 05:47
      Completed NSE at 05:47, 0.00s elapsed
      Initiating Parallel DNS resolution of 1 host. at 05:47
      Completed Parallel DNS resolution of 1 host. at 05:47, 0.07s elapsed
      DNS resolution of 1 IPs took 0.07s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 05:47
      Scanning 192.168.215.76 [5 ports]
      Discovered open port 80/tcp on 192.168.215.76
      Discovered open port 139/tcp on 192.168.215.76
      Discovered open port 445/tcp on 192.168.215.76
      Discovered open port 22/tcp on 192.168.215.76
      Discovered open port 8000/tcp on 192.168.215.76
      Completed Connect Scan at 05:47, 0.19s elapsed (5 total ports)
      Initiating Service scan at 05:47
      Scanning 5 services on 192.168.215.76
      Completed Service scan at 05:47, 16.93s elapsed (5 services on 1 host)
      NSE: Script scanning 192.168.215.76.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 05:47
      Completed NSE at 05:47, 11.70s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 05:47
      Completed NSE at 05:47, 0.78s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 05:47
      Completed NSE at 05:47, 0.00s elapsed
      Nmap scan report for 192.168.215.76
      Host is up, received user-set (0.19s latency).
      Scanned at 2024-09-25 05:47:10 EDT for 30s
      
      PORT     STATE SERVICE     REASON  VERSION
      22/tcp   open  ssh         syn-ack OpenSSH 7.2p2 Ubuntu 4ubuntu2.10 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   2048 41:4d:aa:18:86:94:8e:88:a7:4c:6b:42:60:76:f1:4f (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCq9GoYsvJTOUcsgHSES9+20Ix4Q8wjm5slMheJ2ME+COokAqxBzXSr458KBmHv3bsTLWAH9FxoXJ6zrzDPmPApcqVifB4aI9l/VYxoeJCj54kKIQlCKkWTZjsAeLBI2Lk2+yJLLFWPTAZ2htwRAwCl9z8YV3xgtqhTa+5BqIm/GInW4PYV0zi9zOMn2g4jNSWvy91FBUboGLwVgNYslGBydNW8Fhz8X/LXHZ1x6ulA76W026VEGOiQfoiIi84IFi9CbP8GIKfQ7BHuDlMqgiN9+w7K0z0oFdtiFhAS/48w89MYn6UOzw7Aaa9eLQi0+zxpW5SpCpw0mC2euzPxow2Z
      |   256 4d:a3:d0:7a:8f:64:ef:82:45:2d:01:13:18:b7:e0:13 (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBMz4UG2gfu7L/Lxcqek1pZf46d8SocbES1A2a/XUYQgTmIqJuCEpLf3ERgVXS+7Lwdi6+F3xkI/lYFCA5MkRUQA=
      |   256 1a:01:7a:4f:cf:95:85:bf:31:a1:4f:15:87:ab:94:e2 (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDL5ZwzA5dpqtWx4ZzjVQ6NMzVUia8/We8txfiAn+mv4
      80/tcp   open  http        syn-ack Apache httpd 2.4.18 ((Ubuntu))
      |_http-server-header: Apache/2.4.18 (Ubuntu)
      |_http-title: Photographer by v1n1v131r4
      | http-methods: 
      |_  Supported Methods: OPTIONS GET HEAD POST
      139/tcp  open  netbios-ssn syn-ack Samba smbd 3.X - 4.X (workgroup: WORKGROUP)
      445/tcp  open  netbios-ssn syn-ack Samba smbd 4.3.11-Ubuntu (workgroup: WORKGROUP)
      8000/tcp open  http        syn-ack Apache httpd 2.4.18 ((Ubuntu))
      |_http-server-header: Apache/2.4.18 (Ubuntu)
      |_http-generator: Koken 0.22.24
      |_http-title: daisa ahomi
      | http-methods: 
      |_  Supported Methods: GET HEAD POST OPTIONS
      Service Info: Host: PHOTOGRAPHER; OS: Linux; CPE: cpe:/o:linux:linux_kernel
      
      Host script results:
      | smb-os-discovery: 
      |   OS: Windows 6.1 (Samba 4.3.11-Ubuntu)
      |   Computer name: photographer
      |   NetBIOS computer name: PHOTOGRAPHER\x00
      |   Domain name: \x00
      |   FQDN: photographer
      |_  System time: 2024-09-25T05:47:14-04:00
      | nbstat: NetBIOS name: PHOTOGRAPHER, NetBIOS user: <unknown>, NetBIOS MAC: <unknown> (unknown)
      | Names:
      |   PHOTOGRAPHER<00>     Flags: <unique><active>
      |   PHOTOGRAPHER<03>     Flags: <unique><active>
      |   PHOTOGRAPHER<20>     Flags: <unique><active>
      |   \x01\x02__MSBROWSE__\x02<01>  Flags: <group><active>
      |   WORKGROUP<00>        Flags: <group><active>
      |   WORKGROUP<1d>        Flags: <unique><active>
      |   WORKGROUP<1e>        Flags: <group><active>
      | Statistics:
      |   00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00
      |   00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00
      |_  00:00:00:00:00:00:00:00:00:00:00:00:00:00
      | smb2-security-mode: 
      |   3:1:1: 
      |_    Message signing enabled but not required
      | p2p-conficker: 
      |   Checking for Conficker.C or higher...
      |   Check 1 (port 50179/tcp): CLEAN (Couldn't connect)
      |   Check 2 (port 10747/tcp): CLEAN (Couldn't connect)
      |   Check 3 (port 19091/udp): CLEAN (Failed to receive data)
      |   Check 4 (port 49309/udp): CLEAN (Failed to receive data)
      |_  0/4 checks are positive: Host is CLEAN or ports are blocked
      | smb-security-mode: 
      |   account_used: guest
      |   authentication_level: user
      |   challenge_response: supported
      |_  message_signing: disabled (dangerous, but default)
      |_clock-skew: mean: 1h19m41s, deviation: 2h18m37s, median: -20s
      | smb2-time: 
      |   date: 2024-09-25T09:47:14
      |_  start_date: N/A

- I enumerated smb shares with smbclient and discovered 3 shares.

![image](https://github.com/user-attachments/assets/7b40db63-1e88-462b-8bef-79ee4a5bebdb)

- I fuzzed for directories with ffuf and discovered an admin page on port 8000 which is hosting an apache web server.

![image](https://github.com/user-attachments/assets/da1d0dc8-fab4-425b-b332-240d60ea6b37)

- I discovered a txt file in share `sambashare`.

![image](https://github.com/user-attachments/assets/bf82d66f-4d37-4550-8223-012b82e9afca)

- Contents of the file indicates the website creator telling `daisa` not to forget her secret.

![image](https://github.com/user-attachments/assets/20da922f-e88d-4dab-b6bb-f5eb208bb049)

- I tried the password `babygirl` and mail `daisa@photographer.com` and was able to access the admin account.

![image](https://github.com/user-attachments/assets/47ceeb7c-ba08-42b2-8f6b-ed703086bdab)

-------------------

### Exploiting KOKEN CMS 0.22.4 Arbitrary file upload

- [Exploit Db](https://www.exploit-db.com/exploits/48706) details a walkthrough on how to exploit koken cms arbitrary file upload.
- I imported a php shell code saved as shell.php.jpg

![image](https://github.com/user-attachments/assets/7f9d1aac-4d21-4fb8-a166-14b43513c59c)

- I intercepted with burpsuite to change the filename to `shell.php` and forwarded the request.

![image](https://github.com/user-attachments/assets/9bde0231-bccb-40fd-aeec-a9259a5a81d0)

- I accessed the file from download file.

![image](https://github.com/user-attachments/assets/da0fc213-699d-48ec-83df-03edfd42a32e)

- RCE achieved

![image](https://github.com/user-attachments/assets/2669db3e-4498-496a-bd30-4254bea445db)

- Initial foothold

![image](https://github.com/user-attachments/assets/011e82be-77c8-449e-a1d9-e279a4b88f5a)

--------------------

### Privesc with SUID PHP binary

- I ran `find / -perm -u=s -type f 2</dev/null` and noticed that `php 7.2` has the suid bit.

![image](https://github.com/user-attachments/assets/8206e6eb-a0f9-4e41-a900-91bee8bfa9e1)

- I used this payload to gain root.

Payload-:```php -r "pcntl_exec('/bin/bash', ['-p']);"```

- Root access....!!!

![image](https://github.com/user-attachments/assets/179a1c55-240f-48bc-9847-e0bf8d4630b2)

### REFERENCE

- [ExploitDB](https://www.exploit-db.com/exploits/48706)












