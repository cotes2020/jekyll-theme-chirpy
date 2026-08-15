------------------

### CTF: TRYHACKME
### LAB: JACK-OF-ALL-TRADES

------------------

![image](https://github.com/user-attachments/assets/8fb027aa-9dda-4797-8df2-15a7ffdb7b2a)

-----------------

### RECONNAISSANCE

- Rustscan's output

      ❯ rustscan -a 10.10.128.125 -- -sC -sV -Pn
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
      Open 10.10.128.125:22
      Open 10.10.128.125:80
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-09-10 11:42 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 11:42
      Completed NSE at 11:42, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 11:42
      Completed NSE at 11:42, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 11:42
      Completed NSE at 11:42, 0.00s elapsed
      Initiating Parallel DNS resolution of 1 host. at 11:42
      Completed Parallel DNS resolution of 1 host. at 11:42, 0.00s elapsed
      DNS resolution of 1 IPs took 0.00s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 11:42
      Scanning 10.10.128.125 [2 ports]
      Discovered open port 80/tcp on 10.10.128.125
      Discovered open port 22/tcp on 10.10.128.125
      Completed Connect Scan at 11:42, 0.16s elapsed (2 total ports)
      Initiating Service scan at 11:42
      Scanning 2 services on 10.10.128.125
      Completed Service scan at 11:43, 11.54s elapsed (2 services on 1 host)
      NSE: Script scanning 10.10.128.125.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 11:43
      Completed NSE at 11:43, 30.22s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 11:43
      Completed NSE at 11:43, 0.70s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 11:43
      Completed NSE at 11:43, 0.01s elapsed
      Nmap scan report for 10.10.128.125
      Host is up, received user-set (0.16s latency).
      Scanned at 2024-09-10 11:42:57 EDT for 43s
      
      PORT   STATE SERVICE REASON  VERSION
      22/tcp open  http    syn-ack Apache httpd 2.4.10 ((Debian))
      |_http-title: Jack-of-all-trades!
      |_ssh-hostkey: ERROR: Script execution failed (use -d to debug)
      |_http-server-header: Apache/2.4.10 (Debian)
      | http-methods: 
      |_  Supported Methods: GET HEAD POST OPTIONS
      80/tcp open  ssh     syn-ack OpenSSH 6.7p1 Debian 5 (protocol 2.0)
      | ssh-hostkey: 
      |   1024 13:b7:f0:a1:14:e2:d3:25:40:ff:4b:94:60:c5:00:3d (DSA)
      | ssh-dss AAAAB3NzaC1kc3MAAACBANucPy+D67M/cKVTYaHYYpt9bqPviYbWW/4+BFnUOQoNordc9Pc+8CauJqNFiebIqpKYKXhpEAt82m1IjQh8EmWdJYcQnkMFgukM3/mGjngXTbUO8vAbi53Zy8wwOaBlmRK9mvfAYEWPkcjzRmYgSp51TgEtSGWIyAkc1Lx6YVtDAAAAFQCsIgZJlrsYvAtF7Rmho7lIdn0WOwAAAIEApri35SyOophhqX45JcDpVASe3CSs8tPMGoOc0I9ZtTGt5qyb1cl7N3tXsP6mlrw4d4YNo8ct0w6TjsxPcJjGitRQ+SILWHy72XZ5Chde6yewKB5BeBjXrYvRR1rW+Tpia5kyjB4s0mGB7o3FMjX/dT+ISqYvZeVa7mQnBo0f0XMAAACAP89Ag2kmcs0FBt7KCBieH3UB6gF+LdeRVJHio5p4VQ8cTY1NZDyWqudS1TJq1BAToJSz9MqwUwzlILjRjuGQtylpssWSRbHyM0aqmJdORSMOCMUiEwyfk6T8+Vmama/AN7/htZeWBjWVeVEnbYJJQ6kPSCvZodMdOggYXcv32CA=
      |   2048 91:0c:d6:43:d9:40:c3:88:b1:be:35:0b:bc:b9:90:88 (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDbCwl2kyYWpv1DPDF0xQ5szNR1muMph6gJMJFw9VubKkSvHMWfg7CaCNcyo1QR5dg9buIygIGab8e9aigJdjQUY4XeBejwGe+vAA8RtPMoiLclR6g5qAqVQSeZ2FBzMrmkyKIgsSDb8tP+czpzn/Gp1HzDtiYUvleTvO2xEZ3k2Xz8YDvPlkV4zAIPzZSSZ8BABPYsBrePIwMpr/ZjeeiE59DlkUIv8x8M0z9KOls9zaeqFsbWrfMZzFgtPP+KILN6GrGijxgcGq5mDwvr67oHL3T3FtpReE+UZ/CafmzO/2Ls8XstmUiNeMaNBYtc6703/84bpL0uLp/pkILS8eqX
      |   256 a3:fb:09:fb:50:80:71:8f:93:1f:8d:43:97:1e:dc:ab (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBO4p2E6NglzDeP40tJ42LjWaVrOcINmy42cspAv8DSzGD0K+V3El/tyGBxCJlMMR7wbN0968CQl61x0AkkAHLFk=
      |   256 65:21:e7:4e:7c:5a:e7:bc:c6:ff:68:ca:f1:cb:75:e3 (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIC6jYsDJq1mWTDx7D+p3mMbqXhu9OhhW2p1ickLCdZ9E
      Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

- There is a twist in the open ports, `http` is running on port `22` and ssh is running on `80`.
- I checked the source code of the page,discovered a `base64` encoded comment and a `recover.php` page.

![image](https://github.com/user-attachments/assets/4061eb40-0993-4b4f-b20d-7f5ae7cc120c)

- The base64 decoded content indicates a password

![image](https://github.com/user-attachments/assets/be736e0b-4294-42b7-b86e-edbeb1446dc0)

- The source code of `recover.php` contains a base32 string.

![image](https://github.com/user-attachments/assets/0b6a2d44-8bb2-4345-92fc-15aba7637569)

- The base32 decoded string reveals a hex string which also contains a rot13 string that finally reveals the text.

![image](https://github.com/user-attachments/assets/2f43ea9b-5072-4332-b559-dbb3ab20f409)

- The hint redirects us  to a wikipedia page on the dinosaur `stegosaurus`.This hint implies that we should use steganography on the jpg images in the webpage.
- I tried steghide on the `header` jpg file  using the password I got and discovered a file containing the recovery page's credentials.

![image](https://github.com/user-attachments/assets/52985dc3-b5de-467c-805c-eab29efb5912)

- The index page takes in a `cmd` parameter and executes the values as shell commands.

![image](https://github.com/user-attachments/assets/868a9a4b-bd6f-4b70-99de-a7bdc39c5267)

- Shell access

![image](https://github.com/user-attachments/assets/6b51292d-8640-47cc-93eb-9d6359880bc1)

-----------------------

### Pivoting to user `jack`

- The home directory contains a password list that containing Jack's password.

![image](https://github.com/user-attachments/assets/fa32d107-83c4-4404-a5ba-f5fcfa57dc52)

- I cracked Jack's ssh password with hydra

![image](https://github.com/user-attachments/assets/e709d82b-9838-4ccf-be32-6dfffc97bc06)

- SSH access

![image](https://github.com/user-attachments/assets/a6e3d7fc-aeab-4f77-aa74-e63465721419)


------------------------

### Privesc: SUID file read with `strings`

- I ran `find / -perm -u=s -type f 2</dev/null` to check for suid binaries.I spotted the `strings` binary.Strings is used to output the printable characters of a file.We can use it to read files.

![image](https://github.com/user-attachments/assets/0455e567-99bd-4c95-ab65-1579ceb37a4e)

- Now, we can read the root flag.

![image](https://github.com/user-attachments/assets/e028be96-6e24-485a-8a73-9f9bdaed441b)

- For the user.jpg file in Jack's directory, use pytesseract to convert the image to string.

![image](https://github.com/user-attachments/assets/3b290169-d049-42be-98a4-2978fded7a58)

------------------------

### THANKS FOR READING

------------------------










  




