----------------------

### PLATFORM: PG
### LAB: POTATO

-----------------------

### RECONNAISSANCE

- Rustscan's output

      ❯ cat rustscan.txt
      
      ❯ rustscan -a 192.168.212.101 -- -sC -sV
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
      Open 192.168.212.101:22
      Open 192.168.212.101:80
      Open 192.168.212.101:2112
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-09-04 06:23 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 06:23
      Completed NSE at 06:23, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 06:23
      Completed NSE at 06:23, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 06:23
      Completed NSE at 06:23, 0.00s elapsed
      Initiating Ping Scan at 06:23
      Scanning 192.168.212.101 [2 ports]
      Completed Ping Scan at 06:23, 0.18s elapsed (1 total hosts)
      Initiating Parallel DNS resolution of 1 host. at 06:23
      Completed Parallel DNS resolution of 1 host. at 06:23, 0.00s elapsed
      DNS resolution of 1 IPs took 0.00s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 06:23
      Scanning 192.168.212.101 [3 ports]
      Discovered open port 22/tcp on 192.168.212.101
      Discovered open port 80/tcp on 192.168.212.101
      Discovered open port 2112/tcp on 192.168.212.101
      Completed Connect Scan at 06:23, 0.15s elapsed (3 total ports)
      Initiating Service scan at 06:23
      Scanning 3 services on 192.168.212.101
      Completed Service scan at 06:23, 11.74s elapsed (3 services on 1 host)
      NSE: Script scanning 192.168.212.101.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 06:23
      NSE: [ftp-bounce 192.168.212.101:2112] PORT response: 500 Illegal PORT command
      Completed NSE at 06:23, 6.72s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 06:23
      Completed NSE at 06:23, 1.47s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 06:23
      Completed NSE at 06:23, 0.00s elapsed
      Nmap scan report for 192.168.212.101
      Host is up, received syn-ack (0.17s latency).
      Scanned at 2024-09-04 06:23:22 EDT for 20s
      
      PORT     STATE SERVICE REASON  VERSION
      22/tcp   open  ssh     syn-ack OpenSSH 8.2p1 Ubuntu 4ubuntu0.1 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   3072 ef:24:0e:ab:d2:b3:16:b4:4b:2e:27:c0:5f:48:79:8b (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDamdAqH2ZyWoYj0tstPK0vbVKI+9OCgtkGDoynffxqV2kE4ceZn77FBuMGFKLU50Uv5RMUTFTX4hm1ijh77KMGG1CmAk2YWvEDhxbCBPCohp+xXMBXHBYoMbEVl/loKL2UW6USnKorOgwxUdoMAwDxIrohGHQ5WNUADRaqt1eHuHxuJ8Bgi8yzqP/26ePQTLCfwAZMq+SYPJedZBmfJJ3Brhb/CGgzgRU8BpJGI8IfBL5791JTn2niEgoMAZ1vdfnSx0m49uk8npd0h5hPQ+ucyMh+Q35lJ1zDq94E24mkgawDhEgmLtb23JDNdY4rv/7mAAHYA5AsRSDDFgmbXEVcC7N1c3cyrwVH/w+zF5SKOqQ8hOF7LRCqv0YQZ05wyiBu2OzbeAvhhiKJteICMuitQAuF6zU/dwjX7oEAxbZ2GsQ66kU3/JnL4clTDATbT01REKJzH9nHpO5sZdebfLJdVfx38qDrlS+risx1QngpnRvWTmJ7XBXt8UrfXGenR3U=
      |   256 f2:d8:35:3f:49:59:85:85:07:e6:a2:0e:65:7a:8c:4b (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBNoh1z4mRbfROqXjtv9CG7ZYGiwN29OQQCVXMLce4ejLzy+0Bvo7tYSb5PKVqgO5jd1JaB3LLGWreXo6ZY3Z8T8=
      |   256 0b:23:89:c3:c0:26:d5:64:5e:93:b7:ba:f5:14:7f:3e (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDXv++bn0YEgaoSEmMm3RzCzm6pyUJJSsSW9FMBqvZQ3
      80/tcp   open  http    syn-ack Apache httpd 2.4.41 ((Ubuntu))
      |_http-server-header: Apache/2.4.41 (Ubuntu)
      | http-methods: 
      |_  Supported Methods: GET HEAD POST OPTIONS
      |_http-title: Potato company
      2112/tcp open  ftp     syn-ack ProFTPD
      | ftp-anon: Anonymous FTP login allowed (FTP code 230)
      | -rw-r--r--   1 ftp      ftp           901 Aug  2  2020 index.php.bak
      |_-rw-r--r--   1 ftp      ftp            54 Aug  2  2020 welcome.msg
      Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

- Three active ports: `ftp`,`http`,`ssh`

- FFUF's ouput

![image](https://github.com/user-attachments/assets/d29d6444-a15b-4e7b-8bc8-63ff501acd62)

- We can login into ftp with anonymous access and ftp contains two files `index.php.bak` and a welcome message.

![image](https://github.com/user-attachments/assets/738d26b0-b2ee-4399-9017-c20956f34e2d)

----------------------

### Source Code

- The index.php contains the code for the admin page.The vulnerable point is the `strcmp()` function in php.If you set,username and password
to an empty array.It will produce the value `0` which will grant login because `strcmp()` reads en empty array as `NULL` which is equals to `0`.

      if (strcmp($_POST['username'], "admin") == 0  && strcmp($_POST['password'], $pass) == 0) {
          echo "Welcome! </br> Go to the <a href=\"dashboard.php\">dashboard</a>";
          setcookie('pass', $pass, time() + 365*24*3600);
        }else{
          echo "<p>Bad login/password! </br> Return to the <a href=\"index.php\">login page</a> <p>";
        }

- I bypassed it by setting username and password to empty arrays.

![image](https://github.com/user-attachments/assets/8b667ee2-21c7-449d-90bb-e565283d849e)

- Admin Dashboard's access

![image](https://github.com/user-attachments/assets/042c8f15-4c18-492c-b06d-1a8100cfdedc)

- After checking the pages, I noticed that most of pages execute shell commands and retun the output. I intercepted the logs page with burpsuite and discovered that it
read files.

![image](https://github.com/user-attachments/assets/f6111d12-735b-4ac5-8a79-7289619a427e)

- At first, I thought it was lfi, then after triggering an error, I noticed it uses cat to read files and was able to execute commands
after closing the `cat` statement with `;`.

![image](https://github.com/user-attachments/assets/72f30ab5-d3f9-4715-a740-c68a1c4bdb1d)

- Shell access as `www-data`

![image](https://github.com/user-attachments/assets/c7c33372-068d-4f40-8a95-d4b1234ab5c0)

-------------------------

### USER `webadmin`

- User `webadmin`'s password hash was exposed in file `/etc/passwd`.

![image](https://github.com/user-attachments/assets/3d4dc7a8-bbea-4715-85de-90e99008b6f6)

- I cracked with john and got his password

![image](https://github.com/user-attachments/assets/c8738ce0-62d4-449b-a45d-861517e7a428)

- User webadmin

![image](https://github.com/user-attachments/assets/c8aa58da-efce-4f05-9d02-d59efb2132ab)

---------------------

### PRIVESC with nice

- Running `sudo -l` shows that user `webadmin` can run binary `nice` on files in directory `/notes/*` with a wildcard.Nice is used to execute scripts and binaries,the wildcard is the main
vulnerability,the `*` allows for other characters to be added.

![image](https://github.com/user-attachments/assets/f6e9fdc5-7121-4e4b-9453-023365431f19)

- To gain root, I added `../bin/bash` to go up one directory and execute a bash root shell.

![image](https://github.com/user-attachments/assets/7e61a9cf-2a28-4d42-87a6-f52c11f299ae)

- Root access.....

![image](https://github.com/user-attachments/assets/9f98a3b9-aef8-4c58-aa01-a88516e3281b)

----------------------

### THANKS FOR READING......!!!!

----------------------

### REFERENCES:

- [Strcmp() bypass](https://www.doyler.net/security-not-included/bypassing-php-strcmp-abctf2016)

----------------------












  

