------------------

### CTF: TRYHACKME
### Lab: ALL in One

-------------------

![image](https://github.com/user-attachments/assets/45db70a1-b486-4861-a1da-898fd98acb54)


-------------------

### RECONNAISSANCE

- Rustscan's output

      ❯ rustscan -a 10.10.185.238 -- -sC -sV -Pn
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
      Open 10.10.185.238:21
      Open 10.10.185.238:22
      Open 10.10.185.238:80
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-09-11 15:31 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 15:31
      Completed NSE at 15:31, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 15:31
      Completed NSE at 15:31, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 15:31
      Completed NSE at 15:31, 0.00s elapsed
      Initiating Parallel DNS resolution of 1 host. at 15:31
      Completed Parallel DNS resolution of 1 host. at 15:31, 0.04s elapsed
      DNS resolution of 1 IPs took 0.04s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 15:31
      Scanning 10.10.185.238 [3 ports]
      Discovered open port 22/tcp on 10.10.185.238
      Discovered open port 21/tcp on 10.10.185.238
      Discovered open port 80/tcp on 10.10.185.238
      Completed Connect Scan at 15:31, 0.17s elapsed (3 total ports)
      Initiating Service scan at 15:31
      Scanning 3 services on 10.10.185.238
      Completed Service scan at 15:31, 6.41s elapsed (3 services on 1 host)
      NSE: Script scanning 10.10.185.238.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 15:31
      NSE: [ftp-bounce 10.10.185.238:21] PORT response: 500 Illegal PORT command.
      Completed NSE at 15:32, 15.85s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 15:32
      Completed NSE at 15:32, 1.24s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 15:32
      Completed NSE at 15:32, 0.01s elapsed
      Nmap scan report for 10.10.185.238
      Host is up, received user-set (0.17s latency).
      Scanned at 2024-09-11 15:31:53 EDT for 24s
      
      PORT   STATE SERVICE REASON  VERSION
      21/tcp open  ftp     syn-ack vsftpd 3.0.3
      | ftp-syst: 
      |   STAT: 
      | FTP server status:
      |      Connected to ::ffff:10.8.158.229
      |      Logged in as ftp
      |      TYPE: ASCII
      |      No session bandwidth limit
      |      Session timeout in seconds is 300
      |      Control connection is plain text
      |      Data connections will be plain text
      |      At session startup, client count was 3
      |      vsFTPd 3.0.3 - secure, fast, stable
      |_End of status
      |_ftp-anon: Anonymous FTP login allowed (FTP code 230)
      22/tcp open  ssh     syn-ack OpenSSH 7.6p1 Ubuntu 4ubuntu0.3 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   2048 e2:5c:33:22:76:5c:93:66:cd:96:9c:16:6a:b3:17:a4 (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDLcG2O5LS7paG07xeOB/4E66h0/DIMR/keWMhbTxlA2cfzaDhYknqxCDdYBc9V3+K7iwduXT9jTFTX0C3NIKsVVYcsLxz6eFX3kUyZjnzxxaURPekEQ0BejITQuJRUz9hghT8IjAnQSTPeA+qBIB7AB+bCD39dgyta5laQcrlo0vebY70Y7FMODJlx4YGgnLce6j+PQjE8dz4oiDmrmBd/BBa9FxLj1bGobjB4CX323sEaXLj9XWkSKbc/49zGX7rhLWcUcy23gHwEHVfPdjkCGPr6oiYj5u6OamBuV/A6hFamq27+hQNh8GgiXSgdgGn/8IZFHZQrnh14WmO8xXW5
      |   256 1b:6a:36:e1:8e:b4:96:5e:c6:ef:0d:91:37:58:59:b6 (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBF1Ww9ui4NQDHA5l+lumRpLsAXHYNk4lkghej9obWBlOwnV+tIDw4mgmuO1C3U/WXRgn0GrESAnMpi1DSxy8t1k=
      |   256 fb:fa:db:ea:4e:ed:20:2b:91:18:9d:58:a0:6a:50:ec (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAOG6ExdDNH+xAyzd4w1G4E9sCfiiooQhmebQX6nIcH/
      80/tcp open  http    syn-ack Apache httpd 2.4.29 ((Ubuntu))
      |_http-title: Apache2 Ubuntu Default Page: It works
      |_http-server-header: Apache/2.4.29 (Ubuntu)
      | http-methods: 
      |_  Supported Methods: HEAD GET POST OPTIONS
      Service Info: OSs: Unix, Linux; CPE: cpe:/o:linux:linux_kernel

- Ffuf discovered a wordpress directory

![image](https://github.com/user-attachments/assets/5c52829b-4ec3-4a55-b624-429778cba850)

- Wpscan result shows two plugins `mail-masta` and `reflex-gallery`.

      ❯ wpscan --url http://10.10.185.238/wordpress
      _______________________________________________________________
               __          _______   _____
               \ \        / /  __ \ / ____|
                \ \  /\  / /| |__) | (___   ___  __ _ _ __ ®
                 \ \/  \/ / |  ___/ \___ \ / __|/ _` | '_ \
                  \  /\  /  | |     ____) | (__| (_| | | | |
                   \/  \/   |_|    |_____/ \___|\__,_|_| |_|
      
               WordPress Security Scanner by the WPScan Team
                               Version 3.8.25
             Sponsored by Automattic - https://automattic.com/
             @_WPScan_, @ethicalhack3r, @erwan_lr, @firefart
      _______________________________________________________________
      
      [+] URL: http://10.10.185.238/wordpress/ [10.10.185.238]
      [+] Started: Wed Sep 11 16:42:35 2024
      
      Interesting Finding(s):
      
      [+] Headers
       | Interesting Entry: Server: Apache/2.4.29 (Ubuntu)
       | Found By: Headers (Passive Detection)
       | Confidence: 100%
      
      [+] XML-RPC seems to be enabled: http://10.10.185.238/wordpress/xmlrpc.php
       | Found By: Direct Access (Aggressive Detection)
       | Confidence: 100%
       | References:
       |  - http://codex.wordpress.org/XML-RPC_Pingback_API
       |  - https://www.rapid7.com/db/modules/auxiliary/scanner/http/wordpress_ghost_scanner/
       |  - https://www.rapid7.com/db/modules/auxiliary/dos/http/wordpress_xmlrpc_dos/
       |  - https://www.rapid7.com/db/modules/auxiliary/scanner/http/wordpress_xmlrpc_login/
       |  - https://www.rapid7.com/db/modules/auxiliary/scanner/http/wordpress_pingback_access/
      
      [+] WordPress readme found: http://10.10.185.238/wordpress/readme.html
       | Found By: Direct Access (Aggressive Detection)
       | Confidence: 100%
      
      [+] Upload directory has listing enabled: http://10.10.185.238/wordpress/wp-content/uploads/
       | Found By: Direct Access (Aggressive Detection)
       | Confidence: 100%
      
      [+] The external WP-Cron seems to be enabled: http://10.10.185.238/wordpress/wp-cron.php
       | Found By: Direct Access (Aggressive Detection)
       | Confidence: 60%
       | References:
       |  - https://www.iplocation.net/defend-wordpress-from-ddos
       |  - https://github.com/wpscanteam/wpscan/issues/1299
      
      [+] WordPress version 5.5.1 identified (Insecure, released on 2020-09-01).
       | Found By: Rss Generator (Passive Detection)
       |  - http://10.10.185.238/wordpress/index.php/feed/, <generator>https://wordpress.org/?v=5.5.1</generator>
       |  - http://10.10.185.238/wordpress/index.php/comments/feed/, <generator>https://wordpress.org/?v=5.5.1</generator>
      
      [+] WordPress theme in use: twentytwenty
       | Location: http://10.10.185.238/wordpress/wp-content/themes/twentytwenty/
       | Last Updated: 2024-07-16T00:00:00.000Z
       | Readme: http://10.10.185.238/wordpress/wp-content/themes/twentytwenty/readme.txt
       | [!] The version is out of date, the latest version is 2.7
       | Style URL: http://10.10.185.238/wordpress/wp-content/themes/twentytwenty/style.css?ver=1.5
       | Style Name: Twenty Twenty
       | Style URI: https://wordpress.org/themes/twentytwenty/
       | Description: Our default theme for 2020 is designed to take full advantage of the flexibility of the block editor...
       | Author: the WordPress team
       | Author URI: https://wordpress.org/
       |
       | Found By: Css Style In Homepage (Passive Detection)
       |
       | Version: 1.5 (80% confidence)
       | Found By: Style (Passive Detection)
       |  - http://10.10.185.238/wordpress/wp-content/themes/twentytwenty/style.css?ver=1.5, Match: 'Version: 1.5'
      
      [+] Enumerating All Plugins (via Passive Methods)
      [+] Checking Plugin Versions (via Passive and Aggressive Methods)
      
      [i] Plugin(s) Identified:
      
      [+] mail-masta
       | Location: http://10.10.185.238/wordpress/wp-content/plugins/mail-masta/
       | Latest Version: 1.0 (up to date)
       | Last Updated: 2014-09-19T07:52:00.000Z
       |
       | Found By: Urls In Homepage (Passive Detection)
       |
       | Version: 1.0 (80% confidence)
       | Found By: Readme - Stable Tag (Aggressive Detection)
       |  - http://10.10.185.238/wordpress/wp-content/plugins/mail-masta/readme.txt
      
      [+] reflex-gallery
       | Location: http://10.10.185.238/wordpress/wp-content/plugins/reflex-gallery/
       | Latest Version: 3.1.7 (up to date)
       | Last Updated: 2021-03-10T02:38:00.000Z
       |
       | Found By: Urls In Homepage (Passive Detection)
       |
       | Version: 3.1.7 (80% confidence)
       | Found By: Readme - Stable Tag (Aggressive Detection)
       |  - http://10.10.185.238/wordpress/wp-content/plugins/reflex-gallery/readme.txt

- `Mail-masta` is vulnerable to `local file inclusion`. I was able to read the `/etc/passwd` file.

![image](https://github.com/user-attachments/assets/c354e1b8-9594-4a38-98e9-5d428fc966e7)

- I escalated LFI to rce with php filters generated with this [script from synacktiv](https://raw.githubusercontent.com/synacktiv/php_filter_chain_generator/main/php_filter_chain_generator.py).

![image](https://github.com/user-attachments/assets/920d91a5-0bb0-4fe5-821f-d6c5facf5077)

----------------------

### Privesc with cron job

- I checked `/etc/crontab` and  discovered a root cron job running a bash file in the background.

![image](https://github.com/user-attachments/assets/4420502d-2644-42c7-9ef3-eeb518068c24)

- We have write access to the file

![image](https://github.com/user-attachments/assets/df347f77-301f-425e-a76d-20eb139c180c)

- I copied a bash rev shell into the file

![image](https://github.com/user-attachments/assets/fdcf6be7-13ab-4716-8a68-b17d1a19dfca)

- Root shell

![image](https://github.com/user-attachments/assets/26b8bbcc-c68d-4ba6-aba9-add2961aad89)

- Root!!!....

![image](https://github.com/user-attachments/assets/9fa6213e-b360-4886-aec2-9bf8087a2d16)

-------------------

### THANKS FOR READING!!!......

-------------------

### REFERENCES:

- [Phpflter2Rce](https://medium.com/@sundaeGAN/php-wrapper-and-lfi2rce-81c536ef7a06)
- [Mail Masta](https://www.exploit-db.com/exploits/40290)
- [Synactivscript](https://raw.githubusercontent.com/synacktiv/php_filter_chain_generator/main/php_filter_chain_generator.py)

--------------------








