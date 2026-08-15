-------------

### CTF: TRYHACKME
### Lab: CheeseCTF

--------------

![image](https://github.com/user-attachments/assets/0a01c68a-e4be-4397-ba17-49a4020885ed)

--------------

- Nmap's network scan

      ❯ sudo nmap -sC -sV -sX 10.10.123.44
      [sudo] password for sensei: 
      Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-10-02 13:06 EDT
      Nmap scan report for 10.10.123.44
      Host is up (0.15s latency).
      Not shown: 997 closed tcp ports (reset)
      PORT     STATE SERVICE VERSION
      22/tcp   open  ssh     OpenSSH 8.2p1 Ubuntu 4ubuntu0.11 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   3072 b1:c1:22:9f:11:10:5f:64:f1:33:72:70:16:3c:80:06 (RSA)
      |   256 6d:33:e3:bd:70:62:59:93:4d:ab:8b:fe:ef:e8:a7:b2 (ECDSA)
      |_  256 89:2e:17:84:ed:48:7a:ae:d9:8c:9b:a5:8e:24:04:bd (ED25519)
      80/tcp   open  http    Apache httpd 2.4.41 ((Ubuntu))
      |_http-title: The Cheese Shop
      |_http-server-header: Apache/2.4.41 (Ubuntu)
      4444/tcp open  krb524?
      | fingerprint-strings: 
      |   DNSStatusRequestTCP, DNSVersionBindReqTCP, GenericLines, GetRequest, HTTPOptions, Help, Kerberos, NULL, RPCCheck, RTSPRequest, SSLSessionReq, SSLv23SessionReq, TLSSessionReq, TerminalServerCookie: 
      |     HTTP/1.1 200 OK
      |     m<TITLE>MGI ZOOM Image Server</TITLE>lVersion: n
      |_    Build: 0build/><BR>
      1 service unrecognized despite returning data. If you know the service/version, please submit the following fingerprint at https://nmap.org/cgi-bin/submit.cgi?new-service :
      SF-Port4444-TCP:V=7.94SVN%I=7%D=10/2%Time=66FD7DD8%P=x86_64-pc-linux-gnu%r
      SF:(NULL,5B,"HTTP/1\.1\x20200\x20OK\r\nm<TITLE>MGI\x20ZOOM\x20Image\x20Ser
      SF:ver</TITLE>lVersion:\x20n\xf99\n\t\tBuild:\x200build/><BR>\n\n")%r(GetR
      SF:equest,5B,"HTTP/1\.1\x20200\x20OK\r\nm<TITLE>MGI\x20ZOOM\x20Image\x20Se
      SF:rver</TITLE>lVersion:\x20n\xf99\n\t\tBuild:\x200build/><BR>\n\n")%r(SSL
      SF:SessionReq,5B,"HTTP/1\.1\x20200\x20OK\r\nm<TITLE>MGI\x20ZOOM\x20Image\x
      SF:20Server</TITLE>lVersion:\x20n\xf99\n\t\tBuild:\x200build/><BR>\n\n")%r
      SF:(TLSSessionReq,5B,"HTTP/1\.1\x20200\x20OK\r\nm<TITLE>MGI\x20ZOOM\x20Ima
      SF:ge\x20Server</TITLE>lVersion:\x20n\xf99\n\t\tBuild:\x200build/><BR>\n\n
      SF:")%r(SSLv23SessionReq,5B,"HTTP/1\.1\x20200\x20OK\r\nm<TITLE>MGI\x20ZOOM
      SF:\x20Image\x20Server</TITLE>lVersion:\x20n\xf99\n\t\tBuild:\x200build/><
      SF:BR>\n\n")%r(GenericLines,5B,"HTTP/1\.1\x20200\x20OK\r\nm<TITLE>MGI\x20Z
      SF:OOM\x20Image\x20Server</TITLE>lVersion:\x20n\xf99\n\t\tBuild:\x200build
      SF:/><BR>\n\n")%r(HTTPOptions,5B,"HTTP/1\.1\x20200\x20OK\r\nm<TITLE>MGI\x2
      SF:0ZOOM\x20Image\x20Server</TITLE>lVersion:\x20n\xf99\n\t\tBuild:\x200bui
      SF:ld/><BR>\n\n")%r(RTSPRequest,5B,"HTTP/1\.1\x20200\x20OK\r\nm<TITLE>MGI\
      SF:x20ZOOM\x20Image\x20Server</TITLE>lVersion:\x20n\xf99\n\t\tBuild:\x200b
      SF:uild/><BR>\n\n")%r(RPCCheck,5B,"HTTP/1\.1\x20200\x20OK\r\nm<TITLE>MGI\x
      SF:20ZOOM\x20Image\x20Server</TITLE>lVersion:\x20n\xf99\n\t\tBuild:\x200bu
      SF:ild/><BR>\n\n")%r(DNSVersionBindReqTCP,5B,"HTTP/1\.1\x20200\x20OK\r\nm<
      SF:TITLE>MGI\x20ZOOM\x20Image\x20Server</TITLE>lVersion:\x20n\xf99\n\t\tBu
      SF:ild:\x200build/><BR>\n\n")%r(DNSStatusRequestTCP,5B,"HTTP/1\.1\x20200\x
      SF:20OK\r\nm<TITLE>MGI\x20ZOOM\x20Image\x20Server</TITLE>lVersion:\x20n\xf
      SF:99\n\t\tBuild:\x200build/><BR>\n\n")%r(Help,5B,"HTTP/1\.1\x20200\x20OK\
      SF:r\nm<TITLE>MGI\x20ZOOM\x20Image\x20Server</TITLE>lVersion:\x20n\xf99\n\
      SF:t\tBuild:\x200build/><BR>\n\n")%r(TerminalServerCookie,5B,"HTTP/1\.1\x2
      SF:0200\x20OK\r\nm<TITLE>MGI\x20ZOOM\x20Image\x20Server</TITLE>lVersion:\x
      SF:20n\xf99\n\t\tBuild:\x200build/><BR>\n\n")%r(Kerberos,5B,"HTTP/1\.1\x20
      SF:200\x20OK\r\nm<TITLE>MGI\x20ZOOM\x20Image\x20Server</TITLE>lVersion:\x2
      SF:0n\xf99\n\t\tBuild:\x200build/><BR>\n\n");
      Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

- FFUF's output for files fuzzing

![image](https://github.com/user-attachments/assets/7b1fef8d-d244-416f-b6df-40636d431218)

- In one of the html files named `messages.html`,I noticed an hyper link to a php page that allows us to read files.

![image](https://github.com/user-attachments/assets/2c55887b-d034-4909-802f-e0bc91373786)

- It uses the include() function which vulnerable to lfi.

![image](https://github.com/user-attachments/assets/0061d5c5-9773-4d66-8b04-8682a89283b5)

- I generated a php filters to gain rce with the synactiv `filter.py` script.

![image](https://github.com/user-attachments/assets/2742b966-cd18-49ba-a3ee-f569272656a8)

- RCE achieved

![image](https://github.com/user-attachments/assets/f893e6f6-e740-49e0-ad57-10abd35d3f5f)

- Reverse shell as user `www-data`

![image](https://github.com/user-attachments/assets/0e078bce-5b5a-4643-91ad-14e4de95650d)

### Pivoting to user `comte`

- We have write access to user `comte`'s authorized keys,we can add our ssh public key to gain access to user `comte` without password.

![image](https://github.com/user-attachments/assets/ac5eff6a-4555-4f20-8d8e-1624907e4620)

![image](https://github.com/user-attachments/assets/581e297d-b111-4402-9902-250a9cd411e4)

- User `comte`

![image](https://github.com/user-attachments/assets/93e9a308-9db7-4f23-8f8b-d021d87ca1e5)

### Privesc with `systemctl` and `root`

- Running `sudo -l` shows we can use `systemctl` to start a service `exploit.timer`.

![image](https://github.com/user-attachments/assets/e9f12afa-3c6d-421e-ad64-c10a68e2ecf3)

- I tried to run it but I noticed an error.

![image](https://github.com/user-attachments/assets/4dcc3415-a62e-4fd9-8de1-a56693535378)

- There is an issue because the `OnbootSec` variable is not set and a variable `Unit` is not set to the target service `exploit.service`.Timers are like schedulers or crons for systemd services.

![image](https://github.com/user-attachments/assets/bc4fa1c7-135e-478a-a4cf-aa199ec309bb)

- Fix

![image](https://github.com/user-attachments/assets/5a5c0ec8-d40c-43a3-a1fc-bac1214f7710)

- The service `exploit.service` runs a bash command that copies binary `xxd` to `/opt` and grants it suid privileges.

![image](https://github.com/user-attachments/assets/4b9f7ea7-d862-429c-bfbb-6b33f4a896df)

- Exploiting it

![image](https://github.com/user-attachments/assets/7bd1e617-3955-48d2-b881-d266daa1ebf6)

### Privesc to `root` with XXD

- `XXD` can be used to read and write files.We can use this to rewrite the `/etc/shadow` and change the root user's hash.

 ![image](https://github.com/user-attachments/assets/4cf9ef00-8b76-496c-841a-48351dfdea93)

- Generate an hash with `openssl`

![image](https://github.com/user-attachments/assets/0c61afd8-cf5d-437e-b3db-8e3c9124c3aa)

- I redirected the output to a file and changed the root hash

![image](https://github.com/user-attachments/assets/451ba02e-df30-477e-8d1b-2d30b4dbb856)

- Then, I rewrote the content with xxd.

![image](https://github.com/user-attachments/assets/746d87e6-8c99-4359-9a5b-bdaf379d4b42)

- Root......!!!

![image](https://github.com/user-attachments/assets/c7709823-0faf-4374-9775-27f7d01d1ea8)

------------------

### THANKS FOR READING!!!!

-------------------

### REFERENCE:

- [Configuring a systemd timer](https://medium.com/@rockibul.islam20/configure-and-implement-of-systemd-timers-8c640cc6d667)

-------------------












