---------------------

### CTF: Tryhackme
### Lab: DAV

----------------------

![image](https://github.com/user-attachments/assets/fe4582b5-be9e-4c00-9006-f7e32b3a60bb)

----------------------

### Reconnaissance

- Rustscan's output

      ❯ rustscan -a 10.10.51.148 -- -sC -sV
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
      Open 10.10.51.148:80
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-08-24 17:06 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 17:06
      Completed NSE at 17:06, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 17:06
      Completed NSE at 17:06, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 17:06
      Completed NSE at 17:06, 0.00s elapsed
      Initiating Ping Scan at 17:06
      Scanning 10.10.51.148 [2 ports]
      Completed Ping Scan at 17:06, 0.22s elapsed (1 total hosts)
      Initiating Parallel DNS resolution of 1 host. at 17:06
      Completed Parallel DNS resolution of 1 host. at 17:06, 0.15s elapsed
      DNS resolution of 1 IPs took 0.15s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 17:06
      Scanning 10.10.51.148 [1 port]
      Discovered open port 80/tcp on 10.10.51.148
      Completed Connect Scan at 17:06, 0.15s elapsed (1 total ports)
      Initiating Service scan at 17:06
      Scanning 1 service on 10.10.51.148
      Completed Service scan at 17:07, 6.62s elapsed (1 service on 1 host)
      NSE: Script scanning 10.10.51.148.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 17:07
      Completed NSE at 17:07, 13.65s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 17:07
      Completed NSE at 17:07, 1.40s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 17:07
      Completed NSE at 17:07, 0.00s elapsed
      Nmap scan report for 10.10.51.148
      Host is up, received syn-ack (0.21s latency).
      Scanned at 2024-08-24 17:06:56 EDT for 22s
      
      PORT   STATE SERVICE REASON  VERSION
      80/tcp open  http    syn-ack Apache httpd 2.4.18 ((Ubuntu))
      | http-methods: 
      |_  Supported Methods: GET HEAD POST OPTIONS
      |_http-title: Apache2 Ubuntu Default Page: It works
      |_http-server-header: Apache/2.4.18 (Ubuntu)

- FFuf's directory fuzzing

![image](https://github.com/user-attachments/assets/4e89ce05-7e7b-4372-aebb-94417dbe1e51)

- The webdav directory requests for credentials,I google searched for web dav default creds and got one `jigsaw:jigsaw` but it didn't work. I got another another from this [site](https://shahjerry33.medium.com/rce-via-webdav-power-of-put-7e1c06c71e60) which
worked. The pair `wampp:xampp` can also serve as default credentials for webdav.

![image](https://github.com/user-attachments/assets/1ca476b9-2e00-49b0-b067-ff33ac57326b)

- Web-based distributed authoring and versioning (WebDAV) is a set of extensions to the HTTP protocol that allows WebDAV clients (such as Microsoft Web Folders) to collaboratively edit and manage files on remote Web servers.We can access webdav with the linux binary `cadaver` as explained in this [blog](https://vk9-sec.com/exploiting-webdav/)

![image](https://github.com/user-attachments/assets/c6e4f88e-c0f7-4dc5-9e66-e9e605fc4563)

- We can add a shell to the webdav directory with the `put` keyword.

![image](https://github.com/user-attachments/assets/bd2ca1a9-3cca-4186-86a0-d99199f8758f)

- Shell access

![image](https://github.com/user-attachments/assets/18904bc1-14cc-4199-b8ca-10be83e535f4)

---------------------------

### Privilege Escalation

- Running `sudo -l` shows that we can use binary `/bin/cat` to read files without password

![image](https://github.com/user-attachments/assets/b5efb7a9-f516-457b-8ea8-7ab6aa724560)

- To read root, run `sudo /bin/cat /root/root.txt`

![image](https://github.com/user-attachments/assets/6eb7d576-22d6-468b-a5ad-2ee56fe6ed55)

---------------------------

### THANKS FOR READING!!!!

----------------------------

### REFERENCES:

- [Webdav](https://shahjerry33.medium.com/rce-via-webdav-power-of-put-7e1c06c71e60)
- [Cadaver](https://vk9-sec.com/exploiting-webdav/)

----------------------------





