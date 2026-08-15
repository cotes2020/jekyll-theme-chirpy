-------------------

### CTF: PROVING GROUNDS
### LAB: Inclusiveness

-------------------

### RECONNAISSANCE

- Rustscan's output

      ❯ rustscan -a 192.168.230.14 -- -sC -sV
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
      Open 192.168.230.14:22
      Open 192.168.230.14:21
      Open 192.168.230.14:80
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-08-30 19:17 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 19:17
      Completed NSE at 19:17, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 19:17
      Completed NSE at 19:17, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 19:17
      Completed NSE at 19:17, 0.00s elapsed
      Initiating Ping Scan at 19:17
      Scanning 192.168.230.14 [2 ports]
      Completed Ping Scan at 19:17, 0.73s elapsed (1 total hosts)
      Initiating Parallel DNS resolution of 1 host. at 19:17
      Completed Parallel DNS resolution of 1 host. at 19:17, 0.51s elapsed
      DNS resolution of 1 IPs took 0.51s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 19:17
      Scanning 192.168.230.14 [3 ports]
      Discovered open port 22/tcp on 192.168.230.14
      Discovered open port 80/tcp on 192.168.230.14
      Discovered open port 21/tcp on 192.168.230.14
      Completed Connect Scan at 19:17, 0.61s elapsed (3 total ports)
      Initiating Service scan at 19:17
      Scanning 3 services on 192.168.230.14
      Completed Service scan at 19:17, 7.64s elapsed (3 services on 1 host)
      NSE: Script scanning 192.168.230.14.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 19:17
      NSE: [ftp-bounce 192.168.230.14:21] PORT response: 500 Illegal PORT command.
      Completed NSE at 19:17, 16.66s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 19:17
      Completed NSE at 19:17, 3.02s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 19:17
      Completed NSE at 19:17, 0.01s elapsed
      Nmap scan report for 192.168.230.14
      Host is up, received syn-ack (0.69s latency).
      Scanned at 2024-08-30 19:17:09 EDT for 31s
      
      PORT   STATE SERVICE REASON  VERSION
      21/tcp open  ftp     syn-ack vsftpd 3.0.3
      | ftp-anon: Anonymous FTP login allowed (FTP code 230)
      |_drwxrwxrwx    2 0        0            4096 Feb 08  2020 pub [NSE: writeable]
      | ftp-syst: 
      |   STAT: 
      | FTP server status:
      |      Connected to ::ffff:192.168.45.189
      |      Logged in as ftp
      |      TYPE: ASCII
      |      No session bandwidth limit
      |      Session timeout in seconds is 300
      |      Control connection is plain text
      |      Data connections will be plain text
      |      At session startup, client count was 4
      |      vsFTPd 3.0.3 - secure, fast, stable
      |_End of status
      22/tcp open  ssh     syn-ack OpenSSH 7.9p1 Debian 10+deb10u1 (protocol 2.0)
      | ssh-hostkey: 
      |   2048 06:1b:a3:92:83:a5:7a:15:bd:40:6e:0c:8d:98:27:7b (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC8Yl88LxuiPiXQGaZ6fB6K88oCmL/yXhY4Y3j/9PjnFHPRCqM18y4Ol7Q9LMr5CN042Zs/WMt05YE99R5j98fPGD0hIqxKpRpW8ZeDsfZdG479t3dSkM0OAL+hY4V4Wwbk768DxnLUw0ujGuh38UDl3gyYVBFpFZgRb7zBuYRzjIdWijpXm23sbXti4TO6KTC4KVm1BTzT4CVFxBakuuvk1Ieraeusc9agTfCVx7dkN2OX79jAc1uzZNE+BtokFGIYMvMAA7ejZT504cp1Bccbn+OUwlcRLFJbOO2jrXPj8j4MKEz6klMO7mIMvaHFRQ1Z5kBtH7QIGG97D5qhkD8X
      |   256 cb:38:83:26:1a:9f:d3:5d:d3:fe:9b:a1:d3:bc:ab:2c (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBGNCidfAh8l1B4elJK42/1YqrUEBlGWDjg7ZWacpptAfCGBbSC+agR4LWiEtsnQYX4aWXRGydjc7UggCgpHbDr0=
      |   256 65:54:fc:2d:12:ac:e1:84:78:3e:00:23:fb:e4:c9:ee (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJEkCe1XYRTFeHyzWuvZ3JkIkWwD4pGHBcTGEGYYcJhv
      80/tcp open  http    syn-ack Apache httpd 2.4.38 ((Debian))
      | http-methods: 
      |_  Supported Methods: HEAD GET POST OPTIONS
      |_http-title: Apache2 Debian Default Page: It works
      |_http-server-header: Apache/2.4.38 (Debian)
      Service Info: OSs: Unix, Linux; CPE: cpe:/o:linux:linux_kernel

- FFUF's output

![image](https://github.com/user-attachments/assets/2d956005-3c07-445d-a0fb-3faf6682efc2)

- I tried to access the `robots.txt` but I met a forbidden error that only search engines can access it.

![image](https://github.com/user-attachments/assets/ba8437b7-cafc-4182-8bc5-84aebf650f8d)

- I bypassed it by setting the `User-Agent` header to `Google` with curl.

      ❯ curl http://192.168.230.14/robots.txt -H "User-Agent: Google"
      User-agent: *
      Disallow: /secret_information/

- The index_page in the secret directory is vulnerable to LFI, I was able to read `/etc/passwd`.

![image](https://github.com/user-attachments/assets/10c4cb18-2a83-4973-86ae-aeea6db0db22)

- Instead of the normal LFI2RCE method of poisoning logs, I gained rce with the aid of the idea from this [blog](https://medium.com/@sundaeGAN/php-wrapper-and-lfi2rce-81c536ef7a06).
It involves chaining php filters to represent characters and gain remote code execution.Now, I can execute shell commands

![image](https://github.com/user-attachments/assets/9dfa2cb7-2dca-4a74-b9a5-a3544c98d08b)

- Shell access

 ![image](https://github.com/user-attachments/assets/4ca32d81-c4f4-4fbe-b435-3d270ab6ab57)

---------------------------
 
 ### PRIVESC with PATH hijacking

 - Tom's home directory contains a binary that allows him to spawn a root shell.

![image](https://github.com/user-attachments/assets/8035e616-5a22-476c-a4c5-85c73f7b7322)

- The binary determines if the user is tom by executing binary `whoami`.The binary does not state the absolute path of whoami,we can hijack the path
and direct to our malicious binary.

      #include <stdio.h>
      #include <unistd.h>
      #include <stdlib.h>
      #include <string.h>
      
      int main() {
      
          printf("checking if you are tom...\n");
          FILE* f = popen("whoami", "r");
      
          char user[80];
          fgets(user, 80, f);

- To gain the root shell,we just need a  script that prints the characters `tom` because the binary compares the output of whoami to
the string `tom`.

       if (strncmp(user, "tom", 3) == 0) {
              printf("access granted.\n");
              setuid(geteuid());
              execlp("sh", "sh", (char *) 0);

- The exploitation process involves adding `/tmp` to the path variable, creating a python script that print `tom` and giving it execute permissions.

![image](https://github.com/user-attachments/assets/fb46a597-c2bc-4dfd-a121-830aecda8dba)

- Root!!!!

![image](https://github.com/user-attachments/assets/e09c603f-7801-4b28-9ad5-893a0362f838)

--------------------------

### THANKS FOR READING!!!

--------------------------

### REFERENCES:

- [PHP FILTER RCE](https://medium.com/@sundaeGAN/php-wrapper-and-lfi2rce-81c536ef7a06)

---------------------------
