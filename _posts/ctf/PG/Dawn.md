-----------------

### CTF: Proving Grounds
### Lab: Dawn

-----------------

![image](https://github.com/user-attachments/assets/8fb3698b-c2de-4d5f-8d45-0cdc95bda43a)

-----------------

### RECONNAISSANCE

- Rustscan's output

      ❯ rustscan -a 192.168.170.11 -- -Pn -sC -sV
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
      Open 192.168.170.11:80
      Open 192.168.170.11:139
      Open 192.168.170.11:445
      Open 192.168.170.11:3306
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-09-19 15:56 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 15:56
      Completed NSE at 15:56, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 15:56
      Completed NSE at 15:56, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 15:56
      Completed NSE at 15:56, 0.00s elapsed
      Initiating Parallel DNS resolution of 1 host. at 15:56
      Completed Parallel DNS resolution of 1 host. at 15:56, 6.49s elapsed
      DNS resolution of 1 IPs took 6.49s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 2, CN: 0]
      Initiating Connect Scan at 15:56
      Scanning 192.168.170.11 [4 ports]
      Discovered open port 80/tcp on 192.168.170.11
      Discovered open port 445/tcp on 192.168.170.11
      Discovered open port 3306/tcp on 192.168.170.11
      Discovered open port 139/tcp on 192.168.170.11
      Completed Connect Scan at 15:56, 0.18s elapsed (4 total ports)
      Initiating Service scan at 15:56
      Scanning 4 services on 192.168.170.11
      Completed Service scan at 15:57, 11.56s elapsed (4 services on 1 host)
      NSE: Script scanning 192.168.170.11.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 15:57
      Completed NSE at 15:57, 7.86s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 15:57
      Completed NSE at 15:57, 1.49s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 15:57
      Completed NSE at 15:57, 0.00s elapsed
      Nmap scan report for 192.168.170.11
      Host is up, received user-set (0.17s latency).
      Scanned at 2024-09-19 15:56:56 EDT for 21s
      
      PORT     STATE SERVICE     REASON  VERSION
      80/tcp   open  http        syn-ack Apache httpd 2.4.38 ((Debian))
      |_http-server-header: Apache/2.4.38 (Debian)
      | http-methods: 
      |_  Supported Methods: GET POST OPTIONS HEAD
      |_http-title: Site doesn't have a title (text/html).
      139/tcp  open  netbios-ssn syn-ack Samba smbd 3.X - 4.X (workgroup: WORKGROUP)
      445/tcp  open  netbios-ssn syn-ack Samba smbd 4.9.5-Debian (workgroup: WORKGROUP)
      3306/tcp open  mysql       syn-ack MySQL 5.5.5-10.3.15-MariaDB-1
      | mysql-info: 
      |   Protocol: 10
      |   Version: 5.5.5-10.3.15-MariaDB-1
      |   Thread ID: 17
      |   Capabilities flags: 63486
      |   Some Capabilities: Support41Auth, FoundRows, SupportsLoadDataLocal, Speaks41ProtocolNew, IgnoreSpaceBeforeParenthesis, SupportsCompression, SupportsTransactions, IgnoreSigpipes, LongColumnFlag, Speaks41ProtocolOld, InteractiveClient, ConnectWithDatabase, DontAllowDatabaseTableColumn, ODBCClient, SupportsMultipleResults, SupportsMultipleStatments, SupportsAuthPlugins
      |   Status: Autocommit
      |   Salt: ;[;TrbvJM#eTW8Rrj;&L
      |_  Auth Plugin Name: mysql_native_password
      Service Info: Host: DAWN
      
      Host script results:
      | smb-security-mode: 
      |   account_used: guest
      |   authentication_level: user
      |   challenge_response: supported
      |_  message_signing: disabled (dangerous, but default)
      |_clock-skew: mean: 1h19m39s, deviation: 2h18m34s, median: -21s
      | smb-os-discovery: 
      |   OS: Windows 6.1 (Samba 4.9.5-Debian)
      |   Computer name: dawn
      |   NetBIOS computer name: DAWN\x00
      |   Domain name: dawn
      |   FQDN: dawn.dawn
      |_  System time: 2024-09-19T15:56:48-04:00
      | p2p-conficker: 
      |   Checking for Conficker.C or higher...
      |   Check 1 (port 36516/tcp): CLEAN (Couldn't connect)
      |   Check 2 (port 61680/tcp): CLEAN (Couldn't connect)
      |   Check 3 (port 33559/udp): CLEAN (Timeout)
      |   Check 4 (port 7645/udp): CLEAN (Failed to receive data)
      |_  0/4 checks are positive: Host is CLEAN or ports are blocked
      | smb2-security-mode: 
      |   3:1:1: 
      |_    Message signing enabled but not required
      | smb2-time: 
      |   date: 2024-09-19T19:56:49
      |_  start_date: N/A

- FFUF's output

![image](https://github.com/user-attachments/assets/c922c4d6-2416-42c5-93c2-be8cc24db5c0)

- Enumerating SMB shares with smbclient

![image](https://github.com/user-attachments/assets/a70dbd2a-bc45-441d-9a0d-50f883875b28)

- We have write access to shares `ITDEPT` with no password, I was able to upload a file.

![image](https://github.com/user-attachments/assets/cc55b7cf-fb50-47e7-b8fe-a219dbc4188f)

- In the `logs` directory, I discovered a `management.log` file.

![image](https://github.com/user-attachments/assets/a57d861b-08cc-4c83-9125-879dfaa05c97)

- The `management.log` file contains processes running on the server. I noticed a process that uses `sh` to run a  file `web-control` in the
`ITDEPT` shares.

![image](https://github.com/user-attachments/assets/06240162-e4f5-4c8f-9044-f75a74ffa102)

- I spawned a shell by adding a bash reverse shell code to a malicious script I named `web-control`

![image](https://github.com/user-attachments/assets/cf5c3f4a-1e9e-4e06-a7d7-e0ab9fe85ea5)

- I uploaded the file

![image](https://github.com/user-attachments/assets/29cfc0bc-5952-40db-9cf4-4d180f24329f)

- Shell access as `www-data`

![image](https://github.com/user-attachments/assets/7831f260-2a04-43b8-bc3f-2b4e8d2d43ec)

-----------------

### Privesc with sudoers rule

- I discovered that I can run sudo as root with no pasword based on the output of `sudo -l`.

![image](https://github.com/user-attachments/assets/a5ac96b1-1993-42eb-9905-447bf4ac6bba)

- I used `sudo -u root /usr/bin/sudo bash -p` to spawn a root shell

![image](https://github.com/user-attachments/assets/8a15cb41-a2e0-4a74-8d1c-effea1a8f495)

- Root shell

![image](https://github.com/user-attachments/assets/39f78003-5405-4afc-bb98-8dc6eb2b8188)

----------------

### THANKS FOR READING

-----------------













