-------------------

### CTF: TRYHACKME
### LAB: Vulnetinternal

--------------------

![image](https://github.com/user-attachments/assets/5ce806ea-a9d7-40ba-b29f-4389331f54ab)

--------------------

### RECONNAISSANCE

- Rustscan's output

      ❯ rustscan -a 10.10.34.72 -- -sC -sV -Pn
      .----. .-. .-. .----..---.  .----. .---.   .--.  .-. .-.
      | {}  }| { } |{ {__ {_   _}{ {__  /  ___} / {} \ |  `| |
      | .-. \| {_} |.-._} } | |  .-._} }\     }/  /\  \| |\  |
      `-' `-'`-----'`----'  `-'  `----'  `---' `-'  `-'`-' `-'
      The Modern Day Port Scanner.
      ________________________________________
      : https://discord.gg/GFrQsGy           :
      : https://github.com/RustScan/RustScan :
       --------------------------------------
      Please contribute more quotes to our GitHub https://github.com/rustscan/rustscan
      
      [~] The config file is expected to be at "/home/sensei/.rustscan.toml"
      [!] File limit is lower than default batch size. Consider upping with --ulimit. May cause harm to sensitive servers
      [!] Your file limit is very small, which negatively impacts RustScan's speed. Use the Docker image, or up the Ulimit with '--ulimit 5000'. 
      Open 10.10.34.72:22
      Open 10.10.34.72:111
      Open 10.10.34.72:139
      Open 10.10.34.72:445
      Open 10.10.34.72:873
      Open 10.10.34.72:2049
      Open 10.10.34.72:6379
      Open 10.10.34.72:36795
      Open 10.10.34.72:38417
      Open 10.10.34.72:45403
      Open 10.10.34.72:51745
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-09-09 04:37 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 04:37
      Completed NSE at 04:37, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 04:37
      Completed NSE at 04:37, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 04:37
      Completed NSE at 04:37, 0.00s elapsed
      Initiating Parallel DNS resolution of 1 host. at 04:37
      Completed Parallel DNS resolution of 1 host. at 04:37, 0.17s elapsed
      DNS resolution of 1 IPs took 0.17s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 04:37
      Scanning 10.10.34.72 [11 ports]
      Discovered open port 139/tcp on 10.10.34.72
      Discovered open port 111/tcp on 10.10.34.72
      Discovered open port 22/tcp on 10.10.34.72
      Discovered open port 445/tcp on 10.10.34.72
      Discovered open port 6379/tcp on 10.10.34.72
      Discovered open port 36795/tcp on 10.10.34.72
      Discovered open port 873/tcp on 10.10.34.72
      Discovered open port 45403/tcp on 10.10.34.72
      Discovered open port 2049/tcp on 10.10.34.72
      Discovered open port 51745/tcp on 10.10.34.72
      Discovered open port 38417/tcp on 10.10.34.72
      Completed Connect Scan at 04:37, 0.33s elapsed (11 total ports)
      Initiating Service scan at 04:37
      Scanning 11 services on 10.10.34.72
      Completed Service scan at 04:37, 16.89s elapsed (11 services on 1 host)
      NSE: Script scanning 10.10.34.72.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 04:37
      Completed NSE at 04:37, 6.26s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 04:37
      Completed NSE at 04:37, 0.86s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 04:37
      Completed NSE at 04:37, 0.01s elapsed
      Nmap scan report for 10.10.34.72
      Host is up, received user-set (0.17s latency).
      Scanned at 2024-09-09 04:37:32 EDT for 25s
      
      PORT      STATE SERVICE     REASON  VERSION
      22/tcp    open  ssh         syn-ack OpenSSH 7.6p1 Ubuntu 4ubuntu0.3 (Ubuntu Linux; protocol 2.0)
      | ssh-hostkey: 
      |   2048 5e:27:8f:48:ae:2f:f8:89:bb:89:13:e3:9a:fd:63:40 (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDagA3GVO7hKpJpO1Vr6+z3Y9xjoeihZFWXSrBG2MImbpPH6jk+1KyJwQpGmhMEGhGADM1LbmYf3goHku11Ttb0gbXaCt+mw1Ea+K0H00jA0ce2gBqev+PwZz0ysxCLUbYXCSv5Dd1XSa67ITSg7A6h+aRfkEVN2zrbM5xBQiQv6aBgyaAvEHqQ73nZbPdtwoIGkm7VL9DATomofcEykaXo3tmjF2vRTN614H0PpfZBteRpHoJI4uzjwXeGVOU/VZcl7EMBd/MRHdspvULJXiI476ID/ZoQLT2zQf5Q2vqI3ulMj5CB29ryxq58TVGSz/sFv1ZBPbfOl9OvuBM5BTBV
      |   256 f4:fe:0b:e2:5c:88:b5:63:13:85:50:dd:d5:86:ab:bd (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBNM0XfxK0hrF7d4C5DCyQGK3ml9U0y3Nhcvm6N9R+qv2iKW21CNEFjYf+ZEEi7lInOU9uP2A0HZG35kEVmuideE=
      |   256 82:ea:48:85:f0:2a:23:7e:0e:a9:d9:14:0a:60:2f:ad (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJPRO3XCBfxEo0XhViW8m/V+IlTWehTvWOyMDOWNJj+i
      111/tcp   open  rpcbind     syn-ack 2-4 (RPC #100000)
      | rpcinfo: 
      |   program version    port/proto  service
      |   100000  2,3,4        111/tcp   rpcbind
      |   100000  2,3,4        111/udp   rpcbind
      |   100000  3,4          111/tcp6  rpcbind
      |   100000  3,4          111/udp6  rpcbind
      |   100003  3           2049/udp   nfs
      |   100003  3           2049/udp6  nfs
      |   100003  3,4         2049/tcp   nfs
      |   100003  3,4         2049/tcp6  nfs
      |   100005  1,2,3      33319/udp   mountd
      |   100005  1,2,3      38004/udp6  mountd
      |   100005  1,2,3      45403/tcp   mountd
      |   100005  1,2,3      54025/tcp6  mountd
      |   100021  1,3,4      36795/tcp   nlockmgr
      |   100021  1,3,4      45263/tcp6  nlockmgr
      |   100021  1,3,4      45833/udp   nlockmgr
      |   100021  1,3,4      47826/udp6  nlockmgr
      |   100227  3           2049/tcp   nfs_acl
      |   100227  3           2049/tcp6  nfs_acl
      |   100227  3           2049/udp   nfs_acl
      |_  100227  3           2049/udp6  nfs_acl
      139/tcp   open  netbios-ssn syn-ack Samba smbd 3.X - 4.X (workgroup: WORKGROUP)
      445/tcp   open  netbios-ssn syn-ack Samba smbd 4.7.6-Ubuntu (workgroup: WORKGROUP)
      873/tcp   open  rsync       syn-ack (protocol version 31)
      2049/tcp  open  nfs         syn-ack 3-4 (RPC #100003)
      6379/tcp  open  redis       syn-ack Redis key-value store
      36795/tcp open  nlockmgr    syn-ack 1-4 (RPC #100021)
      38417/tcp open  mountd      syn-ack 1-3 (RPC #100005)
      45403/tcp open  mountd      syn-ack 1-3 (RPC #100005)
      51745/tcp open  mountd      syn-ack 1-3 (RPC #100005)
      Service Info: Host: VULNNET-INTERNAL; OS: Linux; CPE: cpe:/o:linux:linux_kernel
      
      Host script results:
      | smb2-time: 
      |   date: 2024-09-09T08:37:31
      |_  start_date: N/A
      |_clock-skew: mean: -40m20s, deviation: 1h09m16s, median: -21s
      | p2p-conficker: 
      |   Checking for Conficker.C or higher...
      |   Check 1 (port 28299/tcp): CLEAN (Couldn't connect)
      |   Check 2 (port 22898/tcp): CLEAN (Couldn't connect)
      |   Check 3 (port 59608/udp): CLEAN (Failed to receive data)
      |   Check 4 (port 48785/udp): CLEAN (Failed to receive data)
      |_  0/4 checks are positive: Host is CLEAN or ports are blocked
      | nbstat: NetBIOS name: VULNNET-INTERNA, NetBIOS user: <unknown>, NetBIOS MAC: <unknown> (unknown)
      | Names:
      |   VULNNET-INTERNA<00>  Flags: <unique><active>
      |   VULNNET-INTERNA<03>  Flags: <unique><active>
      |   VULNNET-INTERNA<20>  Flags: <unique><active>
      |   WORKGROUP<00>        Flags: <group><active>
      |   WORKGROUP<1e>        Flags: <group><active>
      | Statistics:
      |   00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00
      |   00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00
      |_  00:00:00:00:00:00:00:00:00:00:00:00:00:00
      | smb2-security-mode: 
      |   3:1:1: 
      |_    Message signing enabled but not required
      | smb-os-discovery: 
      |   OS: Windows 6.1 (Samba 4.7.6-Ubuntu)
      |   Computer name: vulnnet-internal
      |   NetBIOS computer name: VULNNET-INTERNAL\x00
      |   Domain name: \x00
      |   FQDN: vulnnet-internal
      |_  System time: 2024-09-09T10:37:31+02:00
      | smb-security-mode: 
      |   account_used: guest
      |   authentication_level: user
      |   challenge_response: supported
      |_  message_signing: disabled (dangerous, but default)

------------------

### Enumerating SMB

- Enumerating port `139` and `445`[smb&&netbios] with `enum4linux` shows listing access to a smb share `shares`.

![image](https://github.com/user-attachments/assets/48ee7f2e-49ad-4a4b-a27b-6f2a8f457d60)

- The smb share contains 2 directories `temp` and `data`.The directories also contain files which I downloaded to my machine with the `get` keyword.

![image](https://github.com/user-attachments/assets/830b388a-1eac-4878-aa20-a87cb9debff9)

- One of the files contains the first flag

![image](https://github.com/user-attachments/assets/552c724e-61d5-47ad-b34b-c6a71995801a)

-----------------------

### Enumerating NFS

- I enumerated NFS with `showmount` to check for shares and got one `/opt/conf`

![image](https://github.com/user-attachments/assets/534cebb0-236b-44c3-85ce-35a35117ddd6)

- I mounted the share to my local machine.

![image](https://github.com/user-attachments/assets/a9b33f5f-344c-45e4-ad56-96af7fd99ca4)

- While checking the redis directory, I discovered a redis.conf containing a password for authentication.

![image](https://github.com/user-attachments/assets/3574c689-6d0e-4246-b504-54c4d41d44a3)

### Enumerating REDIS on port 6379

- I connected to the redis port with `redis-cli -h [host]`.Then, `AUTH [password]` to access the db.

![image](https://github.com/user-attachments/assets/a12a287e-31cd-4791-bb01-feb17b235496)

- I listed all the db keys with `KEYS *` and discovered base64 encoded values in a key `authlist`

![image](https://github.com/user-attachments/assets/8fb31421-2ec6-490a-a6bf-24e7ebb9c6c3)

- I decoded the base64 string and got the password for the `rsync` service.

![image](https://github.com/user-attachments/assets/e69c4e1c-59c8-4103-b097-43a4aac4bbc9)

--------------------

### ENUMERATING rsync on port `879`

- Rsync allows file downloading and uploading.You can interact with a rsync service with `rsync`. I enumerated the directories with `rsync [host]::`.We have write access to `.ssh` of the home user `sys-internal`.

![image](https://github.com/user-attachments/assets/cae5215c-afc4-4467-b6ef-59b8a2bcb03a)

- We can access ssh by copying public key in `~/.ssh/id_rsa.pub` to authorized keys in user `sys-internal` .ssh directory.

![image](https://github.com/user-attachments/assets/49f5abf4-7b28-4321-b4a3-4fc6c123d94a)

- SSH Access

![image](https://github.com/user-attachments/assets/afc42f2c-6de1-43bf-a277-80fb88fd4231)

-----------------------

### PRIVESC with TEAM-CITY

- I ran `ps aux` to check for running processes,I discovered team city running as root on the server.

![image](https://github.com/user-attachments/assets/3f051be5-dc07-4e3f-9848-0e1b80da7944)

- Then, I checked for internal services with `ss -atur` and spotted a service on 8111.I used to `nc` to test the server and discovered it is an http server.I summed up that team city is running on that service.

![image](https://github.com/user-attachments/assets/594aa799-9268-43d6-89f0-636e0556f76f)

- I port forwarded the service with chisel.

![image](https://github.com/user-attachments/assets/64debe25-8e1b-4585-a4d3-35e32cab077e)

![image](https://github.com/user-attachments/assets/73d89cc0-f0b6-4354-8a53-d31f9d39309e)

- Team city requires an authentication token to access the super user account. I got a grep one liner from this [site](https://exploit-notes.hdks.org/exploit/web/teamcity-pentesting/) to check for it .

ONE-LINER-:```grep -rni 'authentication token' TeamCity/logs 2</dev/null```

![image](https://github.com/user-attachments/assets/34bb04e5-2be1-49a3-bb18-35aa42cf1950)

- The last token worked

![image](https://github.com/user-attachments/assets/b4d25510-1e71-4e49-a3b7-cd2e27dccb43)

- I followed the steps in this [site](https://exploit-notes.hdks.org/exploit/web/teamcity-pentesting/) to spawn a revshell.

- Create a new project, pick "manually"

![image](https://github.com/user-attachments/assets/e0a9ec77-75e0-4dce-96f0-010b60570c8a)

- Create a build configuration

![image](https://github.com/user-attachments/assets/86bf10a6-27d2-4552-852a-5d0c604b4094)

- Click on build steps

![image](https://github.com/user-attachments/assets/57a9b4d2-6687-4eea-be50-6c77b5c97ce8)

- Then, add a build step, set the runner type to `command line` and copy any bash reverse shell code and save

![image](https://github.com/user-attachments/assets/09f05185-e513-4543-984c-68bfdbb22b09)

- Click on run to spawn a reverse shell as root

![image](https://github.com/user-attachments/assets/230358c8-5466-465e-a4cc-af12c31320e0)

- Reverse shell as root

![image](https://github.com/user-attachments/assets/32893314-b416-4008-9a94-fa6d8c90a3b0)

- Root access.....!!!

![image](https://github.com/user-attachments/assets/e7e16f40-46de-4e41-a511-082d1c13de0e)

--------------------

### THANKS FOR READING.....!!!!!

---------------------

### REFERENCES:

- [Redis Commands](https://redis.io/learn/howtos/quick-start/cheat-sheet)
- [Team City Pentest](https://exploit-notes.hdks.org/exploit/web/teamcity-pentesting/)

----------------------






  
























  


