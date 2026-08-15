----------------

### CTF: Proving Grounds
### Lab: Seppuku

----------------

![image](https://github.com/user-attachments/assets/61a92159-250b-4618-a619-5d9688328f35)

----------------

### Reconnaissance:

- Rustscan's output:

      ❯ rustscan -a 192.168.201.90 -- -sC -sV -Pn
      .----. .-. .-. .----..---.  .----. .---.   .--.  .-. .-.
      | {}  }| { } |{ {__ {_   _}{ {__  /  ___} / {} \ |  `| |
      | .-. \| {_} |.-._} } | |  .-._} }\     }/  /\  \| |\  |
      `-' `-'`-----'`----'  `-'  `----'  `---' `-'  `-'`-' `-'
      The Modern Day Port Scanner.
      ________________________________________
      : https://discord.gg/GFrQsGy           :
      : https://github.com/RustScan/RustScan :
       --------------------------------------
      😵 https://admin.tryhackme.com
      
      [~] The config file is expected to be at "/home/sensei/.rustscan.toml"
      [!] File limit is lower than default batch size. Consider upping with --ulimit. May cause harm to sensitive servers
      [!] Your file limit is very small, which negatively impacts RustScan's speed. Use the Docker image, or up the Ulimit with '--ulimit 5000'. 
      Open 192.168.201.90:21
      Open 192.168.201.90:22
      Open 192.168.201.90:80
      Open 192.168.201.90:139
      Open 192.168.201.90:445
      Open 192.168.201.90:7080
      Open 192.168.201.90:7601
      [~] Starting Script(s)
      [>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")
      
      Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
      [~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-09-11 06:22 EDT
      NSE: Loaded 156 scripts for scanning.
      NSE: Script Pre-scanning.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 06:22
      Completed NSE at 06:22, 0.00s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 06:22
      Completed NSE at 06:22, 0.00s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 06:22
      Completed NSE at 06:22, 0.00s elapsed
      Initiating Parallel DNS resolution of 1 host. at 06:22
      Completed Parallel DNS resolution of 1 host. at 06:22, 0.06s elapsed
      DNS resolution of 1 IPs took 0.06s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
      Initiating Connect Scan at 06:22
      Scanning 192.168.201.90 [7 ports]
      Discovered open port 21/tcp on 192.168.201.90
      Discovered open port 80/tcp on 192.168.201.90
      Discovered open port 139/tcp on 192.168.201.90
      Discovered open port 22/tcp on 192.168.201.90
      Discovered open port 445/tcp on 192.168.201.90
      Discovered open port 7080/tcp on 192.168.201.90
      Discovered open port 7601/tcp on 192.168.201.90
      Completed Connect Scan at 06:22, 0.17s elapsed (7 total ports)
      Initiating Service scan at 06:22
      Scanning 7 services on 192.168.201.90
      Completed Service scan at 06:23, 46.92s elapsed (7 services on 1 host)
      NSE: Script scanning 192.168.201.90.
      NSE: Starting runlevel 1 (of 3) scan.
      Initiating NSE at 06:23
      Completed NSE at 06:23, 19.56s elapsed
      NSE: Starting runlevel 2 (of 3) scan.
      Initiating NSE at 06:23
      Completed NSE at 06:23, 9.39s elapsed
      NSE: Starting runlevel 3 (of 3) scan.
      Initiating NSE at 06:23
      Completed NSE at 06:23, 0.00s elapsed
      Nmap scan report for 192.168.201.90
      Host is up, received user-set (0.16s latency).
      Scanned at 2024-09-11 06:22:33 EDT for 76s
      
      PORT     STATE SERVICE       REASON  VERSION
      21/tcp   open  ftp           syn-ack vsftpd 3.0.3
      22/tcp   open  ssh           syn-ack OpenSSH 7.9p1 Debian 10+deb10u2 (protocol 2.0)
      | ssh-hostkey: 
      |   2048 cd:55:a8:e4:0f:28:bc:b2:a6:7d:41:76:bb:9f:71:f4 (RSA)
      | ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDhKnaNVJ/YnScPD1GDZSIfyC/a4jjHhSnoEgi2c/c03kE4JVZbA4cTFeEHGq4PFTyiuchv9w9zNu8XtVIDhILb9K4D38EssujmpekrrAnYkS0yU8Kqas1+3FCY8xjz6a5yVdMk/aQVa4BfFXWnv+rdlio0ZFVdLDaRaG90KMUEVw18Ogzt9lBbnbf7gOR0EGPKW0xzyDyI70u5FJnarDFV9jCZL/flcCL0m+MAycgdFyFqCOTjNxd8Qn2R3rnhgjSER5C9c+qEI/htLmtnXTC0p6AMeTDjO3J57LEB1WFYJ4wkeuEUtPadfhwgDR16XqWmqw2HcBIj1W9H9V47KFfR
      |   256 16:fa:29:e4:e0:8a:2e:7d:37:d2:6f:42:b2:dc:e9:22 (ECDSA)
      | ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBC+yj9GRgyn2boC7Dw9un6PEwviM8NZ1CRTjmrHRFiOT+0co+OOwxD5RRQCxuS22zJgsiDIEka8ypTjYWlnJ9T8=
      |   256 bb:74:e8:97:fa:30:8d:da:f9:5c:99:f0:d9:24:8a:d5 (ED25519)
      |_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIESejQ038eElmlRfbqAgaRSK120jvrz9WQ5UcjxJdJ71
      80/tcp   open  http          syn-ack nginx 1.14.2
      |_http-server-header: nginx/1.14.2
      139/tcp  open  netbios-ssn   syn-ack Samba smbd 3.X - 4.X (workgroup: WORKGROUP)
      445/tcp  open  netbios-ssn   syn-ack Samba smbd 4.9.5-Debian (workgroup: WORKGROUP)
      7080/tcp open  ssl/empowerid syn-ack
      |_http-title: Did not follow redirect to https://192.168.201.90:7080/
      | ssl-cert: Subject: commonName=seppuku/organizationName=LiteSpeedCommunity/stateOrProvinceName=NJ/countryName=US/emailAddress=mail@seppuku/name=openlitespeed/organizationalUnitName=Testing/localityName=Virtual/dnQualifier=openlitespeed/initials=CP
      | Issuer: commonName=seppuku/organizationName=LiteSpeedCommunity/stateOrProvinceName=NJ/countryName=US/emailAddress=mail@seppuku/name=openlitespeed/organizationalUnitName=Testing/localityName=Virtual/dnQualifier=openlitespeed/initials=CP
      | Public Key type: rsa
      | Public Key bits: 2048
      | Signature Algorithm: sha256WithRSAEncryption
      | Not valid before: 2020-05-13T06:51:35
      | Not valid after:  2022-08-11T06:51:35
      | MD5:   2002:61c4:9f2d:6bfa:21d1:477c:21d9:e703
      | SHA-1: e44a:c855:93ba:b3f8:b2f3:7ce5:db7f:a350:2f49:c7ca
      | -----BEGIN CERTIFICATE-----
      | MIIENTCCAx2gAwIBAgIUTA/1/lqL0wXtcQz9EwctzIvjfkYwDQYJKoZIhvcNAQEL
      | BQAwgccxEDAOBgNVBAMMB3NlcHB1a3UxCzAJBgNVBAYTAlVTMRAwDgYDVQQHDAdW
      | aXJ0dWFsMRswGQYDVQQKDBJMaXRlU3BlZWRDb21tdW5pdHkxEDAOBgNVBAsMB1Rl
      | c3RpbmcxCzAJBgNVBAgMAk5KMRswGQYJKoZIhvcNAQkBFgxtYWlsQHNlcHB1a3Ux
      | FjAUBgNVBCkMDW9wZW5saXRlc3BlZWQxCzAJBgNVBCsMAkNQMRYwFAYDVQQuEw1v
      | cGVubGl0ZXNwZWVkMB4XDTIwMDUxMzA2NTEzNVoXDTIyMDgxMTA2NTEzNVowgccx
      | EDAOBgNVBAMMB3NlcHB1a3UxCzAJBgNVBAYTAlVTMRAwDgYDVQQHDAdWaXJ0dWFs
      | MRswGQYDVQQKDBJMaXRlU3BlZWRDb21tdW5pdHkxEDAOBgNVBAsMB1Rlc3Rpbmcx
      | CzAJBgNVBAgMAk5KMRswGQYJKoZIhvcNAQkBFgxtYWlsQHNlcHB1a3UxFjAUBgNV
      | BCkMDW9wZW5saXRlc3BlZWQxCzAJBgNVBCsMAkNQMRYwFAYDVQQuEw1vcGVubGl0
      | ZXNwZWVkMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA8SVGtfXTfTSO
      | N6Umrvf+GIwkhWZe0KJ37rASVks61rn4yIVuQNzQwDWDBuw1IZD9SHnWWm8ejHmb
      | M84sP4n9OCJYlnWrjFfAouH3IFku40Zx9JyVkGTeNA3HrFNN7WkX6yq2wHDHTqn+
      | SeEX9pax9RAk1mm+DZBfZGqkkiZCu/IO2Ro1kHYTnlnvQmj1y07RkdcumVyVNZzi
      | qJxrIZSl7EIUMEQfmkaX8RYigcfn6RsFkFdWPZ9JanNTBVBNrZptegtW6zH/R/Gu
      | CUk7nbzqDm0u6Cs+6IWwENDkfELUBFkEW0rrDFxYhhJ1NmPa3bnLRYuU8RxGiVyN
      | 9BEXNFg1rwIDAQABoxcwFTATBgNVHSUEDDAKBggrBgEFBQcDATANBgkqhkiG9w0B
      | AQsFAAOCAQEA1n5K+UR3K91RltYeVilcq5/ynOHQiDrUZ5zi+/ZmYIUpoOakXzHv
      | Pz8+gOSQ8fLch1ZUtkkAv8i5zaYJZ/WDMs4V6R80h9w9NOANKNOPCrWB1jWteBGG
      | OSGn2Wbd4Ii0rKYFfmxoEags6MRklyFXE0rQoSlgUFsIQaPiisjv2xnm0GgoVmS8
      | tUfRimAXsoBLgl5ZzT56MlfX5QSrqYy6UAtBeIc7R4C7lWcpay91b8JCXsGspjfX
      | OBnzFQJ3tuMvtsDWD1NBPGWH5LpWRiaLalyz63KvWKdD3pr/5l2OKgU49qOVU/lQ
      | NLEdNCP2sRzfHH/lXlwPhsm5MEtbf5tDKg==
      |_-----END CERTIFICATE-----
      | http-methods: 
      |_  Supported Methods: POST
      | tls-alpn: 
      |   h2
      |   spdy/3
      |   spdy/2
      |_  http/1.1
      7601/tcp open  http          syn-ack Apache httpd 2.4.38 ((Debian))
      | http-methods: 
      |_  Supported Methods: POST OPTIONS HEAD GET
      |_http-title: Seppuku
      |_http-server-header: Apache/2.4.38 (Debian)
      Service Info: Host: SEPPUKU; OSs: Unix, Linux; CPE: cpe:/o:linux:linux_kernel
      
      Host script results:
      |_clock-skew: mean: 1h19m39s, deviation: 2h18m35s, median: -21s
      | smb-security-mode: 
      |   account_used: guest
      |   authentication_level: user
      |   challenge_response: supported
      |_  message_signing: disabled (dangerous, but default)
      | p2p-conficker: 
      |   Checking for Conficker.C or higher...
      |   Check 1 (port 53308/tcp): CLEAN (Couldn't connect)
      |   Check 2 (port 10822/tcp): CLEAN (Couldn't connect)
      |   Check 3 (port 30555/udp): CLEAN (Failed to receive data)
      |   Check 4 (port 35837/udp): CLEAN (Failed to receive data)
      |_  0/4 checks are positive: Host is CLEAN or ports are blocked
      | nbstat: NetBIOS name: SEPPUKU, NetBIOS user: <unknown>, NetBIOS MAC: <unknown> (unknown)
      | Names:
      |   SEPPUKU<00>          Flags: <unique><active>
      |   SEPPUKU<03>          Flags: <unique><active>
      |   SEPPUKU<20>          Flags: <unique><active>
      |   \x01\x02__MSBROWSE__\x02<01>  Flags: <group><active>
      |   WORKGROUP<00>        Flags: <group><active>
      |   WORKGROUP<1d>        Flags: <unique><active>
      |   WORKGROUP<1e>        Flags: <group><active>
      | Statistics:
      |   00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00
      |   00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00
      |_  00:00:00:00:00:00:00:00:00:00:00:00:00:00
      | smb-os-discovery: 
      |   OS: Windows 6.1 (Samba 4.9.5-Debian)
      |   NetBIOS computer name: SEPPUKU\x00
      |   Workgroup: WORKGROUP\x00
      |_  System time: 2024-09-11T06:23:02-04:00
      | smb2-time: 
      |   date: 2024-09-11T10:23:12
      |_  start_date: N/A
      | smb2-security-mode: 
      |   3:1:1: 
      |_    Message signing enabled but not required

- FFUF's output for the http server on port `7601`

![image](https://github.com/user-attachments/assets/cbd48e1b-261f-47da-9bfc-baa38db68610)

- I checked route `secret` and discovered a `shadow` file containing a hash, wordlist and a passwd file.

![image](https://github.com/user-attachments/assets/3487f946-f2cb-42a1-984e-424c539873ba)

- Route `keys` contains a ssh private key but I could not find the username.

![image](https://github.com/user-attachments/assets/d4aca6e7-e9ba-4e7b-ad66-ad61857594d3)

- Then, I tried to bruteforce ssh with the wordlist I got at first using the username `seppuku` and I got a password.

![image](https://github.com/user-attachments/assets/657be298-6238-480e-a9c3-c41d49bcd2c6)

### User Seppuku

- Ssh access as user `seppuku`

![image](https://github.com/user-attachments/assets/7f7698a1-ae08-44e1-bdc7-2b0ef701c826)

- The shell spawned was a restricted bash shell, so I spawned a new bash shell with bash

![image](https://github.com/user-attachments/assets/4352ce7e-5669-4b99-866b-b578db5f1739)

- I discovered a password file which worked for another user `samurai`.

![image](https://github.com/user-attachments/assets/4e1f8509-fcc1-4d9a-a4d9-34590bfa2594)

- User `samurai` access

![image](https://github.com/user-attachments/assets/4e2f0a73-16e5-4f0e-a792-b2cc11a7b2de)

-----------------------

### Privesc with user tanto and `samurai`

- After running `sudo -l` on user `samurai`, I discovered that user `samurai` can run a command as root but we need to access user tanto to create a binary `bin` in a directory `.cgi-bin`.

![image](https://github.com/user-attachments/assets/8515cec3-db10-44a8-8ab5-fa5d5c5bfe9a)

- I was able to login to user tanto with the aid of the private key I downloaded from the `keys` directory.

![image](https://github.com/user-attachments/assets/dbcf0ede-a9e6-4186-9113-57fb79ee8a6c)

- Then, I create the `/.cgi_bin/bin` file and added this python payload

![image](https://github.com/user-attachments/assets/c980c41e-5ddf-40d3-8ace-f60719082ba9)

- Now we can trigger the payload as user `samurai`.

![image](https://github.com/user-attachments/assets/963cbce7-a877-45e1-8c78-e19cb3cb6e9a)

- Root access...!!!!

![image](https://github.com/user-attachments/assets/b9038f46-e83c-4df1-83e6-d33e84db501e)


-------------------------

### THANKS FOR READING!!!!

--------------------------











