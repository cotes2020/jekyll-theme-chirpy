------------

### CTF-: TRYHACKME
### LAB-: Include

------------

![image](https://github.com/user-attachments/assets/8da533d0-a890-4d13-9bdf-c6a0ee177516)

------------

- Rustscan's output

```bash
❯ rustscan -a 10.10.95.69 -- -Pn -sC -sV
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
Open 10.10.95.69:22
Open 10.10.95.69:25
Open 10.10.95.69:110
Open 10.10.95.69:143
Open 10.10.95.69:993
Open 10.10.95.69:995
Open 10.10.95.69:4000
Open 10.10.95.69:50000
[~] Starting Script(s)
[>] Script to be run Some("nmap -vvv -p {{port}} {{ip}}")

Host discovery disabled (-Pn). All addresses will be marked 'up' and scan times may be slower.
[~] Starting Nmap 7.94SVN ( https://nmap.org ) at 2024-11-25 02:28 WAT
NSE: Loaded 156 scripts for scanning.
NSE: Script Pre-scanning.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 02:28
Completed NSE at 02:28, 0.00s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 02:28
Completed NSE at 02:28, 0.02s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 02:28
Completed NSE at 02:28, 0.01s elapsed
Initiating Parallel DNS resolution of 1 host. at 02:28
Completed Parallel DNS resolution of 1 host. at 02:28, 0.02s elapsed
DNS resolution of 1 IPs took 0.03s. Mode: Async [#: 1, OK: 0, NX: 1, DR: 0, SF: 0, TR: 1, CN: 0]
Initiating Connect Scan at 02:28
Scanning 10.10.95.69 [8 ports]
Discovered open port 143/tcp on 10.10.95.69
Discovered open port 22/tcp on 10.10.95.69
Discovered open port 110/tcp on 10.10.95.69
Discovered open port 993/tcp on 10.10.95.69
Discovered open port 25/tcp on 10.10.95.69
Discovered open port 995/tcp on 10.10.95.69
Discovered open port 4000/tcp on 10.10.95.69
Discovered open port 50000/tcp on 10.10.95.69
Completed Connect Scan at 02:28, 0.18s elapsed (8 total ports)
Initiating Service scan at 02:28
Scanning 8 services on 10.10.95.69
Completed Service scan at 02:29, 13.66s elapsed (8 services on 1 host)
NSE: Script scanning 10.10.95.69.
NSE: Starting runlevel 1 (of 3) scan.
Initiating NSE at 02:29
Completed NSE at 02:29, 22.18s elapsed
NSE: Starting runlevel 2 (of 3) scan.
Initiating NSE at 02:29
Completed NSE at 02:29, 4.32s elapsed
NSE: Starting runlevel 3 (of 3) scan.
Initiating NSE at 02:29
Completed NSE at 02:29, 0.00s elapsed
Nmap scan report for 10.10.95.69
Host is up, received user-set (0.18s latency).
Scanned at 2024-11-25 02:28:51 WAT for 41s

PORT      STATE SERVICE  REASON  VERSION
22/tcp    open  ssh      syn-ack OpenSSH 8.2p1 Ubuntu 4ubuntu0.11 (Ubuntu Linux; protocol 2.0)
| ssh-hostkey: 
|   3072 8e:0e:a6:ba:7c:14:09:64:a6:6e:74:af:ba:36:5d:88 (RSA)
| ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDlJ4DGMMAGGZxFpyZcPbN+COCqbIlfsYf23wtRzj/KMzX/6+Aj+4nTQdAjVIUa1OT4JV55r0+OlSE2T/FgWVwQvogrcfulp9GtI+pdxOVMKLl6Mbw0hcYNNT1XmKx79Z5liL+foIY1lU1/0S8P4bLLpSWDEpuu0ZrnheNVx8bTlHRUhkD8+e8rqb8pk2nhDm4YyBHNtMLe+TiS7uhrqwlS8ua+AZu6FGj7rXKG2Gioq44OaAEFLRbJiR0Ocvf90zyJVa9s3+3cKPQMlZpBM1uT94RCvh6qi1SZvF2kQLCRvqo6juW+yOJ3+w6IeeXcUtJ11vrbrQfvmHVZuWYlG0yonizmgsf6Fjr6kCa9DIiRSwEVBitOTQTsgRQw128SBDzK6/DHn8XrOnv0++7xsdjkL8GXfJNCwPIyqoL7x5IVWCMx+dZtNkxrZzrTuiXWvGrcvjUDZwwg4j42RSizsaejWauNp1TzoggR+Ez7rBOKMjqsji2hH13lYuq+SQDiSn8=
|   256 fd:3c:11:3b:65:d3:ea:ac:83:8b:f6:1f:71:03:e9:56 (ECDSA)
| ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBH/vtCP9ghxkO/i1dUNqeUDXOrtaOHm7P7/uhJAtOHxJ6vX7BN9BdsLNvff2vA1gc5j/UwFtpV9EguUwlrQgr58=
|   256 79:92:22:db:69:be:ff:d1:fa:79:5a:df:b7:ac:40:a6 (ED25519)
|_ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIO5KtQn3k7uRwQIiMRZr5R9WSkkc1u0AFndj5cX2pMnX
25/tcp    open  smtp     syn-ack Postfix smtpd
| ssl-cert: Subject: commonName=ip-10-10-31-82.eu-west-1.compute.internal
| Subject Alternative Name: DNS:ip-10-10-31-82.eu-west-1.compute.internal
| Issuer: commonName=ip-10-10-31-82.eu-west-1.compute.internal
| Public Key type: rsa
| Public Key bits: 2048
| Signature Algorithm: sha256WithRSAEncryption
| Not valid before: 2021-11-10T16:53:34
| Not valid after:  2031-11-08T16:53:34
| MD5:   05c8:4559:9811:a54f:9c53:b3ee:f6ad:f0fd
| SHA-1: a24d:7a7f:9ac1:8045:5c5f:5b7c:721a:4e21:0599:ed7c
| -----BEGIN CERTIFICATE-----
| MIIDOTCCAiGgAwIBAgIUZOpVp2fjesLBhoJHfQXOvrRFh2AwDQYJKoZIhvcNAQEL
| BQAwNDEyMDAGA1UEAwwpaXAtMTAtMTAtMzEtODIuZXUtd2VzdC0xLmNvbXB1dGUu
| aW50ZXJuYWwwHhcNMjExMTEwMTY1MzM0WhcNMzExMTA4MTY1MzM0WjA0MTIwMAYD
| VQQDDClpcC0xMC0xMC0zMS04Mi5ldS13ZXN0LTEuY29tcHV0ZS5pbnRlcm5hbDCC
| ASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBANCGzRe8Ucyrg1ICrmylNf/G
| Dhe8gGUJsnSdBBwzEhofznOXvjGJ/P+5/ScXSNm5Bb632xtPcZ3wSq9xHq8JqZMu
| oXjoyd4U6VK6aV4xjxlwdE33DgsAHXORv9PkMi+NeYDFsJrdRznSV64mc9xhIqSk
| WdnALkBXvNZcTwy6feITP3F4YTGa5ewRNJSVutU4hBY1CfroZxRnff6kkbF0iqQc
| dSHPjK3NeAZnp4iVID8rBuV/fjjOtZ53z1u6cXmQVc2fljvD4GN3TxV4MKbazqOb
| +kEYdT5MiBEIJjQddhagbMWDYPF7McDSS/I3y4KdL1mI40Fjr6sXKOetrFvRZ+cC
| AwEAAaNDMEEwCQYDVR0TBAIwADA0BgNVHREELTArgilpcC0xMC0xMC0zMS04Mi5l
| dS13ZXN0LTEuY29tcHV0ZS5pbnRlcm5hbDANBgkqhkiG9w0BAQsFAAOCAQEArHg4
| zvCqUMzbSvusDU3d4cPDYnh7a7fAdOeVxHWo8/z/gzB8/ojJ8oYtfDV3qdKRhg0m
| pGSG3A2MZvl9u4FYj2tI8sne5HNTGRNg+3DLR/O9lFR90TH4v4piyAJrc29nFmpe
| Mq8I+JOizeSVG9qMSp6s0hDcHGAs111avS5TkEUvL0GybJIIQabOMDJ1e+Mptca+
| iV+Z+rdfirNzw87twkMxEpwTVPf3h5G0EKwE62Ih8cG1Pk/NrZCz5lN5P2b2BwHQ
| wbmbTgiA+hBmWajlHVu7EwEIsnMGrzTgSacVhHd7WsThLlMQwgNIowzUMagIA0yD
| s6SpR/+RIiQzeFiuTw==
|_-----END CERTIFICATE-----
|_smtp-commands: mail.filepath.lab, PIPELINING, SIZE 10240000, VRFY, ETRN, STARTTLS, ENHANCEDSTATUSCODES, 8BITMIME, DSN, SMTPUTF8, CHUNKING
|_ssl-date: TLS randomness does not represent time
110/tcp   open  pop3     syn-ack Dovecot pop3d
|_ssl-date: TLS randomness does not represent time
| ssl-cert: Subject: commonName=ip-10-10-31-82.eu-west-1.compute.internal
| Subject Alternative Name: DNS:ip-10-10-31-82.eu-west-1.compute.internal
| Issuer: commonName=ip-10-10-31-82.eu-west-1.compute.internal
| Public Key type: rsa
| Public Key bits: 2048
| Signature Algorithm: sha256WithRSAEncryption
| Not valid before: 2021-11-10T16:53:34
| Not valid after:  2031-11-08T16:53:34
| MD5:   05c8:4559:9811:a54f:9c53:b3ee:f6ad:f0fd
| SHA-1: a24d:7a7f:9ac1:8045:5c5f:5b7c:721a:4e21:0599:ed7c
| -----BEGIN CERTIFICATE-----
| MIIDOTCCAiGgAwIBAgIUZOpVp2fjesLBhoJHfQXOvrRFh2AwDQYJKoZIhvcNAQEL
| BQAwNDEyMDAGA1UEAwwpaXAtMTAtMTAtMzEtODIuZXUtd2VzdC0xLmNvbXB1dGUu
| aW50ZXJuYWwwHhcNMjExMTEwMTY1MzM0WhcNMzExMTA4MTY1MzM0WjA0MTIwMAYD
| VQQDDClpcC0xMC0xMC0zMS04Mi5ldS13ZXN0LTEuY29tcHV0ZS5pbnRlcm5hbDCC
| ASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBANCGzRe8Ucyrg1ICrmylNf/G
| Dhe8gGUJsnSdBBwzEhofznOXvjGJ/P+5/ScXSNm5Bb632xtPcZ3wSq9xHq8JqZMu
| oXjoyd4U6VK6aV4xjxlwdE33DgsAHXORv9PkMi+NeYDFsJrdRznSV64mc9xhIqSk
| WdnALkBXvNZcTwy6feITP3F4YTGa5ewRNJSVutU4hBY1CfroZxRnff6kkbF0iqQc
| dSHPjK3NeAZnp4iVID8rBuV/fjjOtZ53z1u6cXmQVc2fljvD4GN3TxV4MKbazqOb
| +kEYdT5MiBEIJjQddhagbMWDYPF7McDSS/I3y4KdL1mI40Fjr6sXKOetrFvRZ+cC
| AwEAAaNDMEEwCQYDVR0TBAIwADA0BgNVHREELTArgilpcC0xMC0xMC0zMS04Mi5l
| dS13ZXN0LTEuY29tcHV0ZS5pbnRlcm5hbDANBgkqhkiG9w0BAQsFAAOCAQEArHg4
| zvCqUMzbSvusDU3d4cPDYnh7a7fAdOeVxHWo8/z/gzB8/ojJ8oYtfDV3qdKRhg0m
| pGSG3A2MZvl9u4FYj2tI8sne5HNTGRNg+3DLR/O9lFR90TH4v4piyAJrc29nFmpe
| Mq8I+JOizeSVG9qMSp6s0hDcHGAs111avS5TkEUvL0GybJIIQabOMDJ1e+Mptca+
| iV+Z+rdfirNzw87twkMxEpwTVPf3h5G0EKwE62Ih8cG1Pk/NrZCz5lN5P2b2BwHQ
| wbmbTgiA+hBmWajlHVu7EwEIsnMGrzTgSacVhHd7WsThLlMQwgNIowzUMagIA0yD
| s6SpR/+RIiQzeFiuTw==
|_-----END CERTIFICATE-----
|_pop3-capabilities: RESP-CODES SASL STLS UIDL CAPA AUTH-RESP-CODE PIPELINING TOP
143/tcp   open  imap     syn-ack Dovecot imapd (Ubuntu)
|_imap-capabilities: ID have LITERAL+ post-login listed capabilities Pre-login IMAP4rev1 ENABLE SASL-IR OK LOGINDISABLEDA0001 more STARTTLS LOGIN-REFERRALS IDLE
| ssl-cert: Subject: commonName=ip-10-10-31-82.eu-west-1.compute.internal
| Subject Alternative Name: DNS:ip-10-10-31-82.eu-west-1.compute.internal
| Issuer: commonName=ip-10-10-31-82.eu-west-1.compute.internal
| Public Key type: rsa
| Public Key bits: 2048
| Signature Algorithm: sha256WithRSAEncryption
| Not valid before: 2021-11-10T16:53:34
| Not valid after:  2031-11-08T16:53:34
| MD5:   05c8:4559:9811:a54f:9c53:b3ee:f6ad:f0fd
| SHA-1: a24d:7a7f:9ac1:8045:5c5f:5b7c:721a:4e21:0599:ed7c
| -----BEGIN CERTIFICATE-----
| MIIDOTCCAiGgAwIBAgIUZOpVp2fjesLBhoJHfQXOvrRFh2AwDQYJKoZIhvcNAQEL
| BQAwNDEyMDAGA1UEAwwpaXAtMTAtMTAtMzEtODIuZXUtd2VzdC0xLmNvbXB1dGUu
| aW50ZXJuYWwwHhcNMjExMTEwMTY1MzM0WhcNMzExMTA4MTY1MzM0WjA0MTIwMAYD
| VQQDDClpcC0xMC0xMC0zMS04Mi5ldS13ZXN0LTEuY29tcHV0ZS5pbnRlcm5hbDCC
| ASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBANCGzRe8Ucyrg1ICrmylNf/G
| Dhe8gGUJsnSdBBwzEhofznOXvjGJ/P+5/ScXSNm5Bb632xtPcZ3wSq9xHq8JqZMu
| oXjoyd4U6VK6aV4xjxlwdE33DgsAHXORv9PkMi+NeYDFsJrdRznSV64mc9xhIqSk
| WdnALkBXvNZcTwy6feITP3F4YTGa5ewRNJSVutU4hBY1CfroZxRnff6kkbF0iqQc
| dSHPjK3NeAZnp4iVID8rBuV/fjjOtZ53z1u6cXmQVc2fljvD4GN3TxV4MKbazqOb
| +kEYdT5MiBEIJjQddhagbMWDYPF7McDSS/I3y4KdL1mI40Fjr6sXKOetrFvRZ+cC
| AwEAAaNDMEEwCQYDVR0TBAIwADA0BgNVHREELTArgilpcC0xMC0xMC0zMS04Mi5l
| dS13ZXN0LTEuY29tcHV0ZS5pbnRlcm5hbDANBgkqhkiG9w0BAQsFAAOCAQEArHg4
| zvCqUMzbSvusDU3d4cPDYnh7a7fAdOeVxHWo8/z/gzB8/ojJ8oYtfDV3qdKRhg0m
| pGSG3A2MZvl9u4FYj2tI8sne5HNTGRNg+3DLR/O9lFR90TH4v4piyAJrc29nFmpe
| Mq8I+JOizeSVG9qMSp6s0hDcHGAs111avS5TkEUvL0GybJIIQabOMDJ1e+Mptca+
| iV+Z+rdfirNzw87twkMxEpwTVPf3h5G0EKwE62Ih8cG1Pk/NrZCz5lN5P2b2BwHQ
| wbmbTgiA+hBmWajlHVu7EwEIsnMGrzTgSacVhHd7WsThLlMQwgNIowzUMagIA0yD
| s6SpR/+RIiQzeFiuTw==
|_-----END CERTIFICATE-----
|_ssl-date: TLS randomness does not represent time
993/tcp   open  ssl/imap syn-ack Dovecot imapd (Ubuntu)
| ssl-cert: Subject: commonName=ip-10-10-31-82.eu-west-1.compute.internal
| Subject Alternative Name: DNS:ip-10-10-31-82.eu-west-1.compute.internal
| Issuer: commonName=ip-10-10-31-82.eu-west-1.compute.internal
| Public Key type: rsa
| Public Key bits: 2048
| Signature Algorithm: sha256WithRSAEncryption
| Not valid before: 2021-11-10T16:53:34
| Not valid after:  2031-11-08T16:53:34
| MD5:   05c8:4559:9811:a54f:9c53:b3ee:f6ad:f0fd
| SHA-1: a24d:7a7f:9ac1:8045:5c5f:5b7c:721a:4e21:0599:ed7c
| -----BEGIN CERTIFICATE-----
| MIIDOTCCAiGgAwIBAgIUZOpVp2fjesLBhoJHfQXOvrRFh2AwDQYJKoZIhvcNAQEL
| BQAwNDEyMDAGA1UEAwwpaXAtMTAtMTAtMzEtODIuZXUtd2VzdC0xLmNvbXB1dGUu
| aW50ZXJuYWwwHhcNMjExMTEwMTY1MzM0WhcNMzExMTA4MTY1MzM0WjA0MTIwMAYD
| VQQDDClpcC0xMC0xMC0zMS04Mi5ldS13ZXN0LTEuY29tcHV0ZS5pbnRlcm5hbDCC
| ASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBANCGzRe8Ucyrg1ICrmylNf/G
| Dhe8gGUJsnSdBBwzEhofznOXvjGJ/P+5/ScXSNm5Bb632xtPcZ3wSq9xHq8JqZMu
| oXjoyd4U6VK6aV4xjxlwdE33DgsAHXORv9PkMi+NeYDFsJrdRznSV64mc9xhIqSk
| WdnALkBXvNZcTwy6feITP3F4YTGa5ewRNJSVutU4hBY1CfroZxRnff6kkbF0iqQc
| dSHPjK3NeAZnp4iVID8rBuV/fjjOtZ53z1u6cXmQVc2fljvD4GN3TxV4MKbazqOb
| +kEYdT5MiBEIJjQddhagbMWDYPF7McDSS/I3y4KdL1mI40Fjr6sXKOetrFvRZ+cC
| AwEAAaNDMEEwCQYDVR0TBAIwADA0BgNVHREELTArgilpcC0xMC0xMC0zMS04Mi5l
| dS13ZXN0LTEuY29tcHV0ZS5pbnRlcm5hbDANBgkqhkiG9w0BAQsFAAOCAQEArHg4
| zvCqUMzbSvusDU3d4cPDYnh7a7fAdOeVxHWo8/z/gzB8/ojJ8oYtfDV3qdKRhg0m
| pGSG3A2MZvl9u4FYj2tI8sne5HNTGRNg+3DLR/O9lFR90TH4v4piyAJrc29nFmpe
| Mq8I+JOizeSVG9qMSp6s0hDcHGAs111avS5TkEUvL0GybJIIQabOMDJ1e+Mptca+
| iV+Z+rdfirNzw87twkMxEpwTVPf3h5G0EKwE62Ih8cG1Pk/NrZCz5lN5P2b2BwHQ
| wbmbTgiA+hBmWajlHVu7EwEIsnMGrzTgSacVhHd7WsThLlMQwgNIowzUMagIA0yD
| s6SpR/+RIiQzeFiuTw==
|_-----END CERTIFICATE-----
|_ssl-date: TLS randomness does not represent time
|_imap-capabilities: more have LITERAL+ post-login listed capabilities Pre-login IMAP4rev1 ENABLE SASL-IR OK ID AUTH=PLAIN LOGIN-REFERRALS AUTH=LOGINA0001 IDLE
995/tcp   open  ssl/pop3 syn-ack Dovecot pop3d
|_pop3-capabilities: RESP-CODES USER SASL(PLAIN LOGIN) UIDL CAPA AUTH-RESP-CODE PIPELINING TOP
|_ssl-date: TLS randomness does not represent time
| ssl-cert: Subject: commonName=ip-10-10-31-82.eu-west-1.compute.internal
| Subject Alternative Name: DNS:ip-10-10-31-82.eu-west-1.compute.internal
| Issuer: commonName=ip-10-10-31-82.eu-west-1.compute.internal
| Public Key type: rsa
| Public Key bits: 2048
| Signature Algorithm: sha256WithRSAEncryption
| Not valid before: 2021-11-10T16:53:34
| Not valid after:  2031-11-08T16:53:34
| MD5:   05c8:4559:9811:a54f:9c53:b3ee:f6ad:f0fd
| SHA-1: a24d:7a7f:9ac1:8045:5c5f:5b7c:721a:4e21:0599:ed7c
| -----BEGIN CERTIFICATE-----
| MIIDOTCCAiGgAwIBAgIUZOpVp2fjesLBhoJHfQXOvrRFh2AwDQYJKoZIhvcNAQEL
| BQAwNDEyMDAGA1UEAwwpaXAtMTAtMTAtMzEtODIuZXUtd2VzdC0xLmNvbXB1dGUu
| aW50ZXJuYWwwHhcNMjExMTEwMTY1MzM0WhcNMzExMTA4MTY1MzM0WjA0MTIwMAYD
| VQQDDClpcC0xMC0xMC0zMS04Mi5ldS13ZXN0LTEuY29tcHV0ZS5pbnRlcm5hbDCC
| ASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBANCGzRe8Ucyrg1ICrmylNf/G
| Dhe8gGUJsnSdBBwzEhofznOXvjGJ/P+5/ScXSNm5Bb632xtPcZ3wSq9xHq8JqZMu
| oXjoyd4U6VK6aV4xjxlwdE33DgsAHXORv9PkMi+NeYDFsJrdRznSV64mc9xhIqSk
| WdnALkBXvNZcTwy6feITP3F4YTGa5ewRNJSVutU4hBY1CfroZxRnff6kkbF0iqQc
| dSHPjK3NeAZnp4iVID8rBuV/fjjOtZ53z1u6cXmQVc2fljvD4GN3TxV4MKbazqOb
| +kEYdT5MiBEIJjQddhagbMWDYPF7McDSS/I3y4KdL1mI40Fjr6sXKOetrFvRZ+cC
| AwEAAaNDMEEwCQYDVR0TBAIwADA0BgNVHREELTArgilpcC0xMC0xMC0zMS04Mi5l
| dS13ZXN0LTEuY29tcHV0ZS5pbnRlcm5hbDANBgkqhkiG9w0BAQsFAAOCAQEArHg4
| zvCqUMzbSvusDU3d4cPDYnh7a7fAdOeVxHWo8/z/gzB8/ojJ8oYtfDV3qdKRhg0m
| pGSG3A2MZvl9u4FYj2tI8sne5HNTGRNg+3DLR/O9lFR90TH4v4piyAJrc29nFmpe
| Mq8I+JOizeSVG9qMSp6s0hDcHGAs111avS5TkEUvL0GybJIIQabOMDJ1e+Mptca+
| iV+Z+rdfirNzw87twkMxEpwTVPf3h5G0EKwE62Ih8cG1Pk/NrZCz5lN5P2b2BwHQ
| wbmbTgiA+hBmWajlHVu7EwEIsnMGrzTgSacVhHd7WsThLlMQwgNIowzUMagIA0yD
| s6SpR/+RIiQzeFiuTw==
|_-----END CERTIFICATE-----
4000/tcp  open  http     syn-ack Node.js (Express middleware)
|_http-title: Sign In
| http-methods: 
|_  Supported Methods: GET HEAD POST OPTIONS
50000/tcp open  http     syn-ack Apache httpd 2.4.41 ((Ubuntu))
|_http-title: System Monitoring Portal
| http-cookie-flags: 
|   /: 
|     PHPSESSID: 
|_      httponly flag not set
| http-methods: 
|_  Supported Methods: GET HEAD POST OPTIONS
|_http-server-header: Apache/2.4.41 (Ubuntu)
Service Info: Host:  mail.filepath.lab; OS: Linux; CPE: cpe:/o:linux:linux_kernel
```

- Port 4000 runs a nodejs server.Luckily for me,defualt creds `guest:guest` works as shown on the page.

![image](https://github.com/user-attachments/assets/ffa00f45-bc59-40c4-a23c-41f3fa0ca1bc)

- I checked the view profile tab and notice an isAdmin value set to `false` and set it to `true` since in case admin access is required for other functionalities.That functionality that allows us to add json keys and values is vulnerable to `broken access control`.

![image](https://github.com/user-attachments/assets/ca929718-76b1-4af4-bd23-74d454299b7b)

- New tabs got unlocked in the navbar

![image](https://github.com/user-attachments/assets/7881ccf9-ba04-4687-99e0-f6ee01b10f80)

- I checked the `api` tab and noticed a new api running on port `5000` which reveals admin users credentials.We need a vulnerability like `Server Side Request Forgery` to communicate to an internal port.

![image](https://github.com/user-attachments/assets/fda30f29-75c0-4020-bd11-f25b78bfa4c3)

- I checked the `settings` endpoint,it allows user to paste url pointing to an image so that it can be downloaded and set as profile image for a user.We can use this endpoint to load the internal port and communicate with internal services.

![image](https://github.com/user-attachments/assets/1fadb7be-2e02-48fa-a555-a2d56f29349d)

- Credentials

![image](https://github.com/user-attachments/assets/dff686bf-86ac-4b87-9359-eb657008c362)

- The sysmon credentials belongs to the admin server for port `50000` which is an http server.Admin access

![image](https://github.com/user-attachments/assets/7b9eb1d5-1902-47b9-a7d9-2e46eef680fa)

- I viewed the source code and noticed a file `profile.php` that reads a file via the `img` parameter.

![image](https://github.com/user-attachments/assets/eaf3991b-bfcc-44d6-8052-832ca6f71c0c)

- I tried the basic attacks `../../` to read `/etc/passwd` but I wasn't getting any positive results.I fuzzed with Jhaddix's LFI wordlist and got a hit.

![image](https://github.com/user-attachments/assets/217d6cf0-46ab-4352-a5f7-437dc903da95)

### LFI2RCE

- I need to elevate the LFI to RCE since it is a php web app which is possible either through php filters or log files that we have write access. Although, the latter worked.

- I was able to read the `/var/log/mail.log` which is the log file for the smtp service running on port `25`.

![image](https://github.com/user-attachments/assets/a6c7ce05-23f1-42f6-bfa5-8499357a2d78)

- I wrote a php shell into the log by trying  to send a mail as seen below.

![image](https://github.com/user-attachments/assets/e828aa79-9b7a-420d-9bce-2e4871530ab1)

- RCE achieved,we can pass shell commands with the `c` get parameter.

![image](https://github.com/user-attachments/assets/6f592267-ff93-499a-82f3-00cd5d707f95)

- Final flag-:

![image](https://github.com/user-attachments/assets/a9d8a72a-ec5c-4550-9a8e-e59f735d92dc)

------------------

### REFERENCES-:

- [LFI2RCE](https://github.com/RoqueNight/LFI---RCE-Cheat-Sheet)
- [Jhaddix's list](https://github.com/danielmiessler/SecLists/blob/master/Fuzzing/LFI/LFI-Jhaddix.txt)

------------------













