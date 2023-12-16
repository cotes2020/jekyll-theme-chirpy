---
layout: post
title: Cyber Security knowledge point
date: 2020-08-25 11:11:11 -0400
description: Cyber Security Interview Questiones
categories: [Note]
img: /assets/img/sample/rabbit.png
tags: [Interview]
image: /assets/img/sample/OSILayer.jpg
---

# Cyber Security knowledge point

- [Cyber Security knowledge point](#cyber-security-knowledge-point)
  - [security field](#security-field)
- [Network security](#network-security)
    - [firewall](#firewall)
    - [traceroute](#traceroute)
  - [pentest](#pentest)
  - [wireless](#wireless)
- [Application security](#application-security)
  - [Software Reverse Engineering](#software-reverse-engineering)
  - [Web APP](#web-app)
  - [Web Security](#web-security)
  - [AAA](#aaa)
  - [Architect](#architect)
    - [TLS/SSL](#tlsssl)
  - [Forensics](#forensics)
- [Security architect](#security-architect)
    - [IPS/IDS](#ipsids)
  - [hack](#hack)
    - [XSS](#xss)
- [Risk management](#risk-management)
- [Security audits, testing & incident response](#security-audits-testing--incident-response)
- [Cryptography](#cryptography)
    - [hashing](#hashing)
- [knowledge point](#knowledge-point)
  - [TCP/UDP](#tcpudp)
  - [API](#api)
  - [Docker Basic Questions](#docker-basic-questions)

---



1. What is information security and how is it achieved?

2. What are the core principles of information security?
   1. CIA

3. What is non-repudiation (as it applies to IT security)?
4. What is the relationship between information security and data availability?
5. What is a security policy and why do we need one?
6. What is the difference between logical and physical security? Can you give an example of both?
7. What’s an acceptable level of risk?
8. What are the most common types of attacks that threaten enterprise data security?
9. What is the difference between a threat and a vulnerability?
10. Can you give me an example of common security vulnerabilities?
11. Are you familiar with any security management frameworks such as ISO/IEC 27002?
12. What is a security control?
13. What are the different types of security control?
14. Can you describe the information lifecycle? How do you ensure information security at each phase?
15. What is Information Security Governance?
16. What are your professional values? Why are professional ethics important in the information

17. What is the difference between threat, vulnerability, and a risk?
   - `Threat`, a process that magnifies the likelihood of a negative event, such as the exploitation of a vulnerability, is from an attacker that will use a vulnerability that was not mitigated because someone forgot to identify it as a risk.
   - `Vulnerability`, a weakness in your infrastructure, networks, or applications that potentially exposes you to threats, is a gap in the protection efforts of a system, a threat is an attacker who exploits that weakness.
   - `Risk`, potential for loss, damage, or destruction of assets or data caused by a cyber threat, is the measure of potential loss when that the vulnerability is exploited by the threat.
   - `Threats x Vulnerabilities = Risk`

18. difference between a vulnerability and an exploit?
   - vulnerability, a potential problem
   - exploit, an active problem.
   - Think of it like this: You have a shed with a broken lock where it won’t latch properly. In some areas such as major cities, that would be a major problem that needs to be resolved immediately, while in others like rural areas its more of a nuisance that can be fixed when you get around to it. In both scenarios it would be a vulnerability, while the major cities shed would be an example of an exploit         - there are people in the area, actively exploiting a known problem.

19. How would you find out what a POST code means?
    - POST is one of the best tools available when a system will not boot. Normally through the use of either display LEDs in more modern systems, or traditionally through audio tones, these specific codes can tell you what the system doesn’t like about its current setup. Because of how rare these events can be, unless you are on a tech bench day in and day out, reference materials such as the Motherboard manual and your search engine of choice can be tremendous assets. Just remember to make sure that everything is seated correctly, you have at least the minimum required components to boot, and most importantly that you have all of your connections on the correct pins.

1. TTPs
   - TTPs stands for tactics, techniques, and procedures.
   - describe the behaviors, processes, actions, and strategies used by a threat actor to develop threats and engage in cyberattacks.


---

## security field

1. Are open-source projects more or less secure than proprietary ones?
2. Who do you look up to within the field of Information Security? Why?
3. Where do you get your security news from?
4. What’s the difference between symmetric and public-key cryptography?
5. What kind of network do you have at home?
6. What are the advantages offered by bug bounty programs over normal testing practices?
7. What are your first three steps when securing a Linux server?
8. What are your first three steps when securing a Windows server?
9. Who’s more dangerous to an organization, insiders or outsiders?
10. Why is DNS monitoring important?

11. Why would you want to use SSH from a Windows PC?
12. How would you find out what a POST code means?
13. What is the difference between a black hat and a white hat?
14. What do you think of social networking sites such as Facebook and LinkedIn?
15. Why are internal threats often more successful than external threats?
16. Why is deleted data not truly gone when you delete it?
17. What is the Chain of Custody?
18. How would you permanently remove the threat of data falling into the wrong hands?
19. What is exfiltration?
20. How do you protect your home wireless access point?
21. If you were going to break into a database-based website, how would you do it?
22. What is the CIA triangle?
23. What is the difference between information protection and information assurance?
24. How would you lock down a mobile device?
25. What is the difference between closed-source and open-source? Which is better?
26. What is your opinion on hacktivist groups such as Anonymous?

---

# Network security

44. Explain what **Address Resolution Protocol** is.
   1. ![v2-cfdda1ceb830edd5a8d28ae31c6ac8f6_hd](/assets/img/sample/arp.jpg)
   2. Data link-layer protocol
   3. resolve IP addresses to MAC addresses
   4. broadcasting requests that queries all the network interfaces on a local-area network, and caching responses for future use
   1. ARP request
   2. ARP reply
   3. vulnerability with ARP:
   - It will believe any ARP reply packet
   - nothing to authenticate the request
   - Attackers can easily create ARP reply packets with spoofed or bogus MAC addresses, reply and poison the ARP cache on systems in the network. Gratuitous ARP

45. What port does **ping** work over?
   1. It doesn’t work over a port. no real ports being used.
   2. ping test uses `ICMP` layer 3 protocol.
   3. `ICMP` basically roofs, or sits on top of, the IP address. not a layer four protocol.


46. Do you prefer **filtered ports or closed ports** on your firewall?
   1. For small company servers / back-end systems / intranet sites: close ports (REJECT).
   - The reason
   - because those server are not usually targeted by DDoS attacks
   - because the external apps that requires to consume services hosted in the the servers can quickly report failures instead to hang the connections during minutes.



6. What are Linux’s strengths and weaknesses vs. Windows?
   1. Price
   2. Ease Of Use
   3. Reliability. Linux is notoriously reliable and secure.
   4. Software
   5. Hardware
   6. Security

### firewall

1. What is a firewall? how a firewall can be bypassed?
   1. `PACKET-FILTERING FIREWALLS`
      1. examines the packet source and destination IP address
      2. either prohibits or allows them to pass based on the established security rule set.
   2. `Stateless`
   3. `Stateful`: remember information associated with previously passed packets and thus provide much better security.
   4. `NEXT-GENERATION FIREWALLS (NGFW)`
      1. traditional firewall features coupled with additional functionality
      - like anti-virus, intrusion prevention systems, encrypted traffic inspection, and deep packet inspection (DPI).
      1. Unlike basic firewalls which only check packet headers, DPI examines the data within the packet, thus allowing users to stop, identify, or categorize packets containing malicious data.
   5. `PROXY FIREWALLS`
      1. application level
      2. the client’s request is evaluated against security rules and based on these rules, it is permitted or blocked.
      3. use both stateful and deep packet inspection.
      4. They are mostly used for monitoring traffics for layer 7 protocols like HTTP and FTP.
   6. **bypass**:
      1. incognito window
      2. `HTTP tunneling` is a firewall evasion technique
      3. lots of things can be wrapped within an HTTP shell (Microsoft Office has been doing this for years).
   - because port 80 is almost never filtered by a firewall
   - can craft port 80 segments to carry payload for protocols the firewall may have otherwise blocked.
   - HTTP beacons and HTTP tunnels are the de facto standard implant technology for hackers.

2. What is worse in Firewall Detection, false negative / false positive?
   - false positive
     - the device generated an alert for an intrusion which has actually not happened
     - calling a legitimate piece of traffic bad.
   - false negative: **bad**
     - the device has not generated any alert and the intrusion has actually happened
     - a piece of malicious traffic being let through without incident

3. Besides firewalls, what other devices are used to enforce network boundaries?
   1. IDS/IPS, Procy, VPN, ACLs, subnetting, NAT, PAT,


4. What is the role of network boundaries in information security?

5. How does a router differ from a switch?
   - Switches create a network. Routers connect networks.


6. What does an intrusion detection system do? How does it do it?

7. What is a honeypot? What type of attack does it defend against?
   1. a server left open or appears to have been sloppily locked down, allowing an attacker relatively easy access.
   2. Divert attackers from the live network:
   3. diverts the attacker away from the live network.
   4. location of the honeypot is of utmost importance, more realistic placement is inside the DMZ.
   5. tool to gather intelligence on the attacker

8. What technologies and approaches are used to secure information and services deployed on cloud computing infrastructure?
9. What information security challenges are faced in a cloud computing environment?

10. an overview of IP multicast?
   1. delivers application source traffic to multiple receivers without burdening the source or the receivers while `using a minimum of network bandwidth`.
   2. multicast sender could send traffic destined for a Class D IP address, known as a multicast group,
   3. devices on a network wanting to receive that transmission could join that multicast group.
   4. send that traffic only to devices in a network wanting to receive that traffic.

11. How many bits do you need for a subnet size?
   5. 32 bits

12. What is packet filtering?

13. Can you explain the difference between a packet filtering firewall and an application layer firewall?


14. layers of the OSI model?
   6. Physical, Data Link, Network, Transport, Session, Presentation, and Application.


15. How would you login to Active Directory from a Linux or Mac box?
16. What is an easy way to configure a network to allow only a single computer to login on a particular jack?
17. What are the three ways to authenticate a person?
18. You find out that there is an active problem on your network. You can fix it, but it is out of your jurisdiction. What do you do?
19. How would you compromise an “office workstation” at a hotel?
20. What is worse in firewall detection, a false negative or a false positive? And why?
21. How would you judge if a remote server is running IIS or Apache?
22. What is the difference between an HIDS and a NIDS?


23. use SSH from a Windows pc?
   - SSH (TCP port 22) is a secure connection used on many different systems and dedicated appliances.
   - Routers, Switches, SFTP servers and insecure programs being tunnelled through this port all can be used to help harden a connection against eavesdropping.
   - the SSH protocol itself is implemented on a wide variety of systems         - Linux, Windows.
   - Programs like PuTTY, Filezilla and others have Windows ports available, which allow Windows users the same ease-of-use connectivity to these devices as do Linux users.

47. TTL?
   - Time To Live.
   - When a TCP packet is sent, its TTL is set, which is the number of routers (hops) it can pass through before the packet is discarded.
   - As the packet passes through a router the TTL is decremented until, when the TTL reaches zero, the packet is destroyed and an `ICMP “time exceeded” message is returned.`
   - The return message’s TTL is set by the terminating router when it creates the packet and decremented normally.
   - The TTL is an IP `header field`
   - to prevent packets from running into endless loops.


### traceroute

1. What protocol does traceroute use?
   - Traceroute works by `sending a packet to an open UDP port` on a destination machine.
   - The type of packet that is sent differs depending on the implementation. By default
   - Windows uses ICMP.
   - Mac OS X and Linux use UDP.
   - All versions of traceroute rely on `ICMP type 11 (Time exceeded) responses` from each hop along the route.
   - If ICMP type 11 responses are being blocked by your firewall, traceroute will not work.
   - These packets are inbound, not outbound.

2. How exactly does **traceroute/tracert** work?
   - client:
     - Traceroute transmits packets with small TTL values.
     - `setting the TTL for a packet to 1`
   - sending it towards the requested destination host, and listening for the reply.
     - When the initiating machine receives a “time exceeded” response, it examines the packet to determine where the packet came from
     - this identifies the machine one hop away.
   - The router
     - discards the packet
     - `sends off an ICMP notification packet to the original host` with the message that the TTL expired from the router.
   - Then the tracing machine `generates a new packet with TTL 2`, and uses the response to determine the machine 2 hops away, and so on.

   - **not all TCP stacks behave correctly**.
     - Some TCP stacks `set the TTL for the ICMP “time exceeded” message` to that of the message being killed.
       - So if the TTL is 0, the packet will be killed by the next machine to which it is passed.
     - This can have two effects on a trace.
       - If the computer is an intermediate machine in the trace,
     - the entry remain blank.
     - No information is returned because the “time exceeded” message never makes it back.
       - If the machine you are doing a trace to has this bug in its TCP stack,
     - return packets won’t reach the originating machine unless the TTL is high enough to cover the round trip.
     - So Trace Route will show a number of failed connections equal to n (the number of hops to the destination machine) minus 1.

3. How would traceroute help you find out where a breakdown in communication is?
   - to `see exactly what routers touched` as move along the chain of connections to final destination.
   - if end up with a problem where can’t connect/ping the destination,
   - a tracert can help to tell exactly `where` the chain of connections stop.
   - contact the correct people – your own firewall, ISP, destination’s ISP or somewhere in the middle.

4. How many packets must leave my NIC to complete a traceroute to twitter.com?
   - `they need to factor in all layers: Ethernet, IP, DNS, ICMP/UDP, etc. And they need to consider round-trip times`.
   - The top 3-4 levels of the OSI model are not used
     - (depending on the the operating system)
       - traceroute makes a request to the networking library
       - send either an ICMP (layer 3) or UDP (layer 4) packet to the destination
       - with a TTL of 1,
     - the response that is returned includes information presented to the user and the TTL is increased to 2 for the next packet and sent.
     - This proceeds until the destination is reached.
   - The packets themselves are constructed at the bit level for level 1, to individual frames in level 2, packed in a ‘packet’ in layer 3.
   - Then the remaining transformations depend on whether UDP is used or ICMP.

5. **configure trace route in a cisco firewall** for a group of windows users?
   - Windows uses ICMP for traceroute, Linux technically uses UDP.
   - Therefore, the responses they get from the devices along the path is different, depending on which source device you used to initiate the trace.
   - For Windows, need an `inbound rule allowing icmp time-exceeded`.
   - For Linux, need an `inbound rule allowing icmp unreachable`.
   - For both, need to add an "`inspect icmp`" statement.

---

## pentest


1. red team or a blue team?
   - the Blue Team has to be good every time, while the Red Team only has to be good once.

2. What’s the difference between a White Box test and a Black Box test?

3. difference between Information Protection and Information Assurance?
   - Information Protection: protecting information through the use of Encryption, Security software and other methods designed to keep it safe.
   - Information Assurance: keeping the data reliable – RAID configurations, backups, non-repudiation techniques, etc.


## wireless

1. protect Wireless Access Point?
   - Using WPA2,
   - stop broadcasting the SSID
   - prevent a man-in-the-middle attack?

Secure/Multipurpose Internet Mail Extensions: Encrypts the email in transit
Use HTTPS
Use VPMS/Proxy

using MAC address filtering


---

# Application security

68. Describe the last program or script that you wrote. What problem did it solve?

69. Can you briefly discuss the role of information security in each phase of the software development lifecycle?

70. How would you implement a secure login field on a high traffic website where performance is a consideration?

71. What are the various ways to handle account brute forcing?

72. What is cross-site request forgery?

73. How does one defend against CSRF?

74. If you were a site administrator looking for incoming CSRF attacks, what would you look for?
75. What’s the difference between HTTP and HTML?
76. How does HTTP handle state?
77. What exactly is cross-site scripting?
78. What’s the difference between stored and reflected XSS?
79. What are the common defenses against XSS?
80. You are remoted in to a headless system in a remote area. You have no physical access to the hardware and you need to perform an OS installation. What do you do?
81. On a Windows network, why is it easier to break into a local account than an AD account?


83. What could attackers do with HTTP Header Injection vulnerability?
   - Carriage returns and line feeds (or %0D & %0A) are means to an end that would allow attackers to control HTTP headers
   - could inject XSS via Referer header
   - could set cookie to a value known by the attacker (session fixation)
   - could redirect to a malicious server

84. Describe the last program or script that you wrote. What problem did it solve?
   - Just looking for signs that the candidate has basic understanding of programming concepts and is at least able to write simple programs

85. How would you implement a secure login field on a high traffic website where performance is a consideration?
   - TLS (regardless of performance) is a must
   - reducing 3rd party library dependencies, could improve performance and reduce security risks [link](https://hackernoon.com/im-harvesting-credit-card-numbers-and-passwords-from-your-site-here-s-how-9a8cb347c5b5)
   - Content-Security Policy (CSP) to enforce stricter execution rules around JS and CSS [link](https://en.wikipedia.org/wiki/Content_Security_Policy)
   - Subresource Integrity (SRI) to ensure only known, trusted resource files are loaded from 3rd-party servers/CDNs [link](https://en.wikipedia.org/wiki/Subresource_Integrity)

86. What are the various ways to handle brute forcing?
   - Account Lockouts/timeouts
   - API rate limiting
   - IP restrictions
   - Fail2ban
   - ...etc



## Software Reverse Engineering







---

## Web APP

1. judge if a remote server is running IIS or Apache?
   - Error messages oftentimes
     - give away what the server is running,
     - if the website administrator has not set up custom error pages for every site, it can give it away as entering a known bad address.
   - telnet
     - see how it responds.
     - Never underestimate the amount of information that can be gained by not getting the right answer but by asking the right questions.





## Web Security

87. What is **Cross-Site Request Forgery**? And how to defend?
   - attacker gets a victim's browser to make requests with the victim's credentials
   - Example:
   - if an image tag `<img>` points to a URL with an associated action, e.g. `https://foo.com/logout`
   - Defense includes but are not limited to:
     - check origins header & referer header
     - check CSRF tokens or nonce

88. What is **Cross-Site Scripting XSS**?
   - attackers get victim's browsers to execute some code (usually JavaScript) within their browser

89. the different types of XSS? defend?
   - Traditionally, types have been categorized into `Stored` and `Reflected` XSS attacks.
   - `Stored XSS` is some code that an attacker was able to persist in a database and gets retrieved and presented to victims (e.g. forum)
   - `Reflected XSS` is usually in the form of a maliciously crafted URL which includes the malicious code. When the user clicks on the link, the code runs in their browser
   - `DOM-based XSS`, occurs when attackers can control DOM elements, thus achieve XSS without sending any requests to the server
   - Server Stored XSS,
   - Server Reflected XSS,
   - Client Stored XSS (e.g. stored DOM-based XSS),
   - Client Reflected XSS (e.g. reflected DOM-based XSS)
   - Defense includes:
     - Output encoding (more important)
     - Input validation (less important)

89. How does HTTP handle state?
   - HTTP is stateless
   - State is stored in cookies


---

## AAA

1. How would you harden user authentication?
   - Generate Memorable Secure Passwords/Password Generators
   - Use password vaults
   - Two Factor Authentication
   - Use HTTPS/Firewalls
   - Use Robust Routers
   - Use good antivirus software

2. What is OAuth?
   - an open-standard authorization protocol/framework
   - describes how unrelated servers and services can safely allow authenticated access to their assets without actually sharing the initial, related, single logon credential.
    - known as secure, third-party, user-agent, delegated authorization.
   - Created and strongly supported from the start by Twitter, Google and other companies, OAuth was released as an open standard in 2010 as RFC 5849, and quickly became widely adopted.
   - example
    - log onto a website and it offers one or more opportunities to log on using another website’s/service’s logon.
    - click on the button linked to the other website, the other website authenticates you, and the website you were originally connecting to logs you on itself afterward using permission gained from the second website.

3. How OAuth works
   - assume a user has already signed into one website or service
    - OAuth only works using HTTPS
    - The user then initiates a feature/transaction that needs to access another unrelated site or service.
    - The following happens (greatly simplified):
   1. The first website
     - connects to the second website on behalf of the user, using OAuth,
     - providing the user’s verified identity.
   2. The second site
     - generates a `one-time token and a one-time secret` unique to the transaction and parties involved.
   3. The first site
     - gives this token and secret to the initiating user’s client software.
   4. The client’s software
     - presents the request token and secret to their authorization provider (which may or may not be the second site)
   - If not authenticated to the authorization provider, the client may be asked to authenticate.
   - After authentication, the client is asked to approve the authorization transaction to the second website.
   5. The user
      - approves (or their software silently approves) a particular transaction type at the first website.
      - The user is given an `approved access token` (notice it’s no longer a request token).
      - The user gives the approved access token to the first website.
   6. The first website
      - gives the access token to the second website as proof of authentication on behalf of the user.
   7. The second website lets the first website access their site on behalf of the user.
   8. The user sees a successfully completed transaction occurring.
   - OAuth is not the first authentication/authorization system to work this way on behalf of the end-user.
    - In fact, many authentication systems, notably Kerberos, work similarly.
    - What is special about OAuth is its ability to work across the web and its wide adoption.
    - It succeeded with adoption rates where previous attempts failed (for various reasons).
    - Although not as simple as it could be, web coders seem to readily understand the involved transactions.
    - Making a website OAuth-compatible can be done in a few hours to a day

---

## Architect
Have you designed security measures that span overlapping information domains?
Can you give me a few examples of security architecture requirements?
What special security challenges does Service-Oriented-Architecture (SOA) present?
Have you architected a security solution that involved SaaS components? What challenges did you face?
Have you worked on a project in which stakeholders choose to accept identified security risks that worried you? How did you handle the situation?
How do you handle demands from different stakeholders who have conflicting requirements?
How do you ensure that solution architects develop secure solutions?
How do you ensure that a solution continues to be resilient in the face of evolving threats?
What do you think the most important technology is right now? How are we going to secure it?
Blue Team
Given an HTTP traffic log between a machine on your network and a 3rd party website (e.g. Google), what would the source and destination ports look like?
Source port might be some number above port 1024 (aka ephemeral port)
Destination port might be 80 (HTTP) or 443 (HTTPS)





### TLS/SSL

1. TLS use symmetric or asymmetric encryption?
   - Both.
   - The initial exchange is done using asymmetric encryption
   - but bulk data encryption is done using symmetric.
   - Resources:
   - [link](https://web.archive.org/web/20150206032944/https://technet.microsoft.com/en-us/library/cc785811.aspx)
   - [link](https://en.wikipedia.org/wiki/Transport_Layer_Security)

2. **SSL / TLS session set up** (visits a secure website).
   - ==========================================
   > Client: “Hello there. I want to establish secure communication between the two of us. Here are my cipher suits and compatible SSL/TLS version.”
   - ==========================================
   - **Client** sends `ClientHello` message:
    - lists cryptographic information, such as SSL/TLS version and the client's order of preference of cipher suites.
    - contains a **Client random byte** string that is used in subsequent calculations.
    - may include data compression methods in the hello message as well.
   - ==========================================
   > Server: “Hello Client. I have checked your cipher suits and SSL/TLS version. I think we’re good to go ahead. Here are my certificate file and my public key. Check ‘em out.”
   - ==========================================
   - **Server** responds with `ServerHello` message:
    - the cipher suite chosen by server
    - the server's digital certificate
    - and **Server random byte** string
   - **Server** send `ServerCertification`
   - **Server** send client `CertificateRequest` to the client. If the server requires client certificate authentication
   - **Server** send `ServerKeyExchange`
   - **Server** send `ServerHelloDone`
   - ==========================================
   > Client: “verify your certificate. to verify your private key, I will generate and encrypt a pre-master (shared secret key) key using your public key. Decrypt it using your private key and we’ll use this master key to encrypt and decrypt the information”
   - ==========================================
   - **Client**:
    - verifies server's digital certificate.
    - get server public key.
   - **Client** send `ClientKeyExchange`
    - `(Client random byte + CipherMethod)` = **PreMasterSecretKey** encrypted by server public key
   - **Client** send `ClientCertification`
    - If server requested a client certificate, the client sends
      - a **Client random byte** encrypted with the client's private key with the client's digital certificate
      - **PreMasterSecretKey** encrypted by server public key
      - or "no digital certificate alert".
    - This alert is only a warning, but some implementations will cause the handshake to fail if client authentication is mandatory.
   - **Client** send `ChangeCipherSec`
    - `(ClientRandom byte + ServerRandom byte + PreMasterSecretKey` = **SessionSecrect**
   - **Client** send `ClientVerify`
    - (Hash all content) encrypted by PreMasterSecretKey
   - **Client** send `Finished`
    - sends finished message encrypted with the calculated secret key
   - ==========================================
   - Client: “I’m sending you this sample message to verify that our master-key works. Send me the decrypted version of this message. If it works, our data is in safe hands.”
   - ==========================================
   - **Server**
    - verified client's digital certificate.
    - got PreMasterSecretKey
   - **Server** send `ChangeCipherSec`
    - `(ClientRandom byte + ServerRandom byte + PreMasterSecretKey` = **SessionSecrect**
   - **Server** send `Finished`
    - sends finished message encrypted with the calculated secret key
   - ==========================================
   - For the duration of the TLS session, the server and client can now exchange messages that are symmetrically encrypted with the shared secret key
   - Resources:

3. How is TLS attacked? How has TLS been attacked in the past? Why was it a problem? How was it fixed?
Weak ciphers
Heartbleed
BEAST
CRIME
POODLE

1. SSL, why is it not enough to encryption?
   - SSL is identity verification, not hard data encryption.
   - It is designed to be able to prove that the person you are talking to on the other end is who they say they are.
   - SSL/TLS are both used almost everyone, but the problem is because of this it is a huge target and is mainly attacked via its implementation (The Heartbleed bug for example) and its known methodology.
   - As a result, SSL can be stripped in certain circumstances, so additional protections for data-in-transit and data-at-rest are very good ideas.





---


What is Forward Secrecy?

Forward Secrecy is a system that uses ephemeral session keys to do the actual encryption of TLS data so that even if the server’s private key were to be compromised, an attacker could not use it to decrypt captured data that had been sent to that server in the past.
Describe how Diffie-Hellman works.

## Forensics

1. Are open source projects more or less secure than proprietary projects?
   - Both models have pros and cons.
   - There are examples of insecure projects that have come out of both camps.
   - Open source model encourages "many eyes" on a project, but that doesn't necessarily translate to more secure products

2. What's important is not open source vs proprietary, but quality control of the project.
Who do you look up to in the Information Security field? Why?

Where do you get your security news from?

---

# Security architect

82. Explain data leakage and give examples of some of the root causes.
83. What are some effective ways to control data leakage?
84. Describe the 80/20 rules of networking.
85. What are web server vulnerabilities and name a few methods to prevent web server attacks?
86. What are the most damaging types of malwares?
87. What’s your preferred method of giving remote employees access to the company network and are there any weaknesses associated to it?
88. List a couple of tests that you would do to a network to identify security flaws.
89. What kind of websites and cloud services would you block?
90. What type of security flaw is there in VPN?
91. What is a DDoS attack?
92. Can you describe the role of security operations in the enterprise?
93. What is layered security architecture? Is it a good approach? Why?
94. Have you designed security measures that span overlapping information domains? Can you give me a brief overview of the solution?
95. How do you ensure that a design anticipates human error?
96. How do you ensure that a design achieves regulatory compliance?
97. What is capability-based security? Have you incorporated this pattern into your designs? How?
98. Can you give me a few examples of security architecture requirements?
99. Who typically owns security architecture requirements and what stakeholders contribute?
100. What special security challenges does  present?
101. What security challenges do unified communications present?
102. Do you take a different approach to security architecture for a COTS vs a custom solution?
103. Have you architected a security solution that involved SaaS components? What challenges did you face?
104. Have you worked on a project in which stakeholders choose to accept identified security risks that worried you? How did you handle the situation?
105. You see a user logging in as root to perform basic functions. Is this a problem?
106. What is data protection in transit vs data protection at rest?
107. You need to reset a password-protected BIOS configuration. What do you do?


108. How would you login to Active Directory from a Linux or Mac box?
     - Active Directory uses an implementation of the SMB protocol, which can be accessed from a Linux or Mac system by using the Samba program.
     - Depending on the version, this can allow for share access, printing, and even Active Directory membership.

109. SMB Protocol?
     - `Server Message Block (SMB)`
       - one version of `Common Internet File System (CIFS)`
       - an application-layer network protocol
       - mainly used for providing shared access to files, printers, and serial ports and miscellaneous communications between nodes on a network.
     - It also provides an authenticated inter-process communication mechanism.
     - Most usage of SMB involves computers running Microsoft Windows, where it was known as “Microsoft Windows Network” before the introduction of Active Directory.
     - Corresponding Windows services are LAN Manager Server (for the server component) and LAN Manager Workstation (for the client component)

110. small company, Why should I care about exploits and computer jibberish?
     - a classic catch-22 situation: a company doesn’t have enough money to secure their networks
     - An SMB will acknowledge what they need to do to keep their store secure and keep receiving payments since following the money will tend to help move things along.



### IPS/IDS

1.   What is an IPS and how does it differs from IDS?
     - intrusion detection system
     - intrusion prevention system.
     - IDS will just detect the intrusion and will leave the rest to the administrator for further action
     - IPS will detect the intrusion and will take further action to prevent the intrusion.
     - the positioning of the devices in the network.

---

## hack

1. compromise an “Office Workstation” at a hotel?
   - Considering how infected these typically are,

2. lock down a mobile device?
   - The baseline for these though would be three key elements:
   - An anti-malware application,
   - a remote wipe utility,
   - and full-disk encryption.

3. prevent a man-in-the-middle attack?
   - Secure/Multipurpose Internet Mail Extensions: Encrypts the email in transit
   - Use HTTPS
   - Use VPNS/Proxy

4. reset a password-protected BIOS configuration
   - While BIOS itself has been superseded by UEFI, most systems still follow the same configuration for how they keep the settings in storage.
   - BIOS itself is a pre-boot system
     - it has its own storage mechanism for its settings and preferences.
   - In the classic scenario, simply popping out the CMOS battery will be enough to have the memory storing these settings lose its power supply, and it will lose its settings.
   - use a jumper or a physical switch on the motherboard.
   - actually remove the memory itself from the device and reprogram it in order to wipe it out.
   - try default password enabled, ‘password’.


### XSS

1. XSS/Cross-site scripting, how will you mitigate it?
   - Cross-site Scripting (XSS)
   - client-side code injection attack
   - a JavaScript vulnerability in the web applications.
     - Javascript can run pages locally on the client system as opposed to running everything on the server side,
     - this cause variables can be changed directly on the client’s webpage.
   - attacker can execute malicious scripts/payload into a legitimate website or web application
   - Countermeasures of XSS
     - input validation
     - implementing a CSP (Content security policy)

2. XSS vs SQL Injection?
   - Cross-site scripting (XSS)
     - a type of computer security vulnerability typically found in Web applications.
     - attackers inject client-side script into Web pages viewed by other users.
     - A cross-site scripting vulnerability may be used by attackers to bypass access controls such as the same origin policy.
   - SQL injection
     - a code injection technique, used to attack data-driven applications, in which malicious SQL statements are inserted into an entry field for execution (e.g. to dump the database contents to the attacker).

---

# Risk management

1.   Is there an acceptable level of risk?
2.   How do you measure risk? Can you give an example of a specific metric that measures information security risk?
3.   Can you give me an example of risk trade-offs (e.g. risk vs cost)?
4.   What is incident management?
5.   What is business continuity management? How does it relate to security?
6.   What is the primary reason most companies haven’t fixed their vulnerabilities?
7.   What’s the goal of information security within an organization?
8.   What’s the difference between a threat, vulnerability, and a risk?
9.   If you were to start a job as head engineer or CSO at a Fortune 500 company due to the previous guy being fired for incompetence, what would your priorities be? [Imagine you start on day one with no knowledge of the environment]
10.  As a corporate information security professional, what’s more important to focus on: threats or vulnerabilities?

11.  How would you build the ultimate botnet?
12.  What are the primary design flaws in HTTP, and how would you improve it?
13.  If you could re-design TCP, what would you fix?
14.  What is the one feature you would add to DNS to improve it the most?
15.  What is likely to be the primary protocol used for the Internet of Things in 10 years?
16.  If you had to get rid of a layer of the OSI model, which would it be?
17.  What is residual risk?
18.  What is the difference between a vulnerability and an exploit?

---

# Security audits, testing & incident response

1.   What is an IT security audit?
2.   What is an RFC?
3.   What type of systems should be audited?
4.   Have you worked in a virtualized environment?
5.   What is the most difficult part of auditing for you?
6.   Describe the most difficult auditing procedure you’ve implemented.
7.   What is change management?
8.   What types of RFC or change management software have you used?
9.   What do you do if a rollout goes wrong?
10.  How do you manage system major incidents?
11.  How do you ask developers to document changes?
12.  How do you compare files that might have changed since the last time you looked at them?
13.  Name a few types of security breaches.
14.  What is a common method of disrupting enterprise systems?
15.  What are some security software tools you can use to monitor the network?
16.  What should you do after you suspect a network has been hacked?
17.  How can you encrypt email to secure transmissions about the company?
18.  What document describes steps to bring up a network that’s had a major outage?
19.  How can you ensure backups are secure?
20.  What is one way to do a cross-script hack?
21.  How can you avoid cross script hacks?
22.  How do you test information security?
23.  What is the difference between black box and white box penetration testing?
24.  What is a vulnerability scan?
25.  In pen testing what’s better, a red team or a blue team?
26.  Why would you bring in an outside contractor to perform a penetration test?


---

# Cryptography


1.   difference between Symmetric and Asymmetric encryption?
     - Symmetric encryption uses the same key to encrypt and decrypt
     - Asymmetric uses different keys for encryption and decryption.
     - Symmetric is usually much faster, but difficult to implement most times due to the fact that you would have to transfer the key over an unencrypted channel.
     - Hence, a hybrid approach should be preferred. Setting up a channel using asymmetric encryption and then sending the data using symmetric process

2.   What is secret-key cryptography?
3.   What is public-key cryptography?
4.   What is a session key?


5.   What is RSA?
    1.  asymmetric encryption
    2.  based on factoring 2 larger primes.
    3.  RSA works with both encryption and digital signatures, used in many environments, like Secure Sockets Layer (SSL), and key exchange.
     1.  3^5 = 3 to the 5th
     2.  B^Number Mod M = bignum
      1.  B^X Mod M = bignum1
      2.  B^Y Mod M = bignum2
     3.  (B^Y Mod M)^X = B^XY Mod M
      3.  B^X Mod M = B^XY Mod M = bignum1^Y = bignumber3
      4.  B^Y Mod M = B^XY Mod M = bignum2^X = bignumber3


6.   How fast is RSA?
7.   What would it take to break RSA?
8.   Are strong primes necessary for RSA?
9.   How large a module (key) should be used in RSA?
10.  How large should the primes be?
11.  How is RSA used for authentication in practice? What are RSA digital signatures?
12.  What are the alternatives to RSA?
13.  Is RSA currently in use today?
14.  What are DSS and DSA?
15.  What is difference between DSA and RSA?
16.  Is DSA secure?
17.  What are special signature schemes?
18.  What is a blind signature scheme?
19.  What is a designated confirmer signatures?
20.  What is a fail-stop signature scheme?
21.  What is a group signature?
22.  What is blowfish?
23.  What is SAFER?
24.  What is FEAL?
25.  What is Shipjack?
26.  What is stream cipher?
27.  What is the advantage of public-key cryptography over secret-key cryptography?
28.  What is the advantage of secret-key cryptography over public-key cryptography?
29.  What is Message Authentication Code (MAC)?
30.  What is a block cipher?
31.  What are different block cipher modes of operation?
32.  What is a stream cipher? Name a most widely used stream cipher.
33.  What is one-way hash function?
34.  What is collision when we talk about hash functions?
35.  What are the applications of a hash function?
36.  What is trapdoor function?
37.  Cryptographically speaking, what is the main method of building a shared secret over a public medium?
38.  What’s the difference between Diffie-Hellman and RSA?
39.  What kind of attack is a standard Diffie-Hellman exchange vulnerable to?
40.  What’s the difference between encoding, encryption, and hashing?
41.  In public-key cryptography you have a public and a private key, and you often perform both encryption and signing functions. Which key is used for which function?

42.  **encrypt and compress** data during transmission, which first?
    4. compression aims to use patterns in data to reduce its size.
    5. Encryption aims to randomize data so that it's uninterpretable without a secret key.
    6. encrypt first, then compress, then compression will be useless. `Compression doesn't work on random data. only use for plain data`
    7. compress first, then encrypt, then an attacker can find patterns in message length (Compression Ratio) to learn something about the data and potentially foil the encryption (like CRIME)

43.  What is SSL and why is it not enough when it comes to encryption?
44.  What is salting, and why is it used?
45.  What are salted hashes?
46.  What is the Three-way handshake? How can it be used to create a DOS attack?
47.  What’s more secure, SSL or HTTPS?
48.  Can you describe rainbow tables?

49.  Can two files generate same checksum?
    - Yes, but only if the contents are identical.
    - Even change a single word, the checksum will be different

50. difference between encoding, encryption, and hashing?
    - Encoding
      - protect the `integrity` of data as it crosses networks and systems
      - i.e. to keep its original message upon arriving
      - it isn’t primarily a security function.
      - easily reversible because the system for encoding is almost necessarily and by definition in wide use.
      - Example: base64
    - Encryption
      - for `confidentiality`
      - reversible only if you have the appropriate key/keys.
    - Hashing
      - ensures `Integrity`
      - one-way (non-reversible)
      - the output is of a fixed length that is usually much smaller than the input.
      - Hashing can be cracked using rainbow tables and collision attacks but is not reversible.

### hashing

1. salted hashes?
   - Salt at its most fundamental level is random data.
   - When a properly protected password system receives a new password
     - it will create a hashed value for that password,
     - create a new random salt value,
     - and then store that combined value in its database.
   - helps defend against `dictionary attacks` and `known hash attacks`.
   - For example, if a user uses the same password on two different systems, if they used the same hashing algorithm, they could end up with the same hash value. However, if even one of the systems uses salt with its hashes, the values will be different.


---


# knowledge point

1. What Is Tcp/ip Model?
   - TCP/IP model is an implementation of OSI reference model.
   - It has 4 layers.
   - Network layer, Internet layer, Transport layer and Application layer.


## TCP/UDP

1. Network traffic mainly categorizes into two types:
   - Transmission Control Protocol (TCP)
   - User Datagram Protocol (UDP).
   - Both protocols help in to establish the connection and transfer data between two ends of the communication. Below are the TCP/UDP interview questions and answers which generally asked in an interview.

2. Explain Transmission Control Protocol, TCP
   - TCP is a `connection-oriented protocol`.
   - when data is transferring from source to destination, `protocol takes care of data integrity` by sending data packet again if it lost during transmission.
   - ensures reliability and error-free data stream.
   - TCP packets contain fields such as Sequence Number, AcK number, Data offset, Reserved, Control bit, Window,  Urgent Pointer, Options, Padding, checksum, Source port, and Destination port.

3. Explain User Datagram Protocol, UDP
   - UDP is a `connection-less protocol`.
   - if one data packet is lost during transmission, it will not send that packet again.
   - This protocol is suitable where minor data loss is not a major issue.

4. How does TCP work?
   - three-way handshake to establish a connection between client and server.
   - It uses SYN, ACK and FIN flags (1 bit) for connecting two endpoints.
   - After the establishment of the connection, data is transferred sequentially.
   - If there is any loss of packet, it retransmits data.

5. List out common TCP/IP protocols.
   - HTTP         - Used between a web client and a web server, for non-secure data transmissions.
   - HTTPS         - Used between a web client and a web server, for secure data transmissions.
   - FTP         - Used between two or more computers to transfer files.

6. Comparison between TCP/IP & OSI model.
   - TCP/IP is the alternate model that also explains the information flow in the network.
   - It is a simpler representation in comparison to the OSI model but contains fewer details of protocols than the OSI model.

7. Is UDP better than TCP?
   - wants error-free and guarantees to deliver data, TCP
   - wants fast transmission of data and little loss of data is not a problem, UDP

8. What is the port number of Telnet and DNS?
   - Telnet is a protocol used to access remote server but insecurely. Port no of Telnet is 23.
   - DNS is a protocol used to translate a domain name to IP address. Port no of DNS is 53.

9. What is the UDP packet format?
   - contains four fields:
   - Source Port and Destination Port fields (16 bits each): Endpoints of the connection.
   - Length field (16 bits) : length of the header and data.
   - Checksum field (16 bits) : It allows packet integrity checking (optional).

10. What is the TCP packet format?
   - The TCP packet format consists of these fields:
   - `Source Port and Destination Port` fields (16 bits each);
   - `Sequence Number field` (32 bits);
   - `Acknowledgement Number field` (32 bits);
   - Data Offset (a.k.a. Header Length) field (variable length);
   - `Reserved field` (6 bits);
   - `Flags field` (6 bits) contains the various flags: URG,  ACK, PSH, RST, SYN, FIN;
   - `Window field` (16 bits);
   - `Checksum field` (16 bits) ;
   - `Urgent pointer` field (16 bits) ;
   - Options field (variable length)
   - Data field (variable length).

11. List out some TCP/IP ports and protocols.
   - Protocol	Port Number	Description
   - File Transfer Protocol (FTP)	                  20/21	Protocol helps in transferring files between client and server.
   - Secure Shell (SSH)	                                             22	This method helps to access remote server securely.
   - Telnet	                                                                        23	This method also helps to access remote server but here, data is transmitted in clear text.
   - Simple Mail Transfer Protocol (SMTP)	25	This protocol helps in managing email services.
   - Domain Name System (DNS)	                           53	This protocol helps in translating domain name into IP addresses.



## API

1. API
   - **Representational State Transfer** <- `XML/JSON`
     - not object, but the status of the object
     - GET/POST
   - **an interface** allows users to interact with a program through a client.
     - A client can be a browser
    - end-user uses to access a website. user interacting with Indeed's API through the browser.
     - A client can also be another application.
    - If you're a software developer, you might write a program that accesses the Indeed API to pull in information about jobs through the client application.
     - the client provides access to the API and its resources (objects application stores info about)


2. What is **REST API**?
   - A REST API is one that follows certain constraints.
   - A RESTful application is one that follows specific rules within the API.
   - For one, a RESTful app allows users to access resources. This could be objects like username or user profile and actions like creating user access or editing or removing a post.
   - RESTful applications are also easier for developers to access and use due to the constraints placed on the API.
   - REST API is one that follows the constraints of REST
   - allowing users to interact with the API in a specific way and making it easier for developers to use in their own applications.

3. benefits of using REST
   - easy to scale, flexible and portable
   - works independently from the client and server, which makes development less complex
   - it would give the user control over the resources needed to make accounts and share from the publication,
   - support massive growth.

4. architectural style for web APIs in REST?
   - REST is a set of constraints that has to be applied for an application to be RESTful.
   - The architectural has to have a few key characteristics.
    - HTTP so that a client can communicate with the enterprise server.
    - A format language specified as XML/JSON.
    - An address to reach services in the form of Uniform Resource Identifier and communicate statelessly.


5. test REST API? What tools are needed?
   - To test API you use specific software designed to assess RESTful constraints.
   - Some popular tools for practical API testing: SoapUI, Katalon Studio and Postman.
   - SoapUI, easy to download and access. to let applications been assessed consistently RESTful, good for the other developers at MetroMind who needed to use too.


6. difference between REST and AJAX?
   - Shared database vs batch file transfer
   - RPC vs MOM
   - PUT vs POST
   - Jax-WS vs Jax-RS
   - Request/response is different in AJAX and REST.
    - In REST, request/response revolves around a URL structure and resources
    - in AJAX request is transmitted via XMLHttpRequest objects and response occurs when JavaScript code makes changes to the page.
    - REST is a software development method
    - AJAX is a set of resources for development.
    - REST requires the customer to interact with internal servers
    - AJAX actively prevents it.

7. main characteristics of REST?
   - Primary characteristics of REST are being stateless and using GET to access resources.
   - In a truly RESTful application, the server can restart between calls as data passes through it.

8. HTTP methods commonly used in REST?
   - The HTTP methods supported by REST are `GET, POST, PUT, DELETE, OPTION and HEAD`.
   - The most commonly used method in REST is GET.


9. Can you use GET instead of PUT to create a new resource?
   - cannot use the GET feature instead of PUT
   - GET has view-rights only.

10. markup languages can be used in a RESTful web API
   - XML and JSON can be used in a RESTful web API.

11. resource in REST?
   - resource: a name for any piece of content in a RESTful piece of architecture.
   - This includes HTML, text files, images, video and more.



## Docker Basic Questions


1. What is Hypervisor?
   - a software that makes virtualization possible. It is also called Virtual Machine Monitor. It divides the host system and allocates the resources to each divided virtual environment. You can basically have multiple OS on a single host system. There are two types of Hypervisors:
   - Type 1: It’s also called Native Hypervisor or Bare metal Hypervisor. It runs directly on the underlying host system. It has direct access to your host’s system hardware and hence does not require a base server operating system.
   - Type 2: This kind of hypervisor makes use of the underlying host operating system. It’s also called Hosted Hypervisor.


2. What is virtualization?
   - Virtualization is the process of creating a software-based, virtual version of something(compute storage, servers, application, etc.).
   - These virtual versions or environments are created from a single physical hardware system.
   - Virtualization lets you split one system into many different sections which act like separate, distinct individual systems.
   - A software called Hypervisor makes this kind of splitting possible.
   - The virtual environment created by the hypervisor is called Virtual Machine.


3. What is containerization?
   - Let me explain this is with an example. Usually, in the software development process, code developed on one machine might not work perfectly fine on any other machine because of the dependencies.
   - This problem was solved by the containerization concept.
   - So basically, an application that is being developed and deployed is `bundled and wrapped together with all its configuration files and dependencies`.
   - This bundle is called a container.
   - Now when you wish to run the application on another system, the container is deployed which will give a bug-free environment as all the dependencies and libraries are wrapped together.
   - Most famous containerization environments are `Docker` and `Kubernetes`.


4. Difference between virtualization and containerization, between virtual machines and containers.
   - `Containers` provide an isolated environment for running the application.
     - The entire user space is explicitly dedicated to the application.
     - Any changes made inside the container is never reflected on the host or even other containers running on the same host.
     - Containers are an `abstraction of the application layer`.
     - Each container is a different application.
   - `Virtualization`,
     - hypervisors provide an entire virtual machine to the guest(including Kernel).
     - Virtual machines are an `abstraction of the hardware layer`.
     - Each VM is a physical machine.


5. What is Docker?
   - Docker is a containerization platform which packages your application and all its dependencies together in the form of containers
   - ensure that your application works seamlessly in any environment, be it development, test or production.
   - Docker containers, wrap a piece of software in a complete filesystem that contains everything needed to run: `code, runtime, system tools, system libraries, etc`. It wraps basically anything that can be installed on a server. This guarantees that the software will always run the same, regardless of its environment.



```
6. What is a Docker Container?
Docker containers include the application and all of its dependencies. It shares the kernel with other containers, running as isolated processes in user space on the host operating system. Docker containers are not tied to any specific infrastructure: they run on any computer, on any infrastructure, and in any cloud. Docker containers are basically runtime instances of Docker images.

7. What are Docker Images?
When you mention Docker images, your very next question will be “what are Docker images”.

Docker image is the source of Docker container. In other words, Docker images are used to create containers. When a user runs a Docker image, an instance of a container is created. These docker images can be deployed to any Docker environment.

8. What is Docker Hub?
Docker images create docker containers. There has to be a registry where these docker images live. This registry is Docker Hub. Users can pick up images from Docker Hub and use them to create customized images and containers. Currently, the Docker Hub is the world’s largest public repository of image containers.

9. Explain Docker Architecture?
Docker Architecture consists of a Docker Engine which is a client-server application with three major components:

A server which is a type of long-running program called a daemon process (the docker command).
A REST API which specifies interfaces that programs can use to talk to the daemon and instruct it what to do.
A command line interface (CLI) client (the docker command).
The CLI uses the Docker REST API to control or interact with the Docker daemon through scripting or direct CLI commands. Many other Docker applications use the underlying API and CLI.
Refer to this blog, to read more about Docker Architecture.

10. What is a Dockerfile?
Let’s start by giving a small explanation of Dockerfile and proceed by giving examples and commands to support your arguments.

Docker can build images automatically by reading the instructions from a file called Dockerfile. A Dockerfile is a text document that contains all the commands a user could call on the command line to assemble an image. Using docker build, users can create an automated build that executes several command-line instructions in succession.

The interviewer does not just expect definitions, hence explain how to use a Dockerfile which comes with experience. Have a look at this tutorial to understand how Dockerfile works.

11. Tell us something about Docker Compose.
Docker Compose is a YAML file which contains details about the services, networks, and volumes for setting up the Docker application. So, you can use Docker Compose to create separate containers, host them and get them to communicate with each other. Each container will expose a port for communicating with other containers.

12. What is Docker Swarm?
You are expected to have worked with Docker Swarm as it’s an important concept of Docker.

Docker Swarm is native clustering for Docker. It turns a pool of Docker hosts into a single, virtual Docker host. Docker Swarm serves the standard Docker API, any tool that already communicates with a Docker daemon can use Swarm to transparently scale to multiple hosts.

13. What is a Docker Namespace?
A namespace is one of the Linux features and an important concept of containers. Namespace adds a layer of isolation in containers. Docker provides various namespaces in order to stay portable and not affect the underlying host system. Few namespace types supported by Docker – PID, Mount, IPC, User, Network

14. What is the lifecycle of a Docker Container?
This is one of the most popular questions asked in Docker interviews. Docker containers have the following lifecycle:

Create a container
Run the container
Pause the container(optional)
Un-pause the container(optional)
Start the container
Stop the container
Restart the container
Kill the container
Destroy the container
15. What is Docker Machine?
Docker machine is a tool that lets you install Docker Engine on virtual hosts. These hosts can now be managed using the docker-machine commands. Docker machine also lets you provision Docker Swarm Clusters.

Docker Basic Commands
Once you’ve aced the basic conceptual questions, the interviewer will increase the difficulty level. So let’s move on to the next section of this Docker Interview Questions article. This section talks about the commands that are very common amongst docker users.

16. How to check for Docker Client and Docker Server version?
The following command gives you information about Docker Client and Server versions:

$ docker version

17. How do you get the number of containers running, paused and stopped?
You can use the following command to get detailed information about the docker installed on your system.

$ docker info

Course Curriculum
DevOps Certification Training
Instructor-led SessionsReal-life Case StudiesAssignmentsLifetime Access
You can get the number of containers running, paused, stopped, the number of images and a lot more.

18. If you vaguely remember the command and you’d like to confirm it, how will you get help on that particular command?
The following command is very useful as it gives you help on how to use a command, the syntax, etc.

$ docker --help

The above command lists all Docker commands. If you need help with one specific command, you can use the following syntax:

$ docker <command> --help

19. How to login into docker repository?
You can use the following command to login into hub.docker.com:

$ docker login

You’ll be prompted for your username and password, insert those and congratulations, you’re logged in.

20. If you wish to use a base image and make modifications or personalize it, how do you do that?
You pull an image from docker hub onto your local system

It’s one simple command to pull an image from docker hub:

$ docker pull <image_name>

21. How do you create a docker container from an image?
Pull an image from docker repository with the above command and run it to create a container. Use the following command:

$ docker run -it -d <image_name>

Most probably the next question would be, what does the ‘-d’ flag mean in the command?

-d means the container needs to start in the detached mode. Explain a little about the detach mode. Have a look at this blog to get a better understanding of different docker commands.

22. How do you list all the running containers?
The following command lists down all the running containers:

$ docker ps

23. Suppose you have 3 containers running and out of these, you wish to access one of them. How do you access a running container?
The following command lets us access a running container:

$ docker exec -it <container id> bash

The exec command lets you get inside a container and work with it.

24. How to start, stop and kill a container?

The following command is used to start a docker container:

$ docker start <container_id>

and the following for stopping a running container:

$ docker stop <container_id>

kill a container with the following command:

$ docker kill <container_id>

25. Can you use a container, edit it, and update it? Also, how do you make it a new and store it on the local system?
Of course, you can use a container, edit it and update it. This sounds complicated but its actually just one command.

$ docker commit <container id> <username/imagename>

26. Once you’ve worked with an image, how do you push it to docker hub?
$ docker push <username/image name>

27. How to delete a stopped container?
Use the following command to delete a stopped container:

$ docker rm <container id>

DevOps Training
28. How to delete an image from the local storage system?
The following command lets you delete an image from the local system:

$ docker rmi <image-id>

29. How to build a Dockerfile?
Once you’ve written a Dockerfile, you need to build it to create an image with those specifications. Use the following command to build a Dockerfile:

$ docker build <path to docker file>

The next question would be when do you use “.dockerfile_name” and when to use the entire path?

Use “.dockerfile_name” when the dockerfile exits in the same file directory and you use the entire path if it lives somewhere else.

30. Do you know why docker system prune is used? What does it do?
$ docker system prune

The above command is used to remove all the stopped containers, all the networks that are not used, all dangling images and all build caches. It’s one of the most useful docker commands.

Docker Advanced Questions
Once the interviewer knows that you’re familiar with the Docker commands, he/she will start asking about practical applications This section of Docker Interview Questions consists of questions that you’ll only be able to answer when you’ve gained some experience working with Docker.

31. Will you lose your data, when a docker container exists?
No, you won’t lose any data when Docker container exits. Any data that your application writes to the container gets preserved on the disk until you explicitly delete the container. The file system for the container persists even after the container halts.

32. Where all do you think Docker is being used?
When asked such a question, respond by talking about applications of Docker. Docker is being used in the following areas:

Simplifying configuration: Docker lets you put your environment and configuration into code and deploy it.
Code Pipeline Management: There are different systems used for development and production. As the code travels from development to testing to production, it goes through a difference in the environment. Docker helps in maintaining the code pipeline consistency.
Developer Productivity: Using Docker for development gives us two things – We’re closer to production and development environment is built faster.
Application Isolation: As containers are applications wrapped together with all dependencies, your apps are isolated. They can work by themselves on any hardware that supports Docker.
Debugging Capabilities: Docker supports various debugging tools that are not specific to containers but work well with containers.
Multi-tenancy: Docker lets you have multi-tenant applications avoiding redundancy in your codes and deployments.
Rapid Deployment: Docker eliminates the need to boost an entire OS from scratch, reducing the deployment time.
33. How is Docker different from other containerization methods?
Docker containers are very easy to deploy in any cloud platform. It can get more applications running on the same hardware when compared to other technologies, it makes it easy for developers to quickly create, ready-to-run containerized applications and it makes managing and deploying applications much easier. You can even share containers with your applications.

If you have some more points to add you can do that but make sure the above explanation is there in your answer.

34. Can I use JSON instead of YAML for my compose file in Docker?
You can use JSON instead of YAML for your compose file, to use JSON file with compose, specify the JSON filename to use, for eg:

$ docker-compose -f docker-compose.json up

35. How have you used Docker in your previous position?
Explain how you have used Docker to help rapid deployment. Explain how you have scripted Docker and used it with other tools like Puppet, Chef or Jenkins. If you have no past practical experience in Docker and instead have experience with other tools in a similar space, be honest and explain the same. In this case, it makes sense if you can compare other tools to Docker in terms of functionality.

36. How far do Docker containers scale? Are there any requirements for the same?
Large web deployments like Google and Twitter and platform providers such as Heroku and dotCloud, all run on container technology. Containers can be scaled to hundreds of thousands or even millions of them running in parallel. Talking about requirements, containers require the memory and the OS at all the times and a way to use this memory efficiently when scaled.

37. What platforms does docker run on?
This is a very straightforward question but can get tricky. Do some company research before going for the interview and find out how the company is using Docker. Make sure you mention the platform company is using in this answer.

Docker runs on various Linux administration:

Ubuntu 12.04, 13.04 et al
Fedora 19/20+
RHEL 6.5+
CentOS 6+
Gentoo
ArchLinux
openSUSE 12.3+
CRUX 3.0+
It can also be used in production with Cloud platforms with the following services:

Amazon EC2
Amazon ECS
Google Compute Engine
Microsoft Azure
Rackspace
38. Is there a way to identify the status of a Docker container?
There are six possible states a container can be at any given point – Created, Running, Paused, Restarting, Exited, Dead.

Use the following command to check for docker state at any given point:

$ docker ps

The above command lists down only running containers by default. To look for all containers, use the following command:

$ docker ps -a

39. Can you remove a paused container from Docker?
The answer is no. You cannot remove a paused container. The container has to be in the stopped state before it can be removed.

40. Can a container restart by itself?
No, it’s not possible for a container to restart by itself. By default the flag -restart is set to false.

41. Is it better to directly remove the container using the rm command or stop the container followed by remove container?
Its always better to stop the container and then remove it using the remove command.

$ docker stop <coontainer_id>
$ docker rm -f <container_id>

Stopping the container and then removing it will allow sending SIG_HUP signal to recipients. This will ensure that all the containers have enough time to clean up their tasks. This method is considered a good practice, avoiding unwanted errors.

42. Will cloud overtake the use of Containerization?
Docker containers are gaining popularity but at the same time, Cloud services are giving a good fight. In my personal opinion, Docker will never be replaced by Cloud. Using cloud services with containerization will definitely hype the game. Organizations need to take their requirements and dependencies into consideration into the picture and decide what’s best for them. Most of the companies have integrated Docker with the cloud. This way they can make the best out of both the technologies.

43. How many containers can run per host?
There can be as many containers as you wish per host. Docker does not put any restrictions on it. But you need to consider every container needs storage space, CPU and memory which the hardware needs to support. You also need to consider the application size. Containers are considered to be lightweight but very dependent on the host OS.

Course Curriculum
DevOps Certification Training
Weekday / Weekend Batches
44. Is it a good practice to run stateful applications on Docker?
The concept behind stateful applications is that they store their data onto the local file system. You need to decide to move the application to another machine, retrieving data becomes painful. I honestly would not prefer running stateful applications on Docker.

45. Suppose you have an application that has many dependent services. Will docker compose wait for the current container to be ready to move to the running of the next service?
The answer is yes. Docker compose always runs in the dependency order. These dependencies are specifications like depends_on, links, volumes_from, etc.

46. How will you monitor Docker in production?
Docker provides functionalities like docker stats and docker events to monitor docker in production. Docker stats provides CPU and memory usage of the container. Docker events provide information about the activities taking place in the docker daemon.

47. Is it a good practice to run Docker compose in production?
Yes, using docker compose in production is the best practical application of docker compose. When you define applications with compose, you can use this compose definition in various production stages like CI, staging, testing, etc.

48. What changes are expected in your docker compose file while moving it to production?
These are the following changes you need make to your compose file before migrating your application to the production environment:

Remove volume bindings, so the code stays inside the container and cannot be changed from outside the container.
Binding to different ports on the host.
Specify a restart policy
Add extra services like log aggregator
49. Have you used Kubernetes? If you have, which one would you prefer amongst Docker and Kubernetes?
Be very honest in such questions. If you have used Kubernetes, talk about your experience with Kubernetes and Docker Swarm. Point out the key areas where you thought docker swarm was more efficient and vice versa. Have a look at this blog for understanding differences between Docker and Kubernetes.

You Docker interview questions are not just limited to the workarounds of docker but also other similar tools. Hence be prepared with tools/technologies that give Docker competition. One such example is Kubernetes.

50. Are you aware of load balancing across containers and hosts? How does it work?
While using docker service with multiple containers across different hosts, you come across the need to load balance the incoming traffic. Load balancing and HAProxy is basically used to balance the incoming traffic across different available(healthy) containers. If one container crashes, another container should automatically start running and the traffic should be re-routed to this new running container. Load balancing and HAProxy works around this concept.

This brings us to the end of the Docker Interview Questions article. With increasing business competition, companies have realized the importance of adapting and taking advantage of the changing market. Few things that kept them in the game were faster scaling of systems, better software delivery, adapting to new technologies, etc. That’s when docker swung into the picture and gave these companies boosting support to continue the race.

If you want to learn more about DevOps, check out the DevOps training by Edureka, a trusted online learning company with a network of more than 250,000 satisfied learners spread across the globe. The Edureka DevOps Certification Training course helps learners gain expertise in various DevOps processes and tools such as Puppet, Jenkins, Nagios and GIT for automating multiple steps in SDLC.
```


---

ref:
[1](https://www.indeed.com/career-advice/interviewing/rest-api-interview-questions)
