---
title: Meow's CyberAttack - Application/Server Attacks - Hijacking - Man-in-the-Middle Attacks
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack, Hijacking]
tags: [CyberAttack, Hijacking]
toc: true
image:
---

- [Meow's CyberAttack - Application/Server Attacks - Hijacking - Man-in-the-Middle Attacks](#meows-cyberattack---applicationserver-attacks---hijacking---man-in-the-middle-attacks)
  - [Man-in-the-Middle Attacks](#man-in-the-middle-attacks)
  - [News](#news)
  - [TCP/IP hijacking](#tcpip-hijacking)
  - [remediation](#remediation)
  - [detection](#detection)

book: S+ 7th ch9

---

# Meow's CyberAttack - Application/Server Attacks - Hijacking - Man-in-the-Middle Attacks

---

## Man-in-the-Middle Attacks

![Pasted Graphic 2](/assets/img/Pasted%20Graphic%202_12uzzw5fi.png)

- <font color=LightSlateBlue> Clandestinely 秘密地 place something (like software, rouge router) between server and user, and no one is aware </font>.
  - Intercepts data and sends the information to the server as if nothing is wrong.
  - The server responds to the software, as it is the legitimate client.
  - may recording information, altering it, or in other way compromising the security of your system and session.

- <font color=OrangeRed> an active attack </font>.
  - Something is <font color=OrangeRed> actively intercepting the data and may altering it </font>.
  - a form of active interception/eavesdropping.

- It uses a separate computer that accepts traffic from each party in a conversation and forwards the traffic between the two.
  - The two computers are unaware of the MITM computer, and it can <font color=LightSlateBlue> interrupt the traffic at will or insert malicious code </font>.

- <font color=OrangeRed> Address Resolution Protocol (ARP) poisoning </font> is one way that an attacker can launch an MITM attack.


Threat of man-in-the-middle attacks on wireless networks has increased.

- no necessary to connect to the wire

- malicious rogue can be outside the building intercepting packets, altering them, and sending them on.

---

## News

POODLE SSLv3 Vulnerability - POODLE (Padding Oracle On Downgraded Legacy Encryption)

- Allows attacker to read information encrypted with SSLv3, decipher the plain text content, using a MITM attack. when decrypting messages encrypted using block ciphers in cipher block chaining (CBC) mode.
  - affects any services or clients communicate using SSLv3.
  - affects every piece of software that can be coerced into communicating with SSLv3. (any software that implements a fallback mechanism that includes SSLv3 support is vulnerable and can be exploited.)
  - Common: web browsers, web servers, VPN servers, mail servers, etc.

- Although SSLv3 is an older version of the protocol which is mainly obsolete, many pieces of software still fall back on SSLv3 if better encryption options are not available. More importantly, it is possible for an attacker to force SSLv3 connections if it is an available alternative for both participants attempting a connection.

- POODLE vulnerability:
  - because the SSLv3 protocol does not adequately check the `padding bytes` that are sent with encrypted messages.
  - Since these cannot be verified by the receiving party,
  - an attacker can replace these and pass them on to the intended destination.
  - When done in a specific way, the modified payload will potentially be accepted by the recipient without complaint.
  - An average of once out of every 256 requests will accepted at the destination, allowing the attacker to decrypt a single byte.
  - This can be repeated easily in order to progressively decrypt additional bytes.
  - Any attacker able to repeatedly force a participant to resend data using this protocol can break the encryption in a very short amount of time.
  - Protect:
    - Actions to ensure that you are not vulnerable as both a client and a server.
    - Since encryption is usually negotiated between clients and servers, it is an issue that involves both parties.
    - Servers and clients should should take steps to disable SSLv3 support completely.
    - Many applications use better encryption by default, but implement SSLv3 support as a fallback option.
      - This should be disabled,
      - a malicious user can force SSLv3 communication if both participants allow it as an acceptable method.

---

## TCP/IP hijacking

- An <font color=LightSlateBlue> older term generically used for all man-in-the-middle attacks </font>.
  - attacker gain access to a host in network and logically <font color=LightSlateBlue> disconnecting it from the network </font>.
  - The attacker then <font color=LightSlateBlue> inserts another machine with the same IP address </font>.
  - This happens quickly, and it gives the attacker access to the session and to all the information on the original system.
  - The server won’t know that this has occurred, and it will respond as if the client is trusted—the attacker forces the server to accept its IP address as valid.
  - The hijacker will hope to acquire privileges and access to all the information on the server.

- Can do little to counter this threat
  - but attacks require fairly sophisticated software
  - are harder to engineer than a simple DoS attack.



issue regarding certificates on a secure website.

- the web gateway proxy on the local network has signed all of the certificates on the local machine.

- the proxy has been legitimately programmed to perform man-in-the-middle attacks

---

## remediation

best remediation:
- <font color=OrangeRed> Requiring client and server PKI certificates for all connections </font>

- <font color=OrangeRed> Kerberos </font> helps prevent man-in-the-middle attacks with <font color=LightSlateBlue> mutual authentication </font>.
  - doesn’t allow a malicious system to insert itself in the middle of the conversation without the knowledge of the other two systems.

- enforce secure wireless authentication protocol, like WPA2.

---

## detection

MITM attack aimed at impersonating the default fateway is underway.

- To detect this attack:
  - Ipconfig
  - Tracert
  - Implement a logon script to prevent MITM:
    - `arp - s 192.168.1.1 00-3a-d1-fa-b1-06`
