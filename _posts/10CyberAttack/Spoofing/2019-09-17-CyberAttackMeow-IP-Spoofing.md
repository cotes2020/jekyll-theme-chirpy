---
title: Meow's CyberAttack - IP Spoofing
date: 2019-09-17 11:11:11 -0400
categories: [10CyberAttack, Spoofing]
tags: [CyberAttack, Spoofing]
toc: true
image:
---

# Meow's CyberAttack - IP Spoofing

[toc]

---

![image004](https://i.imgur.com/5L9wA6Y.jpg)

---

### IP Spoofing

Each IP packet includes a place to specify the destination and source IP addresses.
- The validity of the source addres
s is never checked, however, and it is trivial for anyone to specify a source address that is different from their actual IP address.

In fact, nearly every operating system provides an interface by which it can make network connections with arbitrary IP header information
- so spoofing an IP address is specifying the desired IP in the source field of an IP packet data structure before transmitting that data to the network.

hacker uses tools to modify the source address in the packet header to make the receiving computer system think the packet is from a trusted source,
- this occurs at the network level, there are no external signs of tampering.


Such modification of the source address to something other than the sender’s IP address is called IP spoofing.

- IP spoofing does not actually allow an attacker to assume a new IP address by simply changing packet headers, however, because his actual IP address stays the same.
- The source address in the header of an IP packet is simply overwritten with a different IP address from the actual source.
- Note the header checksum field also needs to be updated.

![Screen Shot 2018-11-07 at 17.55.52](https://i.imgur.com/inR65UN.png)
￼

### How IP Spoofing is Used in Other Attacks
with a spoofed source IP address on an outbound packet, the machine with the spoofed IP address will receive any response from the destination server.
If attacker is using IP spoofing on his outbound packets, he must either not care about any responses for these packets or he has some other way of receiving responses.
- example:
- denial-of-service attacks:
  - the attacker doesn’t want to receive any responses back,
  - he just wants to overwhelm some other Internet host with data requests.
- IP spoofing attacks designed for circumventing firewall policy or TCP session hijacking, the attacker has another, nonstandard way of getting response packets.
- A variation on this approach uses thousands of computers to send messages with the same spoofed source IP address to a huge number of recipients. The receiving machines automatically transmit acknowledgement to the spoofed IP address and flood the targeted server.


### Steps to Avoid Spoofing
- **ingress filtering**
  - works only if all routers use it.
- monitoring networks for atypical activity,
- using robust verification methods (even among networked computers),
- **authenticating all IP addresses**,
- Set up a comprehensive **packet filtering system for router or security gateway**. 
  - This should analyze and discard incoming data packets if they have source addresses of devices within your network.
  - Outgoing packets with sender addresses outside of the network should also be watched for and filtered.
  - detect inconsistencies (like outgoing packets with source IP addresses that don't match those on the organization's network),
- avoid host-based authentication systems. 
  - Make sure that all log-in methods take place via encrypted connections.
  - This minimizes the risk of an IP spoofing attack within your own network while also setting important standards for overall security.
- using a network attack blocker.
- Placing at least a portion of computing resources behind a firewall is also a good idea.
- Web designers are encouraged to migrate sites to IPv6, the newest Internet Protocol.
  - It makes IP spoofing harder by including encryption and authentication steps.
- For end users, detecting IP spoofing is virtually impossible.
  - They can minimize the risk of other types of spoofing, however, by using secure encryption protocols like HTTPS — and only surfing sites that also use them.
