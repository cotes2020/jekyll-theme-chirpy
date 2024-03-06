---
title: Meow's CyberAttack - ARP Poisoning, ARP spoofing
date: 2019-09-17 11:11:11 -0400
categories: [10CyberAttack, Spoofing]
tags: [CyberAttack, Spoofing]
toc: true
image:
---

# Meow's CyberAttack - ARP Poisoning ARP spoofing

[toc]

---

## ARP

1. ARP `resolves the IP addresses to MAC address`
   - stores the result in an memory, ARP cache.
   - TCP/IP uses IP address to get a packet to a destination network.
   - In destination network, uses the MAC address to get it to the correct host.
   - ARP Poisoning occurs In Data link Layer
2. ARP uses two primary messages:
   - `ARP request`: broadcasts the IP address, asks, “Who has this IP address?”
   - `ARP reply`: The computer with the IP address in the ARP request responds with its MAC address. The computer that sent the ARP request caches the MAC address for the IP.
   - In many operating systems, all computers that hear the ARP reply also cache the MAC address.
3. vulnerability of ARP:
   - It will **believe any ARP reply packet**. Gratuitous ARP
   - nothing to authenticate the request

---

## Threats of ARP Poisoning:
- gain access to the network,
- fool the router to send data that was intended for another host,
- DoS attack.
- In all cases, the address being faked is an address of a legitimate user, and that makes it possible to get around such measures as allow/deny lists.
- man-in-the-middle attack
- Run Denial of Service (DoS) attacks.
- Intercept data.
- Collect passwords.
- Manipulate data.
- Tap VoIP phone calls.

---

## ARP poisoning achieve

- ARP poisoning: misleads computers or switches about the actual MAC address of a system.
- ARP poisoning achieved in 2 steps:
   1. Fake the MAC address of the data.
   2. make it look as if the data came from a network that it did not.


=================================================================================

- ARP poisoning:
   - create ARP reply packets
   - spoofed or bogus MAC addresses
   - reply and poison the ARP cache on systems in the network.
   - **Gratuitous ARP**:
     - not waiting for request just sending the reply.
     - in order to be efficient, systems will take any reply, even didn’t ask, they cache the mapping. (For avoid always ask)

- By MAC flooding a switch's ARP table with spoofed ARP replies, the attacker can overload the switches and then packet sniff the network while the switch is in “forwarding mode”.

- problem:
   - the length of time ARP entries are cached for: need keep sending gratuitous ARP response.
   - linux: /proc pseudo filesystem: default cache length is 60s. (Can replace it)
   - cat /proc/sys/net/ipv4/neigh/default/gc_stale_time 60
   - windows: cache duration, system different from one and other.
   - base time of 30,000 milliseconds. Multiplied by random value (0.5-1.5)
- to keep conversation finished, need to forwards back message to the true dst address.

- Tools:
   - Ettertap.
=================================================================================

### Attack tools
To initiate an ARP poisoning is fairly easy and can be done by many programs:
- [arpspoof](https://ocholuo.github.io/posts/CyberAttackTools-arpspoof/)
- [ettercap](https://ocholuo.github.io/posts/CyberAttackTools-ettercap/)
- [Cain and Abel](https://ocholuo.github.io/posts/CyberAttackTools-CainandAbel/)


---

## ARP Man-in-the-Middle Attacks

![ARP-MITM](https://i.imgur.com/nkpjGN8.jpg)

man-in-the-middle attack:
- attacker redirect network traffic and insert malicious code.
- Normally, traffic from the user to the Internet will go through the switch directly to the router, after poisoning the ARP cache of the victim, traffic is redirected to the attacker.

- The victim’s ARP cache should include this entry to send data to the router:192.168.1.1, 01-23-45-01-01-01
- However, after poisoning the ARP cache, it includes this entry:192.168.1.1, 01-23-45-66-66-66
- The victim now sends all traffic destined for the router to the attacker.
- The attacker captures the data for analysis later.
- It also uses another method such as IP forwarding to send the traffic to the router so that the victim is unaware of the attack.

---

## ARP DoS Attacks

- use ARP poisoning in a DoS attack.
- Example:
- attacker send an ARP reply with a bogus MAC address for the default gateway.
- The default gateway is the IP address of a router connection that provides a path out of the network.
- If all the computers cache a bogus MAC address for the default gateway, none of them can reach it, and it stops all traffic out of the network.


---

## defend against ARP Spoofing

1. Use ARPWALL system and block ARP spoofing attacks
2. Use **private VLANS**
3. **Place static ARP entries** on servers, workstation and routers
   - can add an extra defense from ARP poisoning.
   - By adding, it is essentially a communication link between devices.
   - As stated above; A communication session with a host, A device sends an ARP request but with a Static entry It knows the target host’s MAC address.
   - eliminating the process of sending an ARP request.
4. make sure all switches are safe from ARP poisoning.
   - use the command: `ip dhcp snooping`
5. **No MAC Address**: 
   - ARP request is IP address to MAC address (IP-to-MAC).
   - If users were to use tools to sniff packets and to find a ARP request without a MAC address. it is highly suspicious.
6. Keeping track of IP-to-MAC: 
   - keeping track of your device’s communication sessions of IP-to-MAC.
   - New IP-to-MAC can cause suspicion.
7. Tools vs Tools: 
   - Many software tools are out there to help protect users from malicious attacks.
   - many ARP poisoning tools: AntiARP, ARPon, ArpStar, XARP.

### XARP

![XARP](https://i.imgur.com/vWTIp91.png)
XARP is security tool users can download and use.
- performs ARP poisoning detection to detect
- Notify and respond to ARP poisoning.
- allows users to have multiple added layers of passive and active defense against ARP Poisoning.
- XARP has several Modules that have specific functions.
- If those conditions are violated they will generate a notification for the User.
- `ChangeFilter`: 
  - Module keeps tracks of IP-to-Mac address mapping.
  - Every ARP Packet contains a mapping of IP-to-MAC addresses. ARP request contains the IP-to-MAC mapping of the sender. ARP replies to contain the IP-to-MAC mapping of the machine resolved. Every mapping is inserted into a database. If a mapping is monitored that break current mapping, an alert is generated. Using Network discoverers, the database is filled quickly and more reliably than without network discoverers.
- `CorruptFilter`: 
  - ARP packets have a special restriction.
  - Ethernet Source Mac Address has to match the ARP source MAC address.
  - Furthermore, there are a field in the ARP packet that has restrictions regarding the values they can adopt. This module checks these values correctness.
  - ProxyARP servers will generate false alerts because they answer ARP request for other machines and thus not contain the saim ethernet source MAC address and ARP mapping source MAC.
- `DirectRequestFilter`: 
  - ARP request needs to be sent to the broadcast MAC address.
  - Some DSL Routers want to know which machines are currently online for their web management interface. Therefore they send out ARP requests which have the specific MAC address entered in the Ethernet packet. Such packets are also used by ARP spoofing software to spoof only a specific machine and not all machines on a network.
- `IPFilter`: 
  - An ARP mappings may contain certain IP addresses. These include broadcast and multicast as well as localhost addresses.
- `MACFIlter`: 
  - Some MAC addresses in ARP packets are highly suspicious. No IP-to-MAC mapping should, for example, have the MAC broadcast address assigned. Furthermore, an ARP reply is suspicious if it maps to one IP addresses to the local machines MAC address. Such Alert might also get generated when you are running a virtual machine. Replies arrive at your real machine with ARP replies containing your MAC address as the sender.
- `RequestedResponseFilter`: 
  - ARP replies should normally follow ARP requests. This filter remembers all ARP requests originating and matches them to an ARP replies. Many ARP spoofing tools send ARP replies that are not requested. This filter might give false positive in some cases as machines want to distribute their IP-to-MAC mapping to other machines that did not request it.
- `StaticPreserveFilter`: 
  - This filter will periodically request local ARP cache and remember to IP-to-MAC mappings that you are static. If an ARP packet violates this static mapping an alert will be generated. If a mapping from an ARP packet tries to collide with a static mapping, someone is trying to spoof your machine.
- `SubnetFilter`: 
  - Every ARP packet IP addresses need to be in the same subnet. An ARP packet with IP addresses that are not in the network interfaces configured subnet are suspicious and will be alerted.”


---














---
