---
title: Meow's Testing Tools - sslstrip
date: 2019-09-17 11:11:11 -0400
categories: [10CyberAttack, CyberAttackTools]
tags: [CyberAttack, CyberAttackTools]
toc: true
image:
---

# sslstrip


Encrypted messages are problematic when it comes to capturing traffic.
- Encryption is intended to be end to end, no way to sit in the middle.
- Any mechanism to sit in the middle defeats the end-to-end expectation of most encryption schemes.
- much easier when SSL was being used.
  - SSL had multiple vulnerabilities over the different versions prior to TLS.
  - the early versions of TLS had vulnerabilities susceptible to be cracked.

The program `sslstrip` was developed to grab SSL messages and strip the encryption from them.
- by Moxie Marlinspike in Black Hat in 2009.
- Today, there is less of a likelihood of success because, ideally, system administrators on top of their game have removed older encryption mechanisms like SSL and TLS 1.0 and 1.1.
- If a server only supports TLS 1.2 and above, SSL strip won't work.

You could use sslstrip as a stand-alone program.
- sslstrip acts as a transparent proxy, sitting between the server and client.
- In doing that, it can `change links from HTTPS to HTTP` in some cases.
- It also uses other techniques to make it appear that the connection is encrypted when, in fact, it isn't.
- As a stand-alone, sslstrip makes use of arpspoof
- also can run sslstrip as a plug-in to Ettercap.

Like DNS spoofing, sslstrip requires ARP spoof
- do this with Ettercap
  - spoof the Internet gateway as we have before.
  - also need to sniff remote connections when we set up the ARP spoofing attack.
- sslstrip is a plug-in to Ettercap needs enabled
  - a configuration change in Ettercap
  - sslstrip needs to know what firewall command is being used so it can set up a redirect in the firewall.
  - They need to be uncommented if using iptables, which is more likely than ipchains, which is the other option.

![sslstrip](https://i.imgur.com/uuOqywf.png)

  - Once done, sslstrip plug-in enabled.
  - It will run the iptables command to start the redirect so the plug-in can receive the messages. Here you can see the log shows the start of sslstrip inside Ettercap .


Once the iptables rule is in place, sslstrip can `capturing any HTTPS traffic`.
- assumes that the HTTPS connection is using a version of SSL/TLS that is vulnerable to the stripping attack.
- If it isn't, you won't get any traffic.

![sslstrip](https://i.imgur.com/K9lZ7F3.png)


to get plaintext traffic. It does not remove SSL requests
- it may be used to convert an HTTPS request to an HTTP request.
- It does not convert SSL to TLS or TLS to SSL
