---
title: Meow's Testing Tools - Ettercap
date: 2019-09-17 11:11:11 -0400
categories: [10CyberAttack, CyberAttackTools]
tags: [CyberAttack, CyberAttackTools]
toc: true
image:
---

# Ettercap

![Ettercap](https://i.imgur.com/0ciXVzn.png)

Ettercap has two modes:
- console-based mode / GUI-based mode.


## ARP Spoofing

In GUI mode
- more easily select which hosts you want to target.
- a sniffer that can also run MitM attacks.
- When run an ARP spoof attack, need to know IP address to MAC address mappings, get Ettercap to check for hosts on the network.
- The first thing to do:
  - tell Ettercap do `Unified sniff` if there is only one interface on the system Ettercap is running on,
  - or `Bridged sniff` if there are multiple interfaces.
  - Once that's done, other menus show up.
  - Once it runs a scan of all of the hosts, bring up a host list.
- Once the host list is in place, `select the hosts want to target`.
  - Since ultimately we're talking about conversations, you can have two targets to place hosts into. This refers to two ends of the conversation.
  - to listen to a conversation between two hosts on your network, like a client system and a local domain controller, so as to potentially grab credentials.
  - You would put one of the systems in Target 1 and the other into Target 2.

## DNS spoofing
- useful, easier to capture the DNS request.
  - Unless we can capture the traffic, it's hard to get the DNS request to know how and when to respond to it.
  - Unlike ARP, we can't just send a spurious response.
  - This is not to say that DNS information isn't cached. Just as with ARP, systems want to be as efficient as possible. DNS requests are time-consuming, so operating systems don't want to make them unless they are necessary.
- Where possible, operating systems will cache DNS mappings from hostname to IP address. That means we poison the cache once and have the system continue to send requests to the wrong address for potentially days.


1. Ettercap requires a configuration file in which you set up the DNS records you want to spoof.
- It will look just like a DNS zone file, provide the record name, the record type, and what it maps to.
- The location: Linux: /etc/ettercap/etter.dns
- There are a number of entries already in place there.
￼

1. Once DNS is in place, we need to go back to set up Ettercap to intercept traffic.
- same process we did before.
- sniff traffic so Ettercap can see the requests come in.
  - use an ARP spoof attack to get traffic on the network to our system
  - to see the DNS requests.
- Once you get to the stage of starting an ARP spoof
  - Plugins menu > Manage Plugins > enable the DNS spoof plug-in.
  - This will automatically load the etter. dns file that was edited earlier.
- In case it's not apparent:
  - these attacks will only work on the local network because the addressing is by MAC address . This requires physical network connectivity for the interface being used to run the spoofing.
- log that is written out in Ettercap from any request that has been captured.
  - While the entries in the preceding code listing were added, none of the default entries in the file were removed.
  - Microsoft 's website is one of the host-names that is being redirected.
  - In that case, it's not being redirected to one of our local systems but instead to another system on the Internet altogether.
  - Since we are using DNS here, the host doesn't have to be on the local network.
  - DNS will respond with an IP address, the requesting system will make a connection to that IP address.
  - The only reason we need local access is to capture the requests. Once the requests have been captured and responded to, everything is layer 3 and above.
