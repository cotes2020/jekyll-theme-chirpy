---
title: Cyber Kill Chain
author: Cotes Chung
date: 2020-09-15 11:11:11 -0400
categories: [10SecConcept, Attack]
tags: [CKC]
toc: true
image:
---

# Cyber Kill Chain

[toc]

The cyber kill chain is a series of steps that trace stages of a cyberattack from the early reconnaissance stages to the exfiltration of data.

The kill chain helps understand and combat ransomware, security breaches, and advanced persistent attacks (APTs).

---
![Screen Shot 2020-09-18 at 00.04.54](https://i.imgur.com/gWRGgpB.png)
# 8 phases of the kill chain

## Reconnaissance [information gathering]
The observation stage: attackers typically assess the situation from the outside-in, in order to identify both targets and tactics for the attack.
- attacker is seeking information that might reveal vulnerabilities and weak points in the system. Firewalls, intrusion prevention systems, perimeter security – these days, even social media accounts – get ID’d and investigated.
- Reconnaissance tools scan corporate networks to search for `points of entry` and `vulnerabilities` to be exploited.


## Intrusion
Based on what the attackers discovered in the reconnaissance phase, they’re able to get into your systems: often leveraging malware or security vulnerabilities.
- Intrusion is when the attack becomes active:
  - attackers can send malware – ransomware, spyware, and adware – to the system to gain entry.
  - This is the delivery phase:
  - it could be delivered by phishing email,
  - it might be a compromised website or that really great coffee shop down the street with free, hacker-prone wifi.
  - Intrusion is the point of entry for an attack, getting the attackers inside.


## Exploitation
The act of exploiting vulnerabilities, and delivering malicious code onto the system, in order to get a better foothold.
- The exploitation stage of the attack, exploits the system
- Attackers can now get into the system and install additional tools, modify security certificates and create new script files for nefarious purposes.
- PowerShell
- Local job scheduling
- Scripting
- Dynamic data exchange


## Privilege Escalation
Attackers often need more privileges on a system to get access to more data and permissions: for this, they need to escalate their privileges often to an Admin.
- Attackers use privilege escalation to get elevated access to resources.
- Privilege escalation techniques often include
  - brute force attacks, password vulnerabilities, and exploiting zero day vulnerabilities.
  - They’ll modify GPO security settings, configuration files, change permissions, and try to extract credentials.
  - Access token manipulation
  - Path interception
  - Sudo attack
  - Process injection


## Lateral Movement
Once they’re in the system, attackers can move laterally to other systems and accounts in order to gain more leverage: whether that’s higher permissions, more data, or greater access to systems.
- Attackers will move from system to system, in a lateral movement, to gain more access and find more assets.
- advanced data discovery mission, where attackers seek out critical data and sensitive information, admin access and email servers – often using the same resources as IT and leveraging built-in tools like PowerShell – and position themselves to do the most damage.
  - SSH hijacking
  - Internal spear phishing
  - Shared webroot
  - Windows remote management



## Obfuscation / Anti-forensics
attackers need to cover their tracks, lay false trails, compromise data, and clear logs to confuse and/or slow down any forensics team.
- conceal their presence and mask activity to avoid detection and thwart the inevitable investigation.
  - wiping files and metadata,
  - overwriting data with false timestamps (timestomping)
  - misleading information,
  - modifying critical information so that it looks like the data was never touched.
  - Binary padding
  - Code signing
  - File deletion
  - Hidden users
  - Process hollowing



## Denial of Service (DoS) attack
Disruption of normal access for users and systems, in order to stop the attack from being monitored, tracked, or blocked
- the attackers target the network and data infrastructure, so the legitimate users can’t get what they need.
  - disrupts and suspends access,
  - and could crash systems and flood services.
  - Data compressed
  - Data encrypted
  - Exfiltration over alternative protocol
  - Exfiltration over a physical medium



## Exfiltration
getting data out of the compromised system.
- copy, transfer, or move sensitive data to a controlled location,
- It can take days to get all of the data out, but once it’s out, it’s in their control.




.
