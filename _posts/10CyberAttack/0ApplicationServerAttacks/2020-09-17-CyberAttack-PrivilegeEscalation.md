---
title: Meow's CyberAttack - Application/Server Attacks - Privilege Escalation
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack]
tags: [CyberAttack]
toc: true
image:
---

- [Meow's CyberAttack - Application/Server Attacks - Privilege Escalation](#meows-cyberattack---applicationserver-attacks---privilege-escalation)
  - [Privilege Escalation](#privilege-escalation)
  - [One Click Lets Them In](#one-click-lets-them-in)

book:
- S+ 7th ch9
- New S+ ch6

---

# Meow's CyberAttack - Application/Server Attacks - Privilege Escalation

---

## Privilege Escalation

- user <font color=LightSlateBlue> gaining more privileges than they should have </font>.

- perform tasks they should not be allowed to do (delete, view data…)

- often associated with bugs left in software:


- Privilege escalation: occurs when a user / process accesses elevated rights and permissions.

- Combined, rights and permissions are privileges. When attackers first compromise a system, they often have minimal privileges.
  - However, **privilege escalation tactics** allow them to get more and more privileges.
  - malware will use various privilege escalation techniques to gain more and more privileges on the user’s computer and within the user’s network.

- If users are logged on with administrative privileges, it makes it much easier for the malware to gain control of the user’s system and within the network.
  - This is one of the reasons organizations require administrators to have two accounts.
    - one account for regular use and one for administrative use.
  	- <font color=LightSlateBlue> The only time they would log on with the administrator account is when they are performing administrative work </font>.
	- This reduces the time the administrative account is in use, and makes it more difficult for the malware to use privilege escalation techniques.



Example:
- <font color=OrangeRed> Backdoor </font>
  - Backdoor that allows developer to become root user to fix something during the debugging phase.
  - After debugging is done, before the software goes live, these abilities are removed.
  - If a developer <font color=LightSlateBlue> forgets to remove the backdoor </font>, it leaves the ability for an attacker to take advantage of the system.

- <font color=OrangeRed> remote access Trojans (RATs) </font>
  - to gain access to a single system.
  - They typically have limited privileges (a combination of rights and permissions) when they first exploit a system.
  - However, they use various privilege escalation techniques to gain more and more privileges.

---

## One Click Lets Them In

- only takes one click by an uneducated user to give an attacker almost unlimited access to an organization’s network.

![Image20](/assets/img/Image20.jpg)
Figure 6.1: Steps in an attack, outlines the process APTs have used to launch attacks.

- the attacker (located in the attacker space) can be located anywhere in the world, and only needs access to the Internet.
- **The neutral space**: might be servers owned or operated by the attackers. the attackers use servers owned by others, but controlled by the attackers, such as servers in a botnet.
- **The victim space**: is the internal network of the target.

steps in an attack:
1. The attacker uses open-source intelligence to identify a target.
2. Next, the attacker crafts a <font color=LightSlateBlue> spear phishing email with a malicious link </font>.
   - might include links to malware hosted on another site and encourage the user to click the link.
   - this link can activate a drive-by download that installs itself on the user’s computer without the user’s knowledge. Cozy Bear (APT 29) used this technique and at least one targeted individual clicked the link.
   - Similarly, criminals commonly use this technique to download ransomware onto a user’s computer.
   - In other cases, the email might indicate that the user’s password has expired and the user needs to change the password or all access will be suspended. Fancy Bear (APT 28) used a similar technique.
3. The attacker sends the spear phishing email to the recipient from a server in the neutral space. This email includes a malicious link and uses words designed to trick the user into clicking it.
4. If the user clicks on the link, it takes the user to a web site that looks legitimate. This web site might attempt a drive-by download, or it might mimic a legitimate web site and encourage the user to enter a username and password.
5. If the malicious link:
   - tricked the user into entering credentials, the web site sends the information back to the attacker.
   - installed malware on the user’s system, such as a RAT, the attacker uses it to collect information on the user’s computer (including the user’s credentials, once discovered) and sends it back to the attacker.
6. The attacker uses the credentials to access targeted systems. In many cases, the attacker uses the infected computer to scan the network for vulnerabilities.
7. The attacker installs malware on the targeted systems.
8. This malware examines all the available data on these systems, such as emails and files on computers and servers.
9. The malware gathers all data of interest and typically divides it into encrypted chunks.
10. These encrypted chunks are exfiltrated out of the network and back to the attacker.
