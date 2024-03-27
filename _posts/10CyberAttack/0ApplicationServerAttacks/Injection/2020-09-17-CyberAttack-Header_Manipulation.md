---
title: Meow's CyberAttack - Application/Server Attacks - Injection - Header Manipulation
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack, Injection]
tags: [CyberAttack, Injection]
toc: true
image:
---

- [Meow's CyberAttack - Application/Server Attacks - Injection - Header Manipulation](#meows-cyberattack---applicationserver-attacks---injection---header-manipulation)
	- [Header Manipulation](#header-manipulation)

---

# Meow's CyberAttack - Application/Server Attacks - Injection - Header Manipulation

book:

<font color=LightSlateBlue></font>
<font color=OrangeRed></font>

---

## Header Manipulation

- Insertion of malicious data, which has not been validated, into a HTTP response header.

- Example
  - HTTP response splitting attack: exploits applications that allow a carriage return or line feed as input.
  - intercepted HTTP traffic and inserted an ASCII line that sets the referrer URL
