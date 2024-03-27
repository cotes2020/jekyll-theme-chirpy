---
title: Meow's CyberAttack - Application/Server Attacks - Injection
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack, Injection]
tags: [CyberAttack, Injection]
toc: true
image:
---

- [Meow's CyberAttack - Application/Server Attacks - Injection](#meows-cyberattack---applicationserver-attacks---injection)
  - [Injection](#injection)

---

# Meow's CyberAttack - Application/Server Attacks - Injection

book: Security+ 7th ch9

<font color=LightSlateBlue></font>
<font color=OrangeRed></font>

---

## Injection


One successful web application attack, <font color=LightSlateBlue>injecting malicious commands into the input string</font>.

- The objective: to pass exploit code to the server through poorly designed input validation in the application.


Many types of injection attacks can occur.

- This can occur using a variety of different methods

- <font color=OrangeRed>file injection</font>: injects a `pointer` in the web form input to an exploit hosted on a remote site.

- <font color=OrangeRed>command injection</font>: injects `commands` into the form fields instead of the expected test entry

- <font color=OrangeRed>shell injection</font>: `gain shell access` using Java or other functions


- <font color=OrangeRed>SQL injection</font>: exploit weaknesses in statements input by users.

- <font color=OrangeRed>LDAP injection</font>: exploits weaknesses in LDAP (Lightweight Directory Access Protocol) implementations.
