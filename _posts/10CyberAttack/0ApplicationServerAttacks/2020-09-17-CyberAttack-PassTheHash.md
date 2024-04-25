---
title: Meow's CyberAttack - Application/Server Attacks - Pass the Hash
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack]
tags: [CyberAttack]
toc: true
image:
---

- [Meow's CyberAttack - Application/Server Attacks - Pass the Hash](#meows-cyberattack---applicationserver-attacks---pass-the-hash)
  - [Pass the Hash](#pass-the-hash)

book:
- S+ 7th ch9

---

# Meow's CyberAttack - Application/Server Attacks - Pass the Hash

---

## Pass the Hash

- This attack takes advantage of a <font color=LightSlateBlue> weakness in the authentication protocol </font> (`NTLM and LanMan`)
  - in which <font color=OrangeRed> the password hash remains static from session to session until the password is changed </font>.
  - Attacker send an authenticated copy of the password hash value (along with a valid username) and authenticate to any remote server (Windows, Unix…) that is accepting LM or NTLM authentication.
  - <font color=OrangeRed> Solution: Disable NTLM </font>
