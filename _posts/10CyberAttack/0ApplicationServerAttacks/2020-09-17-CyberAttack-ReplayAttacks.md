---
title: Meow's CyberAttack - Application/Server Attacks - Replay Attacks
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack]
tags: [CyberAttack]
toc: true
image:
---

- [Meow's CyberAttack - Application/Server Attacks - Replay Attacks](#meows-cyberattack---applicationserver-attacks---replay-attacks)
  - [Replay Attacks](#replay-attacks)

book:
- S+ 7th ch9

---

# Meow's CyberAttack - Application/Server Attacks - Replay Attacks

---

## Replay Attacks

> Replay attacks: becoming quite common

- a kind of access or modification attack.

- occur when <font color=LightSlateBlue> information is captured over a network. The attacker capture the information and replay it later </font>.

- also <font color=LightSlateBlue> occur with security certificates from systems </font>

  - the attacker will have all of the rights and privileges from the original certificate.

  - This is the primary reason that most certificates contain a unique session identifier and a time stamp.

    - If the certificate has expired, it will be rejected,

    - and an entry should be made in a security log to notify system administrators.



Example:

- <font color=OrangeRed> Kerberos </font>:

  - the attacker resubmits the certificate, hoping to be validated by the authentication system and circumvent any time sensitivity.

  - the attacker gets legitimate information, records it.

  - the attacker later relays information to gain access.

  - ![Pasted Graphic 4](/assets/img/Pasted%20Graphic%204_kqpjk952x.png)
