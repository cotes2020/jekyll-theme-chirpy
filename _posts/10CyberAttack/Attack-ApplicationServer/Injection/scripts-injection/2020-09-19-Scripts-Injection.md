---
title: Scripts injection
# author: Grace JyL
date: 2020-09-19 11:11:11 -0400
description:
excerpt_separator:
categories: [10CyberAttack]
tags: [CyberAttack]
math: true
# pin: true
toc: true
# image: /assets/img/sample/devices-mockup.png
---

[toc]

---

# Scripts injection


XSS
- defined by the `Open Web Application Security Projection (OWASP)` as “a type of injection problem, in which malicious scripts are injected into the otherwise benign and trusted web sites.”
- According to the `Web Application Security Consortium`, XSS “is an attack technique that involves echoing attacker-supplied code into a user’s browser instance. A browser instance can be a standard web browser client, or a browser object embedded in a software product.”


CSRF
- defined by the `Open Web Application Security Projection (OWASP)` as “an attack which forces an end user to execute unwanted actions on a web application in which he/she is currently authenticated.”
- According to the `Web Application Security Consortium`, CSRF “is an attack that involves `forcing a victim to send an HTTP request` to a target destination without their knowledge or intent in order to perform an action as the victim.”



> XSS更偏向于代码实现（即写一段拥有跨站请求功能的JavaScript脚本注入到一条帖子里，然后有用户访问了这个帖子，这就算是中了XSS攻击了），
> CSRF更偏向于一个攻击结果，只要发起了冒牌请求那么就算是CSRF了。
> 条条大路（XSS路，命令行路）通罗马（CSRF马，XSRF马）。

Unlike SQL Injection, which affects any application type, CSRF and XSS affect only web-based applications and technologies.
- internal threats can be even more dangerous, have access to more resources than external attackers, which makes the combination of XSS and CSRF a lethal combination.
- In an attack scenario, an external attacker combines a CSRF attack with an XSS attack, allowing infiltration, escalation of privilege, and other gains to internal resources.
- One common form of this combination is called phishing, which utilizes email to entice a user to click a link to a malicious site that contains a CSRF attack signature along with malicious XSS in order to capture and send information or download malicious content without the unsuspecting user’s knowledge.

prevent Cross-site Scripting (XSS) flaws in software applications
- Validate and escape all information sent to a server
