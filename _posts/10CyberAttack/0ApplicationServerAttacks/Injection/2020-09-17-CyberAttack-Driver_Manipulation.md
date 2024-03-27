---
title: Meow's CyberAttack - Application/Server Attacks - Injection - Driver Manipulation
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack, Injection]
tags: [CyberAttack, Injection]
toc: true
image:
---

- [Meow's CyberAttack - Application/Server Attacks - Injection - Driver Manipulation](#meows-cyberattack---applicationserver-attacks---injection---driver-manipulation)
  - [Drivers](#drivers)
  - [Drivers manipulation](#drivers-manipulation)
    - [Shimming](#shimming)
    - [Refactoring](#refactoring)

---

# Meow's CyberAttack - Application/Server Attacks - Injection - Driver Manipulation

book: Security+ ch7

<font color=LightSlateBlue></font>
<font color=OrangeRed></font>

---

## Drivers

- Operating systems use <font color=OrangeRed>drivers</font> to interact with hardware devices or software components.

Example:

- print a page using Microsoft Word,
  - Word accesses the appropriate <font color=LightSlateBlue>print driver</font> via the Windows operating system.

- access encrypted data on the system,
  - the operating system typically accesses a <font color=LightSlateBlue>software driver</font> to decrypt the data so that you can view it.

---

## Drivers manipulation

- causes the driver(s) to `be bypassed altogether` or `to do what it was programmed to do—just not with the values that it should be receiving`.

- <font color=LightSlateBlue>change the data with which the driver is working</font>.

Occasionally, an application needs to support an older driver. Example:

- Windows 10 needed to be compatible with drivers used in Windows 8, but all the drivers weren’t compatible at first.

- <font color=OrangeRed>Shimming</font> provides the solution that makes it appear that the older drivers are compatible.

A <font color=OrangeRed>driver shim</font> is additional code that can be run instead of the original driver.

when a driver is no longer compatible. They Developers can:
- Shimming
- Refactoring

In the malware world, Attackers with strong programming skills can use their knowledge to manipulate drivers by either creating shims, or by rewriting the internal code.

- If the attackers can <font color=LightSlateBlue>fool the operating system into using a manipulated driver</font>, they can cause it to run malicious code contained within the manipulated driver.

---

### Shimming

- Shim: (填隙用木片；夹铁)

- a small library

- <font color=OrangeRed>write a shim</font> to provide compatibility
  - When an app attempts to call an older driver,
  - the operating system intercepts the call and redirects it to <font color=LightSlateBlue>run the shim code instead</font>.

- created to intercept API calls transparently and do one of three things
  - handle the operation itself
  - change the arguments passed
  - or redirect the request elsewhere

- Often, shims are written to support old APIs

- Conversely, shims can be written to <font color=LightSlateBlue>support a new API in an older environment (less common)</font>, give them functionality that weren’t originally written to have (which they weren’t developed, like run on OS versions).


In terms of malware:

- Shimming: involves <font color=LightSlateBlue>creating or modifying a library</font>, to <font color=OrangeRed>bypass a driver and perform a function other than the one for which the API was created</font>.

---

### Refactoring

- a set of techniques

- to <font color=OrangeRed>identify the flow and then modify the internal structure of code</font> <font color=LightSlateBlue>without changing the code’s visible behavior</font>.

- <font color=OrangeRed>Refactoring code</font> is the process of rewriting the internal processing of the code, without changing its external behavior.

- completely rewrite the driver to refactor the relevant code.
  - If the code is clunky, better rewrite the driver.
  - Usually for correct problems of software design.


In the non-malware world:
- done in order to improve the design, to remove unnecessary steps, and to create better code.


In the malware world:
- done to look for opportunities to take advantage of weak code and look for holes that can be exploited.
