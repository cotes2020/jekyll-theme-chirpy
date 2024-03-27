---
title: Meow's CyberAttack - Application/Server Attacks - Injection - Driver Manipulation 
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack, Injection]
tags: [CyberAttack, Injection]
toc: true
image:
---

- [Meow's CyberAttack - Application/Server Attacks - Injection - Driver Manipulation](#meows-cyberattack---applicationserver-attacks---injection---driver-manipulation)
  - [Driver Manipulation](#driver-manipulation)
    - [Drivers manipulation](#drivers-manipulation)
    - [Shimming](#shimming)
    - [Refactoring](#refactoring)

---

# Meow's CyberAttack - Application/Server Attacks - Injection - Driver Manipulation 

book: Security+ ch7
 
<font color=LightSlateBlue></font>
<font color=OrangeRed></font>

---

## Driver Manipulation 

Drivers:

- Operating systems use drivers to interact with hardware devices or software components. 

- Example:

- print a page using Microsoft Word, 
  - Word accesses the appropriate print driver via the Windows operating system. 

- access encrypted data on your system, 
  - the operating system typically accesses a software driver to decrypt the data so that you can view it.

### Drivers manipulation

- causes the driver(s) to be bypassed altogether or to do what it was programmed to do—just not with the values that it should be receiving. 

- change the data with which the driver is working. 



Occasionally, an application needs to support an older driver. 

- Example:

- Windows 10 needed to be compatible with drivers used in Windows 8, but all the drivers weren’t compatible at first. 

- Shimming provides the solution that makes it appear that the older drivers are compatible.


A driver shim is additional code that can be run instead of the original driver. 
when a driver is no longer compatible. They Developers can:

### Shimming 

- Shim: (填隙用木片；夹铁)
  - a small library
  - write a shim to provide compatibility
  - When an app attempts to call an older driver, 
  - the operating system intercepts the call and redirects it to run the shim code instead. 

  - created to intercept API calls transparently and do one of three things: 
  - handle the operation itself, 
  - change the arguments passed, 
  - or redirect the request elsewhere. 

- Often, shims are written to support old APIs 

- Conversely, shims can be written to support a new API in an older environment (less common), give them functionality that weren’t originally written to have (which they weren’t developed, like run on OS versions). 

- In terms of malware:

- Shimming: involves creating or modifying a library, to bypass a driver and perform a function other than the one for which the API was created.


### Refactoring 

- a set of techniques

- to identify the flow and then modify the internal structure of code without changing the code’s visible behavior. 

- Refactoring code is the process of rewriting the internal processing of the code, without changing its external behavior.

- completely rewrite the driver to refactor the relevant code. 
  - If the code is clunky, better rewrite the driver.
  - Usually for correct problems of software design.


- In the non-malware world:
  - done in order to improve the design, to remove unnecessary steps, and to create better code. 

- In the malware world:
  - done to look for opportunities to take advantage of weak code and look for holes that can be exploited.


Attackers with strong programming skills can use their knowledge to manipulate drivers by either creating shims, or by rewriting the internal code. 

- If the attackers can fool the operating system into using a manipulated driver, they can cause it to run malicious code contained within the manipulated driver.

