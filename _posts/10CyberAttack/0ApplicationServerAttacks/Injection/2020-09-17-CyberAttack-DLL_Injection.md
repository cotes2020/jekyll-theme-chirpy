---
title: Meow's CyberAttack - Application/Server Attacks - Injection - DLL injection
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack, Injection]
tags: [CyberAttack, Injection]
toc: true
image:
---

- [Meow's CyberAttack - Application/Server Attacks - Injection - DLL injection](#meows-cyberattack---applicationserver-attacks---injection---dll-injection)
  - [DLL injection](#dll-injection)

book: S+ 7th ch9

---

# Meow's CyberAttack - Application/Server Attacks - Injection - DLL injection

---

## DLL injection


DLL
- a <font color=LightSlateBlue> compiled set of code </font> that an application can use without recreating the code.

- Windows programs frequently make use of <font color=OrangeRed> dynamic linked libraries (DLLs) </font> that are loaded into the memory space of the application.

- Applications commonly use `a Dynamic Link Library (DLL)` or `multiple DLLs`.

- Example:

  - most programming languages include math-based DLLs.

  - Instead of writing the code to discover the square root of a number, a developer can include the appropriate DLL and access the square root function within it.



DLL injection:

- the malware tries to inject <font color=OrangeRed> code </font> into the <font color=LightSlateBlue> memory process space of a library </font>.

- injects a DLL into a system’s memory and causes it to run.

  - to <font color=LightSlateBlue> compromise the program </font> calling the DLL.

  - a rather sophisticated attack.

Example:

- attacker creates a DLL `malware.dll`, includes several malicious functions.

  - the attacker attaches to a running process,

  - allocates memory within the running process,

  - connects the malicious DLL within the allocated memory, executes functions within the DLL.

![Pasted Graphic 1](/assets/img/Pasted%20Graphic%201.png)
