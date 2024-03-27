---
title: Meow's CyberAttack - Application/Server Attacks - MemoryBuffer - Buffer Overflows
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack, MemoryBuffer]
tags: [CyberAttack, MemoryBuffer]
toc: true
image:
---

- [Meow's CyberAttack - Application/Server Attacks - Memory Buffer Vulnerabilities - Buffer Overflows](#meows-cyberattack---applicationserver-attacks---memory-buffer-vulnerabilities---buffer-overflows)
  - [Buffer Overflows Attack](#buffer-overflows-attack)
    - [Attack example](#attack-example)
    - [prevention](#prevention)

---

# Meow's CyberAttack - Application/Server Attacks - Memory Buffer Vulnerabilities - Buffer Overflows

book: Security+ 7th ch9

<font color=LightSlateBlue></font>
<font color=OrangeRed></font>

---

## Buffer Overflows Attack

- an application <font color=OrangeRed>receives more /different input than it’s programmed to accept</font>.

- an attempt to write more data into an application’s prebuilt buffer area in order to overwrite adjacent memory, execute code, or crash a system (application).

- Buffer overflow is also referred to as <font color=OrangeRed>smashing the stack</font>.

- sends data that exceed a buffer capacity, causing an overﬂow of data. These data can be interpreted as executable code, run and give the attacker system level privileges on the web server.

- an input validation attack, Usually is the result of:
  - a programming error in the development of the software.
  - weak or nonexistent parameter checking in the processing software.


The buffer overflow exposes a vulnerability

- it doesn’t necessarily cause damage by itself.

- However, once attackers discover the vulnerability, they exploit it and overwrite memory locations with their own code.

- can cause an application
  - <font color=LightSlateBlue>to terminate</font>
    - leave the system sending the data with temporary access to privileged levels in the attacked system.
  - <font color=LightSlateBlue>To overwriting</font>: write data beyond the end of the allocated space.
    - can cause important data to be lost.

- If the attacker uses the buffer overflow to crash the system or disrupt its services, it is a DoS attack.

- More often, the attacker’s goal is to
<font color=LightSlateBlue>insert malicious code in a memory location that the system will execute</font>.
  - It’s not easy for an attacker to `know the exact memory location where the malicious code is stored`, making it difficult to get the computer to execute it. However, an attacker can make educated guesses to get close.
  - A popular method that makes guessing easier is with `nop operation (NOP, pronounced as “no-op”) commands`, written as a NOP slide or NOP sled.
  - Many Intel processors use hexadecimal 90 (x90) as a NOP command, so a string of x90 characters is a NOP sled.
  - The attacker writes a long string of x90 instructions into memory, followed by malicious code.
  - When a computer is executing code from memory and it comes to a NOP, it just goes to the next memory location.
  - With a long string of NOPs, the computer simply slides through all of them until it gets to the last one and then executes the code in the next instruction.
  - If the attacker can get the computer to execute code from a memory location anywhere in the NOP slide, the system will execute the attacker’s malicious code.


---

### Attack example

> less common than past, but still represent a large problem.

The malicious code varies.

- code to <font color=LightSlateBlue>spread a worm through the web server’s network</font>.

- code modifies the web application so that the web application tries to <font color=LightSlateBlue>infect every user who visits the web site with other malware</font>.


A buffer overflow attack includes several different elements, but they happen all at once.

- The attacker sends a single string of data to the application.

- The first part of the string causes the buffer overflow.

- The next part of the string is a long string of NOPs followed by the attacker’s malicious code, stored in the attacked system’s memory.

- Last, the malicious code goes to work.

In some cases, an attacker writes a malicious script to discover buffer overflow vulnerabilities.

- Example:
	the attacker use JavaScript to send random data to another service on the same system.


A buffer overflow attack is one that should never be successful in modern technology but still remains a great weapon in your arsenal because of poorly designed applications.

- To truly use this attack, you have to become a good computer programmer,

- many Metasploit-like tools make this much easier for you to attempt.

- In the real world, the best hackers are usually exceptional programmers—it’s just a fact of life.

Example:

1. ![Pasted Graphic 4](/assets/img/Pasted%20Graphic%204.png)

2. ![Pasted Graphic 8](/assets/img/Pasted%20Graphic%208.png)

3. sending a long string to the system to create a buffer overflow:
   - GET /index.php?
   - username=ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ

---

### prevention

<font color=OrangeRed>error-handling</font> routines and <font color=OrangeRed>input validation</font> can prevent buffer overflows, but don’t prevent them all.

- Attackers occasionally discover a bug allowing them to send a specific string of data to an application, causing a buffer overflow.

- When vendors discover buffer overflow vulnerabilities, they are usually quick to release a patch or hotfix.

- For administrator solution: Keep the systems up to date with current patches.

In addition to good coding techniques, to avoid allowing the overflow in the first place, sometimes developers can use “canaries” or “canary words.”

- canary words are known values placed between the buffer and control data. If a buffer overflow occurs, the canary word will be altered first, triggering a halt to the system.

- Tools such as `StackGuard` make use of this for stack protection.

- NOTE All of these are memory management attacks that take advantage of how operating systems store information. While canary words are good for test purposes, address space layout randomization (ASLR) and data execution prevention (DEP) are extremely common mechanisms to fight most of these attacks.
