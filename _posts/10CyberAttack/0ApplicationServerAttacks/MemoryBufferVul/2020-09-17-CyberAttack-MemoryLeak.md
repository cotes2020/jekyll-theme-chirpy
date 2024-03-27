---
title: Meow's CyberAttack - Application/Server Attacks - MemoryBuffer - Memory Leak
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack, MemoryBuffer]
tags: [CyberAttack, MemoryBuffer]
toc: true
image:
---

- [Meow's CyberAttack - Application/Server Attacks - Memory Buffer Vulnerabilities - Memory Leak](#meows-cyberattack---applicationserver-attacks---memory-buffer-vulnerabilities---memory-leak)
  - [Memory Leak](#memory-leak)

---

# Meow's CyberAttack - Application/Server Attacks - Memory Buffer Vulnerabilities - Memory Leak
 
book: Security+ 7th 
 
<font color=LightSlateBlue></font>
<font color=OrangeRed></font>

---

## Memory Leak

- a bug in a computer application 

- causes the application to consumes more and more memory the longer it runs. 

- In extreme cases,  
<font color=LightSlateBlue>the app consume so much memory that the operating system crashes</font>.

- typically caused by an application that  
<font color=LightSlateBlue>reserves memory for short-term use, but never releases it</font>. 


Example

- a web application 

- collects user profile data to personalize the browsing experience for users. 

- But collects data every time a user accesses a web page and it never releases the memory used to store the data. 

- Over time, the web server will run slower and slower and eventually need to be rebooted.
