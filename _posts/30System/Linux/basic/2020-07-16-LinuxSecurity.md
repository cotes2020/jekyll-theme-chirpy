---
title: Linux - Security of linux
date: 2020-07-16 11:11:11 -0400
categories: [30System, Basic]
tags: [Linux]
math: true
image:
---

# Security of linux

[toc]

## Linux Secure?

Linux can be configured in a way that it is as secure as every other OS out there

Linux can be configured nearly any way to be configured
- provides pitfalls if we provide secure measures incorrectly


---

## How Linux may be compromised

4 area

### **Software vulnerabilities**

Buffer overflows are still the number one software vulnerability

Linux software is not infallible

“Linus Law” states – “given enough eyeballs, all bugs are shallow” – meaning the more people you have looking at software the more it is secure and bugs are patched, however some bugs still make it through

Developers of custom software may not have luxury of testing software

Software may not be patched, use `package Management`


### **Configurations errors**

configuration pages may have enhanced privileges

easy to do something in Linux

hard to undo something in Linux

Forgetting to close ports or remove configuration pages is a common issue: 3306 for mysql.



### **Social Engineering or Users in general**

Users may make mistakes


### **Rootkits, Viruses, and Trojans**

linux designed to be least privileges, but there are still rootkits, viruses and Trojans that are developed for Linux, but not in the same ballpark as other OS’s

Moris Worm, 1988 was the first Linux worm

Linux has open source AntiVirus however
chrootkit
Rkhunter
ClamAV

















.
