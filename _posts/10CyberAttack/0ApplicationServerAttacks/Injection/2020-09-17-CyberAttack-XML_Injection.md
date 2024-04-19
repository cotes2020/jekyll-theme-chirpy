---
title: Meow's CyberAttack - Application/Server Attacks - Injection - XML injection
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack, Injection]
tags: [CyberAttack, Injection]
toc: true
image:
---

- [Meow's CyberAttack - Application/Server Attacks - Injection - XML injection](#meows-cyberattack---applicationserver-attacks---injection---xml-injection)
	- [XML Injection](#xml-injection)
	- [prevention](#prevention)

book: S+ 7th ch9

---

# Meow's CyberAttack - Application/Server Attacks - Injection - XML injection

---

## XML Injection

> XML 指可扩展标记语言(extensible markup language), XML 被设计用来传输和存储数据。

XML injection attack:

- users <font color=OrangeRed> enter values that query XML (known as XPath) </font>> with values that take advantage of exploits
  - XPath works in a similar manner to SQL
  - does not have the same levels of access control, but taking advantage of weaknesses and return entire documents.

## prevention

- Best way to prevent: <font color=LightSlateBlue> filter input </font>>, sanitize it to make certain that it <font color=LightSlateBlue> does not cause XPath to return more data </font>> than it should.
