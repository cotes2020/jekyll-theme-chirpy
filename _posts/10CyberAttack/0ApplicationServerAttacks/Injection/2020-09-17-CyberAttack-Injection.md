---
title: Meow's CyberAttack - Application/Server Attacks - Injection
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack, Injection]
tags: [CyberAttack, Injection]
toc: true
image:
---

- [Meow's CyberAttack - Application/Server Attacks - Injection](#meows-cyberattack---applicationserver-attacks---injection)
  - [Injection](#injection)
  - [Protection](#protection)

book: Security+ 7th ch9

---

# Meow's CyberAttack - Application/Server Attacks - Injection

---

## Injection


One successful web application attack, <font color=LightSlateBlue>injecting malicious commands into the input string</font>.

- The objective: to pass exploit code to the server through poorly designed input validation in the application.


Many types of injection attacks can occur.

- This can occur using a variety of different methods

- <font color=OrangeRed>file injection</font>: injects a `pointer` in the web form input to an exploit hosted on a remote site.

- <font color=OrangeRed>command injection</font>: injects `commands` into the form fields instead of the expected test entry

- <font color=OrangeRed>shell injection</font>: `gain shell access` using Java or other functions


- <font color=OrangeRed>SQL injection</font>: exploit weaknesses in statements input by users.

- <font color=OrangeRed>LDAP injection</font>: exploits weaknesses in LDAP (Lightweight Directory Access Protocol) implementations.

- <font color=OrangeRed> XML injection attack </font>>: users enter values that query XML (known as XPath) with values that take advantage of exploits


## Protection

The following guidelines provide the ultimate protection for any web application:

1. <font color=OrangeRed> Input Validation </font> – do not trust any data from any source. Validate the information for content, length, format, and other factors prior to use.

2. <font color=OrangeRed> Parameterized statements </font> – avoid dynamic SQL statements. Always bind data to parameters that clearly identify the data type of the bind value.

3. <font color=OrangeRed> Business rule validation </font> – always apply business validation to input. Business validations include length, type, and expected value.

4. <font color=OrangeRed> Least privilege </font> – only allow read only access to the data as a general rule, and other access as an exception. If a form within an application simply views the data, only call the database with a read-only database user. If adding or modifying data, call the database with a modify and add database user.

5. <font color=OrangeRed> Logging </font> – always log access to data, modification of data, and, if necessary, access to the data.

6. <font color=OrangeRed> As a general rule, do not allow deletion </font> – mark record for deletion and create a separate process to delete.

7. <font color=OrangeRed> Threat modeling </font> – always threat model an application to understand access points to the database, input points to the application, and what boundaries and layers are involved through the data flow of the application.

8. <font color=OrangeRed> Error handling </font> – do not throw detailed error messages to the screen for viewing by the user. The detailed information that is included in an error message is invaluable to an attacker providing valuable clues on how to modify the attack to allow the attack to execute without error.

9. <font color=OrangeRed> Trust but verify </font> – verify and validate any requests, data, and calls into your application, even if you trust the source, because the source itself could have been compromised.
