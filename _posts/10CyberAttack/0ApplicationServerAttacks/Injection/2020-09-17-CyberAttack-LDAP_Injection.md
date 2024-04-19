---
title: Meow's CyberAttack - Application/Server Attacks - Injection - LDAP Injection
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack, Injection]
tags: [CyberAttack, Injection]
toc: true
image:
---

- [Meow's CyberAttack - Application/Server Attacks - Injection - LDAP Injection](#meows-cyberattack---applicationserver-attacks---injection---ldap-injection)
  - [LDAP Injection](#ldap-injection)
  - [Example](#example)
  - [prevention](#prevention)

book: CEH Prep ch6

---

# Meow's CyberAttack - Application/Server Attacks - Injection - LDAP Injection

---

## LDAP Injection

- exploits weaknesses in <font color=LightSlateBlue> LDAP (Lightweight Directory Access Protocol) </font>> implementations.

- an attack that exploits applications that <font color=LightSlateBlue> construct LDAP statements based on user input </font>>. it exploits nonvalidated web input that passes LDAP queries.

- occur when the user’s input is <font color=LightSlateBlue>not properly filtered</font>
  - the result can be executed commands, modified content, unauthorized queries.

- One of the most common uses of LDAP is associated with user information.
  - Numerous applications exist
  - users find other users by typing in a portion of their name.
  - queries looking at the `cn` value or other fields (for department, home directory…)
  - feed unexpected values, finding employee information equates to finding usernames and values relates to passwords.

- if a web application takes whatever is entered into the form field and passes it directly as an LDAP query, an attacker can inject code to do all sorts of stuff. You’d think this kind of thing could never happen, but you’d be surprised just how lazy a lot of code guys are.

## Example

- a web application allows managers to pull information about their projects and employees by logging in, setting permissions, and providing answers to queries based on those permissions.

- Manager Matt logs in every morning to check on his folks by `entering his username and password` into two boxes on a form, and his login is parsed into an LDAP query (to validate who he is). The LDAP query would basically says, “Check to see whether the username Matt matches the password MyPwd! If it’s valid, login is successful and off he goes.”
- In an LDAP injection attack, the attacker changes what’s entered into the form field by adding the characters <font color=OrangeRed> )(&) </font> after the username and then providing any password.

- Because the `&` symbol ends the query, only the first part—“check to see whether Matt is a valid user”—is processed and, therefore, any password will work.

- The LDAP query looks like this in the attack:
  - This basically says, “Check to see whether you have a user named Matt. If he’s there, cool—let’s just let him do whatever he wants.”

![page179image36289568](/assets/img/page179image36289568.jpg)

- While there’s a lot of other things you can do with this, I think the point is made; don’t discount something even this simple because you never know what you’ll be able to find with it.

## prevention

- Best way to prevent; filter input, use a <font color=LightSlateBlue> validation scheme </font>> to make certain that queries do not contain exploits.
