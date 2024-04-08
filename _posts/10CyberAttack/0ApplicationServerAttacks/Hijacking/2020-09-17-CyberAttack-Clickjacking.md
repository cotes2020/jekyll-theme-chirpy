---
title: Meow's CyberAttack - Application/Server Attacks - Hijacking - Clickjacking
date: 2020-09-17 11:11:11 -0400
categories: [10CyberAttack, Hijacking]
tags: [CyberAttack, Hijacking]
toc: true
image:
---

- [Meow's CyberAttack - Application/Server Attacks - Hijacking - Clickjacking](#meows-cyberattack---applicationserver-attacks---hijacking---clickjacking)
	- [Clickjacking](#clickjacking)
	- [Prevention](#prevention)

book: S+ 7th ch9

---

# Meow's CyberAttack - Application/Server Attacks - Hijacking - Clickjacking

---

## Clickjacking

- attacker using multiple transparent or opaque layers

- to trick a user into clicking a `button / link` on another page when they were intending to click on the top-level page.

- When an user thinks that they are clicking on the link, they are actually activating the <font color=OrangeRed> invisible button </font> to a completely different site
  - often then asking information that is collected by the miscreant for future malevolent purposes.

- most clickjacking attacks use `Hypertext Markup Language (HTML)` frames.
  - A frame `allows one web page to display another web page within an area` defined as a `frame or iframe`.

## Prevention
web developers implement new standards to defeat them.
- Most methods focus on <font color=OrangeRed> breaking or disabling frames </font>.
- ensures that attackers cannot display your web page within a frame on their web page.

Example
- the Facebook share example is thwarted by Facebook web developers adding code to their web pages preventing the use of frames.
