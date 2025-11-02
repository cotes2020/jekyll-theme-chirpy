---
title: Open-Source Software (OSS)
# author: Grace JyL
date: 2020-09-28 11:11:11 -0400
description:
excerpt_separator:
categories: [14AppSec, SecureSoftwareDevelopment]
tags: [SecureSoftwareDevelopment, OSS]
math: true
# pin: true
toc: true
image: /assets/img/note/tls-ssl-handshake.png
---

# OSS

- [OSS](#oss)
  - [Open-Source Software (OSS)](#open-source-software-oss)

---

## Open-Source Software (OSS)

This is a great starting point for notes! I can structure this information for you using the Outline Method, which is perfect for lectures like this.
Here are your structured notes on Open Source Software (OSS) and its Security Implications:


> - OSS (Open Source Software): Software with source code made available for use and modification.
> - Dependencies: External pieces of software (often other OSS projects) that a main project relies on to function.
> - Zero-Day Vulnerability: A software flaw unknown to those who should be interested in its mitigation (vendor, etc.) that is exploited by attackers before the vendor can develop a patch.
> - Application Security (AppSec): The process of developing, adding, and testing security features within applications to prevent vulnerabilities.

- What is Open Source Software (OSS)?
  - Definition: Software that can be used and integrated into new features or products to avoid reinventing the solution.
    - Saves hours upon hours of hard work.
    - A common solution for modern development.
  - Licensing: Software is only considered OSS if it is licensed under certain types of acceptable licenses.
    - Source of Truth: The generally recognized source of truth for OSS is the Open Source Initiative (OSI).

- Example of an OSS Project: Pandas
  - Project: Pandas (for the Python programming language).
  - Purpose: A powerful library for data analysis and manipulation.
  - License Example: The Pandas project uses an open source license (specifically, the BSD three-clause license).
  - Usage: You can pull this project into your own software to use its functionality.

- The Concept of Dependencies
  - Definition: OSS projects often use many other open-source projects within them. These are called dependencies.
    - Example: Looking at the environment.yaml file for a large project shows over 100 lines of dependencies.
  - Dependency Example (Python dateutil):
    - If your software needs to manage dates and times, you can simply add a separate project like python-dateutil as a dependency in your own code.
- The Downside: Security Vulnerabilities
  - The Problem: When you pull in a project as a dependency, you create a reliance on it.
  - Vulnerability Inheritance: If the dependency project has a security vulnerability, your software can end up inheriting that vulnerability.
    - This happens all the time.
  - Real-World Example (Log4j):
    - Log4j is an open-source software used globally.
    - When a zero-day vulnerability was discovered in Log4j, it affected millions of different applications that relied on it as a dependency.

- Solutions and Future Focus
  - Outlook: It is not "all doom and gloom."
  - Mitigation: There are ways to account for and manage this issue.
  - Importance: Managing these risks is becoming a more and more important part of application security.







.
