---
title: Software Composition Analysis (SCA)
# author: Grace JyL
date: 2020-09-28 11:11:11 -0400
description:
excerpt_separator:
categories: [14AppSec, SecureSoftwareDevelopment]
tags: [SecureSoftwareDevelopment, SCA]
math: true
# pin: true
toc: true
image: /assets/img/note/tls-ssl-handshake.png
---

# SCA

- [SCA](#sca)
  - [Software Composition Analysis (SCA)](#software-composition-analysis-sca)
  - [SCA Scanning Tools](#sca-scanning-tools)

---

## Software Composition Analysis (SCA)

SCA helps verify that third-party components:

- like libraries, frameworks, etc... don't have vulnerabilities in them
- do not have vulnerabilities that will affect our software

This can be done manually or (more likely) with automated tooling

---

## SCA Scanning Tools

SCA tooling typically involves a combination of features that can help us with:

- Container Scanning
- License Scanning
- Dependency Scanning
- Etc...

SCA tools maintain a `list of known vulnerable components`

- scan your code repository on a regular basis
- have them run a scan each time you're writing new code as part of your deployment pipeline

