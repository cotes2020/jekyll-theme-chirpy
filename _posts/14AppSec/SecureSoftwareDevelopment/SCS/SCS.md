---
title: Supply Chain Security (SCS)
# author: Grace JyL
date: 2020-09-28 11:11:11 -0400
description:
excerpt_separator:
categories: [14AppSec, SecureSoftwareDevelopment]
tags: [SecureSoftwareDevelopment, SCS]
math: true
# pin: true
toc: true
image: /assets/img/note/tls-ssl-handshake.png
---

# SCS

- [SCS](#scs)
  - [Supply Chain Security (SCS)](#supply-chain-security-scs)
  - [Case Study: The SolarWinds Attack](#case-study-the-solarwinds-attack)

---

## Supply Chain Security (SCS)

Defining the Supply Chain

- Core Concept: A supply chain includes everything an application or organization relies on to function properly and deliver its solutions.

- Components: A supply chain includes:
  - External suppliers and vendors
  - Logistics and transportation
  - Software licenses (even basic ones)

- Software Context: Anytime an application uses an outside organization to function, that organization is part of the supply chain.
  - Payment Processors: Services like PayPal or Stripe.
  - Cloud Providers: The vendor running your application (e.g., AWS, Azure, Google Cloud).
    - Note: The cloud provider itself has its own supply chain (server vendors, utilities, security for data centers, etc.).

- Focus: While a supply chain can be physical, the focus here is on the software side.

Importance and Impact on Security

- Massive Impact: The supply chain has a massive impact on application security and the CIA Triad.
  - Vulnerability Risk: Dependencies (external software) are part of the supply chain, and they introduce vulnerabilities (affecting Confidentiality and Integrity).
  - Availability Risk: A cloud provider experiencing an outage affects your system's Availability.

- Attack Vector: Attackers can target every single step along the supply chain in many different ways.

## Case Study: The SolarWinds Attack

- Summary: A very serious, real-world supply chain breach that compromised organizations and governments.
- Product: SolarWinds Orion, a performance monitoring platform used to manage and optimize IT infrastructure.
- Attack Method (2021):
  - Orion had to be installed and running on the servers it monitored.
  - Malicious actors found a way to inject their own code into the legitimate Orion update process.
  - Users downloading the official update were unknowingly installing malicious code on their servers.
- Scope of Compromise:
  - Approximately 100 companies and a dozen government agencies were compromised.
  - Victims included Microsoft, Intel, Cisco, and U.S. government departments (Treasury, Justice, Energy, and the Pentagon, according to NPR).
