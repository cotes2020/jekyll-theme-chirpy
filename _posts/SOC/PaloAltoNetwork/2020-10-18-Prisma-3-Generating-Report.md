---
title: Palo Alto Networks - Prisma Generating Report
# author: Grace JyL
date: 2020-10-18 11:11:11 -0400
description:
excerpt_separator:
categories: [SOC, PaloAlto]
tags: [SOC, Prisma]
math: true
# pin: true
toc: true
image: /assets/img/note/prisma.png
---

[toc]

---

# Prisma Cloud - Generating Report

--

## Generate and Download Compliance Reports

Available Compliance Reports in Prisma Cloud
- Unlike the Asset Inventory Dashboard that aggregates all your resources and displays the pass and fail count for all monitored resources,
- the Compliance Dashboard only displays the results for monitored resources that match the policies included within a compliance standard.


Compliance Overview
- Compliance Posture
  - `Gain risk oversight` across all the supported cloud platforms and `gauge the effectiveness of the security processes and controls` you have implemented.
- Compliance Standards
  - For a complete list of the compliance standards supported by Prisma Cloud, reference the Prisma Cloud Administrators Guide.


![Screen Shot 2020-10-21 at 01.24.31](https://i.imgur.com/PUhANEC.png)

![Screen Shot 2020-10-21 at 01.26.05](https://i.imgur.com/tRDmFT4.png)

![Screen Shot 2020-10-21 at 01.26.19](https://i.imgur.com/CeeyT6H.png)

![Screen Shot 2020-10-21 at 01.26.37](https://i.imgur.com/XlKSoKJ.png)

---

## Compliance Standards Reports

- create compliance reports based on a `cloud compliance standard` for immediate online viewing or download, or schedule recurring reports, monitor compliance to the standard over time.

- Compliance Standards
  - Prisma Cloud also supports the downloading of reports on the details of the compliance standards configured in environment.
  - This includes the name of the standard, and the requirements, sections, and descriptions that define the standard.

- Custom Compliance Standards
    - Prisma Cloud also supports creating a custom compliance standard.
    - Once a new custom compliance standard has been created, policies can then associated to the new standard.
    - The policies that are associated to the new standard can be out of the box default policies, or custom policies that you define.

Compliance > Standards

<kbd>Demo: Create a Custom Compliance Standard</kbd>
- <kbd>Compliance</kbd> > <kbd>Standards</kbd> > <kbd>add New</kbd>
- Default: no Policy

![Screen Shot 2020-10-21 at 01.32.25](https://i.imgur.com/r6QWSa2.png)

![Screen Shot 2020-10-21 at 01.32.35](https://i.imgur.com/P271Joo.png)

![Screen Shot 2020-10-21 at 01.32.55](https://i.imgur.com/XldktU3.png)


<kbd>Demo: Add a Policy to a Custom Compliance Standard</kbd>

![Screen Shot 2020-10-21 at 01.33.45](https://i.imgur.com/ULPGpRt.png)

![Screen Shot 2020-10-21 at 01.34.01](https://i.imgur.com/L4vG3eI.png)

![Screen Shot 2020-10-21 at 01.34.23](https://i.imgur.com/bKnjAAY.png)

![Screen Shot 2020-10-21 at 01.34.41](https://i.imgur.com/XZLzD44.png)


![Screen Shot 2020-10-21 at 01.35.04](https://i.imgur.com/a1fybHs.png)


<kbd>Demo: Create a Custom Policy</kbd>


![Screen Shot 2020-10-21 at 01.35.34](https://i.imgur.com/aMk6dJG.png)

![Screen Shot 2020-10-21 at 01.36.22](https://i.imgur.com/OEmeUz0.png)

![Screen Shot 2020-10-21 at 01.36.50](https://i.imgur.com/6vOtq6m.png)

![Screen Shot 2020-10-21 at 01.37.16](https://i.imgur.com/1Icbn2J.png)

![Screen Shot 2020-10-21 at 01.37.27](https://i.imgur.com/YZqkI1h.png)

---

## Alerts Report

Prisma Cloud correlates `configuration data with user behavior and network traffic` to provide context around misconfigurations and threats, in the form of actionable alerts.

As soon as associate the account groups with an active alert rule, Prisma Cloud generates an alert when it detects a violation in a policy that is included in the alert rule.


![Screen Shot 2020-10-21 at 01.40.39](https://i.imgur.com/YSy1bKo.png)

Alert Statuses
- 4 alert statuses in Prism Cloud:
- open, resolved, snoozed, and dismissed.
- Open
  - Prisma Cloud identified a policy violation that triggered the alert and the violation has `not yet been resolved`.
- Resolved
  - Alerts transition to Resolved when the issue that caused the policy violation is resolved.
  - Alerts can change to Resolve due to a change in the `policy` or `alert rule` that triggered the alert.
  - A resolved alert can transition back to the Open state.
- Snoozed
  - A Prisma Cloud administrator temporarily dismissed an alert for a specified time period.
  - When the timer expires, the alert is automatically in an Open or Resolved state.
- Dismissed
  - A Prisma Cloud administrator manually dismissed the alert.
  - Dismissed alerts can be manually re-opened, if needed.

![Screen Shot 2020-10-21 at 01.42.57](https://i.imgur.com/scUCwE0.jpg)


---

## Audit Logs Report

Audit Logs section enables companies to prepare for such audits and demonstrate compliance.

The Audit logs section lists out the actions performed by the users of the system.

- Who: specifies the user who performed the action.
- When: provided by Timestamp.
- Where: the source IP address of the user.
- What: provides the details of what has been updated, created, or deleted.

![Screen Shot 2020-10-21 at 01.45.14](https://i.imgur.com/ARnsxeE.png)

![Screen Shot 2020-10-21 at 01.45.52](https://i.imgur.com/PeTHHux.png)

![Screen Shot 2020-10-21 at 01.46.11](https://i.imgur.com/5pJypwe.png)

---

## check

How are compliance reports generated in Prisma Cloud?
- Compliance tab -> select the report to download in the Reports section

How can a user associate custom policies to a compliance standard?
- When creating a new policy, select one or more compliance standards for the new rule

An Alert report can be downloaded in which file format?
- `.CSV`





"groups": {},
"toPort": 15000,
"fromPort": 80,
"ipRanges": {
  "items": [
    {
      "cidrIp": "10.0.0.0/8"
    }
  ]
},
"ipProtocol": "tcp",
"ipv6Ranges": {},
"prefixListIds": {}






.
