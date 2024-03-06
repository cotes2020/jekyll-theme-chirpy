---
title: Palo Alto Networks - Prisma Cloud - 8
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

# Prisma Cloud - Troubleshooting and Support

---


## Overview of Troubleshooting and Support

Troubleshooting Scenarios

Prisma Cloud will scale as necessary to support an enterprise’s cloud threat defense.

Onboarding
- Onboarding public cloud accounts into Prisma Cloud

Monitor
- Using Prisma Cloud to monitor those accounts for security violations and standards compliance

Investigation
- Investigating violation and compliance issues as they occur


Prisma Cloud Troubleshooting Support
- Administrators should be aware of common Prisma Cloud issues they may encounter, as well as where they can go to get help if support from Palo Alto Networks is needed.
- Common issues
  - Common issues may involve alerts not being triggered, the use of RQL to investigate security and compliance issues.
- Monitoring deployments
  - AWS value-added services, Amazon GuardDuty and Amazon Inspector, can be used to augment the monitoring of your AWS public cloud deployments.
  - Data from these services can be ingested into Prisma Cloud.

<kbd>Demo: Using Amazon GuardDuty</kbd>


![Screen Shot 2020-10-22 at 20.09.55](https://i.imgur.com/cRgOt2P.png)

![Screen Shot 2020-10-22 at 20.10.22](https://i.imgur.com/E9I3oOB.png)

![Screen Shot 2020-10-22 at 20.10.49](https://i.imgur.com/rxkLPLa.png)

<kbd>Demo: Using Amazon Inspector</kbd>

![Screen Shot 2020-10-22 at 20.12.01](https://i.imgur.com/1hBHSbY.png)

![Screen Shot 2020-10-22 at 20.12.10](https://i.imgur.com/lAnNsan.png)



---

## Troubleshooting Common Issues

In Prisma Cloud, a common issue is that alerts are not being generated when you first onboard your public cloud accounts.

Common Onboarding Issues
- Issues with onboarding
  - These include issues with connecting to the cloud account, and ingesting resource and vulnerability data.
- Issues with alert generation
  - In order to generate alerts, you must include your cloud accounts in an `alert rule`, and verify that the `policies` associated with the alert rule are enabled.
- Issues with RQL queries
  - These can be caused by focusing the query on the wrong type of resource, or the fact that resources in a public cloud infrastructure can be ephemeral in nature and the resources may be deleted or terminated when we submit the query.


<kbd>Demo: Troubleshoot Onboarding Issues</kbd>

![Screen Shot 2020-10-22 at 20.14.44](https://i.imgur.com/8kLOQ0Y.png)

![Screen Shot 2020-10-22 at 20.15.07](https://i.imgur.com/bXOCqO8.png)

![Screen Shot 2020-10-22 at 20.16.35](https://i.imgur.com/oto5Jif.png)


<kbd>Demo: Troubleshoot Alert Issues</kbd>

alert > alert rules > `target` > `select policy` > alert notification

![Screen Shot 2020-10-22 at 20.18.19](https://i.imgur.com/C1wj5iH.png)

![Screen Shot 2020-10-22 at 20.19.11](https://i.imgur.com/ZXFbW1P.png)



<kbd>Demo: Troubleshoot RQL Query Issues</kbd>

investigate > RQL

no data or 3-party integrate tools

![Screen Shot 2020-10-22 at 20.21.40](https://i.imgur.com/66NhrIq.png)

---

## Getting Help in Prisma Cloud

Prisma Cloud Help Resources
- The Quick Start Checklist
  - These are embedded online tutorials to help get you started with Prisma Cloud.
- The Help Center
  - This includes links to `TechDOCS` and the `Prisma Cloud Live Community` page, along with What’s new information on recently released Prisma Cloud platform features.


<kbd>Demo: Access Online Help in Prisma Cloud</kbd>

![Screen Shot 2020-10-22 at 20.23.16](https://i.imgur.com/HUYAFWh.png)

![Screen Shot 2020-10-22 at 20.23.26](https://i.imgur.com/v1rToVx.png)



<kbd>Demo: Create a Help Ticket in Prisma Cloud</kbd>

![Screen Shot 2020-10-22 at 20.24.33](https://i.imgur.com/wY8AaSM.png)

![Screen Shot 2020-10-22 at 20.24.51](https://i.imgur.com/elXOB4A.png)

![Screen Shot 2020-10-22 at 20.25.09](https://i.imgur.com/iGkUpwX.png)

![Screen Shot 2020-10-22 at 20.25.27](https://i.imgur.com/YDxsOLy.png)

---

## check

Which two steps can be followed to verify that Amazon GuardDuty and Inspector logs are being ingested? (Choose two.)
- Navigate to Settings, select `Cloud Accounts`, select the account, and click `Status`.
- Navigate to Investigate and perform an RQL query for GuardDuty or Inspector data.


Which two options show possible causes for an RQL query not returning data? (Choose two.)
- An attribute of the RQL query may limit the scope of the search.
- Resources may have been deleted from the cloud account.


Which step is required to create a support ticket in Prisma Cloud?
- From the Help Center, select Other resources, select Get help, and then select Create a Support Case Now.

---


Which permission group is used to provide read-only access to Prisma Cloud?
- Account Group Read Only


Which two methods are used to access the Compute Console? 5245097
- Prisma Cloud Enterprise Edition.
- Prisma Cloud Compute Edition


Which two requirements does an alert rule need to generate alerts? (Choose two)
- one or more Account Groups + Policies


Which two requirements does a new alert rule need to support Automated Remediation (Choose two)?
- Automated Remediation is enabled for the Alert Rule.
- The policies in the alert rule include the required CLI commands for remediation.


Which two methods can be used to resolve alerts? (Choose two)
- automatically by configuring Automated Remediation in the alert rule
- accessing the public cloud account and executing the necessary CLI commands


Prisma Cloud supports the downloading of compliance reports.
- True

Alerts can be forwarded to third-party integrations in Prisma Cloud
- True

use an RQL Query expression to create a custom policy.
- True


Which Dashboard information verifies that Prisma Cloud is ingesting data? 5245097
- number of Resources


Which two platforms support outbound integration? (Choose two)
- "Splunk, Jira"


Prisma Cloud by default and with no initial setup always will generate alerts.
- False

view your public cloud resources in the Dashboard.
- True

the requirement for most API endpoint requests in Prisma Cloud?
- authentication token returned by the login API call


Prisma Cloud resides in the public cloud.
- True

Alerts can be in which two states? (Choose two)
- Dismissed + Resolved


Prisma Cloud provides support for which two compliance standards? (Choose two)
- HIPAA + GDPR


If no alerts are being triggered in Prisma Cloud, what most likely is the problem?
- Account Groups are not included in an alert rule.


Prisma Cloud can access the data generated by Amazon GuardDuty and Inspector
- True

Compute Console can monitor and protect which two types of resources?
- Containers + Hosts


RQL supports which two query types?
- Config + Network


Compute Console can be used to deploy Cloud Native firewalls?
- True


Prisma Cloud provides support for onboarding which public cloud providers?
- Google Cloud Platform


Which alert type does not use RQL?
- Anomaly

What does Prisma Cloud technical support recommend as the best way to get support?
- From the Prisma LIVE Community page click Create a Support Case Now.



.







.
