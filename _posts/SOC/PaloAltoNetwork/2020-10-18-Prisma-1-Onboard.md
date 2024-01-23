---
title: Palo Alto Networks - Prisma Onboarding and Initial Setup
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

# Onboarding and Initial Setup

---

## Prisma Cloud Overview

Prisma Cloud
- a cloud infrastructure security solution
- a Security Operations Center (SOC) enablement tool
- to address risks and secure workloads in a heterogeneous environment (hybrid and multi-cloud) from a single console.

Cloud Security Posture Management with Prisma Cloud

![Screen Shot 2020-10-20 at 18.12.04](https://i.imgur.com/v8I7aDF.png)

1. Comprehensive <kbd>Cloud Configuration Management Database</kbd>
   1. provide a comprehensive cloud `Configuration Management Database`, the data needed for `compliance reporting` and to address `compliance violations,` `threat detection and response`, and `data security`.
2. Integration With Third-Party Applications
   1. can also integrate with additional third-party applications for outbound alert notifications,
   2. such as Splunk, Jira, and many others.
3. Visibility, Detection, and Response
   1. a cloud native security platform
   2. provides visibility, detection, and response to security threats to `public cloud accounts`.
4. Data Collection and Aggregation
   1. It accomplishes this through data collection from the cloud accounts and aggregation of that data.
5. Deployment and Tracking
   1. dynamically discovers resources that are deployed in the cloud,
   2. and tracks historical changes to those resources for auditing and forensics purposes.
6. Ingestion Through APIs
   1. `Resource configurations, user activity, network traffic logs, and host activity and vulnerabilities data` is ingested into Prisma Cloud though the `public cloud APIs`.
7. Supported Cloud Platforms
   1. Prisma Cloud currently supports Amazon Web Services, Alibaba Cloud, Azure, and Google Cloud.
   2. Third-Party Feeds
      1. There is also support for the ingest of data from third-party platforms
      2. such as Tenable and Qualys.


Prisma Cloud Compute
- Twistlock is now branded `Prisma Cloud Compute`.
- Prisma Cloud Compute can be deployed in one of two ways.
  - **SaaS Version**
    - **Prisma Cloud Compute** has been integrated into the **Prisma Cloud SaaS security platform**
    - accessible through the <knd>Compute tab</kbd>.
  - **Self-Hosted Software**
    - **Prisma Cloud Compute Edition** can be deployed as a self-hosted application.


![Screen Shot 2020-10-20 at 18.46.55](https://i.imgur.com/LGlL7xD.png)

![Screen Shot 2020-10-20 at 18.47.42](https://i.imgur.com/tXduNRe.png)

![Screen Shot 2020-10-20 at 18.48.49](https://i.imgur.com/IixaHoC.png)

![Screen Shot 2020-10-20 at 18.49.02](https://i.imgur.com/0QurNUH.png)

![Screen Shot 2020-10-20 at 18.49.20](https://i.imgur.com/pUeOALM.png)

![Screen Shot 2020-10-20 at 18.49.35](https://i.imgur.com/n0axXSN.png)

![Screen Shot 2020-10-20 at 18.50.02](https://i.imgur.com/dqf2kTd.png)

- sys admin

![Screen Shot 2020-10-20 at 18.50.55](https://i.imgur.com/VDXLJBk.png)

![Screen Shot 2020-10-20 at 18.51.12](https://i.imgur.com/xt7gsOL.png)


---

## Onboarding Public Cloud Accounts

Prisma Cloud administrators can use the cloud account onboarding with all supported cloud platforms: `AWS, Alibaba Cloud, Azure, and Google Cloud`.

requirements for each cloud provider.

### AWS requirements

1. Create a `Prisma Cloud read-only custom role` in AWS to be used to connect to AWS environment.
   1. Read-write permissions are required to monitor and protect account through auto-remediation of policy violations.
   2. to allow/authenticate Prisma Cloud to make the required API calls to cloud account for collecting the metadata for cloud resources.
2. `CloudFormation templates` are available to automate the process of creating the custom role required to add AWS account to Prisma Cloud.
3. to ingest network traffic data from cloud account.
   1. Configure `VPC Flow Logs` to monitor network traffic. Make sure the filter setting is configured for all.
   2. Configure the VPCs to `send Flow Log data to CloudWatch` so that it can be ingested by Prisma Cloud.
      1. need to Enable <kbd>trust relationship</kbd> so that the IAM role can access the CloudWatch Log group.
4. Verify that `CloudTrail` is enabled (typically enabled by default).
   1. CloudTrail is required for ingesting user and event data from AWS cloud account.
5. On the Prisma Cloud console, enter two pieces of information:
   1. The External ID: defined when the role is created. Services > IAM > Roles > Trust Relationships > Conditions.
   2. The Amazon Resource Name (ARN) for the role. Services > IAM > Roles > Trust Relationships.

> AWS Public Cloud—AWS account and AWS Organization,
> master account
> Read-Only https://s3.amazonaws.com/redlockpublic/cft/rl-read-only.template
> Read-Write (Limited) https://s3.amazonaws.com/redlock-public/cft/rl-read-andwrite.template
>
> For member accounts within AWS Organizations
> Read-Only https://s3.amazonaws.com/redlock-public/cft/rl-read-onlymember.template
> For member accounts within AWS Organizations
> Read-Write (Limited) https://s3.amazonaws.com/redlockpublic/cft/rl-read-and-writemember.template


![Screen Shot 2020-10-20 at 18.59.24](https://i.imgur.com/NTXuPfB.png)

![Screen Shot 2020-10-20 at 19.00.21](https://i.imgur.com/kpW3jrz.png)

![Screen Shot 2020-10-20 at 19.02.14](https://i.imgur.com/LFl5gvP.png)

![Screen Shot 2020-10-20 at 19.02.43](https://i.imgur.com/QRMMRkK.png)

![Screen Shot 2020-10-20 at 19.03.24](https://i.imgur.com/E2aa2cB.png)


### Azure requirements

> Collect Azure subscription information, which includes and Subscription ID and Azure Active Directory ID or Tenant ID.
> Setup `access control` for the Prisma Cloud service. Register the Prisma Cloud service in Azure by `adding the Prisma Cloud application to the Azure Active Directory`.
> Grant permissions to the Prisma Cloud application. enable permissions to monitor (read-only permission), or to monitor and protect (read-write permission).
> Configure the `Azure Network Security Groups Flow Logs` and `assign a storage account to enable Flow Log ingestion`.


### GCP requirements

> In GCP account, create a `custom role` such as Prisma cloud viewer.
> Create a `service account` and generate the required security keys.
> The service account should include the `getACL permission for read access.`
> For auto-remediation, or to write to the gcp account, `computer security admin permission` is required.
> Verify that the `Compute Engine API along with additional APIs` is defined in the documentation.
> `Associate the service account with the GCP project` that you want to monitor.
> Prisma Cloud also supports onboarding multiple GCP projects or an entire organization in a single operation.


### Alibaba requirementsAlibaba

> Permissions
> Custom Policy vs. System Policy
> Create RAM Role
> Enter Prisma Cloud Account ID
> Obtain the Alibaba Cloud Resource Name (ARN)


## Validating Cloud Permissions

![Screen Shot 2020-10-20 at 19.07.24](https://i.imgur.com/N1yDLjj.png)

---

## delete Public Cloud Accounts
