---
title: Palo Alto Networks - Prisma Investigating Alerts with RQL
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

# Investigating Alerts with RQL

--


Prisma Cloud helps visualize entire cloud infrastructure and provides insights into security and compliance risks and provides you with a proprietary query language called RQL, `Resource Query Language`

## RQL Core Concepts

Prisma Cloud helps connect the dots between configuration, user activity, and network traffic data, to have the context necessary to define appropriate policies and create alert rules.

Insights of RQL
- structured query language
- to discover, detect, and respond to security incidents in cloud environment.
- easy to use and extensible.
- Helps administrators visualize their entire cloud infrastructure and provides insights into security and compliance risks
- Connects `configuration, user activity, and network traffic data` to **define appropriate policies and create alert rules**

Types of RQL Constructs
- `Config queries`
  - resource configurations and vulnerabilities in the cloud environment.
- `Event queries`
  - `search and audit console and API access events` in the cloud environment.
- `Network queries`
  - `monitor network traffic to and from` assets deployed in the cloud environment, to find previously unidentified network security risks.

The query syntax includes two main elements:
- Query type and Filter.
- The filter incorporates the API that was used to ingest the JSON attribute that defines the resource or event.
  - Combining Filters
    - combine more than one filter or attribute using `AND` and `OR`.
  - Using Operators
    - use `= , > , < , IN, and others`
  - Using the JSON Rule Attribute
    - Use the `json.rule attribute` to query or filter specific elements included in the JSON document.
    - `config where cloud.type = 'aws' AND api.name = 'aws-cloudfront-list-distributions' AND json.rule = "viewerCertificate.certificateSource CONTAINS cloudfront"`


![large](https://i.imgur.com/LfAywqp.png)

<kbd>Demo: Prisma Cloud Investigate</kbd>

![Screen Shot 2020-10-22 at 15.58.50](https://i.imgur.com/J8IMB4S.png)

![Screen Shot 2020-10-22 at 15.59.42](https://i.imgur.com/ZwFm4nG.png)

![Screen Shot 2020-10-22 at 16.00.06](https://i.imgur.com/xHHwDwO.png)

![Screen Shot 2020-10-22 at 16.00.43](https://i.imgur.com/3MSjMbV.png)


---

## Config Query

Prisma Cloud ingests various services and associated configuration data from AWS, Azure, GCP, and Alibaba cloud services.

Features of `Config Queries`
- Retrieve
  - Retrieve resource information and identify misconfigurations
- Gain
  - Gain operational insights
- Identify
  - Identify policy and compliance violations


Config Query Options
- select `“config where”` for query, number of choices available:
- `api.name​`
  - Cloud APIs are integral to the cloud platform.
  - to identify a specific configuration aspect of the resource.
- `cloud.type`
  - to narrow queries to a cloud type
- `cloud.service`
  - to narrow queries to a cloud service
- `cloud.account`
  - to specify one or more cloud accounts
- `cloud.region`
  - to narrow the cloud region
- `finding.severity` and `finding.type`
  - to identify host related security findings.
  - This option requires data that is ingested from third-party platforms that have been integrated with Prisma Cloud.
- `json.rule`
  - The `json.rule` is applied with an `operator`.
  - Examples of operators include `greater than, equals, does not equal, contains, and exist`.
- `JOIN` and `ADDCOLUMN`
  - `JOIN`: search against multiple resource types (up to three).
  - Up to three API calls are included in a query.
  - `ADDCOLUMN`: dynamically display columns for the config query results that are displayed on screen.
- `Functions`
  - performs a calculation on specific data that matches the clause contained in the function.
  - Examples of functions include `_DateTime.ageInDays`, `_IPAddress.inRange`, and `_Port.inRange`.



<kbd>Demo: RQL Config Query</kbd>

![Screen Shot 2020-10-22 at 16.07.19](https://i.imgur.com/VJgv8x5.png)


<kbd>Demo: Investigate Config Query Resources</kbd>

![Screen Shot 2020-10-22 at 16.08.11](https://i.imgur.com/0T4GIZj.png)

![Screen Shot 2020-10-22 at 16.08.45](https://i.imgur.com/78wlwUp.png)

![Screen Shot 2020-10-22 at 16.09.15](https://i.imgur.com/X5okbVB.png)

![Screen Shot 2020-10-22 at 16.09.47](https://i.imgur.com/EWDXSuE.png)

![Screen Shot 2020-10-22 at 16.09.59](https://i.imgur.com/2QwhsnS.png)

---

## Event Query

Event queries can be used to determine all root user activity without MFA, look for stolen access keys, and find account compromises.

Features of Event Queries
- Investigate
  - Detects and investigates console and API access
- Monitor
  - Monitors and gains insight into privileged activities
- Detect
  - Detects account compromise and unusual user behavior in the cloud environment

Event Query Options
- `cloud.account, cloud.region, cloud.service, cloud.type`
  - narrow the scope of the query
- `crud`
  - to search for users or entities who performed `create, read, update, or delete` operations.
- `ip`
  - to specify an IP address
- `json.rule`
  - to specify a json rule in the query
- `operation`
  - action performed by users on resources.
  - If an operation is specified, Prisma Cloud will offer a list of matches to the operation criteria.
- `user` or `role`
  - to identify a specific user or users
- `Anomaly.type`


<kbd>Demo: RQL Event Query</kbd>

![Screen Shot 2020-10-22 at 16.15.29](https://i.imgur.com/QSsIsBB.png)

![Screen Shot 2020-10-22 at 16.16.06](https://i.imgur.com/ksjdzsB.png)


---

## Network Query

Network queries can be used to discover network security risks and is currently supported only for AWS, Azure, and GCP cloud accounts.

Features of Network Queries
- Environment
  - Customers can query network events in their cloud environments.
- Detect
  - detect internet exposures and potential data exfiltration attempts
- Discover
  - discover network traffic patterns and security risks.


> this query type does not have `api.name or json.rule` as attributes.

Network Query Attributes
- Bytes, accepted.bytes, response.bytes, packets
- Dest.ip / port / publicnetwork / resource / state / country
- Source.ip / publicnetwork / resource / state / country
- IN resource where finding.severity, finding.type, finding.source, securitygroup.name, virtualnetwork.name, role
- Host vulnerability data from third party feeds:
  - Qualys
  - Tenable
  - AWS GuardDuty
  - AWS Inspector
- Protocol
- Tag


<kbd>Demo: RQL Network Query</kbd>

![Screen Shot 2020-10-22 at 16.18.44](https://i.imgur.com/JNxu9pI.png)


<kbd>Demo: Investigate Network Query Resources</kbd>

![Screen Shot 2020-10-22 at 16.28.19](https://i.imgur.com/KDxBuAU.png)

![Screen Shot 2020-10-22 at 16.28.51](https://i.imgur.com/v2rVeLX.png)

![Screen Shot 2020-10-22 at 16.29.06](https://i.imgur.com/svDOcS2.png)

![Screen Shot 2020-10-22 at 16.29.55](https://i.imgur.com/SSR2ui3.png)


---

## Advanced RQL Queries

**Operators with JSON Arrays**
- Operator `?` opens the array.
- Operator `@` represents the current item being processed.

example
- examine a particular block in the JSON object so that you are matching only that block and no others:
- `config where api.name='aws-ec2-describe-security-groups' AND json.rule='ipPermissions[?(@.fromPort==0)].ipRanges[*] contains 0.0.0.0/0'`

![Screen Shot 2020-11-04 at 00.21.37](https://i.imgur.com/yscbWu3.png)


**JOIN**
- Use JOINs to get configuration data from two different APIs by combining two different conditions.
- Use JOINs for two different APIs
  - to get configuration data from two different APIs by combining two different conditions:
  - `config where api.name=".." as X; config where api.name="..." as Y; filter "$.X... <operator> $.Y"; show (X;|Y;)`
- List EC2 instances as X
  - `config where api.name = 'aws-ec2-describe-instances' as X;`
- List subnets as Y
  - `config where api.name = 'aws-ec2-describe-subnets' as Y;`
- Set the filter
  - `filter '$.X.subnetId == $.Y.subnetId and $.Y.mapPublicIpOnLaunch is true'; show X;`
- List instances in subnets that have public IPs auto-assigned to them
  - `config where api.name = 'aws-ec2-describe-instances' as X; config where api.name = 'aws-ec2-describe-subnets' as Y; filter '$.X.subnetId == $.X.subnetId and $.Y.mapPublicIpOnLaunch is true'; show X;`
- JSON for the subnetId
  - You can display the Y results for the query and then open the JSON config for a subnet resource listed in the results table:
  - `config where api.name = 'aws-ec2-describe-instances' as X; config where api.name = 'aws-ec2-describe-subnets' as Y; filter '$.X.subnetId == $.X.subnetId and $.Y.mapPublicIpOnLaunch is true'; show Y;`



---

## Custom Policy

create new policies using RQL queries that you develop.
- create a custom policy and also use a saved search query in the custom policy.
- RQL can be used to investigate issues as they occur.
- The queries that are developed in the investigations can be saved.
- Saved queries can also be used to develop new custom policies.


<kbd>Demo: RQL Saved Query</kbd>

![Screen Shot 2020-10-22 at 16.50.41](https://i.imgur.com/tgIDutc.png)


---

## Investigate from Alerts

investigate a security policy violation from the Alerts page.

Investigation Methods
- to initiate an investigation of a security incident in Prisma Cloud.
- Use `RQL queries` from the Investigate page
- Launch an investigation from the Alert details page


<kbd>Demo: Investigate from Alerts</kbd>

![Screen Shot 2020-10-22 at 16.52.53](https://i.imgur.com/euBJle2.png)

![Screen Shot 2020-10-22 at 16.53.19](https://i.imgur.com/wFQPQOp.png)

![Screen Shot 2020-10-22 at 16.53.42](https://i.imgur.com/PBxQrg4.png)

![Screen Shot 2020-10-22 at 16.54.27](https://i.imgur.com/rSxBUb1.png)

---

## Knowledge Check


Which two types of queries does RQL support? (Choose two.)
- Audit event
- Network


A config query can start with which two expressions? (Choose two.)
- `config where cloud.region =`
- `config where api.name =`


Which option shows how to use an alert to investigate a resource with RQL?
- Click the alert, hover on the Resource Name, and click the Investigate button.

















.
