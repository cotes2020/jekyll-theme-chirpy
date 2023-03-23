---
title: AWS - Security - AWS Trusted Advisor
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, SecurityControl]
tags: [AWS, SecurityControl]
toc: true
image:
---

[toc]

---

# AWS Trusted Advisor

![Trusted Advisor](https://i.imgur.com/Ighxcn9.png)

- online tool

- offers recommendations for <font color=red> cost optimization, performance, security, fault tolerance and service limits </font>


- <font color=red> optimize performance and security </font>
  - how and where you can get the most impact for your AWS spend.
  - possible reduce the monthly spend and retain or increase productivity.
  - Guidance on getting the optimal performance and availability based on your requirements.
  - confidence that your environment is secure


- <font color=red> real-time guidance </font>
  - to provision resources guid following AWS <font color=blue> best practices </font> and staying within limits.
  - provides valuable guidance for architecting your AWS environment and workloads,
  - but doesn't include AWS service health information.

- auto service, during implement ells right and problems.

- The best practices that are available to all customers include:
  - Offers <font color=red> Service Limits check </font> (in the Performance category)
    - the check displays your usage and limits for some aspects of some services.
    - Business and enterprise can use all checks.

  - Security group rules that allow unrestricted access to specific ports
  - IAM use
  - Is Multi factor authentication, MFA, available on the root account
  - Amazon S3 bucket permissions
  - Amazon EBS public snapshots
  - Amazon RDS public snapshots


5 components
1. <font color=red> Cost Optimization </font>
   - resource use and makes recommendations to optimize cost by
     - eliminating unused and idle resources,
     - or by making commitments to reserved capacity.
2. <font color=red> Performance </font>
   - Improve the performance of your service by
     - checking your service limits,
     - ensuring you take advantage of provisioned throughput,
     - and monitoring for overutilized instances.
3. <font color=red> Securit </font>
   - Improve the security of your application by
     - closing gaps,
     - enabling various AWS security features,
     - and examining your permissions.
4. <font color=red> Fault Tolerance </font>
   - Increase the availability and redundancy of your AWS application by
     - taking advantage of automatic scaling, health checks, Multi-AZ deployments, and backup capabilities.
5. <font color=red> Service Limits </font>
   - checks for <font color=blue> service usage that is more than 80% of the service limit. </font>
   - Values are based on a snapshot, so your current usage might differ.
   - Limit and usage data can take up to 24 hours to reflect any changes.

---

## functionality

Within the console, you have:

- <font color=red> AWS Trusted Advisor Notifications </font>
  - stay up to date with your AWS resource deployment.
  - notified by weekly email when you opt in for this service, and it is free.

- use <font color=red> AWS Identity and Access Management, IAM </font>, to control access to specific checks or check categories.

- can retrieve and refresh Trusted Advisor results programmatically by using the <font color=red> AWS Support API </font>

- <font color=red> Action Links </font>
  - hyperlinks on items within a Trusted Advisor report.
  - takes you directly to the console, where you can take action on the Trusted Advisor recommendations.

- <font color=red> Recent Changes </font>
  - ![Screen Shot 2020-07-21 at 13.05.04](https://i.imgur.com/kndS5jA.png)
  - track recent changes of a check status on the console dashboard.
  - The most recent changes appear at the top of the list to bring attention.

- The <font color=red> Exclude Items </font> feature
  - ![Screen Shot 2020-07-21 at 13.05.58](https://i.imgur.com/SS3uaG3.png)
  - customize the Trusted Advisor report
  - can exclude items from the check result if they are not relevant.
  - can refresh individual checks or refresh all the checks at once by choosing the Refresh All button in the summary dashboard.

- refresh 5 minutes
  - A check is eligible for refresh 5 minutes after it was last refreshed.









.
