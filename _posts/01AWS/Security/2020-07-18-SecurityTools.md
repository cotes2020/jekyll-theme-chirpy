---
title: AWS - Security
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, CloudSecurity]
tags: [AWS, AWSSecurity]
toc: true
image:
---

[toc]

---


# AWS - Security

click for detailed note


[AWS Organizations]()
- restrict what services and actions are allowed in accounts.
- consolidate multiple AWS accounts into an organization to create and centrally manage.
- Each paying account is an independent entity and is not able to access resources of other accounts in the Organization.
- <font color=red> The billing is performed centrally on the root account </font> in the AWS Organization.



[AWS Systems Manager]()
- gives visibility and control of the <font color=red> infrastructure on AWS </font>
- provides a unified user interface, view operational data from multiple AWS services
- to <font color=red> automate operational tasks across your AWS resources </font>
￼


[AWS Secrets Manager ]()
- <font color=red> protects the secrets you use for access </font> to applications, services and IT resources.
- easily rotate, manage, and retrieve database credentials, API keys, and other secrets throughout their lifecycle.


[TAM: Technical account manager: ]()
- Only for the <font color=red> enterprise support plan </font>
- A dedicated voice in AWS to serve as your advocate.
- <font color=red> Proactive guidance and insight </font> into ways to <font color=red> optimize AWS through business and performance reviews </font>
  - Orchestration and access to the full breadth and depth of technical expertise across the full range of AWS.
  - And access to <font color=red> resources and best practice recommendations </font>


[Infrastructure Event Management]()
- A common understanding of <font color=red> event objectives and use cases </font> through pre-event planning and preparation.
- Resource recommendations and deployment guidance based on anticipated capacity needs.
- Dedicated attention of your AWS Support team during your event.
- And the ability to <font color=red> immediately scale down resources to normal operating levels post-event </font>


[Service Health Dashboard]()
- general status of AWS services,  
- shows the current status of services across regions.
- but not provide proactive notifications of scheduled activities or guidance of any kind.


[AWS Personal Health Dashboard ]()
- provides <font color=red> alerts and remediation guidance </font> when AWS is experiencing events that may impact you.
- <font color=red> personalized view </font> into the performance and availability of the AWS services underlying your AWS resources.
  - displays relevant and timely information to help manage events in progress
  - provides <font color=red> proactive notifications to help plan for scheduled activities </font>
    - forward-looking notifications.
    - can set alerts across multiple channels, including email and mobile notifications.
    - alerts are triggered by changes in the health of AWS resources,
    - giving you event visibility, and guidance to help quickly diagnose and resolve issues.
- Having an AWS account grants you access to the Personal Health Dashboard to receive alerts and remediation guidance regarding events affecting the services underlying your resources.
- Business or Enterprise support plan, also get AWS Health API for integrating health data and notifications with your existing in-house and third-party IT management tools.



[AWS Security Hub]()
- <font color=red> consolidates view of your security and compliance status </font> in the cloud.
- Unified security and compliance center



[AWS support concierge 门房:]()
- A primary contact to help manage AWS resources.
- account assistance
- Only for the <font color=red> enterprise support plan </font>
- <font color=red> non-tech billing and account level inquiries </font>
  - Personalized handling of billing inquiries, tax questions, service limits, and bulk reserve instance purchases.
  - answering billing and account questions
  - direct access to an agent to help optimize costs to identify underused resources




[Guard Duty]()
- a <font color=red> threat detection service </font>
- Designed to <font color=red> actively protect </font> the environment from threats.
- <font color=red> monitors environment, and identify malicious/unauthorized activity </font> in AWS account and workloads
  - such as <font color=blue> unusual API calls or potentially unauthorized deployments </font> that indicate a possible account compromise.
  - detects <font color=blue> potentially compromised instances or reconnaissance </font> by attackers.
  - continuously monitor and <font color=red> protect AWS accounts and workloads </font>
  - can <font color=red> identify malicious or unauthorized activities </font> in AWS accounts
- Use <font color=red> threat intelligence </font> feeds to detect threats to the environment.


[AWS Config]()
- fully-managed service
- a service assess,
- enables and simplify:

- <font color=red> security analysis </font>
  - continuously monitors and records AWS resource configurationsd
  - discover existing and deleted AWS resources
  - dive into configuration details of a resource at any point in time.

- <font color=red> change management </font>
  - <font color=blue> audit, evaluate, and monitor changes and Aconfigurations </font> of AWS resources.
  - track resource inventory and changes.
  - review changes in configurations and relationships between AWS resources
  - dive into detailed resource configuration histories,
  - provides an <font color=blue> AWS resource inventory, configuration history, and configuration change notifications </font> to <font color=red> enable security and regulatory compliance </font>
  - allows to <font color=blue> automate the evaluation of recorded configurations </font> against desired configurations.

- <font color=red> compliance auditing </font>
  - determine your overall compliance against rules/configurations specified in your internal guidelines.
- and <font color=red> operational troubleshooting </font>


## audit

[CloudTrail]() <font color=blacko> got all the log, auditing </font>
- simplifies security analysis, resource change tracking, and troubleshooting.
- <font color=red> enables governance, compliance, operational auditing, and risk auditing </font> of AWS account.
  - <font color=blue> enabled automatically </font> when create AWS account
  - delivers log files within 15min of account activity.
  - enable CloudTrail for all regions in your environment
    - can create a trail applies to one Region or to all Regions
    - By default, the trail applies to all AWS Regions.
    - provides a complete audit trail of all AWS services within an account
  - CloudTrail can deliver all log files from all regions to one S3 bucket.
    - By default, <font color=red> CloudTrail event log files are encrypted using S3 server-side encryption </font>
- <font color=red> continuously monitor, tracks user activity and API usage </font> for all regions in AWS
  - provides event history of your AWS account activity
  - including actions taken OR API calls made via:
    - AWS Management Console.
    - AWS SDKs.
    - Command line tools.
    - Higher-level AWS services (such as CloudFormation).


[CloudWatch Logs]() <font color=blacko> collect log, create alarm, does not debug or log errors </font>
- monitor, collect, store, and access logs from resources, applications, and services in <font color=red> near real-time </font>
  - <font color=blue> Basic monitoring collects metrics every 5min </font>
  - <font color=blue> detailed monitoring collects metrics every 1min </font>
- <font color=red> collect and track metrics, collect and monitor log files, and set alarms.  </font>
  - <font color=blue> Compute </font> (EC2 insatnces, autoscaling groups, elastic load balancers, route53 health checks)
    - <font color=blue> CPU, Disk, Network utilization, and others. </font>
    - <font color=blue> aggregate 聚集 logs from your EC2 instance.  </font>
    - <font color=blue> centrally upload logs from all the servers. </font>
    - <font color=blue> Content Delivery </font> (EBS Volumes, Storage Gateways, CloudFront)
  - <font color=blue> Storage, CloudTrail, Lambda functions, and Amazon SQS queues </font>
  - <font color=blue> allow real-time monitoring as well as adjustable retention. </font>
- providing a unified view of AWS resources, applications and services that run on AWS, and on-premises servers.
- actionable insights to monitor applications, respond to system-wide performance changes, and optimize resource utilization to get a view of your overall operational health.



[AWS Trusted Advisor]() <font color=blacko> what should use </font>
- <font color=red> optimize performance and security </font>
- <font color=red> real-time guidance </font> to provision 提供 resources guid following AWS <font color=red> best practices </font> and staying within limits.
- auto service, during implement ells right and problems.
- provides valuable guidance for architecting your AWS environment and workloads, but doesn't include AWS service health information.
- offers recommendations for <font color=red> cost optimization, performance, security, fault tolerance and service limits </font>
- Offers a Service Limits check (in the Performance category)
  - the check displays your usage and limits for some aspects of some services.
  - Business and enterprise can use all checks.










.
