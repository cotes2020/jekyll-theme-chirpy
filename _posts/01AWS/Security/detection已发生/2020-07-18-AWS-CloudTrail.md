---
title: AWS - Security - Amazon CloudTrail (log for audit)
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, Security]
tags: [AWS, Security]
toc: true
image:
---

[toc]

---

# Amazon CloudTrail `got all the log, auditing`

- simplifies security analysis, resource change tracking, and troubleshooting.


- <font color=red> enables governance, compliance, operational auditing, and risk auditing </font> of AWS account.

- <font color=blue> enabled automatically </font> when create AWS account
  - delivers log files within 15min of account activity.
  - enable CloudTrail for all regions in your environment
    - can create a trail applies to one Region or to all Regions
    - By default, the trail applies to all AWS Regions.
    - provides a complete audit trail of all AWS services within an account
  - <font color=red> deliver log files from all regions to one S3 bucket </font>
    - By default, <font color=red> CloudTrail event log files are encrypted using S3 server-side encryption </font>


- <font color=red> tracks, records user activity and API activity </font> for all regions in AWS
  - continuously monitor, tracks <font color=blue> user activity and API usage </font>
  - provides <font color=blue> event history </font> of your AWS account activity
  - including actions taken OR API calls made via:
    - AWS Management Console.
    - AWS SDKs.
    - Command line tools.
    - Higher-level AWS services (such as CloudFormation).

---

## event

![Screen Shot 2020-08-09 at 22.24.33](https://i.imgur.com/R2zwpkX.png)

An event in CloudTrail is the record of an activity in an AWS account.

- This activity can be an action taken by a user, role, or service that is monitorable by CloudTrail.
- CloudTrail events provide a history of both API and non-API account activity
  - made through the AWS Management Console, AWS SDKs, command line tools, and other AWS services.
- types of events that can be logged in CloudTrail:
  - <font color=blue> management events </font>
  - <font color=blue> data events </font>
  - <font color=blue> insight events </font>
    - By default, trails log management events
  - Both use the same CloudTrail JSON log format.
- CloudTrail can save event history for up to 90 days.



### Management Events
- information about <font color=red> control plane/ management operations </font> performed on resources in AWS account.
- Example management events include:
  - Configuring security
    - IAM `AttachRolePolicy` API operations...
  - Registering devices
    - EC2 `CreateDefaultVpc` API operations...
  - Configuring rules for routing data
    - EC2 `CreateSubnet` API operations...
  - Setting up logging
    - CloudTrail `CreateTrail` API operations...
- Management events can also include non-API events that occur in your account.
  - For example,
  - when a user signs in to your account, CloudTrail logs the `ConsoleLogin` event.


### Data Events

- Data events are <font color=blue> disabled by default </font> when create a trail
  - must explicitly add to a trail the supported resources or resource types for which you want to collect activity.

- information about the <font color=red> data plane/resource operations </font> performed on or in a resource.
  - Data events are often high-volume activities.

- The following two data types are recorded:
  - S3 object-level API activity
    - `GetObject, DeleteObject, and PutObject` API operations...
  - Lambda function execution activity
    - the `Invoke` API...



### Insights Events
- capture unusual activity in your AWS account.
  - Insights events are disabled by default when you create a trail.
    - must explicitly enable Insights event collection on a new or existing trail.
  - If you have Insights events enabled, CloudTrail detects unusual activity,
- Insights events are logged to a different folder or prefix in the destination S3 bucket for your trail.

- Insights events provide relevant information,
  - such as the associated API, incident time, and statistics
  - can also see the type of insight and the incident time period when you view Insights events on the CloudTrail console.
  - help you understand and act on unusual activity.

- Insights events are logged only when CloudTrail detects changes in your account's API usage that <font color=red> differ significantly from the account's typical usage patterns </font>

- Examples of activity that might generate Insights events include:
  - Your account typically logs no more than 20 Amazon S3 deleteBucket API calls per minute,
    - but your account starts to log an average of 100 deleteBucket API calls per minute.
    - An Insights event is logged at the start of the unusual activity,
    - and another Insights event is logged to mark the end of the unusual activity.
  - Your account typically logs 20 calls per minute to the Amazon EC2 `AuthorizeSecurityGroupIngress` API,
    - but your account starts to log zero calls to AuthorizeSecurityGroupIngress.
    - An Insights event is logged at the start of the unusual activity, and ten minutes later, when the unusual activity ends, another Insights event is logged to mark the end of the unusual activity.



### CloudTrail Event History
- provides a viewable, searchable, and downloadable record of the past 90 days of CloudTrail events.
- gain visibility into actions taken in your AWS account in the AWS Management Console, AWS SDKs, command line tools, and other AWS services.
- You can customize your view of event history in the CloudTrail console by selecting which columns are displayed.



---

## CloudTrail configuration

### Trails
- a configuration that enables delivery of CloudTrail events to an S3 bucket, CloudWatch Logs, and CloudWatch Events.
- You can use a trail to
  - filter the CloudTrail events you want delivered,
  - encrypt your CloudTrail event log files with an AWS KMS key,
  - set up Amazon SNS notifications for log file delivery.


### Organization Trails
- a configuration
- enables <font color=red> delivery of CloudTrail events in the master account and all member accounts in an organization </font> to the same Amazon S3 bucket, CloudWatch Logs, and CloudWatch Events.
- helps define a uniform event logging strategy for your organization.
- create an organization trail
  - a trail with the name that you give it will be created in every AWS account that belongs to your organization.
  - Users with CloudTrail permissions in member accounts will be able to see this trail (including the trail ARN) when they
    - log into the AWS CloudTrail console from their AWS accounts,
    - or run AWS CLI commands such as describe-trails (although member accounts must use the ARN for the organization trail, and not the name, when using the AWS CLI).
  - but users in member accounts will not have sufficient permissions to
    - delete the organization trail,
    - turn logging on or off,
    - change what types of events are logged,
    - or otherwise alter the organization trail in any way.



















。
