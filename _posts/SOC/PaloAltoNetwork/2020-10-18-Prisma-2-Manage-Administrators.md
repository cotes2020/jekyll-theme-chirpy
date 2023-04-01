---
title: Palo Alto Networks - Prisma Manage Administrators
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

# Manage Administrators

---

## Prisma Cloud Administrator Roles

Manage the individuals who can access instance by
1. defining them as `users` in the system.
2. Create `account groups`, and assign users to them.
3. Use `roles` to specify what different users and account groups can see and do.

### Role-Based Access Control

Prisma Cloud provides support for role-based access control (RBAC)
- Three Settings to Manage Access Using RBAC: **Account**, **Account Groups**, and **Roles**.

![Screen Shot 2020-10-28 at 14.47.28](https://i.imgur.com/vbo94ob.png)


**Prisma Cloud Administrators**
- different <kbd>Prisma Cloud Administrator roles</kbd>.
- Assigned Admin
  - An individual who has been assigned admin privileges
- Perform Admin tasks
  - Can log in to the system and perform admin tasks
- Tasks as per role
  - Tasks a user can perform are defined by their role

![Screen Shot 2020-10-20 at 19.13.07](https://i.imgur.com/I5YSZYs.png)


**Defining Administrator Roles**
- Every user is assigned a <kbd>role</kbd>
- may be one of the `default roles` included in Prisma Cloud,
- or a `custom role` that has been created by another administrator.
- <kbd>Roles</kbd> are associated with
  - <kbd>permission groups</kbd> that define what tasks the user may perform.
  - <kbd>account groups</kbd>, define the scope of cloud accounts that the user may access.

**Prisma Cloud Permission Groups**
- 4 <kbd>permission groups</kbd> predefined
- and a fifth permission group that combines the permissions of two of the predefined groups.



#### Roles

1. <kbd>System Admin</kbd> :
   - Full control (read/write permissions) to the service
   - create, edit, or delete account groups or cloud accounts.
   - **Only System admin** have access to `all Settings on Prisma Cloud` and can view `audit logs` to analyze actions performed by other users who have been assigned admin privileges.
2. <kbd>Account Group Admin</kbd> :
   - Full control and access to `all Prisma Cloud settings`
   - Read/write permissions for the cloud accounts and account groups to which they are granted access.
   - can only view resources deployed within the cloud accounts to which they have access.
   - Resources deployed on other cloud accounts that Prisma Cloud monitors are excluded from the search or investigation results.
4. <kbd>Account Group Read-Only</kbd> :
   - Read-only access to specified sections
   - This role `does not have permissions to modify any settings`.
3. <kbd>Cloud Provisioning Admin</kbd> :
   - Permissions to onboard and manage cloud accounts and the APIs
   - create and manage the account groups.
   - access is limited to
     - Settings > Cloud Accounts
     - and Settings > Account Groups on the admin console.
5. <kbd>Account and Cloud Provisioning Admin</kbd> :
   - This role combines the permissions of Account Group Admin and Cloud Provisioning Admin.
   - able to onboard cloud accounts, access the dashboard, manage the security policies, investigate issues, and view alerts and compliance details for the designated accounts only.
6. <kbd>Build and Deploy Security</kbd>
   1. Restricted permissions to DevOps users who need access to a subset of Compute capabilities and/or API access to run IDE, SCM and CI/CD plugins for Infrastructure as Code and image vulnerabilities scans.
   2. For example, the Build and Deploy Security role enables read-only permissions to review vulnerability and compliance scan reports on Compute and to manage and download utilities such as Defender images, plugins and twistcli.
   3. use the Build and Deploy Security role with Access key only enabled, the administrator can create one access key to use the Prisma Cloud Compute APIs.


#### Account Groups

1. Navigate to Settings > Cloud Accounts.

2. Click > Add New.

1. Give the Account Group a Name and a Description.

2. Add individual cloud accounts by using the check boxes.

Manage Account Groups
- To view and manage account groups:
- STEP 1 | Select Settings > Account Groups.
- STEP 2 | To edit the details of an Account Group, click the record and change any details.
- STEP 3 | To clone an Account Group, hover over the account group and click Clone. Cloning an account group is creating a copy of an existing account group. Cloning serves as a quick method of creating a new account group if you choose to change few details of the source account group.
- STEP 4 | To delete an Account Group, hover over the account group and click Delete.


![Screen Shot 2020-10-28 at 14.54.39](https://i.imgur.com/npCT8qn.png)



Demo: Creating a Role

`Settings > Roles > +Add New`

![Screen Shot 2020-10-20 at 19.20.12](https://i.imgur.com/GDWQMQQ.png)

![Screen Shot 2020-10-20 at 19.21.53](https://i.imgur.com/7A9LeXr.png)

![Screen Shot 2020-10-20 at 19.22.13](https://i.imgur.com/PZ4tE01.png)


Demo: Creating a User

![Screen Shot 2020-10-20 at 19.23.10](https://i.imgur.com/dkSxKzZ.png)

![Screen Shot 2020-10-20 at 19.23.29](https://i.imgur.com/28AKllv.png)

---

## Create and Manage Access Keys
- Access Keys are a secure way to `enable programmatic access` to the Prisma Cloud API, if you are setting up an external integration or automation.
- By default, only the **System Admin** has API access and can enable API access for other administrators.
- You can enable API access either when `Add Administrative Users On Prisma Cloud`, or `modify the user permissions` to enable API access.
- If you have API access, you can create up to two access keys per role for most roles; some roles such the Build and Deploy Security role can generate one access key only. When you create an access key, the key is tied to the role with which you logged in.
- create an access key for a limited time period and regenerate your API keys periodically to minimize exposure and follow security best practices.
- On the Settings > Audit Logs, you can view a record of all access key related activities such as an update to extend its validity, deletion, or a revocation.
Watch this!

![Screen Shot 2020-10-29 at 12.30.11](https://i.imgur.com/t3e92BZ.png)

![Screen Shot 2020-10-29 at 12.30.24](https://i.imgur.com/28cGDus.png)


---
---

## General Settings

Prisma Cloud General Settings
1. <kbd>Access Key</kbd>
   1. Access keys provide a secure method
   2. to enable access to the Prisma Cloud API,
   3. and can be used with external integrations that require access to the APIs,
   4. or for automation systems that may perform operations such as remediation on Prisma Cloud.
2. <kbd>IP Whitelisting</kbd>
   1. `Alert IP Whitelisting` vs `Log IP Whitelisting`
   2. If you have internal networks within global cloud infrastructure, AWS infrastructure for example,
   3. you can provide IP ranges in the form of a CIDR block so that those internal networks are whitelisted from some of the functions within Prisma Cloud.
   4. This is so that these addresses are not flagged as external IPs for analysis purposes.
3. <kbd>Licensing</kbd>
   1. Prisma Cloud administrators can view a resource count in a graphical view along with a resource trend.
   2. The information can be filtered by cloud account and by time range.


<kbd>Demo: Access Keys</kbd>

![Screen Shot 2020-10-20 at 19.31.26](https://i.imgur.com/ytcnZWA.png)

![Screen Shot 2020-10-20 at 19.32.07](https://i.imgur.com/5cCXJkW.png)


<kbd>Demo: IP Whitelisting</kbd>

![Screen Shot 2020-10-20 at 19.34.32](https://i.imgur.com/pyKSJM9.png)


<kbd>Demo: Licensing</kbd>

![Screen Shot 2020-10-20 at 19.35.36](https://i.imgur.com/patzSdB.png)



---

## sso

On Prisma Cloud, you can enable single sign-on (SSO) using an `Identity Provider (IdP)` that supports `Security Assertion Markup Language (SAML 2.0)`, such as `Okta, Azure Active Directory, or PingID`.
- You can configure only one IdP for all the cloud accounts that Prisma Cloud monitors.

To access Prisma Cloud using SSO, every administrative user requires a local account on Prisma Cloud.
- `Add Administrative Users On Prisma Cloud` to create the local account in advance of enabling SSO,
- or use `Just-In-Time (JIT) Provisioning` on the SSO configuration on Prisma Cloud to create the local account automatically.
  - With JIT Provisioning, the first time a user logs in and successfully authenticates with your SSO IdP, the SAML assertions are used to create a local user account on Prisma Cloud.
  - JIT will automatically create users in Prisma Cloud that do not already exist, based on the assertion details.


To enable SSO
- first complete the setup on the IdP.
- Then, log in to Prisma Cloud using an account with System Admin privileges to
  - configure SSO
  - redirect login requests to the IdP’s login page, so Prisma Cloud administrative users can log in using SSO.
  - After enable SSO, must access Prisma Cloud from the IdP’s portal.
- Prisma Cloud supports IdP initiated SSO, and it’s SAML endpoint supports the POST method only.


As a best practice, enable a couple administrative users with both `local authentication credentials on Prisma Cloud` and `SSO access` so that they can log in to the administrative console and modify the SSO configuration when needed, without risk of account lockout.
- Make sure that each administrator has activated their Palo Alto Networks Customer Support Portal (CSP) account using the Welcome to Palo Alto Networks Support email and set a password to access the portal.

> Also, any administrator who needs to access the Prisma Cloud API cannot use SSO and must authenticate directly to Prisma Cloud using the email address and password registered with Prisma Cloud.


Prisma Cloud provides support for single sign-on
- SAML 2.0 compatible.
- Two SSO Methods
- <kbd>Palo Alto Networks (PANW) SSO</kbd>
  - Automatically Setup
  - Uses Ping
  - No Configuration
  - SP Initiated Mode
    - It works in SP initiated mode.
    - If you go directly to tenant URL (i.e. - https://app3.prismacloud.io),
    - it will redirect you to the PANW login page to log in and then back to Prisma Cloud.
  - PANW Hub:
    - It uses the PANW hub for administrative control.
- <kbd>Prisma Cloud SSO</kbd>
  - Must Be Setup
  - IdP of Choice
  - IdP Initiated Mode
    - It works only in IdP initiated mode. This means the login is initiated using a URL that the IdP provides.
  - Bypasses PANW Hub
  - Provide Two Admin Users
    - provide two admin users who can bypass the third-party SSO.
    - This is needed in the event there are SSO issues, such as a SSL certificate expiration or an IdP problem.
  - **Just-In-Time Provisioning**



Prisma Cloud SSO Support
- Prisma Cloud supports Identity Provider (IdP)-initiated SSO
- end users must log in to Identity Provider's SSO page as the first step for authenticating to Prisma Cloud.
- There is one IdP for the entire tenant, and the admin user must have a local account on Prisma Cloud.
- You also can create just-in-time provisioning of local users.


Demo: SSO

![Screen Shot 2020-10-20 at 19.26.42](https://i.imgur.com/8C51qbY.png)

---

## View Audit Logs


![Screen Shot 2020-10-29 at 12.45.33](https://i.imgur.com/OeRtOsX.png)



---

## Define Prisma Cloud Enterprise and Anomaly Settings

### Set Up Inactivity Timeout

Specify a timeout period after which an inactive administrative user will be automatically logged out of Prisma Cloud.
- STEP 1 | Select Settings > Enterprise Settings.
- STEP 2 | User Idle Timeout


### Prisma Cloud Enterprise Settings

set the `enterprise level settings` to build standard training models for `Unusual User Activity (UEBA), alert disposition, browser, and user attribution for alerts`

![Screen Shot 2020-10-20 at 23.55.05](https://i.imgur.com/PNjubL9.png)

4 settings under the <kbd>Unusual User Activity/UEBA</kbd> panel on the bottom portion of the page.

1. <kbd>Unusual User Activity/UEBA</kbd>
   - two settings within this section.
   - <kbd>Training Model Threshold</kbd>
     - `Low, Medium, or High`
     - to define the training model that will be used for UEBA or user and entity behavior analytics.
     - Low: 25 event for 7 days
     - Medium: 100 event for 30 days
     - High: 300 event for 90 days
   - <kbd>Alert Disposition</kbd>

2. <kbd>User Idle Timeout</kbd>
   - specifies the time interval when inactive users will be logged out of the system.
   - `Custom: xx mins`

3. <kbd>Auto Enable New Default Policies of the Type</kbd>
   - allows administrators to specify policies to be enabled by default and can be configured by policy severity.
   - `Low, Medium, or High`

4. <kbd>Make Alert Dismissal Reason Mandatory</kbd>
   - on: users may dismiss an alert only after providing an informative note.

5. <kbd>Populate User Attribution In Alerts Notification<kbd>
   - User attribution tries to identify the user who made the change to a resource that resulted in a policy violation.
   - mandates that user attribution be included in the alert payloads.

### Set Up Anomaly Policy Thresholds
`to define different thresholds for anomaly detection` for **Unusual Entity Behavior Analysis (UEBA)** that correspond to policies which analyze audit events, and for **unusual network activity** that correspond to policies which analyze network flow logs.

to define your preference for when you want to alert notifications based on the severity assigned to the anomaly policy.

If you want to exclude one or more IP addresses or a CIDR block from generating alerts against Anomaly policies, see `Trusted IP Addresses on Prisma Cloud`.

**For UEBA policies:**
1. Settings > Anomaly Settings > Alerts and Thresholds.
2. Select a policy.
3. Define the Training Model Threshold.
   1. The Training Model Threshold informs Prisma Cloud on the values to use for setting the baseline for the machine learning (ML) models.
   2. For production environments, set the Training Model Threshold to High so that you allow for more time and have more data to analyze for determining the baseline
   3. For unusual user activity:
      1. Low: The behavioral models are based on observing at least 25 events over 7 days.
      2. Medium: The behavioral models are based on observing at least 100 events over 30 days.
      3. High: The behavioral models are based on observing at least 300 events over 90 days.
   4. For account hijacking:
      1. Low: The behavioral models are based on observing at least 10 events over 7 days.
      2. Medium: The behavioral models are based on observing at least 25 events over 15 days.
      3. High: The behavioral models are based on observing at least 50 events over 30 days.
4. Define Alert Disposition.
   1. when want to be notified of an alert, based on the severity of the issue —low, medium, high.
   2. The alert severity is based on the severity associated with the policy that triggers an alert.
   3. You can profile every activity by location or user activity.
      1. The `activity-based anomalies` identify any activities which have not been consistently performed in the past.
      2. The `location based anomalies` identify locations from which activities have not been performed in the past.
   4. Choose the disposition (in some cases you may only have two to choose from):
      1. Conservative:
         1. For unusual user activity—Report on unknown location and service to classify an anomaly.
         2. For account hijacking—Reports on location and activity to login under travel conditions that are not possible, such as logging in from India and US within 8 hours.
         3. Set the Alert Disposition to Conservative to reduce false positives.
      2. Moderate:
         1. For unusual user activity—Report on unknown location, or both unknown location and service to classify an anomaly.
      3. Aggressive:
         1. For unusual user activity—Report on either unknown location or service to classify an anomaly.
         2. For account hijacking—Report on unknown browser and Operating System, or impossible time travel.

> When you change Training Model Threshold or Alert Disposition the existing alerts are resolved and new ones are regenerated based on the new setting.
> It might take a while for the new anomaly alerts to show on the Alerts page.

**For unusual network activity**
---

## check

Which two types of data does Prisma Cloud provide visibility for from public cloud accounts? (Choose two.)
- resource configuration and network traffic.

Which two configuration requirements are needed when onboarding an AWS public cloud? (Choose two.)
- VPC Flow logs
- Prisma Cloud Policy and Read-Only role

---















.
