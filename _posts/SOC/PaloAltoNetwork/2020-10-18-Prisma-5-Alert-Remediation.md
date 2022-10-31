---
title: Palo Alto Networks - Prisma Remediating Security Issues
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

# Remediating Security Issues

--

## Remediation Core Concepts

In Prisma Cloud, an `alert` occurs when an `alert rule` has been configured that enforces `one or more policies` and for `one or more account groups`.

Resolving Open Alerts
- Resources in the targeted environment that are in violation of any policy in the alert rule will trigger the alert.
- Once the alert is asserted, open alerts can be resolved in one of three ways.
- **Manual Remediation**
  - log in to public cloud account directly and `reconfigure the resource` so that it is no longer in violation of the policy.
  - applies to all alert types and not just to config alerts.
  - It provides the steps to remediate the policy violation from cloud account console.
  - Note that no CLI command is provided for it.
- **Automatic Remediation**
  - allow Prisma Cloud to automatically remediate the security violation.
  - Auto-remediation needs to be enabled.
  - It applies only to some config policies.
  - Note that Prisma Cloud requires `write permissions` for the cloud account.
- **Guided Remediation**
  - invoke the required CLI from within Prisma Cloud.
  - the CLI command is displayed but not automatically executed,
  - the user needs to manually invoke the steps to execute the CLI command.
  - It applies only to some config policies.


Remediation Core Concepts
- the requirements for auto-remediation.
- **Read and Write Permissions**
  - Prisma Cloud does require read and write permissions to perform remediation.
  - `read-only permissions`: monitor or scan public clouds and gather the information and data regarding those public account.
  - `read-write permissions`: invoke **remediation** from the Prisma Cloud.
- **Rapid Response**
  - To facilitate rapid incident response,
  - Prisma Cloud allows remediation of cloud security alerts in cloud environments using CLI commands.
  - Prisma Cloud allows automatic remediation of Prisma Cloud System Default policies.
  - When the alert is asserted, Prisma Cloud automatically accesses public cloud account and executes the CLI commands necessary to resolve the issue.
- **Remediation CLI Commands**
  - prisma Cloud provides the CLI commands for Prisma Cloud system default policies that have a remediation CLI commands associated with them.
  - Not all policies are applicable for CLI command remediation.
  - The CLI description has the necessary permission listed.
  - may need to make sure the Prisma Cloud application has these necessary additional privileges.


Public Cloud Account Requirements
- Administrators need to give Prisma Cloud write permissions for some resources to leverage the auto-remediation feature.
- AWS: Write permissions IAM role
  - To leverage the auto-remediation feature and enable Prisma Cloud to automatically resolve alerts generated due to policy violations, need to give Prisma Cloud write permissions for some resources.
  - In AWS, create an IAM role, which is an entity that includes permissions but that isn't associated with a specific user.
  - Users from other accounts can then use that role and access resources according to the permissions.
  - do this in the AWS environment by using a CloudFormation template to easily create a read and write permission role to be used by Prisma Cloud.
- Azure: Storage Account Contributor role
  - For auto-remediation of Azure policies, the Prisma Cloud application must have a storage account contributor role at the subscription level.
  - This role only is needed for auto-remediation.
  - The action supported for this role allows to manage resources.
- GCP: Compute Security Admin role
  - In the GCP environment, can grant permissions by granting roles to a user, a group, or a service account.
  - The `compute.securityAdmin` role is required for auto-remediation.
  - This allows the role to create, modify, and delete firewall rules along with other security-related configuration changes.


<kbd>Demo: Policies that Support Remediation</kbd>

![Screen Shot 2020-10-22 at 17.12.52](https://i.imgur.com/IXGyDlO.png)

![Screen Shot 2020-10-22 at 17.13.25](https://i.imgur.com/YuoA13S.png)

---

## Manual Remediation

![original](https://i.imgur.com/16ZX9bN.png)

---

## `Guided Remediation`

In Prisma Cloud, can perform guided remediation to resolve an alert. This can be performed on policies that have the `built-in remediation option`.

Guided Remediation
- In Prisma Cloud, all open alerts are displayed by default in the <kbd>Alerts Overview page</kbd>.
- This page displays all the associated information about the alerts.
- Open alerts are displayed in the Alerts Overview page.
- Some alerts are triggered by policy violations that have built-in remediation options.
- There is an option to filter open alerts on remediable is true.
- Remediable alerts are tagged with a green check mark icon next to the policy name.


<kbd>Demo: Guided Remediation</kbd>

alert > violating resource > audit trail

![Screen Shot 2020-10-22 at 17.24.15](https://i.imgur.com/CGsFuBz.png)

![Screen Shot 2020-10-22 at 17.25.06](https://i.imgur.com/xJgxknC.png)

![extraLarge](https://i.imgur.com/JdREpKi.png)

![Screen Shot 2020-10-22 at 17.25.52](https://i.imgur.com/A5N84bP.png)

---

## Alert Rules Configured for `Auto-Remediation`

In order to invoke auto-remediation, policies that are configured for remediation also need to be incorporated into an alert rule.

Alert Rules for Auto-Remediation Overview
- To `enable automated remediation`, identify the set of policies that want to remediate automatically and verify that Prisma Cloud has the required permissions in the associated cloud environments.
- `Write Access`
  - need write access to the cloud services to execute the remediation CLI commands.
- `Resolution`
  - After remediation has been applied to the violating resource, the alert should be resolved and therefore not reappear after the next scan of the cloud accounts.
- `Auto-Remediation`
  - Prisma Cloud honors alert rules that have auto-remediation turned on.
  - These alerts will be auto-resolved.


<kbd>Demo: Alert Rule Support for Auto-Remediation</kbd>

![Screen Shot 2020-10-22 at 17.20.31](https://i.imgur.com/SlxbNED.png)


---

## Add Remediation to a Policy - `Custom Policies`

When create a new custom policy in Prisma Cloud, configure the remediation steps or commands necessary to resolve policy violations.

Adding Remediation
- to create remediation commands. Prisma Cloud will need `write access` to the necessary cloud services to execute the remediation commands.

Custom Policies
- Remediation Steps
  - **Administrators** can configure the remediation steps or commands necessary to resolve policy violations.
- CLI Commands
  - Only **Prisma Cloud system admins** can specify `CLI commands and CLI command descriptions`.


Remediation Commands
- `resourceid`: Identification of the resource on which the alert is generated
- `resourcename`: Name of the resource on which alert is generated
- `account`: Account ID of the cloud account in Prisma Cloud
- `region`: Name of the cloud region the resource belongs to


<kbd>Demo: Create a Remediable Custom Poli</kbd>

policy > new policy > config:

![Screen Shot 2020-10-22 at 17.32.34](https://i.imgur.com/5x0mZL9.png)

![Screen Shot 2020-10-22 at 17.33.06](https://i.imgur.com/bWWyXjb.png)

![Screen Shot 2020-10-22 at 17.34.28](https://i.imgur.com/08tWJje.png)

![Screen Shot 2020-10-22 at 17.34.47](https://i.imgur.com/wUA1fJi.png)


---

## Knowledge Check


CLI commands that are copied to the clipboard can be used in which two ways? (Choose two.)
- They can be manually executed at the command line for the cloud account to resolve the security violation
- They can be used to define a new policy that includes remediation

Which two requirements are needed for automatic remediation? (Choose two.)
- Alert rule that includes a policy that supports remediation
- Policy that incorporates CLI commands that can remediate a policy violation


Which action resolves an alert?
- When the user clicks the Remediate button for an open alert




.
