---
title: AWS - IdenAccessManage - AWS Organizations and SCP
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, IdenAccessManage]
tags: [AWS, IdenAccessManage]
toc: true
image:
---

[toc]

---

# AWS Organizations

- an account management service.

![xFQwzro](https://i.imgur.com/naqy9zK.png)
￼
AWS Organizations enables you to:

- **policy-based account management**:
  - Create <font color=red> service control policies (SCPs) </font> that centrally control AWS services across multiple AWS accounts.


- **group-based account management**:
  - <font color=red> Create groups of accounts and then attach policies to a group </font> to ensure that the correct policies are applied across the accounts.


- **Simplify account management by using application programming interfaces (APIs)** 
  - automate the creation and management of new AWS accounts.
  - Reduced admin overhead


- **Simplify the billing process**
  - set up single payment method for all AWS accounts in organization.
  - <font color=blue> consolidated billing </font>:
    - combined view of charges off all accounts, take advantage of pricing benefits from aggregated usage. 
    - Reserved purchases can be used by member accounts.
    - provides a central location to manage billing across all AWS accounts, and benefit from volume discounts.



---


## benefits of using AWS Organizations

- to <font color=red> consolidate multiple AWS accounts into an organization and centrally manage </font>
  - An OU is a container for accounts with a root.
    - on OU can have different account inside.
    - An OU can contain other OUs.
    - An OU can have only one parent
    - <font color=red> account can be a member of exactly one OU, 1v1 </font>

  - An organization has one master account along with zero or more member accounts.
    - each account can be located directly in the root, or placed in one of the OUs in the hierarchy.
    - The root is the parent container for all the accounts for your organization.


- <font color=red> Organization permissions overrule account permissions </font>. hierarchical grouping of accounts
  - group accounts into organizational units(OUs) and attach different access policies to each OU.
  - integrates with and supports IAM.
  - provides <font color=blue> service control policies (SCPs) </font>
  - centrally apply policy-based controls across multiple accounts in the AWS Cloud.
    - administrator of the master account of an organization, can restrict which AWS services the users and roles in each member account can access.
    - apply policy to the root, it applies to all the organizational unit and accounts in the organization
    - attach a policy to one of the nodes in the hierarchy, it flows down and it affects all the branches and leaves.
    - This restriction <font color=blue> overrides the administrators of member accounts in the organization </font>
      - When Organizations blocks access to a service or action for a member account,
      - a user or role in that account can't access any prohibited service or action,
      - even if an administrator of a member account explicitly grants such permissions in an IAM policy.
  - Integration and sepport for AWS IAM.
  - can attach a policy to an account to apply controls to only that one account
    - IAM provides granular control over users and roles in individual accounts.
    - Organizations expands that <font color=blue> control to the account level </font>, control over what users and roles in an account or a group of accounts can do.
    - The resulting permissions are the logical intersection of what is allowed by Organizations at the account level, and what permissions are explicitly granted by IAM at the user or role level within that account.
    - the user can access only what is allowed by both <font color=blue> Organizations policies </font> and <font color=blue> IAM policies </font>
    - If either blocks an operation, the user can't access that operation.


- set up **role switching** functionality
  - newly linked account
  - An IAM user from the other account not necessary because your current IAM user will be displayed in the member account and the permissions are assumed via the member accounts IAM role.
  - `OrganizationAccountAccessRole` as the role
  - The account ID of the member account is needed when switching roles to the linked account.
  - The display name of the role will help you identify the account you are in.


- <font color=red> Consolidated billing and account management </font>
  - better meet the budgetary, security, and compliance needs of your business.


- <font color=red> Organizations integrate with other Amazon web services </font>
  - enable <font color=blue> select Amazon web services </font> to access accounts in organization and perform actions on the resources in the accounts.
  - For this service to work, all accounts in an organization have a <font color=blue> service-linked role </font> that enables AWS Organizations to create other service-linked roles.
    - These other roles are required by the AWS services that you configure to perform organizational-level tasks.
    - When you configure another service and authorize it to integrate with your organization, Organizations creates an <font color=blue> IAM service-linked role </font> in each member account.
      - The additional service-linked role:
        - predefined IAM permissions
        - allow the other service to perform specific tasks in your organization's accounts.
        - come with policies
        - enable the specified service to perform only tasks that are required by your configuration choices.


- Data replication with eventual consistency 最终一致性, distributed computing model,
  - Any change that you make in Organizations takes time to become visible from all possible endpoints.
  - Some of the delay results from the time it takes to send the data from server to server, or from replication zone to replication zone.

- uses caching to improve performance
  - but in some cases this can add time.
  - The change might not be visible until the previously cached data times out.

- Design your global applications to account for these potential delays and ensure that they work as expected, even when a change made in one location is not instantly visible at another.

---

### organization master account
- can
  - create accounts in the organization,
  - invite other existing accounts to the organization,
  - remove accounts from the organization,
  - manage invitations,
  - and apply policies to entities within the organization, such as roots, OUs, or accounts.
- An account can be a member of only one organization at a time.
- has the responsibilities of a payer account
  - responsible for paying all charges that are accrued by the member accounts.
- If you previously had a Consolidated Billing family of accounts,
  - your accounts were migrated to a new organization in AWS Organizations
  - the payer account in your Consolidated Billing family has become the master account in your organization.
  - All linked accounts in the Consolidated Billing family become member accounts in the organization, and continue to be linked to the master account.
  - Your bills continue to reflect the relationship between the payer account and the linked accounts.

---

### invitation
- the process of asking another account to join your organization.

- invitation can be issued <font color=blue> only by the organization's master account </font>, and it’s extended to either the account ID or email address that is associated with the invited account.
  - After the invited account accepts an invitation, it becomes a member account in the organization.

- Invitations also can be sent to all current member accounts when the organization needs all members to approve the change from `supporting only consolidated billing features` to `supporting all features in the organization`.

- Invitations work by accounts exchanging <font color=red> handshakes </font>
  - might not see handshakes in AWS Organizations console,
  - if use the AWS CLI, or AWS Organizations API must work directly with handshakes
  - A handshake
    - a multi-step process
    - exchanging information between two parties.
    - One of its primary uses in AWS Organizations is to serve as the underlying implementation for invitations.
    - Handshake messages are passed between and responded to by the <font color=blue> handshake initiator and the recipient </font> so that it ensures that both parties always know the current status.
    - Handshakes are also used when you change the organization from `supporting only consolidated billing features` to `supporting all features that Organizations offer`.



---


# SCP Service Control Policy

> The specified actions from an attached SCP affect all IAM users, groups, and roles for an account,
> including the AWS account root user !!!!!!!!

- one type of policy that can <font color=blue> use to manage organization </font>
  - Attaching an SCP to an AWS Organizations entity (root, OU, or account) defines a guardrail for what actions the principals can perform.

- policy that specifies <font color=blue> the services and actions </font> that users/roles can use in the accounts that the SCP affects.

- can <font color=red> limit account usage </font> to organizational units or linked accounts.


- <font color=red> enables permission controls </font>, similar to IAM permissions policies
  - almost the same syntax, use JSON
  - but SCP policies <font color=red> never grants permissions </font>
  - it specify the maximum permissions for an organization or OU.
  - SCPs are filters that <font color=blue> allow only the specified services and actions to be used </font> in affected accounts.
  - <font color=red> overwrite the admin permissions </font>
    - Even if a user is granted full administrator permissions with an IAM permission policy,
    - offer central control over the maximum available permissions for all accounts in organization, ensure accounts stay in organization’s access control guidelines.
  - any access that is <font color=red> not explicitly allowed or is explicitly denied </font> by the SCPs that affect that account is blocked.

- <font color=red> restrict a root user of an Organization Unit account </font>
  - defines a safeguard for the actions that accounts in the organization root or OU can do.
  - Attaching an SCP to the organization root/unit (OU)
    - Log in to the master account and `create the SCP`
    - `Select the Organizational Unit`
    - `Enable the SCP` for the Organizational Unit
      - SCPs are not automatically enabled;
    - `Attach the SCP to the member account` within the Organizational Unit

- <font color=red> not a substitute for well-managed each account </font>
  - still need to attach IAM policies to users and roles in organization's accounts <font color=blue> to actually grant permissions to them. </font>
    - IAM Identity policy
      - policy attached to an identity in IAM
      - less overhead to use a SCP for the entire AWS account.
      - identity policies can be attached only to IAM Users in your account.



- `FullAWSAccess`
  - a service control policy
  - allows users to access services/resources on an attached account.
  - allows access to all AWS services within an attached member account

- SCPs are available only when you enable all features in your organization.
  - has all features enabled,
  - including consolidated billing.


> For example
> - SCP: allows only database service access to "database" account
> - any user, group, or role in that account is denied access to any other service's operations.

- **You can attach an SCP to the following entities**:
  - A root: <font color=blue> affects all accounts in the organization </font>
  - An OU: <font color=blue> affects all accounts in that OU and all accounts in any OUs in that OU subtree </font>
  - Or an individual account




---


## SCPs vs IAM Identity policy

> AWS Organizations does not replace but associating AWS Identity and Access Management (IAM) policies with users, groups, and roles within an AWS account.

With IAM policies
- can <font color=red> allow or deny access </font> to
  - <font color=blue> AWS services </font> (such as Amazon S3)
  - <font color=blue> individual AWS resources </font> (such as a specific S3 bucket)
  - or <font color=blue> individual API actions </font> (such as s3:CreateBucket)
- An IAM policy can applied only to `IAM users, groups, or roles`,
- but <font color=red> can never restrict the AWS account root user </font>

In contrast, with Organizations,
- use service control policies (SCPs) to <font color=red> allow or deny access </font> to <font color=blue> particular AWS services </font> for `individual AWS accounts` or `groups of accounts in an OU`.
- The specified actions from an attached SCP affect `all IAM users, groups, and roles` for an account,
- <font coloe=red> including the AWS account root user </font>

￼




---

## AWS Organizations Setup

![3A0C5If](https://i.imgur.com/DHrovFl.png)

---


## access AWS Organizations

![fopYerU](https://i.imgur.com/RMGqWu0.png)

- **AWS Management Console**:
  - browser-based interface to manage organization and AWS resources.
  - You can perform any task in your organization by using the console.
- **AWS Command Line Interface(AWS CLI)**:
  - issue commands at your system's command line to perform AWS Organizations tasks and AWS tasks.
  - faster and more convenient than using the console.
- **AWS software development kits (SDKs)**:
  - to handle tasks such as cryptographically signing requests, managing errors, and retrying requests automatically.
  - AWS SDKs consist of libraries and sample code for various programming languages and platforms (Java, Python, Ruby, .NET, iOS, and Android).
- **AWS Organizations HTTPS Query API**:
  - programmatic access to AWS Organizations and AWS.
  - use the API to issue HTTPS requests directly to the service.
  - to use HTTPS API, must include code to digitally sign requests by using your credentials.
