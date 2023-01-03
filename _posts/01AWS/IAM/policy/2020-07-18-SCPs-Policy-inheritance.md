---
title: AWS - IdenAccessManage - SCPs policy inheritance
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, IdenAccessManage]
tags: [AWS, IdenAccessManage, SCPs]
toc: true
image:
---

[toc]

---

# SCPs policy inheritance


attach policies to organization entities (organization root, organizational unit (OU), or account) in your organization:

- attach a policy to the <font color=red> organization root </font>
  - <font color=blue> all OUs and accounts </font> in the organization inherit that policy.
- attach a policy to a <font color=red> specific OU </font>
  - <font color=blue> accounts or child OU </font> that are directly under that OU inherit the policy.
- attach a policy to a <font color=red> specific account </font>
  - it affects <font color=blue> only that account </font>

Because you can attach policies to multiple levels in the organization, accounts can inherit multiple policies.

Exactly how policies affect the OUs and accounts that inherit them depends on the type of policy:
1. Service control policies (SCPs)
2. Management policy types
   - AI services opt-out policies
   - Backup policies
   - Tag policies

---

## Inheritance for service control policies

---

### Inheritance for service control policies (SCPs)

- behaves like a filter
- tree structure of the organization
  - from the root to all of the OUs and end at the accounts.
  - All AWS permissions flow into the root of the tree.
  - <font color=blue> permissions must then flow past the SCPs attached to the root, OUs, and the account to get to the principal (an IAM role or user) making a request </font>
- Each SCP can filter the permissions passing through to the levels below it.
  - If an action is blocked by a `Deny` statement,
    - then all OUs and accounts affected by that SCP are denied access to that action.
- SCPs can **_only_** filter; they never add permissions.

- SCPs do not support inheritance operators that alter how elements of the policy are inherited by child OUs and accounts.

- An SCP at a lower level can't add a permission after it is blocked by an SCP at a higher level.
  - traverse down the hierarchy of OUs to the account
  - At each level, the result of the evaluation at the parent becomes the policy on the left of the diagram and is compared to the SCPs attached to the child OU.


### Example

![How_SCP_Permissions_Work](https://i.imgur.com/82X5eFC.png)

- Because the SCP attached to the root doesn't allow D or E, no OUs or accounts in the organization can use them.
  - Even though the SCP attached to the OU explicitly allows D and E, they are blocked because they're blocked by the SCP attached to the root.
  - Because the OU's SCP doesn't allow A or B, those permissions are blocked for the OU and any of its child OUs or accounts.
  - However, other OUs that might exist under the root can still allow A and B.

- some child OU called X
  - **the oval on the left** represents the inherited, effective permissions permitted by all of the SCPs above OU X in the hierarchy.
  - **The oval on the right** represents the SCP attached to an OU or an AWS account contained in OU X.
  - **the intersection of those permissions** is what is available to be used the entity on the right.
- If that entity is an AWS account
  - intersection is the set of permissions that can be granted to users and roles in that account.
- If the entity is an OU
  - intersection is the set of permissions that can be inherited by that OU's children.
- Repeat the process for each level down the OU hierarchy
  - until reach the account itself.
  - That final effective policy is the list of permissions that were allowed by every SCP above that account and attached to it.



---


### service control policies (SCPs) with IAM permission policies

<font color=red> Users and roles in accounts must still be granted permissions using AWS IAM permission policies attached </font>

- The SCPs
  - only determine what permissions are **_available_** to be granted by such policies.
  - The user can't perform any actions that the applicable SCPs don't allow.
- IAM permission policies
  - Actions allowed by the SCPs can be used if they are granted to the user or role by IAM permission policies.

When you attach SCPs to the organization root, OUs, or directly to accounts
- all policies that affect a given account are evaluated together using the same rules that govern IAM permission policies:

  * Any action that <font color=red> isn't explicitly allowed by an SCP is implicitly denied </font> and can't be delegated to users or roles in the affected accounts.

  * Users and roles in affected accounts can't perform any actions that are listed in the SCP's `Deny` statement.
    * An <font color=red> explicit Deny </font> overrides <font color=red> any Allow that other SCPs might grant </font>

  * Any action that has an explicit `Allow` in an SCP can be delegated to users and roles in the affected accounts.
    * such as the default "\*" SCP or by any other SCP that calls out a specific service or action


---


### SCPs' `FullAWSAccess`


- By default, an SCP named `FullAWSAccess` is attached to every organization root, OU, and account.
  - This default SCP allows all actions and all services.
  - So in a new organization, until creating or manipulating the SCPs, all existing IAM permissions continue to operate as they did.
  - As apply a new or modified SCP to the organization root or an OU that contains an account, the permissions that your users have in that account become filtered by the SCP.
  - Permissions that used to work might now be denied if they're not allowed by the SCP at every level of the hierarchy down to the specified account.

- If disable the SCP policy type on the organization root
  - all SCPs are automatically detached from all entities in the organization root.
- If re-enable SCPs policy type  on the organization root
  - all the original attachments are lost
  - and all entities are reset to being attached to only the default `FullAWSAccess` SCP.


see list of policies applied to an account and where that policy comes from.
1. choose an account in the AWS Organizations console.
2. On the account details page, choose **Policies** and then choose **Service Control Policies** in the right-hand details pane.
3. The same policy might apply to the account multiple times because the policy can be attached to any or all of the parent containers of the account.
4. The effective policy that applies to the account is the intersection of allowed permissions of all applicable policies.

---


### Example: to allow an AWS service API at the member account level

- must allow that API at **_every_** level between the member account and the root of your organization.
- must attach SCPs to every level from your organization’s root to the member account that allows the given AWS service API (such as ec2:RunInstances).
- use either of the following strategies

  * A deny list strategy
    * makes use of the `FullAWSAccess` SCP attached by default to every OU and account.
    * This SCP
      * overrides the default implicit deny
      * explicitly allows all permissions from the root to every account
      * unless explicitly deny a permission with an additional SCP created and attached to the appropriate OU or account.
    * This strategy works because an <font color=red> explicit deny in a policy always overrides allow </font>
    * No account below the level of the OU with the deny policy can use the denied API
    * and there is no way to add the permission back lower in the hierarchy.

  * An allow list strategy
    * remove the `FullAWSAccess` SCP ttached by default to every OU and account.
    * no APIs are permitted anywhere unless you explicitly allow them.
    * To allow a service API to operate in an AWS account, you must create your own SCPs and attach them to the account and every OU above it, up to and including the root.
    * Every SCP in the hierarchy, starting at the root, must explicitly allow the APIs to be usable in the OUs and accounts below it.
    * This strategy works because an <font color=red> explicit allow in an SCP overrides an implicit 暗示的 deny </font>
