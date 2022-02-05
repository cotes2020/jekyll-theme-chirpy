---
title: AWS - IdenAccessManage - Permissions boundaries 
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, IdenAccessManage]
tags: [AWS, IdenAccessManage]
toc: true
image:
---
 
[toc]

---

# Permissions boundaries for IAM entities  

---

## in short

![permissions_boundary](https://i.imgur.com/tnA8au6.png)

---

## permissions boundary

- an advanced feature
- AWS supports `permissions boundaries` for IAM entities (users or roles)
- using a managed policy to set the maximum permissions that an <font color=blue> identity-based policy can grant to an IAM entity </font>
- An entity's permissions boundary allows it to <font color=blue> perform only the actions that are allowed by both its identity-based policies and permissions boundaries </font>

- can use an <font color=blue> AWS managed policy </font> or a <font color=blue> customer managed policy </font> to set the boundary for an IAM entity (user or role).
- That policy limits the maximum permissions for the user or role.


Example:

```json
// assume IAM user `ShirleyRodriguez` is allowed to manage only Amazon S3, Amazon CloudWatch, and Amazon EC2.
// To enforce this rule, use the following policy to set the permissions boundary for the `ShirleyRodriguez` user:
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:*",
                "cloudwatch:*",
                "ec2:*"
            ],
            "Resource": "*"
        }
    ]
}
```

- When you use a policy to set the permissions boundary for a user
  - it limits the user's permissions but does not provide permissions on its own.
- example:
  - the policy sets the maximum permissions of `ShirleyRodriguez` as all operations in Amazon S3, CloudWatch, and Amazon EC2.
  - Shirley can never perform operations in any other service, including IAM, even if she has a permissions policy that allows it.

  - you can add the following policy to the `ShirleyRodriguez` user:
  - allows creating a user in IAM.
  - If you attach this permissions policy to the `ShirleyRodriguez` user, and Shirley tries to create a user, the operation fails.
  - It fails because the permissions boundary does not allow the `iam:CreateUser` operation.
  - Given these two policies, Shirley does not have permission to perform any operations in AWS.
  - You must add a different permissions policy to allow actions in other services, such as Amazon S3. Alternatively, you could update the permissions boundary to allow her to create a user in IAM.

```json

{
  "Version": "2012-10-17",
  "Statement": {
    "Effect": "Allow",
    "Action": "iam:CreateUser",
    "Resource": "*"
  }
}
```

---

## Evaluating effective permissions with boundaries

The permissions boundary for an IAM entity (user or role) sets the maximum permissions that the entity can have.
- This can change the effective permissions for that user or role.
- The effective permissions for an entity are the permissions that are granted by all the policies that affect the user or role.
- Within an account, the permissions for an entity can be affected by
  - identity-based policies,
  - resource-based policies,
  - permissions boundaries,
  - Organizations SCPs
  - or session policies. 

- If any one of these policy types explicitly denies access for an operation, then the request is denied.
- The permissions granted to an entity by multiple permissions types are more complex.
- check [Policy evaluation logic]().

---

### Identity-based policies with boundaries


Identity-based policies
- are <font color=red> inline or managed policies attached to a user, group of users, or role </font>
- effective permissions
  - Identity-based policies grant permission to the entity
  - and permissions boundaries limit those permissions.
  - The effective permissions are the intersection of both policy types.
  - An explicit deny in either of these policies overrides the allow.

![permissions_boundary](https://i.imgur.com/tnA8au6.png)

---

### Resource-based policies

Resource-based policies
- control how the specified principal can access the resource to which the policy is attached.

![EffectivePermissions-rbp-boundary-id](https://i.imgur.com/IaeaipQ.png)

1. <font color=red> Resource-based policies for IAM users </font>
   - Within an account
     - an <font color=blue> implicit deny in a permissions boundary </font> <font color=red> does not limit the permissions granted </font> to an IAM user by a resource-based policy.
   - Permissions boundaries reduce permissions that are granted to a user by identity-based policies.
   - Resource-based policies can provide additional permissions to the user.

2. <font color=red> Resource-based policies for IAM roles and federated users </font>
   - Within an account
     - an <font color=blue> implicit deny in a permissions boundary </font> <font color=red> does limit the permissions granted </font>  to the ARN of the underlying IAM role/user by the resource-based policy.
   - However
     - <font color=blue> if resource-based policy grants permissions directly to the session principal </font> (the assumed-role ARN or federated user ARN)
     - an <font color=blue> implicit deny in the permissions boundary </font> <font color=red> does not limit those permissions </font>
   - [Session policies](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html#policies_session).

---

### Organizations SCPs

![EffectivePermissions-scp-boundary-id](https://i.imgur.com/UfyiKlF.png)

SCPs
- are applied to an entire AWS account.
- limit permissions for every request made by a principal within the account.
- An IAM entity (user or role) can make a request that is affected by an SCP, a permissions boundary, and an identity-based policy.
  - the request is allowed only if all three policy types allow it.
  - The effective permissions are the intersection of all three policy types.
  - An explicit deny in any of these policies overrides the allow.

account member in AWS Organizations.
- Organization members might be affected by an SCP.
- To view this data using the AWS CLI command or AWS API operation, permissions: `organizations:DescribeOrganization` action for your Organizations entity
- You must have additional permissions to perform the operation in the Organizations console. 

---

### Session policies

![EffectivePermissions-session-boundary-id](https://i.imgur.com/LCcnM4u.png)

Session policies
- advanced policies
- pass as a parameter when programmatically create a temporary session for a role or federated user. 
- The permissions for a session come from the `IAM entity (user or role) used` to create the session and from the session policy. 
- The entity's identity-based policy permissions are limited by the session policy and the permissions boundary. 
  - The effective permissions for this set of policy types are theintersection of all three policy types. 
  - An explicit deny in any of these policies overrides the allow. 

---

## Delegate 托付 responsibility to others using permissions boundaries

use permissions boundaries to delegate permissions management tasks
- such as user creation, to IAM users in your account. 
- This permits others to perform tasks on your behalf within a specific boundary of permissions.

Example
- María is the administrator of the X-Company AWS account. 
  - She wants to delegate user creation duties to Zhang. 
  - However, she must ensure that Zhang creates users that adhere to the following company rules:
    * Users cannot use IAM to create or manage users, groups, roles, or policies.
    * Users are denied access to the Amazon S3 `logs` bucket and cannot access the `i-1234567890abcdef0` Amazon EC2 instance.
    * Users cannot remove their own boundary policies.

- To enforce these rules, María completes the following tasks 
  1. María creates the `XCompanyBoundaries` managed policy 
     1. to use as a permissions boundary for all new users in the account.
  2. María creates the `DelegatedUserBoundary` managed policy 
     1. and assigns it as the permissions boundary for Zhang. 
     2. Maria makes a note of her admin IAM user's ARN and uses it in the policy to prevent Zhang from accessing it.
  3. María creates the `DelegatedUserPermissions` managed policy 
     1. and attaches it as a permissions policy for Zhang.
  4. María tells Zhang about his new responsibilities and limitations.


**Task 1:** 

María must first create a managed policy to define the boundary for the new users. 
- María will allow Zhang to give users the permissions policies they need, but she wants those users to be restricted. 
- To do this, she creates the following customer managed policy with the name `XCompanyBoundaries`. 
- This policy does the following:
  * Allows users full access to several services
  * Allows limited self-managing access in the IAM console. 
    * This means they can change their password after signing into the console. 
    * They can't set their initial password. 
    * To allow this, add the `"*LoginProfile"` action to the `AllowManageOwnPasswordAndAccessKeys` statement.
  * Denies users access to the Amazon S3 logs bucket or the `i-1234567890abcdef0` Amazon EC2 instance


```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            // allows full access to the specified AWS services. 
            // This means that a new user's actions in these services are limited only by the permissions policies that are attached to the user.
            "Sid": "ServiceBoundaries",
            "Effect": "Allow",
            "Action": [
                "s3:*",
                "cloudwatch:*",
                "ec2:*",
                "dynamodb:*"
            ],
            "Resource": "*"
        },
        {
            // allows access to list all IAM users. 
            // This access is necessary to navigate the **Users** page in the AWS Management Console. 
            // It also allows viewing the password requirements for the account, which is necessary when changing your own password.
            "Sid": "AllowIAMConsoleForCredentials",
            "Effect": "Allow",
            "Action": [
                "iam:ListUsers",
                "iam:GetAccountPasswordPolicy"
            ],
            "Resource": "*"
        },
        {
            // allows the users manage only their own console password and programmatic access keys. 
            // This is important if Zhang or another administrator gives a new user a permissions policy with full IAM access. 
            // In that case, that user could then change their own or other users' permissions. 
            // This statement prevents that from happening.
            "Sid": "AllowManageOwnPasswordAndAccessKeys",
            "Effect": "Allow",
            "Action": [
                "iam:*AccessKey*",
                "iam:ChangePassword",
                "iam:GetUser",
                "iam:*ServiceSpecificCredential*",
                "iam:*SigningCertificate*"
            ],
            "Resource": ["arn:aws:iam::*:user/${aws:username}"]
        },
        {
            // explicitly denies access to the `logs` bucket.
            "Sid": "DenyS3Logs",
            "Effect": "Deny",
            "Action": "s3:*",
            "Resource": [
                "arn:aws:s3:::logs",
                "arn:aws:s3:::logs/*"
            ]
        },
        {
            // explicitly denies access to the `i-1234567890abcdef0` instance.
            "Sid": "DenyEC2Production",
            "Effect": "Deny",
            "Action": "ec2:*",
            "Resource": "arn:aws:ec2:*:*:instance/i-1234567890abcdef0"
        }
    ]
}
```  

**Task 2:** 

María wants to allow Zhang to create all X-Company users, but only with the `XCompanyBoundaries` permissions boundary. 
- She creates the following customer managed policy named `DelegatedUserBoundary`. 
  - This policy defines the maximum permissions that Zhang can have.
  - Each statement serves a different purpose:
  1. The `CreateOrChangeOnlyWithBoundary` statement allows Zhang to create IAM users but only if he uses the `XCompanyBoundaries` policy to set the permissions boundary. This statement also allows him to set the permissions boundary for existing users but only using that same policy. Finally, this statement allows Zhang to manage permissions policies for users with this permissions boundary set.
  2. The `CloudWatchAndOtherIAMTasks` statement allows Zhang to complete other user, group, and policy management tasks. He has permissions to reset passwords and create access keys for any IAM user not listed in the condition key. This allows him to help users with sign-in issues.
  3. The `NoBoundaryPolicyEdit` statement denies Zhang access to update the `XCompanyBoundaries` policy. He is not allowed to change any policy that is used to set the permissions boundary for himself or other users.
  4. The `NoBoundaryUserDelete` statement denies Zhang access to delete the permissions boundary for himself or other users.
- María then assigns the `DelegatedUserBoundary` policy <font color=red> as the permissions boundary </font> for the `Zhang` user.


```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            // allows Zhang to create IAM users but only if he uses the `XCompanyBoundaries` policy to set the permissions boundary. 
            // allows him to set the permissions boundary for existing users but only using that same policy. 
            // allows Zhang to manage permissions policies for users with this permissions boundary set.
            "Sid": "CreateOrChangeOnlyWithBoundary",
            "Effect": "Allow",
            "Action": [
                "iam:CreateUser",
                "iam:DeleteUserPolicy",
                "iam:AttachUserPolicy",
                "iam:DetachUserPolicy",
                "iam:PutUserPermissionsBoundary",
                "iam:PutUserPolicy"
            ],
            "Resource": "*",
            "Condition": {
                "StringEquals": {
                    "iam:PermissionsBoundary": "arn:aws:iam::123456789012:policy/XCompanyBoundaries"
                }
            }
        },
        {
            // allows Zhang to complete other user, group, and policy management tasks. 
            // He has permissions to reset passwords and create access keys for any IAM user not listed in the condition key. 
            // This allows him to help users with sign-in issues.
            "Sid": "CloudWatchAndOtherIAMTasks",
            "Effect": "Allow",
            "Action": [
                "cloudwatch:*",
                "iam:GetUser",
                "iam:ListUsers",
                "iam:DeleteUser",
                "iam:UpdateUser",
                "iam:CreateAccessKey",
                "iam:CreateLoginProfile",
                "iam:GetAccountPasswordPolicy",
                "iam:GetLoginProfile",
                "iam:ListGroups",
                "iam:ListGroupsForUser",
                "iam:CreateGroup",
                "iam:GetGroup",
                "iam:DeleteGroup",
                "iam:UpdateGroup",
                "iam:CreatePolicy",
                "iam:DeletePolicy",
                "iam:DeletePolicyVersion",
                "iam:GetPolicy",
                "iam:GetPolicyVersion",
                "iam:GetUserPolicy",
                "iam:GetRolePolicy",
                "iam:ListPolicies",
                "iam:ListPolicyVersions",
                "iam:ListEntitiesForPolicy",
                "iam:ListUserPolicies",
                "iam:ListAttachedUserPolicies",
                "iam:ListRolePolicies",
                "iam:ListAttachedRolePolicies",
                "iam:SetDefaultPolicyVersion",
                "iam:SimulatePrincipalPolicy",
                "iam:SimulateCustomPolicy"
            ],
            "NotResource": "arn:aws:iam::123456789012:user/Maria"
        },
        {
            // denies Zhang access to update the `XCompanyBoundaries` policy. 
            // He is not allowed to change any policy that is used to set the permissions boundary for himself or other users.
            "Sid": "NoBoundaryPolicyEdit",
            "Effect": "Deny",
            "Action": [
                "iam:CreatePolicyVersion",
                "iam:DeletePolicy",
                "iam:DeletePolicyVersion",
                "iam:SetDefaultPolicyVersion"
            ],
            "Resource": [
                "arn:aws:iam::123456789012:policy/XCompanyBoundaries",
                "arn:aws:iam::123456789012:policy/DelegatedUserBoundary"
            ]
        },
        {
            // denies Zhang access to delete the permissions boundary for himself or other users.

            "Sid": "NoBoundaryUserDelete",
            "Effect": "Deny",
            "Action": "iam:DeleteUserPermissionsBoundary",
            "Resource": "*"
        }
    ]
}
```




**Task 3:** 

- the <font color=red> permissions boundary </font> 
  - limits the maximum permissions
  - does not grant access on its own, 
  - Maria must create a permissions policy for Zhang. 
- She creates the following policy named `DelegatedUserPermissions`. 
  - This policy defines the operations that Zhang can perform, within the defined boundary.
  - Each statement serves a different purpose:
    1. The `IAM` statement of the policy allows Zhang full access to IAM. However, because his permissions boundary allows only some IAM operations, his effective IAM permissions are limited only by his permissions boundary.
    2. The `CloudWatchLimited` statement allows Zhang to perform five actions in CloudWatch. His permissions boundary allows all actions in CloudWatch, so his effective CloudWatch permissions are limited only by his permissions policy.
    3. The `S3BucketContents` statement allows Zhang to list the `ZhangBucket` Amazon S3 bucket. However, his permissions boundary does not allow any Amazon S3 action, so he cannot perform any S3 operations, regardless of his permissions policy.
  - Note
    - Zhang's policies allow him to create a user that can then access Amazon S3 resources that he can't access. 
    - By delegating these administrative actions, Maria effectively trusts Zhang with access to Amazon S3.

- María then attaches the `DelegatedUserPermissions` policy as the permissions policy for the `Zhang` user.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            // allows Zhang full access to IAM. 
            // However, because his permissions boundary allows only some IAM operations, his effective IAM permissions are limited only by his permissions boundary.
            "Sid": "IAM",
            "Effect": "Allow",
            "Action": "iam:*",
            "Resource": "*"
        },
        {
            // allows Zhang to perform five actions in CloudWatch. 
            // His permissions boundary allows all actions in CloudWatch, so his effective CloudWatch permissions are limited only by his permissions policy.
            "Sid": "CloudWatchLimited",
            "Effect": "Allow",
            "Action": [
                "cloudwatch:GetDashboard",
                "cloudwatch:GetMetricData",
                "cloudwatch:ListDashboards",
                "cloudwatch:GetMetricStatistics",
                "cloudwatch:ListMetrics"
            ],
            "Resource": "*"
        },
        {
            // allows Zhang to list the `ZhangBucket` Amazon S3 bucket. 
            // However, his permissions boundary does not allow any Amazon S3 action, so he cannot perform any S3 operations, regardless of his permissions policy.
            "Sid": "S3BucketContents",
            "Effect": "Allow",
            "Action": "s3:ListBucket",
            "Resource": "arn:aws:s3:::ZhangBucket"
        }
    ]
}
```


**Task 4:** 

She gives Zhang instructions to create a new user. 
- She tells him that he can create new users with any permissions that they need, 
- but he must assign them the `XCompanyBoundaries` policy as a permissions boundary.

- Zhang completes the following tasks:
  1. Zhang creates a user with the AWS Management Console. 
     1. He types the user name `Nikhil` and enables console access for the user. 
     2. He clears the checkbox next to **Requires password reset**, because the policies above allow users to change their passwords only after they are signed in to the IAM console.
  2. On the **Set permissions** page
     1. Zhang chooses the **IAMFullAccess** and **AmazonS3ReadOnlyAccess** permissions policies that allow Nikhil to do his work.
  3. Zhang skips the **Set permissions boundary** section
     1. <font color=red> forgetting María's instructions </font>
  4. Zhang reviews the user details and chooses **Create user**.
     1. The operation fails and access is denied. 
     2. Zhang's `DelegatedUserBoundary` permissions boundary requires that any user he creates have the `XCompanyBoundaries` policy used as a permissions boundary.
  5. Zhang returns to the previous page. 
     1. In the **Set permissions boundary** section, he chooses the `XCompanyBoundaries` policy.
  6. Zhang reviews the user details and chooses **Create user**.
  7. The user is created.

- When Nikhil signs in
  - he has access to IAM and Amazon S3
  - except those operations that are denied by the permissions boundary. 
  - For example
    - he can change his own password in IAM but can't create another user or edit his policies. 
    - Nikhil has read-only access to Amazon S3.

- If someone adds a resource-based policy to the `logs` bucket that allows Nikhil to put an object in the bucket, he still cannot access the bucket. 
  - The reason 
  - any actions on the `logs` bucket are explicitly denied by his permissions boundary. 
  - An explicit deny in any policy type results in a request being denied. 

- However
  - if a resource-based policy attached to a Secrets Manager secret allows Nikhil to perform the `secretsmanager:GetSecretValue` action, then Nikhil can retrieve and decrypt the secret. 
  - The reason is that Secrets Manager operations are not explicitly denied by his permissions boundary, and implicit denies in permissions boundaries do not limit resource-based policies.

