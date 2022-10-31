---
title: AWS - IdenAccessManage - SCPs Template
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, IdenAccessManage]
tags: [AWS, IdenAccessManage, SCPs]
toc: true
image:
---

[toc]

---

# SCPs Template

---

# General Example

---

## Example: Deny access to AWS based on the requested AWS Region

This SCP
- denies access to any operations outside of the specified Regions.
  - uses the `Deny` effect to deny access to all requests for operations that don't target the approved regions.
- provides exemptions for operations in approved global services.
  - The [NotAction](https://docs.aws.amazon.com/IAM/latest/UserGuide/referencepolicieselementsnotaction.html) element enables you to list services whose operations (or individual operations) are exempted from this restriction.
  - Because global services have endpoints that are physically hosted by the `us-east-1` Region , they must be exempted in this way.
    - With an SCP structured this way
    - requests made to global services in the `us-east-1` Region are allowed if the requested service is included in the `NotAction` element.
    - Any other requests to services in the `us-east-1` Region are denied by this example policy.

- This example also exempt requests made by either of two specified administrator roles.


Considerations

- If you use `AWS Control Tower` in the organization, we recommend that you do not use this example policy.
  - `AWS Control Tower` works across AWS Regions in a way that is not compatible with this example policy.

- AWS KMS and AWS Certificate Manager support Regional endpoints.
  - However, if you want to use them with a global service such as Amazon CloudFront you must include them in the global service exclusion list in the following example SCP.
  - A global service like AWS CloudFormation typically requires access to AWS KMS and ACM in the same region, which for a global service is the US East (N. Virginia) Region (`us-east-1`).

- By default, AWS STS is a global service and must be included in the global service exclusion list.
  - However, you can enable AWS STS to use Region endpoints instead of a single global endpoint.
  - If you do this, you can remove STS from the global service exemption list in the following example SCP.
  - For more information see [Managing AWS STS in an AWS Region](https://docs.aws.amazon.com/IAM/latest/UserGuide/idcredentialstempenable-regions.html).

> This example might not include all of the latest global AWS services or operations.
> - Replace the list of services and operations with the global services used by accounts in the organization.
> view the [service last accessed data in the IAM console](https://docs.aws.amazon.com/IAM/latest/UserGuide/accesspoliciesaccess-advisor.html) to determine what global services the organization uses.
> - The **Access Advisor** tab on the details page for an IAM user, group, or role displays the AWS services that have been used by that entity, sorted by most recent access.


```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "DenyAllOutsideEU",
            "Effect": "Deny",
            "NotAction": [
                "a4b:*",
                "acm:*",
                "aws-marketplace-management:*",
                "aws-marketplace:*",
                "aws-portal:*",
                "awsbillingconsole:*",
                "budgets:*",
                "ce:*",
                "chime:*",
                "cloudfront:*",
                "config:*",
                "cur:*",
                "directconnect:*",
                "ec2:DescribeRegions",
                "ec2:DescribeTransitGateways",
                "ec2:DescribeVpnGateways",
                "fms:*",
                "globalaccelerator:*",
                "health:*",
                "iam:*",
                "importexport:*",
                "kms:*",
                "mobileanalytics:*",
                "networkmanager:*",
                "organizations:*",
                "pricing:*",
                "route53:*",
                "route53domains:*",
                "s3:GetAccountPublic*",
                "s3:ListAllMyBuckets",
                "s3:PutAccountPublic*",
                "shield:*",
                "sts:*",
                "support:*",
                "trustedadvisor:*",
                "waf-regional:*",
                "waf:*",
                "wafv2:*",
                "wellarchitected:*"
            ],
            "Resource": "*",
            "Condition": {
                "StringNotEquals": {
                    "aws:RequestedRegion": [
                        "eu-central-1",
                        "eu-west-1"
                    ]
                },
                "ArnNotLike": {
                    "aws:PrincipalARN": [
                        "arn:aws:iam::*:role/Role1AllowedToBypassThisSCP",
                        "arn:aws:iam::*:role/Role2AllowedToBypassThisSCP"
                    ]
                    // exempt requests made by either of two specified administrator roles.
                }
            }
        }
    ]
}
```

---


## Example: Prevent IAM users/roles from making certain changes

This SCP
- restricts IAM users/roles from making changes to the specified IAM role that you created in all accounts in the organization.


```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyAccessToASpecificRole",
      "Effect": "Deny",
      "Action": [
        "iam:AttachRolePolicy",
        "iam:DeleteRole",
        "iam:DeleteRolePermissionsBoundary",
        "iam:DeleteRolePolicy",
        "iam:DetachRolePolicy",
        "iam:PutRolePermissionsBoundary",
        "iam:PutRolePolicy",
        "iam:UpdateAssumeRolePolicy",
        "iam:UpdateRole",
        "iam:UpdateRoleDescription"
      ],
      "Resource": [
        "arn:aws:iam::*:role/name-of-role-to-deny"
      ]
    }
  ]
}
```

---


## Example: Prevent IAM users/roles from making specified changes, with exception for specified admin role

This SCP
- builds on the previous example to make an exception for administrators.
- prevents IAM users/roles in affected accounts from making changes to a common administrative IAM role created in all accounts in the organization except for administrators using a specified role.


```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyAccessWithException",
      "Effect": "Deny",
      "Action": [
        "iam:AttachRolePolicy",
        "iam:DeleteRole",
        "iam:DeleteRolePermissionsBoundary",
        "iam:DeleteRolePolicy",
        "iam:DetachRolePolicy",
        "iam:PutRolePermissionsBoundary",
        "iam:PutRolePolicy",
        "iam:UpdateAssumeRolePolicy",
        "iam:UpdateRole",
        "iam:UpdateRoleDescription"
      ],
      "Resource": [
        "arn:aws:iam::*:role/name-of-role-to-deny"
      ],
      "Condition": {
        "StringNotLike": {
          "aws:PrincipalARN":"arn:aws:iam::*:role/name-of-admin-role-to-allow"
        }
      }
    }
  ]
}
```

---


## Example: Require MFA to perform an API action

This SCP
- require that multi-factor authentication (MFA) is enabled before an IAM user or role can perform an action.
- In this example, the action is to stop an Amazon EC2 instance.


```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyStopAndTerminateWhenMFAIsNotPresent",
      "Effect": "Deny",
      "Action": [
        "ec2:StopInstances",
        "ec2:TerminateInstances"
      ],
      "Resource": "*",
      "Condition": {
          "BoolIfExists": {
              "aws:MultiFactorAuthPresent": false
          }
      }
    }
  ]
}
```

---


## Example: Block service access for the root user

This SCP
- restricts all access to the specified actions for the [root user](https://docs.aws.amazon.com/IAM/latest/UserGuide/idroot-user.html) in an account.
- to prevent the accounts from using root credentials in specific ways, add the own actions to this policy.


```json
{
  "Version": "2012-10-17",
  "Statement": [
      {
          "Sid": "RestrictEC2ForRoot",
          "Effect": "Deny",
          "Action": [
            "ec2:*"
          ],
          "Resource": "*",
          // "Resource": ["*"],
          "Condition": {
              "StringLike": {
                  "aws:PrincipalArn": ["arn:aws:iam::*:root"]
            }
          }
      }
  ]
}
```




---

# Example: SCPs for AWS Config

---


## Example: Prevent users from disabling AWS Config or changing its rules

This SCP
- prevents users/roles in any affected account from running AWS Config operations that could disable AWS Config or alter its rules or triggers.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Deny",
      "Action": [
        "config:DeleteConfigRule",
        "config:DeleteConfigurationRecorder",
        "config:DeleteDeliveryChannel",
        "config:StopConfigurationRecorder"
      ],
      "Resource": "*"
    }
  ]
}
```


---

# Example SCPs for Amazon CloudWatch

---

## Example: Prevent users from disabling CloudWatch or altering its configuration

- A lower-level CloudWatch operator needs to monitor dashboards and alarms.
- However, the operator must not be able to delete or change any dashboard or alarm that senior people might put into place.

This SCP
- prevents users/roles in any affected account from running any of the CloudWatch commands that could delete or change the dashboards or alarms.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Deny",
      "Action": [
        "cloudwatch:DeleteAlarms",
        "cloudwatch:DeleteDashboards",
        "cloudwatch:DisableAlarmActions",
        "cloudwatch:PutDashboard",
        "cloudwatch:PutMetricAlarm",
        "cloudwatch:SetAlarmState"
      ],
      "Resource": "*"
    }
  ]
}
```

---

# Example SCPs for Amazon EC2

---


## Example: Require Amazon EC2 instances to use a specific type

This SCP
- any instance launches not using the `t2.micro` instance type are denied.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "RequireMicroInstanceType",
      "Effect": "Deny",
      "Action": "ec2:RunInstances",
      "Resource": "arn:aws:ec2:*:*:instance/*",
      "Condition": {
        "StringNotEquals":{
          "ec2:InstanceType":"t2.micro"
        }
      }
    }
  ]
}
```

---

# Example SCPs for Amazon GuardDuty

---


## Example: Prevent users from disabling GuardDuty or modifying its configuration

This SCP
- prevents users/roles in any affected account from disabling GuardDuty or altering its configuration,
  - either directly as a command or through the console.
- It effectively enables read-only access to the GuardDuty information and resources.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Deny",
            "Action": [
                "guardduty:AcceptInvitation",
                "guardduty:ArchiveFindings",
                "guardduty:CreateDetector",
                "guardduty:CreateFilter",
                "guardduty:CreateIPSet",
                "guardduty:CreateMembers",
                "guardduty:CreatePublishingDestination",
                "guardduty:CreateSampleFindings",
                "guardduty:CreateThreatIntelSet",
                "guardduty:DeclineInvitations",
                "guardduty:DeleteDetector",
                "guardduty:DeleteFilter",
                "guardduty:DeleteInvitations",
                "guardduty:DeleteIPSet",
                "guardduty:DeleteMembers",
                "guardduty:DeletePublishingDestination",
                "guardduty:DeleteThreatIntelSet",
                "guardduty:DisassociateFromMasterAccount",
                "guardduty:DisassociateMembers",
                "guardduty:InviteMembers",
                "guardduty:StartMonitoringMembers",
                "guardduty:StopMonitoringMembers",
                "guardduty:TagResource",
                "guardduty:UnarchiveFindings",
                "guardduty:UntagResource",
                "guardduty:UpdateDetector",
                "guardduty:UpdateFilter",
                "guardduty:UpdateFindingsFeedback",
                "guardduty:UpdateIPSet",
                "guardduty:UpdatePublishingDestination",
                "guardduty:UpdateThreatIntelSet"
            ],
            "Resource": "*"
        }
    ]
}
```


---

# Example SCPs for AWS Resource Access Manager

---

## Example: Preventing external sharing

This SCP
- prevents users from creating resource shares that allow sharing with IAM users ad roles that aren't part of the organization.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Deny",
            "Action": [
                "ram:CreateResourceShare",
                "ram:UpdateResourceShare"
            ],
            "Resource": "*",
            "Condition": {
                "Bool": {
                    "ram:RequestedAllowsExternalPrincipals": "true"
                }
            }
        }
    ]
}
```

---


## Example: Allowing specific accounts to share only specified resource types

This SCP
- allows accounts `111111111111` and `222222222222` to create resource shares that share prefix lists,
- and to associate prefix lists with existing resource shares.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "OnlyNamedAccountsCanSharePrefixLists",
            "Effect": "Deny",
            "Action": [
                "ram:AssociateResourceShare",
                "ram:CreateResourceShare"
            ],
            "Resource": "*",
            "Condition": {
                "StringNotEquals": {
                    "aws:PrincipalAccount": [
                        "111111111111",
                        "222222222222"
                    ]
                },
                "StringEquals": {
                    "ram:RequestedResourceType": "ec2:PrefixList"
                }
            }
        }
    ]
}
```

---


## Example: Prevent sharing with organizations or organizational units (OUs)

This SCP
- prevents users from creating resource shares that share resources with an AWS Organization or OUs.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Deny",
            "Action": [
                "ram:CreateResourceShare",
                "ram:AssociateResourceShare"
            ],
            "Resource": "*",
            "Condition": {
                "ForAnyValue:StringLike": {
                    "ram:Principal": [
                        "arn:aws:organizations::*:organization/*",
                        "arn:aws:organizations::*:ou/*"
                    ]
                }
            }
        }
    ]
}
```

---


## Example: Allow sharing with only specified IAM users/roles

This SCP
- allows users to share resources with only organization `o-12345abcdef`, organizational unit `ou-98765fedcba`, and account `111111111111`.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Deny",
            "Action": [
                "ram:AssociateResourceShare",
                "ram:CreateResourceShare"
            ],
            "Resource": "*",
            "Condition": {
                "ForAnyValue:StringNotEquals": {
                    "ram:Principal": [
                        "arn:aws:organizations::123456789012:organization/o-12345abcdef",
                        "arn:aws:organizations::123456789012:ou/o-12345abcdef/ou-98765fedcba",
                        "111111111111"
                    ]
                }
            }
        }
    ]
}
```

---

# Example SCPs for Amazon VPC

---


## Example: Prevent users from deleting Amazon VPC flow logs

This SCP
- prevents users/roles in any affected account from deleting Amazon EC2 flow logs or CloudWatch log groups or log streams.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Deny",
      "Action": [
        "ec2:DeleteFlowLogs",
        "logs:DeleteLogGroup",
        "logs:DeleteLogStream"
      ],
      "Resource": "*"
    }
  ]
 }
 ```

---


## Example: Prevent any VPC that doesn't already have internet access from getting it

This SCP
- prevents users/roles in any affected account from changing the configuration of the Amazon EC2 VPCs to grant them direct access to the internet.
- It doesn't block existing direct access or any access that routes through the on-premises network environment.


```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Deny",
      "Action": [
        "ec2:AttachInternetGateway",
        "ec2:CreateInternetGateway",
        "ec2:CreateEgressOnlyInternetGateway",
        "ec2:CreateVpcPeeringConnection",
        "ec2:AcceptVpcPeeringConnection",
        "globalaccelerator:Create*",
        "globalaccelerator:Update*"
      ],
      "Resource": "*"
    }
  ]
}
```

---

# Example SCPs for tagging resources

---


## Example: Require a tag on specified created resources

This SCP
- prevents IAM users/roles in the affected accounts from creating certain resource types if the request doesn't include the specified tags.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyCreateSecretWithNoProjectTag",
      "Effect": "Deny",
      "Action": "secretsmanager:CreateSecret",
      "Resource": "*",
      "Condition": {
        "Null": {
          "aws:RequestTag/Project": "true"
        }
      }
    },
    {
      "Sid": "DenyRunInstanceWithNoProjectTag",
      "Effect": "Deny",
      "Action": "ec2:RunInstances",
      "Resource": [
        "arn:aws:ec2:*:*:instance/*",
        "arn:aws:ec2:*:*:volume/*"
      ],
      "Condition": {
        "Null": {
          "aws:RequestTag/Project": "true"
        }
      }
    },
    {
      "Sid": "DenyCreateSecretWithNoCostCenterTag",
      "Effect": "Deny",
      "Action": "secretsmanager:CreateSecret",
      "Resource": "*",
      "Condition": {
        "Null": {
          "aws:RequestTag/CostCenter": "true"
        }
      }
    },
    {
      "Sid": "DenyRunInstanceWithNoCostCenterTag",
      "Effect": "Deny",
      "Action": "ec2:RunInstances",
      "Resource": [
        "arn:aws:ec2:*:*:instance/*",
        "arn:aws:ec2:*:*:volume/*"
      ],
      "Condition": {
        "Null": {
          "aws:RequestTag/CostCenter": "true"
        }
      }
    }
  ]
}
```

For a list of all the services and the actions that they support in both AWS Organizations SCPs and IAM permission policies, see [Actions, Resources, and Condition Keys for AWS Services](https://docs.aws.amazon.com/IAM/latest/UserGuide/referencepoliciesactions-resources-contextkeys.html) in the IAM User Guide.



.
