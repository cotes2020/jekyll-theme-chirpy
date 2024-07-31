---
title: AWS - IdenAccessManage - IAM policy
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, IdenAccessManage]
tags: [AWS, IdenAccessManage]
toc: true
image:
---

- [IAM policy](#iam-policy)
  - [IAM JSON policy elements reference](#iam-json-policy-elements-reference)
    - [IAM JSON policy elements: Version](#iam-json-policy-elements-version)
    - [IAM JSON policy elements: Id](#iam-json-policy-elements-id)
    - [IAM JSON policy elements: Statement (required)](#iam-json-policy-elements-statement-required)
    - [IAM JSON policy elements: Sid (statement ID)](#iam-json-policy-elements-sid-statement-id)
    - [IAM JSON policy elements: Effect (required)](#iam-json-policy-elements-effect-required)
    - [AWS JSON policy elements: Principal](#aws-json-policy-elements-principal)
    - [AWS JSON policy elements: NotPrincipal](#aws-json-policy-elements-notprincipal)
    - [IAM JSON policy elements: Action](#iam-json-policy-elements-action)
    - [IAM JSON policy elements: NotAction](#iam-json-policy-elements-notaction)
    - [IAM JSON policy elements: Resource](#iam-json-policy-elements-resource)
    - [IAM JSON policy elements: NotResource](#iam-json-policy-elements-notresource)
    - [IAM JSON policy elements: Condition](#iam-json-policy-elements-condition)
      - [String condition operators](#string-condition-operators)
      - [Numeric condition operators](#numeric-condition-operators)
      - [Date condition operators](#date-condition-operators)
      - [Boolean condition operators](#boolean-condition-operators)
      - [Binary condition operators](#binary-condition-operators)
      - [IP address condition operators](#ip-address-condition-operators)
      - [Amazon Resource Name (ARN) condition operators](#amazon-resource-name-arn-condition-operators)
      - [...IfExists condition operators](#ifexists-condition-operators)
      - [Condition operator to check existence of condition keys](#condition-operator-to-check-existence-of-condition-keys)
    - [Variables and tags](#variables-and-tags)
    - [Supported data types](#supported-data-types)
  - [example](#example)
  - [AWS: Allows access based on date and time](#aws-allows-access-based-on-date-and-time)
  - [AWS: Denies access to AWS based on the requested Region](#aws-denies-access-to-aws-based-on-the-requested-region)
  - [AWS: Denies access to AWS based on the source IP](#aws-denies-access-to-aws-based-on-the-source-ip)

---

# IAM policy

---

## IAM JSON policy elements reference

---


### IAM JSON policy elements: Version


```yaml
Version : "2012-10-17"
```

---


### IAM JSON policy elements: Id

The Id element specifies an optional identifier for the policy.
- The ID is used differently in different services.
- For services that let you set an ID element, we recommend you use a UUID (GUID) for the value, or incorporate a UUID as part of the ID to ensure uniqueness.

```yaml
"Id": "cd3ad3d9-2776-4ef1-a904-4c229d1642ee"
```

---


### IAM JSON policy elements: Statement (required)


The Statement element is the main element for a policy.
- The Statement element can contain a single statement or an array of individual statements.
- Each individual statement block must be enclosed in curly braces { }.
- For multiple statements, the array must be enclosed in square brackets [ ].

```yaml
"Statement": [
    {...},
    {...},
    {...}
]

Version: '2012-10-17'
Statement:
- Effect: Allow
  Action:
  - s3:ListAllMyBuckets
  - s3:GetBucketLocation
  Resource: arn:aws:s3:::*

- Effect: Allow
  Action: s3:ListBucket
  Resource: arn:aws:s3:::BUCKET-NAME
  Condition:
    StringLike:
      s3:prefix:
      - ''
      - home/
      - home/${aws:username}/

- Effect: Allow
  Action: s3:*
  Resource:
  - arn:aws:s3:::BUCKET-NAME/home/${aws:username}
  - arn:aws:s3:::BUCKET-NAME/home/${aws:username}/*
```


---


### IAM JSON policy elements: Sid (statement ID)

The Sid (statement ID) is an optional identifier that you provide for the policy statement.
- You can assign a Sid value to each statement in a statement array.
- In services that let you specify an ID element, such as SQS and SNS,
- the Sid value is just a sub-ID of the policy document's ID.
- In IAM, the Sid value must be unique within a JSON policy.

```yaml
"Sid": "1"
```

---


### IAM JSON policy elements: Effect (required)

- specifies whether the statement results in an allow or an explicit deny.
- Valid values for Effect are `Allow` and `Deny`.

```yaml
"Effect":"Allow/Deny"
```


---


### AWS JSON policy elements: Principal

- to specify the principal that is allowed or denied access to a resource.
- You cannot use the Principal element in an IAM identity-based policy.
- You can use it in the trust policies for IAM roles and in resource-based policies.
  - Resource-based policies are policies that you embed directly in a resource.
  - For example, you can embed policies in an Amazon S3 bucket or an AWS KMS customer master key (CMK).

You can specify any of the following principals in a policy:

- AWS account and root user

- Specific AWS accounts
  - All identities inside the account can access the resource if they have the appropriate IAM permissions attached to explicitly allow access.
  - This includes IAM users and roles in that account.
  - to specify an AWS account,use the account ARN `arn:aws:iam::AWS-account-ID:root`, or a shortened form `AWS:account ID`

    ```yaml
    "Principal": { "AWS": "arn:aws:iam::123456789012:root" }
    "Principal": { "AWS": "123456789012" }

    # more than one AWS account as a principal using an array,
    "Principal": {
        "AWS": [
            "arn:aws:iam::123456789012:root",
            "999999999999"
        ]
    }
    ```

- IAM users

    ```yaml
    "Principal": { "AWS": "arn:aws:iam::AWS-account-ID:user/user-name" }
    "Principal": {
    "AWS": [
        "arn:aws:iam::AWS-account-ID:user/user-name-1",
        "arn:aws:iam::AWS-account-ID:user/UserName2"
    ]
    }
    ```


- Federated users (using web identity or SAML federation)

    ```yaml
    # "Principal": { "Federated": "arn:aws:iam::AWS-account-ID:saml-provider/provider-name" }
    "Principal": { "Federated": "cognito-identity.amazonaws.com" }
    "Principal": { "Federated": "www.amazon.com" }
    "Principal": { "Federated": "graph.facebook.com" }
    "Principal": { "Federated": "accounts.google.com" }
    ```


- IAM roles

    ```yaml
    "Principal": { "AWS": "arn:aws:iam::AWS-account-ID:role/role-name" }
    ```


- Assumed-role sessions

    ```yaml
    "Principal": { "AWS": "arn:aws:sts::AWS-account-ID:assumed-role/role-name/role-session-name" }
    ```

- AWS services

    ```yaml
    Version: '2012-10-17'
    Statement:
    - Effect: Allow
    Principal:
        Service:
        - ecs.amazonaws.com
        - elasticloadbalancing.amazonaws.com
    Action: sts:AssumeRole
    ```

- Anonymous users (not recommended)

    ```yaml
    "Principal": "*"
    "Principal" : { "AWS" : "*" }
    ```


Use the Principal element in these ways:
- In IAM roles, use the Principal element in the role's trust policy to specify who can assume the role.
- For cross-account access, you must specify the 12-digit identifier of the trusted account.
- In resource-based policies, use the Principal element to specify the accounts or users who are allowed to access the resource.

Note:
1. Do not use the Principal element in policies that you attach to IAM users and groups.
2. do not specify a principal in the permission policy for an IAM role.
   - In those cases, the principal is implicitly the user that the policy is attached to (for IAM users) or the user who assumes the role (for role access policies).
   - When the policy is attached to an IAM group, the principal is the IAM user in that group who is making the request.




---


### AWS JSON policy elements: NotPrincipal

Example IAM user in the same or a different account

```yaml
Version: '2012-10-17'
Statement:


# all principals  are explicitly denied access to a resource.
# except the user named Bob in AWS account 444455556666
- Effect: Deny
  Action: s3:*
  Resource:
  - arn:aws:s3:::BUCKETNAME
  - arn:aws:s3:::BUCKETNAME/*
  NotPrincipal:
    AWS:
    - arn:aws:iam::444455556666:user/Bob
    - arn:aws:iam::444455556666:root


# all principals are explicitly denied access to a resource.
# except the assumed-role user named cross-account-audit-app in AWS account 444455556666
- Effect: Deny
  Action: s3:*
  Resource:
  - arn:aws:s3:::BUCKETNAME
  - arn:aws:s3:::BUCKETNAME/*
  NotPrincipal:
    AWS:
    - arn:aws:sts::444455556666:assumed-role/cross-account-read-only-role/cross-account-audit-app
    - arn:aws:iam::444455556666:role/cross-account-read-only-role
    - arn:aws:iam::444455556666:root

```


---


### IAM JSON policy elements: Action


---


### IAM JSON policy elements: NotAction


---


### IAM JSON policy elements: Resource


---


### IAM JSON policy elements: NotResource


---


### IAM JSON policy elements: Condition


Use condition operators in the `Condition` element
- to match the condition key and value in the policy against values in the request context.
- The condition operator use in a policy depends on the condition key
  - can choose a global condition key or a service-specific condition key.
  - If the key that you specify in a policy condition is not present in the request context, the values do not match.
  - This applies to all condition operators `except ...IfExists` and `Null check`. These operators test whether the key is present (exists) in the request context.

The condition operators can be grouped into the following categories:

- `String`

- `Numeric`

- `Date`

- `Boolean`

- `Binary`

- `IPAddress`

- `Arn` (available for only some services.)

- `IfExists` (checks if the key value exists as part of another check)

- `Null check` (checks if the key value exists as a standalone check)

---


#### String condition operators

- restrict access based on comparing a key to a string value.

Condition operator | Description
---|---
`StringEquals` | Exact matching, case sensitive
`StringNotEquals` | Negated matching
`StringEqualsIgnoreCase` | Exact matching, ignoring case
`StringNotEqualsIgnoreCase` | Negated matching, ignoring case
`StringLike` | Case-sensitive matching. The values can include a multi-character match wildcard `*` or a single-character match wildcard `?` anywhere in the string. <br> If a key contains multiple values, `StringLike` can be qualified with set operators—`ForAllValues:StringLike` and `ForAnyValue:StringLike`.
`StringNotLike` | Negated case-sensitive matching. The values can include a multi-character match wildcard `*` or a single-character match wildcard `?` anywhere in the string.

Example, the following statement contains a `Condition` element that

```json

{
  "Version": "2012-10-17",
  "Statement": {
    "Effect": "Allow",
    "Action": "iam:*AccessKey*",
    "Resource": "arn:aws:iam::ACCOUNT-ID-WITHOUT-HYPHENS:user/*",
    "Condition": {
        "StringEquals": {
            "aws:PrincipalTag/job-category": "iamuser-admin"
        }
        // uses the `StringEquals` condition operator with the `aws:PrincipalTag` key
        // to specify that the principal making the request must be tagged with the `iamuser-admin` job category.
    }
  }
}

// the `aws:PrincipalTag/job-category` key is present in the request context
// if the principal is using an IAM user with attached tags. It is also included for a principal using an IAM role with attached tags or session tags.
// If a user without the tag attempts to view or edit an access key, the condition returns `false` and the request is implicitly denied by this statement.
```



```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListAllMyBuckets",
        "s3:GetBucketLocation"
      ],
      "Resource": "arn:aws:s3:::*"
    },
    {
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::BUCKET-NAME",
      "Condition": {"StringLike": {"s3:prefix": ["","home/","home/${aws:username}/"]}}
    },
    // The policy allows the specified actions on an S3 bucket
    // as long as the `s3:prefix` matches any one of the specified patterns.
    {
      "Effect": "Allow",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::BUCKET-NAME/home/${aws:username}",
        "arn:aws:s3:::BUCKET-NAME/home/${aws:username}/*"
      ]
    //   lets an IAM user use the Amazon S3 console to manage his or her own "home directory" in an Amazon S3 bucket.
    }
  ]
}
```

restrict access to resources based on an application ID and a user ID for web identity federation, [link](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_examples_s3_cognito-bucket.html)


---


#### Numeric condition operators

Numeric condition operators let you construct `Condition` elements that restrict access based on comparing a key to an integer or decimal value.

Condition operator | Description
---|---
`NumericEquals` | Matching
`NumericNotEquals` | Negated matching
`NumericLessThan` | "Less than" matching
`NumericLessThanEquals` | "Less than or equals" matching
`NumericGreaterThan` | "Greater than" matching
`NumericGreaterThanEquals` | "Greater than or equals" matching


```json
{
  "Version": "2012-10-17",
  "Statement": {
    "Effect": "Allow",
    "Action": "s3:ListBucket",
    "Resource": "arn:aws:s3:::example_bucket",
    "Condition": {"NumericLessThanEquals": {"s3:max-keys": "10"}}
    // specify that the requester can list _up to_ 10 objects in `example_bucket` at a time.
  }
}
// the `s3:max-keys` key is always present in the request when perform the `ListBucket` operation.
// If this policy allowed all Amazon S3 operations, then only the operations that include the `max-keys` context key with a value of less than or equal to 10 would be allowed.

```



---


#### Date condition operators

- restrict access based on comparing a key to a date/time value.
- use these condition operators with the `aws:CurrentTime` key or `aws:EpochTime` keys.
- Wildcards are not permitted for date condition operators.

Condition operator | Description
---|---
`DateEquals` | Matching a specific date
`DateNotEquals` | Negated matching
`DateLessThan` | Matching before a specific date and time
`DateLessThanEquals` | Matching at or before a specific date and time
`DateGreaterThan` | Matching after a specific a date and time
`DateGreaterThanEquals` | Matching at or after a specific date and time

Example, the following statement contains a `Condition` element that uses the `DateGreaterThan` condition operator with the `aws:TokenIssueTime` key. This condition

```json
{
  "Version": "2012-10-17",
  "Statement": {
    "Effect": "Allow",
    "Action": "iam:*AccessKey*",
    "Resource": "arn:aws:iam::ACCOUNT-ID-WITHOUT-HYPHENS:user/*",
    "Condition": {"DateGreaterThan": {"aws:TokenIssueTime": "2020-01-01T00:00:01Z"}}
    // specifies that the temporary security credentials used to make the request were issued in 2020.
    // This policy can be updated programmatically every day to ensure that account members use fresh credentials.
  }
}
// If the key that you specify in a policy condition is not present in the request context, the values do not match.
// The `aws:TokenIssueTime` key is present in the request context only when the principal uses temporary credentials to make the request.
// They key is not present in AWS CLI, AWS API, or AWS SDK requests that are made using access keys.
// In this example, if an IAM user attempts to view or edit an access key, the request is denied.

```


---


#### Boolean condition operators

- restrict access based on comparing a key to "true" or "false."

Condition operator | Description
---|---
`Bool` | Boolean matching

Example
- uses the `Bool` condition operator with the `aws:SecureTransport` key to specify that the request must use SSL.
- If the key that you specify in a policy condition is not present in the request context, the values do not match.
- The `aws:SecureTransport` key is always present in the request context.


```json
{
  "Version": "2012-10-17",
  "Statement": {
    "Effect": "Allow",
    "Action": "iam:*AccessKey*",
    "Resource": "arn:aws:iam::ACCOUNT-ID-WITHOUT-HYPHENS:user/*",
    "Condition": {"Bool": {"aws:SecureTransport": "true"}}
  }
}
```


---


#### Binary condition operators

- onstruct `Condition` elements that test key values that are in binary format.
- It compares the value of the specified key byte for byte against a [base-64](https://en.wikipedia.org/wiki/Base64) encoded representation of the binary value in the policy.

If the key that you specify in a policy condition is not present in the request context, the values do not match.

```json
"Condition" : {
  "BinaryEquals": {
    "`key`" : "QmluYXJ5VmFsdWVJbkJhc2U2NA=="
  }
}
```


---


#### IP address condition operators

- construct `Condition` elements that restrict access based on comparing a key to an IPv4 or IPv6 address or range of IP addresses.
- use these with the `aws:SourceIp` key.
- The value must be in the standard CIDR format (Example, 203.0.113.0/24 or 2001:DB8:1234:5678::/64).
- If you specify an IP address without the associated routing prefix, IAM uses the default prefix value of `/32`.
- Some AWS services support IPv6, using `::` to represent a range of 0s.

Condition operator | Description
---|---
`IpAddress` | The specified IP address or range
`NotIpAddress` | All IP addresses except the specified IP address or range

Example
- uses the `IpAddress` condition operator with the `aws:SourceIp` key to specify that the request must come from the IP range 203.0.113.0 to 203.0.113.255.
- The `aws:SourceIp` condition key resolves to the IP address that the request originates from.
- If the requests originates from an Amazon EC2 instance, `aws:SourceIp` evaluates to the instance's public IP address.
- If the key that you specify in a policy condition is not present in the request context, the values do not match.
- The `aws:SourceIp` key is always present in the request context, except when the requester uses a VPC endpoint to make the request.
- In this case, the condition returns `false` and the request is implicitly denied by this statement.



```json
{
  "Version": "2012-10-17",
  "Statement": {
    "Effect": "Allow",
    "Action": "iam:*AccessKey*",
    "Resource": "arn:aws:iam::ACCOUNT-ID-WITHOUT-HYPHENS:user/*",
    "Condition": {"IpAddress": {"aws:SourceIp": "203.0.113.0/24"}}
  }
}
```

Example
- mix IPv4 and IPv6 addresses to cover all of your organization's valid IP addresses.
- The `aws:SourceIp` condition key works only in a JSON policy if you are calling the tested API directly as a user.
- If you instead use a service to call the target service on your behalf, the target service sees the IP address of the calling service rather than the IP address of the originating user.
- This can happen, Example, if you use AWS CloudFormation to call Amazon EC2 to construct instances for you. There is currently no way to pass the originating IP address through a calling service to the target service for evaluation in a JSON policy.
- For these types of service API calls, do not use the `aws:SourceIp` condition key.

```json
{
  "Version": "2012-10-17",
  "Statement": {
    "Effect": "Allow",
    "Action": "`someservice`:*",
    "Resource": "*",
    "Condition": {
      "IpAddress": {
        "aws:SourceIp": [
          "203.0.113.0/24",
          "2001:DB8:1234:5678::/64"
        ]
      }
    }
  }
}
```


---


#### Amazon Resource Name (ARN) condition operators

- construct `Condition` elements that restrict access based on comparing a key to an ARN.
- The ARN is considered a string.
- Not all services support comparing ARNs using this operator.

Condition operator | Description
---|---
`ArnEquals`, `ArnLike` | Case-sensitive matching of the ARN. Each of the six colon-delimited components of the ARN is checked separately and each can include a multi-character match wildcard `*` or a single-character match wildcard `?`. These behave identically.
`ArnNotEquals`, `ArnNotLike` | Negated matching for ARN. These behave identically.


Example
- policy attached to an Amazon SQS queue to which you want to send SNS messages.
- It gives Amazon SNS permission to send messages to the queue (or queues) of your choice, but only if the service is sending the messages on behalf of a particular Amazon SNS topic (or topics).
- specify the queue in the `Resource` field, and the Amazon SNS topic as the value for the `SourceArn` key.
- If the key that you specify in a policy condition is not present in the request context, the values do not match.
- The `aws:SourceArn` key is present in the request context only if a resource triggers a service to call another service on behalf of the resource owner.
- If an IAM user attempts to perform this operation directly, the condition returns `false` and the request is implicitly denied by this statement.

```json
{
  "Version": "2012-10-17",
  "Statement": {
    "Effect": "Allow",
    "Action": "SQS:SendMessage",
    "Principal": {"AWS": "`123456789012`"},
    "Resource": "arn:aws:sqs:`REGION`:`123456789012`:`QUEUE-ID`",
    "Condition": {"ArnEquals":
        {"aws:SourceArn": "arn:aws:sns:`REGION`:`123456789012`:`TOPIC-ID`"}}
  }
}
```

---


#### ...IfExists condition operators

- add `IfExists` to the end of any condition operator name except the `Null` condition—Example, `StringLikeIfExists`.
- You do this to say "If the policy key is present in the context of the request, process the key as specified in the policy. If the key is not present, evaluate the condition element as true."
- Other condition elements in the statement can still result in a nonmatch, but not a missing key when checked with `...IfExists`.

- Many condition keys describe information about a certain type of resource and only exist when accessing that type of resource.
- These condition keys are not present on other types of resources.
- This doesn't cause an issue when the policy statement applies to only one type of resource.
- However, there are cases where a single statement can apply to multiple types of resources, such as when the policy statement references actions from multiple services or when a given action within a service accesses several different resource types within the same service.
- In such cases, including a condition key that applies to only one of the resources in the policy statement can cause the `Condition` element in the policy statement to fail such that the statement's `"Effect"` does not apply.


Example:


```json
{
  "Version": "2012-10-17",
  "Statement": {
    "Sid": "`THISPOLICYDOESNOTWORK`",
    "Effect": "Allow",
    "Action": "ec2:RunInstances",
    "Resource": "*",
    "Condition": {"StringLike":
        {"ec2:InstanceType": ["t1.*","t2.*","m3.*"]}}
  }
}
```

- enable the user to launch any instance that is type t1, t2 or m3.
- However, launching an instance actually requires accessing many resources in addition to the instance itself; Example, images, key pairs, security groups, etc.
- The entire statement is evaluated against every resource that is required to launch the instance.
- These additional resources do not have the `ec2:InstanceType` condition key, so the `StringLike` check fails, and the user is not granted the ability to launch _any_ instance type.
- To address this, use the `StringLikeIfExists` condition operator instead.
  - the test only happens if the condition key exists.
- You could read the following as:
  - If the resource being checked has an "`ec2:InstanceType`" condition key, then allow the action only if the key value begins with `"t1.\*", "t2.\*", or "m3.\*"`.
  - If the resource being checked does not have that condition key, then don't worry about it.
- The `DescribeActions` statement includes the actions required to view the instance in the console.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "RunInstance",
            "Effect": "Allow",
            "Action": "ec2:RunInstances",
            "Resource": "*",
            "Condition": {
                "StringLikeIfExists": {"ec2:InstanceType": ["t1.*","t2.*","m3.*"]}}
        },
        {
            "Sid": "DescribeActions",
            "Effect": "Allow",
            "Action": [
                "ec2:DescribeImages",
                "ec2:DescribeInstances",
                "ec2:DescribeVpcs",
                "ec2:DescribeKeyPairs",
                "ec2:DescribeSubnets",
                "ec2:DescribeSecurityGroups"
            ],
            "Resource": "*"
        }]
}
```


---


#### Condition operator to check existence of condition keys

Use a `Null` condition operator to check if a condition key is present at the time of authorization.
- `true` (the key doesn't exist — it is null) or
- `false` (the key exists and its value is not null).


Example
- determine whether a user is using their own credentials for the operation or temporary credentials.
- If the user is using temporary credentials, then the key `aws:TokenIssueTime` exists and has a value.
- condition that states that the user must not be using temporary credentials (the key must not exist) for the user to use the Amazon EC2 API.

```json
{
  "Version": "2012-10-17",
  "Statement":{
      "Action":"ec2:*",
      "Effect":"Allow",
      "Resource":"*",
      "Condition":{
          "Null":{"aws:TokenIssueTime":"true"}
      }
  }
}
```


---

### Variables and tags




---

### Supported data types







---

## example

## AWS: Allows access based on date and time

```json
// restricts access to actions that occur between April 1, 2020 and June 30, 2020
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "service-prefix:action-name",
            "Resource": "*",
            "Condition": {
                "DateGreaterThan": {"aws:CurrentTime": "2020-04-01T00:00:00Z"},
                "DateLessThan": {"aws:CurrentTime": "2020-06-30T23:59:59Z"}
            }
        }
    ]
}
```


---

## AWS: Denies access to AWS based on the requested Region


```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "DenyAllOutsideRequestedRegions",
            "Effect": "Deny",
            "NotAction": [
                "cloudfront:*",
                "iam:*",
                "route53:*",
                "support:*"
            ],
            // uses the NotAction element with the Deny effect, which explicitly denies access to all of the actions not listed in the statement.
            // Actions in the CloudFront, IAM, Route 53, and AWS Support services should not be denied
            "Resource": "*",
            "Condition": {
                "StringNotEquals": {
                    "aws:RequestedRegion": [
                        "eu-central-1",
                        "eu-west-1",
                        "eu-west-2",
                        "eu-west-3"
                    ]
                }
            }
        }
    ]
}
```



---

## AWS: Denies access to AWS based on the source IP

denies access to all AWS actions in the account when the request comes `from principals outside the specified IP range`.

```json
{
    "Version": "2012-10-17",
    "Statement": {
        "Effect": "Deny",
        "Action": "*",
        "Resource": "*",
        "Condition": {
            "NotIpAddress": {
                "aws:SourceIp": [
                    "192.0.2.0/24",
                    "203.0.113.0/24"
                ]
            },
            "Bool": {"aws:ViaAWSService": "false"}
        }
    }
}
```
