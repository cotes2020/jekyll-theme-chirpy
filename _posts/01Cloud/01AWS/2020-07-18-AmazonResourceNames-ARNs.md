
- [Amazon Resource Name (ARNs)](#amazon-resource-name-arns)
  - [Format](#format)


# Amazon Resource Name (ARNs)

Amazon Resource Names (ARNs) uniquely identify AWS resources.
- require an ARN when you need to specify a resource unambiguously 明白地 across all of AWS, such as in IAM policies, Amazon Relational Database Service (Amazon RDS) tags, and API calls.

## Format
The following are the general formats for ARNs.
- The specific formats depend on the resource.
- the ARNs for some resources omit the Region, the account ID, or both the Region and the account ID.

```
arn:partition:service:region:account-id:resource-id
arn:partition:service:region:account-id:resource-type/resource-id
arn:partition:service:region:account-id:resource-type:resource-id
```

1. partition:
   1. The partition in which the resource is located.
   2. A partition is a group of AWS Regions. Each AWS account is scoped to one partition.
   3. the supported partitions:
      1. `aws` - AWS Regions
      2. `aws-cn` - AWS China Regions
      3. `aws-us-gov` - AWS GovCloud (US) Regions

2. service
   1. The service namespace that identifies the AWS product.
   2. For example, `s3` for Amazon S3 resources.

3. region
   1. The Region.

> For example, `us-east-2` for US East (Ohio).


1. account-id
   1. The ID of the AWS account that owns the resource, without the hyphens.
   2. For example, `123456789012`.

2. resource-id
   1. The resource identifier.
   2. This part of the ARN can be the name or ID of the resource or a resource path.
   3. For example,
   4. user/Bob for an IAM user
   5. instance/i-1234567890abcdef0 for an EC2 instance.
   6. Some resource identifiers include a parent resource (sub-resource-type/parent-resource/sub-resource) or a qualifier such as a version (resource-type:resource-name:qualifier).

3. Paths in ARNs
   1. Resource ARNs can include a path.
   2. For example, in Amazon S3, the resource identifier is an object name that can include slashes (/) to form a path.
   3. Similarly, IAM user names and group names can include paths.
   4. Paths can include a wildcard character, namely an asterisk (*).

> For example, if you are writing an IAM policy, you can specify all IAM users that have the path product_1234 using a wildcard as follows:
>
> arn:aws:iam::123456789012:`user/Development/product_1234/*`
>
> Similarly, you can specify `user/*` to mean all users or group/* to mean all groups, as in the following examples:
>
> "Resource":"arn:aws:iam::123456789012:user/*"
> "Resource":"arn:aws:iam::123456789012:group/*"
>
> You cannot use a wildcard to specify all users in the Principal element in a resource-based policy or a role trust policy. Groups are not supported as principals in any policy.
>
> The following example shows ARNs for an Amazon S3 bucket in which the resource name includes a path:
>
> arn:aws:s3:::my_corporate_bucket/*
> arn:aws:s3:::my_corporate_bucket/Development/*
>
> You cannot use a wildcard in the portion of the ARN that specifies the resource type, such as the term user in an IAM ARN.
> The following is not allowed:
>
> arn:aws:iam::123456789012:u*



Resource ARNs
The documentation for AWS Identity and Access Management (IAM) lists the ARNs supported by each service for use in resource-level permissions. For more information, see Actions, Resources, and Condition Keys for AWS Services in the IAM User Guide.


.
