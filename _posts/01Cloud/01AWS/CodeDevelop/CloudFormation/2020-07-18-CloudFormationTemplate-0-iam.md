---
title: AWS - CodeDevelop - CloudFormation - Template `AWS::IAM`
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, CodeDevelop]
tags: [AWS]
toc: true
image:
---

[toc]

- ref
  - [AWSdoc](https://docs.aws.amazon.com/zh_cn/AWSCloudFormation/latest/UserGuide/aws-resource-iam-role.html)


---

# AWS::IAM

---

## AWS::IAM::User

1. Declaring an IAM user resource

The policy document named giveaccesstoqueueonly gives the user permission to perform all Amazon SQS actions on the Amazon SQS queue resource myqueue, and denies access to all other Amazon SQS queue resources. The Fn::GetAtt function gets the Arn attribute of the AWS::SQS::Queue resource myqueue.

The policy document named giveaccesstotopiconly is added to the user to give the user permission to perform all Amazon SNS actions on the Amazon SNS topic resource mytopic and to deny access to all other Amazon SNS resources. The Ref function gets the ARN of the AWS::SNS::Topic resource mytopic.


```yaml
AWSTemplateFormatVersion: "2010-09-09"
Resources:
  myuser:
    # declare an AWS::IAM::User resource to create an IAM user.
    Type: AWS::IAM::User
    Properties:
      # The user is declared with the path ("/")
      # and a login profile with the password (myP@ssW0rd).
      Path: "/"
      LoginProfile:
        Password: myP@ssW0rd

      Policies:

      - PolicyName: giveaccesstoqueueonly
        PolicyDocument:
          Version: '2012-10-17'
          Statement:
          # gives the user permission to perform all Amazon SQS actions on the Amazon SQS queue resource myqueue,
          # and denies access to all other Amazon SQS queue resources.
          # The Fn::GetAtt function gets the Arn attribute of the AWS::SQS::Queue resource myqueue.
          - Effect: Allow
            Action: sqs:*
            Resource: !GetAtt myqueue.Arn
          - Effect: Deny
            Action: sqs:*
            NotResource: !GetAtt myqueue.Arn

      - PolicyName: giveaccesstotopiconly
        PolicyDocument:
          Version: '2012-10-17'
          Statement:
          # give the user permission to perform all Amazon SNS actions on the Amazon SNS topic resource mytopic
          # and to deny access to all other Amazon SNS resources.
          # The Ref function gets the ARN of the AWS::SNS::Topic resource mytopic.
          - Effect: Allow
            Action: sns:*
            Resource: !Ref mytopic
          - Effect: Deny
            Action: sns:*
            NotResource: Ref mytopic
```


---

## AWS::IAM::Role

The AssumeRolePolicyDocument
- describes who can assume the role, and under what conditions.
- The trust policy that is associated with this role. Trust policies define which entities can assume the role.
- can associate only one trust policy with a role.

The ManagedPolicyArns
- ARNs of policies that describe what someone assuming that role can do.
- reference them instead of copy their contents.
- A list of Amazon Resource Names (ARNs) of the IAM managed policies that you want to attach to the role.
- This way if the service adds new features or something that require new permissions, they'll just work instead of you having to go in and change them.

Policies
- Adds or updates an inline policy document that is embedded in the specified IAM role.
- When you embed an inline policy in a role, the inline policy is used as part of the role's access (permissions) policy.
- The role's trust policy is created at the same time as the role. You can update a role's trust policy later.
- A role can also have an attached managed policy.

```yaml
AWSTemplateFormatVersion: "2010-09-09"
Resources:
  Role:
    Type: 'AWS::IAM::Role'
    Properties:
      RoleName: my-role1
      Path: /
      AssumeRolePolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Sid: myAssumePolicy
            Effect: Allow
            Action: 'sts:AssumeRole'
            Principal:
              Service: ec2.amazonaws.com
              AWS: !Sub 'arn:aws:iam::12345678:role/role2'
      Policies:
        - PolicyName: myPolicy
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Effect: Allow
                Action: '*'
                Resource: '*'
      # apply your existing IAM managed policy to your new IAM role
      ManagedPolicyArns:
        - 'arn:aws:iam::aws:policy/ReadOnlyAccess'
```



### IAM Role with Embedded Policy and Instance Profiles
- This example shows an embedded policy in the `AWS::IAM::Role`.
- The policy is specified inline in the Policies property of the `AWS::IAM::Role`.

```yaml
AWSTemplateFormatVersion: "2010-09-09"
Resources:

  RootRole:
    Type: 'AWS::IAM::Role'
    Properties:
      AssumeRolePolicyDocument:
        # policy yaml
        Version: "2012-10-17"
        Statement:
          - Sid: mypolicy
            Effect: Allow
            Principal:
              Service: ec2.amazonaws.com
            Action: 'sts:AssumeRole'

      Path: /
      Policies:
        - PolicyName: root
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Effect: Allow
                Action: '*'
                Resource: '*'

  RootInstanceProfile:
    Type: 'AWS::IAM::InstanceProfile'
    Properties:
      Path: /
      Roles:
        - !Ref RootRole
```

### IAM Role with External Policy and Instance Profiles
- the Policy and InstanceProfile resources are specified externally to the IAM Role.
- They refer to the role by specifying its name, "RootRole", in their respective Roles properties.

```yaml
AWSTemplateFormatVersion: "2010-09-09"
Resources:

  RootRole:
    Type: "AWS::IAM::Role"
    Properties:
      AssumeRolePolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Effect: "Allow"
            Principal:
              Service: "ec2.amazonaws.com"
            Action: "sts:AssumeRole"
      Path: "/"

  RolePolicies:
    Type: "AWS::IAM::Policy"
    Properties:
      PolicyName: "root"
      PolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Effect: "Allow"
            Action: "*"
            Resource: "*"
      Roles:
        - Ref: "RootRole"

  RootInstanceProfile:
    Type: "AWS::IAM::InstanceProfile"
    Properties:
      Path: "/"
      Roles:
        - Ref: "RootRole"

```



### IAM role with EC2



```yaml
# the instance profile is referenced by the IamInstanceProfile property of the EC2 Instance.
# Both the instance policy and role policy reference AWS::IAM::Role.
AWSTemplateFormatVersion: '2010-09-09'
Resources:

  RootRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
        - Effect: Allow
          Principal:
            Service: ec2.amazonaws.com
          Action: sts:AssumeRole
      Path: "/"

  RolePolicies:
    Type: AWS::IAM::Policy
    Properties:
      PolicyName: root
      PolicyDocument:
        Version: '2012-10-17'
        Statement:
        - Effect: Allow
          Action: "*"
          Resource: "*"
      Roles: !Ref RootRole

  RootInstanceProfile:
    Type: AWS::IAM::InstanceProfile
    Properties:
      Path: "/"
      Roles: !Ref RootRole

  myEC2Instance:
    Type: AWS::EC2::Instance
    Version: '2009-05-15'
    Properties:
      ImageId: ami-0ff8a91507f77f867
      InstanceType: m1.small
      Monitoring: 'true'
      DisableApiTermination: 'false'
      IamInstanceProfile: !Ref RootInstanceProfile

```


### IAM role with AutoScaling group

```yaml
AWSTemplateFormatVersion: '2010-09-09'

Resources:

  RootRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
        - Effect: Allow
          Principal:
            Service: ec2.amazonaws.com
          Action: sts:AssumeRole
      Path: "/"

  RolePolicies:
    Type: AWS::IAM::Policy
    Properties:
      PolicyName: root
      PolicyDocument:
        Version: '2012-10-17'
        Statement:
        - Effect: Allow
          Action: "*"
          Resource: "*"
      Roles: !Ref RootRole

  RootInstanceProfile:
    Type: AWS::IAM::InstanceProfile
    Properties:
      Path: "/"
      Roles: !Ref RootRole

  myLCOne:
    Type: AWS::AutoScaling::LaunchConfiguration
    Version: '2009-05-15'
    Properties:
      ImageId: ami-0ff8a91507f77f867
      InstanceType: m1.small
      InstanceMonitoring: 'true'
      IamInstanceProfile: !Ref RootInstanceProfile

  myASGrpOne:
    Type: AWS::AutoScaling::AutoScalingGroup
    Version: '2009-05-15'
    Properties:
      AvailabilityZones: "us-east-1a"
      LaunchConfigurationName: !Ref myLCOne
      MinSize: '0'
      MaxSize: '0'
      HealthCheckType: EC2
      HealthCheckGracePeriod: '120'
```





---


## AWS::IAM::AccessKey

1. Declaring an IAM access key resource


```yaml
# The myaccesskey resource creates an access key
# and assigns it to an IAM user that is declared as an AWS::IAM::User resource in the template.
myaccesskey:
  Type: AWS::IAM::AccessKey
  Properties:
    UserName: !Ref myuser


# get the secret key for an AWS::IAM::AccessKey resource using the Fn::GetAtt function.
# The only time that you can get the secret key for an AWS access key is when it is created.
# One way to retrieve the secret key is to put it into an Output value.
# You can get the access key using the Ref function.
# The following Output value declarations get the access key and secret key for myaccesskey.
AccessKeyformyaccesskey:
  Value: !Ref myaccesskey
SecretKeyformyaccesskey:
  Value: !GetAtt myaccesskey.SecretAccessKey


# You can also pass the AWS access key and secret key to an EC2 instance or Auto Scaling group defined in the template.
# uses the UserData property to pass the access key and secret key for the myaccesskey resource.
myinstance:
  Type: AWS::EC2::Instance
  Properties:
    AvailabilityZone: "us-east-1a"
    ImageId: ami-0ff8a91507f77f867
    UserData:
      Fn::Base64: !Sub "ACCESS_KEY=${myaccesskey}&SECRET_KEY=${myaccesskey.SecretAccessKey}

```


---


## AWS::IAM::Group

1. Declaring an IAM group resource

```yaml
mygroup:
  Type: AWS::IAM::Group
  Properties:
    # The group has a path ("/myapplication/").
    Path: "/myapplication/"
    Policies:
    - PolicyName: myapppolicy
      # The policy document named myapppolicy is added to the group
      # to allow the group's users to perform all Amazon SQS actions on the Amazon SQS queue resource myqueue and deny access to all other Amazon SQS resources except myqueue.
      PolicyDocument:
        Version: '2012-10-17'
        Statement:
        - Effect: Allow
          Action: sqs:*
          Resource: !GetAtt myqueue.Arn
          # To assign a policy to a resource, IAM requires the Amazon Resource Name (ARN) for the resource.
          # Fn::GetAtt function gets the ARN of the AWS::SQS::Queue resource queue.
        - Effect: Deny
          Action: sqs:*
          NotResource: !GetAtt myqueue.Arn

```

2. Adding users to a group


```yaml

# The AWS::IAM::UserToGroupAddition resource adds users to a group.
# the addUserToGroup resource adds the following users to an existing group named myexistinggroup2:
# the existing user existinguser1 and the user myuser which is declared as an AWS::IAM::User resource in the template.
addUserToGroup:
  Type: AWS::IAM::UserToGroupAddition
  Properties:
    GroupName: myexistinggroup2
    Users:
    - existinguser1  # 1st user
    - !Ref myuser    # 2nd user
```



## AWS::IAM::Policy

1. Declaring an IAM policy

```yaml
# create a policy and apply it to multiple groups using an AWS::IAM::Policy resource named mypolicy.
mypolicy:
  Type: AWS::IAM::Policy
  Properties:
    PolicyName: mygrouppolicy
    PolicyDocument:
      Version: '2012-10-17'
      Statement:
      #a PolicyDocument property that allows GetObject, PutObject, and PutObjectAcl actions on the objects in the S3 bucket represented by the ARN arn:aws:s3:::myAWSBucket
      - Effect: Allow
        Action:
        - s3:GetObject
        - s3:PutObject
        - s3:PutObjectAcl
        Resource: arn:aws:s3:::myAWSBucket/*
    # applies the policy to an existing group named myexistinggroup1 and a group mygroup
    Groups:
    - myexistinggroup1
    - !Ref mygroup
    Users:
    - existinguser1  # 1st user
    - !Ref myuser    # 2nd user
```









.
