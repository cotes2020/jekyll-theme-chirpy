---
title: AWS - Management - SSM - Session Manager
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, Management]
tags: [AWS]
toc: true
image:
---

- [Session Manager](#session-manager)
  - [session](#session)
  - [Session document schema](#session-document-schema)
  - [赋予System Manager 对实例可执行操作的权限：](#赋予system-manager-对实例可执行操作的权限)
    - [0. setup](#0-setup)
    - [1. 修改 instance profile 和加裝 ssm agent](#1-修改-instance-profile-和加裝-ssm-agent)
      - [Embed permissions for Session Manager actions in a custom instance profile](#embed-permissions-for-session-manager-actions-in-a-custom-instance-profile)
      - [Create a custom IAM instance profile for Session Manager](#create-a-custom-iam-instance-profile-for-session-manager)
    - [2. 確認 instance 上面都有安裝好 SSM agent](#2-確認-instance-上面都有安裝好-ssm-agent)
    - [3: Control user session access to instances](#3-control-user-session-access-to-instances)
      - [Enforce a session document permission check for the AWS CLI](#enforce-a-session-document-permission-check-for-the-aws-cli)
    - [4. 設定 user 的 iam policy](#4-設定-user-的-iam-policy)
      - [end user policies for Session Manager](#end-user-policies-for-session-manager)
      - [administrator policy for Session Manager](#administrator-policy-for-session-manager)
      - [Allow full (administrative) access to all sessions](#allow-full-administrative-access-to-all-sessions)
    - [5. user IAM group](#5-user-iam-group)
    - [6. 設定完以上的基本設定後，登入機器](#6-設定完以上的基本設定後登入機器)
    - [more session preferencse:](#more-session-preferencse)
    - [修改 ssm-user sudo 权限](#修改-ssm-user-sudo-权限)
    - [配置ec2 的安全组：](#配置ec2-的安全组)
    - [记录会话数据](#记录会话数据)
  - [使用 scp](#使用-scp)
    - [設定](#設定)
    - [進階設定](#進階設定)
  - [Port forwarding](#port-forwarding)
  - [Monitoring](#monitoring)
    - [Logging AWS Systems Manager API calls with AWS CloudTrail.](#logging-aws-systems-manager-api-calls-with-aws-cloudtrail)
  - [Reference](#reference)

- ref
  - [aws doc](https://docs.aws.amazon.com/systems-manager/latest/userguide/setup-instance-profile.html)
  - [AWS SSM session manager 筆記 Posted by Kakashi on 2020-04-11](https://kkc.github.io/2020/04/11/aws-ssm-session-manager-note/)
  - [AWS-AWS Systems Manager Session (SSM 会话管理器)试用](https://blog.csdn.net/kozazyh/article/details/88957448)

---


# Session Manager

- 使用 session manger 可以減少 key 的管理，減少資安漏洞
- 透過 proxycommand 可以讓我們建立 ssh tunnel，進而可以使用 scp 等等工具
- port forwarding 可以幫助 developer 測試在 private subnet 的服務
- 搭配 aws cliv2 可以透過 SSO 增加系統安全

- makes it easy to comply with corporate policies that require controlled access to instances, strict security practices, and fully auditable logs with instance access details, while still providing end users with simple one-click cross-platform access to the managed instances.
  - automating common administrative tasks across groups of instances
   - such as registry edits, user management, and software and patch installations.
  - improve security and audit posture, reduce operational overhead by centralizing access control on instances, and reduce inbound instance access.
  - monitor and track instance access and activity, close down inbound ports on instances, or enable connections to instances that do not have a public IP address.
  - grant and revoke access from a single location, and who want to provide one solution to users for Linux, macOS, and Windows Server instances.
  - connect to an instance with just one click from the browser or AWS CLI without having to provide SSH keys.

- integration with AWS IAM
  - can apply granular permissions to control the actions users can perform on instances.
  - Centralized access control to instances using IAM policies
  - single place to grant and revoke access to instances.
  - Using only AWS IAM policies, control which individual users or groups in organization can use Session Manager and which instances they can access.
  - You can also provide temporary access to instances.
   - Example
   - give an on-call user/group of user access to production servers only for the duration of their rotation.


- No open inbound ports and no need to manage bastion hosts or SSH keys
  - provides safe, secure remote management of the instances at scale
  - without logging into the servers,
   - replacing the need for bastion hosts, SSH, or remote PowerShell.
   - Leaving inbound SSH ports and remote PowerShell ports open on the instances greatly increases the risk of entities running unauthorized or malicious commands on the instances.
   - without the need to open inbound ports, manage SSH keys and certificates, bastion hosts, and jump boxes.
  - manage EC2/on-premises instances, and VMs through an `interactive one-click browser-based shell` or through the `AWS CLI`.


- Port forwarding
  - Redirect any port inside the remote instance to a local port on a client.
  - connect to the local port and access the server application that is running inside the instance.
  - Tunneling
   - In a session, use a Session-type SSM document to tunnel traffic, such as http or a custom protocol, between a local port on a client machine and a remote port on an instance.

- AWS PrivateLink support for instances without public IP addresses
  - can set up VPC Endpoints for Systems Manager using AWS PrivateLink to further secure the sessions.
  - AWS PrivateLink limits all network traffic between managed instances, Systems Manager, and Amazon EC2 to the Amazon network.


- One-click access to instances from the console and CLI
  - start a session with a single click.
  - Using the AWS CLI, can also start a session that runs a single command or a sequence of commands.
  - as permissions to instances are provided by IAM policies not SSH keys or other mechanisms, the connection time is greatly reduced.
  - Interactive commands
   - Create a Session-type SSM document that uses a session to interactively run a single command, giving you a way to manage what users can do on an instance.


- Cross-platform support for Windows, Linux, and macOS
  - support for Windows, Linux, and macOS from a single tool.
  - establish secure connections to EC2/on-premises instances, and VMs.
   - support for on-premises servers is provided for the advanced-instances tier only.
  - Example
   - SSH client for Linux and macOS instances
   - RDP connection for Windows Server instances.

- Logging and auditing session activity
  - All actions taken with Systems Manager are recorded by AWS CloudTrail
   - audit changes throughout the environment.
   - receive notifications when a user in the organization starts or ends session activity.
  - provides secure and auditable instance management
   - Note:
   - <font color=blue> Logging is not available for Session Manager sessions that connect through port forwarding or SSH </font>
   - because SSH encrypts all session data, and Session Manager only serves as a tunnel for SSH connections.
  - Logging and auditing capabilities are provided through integration with the following AWS services:
   - AWS CloudTrail
     - captures information about Session Manager API calls made in the AWS account and writes it to log files that are stored in an S3 bucket you specify.
     - One bucket is used for all CloudTrail logs for the account.

   - Amazon S3
     - choose to store session log data in an S3 bucket of the choice for debugging and troubleshooting purposes.
     - Log data can be sent to the S3 bucket with or without encryption using the AWS Key Management Service (AWS KMS) key.

   - Amazon CloudWatch Logs
     - monitor, store, and access log files from various AWS services.
     - You can send session log data to a CloudWatch Logs log group for debugging and troubleshooting purposes.
     - Log data can be sent to the log group with or without AWS KMS encryption using the AWS KMS key. For more information, see Logging session data using Amazon CloudWatch Logs (console).

   - Amazon EventBridge and Amazon Simple Notification Service
     - EventBridge lets you set up rules to detect when changes happen to AWS resources that you specify. You can create a rule to detect when a user in the organization starts or stops a session, and then receive a notification through Amazon SNS (for example, a text or email message) about the event.
     - You can also configure a CloudWatch event to initiate other responses.

   - Console, CLI, and SDK access to Session Manager capabilities
     - The AWS Systems Manager console:
       - includes access to all the Session Manager capabilities for both administrators and end users. You can perform any task that's related to sessions by using the Systems Manager console.
     - The Amazon EC2 console
       - provides the ability for end users to connect to the EC2 instances for which they have been granted session permissions.
     - The AWS CLI
       - includes access to Session Manager capabilities for end users.
       - can start a session, view a list of sessions, and permanently end a session by using the AWS CLI.
       - To use the AWS CLI to run session commands, you must be using version 1.16.12 of the CLI (or later), and you must have installed the Session Manager plugin on local machine.
       - Configurable shell profiles
       - Session Manager provides you with options to configure preferences within sessions. These customizable profiles enable you to define preferences such as shell preferences, environment variables, working directories, and running multiple commands when a session is started.
     - The Session Manager SDK
       - consists of libraries and sample code that enables application developers to build front-end applications, such as custom shells or self-service portals for internal users that natively use Session Manager to connect to instances.
       - Developers and partners can integrate Session Manager into their client-side tooling or Automation workflows using the Session Manager APIs.
       - You can even build custom solutions.

   - Customer key data encryption support
     - configure Session Manager to <font color=blue> encrypt the session data logs send to S3 bucket or stream to a CloudWatch Logs log group </font>
     - configure Session Manager to further encrypt the data transmitted between client machines and instances during sessions.


---

## session

A session is a connection made to an instance using Session Manager.
- Sessions are based on a secure bi-directional communication channel between the client (you) and the remote managed instance that streams inputs and outputs for commands.
- Traffic between a client and a managed instance is encrypted using TLS 1.2, and requests to create the connection are signed using Sigv4.
- This two-way communication enables interactive bash and PowerShell access to instances.
- You can also use an AWS Key Management Service (AWS KMS) key to further encrypt data beyond the default TLS encryption.

Example
- John is an on-call engineer in IT department.
- He receives notification of an issue that requires him to remotely connect to an instance
  - such as a failure that requires troubleshooting or a directive to change a simple configuration option on an instance.
- Using the AWS Systems Manager console, the EC2 console, or the AWS CLI, John starts a session connect to the instance
- When John sends that first command to start the session
  - the Session Manager service authenticates his ID,
  - verifies the permissions granted to him by an IAM policy,
  - checks configuration settings (such as verifying allowed limits for the sessions),
  - and sends a message to SSM Agent to open the two-way connection.
- After the connection is established
- John types the next command, the command output from SSM Agent is uploaded to this communication channel and sent back to his local machine.



---

## Session document schema


The schema elements of a Session document.
- Session Manager uses Session documents to determine which type of session to start
- such as a standard session, a port forwarding session, or a session to run an interactive command.


```yaml
schemaVersion: '1.0'
description: Document to hold regional settings for Session Manager
# Valid values: InteractiveCommands | Port | Standard_Stream

sessionType: Standard_Stream
# The session preferences to use for sessions established using this Session document.
# This element is required for Session documents that are used to create Standard_Stream sessions.
inputs:
  # S3 bucket to send session logs at the end of the sessions.
  s3BucketName: ''
  # The prefix to use when sending logs to the S3 bucket you specified in the s3BucketName input.
  s3KeyPrefix: ''
  s3EncryptionEnabled: true
  # The name of the Amazon CloudWatch Logs (CloudWatch Logs) group to send session logs at the end of the sessions.
  cloudWatchLogGroupName: ''
  # the log group you specified in the cloudWatchLogGroupName input must be encrypted.
  cloudWatchEncryptionEnabled: true
  # If set to true, a continuous stream of session data logs are sent to the cloudWatch log group
  # If set to false, session logs are sent to the log group at the end of the sessions.
  cloudWatchStreamingEnabled: true
  # The ID of the AWS KMS to use to further encrypt data between the local client machines and the EC2 instances be connected to.
  kmsKeyId: ''
  # The Run As feature is only supported for Linux instances.
  # If set to true, must specify a user account exists on the instances you connecting to (if not, sessions will fail to start)
  # By default, sessions are started using the ssm-user account created by the SSM Agent.
  runAsEnabled: true
  runAsDefaultUser: ''
  # The amount of time of inactivity allowed before a session ends. (minutes)
  idleSessionTimeout: '20'
  # The preferences you specify per operating system to apply within sessions
  # such as shell preferences, environment variables, working directories, and running multiple commands when a session is started.
  shellProfile:
    # The shell preferences, environment variables, working directories, and commands you specify for sessions on Windows instances.
    windows: ''
    linux: ''

schemaVersion: '1.0'
description: Document to view a log file on a Linux instance
sessionType: InteractiveCommands
parameters:
  logpath:
    type: String
    description: The log file path to read.
    default: "/var/log/amazon/ssm/amazon-ssm-agent.log"
    allowedPattern: "^[a-zA-Z0-9-_/]+(.log)$"
properties:
  linux:
    commands: "tail -f {{ logpath }}"
    runAsElevated: true


schemaVersion: '1.0'
description: Example document with quotation marks
sessionType: InteractiveCommands
parameters:
  Test:
    type: String
    description: Test Input
    maxChars: 32
properties:
  windows:
    commands: |
        $Test = '{{ Test }}'
        $myVariable = "Computer name is $env:COMPUTERNAME"
        Write-Host "Test variable: $myVariable`.`nInput parameter: $Test"
    runAsElevated: false


schemaVersion: '1.0'
description: Document to open given port connection over Session Manager
sessionType: Port
parameters:
  paramExample:
    type: string
    description: document parameter
properties:
  portNumber: anyPortNumber
```


---

## 赋予System Manager 对实例可执行操作的权限：

---

### 0. setup


1. 创建一个EC2实例：ID:为：i-xxxxxxxxxx
2. 创建一个角色：
   - IAM > 角色 > 创建角色（role-test）
     1. 选择 受信任实体的类型：AWS产品；
     2. 选择 将要使用此角色的服务：EC2 - >  Attach权限策略：选择 AmazonEC2RoleforSSM

3. 把角色(role-test)附加到EC2(i-xxxxxxxxxx)中 ;

![20190402101438357](https://i.imgur.com/qqx5AxQ.png)


---

### 1. 修改 instance profile 和加裝 ssm agent

An <font color=red> instance profile </font>
- a container that passes IAM role information to an EC2 instance at launch.
- This requirement applies to permissions for all AWS Systems Manager capabilities, not only those specific to Session Manager.
- can attach an IAM instance profile to an EC2 instance as launch it or to a previously launched instance.

By default, AWS Systems Manager doesn't have permission to perform actions on the instances.
- must grant access by using an AWS IAM instance profile.
- <font color=blue> AmazonSSMManagedInstanceCore </font>
  - enables an instance to use AWS Systems Manager service core functionality.
  - Depending on the operations plan, might need permissions
- <font color=blue> custom policy for S3 bucket access </font>
  - Case 1:
    - using a VPC endpoint to privately connect VPC to supported AWS services and VPC endpoint services powered by PrivateLink.
    - SSM Agent is Amazon software that is installed on the instances and performs Systems Manager tasks.
    - This agent requires access to specific Amazon-owned S3 buckets.
    - These buckets are publicly accessible.
    - In a private VPC endpoint environment, however, you must explicitly provide access to these buckets:
      ```
      arn:aws:s3:::patch-baseline-snapshot-region/*
      arn:aws:s3:::aws-ssm-region/*
      ```
  - Case 2:
    - use an S3 bucket as part of the Systems Manager operations.
    - the EC2 instance profile for Systems Manager must grant access to an S3 bucket that you own for tasks like the following:
      - To access scripts you store in the S3 bucket to use in commands you run.
      - To store the full output of Run Command commands or Session Manager sessions.
      - To access custom patch lists for use when patching the instances.
- <font color=blue> AmazonSSMDirectoryServiceAccess </font>
  - Required only if you plan to join EC2 instance for Windows Server to a Microsoft AD directory.
  - This AWS managed policy allows SSM Agent to access AWS Directory Service on your behalf for requests to join the domain by the managed instance.
- <font color=blue> CloudWatchAgentServerPolicy </font>
  - Required only if you plan to install and run the CloudWatch agent on the instances to read metric and log data on an instance and write it to Amazon CloudWatch.
  - These help you monitor, analyze, and quickly respond to issues or changes to your AWS resources.
  - Your instance profile needs this policy only if you will use features such as Amazon EventBridge or CloudWatch Logs.
  - (You can also create a more restrictive policy that, for example, limits writing access to a specific CloudWatch Logs log stream.)


use case
- If already use other Systems Manager capabilities, such as Run Command or Parameter Store, an instance profile with the required basic permissions for Session Manager might already be attached to the instances.
- If an instance profile that contains the AWS managed policy `AmazonSSMManagedInstanceCore` is attached to the instances, the required permissions for Session Manager are already provided.
- in some cases, might need to modify the permissions attached to the instance profile.
  - example,
  - to provide a narrower set of instance permissions, you have created a custom policy for the instance profile,
  - or you want to use Amazon S3 encryption or AWS KMS encryption options for securing session data.

For these cases, do one of the following to allow Session Manager actions to be performed on the instances:


#### Embed permissions for Session Manager actions in a custom instance profile
- add permissions for Session Manager actions to an existing IAM instance profile that does not rely on the AWS-provided default policy `AmazonSSMManagedInstanceCore` for instance permissions.
- assumes the existing profile already includes other Systems Manager ssm permissions for actions you want to allow access to. This policy alone is not enough to use Session Manager.
- Roles (the role to embed a policy in) > Permissions > <font color=blue> Add inline policy </font>
- Replace the default content with the following
- Review policy page, for Name, enter a name for the inline policy, such as `SessionManagerPermissions`.
- Choose Create policy.

```yaml
Version: '2012-10-17'
Statement:
- Effect: Allow
  Action:
  - ssmmessages:CreateControlChannel
  - ssmmessages:CreateDataChannel
  - ssmmessages:OpenControlChannel
  - ssmmessages:OpenDataChannel
  Resource: "*"
- Effect: Allow
  Action: s3:GetEncryptionConfiguration
  Resource: "*"
# kms:Decrypt permission enables customer key encryption and decryption for session data. If you will use AWS Key Management Service (AWS KMS) encryption for the session data, replace key-name with the Amazon Resource Name (ARN) of the customer master key (CMK) you want to use, in the format arn:aws:kms:us-west-2:111122223333:key/1234abcd-12ab-34cd-56ef-12345EXAMPLE.
# If you will not use AWS KMS encryption for the session data, you can remove the following content from the policy.
# - Effect: Allow
#   Action: kms:Decrypt
#   Resource: key-name
```

#### Create a custom IAM instance profile for Session Manager
1. create a custom AWS IAM instance profile that provides permissions for only Session Manager actions on the instances.
2. can create a policy to provide the permissions needed for logs of session activity to be sent to Amazon S3 and CloudWatch Logs.
   - IAM console > Policies > Create policy > `SessionManagerPermissions`
3. Attach an IAM Role to an Instance and <font color=blue> Attach or Replace an Instance Profile </font>
   - Roles > Create role > AWS service > EC2 > Permissions > `SessionManagerPermissions`
   - Role name (name for the IAM instance profile) > `MySessionManagerInstanceProfile`.


```yaml
# Create instance profile with minimal Session Manager permissions
Version: '2012-10-17'
Statement:
- Effect: Allow
  Action:
  - ssm:UpdateInstanceInformation
  - ssmmessages:CreateControlChannel
  - ssmmessages:CreateDataChannel
  - ssmmessages:OpenControlChannel
  - ssmmessages:OpenDataChannel
  Resource: "*"
- Effect: Allow
  Action: s3:GetEncryptionConfiguration
  Resource: "*"
- Effect: Allow
  Action: kms:Decrypt
  Resource: key-name # arn:aws:kms:us-west-2:111122223333:key/1234abcd-12ab-34cd-56ef-12345EXAMPLE.


# Create instance profile with permissions for Session Manager and S3 and CloudWatch Logs
Version: '2012-10-17'
Statement:
- Effect: Allow
  Action:
  - ssm:UpdateInstanceInformation
  - ssmmessages:CreateControlChannel
  - ssmmessages:CreateDataChannel
  - ssmmessages:OpenControlChannel
  - ssmmessages:OpenDataChannel
  Resource: "*"
- Effect: Allow
  Action:
  - logs:CreateLogStream
  - logs:PutLogEvents
  - logs:DescribeLogGroups
  - logs:DescribeLogStreams
  Resource: "*"
- Effect: Allow
  Action: s3:PutObject
  #To output session logs to an S3 bucket owned by a different AWS account, you must add the IAM s3:PutObject Acl permission to this policy.
  # If this permission isn't added, the account that owns the S3 bucket cannot access the session output logs.
  Resource: arn:aws:s3:::DOC-EXAMPLE-BUCKET/s3-bucket-prefix
- Effect: Allow
  Action: s3:GetEncryptionConfiguration
  Resource: "*"
- Effect: Allow
  Action: kms:GenerateDataKey
  Resource: "*"


# Create a custom policy for S3 bucket access
# required only if you are using a VPC endpoint or using an S3 bucket of your own in your Systems Manager operations.
Version: '2012-10-17'
Statement:
# required only if you are using a VPC endpoint.
- Effect: Allow
  Action: s3:GetObject
  Resource:
  - arn:aws:s3:::aws-ssm-region/*
  - arn:aws:s3:::aws-windows-downloads-region/*
  - arn:aws:s3:::amazon-ssm-region/*
  - arn:aws:s3:::amazon-ssm-packages-region/*
  - arn:aws:s3:::region-birdwatcher-prod/*
  - arn:aws:s3:::aws-ssm-distributor-file-region/*
  - arn:aws:s3:::aws-ssm-document-attachments-region/*
  - arn:aws:s3:::patch-baseline-snapshot-region/*
# required only if you are using an S3 bucket that you created to use in your Systems Manager operations.
- Effect: Allow
  Action:
  - s3:GetObject
  - s3:PutObject
  # required only if you plan to support cross-account access to S3 buckets in other accounts.
  - s3:PutObjectAcl
  - s3:GetEncryptionConfiguration
  # required if your S3 bucket is configured to use encryption.
  Resource:
  - arn:aws:s3:::DOC-EXAMPLE-BUCKET/*
  - arn:aws:s3:::DOC-EXAMPLE-BUCKET
  # If your S3 bucket is configured to use encryption, then the S3 bucket root (for example, arn:aws:s3:::DOC-EXAMPLE-BUCKET) must be listed in the Resource section.
  # Your IAM user, group, or role must be configured with access to the root bucket.
```

---

### 2. 確認 instance 上面都有安裝好 SSM agent

AWS 上面新版的 ubuntu & amazon linux2 都有先裝好了，不過舊的 AMI 就需要自己去安裝。


---

### 3: Control user session access to instances

Session Manager allows <font color=red> centrally grant and revoke user access to instances </font>
- Using IAM policies, control which instances specific users or groups can connect to,
- and you control what Session Manager API actions they can perform on the instances they are given access to.


`Session ID` ARN Formats
- IAM policies for Session Manager access use variables for user names as part of session IDs.
- Session IDs in turn are used in session Amazon Resource Names (ARNs) to control access.
- Session ARNs have the following format:

```
arn:aws:ssm:region-id:account-id:session/session-id
arn:aws:ssm:us-east-2:123456789012:session/JohnDoe-1a2b3c4d5eEXAMPLE
```

1. use a pair of default IAM policies supplied by AWS
   - one for end users and one for administrators, to supply permissions for Session Manager activities.
2. Or create custom IAM policies for different permissions requirements you might have.



#### Enforce a session document permission check for the AWS CLI

When configure Session Manager for your account, the system creates a Session-type SSM document `SSM-SessionManagerRunShell`.
- This SSM document stores your session preferences, such as
  - whether session data is saved in an S3 bucket or Amazon CloudWatch Logs log group,
  - whether session data is encrypted using AWS Key Management Service,
  - whether Run As support is enabled for your sessions.

```yaml
schemaVersion: '1.0'
description: Document to hold regional settings for Session Manager
sessionType: Standard_Stream
inputs:
  s3BucketName: DOC-EXAMPLE-BUCKET
  s3KeyPrefix: MyBucketPrefix
  s3EncryptionEnabled: true
  cloudWatchLogGroupName: MyLogGroupName
  cloudWatchEncryptionEnabled: true
  kmsKeyId: MyKMSKeyID
  runAsEnabled: true
  runAsDefaultUser: MyDefaultRunAsUser
```


By default, if a user in your account was granted permission in their IAM user policy to start sessions, that user has access to the `SSM-SessionManagerRunShell` SSM document.
- This means that when they use the AWS CLI to run the start-session command, and they do not specify a document in the --document-name option, the system uses SSM-SessionManagerRunShell and launches the session.
- The session starts even if the user’s IAM policy doesn’t grant explicit permission to access the `SSM-SessionManagerRunShell` document.

```bash
# doesn’t specify a session document.
aws ssm start-session \
    --target i-02573cafcfEXAMPLE

# specifies the default Session Manager session document.
aws ssm start-session \
    --document-name SSM-SessionManagerRunShell \
    --target i-02573cafcfEXAMPLE
```

To restrict access to the default or any session document
- add a condition element to the user's IAM policy that validates whether the user has explicit access to a session document.
- When this condition is applied, the user must specify a value for the `--document-name` option of the `start-session` AWS CLI command.
- This value is either the default Session Manager session document or a custom session document you created.

```json
// performs a session document access check.
// With this condition element set to true, explicit access to a session document must be granted in the IAM policy for the user to start a session. The following is an example.
{
    "Effect": "Allow",
    "Action": [
        "ssm:StartSession"
    ],
    "Resource": [
        "arn:aws:ec2:region:account-id:instance/instance-id",
        "arn:aws:ssm:region:account-id:document/SSM-SessionManagerRunShell"
    ],
    "Condition": {
        "BoolIfExists": {
            "ssm:SessionDocumentAccessCheck": "true"
        }
    }
}
```

- Using the default `SSM-SessionManagerRunShell` session document is the only case when a document name can be omitted from the start-session CLI command.
- In other cases, the user must specify a value for the `--document-name` option of the start-session AWS CLI command.
- The system checks whether the user has explicit access to the session document they specify.
- Example
  - if a user specifies the name of a custom session document created, the user’s IAM policy must grant them permission to access that document.
  - If a user runs a command to start a session using SSH, the user’s policy must grant them access to the `AWS-StartSSHSession` session document.
    - To start a session using SSH, configuration steps must be completed on both the target instance and the user's local machine. For information, see (Optional) Enable SSH connections through Session Manager.





---

### 4. 設定 user 的 iam policy

#### end user policies for Session Manager
- create IAM end user policies for Session Manager.
- allows users to start sessions from only the Session Manager console / AWS CLI / EC2 console, or from all three.
- These policies provide end users the ability to start a session to a particular instance and the ability to end only their own sessions.


```yaml
Version: '2012-10-17'
Statement:
- Effect: Allow
  Action:
  - ssm:StartSession
  - ssm:SendCommand
  # needed for cases where a user attempts to start a session from the Amazon EC2 console, but a command must be sent to update SSM Agent first.
  Resource:
  - arn:aws:ec2:*:*:instance/*
  # no restrict
  - arn:aws:ec2:region:987654321098:instance/i-02573cafcfEXAMPLE
  # restrict access to specific instances by creating an IAM user policy that includes the IDs of the instances
  - arn:aws:ssm:region:account-id:document/SSM-SessionManagerRunShell
  # the default name of the SSM document that Session Manager creates to store your session configuration preferences. You can create a custom session document and specify it in this policy instead. You can also specify the AWS-provided document AWS-StartSSHSession for users who are starting sessions using SSH.
  Condition:
    BoolIfExists:
      ssm:SessionDocumentAccessCheck: 'true'
      # the system checks that a user has explicit access to the defined session document, SSM-SessionManagerRunShell, before a session is established.
    StringLike":
      ssm:resourceTag/Environment: "staging"
      # 使用 tag 去區別用戶能夠存取的環境，像是 staging or production

- Effect: Allow
  Action:
  - ssm:DescribeSessions
  - ssm:GetConnectionStatus
  - ssm:DescribeInstanceInformation
  - ssm:DescribeInstanceProperties
  - ec2:DescribeInstances
  Resource: "*"

- Effect: Allow
  Action: kms:GenerateDataKey
  # enables the creation of a data encryption key that will be used to encrypt session data. If you won't use AWS KMS key encryption for session data, remove the following content from the policy.
  Resource: key-name

# Allow a user to end only sessions they started
# Method 1: Grant TerminateSession privileges using the variable {aws:username}
# allowed to end only their sessions on those instances.
- Effect: Allow
  Action: ssm:TerminateSession
  Resource: arn:aws:ssm:*:*:session/${aws:username}-*

# Method 2: Grant TerminateSession privileges using tags supplied by AWS
# control which sessions a user can end by using a condition with specific tag key variables in an IAM user policy. The condition specifies that the user can only end sessions that are tagged with one or both of these specific tag key variables and a specified value.
# When a user in your AWS account starts a session, Session Manager applies two resource tags to the session. The first resource tag is aws:ssmmessages:target-id, with which you specify the ID of the target the user is allowed to end. The other resource tag is aws:ssmmessages:session-id, with a value in the format of role-id:caller-specified-role-name.
- Effect: Allow
  Action: ssm:TerminateSession
  Resource: ''
  Condition:
    StringLike:
      # the condition statement lets a user end only the instance i-02573cafcfEXAMPLE.
      ssm:resourceTag/aws:ssmmessages:target-id:
      - i-02573cafcfEXAMPLE
      # for cases where the caller type is User. The value you supply for aws:ssmmessages:session-id is the ID of the user.
      ssm:resourceTag/aws:ssmmessages:session-id"
      - "AIDIODR4TAW7CSEXAMPLE"
      # for cases where the caller type is AssumedRole. You can use the {aws:userid} variable for the value you supply for aws:ssmmessages:session-id. Alternatively, you can hardcode a role ID for the value you supply for aws:ssmmessages:session-id. If you hardcode a role ID, you must provide the value in the format role-id:caller-specified-role-name. For example, AIDIODR4TAW7CSEXAMPLE:MyRole.
      - ${aws:userid}


```

#### administrator policy for Session Manager

- create IAM administrator policies for Session Manager.
- provide administrators the ability to
  - start a session to instances that are tagged with `Key=Finance,Value=WebServers`,
  - create, update, and delete preferences,
  - end only their own sessions.


```yaml
Version: '2012-10-17'
Statement:
- Effect: Allow
  Action:
  - ssm:StartSession
  - ssm:SendCommand
  # needed for cases where a user attempts to start a session from the Amazon EC2 console, but a command must be sent to update SSM Agent first.
  Resource: arn:aws:ec2:us-west-2:987654321098:instance/*
  Condition:
    StringLike:
      ssm:resourceTag/tag-key1: tag-value1
      ssm:resourceTag/Finance": "WebServers"
      # restrict access to instances based on specific Amazon EC2 tags.
      # allowed to start sessions with the condition that the instance is a Finance WebServer (ssm:resourceTag/Finance: WebServer). If the user sends a command to an instance that is not tagged or that has any tag other than Finance: WebServer, the command result will include AccessDenied.
- Effect: Allow
  Action:
  - ssm:DescribeSessions
  - ssm:GetConnectionStatus
  - ssm:DescribeInstanceInformation
  - ssm:DescribeInstanceProperties
  - ec2:DescribeInstances
  Resource: "*"
- Effect: Allow
  Action:
  - ssm:CreateDocument
  - ssm:UpdateDocument
  - ssm:GetDocument
  Resource: arn:aws:ssm:region:account-id:document/SSM-SessionManagerRunShell
- Effect: Allow
  Action: ssm:TerminateSession
  Resource: rn:aws:ssm:*:*:session/${aws:username}-*
```


#### Allow full (administrative) access to all sessions

- allows a user to fully interact with all instances/sessions created by all users for all instances.
- It should be granted only to an Administrator who needs full control over your organization's Session Manager activities.

```yaml
Version: '2012-10-17'
Statement:
- Action:
  - ssm:StartSession
  - ssm:TerminateSession
  - ssm:ResumeSession
  - ssm:DescribeSessions
  - ssm:GetConnectionStatus
  Effect: Allow
  Resource:
  - "*"
```

### 5. user IAM group

1. 在IAM中创建组:group-test
   - IAM > 选择组(group-test) > 添加权限 > 直接附加现有策略  > 选:policy-test
2. 在IAM中创建用户：user-test
   - 把你的帐号加入group-test组中
   - (注意：你的帐号只分配凭证就行，不需要设置密码,如果需要登录web控制,才需要设置密码)



### 6. 設定完以上的基本設定後，登入機器

```bash
# 安装 AWS CLI
$ sudo pip install --upgrade awscli
# 为 AWS CLI 安装 Session Manager Plugin
$ curl "https://s3.amazonaws.com/session-manager-downloads/plugin/latest/linux_64bit/session-manager-plugin.rpm" -o "session-manager-plugin.rpm"
$ sudo yum install -y session-manager-plugin.rpm
# 验证安装
$ session-manager-plugin
# Session-Manager-Plugin is installed successfully. Use AWSCLI to start a session.


# 启动SSM Session
$ aws ssm start-session \
  --target i-0b0d92751733d1234
# Starting session with SessionId: user-test-xxxxab2b33333333
# sh-4.2$
# sh-4.2$ ls
# /tmp
# 通过SSM登录上EC2(i-xxxxxxxxxx)，并不需要ssh。
```

### more session preferencse:

[link](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-getting-started-configure-preferences.html)


### 修改 ssm-user sudo 权限

当支持 Session Manager 的某个 SSM 代理版本在实例上启动时，它会创建一个名为 ssm-user 的具有根或管理员权限的用户账户。
- 在 Linux 计算机上，此账户添加到 /etc/sudoers。


```bash
# 连接到实例并运行以下命令：
sudo cd /etc/sudoers.d

# 打开名为 ssm-agent-users 的文件进行编辑。

# 删除 sudo 访问权限，请删除以下行：
ssm-user ALL=(ALL) NOPASSWD:ALL

# 要恢复 sudo 访问权限，请添加以下行：
ssm-user ALL=(ALL) NOPASSWD:ALL
```


###  配置ec2 的安全组：

入站：取消所有规则

出站：
1. 开放tcp/udp 53端口
2. 开放tcp/443 端口 （因为需要此端口访问aws ssm服务器）
3. 开放tcp/6443 连接k8s的端口以及只允许k8s master IP（如果想只允许访问k8s API）;


### 记录会话数据


启动session后，所有对ec2执行的命令以及返回的结果，可以记录在CloudWatch Logs中。
1. 在CloudWatch中 新建日志组：session-logs
2. 在System Manager 的会话管理器中启用日志流
   - 日志组选择:session-logs
3. 启动ssm session 后，执行的所有操作，在CloudWatch的日志组（session-logs)都可以查到：


![20190402100145503](https://i.imgur.com/iIqBMgE.png)

![2019040210063670](https://i.imgur.com/JlqkmpI.png)

---

## 使用 scp

### 設定

透過 session manager 去達成 scp ，基本上透過 AWS 文件 [session-manager-getting-started-enable-ssh-connections](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-getting-started-enable-ssh-connections.html) 上的描述，可以得知是利用 Proxycommand 透過 AWS tunnel 直接連接到我們的 EC2 機器上。

編輯 `~/.ssh/config` 並加入

```bash
# SSH over Session Manager
host i-* mi-*
 ProxyCommand sh -c "aws ssm start-session --target %h --document-name AWS-StartSSHSession --parameters'portNumber=%p'"

# 就可以使用
scp -i -i /path/my-key-pair.pem test123 ubuntu@i-0b0d92751733d1234:~/test123
# 還是要利用一開始設定好的 key pair 去做連線。
```



### 進階設定

上面提供的方法雖然可以讓我們使用 scp & ssh，但是還是得設定 EC2 機器的 key

繞過去
- 網路上有人寫好了這個 proxy command 的 [script](https://gist.github.com/qoomon/fcf2c85194c55aee34b78ddcaa9e83a1)
- 使用的方式很簡單
  1. 下載並且把這個 script 放到 `~/.ssh/aws-ssm-ec2-proxy-command.sh`
  2. 修改 `aws-ssm-ec2-proxy-command.sh` 成為可以執行
  3. 修改 `~/.ssh/config` 裡面的指令
- 原理很簡單
  1. 利用 `aws ec2-instance-connect send-ssh-public-key` 去建立一個 short-lived 的 key
     - 這個指令詳細的好處可以看這篇 aws 文章 [new-using-amazon-ec2-instance-connect-for-ssh-access-to-your-ec2-instances](https://aws.amazon.com/blogs/compute/new-using-amazon-ec2-instance-connect-for-ssh-access-to-your-ec2-instances/)
  2. 接著再使用這把 key 透過原本的 start session 那條路連上遠端的 ec2 機器。


```bash
host i-* mi-*
 ProxyCommand ~/.ssh/aws-ssm-ec2-proxy-command.sh %h %r %p

# 就不用在帶一把 key 去做認證了
scp test123 ubuntu@i-0b0d92751733d1234:~/test123
```



---

## Port forwarding

透過 port forwarding 去連接 EC2 上面的服務，
- 很多時候我們會把服務都放進 private subnet 內， 而 developer 想要測試這些 services 時，往往要利用 VPN 或是開一台在內網的 EC2 去連結，
- 而使用 port forwarding 可以讓我們更容易地達成這個需求。

```bash
aws ssm start-session \
  --target i-0b0d92751733d1234 \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["80"],"localPortNumber":["9999"]}'
```


這樣就可以透過 `localhost:9999` 去連結到 EC2 上面 service 的 80 port 了
- 詳細的內容也可以看這篇 AWS 的文章 [new-port-forwarding-using-aws-system-manager-sessions-manager](https://aws.amazon.com/blogs/aws/new-port-forwarding-using-aws-system-manager-sessions-manager/)










---

## Monitoring


### Logging AWS Systems Manager API calls with AWS CloudTrail.















---


## Reference


---

- [https://www.youtube.com/watch?v=nzjTIjFLiow](https://www.youtube.com/watch?v=nzjTIjFLiow)
- [https://www.youtube.com/watch?v=kj9NgFfUIHQ](https://www.youtube.com/watch?v=kj9NgFfUIHQ)
- [https://aws.amazon.com/blogs/compute/new-using-amazon-ec2-instance-connect-for-ssh-access-to-your-ec2-instances/](https://aws.amazon.com/blogs/compute/new-using-amazon-ec2-instance-connect-for-ssh-access-to-your-ec2-instances/)
- [https://aws.amazon.com/blogs/aws/new-port-forwarding-using-aws-system-manager-sessions-manager/](https://aws.amazon.com/blogs/aws/new-port-forwarding-using-aws-system-manager-sessions-manager/)
- [https://globaldatanet.com/blog/ssh-and-scp-with-aws-ssm](https://globaldatanet.com/blog/ssh-and-scp-with-aws-ssm)
