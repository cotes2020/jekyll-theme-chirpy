---
title: AWS - IdenAccessManage - IAM User Login
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, IdenAccessManage]
tags: [AWS, IdenAccessManage]
toc: true
image:
---

- [IAM User Login](#iam-user-login)
  - [Console password:](#console-password)
  - [Security Token Service](#security-token-service)
  - [Server certificates:](#server-certificates)
  - [if your account compromised](#if-your-account-compromised)

---


# IAM User Login

---

The following diagram shows the different methods of authentication available with IAM:
￼
![AWS-IAM-Authentication-Methods-1024x450](https://i.imgur.com/yAw4RLW.jpg)

---

## Console password:
- sign in to interactive sessions such as the AWS Management Console.
- allow selected IAM users to change their passwords by disabling the option for all users and using an IAM policy to grant permissions for the selected users.
- The password policy
  - at least one numerical character from 0 to 9.
  - contain between 6 to 128 characters
- Users can be given access to change their own keys through IAM policy (not from the console).

---

## Security Token Service
- provides <font color=red> short-term authorization </font> that IAM roles utilize.

**Access Keys**:
- A combination of an <font color=red> access key ID </font> and a <font color=red> secret access key </font>
  - can assign two active access keys to a user at a time.
  - max limit of access keys an IAM User may possess at a time: 2
  - IAM users are allowed two sets of access keys.
- can be used to
  - <font color=red> make programmatic calls </font> to AWS when using the API in program code
  - or at a command prompt when using the AWS CLI or the AWS PowerShell tools.
- `aws configure`
  - the command needed to allow access key configuration.
- can create, modify, view or rotate access keys.
- Ensure access keys and secret access keys are stored securely.
  - When created IAM returns the access key ID and secret access key.
  - The secret access is returned only at creation time and if lost a new key must be created.

- disable a user’s access key will prevents it from being used for API calls.

- Access keys are updated immediately, once the associated IAM User's access is updated.

- Once the secret key has been lost, generating new access keys for the application is necessary.

- configuring access key entry to AWS account.
  - Region name
  - Output format
  - Installation of the AWS CLI Interface
  - Access key ID
  - Secret access key

---

## Server certificates:
- can use SSL/TLS certificates to authenticate with some AWS services.
- use the **AWS Certificate Manager (ACM)** to provision, manage and deploy server certificates.
  - Use IAM only when you must support HTTPS connections in a region that is not supported by ACM.


---


## if your account compromised


![Screen Shot 2020-06-12 at 18.14.58](https://i.imgur.com/wQuXgIF.png)














.
