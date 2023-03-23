---
title: AWS Lab - Readme
date: 2020-07-16 11:11:11 -0400
categories: [01AWS, AWSLab]
tags: [AWS, Lab]
math: true
image:
---

# Readme

- [Readme](#readme)
  - [AWS cloud architect project](#aws-cloud-architect-project)

```
    __∧_∧__    ~~~~~
　／(*´O｀)／＼
／|￣∪∪￣|＼／
　|＿＿ ＿|／
```

---

## AWS cloud architect project

1. Lambda interact with S3, auto make modify to the upload file...
2. Lambda interact with API gateway, calculate the data it get.
3. with ECS and Docker Hub to create a static web site.
4. configure VPC gateway to create endpoint, grant access to S3 or other cloud service from a private EC2 instance that do not have public internet.
5. configure Egress-only internet gateway for IPv6 EC2 instance.
6. Configure EFS and mount it to multiple EC2 instance in multiple AZ.
7. create a VPC peer connection
8. create database with word-press
    - RDS with multi-AZ and read-replica
    - Aurora and Aurora serverless
    - DynamoDB, create the table and query/scan the item.
9. use CLB to balanace the traffic to 3 different web instance, make the web instance only be connected through CLB
