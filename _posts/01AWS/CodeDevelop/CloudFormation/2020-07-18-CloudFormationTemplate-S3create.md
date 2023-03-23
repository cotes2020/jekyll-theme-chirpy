---
title: AWS - CodeDevelop - CloudFormation Template - S3 Create
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, CodeDevelop]
tags: [AWS]
toc: true
image:
---

[toc]

---


# Template - setup S3_Website_Bucket_With_No_Retain_On_Delete

S3_Website_Bucket_With_No_Retain_On_Delete:
- create a publicly accessible S3 bucket configured for website access
- with no deletion policy


**WARNING** This template creates an S3 bucket that will be deleted when the stack is deleted.


```json
{
    "AWSTemplateFormatVersion": "2010-09-09",
    "Resources": {
        "S3Bucket": {
            "Type": "AWS::S3::Bucket",
            "Properties": {
                "AccessControl": "PublicRead",
                "WebsiteConfiguration": {
                    "IndexDocument": "index.html",
                    "ErrorDocument": "error.html"
                }
            }
        },
        "BucketPolicy": {
            "Type": "AWS::S3::BucketPolicy",
            "Properties": {
                "PolicyDocument": {
                    "Id": "MyPolicy",
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Sid": "PublicReadForGetBucketObjects",
                            "Effect": "Allow",
                            "Principal": "*",
                            "Action": "s3:GetObject",
                            "Resource": { "Fn::Join": [ "", [  "arn:aws:s3:::", { "Ref": "S3Bucket" }, "/*" ] ] }
                        }
                    ]
                },
                "Bucket": { "Ref": "S3Bucket" }
            }
        }
    },
    "Outputs": {
        "WebsiteURL": {
            "Value": { "Fn::GetAtt": [ "S3Bucket", "WebsiteURL" ] },
            "Description": "URL for website hosted on S3"
        },
        "S3BucketSecureURL": {
            "Value": { "Fn::Join": [ "", [ "https://", { "Fn::GetAtt": [ "S3Bucket", "DomainName" ] } ] ] },
            "Description": "Name of S3 bucket to hold website content"
        }
    }
}
```






---

# Template - setup S3_Website_Bucket_With_Retain_On_Delete

S3_Website_Bucket_With_Retain_On_Delete:
- create a publicly accessible S3 bucket configured for website access
- with a deletion policy of retail on delete.

**WARNING** This template creates an S3 bucket that will NOT be deleted when the stack is deleted.


```yaml
AWSTemplateFormatVersion: 2010-09-09
Description: >-
  AWS CloudFormation Sample Template S3_Website_Bucket_With_Retain_On_Delete:
  Sample template showing how to create a publicly accessible S3 bucket
  configured for website access with a deletion policy of retail on delete.
  **WARNING** This template creates an S3 bucket that will NOT be deleted when
  the stack is deleted. You will be billed for the AWS resources used if you
  create a stack from this template.
Resources:
  S3Bucket:
    Type: 'AWS::S3::Bucket'
    Properties:
      AccessControl: PublicRead
      WebsiteConfiguration:
        IndexDocument: index.html
        ErrorDocument: error.html
    DeletionPolicy: Retain
Outputs:
  WebsiteURL:
    Value: !GetAtt
      - S3Bucket
      - WebsiteURL
    Description: URL for website hosted on S3
  S3BucketSecureURL:
    Value: !Join
      - ''
      - - 'https://'
        - !GetAtt
          - S3Bucket
          - DomainName
    Description: Name of S3 bucket to hold website content
```



```json
{
    "AWSTemplateFormatVersion": "2010-09-09",
    "Resources": {
        "S3Bucket": {
            "Type": "AWS::S3::Bucket",
            "Properties": {
                "AccessControl": "PublicRead",
                "WebsiteConfiguration": {
                    "IndexDocument": "index.html",
                    "ErrorDocument": "error.html"
                }
            },
            "DeletionPolicy": "Retain"
        },
        "BucketPolicy": {
            "Type": "AWS::S3::BucketPolicy",
            "Properties": {
                "PolicyDocument": {
                    "Id": "MyPolicy",
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Sid": "PublicReadForGetBucketObjects",
                            "Effect": "Allow",
                            "Principal": "*",
                            "Action": "s3:GetObject",
                            "Resource": {
                                "Fn::Join": [
                                    "",
                                    [
                                        "arn:aws:s3:::",
                                        {
                                            "Ref": "S3Bucket"
                                        },
                                        "/*"
                                    ]
                                ]
                            }
                        }
                    ]
                },
                "Bucket": { "Ref": "S3Bucket"}
            }
        }
    },
    "Outputs": {
        "WebsiteURL": {
            "Value": { "Fn::GetAtt": [ "S3Bucket", "WebsiteURL" ] },
            "Description": "URL for website hosted on S3"
        },
        "S3BucketSecureURL": {
            "Value": { "Fn::Join": [ "",[ "https://", { "Fn::GetAtt": [ "S3Bucket", "DomainName" ] } ] ] },
            "Description": "Name of S3 bucket to hold website content"
        }
    }
}
```
