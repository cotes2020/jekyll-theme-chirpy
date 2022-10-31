---
title: AWS - CodeDevelop - CloudFormation Template - Nested Stack
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, CodeDevelop]
tags: [AWS]
toc: true
image:
---

[toc]

---


# Template - Nested Stack


```json
// root.json
{
    "AWSTemplateFormatVersion" : "2010-09-09",
    "Resources" : {
        "myStack" : {
	       "Type" : "AWS::CloudFormation::Stack",
	       "Properties" : {
              "TemplateURL" : "https://s3.amazonaws.com/stacker730/noretain.json",
              "TimeoutInMinutes" : "60"
	       }
        }
    }
}
```


---


# Template - multinest.json

```json
// multinest.json
{
    "AWSTemplateFormatVersion" : "2010-09-09",
    "Resources" : {
        "myStack" : {
	       "Type" : "AWS::CloudFormation::Stack",
	       "Properties" : {
              "TemplateURL" : "https://s3.amazonaws.com/stacker730/s3static.json",
              "TimeoutInMinutes" : "60"
	       }
        },
        "myStack2" : {
            "Type" : "AWS::CloudFormation::Stack",
            "Properties" : {
               "TemplateURL" : "https://s3.amazonaws.com/stacker730/noretain.json",
               "TimeoutInMinutes" : "60"
            }
         }
    }
}
```









.
