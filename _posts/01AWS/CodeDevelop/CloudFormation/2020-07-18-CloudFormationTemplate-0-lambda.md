---
title: AWS - CodeDevelop - CloudFormation - Template `AWS::Lambda`
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, CodeDevelop]
tags: [AWS]
toc: true
image:
---

[toc]

- ref
  - [aws doc](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-lambda-function.html)

---

# Lambda


---

## AWS::Lambda::Function

<font color=red> AWS::Lambda::Function </font>

- To create a function, you need a deployment package and an execution role.
- The **deployment package** is a `.zip file archive` or `container image` that contains your function code.
- The **execution role** grants the function permission to use AWS services, such as Amazon CloudWatch Logs for log streaming and AWS X-Ray for request tracing.

- package type
  - Image
    - if the deployment package is a container image.
    - the code property must include the URI of a container image in the Amazon ECR registry.
    - <font color=blue> do not need to specify the handler and runtime properties </font>
  - Zip
    - if the deployment package is a .zip file archive.
    - the code property specifies the location of the .zip file.
    - <font color=blue> must specify the handler and runtime properties </font>
      - You can use **code signing** if your deployment package is a .zip file archive.
      - To enable code signing for this function, specify the ARN of a code-signing configuration.
      - When a user attempts to deploy a code package with `UpdateFunctionCode`, Lambda checks that the code package has a valid signature from a trusted publisher.
      - The code-signing configuration includes set set of signing profiles, which define the trusted publishers for this function.


```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: Lambda function with cfn-response.
Resources:
  primer:
    Type: AWS::Lambda::Function
    Properties:
      Description: Invoke a function during stack creation.
      FunctionName: String
      Code:
        ZipFile: |
          var aws = require('aws-sdk')
          var response = require('cfn-response')
          exports.handler = function(event, context) {
              console.log("REQUEST RECEIVED:\n" + JSON.stringify(event))
              // For Delete requests, immediately send a SUCCESS response.
              if (event.RequestType == "Delete") {
                  response.send(event, context, "SUCCESS")
                  return
              }
              var responseStatus = "FAILED"
              var responseData = {}
              var functionName = event.ResourceProperties.FunctionName
              var lambda = new aws.Lambda()
              lambda.invoke({ FunctionName: functionName }, function(err, invokeResult) {
                  if (err) {
                      responseData = {Error: "Invoke call failed"}
                      console.log(responseData.Error + ":\n", err)
                  }
                  else responseStatus = "SUCCESS"
                  response.send(event, context, responseStatus, responseData)
              })
          }
      CodeSigningConfigArn: String
      # Not currently supported by AWS CloudFormation.
      DeadLetterConfig:
      # A dead letter queue configuration that specifies the queue or topic where Lambda sends asynchronous events when they fail processing.
        DeadLetterConfig
      Environment:
      # Environment variables that are accessible from function code during execution.
        Variables:
          Key : Value
        Variables:
          databaseName: lambdadb
          databaseUser: admin
      FileSystemConfigs:
      # Connection settings for an Amazon EFS file system.
      # To connect a function to a file system, a mount target must be available in every Availability Zone that your function connects to. If your template contains an AWS::EFS::MountTarget resource, you must also specify a DependsOn attribute to ensure that the mount target is created or updated before the function.
        - FileSystemConfig
      Handler: index.handler
      # The name of the method within your code that Lambda calls to execute your function. The format includes the file name. It can also include namespaces and other qualifiers, depending on the runtime.
      ImageConfig:
      # Configuration values that override the container image Dockerfile settings.
        ImageConfig
      KmsKeyArn: String
      # The ARN of the AWS KMS key used to encrypt function's environment variables. If it's not provided, AWS Lambda uses a default service key.
      Layers:
      # A list of function layers to add to the function's execution environment. Specify each layer by its ARN, including the version.
        - String
      MemorySize: Integer
      PackageType: Image | Zip
      ReservedConcurrentExecutions: Integer
      # The number of simultaneous executions to reserve for the function.
      Role: arn:aws:iam::123456789012:role/lambda-role
      # The Amazon Resource Name (ARN) of the function's execution role.
      Runtime: nodejs12.x
      Tags:
      # A list of tags to apply to the function.
        - Tag
      Timeout: Integer
      # The amount of time that Lambda allows a function to run before stopping it. The default is 3 seconds. The maximum allowed value is 900 seconds.
      TracingConfig:
      # Set Mode to Active to sample and trace a subset of incoming requests with AWS X-Ray.
        Mode: Active
      VpcConfig:
      # For network connectivity to AWS resources in a VPC, specify a list of security groups and subnets in the VPC.
        VpcConfig
```


code


```yaml
Code:
  ImageUri: String
  S3Bucket: String
  S3Key: String
  S3ObjectVersion: String
  # For versioned objects, the version of the deployment package object to use.
  ZipFile: String

      Code:
        S3Bucket: my-bucket
        S3Key: function.zip

      Code:
        ZipFile: |
          var aws = require('aws-sdk')
          var response = require('cfn-response')
```























.
