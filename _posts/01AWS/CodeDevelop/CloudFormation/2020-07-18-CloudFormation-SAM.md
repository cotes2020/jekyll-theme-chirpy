---
title: AWS - CodeDevelop - CloudFormation - SAM
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, CodeDevelop]
tags: [AWS]
toc: true
image:
---

[toc]

---

# AWS Serverless Application Model (AWS SAM)

![Screen Shot 2021-01-04 at 22.17.51](https://i.imgur.com/idVQYiw.png)

- an open-source framework
- to build serverless applications on AWS.

A serverless application
- a combination of Lambda functions, event sources, and other resources (such as APIs, databases) that work together to perform tasks.


use AWS SAM to define the serverless applications.

AWS SAM consists of the following components:
1. AWS SAM template specification.
   - use this specification to define the serverless application.
   - simple and clean syntax to describe the functions, APIs, permissions, configurations, and events that make up a serverless application.
   - use an AWS SAM template file to operate on a single, deployable, versioned entity that's your serverless application.  


2. AWS SAM command line interface (AWS SAM CLI).
   - use this tool to build serverless applications that are defined by AWS SAM templates.
   - The CLI provides commands enable you to
     - verify that AWS SAM template files are written according to the specification,
     - invoke Lambda functions locally,
     - step-through debug Lambda functions,
     - package and deploy serverless applications to the AWS Cloud, and so on.  


---

## Benefits of using AWS SAM

SAM integrates with other AWS services, creating serverless applications with AWS SAM provides the following benefits:

1. Single-deployment configuration.
   - easy to organize related components and resources, and operate on a single stack.
   - can use AWS SAM to share configuration (such as memory and timeouts) between resources, and deploy all related resources together as a single, versioned entity.



2. Extension of AWS CloudFormation.
   - as an extension of AWS CloudFormation, get the reliable deployment capabilities of AWS CloudFormation.
   - can define resources by using AWS CloudFormation in your AWS SAM template.
   - can use the full suite of resources, intrinsic functions, and other template features that are available in AWS CloudFormation.



3. Built-in best practices.
   - can use AWS SAM to define and deploy your infrastructure as config.
   - possible to use and enforce best practices such as code reviews.
   - Also, with a few lines of configuration, enable safe deployments through CodeDeploy, and can enable tracing by using AWS X-Ray.


4. Local debugging and testing.
   - The AWS SAM CLI lets you locally build, test, and debug serverless applications that are defined by AWS SAM templates.
   - The CLI provides a Lambda-like execution environment locally.
   - It helps you catch issues upfront by providing parity with the actual Lambda execution environment.
   - To step through and debug your code to understand what the code is doing, you can use AWS SAM with AWS toolkits like the AWS Toolkit for JetBrains, AWS Toolkit for PyCharm, AWS Toolkit for IntelliJ, and AWS Toolkit for Visual Studio Code.
   - This tightens the feedback loop by making it possible for you to find and troubleshoot issues that you might run into in the cloud.


5. Deep integration with development tools.
   - can use AWS SAM with a suite of AWS tools for building serverless applications.
   - can discover new applications in the AWS Serverless Application Repository.
   - For authoring, testing, and debugging AWS SAM–based serverless applications, you can use the AWS Cloud9 IDE.
   - To build a deployment pipeline for your serverless applications, you can use CodeBuild, CodeDeploy, and CodePipeline.
   - can also use AWS CodeStar to get started with a project structure, code repository, and a CI/CD pipeline that's automatically configured for you.
   - To deploy your serverless application, you can use the Jenkins plugin.
   - You can use the Stackery.io toolkit to build production-ready applications.


---


## example

### Deploying a Hello World application

This application implements a basic API backend. It consists of an Amazon API Gateway endpoint and an AWS Lambda function.
- When you send a GET request to the API Gateway endpoint, the Lambda function is invoked.
- This function returns a hello world message.


![sam-getting-started-hello-world](https://i.imgur.com/s2CrYTD.png)

```bash
# Prerequisites
# Creating an AWS account.
# Configuring AWS Identity and Access Management (IAM) permissions.

# Installing Docker. prerequisite only for testing your application locally.

# Installing Homebrew.  
# Installing the AWS SAM command line interface (CLI).
# Check the version
sam --version command.

# If you select the Image package type, having an Amazon Elastic Container Registry (Amazon ECR) repository URI to perform a deployment.


# ------------------ Step 1 - Download a sample application
sam init
# This command creates a directory with the name that you provided as the project name. The contents of the project directory are similar to the following:
 sam-app/
   ├── README.md
   ├── events/
   │   └── event.json
   ├── hello_world/
   │   ├── __init__.py
   │   ├── app.py            #Contains your AWS Lambda handler logic.
   │   └── requirements.txt  #Contains any Python dependencies the application requires, used for sam build
   ├── template.yaml         #Contains the AWS SAM template defining your application's AWS resources.
   └── tests/
       └── unit/
           ├── __init__.py
           └── test_handler.p


# ------------------ Step 2 - Build your application
cd sam-app
sam build
# sam build command builds any dependencies that your application has,
# and copies your application source code to folders under .aws-sam/build to be zipped and uploaded to Lambda.
# You can see the following top-level tree under .aws-sam:
 .aws_sam/
   └── build/
       ├── HelloWorldFunction/ # directory contains app.py file, third-party dependencies that app uses.
       └── template.yaml


# ------------------ Step 3 - Deploy your application
sam deploy --guided
# This command deploys your application to the AWS Cloud.
# It takes the deployment artifacts that you build with the sam build command, packages and uploads them to an Amazon Simple Storage Service (Amazon S3) bucket that the AWS SAM CLI creates, and deploys the application using AWS CloudFormation.
# In the output of the sam deploy command, you can see the changes being made to your AWS CloudFormation stack.


# ------------------ Step 4 - test your application
# If your application created an HTTP endpoint, the outputs that sam deploy generates also show you the endpoint URL for your test application. You can use curl to send a request to your application using that endpoint URL. For example:
curl https://<restapiid>.execute-api.us-east-1.amazonaws.com/Prod/hello/
#  {"message": "hello world"}


# ------------------ Step 4: (Optional) Test your application locally
# When you're developing your application, you might find it useful to test locally. The AWS SAM CLI provides the sam local command to run your application using Docker containers that simulate the execution environment of Lambda. There are two options to do this:
# Host your API locally
sam local start-api
curl http://127.0.0.1:3000/hello

# Invoke your Lambda function directly
sam local invoke "HelloWorldFunction" -e events/event.json



```



### Process Amazon S3 events


```bash
# Step 1: Initialize the application
# download the sample application
# consists of an AWS SAM template and application code.

sam init \
--location https://github.com/aws-samples/cookiecutter-aws-sam-s3-rekognition-dynamodb-python \
--no-input

# Review the contents of the directory that the command created (aws_sam_ocr/):
# template.yaml – Defines three AWS resources that the Amazon S3 application needs: a Lambda function, an Amazon S3 bucket, and a DynamoDB table. The template also defines the mappings and permissions between these resources.
# src/ directory – Contains the Amazon S3 application code.
# SampleEvent.json – The sample event source, which is used for local testing.



# Step 2: Package the application
# create a Lambda deployment package, which you use to deploy the application to the AWS Cloud.
# This deployment creates the necessary AWS resources and permissions that are required to test the application locally.

# Create an S3 bucket where to save the packaged code.
aws s3 mb s3://bucketname

# Create the deployment package
sam package \
    --template-file template.yaml \
    --output-template-file packaged.yaml \  # specify the new template file
    --s3-bucket bucketname



# Step 3: Deploy the application
# test the application by invoking it in the AWS Cloud.

# To deploy the serverless application to the AWS Cloud
sam deploy \
    --template-file packaged.yaml \
    --stack-name aws-sam-ocr \
    --capabilities CAPABILITY_IAM \   # allows AWS CloudFormation to create an IAM role.
    --region us-east-1



# Step 4: test the serverless application in the AWS Cloud
# Upload an image to the Amazon S3 bucket that you created for this sample application.
# Open the DynamoDB console and find the table that was created. See the table for results returned by Amazon Rekognition.
# Verify that the DynamoDB table contains new records that contain text that Amazon Rekognition found in the uploaded image.



# Step 4: Test the application locally
# retrieve the names of the AWS resources that were created by AWS CloudFormation.

# Retrieve the Amazon S3 key name and bucket name from AWS CloudFormation.
# Modify the SampleEvent.json file by replacing the values for the object key, bucket name, and bucket ARN.

# Retrieve the DynamoDB table name. This name is used for the following sam local invoke command.

# generate a sample Amazon S3 event and invoke the Lambda function:
TABLE_NAME="Table name obtained from AWS CloudFormation console" sam local invoke --event SampleEvent.json

# The TABLE_NAME= portion sets the DynamoDB table name.
# The --event parameter specifies the file that contains the test event message to pass to the Lambda function.
#  now verify that the expected DynamoDB records were created, based on the results returned by Amazon Rekognition.
```


















.
