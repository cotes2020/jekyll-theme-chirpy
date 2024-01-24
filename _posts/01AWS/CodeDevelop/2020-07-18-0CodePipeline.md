---
title: AWS - CodeDevelop - CodePipeline
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, CodeDevelop]
tags: [AWS]
toc: true
image:
---

- [CodePipeline](#codepipeline)
  - [basic](#basic)
  - [Components of AWS Data Pipeline](#components-of-aws-data-pipeline)
  - [Accessing AWS Data Pipeline](#accessing-aws-data-pipeline)
  - [Related Services](#related-services)
  - [Pipeline Components, Instances, and Attempts](#pipeline-components-instances-and-attempts)
  - [Pipeline Definition](#pipeline-definition)
  - [Pipeline Components](#pipeline-components)
    - [Data Nodes](#data-nodes)
  - [setup](#setup)
    - [CodePipeline: CodeCommit - ManualAppove - CloudFormation](#codepipeline-codecommit---manualappove---cloudformation)
    - [CodePipeline: S3 - CodeDeploy - CloudFormation](#codepipeline-s3---codedeploy---cloudformation)

---

# CodePipeline

---

## basic

- a web service
- automate the movement and transformation of data.
- define data-driven workflows
  - tasks can be dependent on the successful completion of previous tasks.
- define the parameters of your data transformations and AWS Data Pipeline enforces the logic that you've set up.


- fully managed
- <font color=red> Continuous integration / continuous delivery service. </font>
  - Orchestrates <font color=blue> Build, Test & Deployment (the end-to-end software release process) </font> based on the workflow pre-defined.
  - A pipeline is a workflow construct that describes how code changes go through a release process.

- <font color=red> Automated release process required to release the code </font>
  - fast, consistent, fewer mistakes
  - The pipeline is triggered every time there is a changeto your code
  - enables quick release of new features and bug fixes.

- <font color=red> CodePipeline integrates with </font>
  - CodeCommit, CodeBuild, CodeDeploy,
  - Github, Jenkins, Elasetic Beanstalk,
  - CloudFormation, Lambda
  - Elastic container Service


![CodePipeline](https://i.imgur.com/IOZKewF.png)


CodePipeline: <font color=blue> Defined Workflow </font>
- the workflow begins when there is a change detected in the source code.

CodeCommit: <font color=blue> New code appears </font>
- New source code appears in the CodeCommit repository

CodeBuild: <font color=blue> Code is built&tested </font>
- CodeBuild immediately compiles source code, runs tests, and produces packages.

CodeDeploy: <font color=blue> Application Deployed </font>
- The newly built application is deployed into a staging or porduction environment.





---

## Components of AWS Data Pipeline

Components of AWS Data Pipeline work together to manage the data:

1. A <font color=red> pipeline definition </font> specifies the business logic of your data management.

2. A <font color=red> pipeline schedules </font> and runs tasks by creating Amazon EC2 instances to perform the defined work activities.
   - upload your pipeline definition to the pipeline,
   - and then activate the pipeline.
   - can edit the pipeline definition for a running pipeline and activate the pipeline again for it to take effect.
   - can deactivate the pipeline, modify a data source, and then activate the pipeline again.
   - When finished with your pipeline, can delete it.

3. <font color=red> Task Runner </font> polls for tasks and then performs those tasks.
   - For example:
   - Task Runner could copy log files to Amazon S3 and launch Amazon EMR clusters.
   - Task Runner is installed and runs automatically on resources created by the pipeline definitions.
   - write a custom task runner application, or you can use the Task Runner application that is provided by AWS Data Pipeline.


For example
- use AWS Data Pipeline to archive your web server's logs to Amazon Simple Storage Service (Amazon S3) each day and then run a weekly Amazon EMR (Amazon EMR) cluster over those logs to generate traffic reports.
- AWS Data Pipeline schedules the daily tasks to copy data and the weekly task to launch the Amazon EMR cluster.
- AWS Data Pipeline ensures that Amazon EMR waits for the final day's data to be uploaded to Amazon S3 before it begins its analysis, even if there is an unforeseen delay in uploading the logs.



## Accessing AWS Data Pipeline

You can create, access, and manage your pipelines using any of the following interfaces:

1. AWS Management Console:
   - web interface to access AWS Data Pipeline.

2. AWS Command Line Interface (AWS CLI):
   - Provides commands for a broad set of AWS services, including AWS Data Pipeline
   - supported on Windows, macOS, and Linux.

3. AWS SDKs
   - Provides language-specific APIs and takes care of many of the connection details, such as calculating signatures, handling request retries, and error handling.

4. Query API
   - Provides low-level APIs that you call using HTTPS requests.
   - Using the Query API is the most direct way to access AWS Data Pipeline, but it requires that your application handle low-level details such as generating the hash to sign the request, and error handling.



---

## Related Services

AWS Data Pipeline works with the following services <font color=red> to store data </font>

- <font color=blue> Amazon DynamoDB </font>
  - fully managed NoSQL database with fast performance at a low cost.

- <font color=blue> Amazon RDS </font>
  - fully managed relational database that scales to large datasets.

- <font color=blue> Amazon Redshift </font>
  - fast, fully managed, petabyte-scale data warehouse that makes it easy and cost-effective to analyze a vast amount of data.

- <font color=blue> Amazon S3 </font>
  - secure, durable, and highly scalable object storage.

AWS Data Pipeline works with the following compute services to <font color=red> transform data </font>

- <font color=red> Amazon EC2 </font>
  - resizable computing capacity to build and host your software systems.

- <font color=red> Amazon EMR </font>
  - easy, fast, and cost-effective to distribute and process vast amounts of data across Amazon EC2 servers,
  - using a framework such as Apache Hadoop or Apache Spark.


---

## Pipeline Components, Instances, and Attempts

3 types of items associated with a scheduled pipeline:

1. Pipeline Components
   - define the rules of data management.
   - represent the business logic of the pipeline
   - represented by the different sections of a pipeline definition.
   - specify the data sources, activities, schedule, and preconditions of the workflow.
   - can inherit properties from parent components. Relationships among components are defined by reference.

2. Instances
   - When AWS Data Pipeline runs a pipeline, it compiles the pipeline components to create a set of actionable instances.
   - Each instance contains all the information for performing a specific task.
   - The complete set of instances is the to-do list of the pipeline.
   - AWS Data Pipeline hands the instances out to task runners to process.

3. Attempt objects
   - To provide robust data management, AWS Data Pipeline retries a failed operation.
   - It continues to do so until the task reaches the maximum number of allowed retry attempts.
   - Attempt objects track the various attempts, results, and failure reasons if applicable.
   - Essentially, it is the instance with a counter.
   - AWS Data Pipeline performs retries using the same resources from the previous attempts, such as Amazon EMR clusters and EC2 instances.


Retrying failed tasks
- an important part of a fault tolerance strategy, and AWS Data Pipeline definitions provide conditions and thresholds to control retries.
- but too many retries can delay detection of an unrecoverable failure
  - because AWS Data Pipeline does not report failure until it has exhausted all the retries that you specify.
  - The extra retries may accrue additional charges if they are running on AWS resources.
  - As a result, carefully consider when it is appropriate to exceed the AWS Data Pipeline default settings that you use to control re-tries and related settings.

![dp-object-types](https://i.imgur.com/CXbguCV.png)


---

## Pipeline Definition

**pipeline definition**: how business logic communicate to AWS Data Pipeline.
- It contains the following information:
  - Names, locations, and formats of the data sources
  - Activities that transform the data
  - The schedule for those activities
  - Resources that run the activities and preconditions
  - Preconditions that must be satisfied before the activities can be scheduled
  - Ways to alert with status updates as pipeline execution proceeds


From your pipeline definition, AWS Data Pipeline determines the tasks, schedules them, and assigns them to task runners.
- If a task is not completed successfully, AWS Data Pipeline retries the task according to your instructions and, if necessary, reassigns it to another task runner.
- If the task fails repeatedly, you can configure the pipeline to notify you.

> For example
> in pipeline definition: specify that log files generated by your application are archived each month in 2013 to an Amazon S3 bucket. AWS Data Pipeline would then create 12 tasks, each copying over a month's worth of data, regardless of whether the month contained 30, 31, 28, or 29 days.


<font color=red> create a pipeline definition </font>
- Graphically, by using the AWS Data Pipeline console
- Textually, JSON file used by the command line interface
- Programmatically, call the web service with either one of the AWS SDKs or the AWS Data Pipeline API

---

## Pipeline Components

### Data Nodes
- the location and type of data that a pipeline activity uses as input data for a task or output data is to be stored.
- AWS Data Pipeline supports the following types of data nodes:
  - DynamoDBDataNode
    - A DynamoDB table that contains data for HiveActivity or EmrActivity to use.
  - SqlDataNode
    - An SQL table and database query that represent data for a pipeline activity to use.
    - Previously, MySqlDataNode was used. Use SqlDataNode instead.

  - RedshiftDataNode
    - An Amazon Redshift table that contains data for RedshiftCopyActivity to use.

  - S3DataNode
    - An Amazon S3 location that contains one or more files for a pipeline activity to use.



Activities
A definition of work to perform on a schedule using a computational resource and typically input and output data nodes.

Preconditions
A conditional statement that must be true before an action can run.

Scheduling Pipelines
Defines the timing of a scheduled event, such as when an activity runs.

Resources
The computational resource that performs the work that a pipeline defines.

Actions
An action that is triggered when specified conditions are met, such as the failure of an activity.

For more information, see Pipeline Definition File Syntax.


---

## setup

---

### CodePipeline: CodeCommit - ManualAppove - CloudFormation

![Screen Shot 2021-01-18 at 19.41.32](https://i.imgur.com/DDhcJmb.png)


1. Create an AWS IAM Role
   - the service that will use this role: `CloudFormation`
   - role name: <font color=blue> pipeRoleFullAdminAcess </font>
   - Click Next: Permissions:
     - + `AdministratorAccess` permissions policy.
     - Allows CloudFormation to create and manage AWS stacks and resources on your behalf.

2. Create an AWS CodeCommit Repository and SNS Topic
   - Create an AWS CodeCommit Repository
     - Repository name: <font color=blue> pipeTestsRepo </font>
   - Navigate to Simple Notification Service (SNS).
     - Enter "manualapprove" as the topic name.
     - Click Next step > Create topic.
     - Create subscription: Email as the protocol.
     - Enter your email address as the endpoint.
     - Click Create subscription.
     - Navigate to your inbox and Confirm subscription link.

3. Create an AWS CodePipeline Pipeline
   - CodePipeline.
   - Create pipeline.
     - pipeline name: "ManualApprove4CF"
       - Ensure `New service role` is selected.
       - Ensure `Allow AWS CodePipeline to create service role so it can be used with this new pipeline` is checked.
     - Advanced settings section
       - ensure the `Default location` and `Default AWS Managed Key options` are selected.
     - Add <font color=red> source stage </font> page:
       - Source provider: <font color=blue> AWS CodeCommit </font>
       - Repository name: <font color=blue> pipeTestsRepo </font>
       - Branch name: master
       - Change detection options: Amazon CloudWatch Events (recommended)
     - Skip build stage
     - Add <font color=red> deploy stage </font> page:
       - Deploy provider: <font color=blue> AWS CloudFormation </font>
       - Region: US East - (N. Virginia)
       - Action mode: Create or update a stack
       - Stack name: deploywithmanualapprove
       - Artifact name: SourceArtifact
       - File name: S3Retain.yaml
       - Role name: <font color=blue> pipeRoleFullAdminAcess </font>
       - Click Next > Create pipeline.
     - Click the AWS CloudFormation link in the Deploy panel.
     - Once CloudFormation shows complete, return to the CodePipeline service and verify the manualapprove pipeline status shows Succeeded in the Deploy panel.
   - Add stage between the Source and Deploy panels.
     - stage name: <font color=blue> manualapprove </font>
     - Add action group.
       - action name: `manualapproval`
       - action provider: `Manual approval`
       - Select the `SNS topic ARN` created earlier in the lab.
     - Click Done > Save > Save.
   - Click Release change to restart the pipeline.
   - Navigate to email, open the APPROVAL NEEDED... message.
   - Navigate back to Code Pipeline.
   - Click Review in the Manual approve panel.
   - Enter "Looks good — approved." in the comments, and click Approve.



---

### CodePipeline: S3 - CodeDeploy - CloudFormation


```
cloudFormation.json

myapp1.zip
- appspec.yml
- index.html
- scripts/
  - scripts.sh

myapp2.zip
myapp3.zip

1. create s3 bucket
2. upload the cloudFormation file into a S3 bucket
3. setup the user
4. run the cloudformation to create the ec2
5. setup CodeDeploy and deploy
6. upload the next version of code (manually triggered)
7. setup the CodePipeline and triggered
8. upload the next version of code (auto triggered)
```


1. create s3 bucket
   - for the application:
     - same region of the source
     - keep all the version
     - upload the reversion file: myapp1.zip
   - for cloudFormation:
     - upload the cloudFormation file: cloudFormation.json

2. upload the cloudFormation file into a S3 bucket

3. setup the user
   - aws configure the key pair
      ```bash
      aws iam get-user
      aws configure list
      aws configure
      # input access key id and secrect access key
      ```
   - setup user permissions:
     - attach needed policy
       - S3FullAccess, CodeDeployFullAccess,
       - YouNewPolicy
          ```json
          {
            "Version":"2012-10-17",
            "Statement":[
              {
                "Sid":"VisualEditor0",
                "Effect":"Allow",
                "Action":"cloudformation:*",
                "Resource":"*"
              },
              {
                "Effect":"Allow",
                "Action":"iam:*",
                "Resource":"*"
              },
              {
                "Effect":"Allow",
                "Action":"ec2:*",
                "Resource":"*"
              }
            ]
          }
          ```

4. run the cloudformation to create the ec2
    ```bash
    # cloudFormation.json
    # create the ec2 instance
    aws cloudformation create-stack --stack-name CodeDeployDemoStack \
    --template-url https://s3-bucket-url/cloudFormation.json \
    --parameters \
    ParameterKey=InstanceCount, ParameterValue=1 \
    ParameterKey=InstanceType, ParameterValue=t2.micro \
    ParameterKey=KeyPairNamexxxx, ParameterValue=irkpyyyyyyyy \
    ParameterKey=OperatingSystem, ParameterValue=Linux \
    ParameterKey=SSHLocation, ParameterValue=0.0.0.0/0 \
    ParameterKey=TagKey, ParameterValue=Name \
    ParameterKey=TagValue, ParameterValue=CodeDepoloyDemo \
    --capabilities CAPABILITY_IAM

    # verify the cloudfromation stack has completed using
    aws cloudformation describe-stacks --stack-name CodeDeployDemoStack --query "Stack[0].StackStatus" --output test

    # login to instance to check the codedeploy agent
    sudo service codedeploy-agent status
    ```

5. <font color=red> CodeDeploy </font> : setup and deploy
   - <font color=blue> create application </font>
     - application name
     - compute plantform
   - <font color=blue> create deployment group </font>
     - deployment group name
     - service role
     - deployment type
     - environment configuration
       - EC2 instance: Key&Value of created EC2
     - deployment setting
     - Load balancer
   - <font color=blue> create deployment <- create the application </font>
     - select deployment group
     - revision type(S3/Github) and revision location.
   - the application is installed and run

6. upload the next version of code (manually triggered)

7. <font color=red> Setup the CodePipeline and triggered</font>
   - pipeline setting:
     - pipeline name
     - service role (create a new service role)
     - role name
   - source stage
     - source provider
       - (S3/Github): bucket name
     - detection option
       - Cloudwatch: pipeline
   - build stage
     - CodeBuild or Jenkins
   - deploy stage
     - deploy provider
     - region
     - application name
     - deployment group
   - start building the application

8. upload the next version of code (auto triggered)












.
