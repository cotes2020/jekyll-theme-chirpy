---
title: AWS - CodeDevelop - CloudFormation - Template `AWS::Config`
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, CodeDevelop]
tags: [AWS]
toc: true
image:
---

[toc]

- ref
  - [aws doc](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-config-configrule.html)
  - [Deploy Managed Config Rules using CloudFormation and CodePipeline](https://stelligent.com/2019/10/31/deploy-managed-config-rules-using-cloudformation-and-codepipeline/)

---

# config


---

## step1

add a new custom AWS Config rule
- must first create the AWS Lambda function that the rule invokes to evaluate your resources.
- When use the PutConfigRule action to add the rule to AWS Config, must specify the Amazon Resource Name (ARN) that AWS Lambda assigns to the function.
- Specify the ARN for the SourceIdentifier key. This key is part of the Source object, which is part of the ConfigRule object.


---


## AWS::Config::ConfigRole

```yaml
Type: AWS::IAM::Role
Properties:
  AssumeRolePolicyDocument:
    Version: '2012-10-17'
    Statement:
    - Sid: RoleforConfig
      Effect: Allow
      Action: 'sts:AssumeRole'
      Principal:
        Service: 'config.amazonaws.com'
  ManagedPolicyArns: 'arn:aws:iam::aws:policy/service-role/AWSConfigRole'
```


---

## step2

To enable AWS Config, must create <font color=red> a configuration recorder and a delivery channel </font>.
- If create the resources separately, must create a configuration recorder before create a delivery channel.
- AWS Config uses the **configuration recorder** to capture configuration changes to your resources.
- AWS Config uses the **delivery channel** to deliver the configuration changes to S3 bucket or Amazon SNS topic.

---

## AWS::Config::ConfigurationRecorder

<font color=red> The configuration recorder </font>

- describes the AWS resource types for which AWS Config records configuration changes.
- stores the configurations of the supported resources in your account as configuration items.
- AWS CloudFormation starts the recorder as soon as the delivery channel is available.
- To stop the recorder, delete the configuration recorder from your stack.

```yaml
Type: AWS::Config::ConfigurationRecorder
Properties:
  Name: String
  # Specifies the types of AWS resource for which AWS Config records configuration changes.
  # whether to record configurations for all supported resources or for a list of resource types.
  # The resource types that you list must be supported by AWS Config.
  RecordingGroup:
      AllSupported: Boolean
      # whether AWS Config records configuration changes for every supported type of regional resource.
      # set to true: when AWS Config adds support for a new type of regional resource, it starts recording resources of that type automatically; you cannot enumerate a list of resourceTypes.
      IncludeGlobalResourceTypes: Boolean
      # whether AWS Config includes all supported types of global resources (for example, IAM resources) with the resources that it records.
      # Before can set to true, must set the AllSupported option to true.
      # If set to true, when AWS Config adds support for a new type of global resource, it starts recording resources of that type automatically.
      # The configuration details for any global resource are the same in all regions. To prevent duplicate configuration items, you should consider customizing AWS Config in only one region to record global resources.
      ResourceTypes:
        - "AWS::EC2::Volume"
      # A comma-separated list that specifies the types of AWS resources for which AWS Config records configuration changes
  # The Amazon Resource Name (ARN) of the IAM role that is used to make read or write requests to the delivery channel and to get configuration details for supported AWS resources.
  RoleARN:
      Fn::GetAtt:
        - ConfigRole
        - Arn
```

[AWS Config supports the following AWS resources types and resource relationships.](https://docs.aws.amazon.com/config/latest/developerguide/resource-config-reference.html#supported-resources)


---


## AWS::Config::DeliveryChannel


Specifies a delivery channel object to <font color=blue> deliver configuration information to an S3 bucket and SNS topic </font>.
- Before you can create a delivery channel, must create a configuration recorder.
- use this action to change the Amazon S3 bucket or an Amazon SNS topic of the existing delivery channel.
- To change the Amazon S3 bucket or an Amazon SNS topic, call this action and specify the changed values for the S3 bucket and the SNS topic.
- If you specify a different value for either the S3 bucket or the SNS topic, this action will keep the existing value for the parameter that is not changed.
- can have only one delivery channel per region in your account.
- When create the delivery channel, you can specify;
  - how often AWS Config delivers configuration snapshots to your Amazon S3 bucket (for example, 24 hours),
  - the S3 bucket to which AWS Config sends configuration snapshots and configuration history files,
  - and the Amazon SNS topic to which AWS Config sends notifications about configuration changes, such as updated resources, AWS Config rule evaluations,
  - and when AWS Config delivers the configuration snapshot to your S3 bucket.

```yaml
DeliveryChannel:
  Type: AWS::Config::DeliveryChannel
  Properties:
    # how often AWS Config delivers configuration snapshots to the S3 bucket.
    ConfigSnapshotDeliveryProperties:
      DeliveryFrequency: "One_Hour | Six_Hours | Three_Hours | Twelve_Hours | TwentyFour_Hours"
    Name: String
    # The name of the S3 bucket to which AWS Config delivers configuration snapshots and configuration history files.
    # If specify a bucket that belongs to another AWS account, that bucket must have policies that grant access permissions to AWS Config
    S3BucketName:
      Ref: ConfigBucket
    S3KeyPrefix: String
    # The Amazon Resource Name (ARN) of the Amazon SNS topic to which AWS Config sends notifications about configuration changes.
    # If choose a topic from another account, the topic must have policies that grant access permissions to AWS Config
    SnsTopicARN:
      Ref: ConfigTopic
```



---

## step3

create rule

---

## AWS::Config::ConfigRule


Specifies an AWS Config rule for evaluating whether your AWS resources comply with your desired configurations.
- use this action for `custom AWS Config rules` and `AWS managed Config rules`.
- A custom AWS Config rule is a rule that you develop and maintain.
- An AWS managed Config rule is a customizable, predefined rule that AWS Config provides.

to add new custom AWS Config rule
1. first create the AWS Lambda function that the rule invokes to evaluate your resources.
2. When you use the PutConfigRule action to add the rule to AWS Config, you must specify the Lambda function ARN.
3. Specify the ARN for the `SourceIdentifier` key. This key is part of the Source object, which is part of the ConfigRule object.

to add new AWS managed Config rule
1. pecify the rule's identifier for the `SourceIdentifier` key.
2. To reference AWS managed Config rule identifiers, see About AWS Managed Config Rules.


<font color=red> ConfigRuleName </font>

- For new rule, specify the `ConfigRuleName` in the ConfigRule object.
  - Do not specify the ConfigRuleArn or the ConfigRuleId. These values are generated by AWS Config for new rules.
- updating a rule, specify the rule by `ConfigRuleName, ConfigRuleId, or ConfigRuleArn` in the ConfigRule data type that you use in this request.



```yaml
Type: AWS::Config::ConfigRule
Properties:
  # If don't specify a name, AWS CloudFormation generates a unique physical ID
  ConfigRuleName: String
  Description: String
  # A string, in JSON format, that is passed to the AWS Config rule Lambda function.
  InputParameters: Json
    application: 'oneNote'
    platformType: 'Win'
  # The maximum frequency with which AWS Config runs evaluations for a rule.
  # You can specify a value for MaximumExecutionFrequency when:
  # use an AWS managed rule that is triggered at a periodic frequency.
  # the custom rule is triggered when AWS Config delivers the configuration snapshot.
  # By default, rules with a periodic trigger are evaluated every 24 hours.
  # Allowed values: One_Hour | Six_Hours | Three_Hours | Twelve_Hours | TwentyFour_Hours
  MaximumExecutionFrequency: String

  # Defines which resources can trigger an evaluation for the rule.
  # - The scope can include one or more resource types
  # - a combination of one resource type and one resource ID,
  # - or a combination of a tag key and value.
  # Specify a scope to constrain the resources that can trigger an evaluation for the rule.
  # If do not specify a scope, evaluations are triggered when any resource in the recording group changes.
  Scope:
    # The ID of the only AWS resource that want to trigger an evaluation for the rule.
    # If specify a resource ID, must specify one resource type for ComplianceResourceTypes.
    ComplianceResourceId: String
    # The resource types of only those AWS resources that you want to trigger an evaluation for the rule.
    # can only specify one type if you also specify a resource ID for ComplianceResourceId.
    ComplianceResourceTypes:
      - String
      - "AWS::EC2::Volume"
    # The tag key that is applied to only those AWS resources that you want to trigger an evaluation for the rule.
    TagKey: String
    # The tag value applied to only those AWS resources that you want to trigger an evaluation for the rule. If you specify a value for TagValue, you must also specify a value for TagKey.
    TagValue: String

  # Provides the rule owner (AWS or customer),
  # the rule identifier
  # and the notifications that cause the function to evaluate your AWS resources.
  Source:
    Owner: String AWS/
    # Indicates whether AWS or the customer owns and manages the AWS Config rule.
    SourceDetails:
      - SourceDetail
    # Provides the source and type of the event
    # that causes AWS Config to evaluate your AWS resources.
    SourceIdentifier: String
    # For AWS Config managed rules, a predefined identifier from a list.
    # - For example, IAM_PASSWORD_POLICY is a managed rule.
    SourceIdentifier: "REQUIRED_TAGS"
    # For custom rules, the identifier is the ARN of the rule's AWS Lambda function
    # - such as arn:aws:lambda:us-east-2:123456789012:function:custom_rule_name.
    SourceIdentifier: "arn:aws:lambda:us-east-2:123456789012:function:custom_rule_name"
```

Return values
- Ref
  - When you pass the logical ID of this resource to the intrinsic Ref function, Ref returns the rule name,
  - such as mystack-MyConfigRule-12ABCFPXHV4OV.

- Fn::GetAtt
  - The Fn::GetAtt intrinsic function returns a value for a specified attribute of this type.
  - The following are the available attributes and sample return values.
  - For more information about using the Fn::GetAtt intrinsic function, see Fn::GetAtt.
    - Arn
      - The Amazon Resource Name (ARN) of the AWS Config rule
      - such as arn:aws:config:us-east-1:123456789012:config-rule/config-rule-a1bzhi.

    - Compliance.Type
      - The compliance status of an AWS Config rule
      - such as COMPLIANT or NON_COMPLIANT.

    - ConfigRuleId
      - The ID of the AWS Config rule
      - such as config-rule-a1bzhi.

---


## example

### ConfigRuleForVolumeTags

```yaml
ConfigRuleForVolumeTags:
  Type: AWS::Config::ConfigRule
  Properties:
    InputParameters:
      tag1Key: CostCenter
    Scope:
      ComplianceResourceTypes:
        - "AWS::EC2::Volume"
    Source:
      Owner: AWS
      SourceIdentifier: "REQUIRED_TAGS"
```

### Rule Using Lambda Function

- creates a custom configuration rule that uses a Lambda function.
- The function checks whether an EC2 volume has the `AutoEnableIO` property set to true.
- Note that the configuration rule has a dependency on the Lambda policy so that the rule calls the function only after it's permitted to do so.


```yaml

ConfigPermissionToCallLambda:
  Type: AWS::Lambda::Permission
  Properties:
    FunctionName:
      Fn::GetAtt:
        - VolumeAutoEnableIOComplianceCheck
        - Arn
    Action: "lambda:InvokeFunction"
    Principal: "config.amazonaws.com"

VolumeAutoEnableIOComplianceCheck:
  Type: AWS::Lambda::Function
  Properties:
    Code:
      ZipFile:
        !Sub |
          var aws  = require('aws-sdk');
          var config = new aws.ConfigService();
          var ec2 = new aws.EC2();
          exports.handler = function(event, context) {
              compliance = evaluateCompliance(event, function(compliance, event) {
                    var configurationItem = JSON.parse(event.invokingEvent).configurationItem;
                    var putEvaluationsRequest = {
                        Evaluations: [{
                            ComplianceResourceType: configurationItem.resourceType,
                            ComplianceResourceId: configurationItem.resourceId,
                            ComplianceType: compliance,
                            OrderingTimestamp: configurationItem.configurationItemCaptureTime
                        }],
                        ResultToken: event.resultToken
                    };
                    config.putEvaluations(putEvaluationsRequest, function(err, data) {
                        if (err) context.fail(err);
                        else context.succeed(data);
                    });
                });
            };
            function evaluateCompliance(event, doReturn) {
                var configurationItem = JSON.parse(event.invokingEvent).configurationItem;
                var status = configurationItem.configurationItemStatus;
                if (configurationItem.resourceType !== 'AWS::EC2::Volume' || event.eventLeftScope || (status !== 'OK' && status !== 'ResourceDiscovered'))
                    doReturn('NOT_APPLICABLE', event);
                else ec2.describeVolumeAttribute({VolumeId: configurationItem.resourceId, Attribute: 'autoEnableIO'}, function(err, data) {
                    if (err) context.fail(err);
                    else if (data.AutoEnableIO.Value) doReturn('COMPLIANT', event);
                    else doReturn('NON_COMPLIANT', event);
                });
            }
    Handler: "index.handler"
    Runtime: nodejs12.x
    Timeout: 30
    Role:
      Fn::GetAtt:
        - LambdaExecutionRole
        - Arn


ConfigRuleForVolumeAutoEnableIO:
  Type: AWS::Config::ConfigRule
  Properties:
    ConfigRuleName: ConfigRuleForVolumeAutoEnableIO
    Scope:
      ComplianceResourceId: Ref: Ec2Volume
      ComplianceResourceTypes: "AWS::EC2::Volume"
    Source:
      Owner: "CUSTOM_LAMBDA"
      SourceDetails:
        - EventSource: "aws.config"
          MessageType: "ConfigurationItemChangeNotification"
      SourceIdentifier:
        Fn::GetAtt:
          - VolumeAutoEnableIOComplianceCheck
          - Arn
  DependsOn: ConfigPermissionToCallLambda

```



## Deploy Managed Config Rules using CloudFormation and CodePipeline


### manually


```bash
aws cloudformation create-stack \
  --stack-name cloud-trail-encryption-enabled \
  --template-url https://s3.amazonaws.com/aws-configservice-us-east-1/cloudformation-templates-for-managed-rules/CLOUD_TRAIL_ENCRYPTION_ENABLED.template \
  --capabilities CAPABILITY_NAMED_IAM \
  --disable-rollback
```


### configuring a deployment pipeline in AWS CodePipeline


zip and upload all of the source files to S3 so that they can be committed to the CodeCommit repository that is automatically provisioned by the stack generated by the managed-config-rules-pipeline.yml template.

```yaml
buildspec.yml
# AWS CodeBuild will use this buildspec to download the latest CloudFormation template for the Managed Config Rules that AWS manages.
version: 0.2
phases:
  build:
    commands:
      - wget https://s3.amazonaws.com/aws-configservice-us-east-1/cloudformation-templates-for-managed-rules/CLOUD_TRAIL_ENCRYPTION_ENABLED.template
  post_build:
    commands:
      - echo Build completed on `date`
artifacts:
  type: zip
  files:
    - CLOUD_TRAIL_ENCRYPTION_ENABLED.template


managed-config-rules-pipeline.yml
AWSTemplateFormatVersion: '2010-09-09'
Description: CodePipeline for Deploying Multiple Managed Config Rules
Parameters:
  RepositoryBranch:
    Description: The name of the branch for the CodeCommit repo
    Type: String
    Default: master
    AllowedPattern: "[\\x20-\\x7E]*"
    ConstraintDescription: Can contain only ASCII characters.
  CodeCommitS3Bucket:
    Description: S3 bucket that holds zip of source code for CodeCommit Repo
    Type: String
  CodeCommitS3Key:
    Description: zipfile key located in CodeCommitS3Bucket
    Type: String
Resources:
  ArtifactBucket:
    Type: AWS::S3::Bucket
    DeletionPolicy: Delete
  CodeBuildRole:
    Type: AWS::IAM::Role
    Properties:
      Path: "/"
      AssumeRolePolicyDocument:
        Statement:
        - Effect: Allow
          Action: sts:AssumeRole
          Principal:
            Service: codebuild.amazonaws.com
      Policies:
      - PolicyName: codebuild-service
        PolicyDocument:
          Statement:
          - Action:
            - logs:*
            - cloudwatch:*
            - codebuild:*
            - s3:*
            Effect: Allow
            Resource: "*"
          Version: '2012-10-17'
  CodeBuildConfigRules:
    Type: AWS::CodeBuild::Project
    DependsOn: CodeBuildRole
    Properties:
      Name:
        Fn::Join:
        - ''
        - - Run
          - "CodePipeline"
          - Ref: AWS::StackName
      Description: Build application
      ServiceRole:
        Fn::GetAtt:
        - CodeBuildRole
        - Arn
      Artifacts:
        Type: no_artifacts
      Environment:
        EnvironmentVariables:
        - Name: S3_BUCKET
          Value:
            Ref: ArtifactBucket
        Type: LINUX_CONTAINER
        ComputeType: BUILD_GENERAL1_SMALL
        Image: aws/codebuild/eb-nodejs-4.4.6-amazonlinux-64:2.1.3
      Source:
        BuildSpec: buildspec.yml
        Location:
          Fn::Join:
          - ''
          - - https://git-codecommit.
            - Ref: AWS::Region
            - ".amazonaws.com/v1/repos/"
            - Ref: AWS::StackName
        Type: CODECOMMIT
      TimeoutInMinutes: 10
      Tags:
      - Key: Owner
        Value: MyCodeBuildProject
  MySNSTopic:
    Type: AWS::SNS::Topic
  CodeCommitRepo:
    Type: AWS::CodeCommit::Repository
    Properties:
      RepositoryName:
        Ref: AWS::StackName
      RepositoryDescription: CodeCommit Repository for Config Rule solution
      Code:
        S3:
          Bucket: !Ref CodeCommitS3Bucket
          Key: !Ref CodeCommitS3Key
      Triggers:
      - Name: MasterTrigger
        CustomData:
          Ref: AWS::StackName
        DestinationArn:
          Ref: MySNSTopic
        Events: all
  CloudFormationTrustRole:
    DependsOn:
    - ArtifactBucket
    Description: Creating service role in IAM for AWS CloudFormation
    Properties:
      AssumeRolePolicyDocument:
        Statement:
        - Action: sts:AssumeRole
          Effect: Allow
          Principal:
            Service: cloudformation.amazonaws.com
      Path: "/"
      Policies:
      - PolicyDocument:
          Statement:
          - Action:
            - s3:PutObject
            - s3:GetObject
            - s3:GetObjectVersion
            Effect: Allow
            Resource:
            - Fn::Join:
              - ''
              - - 'arn:aws:s3:::'
                - Ref: ArtifactBucket
            - Fn::Join:
              - ''
              - - 'arn:aws:s3:::'
                - Ref: ArtifactBucket
                - "/*"
          - Action:
            - sns:CreateTopic
            - sns:DeleteTopic
            - sns:ListTopics
            - sns:GetTopicAttributes
            - sns:SetTopicAttributes
            - s3:CreateBucket
            - s3:DeleteBucket
            - events:*
            - config:*
            Effect: Allow
            Resource: "*"
          - Action: iam:PassRole
            Effect: Allow
            Resource: "*"
          - Action:
            - cloudformation:CreateChangeSet
            - config:*
            Effect: Allow
            Resource: arn:aws:cloudformation:us-east-1:aws:transform/Serverless-2016-10-31
        PolicyName: CloudFormationRolePolicy
      RoleName:
        Fn::Join:
        - "-"
        - - stelligent
          - Ref: AWS::StackName
          - CloudFormation
    Type: AWS::IAM::Role
  CodePipelineRole:
    Type: AWS::IAM::Role
    Properties:
      Path: "/"
      AssumeRolePolicyDocument:
        Statement:
        - Effect: Allow
          Action: sts:AssumeRole
          Principal:
            Service: codepipeline.amazonaws.com
      Policies:
      - PolicyName: codepipeline-service
        PolicyDocument:
          Statement:
          - Action:
            - s3:GetObject
            - s3:GetObjectVersion
            - s3:GetBucketVersioning
            Resource: "*"
            Effect: Allow
          - Action: s3:PutObject
            Resource: arn:aws:s3:::codepipeline*
            Effect: Allow
          - Action:
            - s3:GetObject
            - s3:GetObjectVersion
            - s3:GetBucketVersioning
            - s3:PutObject
            - iam:PassRole
            Resource: "*"
            Effect: Allow
          - Action:
            - codecommit:*
            - codebuild:*
            - cloudformation:*
            Resource: "*"
            Effect: Allow
          Version: '2012-10-17'
  PipelineBucket:
    Type: AWS::S3::Bucket
    DeletionPolicy: Delete
  Pipeline:
    Type: AWS::CodePipeline::Pipeline
    Properties:
      RoleArn: !GetAtt CodePipelineRole.Arn
    Properties:
      RoleArn:
        Fn::Join:
        - ''
        - - 'arn:aws:iam::'
          - Ref: AWS::AccountId
          - ":role/"
          - Ref: CodePipelineRole
      Stages:
      - Name: Source
        Actions:
        - InputArtifacts: []
          Name: Source
          ActionTypeId:
            Category: Source
            Owner: AWS
            Version: '1'
            Provider: CodeCommit
          OutputArtifacts:
          - Name: MyApp
          Configuration:
            BranchName:
              Ref: RepositoryBranch
            RepositoryName:
              Ref: AWS::StackName
          RunOrder: 1
      - Name: Build
        Actions:
        - InputArtifacts:
          - Name: MyApp
          Name: StoreConfigRules
          ActionTypeId:
            Category: Build
            Owner: AWS
            Version: '1'
            Provider: CodeBuild
          OutputArtifacts:
          - Name: ConfigRuleTemplateArtifacts
          Configuration:
            ProjectName:
              Ref: CodeBuildConfigRules
          RunOrder: 1
      - Name: Deploy
        Actions:
        - InputArtifacts:
          - Name: ConfigRuleTemplateArtifacts
          Name: DeployCloudTrailEncryptionTemplate
          ActionTypeId:
            Category: Deploy
            Owner: AWS
            Version: '1'
            Provider: CloudFormation
          OutputArtifacts: []
          Configuration:
            ActionMode: CHANGE_SET_REPLACE
            ChangeSetName: pipeline-changeset
            RoleArn:
              Fn::GetAtt:
              - CloudFormationTrustRole
              - Arn
            Capabilities: CAPABILITY_IAM
            StackName:
              Fn::Join:
              - ''
              - - ""
                - Ref: AWS::StackName
                - "-"
                - Ref: AWS::Region
                - ""
            TemplatePath: ConfigRuleTemplateArtifacts::CLOUD_TRAIL_ENCRYPTION_ENABLED.template
          RunOrder: 1
        - ActionTypeId:
            Category: Deploy
            Owner: AWS
            Provider: CloudFormation
            Version: 1
          Configuration:
            ActionMode: CHANGE_SET_EXECUTE
            ChangeSetName: pipeline-changeset
            StackName:
              Fn::Join:
              - ''
              - - ""
                - Ref: AWS::StackName
                - "-"
                - Ref: AWS::Region
                - ""
          InputArtifacts: []
          Name: ExecuteChangeSetCloudTrailEncryption
          OutputArtifacts: []
          RunOrder: 2
      ArtifactStore:
        Type: S3
        Location: !Ref PipelineBucket
Outputs:
  PipelineUrl:
    Value: !Sub https://console.aws.amazon.com/codepipeline/home?region=${AWS::Region}#/view/${Pipeline}
    Description: CodePipeline URL

```


```bash

aws cloudformation create-stack \
  --stack-name managed-config-rules-pipeline \
  --template-body file:///home/ec2-user/environment/aws-compliance-workshop/lesson0-setup/managed-config-rules/managed-config-rules-pipeline.yml \
  --parameters ParameterKey=CodeCommitS3Bucket, ParameterValue=ccoa-mcr-$(aws sts get-caller-identity --output text --query 'Account') ParameterKey=CodeCommitS3Key, ParameterValue=ccoa-mcr-examples.zip \
  --capabilities CAPABILITY_NAMED_IAM \
  --disable-rollback

```

























.
