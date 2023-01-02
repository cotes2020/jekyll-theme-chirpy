---
title: AWS - CodeDevelop - CloudFormation - Template setup EC2forCodeBuild
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, CodeDevelop]
tags: [AWS]
toc: true
image:
---

[toc]

---

# Template setup EC2forCodeBuild

Automate provisioning of CodeBuild with CodePipeline, CodeCommit, and CodeDeploy.

**WARNING** This template creates one or more Amazon EC2 instances. You will be billed for the AWS resources used if you create a stack from this template.",

```json

// example from AWS WhitePaper

{
  "Description":"create instance for codedeploy",
  "AWSTemplateFormatVersion":"2010-09-09",
  "Parameters":{
    // "EmailAddress":{
    //   "Description":"Email Address for sending SNS notifications for CodeCommit",
    //   "Type":"String"
    // },
    // "RepositoryBranch":{
    //   "Description":"The name of the branch for the CodeCommit repo",
    //   "Type":"String",
    //   "Default":"master",
    //   "AllowedPattern":"[\\x20-\\x7E]*",
    //   "ConstraintDescription":"Can contain only ASCII characters."
    // },
    "TagKey":{
      "Description":"The EC2 tag key that is associated with EC2 instances on which CodeDeploy agent is installed, target for deployments",
      "Type":"String",
      "Default":"Name",
      "AllowedPattern": "[\x20-\\x7E\*]",
      "ConstraintDescription":"Can contain only ASCII characters."
    },
    "TagValue":{
      "Description":"The EC2 tag value that identifies this as a target for deployments.",
      "Type":"String",
      "Default":"CodeDeployEC2Tag",
      "AllowedPattern":"[\\x20-\\x7E]*",
      "ConstraintDescription":"Can contain only ASCII characters."
    },
    "KeyPairName":{
      "Description":"Name of an existing Amazon EC2 key pair to enable SSH access to the instances.",
      "Type":"String",
      "MinLength":"1",
      "MaxLength":"255",
      "AllowedPattern":"[\\x20-\\x7E]*",
      "ConstraintDescription":"Can contain only ASCII characters."
    },
      "InstanceType":{
      "Description":"EC2 instance type",
      "Type":"String",
      "Default": "ta.micro",
      "ConstraintDescription":"a valid EC2 type."
    },
      "InstanceCount":{
      "Description":"Number of the ec2 instance",
      "Type":"Number",
      "Default": "1",
      "ConstraintDescription":"a valid EC2 type."
    },
      "OperatingSystem":{
      "Description":"ec2 instance OS",
      "Type":"String",
      "Default": "Linux",
      "ConstraintDescription":"windows or linux",
      "AllowedValues": ["Linux", "Windows"]
    },
      "SSHLocation":{
      "Description":"the IP that can connect to ec2 by ssh",
      "Type":"String",
      "MinLength":"9",
      "MaxLength":"18",
      "Default": "0.0.0.0/0",
      "AllowedPattern":"(\\d{1,3})\\.(\\d{1,3})\\.(\\d{1,3})\\.(\\d{1,3})/(\\d{1,2",
      "ConstraintDescription":"in format of x.x.x.x/x"
    }
  },
  "Mappings": {
      "RegionOS2AMI": {
          "us-east-1":{
              "Linux": "ami-xxx",
              "Windows": "ami-ssss"
          },
          "us-east-2":{
              "Linux": "ami-xxx",
              "Windows": "ami-ssss"
          },
          "us-west-1":{
              "Linux": "ami-xxx",
              "Windows": "ami-ssss"
          },
          "us-west-2":{
              "Linux": "ami-xxx",
              "Windows": "ami-ssss"
          }
      }
  },
  "OS2SSHPort":{
      "Linux": {"SSHPort": "22"},
      "Windows": {"SSHPort": "22"}
  }
  "Metadata":{
    "AWS::CloudFormation::Interface":{
      "ParameterGroups":[
        {
          "Label":{
            "default":"Dynamic Configuration"
          },
          "Parameters":[
            "EC2KeyPairName",
            "RepositoryBranch"
          ]
        }
      ],
      "ParameterLabels":{
        "EC2KeyPairName":{
          "default":"EC2 KeyPair Name"
        },
        "RepositoryName":{
          "default":"CodeCommit Repository Name"
        },
        "RepositoryBranch":{
          "default":"CodeCommit Repository Branch"
        }
      }
    }
  },
  "Resources":{
    "CodeBuildRole":{
      "Type":"AWS::IAM::Role",
      "Properties":{
        "AssumeRolePolicyDocument":{
          "Statement":[
            {
              "Effect":"Allow",
              "Principal":{
                "Service":[
                  "codebuild.amazonaws.com"
                ]
              },
              "Action":[
                "sts:AssumeRole"
              ]
            }
          ]
        },
        "Path":"/",
        "Policies":[
          {
            "PolicyName":"codebuild-service",
            "PolicyDocument":{
              "Statement":[
                {
                  "Effect":"Allow",
                  "Action":"*",
                  "Resource":"*"
                }
              ],
              "Version":"2012-10-17"
            }
          }
        ]
      }
    },
    "CodePipelineRole":{
      "Type":"AWS::IAM::Role",
      "Properties":{
        "AssumeRolePolicyDocument":{
          "Statement":[
            {
              "Effect":"Allow",
              "Principal":{
                "Service":[
                  "codepipeline.amazonaws.com"
                ]
              },
              "Action":[
                "sts:AssumeRole"
              ]
            }
          ]
        },
        "Path":"/",
        "Policies":[
          {
            "PolicyName":"codepipeline-service",
            "PolicyDocument":{
              "Statement":[
                {
                  "Action":[
                    "codecommit:GetBranch",
                    "codecommit:GetCommit",
                    "codecommit:UploadArchive",
                    "codecommit:GetUploadArchiveStatus",
                    "codecommit:CancelUploadArchive",
                    "codebuild:*"
                  ],
                  "Resource":"*",
                  "Effect":"Allow"
                },
                {
                  "Action":[
                    "s3:GetObject",
                    "s3:GetObjectVersion",
                    "s3:GetBucketVersioning"
                  ],
                  "Resource":"*",
                  "Effect":"Allow"
                },
                {
                  "Action":[
                    "s3:PutObject"
                  ],
                  "Resource":[
                    "arn:aws:s3:::codepipeline*",
                    "arn:aws:s3:::elasticbeanstalk*"
                  ],
                  "Effect":"Allow"
                },
                {
                  "Action":[
                    "codedeploy:CreateDeployment",
                    "codedeploy:GetApplicationRevision",
                    "codedeploy:GetDeployment",
                    "codedeploy:GetDeploymentConfig",
                    "codedeploy:RegisterApplicationRevision"
                  ],
                  "Resource":"*",
                  "Effect":"Allow"
                },
                {
                  "Action":[
                    "elasticbeanstalk:*",
                    "ec2:*",
                    "elasticloadbalancing:*",
                    "autoscaling:*",
                    "cloudwatch:*",
                    "s3:*",
                    "sns:*",
                    "cloudformation:*",
                    "rds:*",
                    "sqs:*",
                    "ecs:*",
                    "iam:PassRole"
                  ],
                  "Resource":"*",
                  "Effect":"Allow"
                },
                {
                  "Action":[
                    "lambda:InvokeFunction",
                    "lambda:ListFunctions"
                  ],
                  "Resource":"*",
                  "Effect":"Allow"
                }
              ],
              "Version":"2012-10-17"
            }
          }
        ]
      }
    },
    "CodeBuildJavaProject":{
      "Type":"AWS::CodeBuild::Project",
      "DependsOn":"CodeBuildRole",
      "Properties":{
        "Name":{
          "Ref":"AWS::StackName"
        },
        "Description":"Build Java application",
        "ServiceRole":{
          "Fn::GetAtt":[
            "CodeBuildRole",
            "Arn"
          ]
        },
        "Artifacts":{
          "Type":"no_artifacts"
        },
        "Environment":{
          "Type":"linuxContainer",
          "ComputeType":"BUILD_GENERAL1_SMALL",
          "Image":"aws/codebuild/java:openjdk-8"
        },
        "Source":{
          "Location":{
            "Fn::Join":[
              "",
              [
                "https://git-codecommit.",
                {
                  "Ref":"AWS::Region"
                },
                ".amazonaws.com/v1/repos/",
                {
                  "Ref":"AWS::StackName"
                }
              ]
            ]
          },
          "Type":"CODECOMMIT"
        },
        "TimeoutInMinutes":10,
        "Tags":[
          {
            "Key":"Owner",
            "Value":"JavaTomcatProject"
          }
        ]
      }
    },
    "MySNSTopic":{
      "Type":"AWS::SNS::Topic",
      "Properties":{
        "Subscription":[
          {
            "Endpoint":{
              "Ref":"EmailAddress"
            },
            "Protocol":"email"
          }
        ]
      }
    },
    "CodeDeployEC2InstancesStack":{
      "Type":"AWS::CloudFormation::Stack",
      "Properties":{
        "TemplateURL":"https://s3.amazonaws.com/stelligent-public/cloudformation-templates/github/labs/codebuild/CodeDeploy_SampleCF_Template.json",
        "TimeoutInMinutes":"60",
        "Parameters":{
          "TagValue":{
            "Ref":"TagValue"
          },
          "KeyPairName":{
            "Ref":"EC2KeyPairName"
          }
        }
      }
    },
    "CodeCommitJavaRepo":{
      "Type":"AWS::CodeCommit::Repository",
      "Properties":{
        "RepositoryName":{
          "Ref":"AWS::StackName"
        },
        "RepositoryDescription":"CodeCommit Repository",
        "Triggers":[
          {
            "Name":"MasterTrigger",
            "CustomData":{
              "Ref":"AWS::StackName"
            },
            "DestinationArn":{
              "Ref":"MySNSTopic"
            },
            "Events":[
              "all"
            ]
          }
        ]
      }
    },
    "MyApplication":{
      "Type":"AWS::CodeDeploy::Application",
      "DependsOn":"CodeDeployEC2InstancesStack"
    },
    "MyDeploymentGroup":{
      "Type":"AWS::CodeDeploy::DeploymentGroup",
      "DependsOn":"MyApplication",
      "Properties":{
        "ApplicationName":{
          "Ref":"MyApplication"
        },
        "DeploymentConfigName":"CodeDeployDefault.AllAtOnce",
        "Ec2TagFilters":[
          {
            "Key":{
              "Ref":"TagKey"
            },
            "Value":{
              "Ref":"TagValue"
            },
            "Type":"KEY_AND_VALUE"
          }
        ],
        "ServiceRoleArn":{
          "Fn::GetAtt":[
            "CodeDeployEC2InstancesStack",
            "Outputs.CodeDeployTrustRoleARN"
          ]
        }
      }
    },
    "CodePipelineStack":{
      "Type":"AWS::CodePipeline::Pipeline",
      "DependsOn":"CodeBuildJavaProject",
      "Properties":{
        "RoleArn":{
          "Fn::Join":[
            "",
            [
              "arn:aws:iam::",
              {
                "Ref":"AWS::AccountId"
              },
              ":role/",
              {
                "Ref":"CodePipelineRole"
              }
            ]
          ]
        },
        "Stages":[
          {
            "Name":"Source",
            "Actions":[
              {
                "InputArtifacts":[

                ],
                "Name":"Source",
                "ActionTypeId":{
                  "Category":"Source",
                  "Owner":"AWS",
                  "Version":"1",
                  "Provider":"CodeCommit"
                },
                "OutputArtifacts":[
                  {
                    "Name":"MyApp"
                  }
                ],
                "Configuration":{
                  "BranchName":{
                    "Ref":"RepositoryBranch"
                  },
                  "RepositoryName":{
                    "Ref":"AWS::StackName"
                  }
                },
                "RunOrder":1
              }
            ]
          },
          {
            "Name":"Build",
            "Actions":[
              {
                "InputArtifacts":[
                  {
                    "Name":"MyApp"
                  }
                ],
                "Name":"Build",
                "ActionTypeId":{
                  "Category":"Build",
                  "Owner":"AWS",
                  "Version":"1",
                  "Provider":"CodeBuild"
                },
                "OutputArtifacts":[
                  {
                    "Name":"MyAppBuild"
                  }
                ],
                "Configuration":{
                  "ProjectName":{
                    "Ref":"CodeBuildJavaProject"
                  }
                },
                "RunOrder":1
              }
            ]
          },
          {
            "Name":"Deploy",
            "Actions":[
              {
                "InputArtifacts":[
                  {
                    "Name":"MyAppBuild"
                  }
                ],
                "Name":"DemoFleet",
                "ActionTypeId":{
                  "Category":"Deploy",
                  "Owner":"AWS",
                  "Version":"1",
                  "Provider":"CodeDeploy"
                },
                "OutputArtifacts":[

                ],
                "Configuration":{
                  "ApplicationName":{
                    "Ref":"MyApplication"
                  },
                  "DeploymentGroupName":{
                    "Ref":"MyDeploymentGroup"
                  }
                },
                "RunOrder":1
              }
            ]
          }
        ],
        "ArtifactStore":{
          "Type":"S3",
          "Location":{
            "Fn::Join":[
              "",
              [
                "codepipeline-",
                {
                  "Ref":"AWS::Region"
                },
                "-",
                {
                  "Ref":"AWS::AccountId"
                }
              ]
            ]
          }
        }
      }
    }
  },
  "Outputs":{
    "CodeBuildURL":{
      "Value":{
        "Fn::Join":[
          "",
          [
            "https://console.aws.amazon.com/codebuild/home?region=",
            {
              "Ref":"AWS::Region"
            },
            "#/projects/",
            {
              "Ref":"CodeBuildJavaProject"
            },
            "/view"
          ]
        ]
      },
      "Description":"CodeBuild URL"
    },
    "CodeCommitURL":{
      "Value":{
        "Fn::Join":[
          "",
          [
            "https://console.aws.amazon.com/codecommit/home?region=",
            {
              "Ref":"AWS::Region"
            },
            "#/repository/",
            {
              "Ref":"AWS::StackName"
            },
            "/browse/HEAD/--/"
          ]
        ]
      },
      "Description":"Git Repository URL"
    },
    "CodeDeployURL":{
      "Value":{
        "Fn::Join":[
          "",
          [
            "https://console.aws.amazon.com/codedeploy/home?region=",
            {
              "Ref":"AWS::Region"
            },
            "#/deployments/"
          ]
        ]
      },
      "Description":"CodeDeploy URL"
    },
    "CloneUrlSsh":{
      "Value":{
        "Fn::Join":[
          "",
          [
            "git clone ",
            {
              "Fn::GetAtt":[
                "CodeCommitJavaRepo",
                "CloneUrlSsh"
              ]
            },
            ""
          ]
        ]
      },
      "Description":"Git command for CodeCommit repository"
    },
    "CodePipelineURL":{
      "Value":{
        "Fn::Join":[
          "",
          [
            "https://console.aws.amazon.com/codepipeline/home?region=",
            {
              "Ref":"AWS::Region"
            },
            "#/view/",
            {
              "Ref":"CodePipelineStack"
            }
          ]
        ]
      },
      "Description":"CodePipeline URL"
    }
  }
}

```
