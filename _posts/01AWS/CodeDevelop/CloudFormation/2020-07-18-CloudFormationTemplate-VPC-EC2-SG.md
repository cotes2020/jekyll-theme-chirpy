---
title: AWS - CodeDevelop - CloudFormation Template - setup VPC_Single_Instance_In_Subnet
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, CodeDevelop]
tags: [AWS]
toc: true
image:
---

[toc]

---

# Template - setup VPC_Single_Instance_In_Subnet


AWS CloudFormation Sample Template VPC_Single_Instance_In_Subnet:
- create a VPC and add an EC2 instance with an Elastic IP address and a security group.

**WARNING** This template creates an Amazon EC2 instance.

![Screen Shot 2021-01-18 at 19.49.45](https://i.imgur.com/ZulzkQY.png)

```json
{
  "AWSTemplateFormatVersion": "2010-09-09",
  "Description": "AWS CloudFormation Sample Template VPC_Single_Instance_In_Subnet: Sample template showing how to create a VPC and add an EC2 instance with an Elastic IP address and a security group. **WARNING** This template creates an Amazon EC2 instance. You will be billed for the AWS resources used if you create a stack from this template.",
  "Parameters": {
    "InstanceType": {
      "Description": "WebServer EC2 instance type",
      "Type": "String",
      "Default": "t2.micro",
      "AllowedValues": [
        "t1.micro",
        "t2.micro",
        "t2.small",
        "t2.medium",
        "m1.small",
        "m1.medium",
        "m1.large",
        "m1.xlarge",
        "m2.xlarge",
        "m2.2xlarge",
        "m2.4xlarge",
        "m3.medium",
        "m3.large",
        "m3.xlarge",
        "m3.2xlarge",
        "c1.medium",
        "c1.xlarge",
        "c3.large",
        "c3.xlarge",
        "c3.2xlarge",
        "c3.4xlarge",
        "c3.8xlarge",
        "c4.large",
        "c4.xlarge",
        "c4.2xlarge",
        "c4.4xlarge",
        "c4.8xlarge",
        "g2.2xlarge",
        "r3.large",
        "r3.xlarge",
        "r3.2xlarge",
        "r3.4xlarge",
        "r3.8xlarge",
        "i2.xlarge",
        "i2.2xlarge",
        "i2.4xlarge",
        "i2.8xlarge",
        "d2.xlarge",
        "d2.2xlarge",
        "d2.4xlarge",
        "d2.8xlarge",
        "hi1.4xlarge",
        "hs1.8xlarge",
        "cr1.8xlarge",
        "cc2.8xlarge",
        "cg1.4xlarge"
      ],
      "ConstraintDescription": "must be a valid EC2 instance type."
    },
    "KeyName": {
      "Description": "Name of an existing EC2 KeyPair to enable SSH access to the instance.",
      "Type": "AWS::EC2::KeyPair::KeyName",
      "ConstraintDescription": "must be the name of an existing EC2 KeyPair."
    },
    "SSHLocation": {
      "Description": " The IP address range that can be used access the web server using SSH.",
      "Type": "String",
      "MinLength": "9",
      "MaxLength": "18",
      "Default": "0.0.0.0/0",
      "AllowedPattern": "(\\d{1,3})\\.(\\d{1,3})\\.(\\d{1,3})\\.(\\d{1,3})/(\\d{1,2})",
      "ConstraintDescription": "must be a valid IP CIDR range of the form x.x.x.x/x."
    }
  },
  "Mappings": {
    "AWSInstanceType2Arch": {
      "t1.micro": {
        "Arch": "PV64"
      },
      "t2.micro": {
        "Arch": "HVM64"
      },
      "t2.small": {
        "Arch": "HVM64"
      },
      "t2.medium": {
        "Arch": "HVM64"
      },
      "m1.small": {
        "Arch": "PV64"
      },
      "m1.medium": {
        "Arch": "PV64"
      },
      "m1.large": {
        "Arch": "PV64"
      },
      "m1.xlarge": {
        "Arch": "PV64"
      },
      "m2.xlarge": {
        "Arch": "PV64"
      },
      "m2.2xlarge": {
        "Arch": "PV64"
      },
      "m2.4xlarge": {
        "Arch": "PV64"
      },
      "m3.medium": {
        "Arch": "HVM64"
      },
      "m3.large": {
        "Arch": "HVM64"
      },
      "m3.xlarge": {
        "Arch": "HVM64"
      },
      "m3.2xlarge": {
        "Arch": "HVM64"
      },
      "c1.medium": {
        "Arch": "PV64"
      },
      "c1.xlarge": {
        "Arch": "PV64"
      },
      "c3.large": {
        "Arch": "HVM64"
      },
      "c3.xlarge": {
        "Arch": "HVM64"
      },
      "c3.2xlarge": {
        "Arch": "HVM64"
      },
      "c3.4xlarge": {
        "Arch": "HVM64"
      },
      "c3.8xlarge": {
        "Arch": "HVM64"
      },
      "c4.large": {
        "Arch": "HVM64"
      },
      "c4.xlarge": {
        "Arch": "HVM64"
      },
      "c4.2xlarge": {
        "Arch": "HVM64"
      },
      "c4.4xlarge": {
        "Arch": "HVM64"
      },
      "c4.8xlarge": {
        "Arch": "HVM64"
      },
      "g2.2xlarge": {
        "Arch": "HVMG2"
      },
      "r3.large": {
        "Arch": "HVM64"
      },
      "r3.xlarge": {
        "Arch": "HVM64"
      },
      "r3.2xlarge": {
        "Arch": "HVM64"
      },
      "r3.4xlarge": {
        "Arch": "HVM64"
      },
      "r3.8xlarge": {
        "Arch": "HVM64"
      },
      "i2.xlarge": {
        "Arch": "HVM64"
      },
      "i2.2xlarge": {
        "Arch": "HVM64"
      },
      "i2.4xlarge": {
        "Arch": "HVM64"
      },
      "i2.8xlarge": {
        "Arch": "HVM64"
      },
      "d2.xlarge": {
        "Arch": "HVM64"
      },
      "d2.2xlarge": {
        "Arch": "HVM64"
      },
      "d2.4xlarge": {
        "Arch": "HVM64"
      },
      "d2.8xlarge": {
        "Arch": "HVM64"
      },
      "hi1.4xlarge": {
        "Arch": "HVM64"
      },
      "hs1.8xlarge": {
        "Arch": "HVM64"
      },
      "cr1.8xlarge": {
        "Arch": "HVM64"
      },
      "cc2.8xlarge": {
        "Arch": "HVM64"
      }
    },
    "AWSRegionArch2AMI": {
      "us-east-1": {
        "PV64": "ami-1ccae774",
        "HVM64": "ami-1ecae776",
        "HVMG2": "ami-8c6b40e4"
      },
      "us-west-2": {
        "PV64": "ami-ff527ecf",
        "HVM64": "ami-e7527ed7",
        "HVMG2": "ami-abbe919b"
      },
      "us-west-1": {
        "PV64": "ami-d514f291",
        "HVM64": "ami-d114f295",
        "HVMG2": "ami-f31ffeb7"
      },
      "eu-west-1": {
        "PV64": "ami-bf0897c8",
        "HVM64": "ami-a10897d6",
        "HVMG2": "ami-d5bc24a2"
      },
      "eu-central-1": {
        "PV64": "ami-ac221fb1",
        "HVM64": "ami-a8221fb5",
        "HVMG2": "ami-7cd2ef61"
      },
      "ap-northeast-1": {
        "PV64": "ami-27f90e27",
        "HVM64": "ami-cbf90ecb",
        "HVMG2": "ami-6318e863"
      },
      "ap-southeast-1": {
        "PV64": "ami-acd9e8fe",
        "HVM64": "ami-68d8e93a",
        "HVMG2": "ami-3807376a"
      },
      "ap-southeast-2": {
        "PV64": "ami-ff9cecc5",
        "HVM64": "ami-fd9cecc7",
        "HVMG2": "ami-89790ab3"
      },
      "sa-east-1": {
        "PV64": "ami-bb2890a6",
        "HVM64": "ami-b52890a8",
        "HVMG2": "NOT_SUPPORTED"
      },
      "cn-north-1": {
        "PV64": "ami-fa39abc3",
        "HVM64": "ami-f239abcb",
        "HVMG2": "NOT_SUPPORTED"
      }
    }
  },
  "Resources": {
    "VPC": {
      "Type": "AWS::EC2::VPC",
      "Properties": {
        "EnableDnsSupport": "true",
        "EnableDnsHostnames": "true",
        "CidrBlock": "10.0.0.0/16"
      },
      "Metadata": {
        "AWS::CloudFormation::Designer": {
          "id": "96a791f0-938b-4ebe-9f3c-b3fe2a588aee"
        }
      }
    },
    "PublicSubnet": {
      "Type": "AWS::EC2::Subnet",
      "Properties": {
        "CidrBlock": "10.0.0.0/24",
        "VpcId": {
          "Ref": "VPC"
        }
      },
      "Metadata": {
        "AWS::CloudFormation::Designer": {
          "id": "3df467ad-673c-4c48-a41c-3ac1626961e3"
        }
      }
    },
    "InternetGateway": {
      "Type": "AWS::EC2::InternetGateway",
      "Metadata": {
        "AWS::CloudFormation::Designer": {
          "id": "a166c4f5-7cc4-429b-b9d8-2c8c43facc63"
        }
      }
    },
    "VPCGatewayAttachment": {
      "Type": "AWS::EC2::VPCGatewayAttachment",
      "Properties": {
        "VpcId": {
          "Ref": "VPC"
        },
        "InternetGatewayId": {
          "Ref": "InternetGateway"
        }
      },
      "Metadata": {
        "AWS::CloudFormation::Designer": {
          "id": "1790ebeb-2e41-4293-8cc1-aaba134fd1e0"
        }
      }
    },
    "PublicRouteTable": {
      "Type": "AWS::EC2::RouteTable",
      "Properties": {
        "VpcId": {
          "Ref": "VPC"
        }
      },
      "Metadata": {
        "AWS::CloudFormation::Designer": {
          "id": "175bad80-0988-4588-a919-331be705b02d"
        }
      }
    },
    "PublicRoute": {
      "Type": "AWS::EC2::Route",
      "DependsOn": "VPCGatewayAttachment",
      "Properties": {
        "RouteTableId": {
          "Ref": "PublicRouteTable"
        },
        "DestinationCidrBlock": "0.0.0.0/0",
        "GatewayId": {
          "Ref": "InternetGateway"
        }
      },
      "Metadata": {
        "AWS::CloudFormation::Designer": {
          "id": "143bbaa1-66a2-42a5-885f-e6300817103c"
        }
      }
    },
    "PublicSubnetRouteTableAssociation": {
      "Type": "AWS::EC2::SubnetRouteTableAssociation",
      "Properties": {
        "SubnetId": {
          "Ref": "PublicSubnet"
        },
        "RouteTableId": {
          "Ref": "PublicRouteTable"
        }
      },
      "Metadata": {
        "AWS::CloudFormation::Designer": {
          "id": "528e2b71-46e6-4e09-815a-f70630755219"
        }
      }
    },
    "WebServerSecurityGroup": {
      "Type": "AWS::EC2::SecurityGroup",
      "Properties": {
        "VpcId": {
          "Ref": "VPC"
        },
        "GroupDescription": "Allow access from HTTP and SSH traffic",
        "SecurityGroupIngress": [
          {
            "IpProtocol": "tcp",
            "FromPort": "80",
            "ToPort": "80",
            "CidrIp": "0.0.0.0/0"
          },
          {
            "IpProtocol": "tcp",
            "FromPort": "22",
            "ToPort": "22",
            "CidrIp": {
              "Ref": "SSHLocation"
            }
          }
        ]
      },
      "Metadata": {
        "AWS::CloudFormation::Designer": {
          "id": "2e76192b-a4f8-48a5-92b6-abbfa8b83263"
        }
      }
    },
    "WebServerInstance": {
      "Type": "AWS::EC2::Instance",
      "Metadata": {
        "AWS::CloudFormation::Init": {
          "configSets": {
            "All": [
              "ConfigureSampleApp"
            ]
          },
          "ConfigureSampleApp": {
            "packages": {
              "yum": {
                "httpd": []
              }
            },
            "files": {
              "/var/www/html/index.html": {
                "content": {
                  "Fn::Join": [
                    "\n",
                    [
                      "<h1>Congratulations, you have successfully launched the AWS CloudFormation sample.</h1>"
                    ]
                  ]
                },
                "mode": "000644",
                "owner": "root",
                "group": "root"
              }
            },
            "services": {
              "sysvinit": {
                "httpd": {
                  "enabled": "true",
                  "ensureRunning": "true"
                }
              }
            }
          }
        },
        "AWS::CloudFormation::Designer": {
          "id": "0f900c9e-1272-4ec2-8a42-790b074baa39"
        }
      },
      "Properties": {
        "InstanceType": {
          "Ref": "InstanceType"
        },
        "ImageId": {
          "Fn::FindInMap": [
            "AWSRegionArch2AMI",
            {
              "Ref": "AWS::Region"
            },
            {
              "Fn::FindInMap": [
                "AWSInstanceType2Arch",
                {
                  "Ref": "InstanceType"
                },
                "Arch"
              ]
            }
          ]
        },
        "KeyName": {
          "Ref": "KeyName"
        },
        "NetworkInterfaces": [
          {
            "GroupSet": [
              {
                "Ref": "WebServerSecurityGroup"
              }
            ],
            "AssociatePublicIpAddress": "true",
            "DeviceIndex": "0",
            "DeleteOnTermination": "true",
            "SubnetId": {
              "Ref": "PublicSubnet"
            }
          }
        ],
        "UserData": {
          "Fn::Base64": {
            "Fn::Join": [
              "",
              [
                "#!/bin/bash -xe\n",
                "yum update -y aws-cfn-bootstrap\n",
                "# Install the files and packages from the metadata\n",
                "/opt/aws/bin/cfn-init -v ",
                "         --stack ",
                {
                  "Ref": "AWS::StackName"
                },
                "         --resource WebServerInstance ",
                "         --configsets All ",
                "         --region ",
                {
                  "Ref": "AWS::Region"
                },
                "\n",
                "# Signal the status from cfn-init\n",
                "/opt/aws/bin/cfn-signal -e $? ",
                "         --stack ",
                {
                  "Ref": "AWS::StackName"
                },
                "         --resource WebServerInstance ",
                "         --region ",
                {
                  "Ref": "AWS::Region"
                },
                "\n"
              ]
            ]
          }
        }
      },
      "CreationPolicy": {
        "ResourceSignal": {
          "Timeout": "PT5M"
        }
      }
    }
  },
  "Outputs": {
    "URL": {
      "Value": {
        "Fn::Join": [
          "",
          [
            "http://",
            {
              "Fn::GetAtt": [
                "WebServerInstance",
                "PublicIp"
              ]
            }
          ]
        ]
      },
      "Description": "Newly created application URL"
    }
  },
  "Metadata": {
    "AWS::CloudFormation::Designer": {
      "a166c4f5-7cc4-429b-b9d8-2c8c43facc63": {
        "size": {"width": 60, "height": 60},
        "position": {"x": -40, "y": 210},
        "z": 1,
        "embeds": []
      },
      "96a791f0-938b-4ebe-9f3c-b3fe2a588aee": {
        "size": {"width": 320, "height": 250},
        "position": {"x": 70, "y": 190},
        "z": 1,
        "embeds": [
          "2e76192b-a4f8-48a5-92b6-abbfa8b83263",
          "175bad80-0988-4588-a919-331be705b02d"
        ]
      },
      "2e76192b-a4f8-48a5-92b6-abbfa8b83263": {
        "size": {
          "width": 60,
          "height": 60
        },
        "position": {
          "x": 280,
          "y": 370
        },
        "z": 2,
        "parent": "96a791f0-938b-4ebe-9f3c-b3fe2a588aee",
        "embeds": []
      },
      "175bad80-0988-4588-a919-331be705b02d": {
        "size": {
          "width": 120,
          "height": 120
        },
        "position": {
          "x": 90,
          "y": 230
        },
        "z": 2,
        "parent": "96a791f0-938b-4ebe-9f3c-b3fe2a588aee",
        "embeds": [
          "143bbaa1-66a2-42a5-885f-e6300817103c"
        ]
      },
      "1790ebeb-2e41-4293-8cc1-aaba134fd1e0": {
        "source": {
          "id": "a166c4f5-7cc4-429b-b9d8-2c8c43facc63"
        },
        "target": {
          "id": "96a791f0-938b-4ebe-9f3c-b3fe2a588aee"
        },
        "z": 1
      },
      "143bbaa1-66a2-42a5-885f-e6300817103c": {
        "size": {
          "width": 60,
          "height": 60
        },
        "position": {
          "x": 120,
          "y": 260
        },
        "z": 3,
        "parent": "175bad80-0988-4588-a919-331be705b02d",
        "embeds": [],
        "references": [
          "a166c4f5-7cc4-429b-b9d8-2c8c43facc63"
        ],
        "dependson": [
          "1790ebeb-2e41-4293-8cc1-aaba134fd1e0"
        ],
        "isrelatedto": [
          "a166c4f5-7cc4-429b-b9d8-2c8c43facc63"
        ]
      },
      "3df467ad-673c-4c48-a41c-3ac1626961e3": {
        "size": {
          "width": 120,
          "height": 120
        },
        "position": {
          "x": 250,
          "y": 230
        },
        "z": 0,
        "embeds": [
          "0f900c9e-1272-4ec2-8a42-790b074baa39"
        ]
      },
      "0f900c9e-1272-4ec2-8a42-790b074baa39": {
        "size": {
          "width": 60,
          "height": 60
        },
        "position": {
          "x": 280,
          "y": 260
        },
        "z": 3,
        "parent": "3df467ad-673c-4c48-a41c-3ac1626961e3",
        "embeds": [],
        "isrelatedto": [
          "2e76192b-a4f8-48a5-92b6-abbfa8b83263"
        ]
      },
      "13e0e0da-40c9-45d0-8460-7732ed20d764": {
        "source": {
          "id": "96a791f0-938b-4ebe-9f3c-b3fe2a588aee"
        },
        "target": {
          "id": "3df467ad-673c-4c48-a41c-3ac1626961e3"
        },
        "z": 2
      },
      "528e2b71-46e6-4e09-815a-f70630755219": {
        "source": {"id": "175bad80-0988-4588-a919-331be705b02d"},
        "target": {"id": "3df467ad-673c-4c48-a41c-3ac1626961e3"},
        "z": 2
      }
    }
  }
}
```
