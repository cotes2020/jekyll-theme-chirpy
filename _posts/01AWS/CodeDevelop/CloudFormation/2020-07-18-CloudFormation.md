---
title: AWS - CodeDevelop - CloudFormation
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, CodeDevelop]
tags: [AWS]
toc: true
image:
---

[toc]

---

#  AWS CloudFormation

> Infrastructure as code solution.

![Screen Shot 2021-01-18 at 16.28.28](https://i.imgur.com/nt8tRpD.png)


---

## benefits

1. <font color=red> manage, configure and provision the AWS infrastructure as code </font>
   - repeatedly and predictably model and provision resources
   - infrastructure is provisioned consistently
     - fewer mistakes
     - less time and effort than configure manually

2. <font color=red> Supports almost all the AWS services and programmable </font>
   - provision a broad range of AWS resources.
     - <font color=blue> compare </font>
     - <font color=blue> Elastic Beanstalk </font>
       - more focused on deploying web applications on EC2 
       - PaaS
     - <font color=blue> CloudFormation </font>
       - can deploy Elastic Beanstalk-hosted applications
       - however the reverse is not possible.

3. resources are <font color=red> defined by CloudFormation template </font>
   - Supports YAML and JSON
     - Logical IDs: reference resources within the template.
     - Physical IDs: identify resources outside of AWS CloudFormation templates, but only after the resources have been created.

4. CloudFormation <font color=red> interprets the template and makes the appropriate API calls </font> to create the resources defined.

5. <font color=red> version control </font> and peer review the templates
   - can be used to manage updates & dependencies
   - can be used to rollback and delete the entire stack as well


6. AWS CloudFormation provides 2 methods for <font color=red> updating stacks </font>
   - <font color=red> direct update a stack </font>
     - submit changes
     - AWS CloudFormation immediately deploys them.
     - Use direct updates to quickly deploy the updates.
   - creating and executing <font color=red> change sets </font>
     - preview the changes AWS CloudFormation will make to your stack
     - and then decide whether to apply those changes.


6. free service (resources created would be charged)



---


## templates, stacks and change sets:

![CloudFormation](https://i.imgur.com/zu1yJYA.png)

![Pasted Graphic](https://i.imgur.com/71RbCIM.jpg)

---


### Template

template is used to <font color=red> describe the endstate of the infrastructure either provisioning or changing </font>
1. after created, upload it to CloudFormation using S3
2. CloudFormation reads the template and makes the API calls
3. the resulting resources are called a <font color=red> Stack </font>

CloudFormation template
- create templates to launch, configure, and connect AWS resources
  - standard templates for deploying infrastructure
- template can be stored in CodeCommit
  - maintain a history of the template and the infrastructure that has been deployed.

- CloudFormation determines the order of provisioning.
  - easy way to create a collection of related AWS resources and provision them in an orderly and predictable fashion.
  - Don’t need to worry about dependencies.
  - Architectural designs.

- Treat it as code
  - manage it by using version control, such as Gitor Subversion
  - Create, update and delete templates.

- is the <font color=red> single source of truth </font> for cloud environment.
  - Define an entire application stack (all the resources required for application) in a JSON  template file.
  - provides a common language to to model, describe and provision all the infrastructure resources and properties in the cloud environment.
  - model and provision in an automated and secure manner all the resources needed for your applications across all regions and accounts.


- Define runtime parameters for a template
  - such as EC2—Instance Size, Amazon EC2 Key Pair, etc.




Templates can be created by
- code editor supports JSON syntax, (Atom or Sublime Text)
- third party WYSIWYG editor
- build visually by CloudFormation Designer tool
  - Available in AWS Management Console
  - allows visualize using a drag and drop interface.
  - drag and drop resources onto a design area to automatically generate a JSON/YAML-formatted CloudFormation template
  - edite properties of the JSON or YAML template on same page.
  - can open and edite Existing CloudFormation templates

YAML or JSON
- JavaScript Object Notation (JSON) / YAML-formatted templates.
- Both YAML/JSON-formatted templates have the same structure, support all the same feature

<font color=red> do not recommend </font>
- build all of an application's within one template
  - Resources should be grouped into templates
  - based on the ownership and the place in the application lifecycle.
  - minimum should separate network, security, and application resources into own templates.
- test environment and production environment should not share same templates.
  - Resources in a test environment need to change frequently
  - resources in a production environment should be relatively stable.
- sharing templates across management teams
  - because different needs and standards can impact teams inappropriately.



Organizing template:
- Avoid sharing a single template across applications for resources of the same type
  - unless you are deliberately centralizing control of that resource type.
  - no too many things inside of one template across numerous applications.
  - application template that supports several applications,
    - changes to the template will affect several applications
    - changes can cause all of the applications to be retested.
- share template could potentially break
  - things that are specific to your environment,
    - such as Amazon EC2 key pairs, security group names, subnet IDs, and EBS—snapshot IDs.
  - It can be fixed by using parameters, mappings, and condition section in temple.
- storing templates contain security resources in a separate repository from other templates.



Template elements:
- Mandatory:
  - File format and version.
  - List of resources and associated configuration values.
- Not Mandatory:
  - Template parameters (limited to 60).
  - Output values (limited to 60).
  - List of data tables.


---


### Engine:
- Aws service component
- Interprets AWS cloudFormation template into stacks of AWS resources.

---


### Group:

![Screen Shot 2020-06-26 at 10.10.37](https://i.imgur.com/k3VHJRq.png)

- allows you to quickly provision a test environment to investigate possible breaches into your EC2 instance.
- Puppet and Chef integration is supported.
- Can use bootstrap scripts.
- Can define deletion policies.
- Can create roles in IAM.
- VPCs can be created and customized.
- VPC peering in the same AWS account can be performed.
- Route 53 is supported.



---


### Stack


---

#### Stack

1. <font color=red> A collection of resources </font> created by AWS cloudFormation templates
   - All the resources in a stack are defined by the stack's `AWS CloudFormation template`.
     - Deployed resources based on templates.
     - Create, update and delete stacks using templates.
     - Deployed through the Management Console, CLI or APIs.
     - Tracked and reviewable in the AWS management console

2. a collection of AWS resources that can <font color=red> manage as a single unit </font>
   - AWS CloudFormation treats the stack resources as a single unit
   - create, update, or delete a collection of resources by <font color=blue> creating, updating, or deleting stacks </font>
   - Example:
     - A stack can include all the resources required to run a web application
       - such as a web server, a database, and networking rules.
     - If no longer require that web application, simply delete the stack, and all of its related resources are deleted.

3. AWS CloudFormation ensures all `stack resources` are created or deleted as appropriate.
   - If a resource cannot be created, AWS CloudFormation rolls the stack back and automatically deletes any resources that were created.
   - If a resource cannot be deleted, any remaining resources are retained until the stack can be successfully deleted.
   - can work with stacks by using the AWS CloudFormation console, API, or AWS CLI.

4. Stacks can't create logical resources
   - stack listens to the direction of a template containing logical resources.
     - template describes a stack,
       - a collection of AWS resources want to deploy together as a group.
     - stack manages physical resources based on a logical resource template
     - stack making resources through the direction of a template.

   - Actions to resources can be tracked in CloudFormation's stack details.

5. Stacks control resources
   - if a stack is removed, the resource also be deleted, so are the resources that it created.
   - Stacks could disrupt a resource when performing updates to the resource.


7. Stack creation errors:
   - Automatic rollback on error is enabled by default.
   - will be charged for resources provisioned even if there is an error.



---



#### Nested stacks

1. Nested stacks are <font color=red> stacks created as part of other stacks </font>

2. Use nested stacks <font color=red> to declare common components (best practice) </font>
   - allow <font color=blue> re-use of CloudFormation code for common use cases </font>
     - such as standard configuration for load balancer, web server, application server, etc.

   - As infrastructure grows, declare the same components in multiple templates
     - separate out these common components and create a standard dedicated templates for each common use case
     - store it in S3
     - and refenrece it in the Resources section of other template using the Stack resource type
     - `Resources: Type: AWS::CloudFormation::Stack`

   - Example:
     - a load balancer configuration that use for most of the stacks.
     - Instead of copying and pasting the same configurations into the templates,
     - create a dedicated template for the load balancer.
     - Then, just use the resource to reference that template from within other templates.

3. Nested stacks can contain other nested stacks,
   - resulting in a <font color=blue> hierarchy of stacks </font>

4. Certain stack operations, such as stack updates, should be initiated from the `root stack` rather than performed directly on `nested stacks themselves`.



---

Example:

```
Resources:
  Type: AWS::CloudFormation::Stack
  Properties:
    NotificationARNs:
      - String
    Parameters:
      AWS CloudFormation Stack Parameters
    Tags:
      - Resource Tag
    TemplateURL: https://s3.amazonaws.com/.../template.yml
    TimeoutInMinutes: Integer
```


<img src="https://i.imgur.com/TU6BesT.png" width="200">

The <font color=red> root stack </font>
- the top-level stack to which all the nested stacks ultimately belong.
- each nested stack has an immediate parent stack.
- For the first level of nested stacks, the root stack is also the parent stack.

> Stack A is the `root stack` for all the other, nested, stacks in the hierarchy.
> For stack B, stack A is both the `parent stack`, as well as the `root stack`.
> For stack D, stack C is the `parent stack`;
> for stack C, stack B is the `parent stack`.



1. <kbd>AWS Management Console</kbd> -> <kbd>AWS CloudFormation console</kbd> -> Select the <kbd>stack</kbd>
2. Nested stacks display `NESTED` next to the stack name.
3. <font color=red> To view the root stack of a nested stack </font>
   - <kbd>Overview tab</kbd>: click the stack name listed as `Root stack`.
4. <font color=red> To view the nested stacks that belong to a root stack </font>
   - <kbd>AWS CloudFormation console</kbd> -> Click the name of the root stack whose nested stacks want to view.
   - Expand the <kbd>Resources</kbd> section.
   - Look for resources of type `AWS::CloudFormation::Stack`.


---



#### Cross stack references
- share outputs from one stack with another stack.
  - share things like IAM—roles, VPC—information, and security groups.
  - Before, use AWS CloudFormation custom resources to accomplish these tasks.
  - Now, export values from one stack and import them to another stack by using the new ImportValueintrinsic function.
- useful for customers who
  - separate their AWS infrastructure into logical components that grouped by stack
    - such as a network stack, an application stack, etc.
  - need a way to loosely couple stacks together as an alternative to nested stacks

---

### StackSets.

AWS CloudFormation StackSets

- extends the functionality of stacks by enabling <font color=red> create, update, or delete stacks across multiple accounts and regions with a single operation </font>

- An administrator account
  - the AWS account in which you create stack sets.
  - define and manage an AWS CloudFormation template
  - use the template as the basis for provisioning stacks into selected target accounts across specified regions.
  - A stack set is managed by signing in to the AWS administrator account in which it was created.

- A target account
  - the account into which you create, update, or delete one or more stacks in your stack set.
  - Before use a stack set to create stacks in a target account, must set up a trust relationship between the administrator and target accounts.

---


## Best Practices.
- AWS provides Python “helper scripts” which can help you install software and start services on your EC2 instances.
- Use CloudFormation to make changes to your landscape rather than going directly into the resources.
- Make use of Change Sets to identify potential trouble spots in your updates.
- Use Stack Policies to explicitly protect sensitive portions of your stack.
- Use a version control system such as CodeCommit or GitHub to track changes to templates.



---


## Charges:
- no additional charge for AWS CloudFormation.
- pay for AWS resources (such as EC2 instances, ELB load balancers, etc.) created using AWS CloudFormation in the same manner as if you created them manually.
- only pay for what you use, as you use it;
- there are no minimum fees and no required upfront commitments.


---


# setup

1. cloudformation
2. create stack
   1. select template
   2. stack name
   3. keypaire
   4. rollback on failure
3. delete stack


---


# CloudFormationTemplate.yml

```yml
AWSTemplateFormatVersion: 2010-09-09

# text string that describes the template
Description: Template to create an EC2 instance and enable SSH



# data about the data, Some AWS CloudFormation features retrieve settings or configuration information that you define from the Metadata section.
Metadata:



# input custom values, pass the value of your template at runtime.
Parameters:
  KeyName:
    Description: Name of SSH KeyPair
    Type: 'AWS::EC2::KeyPair::KeyName'
    ConstraintDescription: Provide the name of an existing SSH key pair
  InstanceTypeParameter:
    Type: String
    Default: t2.micro
    AllowedValues: ["t2.micro", "m1.small", "m1.large"]
    Description: 'Enter t2.micro, m1.small or m1.large'



# provision resources based on environment
Conditions:



# Mandatory
# the AWS resource be included / created in the stack
Resources:
  # Logical ID:
  #   Type: 'ARNs'
  #   Properties:
  MyEC2Instance:
    Type: 'AWS::EC2::Instance'
    Properties:
      # InstanceType: t2.micro
      InstanceType: ('Ref': InstanceTypeParameter)
      ImageId: ami-0bdb1d6c15a40392c
      KeyName: !Ref KeyName
      SecurityGroups:
       - Ref: InstanceSecurityGroup
      Tags:
        - Key: Name
          Value: My CF Instance
      # How AWS CloudFormation should wait to launch a resource
      # until a specific, different resource has finished being created.
      DependsOn: myDB
  InstanceSecurityGroup:
    Type: 'AWS::EC2::SecurityGroup'
    Properties:
      GroupDescription: Enable SSH access via port 22
      SecurityGroupIngress:
        IpProtocol: tcp
        FromPort: 22
        ToPort: 22
        CidrIp: 0.0.0.0/0


# create custom mappings
# like different Region for different AMI
# customize the properties of a resource based on certain conditions, which enables you to have fine-grained control over how your templates are launched.
Mappings:
  RegionMap:
    us-east-1: (t2.micro: ami-0bdb1d6c15a40392c)
    us-west-1: (t2.micro: ami-0bdb1d6c15a40392c)

# reference code located in S3
# Lambda code or reusable snippets of CloudFormation code
Transforms:


# values that are returned whenever you view the properties of your stack.
Outputs:
  InstanceID:
    Description: The Instance ID
    Value: !Ref MyEC2Instance
```




---

# template Section


---

## Intrinsic function reference

### Ref
- provided `logical ID of this resource` to the Ref intrinsic function,
- Ref returns `the resource name`.

```yaml
{ "Ref": "RootRole" }
# Ref will return the role name for the AWS::IAM::Role resource with the logical ID "RootRole"


MyEIP:
  Type: "AWS::EC2::EIP"
  Properties:
    InstanceId: !Ref MyEC2Instance
```



### Fn::GetAtt
- returns the value of an attribute from a resource in the template
- returns a value for a specified attribute of this type.
- The following are the available attributes and sample return values.

```yaml
{"Fn::GetAtt" : ["MyRole", "Arn"] }
# Returns the Amazon Resource Name (ARN) for the role.
# This will return a value such as arn:aws:iam::1234567890:role/MyRole-AJJHDSKSDF.

{"Fn::GetAtt" : ["MyRole", "RoleId"] }
# Returns the stable and unique string identifying the role. For example, AIDAJQABLZS4A3QDU576Q.

!GetAtt myELB.DNSName
# returns a string containing the DNS name of the load balancer with the logical name myELB.


AWSTemplateFormatVersion: 2010-09-09
Resources:

  myELB:
    Type: AWS::ElasticLoadBalancing::LoadBalancer
    Properties:
      AvailabilityZones: eu-west-1a
      Listeners:
        - LoadBalancerPort: '80'
          InstancePort: '80'
          Protocol: HTTP

  myELBIngressGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: ELB ingress group
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          SourceSecurityGroupOwnerId: !GetAtt myELB.SourceSecurityGroup.OwnerAlias
          SourceSecurityGroupName: !GetAtt myELB.SourceSecurityGroup.GroupName

```


### Fn::Sub

- The intrinsic function `Fn::Sub` substitutes variables in an input string with values that you specify.
- In your templates, you can use this function to construct commands or outputs that include values that aren't available until you create or update a stack.

```yaml
# Fn::Sub with a mapping
# uses a mapping to substitute the ${Domain} variable with the resulting value from the Ref function.
Name: !Sub
  - www.${Domain}
  - { Domain: !Ref RootDomainName }


# Fn::Sub without a mapping
# uses Fn::Sub with the AWS::Region and AWS::AccountId pseudo parameters and the vpc resource logical ID to create an Amazon Resource Name (ARN) for a VPC.
!Sub 'arn:aws:ec2:${AWS::Region}:${AWS::AccountId}:vpc/${vpc}'


# UserData commands
# The following example uses Fn::Sub to substitute the AWS::StackName and AWS::Region pseudo parameters for the actual stack name and region at runtime.
UserData:
  Fn::Base64:
    !Sub |
      #!/bin/bash -xe
      yum update -y aws-cfn-bootstrap
      /opt/aws/bin/cfn-init -v --stack ${AWS::StackName} --resource LaunchConfig --configsets wordpress_install --region ${AWS::Region}
      /opt/aws/bin/cfn-signal -e $? --stack ${AWS::StackName} --resource WebServerGroup --region ${AWS::Region}


```



---

## Format version

The AWSTemplateFormatVersion section (optional) identifies the capabilities of the template.
- The latest template format version is 2010-09-09 and is currently the only valid value.

```yaml
AWSTemplateFormatVersion: "2010-09-09"
```

---

## Description: (optional)
￼
![Screen Shot 2020-06-26 at 10.13.23](https://i.imgur.com/ZcU1VMh.png)

The Description section include comments about your template.

```yaml
Description: >
  Here are some
  details about
  the template.
```

---



## Parameter : to pass the value of the template at runtime

Use the optional `Parameters` section to customize your templates.
- to input custom values to your template each time you create or update a stack.
- can specify allowed and default values for each parameter.
  - specify details like
  - the range of acceptable AMI ImageIdnumbers,
  - key pairs,
  - subnets,
  - or any properties that must be specified for a resource.
- A parameter contains a list of attributes that define its value, and constraints against its value.


### example

```json
"Parameters" : {
  "ParameterLogicalID" : {
    "Type" : "DataType",
    "ParameterPropertyABCD" : "value"
  }
}

"Parameters" : {
    "InstanceTypeParameter" : {
        "Type" : "String",
        "Description" : "Enter t2.micro, m1.small, m1.large. Default is t2.micro",
        "Default" : "t2.micro",
        "AllowedValues" : [ "t2.micro", "m1.small", "m1.large"]
        // appears in the AWS CloudFormationConsole when the template is launched.
    }
},
"Resources" : {
    // when an EC2 instance is launched in the Resources section
    "Instances" : {
        "Type" : "AWS::EC2::Instance",
        "Properties" : {
            // the Properties section of the instance can reference the InstanceTypeParameter specification.
            // the "Ec2Instance" resource references the InstanceTypeParameter specification for its instancetype.
            "InstanceType" : { "Ref" : "InstanceTypeParameter" },
            "ImageId" : "ami-20b65349",
        }
    }
}
```

```yaml
Parameters:
  ParameterLogicalID:
    Type: DataType
    ParameterPropertyABCD: value

Parameters:
  InstanceTypeParameter:
    Type: String
    Default: t2.micro
    AllowedValues:
      - t2.micro
      - m1.small
      - m1.large
    Description: Enter t2.micro, m1.small, or m1.large. Default is t2.micro.

Resources:
  Ec2Instance:
    Type: AWS::EC2::Instance
    Properties:
      InstanceType:
        Ref: InstanceTypeParameter
      ImageId: ami-0ff8a91507f77f867
```



### Type

The only required attribute
- `String`: "MyUserName"
- `Number`: "8888"
- `List<Number>`: ["80","20"]

```yaml
Parameters:
  DBPort:
    Default: 3306
    Description: TCP/IP port for the database
    Type: Number
    MinValue: 1150
    MaxValue: 65535
  DBPwd:
    NoEcho: true
    Description: The database admin account password
    Type: String
    MinLength: 1
    MaxLength: 41
    AllowedPattern: ^[a-zA-Z0-9]*$
```

- `CommaDelimitedList`: ["test","dev","prod"]
  - to specify multiple string values in a single parameter.
  - can use a single parameter instead of many different parameters to specify multiple values.
  - For example
    - create three different subnets with their own CIDR blocks,
    - use three different parameters to specify three different CIDR blocks.
    - But it's simpler just to use a single parameter that takes a list of three CIDR blocks
  - To refer to a specific value in a list
    - use the `Fn::Select` intrinsic function in the Resources section of your template.
    - pass the index value of the object that you want and a list of objects


```yaml
Parameters:
  DbSubnetIpBlocks:
    Description: "Comma-delimited list of three CIDR blocks"
    Type: CommaDelimitedList
    Default: "10.0.48.0/24, 10.0.112.0/24, 10.0.176.0/24"

Resources:

  DbSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      AvailabilityZone: !Sub
        - "${AWS::Region}${AZ}"
        - AZ: !Select [0, !Ref VpcAzs]
      VpcId: !Ref VPC
      CidrBlock: !Select [0, !Ref DbSubnetIpBlocks]

  DbSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      AvailabilityZone: !Sub
        - "${AWS::Region}${AZ}"
        - AZ: !Select [1, !Ref VpcAzs]
      VpcId: !Ref VPC
      CidrBlock: !Select [1, !Ref DbSubnetIpBlocks]

  DbSubnet3:
    Type: AWS::EC2::Subnet
    Properties:
      AvailabilityZone: !Sub
        - "${AWS::Region}${AZ}"
        - AZ: !Select [2, !Ref VpcAzs]
      VpcId: !Ref VPC
      CidrBlock: !Select [2, !Ref DbSubnetIpBlocks]
```

- `SSM Parameter` Types:
  - Parameters that correspond to existing parameters in Systems Manager Parameter Store. You specify a Systems Manager parameter key as the value of the SSM parameter, and AWS CloudFormation fetches the latest value from Parameter Store to use for the stack. For more information, see SSM parameter types.


- `AWS-Specific Parameter` Types:
  - AWS-specific
  - catching invalid values at the start of creating or updating a stack.
  - to create or update a stack <font color=blue> must specify existing AWS values that are in the user's account and in the region for the current stack. </font>
  - help ensure that input values for these types exist and are correct before AWS CloudFormation creates or updates any resources.
  - If a user uses the AWS Management Console,
    - AWS CloudFormation prepopulates AWS-specific parameter types with valid values.
    - user doesn't have to remember and correctly enter a specific name or ID.
      - just select from a drop-down list.
      - can search for values by ID, name, or Name tag value.
  - For example
    - AWS values such as Amazon EC2 key pair names and VPC IDs.
    - use the `AWS::EC2::KeyPair::KeyName` parameter type,
    - AWS CloudFormation validates the input value against users' existing key pair names before it creates any resources, such as Amazon EC2 instances.

    - The following example declares two parameters with the types `AWS::EC2::KeyPair::KeyName` and `AWS::EC2::Subnet::Id`.
    - These types limit valid values to existing key pair names and subnet IDs.
    - Because the mySubnetIDs parameter is specified as a list, a user can specify one or more subnet

```yaml
Parameters:
  myKeyPair:
    Description: Amazon EC2 Key Pair
    Type: "AWS::EC2::KeyPair::KeyName"
  mySubnetIDs:
    Description: Subnet IDs
    Type: "List<AWS::EC2::Subnet::Id>"
```




#### AWS-specific parameter type

- Supported AWS-specific parameter types

  - `AWS::EC2::AvailabilityZone::Name`
  - An Availability Zone
  - such as `us-west-2a`.

  - `AWS::EC2::Image::Id`
  - An Amazon EC2 image ID,
  - such as `ami-0ff8a91507f77f867`.
  - Note that the AWS CloudFormation console doesn't show a drop-down list of values for this parameter type.

  - `AWS::EC2::Instance::Id`
  - An Amazon EC2 instance ID,
  - such as `i-1e731a32`

  - `AWS::EC2::KeyPair::KeyName`
  - An Amazon EC2 key pair name.

  - `AWS::EC2::SecurityGroup::GroupName`
  - An EC2-Classic or default VPC security group name,
  - such as `my-sg-abc`.

  - `AWS::EC2::SecurityGroup::Id`
  - A security group ID,
  - such as `sg-a123fd85`.

  - `AWS::EC2::Subnet::Id`
  - A subnet ID,
  - such as `subnet-123a351e`.

  - `AWS::EC2::Volume::Id`
  - An Amazon EBS volume ID,
  - such as `vol-3cdd3f56`.

  - `AWS::EC2::VPC::Id`
  - A VPC ID,
  - such as `vpc-a123baa3`

  - `AWS::Route53::HostedZone::Id`
  - An Amazon Route 53 hosted zone ID,
  - such as `Z23YXV4OVPL04A`

  - `List<AWS::EC2::AvailabilityZone::Name>`
  - An array of Availability Zones for a region,
  - such as `us-west-2a, us-west-2b`

  - `List<AWS::EC2::Image::Id>`
  - An array of Amazon EC2 image IDs,
  - such as `ami-0ff8a91507f77f867, ami-0a584ac55a7631c0c`.
  - Note that the AWS CloudFormation console doesn't show a drop-down list of values for this parameter type.

  - `List<AWS::EC2::Instance::Id>`
  - An array of Amazon EC2 instance IDs,
  - such as `i-1e731a32, i-1e731a34`

  - `List<AWS::EC2::SecurityGroup::GroupName>`
  - An array of EC2-Classic or default VPC security group names,
  - such as `my-sg-abc, my-sg-def`

  - `List<AWS::EC2::SecurityGroup::Id>`
  - An array of security group IDs,
  - such as `sg-a123fd85, sg-b456fd85`

  - `List<AWS::EC2::Subnet::Id>`
  - An array of subnet IDs,
  - such as `subnet-123a351e, subnet-456b351e`

  - `List<AWS::EC2::Volume::Id>`
  - An array of Amazon EBS volume IDs,
  - such as `vol-3cdd3f56, vol-4cdd3f56`

  - `List<AWS::EC2::VPC::Id>`
  - An array of VPC IDs,
  - such as `vpc-a123baa3, vpc-b456baa3`

  - `List<AWS::Route53::HostedZone::Id>`
  - An array of Amazon Route 53 hosted zone IDs,
  - such as `Z23YXV4OVPL04A, Z23YXV4OVPL04B`



### Description
- value they should specify.
- The parameter's name and description appear in the Specify Parameters page when a user uses the template in the Create Stack wizard
- A string of up to 4000 characters that describes the parameter.

### AllowedPattern
- A regular expression that represents the patterns to allow for String types. The pattern must match the entire parameter value provided.

### AllowedValues
- An array containing the list of values allowed for the parameter.

### Default
- A value of the appropriate type for the template to use if no value is specified when a stack is created.
- If you define constraints for the parameter, you must specify a value that adheres to those constraints.
### ConstraintDescription

- A string that explains a constraint when the constraint is violated.
- For example, without a constraint description, a parameter that has an allowed pattern of [A-Za-z0-9]+ displays the following error message when the user specifies an invalid value: `Malformed input-Parameter MyParameter must match pattern [A-Za-z0-9]+`
- By adding a constraint description, such as must only contain letters (uppercase and lowercase) and numbers, you can display the following customized error message: `Malformed input-Parameter MyParameter must only contain uppercase and lowercase letters and numbers`
### MaxLength

- An integer value that determines the largest number of characters you want to allow for String types.
### MaxValue

- A numeric value that determines the largest numeric value you want to allow for Number types.
### MinLength

- An integer value that determines the smallest number of characters you want to allow for String types.
### MinValue

- A numeric value that determines the smallest numeric value you want to allow for Number types.
### NoEcho

- Whether to mask the parameter value to prevent it from being displayed in the console, command line tools, or API. If you set the NoEcho attribute to true, CloudFormation returns the parameter value masked as asterisks (*****) for any calls that describe the stack or stack events, except for information stored in the locations specified below.

> General requirements for parameters
> - maximum of 200 parameters in an AWS CloudFormation template.
> - Each parameter must be given a logical name (also called logical ID), unique among all logical names within the template.
> - Each parameter must be assigned a parameter type that is supported by AWS CloudFormation.
> - Each parameter must be assigned a value at runtime for AWS CloudFormation to successfully provision the stack. You can optionally specify a default value for AWS CloudFormation to use unless another value is provided.
> - Parameters must be declared and referenced from within the same template. You can reference parameters from the Resources and Outputs sections of the template.


use the `Ref` intrinsic function to reference a parameter, and AWS CloudFormation uses the parameter's value to provision the stack.
- You can reference parameters from the `Resources` and `Outputs` sections of the same template.

---



## Conditions section : includes statements, control (optional)

![Screen Shot 2020-06-26 at 13.25.07](https://i.imgur.com/wr4a92A.png)

- The optional Conditions section contains statements that define the circumstances under which entities are created or configured.
- whether certain resources are created, or certain properties are assigned a value during the creation or update of a stack.
  - can compare whether a value is equal to another value.
  - Based on the result of that condition, conditionally create resources.
  - If multiple conditions, separate them with commas.


- use conditions when
  - reuse a template that can create resources in different contexts,
    - such as a test environment vs a production environment.
    - In template, add an EnvironmentType input parameter, which accepts either “prod” or “test” as inputs.
      - For the production environment,
        - include EC2 instances with certain capabilities;
      - for the test environment,
        - use reduced capabilities to save money.
    - define which resources are created, and how they're configured for each environment type.



At stack creation or stack update,
- AWS CloudFormation evaluates all the conditions in template <font color=blue> before creating any resources </font>
  - Resources that are associated with a true condition are created.
  - Resources that are associated with a false condition are ignored.
- AWS CloudFormation also re-evaluates these conditions at each stack update before <font color=blue> updating any resources </font>
  - Resources that are still associated with a true condition are updated.
  - Resources that are now associated with a false condition are deleted.


- Conditions are evaluated based on input parameter values specified when create or update a stack.
  - if values or tags have been assigned,
  - the template will do something different based on the assigned value.

- Within each condition, you can reference another condition, a parameter value, or a mapping.
  - After define all conditions,
  - associate them with `resources` and `resource properties` in the `Resources` and `Outputs` sections of a template.

- For example
  - can create a condition and then associate it with a `resource` or `output`
    - AWS CloudFormation only creates the resource or output if the condition is true.
  - can associate the condition with a `property`
    - AWS CloudFormation only sets the property to a specific value if the condition is true.
    - If the condition is false, AWS CloudFormation sets the property to a different value that you specify.

1. Parameters section
   - Define the inputs that you want your conditions to evaluate.
   - The conditions evaluate to true or false based on the values of these input parameters.
   - If you want your conditions to evaluate pseudo parameters, you don't need to define the pseudo parameters in this section; pseudo parameters are predefined by AWS CloudFormation.

2. Conditions section
   - Define conditions by using the intrinsic condition functions.
   - These conditions determine when AWS CloudFormation creates the associated resources.

3. Resources and Outputs sections
   - Associate conditions with the resources or outputs that you want to conditionally create.
   - AWS CloudFormation creates entities that are associated with a true condition and ignores entities that are associated with a false condition.
   - Use the Condition key and a condition's logical ID to associate it with a resource or output.
   - To conditionally specify a property, use the Fn::If function. For more information, see Condition functions.


### Condition intrinsic functions
- You can use the following intrinsic functions to define conditions:

```yaml
Fn::And
Fn::Equals
Fn::If
Fn::Not
Fn::Or
```


### Examples

```json
"Conditions" : {
  "Logical ID" : {Intrinsic function}
}


"Parameters" : {
    "InstanceTypeParameter" : {
        "Type" : "String",
        "Default" : "t2.micro",
        "AllowedValues" : [ "t2.micro", "m1.small", "m1.large"],
        "Description" : "Enter t2.micro, m1.small, m1.large. Default is t2.micro"
    },
    "EnvType" : {
        "Type" : "String",
        "Default" : "Dev",
        "AllowedValues" : [ "Dev", "QA", "Prod"],
        "Description" : "Enter the environment"
    },
},

"Resources" : {
    "Instances" : {
        "Type" : "AWS::EC2::Instance",
        "Properties" : {
            "InstanceType" : { "Ref" : "InstanceTypeParameter" },
            "ImageId" : "ami-20b65349",
        }
    }
},

// use “Condition” to evaluate this, and specify appropriate resources for each environment.
"Conditions" : {
    "CreateProdResources" : {
      "Fn::Equals" : [{ "Ref" : "EnvType"}, "Prod" ]
    }
}
```


```yaml
AWSTemplateFormatVersion: 2010-09-09
Parameters:
  EnvType:
    Description: Environment type.
    Default: test
    Type: String
    AllowedValues: [prod, test]
    ConstraintDescription: must specify prod or test.

# setup condition
Conditions:
  CreateProdResources: !Equals [!Ref EnvType, prod]

Resources:
  EC2Instance:
    Type: 'AWS::EC2::Instance'
    Properties:
      ImageId: ami-0ff8a91507f77f867

  MountPoint:
    Type: 'AWS::EC2::VolumeAttachment'
    # use the condition
    Condition: CreateProdResources
    Properties:
      InstanceId: !Ref EC2Instance
      VolumeId: !Ref NewVolume
      Device: /dev/sdh

  NewVolume:
    Type: 'AWS::EC2::Volume'
    # use the condition
    Condition: CreateProdResources
    Properties:
      Size: 100
      AvailabilityZone: !GetAtt
        - EC2Instance
        - AvailabilityZone
```


### Nested condition
The following sample template references a condition within another condition.
- create a stack that creates an s3 bucket.
- For a stack deployed in a production environment, AWS CloudFormation creates a policy for the S3 bucket.

```yaml
Parameters:
  EnvType:
    Type: String
    AllowedValues: [prod, test]
  BucketName:
    Default: ''
    Type: String

Conditions:
  IsProduction: !Equals [!Ref EnvType, prod]

  CreateBucket: !Not
    - !Equals [!Ref BucketName, '']

  CreateBucketPolicy: !And
    - !Condition IsProduction
    - !Condition CreateBucket

Resources:
  Bucket:
    Type: 'AWS::S3::Bucket'
    # use the condition, true than got the bucket
    Condition: CreateBucket

  Policy:
    Type: 'AWS::S3::BucketPolicy'
    # use the condition, true than got the policy
    Condition: CreateBucketPolicy
    Properties:
      Bucket: !Ref Bucket
      PolicyDocument: ...
```





### Example

![Screen Shot 2020-06-26 at 15.06.07](https://i.imgur.com/YdjU8aF.png)

- the EnvType parameter specifies whether to create a Dev environment, a QA—environment, or a Prod environment.
- Depending on the environment, to specify different configurations, such as which database it points to.
- use “Condition” to evaluate this, and specify appropriate resources for each environment.

- Build environment with conditions:
  - when the target environment is development DEV.
    - only one set of resources in one Availability Zone is launched
  - When this template is used in production PROD
    - the solution launches two sets of resources in two different AZ.
  - get a redundant environment from the same template without single change

- production environment and DEV environment
  - must have the same stack
  - in order to ensure that application works the way that it was designed.

- DEV environment and QA environment
  - must have the same stack of applications and the same configuration.
  - You might have several QA environments for functional testing, user acceptance testing, load testing, and so on.
  - The process of creating those environments manually can be -prone.
  - use a Conditions statement in the template to solve this problem.


---

## Mapping

- atches a key to a corresponding set of named values.

- Example
  - set values based on a region,
  - create a mapping
    - uses the region name as a key
    - and contains the values you want to specify for each specific region.
  - use the `Fn::FindInMap` intrinsic function to retrieve values in a map.

- You cannot include parameters, pseudo parameters, or intrinsic functions in the Mappings section.


### example: mapping


```yaml

Mappings:
  Mapping01:
    Key01:
      Name: Value01
    Key02:
      Name: Value02
    Key03:
      Name: Value03


AWSTemplateFormatVersion: "2010-09-09"

Mappings:  # section
  RegionMap:  # map ID
    us-east-1:  # Key
      "HVM64": "ami-0ff8a91507f77f867"   # Name: Value
    us-west-1:
      "HVM64": "ami-0bdb828fd58c52235"
    eu-west-1:
      HVM64: ami-047bb4163c506cd98
      HVMG2: ami-0a7c483d527806435
    ap-northeast-1:
      HVM64: ami-06cd52961ce9f0d85
      HVMG2: ami-053cdd503598e4a9d
    ap-southeast-1:
      HVM64: ami-08569b978cc4dfa10
      HVMG2: ami-0be9df32ae9f92309

Resources:
  myEC2Instance:
    Type: "AWS::EC2::Instance"
    Properties:
      ImageId: !FindInMap [RegionMap, !Ref "AWS::Region", HVM64]
      #        !FindInMap [mapID, !Ref keyID, ValueName]
      InstanceType: m1.small

```

### Input parameter and FindInMap

- use an input parameter with the `Fn::FindInMap` function to refer to a specific value in a map.
- example
  - have a list of regions and environment types that map to a specific AMI ID.
  - select the AMI ID that your stack uses by using an input parameter (EnvironmentType).
  - To determine the region, use the `AWS::Region` pseudo parameter, which gets the AWS region in which you create the stack.

```yaml
  Parameters:
    EnvironmentType:
      Description: The environment type
      Type: String
      Default: test
      AllowedValues: [prod, test]
      ConstraintDescription: must be a prod or test

  Mappings:
    RegionAndInstanceTypeToAMIID:
      us-east-1:
        test: "ami-8ff710e2"
        prod: "ami-f5f41398"
      us-west-2:
        test: "ami-eff1028f"
        prod: "ami-d0f506b0"

  Resources:
   ...other resources...

  Outputs:
    TestOutput:
      Description: Return the name of the AMI ID that matches the region and environment type keys
      Value: !FindInMap [RegionAndInstanceTypeToAMIID, !Ref "AWS::Region", !Ref EnvironmentType]
      #      !FindInMap [which map ID, !Ref which key, !Ref EnvironmentType]
```


---



## Metadata: data about the data
  ￼
![Screen Shot 2020-06-26 at 10.44.06](https://i.imgur.com/xGuoh3a.png)

- Some AWS CloudFormation features retrieve settings or configuration information defined from the Metadata section.

- Define in the AWS CloudFormation-specific metadata keys:
  - ``AWS::CloudFormation::Init``
    - Defines configuration tasks for the cfn-init helper script.
    - This script is useful for configuring and installing app on EC2 instances.
  - `AWS::CloudFormation::Interface`
    - Defines the grouping and ordering of input parameters when they are displayed in the AWS CloudFormation console.
    - By default, the AWS CloudFormationconsole alphabetically sorts parameters by their logical ID.
  - `AWS::CloudFormation::Designer`
    - Describes how your resources are laid out in AWS CloudFormationDesigner.
    - Designer automatically adds this information when you use it create and update templates.

```yaml
Metadata:
  Instances:
    Description: "Information about the instances"
  Databases:
    Description: "Information about the databases"
```


### `AWS::CloudFormation::Authentication`

- to specify authentication credentials for files or sources that you specify with the `AWS::CloudFormation::Init` resource.
- To include authentication information,
  - use the uris property if the source is a URI
  - use the buckets property if the source is an Amazon S3 bucket.

  - can also specify authentication information for files directly in the `AWS::CloudFormation::Init` resource.
    - The files key of the resource contains a property named authentication.
    - You can use the authentication property to associate authentication information defined in an `AWS::CloudFormation::Authentication` resource directly with a file.

For files, AWS CloudFormation looks for authentication information in the following order:
1. The `authentication property` of the `AWS::CloudFormation::Init` files key.
2. The `uris or buckets property` of the `AWS::CloudFormation::Authentication` resource.

- For sources, AWS CloudFormation looks for authentication information in the uris or buckets property of the `AWS::CloudFormation::Authentication` resource.

Examples
- Unlike most resources, the `AWS::CloudFormation::Authentication` type defines a list of user-named blocks,
- each of which contains authentication properties that use lower camel case naming.

1. EC2 web server authentication
   - how to get a file from a private S3 bucket within an EC2 instance.
   - The credentials used for authentication are defined in the `AWS::CloudFormation::Authentication` resource, and referenced by the `AWS::CloudFormation::Init` resource in the files section.

```yaml
Metadata:
  MetadataID:
    Type: AWS::CloudFormation::Authentication
    String:
      accessKeyId: String
      buckets:
        - String
      password: String
      secretKey: String
      type: String
      uris:
        - String
      username: String
      roleName: String

WebServer:
  Type: AWS::EC2::Instance
  DependsOn: "BucketPolicy"

  Metadata:
    AWS::CloudFormation::Init:
      config:
        packages:
          yum:
            httpd: []
        files:
          /var/www/html/index.html:
            source:
              Fn::Join:
                - ""
                -
                  - "http://s3.amazonaws.com/"
                  - Ref: "BucketName"
                  - "/index.html"
            mode: "000400"
            owner: "apache"
            group: "apache"
            authentication: "S3AccessCreds"
        services:
          sysvinit:
            httpd:
              enabled: "true"
              ensureRunning: "true"

    AWS::CloudFormation::Authentication:
      S3AccessCreds:
        type: "S3"
        accessKeyId:
          Ref: "CfnKeys"
        secretKey:
          Fn::GetAtt:
            - "CfnKeys"
            - "SecretAccessKey"
Properties:
  EC2 Resource Properties ...
```


### `AWS::CloudFormation::Interface`

- a metadata key that defines how parameters are grouped and sorted in the AWS CloudFormation console.
- When you create or update stacks in the console, the console lists input parameters in alphabetical order by their logical IDs.
- grouping and ordering parameters
  - By using this key, you can define your own parameter grouping and ordering so that users can efficiently specify parameter values.
  - example
    - group all EC2-related parameters in one group and all VPC-related parameters in another group.

- define labels for parameters.
  - A label is a friendly name or description that the console displays instead of a parameter's logical ID.
  - Labels are useful for helping users understand the values to specify for each parameter.
  - Example,
  - label a KeyPair parameter Select an EC2 key pair.

```yaml
Metadata:
  AWS::CloudFormation::Interface:
    ParameterGroups:
      - ParameterGroup
    ParameterLabels:
      ParameterLabel


Metadata:

  AWS::CloudFormation::Interface:
    ParameterGroups:

      - Label:
          default: "Network Configuration"
        Parameters:
          - VPCID
          - SubnetId
          - SecurityGroupID

      - Label:
          default: "Amazon EC2 Configuration"
        Parameters:
          - InstanceType
          - KeyName

    ParameterLabels:
      VPCID:
        default: "Which VPC should this be deployed to?"

```

Using the metadata key from this example, the following figure shows how the console displays parameter groups when a stack is created or updated: Parameter groups in the console

![console-create-stack-parameter-groups](https://i.imgur.com/0XOUxcX.png)


---


## Resources: declare the AWS resources be included/created in stack

![Screen Shot 2020-06-26 at 10.44.06](https://i.imgur.com/fQBHEj4.png)

- declare the AWS resources be included / created in the stack
  - such as an EC2 instance, an S3 bucket.
- These properties could also be set in the Parameters or Conditions sections

- must declare each resource separately;
- can specify multiple resources of the same type.
  - declare multiple resources, separate them with commas


```json
"Resources" : {
  "Instances1" : {
      "Type" : "AWS::EC2::Instance",
      "Properties" : {
          // MyQueue resource as part of its UserData property,
          "UserData" : { "Fn::Base64" : { "Fn::Join" : [ "", [ "Queue=", { "Ref" : "MyQueue" }]]}},
          // AvailabilityZone setting: the EC2 instance will be hosted in Northern Virginia us-east-1a.
          "AvailabilityZOne" : "us-east-1a",
          "ImageId" : "ami-20b65349"
      },
      // DependsOn:
      // How AWS CloudFormation should wait to launch a resource until a specific, different resource has finished being created.
      // create the EC@ after the myDB instance has been created
      "DependsOn" : "myDB"
  },
  // 2nd resource is an Amazon Simple Queue Service SQS—queue "MyQueue".
  "MyQueue" : {
      "Type" : "AWS::SQS::Queue",
      "Properties" : {}
  },
  "myDB" : {
      "Type" : "AWS::RDS::DBInstance",
      "Properties" : {}
  }
}
```

```yaml
Resources:
  Logical ID:
    Type: Resource type
    Properties:
      String: OneStringValue
      String: A longer string value
      Number: 123
      LiteralList:
        - "[first]-string-value with a special characters"
        - "[second]-string-value with a special characters"
      Boolean: true
      ReferenceForOneValue:
        Ref: MyLogicalResourceName
      ReferenceForOneValueShortCut: !Ref MyLogicalResourceName
      FunctionResultWithFunctionParams: !Sub |
        Key=%${MyParameter}

Resources:
  MyEC2Instance:
    Type: "AWS::EC2::Instance"
    Properties:
      ImageId: "ami-0ff8a91507f77f867"

  MyInstance:
    Type: "AWS::EC2::Instance"
    Properties:
      UserData:
        "Fn::Base64":
          !Sub |
            Queue=${MyQueue}
      AvailabilityZone: "us-east-1a"
      ImageId: "ami-0ff8a91507f77f867"

  MyQueue:
    Type: "AWS::SQS::Queue"
    Properties: {}
```








### DependsOn

```json
"Resources" : {
  "Instances1" : {
      "Type" : "AWS::EC2::Instance",
      "Properties" : {
          "UserData" : { "Fn::Base64" : { "Fn::Join" : [ "", [ "Queue=", { "Ref" : "MyQueue" }]]}},
          "AvailabilityZOne" : "us-east-1a",
          "ImageId" : "ami-20b65349"
      },
      // DependsOn:
      // How AWS CloudFormation should wait to launch a resource until a specific, different resource has finished being created.
      // create the EC@ after the myDB instance has been created
      "DependsOn" : "myDB"
  },
  "myDB" : {
      "Type" : "AWS::RDS::DBInstance",
      "Properties" : {}
  }
}
```

The DependsOn attribute should be used when
- need to wait for something.
- Some resources in a VPC require a gateway (either internet gateway / VPN gateway)
  - If AWS CloudFormation template defines a VPC, a gateway, and a gateway attachment,
  - any resources that require the gateway depend on the gateway attachment.

  - Other VPC-dependent resources
    - Auto Scaling groups,
    - Amazon EC2 instances,
      - an EC2 instance with a public IP address depends on the VPC gateway attachment if the VPC and internet gateway resources are also declared in the same template.
    - Elastic Load Balancing load balancers,
    - Elastic IP addresses,
    - Amazon RDS—database instances,
    - Amazon Virtual Private Cloud VPC—routes that include the internet gateway



### wait condition: wait/pause and receive a signal to continue

```json
"Resources" : {
  "Instances1" : {
      "Type" : "AWS::EC2::Instance",
      "Properties" : {
          "UserData" : { "Fn::Base64" : { "Fn::Join" : [ "", [ "Queue=", { "Ref" : "MyQueue" }]]}},
          "AvailabilityZOne" : "us-east-1a",
          "ImageId" : "ami-20b65349"
      },
      // DependsOn:
      // How AWS CloudFormation should wait to launch a resource until a specific, different resource has finished being created.
      // create the EC2 after the myDB instance has been created
      "DependsOn" : "myDB"
  },
  "myWaitCondition" : {
      "Type" : "AWS::CloudFormation::WaitCondition",
      // create the EC after the myDB instance has been created
      "DependsOn" : "myDB",
      "Properties" : {
          "Handle" : { "R2ef" : "myWaitHandle"},
          "Timeout" : "4500"
          // It will wait for that EC2 instance or it will time out after 4,500 seconds.
      }
  },
  "myDB" : {
      "Type" : "AWS::RDS::DBInstance",
      "Properties" : {}
  }
}
```

- `AWS::CloudFormation::WaitConditionHandle`
  - has no properties.
  - reference the `WaitCondition Handlere source` by using the Ref function,
    - AWS CloudFormation returns a pre-signed URL.
    - You pass this URL to applications or scripts that are running on your EC2 instances to send signals to that URL.
  - An associated `AWS::CloudFormation::WaitCondition` resource checks the URL for the required number of success signals or for a failure signal.
  - The timeout value is in seconds


### creation policy: pause stack creation and wait for specified number of successful signals.

```json
"Resources" : {
  "AutoScalingGroup" : {
      "Type" : "AWS::AutoScaling::AutoScalingGroup",
      "Properties" : {
          "AvailabilityZOne" : {"Fn::GetAZs" : ""},
          "LaunchConfigurationName" : { "Ref" : "LaunchConfig" },
          "DesiredCapacity" : "3",
          "MinSize" : "1",
          "MaxSize" : "4"
      },
      "CreationPolicy" : {
          " ResourceSignal" : {
              "Count" : "3",
              // “PT#H#M#S: # is the number of hours, minutes, and seconds.
              "Timeout" : "PT15M"
              // wait for 3 AutoCaling instance but time out after 15m
          }
      }
  }
}
```

- This creation policy is associated with <font color=red> the creation of an Auto Scaling group </font>
  - <font color=blue> three successful signals within fifteen minutes are required or it will time out </font>
    - Set timeouts to give resources enough time to get up and running.
  - When the timeout period expires, or a failure signal is received,
  - the creation of the resource fails,
  - and AWS CloudFormation rolls the stack back.

---


## Mappings : keys and their associated values

![Screen Shot 2020-06-26 at 13.18.49](https://i.imgur.com/lTWjdxg.png)


- specify conditional parameter values.
- customize the properties of a resource based on certain conditions
  - enables fine-grained control over how the templates launched.


```json
"Mappings" : {
  "RegionAndAMIID" : {
    "us-east-1" : {
        "m1.small" : " ami-aa",
        "te.micro" : " ami-bb",
    },
    "us-east-2" : {
        "m1.small" : " ami-cc",
        "te.micro" : " ami-dd",
    }
  }
}
```


- For example,
  - use Regions and specify multiple mapping levels
    - an AMI ImageId number is unique to a Region, and the person who use the template not necessarily know which AMI to use.
    - provide the `AMI lookup list` using the Mappings parameter.
    - contains a map for Regions.
    - The mapping
      - lists the AMI that should be used, based on the Region the instance will launch in
      - specifies an AMI based on the type of instance that is launched within a specific Region.
      - if an m1.small instance is used, the AMI be used is ami-1ccae774.
      - This mapping ties specific machine images to instances.


---


## Output     ￼


![Screen Shot 2020-06-26 at 15.17.56](https://i.imgur.com/pI7Bf8n.png)

- Outputs are values that are returned whenever view the properties of the stack.

- For example,
  - if something executes properly,
  - it is helpful to provide an indication that the execution completed and was successful.

- Outputs can specify the string output of any logical identifier that is available in the template.

- It's a convenient way to capture important information about your resources or input parameters


### example


1. Stack output

```yaml
# the output named BackupLoadBalancerDNSName
# returns the DNS name for the resource with the logical ID BackupLoadBalancer only when the CreateProdResources condition is true.
# (The second output shows how to specify multiple outputs.)
Outputs:
  BackupLoadBalancerDNSName:
    Description: The DNSName of the backup load balancer
    Value: !GetAtt BackupLoadBalancer.DNSName
    Condition: CreateProdResources
  InstanceID:
    Description: The Instance ID
    Value: !Ref EC2Instance
```

2. Cross-stack output

```yaml
# the output named StackVPC returns the ID of a VPC,
# and then exports the value for cross-stack referencing with the name VPCID appended to the stack's name.
Outputs:
  StackVPC:
    Description: The ID of the VPC
    Value: !Ref MyVPC
    Export:
      Name: !Sub "${AWS::StackName}-VPCID"
```



---

# Resources and Features outside AWS CloudFormation

For Resources and Features Not Directly Supported by AWS CloudFormation

- AWS CloudFormation is extensible with custom resources
  - so can use part of your own logic to create stacks.
  - With custom resources, write custom provisioning logic in templates.
  - CloudFormation runs the custom logic when you create, update, or delete stacks.

- For example
- to include resources that are not available as AWS CloudFormation resource types.
  - include those resources by using custom resources,
  - which means that you can still manage all your related resources in a single stack.
  - Use the `AWS::CloudFormation::CustomResource` or `Custom::String` resource type to define custom resources in your templates.
  - Custom resources require one property: <font color=red> the service token </font>
    - specifies where AWS CloudFormationsends requests to, such as an Amazon SNS topic.
  - Examples include
    - provisioning a third-party application subscription and passing the authentication key back to the EC2 instance that needs it.
    - use an AWS Lambda function to peer a new VPC with another VPC



example:


![Screen Shot 2020-06-26 at 15.24.01](https://i.imgur.com/RSa5TWf.png)


```json
cfnVerifier
    Type: AWS::CloudFormation::CustomResource
    Properties:
        ServiceToken
          Fn::Join [ "", [ "arn:aws:lambda:", !Ref: "AWS::Region", ":", !Ref: "AWS::AccountId", ":function:cfnVerifierLambda"]]
```

- user creates an AWS CloudFormation template by using a stack that has a `custom resource operation`.
  - This custom resource operation was defined by using `AWS::CloudFormation::CustomResource` or `Custom::CustomResource`.
- The template includes a <font color=red> ServiceToken </font>
  - from the third-party resource provider
  - used for authentication.
- The template also includes any provider-defined parameters required for the custom resource.

- AWS CloudFormation
  - communicates with the custom resource provider by using Amazon Simple Notification Service SNS—message that includes
    - a Create, Update, or Delete request.
    - any input data that is stored in the stack template
    - and an Amazon S3 URL for the response.
- The custom resource provider
  - processes the message
  - returns a Success or Fail response to AWS CloudFormation.
  - can also return
    - the names and values of resource attributes if the request succeeded (output data)
    - or send a string that provides details when the request fails.

- AWS CloudFormation
  - sets the stack status according to the response that is received,
  - provides the values of any custom resource output data.

- can use an AWS Lambda function to act as a custom resource.
  - To implement this, can replace the ServiceToken for custom resource with the Amazon Resource Name, ARN, of your Lambda custom resource.
  - do not need to create an Amazon SNS topic for a custom resource when you use AWS Lambda because AWS CloudFormation is Lambda-aware.

- As in the previous scenarios, your code is responsible for doing any required processing.

- It uses the pre-signed URL (sent by AWS CloudFormation) to signal to the service that the creation of the custom resource either succeeded or failed.














.
