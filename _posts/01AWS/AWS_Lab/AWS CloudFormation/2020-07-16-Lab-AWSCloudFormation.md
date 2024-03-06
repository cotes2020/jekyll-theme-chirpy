---
title: AWS Lab - AWS CloudFormation
date: 2020-07-16 11:11:11 -0400
categories: [01AWS, AWSLab]
tags: [AWS, Lab]
math: true
image:
---

# AWS CloudFormation

- [AWS CloudFormation](#aws-cloudformation)
  - [auto infrastructure development](#auto-infrastructure-development)
    - [Task 1: Deploy a Networking Layer](#task-1-deploy-a-networking-layer)
    - [Task 2: Deploy an Application Layer](#task-2-deploy-an-application-layer)
    - [Task 3: Update a Stack](#task-3-update-a-stack)
    - [Task 4: Explore Templates with AWS CloudFormation Designer](#task-4-explore-templates-with-aws-cloudformation-designer)
    - [Task 5: Delete the Stack](#task-5-delete-the-stack)

---

## auto infrastructure development

### Task 1: Deploy a Networking Layer
deploy infrastructure in layers. Common layers are:
- Network (Amazon VPC)
- Database
- Application

This way, templates can be re-used between systems, such as `deploying a common network topology between Dev/Test/Production` or `deploying a standard database for multiple application`.


1. deploy an AWS CloudFormation template that creates a Networking layer using Amazon VPC.
    - `lab-network.yaml`

2. AWS Management Console > Services > CloudFormation

3. Create stack and configure:
    - Step 1: Specify template
        - Template source:  `Upload a template file`
        - Upload a template file: `lab-network.yaml`
        - Click Next
    - Step 2: Create Stack
        - Stack name: `lab-network`
        - Click Next
    - Step 3: Configure stack options
        - Tags:
        - Key: application
        - Value: inventory
        - Click Next
    - Step 4: Review lab-network
        - Click Create stack

4. The `template` will now be used by `CloudFormation` to `generate a stack of resources`.
    - The specified tags will be automatically propagated to the resources that are created, easier to identify resources used by particular applications.

![Screen Shot 2020-07-10 at 08.33.46](https://i.imgur.com/NmqccXZ.png)

5. **Resources** tab.
    - list of the `resources` created by the template.

![Screen Shot 2020-07-10 at 08.35.02](https://i.imgur.com/Ea92PCU.png)

6. **Events** tab
  - The list shows (in reverse order) the activities performed by CloudFormation,
  - such as starting to create a resource and then completing the resource creation.
  - Any errors encountered during the creation of the stack will be listed in this tab.

7. **Outputs** tab.
   - A CloudFormation stack can provide output information, such as the ID of specific resources and links to resources.
   - You will see two outputs:
     - PublicSubnet: The ID of the Public Subnet that was created (eg subnet-08aafd57f745035f1)
     - VPC: The ID of the VPC that was created (eg vpc-08e2b7d1272ee9fb4)
   - Outputs can also provide values that will be used by other stacks.
     - Export name column.
     - the VPC and Subnet IDs are given an export name so that other stacks can retrieve the values and build resources inside the VPC and Subnet.

8. **Template** tab.
   - the template that was used to create the stack. It shows the template that you uploaded while creating the stack. Feel free to examine the template and see the resources that were created, and the Outputs section at the end that defined which values to export.


### Task 2: Deploy an Application Layer
deploy an application layer that contains an EC2 instance and a Security Group.

The CloudFormation template: `lab-application.yaml`
- will `import the VPC and Subnet IDs` from the Outputs of the existing CloudFormation stack.
- will then use this information to create the Security Group in the VPC and the EC2 instance in the Subnet.

1. Stacks

2. Create stack and configure:
   - Step 1: Specify template
     - Template source:  `Upload a template file`
     - Upload a template file: `lab-application.yaml`
     - Click Next
   - Step 2: Specify stack details
     - Stack name: `lab-application`
     - **NetworkStackName**: `lab-network`
     - Click Next
     - The **NetworkStackName** parameter tells the template the name of the first stack you created (lab-network) so that it can retrieve values from the Outputs.
   - Step 3: Configure stack options
     - Tags:
     - Key: `application`
     - Value: `inventory`
     - Click Next
   - Step 4: Review lab-application
     - Click Create stack

3. application is now ready!

4. Outputs tab.
   - Copy the URL that is displayed, then open a new web browser tab, paste the URL and press Enter.
   - A new browser tab will open, taking you to the application running on the web server.

![Screen Shot 2020-07-10 at 09.17.57](https://i.imgur.com/JP6vFIz.png)


### Task 3: Update a Stack
CloudFormation can also update a stack that has been deployed.
- When updating a stack, CloudFormation will only modify or replace the resources that are being changed. Any resources that are not being changed will be left as-is.

update the lab-application stack to modify a setting in the Security Group. CloudFormation will leave all other resources as-is, without being modified by the update.

1. examine the current settings on the Security Group.
   - AWS Management Console > Services > EC2 > Security Groups > Web Server Security Group.

2. Inbound tab.
   - only one rule in the Security Group, which permits HTTP traffic.

3. return to CloudFormation to update the stack.
   - Services > CloudFormation.
   - `lab-application2.yaml`
   - This template has an additional configuration to permit inbound SSH traffic on port 22:

```c
- IpProtocol: tcp
  FromPort: 22
  ToPort: 22
  CidrIp: 0.0.0.0/0
```

4. lab-application
   - Update button and configure:
     - Replace current template
     - Template source:  Upload a template file
     - Upload a template file: ab-application2.yaml
     - Click Next three times to advance to the Review page.
   - Change set preview
     - In the Change set preview section at the bottom of the page, CloudFormation will display what resources need to be updated:
     - indicating that CloudFormation will Modify the Web Server security group without needing to replace it (Replacement = False).
     - This means there will be a minor change to the Security Group and no references to the security group will need to change.
   - Click Update stack

![change-set-preview](https://i.imgur.com/5Oj8Uw1.png)


5. Return to the EC2 console and select the Web Server security group.
   - The Inbound tab should display an additional rule for SSH traffic.


### Task 4: Explore Templates with AWS CloudFormation Designer

1. Services > CloudFormation > Designer

2. open Local lab-application2.yaml template

3. Designer will display a graphical representation of the template:

4. Experiment with the features of the Designer.
   - Click on the displayed resources. The lower pane will display the portion of the template that defines the resources.
   - Try dragging a new resource from the Resource Types pane on the left into the design area. The definition of the resource will be automatically inserted into the template.
   - Try dragging the resource connector circles to create relationships between resources
   - Open the lab-network.yaml template you downloaded earlier in the lab and explore its resources too

### Task 5: Delete the Stack
When resources are no longer required, CloudFormation can delete the resources built for the stack.
- A Deletion Policy can also be specified against resources, which can preserve or (in some cases) backup a resource when its stack is deleted.
- This is useful for retaining databases, disk volumes or any resource that might be required after stack deletion.

The `lab-application` stack has been configured to take a snapshot of an Amazon EBS disk volume before it is deleted:

```C
DiskVolume:
  Type: AWS::EC2::Volume
  Properties:
    Size: 100
    AvailabilityZone: !GetAtt WebServerInstance.AvailabilityZone
    Tags:
      - Key: Name
        Value: Web Data
  DeletionPolicy: Snapshot
```

The DeletionPolicy in the final line is directing CloudFormation to create a snapshot of the disk volume before it is deleted.

1. CloudFormation console > lab-application > Delete > Delete stack.

2. monitor the deletion process in the Events tab
   - see a reference to the EBS snapshot being created.

3. Wait for the stack to be deleted. It will disappear from the list.

4. check that a snapshot was created of the EBS volume before it was deleted.
  - Services > EC2 > Snapshots > a snapshot with a Started time in the last few minutes.













---
