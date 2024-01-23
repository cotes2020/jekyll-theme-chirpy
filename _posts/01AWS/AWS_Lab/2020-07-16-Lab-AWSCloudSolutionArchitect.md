---
title: AWS Lab - for SAA
date: 2020-07-16 11:11:11 -0400
categories: [01AWS, AWSLab]
tags: [AWS, Lab]
math: true
image:
---

# Lab for SAA

- [Lab for SAA](#lab-for-saa)
- [lab 1: Making Your Environment Highly Available](#lab-1-making-your-environment-highly-available)
  - [Task 1: Inspect Your environment](#task-1-inspect-your-environment)
  - [Task 2: Login to EC2 instance](#task-2-login-to-ec2-instance)
  - [Task 3: Download, Install, and Launch Your Web Server's PHP Application](#task-3-download-install-and-launch-your-web-servers-php-application)
  - [Task 4: Create an Amazon Machine Image (AMI)](#task-4-create-an-amazon-machine-image-ami)
  - [Task 5: Configure a Second Availability Zone](#task-5-configure-a-second-availability-zone)
  - [Task 6: Create an Application Load Balancer](#task-6-create-an-application-load-balancer)
  - [Task 7: Create an Auto Scaling Group](#task-7-create-an-auto-scaling-group)
  - [Task 8: Test the Application](#task-8-test-the-application)
  - [Task 9: Test High Availability](#task-9-test-high-availability)
- [lab2: Using Notifications to Trigger AWS Lambda](#lab2-using-notifications-to-trigger-aws-lambda)
  - [Task 1: Create an SNS Topic](#task-1-create-an-sns-topic)
  - [Task 2: Configure Auto Scaling to Send Events](#task-2-configure-auto-scaling-to-send-events)
  - [Task 3: An IAM Role for the Lambda function](#task-3-an-iam-role-for-the-lambda-function)
  - [Task 4: Create a Lambda Function](#task-4-create-a-lambda-function)
  - [Task 5: Scale-Out the Auto Scaling Group to Trigger the Lambda function](#task-5-scale-out-the-auto-scaling-group-to-trigger-the-lambda-function)
- [lab 7 Implementing a Serverless Architecture with AWS Managed Services](#lab-7-implementing-a-serverless-architecture-with-aws-managed-services)
  - [Task 1: Create a Lambda Function to Process a Transactions File](#task-1-create-a-lambda-function-to-process-a-transactions-file)
  - [Task 2: Create a Lambda Function to Calculate Transaction Totals and Notify About High Account Balances](#task-2-create-a-lambda-function-to-calculate-transaction-totals-and-notify-about-high-account-balances)
  - [Task 3: Create a Simple Notification Service (SNS) Topic](#task-3-create-a-simple-notification-service-sns-topic)
  - [Task 4: Create Two Simple Queue Service Queues](#task-4-create-two-simple-queue-service-queues)
  - [Task 5: Testing the Serverless Architecture by Uploading a Transactions File](#task-5-testing-the-serverless-architecture-by-uploading-a-transactions-file)
- [lab 8: multi-Region failover with Amazon Route 53](#lab-8-multi-region-failover-with-amazon-route-53)
  - [Task 1: Inspect Your Environment](#task-1-inspect-your-environment-1)
  - [Task 2: Configure a Health Check](#task-2-configure-a-health-check)
  - [Task 3: Configure your Domain in Route 53](#task-3-configure-your-domain-in-route-53)
  - [Task 4: Check the DNS Resolution](#task-4-check-the-dns-resolution)
  - [Task 5 - Test Your Failover](#task-5---test-your-failover)


---

# lab 1: Making Your Environment Highly Available

Objectives
Create an image of an existing Amazon EC2 instance and use it to launch new instances.
Expand an Amazon VPC to additional Availability Zones.
Create VPC Subnets and Route Tables.
Create an AWS NAT Gateway.
Create a Load Balancer.
Create an Auto Scaling group.

![P1](https://i.imgur.com/4BEx0e4.jpg)

## Task 1: Inspect Your environment
This lab begins with an environment already deployed via AWS CloudFormation including:
- An Amazon VPC
- A public subnet and a private subnet in one Availability Zone
- An Internet Gateway associated with the public subnet
- A NAT Gateway in the public subnet
- An Amazon EC2 instance in the public subnet

![P2](https://i.imgur.com/goKUVbr.jpg)

---

**Task 1.1: Inspect Your VPC**
1. AWS Management Console > Services menu > VPC > Your VPCs.
    - In the IPv4 CIDR column: 10.200.0.0/20, which means this VPC includes 4,096 IPs between 10.200.0.0 and 10.200.15.255 (with some reserved and unusable).
    - also attached to a Route Table and a Network ACL.
    - also has a Tenancy of default, instances launched into this VPC will by default use shared tenancy hardware.

2. AWS Management Console > Services menu > VPC > Subnets.
    - **Public Subnet** 1 subnet:
        - VPC column, subnet exists inside of Lab VPC.
        - IPv4 CIDR column: 10.200.0.0/24, which means this subnet includes the 256 IPs (5 of which are reserved and unusable) between 10.200.0.0 and 10.200.0.255.
        - Availability Zone column: the Availability Zone in which this subnet resides.
        - Click on the row containing Public Subnet 1 to reveal more details at the bottom of the page.
        - Route Table tab
            - Destination     Target
            - 10.200.0.0/20   local
            - 0.0.0.0/0       igw-
        - Network ACL tab
        - <img alt="pic" src="https://i.imgur.com/5Xbn8cW.png" width="500">

1. AWS Management Console > Services menu > VPC > **Internet Gateways.**
   - <img alt="pic" src="https://i.imgur.com/nYdD6Up.png" width="500">

2. AWS Management Console > Services menu > VPC > **Security Groups**
    - Click Configuration Server SG.
        - Inbound Rules tab
        - <img alt="pic" src="https://i.imgur.com/3b5gxv7.png" width="500">
        - Outbound Rules tab.
        - <img alt="pic" src="https://i.imgur.com/PURZ7AQ.png" width="500">
        -

**Task 1.2: Inspect Your Amazon EC2 Instance**

1. Services menu > EC2 > click Instances.
    - In the Actions menu, click Instance Settings > View/Change User Data.
    - no User Data appears
    - the instance has not yet been configured to run your web application.
    - When launching an Amazon EC2 instance, you can provide a User Data script that is executed when the instance first starts and is used to configure the instance. However, in this lab you will configure the instance yourself!





## Task 2: Login to EC2 instance

```c
cd ~/Downloads
chmod 400 labsuser.pem
ssh -i labsuser.pem ec2-user@<public-ip>
```




## Task 3: Download, Install, and Launch Your Web Server's PHP Application

```c
sudo yum -y update

sudo yum -y install httpd php
// To install a package that creates a web server

sudo chkconfig httpd on
// configures the Apache web server to automatically start when the instance starts.

wget https://aws-tc-largeobjects.s3-us-west-2.amazonaws.com/CUR-TF-200-ACACAD/studentdownload/phpapp.zip
// downloads a zip file containing the PHP web application.


sudo unzip phpapp.zip -d /var/www/html/
// unzips the PHP application into the default Apache web server directory.

sudo service httpd start
// This starts the Apache web server.
```


Your web application is now configured

now access application to confirm that it is working.


The web application should appear and will display information about your location (actually, the location of your Amazon EC2 instance). This information is obtained from freegeoip.app.

<img alt="pic" src="https://i.imgur.com/4fX1Dnt.png" width="500">

Return to your SSH session, execute the following command:

exit
This ends your SSH session.




## Task 4: Create an Amazon Machine Image (AMI)

web application is configured on your instance

to clone instances to run an application on multiple instances, even across multiple Availability Zones.

create an AMI from your Amazon EC2 instance. You will later use this image to launch additional, fully-configured instances to provide a Highly Available solution.

1. EC2 Management Console > Configuration Server is selected, and click Actions > Image > Create Image.
    - a Root Volume is currently associated with the instance. This volume will be copied into the AMI.
    - For Image name, type: Web application
    - Leave other values at their default settings and click Create Image.
    - <img alt="pic" src="https://i.imgur.com/LwqrPal.png" width="500">




## Task 5: Configure a Second Availability Zone
To build a highly available application, it is a best practice to launch resources in multiple Availability Zones.

duplicate your network environment into a second Availability Zone.
You will create:
- A second public subnet
- A second private subnet
- A second NAT Gateway
- A second private Route Table
- 2nd AZ
- <img alt="pic" src="https://i.imgur.com/q6x7wWZ.jpg" width="500">


**Task 5.1: Create a second Public Subnet**
1. Services menu > VPC > Subnets.

2. Create Subnet.
   - <img alt="pic" src="https://i.imgur.com/M5vB5h4.png" width="500">
   - <img alt="pic" src="https://i.imgur.com/VXC5NfS.png" width="500">



**Task 5.2: Create a Second Private Subnet**
1. Create subnet.
   - <img alt="pic" src="https://i.imgur.com/nF3i8sq.png" width="500">



**Task 5.3: Create a Second NAT Gateway**
1. left navigation pane, click NAT Gateways.

2. Create NAT Gateway.
   - <img alt="pic" src="https://i.imgur.com/dvlRVj5.png" width="500">


**Task 5.4: Create a Second Private Route Table**

1. In the navigation pane, click Route Tables.
2. Create route table.
   - <img alt="pic" src="https://i.imgur.com/dC8D6re.png" width="500">


3. Highlight the Private Route Table 2 > Routes tab > Edit routes.
   - <img alt="pic" src="https://i.imgur.com/aWCnJ8S.png" width="500">


4. Highlight the Private Route Table 2 > Subnet Associations tab > Edit subnet associations.
    - Select (tick) the checkbox beside `Private Subnet 2`.
    - <img alt="pic" src="https://i.imgur.com/F2KlhYJ.png" width="500">

5. Private Subnet 2 will now route Internet-bound traffic through the second NAT Gateway.







## Task 6: Create an Application Load Balancer

<img alt="pic" src="https://i.imgur.com/YaiN4p2.jpg" width="500">

do not have any instances yet
- created by the Auto Scaling group in the next task.


1. Services menu > EC2 > Load Balancers
    - Create Load Balancer > Application Load Balancer
    - <img alt="pic" src="https://i.imgur.com/YkmN5md.png" width="500">

- Next: Configure Security Settings
    - Select: Security group for the web servers
    - Note: This Security Group permits only HTTP incoming traffic, so it can be used on both the Load Balancer and the web servers.

- Click Next: Configure Routing.
- <img alt="pic" src="https://i.imgur.com/JxE0NzM.png" width="500">




## Task 7: Create an Auto Scaling Group

<img alt="pic" src="https://i.imgur.com/oxZ7lNl.jpg" width="500">

1. In the left navigation pane > Auto Scaling Groups > Create Auto Scaling group > My AMIs > Web application
    - Accept the default (t2.micro) instance type and click Next: Configure details .
    - Name: Web-Configuration
    - Click Next: Add Storage.
    - Click Next: Configure Security Group.
    - Click Select an existing security group.
        - Select the Security Group with a Description of Security group for the web servers.
    - Click Review.
    - Create launch configuration.
    - accept the vockey keypair, select the acknowledgement check box, then click Create launch configuration.

2. create the Auto Scaling group.
   - <img alt="pic" src="https://i.imgur.com/0gcS4bq.png" width="500">
   - <img alt="pic" src="https://i.imgur.com/lmJcPbb.png" width="500">

Ensure Keep this group at its initial size is selected.

> This configuration tells Auto Scaling to always maintain two instances in the Auto Scaling group. This is ideal for a Highly Available application because the application will continue to operate even if one instance fails. In such an event, Auto Scaling will automatically launch a replacement instance.

Click Next: Configure Notifications.

Click Next: Configure Tags.
- For Key, type: Name
- For Value, type: Web application

Click Review.
click Create Auto Scaling group.

Auto Scaling group will initially show zero instances.
should soon update to two instances.

Your application will soon be running across two Availability Zones and Auto Scaling will maintain that configuration even if an instance or Availability Zone fails.


## Task 8: Test the Application
1. left navigation pane > Target Groups.

Click the Targets tab in the lower half of the window.

You should see two Registered instances. The Status column shows the results of the Load Balancer Health Check that is performed against the instances.

Occasionally click the refresh icon in the top-right until the Status for both instances appears as healthy.
 If the status does not eventually change to healthy, ask your instructor for assistance in diagnosing the configuration. Hovering over the  icon in the Status column will provide more information about the status.

testing the application by connecting to the Load Balancer, which will then send your request to one of the Amazon EC2 instances. You will need to retrieve the DNS Name of the Load Balancer.

2. left navigation pane > Load Balancers > copy the DNS Name: LB1-xxxx.elb.amazonaws.com
    - Open a new web browser tab, paste
    - The Load Balancer forwarded your request to one of the Amazon EC2 instances. The Instance ID and Availability Zone are shown at the bottom of the web application.
        - `You are connected to instance i-0315fbffecb6ab3af in us-east-1a.`
    - Reload the page in your web browser. You should notice that the Instance ID and Availability Zone sometimes changes between the two instances.
        - `You are connected to instance i-082fccc5cfb52db95 in us-east-1b.`


The flow of information when displaying this web application is:
<img alt="pic" src="https://i.imgur.com/ctic2dc.jpg" width="500">

Data flow
- sent the request to the Load Balancer, which resides in the public subnets that are connected to the Internet.
- The Load Balancer chose one of the Amazon EC2 instances that reside in the private subnets and forwarded the request to it.
- EC2 instance requested geographic information from freegeoip.app. This request went out to the Internet through the NAT Gateway in the same Availability Zone as the instance.
- The Amazon EC2 instance then returned the web page to the Load Balancer, which returned it to your web browser.


## Task 9: Test High Availability

Your application has been configured to be Highly Available. This can be proven by stopping one of the Amazon EC2 instances.

1. EC2 Management Console > Instances > Select the `Configuration Server` > Actions > Instance State > Terminate

2. In a short time, the Load Balancer will notice that the instance is not responding and will automatically route all requests to the remaining instance.





---

# lab2: Using Notifications to Trigger AWS Lambda


<img alt="pic" src="https://i.imgur.com/6BND5ml.png" width="500">

<img alt="pic" src="https://i.imgur.com/wDkrKIN.png" width="500">

Many AWS services can automatically generate notifications when events occur.
- These notifications can be used to trigger automated actions without requiring human intervention.
- create an **AWS Lambda function** that  automatically **snapshot and tag new Amazon EC2 instances** launched by Auto Scaling.

The lab scenario is:
- An Auto Scaling group has already been configured.
- trigger Auto Scaling to scale-out and launch a new EC2 instance.
- This will send a notification to an Amazon Simple Notification Service (SNS) topic.
- The SNS topic will trigger an **AWS Lambda function** which will:
    - Create a snapshot of the Amazon EBS volumes attached to the instance.
    - Add a tag to the instance.
    - Sent log information to Amazon CloudWatch Logs.




## Task 1: Create an SNS Topic
create an Amazon Simple Notification Service (SNS) **topic** that the Auto Scaling group will use as a notification target.

1. AWS Management Console > Services > Simple Notification Service > to reveal the Amazon SNS menu > Topics > Create topic.
   - <img alt="pic" src="https://i.imgur.com/grlMWBr.png" width="500">

The topic is now ready to receive notifications.



## Task 2: Configure Auto Scaling to Send Events

- configure an Auto Scaling group to
  - send notifications to the SNS topic
  - when new EC2 instances are launched in the group.

1. AWS Management Console > EC2 > Auto Scaling Groups > the Auto Scaling group created > Notifications tab > Create notification.
   - <img alt="pic" src="https://i.imgur.com/GChiJl9.png" width="500">



## Task 3: An IAM Role for the Lambda function

An IAM role named `SnapAndTagRole` that has permission to perform operations on EC2 instances and to log messages in Amazon CloudWatch Logs has been pre-created for you.
You will later associate this role with your Lambda function.



## Task 4: Create a Lambda Function
- create an **AWS Lambda function** that will be invoked by Amazon SNS when Auto Scaling launches a new EC2 instance.
- The Lambda function will create a snapshot of the Amazon EBS volumes attached to the instance and then add a tag to the instance.

1. AWS Management Console > Lambda > Create a function > Author from scratch
   - <img alt="pic" src="https://i.imgur.com/zx5Y8vJ.png" width="500">

This role grants permission to the Lambda function to create an EBS Snapshot and to tag the EC2 instance.


> Blueprints
> code templates for writing Lambda functions. Blueprints are provided for standard Lambda triggers such as creating Alexa skills and processing Amazon Kinesis Firehose streams.


2. Function code section:

```py
# Snap_and_Tag Lambda function
#
# This function is triggered when Auto Scaling launches a new instance.
# A snapshot of EBS volumes will be created and a tag will be added.

from __future__ import print_function

import json, boto3

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))

    # Extract the EC2 instance ID from the Auto Scaling event notification
    message = event['Records'][0]['Sns']['Message']
    autoscalingInfo = json.loads(message)
    ec2InstanceId = autoscalingInfo['EC2InstanceId']

    # Snapshot all EBS volumes attached to the instance
    ec2 = boto3.resource('ec2')
    for v in ec2.volumes.filter(Filters=[{'Name': 'attachment.instance-id', 'Values': [ec2InstanceId]}]):
        description = 'Autosnap-%s-%s' % ( ec2InstanceId, v.volume_id )

        if v.create_snapshot(Description = description):
            print("\t\tSnapshot created with description [%s]" % description)

    # Add a tag to the EC2 instance: Key = Snapshots, Value = Created
    ec2 = boto3.client('ec2')
    response = ec2.create_tags(
        Resources=[ec2InstanceId],
        Tags=[{'Key': 'Snapshots', 'Value': 'Created'}]
    )
    print ("***Tag added to EC2 instance with id: " + ec2InstanceId)

    # Finished!
    return ec2InstanceId

```
Examine the code.
- Extract the EC2 instance ID from the notification message
- Create a snapshot of all EBS volumes attached to the instance
- Add a tag to the instance to indicate that snapshots were created

3. the Basic settings section at the bottom
    - <img alt="pic" src="https://i.imgur.com/ixPP4zM.png" width="500">

4. configure the trigger that will activate the Lambda function.
    - Add triggers at the top of the page.
    - Configure triggers
    - SNS topic: ScaleEvent

Note: the topic may already be pre-populated in the text box.

Amazon SNS will invoke this Lambda function when the ScaleEvent topic receives a notification from Auto Scaling.

Your Lambda function will now automatically execute whenever Auto Scaling launches a new instance.


## Task 5: Scale-Out the Auto Scaling Group to Trigger the Lambda function

- increase the desired capacity of the Auto Scaling group.
- This will cause the Auto Scaling group to launch a new Amazon EC2 instance to meet the increased capacity requirement.
- Auto Scaling will then send a notification to the ScaleEvent SNS topic.
- Amazon SNS will then invoke the Snap_and_Tag Lambda function.


1. AWS Management Console > EC2 > Auto Scaling Groups > Details tab > click Edit > Desired Capacity: `2`
    - cause Auto Scaling to launch an additional Amazon EC2 instance.
    - the Activity History tab and monitor the progress of the new EC2 instance that is being launched.
    - Wait for the status to change to show 2 rows with a Status of Successful. You can occasionally click refresh  to update the status.
    - <img alt="pic" src="https://i.imgur.com/reYRdBp.png" width="500">

Once the status has updated, you can confirm that the Lambda function executed correctly.

2. AWS Management Console > Instances > the instance that has the most recent launch time > Tags tab
    - see a tag with Snapshots as the key, and Created as the value.
    - This tag was added to the EC2 instance by your Lambda function.

3. AWS Management Console > Snapshots.
    - two snapshots that were created by the Lambda function.
    - Your Auto Scaling group successfully triggered the Lambda function, which created the tag and snapshots.
    - This provides an example serverless solution on AWS.


---

# lab 7 Implementing a Serverless Architecture with AWS Managed Services
![Screen Shot 2020-07-12 at 23.22.49](https://i.imgur.com/zAKe31Q.png)

- The scenario workflow is:
    - upload a transactions file to an **Amazon S3 bucket**
    - This will trigger an **AWS Lambda function** that read the file and insert records into two **Amazon DynamoDB tables**
    - This will trigger another **AWS Lambda function** that calculate customer totals and will send a message to an **Amazon Simple Notification Service (SNS) Topic** if the account balance is over $1500
    - **Amazon SNS** will send an email notification and will store a message in **Amazon Simple Queue Service (SQS) queues** to notify the customer and your credit collection department.


## Task 1: Create a Lambda Function to Process a Transactions File

<img alt="pic" src="https://i.imgur.com/c6WI7Yy.png" width="320">

- create an **AWS Lambda function** to process a transactions file.
  - The Lambda function will read the file and insert information into the Customer and Transactions DynamoDB tables.

Step:

1. AWS Management Console > Lambda
2. Create a function > `Author from scratch`
3. Configure the following:
    - Name: `TransactionProcessor`
    - Runtime: `Python 2.7`
    - Execution Role: `Choose Use an existing role`
    - Existing role: `TransactionProcessorRole`
4. **Function code section**

```py
# TransactionProcessor Lambda function
# This function is triggered by an object being created in an Amazon S3 bucket.
# The file is downloaded and each line is inserted into DynamoDB tables.

from __future__ import print_function
import json, urllib, boto3, csv

# Connect to S3 and DynamoDB
s3 = boto3.resource('s3')
dynamodb = boto3.resource('dynamodb')

# Connect to the DynamoDB tables
customerTable     = dynamodb.Table('Customer');
transactionsTable = dynamodb.Table('Transactions');

# This handler is executed every time the Lambda function is triggered
def lambda_handler(event, context):

  # Show the incoming event in the debug log
  print("Event received by Lambda function: " + json.dumps(event, indent=2))

  # Get the bucket and object key from the Event
  bucket = event['Records'][0]['s3']['bucket']['name']
  key = urllib.unquote_plus(event['Records'][0]['s3']['object']['key']).decode('utf8')
  localFilename = '/tmp/transactions.txt'

  # Download the file from S3 to the local filesystem
  try:
    s3.meta.client.download_file(bucket, key, localFilename)
  except Exception as e:
    print(e)
    print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
    raise e

  # Read the Transactions CSV file. Delimiter is the '|' character
  with open(localFilename) as csvfile:
    reader = csv.DictReader(csvfile, delimiter='|')

    # Read each row in the file
    rowCount = 0
    for row in reader:
      rowCount += 1

      # Show the row in the debug log
      print(row['customer_id'], row['customer_address'], row['trn_id'], row['trn_date'], row['trn_amount'])

      try:
        # Insert Customer ID and Address into Customer DynamoDB table
        customerTable.put_item(
          Item={
            'CustomerId': row['customer_id'],
            'Address':  row['customer_address']})

        # Insert transaction details into Transactions DynamoDB table
        transactionsTable.put_item(
          Item={
            'CustomerId':    row['customer_id'],
            'TransactionId':   row['trn_id'],
            'TransactionDate':  row['trn_date'],
            'TransactionAmount': int(row['trn_amount'])})

      except Exception as e:
         print(e)
         print("Unable to insert data into DynamoDB table".format(e))

    # Finished!
    return "%d transactions inserted" % rowCount
```

5. Basic settings
   - <img alt="pic" src="https://i.imgur.com/pJIni1S.png" width="300">

6. Add triggers > S3
   - <img alt="pic" src="https://i.imgur.com/4L2oqF9.png" width="500">


## Task 2: Create a Lambda Function to Calculate Transaction Totals and Notify About High Account Balances

<img alt="pic" src="https://i.imgur.com/StldKFD.png" width="500">

- create an **AWS Lambda function** to calculate transaction totals and send a **Simple Notification Service** notification if an account balance exceeds $1500.

1. AWS Management Console > Lambda
2. Create a function > `Author from scratch`
3. Configure the following
    - <img alt="pic" src="https://i.imgur.com/4aIfHkv.png" width="500">

4. **Function code** section

```py
# TotalNotifier Lambda function
#
# This function is triggered when values are inserted into the Transactions DynamoDB table.
# Transaction totals are calculated and notifications are sent to SNS if limits are exceeded.

from __future__ import print_function
import json, boto3

# Connect to SNS
sns = boto3.client('sns')
alertTopic = 'HighBalanceAlert'
snsTopicArn = [t['TopicArn'] for t in sns.list_topics()['Topics'] if t['TopicArn'].endswith(':' + alertTopic)][0]

# Connect to DynamoDB
dynamodb = boto3.resource('dynamodb')
transactionTotalTableName = 'TransactionTotal'
transactionsTotalTable = dynamodb.Table(transactionTotalTableName);

# This handler is executed every time the Lambda function is triggered
def lambda_handler(event, context):

  # Show the incoming event in the debug log
  print("Event received by Lambda function: " + json.dumps(event, indent=2))

  # For each transaction added, calculate the new Transactions Total
  for record in event['Records']:
    customerId = record['dynamodb']['NewImage']['CustomerId']['S']
    transactionAmount = int(record['dynamodb']['NewImage']['TransactionAmount']['N'])

    # Update the customer's total in the TransactionTotal DynamoDB table
    response = transactionsTotalTable.update_item(
      Key={
        'CustomerId': customerId
      },
      UpdateExpression="add accountBalance :val",
      ExpressionAttributeValues={
        ':val': transactionAmount
      },
      ReturnValues="UPDATED_NEW"
    )

    # Retrieve the latest account balance
    latestAccountBalance = response['Attributes']['accountBalance']
    print("Latest account balance: " + format(latestAccountBalance))

    # If balance > $1500, send a message to SNS
    if latestAccountBalance >= 1500:

      # Construct message to be sent
      message = '{"customerID": "' + customerId + '", ' + '"accountBalance": "' + str(latestAccountBalance) + '"}'
      print(message)

      # Send message to SNS
      sns.publish(
        TopicArn=snsTopicArn,
        Message=message,
        Subject='Warning! Account balance is very high',
        MessageStructure='raw'
      )

  # Finished!
  return 'Successfully processed {} records.'.format(len(event['Records']))
```

6. Basic settings section
    - <img alt="pic" src="https://i.imgur.com/zSXqVCP.png" width="500">

7. Add trigger
    - <img alt="pic" src="https://i.imgur.com/AiUB6eg.png" width="500">





## Task 3: Create a Simple Notification Service (SNS) Topic

<img alt="pic" src="https://i.imgur.com/nrgegoF.png" width="500">


- create a **Simple Notification Service (SNS)** topic that will receive a notification from your Lambda function when an account balance exceeds $1500. You will also subscribe to the topic with an email and via SMS.

1. Services > Simple Notification Service.
2. Create topic
    - <img alt="pic" src="https://i.imgur.com/2nyhKV3.png" width="500">

3. Create subscription
    - <img alt="pic" src="https://i.imgur.com/mTjk6zB.png" width="500">




## Task 4: Create Two Simple Queue Service Queues

<img alt="pic" src="https://i.imgur.com/yesnLBb.png" width="500">

- create two Simple Queue Service (SQS) queues.
  - subscribe 3 queues to the Simple Notification Service (SNS) topic created.
  - This setup is known as a fan-out scenario because each SNS notification is sent to multiple subscribers and those subscribers can independently consume messages from their own queue.

1. Simple Queue Service
2. create queue
    - Queue Name, type: `CreditCollection`
    - Queue Name, type: `CustomerNotify`
3. Subscribe Queues to SNS Topic.
   - <img alt="pic" src="https://i.imgur.com/5RFOqaA.png" width="500">


## Task 5: Testing the Serverless Architecture by Uploading a Transactions File


---

# lab 8: multi-Region failover with Amazon Route 53

1. Inspecting resources in `two Regions`.
2. Creating an **Amazon Route 53** `health check`
3. Creating an **Amazon Route 53** `domain`.
4. Configuring `primary and secondary settings`.
5. And testing the failover

![Screen Shot 2020-07-18 at 22.48.39](https://i.imgur.com/HmhQ9lz.png)


Objectives

After completing this lab, you will be able to:
- Use Route 53 to configure cross-region failover of a web application.
- Use Route 53 health checks to determine the health of a resource.

## Task 1: Inspect Your Environment

![mdtable1](https://i.imgur.com/1PZSWg4.png)

- Web-Application-1
    - IPv4 Public IP: `34.234.178.0`
- Web-Application-2
    - IPv4 Public IP: `35.163.208.11`

## Task 2: Configure a Health Check

Route 53 > Health checks
![Screen Shot 2020-07-18 at 23.28.51](https://i.imgur.com/h3kV6Bt.png)

## Task 3: Configure your Domain in Route 53

Route 53 > Hosted zones > `domain name`

1. create a `DNS A-record` to `point to your Primary web server`.
    - An `A-record` resolves a domain name by returning an IP address.
    - also associate this Record Set with the Health Check so traffic will only be sent to your Primary web server if the Health Check indicates that the server is healthy.

![Screen Shot 2020-07-18 at 23.47.42](https://i.imgur.com/GTBwLHZ.png)

2. create a `DNS A-record` to `point to your 2nd web server`.

![Screen Shot 2020-07-18 at 23.49.27](https://i.imgur.com/9u8PE2F.png)

3. check `heath check` > Health checkers tab
    - The health check is performed independently from multiple locations around the world, with each location requesting the page every 10 seconds.


## Task 4: Check the DNS Resolution

Hosted zones > domain > Test Record Set.
- Check response from Route 53
- Record name: `www`
- Type: `A`
- Get response.
- Response returned by Route 53 value. Confirm that it is the same IP address as your Primary web server.


## Task 5 - Test Your Failover

1. stop Primary web server instance

![Screen Shot 2020-07-19 at 00.01.03](https://i.imgur.com/rD1UwFN.png)

2. test again

3. 2nd web server ip.






















.
