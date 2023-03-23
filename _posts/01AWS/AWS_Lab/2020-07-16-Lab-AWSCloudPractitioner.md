---
title: AWS Lab - for CCP
date: 2020-07-16 11:11:11 -0400
categories: [01AWS, AWSLab]
tags: [AWS, Lab]
math: true
image:
---


# Lab for AWS Certified Cloud Practitioner

- [Lab for AWS Certified Cloud Practitioner](#lab-for-aws-certified-cloud-practitioner)
  - [lab 1: `IAM Policy`](#lab-1-iam-policy)
  - [lab 2: `Build VPC, Launch a Web Server`](#lab-2-build-vpc-launch-a-web-server)
  - [lab 3 `Amazon EC2`](#lab-3-amazon-ec2)
  - [lab 4 `amazon lambda`](#lab-4-amazon-lambda)
  - [lab 5 `AWS Elastic Beanstalk`](#lab-5-aws-elastic-beanstalk)
  - [lab 6 `amazon EBS`](#lab-6-amazon-ebs)
  - [lab 7 `amazon S3`](#lab-7-amazon-s3)
  - [lab 8 `amazon EFS`](#lab-8-amazon-efs)
  - [lab 9 `amazon RDS`](#lab-9-amazon-rds)
  - [lab 9 `amazon DynamoDB`](#lab-9-amazon-dynamodb)
  - [lab 10 `Balancing`](#lab-10-balancing)
    - [Task 1: Create an AMI for Auto Scaling](#task-1-create-an-ami-for-auto-scaling)
    - [Task 2: Create a Load Balancer](#task-2-create-a-load-balancer)
    - [Task 3: Create a Launch Configuration and an Auto Scaling Group](#task-3-create-a-launch-configuration-and-an-auto-scaling-group)
    - [Task 4: Verify that Load Balancing is Working](#task-4-verify-that-load-balancing-is-working)
    - [Task 5: Test Auto Scaling](#task-5-test-auto-scaling)
    - [Task 6: Terminate Web Server 1](#task-6-terminate-web-server-1)


---

## lab 1: `IAM Policy`

![lab-scenario](https://i.imgur.com/qbcGVzc.jpg)

structure of the statements in an IAM Policy:
- **Effect**: Allow or Deny the permissions.
- **Action** specifies the API calls that can be made against an AWS Service (eg `cloudwatch:ListMetrics`).
- **Resource**: the scope of entities covered by the policy rule (eg a specific Amazon S3 bucket or Amazon EC2 instance, or * means any resource)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "ec2:Describe*",
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": "elasticloadbalancing:Describe*",
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "cloudwatch:ListMetrics",
        "cloudwatch:GetMetricStatistics",
        "cloudwatch:Describe*"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": "autoscaling:Describe*",
      "Resource": "*"
    }
  ]
}
```

---

## lab 2: `Build VPC, Launch a Web Server`

![architecture](https://i.imgur.com/LlXN8fA.png)

1. **Create VPC**
    - AWS Management Console -> select region -> VPC -> Launch VPC Wizard
    - select a VPC configuration:
      - VPC with a single public subnet
      - *VPC with public and private subnets* -> select
      - VPC with public and private subnets & hardware VPN access
      - VPC with a private subnets & hardware VPN access
    - with public and private subnets:
      - *VPC name* Lab VPC
      - *Availability Zone*: Select the first Availability Zone
      - *Public / Private subnet name*: Private Subnet 1
      - *Elastic IP Allocation ID*: Click in the box and select the displayed IP address
      - *create*
2. Create Additional Subnets
    - VPC dashboard -> Subnets.
    - create a second Public Subnet.
    - Click *Create subnet* then configure:
      - *Name tag*: Public Subnet 2
      - *VPC*: Lab VPC
      - *Availability Zone*: Select the second Availability Zone
      - *IPv4 CIDR block*: 10.0.2.0/24
    - now create a second Private Subnet.
      - CIDR block: 10.0.3.0/24
3. configure Route Table for the private Subnets.
    - configure the Private Subnets to route internet-bound traffic to the NAT Gateway
    - so that resources in the Private Subnet are able to connect to the Internet, while still keeping the resources private.
    - VPC dashboard -> *Route Tables*.
    - Select the route table with *Main = Yes* and *VPC = Lab VPC*.
    - *Name column*: Private Route Table
    - In the lower pane -> Routes tab.
      - Destination 0.0.0.0/0
      - Target nat-xxxxxxxx: traffic destined for the internet (0.0.0.0/0) will be sent to the NAT Gateway. The NAT Gateway will then forward the traffic to the internet.
      - This route table is used to route traffic from Private Subnets.
    - In the lower pane -> *Subnet Associations tab*
      - -> *Edit subnet associations*
      - Select both *Private Subnet 1* and *Private Subnet 2*.
4. configure Route Table for the Public Subnets.
    - Select the route table with *Main = no* and *VPC = Lab VPC*.
    - In the lower pane -> Routes tab.
      - Destination 0.0.0.0/0
      - Target igw-xxxxxxxx, which is the Internet Gateway. This means that internet-bound traffic will be sent straight to the internet via the Internet Gateway.
    - *Name column*: Public Route Table
    - In the lower pane -> *Subnet Associations tab*
      - -> *Edit subnet associations*
      - Select both *Public Subnet 1* and *Public Subnet 2*.
5. Create a VPC Security Group
    - VPC dashboard -> *Security Groups*
    - -> *Create security group*
      - *Security group name*: Web Security Group
      - *Description*: Enable HTTP access
      - *VPC*: Lab VPC
    - add a rule to the security group to permit inbound web requests.
      - Select Web Security Group -> Inbound Rules tab -> *Edit rules*
      - -> *Add Rule*
        - *Type*: HTTP
        - *Source*: Anywhere
        - *Description*: Permit web requests
6. Launch a Web Server Instance
    - launch an Amazon EC2 instance into the new VPC. You will configure the instance to act as a web server.
    - Services -> EC2 -> Launch Instance
    - *Amazon Machine Image (AMI)*: Amazon Linux 2 (at the top)
    - *The Instance Type*: t2.micro
    - -> *Configure Instance Details* configure the instance to launch in a Public Subnet of the new VPC.
      - *Network*: Lab VPC
      - *Subnet*: Public Subnet 2 (not Private!)
      - *Auto-assign Public IP*: Enable
      - Advanced Details section: *User data box*:

    ```c
    #!/bin/bash
    # Install Apache Web Server and PHP
    yum install -y httpd mysql php
    # Download Lab files
    wget https://aws-tc-largeobjects.s3.amazonaws.com/AWS-TC-AcademyACF/acf-lab3-vpc/lab-app.zip
    unzip lab-app.zip -d /var/www/html/
    # Turn on web server
    chkconfig httpd on
    service httpd start

    // This script will be run automatically when the instance launches for the first time. The script loads and configures a PHP web application.
    ```

    - -> Next: Add Storage
    - -> Next: Add Tags   -
      - Key: Name
      - Value: Web Server 1
    - -> Next: Configure Security Group
      - *Select an existing security group*: Web Security Group.
    - -> Review and Launch
    - When prompted with a warning that you will not be able to connect to the instance through port 22, click Continue
    - Launch
    - *Select an existing keypair dialog*:
      - I acknowledge
      - Launch Instances
      - View Instances
    - Copy the Public DNS (IPv4) value: `ec2-3-90-11-85.compute-1.amazonaws.com`
    - Open a new web browser tab, paste the Public DNS value and press Enter.
    - a web page displaying the AWS logo and instance meta-data values.


![Screen Shot 2020-05-06 at 00.19.42](https://i.imgur.com/nzNIhKA.png)

---

## lab 3 `Amazon EC2`

![lab-scenario](https://i.imgur.com/GRWyQoi.jpg)

Task 1: Launch Your Amazon EC2 Instance
- Step 1: Choose an Amazon Machine Image (AMI): Amazon Linux 2 AMI
- Step 2: Choose an Instance Type: t2.micro
- Step 3: Configure Instance Details:
  - *Network*: Lab VPC.
  - *Enable termination protection* -> Protect against accidental termination (prevent the instance from being accidentally terminated, you can enable termination protection for the instance, which prevents it from being terminated.)
  - add *user data*
    ```c
    #!/bin/bash
    yum -y install httpd
    systemctl enable httpd
    systemctl start httpd
    echo '<html><h1>Hello From Your Web Server!</h1></html>' > /var/www/html/index.html
    ```
- Step 4: Add Storage
- Step 5: Add Tags
- Step 6: Configure Security Group
  - *Security group name*: Web Server security group
  - *Description*: Security group for my web server
  - Delete the existing SSH rule.
- Step 7: Review Instance Launch
  - this lab will not log into instance, do not require a key pair.
  - Click the *Choose an existing key pair*: Proceed without a key pair.
  - Select I acknowledge that ....
  - Launch Instances
  - View Instances

Task 2: Monitor Your Instance
- Click the *Status Checks* tab.
  - both the `System reachability` and `Instance reachability` checks have passed.
- Click the *Monitoring* tab.
  - not many metrics to display because the instance was recently launched.
- In the *Actions* menu -> *Instance Settings* -> *System Log*.
- In the *Actions* menu -> *Instance Settings* -> *Instance Screenshot*.

Task 3: Update Your Security Group and Access the Web Server
- *Description* tab -> *IPv4 Public IP* -> web browser Enter.
  - not allow.
- In the left navigation pane, click *Security Groups*.
- Select Web *Server security group*.
- Click the *Inbound* tab.
  - Click *Edit then configure*:
  - *Type*: HTTP
  - *Source*: Anywhere
  - Click Save

![Screen Shot 2020-05-06 at 17.02.57](https://i.imgur.com/0wcBGz9.png)

Task 4: Resize Your Instance: Instance Type and EBS Volume
- EC2 Management Console -> Instances -> Web Server
- Actions -> Instance State -> Stop
- *Change The Instance Type*
  - Actions -> Instance Settings -> Change Instance Type,
  - Instance Type: t2.small
  - Click Apply
- *Resize the EBS Volume*
  - In the left navigation menu, click Volumes.
  - Actions -> Modify Volume.
  - Change the size to: 10
  - Click Apply
- Start the Resized Instance again

Task 5: Explore EC2 Limits
- In the left navigation pane, click Limits.

Task 6: Test Termination Protection
- *Actions* -> *Instance State* -> Terminate.
  - the Yes, Terminate button is dimmed and cannot be clicked.
- *Actions* -> *Instance Settings* -> *Change Termination Protection*.
  - Click Yes, Disable
- You can now terminate the instance.

---

## lab 4 `amazon lambda`

![lambda-activity](https://i.imgur.com/Mt5defg.png)

create an AWS Lambda function.
- create an Amazon CloudWatch event to trigger the function every minute.
- The function uses an AWS Identity and Access Management (IAM) role. This IAM role allows the function to stop an Amazon Elastic Compute Cloud (Amazon EC2) instance that is running in the Amazon Web Services (AWS) account.

Task 1: **Create a Lambda function**
- Services menu -> Lambda -> Create function.
  - Choose *Author from scratch*
  - *Function name*: myStopinator
  - *Runtime*: Python 3.8
  - Click *Choose or create an execution role*
  - *Execution role*: Use an existing role
  - *Existing role*: myStopinatorRole
  - Click Create function.

Task 2: **Configure the trigger**
- Click *+ Add trigger*.
- *Select a trigger*: CloudWatch Events.
- Create a new rule
  - *Rule name*: everyMinute
  - *Rule type*: Schedule expression
  - *Schedule expression*: rate(1 minute)
  - Click Add.

Task 3: **Configure the Lambda function**
- click myStopinator
- *Function code* box:
    ```py
    import boto3
    region = '<REPLACE_WITH_REGION>'
    instances = ['<REPLACE_WITH_INSTANCE_ID>']
    ec2 = boto3.client('ec2', region_name=region)

    def lambda_handler(event, context):
        ec2.stop_instances(InstanceIds=instances)
        print('stopped your instances: ' + str(instances))

    // Replace the <REPLACE_WITH_REGION> placeholder with the actual Region that you are using. 'us-east-1'
    // replace <REPLACE_WITH_INSTANCE_ID> with the actual instance ID
    ```
- click Save.
- Lambda function is now fully configured. It should attempt to stop instance every minute.
- Click *Monitoring* (the tab near the top of the page).

Task 4: **Verify that the Lambda function worked**
- Return to the Amazon EC2 console
- instance was stopped.

---

## lab 5 `AWS Elastic Beanstalk`

Task 1: **Access the Elastic Beanstalk environment**
- Services -> *Elastic Beanstalk*.
  - click on the *name* of the environment -> The Dashboard page
  - Green (good): The Elastic Beanstalk environment is ready to host an application. However, it does not yet have running code.
  - click the URL (the URL ends in elasticbeanstalk.com).
  - a new browser tab opens. "HTTP Status 404 - Not Found" message. This behavior is expected because this application server doesn't have an application running on it yet.
  - Return to the Elastic Beanstalk console.

Task 2: **Deploy a sample application to Elastic Beanstalk**
- download a sample application, click [link](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/samples/java-tomcat-v3.zip)
- Elastic Beanstalk Dashboard,
  - click *Upload and Deploy*: java-tomcat-v3.zip file downloaded.
  - Click Deploy.
- After the deployment is complete, click the URL again.
  - ![web-app](https://i.imgur.com/jaAIj6C.png)
- Elastic Beanstalk console -> *Configuration* in the left pane.
  - in the *Instances* row, it indicates the Monitoring interval, EC2 Security groups, and Root volume type details of the Amazon Elastic Compute Cloud (Amazon EC2) instances that are hosting your web application.
  - the *Database* row: does not have details because the environment does not include a database.
    - click Edit. could easily add a database to this environment if you wanted to: you only need to set a few basic configurations and click Apply.
- In the left panel, click *Monitoring*: charts to see the kinds of information that are available to you.


Task 3: **Explore the AWS resources that support your application**
- Services menu -> EC2 -> Instances
  - two instances are running (they both contain samp in their names). Both instances support your web application.
- continue exploring the Amazon EC2 service resources created by Elastic Beanstalk
  - A `security group` with port 80 open
  - A `load balancer` that both instances belong to
  - An `Auto Scaling group` that runs from two to six instances, depending on the network load
  - Though Elastic Beanstalk created these resources for you, you still have access to them.


---

## lab 6 `amazon EBS`

create an Amazon EBS volume, attach it to an instance, apply a file system to the volume, and then take a snapshot backup.

![lab-scenario](https://i.imgur.com/PQEWwie.jpg)

1. Create a New EBS Volume

![Screen Shot 2020-05-07 at 01.16.27](https://i.imgur.com/gBKin2k.png)

2. Attach the Volume to an Instance


3. Connect to Your Amazon EC2 Instance
    - Windows Users: Using SSH to Connect
      - Download **PPK** file
      - Configure PuTTY to not timeout:
        - Click **Connection**
        - Set **Seconds between keep alives** to 30
      - Configure your PuTTY session:
        - Click **Session**
        - **Host Name (or IP address)**: the IPv4 Public IP address of instance.
        - in the **Connection** list: SSH
        - Click **Auth** (don't expand it)
        - Click **Browse**: labsuser.ppk file
        - Click **Open**
      - Click **Yes**, to trust the host and connect to it.
        - **login as**: ec2-user
        - This will connect you to the EC2 instance.
    - macOS and Linux Users
      - Download **PEM** file
      ```c
      J:~ luo$ cd Downloads/
      J:Downloads luo$ chmod 400 labsuser.pem
      ssh -i labsuser.pem ec2-user@54.158.249.24
      ```
4. Task 4: Create and Configure Your File System

```c
[ec2-user@ip-10-1-11-200 ~]$ df -h
Filesystem      Size  Used Avail Use% Mounted on
devtmpfs        483M   64K  483M   1% /dev
tmpfs           493M     0  493M   0% /dev/shm
/dev/xvda1      7.9G  1.2G  6.7G  15% /

[ec2-user@ip-10-1-11-200 ~]$ sudo mkfs /dev/sdf
Writing superblocks and filesystem accounting information: done

[ec2-user@ip-10-1-11-200 ~]$ sudo mkdir /mnt/data-store
[ec2-user@ip-10-1-11-200 ~]$ sudo mount /dev/sdf /mnt/data-store/
[ec2-user@ip-10-1-11-200 ~]$ echo "/dev/sdf   /mnt/data-store ext3 defaults,noatime 1 2" | sudo tee -a /etc/fstab
/dev/sdf   /mnt/data-store ext3 defaults,noatime 1 2

[ec2-user@ip-10-1-11-200 ~]$ cat /etc/fstab
LABEL=/     /           ext4    defaults,noatime  1   1
/dev/sdf   /mnt/data-store ext3 defaults,noatime 1 2

[ec2-user@ip-10-1-11-200 ~]$ df -h
Filesystem      Size  Used Avail Use% Mounted on
devtmpfs        483M   64K  483M   1% /dev
tmpfs           493M     0  493M   0% /dev/shm
/dev/xvda1      7.9G  1.2G  6.7G  15% /
/dev/xvdf      1008M  1.3M  956M   1% /mnt/data-store

[ec2-user@ip-10-1-11-200 ~]$ sudo su -c "echo some text has been written > /mnt/data-store/file.txt"
[ec2-user@ip-10-1-11-200 ~]$ cat /mnt/data-store/file.txt
some text has been written

```

5. Create an Amazon EBS Snapshot

![Screen Shot 2020-05-07 at 01.36.39](https://i.imgur.com/MYyZDnW.png)

```c
[ec2-user@ip-10-1-11-200 ~]$ sudo rm /mnt/data-store/file.txt
[ec2-user@ip-10-1-11-200 ~]$ ls /mnt/data-store/
lost+found
```

6. Restore the Amazon EBS Snapshot

![Screen Shot 2020-05-07 at 01.38.25](https://i.imgur.com/N2mcn1F.png)

![Screen Shot 2020-05-07 at 01.39.20](https://i.imgur.com/A6DuTU3.png)

7. Attach the Restored Volume to EC2 Instance

![Screen Shot 2020-05-07 at 01.39.54](https://i.imgur.com/9LwGYQj.png)

8. Mount the Restored Volume

```c
[ec2-user@ip-10-1-11-200 ~]$ sudo mkdir /mnt/data-store2
[ec2-user@ip-10-1-11-200 ~]$ sudo mount /dev/sdg /mnt/data-store2
[ec2-user@ip-10-1-11-200 ~]$ ls /mnt/data-store2/
file.txt  lost+found
```

---

## lab 7 `amazon S3`

- create bucket
  - bucket name: global unique name
  - set properties:
  - set permissions:
  - review -> create bucket

---

## lab 8 `amazon EFS`

- create file system
  - select VPC
  - create mount targets:

![Screen Shot 2020-05-07 at 01.53.15](https://i.imgur.com/4q0dcHK.png)

![Screen Shot 2020-05-07 at 01.53.50](https://i.imgur.com/H3RquF3.png)

- create security group

![Screen Shot 2020-05-07 at 01.54.52](https://i.imgur.com/eMdHC4e.png)

![Screen Shot 2020-05-07 at 01.55.44](https://i.imgur.com/dbwDfbP.png)

```c
yum install nfs-common

sudo mkdir /mnt/efs

sudo mount -t nfs4 -o nfsvers=4.1 ..... /mnt/efs
```

---

## lab 9 `amazon RDS`

![lab-5-final-lab-architecture](https://i.imgur.com/SKlcGdW.png)

Task 1: **Create a Security Group for the RDS DB Instance**
- Services -> *VPC* -> Security Groups.
  - Create security
![Screen Shot 2020-05-07 at 21.51.12](https://i.imgur.com/EfAi5oq.png)
  - Click the Inbound Rules tab. This configures the `Database security group` to permit inbound traffic on port 3306 from `EC2 instance associated with the Web Security Group`.
![Screen Shot 2020-05-07 at 21.56.32](https://i.imgur.com/EBUW1mW.png)

Task 2: **Create a DB Subnet Group**
- Services -> *RDS* -> Subnet groups.
![Screen Shot 2020-05-07 at 22.01.36](https://i.imgur.com/J6pEqdo.png)

Task 3: **Create an Amazon RDS DB Instance**
- Create database -> MySQL.
  - Settings
    - DB instance identifier: lab-db
    - Master username: master
    - Master password: lab-password
    - Confirm password: lab-password
  - Under DB instance size:
    - Burstable classes: db.t3.micro
  - Under Storage
    - Storage type: General Purpose (SSD)
    - Allocated storage: 20
  - Under *Connectivity*
    - Virtual Private Cloud (VPC): Lab VPC
    - Additional connectivity configuration
      - *Existing VPC security groups*: DB Security Group
  - Under Additional configuration
    - Initial database name: lab
    - Uncheck Enable automatic backups.
    - Uncheck Enable Enhanced monitoring.
  - Create database
- Click lab-db
  - Connectivity & security section: Endpoint field `lab-db.cggq8lhnxvnv.us-west-2.rds.amazonaws.com`

Task 4: **Interact with Database**
- open a web application running on web server and configure it to use the database.
  - SecretKey: my_key
  - WebServerIP: 34.203.248.37
  - AccessKey: my_access_key
- browser -> WebServerIP -> RDS
![Screen Shot 2020-05-07 at 22.17.25](https://i.imgur.com/rEzmE6h.png)
![Screen Shot 2020-05-07 at 22.44.08](https://i.imgur.com/u6UhopD.png)


---

## lab 9 `amazon DynamoDB`

![Screen Shot 2020-05-07 at 22.56.30](https://i.imgur.com/15EJrRY.png)

- using AWS CLI


---

## lab 10 `Balancing`

![starting-architecture](https://i.imgur.com/FflbBOB.png)

![final-architecture](https://i.imgur.com/JtO5TjW.png)

### Task 1: Create an AMI for Auto Scaling
- create an AMI from the existing Web Server 1. This will `save the contents of the boot disk` so that new instances can be launched with identical content.
- Services -> EC2 -> Instances: running Wait until 2/2 checks passed -> refresh update. (create an AMI based upon this instance).
- Select Web Server 1
  - Actions -> Image > **Create Image**:
    - Image name: Web Server AMI
    - Image description: Lab AMI for Web Server
    - Click Create Image
    - the AMI ID for your new AMI. use this AMI when launching the Auto Scaling group later in the lab.


### Task 2: Create a Load Balancer
- balance traffic across multiple EC2 instances and Availability Zones.
- Services -> EC2 -> Load Balancers -> **Create Load Balancer**
  - using an **Application Load Balancer**:
  - routing traffic to targets — EC2 instances, containers, IP addresses and Lambda functions — based on the content of the request.
  - click Create and configure:
    - **Name**: LabELB
    - **VPC**: Lab VPC
    - **Availability Zones**:
      - both to see the available subnets
      - Select *Public Subnet 1* and *Public Subnet 2*
      - This configures the load balancer to operate across multiple Availability Zones.
    - Next: Configure **Security Settings**
    - Next: Configure **Security Groups**
      - Select *Web Security Group*: permits HTTP access
      - deselect default.
    - Next: Configure **Routing**:
      - where to send requests that are sent to the load balancer. You will create a Target Group that will be used by Auto Scaling.
      - *Name*: `LabGroup`
    - Next: **Register Targets**: automatically register instances as targets later in the lab.
    - Next: Review
    - Create then click Close

The load balancer will show a state of provisioning. There is no need to wait until it is ready.


### Task 3: Create a Launch Configuration and an Auto Scaling Group

- Launch Configurations -> **Create launch configuration**
  - My AMIs: the AMI created from the existing Web Server 1.
  - instance type:
    - t3.micro
    - launched the lab in the us-east-1 Region: t2.micro
  - click Next: Configure details
    - Name: LabConfig
    - Monitoring: Select *Enable CloudWatch detailed monitoring*: allows Auto Scaling to react quickly to changing utilization.
  - Next: **Add Storage**: default storage settings.
  - Next: **Configure Security Group**
    - Select an `existing security group`
    - Select `Web Security Group`
  - Review
  - Create launch configuration
    - Select an existing key pair dialog:
  - Create launch configuration
  - now **create an Auto Scaling group** that uses this Launch Configuration.

- **create an Auto Scaling group**
  - Configure
    - **Group name**: Lab Auto Scaling Group
    - **Group size**: Start with: 2 instances
    - **Network**: Lab VPC
    - **Subnet**: Select `Private Subnet 1 (10.0.1.0/24)` and `Private Subnet 2 (10.0.3.0/24)`
      - launch EC2 instances in private subnets across both Availability Zones.
  - Expand **Advanced Details**:
    - **Load Balancing**: `Receive traffic from one or more load balancers`
    - **Target Groups**: `LabGroup`
    - **Monitoring**: Select `Enable CloudWatch detailed monitoring`
      - This will capture metrics at 1-minute intervals, which allows Auto Scaling to react quickly to changing usage patterns.
  - Next: **Configure scaling policies**
    - Select `Use scaling policies to adjust the capacity of this group`
    - Modify the Scale between text boxes to scale between `2` and `6` instances.
      - allow Auto Scaling to automatically add/remove instances, always keeping between 2 and 6 instances running.
    - In **Scale Group Size**:
      - **Metric type**: Average CPU Utilization
      - **Target value**: 60
        - This tells Auto Scaling to maintain an average CPU utilization across all instances at 60%. Auto Scaling will automatically add or remove capacity as required to keep the metric at, or close to, the specified target value. It adjusts to fluctuations in the metric due to a fluctuating load pattern.
  - Next: **Configure Notifications**
    - send a notification when a scaling event takes place. You will use the default settings.
  - Click Next: **Configure Tags**:
    - Tags applied to the Auto Scaling group will be automatically propagated to the instances that are launched.
    - **Key**: Name
    - **Value**: Lab Instance
    - Click Review
    - Review the details of your Auto Scaling group, then click Create Auto Scaling group. If you encounter an error Failed to create Auto Scaling group, then click Retry Failed Tasks.
    - Auto Scaling group has been created.

> Your Auto Scaling group will initially show an instance count of zero, but new instances will be launched to reach the Desired count of 2 instances.


### Task 4: Verify that Load Balancing is Working
Instances: two new instances named Lab Instance. These were launched by Auto Scaling. If the instances or names are not displayed, wait 30 seconds and click refresh  in the top-right.

- confirm that the new instances have passed their Health Check.
- Target Groups (in the Load Balancing section).
  - **LabGroup**
  - Click the **Targets** tab.
    - Two Lab Instance targets should be listed for this target group.
    - Wait until the Status of both instances transitions to healthy. Click Refresh in the upper-right to check for updates.
    - Healthy indicates that an instance has passed the Load Balancer's health check. This means that the Load Balancer will send traffic to the instance.

now access the Auto Scaling group via the Load Balancer.
- In the left navigation pane, click Load Balancers.
- copy the **DNS** name of the load balancer, making sure to omit "(A Record)". `LabELB-931928727.us-east-1.elb.amazonaws.com`
- Open in web browser

The application should appear in your browser. This indicates that the Load Balancer received the request, sent it to one of the EC2 instances, then passed back the result.


### Task 5: Test Auto Scaling

created an Auto Scaling group with a minimum of two instances and a maximum of six instances. Currently two instances are running because the minimum size is two and the group is currently not under any load.

now increase the load to cause Auto Scaling to add additional instances.

Return to the AWS management console, do not close the application tab

- Services -> **CloudWatch** -> click Alarms (not ALARM).
  - Two alarms displayed. created automatically by the Auto Scaling group. They will automatically keep the average CPU load close to 60% while also staying within the limitation of having two to six instances.
> Note: Please follow these steps only if you do not see the alarms in 60 seconds.
On the Services  menu, click EC2.
In the left navigation pane, click Auto Scaling Groups and then click on Scaling Policies.
Click Actions⌄ and Edit.
Change the Target Value to 50.
Click Save.
On the Services  menu, click CloudWatch.
In the left navigation pane, click Alarms (not ALARM) and verify you see two alarms.

- Click the **OK alarm**: which has AlarmHigh in its name.
  - The OK indicates that the alarm has not been triggered.
  - It is the alarm for CPU Utilization > 60, which will add instances when average CPU is high. The chart should show very low levels of CPU at the moment.
- now tell the application to perform calculations that should raise the CPU level.
- Return to the browser tab with the web application.
- Click **Load Test** beside the AWS logo.
  - This will cause the application to generate high loads. The browser page will automatically refresh so that all instances in the Auto Scaling group will generate load. Do not close this tab.
- Return to browser tab with the **CloudWatch console**.
  - In less than 5 minutes, the AlarmLow alarm should change to `OK` and the `AlarmHigh alarm` status should change to ALARM.
- the AlarmHigh chart indicating an increasing CPU percentage.
- Once it crosses the 60% line for more than 3 minutes, it will trigger Auto Scaling to add additional instances.
- Wait until the AlarmHigh alarm enters the ALARM state.
- now view the additional instance(s) that were launched.
- Services -> EC2. -> Instances.
- More than two instances labeled Lab Instance should now be running. The new instance(s) were created by Auto Scaling in response to the Alarm.


### Task 6: Terminate Web Server 1

In this task, you will terminate Web Server 1. This instance was used to create the AMI used by your Auto Scaling group, but it is no longer needed.

Select  Web Server 1 (and ensure it is the only instance selected).

In the Actions  menu, click Instance State > Terminate.

Click Yes, Terminate










.
