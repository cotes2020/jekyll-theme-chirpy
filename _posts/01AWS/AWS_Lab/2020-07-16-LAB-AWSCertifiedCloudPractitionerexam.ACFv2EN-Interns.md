---
title: AWS Lab - ACFv2EN-Interns
date: 2020-07-16 11:11:11 -0400
categories: [01AWS, AWSLab]
tags: [AWS, Lab]
math: true
image:
---


# ACFv2EN-Interns.

- [ACFv2EN-Interns.](#acfv2en-interns)
  - [Module 1 Overview](#module-1-overview)
    - [cloud computing:](#cloud-computing)
    - [different types of cloud computing models](#different-types-of-cloud-computing-models)
    - [three main cloud computing deployment models](#three-main-cloud-computing-deployment-models)
    - [six advantages of cloud computing](#six-advantages-of-cloud-computing)
    - [the main Amazon Web Services AWS service categories and core services](#the-main-amazon-web-services-aws-service-categories-and-core-services)
    - [the AWS Cloud Adoption Framework (AWS CAF)](#the-aws-cloud-adoption-framework-aws-caf)
  - [Module 2 Overview](#module-2-overview)
    - [fundamentals of pricing](#fundamentals-of-pricing)
    - [TCO total cost of owernship](#tco-total-cost-of-owernship)
    - [AWS organizations](#aws-organizations)
    - [access AWS Organizations](#access-aws-organizations)
    - [billing and cost management](#billing-and-cost-management)
    - [Reserved Instances (RIs)](#reserved-instances-ris)
    - [aws Support](#aws-support)
  - [Module 3 Overview](#module-3-overview)
    - [AWS Regions, Availability Zones, and edge locations](#aws-regions-availability-zones-and-edge-locations)
    - [AWS services and service categories](#aws-services-and-service-categories)
      - [AWS Storage](#aws-storage)
      - [AWS compute](#aws-compute)
      - [AWS database](#aws-database)
      - [AWS networking and content delivery](#aws-networking-and-content-delivery)
      - [AWS security, identity and compliance](#aws-security-identity-and-compliance)
      - [AWS cost management](#aws-cost-management)
      - [AWS management and governance](#aws-management-and-governance)
  - [Module 4 Overview](#module-4-overview)
    - [AWS shared responsibility model](#aws-shared-responsibility-model)
    - [AWS Identity and Access Management (IAM)](#aws-identity-and-access-management-iam)
      - [authenticate as an IAM user](#authenticate-as-an-iam-user)
      - [IAM policy](#iam-policy)
      - [IAM group](#iam-group)
      - [IAM roles](#iam-roles)
    - [Securing accounts](#securing-accounts)
      - [AWS Organizations](#aws-organizations-1)
      - [Service control policies (SCPs)](#service-control-policies-scps)
      - [AWS Key Management Service (AWS KMS)](#aws-key-management-service-aws-kms)
      - [Amazon Cognito](#amazon-cognito)
      - [AWS Shield](#aws-shield)
    - [secure AWS data](#secure-aws-data)
    - [AWS compliance programs](#aws-compliance-programs)
  - [Module 5 Overview](#module-5-overview)
    - [Amazon Virtual Private Cloud (Amazon VPC)](#amazon-virtual-private-cloud-amazon-vpc)
    - [IP addressing](#ip-addressing)
    - [VPC Networking](#vpc-networking)


![learning-paths_architect_combined.8aef9e5a292422b17411b2cec6d1009ea786e422](https://i.imgur.com/huJhEhq.png)

https://aws.amazon.com/training/path-architecting/

[toc]

---

## Module 1 Overview

### cloud computing:
- Cloud computing is the *on-demand* delivery of `compute power, database, storage, applications, and other IT resources` via the internet with *pay-as-you-go pricing*.
- These resources run on server computers that are located in large data centers in different locations around the world.
- use a cloud service, service provider owns the computers that you are using. These resources can be used together like building blocks to build solutions that help meet business goals and satisfy technology requirements.


### different types of cloud computing models

- **traditional computing model**
  - infrastructure is thought of as `hardware. Hardware solutions` are physical
  - require `space, staff, physical security, planning, and capital expenditure`
  - the `long hardware procurement 采购 cycle` that involves acquiring, provisioning, and maintaining on-premises infrastructure.
    - enough resource capacity or sufficient storage?
    - provision capacity by guessing theoretical maximum peaks.(pay resources stay idle / don’t have sufficient capacity to meet your needs).
    - if your needs change, then spend the time, effort, and money required to implement a new solution.
    - For example, if you wanted to provision a new website, you would need to buy the hardware, rack and stack it, put it in a data center, and then manage it or have someone else manage it.
  - This approach is expensive and time-consuming.

- **Infrastructure as a Service (IaaS)**
  - IaaS contains the basic building blocks for cloud IT. *software solutions*
  - *flexible*.
    - select the cloud services best match the needs and provision, terminate those resources on-demand, pay as use. You can elastically scale resources up and down in an automated fashion. With the cloud computing model, you can treat resources as temporary and disposable.
    - The flexibility that cloud computing offers enables businesses to implement new solutions quickly and with low upfront costs.
    - Compared to hardware solutions, software solutions can change much more quickly, easily, and cost-effectively.
  - Cloud computing helps *eliminate undifferentiated heavy-lifting tasks* like procurement, maintenance, and capacity planning, thus enabling them to focus on what matters most.
  - *several different service models and deployment strategies*  emerged to meet the specific needs of different users. Each type of cloud service model and deployment strategy provides you with a different level of control, flexibility, and management. Understanding the differences between these cloud service models and deployment strategies can help you decide what set of services is right for your needs.
  - It typically *provides access to networking features, computers (virtual or on dedicated hardware), and data storage space*.
  - IaaS gives you the highest level of flexibility and management control over your IT resources. It is most similar to the existing IT resources with which many IT departments and developers are familiar.

- **Platform as a Service (PaaS)**
  - PaaS *removes the management of underlying infrastructure* (usually hardware and operating systems), and allows you to focus on the *deployment and management of your applications*.
  - This helps you be more efficient as you don’t need to worry about resource procurement, capacity planning, software maintenance, patching, or any of the other undifferentiated heavy lifting involved in running your application.


- **Software as a Service (SaaS)**
  - SaaS provides you with a complete product that is run and managed by the service provider. In most cases, people referring to SaaS are referring to *end-user applications* (such as web-based email).
  - With a SaaS offering, you don’t have to think about how the service is maintained or how the underlying infrastructure is managed. You only need to think about how you will use that particular software.


### three main cloud computing deployment models

 - represent the cloud environments that your applications can be deployed in:
 - **Cloud**:
  - A *cloud-based application* is fully deployed in the cloud, and all parts of the application run in the cloud. Applications in the cloud have either been created in the cloud or have been migrated from an existing infrastructure.
  - Cloud-based applications can be built on low-level infrastructure pieces or they can use higher-level services that provide abstraction from the management, architecting, and scaling requirements of core infrastructure.
 - **Hybrid**:
  - A hybrid deployment is a way to *connect infrastructure/applications between cloud-based resources and existing resources not in cloud*. The most common method of hybrid deployment is between the cloud and *existing on-premises infrastructure*.
  - This model enables an organization to extend and grow their infrastructure into the cloud while connecting cloud resources to internal systems.
 - **On-premises**:
  - Deploying resources on-premises, using virtualization and resource management tools, sometimes called private cloud.
  - While on-premises deployment does not provide many of the benefits of cloud computing, it is sometimes sought for its ability to provide dedicated resources.
  - In most cases, this deployment model is the same as legacy IT infrastructure, but it might also use application management and virtualization technologies to increase resource utilization.



### six advantages of cloud computing

1. Advantage #1 — Trade `capital expense` for `variable expense`:
  - Capital expenses (capex): funds that a company uses to acquire, upgrade, and maintain physical assets such as property, industrial buildings, or equipment.
  - variable expense: expense that the person who bears the cost can easily alter or avoid.
  - Maintenance is reduced

2. Advantage #2 — Benefit from `massive economies of scale`:
  - a lower variable cost than you can get on your own. Because usage from hundreds of thousands of customers is aggregated in the cloud, providers such as AWS can achieve higher economies of scale, which translates into lower pay-as-you-go prices.

3. Advantage #3 — Stop guessing `capacity`:
  - Eliminate guessing infrastructure capacity needs. When you make a capacity decision before you deploy an application, expensive idle resources / limited capacity. With cloud computing, these problems go away. You can access as much or as little as you need, and scale up and down as required with only a few minutes’ notice.

4. Advantage #4 — Increase `speed and agility`:
  - In cloud computing environment, new IT resources are only a click away, reduce the time to make those resources available. dramatic increase agility for the organization because the cost and time that it takes to experiment and develop are significantly lower.

5. Advantage #5 — Stop spending money on running and maintaining data centers:
  - Focus on projects that differentiate your business instead of focusing on the infrastructure.

6. Advantage #6 — `Go global` in minutes:
  - You can deploy your application in multiple AWSRegions around the world with just a few clicks. provide a lower latency and better experience for your customers simply and at minimal cost.

- Agility 敏捷 `/ə'dʒiləti/`
  - *easy access* to a broad range of technologies to innovate faster and build nearly anything.
  - You can quickly spin up resources as you need them from infrastructure services, such as compute, storage, and databases, to IoT, machine learning, data lakes and analytics...
  - You can deploy technology services in a matter of minutes, and get from idea to implementation several orders of magnitude faster than before.
  - the freedom to experiment, test new ideas to differentiate customer experiences, and transform your business.
- Elasticity 弹力 `/elæ'stɪsɪtɪ/`
  - avoid over-provision resources up front to handle peak levels of business activity in the future. but the amount of resources that actually need. You can scale these resources up or down to instantly to grow and shrink capacity as your business needs change.
- Cost savings
  - The cloud allows you to trade capital expenses (such as data centers and physical servers) for variable expenses, and only pay for IT as you consume it. Plus, the variable expenses are much lower than what you would pay to do it yourself because of the economies of scale.
- Deploy globally in minutes
  - With the cloud, you can expand to new geographic regions and deploy globally in minutes.
  - For example, AWS has infrastructure all over the world, so you can deploy your application in multiple physical locations with just a few clicks.
  - Putting applications in closer proximity to end users reduces latency and improves their experience.

similarities between AWS and the traditional, on-premises IT space:

![Screen Shot 2020-05-04 at 16.22.11](https://i.imgur.com/PGeBOJo.png)



### the main Amazon Web Services AWS service categories and core services

web service
- is any piece of software that makes itself available over the internet or on private (intranet) networks.
- uses a standardized format — such as Extensible Markup Language (XML) or JavaScript Object Notation (JSON)—for the request and the response of an application programming interface (API) interaction. It is not tied to any one operating system or programming language.It’s self-describing via an interface definition file and it is discoverable.

**Amazon Web Services (AWS)**
- a *secure cloud platform* that offers a broad set of global *cloud-based products*.
- Because these products are delivered over the internet, you have on-demand access to the `compute, storage, network, database, and other IT resources and the tools to manage them`. You can immediately provision and launch AWS resources. The resources are ready for you to use in minutes.
- AWS offers *flexibility*. Your AWS environment can be reconfigured and updated on demand, scaled up or down automatically to meet usage patterns and optimize spending, or shut down temporarily or permanently. The billing for AWS services becomes an operational expense instead of a capital expense.
- AWS services are designed to work together to support virtually any type of application or workload. Think of these services like building blocks, which you can assemble quickly to build sophisticated, scalable solutions, and then adjust them as your needs change.

![Screen Shot 2020-05-04 at 16.43.39](https://i.imgur.com/bNMqvxW.png)

Which service depend on business goals and technology requirements.
- Amazon EC2: complete control over your AWS computing resources.
- AWS Lambda: to run your code and not manage or provision servers.
- **AWS Elastic Beanstalk**: a service that deploys, manages, and scales your web applications for you.
- Amazon Lightsail: a lightweight cloud platform for a simple web application.
- AWS Batch: need to run hundreds of thousands of batch workloads.
- AWS Outposts: to run AWS infrastructure in your on-premises data center.
- **Amazon Elastic Container Service(Amazon ECS)**, **Amazon Elastic Kubernetes Service(Amazon EKS)**, or **AWS Fargate**: to implement a containers or microservices architecture.
- **VMware Cloud on AWS**: You have an on-premises server virtualization platform that you want to migrate to AWS.

![Screen Shot 2020-05-04 at 16.40.40](https://i.imgur.com/QeVxHRq.png)

![Screen Shot 2020-05-04 at 16.48.22](https://i.imgur.com/OZehrc9.png)


For example
- a database application
- customers might be sending data to Amazon Elastic Compute Cloud (Amazon EC2) instances, a service in the compute category.
- These EC2 servers batch the data in one-minute increments and add an object per customer to `Amazon Simple Storage Service (Amazon S3)`, the AWS storage service you’ve chosen to use.
- then use a non relational database like `Amazon DynamoDB` to power your application, for example, to build an index so that you can find all the objects for a given customer that were collected over a certain period.
- You might run these services inside an `Amazon Virtual Private Cloud (Amazon VPC)`, which is a service in the networking category.


to access the broad array of services offered by AWS.
- AWS Management Console: The console provides a rich graphical interface to a majority of the features offered by AWS.
- AWS Command Line Interface (AWS CLI):The AWS CLI provides a suite of utilities that can be launched from a command script in Linux, macOS, or Microsoft Windows.
- Software development kits (SDKs): AWS provides packages that enable accessing AWS in a variety of popular programming languages.

All three options are built on a common REST-like API that serves as the foundation of AWS. To learn more about tools you can use to develop and manage applications on AWS,


### the AWS Cloud Adoption Framework (AWS CAF)

AWS CAF:
- help organizations design and travel an accelerated path to successful cloud adoption.
- provides guidance and best practices to help organizations identify gaps in skills and processes.
- helps organizations build a comprehensive approach to cloud computing—both across the organization and throughout the IT lifecycle—to accelerate successful cloud adoption.

At the highest level, the `AWS CAF` organizes guidance into six areas of focus/perspectives: Business(business, people, governance), technical (platform, security, operations).
- Each perspective consists of a set of capabilities, which covers distinct responsibilities that are owned or managed by functionally related stakeholders.
- Capabilities within each perspective are used to identify which areas of an organization require attention. By identifying gaps, prescriptive work streams can be created that support a successful cloud journey.

1. Stakeholders from the `Business` perspective
  - (business/finance managers, budget owners, and strategy stakeholders)
  - can use the AWS CAF to create a strong business case for cloud adoption and prioritize cloud adoption initiatives.
  - *ensure that business strategies and goals align with its IT strategies and goals*.

2. Stakeholders from the `People` perspective
  - (human resources, staffing, and people managers: `resource/incentive/career/training/organization change management`)
  - can use the AWS CAF *to evaluate organizational structures and roles, new skill and process requirements, and identify gaps*.
  - Performing an analysis of needs and gaps can help prioritize training, staffing, and organizational changes to build an agile organization.

3. Stakeholders from the `Governance` perspective
  - (the Chief Information Officer, CIO, program managers, enterprise architects, business analysts, and portfolio managers: `portfolio/program and project/business performance/license management`)
  - can use the AWS CAF to *focus on the skills and processes that are needed to align 使结盟 IT strategy* and goals with business strategy and goals.
  - This focus helps the organization maximize the business value of its IT investment and minimize the business risks.

4. Stakeholders from the `Platform` perspective
  - (Chief Technology Officer, CTO, IT managers, solutions architects: `compute/network/storage/database provisioning, systems and solution architecture, application development`)
  - use a variety of architectural dimensions and models to *understand and communicate the nature of IT systems and their relationships*.
  - They must be able to describe the architecture of the target state environment in detail. The AWS CAF includes principles and patterns for implementing new solutions on the cloud, and for migrating on-premises workloads to the cloud.

5. Stakeholders from the `Security` perspective
  - (Chief Information Security Officer, CISO, IT security managers/analysts: `identity and access management, detective control, infrastructure security, data protection, incident response`)
  - must *ensure that the organization meets security objectives* for visibility, audit ability, control, and agility.
  - can use the AWS CAF to structure the selection and implementation of security controls that meet the organization’s needs.

6. Stakeholders from the `Operations` perspective
  - (IT operations/support managers: `service monitoring, application performance monitoring, resource inventory management, release/change management, reporting and analytics, business continuity/disaster recovery, IT service catalog`)
  - define how day-to-day, quarter-to-quarter, and year-to-year business is conducted.
  - Stakeholders from the Operations perspective align with and support the operations of the business. The AWS CAF helps these stakeholders *define current operating procedures*.
  - It also helps them identify the process changes and training that are needed to implement successful cloud adoption.


---

## Module 2 Overview

### fundamentals of pricing

In most cases, no charge for `inbound data transfer` or for data transfer between other AWS services within the same AWS Region.
`Outbound data transfer `is aggregated across services and then charged at the outbound data transfer rate.
- This charge appears on the monthly statement as AWS Data Transfer Out.

Pay for what you use
Pay less when you reserve/use more/as AWS grows
pay only for the services that you consume with no large upfront expenses.
lower variable costs,
All AWS services are available on demand, require no long-term contracts, and have no complex licensing dependencies.

![Screen Shot 2020-05-04 at 20.20.01](https://i.imgur.com/Wt4pZqj.png)

---

### TCO total cost of owernship

![Screen Shot 2020-05-04 at 20.21.10](https://i.imgur.com/YuUNMPO.png)

TCO
- `the cost of a service, plus all the costs that are associated with owning the service`.
- In the cloud environment, TCO is used for comparing the costs of `running an entire infrastructure environment for a specific workload in an on-premises or collocation facility`, to the same workload `running on a cloud-based infrastructure`.

![Screen Shot 2020-05-04 at 20.23.35](https://i.imgur.com/QkeAe7u.png)


### AWS organizations

![Screen Shot 2020-05-04 at 21.15.44](https://i.imgur.com/ldAOLXn.jpg)

structure of AWS Organizations.
- a basic organization, or root
- seven accounts, organized into four organizational units (OUs).
  - An OU is a `container for accounts` within a root.
  - An OU can also contain other OUs.
- attach a policy to one of the nodes in the hierarchy, it flows down and it affects all the branches and leaves.
- An OU can have only one parent and each account can be a member of exactly one OU.
- An account is a standard AWS account that contains your AWS resources.
- You can attach a policy to an account to apply controls to only that one account

![Screen Shot 2020-05-04 at 21.21.09](https://i.imgur.com/xFQwzro.png)

AWS Organizations enables you to:
- `policy-based account management`: Create `service control policies (SCPs)` that centrally control AWS services across multiple AWS accounts.
- `group-based account management`: Create groups of accounts and then attach policies to a group to ensure that the correct policies are applied across the accounts.
- Simplify account management by using `application programming interfaces (APIs)` to automate the creation and management of new AWS accounts.
- Simplify the billing process, set up a single payment method for all the AWS accounts in organization. With `consolidated billing`, see a combined view of charges off all accounts, and you can take advantage of pricing benefits from aggregated usage. `Consolidated billing` provides a central location to manage billing across all of your AWS accounts, and the ability to benefit from volume discounts.


AWS Organizations does not replace associating `AWS Identity and Access Management (IAM) policies` with users, groups, and roles within an AWS account.
  - With IAM policies, you can allow or deny access to AWS services(such as Amazon S3), individual AWS resources(such as a specific S3 bucket), or individual API actions(such as s3:CreateBucket).
  - An IAM policy can be applied only to IAM users, groups, or roles, and *can never restrict the AWS account root user*.

In contrast, with Organizations, you use `service control policies (SCPs)` to allow or deny access to particular AWS services for individual AWS accounts or for groups of accounts in an OU.
  - The specified actions from an attached SCP affect *all IAM users, groups, and roles for an account, including the AWS account root user*.

![Screen Shot 2020-05-04 at 23.07.24](https://i.imgur.com/3A0C5If.png)


### access AWS Organizations

![Screen Shot 2020-05-04 at 23.09.10](https://i.imgur.com/fopYerU.png)

- AWS `Management Console`: a browser-based interface that you can use to manage your organization and your AWS resources. You can perform any task in your organization by using the console.
- AWS `Command Line Interface(AWS CLI)` tools enable you to issue commands at your system's command line to perform AWS Organizations tasks and AWS tasks. faster and more convenient than using the console.
- AWS `software development kits (SDKs)`: to handle tasks such as cryptographically signing requests, managing errors, and retrying requests automatically. AWS SDKs consist of libraries and sample code for various programming languages and platforms (Java, Python, Ruby, .NET, iOS, and Android).
- AWS `Organizations HTTPS Query API`: gives you programmatic access to AWS Organizations and AWS. use the API to issue HTTPS requests directly to the service. When you use the HTTPS API, you must include code to digitally sign requests by using your credentials.

### billing and cost management

From the billing dashboard, you can access several other cost management tools that you can use to estimate and plan your AWS costs.These tools include AWS Bills, AWS Cost Explorer, AWS Budgets, and AWS Cost and Usage Reports


### Reserved Instances (RIs)

Save Money and Maintain Flexibility

Right sizing is the most effective way to control cloud costs.
- involves `continually analyzing` instance performance and usage needs and patterns
- then `turning off idle instances` and `right sizing instances` that are either over provisioned or poorly matched to the workload.
- resource needs are always changing, `right sizing` must become an ongoing process to continually achieve cost optimization.
  - make `right sizing` a smooth process by establishing a `right-sizing schedule` for each team, enforcing tagging for all instances, and taking full advantage of the powerful tools that AWS provide to resource monitoring and analysis.


For certain services like Amazon EC2 and Amazon RDS, you can invest in reserved capacity.
- save up to 75% over equivalent on-demand capacity.
- buy Reserved Instances, the larger the upfront payment, the greater the discount.
- can minimize risks, more predictably manage budgets, and comply with policies that require longer-term commitments.

- Reserved Instances are available in 3 options
  - `All up-front (AURI)`: receive the largest discount.
  - `partial up-front (PURI)`: lower discounts but to spend less up front.
  - `no upfront payments (NURI)`: receive a smaller discount, but allowing you to free up capital to spend in other projects.


### aws Support

- proactive guidance:
  - TAM: Technical account manager: via the enterprise support plan.
- best practices:
  - `AWS Trusted Advisor`: auto service, during implement ells right and problems.
- account assistance:
  - `AWS support concierge` 门房: non-tech billing and account level inquiries.

![Screen Shot 2020-05-05 at 00.28.49](https://i.imgur.com/yDZH3Qz.jpg)


---

## Module 3 Overview

### AWS Regions, Availability Zones, and edge locations

![Screen Shot 2020-05-05 at 02.03.41](https://i.imgur.com/sOqGghh.png)

use the `AWS management console` to enable or disable the region.

![Screen Shot 2020-05-05 at 02.10.02](https://i.imgur.com/5QMqWup.png)

![Screen Shot 2020-05-05 at 02.11.18](https://i.imgur.com/p2O3jYq.jpg)

AWS uses custom networking equipment source from multiple `original device manufactures ODMs`

![Screen Shot 2020-05-05 at 02.16.53](https://i.imgur.com/EuRHTXS.png)

`Amazon CloudFront` is a content delivery network(CDN) used to distribute content to end users to reduce latency. Amazon Route 53 is a Domain Name System (DNS) service. Requests going to either one of these services will be routed to the nearest edge location automatically in order to lower latency.
AWS `Points of Presence` are located in most of the major cities (69 cities in total) across 30 countries around the world. By continuously measuring internet connectivity, performance and computing to find the best way to route requests, the Points of Presence deliver a better near real-time user experience. They are used by many AWS services, including Amazon CloudFront, Amazon Route 53, AWS Shield, and AWS Web Application Firewall (AWSWAF) services. `Regional edge caches` are used by default with Amazon CloudFront. are used when you have content that is not accessed frequently enough to remain in an edge location. Regional edge caches absorb this content and provide an alternative to that content having to be fetched from the origin serve

![Screen Shot 2020-05-05 at 02.19.39](https://i.imgur.com/IL9c1Cy.jpg)

### AWS services and service categories

![Screen Shot 2020-05-05 at 02.21.20](https://i.imgur.com/qZvY9bq.png)

#### AWS Storage

![Screen Shot 2020-05-05 at 02.24.22](https://i.imgur.com/TCkfBd0.png)

`Amazon Simple Storage Service (Amazon S3)` is an object storage service that offers scalability, data availability, security, and performance. Use it to store and protect any amount of data for websites, mobile apps, backup and restore, archive, enterprise applications, Internet of Things (IoT) devices, and big data analytics.
`Amazon Elastic Block Store (Amazon EBS)`is high-performance block storage that is designed for use with Amazon EC2 for both throughput and transaction intensive workloads. It is used for a broad range of workloads, such as relational and non-relational databases, enterprise applications, containerized applications, big data analytics engines, file systems, and media workflows.
`Amazon Elastic File System (Amazon EFS)` provides a scalable, fully managed elastic Network File System (NFS) file system for use with AWS Cloud services and on-premises resources. It is built to scale on demand to petabytes, growing and shrinking automatically as you add and remove files. It reduces the need to provision and manage capacity to accommodate growth.
`Amazon Simple Storage Service Glacier` is a secure, durable, and extremely low-cost Amazon S3 cloud storage class for data archiving and long-term backup. It is designed to deliver 11 9s of durability, and to provide comprehensive security and compliance capabilities to meet stringent regulatory requirements

#### AWS compute

![Screen Shot 2020-05-05 at 02.24.43](https://i.imgur.com/L6z7C7o.jpg)

- `Amazon Elastic Compute Cloud (Amazon EC2)` provides resizable compute capacity as virtual machines in the cloud.
- `Amazon EC2 Auto Scaling`: automatically add or remove EC2 instances according to conditions that you define.
- `Amazon Elastic Container Service (Amazon ECS)` is a highly scalable, high-performance container orchestration service that supports Docker containers.
- `Amazon Elastic Container Registry (Amazon ECR)` is a fully-managed Docker container registry that makes it easy for developers to store, manage, and deploy Docker container images.
- `AWS Elastic Beanstalk` is a service for deploying and scaling web applications and services on familiar servers such as Apache and Microsoft Internet Information Services (IIS).
- `AWS Lambda` enables you to run code without provisioning or managing servers. You pay only for the compute time that you consume. There is no charge when your code is not running.
- `Amazon Elastic Kubernetes Service (Amazon EKS)` makes it easy to deploy, manage, and scale containerized applications that use Kubernetes on AWS.
- `AWS Fargateis` a compute engine for Amazon ECS, to run containers without having to manage servers or clusters.

#### AWS database

![Screen Shot 2020-05-05 at 02.32.32](https://i.imgur.com/zDf1oWI.png)

- `Amazon Relational Database Service (Amazon RDS)` makes it easy to set up, operate, and scale a relational database in the cloud. It provides resizable capacity while automating time-consuming administration tasks such as hardware provisioning, database setup, patching, and backups.
- `Amazon Aurora` is a MySQL and PostgreSQL-compatible relational database. up to 5 times faster than standard MySQL databases and 3 times faster than standard PostgreSQL databases.
- `Amazon Redshift` run analytic queries against petabytes of data that is stored locally in Amazon, and directly against exabytes of data that are stored in Amazon S3. It delivers fast performance at any scale.
- `Amazon DynamoDB` is a key-value and document database that delivers single-digit millisecond performance at any scale, with built-in security, backup and restore, and in-memory caching

#### AWS networking and content delivery

![Screen Shot 2020-05-05 at 02.35.39](https://i.imgur.com/BPB6Oab.png)

- `Amazon Virtual Private Cloud (Amazon VPC)` to provision logically isolated sections of the AWS Cloud.
- `Elastic Load Balancing` automatically distributes incoming application traffic across multiple targets, such as Amazon EC2 instances, containers, IP addresses, and Lambda functions.
- `Amazon CloudFront` is a fast content delivery network (CDN) service that securely delivers data, videos, applications, and application programming interfaces (APIs) to customers globally, with low latency and high transfer speeds.
- `AWS Transit Gateway` a service that enables customers to connect their Amazon `Virtual Private Clouds (VPCs)` and their on-premises networks to a single gateway.
- `Amazon Route 53`is a scalable cloud Domain Name System (DNS) web service designed to give you a reliable way to route end users to internet applications. It translates names (like www.example.com) into the numeric IP addresses (like 192.0.2.1) that computers use to connect to each other.
- `AWS Direct Connect` provides a way to establish a dedicated private network connection from your data center or office to AWS, reduce network costs and increase bandwidth throughput.
- `AWS VPN` provides a secure private tunnel from your network or device to the AWS global network.

#### AWS security, identity and compliance

![Screen Shot 2020-05-05 at 02.39.54](https://i.imgur.com/cB2DuX1.png)

- `AWS Identity and Access Management (IAM)` enables you to manage access to AWS services and resources securely. By using IAM, you can create and manage AWS users and groups. You can use IAM permissions to allow and deny user and group access to AWS resources.
- `AWS Organizations` allows you to restrict what services and actions are allowed in your accounts.
- `Amazon Cognito` lets you add user sign-up, sign-in, and access control to your web and mobile apps.
- `AWS Artifact` provides on-demand access to AWS security and compliance reports and select online agreements.
- `AWS Key Management Service (AWS KMS)` to create and manage keys. to control the use of encryption across a wide range of AWS services and in your applications.
- `AWS Shield` is a managed Distributed Denial of Service (DDoS) protection service that safeguards applications running on AWS

#### AWS cost management

![Screen Shot 2020-05-05 at 02.41.27](https://i.imgur.com/C4kLRBa.png)

- `The AWS Cost and Usage Report` contains the most comprehensive set of AWS cost and usage data available, including additional metadata about AWS services, pricing, and reservations.
- `AWS Budgets` set custom budgets alert when your costs or usage exceed (or are forecasted to exceed) your budgeted amount.
- `AWS Cost Explorer` has an easy-to-use interface that enables you to visualize, understand, and manage AWS costs and usage over time

#### AWS management and governance

![Screen Shot 2020-05-05 at 02.42.12](https://i.imgur.com/OH8WNTH.jpg)

- `The AWS Management Console` provides a web-based user interface for accessing your AWS account.
- `AWS Config` provides a service that helps you track resource inventory and changes.
- `Amazon CloudWatch` allows you to monitor resources and applications.
- `AWS Auto Scaling` to scale multiple resources to meet demand.
- `AWS Command Line Interface` provides a unified tool to manage AWS services.
- `AWS Trusted Advisor` helps you optimize performance and security.
- `AWS Well-Architected Tool` provides help in reviewing and improving your workloads.
- `AWS CloudTrail` tracks user activity and API usage

---

## Module 4 Overview

### AWS shared responsibility model

AWS shared responsibility model

![Screen Shot 2020-05-05 at 13.07.00](https://i.imgur.com/gOOD7zI.png)

![Screen Shot 2020-05-05 at 13.16.04](https://i.imgur.com/KDijCLE.png)

**AWS**: security “of” the cloud
- operates, manages, and controls the components from the `bare metal host operating system and hypervisor virtualization layer` down to the `physical security of the facilities` where the services operate.
- responsible for protecting the infrastructure that runs all the services offered in the AWS Cloud.
- The *global infrastructure* includes `AWS Regions, Availability Zones, and edge locations`.
- This *physical infrastructure*:
  - `Physical security of data centers` controlled, need-based access; nondescript facilities, 24/7 security guards; two-factor authentication; access logging and review; video surveillance; and disk degaussing and destruction.
  - `Hardware infrastructure` servers, storage devices, appliances that AWS relies on.
  - `Software infrastructure`, which hosts operating systems, service applications, and virtualization software.
  - `Network infrastructure`, such as routers, switches, load balancers, firewalls, and cabling. AWS also continuously monitors the network at external boundaries, secures access points, and provides redundant infrastructure with intrusion detection.

![Screen Shot 2020-05-05 at 13.16.51](https://i.imgur.com/mglqjcX.png)

**customer**: security “in” the cloud
- When customers use AWS services, they maintain complete control over their content.
- the *configuration of security groups/operating system* that run on compute instances (including updates and patches). secure account management.
- securing the applications that are launched on AWS resources,
- the *encryption* of data at rest and data in transit.
- ensure the *network* is configured for security, security credentials and logins are managed safely. firewall configurations, network configurations
- managing *critical content security requirements*: What content they choose to store on AWS•Which AWS services are used with the content•In what country that content is stored•The format and structure of that content and whether it is masked, anonymized, or encrypted•Who has access to that content and how those access rights are granted, managed, and revoked
- Customers retain control of *what security they choose to implement* to protect their own data, environment, applications, IAM configurations, and operating systems.

IaaS: `Amazon EC2`, `amazon Elastic Block Store(EBS)`, `Amazon Virtual Private Cloud(VPC)`
PaaS: `AWS Lambda`, `Amazon relational database service(RDS)`, `Amazon elastic Beanstalk`
SaaS: `AWS trusted advisor`, `AWS shield`, `Amazon chime`

![Screen Shot 2020-05-05 at 13.37.03](https://i.imgur.com/Akucjg2.png)

![Screen Shot 2020-05-05 at 13.36.28](https://i.imgur.com/2vVuFyK.png)


<font color=red>customers</font>
- updates and patched OS / Oracleon in EC2 instances
- EC2 security group setting
- configuration of App / S3 buckekt access / subnet / VPC / Authentication for account user login
- secure SSH keys

<font color=red>AWS</font>:
- physical.. / virtulization infrastructure / against outage / network isolation between customer / low lantency connection between web server and S3 buckekt
- updates and patched Oracleon in RDS Instances
- sercure AWS management console

### AWS Identity and Access Management (IAM)

- a tool that `centrally manages access` to `launching, configuring, managing, and terminating resources` in your AWS account.
- provides granular control over access to resources, including the ability to specify exactly which `API calls` the user is authorized to make to each service.
  - for AWS Management Console, AWS CLI, or AWS software development kits (SDKs), every call to an AWS service is an API call.
- grant different permissions to different people for different resources. `who, which, how`
- feature of your AWS account, and it is offered at no additional charge.

essential components
- `IAM user`: a person/application that is defined in an AWS account, and that must make API calls to AWS products.
  - Each user must have a unique name (no spaces in name) within the AWS account, and a set of security credentials. These credentials are different from the AWS account root user security credentials. Each user is defined in one and only one AWS account.
- `IAM group`: a collection of IAM users. use IAM groups to simplify specifying and managing permissions for multiple users.
- `IAM policy`: a document that defines permissions to determine what users can do in the AWS account. A policy typically grants access to specific resources to user / explicitly deny access.
- `IAM role`: a tool for granting `temporary access to specific AWS resources` in an AWS account. only selected users or applications.

#### authenticate as an IAM user

how the user is permitted to use to access AWS resources.
- 2 types: `programmatic access` and `AWS Management Console access`. only or both
- **programmatic access**:
  - `access key ID` and a `secret access key`
  - to make an `AWS API call` by AWS CLI/AWS SDK/other development tool.
- **AWS Management Console access**
  - browser login window.
    - 12-digit account ID / corresponding account alias.
    - IAM user name and password.
  - If `multi-factor authentication (MFA)` is enabled: an authentication code.
    - With MFA, users and systems must provide an `MFA token` + the regular sign-in credentials, before access AWS services and resources.
    - generating the MFA authentication token:
      - `virtual MFA-compliant applications`(Google Authenticator / Authy 2-Factor Authentication...),
      - `U2F security key devices` (Yubikey)
      - `hardware MFA devices` (Gemalto)
    - ![Screen Shot 2020-05-05 at 14.11.46](https://i.imgur.com/dwVKXzD.png)

#### IAM policy

By default, IAM users do not have permissions to any resources / data in AWS account.
- must explicitly grant permissions to a user/group/role by creating a IAM `policy`

IAM `policy`
- a document in `JavaScript Object Notation (JSON)` format
- lists permissions that allow / deny access to resources in the AWS account.
- principle of least privilege
- the scope of the IAM service configurations is global. The settings are not defined at an AWS Region level. IAM settings apply across all AWS Regions.
- When there is a conflict, the most restrictive policy applies.
- allow vs deny: deny win.

2 types of IAM policy:
1. **Identity-based policies**
    - permissions policies that attach to a `principal/identity (IAM user/role/group)`. control what actions that identity can perform, on which resources, and under what conditions.
    - Identity-based policies categorized:
        - **Managed policies** – prebuild, Standalone identity-based policies that can attach to multiple users/groups/roles in AWS account
        - **Inline policies** – Policies embedded directly to a single user/group/role

2. **Resource-based policies**
    - JSON policy documents that you attach to a `resource (S3 bucket...)`. control what actions a specified principal can perform on that resource, and under what conditions.
    - `inline only`: define the policy on the resource itself, instead of creating a separate IAM policy document that you attach.
        - For example
        - create an S3 bucket policy (a type of resource-based policy) on an S3 bucket -> the bucket -> Permissions tab -> Bucket Policy button -> define the JSON-formatted policy document there.
        - An Amazon S3 access control list (ACL) is another example of a resource-based policy

3. **identity-based policy**.
    - An IAM policy that grants access to the S3 bucket is attached to the MaryMajor user.

4. **resource-based policy**.
    - The S3 bucket policy for the photos bucket specifies that the user MaryMajor is allowed to list and read the objects in the bucket.
    - An explicit deny statement will always take precedence over any allow statement.
    - could define a deny statement in a bucket policy to restrict access to specific IAM users, even if the users are granted access in a separate identity-based policy.

<img alt="pic" src="https://i.imgur.com/HVBkIYM.png" width="320" alt="function_calls">

`explicit denial policy -> explicit allow policy -> deny`

<img alt="pic" src="https://i.imgur.com/X79K6Ni.png" width="500" alt="function_calls">


1. "Version": "2012-10-17",
2. "Statement": [ ]
    - "Action": [ ]
        - ec2:
            - "ec2:`Describe*`",
            - "ec2:StartInstances",
            - "ec2:StopInstances"
        - s3
            - "s3:Get*",
            - "s3:List*"
        - elasticloadbalancing:
            - "elasticloadbalancing:Describe*",
        - cloudwatch:
            - "cloudwatch:ListMetrics",
            - "cloudwatch:GetMetricStatistics",
            - "cloudwatch:Describe*"
        - autoscaling:
            - "autoscaling:Describe*",
    - "Resource": `"*"`,
    - "Effect": "Allow"


```JSON
{
  "Version": "2012-10-17",
  "Statement": [

    {
      "Effect": "Allow",
      "Action": [
        "s3:Get*",
        "s3:List*"
      ],
      "Resource": "*"
    }

    {
      "Action": [
        "ec2:Describe*",
        "ec2:StartInstances",
        "ec2:StopInstances"
      ],
      "Resource": "*",
      "Effect": "Allow"
    },

    {
      "Action": "elasticloadbalancing:Describe*",
      "Resource": "*",
      "Effect": "Allow"
    },

    {
      "Action": [
        "cloudwatch:ListMetrics",
        "cloudwatch:GetMetricStatistics",
        "cloudwatch:Describe*"
      ],
      "Resource": "*",
      "Effect": "Allow"
    },
    {
      "Action": "autoscaling:Describe*",
      "Resource": "*",
      "Effect": "Allow"
    }
  ]
}
```



#### IAM group
Important characteristics of IAM groups:
- A group can contain many users, and a user can belong to multiple groups.
- Groups cannot be nested. A group can contain only users, and a group cannot contain other groups.
- There is no default group that automatically includes all users in the AWS account. group with all account users in it, you need to create the group and add each new user to it.A

#### IAM roles
- an IAM identity in account that has specific permissions.
- similar to IAM user, also an AWS identity that can attach permissions policies to, determine what the identity can/cannot do in AWS.
- but instead of being uniquely associated with one person, a role is intended to be assumable by anyone who needs it.
  - does not have standard long-term credentials (password/access keys associated with it...)
  - assume a role, the role provides temporary security credentials for role session.
- use roles to delegate 托付 access to users/app/services that do not normally have access to your AWS resources.
  - For example
  - grant users in your AWS account access to resources don't usually have, or grant users in one AWS account access to resources in another account.
  - Or allow a mobile app to use AWS resources, but you do not want to embed AWS keys within the app (difficult to rotate and can potentially extract).
  - grant AWS access to users who already have identities that are defined outside of AWS, such as in your corporate directory.
  - Or, to grant access to your account to third parties to perform an audit on your resources.
  - For all of these example use cases, IAM roles are an essential component to implementing the cloud deployment.

![Screen Shot 2020-05-05 at 14.45.29](https://i.imgur.com/ilJwlvj.png)

### Securing accounts

first create an AWS account, begin with a `single sign-in identity` that has complete access to all AWS services and resources in the account. the `AWS account root user`
- it is accessed by signing into the AWS Management Console with the email address and password
- have (and retain) full access to all resources in the account.
- do not use account root user credentials for day-to-day interactions with the account.

step 1: stop using the account root user
  - create an IAM user for yourself with AWS Management Console access enabled (do not attach any permissions to the user yet). Save the IAM user access keys if needed.
  - create an IAM group, give it a name (such as FullAccess), and attach IAM policies to the group that grant full access to at least a few of the services you will use.
  - add the IAM user to the group.
  - Disable and remove your account root user access keys, if they exist.
  - Enable a password policy for all users. Copy the IAM users sign-in link from the IAM Dashboard page. Then, sign out as the account root user.
  - Browse to the IAM users sign-in link that you copied, and sign in to the account by using your new IAM user credentials.
  - Store your account root user credentials in a secure place.

step2: enable  multi-factor authentication (MFA)
  - for the root/other IAM user logins.
  - You can also use MFA to control programmatic access.
  - for retrieving the MFA token needed to log in when MFA is enabled: `virtual MFA-compliant applications` (such as Google Authenticator and Authy Authenticator), `U2F security key devices`, and `hardware MFA options` that provide a key fob or display card.

step 3: AWS CloudTrail
- a service that logs all API requests to resources in your account. enables operational auditing on account.
- enabled by default on all AWS accounts
- keeps record of the last 90 days of account management event activity. create, modify, and delete operations of services that are supported by CloudTrail without needing to manually create another trail.

To enable CloudTrail log retention beyond the last 90 days and to enable alerting whenever specified events occur, create a new trail (which is described at a highlevel on the slide).

![Screen Shot 2020-05-05 at 16.33.42](https://i.imgur.com/D7DLTrs.png)

setp 4: enable billing report


#### AWS Organizations
an account management service that enables you to consolidate multiple AWS accounts into an organization and centrally manage.
- group accounts into organizational units(OUs) and attach different access policies to each OU.
- integrates with and supports IAM.
- provides `service control policies (SCPs)`
- on OU can have different account inside.


#### Service control policies (SCPs)
- offer central control over the `maximum available permissions` for all accounts in your organization, ensure that accounts stay in organization’s access control guidelines.
- SCPs are available only in an organization that has all features enabled, including consolidated billing.
- similar to IAM permissions policies
  - almost the same syntax.
  - However, SCP never grants permissions. but JSON policies that `specify the maximum permissions` for an organization or OU.
- Attaching an SCP to the organization root/organizational unit (OU) defines a safeguard for the actions that accounts in the organization root or OU can do.
- not a substitute for well-managed each account. still attach IAM policies to users and roles in organization's accounts to actually grant permissions to them.

#### AWS Key Management Service (AWS KMS)
- a service to *create and manage encryption keys*, *control the encryption* across a wide range of AWS services and applications.
- a secure and resilient service that uses hardware security modules (HSMs) that were validated under Federal Information ProcessingStandards (FIPS) 140-2 to protect your keys.
- integrates with `AWS CloudTrail` to logs of all key usage
- `Customer master keys (CMKs)` are used to control access to data encryption keys that encrypt and decrypt your data.
  - create new master keys when you want, manage who has access to these key, which services they can be used with.
  - can also import keys from your own key management infrastructure into AWS KMS. AWS KMS integrates with most AWS services, which means that you can use AWS KMS master keys to control the encryption of the data that you store in these services.

#### Amazon Cognito
- control access to AWS resources from your application.
  - define roles and map users to different roles, so application can access only the resources that are authorized for each user.
- uses common identity management standards, such as Security Assertion Markup Language (SAML) 2.0.
  - SAML: open standard for exchanging identity and security information with applications and service providers.
  - Applications and service providers that support SAML enable you to sign in by using your corporate directory credentials (user name and password from Microsoft Active Directory)
  - With SAML, you can use single sign-on (SSO) to sign in to all of your SAML-enabled applications by using a single set of credentials.
- meet multiple security and compliance requirements, including requirements for highly regulated organizations such as healthcare companies and merchants.
  - eligible for use with the US Health Insurance Portability and Accountability Act (HIPAA).
  - for workloads that are compliant with the Payment Card IndustryData Security Standard(PCI DSS); theAmerican Institute of CPAs (AICPA) Service Organization Control (SOC); the International Organization for Standardization (ISO) and International Electrotechnical Commission (IEC) standardsISO/IEC 27001,ISO/IEC 27017, and ISO/IEC 27018; andISO 9001

#### AWS Shield
- a managed distributed denial of service (DDoS) protection service that safeguards applications that run on AWS.
- provides always-on detection and automatic inline mitigations that minimize application downtime and latency, no need to engage AWS Support to benefit from DDoS protection.
  - helps protects your website from all types of DDoS attacks, including Infrastructure layer attacks (like UserDatagram Protocol—or UDP—floods), state exhaustion attacks (like TCP SYN floods), and application-layer attacks (like HTTP GET or POST floods).
- auto enabled at no additional cost.
  - AWS Shield Advanced: optional paid service.
    - provides additional protections against more sophisticated and larger attacks for your applications that run on Amazon EC2, Elastic Load Balancing, Amazon CloudFront, AWS Global Accelerator, and Amazon Route 53.
    - available to all customers. However, to contact the DDoS Response Team, customers need to have either Enterprise Support or Business Support from AWS Support.


### secure AWS data
Data encryption is an essential tool to protect digital data.

**data at rest**
- create encrypted file systems on AWS so that all your data and metadata is encrypted at rest by AdvancedEncryption Standard (AES)-256 encryption algorithm.

**data in transit**
- by using Transport Layer Security (TLS) 1.2 with an open standard AES-256 cipher.
  - `AWS Certificate Manager`: service to provision, manage, and deploy SSL/TLS certificates for use with AWS services and internal connected resources.
- traffic runs over Secure HTTP (HTTPS) is encrypted by using TLS or SSL.
  - protected against eavesdropping and man-in-the-middle attacks as bidirectional encryption of the communication.

![Screen Shot 2020-05-05 at 17.12.22](https://i.imgur.com/g9a4e71.png)

Amazon S3 buckets
- Amazon S3 buckets are private and can be accessed only by users who are explicitly granted access.
- when share data onS3
  - It is essential to manage and control access to Amazon S3 data.
- tools and options for controlling access to your S3 buckets or objects,
  - `Amazon S3 Block Public Access`. These settings override any other policies or object permissions. Enable Block Public Access for all buckets that not publicly accessible. avoiding unintended exposure of Amazon S3 data.
  - `IAM policies`: specify the users or roles that can access specific buckets and objects.
  - `bucket policies`: define access to specific buckets or objects. typically used when the user/system cannot authenticate by using IAM. Bucket policies can be configured to grant access across AWS accounts or to grant public or anonymous access to Amazon S3 data.
    - bucket policies should be written carefully and tested fully.
    - specify a deny statement in a bucket policy to restrict access. Access will be restricted even if the users have permissions that are granted in an identity-based policy that is attached to the users.
  - `Setting access control lists (ACLs)` on your buckets and objects. ACLs are less commonly used (ACLs predate IAM). If you do use ACLs, do not set access that is too open or permissive.
  - `AWS Trusted Advisor` provides a bucket permission check feature for discovering if any of the buckets in your account have permissions that grant global access.

### AWS compliance programs

AWS engages with external certifying bodies and independent auditors to provide customers with information about the policies, processes, and controls that are established and operated by AWS.

![Screen Shot 2020-05-05 at 17.37.51](https://i.imgur.com/v3zXuMZ.png)

**AWS Config**
- service to assess, audit, and evaluate the configurations of your AWS resources.
  - continuously monitors and records your AWS resource configurations
  - automate the evaluation of recorded configurations against desired configurations.
  - review changes in configurations and relationships between AWS resources,
  - review detailed resource configuration histories,
  - determine overall compliance against the configurations that are specified in your internal guidelines.
  - simplify compliance auditing, security analysis, change management, and operational troubleshooting.
- As you can see in the AWS Config Dashboard screen capture shown here, AWS Config keeps an inventory listing of all resources that exist in the account, and it then checks for configuration rule compliance and resource compliance. Resources that are found to be noncompliant are flagged, which alerts you to the configuration issues that should be addressed within the account.
- AWS Config is a `Regional service`.
  - track resources across Regions, enable it in every Region that you use.
  - AWS Config offers an aggregator feature that can show an aggregated view of resources across multiple Regions and even multiple accounts.

![Screen Shot 2020-05-05 at 17.45.47](https://i.imgur.com/7JDYhrX.png)

**AWS Artifact**
- provides on-demand downloads of `AWS security and compliance documents`,
  - such as AWS ISO certifications, Payment Card Industry (PCI), and Service Organization Control (SOC) reports.
  - submit the security and compliance documents (asaudit artifacts) to your auditors or regulators to demonstrate the security and compliance of the AWS infrastructure and services that you use.
- use these documents as guidelines
  - to evaluate your own cloud architecture and assess the effectiveness of your company's internal controls.
  - AWS Artifact provides documents about AWS only.
  - AWS customers are responsible for developing or obtaining documents that demonstrate the security and compliance of their companies.
- to review, accept, and track the status of AWS agreements such as the Business Associate Agreement (BAA).
  - A BAA typically is required for companies that are subject to HIPAA to ensure that protected health information (PHI) is appropriately safeguarded.
- With AWS Artifact, you can accept agreements with AWS and designate AWS accounts that can legally process restricted information. You can accept an agreement on behalf of multiple accounts. To accept agreements for multiple accounts, use AWS Organizations to create an organization.

---

## Module 5 Overview

![Screen Shot 2020-05-05 at 21.21.26](https://i.imgur.com/JJJO3yq.png)

### Amazon Virtual Private Cloud (Amazon VPC)
- a service provision a `logically isolated section` of the AWS Cloud (virtual private cloud/VPC) to launch your AWS resources.
  - control over your virtual networking resources: the selection of your own *IP address range*, the creation of *subnets*, and the configuration of *route tables and network gateways*.
  - can use both IPv4 and IPv6 in VPC for secure access to resources and applications.
  - can also customize the network configuration for VPC.
    - example,
    - create a public subnet for web servers that can access the public internet.
    - place backend systems (databases/application servers...) in a private subnet with no public internet access.
    - use multiple layers of security (security groups and network access control lists (networkACLs)...) to control access to Amazon Elastic Compute Cloud (Amazon EC2) instances in each subnet.
- A VPC is dedicated to your account.
- VPCs belong to a single AWS Region and can span multiple Availability Zones.
- a VPC can divide it into one or more subnets.

subnet
- a range of IP addresses in a VPC.
- Subnets belong to a single Availability Zone.
- Subnets are generally classified as public or private.
  - Public subnets have direct access to the internet
  - private subnet do not.

![Screen Shot 2020-05-05 at 21.37.13](https://i.imgur.com/279Kh2r.png)

### IP addressing

A common method to describe networks is `Classless Inter-Domain Routing (CIDR)`.

IP addresses enable resources in VPC to communicate with each other and with resources over the internet.

- When create a VPC, assign an CIDR block (a range of private addresses) to it.
  - After create a VPC, cannot change the address range
- The IPv4 CIDR block:
  - large as /16 (2^16, or 65,536 addresses)
  - small as /28 (2^4, or 16 addresses)
- associate an IPv6 CIDR block with your VPC and subnets, and assign IPv6 addresses from that block to the resources in your VPC.
  - IPv6 CIDR blocks have a different block size limit.
- The CIDR block of a subnet = the CIDR block for a VPC.
  - the VPC and the subnet are the same size (a single subnet in the VPC).
- the CIDR block of a subnet can be a subset < the CIDR block for the VPC.
  - This structure enables multiple subnets.
  - create more than one subnet in a VPC, the CIDR blocks of the subnets cannot overlap. You cannot have duplicate IP addresses in the same VPC.

- AWS reserves these IP addresses for:
  - 10.0.0.0 : Network address
  - 10.0.0.1 : VPC local router (internal communications)
  - 10.0.0.2 : Domain Name System (DNS) resolution
  - 10.0.0.3 : Future use
  - 10.0.0.255 : Network broadcast address
  - For example, create a subnet with an IPv4 CIDR block of 10.0.0.0/24 (which has 256 total IP addresses). The subnet has 256 IP addresses, but only 251 are available because five are reserved.

When create a VPC
*public IP*
- every instance in VPC gets a `public IP` address automatically
- can also request a `public IP` address to be assigned when create the instance by modifying the subnet’s `auto-assign public IP address properties`.


*Elastic IP* address
- a static and public IPv4 address designed for dynamic cloud computing.
- associate an Elastic IP address with any instance or network interface for any VPC in your account.
- With an Elastic IP address, you can mask the failure of an instance by rapidly `remapping` the address to another instance in your VPC.
- Associating the Elastic IP address with the network interface has an advantage over associating it directly with the instance.
- move all of the attributes of the network interface from one instance to another in a single step.
- Additional costs might apply

An *elastic network interface*
- a `virtual network interface` that can attach or detach from an instance in a VPC.
  - A `network interface's attributes` follow it when it is reattached to another instance.
  - move a network interface from one instance to another, `network traffic` is redirected to the new instance.
- Each instance in VPC has a default network interface (primary network interface), assigned a private IPv4 address from the IPv4 address range of your VPC.
  - cannot detach a primary network interface from an instance.
- can create and attach an additional network interface to any instance in your VPC. The number of network interfaces you can attach varies by instance type.

### VPC Networking

**internet gateway**
- a scalable, redundant, and highly available VPC component
- allows communication between instances in your VPC and the internet.
- serves two purposes:
  - to provide a target in your VPC route tables for internet-routable traffic
  - to perform network address translation for instances that were assigned public IPv4 addresses.
- To make a subnet public
  - attach an internet gateway to your VPC
  - and add a route to the route table: send non-local traffic through the internet gateway to the internet (0.0.0.0/0).

**network address translation (NAT) gateway**
enables instances in a private subnet to connect to the internet or other AWS services, but prevents the internet from initiating a connection with those instances.
1. create a NAT gateway
  - specify the public subnet which the NAT gateway should reside and associate the NAT gateway an Elastic IP address
2. After you create a NAT gateway, update the route table
  - route table associated to your private subnets to `point internet-bound traffic to the NAT gateway`.
3. Thus, instances in private subnets can communicate with the internet.

can also use NAT instance in public subnet in VPC instead of a NAT gateway.
- However, a `NAT gateway` is a managed NAT service that provides better availability, higher bandwidth, and less administrative effort.

**VPC sharing**
- share subnets with other AWS accounts in the same organization in AWS Organizations.
- enables multiple AWS accounts to create application resources into shared, centrally managed VPCs.
  - such as Amazon EC2 instances, Amazon Relational Database Service (Amazon RDS) databases, Amazon Redshift clusters, and AWS Lambda functions
- After a subnet is shared
  - the participants can view, create, modify, and delete their application resources in the subnets that are shared with them.
  - Participants cannot view, modify, or delete resources that belong to other participants or the VPC owner.

- VPC sharing offers several benefits:
  - Separation of duties: Centrally controlled VPC structure, routing, IP address allocation
  - Ownership: Application owners continue to own resources, accounts, and security groups
  - Security groups: VPC sharing participants can reference the security group IDs of each other
  - Efficiencies: Higher density in subnets, efficient use of VPNs and AWS Direct Connect
  - No hard limits: Hard limits can be avoided—for example, 50 virtual interfaces per AWS Direct Connect connection through simplified network architecture
  - Optimized costs: Costs can be optimized through the reuse of NAT gateways, VPC interface endpoints, and intra-Availability Zone traffic

**A VPC peering connection**
- a networking connection between two VPCs that enables you to route traffic between them privately.
- Instances in either VPC can communicate with each other as if they are within the same network.
- create a VPC peering connection
  - between your own VPCs, with a VPC in another AWS account, or in a different AWS Region.
  - you create rules in your route table to allow the VPCs to communicate with each other through the peering resource.
  - the destination: `IP of VPC `
  - the target: `the peering resource ID`

VPC peering has some restrictions:
- IP address ranges cannot overlap.
- Transitive peering is not supported.
  - three VPCs: A, B, and C.
  - VPC A connected to VPC B, VPC A connected to VPC C.
  - However, VPC B is not connected to VPC C implicitly.
  - To connect VPC B to VPC C, you must explicitly establish that connectivity.
- You can only have one peering resource between the same two VPCs.

**AWS site-to-site VPN**
- By default, instances launch in VPC cannot communicate with a remote network.
- To connect your VPC to your remote network (create a virtual private network or VPN connection)

1. Create a virtual gateway device (virtual private network (VPN) gateway) and attach it to your VPC.
2. Define the configuration of the VPN device or the customer gateway. The customer gateway is not a device but an AWS resource that provides information to AWS about your VPN device.
3. Create a custom route table to point corporate data center-bound traffic to the VPN gateway. You also must update security group rules.
4. Establish an AWS Site-to-Site VPN (Site-to-Site VPN) connection to link the two systems together.
5. Configure routing to pass traffic through the connection.


**AWS Direct Connect DX**
network performance can be affected if data center is located far away from your AWS Region.

`AWS Direct Connect`
- establish a dedicated, private network connection between your network and one of the DX locations.
- This private connection can reduce your network costs, increase bandwidth throughput, and provide a more consistent network experience than internet-based connections.
- DX uses open standard 802.1q virtual local area networks (VLANs).

A VPC endpointis a virtual device that enables you to privately connect your VPC to supported AWS services and VPC endpoint services that are powered by AWS PrivateLink. Connection to these services does not require an internet gateway, NAT device, VPN connection, or AWS Direct Connect connection. Instances in your VPC do not require public IP addresses to communicate with resources in the service. Traffic between your VPC and the other service does not leave the Amazon network. There are two types of VPC endpoints:•An interface VPC endpoint(interface endpoint) enables you to connect to services that are powered by AWS PrivateLink. These services include some AWS services, services that are hosted by other AWS customers and AWS Partner Network (APN) Partners in their own VPCs (referred to as endpoint services), and supported AWS Marketplace APN Partner services. The owner of the service is the service provider, and you—as the principal who creates the interface endpoint—are the service consumer. You are charged for creating and using an interface endpoint to a service. Hourly usage rates and data processing rates apply. See the AWS Documentation for a list of supported interface endpointsand for more information about the example shown here.•Gateway endpoints: The use of gateway endpoints incurs no additional charge. Standard charges for data transfer and resource usage apply.
















.
