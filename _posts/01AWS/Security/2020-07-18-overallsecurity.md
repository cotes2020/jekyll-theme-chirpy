---
title: AWS - Security - Cloud Proactive Security and Forensic Readiness
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, Security]
tags: [AWS, SecurityControl, ZeroTrust]
toc: true
image:
---

- [AWS Cloud: Proactive Security and Forensic Readiness](#aws-cloud-proactive-security-and-forensic-readiness)
  - [1. Identity and Access Management in AWS](#1-identity-and-access-management-in-aws)
    - [Best-practice checklist](#best-practice-checklist)
  - [2. Infrastructure-level protection in AWS](#2-infrastructure-level-protection-in-aws)
    - [Best-practice checklist](#best-practice-checklist-1)
  - [3. Data protection in AWS](#3-data-protection-in-aws)
    - [Best-practice checklist](#best-practice-checklist-2)
  - [4. Detective controls in AWS Cloud](#4-detective-controls-in-aws-cloud)
    - [best practice checklist](#best-practice-checklist-3)
  - [5. Incident Response in AWS](#5-incident-response-in-aws)
    - [Best-practice checklist](#best-practice-checklist-4)
- [security](#security)
  - [Zero Trust](#zero-trust)
      - [Designing on AWS using Zero Trust principles](#designing-on-aws-using-zero-trust-principles)
    - [AWS best practice architectures](#aws-best-practice-architectures)
    - [Apply Zero Trust](#apply-zero-trust)


ref:
- [web](https://cloudsecurityalliance.org/search/?page=1&s=AWS+Cloud%3A+Proactive+Security+and+Forensic+Readiness+)

---

# AWS Cloud: Proactive Security and Forensic Readiness

checklist for proactive security and forensic readiness in the AWS cloud environment.

Five-part best practice checklist
1. Identity and Access Management in AWS
1. Infrastructure Level Protection in AWS
1. Data Protection in AWS
1. Detective Controls in AWS
1. Incident Response in AWS

---


## 1. Identity and Access Management in AWS

AWS has numerous features that enable <font color=red> granular control for access </font> to an account’s resources by means of the Identity and Access Management (IAM) service.
- IAM provides control over
  - who can use AWS resources (authentication)
  - and how they can use those resources (authorization).

The following list focuses on
- limiting access to, and use of, root account and user credentials;
- defining roles and responsibilities of system users;
- limiting automated access to AWS resources;
- protecting access to data stored in storage buckets – including important data stored by services such as CloudTrail.

### Best-practice checklist

1. <font color=red> protect the access/use of AWS root account credentials </font>
   - Lock away AWS account (root) login credentials
   - Use multi-factor authentication (MFA) on root account
   - Make minimal use of root account (or no use of root account at all if possible).
     - Use IAM user instead to manage the account
   - Do not use AWS root account to create API keys.

2. defining roles and responsibilities of system users to <font color=red> control human access to the AWS Management Console and API </font>
   - Create individual IAM users
   - Configure a strong password policy for the users
   - Enable MFA for privileged users
   - Segregate defined roles and responsibilities of system users by creating user groups.
     - Use groups to assign permissions to IAM users
   - Clearly define and grant only the minimum privileges to users, groups, and roles that are needed to accomplish business requirements.
   - Use AWS defined policies to assign permissions whenever possible
   - Define and enforce user life-cycle policies
   - Use roles to delegate access to users, applications, or services that don’t normally have access to the AWS resources
   - Use roles for applications that run on Amazon EC2 instances
   - Use access levels (list, read, write and permissions management) to review IAM permissions
   - Use policy conditions for extra security
   - Regularly monitor user activity in the AWS account(s).

1. <font color=red> protecting the access/use of user account credentials </font>
   - <font color=blue> Rotate credentials regularly </font>
   - Remove/deactivate unnecessary credentials
   - Protect EC2 key pairs.
     - Password protect the .pem and .ppk file on user machines
   - Delete keys on the instances when someone leaves the organization or no longer requires access
   - Regularly run least privilege checks using <font color=blue> IAM user Access Advisor and IAM user Last Used Access Keys </font>
   - Delegate access by using roles instead of by sharing credentials
   - Use IAM roles for cross-account access and identity federation
   - Use temporary security instead of long-term access keys.

1. <font color=red> limiting automated access </font> to AWS resources
   - Use IAM roles for EC2 and an AWS SDK or CLI
   - Store static credentials securely that are used for automated access
   - Use instance profiles or Amazon STS for dynamic authentication
   - For increased security, implement alternative authentication mechanisms (e.g. LDAP or Active Directory)
   - Protect API access using Multi-factor authentication (MFA).

2. <font color=red> protecting CloudTrail logs stored in S3 </font>
   - Limit access to users and roles on a “need-to-know” basis for data stored in S3
   - Use <font color=blue> bucket access permissions and object access permissions </font> for fine-grained control over S3 resources
   - Use bucket policies to grant other AWS accounts or IAM
   - For more details, refer to the following AWS resources:

---

## 2. Infrastructure-level protection in AWS


Protecting any computing infrastructure requires a layered or defense-in-depth approach.
- The layers are typically divided into
  - physical,
  - network (perimeter and internal),
  - system (or host),
  - application,
  - and data.
- In an Infrastructure as a Service (IaaS) environment, AWS is responsible for security ‘of’ the cloud including the physical perimeter, hardware, compute, storage and networking
- while customers are responsible for security ‘in’ the cloud, or on layers above the hypervisor. This includes the operating system, perimeter and internal network, application and data.

Infrastructure protection requires defining:
- <font color=red> trust boundaries </font> (e.g., network boundaries and packet filtering),
- <font color=red> system security configuration and maintenance </font> (e.g., hardening and patching),
- <font color=red> operating system authentication and authorizations </font> (e.g., users, keys, and access levels),
- and <font color=red> other appropriate policy enforcement points </font> (e.g., web application firewalls and/or API gateways).

The key AWS service that supports <font color=red> service-level protection </font> is AWS Identity and Access Management (IAM) while Virtual Private Cloud (VPC) is the fundamental service that contributes to securing infrastructure hosted on AWS.
- VPC is the virtual equivalent of a traditional network operating in a data center, albeit with the scalability benefits of the AWS infrastructure. In addition, there are several other services or features provided by AWS that can be leveraged for infrastructure protection.

The following list mainly focuses on network and host-level boundary protection

### Best-practice checklist

1. <font color=red> enforcing network and host-level boundary protection </font>
   - Establish appropriate network design for workload
     - to ensure only desired network paths and routing are allowed
   - For large-scale deployments, design <font color=blue> network security in layers </font>
     - external, DMZ, and internal
   - NACL rules is stateless firewall, ensure to define both outbound and inbound rules
   - Create secure VPCs using network segmentation and security zoning
   - Carefully plan routing and server placement in public and private subnets.
   - Place instances (EC2 and RDS) within VPC subnets and <font color=blue> restrict access using security groups and NACLs </font>
   - Use non-overlapping IP addresses with other VPCs or data centre in use
   - Control network traffic by using
     - security groups (stateful firewall, outside OS layer),
     - NACLs (stateless firewall, at subnet level),
     - bastion host,
     - host based firewalls, etc.
   - Use Virtual Gateway (VGW) where Amazon VPC-based resources require remote network connectivity
   - Use IPSec or AWS Direct Connect
     - for trusted connections to other sites
   - Use VPC Flow Logs
     - for information about the IP traffic going to and from network interfaces in the VPC
   - Protect data in transit to ensure the confidentiality and integrity of data, as well as the identities of the communicating parties.


2. <font color=red> protecting against DDoS at network and application level </font>
   - Use firewalls including Security groups, network access control lists, and host based firewalls
   - Use rate limiting to protect scarce resources from overconsumption
   - Use <font color=blue> Elastic Load Balancing and Auto Scaling </font>
     - to configure web servers to scale out when under attack (based on load)
     - and shrink back when the attack stops
   - Use <font color=blue> AWS Shield </font>
     - a managed Distributed Denial of Service (DDoS) protection service, that safeguards web applications running on AWS
   - Use <font color=blue> Amazon CloudFront </font>
     - to absorb DoS/DDoS flooding attacks
   - Use <font color=blue> AWS WAF with AWS CloudFront </font>
     - help protect the web applications from common web exploits that could affect application availability, compromise security, or consume excessive resources
   - Use <font color=blue> Amazon CloudWatch </font>
     - to detect DDoS attacks against the application
   - Use <font color=blue> VPC Flow Logs </font>
     - to gain visibility into traffic targeting the application.


3. <font color=red> managing malware </font>
   - Give users the minimum privileges they need to carry out their tasks
   - Patch external-facing and internal systems to the latest security level.
   - Use a reputable and up-to-date antivirus and antispam solution on the system.
   - Install host based IDS with file integrity checking and rootkit detection
   - Use IDS/IPS systems
     - for statistical/behavioural or signature-based algorithms to detect and contain network attacks and Trojans.
   - Launch instances from trusted AMIs only
   - Only install and run trusted software from a trusted software provider (note: MD5 or SHA-1 should not be trusted if software is downloaded from random source on the internet)
   - Avoid SMTP open relay, which can be used to spread spam, and which might also represent a breach of the AWS Acceptable Use Policy.


4. <font color=red> identify vulnerability/misconfigurations in the os of EC2 </font>
   - Define approach for securing the system, consider the level of access needed and take a least-privilege approach
   - Open only the ports needed for communication, harden OS and disable permissive configurations
   - Remove or disable unnecessary user accounts.
   - Remove or disable all unnecessary functionality.
   - Change vendor-supplied defaults prior to deploying new applications.
   - Automate deployments and remove operator access
     - to reduce attack surface area using tools such as EC2 Systems Manager Run Command
   - Ensure operating system and application configurations, such as firewall settings and anti-malware definitions, are correct and up-to-date;
   - Use <font color=blue> EC2 Systems Manager State Manager </font>
     - to define and maintain consistent operating system configurations
   - Ensure an inventory of instances and installed software is maintained;
     - Use <font color=blue> EC2 Systems Manager Inventory </font>
       - to collect and query configuration about the instances and installed software
   - Perform routine vulnerability assessments when updates or deployments are pushed;
     - Use <font color=blue> Amazon Inspector </font>
       - to identify vulnerabilities or deviations from best practices in the guest operating systems and applications
   - Leverage automated patching tools such as EC2 Systems Manager Patch Manager to help you deploy operating system and software patches automatically across large groups of instances
   - Use <font color=blue> AWS CloudTrail, AWS Config, and AWS Config Rules </font>
     - as they provide audit and change tracking features for auditing AWS resource changes.
   - Use <font color=blue> template definition and management tools, including AWS CloudFormation </font>
     - to create standard, preconfigured environments.


5. <font color=red> protect the integrity of the os of EC2 instances </font>
   - Use file integrity controls for Amazon EC2 instances
   - Use host-based intrusion detection controls for Amazon EC2 instances
   - Use a custom Amazon Machine Image (AMI) or configuration management tools (such as Puppet or Chef) that provide secure settings by default.

6. <font color=red> ensure security of containers on AWS </font>
   - Run containers on top of virtual machines
   - Run small images, remove unnecessary binaries
   - Use many small instances to reduce attack surface
   - Segregate containers based on criteria such as role or customer and risk
   - Set containers to run as non-root user
   - Set filesystems to be read-only
   - Limit container networking;
     - Use <font color=blue> AWS ECS </font>
       - to manage containers and define communication between containers
   - Leverage Linux kernel security features using tools like SELinux, Seccomp, AppArmor
   - Perform vulnerability scans of container images
   - Allow only approved images during build
   - Use tools such as Docker Bench to automate security checks
   - Avoid embedding secrets into images or environment variables, Use S3-based secrets storage instead.

7. <font color=red> ensuring only trusted Amazon Machine Images (AMIs) are launched </font>
   - Treat shared AMIs as any foreign code that you might consider deploying in the own data centre and perform the appropriate due diligence
   - Look for description of shared AMI, and the AMI ID, in the Amazon EC2 forum
   - Check aliased owner in the account field to find public AMIs from Amazon.

8. <font color=red> creating secure custom (private or public) AMIs </font>
   - Disable root API access keys and secret key
   - Configure Public Key authentication for remote login
   - Restrict access to instances from limited IP ranges using Security Groups
   - Use bastion hosts
     - to enforce control and visibility
   - Protect the .pem file on user machines
   - Delete keys from the authorized_keys file on the instances when someone leaves the organization or no longer requires access
   - Rotate credentials (DB, Access Keys)
   - Regularly run least privilege checks using IAM user Access Advisor and IAM user Last Used Access Keys
   - Ensure that software installed does not use default internal accounts and passwords.
   - Change vendor-supplied defaults before creating new AMIs
   - Disable services and protocols that authenticate users in clear text over the network, or otherwise insecurely.
   - Disable non-essential network services on startup.
     - Only administrative services (SSH/RDP) and the services required for essential applications should be started.
   - Ensure all software is up to date with relevant security patches
   - For in instantiated AMIs, update security controls by running custom bootstrapping Bash or Microsoft Windows PowerShell scripts; or use bootstrapping applications such as Puppet, Chef, Capistrano, Cloud-Init and Cfn-Init
   - Follow a formalised patch management procedure for AMIs
   - Ensure that the published AMI does not violate the Amazon Web Services Acceptable Use Policy. Examples of violations include open SMTP relays or proxy servers.
   - Security at the infrastructure level, or any level for that matter, certainly requires more than just a checklist. For a comprehensive insight into infrastructure security within AWS, we suggest reading the following AWS whitepapers – AWS Security Pillar and AWS Security Best Practises.


---


## 3. Data protection in AWS


The checklist mainly focuses on protection of data (at rest and in transit), protection of encryption keys, removal of sensitive data from AMIs, and, understanding access data requests in AWS.

### Best-practice checklist
1. <font color=red> protecting data at rest </font>
   - Define polices for data classification, access control, retention and deletion
   - Tag information assets stored in AWS based on adopted classification scheme
   - Determine where the data will be located by selecting a suitable AWS region
   - Use geo restriction (or geoblocking),
     - to prevent users in specific geographic locations from accessing content that you are distributing through a CloudFront web distribution
   - Control the format, structure and security of the data by masking, making it anonymised or encrypted in accordance with the classification
   - Encrypt data at rest using server-side or client-side encryption
   - Manage other access controls, such as identity, access management, permissions and security credentials
   - Restrict access to data using IAM policies, resource policies and capability policies


2. <font color=red> protecting data at rest on Amazon S3 </font>
   - Use <font color=blue> bucket-level or object-level permissions </font> alongside IAM policies
   - <font color=blue> Don’t create any publicly accessible S3 buckets </font>
     - Instead, create pre-signed URLs to grant time-limited permission to download the objects
   - encrypt
     - Protect sensitive data by <font color=blue> encrypting data at rest </font> in S3.
       - Amazon S3 supports server-side encryption and client-side encryption of user data,
       - using which you create and manage the own encryption keys
     - <font color=blue> Encrypt inbound and outbound S3 data traffic </font>
   - <font color=blue> data replication and versioning </font> instead of automatic backups.
     - Implement S3 Versioning and S3 Lifecycle Policies
   - Automate the lifecycle of the S3 objects with rule-based actions
   - Enable MFA Delete on S3 bucket
   - <font color=blue> enable logging </font>
   - Be familiar with the durability and availability options for different S3 storage types – S3, S3-IA and S3-RR.


3. <font color=red> protecting data at rest on Amazon EBS </font>
   - only use encrypted EBS volume
     - encrypt data, snapshots, and disk I/O using the customary AWS-256 algorithm
   - active VPC Flow log
   - AWS creates two copies of the EBS volume for redundancy.
     - However, since both copies are in the same Availability Zone, replicate data at the application level, and/or create backups using EBS snapshots
   - On Windows Server 2008 and later:
     - use <font color=blue> BitLocker encryption </font>
     - to protect sensitive data stored on system or data partitions (this needs to be configured with a password as Amazon EC2 does not support Trusted Platform Module (TPM) to store keys)
   - On Windows Server
     - implement <font color=blue> Encrypted File System (EFS) </font>
     - to further protect sensitive data stored on system or data partitions
   - On Linux instances running kernel versions 2.6 and later
     - use <font color=blue> dmcrypt and Linux Unified Key Setup (LUKS) </font>
     - for key management


4. <font color=red> protecting data at rest on Amazon RDS </font>
   - (Note: Amazon RDS leverages the same secure infrastructure as Amazon EC2. You can use the Amazon RDS service without additional protection, but it is suggested to encrypt data at application layer)
   - Use built-in encryption function that encrypts all sensitive database fields, using an application key, before storing them in the database
   - Use platform level encryption
   - Use MySQL cryptographic functions – encryption, hashing, and compression
   - Use Microsoft Transact-SQL cryptographic functions – encryption, signing, and hashing
   - Use Oracle Transparent Data Encryption on Amazon RDS for Oracle Enterprise Edition under the Bring Your Own License (BYOL) model


5. <font color=red> protecting data at rest on Amazon Glacier </font>
   - Data stored on Amazon Glacier is protected using server-side encryption.
   - AWS generates separate unique encryption keys for each Amazon Glacier archive, and encrypts it using AES-256
   - Encrypt data prior to uploading it to Amazon Glacier for added protection

6. <font color=red> protecting data at rest on Amazon DynamoDB </font>
   - DynamoDB is a shared service from AWS and can be used without added protection
   - implement a data encryption layer over the standard DynamoDB service
   - Use raw binary fields or Base64-encoded string fields, when storing encrypted fields in DynamoDB

7. <font color=red> protecting data at rest on Amazon EMR </font>
   - Store data permanently on Amazon S3 only, and do not copy to HDFS at all. Apply server-side or client-side encryption to data in Amazon S3
   - Protect the integrity of individual fields or entire file (for example, by using HMAC-SHA1) at the application level while you store data in Amazon S3 or DynamoDB
   - Or, employ a combination of Amazon S3 server-side encryption and client-side encryption, as well as application-level encryption

8. <font color=red> protecting data in transit </font>
   - Encrypt data in transit
     - using IPSec ESP and/or SSL/TLS
   - Encrypt all non-console administrative access using strong cryptographic mechanisms using SSH, user and site-to-site IPSec VPNs, or SSL/TLS to further secure remote system management
   - Authenticate data integrity
     - using IPSec ESP/AH, and/or SSL/TLS
   - Authenticate remote end
     - using IPSec with IKE with pre-shared keys or X.509 certificates
     - using SSL/TLS with server certificate authentication based on the server common name(CN), or Alternative Name (AN/SAN)
   - Offload HTTPS processing on Elastic Load Balancing to minimise impact on web servers
   - Protect the backend connection to instances using an application protocol such as HTTPS
   - On Windows servers use X.509 certificates for authentication
   - On Linux servers, use SSH version 2 and use non-privileged user accounts for authentication
   - Use HTTP over SSL/TLS (HTTPS) for connecting to RDS, DynamoDB over the internet
   - Use SSH for access to Amazon EMR master node
   - Use SSH for clients or applications to access Amazon EMR clusters across the internet using scripts
   - Use SSL/TLS for Thrift, REST, or Avro

9. <font color=red> managing and protecting encryption keys </font>
   - Define key rotation policy
   - Do not hard code keys in scripts and applications
   - Securely manage keys
     - at server side (SSE-S3, SSE-KMS) or at client side (SSE-C)
   - Use tamper-proof storage
     - such as Hardware Security Modules (AWS CloudHSM)
   - Use a key management solution
     - from the AWS Marketplace or from an APN Partner. (e.g., SafeNet, TrendMicro, etc.)

10. <font color=red> ensuring custom Amazon Machine Images (AMIs) are secure and free of sensitive data </font> before publishing for internal (private) or external (public) use
    - Securely delete all sensitive data including AWS credentials, third-party credentials and certificates or keys from disk and configuration files
    - Delete log files containing sensitive information
    - Delete all shell history on Linux

11. <font color=red> understand who has the right to access the data stored in AWS </font>
    - Understand the applicable laws to the business and operations
      - consider whether laws in other jurisdictions may apply
    - Understand that relevant government bodies may have rights to issue requests for content, each relevant law will contain criteria that must be satisfied for the relevant law enforcement body to make a valid request.
    - Understand that AWS notifies customers where practicable before disclosing their data so they can seek protection from disclosure, unless AWS is legally prohibited from doing so or there is clear indication of illegal conduct regarding the use of AWS services. For additional information, visit Amazon Information Requests Portal.



---

## 4. Detective controls in AWS Cloud
AWS detective controls include
- processing of logs and monitoring of events
- that allow for auditing, automated analysis, and alarming.

These controls can be implemented using
- AWS CloudTrail logs to record AWS API calls, Service-specific logs (for Amazon S3, Amazon CloudFront, CloudWatch logs, VPC flow logs, ELB logs, etc)
- and AWS Config to maintain a detailed inventory of AWS resources and configuration.
- Amazon CloudWatch is a monitoring service for AWS resources and can be used to trigger CloudWatch events to automate security responses.
- Another useful tool is Amazon GuardDuty which is a managed threat detection service in AWS and continuously monitors for malicious or unauthorized.


### best practice checklist

1. using <font color=red> Trusted Advisor  </font>
   - to check for security compliance.

2. <font color=red> capturing and storing logs  </font>
   - Activate AWS Cloud Trail.
   - Collect logs from various locations/services including
     - AWS APIs and user-related logs (e.g. AWS CloudTrail),
     - AWS service-specific logs (e.g. Amazon S3, Amazon CloudFront, CloudWatch logs, VPC flow logs, ELB logs, etc.)
     - operating system-generated logs,
     - IDS/IPS logs
     - and third-party application-specific logs
   - Use services and features such as AWS CloudFormation, AWS OpsWorks, or Amazon Elastic Compute Cloud (EC2) user data, to ensure that instances have agents installed for log collection
   - Move logs periodically from the source either directly into a log processing system (e.g., CloudWatch Logs) or stored in an Amazon S3 bucket for later processing based on business needs

3. <font color=red> analyzing logs </font>
   - Parse and analyse security data using solutions
   - such as <font color=blue> AWS Config, AWS CloudWatch, Amazon EMR, Amazon Elasticsearch Service, etc. </font>
   - Perform analysis and visualization with Kibana.

4. <font color=red> retaining logs  </font>
   - Store data centrally using Amazon S3, and, for long-term archiving if required, using Amazon Glacier
   - Define data-retention lifecycle for logs.
     - By default, CloudWatch logs are kept indefinitely and never expire.
     - You can adjust the retention policy for each log group, keeping the indefinite retention, or choosing a retention period between 10 years and one day
   - Manage log retention automatically using AWS Lambda.

5. <font color=red> receiving notification and alerts </font>
   - Use Amazon CloudWatch Events for routing events of interest and information reflecting potentially unwanted changes into a proper workflow
   - Use Amazon GuardDuty to continuously monitor for malicious or unauthorized behavior
   - Send events to targets like an AWS Lambda function, Amazon SNS, or other targets for alerts and notifications

6. <font color=red> monitoring billing in the AWS account </font>
   - Use detailed billing to monitor the monthly usage regularly
   - Use consolidated billing for multiple accounts


---

## 5. Incident Response in AWS

NIST defines a security incident as “an occurrence that actually or potentially jeopardises the confidentiality, integrity, or availability of an information system or the information the system processes, stores, or transmits or that constitutes a violation or imminent threat of violation of security policies, security procedures, or acceptable use policies”.

The figure below outlines the typical phases of an incident response lifecycle.

![Screen Shot 2020-12-15 at 2.45.07 PM](https://i.imgur.com/NgsIePn.png)


there are several tools in the AWS cloud environment to help the incident response process,
- such as AWS CloudTrail, Amazon CloudWatch, AWS Config, AWS CloudFormation, AWS Step Functions, etc.
- These tools enable you to track, monitor, analyse, and audit events.

- Audit logs are treasure troves and are indispensable during investigations.
  - AWS provides detailed audit logs that record important events such as file access and modification.
- Events can be automatically processed and trigger tools that automate responses through the use of AWS APIs.
- You can pre-provision tooling and a “clean room” which allows you to carry out forensics in a safe, isolated environment.


### Best-practice checklist

1. <font color=red> ensure an appropriate incident response strategy in place </font>
   - Make sure the security team has the right tools pre-deployed into AWS so that the incident can be responded to in a timely manner.
   - Pre-provision a ‘clean room’ for automated incident handling.
   - Have a list of relevant contacts that may need to be notified.
   - Decide on the medium of communication. If the compromised account contains personal data, you may be required to contact the Data Protection Commission (DPC) within 72 hours to comply with GDPR.
   - Conduct incident response simulations regularly in the non-production and the production environments as well. Incorporate lessons learned into the architecture and operations.

2. <font color=red> AWS tools for prepare in advance for incident handling </font>
   - Tags in AWS allow you to proactively label resources with a data classification or a criticality attribute so you can quickly estimate the impact when the incident occurs.
   - <font color=blue> AWS Organisations </font>
     - allows you to create separate accounts along business lines or mission areas which also limits the “blast radius” should a breach occur;
     - for governance, you can apply policies to each of those sub accounts from the AWS master account.
   - <font color=blue> IAM </font>
     - grants appropriate authorisation to incident response teams in advance.
   - <font color=blue> Security Groups </font>
     - enables isolation of Amazon EC2 instances.
   - <font color=blue> AWS Cloud Formation </font>
     - automates the creation of trusted environments for conducting deeper investigations.
   - <font color=blue> AWS CloudTrail </font>
     - provides a history of AWS API calls that can assist in response and trigger automated detection and response systems.
   - <font color=blue> VPC Flow Logs </font>
     - enables you to capture information about the IP traffic going to and from network interfaces in the VPC.
   - <font color=blue> AWS Key Management Service (KMS) </font>
     - encrypts sensitive data at rest including logs aggregated and stored centrally.
   - <font color=blue> Amazon GuardDuty </font>
     - is a managed threat detection service that continuously monitors for malicious or unauthorised behaviour.
   - <font color=blue> Amazon CloudWatch Events </font>
     - triggers different automated actions from changes in AWS resources including CloudTrail.
   - <font color=blue> Amazon S3 </font>
     - stores snapshots and related incident artefacts.
   - <font color=blue> AWS Step Functions </font>
     - coordinates a sequence of steps to automate an incident response process.
   - APIs automate many of the routine tasks that need to be performed during incident handling.

3. <font color=red> respond to AWS abuse warnings </font>
   - Set up a dedicated security communication email address.
   - Do not ignore abuse warnings. Take action to stop the malicious activities, and prevent future re-occurrence.
   - Open a case number with AWS Support for cross-validation.

4. <font color=red> isolate and restrict user access to a compromised Amazon EC2 instance </font>
   - containing the instance manually,
     - use IAM to restrict access permissions to compromised Amazon EC2 instance.
     - Isolate the instance using restrictive ingress and egress security group rules or remove it from a load balancer.
   - Tag the instance as appropriate to indicate isolation.
   - Create snapshots of EBS volumes.
   - Notify relevant contacts.
   - Use CloudFormation
     - to quickly create a new, trusted environment in which to conduct deeper investigation.
   - automate the above steps using Lambda, Step Functions, CloudFormation and SNS Topic to prepare an EC2 auto clean room for containing the instance.
   - You could also use aws-security-automation code on GitHub, which is a collection of scripts and resources for DevSecOps, Security Automation and Automated Incident Response Remediation.

5. <font color=red> ensure sensitive information is wiped post investigation </font>
   - Secure wipe-files and delete any KMS data keys, if used.


---

# security
## Zero Trust


Traditional network security relies on a **secure perimeter**
- everything within the perimeter is trusted and anything outside the perimeter is not.


Zero Trust
- a model where application components or microservices are considered discrete from each other and no component or microservice trusts any other.
- This manifests as a security posture designed to consider input from any source as potentially malicious.
- It starts with not trusting the underlying internal network fabric, and extending to things such as input and output validation at every microservice. Additional efforts can include designing a defense-in-depth approach to protect against individual components, microservices, or identities compromise.
- A Zero Trust network evaluates all actions and resources in real time to reduce the risk of unintended access to business data and sensitive resources.


Zero Trust building a network boundary between each microservice in the architecture.
- strengthening the component perimeters; rethink threat sources and investment to protect against them.

#### Designing on AWS using Zero Trust principles

In threat modeling, users attempt to determine all of the potential attack possibilities to define risk and identify mitigations.

One threat model that can be used for illustrative purposes, [STRIDE](https://en.wikipedia.org/wiki/STRIDE_(security)) , identifies threats in these categories:

* **Spoofing** of user identity
* **Tampering** with data
* **Repudiation** the source
* **Information** disclosure
* **Denial** of service
* **Elevation** of privilege

### AWS best practice architectures

AWS offers foundational tools for designing Well-Architected applications on AWS.
- The AWS Well-Architected Framework includes strategies to help you compare the workload against AWS best practices, and obtain guidance to produce stable and efficient systems.
- The Well-Architected Framework contains five distinct pillars, including one for [security](https://d1.awsstatic.com/whitepapers/architecture/AWS-Security-Pillar.pdf) .




well architected example

- a [web application](https://d0.awsstatic.com/whitepapers/aws-web-hosting-best-practices.pdf)
- The architecture represented is a well architected and secure design for the web application.

[![Figure 1. An example of a web hosting architecture on AWS ](https://d2908q01vomqb2.cloudfront.net/9e6a55b6b4563e652a23be9d623ca5055c356940/2020/01/20/Zero-Trust-Figure-1.png)](https://d2908q01vomqb2.cloudfront.net/9e6a55b6b4563e652a23be9d623ca5055c356940/2020/01/20/Zero-Trust-Figure-1.png)

The system is protected against common attack vectors leveraging the following services:
1. **Load balancing with [Elastic Load Balancing (ELB)](https://aws.amazon.com/elasticloadbalancing/) / [Application Load Balancer (ALB)](https://aws.amazon.com/blogs/aws/new-aws-application-load-balancer/)**
   1. spread loads across multiple AZs and EC2 Auto Scaling groups for **redundancy** and **decoupling** of services.

2. [**Virtual firewalls using AWS security groups**](https://docs.aws.amazon.com/vpc/latest/userguide/VPC_SecurityGroups.html)
    1. moves security to the instance to provide a stateful, host-level firewall for both web and application servers.

3. **Domain Name System (DNS) services with [Amazon Route 53](https://aws.amazon.com/route53/)**
   1. provides DNS services to simplify domain management.

4. **Edge caching with [Amazon CloudFront](https://aws.amazon.com/cloudfront/)**
   1. decreases the latency to customers.

5. **Edge security for [Amazon CloudFront](https://aws.amazon.com/cloudfront/) with [AWS Web Application Firewall (AWS WAF)](https://aws.amazon.com/waf/)**
   1. filters malicious traffic, including XSS and SQL injection via customer-defined rules.

6. **DDoS attack protection with [AWS Shield](https://aws.amazon.com/shield/)**
   1. safeguards the infrastructure against the most common network and transport layer DDoS attacks automatically.

### Apply Zero Trust

Let’s reevaluate the architecture, but protect each component as a microservice instead of as part of a larger trusted system.

In a Zero Trust model:

- protect against tampering and information disclosure by a SQL Injection attack using the AWS WAF service.
  - By design, customers using the website will come through Amazon CloudFront to access both static and dynamic content. While it makes sense to apply the AWS WAF rules to the CloudFront distribution, ELB/ALB will use a public IP address that could be discoverable by someone else.
  - One mitigation would be to apply the same WAF rules directly against the load balancer.

- protection between the web server and app server tiers?
  - Those are traditionally considered “internal” components and data flowing between them is not subject to the same scrutiny.
  - However, the Zero Trust model requires all components and communications be considered untrusted.
  - AWS WAF might not be the right solution depending on the communication methods, but another layer of filtering – either host-based or network-based – would be implemented to do input validation before the input is ingested by the app tier.
  - Additionally, authentication and authorization of commands between these two tiers would be continuous similar to how AWS uses the [AWS Signature Version 4](https://docs.aws.amazon.com/AmazonS3/latest/API/sig-v4-authenticating-requests.html) for API signing.


- AWS WAF rules and local input validation are effective at protecting against some attacks


- but what about DDoS?
  - AWS Shield protects against the most common volumetric and state exhaustion attacks, but you should evaluate potential threats coming from other elements of the system.
  - The best practice architecture does not address the potential of a web server instance flooding the application server with valid but meaningless work, or addresses an inadvertent misconfigured security group.

    * Implement additional metrics and monitoring so a consistent amount of traffic flows from each instance.

    * Implement [Amazon CloudWatch Anomaly Detection](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch_Anomaly_Detection.html) to use machine learning (ML) algorithms to analyze specific metrics such as Amazon EC2 instances generating unusually large amounts of network traffic.

    * Use the [alarm to notify](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Create_Anomaly_Detection_Alarm.html) an Amazon SNS topic, which will then trigger a custom Lambda function that removes the offending Amazon EC2 instance for the auto-scaling group, stops it, and isolates it for further analysis.

[![Figure 2. An example of a Zero Trust web hosting architecture on AWS](https://d2908q01vomqb2.cloudfront.net/9e6a55b6b4563e652a23be9d623ca5055c356940/2020/01/20/ZeroTrustDiagram-2.png)](https://d2908q01vomqb2.cloudfront.net/9e6a55b6b4563e652a23be9d623ca5055c356940/2020/01/20/ZeroTrustDiagram-2.png)

_Figure 2. An example of a Zero Trust web hosting architecture on AWS_

example

- a web tier
  - creates backups, use an AWS KMS key that only that instance role has `KMS:Encrypt` permissions on.
  - Since the web tier shouldn’t need to decrypt its own backups, deny or omit KMS:Decrypt to that role.
  - Since that instance role is the only entity with the ability to use the KMS key to encrypt data, and coupled with CloudTrail logs for auditing, you can validate that backups were written by those instances and have not been tampered with.
  - If those instances are accessed by an unauthorized user, they cannot read from past backups.

- You could also
  - add [user authentication between tiers on the Application Load Balancer](https://aws.amazon.com/blogs/aws/built-in-authentication-in-alb/)
  - or use [API Gateway between the App Tier and the databases](https://aws.amazon.com/blogs/database/query-the-aws-database-from-the-serverless-application/) to execute validation of queries.


.
