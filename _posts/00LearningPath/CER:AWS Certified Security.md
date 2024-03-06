
- [SCS-C01 AWS Certified Security](#scs-c01-aws-certified-security)
  - [Introduction](#introduction)
  - [Exam content](#exam-content)
  - [Content outline](#content-outline)
  - [Official Practice Question Set](#official-practice-question-set)


# SCS-C01 AWS Certified Security

## Introduction
The AWS Certified Security – Specialty (SCS-C01) exam is intended for individuals who perform a security role.

The exam validates a candidate’s ability to effectively demonstrate knowledge about securing the AWS platform.

The exam validates whether a candidate has the following:
- An understanding of specialized data classifications and `AWS data protection mechanisms`
- An understanding of `data-encryption methods` and AWS mechanisms to implement them
- An understanding of `secure internet protocols` and AWS mechanisms to implement them
- A working knowledge of AWS security services and features of services to provide a secure production environment
- Competency from 2 or more years of production deployment experience in using AWS security
services and features
- The ability to make tradeoff decisions with regard to `cost, security, and deployment complexity` to meet a set of application requirements
- An understanding of `security operations and risks`


Target candidate description
The target candidate should have 5 years of IT security experience in designing and implementing security solutions. Additionally, the target candidate should have 2 or more years of hands-on experience in securing AWS workloads.

Recommended AWS knowledge
The target candidate should have the following knowledge:
- The AWS shared responsibility model and its application
- Security controls for workloads on AWS
- Logging and monitoring strategies
- Cloud security threat models
- Patch management and security automation
- Ways to enhance AWS security services with third-party tools and services
- Disaster recovery controls, including BCP and backups
- Encryption
- Access control
- Data retention

What is considered out of scope for the target candidate?
The following is a non-exhaustive list of related job tasks that the target candidate is not expected to be able to perform. These items are considered out of scope for the exam:
- Create or write configurations
- Implement (SysOps)
- Demonstrate scripting in a specific language (for example, Perl or Java)

For a detailed list of specific tools and technologies that might be covered on the exam, as well as lists of in-scope and out-of-scope AWS services, refer to the Appendix.


## Exam content

**Response types**
There are two types of questions on the exam:
- Multiple choice: Has one correct response and three incorrect responses (distractors)
- Multiple response: Has two or more correct responses out of five or more response options

Select one or more responses that best complete the statement or answer the question. Distractors, or incorrect answers, are response options that a candidate with incomplete knowledge or skill might choose. Distractors are generally plausible responses that match the content area.

Unanswered questions are scored as incorrect; there is no penalty for guessing. The exam includes 50 questions that will affect your score.

**Unscored content**
The exam includes 15 unscored questions that do not affect your score. AWS collects information about candidate performance on these unscored questions to evaluate these questions for future use as scored questions. These unscored questions are not identified on the exam.

**Exam results**
The AWS Certified Security – Specialty (SCS-C01) exam is a pass or fail exam. The exam is scored against a minimum standard established by AWS professionals who follow certification industry best practices and guidelines.
Your results for the exam are reported as a scaled score of 100–1,000. The minimum passing score is 750. Your score shows how you performed on the exam as a whole and whether or not you passed. Scaled scoring models help equate scores across multiple exam forms that might have slightly different difficulty levels.
Your score report could contain a table of classifications of your performance at each section level. This information is intended to provide general feedback about your exam performance. The exam uses a compensatory scoring model, which means that you do not need to achieve a passing score in each section. You need to pass only the overall exam.
Each section of the exam has a specific weighting, so some sections have more questions than other sections have. The table contains general information that highlights your strengths and weaknesses. Use caution when interpreting section-level feedback.
Version 2.0 SCS-C01 2 | PAGE


## Content outline
This exam guide includes weightings, test domains, and objectives for the exam. It is not a comprehensive listing of the content on the exam. However, additional context for each of the objectives is available to help guide your preparation for the exam. The following table lists the main content domains and their weightings. The table precedes the complete exam content outline, which includes the additional context. The percentage in each domain represents only scored content.

Domain 1: Incident Response 12%
Domain 2: Logging and Monitoring 20%
Domain 3: Infrastructure Security 26%
Domain 4: Identity and Access Management 20%
Domain 5: Data Protection 22%
TOTAL 100%

---

Domain 1: Incident Response

1.1 Given an AWS abuse notice, <font color=red> evaluate the suspected compromised instance or exposed access keys </font>.
- Given an AWS Abuse report about an EC2 instance, <font color=blue> securely isolate the instance </font> as part of a
forensic investigation.
- <font color=blue> Analyze logs </font> relevant to a reported instance to verify a breach, and collect relevant data.
- <font color=blue> Capture a memory dump </font> from a suspected instance for later deep analysis or for legal compliance reasons.

1.2 Verify that the <font color=red> Incident Response plan </font> includes relevant AWS services.
- Determine <font color=blue> if changes to baseline security configuration </font> have been made.
- Determine <font color=blue> if list omits services, processes, or procedures </font> which facilitate Incident Response.
- Recommend services, processes, procedures to remediate gaps.

1.3 Evaluate the configuration of <font color=red> automated alerting, and execute possible remediation </font> of security-related incidents and emerging issues.
- <font color=blue> Automate evaluation of conformance </font> with rules for new/changed/removed resources.
- Apply <font color=blue> rule-based alerts </font> for common infrastructure misconfigurations.
- Review previous security incidents and recommend improvements to existing systems.

---

Domain 2: Logging and Monitoring

2.1 <font color=red> Design and implement </font> security monitoring and alerting.
- Analyze architecture and identify <font color=blue> monitoring requirements </font> and sources for monitoring statistics.
- Analyze architecture to determine which AWS services can be used to automate monitoring and alerting.
- Analyze the requirements for <font color=blue> custom application monitoring </font>, and determine how this could be achieved.
- Set up <font color=blue>  </font> to perform regular audits.

2.2 <font color=red> Troubleshoot </font> security monitoring and alerting.
- Given an occurrence of a known event without the expected alerting, analyze the service functionality and configuration and remediate.
- Given an occurrence of a known event without the expected alerting, analyze the permissions and remediate.
- Given a custom application which is not reporting its statistics, analyze the configuration and remediate.
- Review audit trails of system and user activity.

2.3 <font color=red> Design and implement </font> a logging solution.
- Analyze architecture and identify <font color=blue> logging requirements </font> and sources for log ingestion.
- Analyze requirements and implement <font color=blue> durable and secure log storage </font> according to AWS best
practices.
- Analyze architecture to determine which AWS services can be used to <font color=blue>  </FONT>.

2.4 <font color=red> Troubleshoot </font> logging solutions.
- Given the absence of logs, determine the incorrect configuration and define remediation steps.
- Analyze logging access permissions to determine incorrect configuration and define remediation steps.
- Based on the security policy requirements, determine the correct log level, type, and sources.

---

Domain 3: Infrastructure Security

3.1 Design <font color=red> edge security </font> on AWS.
- For a given workload, assess and limit the attack surface.
- Reduce blast radius (e.g. by distributing applications across accounts and regions).
- Choose appropriate AWS and/or third-party edge services such as WAF, CloudFront and Route 53 to protect against DDoS or filter application-level attacks.
- Given a set of edge protection requirements for an application, evaluate the mechanisms to prevent and detect intrusions for compliance and recommend required changes.
- Test WAF rules to ensure they block malicious traffic.


3.2 Design and implement a <font color=red> secure network infrastructure </FONT>.
- Disable any unnecessary network ports and protocols.
- Given a set of edge protection requirements, evaluate the security groups and NACLs of an application for compliance and recommend required changes.
- Given security requirements, decide on network segmentation (e.g. security groups and NACLs) that allow the minimum ingress/egress access required.
- Determine the use case for VPN or Direct Connect.
- Determine the use case for enabling VPC Flow Logs.
- Given a description of the network infrastructure for a VPC, analyze the use of subnets and gateways for secure operation.

3.3 <font color=red> Troubleshoot </font> a secure network infrastructure.
- Determine where network traffic flow is being denied.
- Given a configuration, confirm security groups and NACLs have been implemented correctly.


3.4 Design and implement <font color=red> host-based security </font>.
- Given security requirements, install and configure host-based protections including Inspector, SSM.
- Decide when to use host-based firewall like iptables.
- Recommend methods for host hardening and monitoring.


---

Domain 4: Identity and Access Management

4.1 Design and implement a <font color=red> scalable authorization and authentication sys. em</font> to access AWS resources.
- Given a description of a workload, analyze the access control configuration for AWS services and make recommendations that reduce risk.
- Given a description how an organization manages their AWS accounts, verify security of their root user.
- Given your organization’s compliance requirements, determine when to apply user policies and resource policies.
- Within an organization’s policy, determine when to federate a directory services to IAM.
- Design a scalable authorization model that includes users, groups, roles, and policies.
- Identify and restrict individual users of data and AWS resources.
- Review policies to establish that users/systems are restricted from performing functions beyond their responsibility, and also enforce proper separation of duties.

4.2 <font color=red> Troubleshoot </font> an authorization and authentication system to access AWS resources.
- Investigate a user’s inability to access S3 bucket contents.
- Investigate a user’s inability to switch roles to a different account.
- Investigate an Amazon EC2 instance’s inability to access a given AWS resource.

---

Domain 5: Data Protection

5.1 Design and implement key management and use.
- Analyze a given scenario to determine an appropriate key management solution.
- Given a set of data protection requirements, evaluate key usage and recommend required changes.
- Determine and control the blast radius of a key compromise event and design a solution to contain the same.

5.2 Troubleshoot key management.
- Break down the difference between a KMS key grant and IAM policy.
- Deduce the precedence given different conflicting policies for a given key.
- Determine when and how to revoke permissions for a user or service in the event of a compromise.

5.3 Design and implement a data encryption solution for data at rest and data in transit.
- Given a set of data protection requirements, evaluate the security of the data at rest in a workload and recommend required changes.
- Verify policy on a key such that it can only be used by specific AWS services.
- Distinguish the compliance state of data through tag-based data classifications and automate remediation.
- Evaluate a number of transport encryption techniques and select the appropriate method (i.e. TLS, IPsec, client-side KMS encryption).

Appendix
Which key tools, technologies, and concepts might be covered on the exam?
The following is a non-exhaustive list of the tools and technologies that could appear on the exam. This list is subject to change and is provided to help you understand the general scope of services, features, or technologies on the exam. The general tools and technologies in this list appear in no particular order. AWS services are grouped according to their primary functions. While some of these technologies will likely be covered more than others on the exam, the order and placement of them in this list is no indication of relative weight or importance:
- AWS CLI
- AWS SDK
- AWS Management Console
- Network analysis tools (packet capture and flow captures)
- SSH/RDP
- Signature Version 4
- TLS
- Certificate management
- Infrastructure as code (IaC)

AWS services and features
Note: Security affects all AWS services. Many services do not appear in this list because the overall service is out of scope, but the security aspects of the service are in scope. For example, a candidate for this exam would not be asked about the steps to set up replication for an S3 bucket, but the candidate might be asked about configuring an S3 bucket policy.
Management and Governance:
- AWS Audit Manager
- AWS CloudTrail
- Amazon CloudWatch
- AWS Config
- AWS Organizations
- AWS Systems Manager
- AWS Trusted Advisor
Networking and Content Delivery:
- Amazon Detective
- AWS Firewall Manager
- AWS Network Firewall
- AWS Security Hub
- AWS Shield
- Amazon VPC
  - VPC endpoints
  - Network ACLs
  - Security groups
- AWS WAF
  Security, Identity, and Compliance:
- AWS Certificate Manager (ACM)
- AWS CloudHSM
- AWS Directory Service
- Amazon GuardDuty
- AWS Identity and Access Management (IAM)
- Amazon Inspector
- AWS Key Management Service (AWS KMS)
- Amazon Macie
- AWS Single Sign-On

Out-of-scope AWS services and features
The following is a non-exhaustive list of AWS services and features that are not covered on the exam. These services and features do not represent every AWS offering that is excluded from the exam content. Services or features that are entirely unrelated to the target job roles for the exam are excluded from this list because they are assumed to be irrelevant.
Out-of-scope AWS services and features include the following:
- Application development services
- IoT services
- Machine learning (ML) services
- Media services
- Migration and transfer services


## Official Practice Question Set

1. [link](https://explore.skillbuilder.aws/learn/course/external/view/elearning/12473/aws-certified-security-specialty-official-practice-question-set-scs-c01-english)

















.
