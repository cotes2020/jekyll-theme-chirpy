---
title: AWS - Management - cloud governance on AWS
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, Management]
tags: [AWS]
toc: true
image:
---

- [cloud governance on AWS](#cloud-governance-on-aws)
  - [overview](#overview)
  - [value of the NIST CSF](#value-of-the-nist-csf)
  - [NIST CSF](#nist-csf)
    - [NIST CSF use case with identity](#nist-csf-use-case-with-identity)
  - [The organizational context: AWS Cloud Adoption Framework](#the-organizational-context-aws-cloud-adoption-framework)
    - [AWS CAF use case with identity](#aws-caf-use-case-with-identity)
  - [Secure and resilient system architecture: AWS Well-Architected Framework](#secure-and-resilient-system-architecture-aws-well-architected-framework)
    - [AWS Well-Architected use case with identity](#aws-well-architected-use-case-with-identity)
  - [Putting it all together](#putting-it-all-together)

---

# cloud governance on AWS


- ref:
  - [Optimizing cloud governance on AWS: Integrating the NIST Cybersecurity Framework, AWS Cloud Adoption Framework, and AWS Well-Architected](https://aws.amazon.com/blogs/security/optimizing-cloud-governance-on-aws-integrating-the-nist-cybersecurity-framework-aws-cloud-adoption-framework-and-aws-well-architected/)

---

## overview

- Integrating the NIST Cybersecurity Framework, AWS Cloud Adoption Framework, and AWS Well-Architected
- approach to security governance, risk management, and compliance can be an enabler to digital transformation and business agility.
- many customers establish a security foundation using technology **agnostic risk management frameworks**
  - such as the <font color=red> National Institute of Standards and Technology (NIST) Cybersecurity Framework (CSF) </font>
- to understand their organization’s current capabilities, set goals, and develop a plan to improve and maintain security posture. However, you still need the right model to optimize security outcomes in the cloud.
- To <font color=red> adapt the security program for the cloud, AWS developed two tools </font>
  - <font color=red> AWS Cloud Adoption Framework (CAF) and AWS Well-Architected Framework </font>.
  - complement the risk-based foundation with the AWS CAF, <font color=blue> integrate the organizational business drivers at scale as you move to the cloud </font>
  - and, when ready to implement specific workloads,use the AWS Well-Architected Framework to <font color=blue> design, measure, and improve the technical implementation </font>.



## value of the NIST CSF

- the value and use of the NIST CSF as a framework to
  - establish the security objectives,
  - assess the organization’s current capabilities,
  - and develop a plan to improve and maintain the desired security posture.
- use the AWS CAF t begin the digital transformation journey in the AWS Cloud with strategies around organizational practices and governance at scale that align to the business drivers.
- AWS Well-Architected Framework can enable security best practices at the workload level.

using these three complementary frameworks can optimize the security outcomes. While they can be used independently, each builds upon the other to strengthen and mature the cloud environment and organizational security program.
- Using the AWS Cloud Adoption Framework (CAF) and AWS Well-Architected Framework to help meet NIST Cybersecurity Framework (CSF) Objectives and Achieve a Target Profile

![CSF-CAF-WA-Graphicrev](https://i.imgur.com/t7lp7No.png)

to use CAF and AWS Well-Architected to help meet NIST CSF objectives, process involves the following steps:
- **Establish** the organization’s `cybersecurity governance` and `desired security outcomes` with the NIST CSF using the Core functions and **implementation** Tiers to create the target profile.
- **Prepare** for cloud migration and **implement** a scalable foundation using AWS CAF to map those capabilities in the cloud
- **Measure and improve** the security architecture and operational practices with AWS Well-Architected and select the AWS services to support the security needs.


## NIST CSF

![NIST](https://d2908q01vomqb2.cloudfront.net/22d200f8670dbdb3e253a90eee5098477c95c23d/2021/04/08/Optimizing-Cloud-Governance-2small.png)

- Establish the security governance and desired security outcomes
- using a framework for the organizational security program

[NIST CSF](https://www.nist.gov/cyberframework), an internationally recognized risk management framework
- The CSF provides a simple and effective method for understanding and communicating security risk across the organization.
- Its technology and industry-agnostic approach allows for an outcome-based common taxonomy that you can use across the business, from the board level to the technical teams.
- We continue to see accelerating adoption of the CSF across industries and countries, and its principles are becoming standardized approaches, as we see in the latest ISO 27103:2018 and draft ISO 27101 standards.

The NIST CSF consists of three elements — <font color=red> Core, Tiers, and Profiles.</font>
- The Core
  - includes five continuous functions: <font color=blue> Identify, Protect, Detect, Respond, and Recover </font>
  - you can map to other standards or control requirements as required by the business. 
- The Tiers
  - characterize an organization’s aptitude and maturity for managing the CSF functions and controls,
- The Profiles
  - intended to convey the organization’s “as is” and “to be” security postures.
- Together, these three elements are designed to enable the organization to prioritize and address security risks consistent with the business and mission needs. 

### NIST CSF use case with identity

Unlike the process for building **on-premises networks and datacenters** that start with `physical facilities, computer and storage hardware, and a network perimeter` to protect what is being built out,
- **adopting the cloud** starts with `identity and access management` with the chosen cloud service provider.

For AWS, this means creating an AWS account and leveraging AWS IAM to create users and groups, define roles, and assign permissions and policies.
- NIST CSF five functions—Identify, Protect, Detect, Respond, and Recover.
- If we look at the Protect function as an example, there are 7 subcategories under the <font color=red> Identity Management, Authentication and Access Control (PR.AC) category </font>:
  - **PR.AC-1:** `Identities and credentials` are issued, managed, verified, revoked, and audited for authorized devices, users and processes
  - **PR.AC-2**: `Physical access` to assets is managed and protected
  - **PR.AC-3:** `Remote access` is managed
  - **PR.AC-4:** `Access permissions and authorizations` are managed, incorporating the principles of least privilege and separation of duties
  - **PR.AC-5:** `Network integrity` is protected (e.g., network segregation, network segmentation)
  - **PR.AC-6:** `Identities are proofed and bound to credentials` and asserted in interactions
  - **PR.AC-7:** `Users, devices, and other assets are authenticated` (e.g., single-factor, multi-factor) commensurate with the risk of the transaction (e.g., individuals’ security and privacy risks and other organizational risks)

> PR.AC-2 is a good example of how the shared responsibility model comes into play.
> With most cloud services, physical security is implemented and managed by the cloud service provider and you get to inherit those controls.
> This is true for AWS, with the exception that when you utilize a hybrid cloud service, such as AWS Snowball or AWS Outposts,
> - where we will ship a physical device for you to use in the on-prem environment or in the field, you are responsible for their physical security while the physical device is in the custody.
> For the purpose of this blog, however, we will focus on non-hybrid AWS cloud services, and so we will skip PR.AC-2 for this use case. Customers retain the responsibility to manage the physical security of their datacenters and their physical assets that connect to and access cloud services.


## The organizational context: AWS Cloud Adoption Framework

AWS Cloud Adoption Framework – _Prepare the organization for the cloud_

Cloud computing introduces a significant shift in <font color=blue> how technology is procured, accessed, used, and managed </font>.
- To operationalize and optimize the security program for the cloud, the organization needs to `understand the new paradigm, and update skills, adapt existing processes, and introduce new processes`.

The [AWS Cloud Adoption Framework (CAF)](https://aws.amazon.com/professional-services/CAF/) 
- helps organizations plan for a successful cloud migration, and not just the technical aspects for a single application lift-and-shift, but with the intent to establish an organizational foundation to facilitate deploying, operating, and securing workloads at scale.
- This may include:
  - establishing a DevSecOps culture and processes,
  - training staff and incorporating new paradigms into assignments and work,
  - building shared cloud infrastructure and management service environments,
  - implementing central governance and logging,
  - and other aspects that will integrate with individual applications and use cases.
- Each organization’s path will be different, so it’s important to plan ahead and connect the business goals and desired security outcomes to the right processes and technologies.

![Figure 2: CAF perspectives](https://d2908q01vomqb2.cloudfront.net/22d200f8670dbdb3e253a90eee5098477c95c23d/2021/04/01/Optimizing-Cloud-Governance-4.jpg)

**CAF perspectives**
- comprised of six perspectives used for planning and strategic considerations
  - based on principles that apply to most organizations.
  - 3 focus on the organization: <font color=blue> Business, People, and Governance </font>
  - technical aspects are considered in the <font color=blue> Platform, Security, and Operations perspectives </font>.
  - As NIST CSF, all these perspectives influence management of security risks and help achieve the security outcomes. 
- Using the AWS CAF, structure the security program to `meet the desired outcomes with agility, scale, speed, and innovation` that comes with AWS. 
- AWS CAF helps customers operationalize the security goals through 4 principles: <font color=blue> Directive, Preventive, Detective, and Responsive </font>.
  - **Directive** principle provides guidance to <font color=blue> understand the environment and data in the cloud </font>
  - **Preventive** provides guidance to <font color=blue> operate selected security controls in AWS </font>;
  - **Detective** provides a means to <font color=blue> analyze the environment and alert on anomalies and risks </font>;
  - **Responsive** looks to <font color=blue> mitigate detected risks, with an emphasis on automation </font>


**AWS CAF Security Perspective**
- comprised of 5 core + 5 augmenting security epics
- Consistent with the principles of the NIST CSF
  - organization’s foundational capabilities focus on identifying, applying, and scaling security best practices at the program and organizational levels to support business outcomes. 
- Security epics begin with identity and access management as the backbone to secure cloud deployment.

![Security epics](https://d2908q01vomqb2.cloudfront.net/22d200f8670dbdb3e253a90eee5098477c95c23d/2021/04/01/Optimizing-Cloud-Governance-5.png)


### AWS CAF use case with identity

few AWS services being applied and configured to govern IAM at scale.
- 3 tiers to consider when <font color=red> designing and building the IAM security </font>
  - Implement <font color=blue> IAM Guardrails, Operationalize IAM, and Privileged Access Management </font>. 

AWS shift the mindset from “locking down” a system: `implies inflexibility that can impact usability and business agility`, to the concept of “guardrails” where `security is defined by outer limits that allow freedom of movement within those constraints`.
- allows for more flexibility to explore new methods and technologies to **meet dynamic market changes**

Specifically for AWS IAM, `implementing guardrails` through services such as **AWS Organizations, AWS IAM, and AWS Control Tower.**
- For example
  - **AWS Control Tower**: provides the easiest way to `set up and govern a new, secure, multi-account AWS environment based on best practices` established through AWS’ experience, working with thousands of enterprises as they move to the cloud.
  - Next, operationalize IAM by **federating with an existing directory service, or creating a cloud directory**, and `implementing account and access control lifecycles`.
  - Finally, explore options to `implement a privileged access management (PAM) solution` to protect these important accounts.
    - **AWS Secrets Manager** and **Systems Manager Sessions Manager**, 2 services that can assist with this objective.
  - Using this small excerpt of the AWS CAF, and with the input into the process, you can design the AWS IAM to meet the NIST CSF subcategories PR.AC-1, 3, 4, 6, and 7 highlighted above.

![CAF Identity sprint](https://d2908q01vomqb2.cloudfront.net/22d200f8670dbdb3e253a90eee5098477c95c23d/2021/04/07/Optimizing-Cloud-Governance-4r.png)



## Secure and resilient system architecture: AWS Well-Architected Framework

The [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- Using AWS Well-Architected Framework to <font color=red> measure and improve the workload architecture </font>
  - helps to understand considerations and key decision points for building systems on AWS
  - a framework for guiding and evaluating the workload architectures.
- learn architectural best practices for designing and operating reliable, secure, efficient, and cost-effective systems in the cloud.
- It provides a way to consistently measure the architectures against best practices and identify areas for improvement.
- The process for `reviewing an architecture` is a **constructive conversation** about architectural decisions and having AWS Well-Architected systems increases the likelihood of business success.

**AWS Well-Architected Tool**
- To assist customers in documenting and measuring their workloads
- a questionnaire available on the AWS Management Console that helps you answer, “Am I well-architected?”
  - use AWS Well-Architected for designing, evaluating, and continuously improving the architectures.
  - After preparing, planning, and scaling for cloud migration using the **Cloud Adoption Framework**, **AWS Well-Architected** can inform how you secure specific workloads in line with the security outcomes (and Target Profile) applied from the <font color=red> NIST CSF </font>.

- focuses on the workload level: <font color=red> the infrastructure, systems, data, and processes </font>
- by examining five core pillars:
  - Operational Excellence,
  - Security,
  - Reliability,
  - Performance Efficiency,
  - and Cost Optimization.

![Figure 5: Well-Architected Tool](https://d2908q01vomqb2.cloudfront.net/22d200f8670dbdb3e253a90eee5098477c95c23d/2021/04/01/Optimizing-Cloud-Governance-8.png)


### AWS Well-Architected use case with identity
- 5 areas in the security pillar of the AWS Well-Architected Framework:
  - Identity and Access Management (IAM),
  - Detection,
  - Infrastructure Protection,
  - Data Protection,
  - and Incident Response.
- AWS Well-Architected **provides guidance for secure implementation and approaches** for `selecting the right AWS services to put these core security practices in place in the workloads`.
- these areas are similar to AWS CAF security perspective.
  - That’s because those capabilities that were identified at the strategic level should be addressed at the technical layer.
  - That traceability from business requirement to technical strategy to technical architecture and operations is a crucial element to make sure security is applied at all levels of the organization, and that it is meeting a business need.

read through the IAM of the AWS Well-Architected security pillar
- will see some workload-level best practices for activities
  - like `multi-factor authentication, temporary credentials, auditing, and least privilege`.
- Following these guidelines to meet NIST CSF subcategories 1, 3, 4, 6, and 7 for individual workloads and applications, uniquely for each if needed.
- For example,
  - **PR.AC-1** includes an objective to `audit for authorized devices, users, and services`.
    - Although there are several options to implement auditing, and areas to focus on, one of the **AWS Well-Architected best practices** is to `audit and rotate credentials periodically`.
    - Following the general guidance below along with prescriptive AWS service guidance, to design the workload to help meet this requirement.
    - In support of this use case, **AWS Well-Architected** recommends that you
      - transition from user and group permissions to the use of inherited roles for human and machine principals,
      - retire long-term credentials and access keys for temporary credentials and MFA where appropriate,
      - then use automation to verify controls are enforced.

## Putting it all together

Using **NIST CSF**, **AWS CAF**, and **AWS Well-Architected**
- tailor the approach to incorporate security management best practices for the cloud journey.
- These three frameworks offer `related, but distinct lenses` on how to approach security for the organization, connecting business goals and outcomes to the security program.
  - NIST CSF -> develop an `organizational understanding` to managing security risks.
  - AWS CAF -> `plan the cloud security approach and map activities` to security controls operating in the cloud and scale them throughout the organization. helps build out the architecture.
  - AWS Well-Architected -> `consistently measure` the workload against best practices and identify areas for improvement.
