---
title: AWS - Security - AWS Inspector (EC2)
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, Security]
tags: [AWS, Security]
toc: true
image:
---

[toc]

---

# AWS Inspector `what’s wrong > vulnerabilities for EC2`

![Screen Shot 2020-07-13 at 21.49.29](https://i.imgur.com/rYTKcT6.png)

- automated security assessment service
- assesses applications for <font color=red> exposure, vulnerabilities, and deviations from best practices </font>

- <font color=red> analyze the behavior of the resources and identify potential security issues </font>
  - Analyzes the VPC encironment for potential security issuse.
  - identify EC2 instances for common security vulnerabilities.
  - asses EC2s for vulnerabilities or deviations from best practices.

- helps <font color=red> improve the security and compliance </font> of applications deployed on AWS.

- Inspector uses a defined template and assesses the environment.
  - Providees the findings and recommends steps to resolve any potential security issues found.
  - define a collection of resources to include in the <font color=blue> assessment target </font>
  - then create an <font color=blue> assessment template </font> to launch a security assessment run of that target.
  - <font color=red> analyze EC2 instances against pre-defined security templates </font> to check for vulnerabilities

- Results is a <font color=red> detailed list of the security findings/issues </font> prioritized by level of severity!
  - The name of the assessment target, which includes the EC2 instance where this finding was registered
  - The name of the assessment template that was used to produce this finding
  - The assessment run start time, end time, and status
  - The name of the rules package that includes the rule that triggered this finding
  - The name of the finding
  - The <font color=blue> severity level </font> of severity of the finding
  - The description of the finding
  - <font color=blue> prioritized steps for remediation  </font>
  - findings can be reviewed directly or as part of detailed assessment reports which are available via the Amazon Inspector console or API.

- Amazon Inspector includes a <font color=red> knowledge base with hundreds of rules </font>
  - Use <font color=blue> rules packages to evaluate an application </font>
  - mapped to common <font color=blue> security compliance standards and vulnerability definitions </font>
    - whether remote root login is enabled
    - whether vulnerable software versions are installed.
    - check for unintended network accessibility and vulnerabilities on EC2 instances.
  - These rules are <font color=blue> regularly updated by AWS security researchers </font>
