---
title: LLMs Paper - Methodology for incident response on Gen-AI workloads
date: 2023-04-24 11:11:11 -0400
description:
categories: [51AI, Paper]
# img: /assets/img/sample/rabbit.png
tags: [LLM]
---

- [Methodology for incident response on Gen-AI workloads](#methodology-for-incident-response-on-gen-ai-workloads)
  - [Components of a Gen-AI workload](#components-of-a-gen-ai-workload)
  - [Prepare for incident response on Gen-AI workloads](#prepare-for-incident-response-on-gen-ai-workloads)
  - [Methodology for incident response on Gen-AI workloads](#methodology-for-incident-response-on-gen-ai-workloads-1)
    - [Access](#access)
    - [Infrastructure changes](#infrastructure-changes)
    - [AI changes](#ai-changes)
    - [Data store changes](#data-store-changes)
    - [Invocation](#invocation)
    - [Private data](#private-data)
    - [Agency](#agency)
    - [Example incident](#example-incident)
      - [AI changes](#ai-changes-1)
      - [Data store changes](#data-store-changes-1)

ref:
- https://aws.amazon.com/blogs/security/methodology-for-incident-response-on-generative-ai-workloads/

---

# Methodology for incident response on Gen-AI workloads

The AWS [Customer Incident Response Team (CIRT)](https://docs.aws.amazon.com/whitepapers/latest/aws-security-incident-response-guide/understand-aws-response-teams-and-support.html#aws-customer-incident-response-team) has developed a methodology to `investigate security incidents involving Gen-AI-based applications`.

To respond to security events related to a Gen-AI workload, you should still follow the guidance and principles outlined in the [AWS Security Incident Response Guide](https://docs.aws.amazon.com/whitepapers/latest/aws-security-incident-response-guide/aws-security-incident-response-guide.html)

Methodology for incident response on Gen-AI workloads

- seven elements to consider when triaging and responding to a security event on a Gen-AI workload.

---

## Components of a Gen-AI workload

![Screenshot 2024-10-03 at 09.16.26](/assets/img/Screenshot%202024-10-03%20at%2009.16.26.png)

Generative AI applications include the following five components:

- **An organization**: owns or is responsible for infrastructure, Gen-AI applications, and the organization’s private data.
- **Infrastructure**: isn’t specifically related to the Gen-AI application itself, include databases, backend servers, and websites.

- **Generative AI applications**, which include the following:

  - **Foundation models** – AI models with a large number of parameters and trained on a massive amount of diverse data.
  - **Custom models** – models that are fine-tuned or trained on an organization’s specific data and use cases, tailored to their unique requirements.
  - **Guardrails** –` mechanisms or constraints` to help make sure that the Gen-AI application operates within desired boundaries. Examples include content filtering, safety constraints, or ethical guidelines.
  - **Agents** – workflows that enable Gen-AI applications to perform multistep tasks across company systems and data sources.
  - **Knowledge bases** – repositories of domain-specific knowledge, rules, or data that the Gen-AI application can access and use.
  - **Training data** – data used to train, fine-tune, or augment the Gen-AI application’s models, including data for techniques such as [retrieval augmented generation (RAG)](https://aws.amazon.com/what-is/retrieval-augmented-generation/)
    > **Note** : Training data is distinct from an organization’s _private data._ A Gen-AI application might not have direct access to private data, although this is configured in some environments.
  - **Plugins** – additional software components or extensions that you can integrate with the Gen-AI application to provide specialized functionalities or access to external services or data sources.

- **Private data** refers to the customer’s privately stored, confidential data that the Gen-AI resources or applications aren’t intended to interact with during normal operation.

- **Users** are the identities that can interact with or access the Gen-AI application. They can be human or non-human (such as machines).

---

## Prepare for incident response on Gen-AI workloads

You should prepare for a security event across three domains: [people, process, and technology](https://docs.aws.amazon.com/whitepapers/latest/aws-security-incident-response-guide/preparation.html).

- [preparation items](https://docs.aws.amazon.com/whitepapers/latest/aws-security-incident-response-guide/preparation-summary.html) from the Security Incident Response Guide.

Preparation for a security event that’s related to a Gen-AI workload should include the following:

- **People: Train incident response and security operations staff on Gen-AI**:

  - make sure that the staff is familiar with Gen-AI concepts and with the AI/ML services in use at the organization.
  - [AWS Skill Builder](https://skillbuilder.aws/) provides both free and paid courses on both of these subjects.

- **Process: Develop new playbooks**:
  - develop new playbooks for security events that are related to a Gen-AI workload.
    - [Responding to Amazon Bedrock Security Events](https://github.com/aws-samples/aws-customer-playbook-framework/blob/main/docs/Bedrock_Response.md)
    - [Responding to SageMaker Security Events](https://github.com/aws-samples/aws-customer-playbook-framework/blob/main/docs/Responding%20to%20SageMaker.md)
    - [Responding to Amazon Q Security Events](https://github.com/aws-samples/aws-customer-playbook-framework/blob/main/docs/Amazon_Q.md).
    You can use these playbooks as a starting point and modify them to best fit the organization and usage of these services.
- **Technology: Log Gen-AI application prompts and invocations**:

  - foundational logs, such as those available in [AWS CloudTrail](https://aws.amazon.com/cloudtrail/)
  - logging [Amazon Bedrock](https://aws.amazon.com/bedrock/) model invocation logs so that you can analyze the prompts coming into the application and the outputs.
  - To learn more, see [Amazon Bedrock model invocation logging](https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html).
  - CloudTrail [data event](https://docs.aws.amazon.com/awscloudtrail/latest/userguide/logging-data-events-with-cloudtrail.html#logging-data-events) logging is also available for Amazon Bedrock, [Amazon Q](https://aws.amazon.com/q/), and [Amazon SageMaker](https://aws.amazon.com/sagemaker/).
  - For general guidance, see [Logging strategies for security incident response](https://aws.amazon.com/blogs/security/logging-strategies-for-security-incident-response/).

> **Important:** Logs can contain sensitive information. To help protect this information, you should
>
> - set up least privilege access to these logs.
> - protect sensitive log data with [data masking](https://aws.amazon.com/what-is/data-masking/). In [Amazon CloudWatch](https://aws.amazon.com/cloudwatch/), you can mask data natively through [log group data protection policies](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/mask-sensitive-log-data.html).

---

## Methodology for incident response on Gen-AI workloads

After the preparation items, use the _Methodology for incident response on Gen-AI_ _workloads_ for active response, to rapidly triage an active security event involving a Gen-AI application.

- The methodology has seven elements, each element describes a method by which the components can interact with another component or a method by which a component can be modified.
  - Access
  -
- Consideration of these elements will help guide the actions during the [Operations](https://docs.aws.amazon.com/whitepapers/latest/aws-security-incident-response-guide/operations.html) phase of a security incident, which includes detection, analysis, containment, eradication, and recovery phases.

---

### Access

- Determine the designed or intended access patterns for the organization that hosts the components of the Gen-AI application, and look for deviations or anomalies from those patterns. Consider whether the application is accessible externally or internally because that will impact the analysis.

- determine whether an organization has access to their AWS account.
  - If the password for the AWS account root user was lost or changed, [reset the password](https://docs.aws.amazon.com/IAM/latest/UserGuide/reset-root-password.html), enable [multi-factor authentication (MFA) device](https://docs.aws.amazon.com/IAM/latest/UserGuide/enable-virt-mfa-for-root.html) for the root user—this should block a threat actor from accessing the root user.


- identify anomalous and potential unauthorized access to the environment,
  - use [Amazon GuardDuty](https://aws.amazon.com/guardduty/).
  - If the application is accessible externally, the threat actor might not be able to access the AWS environment directly and thus GuardDuty won’t detect it. The way that you’ve set up authentication to the application will drive how you detect and analyze unauthorized access.

- If evidence of unauthorized access to the AWS account or associated infrastructure exists
    - determine the scope of the unauthorized access, such as the associated privileges and timeline.

    - If the unauthorized access involves service credentials, for example, [Amazon Elastic Compute Cloud (Amazon EC2)](https://aws.amazon.com/ec2/) instance credentials, review the service for vulnerabilities.

    - determine whether unauthorized access to the account persists.

      - to identify mutative actions logged by [AWS Identity and Access Management (IAM)](https://aws.amazon.com/iam/) and [AWS Security Token Service (Amazon STS)](https://docs.aws.amazon.com/STS/latest/APIReference/welcome.html), see the [Analysis](https://github.com/aws-samples/aws-customer-playbook-framework/blob/main/docs/Compromised_IAM_Credentials.md#analysis) section of the [Compromised IAM Credentials playbook](https://github.com/aws-samples/aws-customer-playbook-framework/blob/main/docs/Compromised_IAM_Credentials.md) on GitHub.
      - make sure that access keys aren’t stored in public repositories or in the application code; for alternatives, see [Alternatives to long-term access keys](https://docs.aws.amazon.com/IAM/latest/UserGuide/security-creds-programmatic-access.html#security-creds-alternatives-to-long-term-access-keys).

---

### Infrastructure changes


- Review the supporting infrastructure, such as servers, databases, serverless computing instances, and internal or external websites, to determine if it was accessed or changed.

- To investigate infrastructure changes, you can analyze CloudTrail logs for modifications of in-scope resources, or analyze other operating system logs or database access logs.

Analyze the infrastructure changes of an application

- [the control plane and data plane](https://docs.aws.amazon.com/whitepapers/latest/aws-fault-isolation-boundaries/control-planes-and-data-planes.html).
  - example, imagine that [Amazon API Gateway](https://aws.amazon.com/api-gateway/) was used for authentication to the downstream components of the Gen-AI application and that other ancillary resources were interacting with the application.

- have additional logging to be turned on to review changes made on the operating system of the resource.

---

### AI changes

Unauthorized changes can include, but are not limited to, system prompts, application code, guardrails, and model availability.

- Investigate whether users have accessed components of the Gen-AI application and whether they made changes to those components.

- Look for signs of unauthorized activities, such as the creation or deletion of custom models, modification of model availability, tampering or deletion of Gen-AI logging capabilities, tampering with the application code, and removal or modification of Gen-AI guardrails.

---

### Data store changes


Typically, you use and access a data store and knowledge base through model invocation.


- Determine the designed or intended data access patterns, whether users accessed the data stores of the Gen-AI application, and whether they made changes to these data stores.

- look for the addition or modification of agents to a Gen-AI application.

  - if an unauthorized user gains access to the environment, they can create, change, or delete the data sources and knowledge bases that the Gen-AI applications integrate with.
  - This could cause data or model exfiltration or destruction, as well as data poisoning, and could create a denial-of-service condition for the model.

---

### Invocation


- Analyze invocations of Gen-AI models, including the strings and file inputs, for threats, such as prompt injection or malware. You can use the [OWASP Top 10 for LLM](https://owasp.org/www-project-top-10-for-large-language-model-applications/) as a starting point to understand invocation related threats, and you can use invocation logs to analyze prompts for suspicious patterns, keywords, or structures that might indicate a prompt injection attempt.

- The logs also capture the model’s outputs and responses, enabling `behavioral analysis` to help identify uncharacteristic or unsafe model behavior indicative of a prompt injection. You can use the timestamps in the logs for temporal analysis to help detect coordinated prompt injection attempts over time and collect information about the user or system that initiated the model invocation, helping to identify the source of potential exploits.

Amazon Bedrock uses specific APIs to register [model invocation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html#model-invocation-logging-console).

- When a model in Amazon Bedrock is invoked, CloudTrail logs it.

- However, to determine the prompts that were sent to the Gen-AI model and the output response that was received from it, you must have configured` model invocation logging`.

  - crucial because they can reveal important information, such as whether a threat actor tried to get the model to divulge information from the data stores or release data that the model was trained or fine-tuned on.

  - For example, the logs could reveal if a threat actor attempted to prompt the model with carefully crafted inputs that were designed to extract sensitive data, bypass security controls, or generate content that violates the policies.
  - Using the logs, learn whether the model was used to generate misinformation, spam, or other malicious outputs that could be used in a security event.

> **Note** : For services such as Amazon Bedrock, invocation logging is disabled by default. recommend enable data events and model invocation logging for Gen-AI services, where available. However, the organization might not want to capture and store invocation logs for privacy and legal reasons. One common concern is users entering sensitive data as input, which widens the scope of assets to protect. This is a business decision that should be taken into consideration.

---

### Private data

- Determine whether the in-scope Gen-AI application was designed to have access to private or confidential data. Then look for unauthorized access to, or tampering with, that data.

From an architectural standpoint, Gen-AI applications shouldn’t have direct access to an organization’s private data.

- should classify data used to train a Gen-AI application or for RAG use as data store data and segregate it from private data, unless the Gen-AI application uses the private data (for example, in the case where a Gen-AI application is tasked to answer questions about medical records for a patient).

- One way to help make sure that an organization’s private data is segregated from Gen-AI applications is to [use a separate account](https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/sec_securely_operate_multi_accounts.html) and to authenticate and authorize access as necessary to adhere to the principle of least privilege.

---

### Agency


- Agency refers to the ability of applications to make changes to an organization’s resources or take actions on a user’s behalf.

  - For example, a Gen-AI application might be configured to generate content that is then used to send an email, invoking another resource or function to do so.

- determine whether the Gen-AI application has the ability to invoke other functions. Then, investigate whether unauthorized changes were made or if the Gen-AI application invoked unauthorized functions.

- [Excessive agency](https://genai.owasp.org/llmrisk/llm08-excessive-agency/) for an LLM refers to `an AI system that has too much autonomy or decision-making power`, leading to unintended and potentially harmful consequences. This can happen when an LLM is deployed with insufficient oversight, constraints, or alignment with human values, resulting in the `model making choices that diverge from what most humans would consider beneficial or ethical`.



The following table lists some questions to help you address the seven elements of the methodology. Use the answers to guide the response.

| Topic                  | Questions to address                                                                                                                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Access                 | Do you still have access to the computing environment? <br> Is there continued evidence of unauthorized access to the organization?                                                              |
| Infrastructure changes | Were supporting infrastructure resources accessed or changed?                                                                                                                                    |
| AI changes             | Were the AI models, code, or resources accessed or changed?                                                                                                                                      |
| Data store changes     | Were the data stores, knowledge bases, agents, plugins, or training data accessed or tampered with?                                                                                              |
| Invocation             | What data, strings, or files were sent as input to the model? <br> What prompts were sent? <br> What responses were produced?                                                                    |
| Private data           | What private or confidential data do Gen-AI resources have access to? <br> Was private data changed or tampered with?                                                                            |
| Agency                 | Can the Gen-AI application resources be used to start computing services in an organization, or do the Gen-AI resources have the authority to make changes? <br> Were unauthorized changes made? |

---

### Example incident

An example security event where an unauthorized user compromises a Gen-AI application that’s hosted on AWS by using credentials that were exposed on a public code repository.
- determine what resources were accessed, modified, created, or deleted.
- the main log sources to review:
  - CloudTrail
  - CloudWatch
  - [VPC Flow Logs](https://docs.aws.amazon.com/vpc/latest/userguide/flow-logs.html)
  - [Amazon Simple Storage Service (Amazon S3) data events](https://docs.aws.amazon.com/AmazonS3/latest/userguide/cloudtrail-logging-s3-info.html) (for evidence of access to an organization’s S3 buckets)
  - Amazon Bedrock [model invocation logs](https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html) (if the application uses this service)

**Access**

- determine whether an organization has access to their AWS account.
  - If the password for the AWS account root user was lost or changed, [reset the password](https://docs.aws.amazon.com/IAM/latest/UserGuide/reset-root-password.html), enable [multi-factor authentication (MFA) device](https://docs.aws.amazon.com/IAM/latest/UserGuide/enable-virt-mfa-for-root.html) for the root user—this should block a threat actor from accessing the root user.

- determine whether unauthorized access to the account persists.

  - to identify mutative actions logged by [AWS Identity and Access Management (IAM)](https://aws.amazon.com/iam/) and [AWS Security Token Service (Amazon STS)](https://docs.aws.amazon.com/STS/latest/APIReference/welcome.html), see the [Analysis](https://github.com/aws-samples/aws-customer-playbook-framework/blob/main/docs/Compromised_IAM_Credentials.md#analysis) section of the [Compromised IAM Credentials playbook](https://github.com/aws-samples/aws-customer-playbook-framework/blob/main/docs/Compromised_IAM_Credentials.md) on GitHub.
  - make sure that access keys aren’t stored in public repositories or in the application code; for alternatives, see [Alternatives to long-term access keys](https://docs.aws.amazon.com/IAM/latest/UserGuide/security-creds-programmatic-access.html#security-creds-alternatives-to-long-term-access-keys).


**Infrastructure changes**

- some common names for control plane events in CloudTrail for this element:

    - `ec2:RunInstances`
    - `ec2:StartInstances`
    - `ec2:TerminateInstances`
    - `ecs:CreateCluster`
    - `cloudformation:CreateStack`
    - `rds:DeleteDBInstance`
    - `rds:ModifyDBClusterSnapshotAttribute`

#### AI changes

Unauthorized changes can include, but are not limited to, system prompts, application code, guardrails, and model availability.

- Internal user access to the Gen-AI resources that AWS hosts are logged in CloudTrail

- **event sources**:

  - `bedrock.amazonaws.com`
  - `sagemaker.amazonaws.com`
  - `qbusiness.amazonaws.com`
  - `q.amazonaws.com`

- **event names** that would represent Gen-AI resource log tampering:

  - `bedrock:PutModelInvocationLoggingConfiguration`
  - `bedrock:DeleteModelInvocationLoggingConfiguration`

- **event names**  that would represent access to the AI/ML model service configuration:

  - `bedrock:GetFoundationModelAvailability`
  - `bedrock:ListProvisionedModelThroughputs`
  - `bedrock:ListCustomModels`
  - `bedrock:ListFoundationModels`
  - `bedrock:ListProvisionedModelThroughput`
  - `bedrock:GetGuardrail`
  - `bedrock:DeleteGuardrail`

In our example scenario
- the unauthorized user has gained access to the AWS account.
- Now imagine that the compromised user has a policy attached that grants them full access to all resources. With this access, the unauthorized user can enumerate each component of Amazon Bedrock and identify the knowledge base and guardrails that are part of the application.

- The unauthorized user then requests model access to other [foundation models (FMs)](https://aws.amazon.com/what-is/foundation-models/) within Amazon Bedrock and removes existing guardrails.

- The access to other foundation models could indicate that the unauthorized user intends to use the Gen-AI application for their own purposes, and the removal of guardrails minimizes filtering or output checks by the model.

- AWS recommends that you implement fine-grained access controls by using  [IAM policies and resource-based policies](https://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html#security_iam_access-manage)  to restrict access to only the necessary Amazon Bedrock resources, [AWS Lambda](https://aws.amazon.com/lambda/) functions, and other components that the application requires.

- Also, you should enforce the use of MFA for IAM users, roles, and service accounts with access to critical components such as Amazon Bedrock and other components of the Gen-AI application.

#### Data store changes

- event names that would represent changes to AI/ML data sources:

  - `bedrock:CreateDataSource`
  - `bedrock:GetKnowledgeBase`
  - `bedrock:DeleteKnowledgeBase`
  - `bedrock:CreateAgent`
  - `bedrock:DeleteAgent`
  - `bedrock:InvokeAgent`
  - `bedrock:Retrieve`
  - `bedrock:RetrieveAndGenerate`

  - For the full list of possible actions, see the [Amazon Bedrock API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_Operations.html).

In this scenario, we have established that the unauthorized user has full access to the Gen-AI application and that some enumeration took place.
- The unauthorized user then identified the S3 bucket that was the knowledge base for the Gen-AI application and uploaded inaccurate data, which corrupted the LLM.
- For examples of this vulnerability, see the section _LLM03 Training Data Poisoning_ in the [OWASP TOP 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-2023-v1_1.pdf).


**Invocation**

In our example scenario, imagine that model invocation wasn’t enabled, the incident responder
- couldn’t collect invocation logs to see the model input or output data for unauthorized invocations.
- wouldn’t be able to determine the prompts and subsequent responses from the LLM.
- couldn’t see the full request data, response data, and metadata associated with invocation calls.

Event names in model invocation logs that would represent model invocation logging in Amazon Bedrock include:

- `bedrock:InvokeModel`
- `bedrock:InvokeModelWithResponseStream`
- `bedrock:Converse`
- `bedrock:ConverseStream`


sample log entry for Amazon Bedrock model invocation logging:

![Figure 2: sample model invocation log including prompt and response](https://d2908q01vomqb2.cloudfront.net/22d200f8670dbdb3e253a90eee5098477c95c23d/2024/09/16/img2-3.png)



**Private data**

From an architectural standpoint, Gen-AI applications shouldn’t have direct access to an organization’s private data.

- should classify data used to train a Gen-AI application or for RAG use as data store data and segregate it from private data, unless the Gen-AI application uses the private data (for example, in the case where a Gen-AI application is tasked to answer questions about medical records for a patient).

- One way to help make sure that an organization’s private data is segregated from Gen-AI applications is to [use a separate account](https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/sec_securely_operate_multi_accounts.html) and to authenticate and authorize access as necessary to adhere to the principle of least privilege.


**Agency**

In our example scenario

- the Gen-AI application has excessive permissions to services that aren’t required by the application.

- Imagine that the application code was running with an execution role with full access to [Amazon Simple Email Service (Amazon SES)](https://aws.amazon.com/ses/). This could allow for the unauthorized user to send spam emails on the users’ behalf in response to a prompt.

- You could help prevent this by limiting permission and functionality of the Gen-AI application plugins and agents.
  - For more information, see [OWASP Top 10 for LLM](https://owasp.org/www-project-top-10-for-large-language-model-applications/), evidence of LLM08 Excessive Agency.

- During an investigation, while analyzing the logs, both the `sourceIPAddress` and the `userAgent` fields will be associated with the Gen-AI application (for example, `sagemaker.amazonaws.com`, `bedrock.amazonaws.com`, or `q.amazonaws.com` ).
  - Some examples of services that might commonly be called or invoked by other services are Lambda, Amazon SNS, and Amazon SES.

---
