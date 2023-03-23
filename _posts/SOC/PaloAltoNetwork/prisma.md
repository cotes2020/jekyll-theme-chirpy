#  Prisma™ Cloud Administrator's Guide
paloaltonetworks.com/documentation
2 PRISMA™ CLOUD ADMINISTRATOR'S GUIDE |
Contact Information
Corporate Headquarters:
Palo Alto Networks
3000 Tannery Way
Santa Clara, CA 95054
www.paloaltonetworks.com/company/contact-support
About the Documentation
- For the most recent version of this guide or for access to related documentation, visit the Technical
Documentation portal www.paloaltonetworks.com/documentation.
- To search for a specific topic, go to our search page www.paloaltonetworks.com/documentation/
document-search.html.
- Have feedback or questions for us? Leave a comment on any page in the portal, or write to us at
documentation@paloaltonetworks.com.
Copyright
Palo Alto Networks, Inc.
www.paloaltonetworks.com
© 2019-2020 Palo Alto Networks, Inc. Palo Alto Networks is a registered trademark of Palo
Alto Networks. A list of our trademarks can be found at www.paloaltonetworks.com/company/
trademarks.html. All other marks mentioned herein may be trademarks of their respective companies.
Last Revised
October 24, 2020
TABLE OF CONTENTS iii
Table of Contents
Get Started with Prisma Cloud........................................................................9
Prisma Cloud.............................................................................................................................................. 11
Cloud Security Posture Management with Prisma Cloud.................................................. 11
Cloud Workload Protection with Prisma Cloud................................................................... 13
Prisma Cloud License Types.................................................................................................................. 14
Prisma Cloud—How it Works................................................................................................................ 16
Get Prisma Cloud From the Palo Alto Networks Marketplace......................................................18
Get Prisma Cloud From the AWS Marketplace................................................................................ 20
Annual or Longer Term Subscription...................................................................................... 20
Hourly PAYG Subscription.........................................................................................................22
Get Prisma Cloud From the GCP Marketplace................................................................................. 25
Access Prisma Cloud................................................................................................................................28
Prisma Cloud—First Look........................................................................................................................30
Prisma Cloud—Next Steps...................................................................................................................... 31
NAT Gateway IP Addresses for Prisma Cloud.................................................................................. 32
Access the Prisma Cloud REST API..................................................................................................... 35
Prisma Cloud FAQs.................................................................................................................................. 37
Account Onboarding and SSO..................................................................................................37
Password and Help......................................................................................................................37
Policy and Alerts...........................................................................................................................38
Connect Cloud Platform to Prisma Cloud........................................ 43
Cloud Account Onboarding....................................................................................................................45
Onboard AWS Account................................................................................................................ 46
Add an AWS Cloud Account on Prisma Cloud.....................................................................46
Add an AWS Organization to Prisma Cloud..........................................................................53
Update an Onboarded AWS Account.....................................................................................62
Set Up the Prisma Cloud Role for AWS—Manual................................................................66
Prisma Cloud on AWS China.................................................................................................... 72
AWS APIs Ingested by Prisma Cloud......................................................................................72
Onboard Azure Account...............................................................................................................77
Azure Cloud Account Onboarding Checklist.........................................................................77
Add an Azure Subscription on Prisma Cloud........................................................................ 80
Update an Onboarded Azure Account................................................................................... 91
Add an Azure Active Directory Account on Prisma Cloud................................................ 91
Set Up Azure Subscription for Prisma Cloud.............................................................. 94
Create a Custom Role on Azure to Enable Prisma Cloud to Access Flow Logs.......... 101
Use the Azure PowerShell Script to Add an Azure Account...........................................102
Troubleshoot Azure Account Onboarding...........................................................................104
Microsoft Azure APIs Ingested by Prisma Cloud...............................................................107
Onboard Google Cloud Platform (GCP) Account.................................................................109
Permissions and Roles for GCP Account on Prisma Cloud..............................................109
Add GCP Project to Prisma Cloud.............................................................................. 117
Update an Onboarded Google Cloud Account.................................................................. 122
Enable Flow Logs for GCP Projects......................................................................................123
Flow Log Compression on GCP.............................................................................................125
Add GCP Organization to Prisma Cloud....................................................................127
Enable Flow Logs for GCP Organization............................................................................. 133
Create a Service Account With a Custom Role for GCP................................................. 134
iv TABLE OF CONTENTS
GCP APIs Ingested by Prisma Cloud.................................................................................... 137
Onboard Alibaba Cloud Account............................................................................................. 140
Set Up Alibaba Account.................................................................................................140
Add an Alibaba Cloud Account on Prisma Cloud.............................................................. 144
Alibaba APIs Ingested by Prisma Cloud............................................................................... 149
Cloud Service Provider Regions on Prisma Cloud..........................................................................150
AWS Regions.............................................................................................................................. 150
Azure............................................................................................................................................. 151
GCP................................................................................................................................................153
Alibaba Cloud..............................................................................................................................154
Manage Prisma Cloud Administrators.......................................................157
Prisma Cloud Administrator Roles..................................................................................................... 159
Create and Manage Account Groups on Prisma Cloud................................................................ 160
Create an Account Group........................................................................................................160
Manage Account Groups......................................................................................................... 160
Create Prisma Cloud Roles.................................................................................................................. 161
Prisma Cloud Administrator Permissions..........................................................................................163
Manage Roles in Prisma Cloud........................................................................................................... 170
Add Administrative Users On Prisma Cloud....................................................................................171
Create and Manage Access Keys.......................................................................................................173
Manage Prisma Cloud Profile.................................................................................................... 176
Set up SSO Integration on Prisma Cloud......................................................................................... 177
Set up Just-in-Time Provisioning on Okta...........................................................................185
View Audit Logs......................................................................................................................................188
Define Prisma Cloud Enterprise and Anomaly Settings................................................................189
Set Up Inactivity Timeout........................................................................................................189
Set Up Global Settings for Policy and Alerts......................................................................189
Set Up Anomaly Policy Thresholds.......................................................................................190
Manage Prisma Cloud Alerts.......................................................................193
Prisma Cloud Alerts and Notifications..............................................................................................195
Trusted IP Addresses on Prisma Cloud.............................................................................................196
Enable Prisma Cloud Alerts..................................................................................................................198
Create an Alert Rule..............................................................................................................................199
Configure Prisma Cloud to Automatically Remediate Alerts.......................................................203
Send Prisma Cloud Alert Notifications to Third-Party Tools.......................................................207
Send Alert Notifications to Amazon SQS............................................................................207
Send Alert Notifications to Azure Service Bus Queue.....................................................208
Send Alert Notifications Through Email...............................................................................209
Send Alert Notifications to a Slack Channel.......................................................................211
Send Alert Notifications to Splunk........................................................................................212
Send Alert Notifications to Jira..............................................................................................212
Send Alert Notifications to Google Cloud SCC..................................................................213
Send Alert Notifications to ServiceNow..............................................................................214
Send Alert Notifications to Webhooks................................................................................ 215
Send Alert Notifications to PagerDuty.................................................................................215
Send Alert Notifications to AWS Security Hub................................................................. 216
Send Alert Notifications to Microsoft Teams.....................................................................216
Send Alert Notifications to Cortex XSOAR.........................................................................218
View and Respond to Prisma Cloud Alerts......................................................................................219
Generate Reports on Prisma Cloud Alerts.......................................................................................222
Alert Payload........................................................................................................................................... 225
TABLE OF CONTENTS v
Prisma Cloud Alert Resolution Reasons............................................................................................228
Prisma Cloud Dashboards............................................................................ 231
Assets, Policies, and Compliance on Prisma Cloud........................................................................233
Prisma Cloud Asset Inventory.............................................................................................................238
SecOps Dashboard.................................................................................................................................240
Monitored Accounts................................................................................................................. 240
Monitored Resources................................................................................................................240
Open Alerts................................................................................................................................. 240
Top Instances by Role.............................................................................................................. 240
Alerts by Severity...................................................................................................................... 241
Policy Violations by Type over Time.................................................................................... 241
Top Policy Violations................................................................................................................ 241
Top Internet Connected Resources...................................................................................... 241
Connections from the Internet...............................................................................................241
Customize the SecOps Dashboard.................................................................................................... 243
Prisma Cloud Policies.................................................................................... 245
Create a Policy on Prisma Cloud........................................................................................................247
Create a Configuration Policy.................................................................................................247
Create a Network or Audit Event Policy............................................................................. 251
Add a JSON Query for Build Policy Subtype..................................................................... 253
Prisma Cloud IAC Scan Policy Operators............................................................................ 262
Manage Prisma Cloud Policies............................................................................................................266
Anomaly Policies.....................................................................................................................................271
Investigate Incidents on Prisma Cloud......................................................275
Investigate Config Incidents on Prisma Cloud................................................................................ 277
Investigate Audit Incidents on Prisma Cloud.................................................................................. 279
Use Prisma Cloud to Investigate Network Incidents.....................................................................281
Prisma Cloud Compliance............................................................................ 285
Compliance Dashboard......................................................................................................................... 287
Create a Custom Compliance Standard............................................................................................290
Add a New Compliance Report.......................................................................................................... 293
Configure External Integrations on Prisma Cloud.................................. 299
Prisma Cloud Integrations.................................................................................................................... 301
Integrate Prisma Cloud with Slack.....................................................................................................303
Integrate Prisma Cloud with Splunk.................................................................................................. 306
Integrate Prisma Cloud with Amazon SQS...................................................................................... 308
Integrate Prisma Cloud with Amazon GuardDuty..........................................................................311
Integrate Prisma Cloud with AWS Inspector...................................................................................313
Integrate Prisma Cloud with AWS Security Hub............................................................................316
Integrate Prisma Cloud with Azure Service Bus Queue............................................................... 321
Integrate Prisma Cloud with Jira........................................................................................................ 324
Integrate Prisma Cloud with Qualys..................................................................................................334
Integrate Prisma Cloud with Google Cloud Security Command Center (SCC)........................ 340
Integrate Prisma Cloud with Tenable................................................................................................344
Integrate Prisma Cloud with ServiceNow........................................................................................ 347
vi TABLE OF CONTENTS
Set Up Permissions on ServiceNow......................................................................................347
Enable the ServiceNow Integration on Prisma Cloud.......................................................349
Set up Notification Templates................................................................................................351
Interpret Error Messages.........................................................................................................353
View Alerts.................................................................................................................................. 354
Integrate Prisma Cloud with Webhooks...........................................................................................356
Integrate Prisma Cloud with PagerDuty...........................................................................................358
Integrate Prisma Cloud with Microsoft Teams............................................................................... 361
Integrate Prisma Cloud with Cortex XSOAR...................................................................................362
Enable the Cortex XSOAR Integration on Prisma Cloud................................................. 362
Set Up the Integration on Cortex XSOAR...........................................................................363
Prisma Cloud Integrations—Supported Capabilities.......................................................................366
Prisma Cloud DevOps Security...................................................................369
Secure Infrastructure Automation........................................................................................... 371
Prisma Cloud Plugins.............................................................................................................................372
Set Up Prisma Cloud Configuration File for IaC Scan.........................................................374
Configure IaC Scan to Support Terraform...........................................................................374
Configure IaC Scan to Support AWS CloudFormation.....................................................375
Configure IaC Scan to Support Kubernetes........................................................................376
Configure Prisma Cloud Tags................................................................................................. 376
Use the Prisma Cloud Extension for AWS DevOps...................................................................... 377
Set Up IaC Scanning with AWS CodePipeline................................................................... 377
Set Up Container Image Scanning with AWS CodeBuild................................................ 386
poll.sh............................................................................................................................................ 389
Use the Prisma Cloud Extension for Azure DevOps.....................................................................399
Install and Configure the Prisma Cloud Extensions.......................................................... 399
Set up a Custom Task for IaC Scanning...............................................................................402
Set Up Container Image Scanning.........................................................................................405
Set Up RASP Defender............................................................................................................ 408
Sample YAML File..................................................................................................................... 409
Use the Prisma Cloud Plugin for CircleCI........................................................................................ 410
Use the Prisma Cloud Plugin for IntelliJ IDEA................................................................................419
Install the Prisma Cloud Plugin for IntelliJ...........................................................................419
Configure the Prisma Cloud Plugin for IntelliJ................................................................... 420
Scan Using the Prisma Cloud Plugin for IntelliJ.................................................................421
Use the Prisma Cloud App for GitHub............................................................................................. 426
Set up the Prisma Cloud App Files for GitHub.................................................................. 426
Install the Prisma Cloud App for GitHub.............................................................................429
Use the Prisma Cloud Extension for GitLab....................................................................................433
Use the Prisma Cloud Extension for the GitLab CI/CD Pipeline....................................433
Use the Prisma Cloud Extension for GitLab SCM............................................................. 447
Use the Prisma Cloud Extension for Visual Studio Code............................................................. 453
Install Prisma Cloud Extension for Visual Studio Code.................................................... 453
Configure the Prisma Cloud Extension for VS Code........................................................ 453
Scan Using the Prisma Cloud VS Code Extension.............................................................455
Use the Prisma Cloud IaC Scan REST API....................................................................................... 457
Use the IaC Scan API Version 2............................................................................................ 457
Scan API Version 1 for Terraform Files (Deprecated)...................................................... 462
Scan API Version 1 for AWS CloudFormation Templates (Deprecated)...................... 467
Scan API Version 1 for Kubernetes Templates (Deprecated)......................................... 468
Prisma Cloud Data Security.........................................................................471
TABLE OF CONTENTS vii
What is Included with Prisma Cloud Data Security.......................................................................473
Enable the Prisma Cloud Data Security Module............................................................................ 474
Add a New AWS Account and Enable Data Security.......................................................476
Edit an AWS Account Onboarded on Prisma Cloud to Enable Data Security.............483
Monitor Data Security Scan Results on Prisma Cloud..................................................................493
Use the Data Policies to Scan................................................................................................493
Data Security settings...............................................................................................................494
Data Dashboard......................................................................................................................... 496
Data Inventory............................................................................................................................499
Resource Explorer......................................................................................................................502
Object Explorer.......................................................................................................................... 503
Exposure Evaluation..................................................................................................................504
Supported File Extensions—Prisma Cloud Data Security.................................................506
Supported Data Profiles & Patterns......................................................................................507
Disable Prisma Cloud Data Security and Offboard AWS account............................................. 515
Guidelines for Optimizing Data Security Cost on Prisma Cloud.................................................517
Cost Implications and Control................................................................................................517
API Throttling and Egress Implications................................................................................ 517
viii TABLE OF CONTENTS
9
Get Started with Prisma Cloud
Prisma™ Cloud is an API-based cloud service that connects to cloud environments in
just minutes and aggregates volumes of raw configuration data, user activity information, and
network traffic to analyze and produce concise and actionable insights.
> Prisma Cloud
> Prisma Cloud—How it Works
> Prisma Cloud License Types
> Get Prisma Cloud From the Palo Alto Networks Marketplace
> Get Prisma Cloud From the AWS Marketplace
> Get Prisma Cloud From the GCP Marketplace
> Access Prisma Cloud
> Prisma Cloud—First Look
> Prisma Cloud—How it Works
> NAT Gateway IP Addresses for Prisma Cloud
> Access the Prisma Cloud REST API
> Prisma Cloud FAQs
10
Prisma Cloud
Prisma™ Cloud is a cloud infrastructure security solution and a Security Operations Center (SOC)
enablement tool that enables to address risks and secure workloads in a heterogeneous
environment (hybrid and multicloud) from a single console. It provides complete visibility and control over
risks within public cloud infrastructure and enables to manage vulnerabilities, detect anomalies, ensure compliance, and provide runtime defense in heterogeneous environments, such as Windows, Linux, Kubernetes, Red Hat OpenShift, AWS Lambda, Azure Functions, and GCP Cloud Functions. The main
capabilities are:
- Continuous security assessment of configuration, compliance monitoring, and integration with
external services for incident management and remediation to address issues identified on your
resources in the public cloud. These capabilities are completely API-based and configure these
capabilities using the different tabs on the Prisma Cloud administrative console. For an overview, see
Cloud Security Posture Management with Prisma Cloud.
- Consistent visibility and runtime defense with least-privilege microsegmentation for physical machines, virtual machines, containers, and serverless workloads—regardless of location. These capabilities require
an agent and the API. Use the Compute tab on the Prisma Cloud administrative console to set up and
monitor this functionality. For an overview, see Cloud Workload Protection with Prisma Cloud.
Cloud Security Posture Management with Prisma Cloud
The API-based service enables granular visibility in to resources deployed on public cloud platforms—Amazon Web Services (AWS), Google Cloud Platform (GCP), and Microsoft Azure—and in to the network
traffic flows to these resources from the internet and between instances. Prisma™ Cloud also provides
threat detection and response for resource misconfigurations and workload vulnerabilities and provides
visibility into user activity within each cloud environment. Tracking user activity helps identify account
compromises, escalation of privileges with privileged accounts, and insider threats from malicious users, unauthorized activity, and inadvertent errors. Prisma Cloud continuously monitors cloud environments
to help ensure that cloud infrastructure is protected from these security threats.
In addition to providing visibility and reducing risks, Prisma Cloud facilitates Security Operations Center
(SOC) enablement and adherence to compliance standards. As the service automatically discovers and
monitors compliance for new resources that are deployed in cloud environment, it enables to
implement policy guardrails to ensure that resource configurations adhere to industry standards and helps
integrate configuration change alerts into DevSecOps workflows that automatically resolve issues
as they are discovered. This capability streamlines the process of identifying issues and detecting and
responding to a list of prioritized risks to maintain an agile development process and operational efficiency.
12
Here are some highlights of Prisma Cloud:
Comprehensive Visibility—Enables to view resources—deployed on multiple cloud
infrastructure platforms—from a single console. In addition to providing a consolidated view of the resources across the cloud platforms, Prisma Cloud integrates with threat intelligence feeds, vulnerability
scanners, and Security Information and Event Management (SIEM) solutions to help build a contextual view of cloud deployments.
Policy Monitoring—Enables to use Prisma Cloud, which includes Security policies based on industry
standards, to continuously monitor for violations. Because cloud platforms enable agility and users can create, modify, and destroy resources on-demand, these user actions often occur without any
security oversight. Prisma Cloud provides hundreds of out-of-the-box policies for common security and
compliance standards, such as GDPR, PCI, CIS, and HIPAA. also create custom policy rules to
address specific needs or to customize the default policy rules.
Anomaly Detection—Automatically detects suspicious user and network behavior using machine
learning. Prisma Cloud consumes data about AWS resources from AWS CloudTrail, AWS Inspector, and Amazon GuardDuty to detect account compromises and insider threats. This service uses machine
learning to score the risk level for each cloud resource based on the severity of business impact, policy violations, and anomalous behavior. Risk scores are then aggregated so that prioritize alerts and benchmark risk postures across entire environment.
Contextual Alerting—Leverages highly contextual alerts for prioritization and rapid response. Because
Prisma Cloud also integrates with external vulnerability services, such as AWS Inspector, Tenable.io, and Qualys, to continuously scan environment, it has additional context to identify unexpected
and potentially unauthorized and malicious activity. For example, the service scans for unpatched hosts, escalation of privileges, and use of exposed credentials, and also scans communication for malicious IP
addresses, URLs, and domains.
Cloud Forensics—Enables to go back to any point in time and investigate an issue within seconds.
To help identify security blind spots and investigate issues, Prisma Cloud monitors network traffic
from sources such as AWS VPC flow logs, Azure flow logs, GCP flow logs, Amazon GuardDuty, and user
activity from AWS CloudTrail and Azure.
Compliance Reporting—Reports risk posture to management team, to board of directors, and to auditors.
Limited GAData Security—Scans data stored on AWS S3 buckets and provides visibility on the scan
results directly on the Prisma Cloud dashboard. The data security capabilities include predefined data policies and associated data classification profiles such as PII, Financial, or Healthcare & Intellectual
Property that scan objects stored in the S3 bucket to identify exposure—how sensitive information is kept private, or exposed or shared externally, or allows unauthorized access. It also uses the WildFire
service to detect known and unknown malware in these objects.
Cloud Workload Protection with Prisma Cloud
Prisma™ Cloud offers cloud workload protection, as either a SaaS option or a self-hosted solution that deploy and manage (review options ).
The SaaS option, available with the Prisma Cloud Enterprise Edition, offers a single management console
for threat detection, prevention, and response for heterogeneous environment where teams are
leveraging public cloud platforms and a rich set of microservices to rapidly build and deliver applications.
The Compute tab on the Prisma Cloud administrative console enables to define policy and to monitor
and protect the hosts, containers, and serverless functions within environment.
To monitor the workloads, must deploy Prisma Cloud Defenders: the agents. All Defenders, regardless
of their type, connect back to the console using WebSocket over port 8084 to retrieve policies and enforce
vulnerability and compliance blocking rules to the environments where they are deployed, and to send data back to the Compute tab within the Prisma Cloud administrative console. For documentation on how to get
started with deploying Defenders, configuring policies, viewing alerts, and interpreting the data on Radar, see the Prisma Cloud Administrator’s Guide (Compute) . For administrative user management, such as integrating single sign-on, setting up custom roles, and creating access keys, use the Settings tab on the Prisma Cloud administrative console outlined in this document.
14
Prisma Cloud License Types
Prisma Cloud is available as a one-, two-, or three-year subscription in the following three editions:
- Prisma Cloud Business Edition—License includes configuration security posture management (CSPM), compliance reporting, automated remediation, custom policy creation, and a standard success plan.
The Business edition is tailored for investigating resource misconfigurations and verifying adherence
to compliance standards so that take steps to implement policies and regulations that enable
to secure public cloud deployments and comply with security standards. The Business edition
is powered entirely through the application programming interface (API) and delivered exclusively as a software-as-a-service (SaaS) model.
- Prisma Cloud Compute Edition—License includes workload protection for hosts, containers, and
serverless deployments in any cloud or on-premises environment. This is an agent-based approach
to protect resources across the application lifecycle. Unlike the Prisma Cloud Business or Enterprise
editions, this is a is a self-operated software solution that download and run in own
environments—whether public, private, or hybrid clouds—including entirely air-gapped environments.
- Prisma Cloud Enterprise Edition—License includes all features included in the Business edition and
Compute edition licenses (including a standard success plan). The Enterprise edition—delivered as a SaaS model—combines API- and agent-based approaches to deliver comprehensive host, container, serverless, IaaS, PaaS, IAM, network, and storage security for cloud and on-premises environments. It protects from the most sophisticated threats with advanced machine learning capabilities, network security
monitoring, user entity behavior analysis (UEBA) for detecting location- and activity-based anomalies, and integration with host vulnerability management tools.
optionally upgrade to a premium success plan that includes 24x7 access to customer
success experts, custom workshops, implementation assistance, configuration and planning, online
documentation, and best practices guides.
Each of these editions has a different capacity unit and unit price in Prisma Cloud Credits. The number of credits required to secure assets can vary across the different Prisma Cloud modules such as Visibility, Compliance & Governance, Compute Security, or Network Security. Refer to the Prisma Cloud Licensing
and Editions Guide for details.
Licensing is sold in increments of 100 credits and estimate the number of units need to monitor
and protect. usage data is based on the number of capacity units that are consuming for each
Prisma Cloud module every hour, and the Time Range is averaged for daily, weekly, monthly and quarterly
usage to prevent overages based on short-term bursts.
After Connect Cloud Platform to Prisma Cloud or deploy Prisma Cloud Defenders, review
the actual number of licensable assets that you’re securing with the service. The Settings > Licensing tab
displays a resource count in a tabular format and a graphical view of the resource trend as a line graph. Use
the License Usage graph to review the average number of billable resources monitored across overall
cloud environments or view the pattern for each cloud type, actual usage trends, and the number of Prisma Cloud credits have purchased. While the table depicts the total number of resources for which
credits are applied, the graph represents the actual credits consumed over the time range and is averaged
across the time period.
If have environments where have deployed Prisma Cloud Defenders either on-premises or on
private or public cloud platforms that are not being monitored by Prisma Cloud, such as on OpenShift, on-prem Kubernetes clusters, or AWS Fargate, select Non-onboarded Cloud Account Resources to view
the details on credits used towards resources deployed on cloud environments that are not onboarded on
Prisma Cloud.
The default time range is three months but select a time range of choice. also
download and share the licensing information in a zip file that includes information on each cloud platform
in CSV file format. For a time period of 3 days or less, download hourly usage data, and data on
daily usage for a time period greater than 3 days.
For details on how credits are calculated for Prisma Cloud Defenders, see Prisma Cloud Compute—Licensing.
16
Prisma Cloud—How it Works
As a Security Operations Center (SOC) enablement tool, Prisma™ Cloud helps identify issues in
cloud deployments and then respond to a list of prioritized risks so that maintain an agile
development process and operational efficiency.
When add a cloud account to Prisma Cloud, the IaaS Integration Services module ingests data from
flow logs, configuration logs, and audit logs in cloud environment over an encrypted connection and
stores the encrypted metadata in RDS3 and Redshift instances within the Prisma Cloud AWS Services
module. then use the Prisma Cloud administrative console or the APIs to interact with this data to
configure policies, to investigate and resolve alerts, to set up external integrations, and to forward alert
notifications. The Enterprise Integration Services module enables to leverage Prisma Cloud as your
cloud orchestration and monitoring tool and to feed relevant information to existing SOC workflows.
The integration service ingests information from existing single sign-on (SSO) identity management
system and allows to feed information back in to existing SIEM tools and to collaboration and
helpdesk workflows.
To ensure the security of data and high availability of Prisma Cloud, Palo Alto Networks makes
Security a priority at every step. The Prisma Cloud architecture uses Cloudflare for DNS resolution of web
requests and for protection against distributed denial-of-service (DDoS) attacks. The following diagram
represents the infrastructure within a region: For data redundancy of stateful components, such as RDS and Redshift, and of stateless components, such
as the application stack and Redis (used primarily as a cache), the service uses native AWS capabilities for automated snapshots or has set up automation scripts using AWS Lambda and SNS for saving copies to S3
buckets.
Additionally, to ensure that these snapshots and other data at rest are safe, Prisma Cloud uses AWS Key
Management Service (KMS) to encrypt and decrypt the data. To protect data in transit, the infrastructure
terminates the TLS connection at the Elastic Load Balancer (ELB) and secures traffic between components
within the data center using an internal certificate until it is terminated at the application node. This ensures
that data in transit is encrypted using SSL. And, lastly, for workload isolation and micro segmentation, the built-in VPC security controls in AWS securely connect and monitor traffic between application workloads
on AWS.
18
Get Prisma Cloud From the Palo Alto Networks
Marketplace
Purchase or try Prisma™ Cloud from Palo Alto Networks Marketplace. Within 24 hours of purchase, will get access to the Prisma Cloud tenant that is provisioned for you.
- STEP 1 | Go to Palo Alto Networks Marketplace.
- STEP 2 | Create an account.
Required only if do not have a Palo Alto Networks Customer Support Portal (CSP) account.
1. Enter the personal and company information requested in the form. Required fields are indicated
with red asterisks.
2. Accept the privacy agreement and Create an account.
3. Look for the welcome email in inbox and click the link in that email to continue the activation
process.
4. Click here on the page that displays and enter account credentials on the Palo Alto Networks
single sign-on (SSO) page.
5. Answer the security questions and Save changes.
will be logged in to Palo Alto Networks Marketplace.
- STEP 3 | Scroll down and select the Prisma Cloud app on Palo Alto Networks Marketplace and then
View app.
- STEP 4 | Select Free Trial or Buy Now.
The Enterprise edition is available in either the Trial or Buy Now option; the Business edition is available
only as a Buy Now subscription. (See Prisma Cloud License Types for details.)
- Buy Now:
1. Select the license edition—Business or Enterprise.
2. Select the Term—1 year.
3. Select Billing cycle—yearly or monthly.
4. Select the Number of workloads need to secure.
The range is 100 to 900 workloads; contact sales if need to secure more than 900 workloads.
5. Opt in to purchase the Premium Success Plan.
If do not opt in, will receive the standard success plan.
6. Select the region in which want to provision Prisma Cloud instance.
Available regions are Australia, Germany, GovCloud, and US.
7. Click Next and review the summary.
8. Enter Billing information.
9. Review and Accept the terms and conditions of the EULA and click Next.
will see a success message on screen. Then, within the next 24 hours,will receive two
emails: the first is an order confirmation email and the second is a welcome email that includes
a link will use to log in to Prisma Cloud tenant. also access Prisma Cloud
tenant from the Palo Alto Networks hub using the Prisma Cloud tile.
10.Enable Auto Renewal to ensure that have uninterrupted monitoring and protection for your
public cloud deployments. -Free trial:
1. Select the region in which want to provision Prisma Cloud instance.
Available regions are Australia, Germany, GovCloud, and US.
2. Accept the EULA and click Next.
A license confirmation displays.
Within the next 24 hours, will receive a welcome email that includes a link to log in to the Prisma Cloud tenant that is provisioned for you. directly access the Prisma Cloud instance
or log in to the hub and click the Prisma tile to log in to Prisma Cloud tenant.
20
Get Prisma Cloud From the AWS Marketplace
Purchase Prisma™ Cloud as a SaaS subscription directly from the AWS Marketplace. choose a 1-, 3-, or 5-year subscription for the Prisma Cloud Business or Enterprise Edition, or as a PAYG subscription
based on hourly usage, in the Prisma Cloud Enterprise Edition. The PAYG model offers a 15-day trial
and is available in the Enterprise Edition only.
Within 24 hours of purchase, will get access to the Prisma Cloud tenant that is provisioned for you.
- Annual or Longer Term Subscription
- Hourly PAYG Subscription
Annual or Longer Term Subscription
- STEP 1 | Go to AWS Marketplace.
- STEP 2 | Search for Palo Alto Networks on AWS Marketplace and select Prisma Cloud Threat Defense
and Compliance Platform.
- STEP 3 | Continue to Subscribe.
- STEP 4 | Enter the details to get a Prisma Cloud subscription.
The subscription is for a 12-month period. 1. Select autorenewal preference.
The default for Renewal Settings is Yes so that contract is automatically renewed before
it expires. This ensures that have uninterrupted coverage for securing public cloud
deployments.
2. Select the license edition and the number of units.
The Prisma Cloud License Types are Business or Enterprise with the standard success plan. If want to protect 1,000 workloads, enter 10 units (1 unit is 100 workloads).
3. Create contract.
4. Confirm the contract and Pay Now.
5. Set up account to continue setting up Prisma Cloud.
This link has a temporary access token that expires in 24 hours. If want to come
back later and provide the details to provision Prisma Cloud instance, must
log in to AWS account. then use the View or Modify link for the active
AWS Marketplace software subscription on Software page of AWS Marketplace.
then Click here to setup account.
22
- STEP 5 | Provide the details for provisioning Prisma Cloud instance.
1. Enter the personal and company information requested in the form.
2. Select the region where want Prisma Cloud tenant provisioned.
3. Select Register.
A message informs whether the registration was successful. Look for the welcome email in your
inbox and click the link in that email to log in to the Palo Alto Networks hub.
4. Log in to the hub and click the Prisma Cloud tile to start using Prisma Cloud.
are now ready for Prisma Cloud—First Look and Prisma Cloud—Next Steps.
Hourly PAYG Subscription
- STEP 1 | Go to AWS Marketplace.
- STEP 2 | Search for Palo Alto Networks on AWS Marketplace and select Prisma Cloud Enterprise
Edition.
- STEP 3 | Continue to Subscribe. - STEP 4 | Subscribe to Prisma Cloud.
The first 15 days of the subscription are free, and will be charged for the subscription hourly after
the free period expires.
24
- STEP 5 | Provide the details for provisioning Prisma Cloud instance.
1. Enter the personal and company information requested in the form.
2. Select the region where want Prisma Cloud tenant provisioned.
3. Select Register.
A message informs whether the registration was successful. Look for the welcome email in your
inbox and click the link in that email to log in to the Palo Alto Networks hub.
4. Log in to the hub and click the Prisma Cloud tile to start using Prisma Cloud.
are now ready for Prisma Cloud—First Look and Prisma Cloud—Next Steps. Get Prisma Cloud From the GCP Marketplace
Purchase Prisma™ Cloud directly from Google Cloud Platform (GCP) Marketplace. Within 24 hours of your
purchase, will get access to the Prisma Cloud tenant that is provisioned for you.
- STEP 1 | Go to GCP Marketplace and search for Prisma Cloud.
- STEP 2 | View All Plans and choose the one to which want to Subscribe.
View the available plans and pick the one that best meets the security and compliance requirements for enterprise (see Prisma Cloud License Types).
- STEP 3 | Activate the subscription selected.
The subscription from the marketplace is for 100 workloads for a 12-month period. For a longer term or
to secure more workloads, please contact Palo Alto Networks Sales Representative.
- STEP 4 | Provide the details for provisioning Prisma Cloud instance.
26
1. Enter the personal and company information requested in the form.
2. Provide the Tenant Name that allows to identify Prisma Cloud instance.
The name enter here is displayed in the GCP marketplace subscription details and on the Prisma Cloud console.
3. Select the region where want Prisma Cloud tenant provisioned.
4. Select Register.
A message informs whether the registration was successful. After successfully register, Palo Alto Networks sends two emails: the first email is titled Welcome to Prisma Cloud
Support and enables to Set Password; the second email is titled Welcome to Prisma Cloud
and it includes a link to Get Started. Use this link to navigate to the Palo Alto Networks hub and log
in using the registered email address and the new password that configured.
5. Verify subscription details on GCP Marketplace.
When the subscription cycle ends, subscription will not be automatically renewed.
upgrade subscription at any time. If want to change plan to a more limited set of capabilities, the change is in effect at the end of the currently
committed subscription term.
- STEP 5 | Log in to the hub and click the Prisma Cloud tile to start using Prisma Cloud.
are now ready for Prisma Cloud—First Look and Prisma Cloud—Next Steps.
28
Access Prisma Cloud
The welcome email receive from Palo Alto Networks (noreply@prismacloud.paloaltonetworks.com)
includes a link to where access instance of Prisma Cloud. If are the first registered user, a Palo Alto Networks Customer Support Portal (CSP) account is created for and log in to
Prisma Cloud to start securing cloud deployments. For all other Prisma Cloud users, when Prisma Cloud system administrator adds to the tenant, receive two emails. Use the Welcome to Palo Alto
Networks Support email to activate the CSP account and set a password to access the Palo Alto Networks
Support portal before click Get Started in the Welcome to Prisma Cloud email to log in to Prisma Cloud instance.
The link in welcome email varies depending on whether are using Palo Alto Networks Customer
Support Portal (CSP) credentials to log in or if are using a third-party identity provider (IdP) for Single
Sign-On (SSO):
If have a Palo Alto Networks CSP account and are not using a third-party IdP, the link enables to
log in directly to Prisma Cloud using the email address and password registered with CSP account.
If are using a third-party IdP and the login URL is configured on Prisma Cloud, the link redirects to
login page for IdP and log in using IdP credentials.
If are using a third-party IdP but the login URL is not configured on Prisma Cloud, must navigate to
IdP and click the Prisma Cloud tile there to log in using the credentials set up on IdP.
Browser Support—To access the Prisma Cloud administrator console, Chrome version 72 or later provides
the optimal user experience. The Prisma Cloud console is not explicitly tested on other browsers and, though we expect it to perform with graceful degradation, it is not guaranteed to work on other browsers.
Status Updates—Use the Prisma Cloud status page to view operational status and subscribe to updates
about the service.
- STEP 1 | Launch a web browser and access the URL for Prisma Cloud or go to the Palo Alto Networks
hub to access the app.
The URL for Prisma Cloud varies depending on the region and cluster on which tenant is deployed.
welcome email will include one of the following URLs that is specific to the tenant provisioned for you:
- https://app.prismacloud.io
- https://app2.prismacloud.io
- https://app3.prismacloud.io
- https://app4.prismacloud.io
- https://app.ca.prismacloud.io
- https://app.eu.prismacloud.io
- https://app2.eu.prismacloud.io -https://app.anz.prismacloud.io
- https://app.gov.prismacloud.io
- https://app.sg.prismacloud.io
- https://app.prismacloud.cn
On the hub, if see the Prisma Cloud tile in to the app because of a SAML error, it
likely means that do not have an account on that Prisma cloud instance. Contact your
system administrator for an account to access that instance.
- STEP 2 | Accept the EULA.
After accept the terms and conditions of use, use the get started guide to learn the basics.
- STEP 3 | Select Licensing and verify that have the correct Prisma Cloud License Types or get started
with Prisma Cloud—First Look.
- STEP 4 | Switch between Prisma Cloud instances.
If are responsible for monitoring clouds belonging to different organizations (tenants), use
Palo Alto Networks login credentials to access all tenants from the hub. To enable multitenant login
access, system administrator must add email address on each Prisma Cloud tenant (see Add
Administrative Users On Prisma Cloud).will then receive an email from Palo Alto Networks to get
started. By default, an administrator on Prisma cloud is designated as an Instance Administrator for that
tenant only on the hub. If want to change a role or enable access to other apps, see roles.
With Palo Alto Networks CSP credentials, click the app switcher on the hub and then select a different tenant to which to switch between instances.
If see the serial number for instance and want to change it to a descriptive label, navigate to the Settings page using gear > Manage Apps in the upper-right. Click directly
on the serial number and rename it. This new name displays only on the hub and it does
not automatically apply to Prisma Cloud instance name.
30
Prisma Cloud—First Look
When Access Prisma Cloud, first see the Alerts. then use the following tabs to interact
with the data and visualize the traffic flow and connection details to and from the different resources in
cloud deployment; review the default policy rules and compliance standards; and explore how the web
interface is organized to help and DevSecOps teams to monitor cloud resources.
- Dashboard—Get a graphical view of the health cloud (security and compliance posture) of assets
deployed across multiple public cloud environments.
- Investigate—Identify security threats and vulnerabilities, create and save investigative queries, and
analyze impacted resources.
- Policies—Configure policies to maintain compliance and security.
- Compliance—Monitor cloud accounts against compliance standards (such as NIST, SOC 2, HIPAA, PCI, GDPR, ISO 27001:2013), create custom policies, and generate reports for offline viewing.
- Alerts—View the list of discovered violations and anomalies, drill in to the details and look up
remediation options, and create alert rules and notification templates.
- Compute—Deploy the Prisma Cloud Defender in hosts, containers, and serverless environments to
identify vulnerabilities, detect anomalous behavior, and provide least privilege micro-segmentation and
runtime defense across the entire application lifecycle from CICD to runtime. deploy Prisma Cloud Defenders on heterogeneous environments, including Windows, Linux, Kubernetes, OpenShift, AWS Lambda, Azure Functions, GCP Cloud Functions, and more. Defenders can create IP table rules on
the host to observe network traffic and to enforce both CNNF and CNAF firewall policies. For details
see the Prisma Cloud Administrator’s Guide (Compute) .
- Settings—Add new cloud accounts and set up Prisma Cloud administrative users. also set up
account groups, create users, associate roles and permissions, add external integrations including SAML
integration (SSO), view audit logs, add trusted IP addresses, view license and usage reporting, and modify
the browser session timeout.
- Profile—Maintain profile, manage credentials, and change the UI display language. Prisma Cloud—Next Steps
Now that have familiarized yourself with Prisma™ Cloud, here are some things to consider next so that
begin protecting cloud resources:
- Add the NAT Gateway IP Addresses for Prisma Cloud so that access Prisma Cloud from your
network.
- Create and Manage Account Groups on Prisma Cloud.
- Connect Cloud Platform to Prisma Cloud.
- Add Administrative Users On Prisma Cloud.
- Create an Alert Rule.
- Prisma Cloud Integrations.
- Prisma Cloud Dashboards.
- Investigate Incidents on Prisma Cloud.
- Create a Policy on Prisma Cloud.
- Add a New Compliance Report.
- Deploy Prisma Cloud Defenders for securing host, container, and serverless functions.
32
NAT Gateway IP Addresses for Prisma Cloud
Prisma™ Cloud uses the following NAT gateway IP addresses. To ensure that access Prisma Cloud
and the API for any integrations that enabled between Prisma Cloud and incidence response
workflows, or for Prisma Cloud Defenders to communicate with the Prisma Cloud Compute Console, review the list and update the IP addresses in allow lists.
URL Source IP Address to Allow
app.prismacloud.io 3.217.51.44
3.218.144.244
34.199.10.120
34.205.176.82
34.228.96.118
52.201.19.205
app2.prismacloud.io 3.16.7.30
13.59.164.228
18.191.115.70
18.218.243.39
18.221.72.80
18.223.141.221
app3.prismacloud.io 34.208.190.79
52.24.59.168
52.39.60.41
52.26.142.61
54.213.143.171
54.218.131.166
app4.prismacloud.io 13.52.27.189
13.52.105.217
13.52.157.154
13.52.175.228
52.52.50.152
52.52.110.223
app5.prismacloud.io 3.128.141.242/32
3.129.241.104/32
3.130.104.173/32
3.136.191.187/32 URL Source IP Address to Allow
13.59.109.178/32
18.190.115.80/32
app.anz.prismacloud.io 3.104.252.91
13.210.254.18
13.239.110.68
52.62.75.140
52.62.194.176
54.66.215.148
app.ca.prismacloud.io 15.223.59.158
15.223.96.201
15.223.127.111
52.60.127.179
99.79.30.121
35.182.209.121
app.prismacloud.cn 52.82.89.61
52.82.102.153
52.82.104.173
52.83.179.1
52.83.70.13
52.83.77.73
app.eu.prismacloud.io 3.121.64.255
3.121.248.165
3.121.107.154
18.184.105.224
18.185.81.104
52.29.141.235
app2.eu.prismacloud.io 18.200.200.125
3.248.26.245
99.81.226.57
52.208.244.121
18.200.207.86
63.32.161.197
app.gov.prismacloud.io 15.200.20.182
15.200.89.211
34
URL Source IP Address to Allow
52.222.38.70
52.61.207.0
15.200.68.21
15.200.146.166
app.sg.prismacloud.io 13.250.248.219
18.139.183.196
52.76.28.40
52.76.70.227
52.221.36.124
52.221.157.53
Prisma Cloud Compute Console On the Compute > Manage > System >
Downloads, find the region in the URL for Path
to Console. Use that region to identify the destination IP address, which must allow
or add as trusted to access the Prisma Cloud
Compute console.
us-east1—35.196.73.150
us-west1—35.233.225.166
asia-northeast1—34.84.195.213
europe-west3—34.89.249.72
These IP addresses are not the outgoing address for alerts or
cloud discovery. Access the Prisma Cloud REST API
Prisma Cloud has a REST API that enables configure custom integrations for cloud security needs.
can, for example, use it to automate sending alert notifications to an in-house tool use or to extend
the DevOps security capabilities for a tool that does not have an extension or plugin for Prisma Cloud. Most
actions supported on the Prisma Cloud web interface are available with the REST API. See the Prisma Cloud
REST API reference for details about the REST API.
Watch this!
Prisma Cloud requires an API access key to enable programmatic access to the REST API. By default, only
the System Admin has API access and can enable API access for other administrators. To generate an access
key, see Create and Manage Access Keys. After obtain an access key, submit it in a REST API
request to generate a JSON Web Token (JWT). The JWT is then used to authenticate all subsequent REST
API requests on Prisma Cloud.
- STEP 1 | Obtain a JWT to authenticate API requests.
The following is an example of a cURL call for a REST API request that returns a JWT. Prisma Cloud
access key provides the request parameters. Note that an access key is made up of two parts: an Access
Key ID and a Secret Key. In the body parameters, specify Access Key ID as the string value for the userName and Secret Key as the string value for the password.

curl -X POST \
https://api.prismacloud.io/login \
- H 'Content-Type: application/json' \
- d '{"username":"<Access Key ID>","password":"<Secret Key>"}'

The following shows the response for a successful request.

{
 "token": "<JWT>",  "message": "login_successful",  "customerNames": [
 {
 "customerName": "Test",  "tosAccepted": true
 }
 ]
}

The value for token in the response is the JWT will use to authorize subsequent REST API
requests.
- STEP 2 | Authenticate Using the JWT.
36
Specify the JWT in an HTTP header parameter for every Prisma Cloud REST API request. The following
table shows the details of the header parameter.
HTTP Header Parameter Key Value
x-redlock-auth <JWT>
- STEP 3 | Refresh the JWT.
The JWT is valid for 10 minutes, so must refresh the token for continued access to the Prisma Cloud API. If make an API request with an expired JWT request, will receive an HTTP 401
(Unauthorized) response. The following example is a cURL call that makes an API request to obtain a refreshed JWT.

curl -X GET \
https://api.prismacloud.io/auth_token/extend \
- H 'Content-Type: application/json' \
- H 'x-redlock-auth:<current JWT>'

The following is an example of the response to a successful request to refresh a JWT.

{
 "token": "<JWT>",  "message": "login_successful",  "customerNames": [
 {
 "customerName": "Test",  "tosAccepted": true
 }
 ]
}

The value for token in the response is the new JWT must specify in the HTTP header parameter of subsequent REST API requests. Prisma Cloud FAQs
- Account Onboarding and SSO
- Password and Help
- Policy and Alerts Account Onboarding and SSO
- Explain VPC flow logs
VPC flow logs provide a unidirectional record of network traffic that inform about how packets
flowed from A to B and from B to A (as in a separate record). They provide no direct insight into which
endpoint is the server in any conversation or which endpoint initiated a conversation. could look at
which flow record has a lower timestamp and assume that the source in that record is the client but, in
the case of VPC flow logs, log collection is aggregated over several-minute windows, which removes the precision required to make this a reliable indicator. Additionally, long-lived connections and connections
that appear on the boundaries of batches of logs will defeat this heuristic. There are other factors to
consider, such as source port vs destination port. also compare the count of distinct peers for a given endpoint IP address and port. Prisma™ Cloud evaluates all of these conditions, plus others, with
a weight given to each measure and a historical bias. However, these measures are heuristics and are
therefore not perfect.
- Why do we need to list permissions for Key Vault in Azure?
To support policies based on Azure Key Vault, Prisma Cloud needs to ingest Key Vault. Prisma Cloud
does not ingest the keys or secretsonly their IDs and other metadata.
- Which SAML binding should I use for SSO?
The Prisma Cloud SAML endpoint supports HTTP POST binding.
Password and Help
- What are the rules for password similarity and reuse in Prisma Cloud?
When create a new password, we check for its similarity with the current password. The measure
of similarity between the new and the current and old password strings is determined by the minimum
number of single-character edits, such as insertions, deletions, substitutions that are required to change
one word in to another. We do not accept the new password string if the similarity with the current or
old passwords is 70% or higher.
Example: If current password isMenloPark.123!, then cannot use M3nl0P@rk.123! but could use ParkMenlo.123!.
- What are the complexity requirements for creating Prisma Cloud passwords?
Passwords must have a minimum of 8 characters and a maximum of 30 characters and include one
or more of each of the following: an uppercase letter, a lowercase letter, a digit (0 to 9), and a special
character (“~”, “`”, “!”, “@”, “#”, “$”, “%”, “^”, “&”, “*”, “(”, “)”, “-”, “_”, “+”, “=”, “{”, “}”, “[”, “]”, “|”, “\”, “:”, “;”, “’”, “,”, ““”, “.”, “<”, “>”, “?”, and “/”).
- Help icon
Check to ensure that browser allows pop-ups. Check whether advertisement blocking
software is blocking Prisma Cloud and, if so, add the URL for Prisma Cloud instance and
app.pendo.io to the allow list. Check and disable any local firewall rules or proxies that are blocking
either or both of these URLs.
- Where do I submit documentation requests or report errors in the documentation?
38
Please let us know how we are doing at documentation@paloaltonetworks.com. When writing to us
about a documentation error, please include the URL for the page where see the issue.
- Where do I find documentation for the Compute tab for securing host, container, and serverless
functions?
If are using the Prisma Cloud Enterprise edition license, see Prisma Cloud Administrator’s Guide
(Compute). If are using Prisma Cloud Compute Edition license and are deploying and hosting it on
own, see Prisma Cloud Compute Edition Administrator’s Guide.
- How do I get technical help or open a support case?
Check the discussions on the Palo Alto Networks Live Community, and to open a support case, log in to
the Customer Support Portal.
Policy and Alerts -What happens when I have two alert rules for the same conditions—one with and one without auto
remediation?
The alert rule with auto remediation enabled takes precedence and the violation is automatically
resolved.
- With which threat intelligence feeds does Prisma Cloud integrate?
Prisma Cloud provides users with comprehensive threat intelligence and vulnerability data sourced
across multiple unique sources:
- Prisma Cloud Intelligence Stream: Our own collection of 30-plus upstream data sources across
commercial, open-source and proprietary feeds; offering vulnerability data for hosts, containers and
functions as well as malware and IP-reputation lists.
- Palo Alto Networks sources: In addition to AutoFocus, Prisma Cloud integrates with WildFire for malware scanning as part of data security capabilities.
When combined with AutoFocus, Prisma Cloud enables unmatched alert accuracy with the risk clarity
required to effectively protect today’s highly dynamic, distributed cloud environments.
- Third-party sources: Prisma Cloud integrates with data provided from Qualys, Tenable, AWS
GuardDuty, AWS Inspector and others to provide a single view into risk within cloud environments.
Each threat intelligence feed provides a classification for each of the IP addresses they include, and
Prisma Cloud uses this data to identify bad actors. Some IP addresses that have been known to launch
Command and Control traffic or DDOS attacks, are classified as outright malicious. Other IP addresses
are listed as suspicious, and these have demonstrated patterns of association with other malicious sites
or have indicators—file properties, behaviors, and activities— that are suspicious or highly suspicious. For details on AutoFocus, see AutoFocus artifacts.
- How often does Prisma Cloud retrieve data from cloud services that provide automated security
assessment or threat detection?
If set up an integration with Qualys, Tenable, Amazon GuardDuty, or AWS Inspector for additional
context on risks in the cloud, Prisma Cloud retrieves data from these services periodically. The data from
Qualys and Tenable is retrieved every hour; the data from AWS Inspector and Amazon GuardDuty is
retrieved at every ingestion cycle.
- After I update a config policy query, how long does it take to automatically resolve alerts that no longer
match this policy?
When a Config-based policy query is changed, all the alerts generated by this policy are re-evaluated at
the next scan. Alerts that are no longer valid because of the policy change are automatically resolved.
- What is the list of web applications that Prisma Cloud automatically classifies? Port Number Application Classification
0 ICMP
21 FTP
22 SSH
23 TELNET
25 SMTP
53 DNS
80 Web (80)
88 Kerberos
111 RPC (111)
135 RPC (135)
143 IMAP
389 LDAP
443 Web (443)
444 SNPP
445 Generic (445)
514 Syslog
587 SMTP
636 LDAP (TLS)
995 IMAP
1433 SQL Server
1515 OSSEC
1521 Oracle
2376 Docker TLS
3128 Web Proxy
3268 Active Directory (GC)
40
Port Number Application Classification
3306 My SQL
3389 RDP
5050 Mesos Server
5432 Postgres
5439 Redshift
5671 RabbitMQ
5672 RabbitMQ
5900 VNC
6168 Generic (6168)
6379 Redis
7200 Generic (7200)
7205 Generic (7205)
7210 MaxDB
8000 HTTP (8000)
8080 HTTP (8080)
8140 Puppet
8332 Bitcoin
8333 Bitcoin
8443 HTTP (8443)
8545 Ethereum (8545)
8888 HTTP (8888)
9000 Generic (9000)
9006 Web (9006)
9092 Kafka
9300 Elastic Search
9997 Splunk Logger Port Number Application Classification
15671 RabbitMQ WebUI
15672 RabbitMQ WebUI
27017 MongoDB
29418 Git
30000 Generic (30000)
30303 Ethereum (30303)
52049 NFS
55514 Syslog
60000 Generic
61420 Minuteman LB
61421 Minuteman LB
61668 Generic (61668) 43

## Connect Cloud Platform to Prisma Cloud
To begin monitoring the resources on cloud infrastructure, must first connect your
public cloud accounts to Prisma™ Cloud. When add cloud account to Prisma Cloud, the API integration between cloud infrastructure provider and Prisma Cloud is established
and begin monitoring the resources and identify potential security risks in your
infrastructure.
> Cloud Account Onboarding
> Onboard AWS Account
> Onboard Azure Account
> Onboard Google Cloud Platform (GCP) Account
> Onboard Alibaba Cloud Account
> Cloud Service Provider Regions on Prisma Cloud Cloud Account Onboarding
To get the most out of investment in Prisma™ Cloud, first need to add cloud accounts to
Prisma Cloud. This process requires that have the correct permissions to authenticate and authorize the connection and retrieval of data.
Prisma Cloud administrators with the System Administrator and Cloud Provisioning Administrator roles
can use the cloud account onboarding guided tour for a good first-run experience with all supported cloud
platforms—Alibaba Cloud, AWS, Azure, and Google Cloud. The workflow provides the context need to
make decisions based on own security and compliance requirements and it uses automation scripts—Cloud Formation templates for AWS or Terraform templates for Azure and GCP—to create the custom roles
and enable the permissions required to add a cloud account.
When log in to Prisma Cloud for the first-time, the guided tour displays after the welcome tour and
prompts to pick a cloud platform to add to Prisma Cloud.
will make a few choices and provide basic account details to retrieve configuration logs and get started
with Prisma Cloud for monitoring and visibility. If want to ingest data from event logs and flow logs, need to perform additional tasks.
- Onboard AWS Account
- Onboard Azure Account
- Onboard Google Cloud Platform (GCP) Account
- Onboard Alibaba Cloud Account Onboard AWS Account
To connect AWS Organizations (only supported on public AWS) or AWS accounts on the public
AWS, AWS China, AWS GovCloud account to Prisma™ Cloud, must complete some tasks on the AWS management console and some on Prisma Cloud. The onboarding workflow enables to create a Prisma Cloud role with either read-only access to traffic flow logs or with limited read-write access
to remediate incidents. With the correct permissions, Prisma Cloud can successfully connect to and access
AWS account(s).
In addition to scanning AWS resources against Prisma Cloud policies for compliance
and governance issues, also scan objects in AWS S3 buckets for data security
issues. The data security capabilities include predefined data policies and associated data classification profiles such as PII, Financial, or Healthcare & Intellectual Property that scan
objects stored in the S3 bucket to identify exposure—how sensitive information is kept
private, or exposed or shared externally, or allows unauthorized access. Prisma Cloud Data Security capability is in Limited GA and available to select Prisma Cloud Enterprise Edition
customers only.
- Add an AWS Cloud Account on Prisma Cloud.
- Add an AWS Organization to Prisma Cloud
- Update an Onboarded AWS Account
- Set Up the Prisma Cloud Role for AWS—Manual
- AWS APIs Ingested by Prisma Cloud
Add an AWS Cloud Account on Prisma Cloud
Use the following workflow to add AWS public, AWS China, or AWS GovCloud accounts to Prisma™
Cloud. To add AWS Organizations on Prisma Cloud, see Add an AWS Organization to Prisma Cloud.
If want to download and review the CloudFormation templates, get the S3 URLs from here:
Role S3 Template URL
AWS Public Cloud—AWS account and AWS Organization, master account
Read-Only https://s3.amazonaws.com/redlockpublic/cft/rl-read-only.template
Read-Write (Limited) https://s3.amazonaws.com/
redlock-public/cft/rl-read-andwrite.template
For member accounts within AWS
OrganizationsRead-Only
https://s3.amazonaws.com/
redlock-public/cft/rl-read-onlymember.template
For member accounts within AWS
OrganizationsRead-Write (Limited)
https://s3.amazonaws.com/redlockpublic/cft/rl-read-and-writemember.template Role S3 Template URL
AWS GovCloud
Read-Only https://s3.amazonaws.com/redlockpublic/cft/redlock-govcloud-readonly.template
Read-Write (Limited) https://s3.amazonaws.com/redlockpublic/cft/redlock-govcloud-readand-write.template
AWS China
Read-Only https://s3.amazonaws.com/redlockpublic/cft/rl-cn-read-only.template
Read-Write (Limited) https://s3.amazonaws.com/redlockpublic/cft/rl-cn-read-andwrite.template
- STEP 1 | Before begin.
If would like Prisma Cloud to ingest VPC flow logs and any other integrations, such as Amazon
GuardDuty or AWS Inspector, must enable these services on the AWS management console. The CFT enables the ingestion of configuration data and AWS CloudTrail logs (audit events) only. VPC flow
logs and any other integrations, such as Amazon GuardDuty or AWS Inspector are retrieved only if previously enabled these services for the AWS account that are onboarding.
1. Decide whether want to manually create the roles to authorize permissions for Prisma Cloud.
The onboarding flow automates the process of creating the Prisma Cloud role and adding the permissions required to monitor and/or protect AWS account. If want to create these roles
manually instead, see Set Up the Prisma Cloud Role for AWS—Manual.
2. Create a CloudWatch log group.
The CloudWatch log group defines where the log streams are recorded.
1. Select Services > CloudWatch > Logs > Actions > Create log group.
2. Enter a name for the log group and click Create log group.
3. Enable flow logs.
1. Select Services > VPC > VPCs.
2. Select the VPC to enable flow logs for and select Actions > Create flow log.
3. Set the Filter to Accept or All.
Setting the filter to All enables Prisma Cloud to retrieve accepted and rejected traffic from the flow logs. Setting the filter to Accept retrieves Accepted traffic only. If set the filter to Reject, Prisma Cloud will not retrieve any flow log data.
4. Verify that the Destination is configured to Send to CloudWatch Logs.
If set the destination as Amazon S3 bucket, Prisma Cloud will be unable to retrieve the data.
5. Select the Destination log group created above.
6. Create new or use existing IAM role to publish flow logs to the CloudWatch log group. If are an existing IAM role to publish logs to the CloudWatch log group, must edit the IAM role to include the following permissions.
 {
"Statement": [
{
"Action": [
"logs:CreateLogGroup", "logs:CreateLogStream", "logs:DescribeLogGroups", "logs:DescribeLogStreams", "logs:PutLogEvents"
], "Effect": "Allow", "Resource": "*"
}
]
}
will also need to Enable trust relationshipso that the IAM role can access the CloudWatch Log
group.
- STEP 2 | Access Prisma Cloud and select Settings > Cloud Accounts > Add New.
- STEP 3 | Select AWS as the Cloud to Protect.
- STEP 4 | Enter a Cloud Account Name.
A cloud account name is auto-populated for you. replace it with a cloud account name that
uniquely identifies AWS account on Prisma™ Cloud. - STEP 5 | Select the Mode.
Decide whether to enable permissions to only monitor (read-only access) or to monitor and protect
(read-write access) the resources in cloud account. selection determines which AWS Cloud
Formation Template (CFT) is used to automate the process of creating the custom role required for Prisma Cloud.
- STEP 6 | Set up the Prisma Cloud role on AWS.
To automate the process of creating the Prisma Cloud role that is trusted and has the permissions
required to retrieve data on AWS deployment, Prisma Cloud uses a CFT. The CFT enables the ingestion of configuration data and AWS CloudTrail logs (audit events) only, and it does not support
the ability to enable VPC flow logs for AWS account or any other integrations, such as Amazon
GuardDuty or AWS Inspector.
1. Open a new tab on browser and sign in to AWS account for AWS public cloud or
AWS GovCloud deployment that want to protect using Prisma Cloud.
To onboard an AWS GovCloud account, Prisma Cloud instance must be on https:/
app.gov.prismacloud.io
2. Click back to the Prisma Cloud console, and in the onboarding flow, select Create Stack.
will be directed to the AWS CloudFormation stack for AWS public or AWS GovCloud
environment, and the following details are automatically filled in for you:
- Stack Name—The default name for the stack is PrismaCloudApp.
- External ID—The Prisma Cloud ID, a randomly generated UUID that is used to enable the trust
relationship in the role's trust policy.
- Prisma Cloud Role Name—The name of the role that will be used by Prisma Cloud to authenticate
and access the resources in AWS account. 3. Accept the IAM acknowledgment for resource creation and select Create Stack.
The stack creation is initiated. Wait for the CREATE_COMPLETE status.
4. Select Outputs and copy the value of the Prisma CloudARN.
The Prisma Cloud ARN has the External ID and permissions required for enabling authentication
between Prisma Cloud and AWS account. 5. Paste the Role ARN and click Next.
- STEP 7 | Select one or more account groups and click Next.
must assign each cloud account to an account group and Create an Alert Rule to associate with that
account group to generate alerts when a policy violation occurs. - STEP 8 | Review the onboarding Status of AWS account on Prisma Cloud.
The status check verifies that VPC flow logs are enabled on at least 1 VPC in account, and audit
events are available in at least one region on AWS CloudTrail.
- Prisma Cloud checks whether Compute permissions are enabled only if have one
or more compute workloads deployed on the AWS cloud accounts that are onboarded.
And the cloud status transitions from green to amber only when have compute workloads deployed and the additional permissions are not enabled for monitor, or
monitor and protect modes.
- If have services that are not enabled on AWS account, the status screen
provides some details.
Add an AWS Organization to Prisma Cloud
If have consolidated access to AWS services and resources across company within AWS
Organizations, onboard the AWS master account on Prisma Cloud. When enable the AWS
Organizations on the AWS management console and add the root or master account that has the role of a payer account that is responsible for paying all charges accrued by the accounts in its organization, all
member accounts within the hierarchy are added in one streamlined operation on Prisma Cloud.
Figure 1: Image from AWS documentation
In this workflow, first deploy a CloudFormation template in the master account to create the Prisma Cloud role to monitor, or monitor and protect resources deployed on the master account. And then, use CloudFormation StackSets to automate the creation of the Prisma Cloud role, which authorizes
Prisma Cloud to access each member account. When then add a new member account to AWS
organization, it is onboarded automatically on Prisma Cloud within a few (up to six) hours. -If want to exclude one or more Organizational Units (OUs) and all the member
accounts it includes, manually disable individual member accounts on Prisma Cloud after they are onboarded. Alternatively, to onboard a subset of accounts, exclude the OUs when deploying the StackSet so that the PrismaCloud role is only
created in the OUs for which want to onboard accounts.
- If had previously onboarded AWS master account as a standalone or individual
account, must re-add the account as an Organization. All existing data on assets
monitored, alerts generated, or account groups created are left unchanged.
After onboard account as an AWS Organization, cannot roll back. To add
the account as a standalone or individual account, must delete the Organization on
Prisma Cloud and use the instructions to Add an AWS Cloud Account on Prisma Cloud.
- If had previously onboarded an AWS account that is a member of the AWS
Organization that now add on Prisma Cloud, all existing data on assets
monitored, alerts generated, or account groups created are left unchanged. On Prisma Cloud, the member account will be logically grouped under the AWS Organization.
When delete the AWS Organization on Prisma Cloud, recover all the existing
data related to these accounts if re-onboarded within 24 hours. After 24 hours, the data is deleted from Prisma Cloud.
- Add a New AWS Organization Account on Prisma Cloud
- Update an Onboarded AWS Organization
Add a New AWS Organization Account on Prisma Cloud
Add AWS Organization on Prisma Cloud.
- STEP 1 | Access Prisma Cloud and select Settings > Cloud Accounts > Add New.
- STEP 2 | Select AWS as the Cloud to Protect.
- STEP 3 | Enter a Cloud Account Name and onboard Organization.
A cloud account name is auto-populated for you. replace it with a cloud account name that
uniquely identifies AWS Organization on Prisma™ Cloud. - STEP 4 | Select the Mode.
Decide whether to enable permissions to only monitor (read-only access) or to monitor and protect
(read-write access) the resources in cloud account. selection determines which AWS Cloud
Formation Template (CFT) is used to automate the process of creating the custom role required for Prisma Cloud.
- STEP 5 | Set up the Prisma Cloud role on the AWS master account.
To automate the process of creating the Prisma Cloud role that is trusted and has the permissions
required to retrieve data on AWS deployment, Prisma Cloud uses a CFT. The CFT enables the ingestion of configuration data and AWS CloudTrail logs (audit events) only, and it does not support
the ability to enable VPC flow logs for AWS account or any other integrations, such as Amazon
GuardDuty or AWS Inspector.
1. Open a new tab on browser and sign in to the AWS master account that want to add on
Prisma Cloud.
2. Click back to the Prisma Cloud console, and in the onboarding flow, select Create Stack.
will be directed to the AWS CloudFormation stack for AWS environment, and the following
details are automatically filled in for you:
- Stack Name—The default name for the stack is PrismaCloudApp.
- External ID—The Prisma Cloud ID, a randomly generated UUID that is used to enable the trust
relationship in the role's trust policy.
- Prisma Cloud Role Name—The name of the role that will be used by Prisma Cloud to authenticate
and access the resources in AWS account. 3. Accept the IAM acknowledgment for resource creation and select Create Stack.
The stack creation is initiated. Wait for the CREATE_COMPLETE status.
4. Select Outputs and copy the value of the Prisma CloudARN.
The Prisma Cloud ARN has the External ID and permissions required for enabling authentication
between Prisma Cloud and AWS account.
5. Paste the Master Role ARN and click Next. - STEP 6 | Create a StackSet to create the Prisma Cloud role within each member account.
AWS StackSets enables to automate the process of creating the Prisma Cloud role across multiple
accounts in a single operation.
1. Download the template file.
Get the template file:
- For member accounts with read-only access permissions (Monitor mode)—https://
s3.amazonaws.com/redlock-public/cft/rl-read-only-member.template
- For member accounts with the read-write access permissions (Monitor & Protect mode)—https://
s3.amazonaws.com/redlock-public/cft/rl-read-and-write-member.template
2. On the AWS management console, select Services > CloudFormation > StackSets > Create StackSet.
Verify that are logged in to the AWS master account.
3. Upload the template file and click Next, then enter a StackSet Name.
4. In Parameters, enter the values for PrismaCloudRoleName and ExternalId.
The PrismaCloudRoleName must include Org within the string.
5. Click Next and select Service managed permissions.
6. Click Next and select Deploy to organization under Deployment targets.
If do not want to onboard all member accounts, select Deploy to organization unit
OUsand deploy the Stackset only to selected OUs only.
7. Set Automatic deployment Enabled, and Account removal behavior Delete stacks.
8. In Specify regions, select a region.
9. In Deployment Options, Maximum concurrent accounts, select Percentage and set it to 100.
10.In Deployment Options, Failure tolerance, select Percentage and set it to 100.
11.Click Next, and review the configuration. 12.Select I acknowledge that AWS CloudFormation might create IAM resources with custom names
and Submit.
The StackSet creation is initiated. Wait for the SUCCEEDED status. When the process completes, each member account where the role was created is listed under Stack instances on the AWS
management console.
13.Select Parameters and copy the values for PrismaCloudRoleName and ExternalId.
- STEP 7 | Configure the member account role details on Prisma Cloud.
Use the details copied from the previous step to set up the trust relationship and retrieve data from
the member accounts.
1. Paste the Member Role Name and Member External ID.
2. Select I confirm the stackset has created Prisma roles in member accounts successfully and click
Next. If have a large number of member accounts, it may take a while to create the role in each
account and list it for verification. If want to verify that the role was created in all accounts, do
not select the checkbox. edit the cloud account settings later and onboard the member
accounts. If do not select the checkbox, only the master account will be onboarded to Prisma Cloud.
- STEP 8 | Select an account group and click Next.
During initial onboarding, must assign all the member cloud accounts with the AWS Organization
hierarchy to an account group. Then, Create an Alert Rule to associate with that account group so that
alerts are generated when a policy violation occurs.
If would like to selectively assign AWS member accounts to different account groups
on Prisma Cloud, edit the cloud account settings later.
- STEP 9 | Review the onboarding Status of AWS Organization on Prisma Cloud.
The status check verifies that VPC flow logs are enabled on at least 1 VPC in master account, and audit events are available in at least one region on AWS CloudTrail. It also displays the number of member accounts that are provisioned with the Prisma Cloud role.
If did not select the I confirm the stackset has created Prisma roles in member
accounts successfully checkbox, the status screen displays the onboarding status of the master account but does not list the number of member accounts. Update an Onboarded AWS Organization
In addition to updating the CFT stack for enabling permissions for new services, use this workflow
to update the account groups that are secured with Prisma Cloud, change the protection mode from
Monitor to Monitor & Protect or the reverse way, and redeploy the Prisma Cloud role in member accounts.
opt to onboard all member accounts under Organizations hierarchy, or selectively add the OUs
whose member accounts want to onboard on Prisma Cloud.
- STEP 1 | Provision the Prisma Cloud role on the AWS master account.
1. Download the template file.
Get the template file for needs:
- For master accounts with the read-only access for Monitor mode—https://s3.amazonaws.com/
redlock-public/cft/rl-read-only.template
- For member accounts with the read-write access for Monitor & Protect mode—https://
s3.amazonaws.com/redlock-public/cft/rl-read-and-write.template
2. Log in to master account on the AWS management console.
3. Select Services > CloudFormation > Stacks.
4. Select PrismaCloudApp Stack and click Update Stack.
5. Replace the existing template with the template downloaded earlier.
6. Click Next, review the configuration.
7. Select I acknowledge that AWS CloudFormation might create IAM resources with custom names
and Submit.
- STEP 2 | Configure the member accounts.
1. Log in to Master Account on the AWS management console.
2. Select Services > CloudFormation > > StackSets.
3. Select the Prisma stack set and Edit StackSet Details.
4. Replace the current template with the downloaded template.
5. Click Next and enter values for PrismaCloudRoleName and ExternalId.
6. Click Next and verify Service managed permissions is selected.
7. Select Deploy To Organizational units (OUs), and Under Organizational units (OUs), select all the OUs that are displayed, or enter the AWS OU ID.
To enter Organization Root ID use the format r-[0-9a-z]{4,32}. For example, r-6usb. 8. In Specify regions, select a region from the drop-down.
9. In Deployment Options, Maximum concurrent accounts, select Percentage and set it to 100.
10.In Deployment Options, Failure tolerance, select Percentage and set it to 100.
11.Click Next, and review the configuration.
12.Select I acknowledge that AWS CloudFormation might create IAM resources with custom names
and Submit.
The StackSet creation is initiated. Wait for the SUCCEEDED status. When the process completes, each member account where the role was created is listed under Stack instances.
13.Select Parameters and copy the values for PrismaCloudRoleName and ExternalId.
- STEP 3 | Access Prisma Cloud and select the AWS Organization account want to modify.
1. Select Settings > Cloud Accounts and select the account.
2. (Optional) Select a differentaccount group and click Next.
During initial onboarding, must assign all the member cloud accounts with the organization
hierarchy to one account group.
cn now edit to selectively assign AWS member accounts to different account
groups on Prisma Cloud.
- STEP 4 | Review the onboarding Status of AWS organization on Prisma Cloud.
The status check verifies that VPC flow logs are enabled on at least 1 VPC in master account, and audit events are available in at least one region on AWS CloudTrail. It also displays the number of member accounts that are provisioned with the Prisma Cloud role. Update an Onboarded AWS Account
After add cloud account to Prisma Cloud, may need to update the PrismaCloud stack to
provide additional permissions for new policies that are frequently added to help monitor cloud
account and ensure that have a good security posture. When update the CFT stack, Prisma Cloud can
ingest data on new services that are supported. These CFTs are available directly from the Prisma Cloud
administrative console and are also accessible from the S3 bucket. For instruction on updating AWS
Organization, see Add an AWS Organization to Prisma Cloud.
Role S3 Template URL
AWS Public Cloud—AWS account and AWS Organization, master account
Read-Only https://s3.amazonaws.com/redlockpublic/cft/rl-read-only.template
Read-Write (Limited) https://s3.amazonaws.com/
redlock-public/cft/rl-read-andwrite.template
For member accounts within AWS
OrganizationsRead-Only
https://s3.amazonaws.com/
redlock-public/cft/rl-read-onlymember.template
For member accounts within AWS
OrganizationsRead-Write (Limited)
https://s3.amazonaws.com/redlockpublic/cft/rl-read-and-writemember.template Role S3 Template URL
AWS GovCloud
Read-Only https://s3.amazonaws.com/redlockpublic/cft/redlock-govcloud-readonly.template
Read-Write (Limited) https://s3.amazonaws.com/redlockpublic/cft/redlock-govcloud-readand-write.template
AWS China
Read-Only https://s3.amazonaws.com/redlockpublic/cft/rl-cn-read-only.template
Read-Write (Limited) https://s3.amazonaws.com/redlockpublic/cft/rl-cn-read-andwrite.template
In addition to updating the CFT stack for enabling permissions for new services, use this workflow
to update the account groups that are secured with Prisma Cloud or to change the protection mode from
Monitor to Monitor & Protect or the reverse way.
- STEP 1 | Log in to the Prisma Cloud administrative console.
- STEP 2 | Select the AWS cloud account want to modify.
Select Settings > Cloud Accounts and click on the name of the cloud account to manage from the list of cloud accounts.
- STEP 3 | (Optional)Change the account groups want to monitor.
- STEP 4 | (To change permissions for the Prisma Cloud role) Update the Prisma Cloud App using the CloudFormation template (CFT).
1. Click the link to download the latest template and follow the instructions to update the stack.
2. Update the stack either using the AWS console or using the AWS CLI.
- Log in to AWS console.
- Select Services > CloudFormation > Stacks.
- Select the PrismaCloudApp stack to update and select Update.
Select Replace current template and Upload a template file downloaded earlier; can
optionally provide the Amazon S3 URL listed in the table above. If decide to create a new stack instead of updating the existing stack, must copy the ExternalID and PrismaCloudRoleARN values from the CFT outputs.
- Configure stack options.
- Click Next and verify the settings.
- Preview changes to the CloudFormation template for the role updated.
- Update CFT.
If created a new stack, must log in to the Prisma Cloud administrative console
and select cloud account on Settings > Cloud Accounts to enter the ExternalID and
PrismaCloudRoleARN values from the CFT outputs. -Check the Status to verify that Prisma Cloud can successfully retrieve information on cloud
resources.
- Use AWS Command Line Interface to deploy the updated Prisma Cloud App stack.
- Using the AWS CLI tool, enter the following command to retrieve the latest CloudFormation
template.
Role CLI Command
AWS Public cloud
Read-Only wget https://s3.amazonaws.com/redlockpublic/cft/rl-read-only.template --quiet -O /
tmp/rl-read-only.template
Read-Write (Limited) wget https://s3.amazonaws.com/redlockpublic/cft/rl-read-and-write.template --quiet -
O /tmp/rl-read-and-write.template
AWS GovCloud
Read-Only wget https://s3.amazonaws.com/redlockpublic/cft/redlock-govcloud-readonly.template --quiet -O /tmp/rl-readonly.template
Read-Write (Limited) wget https://s3.amazonaws.com/redlockpublic/cft/redlock-govcloud-read-andwrite.template --quiet -O /tmp/rl-read-andwrite.template Role CLI Command
AWS China
Read-Only wget https://s3.amazonaws.com/redlockpublic/cft/rl-cn-read-only.template --quiet -
O /tmp/rl-cn-read-only.template
Read-Write (Limited) wget https://s3.amazonaws.com/redlockpublic/cft/rl-cn-read-and-write.template --
quiet -O /tmp/rl-cn-read-and-write.template
- Enter the following command to deploy the updated CloudFormation template.
Replace with the correct name for the CloudFormation template, current stack name, role ARN, and External ID to overwrite the current stack or enter new values to create a new stack.
- Read-Only—aws cloudformation deploy --template-file /tmp/<RedLockcloudformation-template-name> --stack-name <Stack Name> --parameteroverrides RedlockRoleARN=<Role ARN> ExternalID=<xxxxxxxxxx> --
capabilities CAPABILITY_NAMED_IAM
- Read-Write (Limited)—aws cloudformation deploy --template-file /tmp/
<RedLock-cloudformation-template-name> --stack-name <Stack Name> --
parameter-overrides RedlockRoleARN=<Role ARN> ExternalID=<xxxxxxxxxx>
- -capabilities CAPABILITY_NAMED_IAM
Set Up the Prisma Cloud Role for AWS—Manual
If do not want to use the guided onboarding flow that automates the process of creating the roles
required for Prisma™ Cloud to monitor or monitor and protect accounts on AWS, must create
the roles manually. In order to monitor AWS account, must create a role that grants Prisma Cloud
access to flow logs and read-only access (to retrieve and view the traffic log data) or a limited readwrite access (to retrieve traffic log data and remediate incidents). To authorize permission, must copy
the policies from the relevant template and attach it to the role. Event logs associated with the monitored
cloud account are automatically retrieved on Prisma Cloud.
- STEP 1 | Log in to the AWS Management Console to create a role for Prisma Cloud.
Refer to the AWS documentation for instructions. Create the role in the same region as AWS
account, and use the following values and options when creating the role:
- Type of trusted entity: Another AWS Account and enter the Account ID*: 188619942792
- Select Require external ID, which is a unique alphanumeric string. generate a secure UUIDv4
at https://www.uuidgenerator.net/version4.
- Do not enable MFA. Verify that Require MFA is not selected. -Click Next and add the AWS Managed Policy for Security Audit.
Then, add a role name and create the role. In this workflow, later, will create the granular policies
and edit the role to attach the additional policies.
- STEP 2 | Get the granular permissions from the AWS CloudFormation template for AWS
environment.
The Prisma Cloud S3 bucket has read-only templates and read-and-write templates for the public AWS, AWS GovCloud, and AWS China environments.
1. Download the template need. Role S3 Template URL
AWS Public Cloud—AWS account and AWS Organization, master account
Read-Only https://s3.amazonaws.com/
redlock-public/cft/rl-readonly.template
Read-Write (Limited) https://s3.amazonaws.com/
redlock-public/cft/rl-read-andwrite.template
For member accounts within AWS
OrganizationsRead-Only https://s3.amazonaws.com/
redlock-public/cft/rl-read-onlymember.template
For member accounts within AWS
OrganizationsRead-Write (Limited) https://s3.amazonaws.com/
redlock-public/cft/rl-read-andwrite-member.template
AWS GovCloud
Read-Only https://s3.amazonaws.com/
redlock-public/cft/redlockgovcloud-read-only.template
Read-Write (Limited) https://s3.amazonaws.com/
redlock-public/cft/redlockgovcloud-read-and-write.template
AWS China
Read-Only https://s3.amazonaws.com/
redlock-public/cft/rl-cn-readonly.template
Read-Write (Limited) https://s3.amazonaws.com/
redlock-public/cft/rl-cn-readand-write.template
2. Identify the permissions need to copy.
To create the policy manually, will need to add the required permissions inline using the JSON
editor. From the read-only template get the granular permissions for the PrismaCloud-IAMReadOnly-Policy, and the read-write template lists the granular permissions for the PrismaCloudIAM-ReadOnly-Policy and the PrismaCloud-IAM-Remediation-Policy. For AWS accounts onboard to Prisma Cloud, if do not use the host, serverless functions, and container capabilities enabled with Prisma Cloud Compute, do not need the permissions associated with these roles:
- PrismaCloud-ReadOnly-Policy-Compute role—CFT used for Monitor mode, includes additional permissions associated with this new role to enable monitoring
of resources that are onboarded for Prisma Cloud Compute.
- PrismaCloud-Remediation-Policy-Compute role—CFT used for Monitor & Protect
mode, includes additional permissions associated with this new role to enable
read-write access for monitoring and remediating resources that are onboarded for Prisma Cloud Compute.
1. Open the appropriate template using a text editor.
2. Find the policies need and copy it to clipboard.
Copy the details for one or both permissions, and make sure to include the open and close
brackets for valid syntax, as shown below.
- STEP 3 | Create the policy that defines the permissions for the Prisma Cloud role.
Both the read-only role and the read-write roles require the AWS Managed Policy SecurityAudit Policy.
In addition, will need to enable granular permissions for the PrismaCloud-IAM-ReadOnly-Policy for the read-only role, or for the read-write role add the PrismaCloud-IAM-ReadOnly-Policy and the limited
permissions for PrismaCloud-IAM-Remediation-Policy.
1. Select IAM on the AWS Management Console.
2. In the navigation pane on the left, choose Access Management > Policies > Create policy.
3. Select the JSON tab.
Paste the JSON policies that copied from the template within the square brackets for Statement. If are enabling read and read-write permissions, make sure to append the readwrite permissions within the same Action statement.
4. Review and create the policy.
- STEP 4 | Edit the role created in Step 1 and attach the policy to the role. - STEP 5 | Required only if want to use the same role to access CloudWatch log group Update the trust
policy to allow access to the CloudWatch log group.
Edit the Trust Relationships to add the permissions listed below. This allow to ensure that role
has a trust relationship for the flow logs service to assume the role and publish logs to the CloudWatch
log group.

 {
 "Effect": "Allow",  "Principal": {
 "Service": "vpc-flow-logs.amazonaws.com"
 },  "Action": "sts:AssumeRole"
 }


- STEP 6 | Copy the Role ARN.
- STEP 7 | Resume with the account onboarding flow at Paste the Role ARN in Add an AWS Cloud
Account on Prisma Cloud Prisma Cloud on AWS China
To use Prisma Cloud to monitor or monitor and protect deployments in the AWS China regions of Ningxia and Beijing, require a Prisma Cloud instance in China. See Add an AWS Cloud Account on
Prisma Cloud for getting started with the configuration logs for resources.
Review the following sections to know what is not currently supported and available for the AWS China
deployments:
- Prisma Cloud Compute, which enables to secure containers and serverless functions is not available
on the Prisma Cloud instance in China.
- Prisma Cloud does not support the following services:
- Amazon GuardDuty
- AWS Inspector
- AWS Organization
- AWS does not support the following services:
- Amazon AppStream
- Amazon CloudSearch
- Amazon Data Pipeline
- AWS Glue
- Amazon Route53
- Amazon Simple Email Service
- AWS WAF
- AWS WAFv2
- Resource and alert attribution
AWS APIs Ingested by Prisma Cloud
The following are AWS APIs that are ingested by Prisma Cloud.
SERVICE API NAME IN PRISMA CLOUD
API Gateway -aws-apigateway-get-rest-apis
- aws-apigateway-get-stages
- aws-apigateway-domain-name
- aws-apigateway-base-path-mapping
- aws-apigateway-method
- aws-apigateway-client-certificates
AWS AutoScaling -aws-describe-auto-scaling-groups
- aws-ec2-autoscaling-launch-configuration
AWS Certificate Manager aws-acm-describe-certificate
Amazon Elastic Container Service (ECS) -aws-ecs-describe-task-definition
- aws-ecs-service
AWS CloudFormation aws-cloudformation-describe-stacks
AWS CloudFront aws-cloudfront-list-distributions SERVICE API NAME IN PRISMA CLOUD
Amazon CloudSearch aws-cloudsearch-domain
AWS CloudTrail -aws-cloudtrail-describe-trails
- aws-cloudtrail-get-event-selectors
- aws-cloudtrail-get-trail-status
AWS CloudWatch -aws-cloudwatch-describe-alarms
- aws-cloudwatch-log-group
- aws-logs-describe-metric-filters
Amazon Cognito -aws-cognito-identity-pool
- aws-cognito-user-pool
AWS Direct Connect Gateway -aws-directconnect-describe-gateway
Amazon EC2 -aws-describe-account-attributes
- aws-ec2-describe-instances
- aws-ec2-describe-images
- aws-ec2-describe-snapshots
- aws-ec2-describe-network-interfaces
- aws-ec2-key-pair
- aws-ec2-describe-volumes
- aws-ec2-elastic-address
Amazon MQ aws-mq-broker
Amazon SageMaker aws-sagemaker-notebook-instance
aws-sagemaker-endpoint
AWS Config aws-configservice-describe-configurationrecorders
Delivery Channels aws-describe-delivery-channels
Amazon DynamoDB aws-dynamodb-describe-table
AWS Database Migration Service -aws-dms-endpoint
- aws-dms-replication-instance
AWS Elastic Beanstalk -aws-elasticbeanstalk-environment
- aws-elasticbeanstalk-configuration-settings
Amazon Elastic Container Registry (ECR) -aws-ecr-image
- aws-ecr-get-repository-policy AWS Elastic File System (EFS) aws-describe-mount-targets
Amazon Elastic Container Service for Kubernetes
(EKS)
aws-eks-describe-cluster SERVICE API NAME IN PRISMA CLOUD
ElastiCache -aws-cache-engine-versions
- aws-elasticache-cache-clusters
- aws-elasticache-describe-replication-groups
- aws-elasticache-reserved-cache-nodes
- aws-elasticache-subnet-groups
- aws-elasticache-snapshots
Amazon Elastic Load Balancing -aws-elb-describe-load-balancers
- aws-describe-ssl-policies
- aws-elbv2-describe-load-balancers
- aws-elbv2-target-group
Amazon ElasticSearch Service aws-es-describe-elasticsearch-domain
Amazon Elastic MapReduce (EMR) -aws-emr-describe-cluster
- aws-emr-public-access-block
Amazon S3 Glacier -aws-glacier-get-vault-access-policy -aws-glacier-get-vault-lock
Amazon GuardDuty aws-guardduty-detector
AWS Glue aws-glue-security-configuration
AWS Identity and Access Management (IAM) -aws-iam-list-access-keys
- aws-iam-get-account-summary
- aws-iam-list-server-certificates
- aws-iam-get-credential-report
- aws-iam-list-mfa-devices
- aws-iam-list-virtual-mfa-devices
- aws-iam-get-account-password-policy -aws-iam-get-policy-version
- aws-iam-list-users -aws-iam-list-user-policies
- aws-iam-list-roles
- aws-iam-list-groups
- aws-iam-list-attached-user-policies
- aws-iam-list-ssh-public-keys
- aws-iam-saml-provider
- aws-iam-service-last-accessed-details
AWS Key Management Service (KMS) aws-kms-get-key-rotation-status
Amazon Kinesis aws-kinesis-list-streams
aws-kinesis-firehose-delivery-stream
AWS Lambda -aws-lambda-list-functions
- aws-lambda-get-region-summary SERVICE API NAME IN PRISMA CLOUD
AWS Organization -aws-organization-account
- aws-organization-ou
- aws-organization-root
- aws-organization-scp
- aws-organization-tag-policy AWS Resource Access Manager (RAM) -aws-ram-principal
- aws ram list-resources
- aws-ram-resource
- aws-ram-resource-share
Amazon Relational Database Service (RDS) -aws-rds-describe-db-instances
- aws-rds-describe-db-snapshots
- aws-rds-describe-event-subscriptions
- aws-rds-db-cluster-snapshots
- aws-rds-db-clusters
Amazon RedShift aws-redshift-describe-clusters
AWS Route53 -aws-route53-list-hosted-zones
- aws-route53-domain
Amazon RDS aws-rds-describe-db-parameter-groups
AWS Secrets Manager aws-secretsmanager-describe-secret
AWS Systems Manager aws-ssm-parameter
Amazon S3 -aws-s3control-public-access-block
- aws-s3api-get-bucket-acl
The list of APIs associated with this API name
are:
- listBuckets
- getS3AccountOwner
- getRegionName
- getBucketLocation
- getBucketAcl
- getBucketPolicy -getBucketPolicyStatus
- getBucketVersioningConfiguration
Amazon Simple Notification Service (SNS) -aws-sns-get-subscription-attributes
- aws-sns-get-topic-attributes
- aws-sns-platform-application
Amazon Simple Queue Service (SQS) aws-sqs-get-queue-attributes
Amazon VPC -aws-ec2-describe-security-groups SERVICE API NAME IN PRISMA CLOUD
- aws-ec2-describe-route-tables
- aws-ec2-describe-subnets
- aws-ec2-describe-vpcs
- aws-ec2-describe-vpc-peering-connections
- aws-describe-vpc-endpoints
- aws-ec2-vpn-connections-summary
- aws-ec2-describe-vpn-connections
- aws-ec2-describe-vpn-gateways
- aws-ec2-describe-vpn-gateways-summary
- aws-vpc-dhcp-options
- aws-vpc-nat-gateway
- aws-ec2-describe-flow-logs
- aws-ec2-describe-internet-gateways
- aws-ec2-describe-network-acls
- aws-ecr-get-repository-policy -aws-vpc-transit-gateway
AWS Web Application Firewall (WAF) -aws-waf-web-acl-resources
- aws-waf-classic-web-acl-resource
- aws-waf-classic-global-web-acl-resource
Amazon WorkSpaces -aws-describe-workspace-directories
- aws-workspaces-describe-workspaces Onboard Azure Account
To begin monitoring and identifying compliance violations and vulnerabilities on Azure commercial
or Government environment, must add Azure subscriptions to Prisma™ Cloud. To successfully
add Azure subscriptions, must enable authentication between Prisma Cloud and Azure
resources and configure the permissions required to read configuration data, flow logs, and audit logs, and
to remediate issues that are identified on Azure resources.
And, if want to retrieve information on users who access resources deployed within Azure
subscription, add Azure Active Directory account to Prisma Cloud.
- Azure Cloud Account Onboarding Checklist.
- Add an Azure Subscription on Prisma Cloud.
- Add an Azure Active Directory Account on Prisma Cloud.
- Set Up Azure Cloud Account Manually or Use the Azure PowerShell Script.
- Troubleshoot Azure Account Onboarding.
Azure Cloud Account Onboarding Checklist
Prisma™ Cloud supports both Azure commercial and Azure Government. For Azure commercial, the onboarding flow enables to provide subscription details as inputs and generates a Terraform
template, which download and run from the Azure Cloud Shell. The workflow automates the process of setting up the Prisma Cloud application on Azure Active Directory and enabling the permissions
for read-only or read-write access to Azure subscription. will however, need to review and
manually enable the permissions for Prisma Cloud to retrieve network traffic data from network security
group (NSG) flow logs.
To successfully onboard and monitor the resources within Azure Government subscription, use the following checklist to authorize the correct set of access rights to Prisma Cloud.
Collect Azure Subscription ID from the Azure portal.
Get Azure Subscription ID. Make sure that have Account Owner or Contributor privileges so
that add Prisma Cloud as an application on Azure Active Directory. To onboard Azure
subscription on Prisma Cloud, set up an Active Directory application object (Application Client ID) and
an Enterprise Application Object ID that together enable API access. The process of setting up Prisma Cloud on Azure Active Directory will provide with the keys and IDs that are required to establish an
identity for sign-in and access to resources in Azure subscription. The Enterprise Application Object
ID defines the permissions and scope that is assumed by Prisma Cloud.
Review the roles and associated permissions required:
Role Description
Reader The Reader role at the subscription level is
required for Prisma Cloud to monitor the configuration of existing Azure resources within
Azure subscription. Prisma Cloud requires
this role to ingest configuration and activity logs.
Reader and Data Access The Reader and Data Access role at the subscription level is required for Prisma Cloud to
fetch flow logs and storage account attributes so
that use Prisma Cloud policies that assess
risks in storage account. This role includes Role Description
the permissions to access the storage account
keys and authenticate to the storage account to
access the data.
- For Prisma Cloud to access
flow logs stored in storage
accounts that belong to
subscriptions that are not
monitored by Prisma Cloud, must provide Reader
and Data Access role on the storage accounts.
- The Reader and Data Access
role is not a superset of the Reader role. Although this
role has read-write access, Prisma Cloud only uses these
permissions to access and
read the flow log from the storage account.
Network Contributor or a custom role to query
flow log status
The built-in Network Contributor role can manage
network data necessary to access and read flow
logs settings for all network security groups
(NSGs) along with the details on the storage
account to which the flow logs are written. It also
enables auto-remediation of network-related
incidents.
use the built-in role or create a custom
role to allow Prisma Cloud to fetch flow log status.
As a best practice, Create a Custom Role on Azure
to Enable Prisma Cloud to Access Flow Logs and
use the least privilege principal to enable access
only to the required permissions. The network
contributor built-in role provides a much broader
set of permissions than required by Prisma Cloud.
To create a custom
role, must have the Microsoft.Authorization/
roleDefinitions/
write permission on all
AssignableScopes, such
as Owner or User Access
Administrator.
then use the Azure
CLI to create a custom role with
the Microsoft.Network/
networkWatchers/
queryFlowLogStatus/action Role Description
permission to query the status of flow logs.
Storage Account Contributor (Optional but required if want to enable auto
- remediation) The Storage Account Contributor
role is required on all storage accounts to allow
auto-remediation of policy violations.
Custom role with permissions (Optional but required if want to enable
ingestion of the listed services) Create a custom
role with the following permissions:
- Microsoft.ContainerRegistry/
registries/webhooks/
getCallbackConfig/action—To ingest
data from Azure Container Registry webhooks
that are triggered when a container image
or Helm chart is pushed to or deleted from a registry.
- Microsoft.Web/sites/config/
list/action—To ingest Authentication/
Authorization data from Azure App Service
that hosts websites and web applications. The Reader Role listed earlier is adequate to ingest
configuration data from the Azure App Service.
Prisma Cloud provides a JSON file that makes it easier for to create a custom role
with the read-only permissions required to monitor Azure resources.
Enable Prisma Cloud to obtain network traffic data from network security group (NSG) flow logs: NSG
flow logs are a feature of Network Watcher, which allows to view information about ingress and
egress IP traffic through an NSG.
- Create one or more network security groups if have none.
- Create Azure Network Watcher instances for the virtual networks in every region where collect
NSG flow logs.
Network Watcher enables to monitor, diagnose, and view metrics to enable and disable logs for resources in an Azure virtual network.
- Create storage accounts. must have a storage account in each region where have NSGs
because flow logs are written to the same region as the NSGs. As a best practice, configure a single
storage account to collect flow logs from all NSGs in a region.
- Enable Network Watcher and register Microsoft.InsightsResource Provider.
Microsoft.Insights is the resource provider namespace for Azure Monitor, which provides features
such as metrics, diagnostic logs, and activity logs.
- Enable NSG flow logs version 1 or 2, based on the regions where NSG flow logs version 2 is
supported on Azure.
- Verify that view the flow logs.
Continue to Add an Azure Subscription on Prisma Cloud. Add an Azure Subscription on Prisma Cloud
Connecting Prisma™ Cloud to Azure cloud account enables to analyze and monitor traffic logs, and detect potential malicious network activity or compliance issues. To enable API access between
Prisma Cloud and Microsoft Azure Subscription, need to gather account information about your
subscription and Azure Active Directory so that Prisma Cloud can monitor the resources in cloud
account, and add one subscription at a time.
If are adding an Azure commercial account, this workflow uses Terraform templates to streamline
the set up process. The template automates the process of creating and registering Prisma Cloud as an
application on Active Directory and creating the Service Principal and associating the roles required to
enable authentication.
If are adding an Azure Government or Azure China subscription, must complete some tasks
manually on the Azure portal.
- Add Azure Commercial Subscription to Prisma Cloud
- Add Azure Government Subscription to Prisma Cloud
- Add an Azure China Subscription on Prisma Cloud
- Create a Custom Role for Prisma Cloud (Not required if use the Terraform template)
Add Azure Commercial Subscription to Prisma Cloud
- STEP 1 | Access Prisma Cloud and select Settings > Cloud Accounts > Add New.
- STEP 2 | Select Azure as the Cloud to Protect.
- STEP 3 | Enter a Cloud Account Name.
A cloud account name is auto-populated for you. replace it with it a cloud account name that
uniquely identifies Azure subscription on Prisma Cloud. - STEP 4 | Select the Mode.
Decide whether to enable permissions to only monitor (read-only access) or to monitor and protect
(read-write access) the resources in cloud account. selection determines which Terraform
template is used to automate the process of creating the custom role required for Prisma Cloud.
- STEP 5 | Register Prisma Cloud as an application on Azure Active Directory.
Prisma Cloud requires Azure Subscription ID so that it can identify Azure cloud account and
retrieve the storage account and key vault information. Prisma Cloud also needs the Directory (Tenant)
ID, Application (Client) ID, Application Client Secret, and Enterprise Application Object ID to establish
the connection between Prisma Cloud and Azure Active Directory so that it can access the resources in
subscription.
1. Fill out the details to set up Prisma Cloud on Azure subscription and click Next.
From the Azure portal, get Azure Active Directory ID, that is referred to as Tenant ID on Prisma Cloud, and Azure Subscription ID.
The Terraform template uses the value enter as inputs to automate the process of setting up
the custom role with the associated permissions for the Monitor or Monitor & Protect mode selected earlier. It also automatically generates a Service principal password. 2. Download the Terraform template.
Prisma Cloud recommends that create a directory to store the Terraform template download. This allows to manage the templates when add a different Azure subscription to
Prisma Cloud. Give the directory a name that uniquely identifies the subscription for which you're
using it (for example, onboard-<subscription-name>).
3. Login to the Azure portal Cloud Shell (Bash).
4. Upload the template to the Cloud Shell.
5. Run the following Terraform commands.
1. terraform init
2. terraform apply
6. Copy the details after applying the Terraform template.
Get the Application Client Secret, Application (Client) ID and the Enterprise Application Object ID
from the output file. - STEP 6 | Select Ingest & Monitor Network Security Group flow logs and click Next.
Network security group (NSG) flow logs are a feature of Network Watcher that allows to view
information about ingress and egress IP traffic through an NSG. must first configure Network
Security Groups on Azure and assign a storage account to enable Flow log ingestion on Prisma Cloud.
Make sure that Azure Flow logs are stored within a storage account in the same region as the NSG. If
want to enable flow log ingestion, must complete the tasks outlined in Steps 6 to Step 10 in
Set Up Azure Subscription for Prisma Cloud. If enable this option without setting it up on the Azure portal, Prisma Cloud will not be able to retrieve any Flow logs.
- STEP 7 | Select the Account Groups want to add and click Next.
must assign each cloud account to an account group, and Create an Alert Rule to associate the account group with it to generate alerts when a policy violation occurs.
- STEP 8 | Verify the Status and clickDone.
If Prisma Cloud is able to successfully connect to Azure subscription and retrieve
information, the status is displayed with a green check mark. If Prisma Cloud is unable
to retrieve the logs, the error message indicates what failed. See Troubleshoot Azure
Account Onboarding for help. - STEP 9 | Verify that view the information on Azure resources on Prisma Cloud.
Wait for approximately 10-24 hours after onboard the Azure subscription to Prisma Cloud, to
review the data that was collected about Azure resources. After Prisma Cloud ingests data, the information is available for compliance checks, configuration review, audit history, and network
visualization.
It takes about four to six hours before view flow logs in Prisma Cloud. Prisma Cloud ingests flow logs from the previous seven days from when onboard the account.
1. Log in to Prisma Cloud.
2. Select Investigate and enter the following RQL query.
This query allows to list all network traffic from the Internet or from Suspicious IP addresses with
over 0 bytes of data transferred to a network interface on any resource on any cloud environment.
 network where cloud.account = ‘{{cloud account name}}’ AND
 source.publicnetwork IN (‘Internet IPs’, ‘Suspicious IPs’) AND bytes >
 0
  Add Azure Government Subscription to Prisma Cloud
Connect Azure Government subscription on Prisma Cloud to monitor resources for potential security
and compliance issues.
- STEP 1 | Set Up Azure Subscription for Prisma Cloud.
- STEP 2 | Add Azure subscription on Prisma Cloud.
1. Access Prisma Cloud and select Settings > Cloud Accounts > Add New.
2. Enter a Cloud Account Name.
3. Select Cloud Type Azure and the Government environment where resources are deployed, click
Next.
4. Enter Azure Subscription ID, Directory (Tenant) ID, Application (Client) ID, Application Client
Secret and Enterprise Application Object ID.
5. Select Ingest & Monitor Network Security Group flow logs and click Next.
Network security group (NSG) flow logs are a feature of Network Watcher that allows to view
information about ingress and egress IP traffic through an NSG. Make sure that Azure Flow logs are
stored within a storage account in the same region as the NSG. See Azure Cloud Account Onboarding
Checklist for the set up details to ensure that Prisma Cloud can successfully ingest NSG flow logs.
6. Select the Account Groups want to add and click Next.
must assign each cloud account to an account group, and Create an Alert Rule to associate the account group with it to generate alerts when a policy violation occurs.
7. Verify the Status and Done to save changes.
If Prisma Cloud was able to successfully make an API request to retrieve the Azure
flow logs, the status is displayed with a green check mark. If Prisma Cloud is unable
to retrieve the logs, the error message indicates what failed. See Troubleshoot Azure
Account Onboarding for help. - STEP 3 | Verify that view the information on Azure resources on Prisma Cloud.
Wait for approximately 10-24 hours after onboard the Azure subscription to Prisma Cloud, to
review the data that was collected about Azure resources. After Prisma Cloud ingests data, the information is available for compliance checks, configuration review, audit history, and network
visualization.
It takes about four to six hours before view flow logs in Prisma Cloud. Prisma Cloud ingests flow logs from the previous seven days from when onboard the account.
1. Log in to Prisma Cloud.
2. Select Investigate and enter the following RQL query.
This query allows to list all network traffic from the Internet or from Suspicious IP addresses with
over 0 bytes of data transferred to a network interface on any resource on any cloud environment.
 network where cloud.account = ‘{{cloud account name}}’ AND
 source.publicnetwork IN (‘Internet IPs’, ‘Suspicious IPs’) AND bytes >
 0

Add an Azure China Subscription on Prisma Cloud
With the Prisma Cloud Business Edition license on app.prismacloud.cn, monitor Microsoft
Azure China subscriptions. To get started, gather the details listed in Set Up Azure Subscription for Prisma Cloud from the Azure China portal and connect subscription to Prisma Cloud. When add the subscription, Prisma Cloud monitors the configuration metadata for IaaS and PaaS services
and identifies potential resource misconfiguration and improper exposure. It also enables to use data ingested from event logs and network flow logs for better visibility and governance. - STEP 1 | Add Azure subscription on Prisma Cloud.
1. Log in to Prisma Cloud.
2. Select Settings > Cloud Accounts > Add New
3. Select Cloud Type Azure and click Next.
4. Enter a Cloud Account Name.
5. Enter Azure Subscription ID, Directory (Tenant) ID, Application (Client) ID, Application Client
Secret and Enterprise Application Object ID.
These are the details collected from the Azure portal. 6. Select Ingest & Monitor Network Security Group flow logs and click Next.
Network security group (NSG) flow logs are a feature of Network Watcher that allows to view
information about ingress and egress IP traffic through an NSG. Make sure that Azure Flow logs are
stored within a storage account in the same region as the NSG. See Azure Cloud Account Onboarding
Checklist for the set up details to ensure that Prisma Cloud can successfully ingest NSG flow logs.
7. Select the Account Groups want to add and click Next.
must assign each cloud account to an account group, and Create an Alert Rule to associate the account group with it to generate alerts when a policy violation occurs.
8. Verify the Status and Save changes.
If Prisma Cloud was able to successfully make an API request to retrieve the configuration metadata, the status is displayed with a green check mark. If Prisma Cloud is unable to retrieve the logs, the error message indicates what failed. Review the details for the account added on Settings > Cloud Accounts. The cloud account
owner name is displayed for you.
- STEP 2 | Verify that view the information on Azure resources on Prisma Cloud.
Wait for approximately 10-24 hours after onboard the Azure subscription to Prisma Cloud, to
review the data that was collected about Azure resources. After Prisma Cloud ingests data, the information is available for asset inventory, compliance checks and configuration review.
1. Log in to Prisma Cloud.
2. Select Inventory > Assets.
View a snapshot of the current state of all cloud resources or assets that are monitoring and
securing using Prisma Cloud. Create a Custom Role for Prisma Cloud
If want to manually create the role and review all the permissions required for monitoring Azure
subscription, instead of using the Terraform template, use the custom role JSON file from Prisma Cloud. To create a custom role on Azure, must have an Azure Active Directory Premium 1 or Premium
2 license plan.
- STEP 1 | Download and save the JSON file from here.
Save the JSON file on local machine or laptop.
If are using this file for onboarding Azure China subscription, remove the permission for Microsoft.Databricks/workspaces/read
because Databricks is not available on Azure China.
- STEP 2 | Install the Azure CLI and log in to Azure.
- STEP 3 | Go to the directory where stored the JSON file.
- STEP 4 | Enter the following Azure CLI command.
If renamed the file, will need to replace the JSON filename to match that in the following
command.
 az role definition create --role-definition
 "azure_prisma_cloud_lp_read_only.json"
For services that are not available in the Azure environment where are creating the role, the following error message displays New-AzRoleDefinition : The resource provider referenced in the action is not returned in the list of providers from Azure Resource Manager. must edit the JSON
file to remove the permissions for services that are not available.
Update an Onboarded Azure Account
For an Azure subscription that have already added to Prisma Cloud, enable flow log ingestion
after manually set it up starting at 6.
- STEP 1 | Log in to Prisma Cloud administrative console.
- STEP 2 |
- Enable Ingest and Monitor Network Security Group Flow Logs.
- Change the account groups want to monitor.
Add an Azure Active Directory Account on Prisma Cloud
Connecting Prisma™ Cloud to monitor Azure Active Directory enables to retrieve information on
users who access resources deployed within Azure subscription.
This feature is available as a Limited GA, and to try it please contact Palo Alto Networks Customer Success.
- STEP 1 | Authorize an Azure Active Directory application to read user profile information.
Watch it.
1. Log in to the Azure portal.
2. Select Azure Active Directory > App Registrations.
3. Select the application from the list and select API Permissions > Add a Permission. 4. Select Microsoft Graph > Application Permissions
This allows to allow the application to access the Microsoft Graph APIs.
5. Select the permission user.read.all, and Add Permission.
The permission added allows the app to read user profiles without a signed in user.
This permission requires admin consent. An Azure AD tenant administrator must grant
these permissions by making a call to the admin consent endpoint.
- STEP 2 | Collect the details for the Active Directory app.
1. Log in to the Azure portal.
2. Get the Directory (tenant) ID.
3. Get the Application (client) ID and Object ID.
must enter the Object ID as the Enterprise Application Object ID in Step 3-d.
4. Get the Application Key. - STEP 3 | Add Azure Active Directory on Prisma Cloud.
1. Access Prisma Cloud and select Settings > Cloud Accounts > Add New.
2. Enter a Cloud Account Name.
3. Select Onboard using Azure Active Directory
4. Select Cloud Type Azure and the Commercial or Government environment where AD resources
are deployed, click Next.
5. Enter Azure Directory (Tenant) ID, Application (Client) ID, Application Client Secret and
Enterprise Application Object ID.
6. Select the Account Groups want to add and click Next.
must assign each cloud account to an account group, and Create an Alert Rule to associate the account group with it to generate alerts when a policy violation occurs.
7. Verify the Status and Done to save changes.
If Prisma Cloud is able to validate the credentials by making an authentication call
using the credentials provided in the previous step, it displays a green check mark. - STEP 4 | Verify that view the information on Azure Active Directory users on Prisma Cloud.
1. Log in to Prisma Cloud.
2. Select Investigate and enter the following RQL query to view details on Azure Active Directory
users.
 config where cloud.type = 'azure' AND api.name = 'azure-activedirectory-user' AND json.rule = userType equals "Guest"

Set Up Azure Subscription for Prisma Cloud
Connect Prisma™ Cloud to Azure cloud environment so that monitor for threats and
compliance violations, enable auto-remediation of incidents, and identify hosts and containers that contain
vulnerabilities. Before Prisma Cloud can monitor the resources within Azure Government subscription, must add Prisma Cloud as an application to Azure Active Directory and configure Azure
subscription to allow Prisma Cloud to analyze flow log data.
do not need to complete this workflow for Azure commercial because the onboarding
flow uses Terraform template to automate this process. Start with Add an Azure Subscription
on Prisma Cloud
Prisma Cloud requires Azure Subscription ID so that it can identify Azure cloud account
and retrieve the storage account and key vault information. Prisma Cloud also needs the Directory ID, Application ID, Application Key, and Service Principal ID to establish the connection between Prisma Cloud
and Azure Active Directory so that it can access the resources in subscription.
- STEP 1 | Locate and copy Azure subscription ID.
Prisma Cloud requires the Subscription ID so that it can identify Azure cloud account and can
retrieve the storage account and key vault information. 1. Log in to Microsoft Azure, select All services > Subscriptions, select subscription, and copy the Subscription ID.
- STEP 2 | Add Prisma Cloud as a new application on Azure Active Directory.
Registering Prisma Cloud as an application on Azure AD generates an Application ID. need this ID
and an Application key to authenticate Prisma Cloud on Azure and to maintain a secure connection.
1. Log in to Microsoft Azure and select Azure Active Directory > App registrations > New application
registration.
2. Enter a Name to identify the Prisma Cloud application, select the Supported account types that can
use the application as Accounts in this organizational directory only, enter login URL for Prisma Cloud as the Redirect URI, and then click Register.
The log in URL for Prisma Cloud is the URL received in order confirmation email, and it
varies depending on region. 3. Generate a client secret for the Prisma Cloud application.
The client secret is the application password for Prisma Cloud.
1. Select Azure Active Directory > App Registrations > All Applications and select the Prisma Cloud
application.
2. Add a client secret or the application password (Certificates & Secrets > New client secret).
3. Enter a Description and select a Duration, which is the term for which the key is valid.
4. Add the new client secret and then copy the value of that new client secret for records
because cannot view this key after close this dialog. will need this new client secret
application key when Add an Azure Subscription on Prisma Cloud.
- STEP 3 | Copy the information on the Prisma Cloud application from Azure Active Directory.
For Prisma Cloud to interact with the Azure APIs and collect information on Azure resources, must capture the following values.
1. Select Azure Active Directory > App Registrations > All Applications, find the Prisma Cloud
application created, and copy the Directory ID. 2. Select Azure Active Directory > Enterprise Applications.
3. Select Prisma Cloud application Properties and copy the Application ID and Object ID.
must enter the Object ID as the Service Principal ID in the next step.
Make sure that get the Object ID for the Prisma Cloud application from Enterprise
Applications > All applications on the Azure portal—not from App Registrations.
- STEP 4 | Grant permissions for the Prisma Cloud application to access information at the Azure
Subscription level.
To assign roles, must have Owner or User Access Administrator privileges on Azure
Subscription.
1. Select All Services > Subscriptions.
2. Select subscription and Add role assignment (Access Control (IAM)). 3. Select the Role, verify that Azure AD user, group, or service principal is selected (Assign access to), and select the Prisma Cluod app to assign the roles.
Review the Azure Cloud Account Onboarding Checklist for a description of the roles and permissions
that are required at the subscription level. Then decide which roles must add for security
and monitoring needs—Reader Role, Reader and Data Access Role, Create a Custom Role on
Azure to Enable Prisma Cloud to Access Flow Logs, Network Contributor Role, or Storage Account
Contributor Role.
- STEP 5 | (Optional) Grant permission for the Prisma Cloud application to access the Azure Key Vault
service.
If use Azure Key Vault to safeguard and manage cryptographic keys and secrets used by cloud
applications and services, Prisma Cloud needs permission to ingest this key vault data.
1. From Azure, select All Services > Key Vaults.
2. Select Key vault name and Access Policies. 3. Select (add) Prisma Cloud application (Add new > Select Principal).
4. Select List for Key permissions and for Secret permissions, select both List and List Certificate
Authorities for Certification permissions, and then click OK.
- STEP 6 | On the Azure portal, Enable Network Watcher and register Insights provider.
- STEP 7 | On the Azure portal,Create a storage account on Azure for NSG flow logs. Azure storage account stores the flow logs that are required for Prisma Cloud to monitor and
analyze network traffic. When Prisma Cloud ingests the data in these logs, interact with the information in Prisma Cloud. For example, run queries against the data, visualize network
topology, and investigate traffic flows between two instances. also apply network policies to
this traffic.
If do not have regulatory guidelines that specify a log retention period to which must adhere, we recommend set retention to at least 15 days.
- STEP 8 | On the Azure portal,Enable NSG flow logs.
- STEP 9 | Configure Prisma Cloud Reader and Data Access role for Azure storage account.
To ingest Azure flow logs, have to grant access to the storage account in which the logs are stored.
The Reader and Data Access role provides the ability to view everything and allows read/write access to
all data contained in a storage account using the associated storage account keys. If flow logs are
stored in storage accounts that belong to one or more subscriptions that are not monitored by Prisma Cloud, must configure the Prisma Cloud application with the Reader and Data Access role for each
storage account.
1. After creating storage account, select Access control (IAM) > Add role assignment.
2. Select Reader and Data Access as the Role, Select the administrative user to whom want to
assign the role, and Save changes.
- STEP 10 | Add an Azure Subscription on Prisma Cloud. Create a Custom Role on Azure to Enable Prisma Cloud to Access
Flow Logs
To enable Prisma™ Cloud to access Azure flow logs and monitor flow-related data (such as volume of traffic generated by a host, top sources of traffic to the host, or to identify which ports are in use), must
provide the required permissions. While use the built-in Network Contributor role that enables a much broader set of permissions, it is a best practice to create a custom role so that follow the principle
of least privilege and limit access rights to the bare-minimum. Use the Azure Cloud Account Onboarding
Checklist to verify on which services want to ingest data and manually assign the permissions for this
custom role that includes the permissions required. To create a custom role, install Azure CLI and create a limited role named Prisma Cloud - Flow Logs Setting Reader and then enable the role to access flow logs.
To create a custom role on Azure, must have an Azure Active Directory Premium 1 or
Premium 2 license plan.
- STEP 1 | Create a custom role using Azure CLI.
If already assigned a Network Contributor Role to an Azure user, skip this step.
- Manually create a custom role JSON file for flow logs only.
The permissions required are { "Name": "Prisma Cloud - Flow Logs Setting
Reader", "Id": null, "IsCustom": true, "Description": "Allows Reading
Flow Logs Settings", "Actions": [ "Microsoft.Network/networkWatchers/
queryFlowLogStatus/action" ], "NotActions": [], "AssignableScopes": [ "/
subscriptions/SUBSCRIPTION-ID-HERE!!!" ]}
1. Install the Azure CLI and log in to Azure.
2. Open a text editor (such as notepad) and enter the following command in the JSON format to
create a custom role. create custom roles using Azure PowerShell, Azure CLI, or the REST
API. These instructions use the Azure CLI command (run on PowerShell or on the DOS command
prompt) to create the custom role with queryFlowLogStatus permission. Make sure to provide
Azure Subscription ID in the last line.
3. Save the JSON file on local Windows system and give it a descriptive name, such as adrole-cli.json.
4. Log in to the Azure portal from the same Windows system and complete the following steps:
1. Open a PowerShell window (or a DOS Command Prompt Window)
2. Go to the directory where stored the JSON file.
3. Enter the following Azure CLI command (replace the JSON filename to match the name specified when saved custom role JSON file.
 az role definition create --role-definition "ad-rolecli.json"

The output is as follows:
 {"assignableScopes": [ "/subscriptions/
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" ], "description": "Allows
 Reading Flow Logs Settings", "id": "/subscriptions/16dfdbcce407-4fbe-9096-e7a97ee23fb5/providers/Microsoft.Authorization/
roleDefinitions/088c8f48-201c-4f8d-893f-7716a8d58fa1", "name":
 "088c8f48-201c-4f8d-893f-7716a8d58fa1", "permissions": [{  "actions": [ "Microsoft.Network/networkWatchers/
queryFlowLogStatus/action"], "dataActions": [],  "notActions": [], "notDataActions": []
 }], "roleName": "RedLock Flow Log Settings Reader",  "roleType": "CustomRole", "type": "Microsoft.Authorization/
roleDefinitions"}

- STEP 2 | Configure the custom role to access the flow logs.
1. Log in to the Microsoft Azure Portal.
2. Select subscription ( All services > Subscriptions
3. Select Access control (IAM) > Add role assignment.
erify that see the new custom role created in the Role drop-down.
4. Assign the Role to Prisma Cloud, enable the permission to query flow log status, and Save your
changes.
Use the Azure PowerShell Script to Add an Azure Account
To save time and reduce the likelihood of human error, use the Azure PowerShell Az module
to automate the process of setting up the Prisma Cloud application on Microsoft Azure cloud
environment. Because this script adds the Prisma Cloud application on Azure Active Directory, must be an administrator with authorization to create roles, and add an application to manage Azure
subscription.
- STEP 1 | Download the RedLock Azure Onboarding script.
- STEP 2 | Launch Azure Cloud Shell.
Verify that are in PowerShell.
- STEP 3 | Upload the RedLock Azure Onboarding script.
- STEP 4 | Verify the script is in the local directory and then enter ./RedLock-Azure-Onboarding-BetaV5.ps1 By default, the file is uploaded to home directory and might need to use cd ~ and then enter ./
RedLock-Azure-Onboarding-Beta-V5.ps1
- STEP 5 | When prompted, enter Azure SubscriptionID, a name to identify the RedLock application
for example, APPNAME-RedLock-V2, and the URL that use to access Prisma Cloud, for example https://app.redlock.io.
- STEP 6 | When prompted, open the link on browser.
- STEP 7 | Log in to Windows Azure PowerShell.
Log in to Azure using an account with Owner or Admin access.
- STEP 8 | Enter the authentication code and click Continue to log in to Windows Azure PowerShell.
- STEP 9 | Verify the details on successful completion of the process. The Azure PowerShell window displays the results of the script. It is also saved to a file named
Subscription-ApplicationName.txt
1. The Application Client ID.
2. The Reader role.
- STEP 10 | Generate the Application client secret for the Prisma Cloud application.
The script does not create the application client secret. To generate an application client secret (Step
2-3), must log in to the Azure portal.
- STEP 11 | Copy the Prisma Cloud application details from Azure Active Directory.
Prisma Cloud requires the Directory (tenant) ID, Application Client ID, Application Client Secret, and
Enterprise Application Object ID to establish the connection between Prisma Cloud and Azure Active
Directory and access the resources in subscription.
- STEP 12 | Continue to Add an Azure Subscription on Prisma Cloud.
Troubleshoot Azure Account Onboarding
After have completed onboarding Azure subscription to Prisma Cloud, use these checks to resolve
issues if Prisma Cloud cannot retrieve logs and metadata including resource configurations, user activities, network traffic, host vulnerabilities/activities on Azure resources. Without enabling the correct
permissions and configuration on the Azure portal and on Prisma Cloud, you’ll be unable to identify, detect, and remediate issues to manage the risks in environment.
- Verify that the cloud account permissions are accurate on Settings > Cloud Accounts >
Cloud_Account_Name > Status.
At every ingestion cycle when Prisma Cloud connects to the Azure subscription to retrieve and
process the data, the service validates that continue to have the permissions required to continue
monitoring the resources within Azure subscription. Periodically, review the status of these
checks to learn of any changes or modifications that limit ability to maintain visibility and security
governance over the resources within Azure subscription. -Reader role, and Reader and Data Access role at the subscription level.
If see the error Subscription does not have Reader role(s) assigned, verify that
have entered the correct Service Principal ID. On the Azure portal, the Object ID
for the Prisma Cloud application is what must provide as the Service Principal ID
on Prisma Cloud. Make sure to get the Object ID for the Prisma Cloud application from
Enterprise Applications > All applications on the Azure portal.
- Checks for the Network Contributor role or Custom role that is required to query flow log status.
- Checks for the Reader and Data Access role on the storage accounts to access the flow logs stored in
storage buckets in subscriptions that are not monitored by Prisma Cloud.
- Checks for the Storage Account Contributor role (optional and required only for remediation) that is
required for auto remediation of policy violations related to storage accounts.
- Checks whether flow logs are published to the storage account.
- Check that Azure flow logs are being generated and written to the storage account.
1. Log in to the Azure portal.
2. Select Storage Accounts and select the storage account that want to check.
3. Select Blobs > Blob Service and navigate through the folders to find the *.json files.
These are the flow logs that Prisma Cloud ingests.
- On the Azure portal, check that have created storage accounts in the same regions as the Network Security Groups.
Network security group (NSG) flow logs are a feature of Network Watcher that allows to view
information about ingress and egress IP traffic through an NSG. Azure flow logs must be stored within a storage account in the same region as the NSG.
1. Log in to Prisma Cloud. 2. Select Investigate and enter the following RQL query network where source.publicnetwork
IN (‘Internet IPs’, ‘Suspicious IPs’) AND bytes > 0
This query allows to list all network traffic from the Internet or from Suspicious IP addresses with
over 0 bytes of data transferred to a network interface on any resource on any cloud environment.
- On the Azure portal, verify that have enabled Network Watcher instance.
The Network Watcher is required to generate flow logs on Azure.
1. Log in to the Azure portal and select Network Watcher > Overview and verify that the status is
Enabled.
2. Log in to Prisma Cloud.
3. Select Investigate and enter the following RQL query config where cloud type =’azure’
AND api.name =’azure-network-nsg-list’ addcolumn provisioningState.
- On the Azure portal, check that have enabled flow logs on the NSGs.
1. Log in to the Azure portal, and select Network Watcher > NSG Flow Logs and verify that the status is
Enabled.
2. Log in to Prisma Cloud.
3. Select Investigate and enter the following RQL query network where source.publicnetwork
IN (‘Internet IPs’, ‘Suspicious IPs’) AND bytes > 0 This query allows to list all network traffic from the Internet or from Suspicious IP addresses with
over 0 bytes of data transferred to a network interface on any resource on any cloud environment.
- The cloud account status displays red and includes the error message "Authentication failed.
Azure Subscription not found.
When the Azure subscription is deleted or disabled on the Azure portal and Prisma Cloud cannot
monitor the subscription, the cloud account status displays red and includes the error message
Authentication failed. Azure Subscription not found.
Microsoft Azure APIs Ingested by Prisma Cloud
The following APIs are ingested by Prisma Cloud:
SERVICE API NAME IN PRISMA CLOUD
Azure Application Gateway azure-application-gateway
Azure Active Directory -azure-active-directory-user
- azure-active-directory-conditional-accesspolicy -azure-active-directory-named-location
Azure API Management azure-api-management-service
Azure Cache azure-redis-cache
Azure Compute -azure-vm-list
- azure-disk-list
Azure Container Registry azure-container-registry
Azure Cosmos DB azure-cosmos-db
Azure Data Factory -azure-data-factory-v1
- azure-data-factory-v2
Azure Data Bricks azure-databricks-workspace
Azure Database for MySQL azure-mysql-server
Azure Event Hubs azure-event-hub-namespace
Azure Key Vault -azure-key-vault-list
- azure-key-vault-certificate
Azure Load Balancer azure-network-lb-list SERVICE API NAME IN PRISMA CLOUD
Azure Logic Apps -azure-logic-app-workflow
- azure-logic-app-custom-connector
Azure Resource Group -azure-resource-group
- azure-role-definition
Azure Resource Manager azure-role-assignment
Azure Policy azure-policy-assignments
azure-policy-definition
Azure Security Center azure-security-center-settings
Azure Virtual Network -azure-network-public-ip-address
- azure-network-usage
- azure-network-application-security-group
- azure-network-vnet-list
- azure-network-subnet-list
- azure-network-route-table
- azure-network-lb-list
- azure-network-peering
- azure-network-nic-list
- azure-network-nsg-list
- azure-network-firewall
Azure SQL Database -azure-sql-db-list
- azure-sql-server-list
- azure-sql-managed-instance
Azure Database for PostgreSQL azure-postgresql-server
Azure Monitor -azure-activity-log-alerts -azure-monitor-log-profiles-list
Azure Network Watcher azure-network-watcher-list
Storage Account azure-storage-account-list
Subnets azure-network-subnet-list
Azure Kubernetes Service azure-kubernetes-cluster
App service azure-app-service
Azure VPN Gateway azure-network-vpn-connection-list Onboard Google Cloud Platform (GCP)
Account
To enable Prisma™ Cloud to retrieve data on Google Cloud Platform (GCP) resources and identify
potential security risks and compliance issues, must connect GCP accounts to Prisma Cloud. In
keeping with the GCP resource hierarchy, choose whether want Prisma Cloud to monitor one
or more GCP Projects or all projects that are under GCP Organization. Regardless of what choose, the process of onboarding automates the process of creating a service account, creating and associating
roles with the service account, and enabling specific APIs.
- Permissions and Roles for GCP Account on Prisma Cloud.
- Add GCP Project to Prisma Cloud.
- Add GCP Organization to Prisma Cloud.
- Create a Service Account With a Custom Role for GCP
- Flow Log Compression on GCP
- Enable Flow Logs for GCP Projects
- Enable Flow Logs for GCP Organization
- GCP APIs Ingested by Prisma Cloud
Permissions and Roles for GCP Account on Prisma Cloud
In order to analyze and monitor Google Cloud Platform (GCP) account, Prisma Cloud requires access
to specific APIs and a service account which is an authorized identity that enables authentication between
Prisma Cloud and GCP. A combination of custom, predefined and primitive roles grant the service account
the permissions it needs to complete specific actions on the resources in GCP project or organization.
Depending on cloud protection needs, the service account requires the following roles for read or
read-write access:
- Viewer—Primitive role on GCP.
- Prisma Cloud Viewer— Custom role. Prisma Cloud needs this custom role to grant cloud storage
bucket permission to read storage bucket metadata and update bucket IAM policies. This role requires
storage.buckets.get to retrieve list of storage buckets, and storage.buckets.getIampolicy to retrieve
the IAM policy for the specified bucket.
- Compute Security Admin—Predefined role on GCP. An optional privilege that is required only if want to enable auto-remediation.
- Organization Role Viewer—Predefined role on GCP. This role is required for onboarding a GCP
Organization.
- Dataflow Admin—Predefined role on GCP. An optional privilege that is required for dataflow log
compression using the Dataflow service. See Flow Log Compression on GCP for details.
- Folder Viewer—Predefined role on GCP. An optional privilege that is required only if are onboarding
Folders in the GCP resource hierarchy.
Prisma Cloud can ingest data from several GCP APIs. In the GCP project where create the service
account, must enable the Stackdriver Logging API (logging.googleapis.com) to monitor audit logs, and any other APIs for which want Prisma Cloud to monitor resources. When use the Terraform
template that Prisma Cloud provides to automate the onboarding of GCP project or organization, the required permissions are automatically enabled for you.
The following table lists the APIs and associated granular permissions if want to create a custom role
to onboard GCP account. When the APIs are enabled and the service account has the correct set of roles and associated permissions, Prisma Cloud can retrieve data about GCP resources and identify
potential security risks and compliance issues across cloud accounts.To create a custom role for the service account, see Create a Service Account With a Custom Role for GCP before continue to Add
GCP Project to Prisma Cloud or Add GCP Organization to Prisma Cloud
To enable the APIs that enable Prisma Cloud to monitor GCP projects, use it
as shown in this example (that uses some of the APIs below): gcloud services
enable serviceusage.googleapis.com appengine.googleapis.com
bigquery.googleapis.com cloudfunctions.googleapis.com
dataflow.googleapis.com dns.googleapis.com dataproc.googleapis.com
cloudresourcemanager.googleapis.com cloudkms.googleapis.com
sqladmin.googleapis.com compute.googleapis.com storagecomponent.googleapis.com recommender.googleapis.com
iam.googleapis.com container.googleapis.com
monitoring.googleapis.com logging.googleapis.com
Verify the APIs that have enabled with gcloud services list.
API Service Name Description Role Name Permissions
App Engine
API appengine.googleapis.com
Allows to access
App Engine, which is
a fully
managed
serverless
platform on
GCP.
App Engine
Viewer appengine.applications.get
Access
Context
Manager API
accesscontextmanager.googleapis.com
Read access
to policies, access levels, and access
zones.
Access
Context
Manager
Reader
accesscontextmanager.accessPolicies.list
 accesscontextmanager.accessLevels.list
accesscontextmanager.servicePerimeters.list
BigQuery
API bigquery.googleapis.com
Allows to create, manage, share, and
query data.
BigQuery
Metadata Viewer
bigquery.datasets.get
bigquery.tables.get
bigquery.tables.list
Cloud
Functions cloudfunctions.googleapis.com
Cloud
Functions
is Google
Cloud’s
Cloud
Functions
Viewer risma Cloud 111
API Service Name Description Role Name Permissions
event-driven
serverless
compute
platform
cloudfunctions.functions.list
Cloud
DataFlow
API
dataflow.googleapis.com
Manages
Google
Cloud
Dataflow
projects.
Dataflow
Admin resourcemanager.projects.get
storage.buckets.get
storage.objects.create
storage.objects.get
storage.objects.list
See Flow Log Compression on GCP
Cloud DNS
API dns.googleapis.com
Cloud DNS
translates
requests
for domain
names into
IP addresses
and manages
and
publishes
DNS zones
and records.
DNS Reader
dns.dnsKeys.list
dns.managedZones.get
dns.managedZones.list
dns.projects.get
Cloud Pub/
Sub pubsub.googleapis.com
Real-time
messaging
service that
allows to send
and receive
messages
between
independent
applications.
Custom Role
pubsub.topics.list
pubsub.topics.getIamPolicy pubsub.subscriptions.list
pubsub.subscriptions.getIamPolicy pubsub.snapshots.list risma Cloud
API Service Name Description Role Name Permissions
Cloud
Resource
Manager API
cloudresourcemanager.googleapis.com
Creates, reads, and
updates
metadata for Google
Cloud
Platform
resource
containers.
Role Viewer
resourcemanager.projects.getIamPolicy Cloud Key
Management
Service
(KMS) API
cloudkms.googleapis.com
Google
Cloud KMS
allows
customers
to manage
encryption
keys and
perform
cryptographic
operations
with those
keys.
Custom Role
cloudkms.cryptoKeys.list
cloudkms.keyRings.list
Cloud
Service
Usage
serviceusage.googleapis.com
API that lists
the available
or enabled
services, or disables
services
that service
consumers
no longer
use on GCP.
Role Viewer
serviceusage.services.list
Google
Cloud
Spanner
spanner.googleapis.com
A globally
distributed
NewSQL
database
service and
storage
solution
designed
to support
global online
transaction
processing
deployments
Cloud
Spanner
Viewer
spanner.instances.list
spanner.instanceConfigs.list API Service Name Description Role Name Permissions
Cloud SQL
Admin API sqladmin.googleapis.com
API for Cloud SQL
database
instance
management.
Custom Role
cloudsql.instances.list
Compute
Engine API compute.googleapis.com
Creates and
runs virtual
machines on
the Google
Cloud
Platform.
Compute
Network
Viewer
compute.backendServices.list
compute.disks.list
compute.firewalls.list
compute.forwardingRules.list
compute.globalForwardingRules.list
compute.images.list
compute.images.getIamPolicy compute.instanceGroups.get
compute.instances.list
compute.instanceGroups.list
compute.networks.list
compute.projects.get
compute.regionBackendServices.list
compute.routers.list
compute.routes.list
compute.sslPolicies.get API Service Name Description Role Name Permissions
compute.sslPolicies.list
compute.subnetworks.list
compute.targetHttpProxies.list
compute.targetHttpsProxies.list
compute.targetPools.list
compute.urlMaps.list
compute.vpnTunnels.list
Google API
Key apikeys.googleapis.com
Google
lets manage your
project's API
keys.
This
service
is
in
Alpha.
serviceusage.apiKeys.list
Cloud
Bigtable API bigtableadmin.googleapis.com
Google
Cloud
Bigtable is
a NoSQL
Big Data database
service.
Custom Role
bigtable.appProfiles.list
bigtable.clusters.list
bigtable.instances.list
bigtable.instances.getIamPolicy bigtable.tables.list loud 115
API Service Name Description Role Name Permissions
Google
Cloud
Storage API
storagecomponent.googleapis.com
Cloud
Storage is
a RESTful
service for storing and
accessing
data on Google’s
infrastructure.
Custom Role
storage.buckets.get
storage.buckets.getIamPolicy storage.buckets.list
Google
Dataproc
Clusters API
dataproc.googleapis.com
Dataproc is
a managed
service for creating
clusters of compute
that can
be used to
run Hadoop
and Spark
applications.
Project
Viewer, or a custom role
with granular
privileges.
dataproc.clusters.listdataproc.clusters.getIamPolicy Google
Recommendations
API
recommender.googleapis.com
GCP IAM
Recommender
Google
Recommender
provides
usage
recommendations
for Google
Cloud
resources.
Recommenders
are specific
to a single
Google
Cloud
product and
resource
type.
IAM
Recommender
Viewer
recommender.iamPolicyRecommendations.list
Google
Cloud Run run.googleapis.com
Cloud Run is
a managed
service for deploy and
manage user
provided
container
images
Role Viewer
run.locations.list
run.services.list loud
API Service Name Description Role Name Permissions
Identity
and Access
Management
(IAM) API
iam.googleapis.com
Manages
identity
and access
control
for GCP
resources, including
the creation
of service
accounts, which use to
authenticate
to Google
and make
API calls.
Role Viewer
iam.roles.get
iam.roles.list
iam.serviceAccountKeys.list
iam.serviceAccounts.list
Kubernetes
Engine API container.googleapis.com
Builds and
manages
containerbased
applications, powered by
the open
source
Kubernetes
technology.
Kubernetes
Engine
Cluster
Viewer
container.clusters.get
container.clusters.list
Services
Usage API serviceusage.googleapis.com
List all
services
available to
the specified
GCP project, and the current state
of those
services with
respect to
the project.
Prisma Cloud
recommends
that
enable
this
API
on
N/A
ServiceUsage.Services.List API Service Name Description Role Name Permissions
all
GCP
projects
that
are
onboarded
to
Prisma Cloud.
Stackdriver
Monitoring
API
monitoring.googleapis.com
Manages
your
Stackdriver
Monitoring
data and
configurations.
Monitoring
Viewer monitoring.alertPolicies.list
monitoring.metricDescriptors.get
redis.instances.list
Stackdriver
Logging API logging.googleapis.com
Writes log
entries and
manages
Logging
configuration.
Logging
Admin logging.logEntries.list
logging.logMetrics.list
logging.sinks.list
GCP Organization - Additional permissions
required to onboard
Organization
Role Viewer
The Organization Role Viewer is
required for onboarding a GCP
Organization. If only provide the individual permissions listed below, the permissions set is not sufficient.
resourcemanager.organizations.get
resourcemanager.projects.list
resourcemanager.organizations.getIamPolicy Add GCP Project to Prisma Cloud
Begin here to add a GCP project to Prisma Cloud. If want to add multiple projects, must either
repeat this process for each project want to onboard, or allow Prisma Cloud to automatically
monitor all GCP projects—current and future—that use the Service Account attached to the project are
adding to Prisma Cloud. Prisma Cloud refers to this service account as a Master Service Account. After start monitoring project using Prisma Cloud, if delete the project on GCP, Prisma Cloud learns about it and automatically deletes the account from the list of monitored
accounts on Settings > Cloud Accounts. To track the automatic deletion of the project, an
audit log is generated with information on the name of the deleted account and the date that
the action was performed.
- STEP 1 | Access Prisma Cloud and select Settings > Cloud Accounts > Add New.
- STEP 2 | Select Google Cloud as the Cloud to Protect.
- STEP 3 | Enter a Cloud Account Name.
A cloud account name is auto-populated for you. replace it with a cloud account name that
uniquely identifies GCP project on Prisma™ Cloud.
- STEP 4 | Select the Mode.
Decide whether to enable permissions to only monitor (read-only access) or to monitor and protect
(read-write access) the resources in cloud account. selection determines which Terraform
template is used to automate the process of creating the service account and attaching the roles
required for Prisma Cloud.
- STEP 5 | Select Project for Onboard Using and enter Project ID and the name of Flow Log
Storage Bucket. The Terraform template does not enable flow logs, and must complete the workflow in Enable Flow
Logs for GCP Projects for Prisma Cloud to retrieve flow logs. Additionally, if want to enable flow
log compression on Prisma cloud and address the lack of native compression support for flow logs sink
setup on GCP, must do it manually too. When enable Use Dataflow to generate compressed
logs, Prisma Cloud sets up the network and compute resources required for flow log compression and
this can take up to five minutes.
When enable flow logs, the service ingests flow log data for the last seven days.
Then if flow logs become unavailable for any reason such as if manually disabled
flow logs, modified API permissions, or an internal error occurred, when access is
restored, logs from the preceding seven days only are ingested.
- STEP 6 | ( Optional) Allow Prisma Cloud to monitor all current and future GCP projects associated with
the service account.
If have multiple GCP projects, enable Automatically onboard projects that are accessible by this
service account. to allow Prisma Cloud to monitor all current and future GCP projects associated with
the Service Account.
- STEP 7 | Set up the Service Account for Prisma Cloud.
1. Download the Terraform template for the mode selected.
Prisma Cloud recommends that create a directory to store the Terraform template download. This allows to manage the templates when add a different Google project to
Prisma Cloud. Give the directory a name that uniquely identifies the subscription for which you're
using it (for example, onboard-<subscription-name>).
2. Open a new tab on browser and sign in to the Google Cloud Shell.
3. Upload the template to the Google Cloud Shell.
4. Run the following Terraform commands to generate the Service Account.
1. terraform init
2. terraform apply 5. Upload Service Account Key (JSON) file, review the GCP onboarding configuration displayed on
screen to verify that it is correct, and click Next.
The service account security key is used for service-to-service authentication within GCP. The private key file is required to authenticate API calls between GCP projects and Prisma Cloud.
If are on a PC, when copy the JSON file output from Google Cloud Shell the content is formatted as text instead of JSON. When upload this file to Prisma Cloud, the Invalid JSON file error displays. To fix the error, use a JSON
formatting tool such as Sublime or Atom to find the errors (for example, the certificate
value should be a single line) and validate the format before upload the file on
Prisma Cloud.
6. Enable the GCP APIs on all projects.
must enable the Stackdriver Logging API (logging.googleapis.com) to monitor audit logs and any
other GCP APIs for which want Prisma Cloud to monitor resources, on all GCP projects; enabling
it only of the project that hosts the service account is not adequate. For example, in the Google Cloud
Shell, enter:
gcloud services enable serviceusage.googleapis.com
 appengine.googleapis.com bigquery.googleapis.com
 cloudfunctions.googleapis.com dataflow.googleapis.com
 dns.googleapis.com dataproc.googleapis.com
 cloudresourcemanager.googleapis.com cloudkms.googleapis.com
 sqladmin.googleapis.com compute.googleapis.com storagecomponent.googleapis.com recommender.googleapis.com  iam.googleapis.com container.googleapis.com monitoring.googleapis.com
 logging.googleapis.com
- STEP 8 | Select the account groups to associate to project and click Next.
must assign each cloud account to an account group, and Create an Alert Rule to associate the account group with it to generate alerts when a policy violation occurs.
- STEP 9 | Verify the onboarding Status of GCP project to Prisma Cloud and click Done.
review the status and take necessary actions to resolve any issues encountered during the onboarding process by viewing the Cloud Accounts page. It takes between 4-24 hours for the flow log
data to be exported and analyzed before review it on Prisma Cloud. To verify if the flow log data from GCP project has been analyzed, run a network query on the Investigate page.
1. Go to Cloud Accounts, locate GCP project and view the status.
If Prisma Cloud GCP IAM role does not have adequate permissions to ingest data on the monitored
resources within project, the status icon displays as red or amber and it lists the permissions that
are missing. 2. Go to Investigate, replace the name with GCP Cloud Account name and enter the following
network query.
This query allows to list all network traffic from the Internet or from Suspicious IP addresses with
over 0 bytes of data transferred to a network interface on any resource on any cloud environment.
 network where cloud.account = ‘{{cloud account name}}’ AND
 source.publicnetwork IN (‘Internet IPs’, ‘Suspicious IPs’) AND bytes >
 0

Update an Onboarded Google Cloud Account
For a Google Cloud project or organization that have already added to Prisma Cloud, update
the following options.
- STEP 1 | Log in to the Prisma Cloud administrative console.
- STEP 2 | Select the Google Cloud account want to modify.
- Select Settings > Cloud Accounts and click on the name of the cloud account to manage from the list
of cloud accounts.
- Change the account groups want to monitor.
- If have onboarded the GCP Organization, select which folders and projects to monitor, or
monitor and protect.
On the Google Cloud console, verify that the IAM permissions for the service account includes the Folder Viewer role. If this role does not have adequate permissions, the following error displays
- Update the flow log bucket name.
- Enable Flow Log Compression on GCP and select Use Dataflow to generate compressed logs
(significantly reduces network egress costs).
Enable Flow Logs for GCP Projects
With VPC flow logs, Prisma Cloud helps visualize flow information for resources deployed in your
GCP projects. VPC flow logs on GCP provide flow-level network information of packets going to and
from network interfaces that are part of a VPC, including a record of packets flowing to a source port and
destination port, the number of distinct peers connecting to an endpoint IP address and port, so that monitor applications from the perspective of network. On the Investigate page, view
the traffic flow between virtual machines in different service-projects and/or host-projects that are using
shared VPC network and firewall rules.
VPC flow logs are supported on VPC networks only, and are not available for legacy
networks on GCP.
To analyze these logs on Prisma Cloud must enable VPC flow logs for each VPC subnet and export the logs to a sink that holds a copy of each log entry. Prisma Cloud requires to export the flow logs to a single Cloud Storage bucket, which functions as the sink destination that holds all VPC flow logs in your
environment. When then configure Prisma Cloud to ingest these logs, the service can analyze this data and provide visibility into network traffic and detect potential network threats such as crypto mining, data exfiltration, and host compromises.
Prisma Cloud automates VPC flow log compression using the Google Cloud Dataflow service, and saves
them to Storage bucket for ingestion. Consider enabling the Google Cloud Dataflow Service and
enabling log compression because transferring raw GCP Flow logs from storage bucket to Prisma Cloud can add to data cost. See Flow Log Compression on GCP to make sure that have the permissions to create and run pipelines for a Cloud Dataflow job.
Enabling flow logs will incur high network egress costs. Palo Alto Networks strongly recommends that enable Flow Log Compression on GCP to significantly reduce the network egress costs associated with
sending uncompressed GCP logs to the Prisma Cloud infrastructure.
- STEP 1 | Enable flow logs for VPC networks on GCP. To analyze network traffic, must enable flow logs for each project want Prisma Cloud to
monitor.
1. Log in to GCP console and select project.
2. Select Navigation menu > VPC network > VPC networks.
3. Select VPC network and click EDIT.
4. Select Flow logs On to enable flow logs.
5. Set the Aggregation Interval to 15 min.
6. Set the Sample rate to 100%. As a best practice, setting the aggregate interval and the sample rate as recommended above generates alerts faster on Prisma Cloud and reduces network
costs incur.
7. Save changes.
- STEP 2 | Create a Sink to export flow logs.
must create a sink and specify a Cloud Storage bucket as the export destination for VPC flow logs.
must configure a sink for every project that want Prisma Cloud to monitor and configure a single Cloud Storage bucket as the sink destination for all projects. When Add GCP Project to
Prisma Cloud, must provide the Cloud Storage bucket from which the service can ingest VPC flow
logs. As a cost reduction best practice, set a lifecycle to delete logs from Cloud Storage bucket.
1. Select Navigation menu > Logging > Logs Viewer > Create Sink
2. Select GCE Subnetwork.
3. Change All logs to compute.googleapis.com/vpc_flows and click OK.
4. Enter a name and select Cloud Storage as the Sink Service.
5. Select an existing Cloud Storage bucket or create a new Cloud Storage bucket as the Sink
Destination, and click Create Sink.
6. Add a lifecycle rule to limit the number of days store flow logs on the Cloud Storage bucket.
By default, logs are never deleted. To manage cost, specify the threshold (in number of days) for which want to store logs.
1. Select Navigation Menu > Storage > Browser.
2. Select the Lifecycle link for the storage bucket want to modify.
3. Add rule and Select object conditions to set Age to 30 days and Select Action as Delete.
Logs that are stored on Cloud Storage bucket will be deleted in 30 days.
4. Select Continue and Save changes.
- STEP 3 | Add the name of the Cloud Storage bucket referenced above in Flow Logs Storage Bucket
when Add GCP Project to Prisma Cloud.
Flow Log Compression on GCP
Prisma Cloud enables to automate the compression of flow logs using the Google Cloud Dataflow
service. This additional automation on Prisma cloud addresses the lack of native compression
support for flow logs sink setup on GCP, and helps reduce the egress costs associated with
transferring large volume of logs to the Prisma Cloud infrastructure. Therefore, Prisma Cloud
recommends that enable flowlog compression.
When enable dataflow compression on Prisma Cloud, the dataflow pipeline resources are created in the same GCP project associated with the Google Cloud Storage bucket to which VPC Flow logs are sent, and it saves the compressed logs also to the Cloud Storage bucket. Therefore, if are onboarding a GCP
Organization and enabling dataflow compression to it or enabling dataflow compression to an existing GCP
Organization that has been added to Prisma cloud, make sure that the Dataflow-enabled Project ID is the same Google Cloud Storage bucket to which send VPC flow logs.
In order to launch the dataflow job and create and stage the compressed files, the following permissions are
required:
- Enable the Dataflow APIs.
The API is dataflow.googleapi.com.
- Grant the service account with permissions to:
- Run and examine jobs— Dataflow Admin role
- Create a network, subnetwork, and firewall rules within VPC— compute.networks.create, compute.subnetworks.create, compute.firewalls.create, compute.networks.updatepolicy To enable connectivity with the Dataflow pipeline resources and the compute instances that perform
log compression within VPC, Prisma Cloud creates a network, subnetwork, and firewall rules
in VPC. view the compute instance that are spun up with the RQL config where
api.name='gcloud-compute-instances-list' AND json.rule = name starts with
"prisma-compress"
For details on enabling the APIs, see Permissions and Roles for GCP Account on Prisma Cloud.
The Cloud Dataflow service spins up short lived compute instances to handle the compression jobs and may have associated costs with the service. Palo Alto Networks recommends keeping Cloud Storage
bucket in the same project in which have enabled the Dataflow service. Based on the location of your
Cloud Storage bucket, Prisma Cloud launches the Cloud Dataflow jobs in the following regions:
Storage Bucket Region Region Where the Dataflow is Launched
us-central1
us-east1
us-west1
europe-west1
europe-west4
asia-east1
asia-northeast1
us-central1
us-east1
us-west1
europe-west1
europe-west4
asia-east1
asia-northeast1
eur4
or
eu
europe-west4
asia asia-east1 Storage Bucket Region Region Where the Dataflow is Launched
us us-central1
or
us-east1
Any other region us-central1
Add GCP Organization to Prisma Cloud
Begin here to add a GCP Organization and folders to Prisma Cloud. If have added a GCP project to
Prisma Cloud and now want to add the GCP Organization to which the project belongs, the existing
GCP project is moved under the Organization in Prisma Cloud.
When add the GCP Organization to Prisma Cloud, specify which folders and projects to include
or exclude under the organization resource hierarchy.
- STEP 1 | Review the best practices for onboarding GCP Organization to Prisma Cloud.
1. Enable the GCP APIs on each GCP project.
For the cloud services that want Prisma Cloud to monitor or monitor and protect, must
enable the APIs listed in Permissions and Roles for GCP Account on Prisma Cloud. If a cloud service
API is not enabled on a GCP project, Prisma Cloud skips the ingestion for the respective service; must, however, ensure that Service Usage API is enabled on each GCP project that want Prisma Cloud to monitor or monitor and protect under GCP Organization hierarchy.
To skip ingestion for a cycle, Prisma cloud watches the response from the Service Usage API for the details on which cloud services are enabled in a GCP project. For example, if have not enabled
cloud functions in one or more GCP projects within the GCP Organization, Prisma cloud can learn
about it and skip the ingestion cycle for this cloud service.
2. Create the service account in a dedicated GCP project.
GCP enforces a limit on the API calls allowed to a GCP project/IAM service account. When create the service account in a dedicated GCP project, ensure that the API calls that Prisma Cloud makes do not interfere with any quota limits against production workloads and services
hosted in the separate GCP project.
3. Verify that have granted all the required permissions to the Prisma Cloud service account.
If the service account does not have the IAM permissions required to retrieve data, Prisma Cloud
skips ingestion of the respective cloud.service(s) for onboarded account.
- STEP 2 | Access Prisma Cloud and select Settings > Cloud Accounts > Add New.
- STEP 3 | Select Google Cloud as the Cloud to Protect.
- STEP 4 | Enter a Cloud Account Name.
A cloud account name is auto-populated for you. replace it with a cloud account name that
uniquely identifies this GCP organization on Prisma™ Cloud. - STEP 5 | Select the Mode.
Decide whether to enable permissions to only monitor (read-only access) or to monitor and protect
(read-write access) the resources in cloud account. selection determines which Terraform
template is used to automate the process of creating the service account and attaching the roles
required for Prisma Cloud.
- STEP 6 | Select Organization for Onboard Using and enter additional details.
1. Enter Organization Name and Organization ID.
All the GCP projects under the Organization hierarchy—current and future—will be monitored
by Prisma Cloud. To find GCP Organization ID, log in to the GCP console and select your
organization.
2. Enter Project ID and the name of Flow Log Storage Bucket.
3. (Optional) Enable Use Dataflow to generate compressed logs. The Terraform template does not enable flow logs, and must complete the workflow in Enable
Flow Logs for GCP Organization for Prisma Cloud to retrieve flow logs. Additionally, if want to
enable flow log compression on Prisma cloud and address the lack of native compression support for flow logs sink setup on GCP, must do it manually too. When enable log compression, Prisma Cloud sets up the network and compute resources required for flow log compression and this can
take up to five minutes.
When enable flow logs, the service ingests flow log data for the last seven days.
Then if flow logs become unavailable for any reason such as if manually disabled
flow logs, modified API permissions, or an internal error occurred, when access is
restored, logs from the preceding seven days only are ingested.
4. Enter the Project ID where enabled the Cloud Dataflow service and click Next.
It is best if this project is where send VPC flow logs too.
- STEP 7 | Set up the Service Account for Prisma Cloud.
A service account is an identity to which grant granular permissions instead of creating
individual user accounts. To monitor all the GCP projects that are within the GCP Organizational
hierarchy, the service account requires four roles. Of the four roles, three are common for granting
permissions at the GCP project level too; the Organization Role Viewer and Folder Viewer roles are
additionally required to grant access to the Organization's properties:
- Viewer—Primitive role.
- (Required for Prisma Cloud Compute, Optional for Prisma Cloud) Compute Security Admin—Predefined role.
- RedLock Viewer—Custom role.
- Organization Role Viewer—Predefined role.
- Folder Viewer- Predefined role.
1. Download the Terraform template for the mode selected.
Prisma Cloud recommends that create a directory to store the Terraform template download. This allows to manage the templates when add a different Google organization
to Prisma Cloud. Give the directory a name that uniquely identifies the subscription for which you're
using it (for example, onboard-<subscription-name>).
2. Open a new tab on browser and sign in to the Google Cloud Shell.
3. Upload the template to the Google Cloud Shell.
4. Run the following Terraform commands to generate the Service Account.
1. terraform init
2. terraform apply
5. Upload Service Account Key (JSON) file, review the GCP onboarding configuration displayed on
screen to verify that it is correct, and click Next.
The service account security key is used for service-to-service authentication within GCP. The private key file is required to authenticate API calls between GCP projects and Prisma Cloud.
If are on a PC, when copy the JSON file output from Google Cloud Shell the content is formatted as text instead of JSON. When upload this file to Prisma Cloud, the Invalid JSON file error displays. To fix the error, use a JSON
formatting tool such as Sublime or Atom (for example, the certificate value should be a single line), and validate the formatting before upload the file on Prisma Cloud. 6. Select the projects want to add to Prisma Cloud.
choose to include:
- All projects included within the organization hierarchy.
- Include a subset or Exclude a subset of projects. Select the relevant tab and choose the projects
to include or exclude.
When select a folder, all existing projects within that folder or sub-folder are onboarded
to Prisma Cloud. The periodic sync also checks for any new projects and sub-folders that subsequently add on the cloud platform and adds them to Prisma Cloud.
7. Resolve any missing permissions or errors.
If the service account does not have adequate permissions, the following error displays. and if there are issues with the following message indicates that there is an issue with the service
account. This error occurs when the service account is deleted, or disabled or when the key is deleted
on the Google Cloud console.
8. Enable the GCP APIs.
In the GCP project where created the service account, must enable the Stackdriver Logging
API (logging.googleapis.com) to monitor audit logs, and any other GCP APIs for which want
Prisma Cloud to monitor resources. For example, in the Google Cloud Shell, enter:
gcloud services enable compute.googleapis.com sqladmin.googleapis.com
 sql-component.googleapis.com storage-component.googleapis.com
 appengine.googleapis.com iam.googleapis.com container.googleapis.com
 logging.googleapis.com monitoring.googleapis.com
 cloudresourcemanager.googleapis.com cloudkms.googleapis.com bigqueryjson.googleapis.com dns.googleapis.com dataflow.googleapis.com
- STEP 8 | Select the account groups to associate to GCP project and click Next.
must assign each cloud account to an account group, and Create an Alert Rule to associate the account group with it to generate alerts when a policy violation occurs.
- STEP 9 | Verify the onboarding Status of GCP Organization to Prisma Cloud and click Done.
If are missing permissions for the GCP IAM role to successfully ingest data from your
GCP Organization, the icon displays red or amber and the details of the permission gaps
display on screen.
review the status and take necessary actions to resolve any issues encountered during the onboarding process by viewing the Cloud Accounts page. It takes between 4-24 hours for the flow log data to be exported and analyzed before review it on Prisma Cloud. To verify if the flow log data from GCP Organization have been analyzed, run a network query on the Investigate page.
- After add the GCP Organization to Prisma Cloud, must create a support
request to delete the GCP Organization or the projects within GCP Organization.
cannot delete the account from Prisma Cloud.
- Because Prisma Cloud has access to all projects associated with a Service Account, if want to remove access to a project that is associated with the Service Account, must remove the project from the Service Account on the GCP IAM console.
In the next scanning cycle, the project is excluded and Prisma Cloud no longer has access to the project.
1. Go to Cloud Accounts, locate GCP account and view the status.
2. Verify the projects that are onboarded to Prisma Cloud.
Select the cloud account name and review the list of projects to verify the include/exclude selections
made earlier.
3. Go to Investigate, replace the name with GCP Cloud Account name and enter the following
network query.
This query allows to list all network traffic from the Internet or from Suspicious IP addresses with
over 0 bytes of data transferred to a network interface on any resource on any cloud environment.
 network where cloud.account = ‘{{cloud account name}}’ AND
 source.publicnetwork IN (‘Internet IPs’, ‘Suspicious IPs’) AND bytes >
 0
  Enable Flow Logs for GCP Organization
Prisma Cloud uses the traffic data in flow logs for GCP organization or folder resource hierarchy to
detect network threats such as cryptomining, data exfiltration, and host compromises. Before Prisma Cloud
can analyze flow log data, must create a sink to export the flow logs to a Cloud Storage bucket. To
configure a sink for whole GCP organization or folder, use the gcloud command line tool.
Enabling flow logs will incur high network egress costs. Palo Alto Networks strongly
recommends that enable Flow Log Compression on GCP to significantly reduce the network egress costs associated with sending uncompressed GCP logs to the Prisma Cloud
infrastructure.
- STEP 1 | Gather the following information from GCP account:
- Cloud Storage bucket name
- Organization ID
- STEP 2 | Download and install the Google Cloud SDK.
During the SDK install, must log in to GCP account. This account must have these three
permissions enabled at the organization level:
- Billing Account Administrator
- Logging Admin
- Organization Administrator
- STEP 3 | Run this command to create a service account needed to configure the sink for Cloud
Storage bucket but replace the Bucket-name with Cloud Storage bucket name and
Organization ID with organization ID.
 $ gcloud logging sinks create <sink-name> storage.googleapis.com/
<bucket-name> --include-children --organization=<organisation-id>
 --log-filter="resource.type="gce_subnetwork" AND logName:"logs/
compute.googleapis.com%2Fvpc_flows"
If are onboarding a GCP folder, must have the Folder Viewer role and
can use the command $ gcloud logging sinks create <sink-name>
storage.googleapis.com/<bucket-name> --include-children --
folder=<folder-id> --log-filter="resource.type="gce_subnetwork"
AND logName:"logs/compute.googleapis.com%2Fvpc_flows" to create a service account needed to configure the sink for Cloud Storage bucket.
- STEP 4 | Grant the service account permission to access Cloud Storage bucket.
1. Select Navigation menu > Storage and select Cloud Storage bucket.
2. Select Permissions > Add members.
3. Add the service account email address for Members, select Storage > Storage Admin and select Add.
- STEP 5 | Add the name of Cloud Storage bucket created above in Flow Logs Storage Bucket when
Add GCP Organization to Prisma Cloud.
- STEP 6 | (Optional) Enable Flow Log Compression on GCP.
Enable flow log compression on Prisma Cloud to automate the compression of flow logs using the Google Cloud Dataflow service. When enabled the compressed logs are stored to the same Storage
bucket as flow logs and mitigates the network egress costs associated with sending uncompressed
GCP logs to the Prisma Cloud infrastructure.
Create a Service Account With a Custom Role for GCP
If prefer to create a service account with more granular permissions to Add GCP Organization
to Prisma Cloudor Add GCP Project to Prisma Cloud, instead of using the Terraform template which
grants the Viewer (primitive) role for read-only access to resources in GCP account, use the following
instructions. -If enable granular permissions, must update the custom role and add additional
permissions that maybe required to ingest data from any new service that is added on
Prisma Cloud.
- To enable dataflow log compression using the Dataflow service, must enable
additional permissions. See Flow Log Compression on GCP for details on ingesting
network log data.
- STEP 1 | Create a YAML file with the custom permissions.
1. Create a YAML file and add the granular permissions for the custom role.
Use this YAML format as an example. must add the permissions for onboarding GCP project
or organization, from the link above, to this file:
 title: prisma-custom-role
 description: prisma-custom-role
 stage: beta
 includedPermissions:
 - compute.networks.list
 - compute.backendServices.list

- STEP 2 | Create the custom role.
When creating a service account, must select a GCP project because GCP does not allow the service
account to belong directly under the GCP Organization.
1. Select the GCP project in which want to create the custom role.
2. Upload the YAML file to the Cloud Shell.
3. Run the gcloud command gcloud iam roles create <prisma customrole name> --
project <project-ID> --file <YAML file name>
- STEP 3 | Create a Service Account and attach the custom role to it.
1. Select IAM & Admin > Service Accounts page in the Google Cloud Console.
2. Create Service Account and add the role created earlier to it.
3. Create a key and download the private key.
- STEP 4 | Continue to Add GCP Project to Prisma Cloud and use the private key for the service
account to complete onboarding.
- STEP 5 | (For onboarding GCP Organization only) Create the custom role in the GCP Organization level.
1. Select GCP Organization.
2. Verify that the YAML file created in Step 1 includes the additional permissions for GCP
Organization.
Run the gcloud command gcloud iam roles create <prisma customrole name> --
organization <org ID> --file <YAML File name> - STEP 6 | (For onboarding GCP Organization only) Set up Service Account to monitor all the GCP
folders and projects within the GCP Organization.
must associate the Service account created in the project in Step 3 to the GCP Organizationlevel and add the custom role created in the previous step. Additionally, must add the predefined role for Organization Viewer to the service account. All these tasks together enable the service account to monitor all the GCP projects that are within the GCP Organizational hierarchy.
1. Copy the service account member address.
Select the project that used to create the service account, and select IAM & admin > IAM to
copy the service account member address.
2. Select Organization, select IAM & Admin > IAM to Add members to the service account.
3. Paste the service account member address copied as New members, and Select a role.
4. Select the custom role created in Step 4, and click + ADD ANOTHER ROLE.
5. Select Resource Manager > Organization Role Viewer, and Folder Viewer role, and Save.
The Organization Viewer role enables permissions to view the Organization name without granting
access to all resources in the Organization. The Folder Viewer roles is also required to onboard your
GCP folders. - STEP 7 | (For onboarding GCP Organization only) Continue to Add GCP Organization to Prisma Cloud and use the private key associated with service account to complete onboarding.
GCP APIs Ingested by Prisma Cloud
The following are GCP APIs that have been ingested by Prisma Cloud.
SERVICE API NAME IN PRISMA CLOUD
(Alpha release) Google API Key gcloud-api-key
Google Compute Engine (GCE) -gcp-compute-disk-list
- gcloud-compute-image
- gcloud-compute-instances-list
- gcloud-compute-interfaces-list
- gcloud-compute-project-info
- gcloud-compute-nat
- gcloud-compute-route
- gcloud-compute-router
- gcloud-compute-vpn-tunnel
- gcloud-virtual-network-interface
Google VPC -gcloud-compute-networks-list
- gcloud-compute-networks-subnets-list
- gcloud-compute-firewall-rules-list SERVICE API NAME IN PRISMA CLOUD
Google Access Context Manager gcloud-access-policy Google App Engine -gcloud-app-engine-firewall-rule
- gcloud-app-engine-application
Google Bigtable -gcloud-bigtable-instance-list
- gcloud-bigtable-table
Google Cloud Function gcloud-cloud-function
Google Cloud SQL gcloud-sql-instances-list
Google Cloud Run gcloud-cloud-run-services-list
Google Cloud Storage gcloud-storage-buckets-list
Google Cloud Spanner -gcloud-cloud-spanner-instance
- gcloud-cloud-spanner-instance-config
Google Dataproc Clusters gcloud-dataproc-clusters-list
Google Recommendations for GCP IAM
Recommender
gcloud-iam-policy-recommendation-list
Google Cloud Identity & Access Management
(Cloud IAM)
- gcloud-iam-service-accounts-list
- gcloud-iam-service-accounts-keys-list
- gcloud-iam-get-audit-config
- gcloud-project-iam-role
- gcloud-organization-iam-role
Google Cloud Resource Manager -gcloud-organization-iam-policy -gcloud-projects-get-iam-user
- gcloud-projects-get-iam-policy Google Cloud SQL gcloud-sql-instances-list
Google BigQuery -gcloud-bigquery-dataset-list
- gcloud-bigquery-table
Google Kubernetes Engine gcloud-container-describe-clusters
Google Cloud DNS -gcloud-dns-project-info
- gcloud-dns-managed-zone
Google Cloud Key Management Service (KMS) gcloud-kms-keyring-list
Google Cloud Memorystore gcloud-redis-instances-list
Google Service Usage gcloud-services-list SERVICE API NAME IN PRISMA CLOUD
Google Stackdriver Logging -gcloud-logging-sinks-list
- gcloud-events-logging-sinks-list
Google Stackdriver Monitoring Policy -gcloud-monitoring-policies-list
- gcloud-logging-metric
Google Cloud Load Balancing -gcloud-compute-internal-lb-backend-service
- gcloud-compute-target-pools
- gcloud-compute-target-http-proxies
- gcloud-compute-target-https-proxies
- gcloud-compute-url-maps
- gcloud-compute-global-forwarding-rule
- gcloud-compute-ssl-policies
Google PubSub -gcloud-pubsub-topic
- gcloud-pubsub-subscription
- gcloud-pubsub-snapshot Onboard Alibaba Cloud Account
Use Prisma™ Cloud for monitoring Alibaba Cloud infrastructure in Mainland China and International
regions.
To begin monitoring the resources on cloud infrastructure, must first create a role on Alibaba
Cloud and then connect Alibaba cloud account to Prisma Cloud. When add cloud account
to Prisma Cloud, the API integration between cloud infrastructure provider and Prisma Cloud is
established and begin monitoring the resources and identify potential security risks in your
infrastructure.
- Set Up Alibaba Account
- Add an Alibaba Cloud Account on Prisma Cloud
- Alibaba APIs Ingested by Prisma Cloud
Set Up Alibaba Account
Prisma Cloud is available for visibility and monitoring of Alibaba Cloud infrastructure in Mainland China
and International regions. The first step to start monitoring resources on Alibaba Cloud is to grant
Prisma Cloud access to account. To do this, must create a role and attach policies that enable
permissions to authorize access to the assets deployed within the account. choose to create a custom policy with granular permissions or use the Alibaba Cloud system policy to enable ReadOnlyAccess.
After create the role and enable permissions, add the Alibaba Cloud Resource Name (ARN) on
Prisma Cloud so that it can assume the role to monitor Alibaba Cloud account.
- STEP 1 | (Required if want to enable granular access permissions) Create a custom policy.
Creating a custom policy allows to use the principle of least privilege and enable the bare-minimum
permissions that Prisma Cloud currently requires to monitor account. If do not want to update
these permissions periodically, skip ahead to Step 2 and use the Alibaba Cloud system policy to
enable ReadOnlyAccess permissions to all aliyun services.
1. Download the permissions for Alibaba China.
The JSON file includes the required permissions.
2. Log in to the Alibaba Cloud console for China region.
3. Select Resource Access Management > Permissions > Policies > Create Policy.
4. Enter a new Policy Name and select Script.
5. Paste the contents in to the Policy Document and click OK. - STEP 2 | Create a RAM role.
must create a RAM role and attach policies to authorize API access to Prisma Cloud. can
attach the custom policy with granular permissions or use the Alibaba Cloud system policy to enable
ReadOnlyAccess.
1. On the Alibaba Cloud console, select Product > Resource Access Management.
2. Select RAM Roles > Create RAM Role. 3. Select Trusted entity type as Alibaba Cloud Account and Next.
4. Enter a RAM Role Name.
5. Enter the Prisma Cloud Account ID as a trusted Alibaba Cloud account.
If Prisma Cloud instance is on https:/app.prismacloud.cn, the Prisma Cloud Account ID is
1306560418200997. Otherwise, the Prisma Cloud Account ID is 5770382605230796.
Enter the appropriate account ID in Select Trusted Alibaba Cloud Account > Other Alibaba Cloud
Account and click OK.
6. Select Add Permissions to RAM Role.
Either attach the permissions associated with the custom policy (if created one), or use the system policy. -System Policy 7. Click Finished.
- STEP 3 | Copy the Alibaba Cloud Resource Name (ARN).
need the ARN to add the Alibaba cloud account on Prisma Cloud.
1. Select RAM Roles and search for the name entered earlier. 2. Note the ARN.
- STEP 4 | Add an Alibaba Cloud Account on Prisma Cloud.
Add an Alibaba Cloud Account on Prisma Cloud
After create a RAM role with permissions that enable Prisma Cloud programmatic access to cloud
resources on Alibaba Cloud, all now need to do for visibility into changes on the cloud infrastructure is
to add the account want to monitor.
- STEP 1 | Access Prisma Cloud.
If have not already activated account, check email for the welcome to Prisma Cloud email
that includes username, and create a new password to log in. On first-time login to Prisma Cloud in the Alibaba Mainland China region, must accept
the EULA. When click the EULA, a new page displays where review the content. As a temporary work around, to Agree and Submit must refresh the page
and log in again with credentials.
- STEP 2 | Select Settings > Cloud Accounts > Add New.
- STEP 3 | Select Alibaba Cloud as the Cloud to Protect. - STEP 4 | Enter a Cloud Account Name.
A cloud account name is auto-populated for you. replace it with a cloud account name that
uniquely identifies Alibaba Cloud account on Prisma™ Cloud.
- STEP 5 | Enter the Alibaba Cloud Resource Name (ARN) as RAM Role and click Next.
The ARN is the unique identifier for the RAM role created to authorize API access for Prisma Cloud.
When enter the ARN, the Alibaba Cloud Account ID gets added automatically. - STEP 6 | Select one or more account groupsand click Next.
must assign each cloud account to an account group, and create an alert rule to associate the account group with it to generate alerts when a policy violation occurs.
- STEP 7 | Verify the onboarding status.
If have set up the RAM role and policies properly, the onboarding process should be successful. - STEP 8 | Select Done.
- STEP 9 | Next Steps:
- Review the Prisma Cloud default Policies for Alibaba Cloud.
Select Policies, set the Cloud Type filter as Alibaba Cloud and view all the Config policies that are
available to detect any misconfigurations in infrastructure. -Start using the Prisma Cloud Asset Inventory for visibility.
Alibaba APIs Ingested by Prisma Cloud
Prisma Cloud ingests the following Alibaba Cloud APIs to retrieve metadata on resources in the Alibaba Cloud environment.
SERVICE API NAME IN PRISMA CLOUD
Alibaba Resource Access Management -alibaba-cloud-ram-password-policy -alibaba-cloud-ram-group
- alibaba-cloud-ram-policy -alibaba-cloud-ram-role
- alibaba-cloud-ram-user
Action Trail alibaba-cloud-action-trail
Elastic Compute Service -alibaba-cloud-ecs-disk
- alibaba-cloud-ecs-instance
- alibaba-cloud-ecs-security-group
Object Storage Services alibaba-cloud-oss-bucket-info
VPC alibaba-cloud-vpc
RDS alibaba-cloud-rds-instance
Server Load Balancer alibaba-cloud-load-balancer Cloud Service Provider Regions on Prisma Cloud
View the list of all cloud regions supported on Prisma Cloud.
- AWS Regions
- Azure
- GCP
- Alibaba Cloud
AWS Regions
Region ID Region Name
AWS Global
ap-east-1 AWS Hong Kong
ap-northeast-1 AWS Tokyo
ap-northeast-2 AWS Seoul
ap-south-1 AWS Mumbai
ap-southeast-1 AWS Singapore
ap-southeast-2 AWS Sydney
ca-central-1 AWS Canada
eu-central-1 AWS Frankfurt
eu-north-1 AWS Stockholm
eu-west-1 AWS Ireland
eu-west-2 AWS London
eu-west-3 AWS Paris
me-south-1 AWS Bahrain
sa-east-1 AWS Sao Paulo
us-east-1 AWS Virginia
us-east-2 AWS Ohio
us-west-1 AWS California Region ID Region Name
us-west-2 AWS Oregon
AWS Gov
us-gov-east-1 AWS GovCloud (US-East)
us-gov-west-1 AWS GovCloud (US-West)
AWS China
cn-north-1 AWS Beijing
cn-northwest-1 AWS Ningixa
Azure
Region ID Region Name
Azure Commercial
australiacentral Azure Australia Central
australiacentral2 Azure Australia Central 2
australiaeast Azure Australia East
australiasoutheast Azure Australia Southeast
brazilsouth Azure Brazil South
canadacentral Azure Canada Central
canadaeast Azure Canada East
centralindia Azure Central India
centralus Azure Central US
chinaeast Azure China East
chinaeast2 Azure China East 2
chinanorth Azure China North
chinanorth2 Azure China North 2
eastasia Azure East Asia
eastus Azure East US Region ID Region Name
eastus2 Azure East US 2
francecentral Azure France Central
francesouth Azure France South
germanycentral Azure Germany Central
germanynorth Azure Germany North
germanynortheast Azure Germany Northeast
germanywestcentral Azure Germany West Central
japaneast Azure Japan East
japanwest Azure Japan West
koreacentral Azure Korea Central
koreasouth Azure Korea South
northcentralus Azure North Central US
northeurope Azure North Europe
norwayeast Azure Norway East
norwaywest Azure Norway West
southafricanorth Azure South Africa North
southafricawest Azure South Africa West
southcentralus Azure South Central US
southeastasia Azure Southeast Asia
southindia Azure South India
switzerlandnorth Azure Switzerland North
switzerlandwest Azure Switzerland West
uaecentral Azure UAE Central
uaenorth Azure UAE North
uksouth Azure UK South
ukwest Azure UK West Region ID Region Name
westcentralus Azure West Central US
westeurope Azure West Europe
westindia Azure West India
westus Azure West US
westus2 Azure West US 2
Azure Government
usgovarizona Azure Gov Arizona (US)
usgoviowa Azure Gov Iowa (US)
usgovtexas Azure Gov Texas (US)
usgovvirginia Azure Gov Virginia (US)
GCP
Region ID Region Name
asia GCP Asia Pacific
asia-east1 GCP Taiwan
asia-east2 GCP Hong Kong
asia-northeast1 GCP Tokyo
asia-northeast2 GCP Osaka
asia-northeast3 GCP Seoul
asia-south1 GCP Mumbai
asia-southeast1 GCP Singapore
australia-southeast1 GCP Sydney
eu GCP European Union
eur4 GCP Finland and Netherlands
europe GCP Europe
europe-north1 GCP Finland Region ID Region Name
europe-west1 GCP Belgium
europe-west2 GCP London
europe-west3 GCP Frankfurt
europe-west4 GCP Netherlands
europe-west6 GCP Switzerland
nam4 GCP Iowa and South Carolina
northamerica-northeast1 GCP Montreal
southamerica-east1 GCP Sao Paulo
us GCP United States
us-central1 GCP Iowa
us-east1 GCP South Carolina
us-east4 GCP Northern Virginia
us-west1 GCP Oregon
us-west2 GCP Los Angeles
asia-northeast3 GCP Seoul
us-west4 GCP Las Vegas asia-southeast2 GCP Jakarta
us-west3 GCP Salt Lake City
Alibaba Cloud
Region ID Region Name
Alibaba China
ali.cn.cn-beijing Alibaba Cloud Beijing
ali.cn.cn-chengdu Alibaba Cloud Chengdu
ali.cn.cn-hangzhou Alibaba Cloud Hangzhou
ali.cn.cn-huhehaote Alibaba Cloud Hohhot Region ID Region Name
ali.cn.cn-qingdao Alibaba Cloud Qingdao
ali.cn.cn-shanghai Alibaba Cloud Shanghai
ali.cn.cn-shenzhen Alibaba Cloud Shenzhen
ali.cn.cn-zhangjiakou Alibaba Cloud Zhangjiakou
Alibaba International
ali.int.ap-northeast-1 Alibaba Cloud Tokyo
ali.int.ap-south-1 Alibaba Cloud Mumbai
ali.int.ap-southeast-1 Alibaba Cloud Singapore
ali.int.ap-southeast-2 Alibaba Cloud Sydney
ali.int.ap-southeast-3 Alibaba Cloud Kuala Lumpur
ali.int.ap-southeast-5 Alibaba Cloud Jakarta
ali.int.cn-hongkong Alibaba Cloud Hong Kong
ali.int.eu-central-1 Alibaba Cloud Frankfurt
ali.int.eu-west-1 Alibaba Cloud London
ali.int.me-east-1 Alibaba Cloud Dubai
ali.int.us-east-1 Alibaba Cloud Virginia
ali.int.us-west-1 Alibaba Cloud Silicon Valley 157

## Manage Prisma Cloud Administrators
Role-based access controls allow to restrict access to the cloud accounts based on a user’s role in the organization. For example, assign groups of accounts to a line of business cloudOps, DevOps, and SecOps owners to restrict their access to Prisma Cloud for the accounts they own.
> Prisma Cloud Administrator Roles
> Create and Manage Account Groups on Prisma Cloud
> Create Prisma Cloud Roles
> Prisma Cloud Administrator Permissions
> Manage Roles in Prisma Cloud
> Add Administrative Users On Prisma Cloud
> Create and Manage Access Keys
> Manage Prisma Cloud Profile
> Set up SSO Integration on Prisma Cloud
> Define Prisma Cloud Enterprise and Anomaly Settings

### Prisma Cloud Administrator Roles
A user on Prisma Cloud is someone who has been assigned administrative privileges, and a role defines the type of access that the administrator has on the service. When define a role, specify the permission group and the account groups that the administrator can manage. Prisma Cloud has four types
of permission groups built-in for administrators.
- System Admin—Full control (read/write permissions) to the service, and they can create, edit, or delete
account groups or cloud accounts. Only System administrators have access to all Settings on Prisma Cloud and can view audit logs to analyze actions performed by other users who have been assigned
administrative privileges.
If use the System Admin role with Only for Compute capabilities enabled, the administrator only has full control (read/write permissions) to the Compute tab and APIs on Prisma Cloud, and does not have
access to the rest of Prisma Cloud capabilities.
- Account Group Admin—Read/write permissions for the cloud accounts and account groups to which
they are granted access.
An account group administrator can only view resources deployed within the cloud accounts to which
they have access. Resources deployed on other cloud accounts that Prisma Cloud monitors are excluded
from the search or investigation results.
- Account Group Read Only—Read only permissions to view designated sections of Prisma Cloud. This
role does not have permissions to modify any settings.
- Account and Cloud Provisioning Admin—Combines the permissions for the Account Group Admin
and the Cloud Provisioning Admin to enable an administrator who is responsible for a line of business.
With this role, in addition to being able to onboard cloud accounts, the administrator can access the dashboard, manage the security policies, investigate issues, view alerts and compliance details for the designated accounts only.
- Cloud Provisioning Admin—Permissions to onboard and manage cloud accounts from Prisma Cloud and
the APIs, and the ability to create and manage the account groups. With this role access is limited to
Settings > Cloud Accounts and Settings > Account Groups on the admin console.
- Build and Deploy Security—Restricted permissions to DevOps users who need access to a subset
of Compute capabilities and/or API access to run IDE, SCM and CI/CD plugins for Infrastructure as Code and image vulnerabilities scans. For example, the Build and Deploy Security role enables readonly permissions to review vulnerability and compliance scan reports on Compute and to manage and
download utilities such as Defender images, plugins and twistcli.
And if use the Build and Deploy Security role with Access key only enabled, the administrator can
create one access key to use the Prisma Cloud Compute APIs.
See Prisma Cloud Compute Roles for more details for the roles and associated permissions.
Add Administrative Users On Prisma Cloud. View permissions associated with each role on
Settings > Roles > +Add New Create and Manage Account Groups on Prisma Cloud
use Account Groups to combine access to multiple cloud accounts with similar or different
applications that span multiple divisions or business units, so that manage administrative access to
these accounts from Prisma Cloud.
When onboard a cloud account to Prisma Cloud, assign the cloud account to one or more
account groups, and then assign the account group to Prisma Cloud Administrator Roles. Assigning an
account group to an administrative user on Prisma Cloud allows to restrict access only to the resources
and data that pertains to the cloud account(s) within an account group. Alerts on Prisma Cloud are applied
at the cloud account group level, which means setup separate alert rules and notification flows for different cloud environments.
Create an Account Group
- STEP 1 | Select Settings > Account Groups and click + Add New.
- STEP 2 | Enter a Name and Description for the Account Group.
- STEP 3 | Select the cloud accounts that want to group together in this account group and click Save.
Manage Account Groups
To view and manage account groups:
- STEP 1 | Select Settings > Account Groups.
- STEP 2 | To edit the details of an Account Group, click the record and change any details.
- STEP 3 | To clone an Account Group, hover over the account group and click Clone.
Cloning an account group is creating a copy of an existing account group. Cloning serves as a quick
method of creating a new account group if choose to change few details of the source account
group.
- STEP 4 | To delete an Account Group, hover over the account group and click Delete. Create Prisma Cloud Roles
Roles on Prisma Cloud enable to specify what permissions an administrator has, and to which cloud
accounts they have access, what policies they can apply, and how they interact with alerts and reports, for example.
When create a cloud account, assign one or more cloud account to account group(s) and then
attach the account group to the role create. This flow allows to ensure the administrator who
has the role can access the information related to only the cloud account(s) to which have authorized
access.
- STEP 1 | Select Settings > Roles and click + Add New.
- STEP 2 | Enter a name and a description for the role.
- STEP 3 | Select a permission group.
See Prisma Cloud Administrator Roles for a description of the following permission groups.
- Select System Admin to allow full access and control over all sections of Prisma Cloud including
overall administration settings and permissions management. To limit access to the Compute
capabilities and APIs that enable to secure host, serverless, and container deployments, select Only for Compute capabilities.
- Select Account Group Admin to allow full access over designated accounts including a subset
of administration settings and permissions management for the designated Account Groups specify. By default an Account Group Admin has the ability to dismiss, snooze, and resolve alerts that are generated against all policies included in an alert rule defined by the System Admin. can, however, restrict the ability dismiss or resolve alerts.
- Select Account ReadOnly to allow read access to designated accounts across Prisma Cloud
administrative console, excluding any administration settings and permissions management.
- Select Account and Cloud Provisioning Admin—to enable an administrator who is responsible for a line of business.With this role the permissions allow to onboard cloud accounts, and access the dashboard, manage the security policies, investigate issues, view alerts and compliance details for the designated Account Groups specify. By default an Account and Cloud Provisioning Admin
has the ability to dismiss, snooze, and resolve alerts that are generated against all policies included in
an alert rule defined by the System Admin. can, however, restrict the ability dismiss or resolve
alerts.
- Select Cloud Provisioning Admin to onboard and manage cloud accounts from the admin console
and through APIs. They can also create and manage Account Groups. They do not have access to any
other Prisma Cloud features.
- Select Build and Deploy Security to enable DevOps users who need access to a subset of Compute
capabilities and/or API access to run IDE, SCM and CI/CD plugins for Infrastructure as Code and
image vulnerabilities scans.
- STEP 4 | Click View Permissions to see the permissions associated with every permission group. The Permission matrix shows what permissions each permission group has within Prisma Cloud. - STEP 5 | (Optional) Restrict the ability dismiss or resolve alerts.
If would like to ensure that only the System Admin role can manage alerts associated with a policy defined by a system administrator, select Restrict alert dismissal. When enabled, an administrator with
any other role such as Account Group Admin or Account and Cloud Provisioning Admin roles cannot
dismiss or resolve alerts.
- STEP 6 | (Optional) Enable access to data collected from Prisma Cloud Defenders.
Select Non-onboarded cloud accounts access, if would like to ensure that the roles—Account Group
Admin, Account Group Read Only, and Account and Cloud Provisioning Admin—can view data sent from
Prisma Cloud Defenders deployed on resources hosted on-premises or on cloud platforms deployed on
private or public cloud platforms that are not being monitored by Prisma Cloud, such as on OpenShift, on-prem Kubernetes clusters, or AWS Fargate. When enable this option, administrators who are
assigned the role can view data collected from Defenders on the Compute tab.
- STEP 7 | Select Account Groups that want to associate with this role and click Save. Prisma Cloud Administrator Permissions
The following table provides a list of the access privileges associated with each role for different parts of the Prisma Cloud administrative console. For details on permissions for Prisma Cloud Compute roles, see
Prisma Cloud Compute roles.
Compute
Role
Sys
Admin
System
Admin
(Only
allow
compute
access)
Auditor Defender
Manager
Auditor DevOps CI DevSecOps
Prisma Cloud
Role
System
Admin
System
Admin
with
Compute
Access
Only
Account
Group
Admin
Cloud
Provisioning
Admin
Account
and Cloud
Provisioning
Admin
Build and
Deploy
Security
Build and
Deploy
Security
Account
Group
Read
Only
Dashboard All
accounts
No Designated
accounts
No Designated
accounts
No No Designated
accounts
Inventory All
accounts
No Designated
accounts
No Designated
accounts
No No Designated
accounts
Investigate
Running
Queries
All
accounts
No Designated
accounts
No Designated
accounts
No No Designated
accounts
Save
Searches
All
accounts
No Designated
accounts
No Designated
accounts
No No No
Edit /
Delete
Saved
Search
Yes No Users in
this role
No Users in
this role
No No No
Policies
View
Policy Yes No Yes No Yes No No Yes
Create
Policy Yes No Yes No Yes No No No
Add/
Edit CLI
Remediation
in Policy Yes No No No No No No No Compute
Role
Sys
Admin
System
Admin
(Only
allow
compute
access)
Auditor Defender
Manager
Auditor DevOps CI DevSecOps
Prisma Cloud
Role
System
Admin
System
Admin
with
Compute
Access
Only
Account
Group
Admin
Cloud
Provisioning
Admin
Account
and Cloud
Provisioning
Admin
Build and
Deploy
Security
Build and
Deploy
Security
Account
Group
Read
Only
Edit /
Delete /
Disable
Policy Yes No Users in
this role
No Users in
this role
No No No
Compliance
Compliance
Dashboard
All
accounts
No Designated
accounts
No Designated
accounts
No No Designated
accounts
Create /
Edit
Reports
All
accounts
No Designated
accounts
No Designated
accounts
No No No
Download
Reports
All
accounts
No Designated
accounts
No Designated
accounts
No No Designated
accounts
Delete
Reports
All
accounts
No Designated
accounts
No Designated
accounts
No No No
Create /
Edit /
Delete
Compliance
Standards
Yes No No No No No No No
View
Compliance
Standards
Yes No Yes No Yes No No Yes
Alerts View /
Search
Alerts All
accounts
No Designated
accounts
No Designated
accounts
No No Designated
accounts
Dismiss /
Resolve /
Snooze
Alerts All
accounts
No Designated
accounts
No Designated
accounts
No No No Compute
Role
Sys
Admin
System
Admin
(Only
allow
compute
access)
Auditor Defender
Manager
Auditor DevOps CI DevSecOps
Prisma Cloud
Role
System
Admin
System
Admin
with
Compute
Access
Only
Account
Group
Admin
Cloud
Provisioning
Admin
Account
and Cloud
Provisioning
Admin
Build and
Deploy
Security
Build and
Deploy
Security
Account
Group
Read
Only
Save
Alert
Filter(s)
All
accounts
No Designated
accounts
No Designated
accounts
No No No
Delete
Alert
Filter(s)
Yes No Users in
this role
No Users in
this role
No No No
Create
Report
All
accounts
No Designated
accounts
No Designated
accounts
No No No
Download
Reports
All
accounts
No Designated
accounts
No Designated
accounts
No No Designated
accounts
Delete
Reports
All
accounts
No Designated
accounts
No Designated
accounts
No No No
View
Alert
Rules
All
accounts
No Designated
accounts
No Designated
accounts
No No Designated
accounts
Create /
Edit /
Delete /
Disable
Alert
Rules
All
accounts
No Designated
accounts
No Designated
accounts
No No No
View
Notification
Templates
Yes No Yes No Yes No No Yes
Create /
Edit /
Delete
Notification
Templates
Yes No No No Yes No No No Compute
Role
Sys
Admin
System
Admin
(Only
allow
compute
access)
Auditor Defender
Manager
Auditor DevOps CI DevSecOps
Prisma Cloud
Role
System
Admin
System
Admin
with
Compute
Access
Only
Account
Group
Admin
Cloud
Provisioning
Admin
Account
and Cloud
Provisioning
Admin
Build and
Deploy
Security
Build and
Deploy
Security
Account
Group
Read
Only
Compute Yes Yes Yes -
Auditor
Yes -
Defender
Manager
Yes -
Auditor
Yes -
DevOps
No
Access
to the APIs for running
IDE, SCM, and CI
plugins
for IaC
and Vuln
scanning
YesDevSecOps
User
Radar Yes Yes Yes
readonly
access
to data relevant
to the account
in
account
group
No Yes
readonly
access
to data relevant
to
account
in
account
group
No No Yes
Defend Yes Yes Yes
readonly
access
to all
data No Yes
readonly
access
to all
data No No No
Monitor Yes Yes Yes
readonly
access
to data relevant
to
account
in
No Yes
readonly
access
to data relevant
to
account
in
Monitor
Vulnerabilities/
Compliance
but only
CI tab
under
Images/
Functions
No Yes Compute
Role
Sys
Admin
System
Admin
(Only
allow
compute
access)
Auditor Defender
Manager
Auditor DevOps CI DevSecOps
Prisma Cloud
Role
System
Admin
System
Admin
with
Compute
Access
Only
Account
Group
Admin
Cloud
Provisioning
Admin
Account
and Cloud
Provisioning
Admin
Build and
Deploy
Security
Build and
Deploy
Security
Account
Group
Read
Only
account
group
account
group
Manage Yes Yes View All
Logs, Defenders
-
Manage
deployed
to
account
group, Alerts - View, Collections
and Tags
- Read
Only, Authentication
- Read
Only, System -
Downloads
- Jenkins
Plugin
and
twistcli
Defenders
-
Manage
current
defenders
and
deploy
new
ones, Authentication
- view
user
certificates, System
- Access
to all
downloads
View All
Logs, Defenders
-
Manage
deployed
to
account
group, Alerts - View, Collections
and Tags
- Read
Only, Authentication
- Read
Only, System -
Downloads
- Jenkins
Plugin
and
twistcli, path to
console
Authentication
- view
user
certificates, System -
Download
Jenkins
Plugin
and
twistcli, path to
console
No Yes
Settings
View
Accounts
All
accounts
No Designated
accounts
Designated
accounts
Designated
accounts
No No Designated
accounts
View
Account
Details
Yes No No Yes Yes No No No
Create /
Edit /
Delete /
Yes No No Yes Yes No No No Compute
Role
Sys
Admin
System
Admin
(Only
allow
compute
access)
Auditor Defender
Manager
Auditor DevOps CI DevSecOps
Prisma Cloud
Role
System
Admin
System
Admin
with
Compute
Access
Only
Account
Group
Admin
Cloud
Provisioning
Admin
Account
and Cloud
Provisioning
Admin
Build and
Deploy
Security
Build and
Deploy
Security
Account
Group
Read
Only
Disable
Accounts
View
Account
Groups
All
accounts
No Designated
accounts
No Designated
accounts
No No Designated
accounts
Create /
Edit /
Delete
Account
Groups
Yes No No Yes Yes No No No
Create /
View /
Edit /
Delete
User
Roles
Yes Readonly
access
Readonly
access
Readonly
access
Readonly
access
Readonly
access
Readonly
access
Readonly
access
Create /
View /
Edit /
Delete /
Disable
Users Yes Readonly
access
to view
the roles
assigned
for self
Readonly
access
to view
the roles
assigned
for self
Readonly
access
to view
the roles
assigned
for self
Readonly
access
to view
the roles
assigned
for self
Readonly
access
to view
the roles
assigned
for self
Readonly
access
to view
the roles
assigned
for self
Readonly
access
to view
the roles
assigned
for self
Add/
Activate/
Deactivate/
Delete
Access
Keys
Yes; Can
manage
access
keys for other
roles
also.
Yes
Can
manage
access
keys for self
Yes
Can
manage
access
keys for self
Yes
Can
manage
access
keys for self
Yes
Can
manage
access
keys for self
Yes
Can
manage
one
access
key for self
Yes
Can
manage
one
access
key for self
Yes
Can
manage
one
access
key for self
View /
Edit SSO
Settings
Yes No No No No No No No Compute
Role
Sys
Admin
System
Admin
(Only
allow
compute
access)
Auditor Defender
Manager
Auditor DevOps CI DevSecOps
Prisma Cloud
Role
System
Admin
System
Admin
with
Compute
Access
Only
Account
Group
Admin
Cloud
Provisioning
Admin
Account
and Cloud
Provisioning
Admin
Build and
Deploy
Security
Build and
Deploy
Security
Account
Group
Read
Only
Create /
View /
Edit /
Delete /
Disable
Integrations
Yes No No No No No No No
View/
Edit
Trusted
IP
Addresses
Yes No No No No No No No
View
Licensing
Info
Yes No No No No No No No
View
Prisma Cloud
Audit
Logs
Yes No No No No No No No
View/
Edit
AnomalySettings
Yes No Yes No Yes No No No
View/
Edit
Enterprise
Settings
Yes No No No No No No No Manage Roles in Prisma Cloud
Use roles to define the permissions for a specific account group.
- STEP 1 | To view roles, select Settings > Roles.
- STEP 2 | To edit the details of a role, click the record and change any details.
- STEP 3 | To clone a role, hover over the role and click Clone.
Cloning a role is creating a copy of an existing role and then updating it meet requirements quickly.
Only the System Admin role can clone a role.
- STEP 4 | To delete a role, hover over the role and click Delete. Add Administrative Users On Prisma Cloud
To provide administrative access to Prisma Cloud—admin and API—must add users locally on Prisma Cloud. choose whether want these administrators to use Palo Alto Networks Customer
Support Portal (CSP) credentials to log in or SSO using a third-party Identity Service Provider.
If want to use Palo Alto Networks Customer Support Portal (CSP) credentials, when add the email address for a user who already has a support account with Palo Alto Networks, they can just log
in to Prisma Cloud using the Prisma Cloud URL or from the Prisma Cloud tile on hub.If Set up SSO
Integration on Prisma Cloud with an Identity Service Provider who supports SAML, configure Justin-Time Provisioning (JIT) to create a local account on the fly, instead of creating the account in advance on
Prisma Cloud. With JIT, do not need to manually create a local user account.
The following instructions are for manually adding a local user account on Prisma Cloud.
- STEP 1 | Select Settings > Users and click + Add New.
- STEP 2 | Enter First Name, Last Name, and Email of the user.
For a user who has a Palo Alto Networks CSP account, must enter the email address that is
associated with the CSP account so that they can log in as soon as save the changes. If the user does
not have a CSP account, as soon as add them here and save changes, they will receive two
emails. One email to activate the CSP account and another email with a link to get started with Prisma Cloud.
- STEP 3 | Assign Roles to the user.
assign up to five roles to a user, and must select one as the Default Role. See Prisma Cloud
Administrator Roles for the different permission groups and associated permissions. Users with multiple
roles can use the Profile to switch between roles. The default role is marked with a star.
The role assumed by the user is tied to policies, saved searches, saved alert filters, and recurring
compliance reports that do not have a cloud account selected. These objects are available to any other
user who has the same role, and it is not tied to the specific user.
- STEP 4 | Specify a Time Zone for the user and click Save.
- STEP 5 | Decide whether to Allow user to create API Access Keys.
By default, API access is enabled for the System Admin role only. When add a new administrator, decide whether or not want to enable API access for the other roles; the key icon in the API Access column indicates that the administrator has API access, and can create up to two access keys per role on
Prisma Cloud. See Create and Manage Access Keys for more information.
- STEP 6 | After add an administrator, edit or delete the user or modify permissions to add
additional roles.
When delete an administrator or modify the role, all the access keys associated with the user and
role are deleted immediately.
- To edit the details of an user, click the record and change the details.
- To disable an user, toggle the Status of the user.
- To delete an user, hover over the user and click Delete.
- STEP 7 | Change the password for an administrative user.
If want to set a new password to periodically change it or if are unable to log in because forgot password. As a security measure, if enter an incorrect password five times, account
is locked and must reset password.
1. Access the URL for Prisma Cloud instance.
2. Click the Forgot password link.
will receive an email at the registered email address (Step 2 above). Use the link in the email to
set a new password. Create and Manage Access Keys
Access Keys are a secure way to enable programmatic access to the Prisma Cloud API, if are setting up
an external integration or automation. By default, only the System Admin has API access and can enable API
access for other administrators.
enable API access either when Add Administrative Users On Prisma Cloud, modify the user permissions to enable API access. If have API access, create up to two access keys per role
for most roles; some roles such the Build and Deploy Security role can generate one access key only. When
create an access key, the key is tied to the role with which logged in.
Create an access key for a limited time period and regenerate API keys periodically to minimize
exposure and follow security best practices. On the Settings > Audit Logs, view a record of all
access key related activities such as an update to extend its validity, deletion, or a revocation.
Watch this!
- STEP 1 | Select Settings > Access Keys > + Add New
If do not see the option to add a new key, it means that do not have the permissions to create
access keys.
- STEP 2 | Enter a descriptive Name for the key.
- STEP 3 | Set the Key Expiry term.
Select the checkbox and specify a term—date and time for the key validity—that adheres
to corporate compliance standards. If do not select key expiry, the key is set to
never expire; if select it, but do not specify a date, the key expires in a month. In the event a key is compromised, administratively disable (Make Inactive) the key.
- STEP 4 | Create the key.
If have multiple roles, must switch roles to create an access key for each role. Copy or download the Access Key ID and the Secret Key as a CSV file. After close the window, cannot view the secret key again, and must delete the existing key and create a new key.
- STEP 5 | View the details for keys. verify the expiry date for the key and can update it here, review when it was last used and the status —Active or Expired.
If have multiple roles, the access key details display only for the role with which are logged in. Manage Prisma Cloud Profile
Manage Prisma Cloud profile.
- STEP 1 | To view profile information, click the icon on the Right hand top corner.
- STEP 2 | Edit Name, Last Name, or Time zone and click Save. Set up SSO Integration on Prisma Cloud
On Prisma Cloud, enable single sign-on (SSO) using an Identity Provider (IdP) that supports Security
Assertion Markup Language (SAML), such as Okta, Azure Active Directory, or PingID. configure
only one IdP for all the cloud accounts that Prisma Cloud monitors.
To access Prisma Cloud using SSO, every administrative user requires a local account on Prisma Cloud. either Add Administrative Users On Prisma Cloud to create the local account in advance of enabling
SSO, or use Just-In-Time (JIT) Provisioning on the SSO configuration on Prisma Cloud if prefer to
create the local account automatically. With JIT Provisioning, the first time a user logs in and successfully
authenticates with SSO IdP, the SAML assertions are used to create a local user account on Prisma Cloud.
To enable SSO, must first complete the setup on the IdP. Then, log in to Prisma Cloud using an account
with System Admin privileges to configure SSO and redirect login requests to the IdP’s login page, so that
Prisma Cloud administrative users can log in using SSO. After enable SSO, must access Prisma Cloud from the IdP’s portal. Prisma Cloud supports IdP initiated SSO, and it’s SAML endpoint supports the POST method only.
As a best practice, enable a couple administrative users with both local authentication credentials on
Prisma Cloud and SSO access so that they can log in to the administrative console and modify the SSO
configuration when needed, without risk of account lockout. Make sure that each administrator has activated their Palo Alto Networks Customer Support Portal (CSP) account using the Welcome to Palo Alto
Networks Support email and set a password to access the portal.
Also, any administrator who needs to access the Prisma Cloud API cannot use SSO and must authenticate
directly to Prisma Cloud using the email address and password registered with Prisma Cloud.
- STEP 1 | Decide whether want to first add Add Administrative Users On Prisma Cloud or prefer
to add users on the fly with JIT Provisioning when Configure SSO on Prisma Cloud.
If want to enable JIT provisioning for users, Create Prisma Cloud Roles before continue to the next step. When configure SSO on the IdP, must attach this role to the user‘s profile so that the user has the appropriate permissions and can monitor the assigned cloud accounts on Prisma Cloud.
- STEP 2 | Copy the Audience URI, for Prisma Cloud, which users need to access from the IdP.
1. Log in to Prisma Cloud and select Settings > SSO.
2. Copy the Audience URI (SP Entity ID) value. This is a read-only field in the format: https://
app.prismacloud.io?customer=<string> to uniquely identify instance of Prisma Cloud. require this value when configure SAML on IdP.
- STEP 3 | Set up the Identity Provider for SSO.
1. This workflow uses Okta as the IdP. Before begin to set up Okta configuration, login to your
Prisma Cloud instance and copy the Audience URI (SP Entity ID) from Prisma Cloud. See For example:
https://app.prismacloud.io/settings/sso.
2. Login to Okta as an Administrator and click Admin. 3. Click Add Applications.
4. Search for Prisma Cloud and Add.
5. On Create a New Application Integration, select Web for Platform and SAML 2.0 for Sign on
method. 6. Click Create.
7. On General Settings, use these values and click Next.
App Name - Prisma Cloud SSO app
App Logo - Use the Prisma Cloud logo
App Visibility - Do not check these options
8. To Configure SAML, specify the Sign On URL.
The format for Sign On URL uses the URL for Prisma Cloud, but must replace app with api
and add saml at the end. For example, if access Prisma Cloud at https://app2.prismacloud.io, 180 Sign On URL should be https://api2.prismacloud.io/saml and if it is https://
app.eu.prismacloud.io, it should be https://api.eu.prismacloud.io/saml.
9. For Audience URI - Use the value displayed on Prisma Cloud Settings > SSO that copied in the first step.
10.Select Name ID format as EmailAddress and Application username as Email.
11.For Advanced Section, select Response as Unsigned, Assertion Signature as Signed, Assertion
Encryption as UnEncrypted. 12.Assign users who can use the Prisma Cloud SSO app to log in to Prisma Cloud.
13.(Required only for JIT provisioning of a local user account automatically on Prisma Cloud) Specify the attributes to send with the SAML assertion.
For more details, see Set up Just-in-Time Provisioning on Okta.
14.(Required only for JIT provisioning of a local user account automatically on Prisma Cloud) Assign the role created on Prisma Cloud to the user profile.
have now successfully created an application for the SAML integration. This application will have
the details of the IdP URL and Certificate which you’ll need to add on Prisma Cloud to complete the SSO integration.
- STEP 4 | Configure SSO on Prisma Cloud.
1. Log in to Prisma Cloud and select Settings > SSO.
2. Enable SSO.
3. Enter the value for Identity Provider Issuer. This is the URL of a trusted provider such as Google, Salesforce, Okta, or Ping who act as IdP
in the authentication flow. On Okta, for example, find the Identity Provider issuer URL at
Applications > Sign On > View Setup Instructions.
In the setup instructions, have Identity Provider Issuer and Prisma Cloud Access SAML URL.
4. Enter the Identity Provider Logout URL to which a user is redirected to, when Prisma Cloud times
out or when the user logs out.
5. Enter IdP Certificate in the standard X.509 format.
must copy and paste this from IdP. 6. Enter the Prisma Cloud Access SAML URL configured in IdP settings.
For example, on Okta this is the Identity Provider Single Sign-On URL. When click this URL, after
authentication with IdP, are redirected to Prisma Cloud. This link along with the Relay State
Parameter is used for all the redirection links embedded in notifications like email, slack, SQS, and
compliance reports.
7. Relay State Param name is SAML specific Relay State parameter name. If provide this parameter
along with Prisma Cloud Access SAML URL, all notification links in Splunk, Slack, SQS, email, and
reports can link directly to the Prisma Cloud application. The relay state parameter or value is specific
to Identity Provider. For example, this value is RelayState for Okta.
When using RelayState functionality, make sure Prisma Cloud Access SAML
URL corresponds to Identity Provider Single Sign-On URL ending in ‘/sso/saml’.
8. (Optional) Clear the Enforce DNS Resolution for Prisma Cloud Access SAML URL.
By default, Prisma Cloud performs a DNS look up to resolve the Prisma Cloud SAML Access URL entered earlier. If IdP is on internal network, and do not need to perform a DNS look
up, clear this option to bypass the DNS lookup.
9. (Optional) Enable Just-in-Time Provisioning for SSO users.
Enable JIT Provisioning, if want to create a local account for users who are authenticated by the IdP. With JIT, the user is provisioned with the first five roles mapped to the user’s profile on the IdP.
10.Provide the user attributes in the SAML assertion or claim that Prisma Cloud can use to create the local user account.
must provide the email, role, first name, and last name for each user. Timezone is optional. The role that specify for the user’s profile on the IdP must match what created
on Prisma Cloud in Step 1.
11.Select Allow select users to authenticate directly with Prisma Cloud to configure some users to
access Prisma Cloud directly using their email address and password registered with Prisma Cloud, in
addition to logging in via the SSO provider.
When enable SSO, make sure to select a few users who can also access Prisma Cloud directly
using the email and password that is registered locally on Prisma Cloud to ensure that are not
locked out of the console in the event have misconfigured SSO and need to modify the IdP
settings. For accessing data through APIs, need to authenticate directly to Prisma Cloud.
12.Select the Users who can access Prisma Cloud either using local authentication credentials on Prisma Cloud or using SSO.
The users listed in the allow list can log in using SSO and also using a local account username and
password that have created on Prisma Cloud.
13.Save changes.
14.Verify access using SSO.
Administrative users for whom have enabled SSO, must access Prisma Cloud from the Identity
Provider’s portal. For example, if have integrated Prisma Cloud with Okta, administrative users must login to Okta and then click on the Prisma Cloud app icon to be logged in to Prisma Cloud.
15.Using View last SSO login failures, see details of last five login issues or errors for SSO
authentication for any users. If the user is logged in already using a username/password and then logs in using SSO, the authentication token in the browser's local storage is replaced with the latest token.
Set up Just-in-Time Provisioning on Okta
To successfully set up local administrators on the fly with Just-in-Time (JIT) provisioning, need to
configure the Prisma Cloud app for Okta to provide the SAML claims or assertions that enable Prisma Cloud
to add the authenticated SSO user on Prisma Cloud. Then, to ensure that the SSO user has the correct
access privileges on Prisma Cloud, need to assign a Prisma Cloud role to the user; if this role is not a default role on Prisma Cloud, must define the role before assign the role to the user on Okta.
- STEP 1 | Create the Prisma Cloud App for Okta.
- STEP 2 | For JIT, create a custom attribute on the Prisma Cloud Okta app.
1. Go to Directory > Profile Editor > Apps.
2. Find the Prisma Cloud app and select Profile, and Add Attribute. Enter a display name, a variable name, and an attribute length that is long enough to accommodate
the role names on Prisma Cloud.
- STEP 3 | Configure the Attribute Statements on the Prisma Cloud Okta app.
Specify the user attributes in the SAML assertion or claim that Prisma Cloud can use to create the local
user account.
1. Select Applications > Applications
2. Select the Prisma Cloud app, General and edit the SAML Settings to add the attribute statements.
must provide the email, role, first name, and last name for each user. Timezone is optional.
- STEP 4 | Assign the Prisma Cloud role for each SSO user.
Each SSO user who is granted access to Prisma Cloud, can have between one to five Prisma Cloud roles
assigned. Each role determines the permissions and account groups that the user can access on Prisma Cloud.
1. Select Applications > Applications
2. Select the Prisma Cloud app and Assignments. For existing users, click the pencil icon to add the Prisma Cloud Role want to give this user. For example, System Admin.
For new users, select Assign > Assign to People, click Assign for the user want to give access to
Prisma Cloud and define the Prisma Cloud Role want to give this user.
- STEP 5 | Continue with 4. View Audit Logs
As part of compliance requirement for organizations, companies need to demonstrate they are pro-actively
tracking security issues and taking steps to remediate issues as they occur. Prisma Cloud Audit Logs section
enables companies to prepare for such audits and demonstrates compliance. The Audit logs section lists out
the actions performed by the users in the system.
- STEP 1 | Select a Time Range to view the activity details by users in the system.
- STEP 2 | Select the columns in the table and Download all administrator activity.
The user activity details are in a CSV format. Define Prisma Cloud Enterprise and Anomaly
Settings
Set the enterprise level settings to build standard training models for anomaly detection, alert disposition, and some other global settings such as the timeout before the user is looked out for inactivity, and user
attribution for alerts.
- Set Up Inactivity Timeout
- Set Up Global Settings for Policy and Alerts -Set Up Anomaly Policy Thresholds
Set Up Inactivity Timeout
Specify a timeout period after which an inactive administrative user will be automatically logged out of Prisma Cloud. An inactive user is one who does not interact with the UI using their keyboard and mouse
within the specified time period.
- STEP 1 | Select Settings > Enterprise Settings.
- STEP 2 | User Idle Timeout
If modify the timeout period, the new value is in effect for all administrative users who log in after
make the change; the previous timeout applies for all currently logged in users.
Set Up Global Settings for Policy and Alerts These settings apply to all Prisma Cloud policies. For Anomaly policies, have more customizable
settings, see Set Up Anomaly Policy Thresholds.
- Auto enable new default policies of the type.
1. Select Settings > Enterprise Settings.
2. Granularly enable new Default policies of severity High, Medium or Low.
While some high severity policies are enabled to provide the best security outcomes, by default, policies of medium or low severity are in a disabled state. When select the checkbox to auto
enable policies of a specific severity, either retroactively enable all policies that match the severity or only enable policies that are added to Prisma Cloud going forward.
If enable policies of a specific severity, when then clear the checkbox, the policies that were enabled previously are not disabled; going forward, policies that
match the severity cleared are no longer enabled to scan cloud resources
and generate alerts.
If want to disable the policies that are currently active, must disable the status
of each policy on the Policies page. -Enable Make Alert Dismissal Note Mandatory, to mandate the users to dismiss alerts only
after specifying a reason.
- Enable Populate User Attribution in Alerts Notifications
User attribution data provides with context on who created or modified the resource that triggered
the alert. Select this option to make sure that the alerts include user attribution data in the alert payload, so that it is sent as part of the JSON data to notification channels such as SQS or Splunk. Enabling this
option can result in a delay of up to two hours in the generation of alerts because the relevant user
information may not be instantly available from the cloud provider.
Set Up Anomaly Policy Thresholds
Prisma Cloud allows to define different thresholds for anomaly detection for Unusual Entity Behavior
Analysis (UEBA) that correspond to policies which analyze audit events, and for unusual network activity
that correspond to policies which analyze network flow logs. also define preference for when
want to alert notifications based on the severity assigned to the anomaly policy.
If want to exclude one or more IP addresses or a CIDR block from generating alerts against Anomaly
policies, see Trusted IP Addresses on Prisma Cloud.
- For UEBA policies:
1. Select Settings > Anomaly Settings > Alerts and Thresholds.
2. Select a policy.
3. Define the Training Model Threshold.
The Training Model Threshold informs Prisma Cloud on the values to use for setting the baseline for the machine learning (ML) models.
For production environments, set the Training Model Threshold to High so that allow for more time and have more data to analyze for determining the baseline.
For unusual user activity:
1. Low: The behavioral models are based on observing at least 25 events over 7 days.
2. Medium: The behavioral models are based on observing at least 100 events over 30 days.
3. High: The behavioral models are based on observing at least 300 events over 90 days.
For account hijacking:
1. Low: The behavioral models are based on observing at least 10 events over 7 days.
2. Medium: The behavioral models are based on observing at least 25 events over 15 days.
3. High: The behavioral models are based on observing at least 50 events over 30 days.
4. Define Alert Disposition. Alert Disposition is preference on when want to be notified of an alert, based on the severity of the issue —low, medium, high. The alert severity is based on the severity associated with
the policy that triggers an alert.
profile every activity by location or user activity. The activity-based anomalies identify any
activities which have not been consistently performed in the past. The location based anomalies
identify locations from which activities have not been performed in the past.
Choose the disposition (in some cases may only have two to choose from):
1. Conservative:
For unusual user activity—Report on unknown location and service to classify an anomaly.
For account hijacking—Reports on location and activity to login under travel conditions that are
not possible, such as logging in from India and US within 8 hours.
2. Moderate:
For unusual user activity—Report on unknown location, or both unknown location and service to
classify an anomaly.
3. Aggressive:
For unusual user activity—Report on either unknown location or service to classify an anomaly.
For account hijacking—Report on unknown browser and Operating System, or impossible time
travel.
Set the Alert Disposition to Conservative to reduce false positives.
When change Training Model Threshold or Alert Disposition the existing alerts are resolved and new ones are regenerated based on the new setting. It might take a while for the new anomaly alerts to show on the Alerts page.
- For unusual network activity.
For anomalies policies that help detect network incidents, such as unusual protocols or port used to
access a server on network, customize the following for each policy.
1. Select Settings > Anomaly Settings > Alerts and Thresholds.
2. Select a policy.
3. Define the Training Model Threshold. The Training Model Threshold informs Prisma Cloud on the values to use for various parameters such
as number of days and packets for creating the ML models. These thresholds are available only for the policies that require model building such as Unusual server port activity and Spambot activity.
1. Low: The behavioral models are based on observing at least 10K packets over 7 days.
2. Medium: The behavioral models are based on observing at least 100k packets over 14 days.
3. High: The behavioral models are based on observing at least 1M packets over 28 days.
4. Define Alert Disposition.
Alert Disposition is preference on when want to be notified of an alert, based on the severity of the issue —low, medium, high. The alert severity is based on the severity associated with
the policy that triggers an alert. choose from three dispositions based on the number of ports, hosts or the volume of traffic generated to a port or host on a resource:
1. Aggressive: Reports High, Medium, and Low severity alerts.
For example, a Spambot policy that sees 250MB traffic to a resource, or a port sweep policy that
scans 10 hosts.
2. Moderate: Reports High and Medium severity alerts.
For example, a Spambot policy that sees 500MB traffic to a resource, or a port sweep policy that
scans 25 hosts.
3. Conservative: Report on High severity alerts only.
For example, a Spambot policy that sees 1GB traffic to a resource, or a port sweep policy that
scans 40 hosts.
193
Manage Prisma Cloud Alerts > Prisma Cloud Alerts and Notifications
> Trusted IP Addresses on Prisma Cloud
> Enable Prisma Cloud Alerts > Create an Alert Rule
> Configure Prisma Cloud to Automatically Remediate Alerts > Send Prisma Cloud Alert Notifications to Third-Party Tools
> View and Respond to Prisma Cloud Alerts > Generate Reports on Prisma Cloud Alerts > Alert Payload
> Prisma Cloud Alert Resolution Reasons Prisma Cloud Alerts and Notifications
Prisma™ Cloud continually monitors all of cloud environments to detect misconfigurations (such as exposed cloud storage instances), advanced network threats (such as cryptojacking and data exfiltration), potentially compromised accounts (such as stolen access keys), and vulnerable hosts. Prisma Cloud
then correlates configuration data with user behavior and network traffic to provide context around
misconfigurations and threats in the form of actionable alerts.
Although Prisma Cloud begins monitoring and correlating data as soon as onboard the cloud account, there are tasks need to perform before see alerts generated by policy violations in cloud
environments. The first task to Enable Prisma Cloud Alerts is to add the cloud account to an account group
during onboarding. Next, create an alert rule that associates all of the cloud accounts in an account group
with the set of policies for which want Prisma Cloud to generate alerts. view the alerts for all
of cloud environments directly from Prisma Cloud and drill down in to each to view specific policy violations. If have internal networks that want to exclude from being flagged in an alert, can
Trusted IP Addresses on Prisma Cloud.
In addition, Prisma Cloud provides out-of-box ability to Configure External Integrations on Prisma Cloud with third-party technologies, such as SIEM platforms, ticketing systems, messaging systems, and automation frameworks so that continue using existing operational, escalation, and
notification tools. To monitor cloud infrastructures more efficiently and provide visibility in to
actionable events across all cloud workloads, also:
- Generate Reports on Prisma Cloud Alerts —on-demand or scheduled reports— on open alerts and email
them to stakeholders.
- Send the Alert Payload to a third-party tool. Trusted IP Addresses on Prisma Cloud
Prisma™ Cloud enables to specify IP addresses or CIDR ranges for:
- Trusted Login IP Addresses—Restrict access to the Prisma Cloud administrator console and API to only
the specified source IP addresses.
- Trusted Alert IP Addresses—If have internal networks that connect to public cloud
infrastructure, add these IP address ranges (or CIDR blocks) as trusted on Prisma Cloud. When
add IP addresses to this list, create a label to identify internal networks that are not
in the private IP address space to make alert analysis easier. When visualize network traffic on the Prisma Cloud Investigate tab, instead of flagging internal IP addresses as internet or external IP
addresses, the service can identify these networks with the labels provide.
Prisma Cloud default network policies that look for internet exposed instances also do not generate
alerts when the source IP address is included in the trusted IP address list and the account hijacking
anomaly policy filters out activities from known IP addresses. Also, when use RQL to query network
traffic, filter out traffic from known networks that are included in the trusted IP address list.
- Anomaly Trusted List—Exclude trusted IP addresses when conducting tests for PCI compliance or
penetration testing on network. Any addresses included in this list do not generate alerts against
the Prisma Cloud Anomaly Policies that detect unusual network activity such as the policies that detect
internal port scan and port sweep activity, which are enabled by default.
To add an IP address to the trusted list:
- Add an Alert IP address.
1. Select Settings > Trusted Alert IP Addresses > + Add New
must have the System Administrator role on Prisma Cloud to view or edit the Trusted IP
Addresses page. See Prisma Cloud Administrator Permissions.
2. Enter a name or label for the Network.
3. Enter the CIDR and, optionally, add a Description and then click Done.
Enter the CIDR block for IP addresses that are routable through the public internet, cannot add a private CIDR block. The IP addresses enter may take up to 15 minutes to take effect.
The trusted IP addresses are appropriately classified when run a network query.
- Add a Login IP address.
1. Select Settings > Trusted Login IP Addresses > + Add New.
must have the System Administrator role on Prisma Cloud to view or edit the Trusted IP
Addresses page. See Prisma Cloud Administrator Permissions.
2. Enter a Name and, optionally a Description.
3. Enter the CIDR and Create the new login IP address entry.
4. Verify that the IP address are logged in with is included in the list.
If are logged in from an IP address that is not listed as a trusted IP address, will be logged out as soon as save changes and can no longer access the Prisma Cloud administrator console and API interface.
5. Enable the IP address.
- Add an IP Address to the Anomaly Trusted List.
1. Select Settings > Anomaly Settings must have the correct role, such as the System Administrator role, on Prisma Cloud to view or
edit the Anomaly Settings page. See Prisma Cloud Administrator Permissions for the roles that have
access.
2. Add New > IP Address.
3. Enter a Trusted List Name and, optionally a Description.
4. Select the Anomaly Policies for which do not want to generate alerts.
5. Enter the IP Addresses.
enter one or more IP addresses in the CIDR format. By default, the IP addresses add
to the trusted list are excluded from generating alerts against any (all) cloud accounts that are
onboarded to Prisma Cloud.
6. (Optional) Toggle Hide Advanced Settings to select an Account ID and VPC ID.
select only one Account and VPC ID, or set it to Any to exclude any account that is added to
Prisma Cloud.
7. Save the list.
When save the list, for the selected anomaly policies that detect network issues such as network
reconnaissance, network evasion, or resource misuse, Prisma Cloud will not generate alerts for the IP
addresses included in this list.
Only the administrator who created the list can modify the name, description, Account
ID and VPC ID; Other administrators with the correct role can add or delete IP address
entries on the trusted list. Enable Prisma Cloud Alerts Although Prisma™ Cloud begins monitoring cloud environments as soon as onboard a cloud
account, must first enable alerting for each cloud account onboard before receive alerts.
Prisma Cloud gives the flexibility to group cloud accounts into account groups so that can
restrict access to information about specific cloud accounts to only those administrators who need it.
Then must assign each account group to an alert rule that allows to select a group of policies
and designate where want to display the Prisma Cloud Alerts and Notifications associated with
those policies. This enables to define different alert rules and notification flows for different cloud
environments, such as for both a production and a development cloud environment. In addition, can
set up different alert rules for sending specific alerts to existing SOC visibility tools. For example, could send one set of alerts to security information and event management (SIEM) system and another
set to Jira for automated ticketing.
- STEP 1 | Make sure have associated all onboarded cloud accounts to an account group.
If did not associate a cloud account with an account group during the onboarding process, do it now
so that see alerts associated with the account.
1. Click Settings ( ) and then select Cloud Accounts.
2. For each cloud account, verify that there is a value in the Account Groups column.
3. For any cloud account that isn’t yet assigned to an account group, select the cloud account to edit it
and select an Account Group to which to add it.
- STEP 2 | Create an Alert Rule.
Alert rules define what policy violations trigger alerts for cloud accounts within the selected account
group and where to send the alert notifications.
- STEP 3 | Verify that the alert rule created is triggering alert notifications.
As soon as save alert rule, any violation of a policy for which enabled alerts results in an
alert notification on the Alerts page, as well as in any third-party integrations designated in the alert
rule. Make sure see the alerts are expecting on the Alerts page as well as in third-party
tools. Create an Alert Rule
Alert rules enable to define the policy violations in a selected set of cloud accounts for which want
to trigger alerts. When create an alert rule, select the account groups to which the rule applies and
the corresponding set of policies for which want to trigger alerts. add more granularity to the rule by excluding some cloud accounts from the selected account groups, by specifying specific regions
for which to send alerts, and even by narrowing down the rule to specific cloud resources identified by
resource tags. This provides with flexibility in how manage alerts and ensures that adhere
to the administrative boundaries defined. create a single alert rule that alerts on all policy rules or define granular alert rules that send very specific sets of alerts for specific cloud accounts, regions, and even resources to specific destinations.
When create an alert rule, Configure Prisma Cloud to Automatically Remediate Alerts, which
enables Prisma Cloud to automatically run the CLI command required to remediate the policy violation
directly in cloud environments. Automated remediation is only available for default policies (Config
policies only) that are designated as Remediable ( ) on the Policies page.
In addition, if Configure External Integrations on Prisma Cloud with third-party tools, defining granular
alert rules enables to send only the alerts need to enhance existing operational, ticketing, notification, and escalation workflows with the addition of Prisma Cloud alerts on policy violations in all
cloud environments. To see any existing integrations, click Settings ( ) and then select Integrations.
- STEP 1 | Select Alerts > Alert Rules and +Add New alert.
- STEP 2 | Enter an Alert Rule Name and, optionally, a Description to communicate the purpose of the rule and then click Next.
- STEP 3 | Select the Account Groups to which want this alert rule to apply and then click Next.
1. Toggle View Advanced Settings to see advanced settings for setting a target.
2. Exclude Cloud Accounts from selected Account Group.
3. Choose Region.
4. Add Tags to easily manage or identify the type of resources.
Tags apply only to Config and Network policies.
5. Click Next.
- STEP 4 | (Optional) If want to add more granularity for which cloud resources trigger alerts for this
alert rule, View Advanced Settings and then provide more criteria as needed:
- Exclude Cloud Accounts—If there are some cloud accounts in the selected account groups for which
do not want to trigger alerts, select the accounts from the list.
- Regions—To trigger alerts only for specific regions for the cloud accounts in the selected account
group, select one or more regions from the list.
- Resource Tags—To trigger alerts only for specific resources in the selected cloud accounts, enter the Key and Value of the resource tag created for the resource in cloud environment. Tags apply only to Config and Network policies. When add multiple resource tags, it uses the boolean logical OR operator.
When finish defining the target cloud resources, click Next.
- STEP 5 | Select the policies for which want this alert rule to trigger alerts and, optionally, Configure
Prisma Cloud to Automatically Remediate Alerts. 1. Either Select All Policies or select the specific policies for which want to trigger alerts on this
alert rule.
If enable Automated Remediation, the list of policies shows only Remediable ( )
policies
.
To help find the specific group of policies for which want this rule to alert
- Filter Results—Enter a search term to filter the list of policies to those with specific keywords.
•
Column Picker—Click Edit ( ) to modify which columns to display.
- Sort—Click the corresponding Sort icon ( ) to sort on a specific column.
- Column Filter—Click the corresponding column Filter icon ( ) to filter on a specific value in a column. For example, to filter on compliance standards related to NIST, click the filter for the Compliance Standard column, select NIST standards, and then Set that filter. 2. Click Next.
- STEP 6 | (Optional) Send Prisma Cloud Alert Notifications to Third-Party Tools.
By default, all alerts triggered by the alert rule display on the Alerts page. If Configure External
Integrations on Prisma Cloud, also send Prisma Cloud alerts triggered by this alert rule to thirdparty tools. For example, Send Alert Notifications to Amazon SQS or Send Alert Notifications to
Jira.
In addition, configure the alert rule to Send Alert Notifications Through Email.
- STEP 7 | (Optional) If want to delay the alert notifications for Config alerts, configure the Prisma Cloud to Trigger notification for Config Alert only after the Alert is Open for a specific
number of minutes.
- STEP 8 | Save the alert rule.
- STEP 9 | To verify that the alert rule triggers the expected alerts, select Alerts > Overview and ensure
that see the alerts that expect to see there.
If configured the rule to Send Prisma Cloud Alert Notifications to Third-Party Tools, make sure also see the alert notifications in those tools. Configure Prisma Cloud to Automatically
Remediate Alerts If want Prisma™ Cloud to automatically resolve policy violations, such as misconfigured security
groups, configure Prisma Cloud for automated remediation. To automatically resolve a policy violation, Prisma Cloud runs the CLI command associated with the policy in the cloud environments where
it discovered the violation. On Prisma Cloud, enable automated remediation for default policies
(Config policies only) that are designated as remediable (indicated by in the Remediable column) and for any cloned or custom policies that add.
To enable automated remediation, identify the set of policies that want to remediate automatically and
verify that Prisma Cloud has the required permissions in the associated cloud environments. Then Create an
Alert Rule that enables automated remediation for the set of policies identified.
Use caution when enable automated remediation because it requires Prisma Cloud to
make changes in cloud environments that can adversely affect applications.
- STEP 1 | Verify that Prisma Cloud has the required privileges to remediate the policies plan to
configure for automated remediation.
1. To view remediable policies, select Policies and set the filter to Remediable > True.
If the Remediable column is not displayed on the Policies page, use the Column
Picker ( ) to display it. 2. Select a policy for which want to enable remediation and go to the Remediation page.
Review the required privileges in the CLI Command Description to identify which permissions Prisma Cloud requires in the associated cloud environments to be able to remediate violations of the policy.
define up to 5 CLI commands in a sequence for a multi-step automatic
remediation workflow. Add the commands in the sequence want them to execute
and separate the commands with a semi colon. If any CLI command included in the sequence fails, the execution stops at that point. See list of supported CLI variables. - STEP 2 | Create an Alert Rule or modify an existing alert rule.
- STEP 3 | On the Select Policies page, enable Automated Remediation and then Continue to
acknowledge the impact of automated remediation on application.
The list of available policies updates to show only those policies that are remediable (as indicated by
in the Remediable column).
If are modifying an existing alert rule that includes non-remediable policies, those
policies will no longer be included in the rule. When modify the rule, Prisma Cloud
notifies all account administrators who have access to that rule. - STEP 4 | Finish configuring and Save the new alert rule or Confirm changes to an existing alert rule. Send Prisma Cloud Alert Notifications to ThirdParty Tools
Alert rules define which policy violations trigger an alert in a selected set of cloud accounts. When Create an Alert Rule, also configure the rule to send the Alert Payload that the rule triggers to one
or more third-party tools. For all channels except email, to enable notification of policy violations in your
cloud environments in existing operational workflows, must Configure External Integrations on
Prisma Cloud. either set up an integration before create the alert rule or use the inline link in
the alert rule creation process to set up the integration when need it.
On some integrations, such as Google CSCC, AWS Security Hub, PagerDuty, and ServiceNow, Prisma Cloud can send a state-change notification to resolve an incident when the issue that generated the alert is
resolved manually or if the resource was updated in the cloud environment and the service learns that the violation is fixed.
Refer to the following topics to enable an alert notification channel with third-party tools:
- Amazon SQS
- Azure Service Bus Queue
- Email
- Slack
- Splunk
- Jira
- Google SCC
- ServiceNow
- Webhooks
- PagerDuty
- AWS Security Hub
- Microsoft Teams
- Cortex XSOAR
Send Alert Notifications to Amazon SQS
send Prisma Cloud alert notifications to Amazon Simple Queue Service (SQS).
- STEP 1 | Integrate Prisma Cloud with Amazon SQS.
- STEP 2 | Select Alerts > Alert Rules and either Create an Alert Rule or select an existing rule to edit.
- STEP 3 | On the Set Alert Notification page for the alert rule, select SQS.
- STEP 4 | Select the SQS Queues to which want to send alerts triggered by this alert rule. - STEP 5 | Save the new alert rule or Confirm changes to an existing alert rule.
Send Alert Notifications to Azure Service Bus Queue
send Prisma Cloud alert notifications to an Azure Service Bus queue.
- STEP 1 | Integrate Prisma Cloud with Azure Service Bus Queue.
- STEP 2 | Select Alerts > Alert Rules and either Create an Alert Rule or select an existing rule to edit.
- STEP 3 | On the Set Alert Notification page for the alert rule, select Azure Service Bus Queue.
- STEP 4 | Select the Azure Service Bus Queue to which want to send alerts triggered by this alert
rule.
- STEP 5 | Save the new alert rule or Confirm changes to an existing alert rule. Send Alert Notifications Through Email
To send email notifications for alerts triggered by an alert rule, Prisma Cloud provides a default email
notification template. customize the message in the template using the in-app rich text editor and
attach the template to an alert rule. In the alert notification, configure Prisma Cloud to send the alert details as an uncompressed CSV file or as a compressed zip file, of 9 MB maximum attachment size.
All email notifications from Prisma Cloud include the domain name to support Domain-based
Message Authentication, Reporting & Conformance (DMARC), and the email address used is
noreply@prismacloud.paloaltonetworks.com.
- STEP 1 | (Optional) Set up a custom message for email notification template.
Prisma Cloud provides a default email template for convenience, and customize the lead-in
message within the body of the email using the rich-text editor.
1. Select Alerts > Notification Templates.
2. Add New notification template, and choose Email template.
3. Enter a Template Name.
4. Enter a Custom Note.
The preview on the right gives an idea of how content will look.
5. Save the email notification template.
- STEP 2 | Select Alerts > Alert Rules and either Create an Alert Rule or select an existing rule to edit.
- STEP 3 | On the Set Alert Notification page for the alert rule, select Email.
- STEP 4 | Enter or select the Emails for which to send the alert notifications.
include multiple email addresses and can send email notifications to email addresses in your
domain and to guests external to organization.
- STEP 5 | (Optional) Select custom email Template, if have one.
- STEP 6 | Set the Frequency at which to send email notifications.
- Instantly—Sends an email to the recipient list each time the alert rule triggers an alert. -Recurring—select the time interval as Daily, Weekly or Monthly. Prisma Cloud sends a single
email to the recipient list that lists all alerts triggered by the alert rule on that day, during that week, or the month.
- STEP 7 | Specify whether to include an attachment to the email.
Including an attachment provides a way for to include information on the alerts generated and
the remediation steps required to fix the violating resource. When select Attach detailed report, choose whether to Include remediation instructions to fix the root cause for the policy that
triggered each alert, and opt to send it as a zip file (Compress attachment(s)).
Each email can include up to 10 attachments. An attachment in the zip file format can have 60000
rows, while a CSV file can have 900 rows. If the number of alerts exceed the maximum number of attachments, the alerts with the older timestamps are omitted.
- STEP 8 | Save the new alert rule or Confirm changes to an existing alert rule.
- STEP 9 | Verify the alert notification emails.
The email alert notification specifies the alert rule, account name, cloud type, policies that were violated, the number of alerts each policy violated, and the affected resources. Click the <number> of alerts view
the Prisma Cloud Alerts > Overview page. Send Alert Notifications to a Slack Channel
send alert notifications associated with an alert rule to a Slack channel.
- STEP 1 | Integrate Prisma Cloud with Slack.
- STEP 2 | Select Alerts > Alert Rules and either Create an Alert Rule or select an existing rule to edit.
- STEP 3 | On the Set Alert Notification page for the alert rule, select Slack.
- STEP 4 | Select the Slack Channels to which want to send alerts triggered by this alert rule.
- STEP 5 | Set the Frequency at which to send email notifications.
- As it Happens—Sends a notification to the selected slack channels each time the alert rule triggers an
alert.
- Daily—Sends a single notification to the selected Slack channels once each day that lists all alerts triggered by the alert rule on that day.
- Weekly—Sends a single notification to the selected Slack channels once each week that lists all alerts triggered by the alert rule during that weekly interval.
- Monthly—Sends a single notification to the selected Slack channels once each month that lists all
alerts triggered by the alert rule monthly interval.
- STEP 6 | Save the new alert rule or Confirm changes to an existing alert rule. Send Alert Notifications to Splunk
send alert notifications associated with an alert rule to a Splunk event collector.
- STEP 1 | Integrate Prisma Cloud with Splunk.
- STEP 2 | Select Alerts > Alert Rules and either Create an Alert Rule or select an existing rule to edit.
- STEP 3 | On the Set Alert Notification page for the alert rule, select Splunk.
- STEP 4 | Select the Splunk Event Collectors to which want to send alerts triggered by this alert rule.
- STEP 5 | Save the new alert rule or Confirm changes to an existing alert rule.
Send Alert Notifications to Jira
configure alert notifications triggered by an alert rule to create Jira tickets.
- STEP 1 | Integrate Prisma Cloud with Jira.
- STEP 2 | Select Alerts > Alert Rules and either Create an Alert Rule or select an existing rule to edit.
- STEP 3 | On the Set Alert Notification page for the alert rule, select Jira. - STEP 4 | Select the Jira Templates to use for creating tickets based on the alert payload data for alerts that are triggered by this alert rule.
- STEP 5 | Save the new alert rule or Confirm changes to an existing alert rule.
Send Alert Notifications to Google Cloud SCC
send alert notifications to Google Cloud Security Command Center (SCC).
- STEP 1 | Integrate Prisma Cloud with Google Cloud Security Command Center (SCC).
- STEP 2 | Select Alerts > Alert Rules and either Create an Alert Rule or select an existing rule to edit.
- STEP 3 | On the Set Alert Notification page for the alert rule, select CSCC.
- STEP 4 | Select the Google CSCC Integrations that want to use to send notifications of alerts triggered by this alert rule. - STEP 5 | Save the new alert rule or Confirm changes to an existing alert rule.
Send Alert Notifications to ServiceNow
send alert notifications to ServiceNow.
- STEP 1 | Integrate Prisma Cloud with ServiceNow.
- STEP 2 | Select Alerts > Alert Rules and either Create an Alert Rule or select an existing rule to edit.
- STEP 3 | On the Set Alert Notification page for the alert rule, select now. - STEP 4 | Select the ServiceNow Templates that want to use to send notifications of alerts triggered
by this alert rule.
- STEP 5 | Save the new alert rule or Confirm changes to an existing alert rule.
Send Alert Notifications to Webhooks
send alert notifications to webhooks.
- STEP 1 | Integrate Prisma Cloud with Webhooks.
- STEP 2 | Select Alerts > Alert Rules and either Create an Alert Rule or select an existing rule to edit.
- STEP 3 | On the Set Alert Notification page for the alert rule, select webhooks.
- STEP 4 | Select the webhook Channels that want to use to send notifications of alerts triggered by
this alert rule.
A webhook notification is delivered as soon as the alert is generated.
- STEP 5 | Save the new alert rule or Confirm changes to an existing alert rule.
Send Alert Notifications to PagerDuty
send alert notifications to PagerDuty.
- STEP 1 | Integrate Prisma Cloud with PagerDuty.
- STEP 2 | Select Alerts > Alert Rules and either Create an Alert Rule or select an existing rule to edit. - STEP 3 | On the Set Alert Notification page for the alert rule, select pagerduty.
- STEP 4 | Select the Integration Key.
- STEP 5 | Save the new alert rule or Confirm changes to an existing alert rule.
Send Alert Notifications to AWS Security Hub
send alert notifications to AWS Security Hub.
- STEP 1 | Integrate Prisma Cloud with AWS Security Hub.
- STEP 2 | Select Alerts > Alert Rules and either Create an Alert Rule or select an existing rule to edit.
- STEP 3 | Select AWS account from AWS Security Hub.
- STEP 4 | Save the new alert rule or Confirm changes to an existing alert rule.
Send Alert Notifications to Microsoft Teams
send alert notifications to Microsoft Teams.
- STEP 1 | Integrate Prisma Cloud with Microsoft Teams.
- STEP 2 | Select Alerts > Alert Rules and either Create an Alert Rule or select an existing rule to edit.
- STEP 3 | On the Set Alert Notification page for the alert rule, select Microsoft Teams. - STEP 4 | Select the Teams channels that want to use to send notifications for alerts triggered by
this alert rule.
- STEP 5 | Set the Frequency at which to send POST notifications.
- As it Happens—Sends a notification to the selected channels each time the alert rule triggers an alert.
- Daily—Sends a single notification to the selected channels once each day that lists all alerts triggered
by the alert rule on that day.
- Weekly—Sends a single notification to the selected channels once each week that lists all alerts triggered by the alert rule during that weekly interval.
- Monthly—Sends a single notification to the selected channels once each month that lists all alerts triggered by the alert rule monthly interval.
- STEP 6 | Save the new alert rule or Confirm changes to an existing alert rule.
When a policy rule is violated, a message card displays on the Microsoft teams conversation. The message card is formatted with a red (high), yellow (medium), or gray (low) line to indicate the severity of the alert. For example, the following screenshot is a daily notification summary. Send Alert Notifications to Cortex XSOAR
send alert notifications associated with an alert rule to a Demisto instance.
- STEP 1 | Integrate Prisma Cloud with Cortex XSOAR.
- STEP 2 | Select Alerts > Alert Rules and either Create an Alert Rule or select an existing rule to edit.
- STEP 3 | On the Set Alert Notification page for the alert rule, select Demisto.
- STEP 4 | Select the Demisto instance to which want to send alerts triggered by this alert rule.
- STEP 5 | Save the new alert rule or Confirm changes to an existing alert rule. View and Respond to Prisma Cloud Alerts As soon as Enable Prisma™ Cloud Alerts, Prisma Cloud generates an alert when it detects a violation in
a policy that is included in an active alert rule. To secure cloud environments, must monitor alerts.
either monitor alerts from Prisma Cloud or Send Prisma Cloud Alert Notifications to ThirdParty Tools to ensure that policy violations in cloud environments are resolved. The status of an alert
can be one the following:
- Open—Prisma Cloud identified a policy violation that triggered the alert and the violation is not yet
resolved.
- Resolved—Alerts automatically transition to Resolved state when the issue that caused the policy violation is resolved. An alert can also change to Resolved state due to a change in the policy or alert rule
that triggered the alert. A resolved alert can also transition back to the open state if the issue resurfaces
or there is a policy or alert rule change that causes the alert to trigger again.
- Snoozed—A Prisma Cloud administrator temporarily dismissed an alert for a specified time period. When
the timer expires, the alert automatically changes to an pen or Resolved state depending on whether the issue is fixed.
- Dismissed—A Prisma Cloud administrator manually dismissed the alert even though the underlying issue
was not resolved. manually reopen a dismissed alert if needed.
If manually dismiss an alert for a Network policy rule violation, Prisma Cloud
automatically reopens the alert when it detects the same violation again.
- View alerts from within Prisma Cloud.
Prisma Cloud displays all alerts for which role gives permission to see. Click Alerts to sort and
filter the alerts as follows:
- To modify which columns display, click Edit ( ) and add or remove columns.
- To sort on a specific column, click the corresponding Sort icon ( ).
- To filter on specific alert criteria, click the corresponding column Filter icon ( ) to filter on a specific
value in a column. also clear filters ( ) or save a filter ( ) for future use.
- To modify which filters are available or to perform a keyword search, click Add ( ) and then
either enter search term to Filter Results or add additional filters. use the following filters—Account Group, Alert ID, Alert Rule Name, Alert Status, Cloud Account, Cloud Region, Cloud Service, Cloud Type, Compliance Requirement, Compliance Section, Compliance Standard, Policy Label, Policy Name, Remediable, Resource ID, Resource Name, and Resource Type.
The filters act as a union operator to combine the results from multiple selections. -As needed, Download ( ) the filtered list of alert details to a CSV file.
- Address alerts.
Prisma Cloud generates an alert each time that it finds policy violations in one or more of the account
groups that are associated with an alert rule. monitor alerts in the cloud accounts for which are responsible to see any security risks have and to ensure that any critical issues get resolved or
remediated. An alert is resolved when the underlying conditions that generated the alert are fixed or
changed such as when the resource is no longer being scanned or the policy is no longer in effect. When
fix the issue on the Cloud Service Provider such as AWS or GCP, the issue is resolved automatically
and the resolution reason is displayed on Prisma Cloud. For a list of different reasons, see Prisma Cloud
Alert Resolution Reasons.
Send Prisma Cloud Alert Notifications to Third-Party Toolsand Configure Prisma Cloud to
Automatically Remediate Alerts, or manually resolve the issues. By reviewing these alerts, also
decide whether need to make a change to a policy or alert rule. Depending on the policy type that
triggered the alert, go directly from the alert to the cloud resource where the violation occurred
or resolve the issue from the Prisma Cloud Alerts page:
1. Filter the alerts to show only Open alerts that are Remediable.
2. Select the policy for which want to remediate alerts.
Review the recommendations for addressing the policy rule violation. also click the policy name to go directly to the policy.
3. Select the alert(s) want Prisma Cloud to resolve and then click Remediate.
To remediate issues, Prisma Cloud requires limited read-write access to cloud accounts. With
the correct permissions, Prisma Cloud can automatically run the CLI commandrequired to remediate
the policy violation directly on cloud platform. -Pivot from an alert into the cloud resource that triggered the alert to manually resolve the issue.
Prisma Cloud allows to pivot directly from an alert to view the violating cloud resource and resolve
the issue manually.
1. Filter the alert list to show alerts with Alert Status Open and select the Policy Type. For example, Network or Config.
2. Select the policy for which want to resolve alerts.
Review the recommendations for resolving the policy violation.
3. Click Resource ( ) to pivot to the cloud resource containing the violation want to resolve and
follow the recommended steps.
When click Resource, Prisma Cloud redirects the request to the cloud platform. To view the resource details in the cloud platform, must be logged in to the same account on the cloud
platform where want to further investigate. Generate Reports on Prisma Cloud Alerts generate two types of reports on alerts—Cloud Security Assessment report and Business Unit
report. These reports enable to inform stakeholders on the status of the cloud assets and how
they are doing against Prisma Cloud security and compliance policy checks. Sharing the reports on a regular
basis enables them to monitor progress without requiring access to the Prisma Cloud administrator console.
The Cloud Security Assessment report is a PDF report that summarizes the risks from open alerts in the monitored cloud accounts for a specific cloud type. The report includes an executive summary and a list
of policy violations, including a page with details for each policy that includes the description and the compliance standards that are associated with it, the number of resources that passed and failed the check
within the specified time period.
The Business Unit report is a .csv file that includes the total number of resources that have open alerts against policies for any compliance standard, and generate the report on-demand or on a recurring
schedule. opt to create an overview report which shows how you’re doing across all your
business units, or get a little more granular about each of the cloud accounts want to monitor. also generate the Business Unit report to review policy violations that are associated with specific
compliance standards.
The overview report lists cloud resources by account group and aggregates information about the number
of resources failing and the failure percentage against each policy. Whereas, the detailed Business Unit
report lists cloud resources by account group, account name, and account ID, and it includes information
about the number of resources failing against each policy and the status of cloud resources that have been
scanned against that policy. The status can be pass or fail, and the status is reported as pass it means that
the count of resources that failed the policy check is zero.
- STEP 1 | Select Alerts > Reports > +Add New.
- STEP 2 | Enter a Name and select a Report Type.
select Cloud Security Assessment report and Business Unit report.
- To generate a Business Unit Report:
1. Select the Account Groups to include in the file. 2. Select Detailed Report, if want to include a breakdown of the policy details for each cloud
account being monitored. Read above for more details on the difference between the overview
and the detailed Business Unit report.
For the detailed report, select Compress Attachments to get a .zip file for download or email
attachment.
3. Filter alerts associated with specific compliance standards.
1. Select Filter by Compliance Standard.
2. Select the Compliance Standards from the drop-down.
This option is useful if want to view the alerts that are associated with policies that are
tied to the selected compliance standards. The report then includes information on alerts that
pertain to the selected compliance standards only.
4. Enter the Email address(es) for the file recipient(s).
5. Select the Notification template, if want to use a custom email template.
A custom email template allows to tailor message and add a URL in the message body.
See Send Alert Notifications Through Email to set up a custom template.
6. Set the Frequency at which to send email notifications.
One-Time—Sends an email to the recipient list only this once.
Recurring—select the time interval as Daily, Weekly or Monthly. Prisma Cloud sends a single email to the recipient list on that day, during that week, or the month.
only edit recurring Alerts Reports to modify some inputs such as the time
interval, whether or not to compress attachments.
7. Save changes.
- To generate a Cloud Security Assessment PDF Report:
1. Select the Cloud Type to include in the file.
2. Select the Account Groups and the Cloud Accounts to include in the file.
3. Select the cloud Regions and the Time Range for the report.
If have a large number of open alerts in the account(s) selected an error message displays.
must remove some accounts from this report or reduce the time range, and create a separate
report for the details need.
4. Save changes. Alert Payload
A Prisma™ Cloud alert payload is a JSON data object that contains detailed information about an alert, such
as the cloud account, resource, compliance standard, and policy.
Alert Payload Field Description
Account ID The ID of the cloud account where the violation
that triggered the alert occurred.
Account Name Name of the cloud account where Prisma Cloud
detected the policy violation.
Alert ID Identification number of the alert.
Alert Rule Name Name of the alert rule that triggered this alert.
Callback URL The URL for the alert in Prisma Cloud.
Cloud Type Type of cloud account: AWS, Azure, or GCP.
Policy Description Description of the policy as shown within Prisma Cloud.
Policy ID Universally unique identification (UUID) number of the policy.
Policy Labels Labels associated with the policy.
Policy Name Name of the policy. Alert Payload Field Description
Policy Recommendation Remediation recommendations for the policy.
Saved Search UUID Universally unique identification (UUID) number of the saved search.
Remediation CLI The CLI commands that use to resolve the policy violation.
Compliance Standard name Name of the compliance standard.
Compliance Standard description Description of the compliance standard.
Requirement ID Identification number of the requirement in the compliance standard.
Requirement Name Name of the requirement in the compliance
standard.
Section ID Identification number of the section in the compliance standard.
Section Description Description of the section in the compliance
standard.
Compliance ID ID number of the compliance standard.
System Default Indicates whether the compliance standard is
Prisma Cloud System Default.
Custom assigned Indicates if the compliance standard is assigned to
a policy.
Resource Cloud Service Cloud service provider of the resource that
triggered the alert.
Resource Data The JSON data of the resource.
Resource ID ID of the resource that triggered the alert.
Resource Name Name of the resource that triggered the alert.
Resource Region Name of the cloud region to which the resource
belongs.
Resource Region ID ID of the region to which the cloud resource
belongs.
Resource Type Type of resource that triggered the alert (for example, EC2 instance or S3 bucket).
Severity Severity of the alert: High, Medium, or Low. Alert Payload Field Description
User Attribution data Data about the user who created or modified the resource and caused the alert.
For alert notifications to include
user attribution data, must
Populate User Attribution In
Alerts Notifications ( Settings >
Enterprise Settings). Including
user attribution data may delay
alert notifications because the information may not be available
from the cloud provider when
Prisma Cloud is ready to generate
the alert. Prisma Cloud Alert Resolution Reasons
When an open alert is resolved, the reason that was alert was closed is included to help with audits. The reason is displayed in the response object in the API, and on the Prisma Cloud administrative console on
Alerts > Overview when select an resolved alert and review the alert details for the violating resource.
The table below lists the reasons:
Reason Description
RESOURCE_DELETED Resource was deleted.
RESOURCE_UPDATED Resource was updated (based on the JSON metadata).
POLICY_UPDATED Policy was updated.
POLICY_DISABLED Policy was disabled.
POLICY_DELETED Policy was deleted.
ALERT_RULE_DISABLED Alert rule was disabled.
ALERT_RULE_UPDATED Alert rule was updated. The list of policies included in the rule, accountgroups being scanned, or cloud regions may have been modified.
ALERT_RULE_DELETED Alert rule was deleted.
ACCOUNT_GROUP_UPDATED Account group was updated.
ACCOUNT_GROUP_DELETEDAccount group was deleted.
ANOMALY_CONFIG_CHANGED Anomaly policy configuration changed.
REMEDIATED Alert was successfully remediated using the Cloud Service Provider’s CLI, either manually or auto-remediation.
USER_DISMISSED Alert was dismissed or snoozed by the Prisma Cloud administrator with role
of System admin, Account Group Admin, or Account and Cloud Provisioning
Admin.
USER_REOPENED A dismissed or snoozed alert was reopened by the Prisma Cloud
administrator with role of System admin, Account Group Admin, or Account
and Cloud Provisioning Admin.
MDC_REOPEN_FOR_ACCIDENTAL_DELETE Alert was reopened during ingestion as resource was rediscovered.
NEW_ALERT A new alert was generated.
RESOURCE_POLICY_RESCOPED Alert was resolved because the policy was updated and the violating
resource is no longer scanned or within the scope of the modified policy. Reason Description
NETWORK_DISMISSED_AUTO_REOPEN Alerts generated against Network policies cannot be resolved, however dismiss the open alert. For an alert that dismissed, Prisma Cloud may
automatically reopen the alert when the same network flow is seen again.
SNOOZED_AUTO_REOPENSnooze time expired for the alert, and it was automatically reopened. Prisma Cloud Dashboards
The interactive Asset Inventory and SecOps dashboards give visibility into the health
and security posture of cloud infrastructure. The dashboards provide a summarized and
graphical view of all Prisma Cloud cloud accounts and resources, and use the predefined or custom time range to view current trends or historical data.
> Assets, Policies, and Compliance on Prisma Cloud
> Prisma Cloud Asset Inventory
> SecOps Dashboard
> Customize the SecOps Dashboard Assets, Policies, and Compliance on Prisma Cloud
To know the state of cloud infrastructure, need visibility into all the assets and infrastructure that
make up cloud environment and a pulse on security posture.
Whether want to detect a misconfiguration or want to continually assess security posture
and adherence to specific compliance standards Prisma Cloud provides out-of-the-box policies (auditable
controls) for ongoing reporting and measurement.
Policies are for risk assessment and they help to reduce the risk of business disruptions. Prisma Cloud
provides policies that map to compliance standards, and a larger set of policies that enable prevention
or detection of security risks to which cloud assets are exposed. Anomaly policies are an example
of policies that are typically not a part of compliance standards, and these policies inform of actions
performed on cloud assets by entities that are users, services, or IAM roles that have authorization to
access and modify cloud assets, but the entities are not cloud assets. Prisma Cloud supports the need to keep track of potential risks and threats to cloud infrastructure
with dashboards for Asset Inventory, Compliance Dashboard, and out-of-the-box policieswhich
generate alerts for cloud assets that are in violation. When a policy is violated, an alert is triggered in real
time.
While alerts help detect policy violations in real time and enable to investigate what happened, the asset inventory and compliance dashboard are hourly snapshots of assets and compliance posture for the last full hour. From the asset inventory and the compliance dashboard, directly access all open alerts by severity, and view asset details from the asset explorer as of the last hour. Prisma Cloud Asset Inventory
The Asset Inventory dashboard (on the Inventory tab) provides a snapshot of the current state of all cloud
resources or assets that are monitoring and securing using Prisma Cloud. From the dashboard, gain
operational insight over all our cloud infrastructure, including assets and services such as Compute Engine
instances, Virtual machines, Cloud Storage buckets, Accounts, Subnets, Gateways, and Load Balancers.
Assets are displayed by default for all account groups, which the service monitors, for the most recent time
range (last full hour). The interactive dashboard provides filters to change the scope of data displayed, so
that analyze information want to view in greater detail.
At a glance the Asset Inventory dashboard four sections:
- Resource Summary - Shows the count of the Total Unique Resources monitored by Prisma Cloud. Click
the link to view all the assets on the Asset Explorer. For all these assets, toggle to view the following details as numeric value or a percentage:
- Pass—Displays the resources without any open alerts. Click the link for the passed resources and
will be redirected to the Asset Explorer that is filtered to display all the resources that have Scan
Status set to Pass. -Low/Medium/High—Displays the resources that have generated low, medium, or high severity alerts.
On the asset inventory, when a resource triggers multiple alerts, the asset severity assigned to it
matches the highest risk to which it is exposed. When click the link, will be redirected to
the Asset Explorer that is filtered to display all the resources that match the corresponding Asset
Severity level.
The View Alerts link enables to view a list of all resources that have open alerts sorted by
severity. Click each link to view the Alerts Overview sorted for low, medium or high severity alerts.
review the policies that triggered the alerts along with a count of the total number of alerts for each policy.
- Fail—Displays the total number of resources that have generated at least one open alert when the hourly snapshot was generated. Click the link and will be redirected to the Asset Explorer that is
filtered to display all resources that have Scan Status set to Failed.
- Asset Trend—Trend line to help monitor the overall health of cloud resources starting when
added the first cloud account on Prisma Cloud through the time when the hourly snapshot was generated. The green, blue and red trend lines are overlaid to visually display the pass and failed
resources against the total resource count. The trends depict the overall security posture of your
resources and how they are performing over time so identify sudden surges with failed policy checks or sustained improvements with passing policy checks.
- Asset Classification—Bar graph for each cloud type (default), region name, or account name that depicts
the ratio of passed to failed resources. This interactive graph allows to drill into the passed and
failed resources for details on the corresponding services that passed or failed policy checks; can
click and drag a section of the chart to zoom in further.
- Tabular data— The table enables to group the results by account name, cloud region, or service
name (default) and then drill down to view granular information on the resource types within cloud
accounts. All global resources for each cloud are grouped under AWS Global, Alibaba Cloud Global, Azure Global, and GCP Global.
Each row displays the service name with details on the cloud type (which filter on), and the percentage of resources that pass policy checks to which want to adhere. The links is each column
help explore and gain the additional context may need to take action.
may see more failed resources on the Compliance Dashboard compared to the Asset
Inventory. This is because the Asset Inventory only counts assets that belong to cloud
account, and the Compliance Dashboard includes foreign entities such as SSO or Federated
Users that are not resources ingested directly from the monitored cloud accounts. SecOps Dashboard
The Dashboard > SecOps provides a graphical view of the performance of resources that are connected to
the internet, the risk rating for all accounts that Prisma Cloud is monitoring, the policy violations over time
and a list of the policies that have generated the maximum number of alerts across cloud resources. It
makes the security challenges visible to as a quick summary, so dig in.
Monitored Accounts
This graph shows the number of accounts Prisma Cloud is monitoring.
Monitored Resources
Prisma Cloud considers any cloud entity that work with as a resource. Examples of resources include
AWS Elastic Compute Cloud, Relational Databases, AWS RedShift, Load Balancers, Security Groups, NAT
Gateways The Resources graph shows the total number of resources that currently manage. It gives
a view into the potential growth in the number of resources in enterprise over a period of time.
Hover over the graph to see data as per the timeline.
Open Alerts Whenever a resource violates a policy, Prisma Cloud generates alerts flagging these policy violations.
The Open Alerts graph shows the number of alerts that were generated. The purpose of this graph is to
demonstrate risk trends over a timeline. Click on the alert number to go to the ‘Alerts’ section and get the detailed view of the alerts.
Top Instances by Role
This graph summarizes top open ports in cloud environments and the percentage of the traffic directed
at each type of port. The purpose of this graph is to show what types of applications (web server, database)
the top workloads are running. Alerts by Severity
Alerts are graphically displayed and classified based on their severity into High, Medium, and Low. By
clicking on the graph, directly reach the alerts section.
Policy Violations by Type over Time
This graph displays the type of policy violations (network, config, audit event) over a period of time.
Top Policy Violations
This graph displays the alerts generated by each type of policy over a period of time.
Top Internet Connected Resources
This graph displays top internet connected workloads by role, so know which workloads are connecting
to the Internet most of the time and are prone to malicious attacks. For this report, ELB & NAT Gateway
data are filtered out, but includes data from other roles. The data in this chart is based on the account and
the time filter.
Connections from the Internet
On a world map, see the inbound and outbound connections to different workloads across the globe , so that visualize where the connections are originating from and see whether the traffic is
regular internet traffic, suspicious traffic and all accepted traffic from suspicious IP addresses. By default, the map shows aggregated numbers by specific regions in the map but zoom in on any
of the regions in the map a get more granular detail on the specific location.
use the multi-select filter option available on the map to only present information for the type of workload(s) are interested in viewing traffic for. By default, traffic to destination resources that are
allowed to accept inbound connections such as NAT Gateways, ELB, Web Servers, and HTTP traffic is
filtered out.
To see the network graph representing connections, click on any of the connections from a specific region
and get redirected to the Investigate page to see the network graph. The network query will have the IP address, destination resources and the time filters carried forward so pin point to a specific
incident. Customize the SecOps Dashboard
customize the screen space used for each widget on the SecOps dashboard. For the Top Instances
by Role widget, view the data as a table or click the graph to view the segment details in a table.
- Select Dashboard > SecOps and click Customize.
- Toggle Show or Hide to view or hide a widget.
- Select the icons on the screen to choose whether want to maximize the screen
space for a widget or fit two or three widgets in a row.
- Select a widget and click View as table to view the data in a tabular format. 245
Prisma Cloud Policies
In Prisma Cloud, a policy is a set of one or more constraints or conditions that must be adhered
to. Prisma Cloud provides predefined policies for configurations and access controls that
adhere to established security best practices such as PCI, GDPR, ISO 27001:2013,and NIST, and a larger set of policies that enable to validate security best practices with an impact
beyond regulatory compliance. These Prisma Cloud default polices cannot be modified.
In addition to these predefined policies, create custom policies to monitor for violations and enforce own organizational standards. use the Default policies as templates to create custom policy. After set up the policies, any new or existing resources
that violate these policies are automatically detected.
> Create a Policy on Prisma Cloud
> Manage Prisma Cloud Policies
> Anomaly Policies Create a Policy on Prisma Cloud
Create a custom policy with remediation rules that are tailored to meet the requirements of your
organization. When creating a new policy, either build the query using RQL or use a saved
search to automatically populate the query need to match on cloud resources. For Prisma Cloud
DevOps Security, also create configuration policies to scan Infrastructure as Code (IaC)
templates that are used to deploy cloud resources. The policies used for scanning IaC templates use a JSON
query instead of RQL.
If want to enable auto-remediation, Prisma Cloud requires write access to the cloud platform to
successfully execute the remediation commands.
create three types of policies:
- Config—Configuration policies monitor resource configurations for potential policy violations.
Configuration policies on Prisma Cloud can be of two sub-types—Build and Run—to enable a layered
approach. Build policies enable to check for security misconfigurations in the IaC templates and
ensure that these issues do not make their way into production. The Run policies monitor resources and
check for potential issues once these cloud resources are deployed. See Create a Configuration Policy.
- Network—Network policies monitor network activities in environment. See Create a Network or
Audit Event Policy.
- Audit Event—Event policies monitor audit events in environment for potential policy violations. create audit policies to flag sensitive events such as root activities or configuration changes that may
potentially put cloud environment at risk. See Create a Network or Audit Event Policy.
Create a Configuration Policy Use these instructions to add a custom configuration policy, for checking resources in the build or run phase
of application lifecycle. Because building the rules takes practice, before start, take a look at a few
Prisma Cloud default policies for directly on the administrative console, and review the query format within
the rules.
- STEP 1 | Select Policies and click New Policy > Config.
- STEP 2 | Enter a Policy Name.
optionally add a Description and Labels. - STEP 3 | Select the policy subtype and click Next.
choose one or both the policy subtypes options:
Run subtype enables to scan cloud resources that are already deployed on a supported cloud
platform.
Build subtype enables to scan IaC templates—Terraform, CloudFormation, Kubernetes manifest—that are used to deploy cloud resources.
- STEP 4 | Select the Severity for the policy and click Next.
For a Run policy, an alert will be generated on a policy violation.
- STEP 5 | Build the query to define the match criteria for policy.
1. Add a rule for the Run phase.
The Configuration—Run policies use RQL. If are using a Saved Search, select from
predefined options to auto-populate the query. For building a New Search, enter config where
and use the auto-suggestion to select the available attributes and complete the query. Config queries require some mandatory attributes. It should at a minimum have api.name in
conjunction with json.rule or it can have hostfinding.type or it can have two api.name
attributes with a filter attribute.
config where cloud.type = 'azure' AND api.name = 'azure-network-usage' AND
 json.rule = StaticPublicIPAddresses.currentValue greater than 1
config where hostfinding.type = 'Host Vulnerability'
config where api.name = 'aws-ec2-describe-internet-gateways' as  X; config where api.name = 'aws-ec2-describe-vpcs' as Y; filter
 '$.X.attachments[*].vpcId == $.Y.vpcId and $.Y.tags[*].key contains
 IsConnected and $.Y.tags[*].value contains true'; show Y;
When creating a custom policy, as a best practice do not include cloud.account, cloud.account.group or cloud.region attributes in the RQL query. If have a saved
search that includes these attributes, make sure to edit the RQL before create
a custom policy. While these attributes are useful to filter the results see on the Investigate tab, they are ignored when used in a custom policy.
2. Add a rule for the Build phase.
If policy will include both Run and Build checks, and have added the RQL query, cloud
type for the build rule is automatically selected. It is based on the cloud type referenced in the RQL
query.
1. Select the Template Type want to scan—CloudFormation, Kubernetes, or Terraform. can
add one or more types.
For scanning Terraform templates, must select the Cloud Type and the Terraform version.
Terraform versions 0.11 and 0.12 are supported.
2. Add the JSON query that specifies the properties or objects for which want to apply policy checks. For more information see Add a JSON Query for Build Policy Subtype and Prisma Cloud
IAC Scan Policy Operators. If choose to upload a template in the next step, the query entered above is validated
against the template. Each time modify the query or upload a new template, the JSON query
is re-validated.
3. (Optional) Upload a file to validate the JSON query.
The JSON Template Validation is optional. upload a single file or a .zip file. The supported
file formats are HCL,YAML, JSON. The uploaded file is converted to JSON and displayed onscreen.
In addition, include a variable name and value to pass to the sample file and verify that
the build rule works before save the policy. For example, if want to check whether EC2
instances include tags to identify the owner, the variables enable to quickly validate against
the sample template attached.
- STEP 6 | Add the compliance standards to policy.
1. Choose the compliance Standard, Requirement, and Section.
2. Click + to add more standards as required and click Next.
- STEP 7 | Enter details in the remediation section, if want to automatically remediate alerts on a policy violation.
1. Select Run or Build
Build phase policies do not support remediation CLI. however add the instructions for manually fixing the issue in the Recommendation for Remediation.
2. (Configuration—Run policies only) Enter Command Line remediation commands in CLI Remediation. CLI remediation is available for config where queries only. add up to 5 CLI commands, and use a semi-colon to separate the commands in the sequence. The sequence is executed in
the order defined in policy, and if a CLI command fails, the execution stops at that command.
The parameters that use to create remediation commands are displayed on the interface
as CLI variables, and a syntax example is: gcloud -q compute --project=${account}
firewall-rules delete ${resourceName}; gsutil versioning set off gs://
${resourceName};:
- $account—Account is the Account ID of account in Prisma Cloud.
- $azurescope—(Azure only) Allows to specify the node in the Azure resource hierarchy
where the resource is deployed.
- $gcpzoneid—(GCP only) Allows to specify the zone in the GCP project, folder, or
organization where the resource is deployed.
- $region—Region is the name of the cloud region to which the resource belongs.
- resourcegroup— (Azure only) Allows to specify the name of the Azure Resource Group
that triggered the alert.
- $resourceid—Resource ID is the identification of the resource that triggered the alert.
- $resourcename—Resource name is the name of the resource that triggered the alert.
3. Click Validate syntax to validate the syntax of code.
If would like to see an example of the CLI syntax in the default remediable policies on Prisma Cloud, clone any existing policy and edit it.
The default policies include additional variables that are restricted for use in default
policies only, and are not supported in custom policies. Syntax validation displays an
error if use the restricted variables.
4. Click Save.
All System Administrators and Account Administrators are notified when there is a change to
the CLI commands.
Create a Network or Audit Event Policy Use the following instructions to add a custom Network or Audit Event policy on Prisma Cloud.
- STEP 1 | Select Policies and click New Policy.
- STEP 2 | Select Audit Event or Network. - STEP 3 | Enter a Policy Name and Severity.
- STEP 4 | Add an optional Description and Labels before click Next.
- STEP 5 | Build the query to define the match criteria for policy by using a New Search or a Saved
Search and click Next.
If are using a Saved Search, select from the list of predefined options to auto-populate the query. The Select Saved Search drop-down displays the RQL for saved searches that match the policy type selected in Step 2 above.
For a building a New Search, the RQL query must begin with event where for an Audit Event policy or network where for a Network policy. then use the auto-suggestion to select the available
attributes and complete the query. - STEP 6 | Select the compliance standards for policy.
1. Choose the compliance Standard, Requirement, and Section.
2. Click + to add more standards as required and click Next.
- STEP 7 | (Optional) Provide a Recommendation for Remediation.
CLI commands to enable automatic remediation are not supported on Audit Event or Network policy.
- STEP 8 | Save the policy.
Add a JSON Query for Build Policy Subtype
Policy rules to Secure Infrastructure Automation are written in JSON. So, need to first convert
template file—CFT, Kubernetes or Terraform templates— to JSON and then parse the JSON structure
to write the JSON query that correctly identifies the parameter and criteria on which want to be
alerted. See Build Policy Query Examples below.
- STEP 1 | Convert template file to JSON.
also use any online tool to convert CFT, Kubernetes or Terraform templates to JSON. In
this workflow, use the following sample Terraform template and upload the template file to Prisma Cloud.
1. Get template file.
For example:
 Resources:
 myTrail:
 DependsOn:
 - BucketPolicy  - TopicPolicy  Type: AWS::CloudTrail::Trail
 Properties:
 S3BucketName:
 Ref: S3Bucket
 SnsTopicName:
 Fn::GetAtt:
 - Topic
 - TopicName
 IsLogging: true
 IsMultiRegionTrail: true
 myTrail2:   DependsOn:
 - BucketPolicy  - TopicPolicy  Type: AWS::CloudTrail::Trail
 Properties:
 S3BucketName:
 Ref: S3Bucket
 SnsTopicName:
 Fn::GetAtt:
 - Topic
 - TopicName
 IsLogging: true
 IsMultiRegionTrail: true
2. Select Policies and click New Policy > Config.
3. Enter a Policy Name.
optionally add a Description and Labels.
4. Select the Build policy subtype and click Next.
The build subtype enables to scan IaC templates that are used to deploy cloud resources. For Run policy subtype, see Create a Configuration Policy.
5. Select the Severity for the policy and click Next.
6. Build the query to define the match criteria for policy.
If policy is for both Run and Build checks and have added the RQL query, cloud type
for the build rule is automatically selected. It is based on the cloud type referenced in the RQL query.
Otherwise, must select the template type and the cloud type.
1. Select the Template Type want to scan—CloudFormation, Kubernetes, or Terraform. The supported files types are HCL, YAML, JSON.
For scanning Terraform templates, must select the Cloud Type and the Terraform version.
Terraform versions 0.11 and 0.12 are supported. For the other templates, do not need to
select the cloud type.
2. Upload the template file for conversion to JSON in the JSON Template Validation section. In order to upload a file must add a JSON query string. enter object exists to
enable the ability to attach a file.
Then, upload a single file or a .zip file.
The sample template above when converted to JSON looks like this:
{
 "Resources": {
 "myTrail": {
 "Type": "AWS::CloudTrail::Trail",  "Properties": {
 "S3BucketName": {
 "Ref": "S3Bucket"
 },  "IsLogging": true,  "IsMultiRegionTrail": true
 }
 },  "myTrail2": {
 "Type": "AWS::CloudTrail::Trail",  "Properties": {
 "S3BucketName": {
 "Ref": "S3Bucket"
 },  "IsLogging": true,  "IsMultiRegionTrail": true
 }
 }
 }
}
- STEP 2 | Use JSON path validators to write JSON query in the policy.
Use the following guidelines to parse the file structure and write the JSON query that
specifies the properties or objects for which want to apply policy checks:
- $ - symbol refers to the root object or element.
- @ – symbol refers to the current object or element.
- . – operator is the dot-child operator, which use to denote a child element of the current element.
- [ ] – is the subscript operator, which use to denote a child element of the current
element (by name or index).
- * – operator is a wildcard, that returns all objects or elements without regard of the name.
- ? ( ) – to query all items that meet a certain criteria.
For validating the JSON path, use validators such as https://jsonpath.com/
or others available on the internet. Each time modify the query or upload a new
template, Prisma Cloud revalidates JSON query.
Refer to the list of Prisma Cloud IAC Scan Policy Operators to define the operators for the match.
If for example, require that all AWS accounts across organization have enabled AWS
CloudTrail, check for violations where AWS CloudTrail is not enabled. -$.Resources.*[?(@.Type=='AWS::CloudTrail::Trail')], enables filtering the Resources
whose type is AWS::CloudTrail::Trail
- and $.Resources.*[?
(@.Type=='AWS::CloudTrail::Trail')].Properties.IsMultiRegionTrail ‘any
null’ or $.Resources.*[?
(@.Type=='AWS::CloudTrail::Trail')].Properties.IsMultiRegionTrail anyFalse
enables to further filter for the value specified for the MultiRegionalTrail parameter.
In this case, are looking to identify a security issue when the value is missing or set to false. So, the match works as follows:If the Type: “AWS::CloudTrail::Trail”, then:
Parameter Value Outcome
IsMultiRegionalTrail missing Because the default is false, the rule will match.
It means there a security issue
detected.
IsMultiRegionalTrail false The rule will match.
It means there is a security
issue detected.
IsMultiRegionalTrail true The rule will not match.
It means there is no security
issue detected.
- STEP 3 | Choose next steps:
Continue to Step 6if want to add compliance standards to the policy rule or to 5.b. Otherwise, Save
the policy rule.
Build Policy Query Examples
The following section shows an example of a Terraform template and an AWS CloudFormation
Template (CFT).
- Sample JSON file (after being converted from Terraform)
1. View the file contents.
 {
 "data": [
 {
 "azurerm_client_config": [
 {
 "current": [
 {}
 ]
 }
 ]
 }
 ],  "provider": [
 {  "azurerm": [
 {
 "features": [
 {
 "key_vault": [
 {
 "purge_soft_delete_on_destroy": true
 }
 ]
 }
 ]
 }
 ]
 }
 ],  "resource": [
 {
 "azurerm_resource_group": [
 {
 "example": [
 {
 "location": "West US",  "name": "resourceGroup1"
 }
 ]
 }
 ]
 },  {
 "azurerm_key_vault": [
 {
 "example": [
 {
 "access_policy": [
 {
 "key_permissions": [
 "list"
 ],  "object_id": "11111111-2222-3333-4444-555555555555",  "secret_permissions": [
 "list"
 ],  "storage_permissions": [
 "get"
 ],  "tenant_id": "2111-3333-4445-555"
 }
 ],  "enabled_for_disk_encryption": true,  "location": "azurerm_resource_group.example.location",  "name": "testvault",  "network_acls": [
 {
 "bypass": "AzureServices",  "default_action": "Deny"
 }
 ],  "purge_protection_enabled": false,  "resource_group_name": "def",  "sku_name": "standard",  "soft_delete_enabled": true,  "tags": [  {
 "environment": "Testing"
 }
 ],  "tenant_id": "2111-3333-4445-555"
 }
 ]
 }
 ]
 }
 ]
}

2. Define the match criteria for the policy.
This following query checks that the template for an Object ID match. It checks whether the object
ID of the user or service principal in Azure Active Directory that is granted permissions matches your
organizational policy.
 $.resource[*].azurerm_key_vault.*[*].*.access_policy exists
 and $.resource[*].azurerm_key_vault.*[*].*.access_policy[*].object_id
 == "11111111-2222-3333-4444-555555555555"

- Original CFT file that defines the security groups and the ports that allow ingress traffic.
1.
 AWSTemplateFormatVersion: '2010-09-09'
Parameters:
 testDescription:
 Description: Tests for blocked ports negative case in AWS Security
 Groups
 Type: String
Resources:
 myELB:
 Type: AWS::ElasticLoadBalancing::LoadBalancer
 Properties:
 AvailabilityZones:
 - eu-west-1a
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
 FromPort: 22
 ToPort: 22
 CidrIp: 0.0.0.0/0
 SourceSecurityGroupOwnerId:
 Fn::GetAtt:
 - myELB
 - SourceSecurityGroup.OwnerAlias  SourceSecurityGroupName:
 Fn::GetAtt:
 - myELB  - SourceSecurityGroup.GroupName
 myELBIngressGroup2:
 Type: AWS::EC2::SecurityGroup
 Properties:
 GroupDescription: ELB ingress group
 SecurityGroupIngress:
 - IpProtocol: tcp
 FromPort: '22'
 ToPort: '22'
 CidrIp6: "::/0"
 SourceSecurityGroupOwnerId:
 Fn::GetAtt:
 - myELB
 - SourceSecurityGroup.OwnerAlias  SourceSecurityGroupName:
 Fn::GetAtt:
 - myELB
 - SourceSecurityGroup.GroupName

2. Sample JSON file (after being converted from AWS CFT)
 {
 "AWSTemplateFormatVersion": "2010-09-09",  "Parameters": {
 "testDescription": {
 "Description": "Tests for blocked ports negative case in AWS
 Security Groups",  "Type": "String"
 }
 },  "Resources": {
 "myELB": {
 "Type": "AWS::ElasticLoadBalancing::LoadBalancer",  "Properties": {
 "AvailabilityZones": [
 "eu-west-1a"
 ],  "Listeners": [
 {
 "LoadBalancerPort": "80",  "InstancePort": "80",  "Protocol": "HTTP"
 }
 ]
 }
 },  "myELBIngressGroup": {
 "Type": "AWS::EC2::SecurityGroup",  "Properties": {
 "GroupDescription": "ELB ingress group",  "SecurityGroupIngress": [
 {
 "IpProtocol": "tcp",  "FromPort": 22,  "ToPort": 22,  "CidrIp": "0.0.0.0/0",  "SourceSecurityGroupOwnerId": {
 "Fn::GetAtt": [
 "myELB",  "SourceSecurityGroup.OwnerAlias"  ]
 },  "SourceSecurityGroupName": {
 "Fn::GetAtt": [
 "myELB",  "SourceSecurityGroup.GroupName"
 ]
 }
 }
 ]
 }
 },  "myELBIngressGroup2": {
 "Type": "AWS::EC2::SecurityGroup",  "Properties": {
 "GroupDescription": "ELB ingress group",  "SecurityGroupIngress": [
 {
 "IpProtocol": "tcp",  "FromPort": "22",  "ToPort": "22",  "CidrIp6": "::/0",  "SourceSecurityGroupOwnerId": {
 "Fn::GetAtt": [
 "myELB",  "SourceSecurityGroup.OwnerAlias"
 ]
 },  "SourceSecurityGroupName": {
 "Fn::GetAtt": [
 "myELB",  "SourceSecurityGroup.GroupName"
 ]
 }
 }
 ]
 }
 }
 }
}

3. Define the policy match
Check for any IPv4 or IPv6 CIDR range that allows unrestricted access for ingress traffic on port 22.
$.Resources.*[?(@.Type ==
'AWS::EC2::SecurityGroup')].Properties.SecurityGroupIngress[?(@.IpProtocol
== 'tcp' && @.FromPort == '22' && @.ToPort == '22' && @.CidrIp
== '0.0.0.0/0')] size greater than 0 or $.Resources.*[?(@.Type ==
'AWS::EC2::SecurityGroup')].Properties.SecurityGroupIngress[?(@.IpProtocol
== 'tcp' && @.FromPort == '22' && @.ToPort == '22' && @.CidrIp6 ==
'::/0')] size greater than 0
This rule will match as follows:
Parameter Value Outcome
IpProtocol
FromPort
tcp
22
The rule will match.
Security issue found. Parameter Value Outcome
ToPort
CidrIp
22
0.0.0.0/0
IpProtocol
FromPort
ToPort
CidrIp6
tcp
22
22
::/0
The rule will match.
Security issue found.
IpProtocol
FromPort
ToPort
CidrIp6
ftp
23
23
0.0.0.0/0 or ::/0
The rule will not match.
No security issue found.
IpProtocol
FromPort
ToPort
CidrIp6
tcp
22
22
4.0.0.0/0 or ::/16
The rule will not match
because an IP address
restriction is in place.
No security issue found.
- The JSON query on Prisma Cloud does not support the following cases:
Embedded policies or similar structures that are represented as a JSON string instead of JSON elements.
Example: If resource template is like this:
 resource "aws_iam_role" "nat" {
 name = "${local.infra}-nat"
 path = "/"
 assume_role_policy =<<EOF
{
 "Version": "2008-10-17",  "Statement": [
 {
 "Action": "sts:AssumeRole",  "Principal": {
 "Service": "ec2.amazonaws.com"
 },  "Effect": "Allow",  "Sid": ""
 }
 ]
}
EOF
}
When converted to JSON, the assume_role_policy is a json string within json. While Prisma Cloud
can fetch the complete policy, it does not support the ability to filter the parameters inside it.  {
 "aws_iam_role": [
 {
 "nat": [
 {
 "path": "/",  "name": "${local.infra}-nat",  "assume_role_policy": "{\n \"Version
\": \"2008-10-17\",\n \"Statement\": [\n {\n \"Action\":
 \"sts:AssumeRole\",\n \"Principal\": {\n \"Service\":
 \"ec2.amazonaws.com\"\n },\n \"Effect\": \"Allow\",\n \"Sid\":
 \"\"\n }\n]\n}\n"
 }
 ]
 }
 ]
 }, -Filtering with multiple criteria or with literal * matching.
Example: An IAM policy that allows full administrative permissions, that is access to all AWS actions and
resources, is a policy that contains a statement with
"Effect": "Allow" for "Action": "*" over "Resource": "*", i.e. "Statement": [ { "Effect": "Allow", "Action": "*", "Resource": "*" } ].
The following JSON equivalent for the policy above is not supported on Prisma Cloud:
$.Resources.*[?
(@.Type=='AWS::IAM::Policy')].Properties.PolicyDocument.Statement[?(
 @.Effect == 'Allow' && @.Action == '*' && @.Resource == '*' )] exists
Prisma Cloud IAC Scan Policy Operators
For Prisma Cloud DevOps Security, create configuration policies to scan Infrastructure as Code (IaC) templates that are used to deploy cloud resources. The policies used for scanning IaC templates
use a JSON query instead of RQL. The following list of operators are available for use in a JSON query, when Add a JSON Query for Build Policy Subtype and specify the properties or objects for which want to apply policy checks.
Operator Usage Examples
'greater than' | ' > '
"$.Resources.*[?(@.Type ==
 'AWS::EC2::SecurityGroup')].Properties.SecurityGroupIngress[?
(@.IpProtocol == 'tcp' && @.FromPort ==
 '22' && @.ToPort == '22' && @.CidrIp ==
 '0.0.0.0/0')] size greater than 0
'less than' | ' < '
.securityContext.runAsUser < 9999
' equals ' | '=='
.password_reuse_prevention == 0 Operator Usage Examples
'does not equal'
.aws_vpc_peering_connection[*].*[*].peer_vpc_id
 does not equal
 $.resource[*].aws_vpc_peering_connection[*].*[*].vpc_id
'starts with' | 'startsWith'
version startsWith 1.9
'does not start with' | '!startsWith'
version !startWith 1.9
'ends with' | 'endsWith'
.member endsWith \".gserviceaccount.com\
'does not end with' | ' !endsWith'
.member does not end with
 \".gserviceaccount.com\"
'contains'
.Properties.KmsMasterKeyId contains alias/
aws/sqs
'includes one'
'does not contain' | '!contains'
.Properties.KmsMasterKeyId does not contain
 alias/aws/sqs
'is empty' | 'isEmpty'
.Properties.Users[*] is
 emptyor .Properties.Users[*] isempty
'is not empty' | ' !isEmpty'
.Properties.Users[*] is not
 emptyor.Properties.Users[*] !isempty
'any empty' | 'anyEmpty'
'none empty' | 'noneEmpty'
'all empty' | 'allEmpty'
.Properties.KMSKeyId anyEmpty
'any null' | 'anyNull'
.Properties.LoggingConfiguration any
 nullor.Properties.LoggingConfiguration
 anyNull Operator Usage Examples
'exists'
.Properties.VPCOptions exists
'does not exist' | '!exists'
.Properties.VersioningConfiguration does
 not exist
'any start with' | 'anyStartWith'
version anyStartWith 1.9
'none start with' | 'noneStartWith'
version noneStartWith 1.9
'all start with' | 'allStartWith'
.*.node_config[*].image_type allStartWith
 cos
'any end with' | 'anyEndWith'
'none end with' | 'noneEndWith'
'all end with' | 'allEndWith'
.members any end with
 \".gserviceaccount.com\"
'any equal' | 'anyEqual'
'none equal' | 'noneEqual'
.Properties.ContainerDefinitions[*].Privileged
 any equal true
 or .Properties.ContainerDefinitions[*].Memory
 any equal 0or.Properties.AccessControl any
 equal PublicReadWrite
'all equal' | 'allEqual'
' size equals ' | ' size == '
' size does not equal ' | ' size != '
.network_rules[*].bypass allEqual
 \"AzureServices\"
' size greater than ' | ' size > '
' size less than ' | ' size < '
'length equals' | 'length =='
'length does not equal' | 'length !='
'length greater than' | 'length >'
'length less than' | 'length <'
"$.Resources.*[?(@.Type ==
 'AWS::EC2::SecurityGroup')].Properties.SecurityGroupIngress[?
(@.IpProtocol == 'tcp' && @.FromPort ==
 '22' && @.ToPort == '22' && @.CidrIp ==
 '0.0.0.0/0')] size greater than 0  Operator Usage Examples
'any true' | 'anyTrue'
'none true' | 'noneTrue'
' all true' | ' allTrue'
.Properties[?(@.SourceType == 'db-securitygroup')].Enabled anyTrue
'any false' | 'anyFalse'
none false' | 'noneFalse'
' all false' | ' allFalse'
.Properties.IsMultiRegionTrail
 anyFalseor .Properties.IsMultiRegionTrail
 any false
'is true' | 'isTrue'
.client_certificate_config[*].issue_client_certificate
 isTrue
 or.client_certificate_config[*].issue_client_certificate
 is true
'is false' | 'isFalse'
'is type' | 'isType'
'is not type' | '!isType'
queue_properties.*.logging.*.delete
 isFalse
or
queue_properties.*.logging.*.delete is
 false
'is member of' | 'isMemberOf'
spec.containers[*].securityContext.capabilities.add[*]
 is member of (FSETID, SETUID, SETGID,SYS)
'is not member of' | '!isMemberOf'
spec.containers[*].securityContext.capabilities.add[*]
 is not member of (FSETID, SETUID,  SETGID,SYS)
IDENTIFIER '[]'
| IDENTIFIER '[*]'
| IDENTIFIER '[' INT ']'
| IDENTIFIER '[?(' query_expr ')]'
| '[*]'
resource[*].google_compute_subnetwork[*]
 $.Resources.*[?
(@.Type=='AWS::S3::Bucket')].Properties Manage Prisma Cloud Policies
To help find the relevant policies based on role, Prisma Cloud policies are grouped based on
a hierarchy of Category, Class, Type, and Subtype. All of these groupings are available as filters on the Policies page.
The main categories are incidents and risks. An incident is likely a policy that identifies a potential security
issue, while a risk is one that checks for risky configurations. The policy type indicates whether the check
is performed against the network logs, audit logs, configuration logs, or user activity logs. Each policy type
has subtypes for more granularity, for example, Anomaly policies are split into two subtypes—Network and
UEBA. Class is another way to logically group policies into buckets such as Misconfiguration or Privileged
Activity Monitoring.
Category Class Type Subtype
Behavioral Anomaly UEBA
Behavioral Anomaly Network
Privileged Activity
Monitoring
Audit Event Audit
Incident
Network Protection Network Network Event
Risk Misconfiguration Config Run
Misconfiguration Config Build
Use the following workflows to manage Prisma Cloud policies. download policy data, clone, enable, delete, or disable policies from the Policies page.
- To enable global settings for Prisma Cloud default policies click Settings and select
Enterprise Settings.
While some high severity policies are enabled to provide the best security outcomes, by default, policies
of medium or low severity are in a disabled state . To enable policies based on severity, select Auto
enable new default policies of the type—High, Medium, or Low. Based on what enable, Prisma Cloud will scan resources in the onboarded cloud accounts against policies that match the severity
and generate alerts.
For Anomaly policies, have more customizable settings, see Set Up Anomaly Policy Thresholds. When Save changes, choose one of the following options:
- Enable and Save—With Enable and Save, are enabling all existing policies that match your
selection criteria and new Prisma Cloud default policies that are periodically added to the service.
This option allows to enable and scan resources against all existing and new policies to help
stay ahead of threats and misconfigurations.
- Save—With Save, are saving selection criteria and enabling new Prisma Cloud default
policies only as they are periodically added to the service. New policies that match selection, are
automatically enabled and resources are scanned against them after made the change.
- If enable policies of a specific severity, when then clear the checkbox, the policies that were enabled previously are not disabled; going forward, policies that
match the severity cleared are no longer enabled to scan cloud resources
and generate alerts.
If want to disable the policies that are currently active, must disable the status
of each policy on the Policies page.
- The audit logs include a record of all activities performed in Prisma Cloud. To view the audit logs click
Settings and select Audit Logs.
- To view policies, select Policies. -To filter Policies enter a keyword in the Filter Results search box or click Add Filters and select
the filtering criteria.
The filters enable to narrow the search results on the page. The values select within a filter use
the AND operator to display results. Across different filters, the selected values work as OR operators.
To find all Prisma Cloud policies of a specific Policy Subtype, when select the values
Build and Run, view all policies that are classified as Build policies OR Run
policies. To find all policies that are classified as Build and Run, must select the filter
value Build, Run.
- To download the details of policies (or a filtered set of policies) in CSV format so that have an offline copy, click Download. -To enable or disable any policy toggle the Status.
- To edit a custom policy, click the policy and edit the details.
cannot edit a Prisma Cloud Default policy.
- To delete a policy, select the policy and click Delete. -To clone a policy, select the policy and click Clone.
Cloning a policy is creating a copy of an existing policy. Cloning serves as a quick method of creating a new policy if choose to change few details of the source policy.
Prisma Cloud comes with default policies. If want to modify any details, clone a policy and
then modify details.
- To view Alerts associated with a policy click View Alerts. Anomaly Policies
Anomaly policies use audit logs and network flow logs to help identify unusual network and user
activity for all users, and is especially critical for privileged users and assumed roles where detecting unusual
activity may indicate the first steps in a potential misuse or account compromise. These policies rely on
threat feeds to resolve IP addresses to geo-locations and perform user entity behavior analysis (UEBA).
When Prisma Cloud identifies a suspicious IP address, the threat feed enables to classify and view more
information on the malicious IP addresses with which the suspicious IP address is communicating, so quickly figure out which alerts to pay attention to and act on.
Before the service can detect unusual activity for enterprise, must Define Prisma Cloud Enterprise
and Anomaly Settings to specify a training threshold and set the baseline for what are normal trends in
network. To set this baseline, Prisma Cloud gathers information about the user or identities used to
access the monitored cloud accounts, the devices used for access, the IP addresses and locations they come
from, the ports and protocols typically used, the cloud services they use and the frequency, the hours within
which they access these applications, and the activities they perform within the cloud services.
The anomaly policies that are predefined and marked as Prisma Cloud Default policies alert to these
issues:
Account hijacking attempts—Detect potential account hijacking attempts discovered by identifying unusual
login activities. These can happen if there are concurrent login attempts made in short duration from two
different geographic locations, which is impossible time travel, or login from a previously unknown browser, operating system, or location.
Excessive login failures—Detect potential account hijacking attempts discovered by identifying brute force
login attempts. Excessive login failure attempts are evaluated dynamically based on the models observed
with continuous learning.
Unusual user activity—Discover insider threat and an account compromise using advanced data science.
The Prisma Cloud machine learning algorithm profiles a user's activities on the console, as well as the usage
of access keys based on the location and the type of cloud resources.
Network evasion and resource misuse—Detects unusual server port activity or unusual protocol activity
from a client within or outside cloud environment to an server host within or outside network
using a server port or an IP protocol that is not typical to network traffic flows.To identify potential
resource misuse, the anomaly policy monitors when a host inside cloud environment that has no prior
mail-related network activity, starts generating outbound SMTP traffic.
Network reconnaissance—Detect port scan or port sweep activities that probe a server or host for open
ports. The port scanning policies identify when an attacker is performing a vertical scan to find any ports on
a target, and the port sweep detects a horizontal scan where an attacker is scanning for a specific port on
many targets hosts.The policies identify whether the source of the attack internal that is the port scan or
sweep originates from an instance within cloud environment, or external where the source of the port
scan or sweep originates from the internet and targets the cloud environment that is monitored by Prisma Cloud. The policies that detect internal port scan and port sweep activity are enabled by default.
To find all anomaly policies on Prisma Cloud, use the Policy type and Policy Subtypes filters
on Policies. Alerts generated for anomaly policies are grouped by policy and then by user. Because the same IP address
can resolve to different locations at different points in time, if there is an unusual user activity from a previously unseen location for an IP address that has been seen before, Prisma Cloud does not generate an
anomaly alert (and reduces false positives).
If want to add one or more IP addresses as trusted sources. see Trusted IP Addresses
on Prisma Cloud. IP addresses included in the trusted list do not generate alerts for network
based anomaly policies such as network reconnaissance, evasion and resource misuse
policies.
To view alerts generated for an anomaly policy, see Alerts > Overview, and filter for alerts generated
against anomaly policies and get the details on what was identified as unusual or suspicious activity. Note
that multiple alerts of the same type (when a user accesses a resource that is flagged as an anomaly), are logged as a single alert, while a distinct alert is generated if the same user accesses another type of resource.
Alerts generated against the anomaly policies also include additional context based on threat feed
information from Autofocus and Facebook Threat Exchange. Use the tooltip to review the threat details. If
have an AutoFocus license, click the IP address link to launch the AutoFocus portal and search
for a Suspicious IP address directly from the Investigate page, see Use Prisma Cloud to Investigate Network
Incidents. From the alert details use the , to pivot to the Investigate page. For UEBA anomaly policies, can
also see a Trending View of all anomalous activities performed by the entity or user. 275
Investigate Incidents on Prisma Cloud
Prisma Cloud helps visualize entire cloud infrastructure and provides insights into
security and compliance risks.
Prisma Cloud helps connect the dots between configuration, user activity, and network
traffic data, so that have the context necessary to define appropriate policies and create
alert rules.
To conduct such investigations, Prisma Cloud provides with a proprietary query language
called RQL that is similar to SQL.
> Investigate Config Incidents on Prisma Cloud
> Investigate Audit Incidents on Prisma Cloud
> Use Prisma Cloud to Investigate Network Incidents Investigate Config Incidents on Prisma Cloud
Prisma Cloud ingests various services and associated configuration data from AWS, Azure, Alibaba, and GCP
cloud services. retrieve resource information to identify resource misconfigurations, and detect
policy violations that expose business to undue risk and non-compliance to industry benchmarks.
To investigate configuration issues ,use Config queries. enter query in the Search
bar and if the search expression is valid and complete, a green check mark displays along with query
results.
choose to save the searches that have created for investigating incidents in My Saved
Searches. A saved search enables to use the same query at a later time, instead of typing the query
again, and it enables to use the saved search to create a policy.
Saved Searches has list of search queries saved by any Prisma Cloud administrator.
Select a record to view the Audit Trail or Host Findings. The alerts are displayed when select the red
exclamation mark. Hover over the configuration record to see the option to view the details of the resource configuration. also search directly within the JSON Resource configuration to easily find something that is part of the metadata ingested on Prisma Cloud, and speed up investigation.
To analyze configuration events offline, download the event search details in a CSV format, click Download on the right hand corner. Investigate Audit Incidents on Prisma Cloud
Prisma Cloud ingests various services and associated user and event data from AWS, Azure, and GCP cloud
services. investigate console and API access, monitor privileged activities and detect account
compromise and unusual user behavior in cloud environment.
To investigate audit data use Event queries. To build Event RQL queries, enter query in the Search; use the auto-suggest for the attribute json.rule with the operators = and IN, (auto suggestion
is not available for array objects). If the search expression is valid and complete, see a green check
mark and results of query. choose to save the searches that have created for investigating
incidents in My Saved Searches. Use these queries for future reuse, instead of typing the queries all over
again. also use the Saved Searches to create a policy. Saved Searches has list of search queries
saved by any user in the system.
After run event search queries, view the results in Table View, Trending View, or in Map View.
By default see the details in the Table view. To pick the columns in the Table view, use the Column
Picker on the Right hand corner.
From the table view, select View Event Details to see the resource configuration details.
To analyze Audit events offline, download the event search details in a CSV format, click
Download on the right hand corner. Select Trending View to see the results in a timeline. Single click the bubble to view the results for a given
timeline. Double click the bubble to drill down further.
Select Map View to see a World map with pinpoints to the locations where there are activities and
anomalies. view usual activities and anomalous activities to their specific locations. Single click on
the bubble in the map view to view results for the given location. Double click on the bubble in the map
view to drill down further. Use Prisma Cloud to Investigate Network
Incidents
Prisma Cloud ingests and monitors network traffic from cloud services and allows customers to query
network events in their cloud environments. detect when services, applications or databases are
exposed to the internet and if there are potential data exfiltration attempts. Network queries are currently
supported for AWS, Azure and GCP.
To view network traffic data, use Network queries. Enter queries in the Search. If the search
expression is valid and complete, see a green check mark and results of query. choose
to save the searches that have created for investigating incidents in My Saved Searches. Use these
queries for future reuse, instead of typing the queries all over again. also use the Saved Searches to
create a policy. Saved Searches has list of search queries saved by any user in the system.
Network queries enable to search for network resources or network flows. By using packets, bytes, source or destination resource, source or destination IP address, and source or destination port information, these queries enable to monitor traffic and the interconnectivity of the resources that belong to your
cloud accounts and regions.
To download network traffic details for entire network, a node or an instance, or for a specific
connection between a source and a destination node in a CSV format, click Download on the top right hand
corner. This report groups all connection details by port and includes details such as source and destination
IP addresses and names, inbound and outbound bytes, inbound and outbound packets, and whether the node accepted the traffic connection
To see the details of a network resource, click the resource and view Instance Summary, Network
Summary, or Alert Summary.
To see the accepted and rejected traffic, use the Traffic Summary link. Note that the attempted bytes count
displays traffic that is either denied by the security group or firewall rules or traffic that was reset by a host
or virtual machine that received the packet and responded with a RST packet. To view details of a connection, click the connection and click View Details. If the traffic is from a suspicious IP address as characterized by a threat feed, get more details on the threat feed source, when it was classified and reason for classification.
And if have an AutoFocus license, click the IP address link to launch the AutoFocus portal and
search for a Suspicious IP address directly from the Investigate page. 285
Prisma Cloud Compliance
Prisma Cloud enables to view, assess, report, monitor and review cloud infrastructure
health and compliance posture. also create reports that contain summary and detailed
findings of security and compliance risks in cloud environment.
> Compliance Dashboard
> Create a Custom Compliance Standard
> Add a New Compliance Report Compliance Dashboard
The Compliance Overview is a dashboard that provides a snapshot of overall compliance posture
across various compliance standards. Use the Compliance Dashboard as a tool for risk oversight across
all the supported cloud platforms and gauge the effectiveness of the security processes and controls have implemented to keep enterprise secure. also create compliance reports and run them
immediately, or schedule them on a recurring basis to measure compliance over time.
The built-in regulatory compliance standards that Prisma Cloud supports are:
Cloud
Type
Compliance Standards Supported
AWS CIS v1.3, CIS v1.2, CSA CCM v3.0.1,CCPA, GDPR, HITRUST v9.3, HIPAA, ISO 27001:2013, MITRE ATT&CK, Multi-Level Protection Scheme (MLPS) v2.0, NIST 800.53 R4, NIST 800-53
Rev 5, NIST 800-171 Rev1, NIST CSF v1.1,PCI DSS v3.2, PIPEDA, SOC 2, NIST 800-53 Rev 5
Azure CIS v1.1, CSA CCM v3.0.1,CCPA, GDPR, HITRUST v9.3, HIPAA, ISO 27001:2013, MITRE
ATT&CK, Multi-Level Protection Scheme (MLPS) v2.0, NIST 800.53 R4, NIST 800-53 Rev 5, NIST CSF v1.1, PCI DSS v3.2, PIPEDA, SOC 2
GCP CIS v.1.1.0, CIS v1.0.0, CSA CCM v3.0.1,CCPA,GDPR, HITRUST v9.3, HIPAA, ISO
27001:2013, MITRE ATT&CK, NIST 800.53 R4, NIST 800-53 Rev 5, NIST CSF v1.1, PCI DSS
v3.2, PIPEDA, SOC 2
Alibaba Multi-Level Protection Scheme (MLPS) v2.0, NIST 800.53 R4, NIST 800-53 Rev 5
To help easily identify the gaps and measure how you’re doing against the benchmarks defined in the governance and compliance frameworks, the Compliance Dashboard (Compliance > Overview combines
rich visuals with an interactive design. The dashboard results include data for the last full hour. The timestamp on the bottom right corner of the screen indicates when the data was aggregated for the results
displayed.
The compliance dashboard is grouped into three main sections that enable to continuously monitor
progress. -Filters—The left pane provides filters that help sharpen the focus on compliance posture across
different cloud types, accounts, regions, and specific compliance mandates—compliance standards and
the requirements and sections within each standard. The compliance time selector allows to specify
the time range for which want to see compliance posture. By default, the dashboard shows
compliance state as of today. Because the Prisma Cloud service ingests data on all assets in the connected cloud accounts, use this data to audit usage/deployment of resources on each cloud
and measure improvement over time. For example, see how were doing three months ago
and analyze trends in adherence to compliance guidelines today.
- Compliance Score and Charts—The colorful and interactive main section presents the overall health of the cloud resources in organization. The rich visual display helps focus attention on the gaps in compliance for a standard or regulation that is important to you.
- The compliance score presents data on the total unique resources that are passing or failing the policy checks that match compliance standards. Use this score to audit how many unique resources
are failing compliance checks and get a quick count on the severity of these failures. The links allow
to view the list of all resources on the Asset Explorer, and the View Alerts link enables to
view all the open alerts of Low, Medium, or High severity.
- The compliance trendline is a line chart that shows how the compliance posture of your
monitored resources have changed over time (on the horizontal X axis). view the total
number of resources monitored (in blue), and the number of resources that passed (in green) and
failed (in red) over that time period.
- The Compliance coverage sunburst chart highlights the passed and failed resource count across all
compliance standards and enables easy comparison. When click on the inner circle, drilldown to the summary for a specific compliance standard that needs attention; click the center
of the donut to toggle and view all the compliance standards. When click on the outer circle, view the alerts that map to the failed resources associated with a standard.
To review all the details, click the link for the description of the compliance standard.
- Compliance Standards Table—The last section is a list of all the built-in and custom standards that may have defined to monitor and audit organization’s performance. Each row in the table includes
a description of a standard and the total number of policies that map to the standard. It also includes
the total number of unique resources monitored for that standard, the pass and fail count, along with
a percentage of the resources that passed the compliance checks. For each failed check, the severity
of the issue affects where it is counted. For example, if a resource fails a high severity policy, it is not
counted towards a medium or low failure even if it fails a medium or low severity policy rule. To learn about each compliance standard, the requirements/sections that it comprises and the policies
that map to each requirement, use the links in each row. also click the description in the table to
open a new tab that automatically filters the data to display information about the selected compliance
standard and then generate a report on demand. To generate compliance reports, see Add a New
Compliance Report.
Unlike the Asset Inventory that aggregates all resources and displays the pass and
fail count for all monitored resources, the Compliance Dashboard only displays the results
for monitored resources that match the policies included within a compliance standard. For example, even if have 30 AWS Redshift instances, if none of the compliance standards
include policies that check the configuration or compliance and security standards for Redshift instances, the 30 Redshift instances are not included in the resource count on the Compliance Dashboard. The results on the Compliance Dashboard therefore, help focus
attention on the gaps in compliance for a standard or regulation that is important to you.
See Assets, Policies, and Compliance on Prisma Cloud for additional context. Create a Custom Compliance Standard
create own custom compliance standards that are tailored to own business needs, standards, and organizational policies. When defining a custom compliance standard, add
requirements and sections. A custom compliance standard that has a minimum of one requirement and one
section can be associated with policies that check for adherence to standards.
create an all new standard or clone an existing compliance standard and edit it.
- Clone an existing compliance standard to customize.
1. On Prisma Cloud, select Compliance > Standards.
2. Hover over the standard want to clone, and click Clone.
When clone, it creates a new standard with the same name with Copy in the prefix. then
edit the cloned compliance standard to include the requirements, sections, and policies need.
- Create a compliance standard from scratch.
1. On Prisma Cloud, select Compliance > Standards > + Add New. 2. Enter a name and description for the new standard and click Save .
3. Add requirements to custom compliance standard.
1. Select the custom compliance standard just added and click + Add New.
2. Enter a requirement, name and a description and click Save .
4. Add sections to custom compliance standard after adding the requirement.
1. Select the requirement for which are adding the section and click +Add New. 2. Enter a name for the Section a Description and click Save .
Although have added the custom standard to Prisma Cloud, it is not listed on the Compliance
Standards table on Compliance > Overview until add at least one policy to it.
5. Add policies to custom compliance standard.
must associate Prisma Cloud Default policies or custom policies to the compliance standard
to monitor cloud resources for adherence to the internal guidelines or benchmarks that matter
to you. The RQL in the policy specifies the check for the resource configuration, anomaly or event.
1. Select Policies.
Filter the policies want to associate with the standard. filter by cloud type, policy type and policy severity, to find the rules want to attach.
2. Select the policy rule to edit, on 3 Compliance Standards click + and associate the policy with the custom compliance standard.
3. Confirm changes. Add a New Compliance Report
Creating compliance reports is the best way to monitor cloud accounts across all cloud types—AWS, Azure, and GCP—and ensure that are adhering to all compliance standards. create compliance
reports based on a cloud compliance standard for immediate online viewing or download, or schedule
recurring reports so monitor compliance to the standard over time. From a single report, have
a consolidated view how well all of cloud accounts are adhering to the selected standard. Each report
details how many resources and accounts are being monitored against the standard, and, of those, how
many of the resources passed or failed the compliance check. In addition, the report provides detailed
findings for each section of the standard including a description of the requirements in each section, what
resources failed the compliance check, and recommendations for fixing the issues, so that prioritize
what need to do to become compliant. From the Compliance Reports dashboard, also view or
download historic reports so that see compliance trend.
- STEP 1 | Log in to Prisma Cloud.
- STEP 2 | Create a new report.
1. Select Compliance > Overview and select the standard for which want to create a new
compliance report. 2. On the page for the compliance standard selected, click Create Report.
3. Enter the following information and Save the report.
- Enter a descriptive Name for the report.
- Enter the Email address to which to send report when scheduled.
- Select whether want to run the report One Time or Recurring.
If select Recurring must also specify how often want to run the report, the interval, day of the week, and time when want to schedule the recurring report to run.
- STEP 3 | View compliance reports.
After create a compliance report, it will automatically run at the time specified. then
view and manage reports as follows:
- To the list of all compliance reports that have run, select Compliance > Reports. use the filters to narrow the list of compliance reports shown, or search for the report.
- To view a compliance report, click the report name.
A graphical view of the report displays showing the number of unique cloud resources and how
many of them passed and the number and severity of those that failed (also toggle this to
show percentages instead) and a graphical representation of how well cloud accounts are doing
against all sections of the standard. If this report has run before, also see the compliance
trend over time. Finally the report shows summarizes compliance against each requirement of the standard. To drill down into details on a particular requirement of the standard, click the requirement
name.
- If want to refine the report so that it only shows the details are interested, clone it. then use the Compliance filters to customize the report to show only the information are
interested. use the Compliance filters set the report timeframe and narrow the report to only
show compliance information for specific cloud accounts, cloud regions, or cloud types. As add
or remove filters, the report updates so that see changes reflected in the report. When
the cloned report shows the information want it to, click Create Report to saveit as a new report
instance. -Download Report for a PDF of the entire option (unless the report has already been
scheduled for download in which case this option is grayed out). also download the details
about compliance with each requirement of the standard to a CSV file by clicking the download icon.
- also download the compliance reports from the Compliance > Reports page by clicking the Download icon that corresponds the specific report want to download. Note that for recurring
reports, this downloads the most recent report generated.
- For recurring reports, view the report history by clicking the corresponding History icon. then view individual instances of the compliance report, or download them. -To edit the recurrence settings of a report added, or to add or remove email addresses of report
recipients, click the corresponding Edit icon.
- For recurring reports, indicate whether want to automatically include a PDF of the report to the recipients defined, or whether want administrators to be able to download the report on demand rather than emailing it by toggling Enable Scheduling. With this setting enabled, the report will automatically be emailed according to the recurrence schedule defined. With it
disabled, the report will not be emailed, but can be downloaded on demand. 299
Configure External Integrations on Prisma Cloud
integrate Prisma Cloud with third-party services such as Jira, Slack, Splunk, Google
CSCC, Qradar, and ServiceNow to enable to receive, view and receive notification of Prisma Cloud alerts in these external systems. By integrating Prisma Cloud with third-party
services have an aggregated view of cloud infrastructure.
Similarly, Prisma Cloud integration with external systems such as Amazon GuardDuty, AWS
Inspector, Qualys, and Tenable allow to import vulnerabilities and provide additional
context on risks in the cloud.
> Prisma Cloud Integrations
> Integrate Prisma Cloud with AWS Inspector
> Integrate Prisma Cloud with Amazon SQS
> Integrate Prisma Cloud with Amazon GuardDuty
> Integrate Prisma Cloud with AWS Security Hub
> Integrate Prisma Cloud with Azure Service Bus Queue
> Integrate Prisma Cloud with Google Cloud Security Command Center (SCC)
> Integrate Prisma Cloud with Jira
> Integrate Prisma Cloud with Qualys
> Integrate Prisma Cloud with Slack
> Integrate Prisma Cloud with Splunk
> Integrate Prisma Cloud with Tenable
> Integrate Prisma Cloud with ServiceNow
> Integrate Prisma Cloud with Webhooks
> Integrate Prisma Cloud with PagerDuty
> Integrate Prisma Cloud with Microsoft Teams
> Integrate Prisma Cloud with Cortex XSOAR
> Prisma Cloud Integrations—Supported Capabilities Prisma Cloud Integrations
Prisma™ Cloud provides multiple out-of-the-box integration options that use to integrate Prisma Cloud in to existing security workflows and with the technologies already use. The Amazon
GuardDuty, AWS Inspector, Qualys, and Tenable integrations are inbound or pull-based integrations where
Prisma Cloud periodically polls for the data and retrieves it from the external integration system; all other
integrations are outbound or push-based integrations where Prisma Cloud sends data about an alert or
error to the external integration system.
Alibaba Cloud in the Mainland China regions does not support all the integrations listed
below. The supported Integrations are Email, Splunk and Webhooks.
- Amazon GuardDuty—Amazon GuardDuty is a threat detection service that continuously monitors for malicious activity and unauthorized behavior to protect AWS accounts and workloads. Prisma Cloud integrates with Amazon GuardDuty and ingests vulnerability data to provide with additional
context on risks in the cloud.
- AWS Inspector—AWS Inspector assesses applications for exposure, vulnerabilities, and deviations
from best practices. It also produces a detailed list of security findings prioritized by level of severity.
Prisma Cloud integrates with AWS inspector and ingests vulnerability data and Security best practices
deviations to provide with additional context about risks in the cloud.
- AWS Security Hub—AWS Security Hub is a central console where view and monitor the security posture of cloud assets directly from the Amazon console. As the Prisma Cloud application
monitors assets on the AWS cloud and sends alerts on resource misconfigurations, compliance
violations, network security risks, and anomalous user activities, have a comprehensive view of all
cloud assets across all AWS accounts directly from the Security Hub console.
- Amazon SQS—Amazon Simple Queue Service (SQS) helps send, receive, and store messages that
pass between software components at any volume without losing messages and without requiring other
services to be always available. Prisma Cloud can send alerts to Amazon SQS, and set up the AWS CloudFormation service to enable custom workflows.
- Azure Service Bus Queue—Azure Service Bus is a managed messaging infrastructure designed to
transfer data between applications as messages. With the Prisma Cloud and Azure Service Bus queue
integration, send alerts to the queue and set up custom workflows to process the alert payload.
- Cortex XSOAR—Cortex XSOAR (formerly Demisto) is a Security Orchestration, Automation and
Response (SOAR) platform that enables to streamline incident management workflows. With
the Prisma Cloud and Cortex XSOAR integration automate the process of managing Prisma Cloud alerts and the incident lifecycle with playbook-driven response actions.
- Email—Configure Prisma Cloud to send alerts as emails to email account.
- Google Cloud SCC—Google Cloud Security Command Center (SCC) is the security and data risk database
for Google Cloud Platform. Google Cloud SCC enables to understand security and data attack surface by providing inventory, discovery, search, and management of assets. Prisma Cloud integrates with Google Cloud SCC and sends alerts to the Google Cloud SCC console to provide
centralized visibility in to security and compliance risks of cloud assets.
- Jira—Jira is an issue tracking, ticketing, and project management tool. Prisma Cloud integrates with Jira
and sends notifications of Prisma Cloud alerts to Jira accounts.
- Microsoft Teams—Microsoft Teams is cloud-based team collaboration software that is part of the Office
365 suite of applications and is used for workplace chat, video meetings, file storage, and application
integration. The Prisma Cloud integration with Microsoft Teams enables to monitor assets and
send alerts on resource misconfigurations, compliance violations, network security risks, and anomalous
user activities—either as they happen or as consolidated summary cards.
- PagerDuty—PagerDuty enables alerting, on-call scheduling, escalation policies, and incident tracking to
increase the uptime of apps, servers, websites, and databases. The PagerDuty integration enables to send Prisma Cloud alert information to PagerDuty service. The incident response teams can
investigate and remediate the security incidents.
- Qualys—Qualys specializes in vulnerability management security software that scans hosts for potential
vulnerabilities. Prisma Cloud integrates with the Qualys platform and ingests vulnerability data to
provide with additional context about risks in the cloud.
- ServiceNow—ServiceNow is an incident, asset, and ticket management tool. Prisma Cloud integrates
with ServiceNow and sends notifications of Prisma Cloud alerts as ServiceNow tickets.
- Slack—Slack is an online instant messaging and collaboration system that enables to centralize all
notifications. configure Prisma Cloud to send notifications of Prisma Cloud alerts through
slack channels.
- Splunk—Splunk is a software platform that searches, analyzes, and visualizes machine-generated data gathered from websites, applications, sensors, and devices. Prisma Cloud integrates with cloud-based
Splunk deployments and enables to view Prisma Cloud alerts through the Splunk event collector.
Prisma Cloud can integrate with on-premises Splunk instances through the AWS SQS integration.
- Tenable—Tenable.io is a cloud-hosted vulnerability management solution that provides visibility and
insight in to dynamic assets and vulnerabilities. Prisma Cloud integrates with Tenable and ingests
vulnerability data to provide with additional context about risks in the cloud.
- Webhooks—The webhooks integration enables to pass information in JSON format to any thirdparty integrations that are not natively supported on Prisma Cloud. With a webhook integration, configure Prisma Cloud to send alerts to the webhook URL as an HTTP POST request so that any
services or applications that subscribe to the webhook URL receive alert notifications as soon as Prisma Cloud detects an issue.
For the outbound integrations—with the exception of PagerDuty and email, Prisma Cloud performs periodic
checks and background validation to identify exceptions or failures in processing notifications. The status
checks are displayed on the Prisma Cloud administrator console: red if the integration fails validation checks
for accessibility or credentials; yellow if one or more templates associated with the integration are invalid;
or green when the integration is working and all templates are valid. Any state transitions are also displayed
on the Prisma Cloud administrator console to help find and fix potential issues. Integrate Prisma Cloud with Slack
Integrate Prisma™ Cloud with Slack to get instant messages on Slack channels. This will help to
collaborate and centralize all notifications.
- STEP 1 | Set up Slack to get the webhook for application.
1. Log in to web Slack using company URL. For example https://<company-name-orabbreviation>.slack.com/apps/manage.
2. Select Manage > Apps.
3. Select Incoming WebHooks.
4. Add Configuration. 5. Select a channel and Add Incoming WebHooks Integration.
6. Save Settings.
Copy the Webhook URL from this page so specify this URL in Prisma Cloud.
- STEP 2 | Configure Slack in Prisma Cloud and complete set up of the integration channel.
1. Log in to Prisma Cloud.
2. Select Settings > Integrations.
3. Create a +New Integration.
4. Set the Integration Type to Slack.
5. Enter a name and a description for this integration.
6. Enter the WebHook URL. 7. Click Next and then Test.
The status check for Slack displays in red text if Prisma Cloud receives any of the following errors:
user not found or channel not found, channel is archived, action prohibited, posting to general
channel denied, no service or no service ID, or no team or team disabled.
- STEP 3 | Create an Alert Rule or modify an existing rule to send alerts to Slack channels. See Send
Prisma Cloud Alert Notifications to Third-Party Tools. Integrate Prisma Cloud with Splunk
Splunk is a software platform to search, analyze, and visualize machine-generated data gathered from
websites, applications, sensors, and devices.
Prisma™ Cloud integrates with Splunk and monitors assets and sends alerts for resource
misconfigurations, compliance violations, network security risks, and anomalous user activities to Splunk.
- STEP 1 | Set up Splunk HTTP Event Collector (HEC) to view alert notifications from Prisma Cloud in
Splunk.
Splunk HTTP Event Collector (HEC) lets send data and application events to a Splunk deployment
over the HTTP and Secure HTTP (HTTPS) protocols. This helps consolidate alert notifications from
Prisma Cloud in to Splunk so that operations team can review and take action on the alerts.
1. To set up HEC, use instructions in Splunk documentation.
For source type, _json is the default; if specify a custom string on Prisma Cloud, that value will
overwrite anything set here.
2. Select Settings > Data inputs > HTTP Event Collector and make sure see HEC added in the list
and that the status shows that it is Enabled.
- STEP 2 | Set up the Splunk integration in Prisma Cloud.
1. Log in to Prisma Cloud.
2. Select Settings > Integrations.
3. Create a +New Integration.
4. Set Splunk as the Integration Type.
5. Enter an Integration Name and, optionally, a Description.
6. Enter the Splunk HTTP Event Collector URL that set up earlier.
The Splunk HTTP Event Collector URL is a Splunk endpoint for sending event notifications to your
Splunk deployment. either use HTTP or HTTPS for this purpose.
7. Enter Auth Token.
The integration uses token-based authentication between Prisma Cloud and Splunk to authenticate
connections to Splunk HTTP Event Collector. A token is a 32-bit number that is presented in Splunk. 8. (Optional) Specify the Source Type if want all Prisma Cloud alerts to include this custom name in
the alert payload.
9. Click Test and then Save changes.
The integration status check for Splunk displays as red if the event collector URL is not reachable or
times out or if the authentication token is invalid or receives an HTTP 403 response.
- STEP 3 | Create an Alert Rule or modify an existing rule to receive alerts in Splunk. (See Send Prisma Cloud Alert Notifications to Third-Party Tools.) Integrate Prisma Cloud with Amazon SQS
If use Amazon Simple Queue Service (SQS) to enable custom workflows for alerts, Prisma™ Cloud
integrates with Amazon SQS. When set up the integration, as soon as an alert is generated, the alert
payload is sent to Amazon SQS.
The integration gives the flexibility to send alerts to a queue in the same AWS account that may
have onboarded to Prisma Cloud or to a queue in a different AWS account. If want to send alerts to an
SQS in a different AWS account, must provide the relevant IAM credentials—Access Key or IAM Role.
- STEP 1 | Configure Amazon SQS to receive Prisma Cloud alerts.
1. Log in to the Amazon console with the necessary credentials to create and configure the SQS.
2. Click Simple Queue Services (under Application Integration).
3. Create New Queue or use an existing queue.
4. Enter a Queue Name and choose a Queue Type—Standard or FIFO.
5. Click Configure Queue.
For the attributes specific to the Queue, use either the AWS default selection or set them per your
company policies. Use SSE to keep all messages in the Queue encrypted, and select the default AWS
KMS Customer Master Key (CMK) or enter CMK ARN.
6. Create Queue.
This creates and displays SQS Queue 7. Click the Queue that created and view the Details and copy the URL for this queue.
provide this value in Prisma Cloud to integrate Prisma Cloud notifications in to this Queue.
- STEP 2 | If are using a Customer Managed Key to encrypt queues in Amazon SQS, must
configure the Prisma Cloud Role with explicit permission to read the key.
1. On the Amazon console, select KMS > Customer Managed Keys and Create Key.
Refer to the AWS documentation for details on creating keys.
2. Enter an Alias and Description, and add any required Tags and click Next.
3. Select the IAM users and roles who can use this key through the KMS API and click Next.
4. Select the IAM users and roles who can use this key to encrypt and decrypt the data.
5. Review the key policy and click Finish.
- STEP 3 | Enable read-access permissions to Amazon SQS on the IAM Role policy.
The Prisma Cloud IAM Role policy use to onboard AWS setup needs these permissions:
"sqs:GetQueueAttributes", "sqs:ListQueues","sqs:SendMessage",  "sqs:SendMessageBatch", "tag:GetResources"
If used the CFT templates to onboard AWS account and the SQS queue belongs to the same
cloud account, Prisma Cloud IAM Role policy already has the permissions required for Amazon SQS. If
the SQS belongs to a different cloud account, must provide the relevant IAM credentials—Access
Key or IAM Role—when enable the SQS integration in the next step.
- STEP 4 | Set up Amazon SQS integration in Prisma Cloud.
1. Log in to Prisma Cloud.
2. Select Settings > Integrations.
3. Set the Integration Type to Amazon SQS.
4. Enter a Name and Description for the integration.
5. Enter the Queue URL that copied when configured Prisma Cloud in Amazon SQS.
6. (Optional) Select More Options to provide IAM credentials associated with a different AWS account.
By default, Prisma Cloud accesses the SQS queue using the same credentials with which onboarded the AWS account to Prisma Cloud. If queue is in a different account or if want
to authorize access using a separate set of IAM security credentials, pick of the following
options.
Select the IAM Access Key and enter the Access Key and Secret Key, or IAM Role and enter the External ID and Role ARN. This IAM permissions for both options must include sqs:SendMessage and
sqs:SendMessageBatch.
7. Click Next and then Test.
should receive a success message.
8. Click Save.
After set up the integration successfully, if the SQS URL is unresponsive, the status red
(Settings > Integrations) and green when the issue is resolved.
- STEP 5 | Create an Alert Rule or modify an existing rule to enable the Amazon SQS Integration. Integrate Prisma Cloud with Amazon
GuardDuty
Amazon GuardDuty is a continuous security monitoring service that analyzes and processes Virtual Private
Cloud (VPC) Flow Logs and AWS CloudTrail event logs. GuardDuty uses security logic and AWS usage
statistics techniques to identify unexpected and potentially unauthorized and malicious activity.
Prisma™ Cloud integrates with GuardDuty and extends its threat visualization capabilities. Prisma Cloud
starts ingesting GuardDuty data, correlates it with the other information that Prisma Cloud already collects, and presents contextualized and actionable information through the Prisma Cloud app.
- STEP 1 | Enable Amazon GuardDuty on AWS instances (see Amazon Documentation).
- STEP 2 | Enable read-access permissions to Amazon GuardDuty on the IAM Role policy.
The Prisma Cloud IAM Role policy use to onboard AWS setup needs to include these
permissions:
 guardduty:List*, guardduty:Get*
If used the CFT templates to onboard AWS account, the Prisma Cloud IAM Role policy already
has the permissions required for Amazon GuardDuty.
- STEP 3 | After Prisma Cloud has access to the Amazon GuardDuty findings, use the following RQL
queries for visibility into the information collected from Amazon GuardDuty.
Config Query:
config where hostfinding.type = 'AWS GuardDuty Host'
Network Query:
network where dest.resource IN ( resource where hostfinding.type = 'AWS
 GuardDuty Host' )
Click on the resource to see the Audit Trail. Click Host Findings for information related to vulnerabilities. Select AWS GuardDuty Host or AWS
GuardDuty IAM in the filter to view vulnerabilities detected by AWS GuardDuty. Integrate Prisma Cloud with AWS Inspector
Prisma™ Cloud ingests vulnerability data and security best practices deviations from AWS Inspector to
provide organizations with additional context about risks in the cloud. identify suspicious traffic to
sensitive workloads, such as databases with known vulnerabilities.
- STEP 1 | Enable AWS Inspector on EC2 instances. To set up AWS Inspector, see Amazon
documentation.
- STEP 2 | Enable read-access permissions to AWS Inspector on the IAM Role policy.
The Prisma Cloud IAM Role policy that use to onboard AWS setup needs these permissions:
inspector:Describe*
 inspector:List*
If used the CFT templates to onboard AWS account, the Prisma Cloud IAM Role policy already
has the permissions required for AWS Inspector.
- STEP 3 | After the Prisma Cloud service begins ingesting AWS Inspector data, use the following
RQL queries for visibility into the host vulnerability information collected from AWS Inspector.
- Config queries:
config where hostfinding.type = 'AWS Inspector Runtime Behavior Analysis'
config where hostfinding.type = 'AWS Inspector Security Best Practices'
AWS Inspector Runtime Behavior Analysis—Fetches all resources which are in violation of one or
more rules reported by the AWS Runtime Behavior Analysis package. AWS Inspector Security Best Practices—Fetches all resources which are in violation of one or more
rules reported by the AWS Inspector security best practices package.
- Network queries:
network where dest.resource IN ( resource where hostfinding.type = 'AWS
 Inspector Runtime Behavior Analysis' )
network where dest.resource IN ( resource where hostfinding.type = 'AWS
 Inspector Security Best Practices' )
Click on the resource to see an Audit trail.
Click Host Findings for information related to vulnerabilities. Integrate Prisma Cloud with AWS Security Hub
use AWS Security Hub as a central console to view and monitor the security posture of cloud
assets on the Amazon AWS Security Hub console.
Integrate Prisma™ Cloud with AWS Security Hub for centralized visibility into security and compliance risks
associated with cloud assets on the AWS Security Hub console.
As part of the integration, Prisma Cloud monitors assets on AWS cloud and sends alerts about
resource misconfigurations, compliance violations, network security risks, and anomalous user activities
directly to the Security Hub console so that have a comprehensive view of the cloud assets deployed
on AWS accounts.
- STEP 1 | Attach a AWS Security Hub read-only policy to AWS administrator user role to enable
this integration on the Amazon console.
1. Log in to the AWS console and select IAM.
2. Select Users and select the AWS administrator who is creating the integration.
3. Add permissions.
4. Attach existing policies Directly.
5. Select AWSSecurityHubReadOnlyAccess and then Next: Review. 6. Add Permissions.
- STEP 2 | Sign up for Prisma Cloud on AWS Security Hub.
1. Log in to the AWS console and select Security Hub.
2. Select Settings > Integrations and enter Palo Alto Networks as the search term. 3. Find Palo Alto Networks: Prisma Cloud and Enable Integration.
- STEP 3 | Set up the AWS Security Hub Integration on Prisma Cloud.
Set up the AWS Security Hub as an integration channel on Prisma Cloud so that view security
alerts and compliance status for all AWS services from the AWS console.
1. Log in to Prisma Cloud. 2. Select Settings > Integrations.
3. Select +New Integration.
4. Select AWS Security Hub as the Integration Type.
5. Set the Integration Name to the AWS account to which assigned AWS Security Hub read-only
access.
6. Enter a Description and select a Region.
select regions only if enabled Prisma Cloud on AWS Security Hub for cloud
account.
7. Click Next and then Test.
After set up the integration successfully, if there is a permission exception for the enabled regions, the status turns red (Settings > Integrations) and turns green when
the issue is resolved.
- STEP 4 | Modify an existing alert rule or create a new alert rule to specify when to send alert
notifications. (See Send Prisma Cloud Alert Notifications to Third-Party Tools.) - STEP 5 | View Prisma Cloud alerts on AWS Security Hub.
1. Log in to the AWS console and select Security Hub.
2. Click Findings to view the alerts.
3. Select the Title to view details about the alert description. Integrate Prisma Cloud with Azure Service Bus
Queue
Prisma™ Cloud can send alerts to a queue on the Azure Service Bus messaging service. To authorize
access, either use a Shared Access Signature for limiting access permissions to the Service Bus
namespace or queue, or use the service principal credentials associated with the Azure Cloud account have onboarded to Prisma Cloud. If plan to use the service principal that uses Azure Active Directory to
authorize requests, must include the additional role—Azure Service Bus Data Sender— and enable send
access to the Service Bus namespace and queues.
When configured, as soon as an alert is generated, the entire alert payload is sent to the queue.
- STEP 1 | Configure the Azure Service Bus to receive Prisma Cloud alerts.
1. Log in to the Azure portal, to create a Service Bus namespace and add a queue.
Copy the queue URL.
2. Choose authentication method.
To authenticate and authorize access to Azure Service Bus resources, either use Azure
Activity Directory (Azure AD) or Shared Access Signatures (SAS).
- If want to use Azure AD Add the Azure Service Bus Data Sender role to the service principal
associated with the Prisma Cloud App registered on Azure AD tenant.
Refer to the Azure documentation on assigning roles.
- If want to use a SAS Get the connection string to enable Prisma Cloud to authenticate to the Azure Service Bus namespace or queue.
define the scope for the connection string to be the namespace or a specific queue. Refer
to the Azure documentation for getting the connection string.
either use the RootManageSharedAccessKey policy that enables access to the Service
Bus namespace, and is created by default. This policy includes a Shared Access Signature (SAS)
rule with an associated pair of primary and secondary keys that use on Prisma Cloud. Or, limit access to a specific queue, and create a policy with the minimum permissions for send access to the Azure Service Bus queue.
- STEP 2 | Add the Azure Service Bus Queue on Prisma Cloud.
1. Log in to Prisma Cloud.
2. Select Settings > Integrations.
3. Set the Integration Type to Azure Service Bus Queue.
4. Enter a Name and Description for the integration.
5. Enter the Queue URL that copied earlier.
6. Select the method to authorize access to the queue.
- Select Azure Account if want to access the queue with the Prisma Cloud credentials which
used to onboard Azure subscription. If missed adding the Azure Service Bus Data Sender role to the service principal, an error message will display when save the integration.
Select the Azure account from the drop-down.
- Select Shared Access Signature, if want to use a role with limited permissions, and paste the connection string value for the scope selection.
7. Click Next and then Test.
should receive a success message.
And can verify that the message count increments on the queue on the Azure portal. 8. Click Save.
When the communication is successful, the status of the integration is green (Settings >
Integrations). If the Queue URL is unreachable or if permissions are insufficient, the status turns red.
- STEP 3 | Create an Alert Rule or modify an existing rule to enable the Azure Service Bus Queue
integration. Integrate Prisma Cloud with Jira
Integrate Prisma™ Cloud with Jira and receive Prisma Cloud alert notifications in Jira accounts.
With this integration, automate the process of generating Jira tickets with existing security
workflow.
To set up this integration, need to coordinate with Jira administrator and gather the inputs needed
to enable communication between Prisma Cloud and Jira.
This integration supports Jira Cloud and Jira On-Premises versions, and is qualified with the most recent GA
versions of Jira.
- STEP 1 | Configure Prisma Cloud in Jira account.
1. Login to Jira as a Jira Administrator.
2. Locate Application Links.
For Jira Cloud, select Jira Settings > Products > Application Links.
For Jira On-Premises, select Settings > Applications > Application Links. 3. Enter the URL for instance of Prisma Cloud in Configure Application Links and Create new link.
See Access Prisma Cloud for details on the URL.
4. Disregard the message in Configure Application URL and Continue.
5. Enter the Application Name and set the Application Type to Generic Application.
6. Create incoming Link and Continue. 7. On Link Applications, specify a Consumer Key and a Consumer Name. Save the Consumer Key
because will need this value when enter the information in Prisma Cloud.
8. Copy the Public Key shown below and Continue.
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAnYoXB
+BZ555jUIFyN+0b3g7haTchsyeWwDcUrTcebbDN1jy5zjZ/vp31//
L9HzA0WCFtmgj5hhaFcMl1bCFY93oiobsiWsJmMLgDyYBghpManIQ73TEHDIAsV49r2TLtX01iRWSW65CefBHD6b/1rvrhxVDDKjfxgCMLojHBPb7nLqXMxOKrY8s1yCLXyzoFGTN6ankFgyJ0BQh
+SMj/hyB59LPVin0bf415ME1FpCJ3yow258sOT7TAJ00ejyyhC3igh
+nVQXP+1V0ztpnpfoXUypA7UKvdI0Qf1ZsviyHNwiNg7xgYc
+H64cBmAgfcfDNzXyPmJZkM7cGC2y4ukQIDAQAB Prisma Cloud is listed in Jira account after successful creation.
- STEP 2 | Setup Jira as one of the integration channels in Prisma Cloud.
1. Login to Prisma Cloud.
2. Select Settings > Integrations.
3. + Add New integration.
4. Set Integration to JIRA.
5. Specify a meaningful Integration Name and, optionally, add a Description.
6. Enter the JIRA Login URL.
Make sure the URL starts with https and does not have a trailing slash (‘/’) at the end.
7. Enter the Consumer Key that created when created the Prisma Cloud application in Jira and
Generate Token.
8. After see the Token Generated message, click Next. 9. Click the secret key URL link to retrieve secret key.
The URL with the verification code is valid for only 10 minutes. 10.When redirected to the Welcome to JIRA page, Allow Prisma Cloud read and write access to data in
Jira account.
11.Copy the verification code displayed on the page, paste it as the Secret Key, and Generate Token, Test and Save. The integration will be listed on the Integrations page.
- STEP 3 | Create Jira notification templates to configure and customize Prisma Cloud alerts.
The Jira fields that are defined as mandatory in project are displayed in the template. The types of fields in Jira (such as text, list, single select check boxes, and option type fields) are supported in Prisma Cloud. If add any other type of fields as mandatory in Jira (such as date fields), it will fail. Do not
configure any Date fields as Mandatory and define any text fields in Jira as free-form text so that alert
data is displayed correctly and completely.
1. Log in to Prisma Cloud.
2. From Alerts, select Notification Templates and +Add New template.
3. Select the Jira Notification template and enter a Template Name.
4. Select an Integration.
5. Select Project.
Select the project where want to see the Prisma Cloud alerts. Because every alert
translates to a Jira ticket, as a best practice, create and use a dedicated project for Prisma Cloud ticketing and issue management.
6. Select Issue Type and click Next. 7. Select the Jira Fields that would like to populate.
The Jira fields that are defined as mandatory in project are already selected and included in the alert.
8. Select information that goes in to Summary and Description from the alert payload.
9. Select the Reporter for alert from users listed in Jira project.
This option is available only if the administrator who set up this integration has the appropriate
privileges to modify the reporter settings on Jira. 10.Click Next to go to the review dialog and review selection.
11.Save changes.
delete or edit the Jira notification from the Notification Template dialog. After set up the integration, Prisma Cloud performs periodic status checks with
Jira red if the Jira Login URL is not reachable or if any request to Jira results in an
HTTP 400 or 403 response.
- STEP 4 | Create an Alert Rule or modify an existing rule to send alerts to Jira. Integrate Prisma Cloud with Qualys
Prisma™ Cloud integrates with the Qualys platform to ingest and visualize vulnerability data for your
resources that are deployed on the AWS and Azure cloud platforms.
- STEP 1 | Gather the information that need to set up the Qualys integration on Prisma Cloud.
must obtain the Qualys Security Operations Center (SOC) server API URL (also known as or
associated with a POD—the point of delivery to which are assigned and connected for access to
Qualys).
Get the API URL from Qualys account (Help > About). The Qualys API URL is listed under
Qualys Scanner Appliances. When enter this URL in as the Qualys API Server URL, do not
include :443.
must provide Qualys users with the privileges required to enable the integration using the Manager role, the Unit Manager role, or both. modify the Manager role to enable readonly access permission if needed. (Refer to the Qualys documentation for details about User Roles
Comparison (Vulnerability Management).)
must enable Vulnerability Management (VM), Cloud Agent (CA), and Asset View (AV) for Qualys
users.
must enable Qualys API and Qualys EC2 API access for Qualys users.
(AWS only) must configure Qualys Sensors for AWS cloud, such as Virtual Scanner Appliances, Cloud Agents, AWS Cloud Connectors, and Internet Scanners.
The cloud agents or cloud connectors enable Prisma Cloud to retrieve vulnerability data so that correlate this data with AWS asset inventory. (Refer to the Qualys documentation for more
information.) (Azure only) For Azure accounts, deploy the Qualys Virtual Scanner Appliance using Microsoft Azure
Resource Manager (ARM) (see the Qualys documentation).
use Qualys Cloud Agents (Windows and Linux) for Azure instances from the Azure Security
Center console to view vulnerability assessment findings within Azure Security Center and your
Qualys subscription (see Qualys Documentation.)
(Azure only) Make sure that Azure VM Information is visible in Qualys.
- STEP 2 | Set up Qualys Integration on Prisma Cloud.
1. Select Settings > Integrations.
2. Create a +New Integration.
3. Set the Integration Type to Qualys.
4. Enter an Integration Name and Description.
5. Enter the Qualys API Server URL (without http[s]).
This is the API URL for Qualys account. When enter this URL, do not include the protocol
(http(s)) or the port (:443).
6. Enter Qualys User Login and Password. 7. Save changes.
The integration will be listed on the Integrations dialog, where enable, disable, or delete
integrations as needed.
- STEP 3 | View Qualys host vulnerability data in Prisma Cloud.
After configure Prisma Cloud with access to the Qualys findings, use RQL queries for visibility in to the host vulnerability information collected by Qualys.
1. Use Config Query for visibility for host vulnerabilities.
config where hostfinding.type = 'Host Vulnerability'  Click a resource to get information about vulnerabilities. View the Audit Trail to see the CVE
numbers.
Click Host Findings for information related to vulnerabilities. The Source column in Host Findings
displays the Qualys icon to help easily identify the source for the vulnerability findings. Network Query
network where dest.resource IN ( resource where hostfinding.type = 'Host
 Vulnerability' )
- STEP 4 | Use the Qualys APIs on the CLI to confirm if API access is enabled for account.
If have trouble connecting with Qualys API, enter username, password, and the URL for the Qualys service in the following Curl examples:
 curl -H “X-Requested-With: Curl Sample” -u “Username:Password”
 “https://qualysapi.qg1.apps.qualys.in/api/2.0/fo/scan/?
action=list&echo_request=1”

 curl -k “https://qualysapi.qg1.apps.qualys.in/msp/
asset_group_list.php” -u “Username:Password”
 curl -k -H “X-Requested-With:curl” “https://
qualysapi.qg1.apps.qualys.in/api/2.0/fo/scan/stats/?action=list” -u
 “Username:Password”
  Integrate Prisma Cloud with Google Cloud
Security Command Center (SCC)
Integrate Prisma™ Cloud with Google Cloud Security Command Center (SCC) for centralized visibility in to
security and compliance risks associated with cloud assets on the Google Cloud Platform (GCP).
set up this integration for a GCP Organization that are monitoring with Prisma Cloud. The alerts generated by Prisma Cloud for GCP accounts based on alert rule are posted to Google Cloud
SCC. To show Prisma Cloud alerts in Google Could SCC for cloud accounts of other cloud types (such as AWS and Azure), contact Prisma Cloud support on the Palo Alto Networks LIVE Community.
- STEP 1 | The service account use to onboard the GCP Organization in to Prisma Cloud should
include Viewer, Organization Viewer, and Security Center Findings Editor roles.
- STEP 2 | To view assets and findings on the Cloud SCC console, enable the Cloud Security Command
Center API.
1. Go to the GCP Console API Library and select GCP project.
Make sure to enable the Cloud Security Command Center API in the project owns the Service
Account that will use to onboard the GCP Organization into Prisma Cloud.
2. Enable APIs and Services.
3. Enable the Cloud Security Command Center API.
- STEP 3 | Sign up for the Prisma Cloud SCC solution on the Google console.
A security center administrator can set up this integration on the Google console.
1. Go to the Google Console and search for Prisma Cloud CSCC.
2. Visit Palo Alto Networks site to Signup.
3. Select the organization that onboarded in to Prisma Cloud.
4. Select the Service account used to onboard the GCP Organization. 5. Copy the Source ID. need the Source ID when set up this integration in Prisma Cloud.
6. Click Done.
- STEP 4 | Set up Google Cloud SCC as one of the integration channels in Prisma Cloud.
1. Log in to Prisma Cloud.
2. Select Settings > Integrations.
3. Create a +New Integration.
4. Select CSCC as the Integration Type.
5. Specify a meaningful Integration Name and Description.
6. Enter the Source ID that copied Prisma Cloud 7. Select the GCP Organization.
8. Click Next and then Test.
For a successful integration, must configure adequate permissions for the service account (as listed above). After successfully set up the integration, the status Settings > Integrations) turns
red when there are any issues and green when there are no issues or all issues are resolved.
- STEP 5 | Create an Alert Rule or modify an existing rule to send alerts to Google Cloud SCC. See Send
Prisma Cloud Alert Notifications to Third-Party Tools.
- STEP 6 | View alerts in Cloud SCC.
1. Go to the Google Console and select Security > Security Command Center.
2. Click Findings to view the alerts. 3. Select the rule to see the details about the alerts. Integrate Prisma Cloud with Tenable
Prisma™ Cloud ingests vulnerability data from Tenable to provide with additional context about risks
in the cloud. This integration enables to, for example, identify suspicious traffic to sensitive workloads, such as databases with known vulnerabilities.
AWS, Azure, and GCP clouds support the Prisma Cloud integration with Tenable.
- STEP 1 | Tenable.IO provides API access to assets and their vulnerability information. Configure the Tenable account to use the Tenable AWS, Azure, and GCP connectors. Without connectors, cannot identify the cloud resource.
The Tenable API requires an access key and secret key in the header. Generate an access key and secret
key per user on the Tenable.io app. (See Tenable documentation for information.) Also, make sure that
the Tenable role that use to enable this integration has administrator permissions that include vulnsrequest-export and assets-request-export API access.
- STEP 2 | Set up Tenable integration on Prisma Cloud.
1. Select Settings > Integrations.
2. Set the Integration Type to Tenable.
3. Enter an Integration Name and Description.
4. Enter the Access Key and the Secret Key that are generated in Tenable.io.
See Tenable documentation for information.
5. Click Next and Test the integration.
- STEP 3 | View vulnerabilities detected by Tenable in Prisma Cloud.
1. After Prisma Cloud has access to the Tenable findings, use the following RQL queries for visibility into the host vulnerability information collected from Tenable.
Config Query config where hostfinding.type = 'Host Vulnerability' AND
 hostfinding.source = 'Tenable' AND hostfinding.severity = 'high'
Select a resource to get information about vulnerabilities. Select Audit Trail to view the CVE
numbers.
Network Query network where dest.resource IN ( resource where hostfinding.type = 'Host
 Vulnerability' )
Click Host Findings to see details. Integrate Prisma Cloud with ServiceNow
Integrate Prisma™ Cloud with ServiceNow and get automatically notified about Prisma Cloud alerts through
ServiceNow tickets to prioritize incidents and vulnerabilities that impact business. Prisma Cloud
integrates with the ITSM module (incident table), the Security Incident Response module (sn_si_incident
table), and the Event Management modules (em_event table) on ServiceNow to generate alerts in the form
of ITSM Incident, Security Incident, and Event tickets. After enable the integration, when Prisma Cloud
scans cloud resources and detects a policy violation, it generates an alert and pushes it to ServiceNow
as a ticket. When dismiss an alert on Prisma Cloud, Prisma Cloud sends a state change notification
to update the ticket status on ServiceNow. This integration seamlessly fits in to the existing workflows
for incident management (ITSM),security operations management (Security Incident Response) or event
management for organization.
The Prisma Cloud integration with ServiceNow is qualified with the most recent GA versions of ServiceNow.
If are using a ServiceNow developer instance, make sure that it is not hibernating.
1. Set Up Permissions on ServiceNow
2. Enable the ServiceNow Integration on Prisma Cloud
3. Set up Notification Templates
4. View Alerts If see errors, review how to Interpret Error Messages.
Set Up Permissions on ServiceNow
To integrate Prisma Cloud and ServiceNow, must have the privileges on ServiceNow to configure users, roles, fields on ServiceNow, which then allow to set up the data mapping for the Notification Templates
on Prisma Cloud.
If do not have the privileges required listed below, must work with ServiceNow administrator.
- Prerequisites for the Prisma Cloud and ServiceNow Integration
1. must have permissions to create a local user account on ServiceNow.
Create a Username and password that are local on the instance itself. A local user account
is a requirement because the ServiceNow web services cannot authenticate against an LDAP
or SSO Identity provider and it is unlike the authentication flow that ServiceNow supports for typical administrative users who access the service using a web browser.Refer to the ServiceNow
documentation for more information. 2. Review the ServiceNow roles required.
PrismaCloud has verified that the following roles provide the required permissions. If your
implementation has different roles and RBAC mechanisms, work with ServiceNow
administrator.
New York, Orlando, and Paris
- (Optional)personalizefor accessing tables.
Personalize role is recommended to support type-ahead fields in notification templates for ServiceNow on Prisma Cloud. With this permission, when enter a minimum of three
characters in a type-ahead field, this role enables to view the list of available options. If do not enable personalize permissions, must give table specific read-access permissions for type-ahead inputs.
- 2. evt_mgmt_integration basic role has create access to the Event [em_event] and Registered
Nodes [em_registered_nodes] tables to integrate with external event sources.
3. For the user added earlier, create a custom role with the permissions listed above.
These permissions are required to create tickets and access the data in the respective ITSM, Events, and Security Incident Response tables and fields on ServiceNow.
Prisma Cloud needs access to the Plugins (V_plugin), Dictionary (sys_dictionary), and Choice Lists
(sys_choices) tables to fetch data from the ServiceNow fields. view this information in the ServiceNow notification templates that enable to customize Prisma Cloud alerts in ServiceNow.
1. Select User Administration > Roles to create a new role and assign it to the local administrative
user created earlier.
2. Pick a table, such as the Plugins table, and select the menu (“hamburger”) icon next to a table
column header to Configure > Table.
3. Elevate the role to security_admin to enable modification of the access control list (ACL).
4. Select Access Controls > New. 5. Set Operation to Read and assign this permission to the role.
6. Enable permissions for the remaining tables and assign them to the same role.
Verify that all three tables—Plugins (V_plugin), Dictionary (sys_dictionary), and Choice Lists
(sys_choices) have the role and the required permission especially if have defined field-level
ACL rules to restrict access to objects in ServiceNow implementation.
4. must be familiar with the fields and field-types in ServiceNow implementation to set up
the Notification templates on Prisma Cloud. Because this knowledge is essential for setting up the mapping of the Prisma Cloud alert payload to the corresponding fields on ServiceNow, must
work with ServiceNow administrator to successfully enable this integration.
- Prerequisites for the Security Incident Module
The Security Incident Response plugin is optional but is required if want to generate Security
Incident tickets. To create Security Incident tickets, must also have the Security Incident Response
plugin installed on ServiceNow instance.
Verify that the Security Incident Response plugin is activated. To activate a plugin must be
ServiceNow administrator; if do not see the plugin in the list, verify that have purchased the subscription.
- Prerequisites for the Event Management Module
The Event Management plugin is optional but is required if want to generate Event tickets on
ServiceNow. To create Event tickets, must have the Event Management subscription and the plugin
installed on ServiceNow instance.
Verify that the Event Management plugin is activated. To activate a plugin must be ServiceNow
administrator; if do not see the plugin in the list, verify that have purchased the subscription.
Enable the ServiceNow Integration on Prisma Cloud
Set up ServiceNow as an external integration on Prisma Cloud.
- STEP 1 | Log in to Prisma Cloud and select Settings > Integrations > +Add New.
- STEP 2 | Set the Integration Type to ServiceNow.
- STEP 3 | Enter a meaningful Integration Name and a Description.
- STEP 4 | Enter FQDN for accessing ServiceNow. Make sure to provide the FQDN for ServiceNow—not the SSO redirect URL or a URL that enables to bypass the SSO provider (such as sidedoor or login.do) for local authentication on ServiceNow. For example, enter <yourservicenowinstance>.com and not any of the following:
https://www.<yourservicenowinstance>.com
<yourservicenowinstance>.com/
<yourservicenowinstance>.com/sidedoor.do
<yourservicenowinstance>.com/login.do
If switch the FQDN from one ServiceNow instance to another, state change
notifications for existing alerts will fail.
- STEP 5 | Enter the Username and Password for the ServiceNow administrative user account.
The ServiceNow web services use the SOAP API that supports basic authentication, whereby the administrative credentials are checked against the instance itself and not against any LDAP or SSO
Identity provider. Therefore, must create a local administrative user account and enter the credentials for that local user account here instead of the SSO credentials of the administrator. This
method is standard for SOAP APIs that pass a basic authentication header with the SOAP request.
- STEP 6 | Select the Service Type for which want to generate tickets—Security, Incidents, and/or
Event.
must have the plugin installed to create Security incident tickets or Event tickets; make sure to work
with ServiceNow administrator to install and configure the Security Incident Response module or
Event Management module. If select Security only, Prisma Cloud generates all tickets as Security
Incident Response (SIR) on ServiceNow.
- STEP 7 | Click Next and then Test.
If have omitted any of the permissions listed in Set Up Permissions on ServiceNow, an HTTP 403
error displays.
- STEP 8 | Test and Save the integration. Continue with setting up the notification template, and then verify the status of the integration on
Settings > Integrations.
Set up Notification Templates
Notification templates allow to map the Prisma Cloud alert payload to the incident fields (referred to as ServiceNow fields on the Prisma Cloud interface in the screenshot) on ServiceNow instance. Because
the incident, security, and event tables are independent on ServiceNow, to view alerts in the corresponding
table, must set up the notification template for each service type — Incidents ,Events or Security
Incidents on Prisma Cloud.
- STEP 1 | Log in to Prisma Cloud and select Alerts > Notification Templates.
- STEP 2 | Add New notification template, and choose the template for ServiceNow.
- STEP 3 | Enter a Template Name and select Integration.
Use descriptive names to easily identify the notification templates.
- STEP 4 | Set the Service Type to Security ,Incident or Event.
The options in this drop-down match what selected when enabled the ServiceNow integration
on Prisma Cloud.
- STEP 5 | Click Next and select the alert status for which want to set up the ServiceNow fields.
choose different fields for the Open, Dismissed, or Resolved states. The fields for the Snoozed
state are the same as that for the Dismissed state. - STEP 6 | Select the ServiceNow Fields that want to include in the alert.
Prisma Cloud retrieves the list of fields from ServiceNow instance dynamically, and it does not
store any data. Depending on how IT administrator has set up ServiceNow instance, the configurable fields may support a drop-down list, long-text field, or type-ahead. For a type-ahead field, must enter a minimum of three characters to view a list of available options. When selecting the configurable fields in the notification template, at a minimum, must include the fields that are
defined as mandatory in ServiceNow implementation.
In this example, Description is a long-text field, hence select and include the Prisma Cloud
Alert Payload fields that want in ServiceNow Alerts. must include a value for each field
select to make sure that it is included in the alert notification. See Alert Payload for details on the context include in alerts.
To generate a ServiceNow Event, Message Key and Severity are required. The Message
key determines whether to create a new alert or update an existing one, and map
the Message Key to Account Name or to Alert ID based on preference for logging
Prisma Cloud alerts as a single alert or multiple alerts on ServiceNow. Severity is required
to ensure that the event is created on ServiceNow and can be processed without error;
without severity, the event is in an Error state on ServiceNow.
For Number, use AlertID from the Prisma Cloud alert payload for ease of scanning and readability of incidents on ServiceNow.
- STEP 7 | Click Next to go to the review pane and review selection.
- STEP 8 | Test and Save changes.
After set up the integration and configure the notification template, Prisma Cloud uses this template
to send alerts to ServiceNow instance. When the communication is successful, the status of the integration is green on Settings > Integrations. If the ServiceNow instance URL is unreachable or if
credentials are invalid, the status turns red. When a failure occurs, Prisma Cloud performs periodic
checks to verify the connection status. The status, however, does not transition to red if:
Prisma Cloud cannot resolve the alert or update an existing alert field for a deleted record
or missing record on ServiceNow instance.
Prisma Cloud is unable to send a test message to ServiceNow because of an HTTP 404
error.
Interpret Error Messages
The following table displays the most common errors when enable the ServiceNow integration on
Prisma Cloud.
What is Wrong? Error Message that Displays
The ServiceNow URL entered is
incorrect.
must provide an IP
address or an FQDN without
the protocol http or https
invalid_snow_base_url
The ServiceNow URL entered is
invalid.
The FQDN is invalid it should
be a valid host name or IP
address.
invalid_snow_fqdn
The ServiceNow URL entered is
not reachable.
The FQDN provided is either
not reachable or is an invalid
ServiceNow instance.
snow_network_error
A required field is missing in the ServiceNow configuration.
Missing Required Field -
{{param}}
missing_required_param,  subject - {{param}}
ServiceNow username or
password is not valid or is inaccurate.
Invalid Credentials invalid_credentials
The ServiceNow permissions have enabled are not adequate.
Required roles or Plugins is/
are missing for {{table}}
missing_role_or_plugin,  subject - {{table}}
Insufficient permission to read
the field from {{table}} table
insufficient_permission_to_read,  subject - {{table}}
The Notification template for this
integration does not have adequate
permissions.
Error Fetching Suggestions For {{table}}
error_fetching_fields_for,  subject - {{table}}
The ServiceNow integration is not
successfully configured.
Failed Service Now Test -
{{reason}}
failed_service_now_test,  subject -
 {{reason}} View Alerts Verify that the integration is working as expected. On the incidents view in ServiceNow, add the Created
timestamp in addition to the same columns enabled in the Prisma Cloud notification template to easily
correlate alerts across both administrative consoles.
- STEP 1 | Modify an existing Alert Rule or create a new Alert Rule to send alert notifications to
ServiceNow. (See Send Prisma Cloud Alert Notifications to Third-Party Tools.)
- STEP 2 | Login to ServiceNow to view Prisma Cloud alerts.
When alert states are updated in Prisma Cloud, they are automatically updated in the corresponding
ServiceNow tickets.
1. To view incidents (incident table), select Incidents.
In ServiceNow, all the Open Prisma Cloud have an incident state of New and all the Resolved or
Dismissed alerts have an incident state of Resolved.
2. To view security incidents (sn_si_incident table), select Security Incidents.
In ServiceNow, all the Open Prisma Cloud alerts have a state of Draft and all the Resolved or
Dismissed alerts have a state of Review. 3. To view event incidents (events table), select Event Management > All Events. Integrate Prisma Cloud with Webhooks
Integrate Prisma™ Cloud with webhooks to send Prisma Cloud alerts to webhooks and pass information
to any third-party integrations that are not natively supported on Prisma Cloud. incident response
teams can monitor the webhook channels to investigate and remediate security incidents. With a webhook
integration, configure Prisma Cloud to send information to the webhook as an HTTP POST request
as soon as an alert is generated. And if have internal services or applications that subscribe to the webhook, these subscribing clients can get data immediately in JSON format.
- STEP 1 | Obtain Webhook URL.
If have additional details that want to include in the payload to enable additional security or to
verify the authenticity of the request, include these as key-value pairs in a custom header.
- STEP 2 | Set up webhooks as an integration channel on Prisma Cloud.
1. Log in to Prisma Cloud and select Settings > Integrations.
2. +Add New integration.
3. Set the Integration Type to webhooks.
4. Enter Webhook URL.
5. Add any custom HTTP Headers as key-value pairs.
can, for example, include an authentication token in the custom header. The integration includes
Content-Type as a default header and cannot edit it.
6. Test and Save the integration.
After set up the integration successfully, the status (Settings > Integrations) turns red when the webhook URL is unreachable or when Prisma Cloud cannot authenticate to it successfully and turns
green when there aren’t any issues or the issues are resolved.
- STEP 3 | Modify an existing alert rule or create a new alert rule to send alert notifications to webhook.
(See Send Prisma Cloud Alert Notifications to Third-Party Tools.)
- STEP 4 | View the alert POST requests on Webhook.
In one POST request, alerts are sent in a batch of 30. Integrate Prisma Cloud with PagerDuty
Integrate Prisma™ Cloud with PagerDuty to aid alerting, on-call scheduling, escalation policies, and incident
tracking to increase uptime of apps, servers, websites, and databases. When integrated, Prisma Cloud
sends alerts to the PagerDuty service so that incident response teams are notified to investigate and
remediate the security incidents.
- STEP 1 | Add a new service in PagerDuty and get the integration key. have to provide this
integration key in Prisma Cloud.
1. Log in to PagerDuty.
2. Click Configuration > Services and add a +New Service.
3. Complete the Add a Service form.
4. In the Integration Settings, set the Integration Type to Use our API Directly and select Events API
V2.
5. After fill out all the details, Add Service.
6. Copy and save the Integration Key .
will need to enter this integration key in Prisma Cloud when add this integration. For more information about integrations with PagerDuty, see PagerDuty Documentation.
- STEP 2 | Set up PagerDuty as an integration channel on Prisma Cloud.
1. Log in to Prisma Cloud and select Settings > Integrations.
2. +Add New integration.
3. Set the Integration Type to pagerduty.
4. Enter the Integration Key of service on PagerDuty.
5. Click Next and then Test.
6. Save the integration.
Prisma Cloud creates a test incident and sends it to service in PagerDuty. To make sure that this
integration is successful, look for the test integration in PagerDuty Service. - STEP 3 | Modify an existing alert rule or create a new alert rule to send alert notifications to PagerDuty.
(See Send Prisma Cloud Alert Notifications to Third-Party Tools.)
- STEP 4 | View Prisma Cloud in PagerDuty.
In PagerDuty, all the open alerts display the Incident State as Triggered and all the resolved alerts display
the Incident State as Resolved. Integrate Prisma Cloud with Microsoft Teams
Microsoft Teams is a cloud-based team collaboration software that is part of the Office 365 suite of applications and is used for workplace chat, video meetings, file storage, and application integration.
Prisma™ Cloud integrates with Microsoft Teams and monitors assets and sends alerts on resource
misconfigurations, compliance violations, network security risks, and anomalous user activities either as they happen or as a consolidated summary card—determined by how configure alert notifications.
Each alert message is a webhook notification that includes details such as the cloud type, policy name, and
the resource that triggered the alert and the message card indicates the severity with a red (high), yellow
(medium) or gray (low) line for easy scanning.
- STEP 1 | Set up Microsoft Teams to view alert notifications from Prisma Cloud.
must create an incoming webhook connector on a new channel or on a pre-existing channel on
Microsoft Teams to enable the integration with Prisma Cloud. This webhook channel helps consolidate
alert notifications from Prisma Cloud in Microsoft Teams so that operations team can review and
take action on the alerts. To enable this integration, must have administrative privileges or contact
the Microsoft 365 administrator who manages Team settings for organization.
1. In Microsoft Teams, click More options (...) next to the channel name and select Connectors.
2. Scroll to Incoming Webhook, Add a webhook, and Install it.
3. Enter a name for the webhook and Create it.
4. Copy the webhook URL to the clipboard and save it before click Done.
- STEP 2 | Set up Microsoft Teams on Prisma Cloud.
1. Log in to Prisma Cloud.
2. Select Settings > Integrations.
3. +Add New integration.
4. Set Microsoft Teams as the Integration Type.
5. Enter the Integration Name and, optionally, a Description.
6. Paste the Webhook URL that previously copied from Microsoft Teams.
7. Test and then Save.
After successful integration in the Microsoft Teams conversation, will receive a test message card with a green line.
- STEP 3 | Create an Alert Rule or modify an existing rule to receive alerts in Microsoft Teams. (See Send
Prisma Cloud Alert Notifications to Third-Party Tools.)
The message card includes information on the policy rules and the resource names that have violated
the policy and it includes a direct link with the relevant filters to access Prisma Cloud and view the alert
or scheduled notification summary directly in the app. Integrate Prisma Cloud with Cortex XSOAR
With the Prisma™ Cloud and Cortex XSOAR (formerly Demisto) outbound or push-based integration, send a Prisma Cloud alert generated by a policy violation to Cortex XSOAR. This integration
enables Security operations team to define custom playbooks or use the out-of-box playbooks on
Cortex XSOAR to create multi-step workflows for incident management of cloud resources; this is an
alternative to the pull-based integration that configure from Cortex XSOAR.
Using the policy ID in the alert, Cortex XSOAR categorizes the alert as a specific incident type. For an
incident type, the Prisma Cloud alert payload is mapped to a Cortex XSOAR layout that specifies the incident fields for data classification and mapping on Cortex XSOAR. The current list of incident types are:
AWS CloudTrail Misconfiguration, AWS EC2 Instance Misconfiguration, AWS IAM Policy Misconfiguration, and Prisma Cloud. If the policy ID is not categorized to a specific incident type, it is automatically mapped
to the generic Prisma Cloud incident type. Every incident type is mapped to a Cortex XSOAR layout and
associated with a playbook to enable autoremediation of the violating resource, except for the generic
Prisma Cloud incident type.
On autoremediation, Prisma Cloud performs a scan that detects that the issue is resolved and marks the alert as resolved.
Currently, this integration does not support the use of notification templates and Prisma Cloud does not
receive state change notifications from Cortex XSOAR after it resolves an open alert.
- Enable the Cortex XSOAR Integration on Prisma Cloud
- Set Up the Integration on Cortex XSOAR
Enable the Cortex XSOAR Integration on Prisma Cloud
Set up Cortex XSOAR as an external integration on Prisma Cloud. If have a firewall or cloud Network
Security Group between the internet and Cortex XSOAR, must add the NAT Gateway IP Addresses for Prisma Cloud to the allow list and enable the connection to Prisma Cloud.
For the push-based integration, must use Cortex XSOAR version 5.0.0 and content release version
19.10.2 or later.
- STEP 1 | Log in to Prisma Cloud and select Settings > Integrations > +Add New.
- STEP 2 | Set the Integration Type to Cortex XSOAR.
- STEP 3 | Enter a meaningful Integration Name and a Description.
- STEP 4 | Enter Cortex XSOAR Instance FQDN/IP address.
If are adding a Cortex XSOAR instance that is part of a multi-tenant deployment, enter the tenant
URL without the protocol (http or https).
- STEP 5 | Enter the API Key associated with the Cortex XSOAR administrative user account.
The API key provide must belong to a Cortex XSOAR administrative user who has read-write
permissions, which are required to enable this push-based integration. Within Cortex XSOAR, navigate
to Settings > Integrations > API Keys and Get Key. - STEP 6 | Click Next and then Test.
- STEP 7 | Save the integration.
After set up the integration, the status indicates whether Prisma Cloud is connected to Cortex
XSOAR.
- STEP 8 | Modify an existing Alert Rule or create a new Alert Rule to send alert notifications to Cortex
XSOAR. (See Send Prisma Cloud Alert Notifications to Third-Party Tools.)
- STEP 9 | Get Prisma Cloud Access Key.
If do not have an access key, see Create and Manage Access Keys. need the Access Key ID and
Secret Key ID to complete the integration on Cortex XSOAR.
- STEP 10 | Set Up the Integration on Cortex XSOAR.
Set Up the Integration on Cortex XSOAR
Before view Prisma Cloud alerts as incidents on Cortex XSOAR, need content release 19.10.2
or a later version. The content release includes the incident fields required for this push-based integration.
When have the content release, the Classifier, incident types, and layouts are available automatically.
Cortex XSOAR maps Prisma Cloud alerts to out-of-the-box incident types such as AWS CloudTrail
Misconfiguration, AWS EC2 Instance Misconfiguration, AWS IAM Policy Misconfiguration, GCP Compute
Engine Misconfiguration, and Prisma Cloud. The out-of-box, Incident Layouts map the Prisma Cloud alert
data to the classification rules. These layouts provide the Incident Classifier & Mapping that is required for classifying incidents to the correct incident type and mapping the fields in the Prisma Cloud alert payload to
the Cortex XSOAR incident fields. When an incident is created, the playbook attached to the incident type
automatically executes.
Find all the Cortex XSOAR playbooks that are available to support remediation on Prisma Cloud for example
Prisma Cloud Remediation - AWS EC2 Instance Misconfiguration and Prisma Cloud Remediation - GCP VPC
Network Misconfiguration; search for playbook-PrismaCloudRemediation_.
If want to use the pull-based integration from Cortex XSOAR, see Cortex
documentation. In a pull-based integration, must enable the instance to Fetches
incidents. - STEP 1 | Install Cortex XSOAR content release 19.10.2 or a later version on Cortex XSOAR 5.0.0
or later instance.
19.10.2 is the minimum content release version that includes the Prisma Cloud incident fields required
for this push-based integration. see the incident fields on Settings > Advanced > Fields.
- STEP 2 | Enable the connection between Cortex XSOAR and Prisma Cloud.
1. Navigate to Settings > Integrations > Servers&Services.
2. Search for Prisma Cloud and Add Instance.
3. Complete the set up.
Provide a Name for the Prisma Cloud instance are integrating (the name must be unique from
other Integrations within cortex XSOAR), the Server URL that corresponds to the API endpoint for the Prisma Cloud instance, and access key and secret keys as username and password.
If access Prisma Cloud instance at https://app2.eu.prismacloud.io, the API endpoint is
https://api2.eu.prismacloud.io
- STEP 3 | Review the classification mapping for incident types.
When Prisma Cloud pushes alerts to the Cortex XSOAR endpoint, the alerts are classified under the Prisma Cloud App (/prismacloud app) path, and listed in Settings > Integrations > Classification
Mapping
and the playbooks associated with each incident type are in Settings > Integrations > Advanced >
Incident Types - STEP 4 | View incidents on Cortex XSOAR.
Verify that the integration is working as expected and that Prisma Cloud alerts display as incidents and
are mapped to specific incident types.
.
- STEP 5 | (Optional) Create additional classification and mapping rules and incident layouts to classify
Prisma Cloud alerts to distinct incident types on Cortex XSOAR.
Cortex XSOAR includes a few incident types for Prisma Cloud to which associate one of the AWS playbooks (listed above) for autoremediation. Refer to the Cortex XSOAR documentation for detailed instructions about customizing incident types, creating different classifications, mapping
and layouts for Prisma Cloud alerts, and to associate different playbooks to take action and enable
incident resolution for other cloud platforms. Refer to the Cortex XSOAR GitHub repository for some
sample packs. Prisma Cloud Integrations—Supported
Capabilities
The following table provides the details of features supported on each integration with Prisma™ Cloud.
Integration Integration
Method
User
Attribution
Notification
Template
Alert
Notification
Delay
Status
Check
State
Change
Notification
Alert
Notification
Grouping
Frequency
of Alert
Notification
AWS
Guard
Duty
Pull — — — — — — —AWS
Inspector
Pull — — — — — — —AWS
Security
Hub
Push — — — — —Amazon
SQS
Push — — (w/
batch
support)
—Aure
Service
Bus
Queue
Push — — — —Demisto Push — — — — — —Email Push — — — —Google
Cloud
SCC
Push — — — — —Jira Push — — —Microsoft
Teams
Push — — —PagerDuty Push — — — — — —ServiceNowPush — — — —Slack Push — — —Splunk Push — — — —PRISMA™ Integration Integration
Method
User
Attribution
Notification
Template
Alert
Notification
Delay
Status
Check
State
Change
Notification
Alert
Notification
Grouping
Frequency
of Alert
Notification
Tenable Pull — — — — — — —Qualys Pull — — — — — — —Webhook Push — — (w/
batch
support) 369
Prisma Cloud DevOps Security
The Prisma Cloud devOps security capabilities are geared to meet the common goal of delivering releases faster and preventing security lapses by applying a consistent set of checks
through the build-to-release process that keep applications and infrastructure secure.
> Secure Infrastructure Automation
> Prisma Cloud Plugins
> Set Up Prisma Cloud Configuration File for IaC Scan
> Use the Prisma Cloud Extension for AWS DevOps
> Use the Prisma Cloud Extension for Azure DevOps
> Use the Prisma Cloud Plugin for CircleCI
> Use the Prisma Cloud App for GitHub
> Use the Prisma Cloud Extension for GitLab
> Use the Prisma Cloud Plugin for IntelliJ IDEA
> Use the Prisma Cloud Extension for Visual Studio Code
> Use the Prisma Cloud IaC Scan REST API
> View the Prisma Cloud IaC scan policies available to perform checks on IaC templates. Secure Infrastructure Automation
Prisma Cloud DevOps Security enables DevOps and security teams to identify insecure configurations in
Infrastructure-as-Code (IaC) templates and vulnerabilities in container images so that security issues are
identified before actual resources are deployed in runtime environments.
To identify potential issues scan content in IaC templates such as AWS CloudFormation
Templates (JSON or YAML format), HashiCorp Terraform templates (HCL format), and Kubernetes App
manifests (JSON or YAML format) against a list of IaC policies.
With a valid Prisma Cloud Enterprise edition license, use the IaC scanning and container image
scanning functionality in any of the following ways:
- Plugins/Extensions—Install and configure the Prisma Cloud Plugins for popular IDEs such as VScode, IntelliJ; Source Control Management systems such as Github ;CI/CD tools such as Jenkins, CircleCI, Azure DevOps. These plugins are designed to easily integrate in to application development and
deployment processes so that scan and fix issues in current workflows without additional
tools, thereby reducing the friction and boosting the adoption of better security checks.
- Prisma Cloud IaC API—Interact with the Prisma Cloud IaC scanning API endpoint using tools such
as Curl, shell scripts, or Postman to scan IaC templates. Prisma Cloud recommends that use
the published plugins/extensions to perform IaC scanning, but use the IaC APIs directly for integrating with custom tools or specific use cases. See Use the Prisma Cloud IaC Scan REST API.
- Twistcli—Install and scan container images using twistcli. Twistcli is a command-line tool supported on
Linux, macOS, and Windows, and it requires a Docker Engine to be installed on the machine where are scanning images for vulnerabilities and malware. Prisma Cloud Plugins
Prisma Cloud plugins enable to check DevOps infrastructure templates for security
misconfigurations and scan container images to proactively prevent issues by shifting left.
The plugins or extensions as called on some environments, scan templates against Prisma Cloud IaC
policies to ensure compliance with security best practices before deploy it into the cloud infrastructure.
These plugins enable to stay secure while being agile because they make it easy to scan files, review any potential security issues, fix and validate code before check it in to source control
repository or integrate it in CI/CD pipeline.
Integration Category Marketplace Documentation
AWS DevOps CI/CD GitHub repository Use the Prisma Cloud
Extension for AWS
DevOps
Azure DevOps CI/CD Azure Visual Studio
Marketplace
Use the Prisma Cloud
Extension for Azure
DevOps
CircleCI CI/CD Circle CI Orb Registry Use the Prisma Cloud
Plugin for CircleCI
GitHub SCM GitHub Marketplace Use the Prisma Cloud
App for GitHub
GitLab SCM and CI/CD Use the Prisma Cloud
Extension for GitLab
IntelliJ IDEA IDE Intellij Marketplace Use the Prisma Cloud
Plugin for IntelliJ IDEA Integration Category Marketplace Documentation
Jenkins CI/CD Get the plugin from
the Prisma Cloud
administrative console
(Compute> Manage >
System > Downloads)
Use the Prisma Cloud
plugin for Jenkins
Visual Studio Code IDE VS Code Marketplace Use the Prisma Cloud
Extension for Visual
Studio Code Set Up Prisma Cloud Configuration File
for IaC Scan
Prisma Cloud IaC Scan requires a Prisma Cloud configuration file in the repository where templates are
stored. This configuration file can include information about IaC module structure, runtime variables, and tags that help refine IaC Scan use. It enables Prisma Cloud IaC scan to support complex module
structures and variable formats.
Create this file as .prismaCloud/config.yml in the root directory of repository branch.
The content of Prisma Cloud configuration file depends on the IaC Scan support need. The following show configuration details.
- Configure IaC Scan to Support Terraform
- Configure IaC Scan to Support AWS CloudFormation
- Configure IaC Scan to Support Kubernetes
- Configure Prisma Cloud Tags
Make sure to use a syntax validation tool when copy and paste content from this page.
Configure IaC Scan to Support Terraform
The following shows the parameters in the Prisma Cloud configuration file that enable to configure the IaC scan for Terraform 0.11 module with a variable file and/or input variables.
Make sure to use a syntax validation tool when copy and paste content from this page.

 # Specify the template type. Valid values are as follows.
 # - For Terraform: TF
 # - For AWS CloudFormation: CFT
 # - For Kubernetes: K8S
 template_type: TF
 # The valid values for terraform_version are 0.11 or 0.12
 terraform_version: 0.11
 # If terraform_version is 0.11, then terraform_011_parameters is
 required.
 # The value for variable_files is an array of custom variable file
 names.The path of each file is relative to repository branch root
 directory
 # The value for variable_values is an array of name/value pairs that
 identify the input variables template uses.
 terraform_011_parameters:
 variable_files:
 - scan/rich-value-types/network/variables.tf  variable_values:
 - name: check
 value: public-read-write


The following shows the parameters in the Prisma Cloud configuration file that enable to configure the IaC scan for Terraform 0.12.

 # Specify the template type. Valid values are as follows.
 # - For Terraform: TF
 # - For AWS CloudFormation: CFT
 # - For Kubernetes: K8S
 template_type: TF
 # Valid values for terraform_version are 0.11 or 0.12.
 terraform_version: 0.12
 # If terraform_version is 0.12, then terraform_012_parameters is
 required.
 # The value of terraform_012_parameters is an array of root_modules.
 The value for root_module is relative to repository branch root
 directory.
 # Each root module can have:
 # - variable_files, which is an array of variable file names relative
 to repository branch root directory
 # - variables, which is an array of name/value pairs that identify the  input variables for the module
 terraform_012_parameters:
 - root_module: scan/rich-value-types/
 variables:
 - name: check
 value: public-read-write
 - name: varName2
 value: varValue2
 - root_module: scan/for-expressions/
 variable_files:
 - scan/rich-value-types/expressions/variables.tf


Configure IaC Scan to Support AWS CloudFormation
The following shows the parameters in the Prisma Cloud configuration file that enable to configure the IaC scan for Amazon CloudFormation templates with variables.

 # Specify the template type. Valid values are as follows.
 # For Terraform: TF
 # For AWS CloudFormation: CFT
 # For Kubernetes: K8S
 template_type: CFT  # If template_type value is CFT, set cft_parameters (optional)
 # variable_values is an array of name/value pairs, which identifies
 the  # template variables
 cft_parameters:
 variable_values:
 - name: KeyName
 value: 10
 - name: AMI
 value: ami-45785


Configure IaC Scan to Support Kubernetes
The following shows the parameters in the Prisma Cloud configuration file that enable to configure the IaC scan for Kubernetes.

 # Specify the template type. Valid values are as follows.
 # For Terraform: TF
 # For AWS CloudFormation: CFT
 # For Kubernetes: K8S
 template_type: K8S


Configure Prisma Cloud Tags
The following shows the parameters in the Prisma Cloud configuration file that enable to identify
Prisma Cloud tags in template. These tags offer a flexible way to identify and organize resources
in Prisma Cloud.

 # Prisma Cloud Tags
 # tags is an array of labels that enable to organize your
 resources
 # with these key/value pairs in Prisma Cloud
 tags:
 - Org:Engineering
 - Team:Shift_Left

  Use the Prisma Cloud Extension for AWS
DevOps
With a Prisma Cloud Enterprise Edition license, integrate compliance and vulnerability checks into
AWS continuous integration/continuous (CI/CD) and build environments. This extension enables
to scan Infrastructure-as-Code (IaC) templates like AWS CFT, Terraform templates, and Kubernetes
deployment files against Prisma Cloud security policies. It also enables to use Prisma Cloud Compute to
scan container images for vulnerabilities.
The sections below show how to integrate the Prisma Cloud extension with AWS CodePipeline
pipelines and AWS CodeBuild projects.
- Set Up IaC Scanning with AWS CodePipeline
- Set Up Container Image Scanning with AWS CodeBuild
Set Up IaC Scanning with AWS CodePipeline
customize AWS CodePipeline to check Infrastructure-as-Code (Iac) templates. The following
examples show how to integrate IaC scan into CodePipeline.
have two options to scan IaC templates against Prisma Cloud security policies. use an
AWS Lambda function with Python scripting, or use a custom action with a Bash shell script.
The prerequisites for IaC scan integration regardless of whether use an AWS Lambda function or a custom action with a Bash shell script are as follows:
- have a valid Prisma Cloud Enterprise Edition license
- have a valid AWS CodePipeline service role to give AWS CodePipeline access to other resources in
account.
- have configured a two-stage pipeline in AWS CodePipeline.
- If customization uses any AWS commands, then have installed and configured the AWS
command line interface.
Use an AWS Lambda Function with Python Scripting
The following table describes the variables need to set for Lambda function, whether are
using the AWS CLI or the AWS console to configure Lambda function. If use the AWS console, these variables are environment variables. If use a script that invokes the AWS CLI, may specify the variables directly in the script.
Variable Value
Prisma_Cloud_API_URL Prisma Cloud API URL (e.g. https://
api.prismacloud.io). The exact URL depends
on the Prisma Cloud region and cluster of your
tenant
Access_Key Prisma Cloud access key for API access. If
do not have an access key, must Create
and Manage Access Keys Variable Value
Secret_Key The secret key that corresponds to Prisma Cloud access key
Asset_Name Identifies the repository want to scan
Tags Organizes the templates that are scanned with this
service connection, for visibility on Prisma Cloud
- STEP 1 | Create a Lambda function.
This example shows how to use the AWS command line interface to create a Lambda function that
scans the IaC templates in AWS CodePipeline for checking against Prisma Cloud security policies.
1. Set Up Prisma Cloud Configuration File for IaC Scan
Create the .prismaCloud/config.yml file and add it to the root directory of repository branch.
The file is required, and it must include the template type, version, and the template specific
parameters and tags use in environment.
2. Download PrismaCloudIaCScan.zip to an accessible location.
find the file at https://github.com/PaloAltoNetworks/Prisma-Cloud-DevOpsSecurity/blob/aws-codepipeline/aws-codepipeline/PrismaCloudIacScan/Lambda/
PrismaCloudIaCScan.zip
3. Run the following command to create Lambda function.
Note that will need to set the variables directly in script.
export AWS_PROFILE=prisma-scan
export AWS_DEFAULT_REGION=us-west-1
export AWS_LAMBDA_FUNCTION=AWSCodePipeline-gn
export AWS_ROLE=iam::/CustomPipelineWithLambda
aws --profile ${AWS_PROFILE} --region ${AWS_DEFAULT_REGION} \
 lambda create-function \
- -role ${AWS_ROLE}
 --function-name ${AWS_LAMBDA_FUNCTION} \
 --runtime python3.6 \
 --handler PrismaCloudIaCScan.lambda_handler \
 --environment
 Variables="{Prisma_Cloud_API_URL<hostname>,Access_Key=<accesskey>,\
 Secret_Key=<secetkey>, Asset_Name=<sssetname>, Tags=<tags>}” \
 --zip-file fileb://PrismaCloudIaCScan.zip
If prefer to use the AWS console instead of the AWS CLI to create Lambda function, can
use the steps below.
1. Set Up Prisma Cloud Configuration File for IaC Scan
Create the .prismaCloud/config.yml file and add it to the root directory of repository branch.
The file is required, and it must include the template type, version, and the template specific
parameters and tags use in environment.
2. Download PrismaCloudIaCScan.zip to an accessible location.
download this file from https://github.com/PaloAltoNetworks/Prisma-CloudDevOps-Security/blob/aws-codepipeline/aws-codepipeline/PrismaCloudIacScan/Lambda/
PrismaCloudIaCScan.zip 3. In the AWS console, set the environment variables listed at the beginning of Use an AWS Lambda
Function with Python Scripting.
4. Open the AWS Lambda console and navigate to the Create function page.
5. Provide a function name (e.g. LambdaFunctionForAWSCodePipeLine).
6. Chose a runtime of either Python 3.6 or Python 3.7.
7. Either create a new execution role or choose an existing role that has the proper permissions.
The proper permissions are:
- Write permission for AWS Code Pipeline
- List, Read, and Write permissions for AWS Cloudwatch Logs
- Read permission for S3 bucket if it is data source
8. Select Create function.
9. Set the handler to PrismaCloudIaCScan.lambda_handler.
The handler is defined in Basic Settings.
10.Choose the Execution role (optional).
11.Set a timeout.
12.Select Save. - STEP 2 | Add the Lambda function to pipeline.
1. In the AWS console, navigate to Services > Developer Tools > CodePipeline > Edit Pipeline. Choose
pipeline and select Edit.
2. Between any phase stage, select + Add Stage and provide a stage name of choice.
3. Select + Add action group. In Edit action, provide the information required to define a custom action.
The table below identifies the fields that have values specific to Prisma Cloud. The value for the User
parameters is in JSON format and specifies the conditions under which the pipeline job status will
fail. For the example in the table, the job will fail if the extension finds one high-severity violation or
two medium-severity violations or five low-severity violations.
Field Value
Action provider AWS Lambda
Function name The function name used when
created the Lambda function (e.g.
PrismaCloudIaCScan)
User parameters Example: {"FailureCriteria":
{"High": 1,"Medium": 2,"Low":
5,"Operator": "or"}}
Example with Tags: {"FailureCriteria":
{"High": 1,"Medium": 2,"Low":
5,"Operator": "or"}, "Tags":
["team: devOps", "env: test"]}
Valid values for “Operator” are “or” and
“and”
The following example shows the Edit action entries.
4. Select Done. 5. Review the results after you’ve executed pipeline.
To start a pipeline manually through the console, select Release change on the pipeline details page.
Select the link to execution details to see the latest CloudWatch logs to view any security violations
that Prisma Cloud identified.
Use a Custom Action with Bash
- STEP 1 | Create a custom action.
The following example shows how to use an AWS custom action with a Bash shell script to scan your
IaC templates and compare them against Prisma Cloud security policies.
It’s assumed that source is created in a GitHub repository. 1. If it’s not already installed, install jq, version 1.6 or higher on the EC2 instance or system where your
job worker will run.
jq is available at
https://stedolan.github.io/jq/.
2. If job worker runs in an EC2 instance, ensure EC2 instance user has permission to run
CodePipeline.
3. Ensure the AWS CLI is available where job worker runs.
The job worker uses the following:
- aws codebuild
- aws codepipeline
4. Create a file CustomAction.json in a working location, such as EC2 instance, and copy the following content to that file.
{
 "category": "Test",  "provider": "Prisma-Cloud-IaC-Scan",  "version": "1e",  "settings": {
 "entityUrlTemplate": "https://s3.console.aws.amazon.com/s3/
buckets/{Config:S3BucketName}/?
region={Config:S3BucketRegion}&tab=overview",  "executionUrlTemplate": "https://
s3.console.aws.amazon.com/s3/buckets/{Config:S3BucketName}/?
region={Config:S3BucketRegion}&tab=overview"
 },
 "configurationProperties": [
 {
 "name": "S3BucketName",  "required": true,  "key": true,  "secret": false,  "queryable": false,  "description": "The S3 bucket name. The results with the  vulnerabilities will be stored in this bucket.",  "type": "String"
 },  {
 "name": "S3BucketRegion",  "required": true,  "key": true,  "secret": false,  "queryable": false,  "description": "The S3 bucket region.",  "type": "String"
 },  {
 "name": "Prisma_Cloud_API_URL",  "required": true,  "key": true,  "secret": false,  "queryable": false,  "description": "Prisma Cloud server URL",  "type": "String"
  {
 "name": "Access_Key",  "required": true,  "key": true,  "secret": false,  "queryable": false,  "description": "Prisma Cloud access key",  "type": "String"
 },  {
 "name": "Secret_Key",  "required": true,  "key": true,  "secret": true,  "queryable": false,  "description": "Prisma Cloud secret key",  "type": "String"
 },  {
 "name": "Asset_Name",  "required": true,  "key": true,  "secret": false,  "queryable": false,  "description": "Provide the asset name for the pipeline",  "type": "Number"
 },  {
 "name": "Failure_Criteria",  "required": true,  "key": true,  "secret": false,  "queryable": false,  "description": "Provide failure threshold for high, medium
 and low severity security issues along with the operator. Ex. high:5,  medium:0, low:2, op:or",  "type": "String"
 },  {
 "name": "Tags",  "required": false,  "key": true,  "secret": false,  "queryable": false,  "description": "Provide the tags for the IaC Scan task.",  "type": "String"
 }
],  "inputArtifactDetails": {
 "maximumCount": 1,  "minimumCount": 0
 },  "outputArtifactDetails": {
 "maximumCount": 1,  "minimumCount": 0
 }
}
Optionally, edit the provider and version fields, but do not modify the configurationProperties field. 5. Execute the following AWS command to create the custom action.
aws codepipeline create-custom-action-type --cli-input-json \
file://CustomAction.json
6. Create the required IAM policies.
- Navigate to IAM > Policies, and create a policy to transfer files to and from the S3 bucket.
This policy enables the worker to pull build artifacts from the S3 bucket for scanning and publish
the logs to the bucket.
- Create the scan job worker for custom actions.
- Execute aws configure and set the default output format to JSON.
- Copy the job worker shell script poll.sh to local machine or the EC2 instance, depending
on the job worker’s running location.
Make sure EC2 instance user has permission to run pipeline and the job worker has permission to access CodePipeline.
- Execute the job worker shell script with the following command:
./poll.sh "category=Test,owner=Custom,version=1,provider=PrismaCloud-IaC-Scan"
The job worker is now configured to listen to requests from CodePipeline.
- STEP 2 | Add the custom action to pipeline.
1. In the AWS console, navigate to Services > Developer Tools > CodePipeline > Create/Edit Pipeline
to add Scan custom action to pipeline.
To add custom action to pipeline as a test step, navigate to Test > Test provider, and select
Custom Action.
2. Configure the values for the pipeline. Field Value
Input artifacts The output artifact from the previous step
Prisma_Cloud_API_URL Prisma Cloud API URL (e.g. https://
api.prismacloud.io). The exact URL
depends on the region and cluster of your
Prisma Cloud tenant
Access_Key Prisma Cloud access key for API access
Secret_Key Prisma Cloud secret key
Asset_Name identifies the repository want to scan
Tags Organizes templates that are scanned, for visibility on Prisma Cloud. Example:
env:test,team:devOps
Failure_Criteria Failure criteria for high, medium, and low
severity issues. Example: high:0, med:0, low:0, operator:or
S3BucketName Although this field is not specific to Prisma Cloud, a valid S3BucketName is required for this custom action
S3BucketRegion Although this field is not specific to Prisma Cloud, a valid S3BucketRegion is required for this custom action 3. Save the pipeline changes.
- STEP 3 | Test pipeline.
use the AWS console to release the pipeline manually. After stage completes, view
the results of the checks against Prisma Cloud security profile in the log report in S3 by selecting the Details link.
Set Up Container Image Scanning with AWS CodeBuild
enable container image scanning with Prisma Cloud Compute. Add the following steps to your
normal AWS CodeBuild build project set-up steps to add container scans to build project.
- STEP 1 | On the Prisma Cloud Compute Console, add a vulnerability scan rule.
1. Select Compute > Defender > Vulnerabilities > Images > CI. 2. Add Rule and enter a Rule name.
3. Specify the Alert and Failure thresholds.
set the vulnerability scan to fail on critical, high, medium, or low severity. The failure
threshold must be greater than the alert threshold.
4. Specify the Grace period.
The grace period is the number of days for which want
For more information about these settings, see the Prisma Cloud Compute Guide.
- STEP 2 | Use the following example as AWS buildspec file, buildspec.yaml, which is in the root
directory of source.
This file runs the twistcli command to scan the specified container image for vulnerabilities.
The following example splits some of the lines of code for documentation formatting. If
choose to copy this example directly, ensure the commands are not split into multiple
lines in code.
version: 0.2
# In this example, we're using environment variables
# to store the username and password of our Prisma Cloud Compute CI user
 account
# and the URL to our console
# PC_COMPUTE_USER: The Prisma Cloud Compute user with the CI User role
# PC_COMPUTE_PASS: The password for this user account
# PC_COMPUTE_CONSOLE_URL: The base URL for the console -- http://
console.<my_company>.com:8083 -- without a trailing /
phases:
 install:
 runtime-versions:
 docker: 18
 build:  commands:
 - echo Build started on `date`
 - echo Building the Docker image..$IMAGE_TAG
 - docker build -t $IMAGE_REPO_NAME:$IMAGE_TAG .
 post_build:
 commands:
 - echo Build completed on `date`
 - curl -k -u $PC_COMPUTE_USER:$PC_COMPUTE_PASS --output ./twistcli
 $PC_COMPUTE_CONSOLE_URL/api/v1/util/twistcli
 - chmod +x ./twistcli
 - echo Scanning with twistcli $PC_COMPUTE_PASS $PC_COMPUTE_USER
 # Run the scan with twistcli, providing detailed results in
 CodeBuild and
 # pushing the results to the Prisma Cloud Compute console.
 # --details returns all vulnerabilities and compliance issues rather
 than just summaries.
 # -address points to our Prisma Cloud Compute console
 # -u and -p provide credentials for the console. These creds only
 need the CI User role.
 # Finally, we provide the name of the image we built with 'docker
 build', above.
 - ./twistcli images scan --details --address $PC_COMPUTE_CONSOLE_URL
 -u $PC_COMPUTE_USER -p $PC_COMPUTE_PASS $IMAGE_REPO_NAME:$IMAGE_TAG

 # See twistcli documentation for more details.
- STEP 3 | In AWS CodeBuild build project, set the following environment variables, which the sample buildspec.yml file will use.
Environment Variable Description
PC_COMPUTE_USER Prisma Cloud Compute user with the CI User role
PC_COMPUTE_PASS Prisma Cloud Compute user password
PC_COMPUTE_CONSOLE_URL Base URL for the Prisma Cloud
Compute console (e.g. http://
console.<example>.com:8083)
IMAGE_REPO_NAME Docker repository for image to be scanned for vulnerabilities
IMAGE_TAG Docker tag for image to be scanned for vulnerabilities - STEP 4 | View the results of the container image scan.
poll.sh
#!/bin/bash
set -u
set -e trap "echo ERR; exit" ERR
# exec&> >(while read line; do echo "$(date +'%h %d %H:%M:%S') $line" >>
 cmds.log; done;)
#set -x
if [[ -z "${1:-}" ]]; then
 echo "Usage: ./poll.sh <action type id>" >&2
 echo -e "Example:\n ./poll.sh
 \"category=Test,owner=Custom,version=1,provider=Prisma-Cloud-IaC-Scan\""
 >&2
 exit 1
fi
echo_ts() {

 echo -e "\n" >> Prisma_Cloud_IaC_Scan.log
 echo "$1" >> Prisma_Cloud_IaC_Scan.log

}
run() {
 local action_type_id="$1"
 echo_ts "actiontypeid: $action_type_id"
 local job_json="$(fetch_job "$action_type_id")"
 if [[ "$job_json" != "null" && "$job_json" != "None" && "$job_json" !
= "" ]]; then

 local job_id="$(echo "$job_json" | jq -r '.id')"
 echo "job_id: $job_id"
 mkdir $job_id
 chmod +x $job_id
 cd $job_id || update_job_status "$job_json" "job id not found"

 acknowledge_job "$job_json"
 echo "job_json: $job_json"
 local build_json=$(create_build "$job_json")
 else
 sleep 10
 fi
}
acknowledge_job() {
 local job_json="$1"
 local job_id="$(echo "$job_json" | jq -r '.id')"
 local nonce="$(echo "$job_json" | jq -r '.nonce')"
 echo_ts "Acknowledging CodePipeline job (id: $job_id nonce: $nonce)" >&2
 aws codepipeline acknowledge-job --job-id "$job_id" --nonce "$nonce" > /
dev/null 2>&1
}
fetch_job() {
 local action_type_id="$1"
 aws codepipeline poll-for-jobs --max-batch-size 1 \
 --action-type-id "$action_type_id" \
 --query 'jobs[0]'
} action_configuration_value() {
 local job_json="$1"
 local configuration_key="$2"

 echo "$job_json" | jq -r ".data.actionConfiguration.configuration | .
[\"$configuration_key\"]"
}
update_job_status() {
 local job_json="$1"
 local build_state="$2"
 local job_id="$(echo "$job_json" | jq -r '.id')"
 echo_ts "Updating CodePipeline job with '$build_state' and job_id
 '$job_id'result" >&2

 if [[ "$build_state" == *"succeeded"* ]]; then
 aws codepipeline put-job-success-result \
 --job-id "$job_id" \
 --execution-details "summary=$build_state,externalExecutionId=
$job_id,percentComplete=100"
 else
 aws codepipeline put-job-failure-result \
 --job-id "$job_id" \
 --failure-details "type=JobFailed,message=Build
 $build_state,externalExecutionId=$job_id"
 fi
}
decide_job_status(){
 local job_json="$1"
 local stats="$2"

 local in_high="$3"
 local in_med="$4"
 local in_low="$5"
 local in_oper="$6"
 local resp_high="$(echo "$stats" | jq -r '.high')"
 local resp_med="$(echo "$stats" | jq -r '.medium')"
 local resp_low="$(echo "$stats" | jq -r '.low')"

 if [[ $in_oper == null ]];then
 in_oper="or"
 fi
 if [[ $in_high == null ]];then
 in_high=0
 fi
 if [[ $in_med == null ]];then
 in_med=0
 fi
 if [[ $in_low == null ]];then
 in_low=0
 fi

 if [[ $stats != null ]] ;then  if [[ "$in_oper" == "or" && ( "$resp_high" -ge "$in_high" ||
 "$resp_med" -ge "$in_med" || "$resp_low" -ge "$in_low" ) ]] ;then
 local failure_message="Prisma Cloud IaC scan failed with
 issues as security issues count (high: $resp_high, medium: $resp_med,  Low: $resp_low) meets or exceeds failure criteria (high: $in_high, medium:
 $in_med, Low: $in_low)"
 echo_ts "$failure_message"
 update_job_status "$job_json" "failed: Prisma Cloud IaC scan
 failed with issues."

 elif [[ "$in_oper" == "and" && ( "$resp_high" -ge "$in_high" &&
 "$resp_med" -ge "$in_med" &&esp_low" -ge "$in_low" ) ]]; then
 local failure_message="Prisma Cloud IaC scan failed with
 issues as security issues count (high: $resp_high, medium: $resp_med,  Low: $resp_low) meets or exceeds failure criteria (high: $in_high, medium:
 $in_med, Low: $in_low)"
 echo_ts "$failure_message"
 update_job_status "$job_json" "failed: Prisma Cloud IaC scan
 failed with issues."

 else
 local partial_success="Prisma Cloud IaC scan succeeded
 with issues as security issues count (high: $resp_high, medium: $resp_med,  Low: $resp_low) does not exceed failure criteria (high: $in_high, medium:
 $in_med, Low: $in_low)"
 echo_ts "$partial_success"
 update_job_status "$job_json" "succeeded: Prisma Cloud IaC
 scan succeeded with issues as security issues."
 fi

 else
 update_job_status "$job_json" "success"
 fi

}
create_build() {
 ls
 local job_json="$1"
 local job_id="$(echo "$job_json" | jq -r '.id')"
 local pipelineName="$(echo "$job_json" | jq -r
 ".data.pipelineContext.pipelineName")"
 echo "pipelineName: $pipelineName"
 local s3_bucket=$(action_configuration_value "$job_json" "S3BucketName")
 local bucketName="$(echo "$job_json" | jq -r
 ".data.inputArtifacts[0].location.s3Location | .[\"bucketName\"]")"
 local object_key="$(echo "$job_json" | jq -r
 ".data.inputArtifacts[0].location.s3Location | .[\"objectKey\"]")"
 local output_object="$(echo "$job_json" | jq -r
 ".data.outputArtifacts[0].location.s3Location | .[\"objectKey\"]")"
 local console_url="$(echo "$job_json" | jq -r
 ".data.actionConfiguration.configuration.Prisma_Cloud_API_URL")"
 local access_key="$(echo "$job_json" | jq -r
 ".data.actionConfiguration.configuration.Access_Key")"
 local secret_key="$(echo "$job_json" | jq -r
 ".data.actionConfiguration.configuration.Secret_Key")"
 aws codepipeline get-pipeline --name "$pipelineName" >
 pipelineDetails.json
 jq '.pipeline.stages[] | select(.name == "Source")' pipelineDetails.json >
 source.json  #cat source.json
 local user_id="$(cat source.json | jq -r
 ".actions[].configuration.Owner")"
 local project_name="$(cat source.json | jq -r
 ".actions[].configuration.Repo")"
 if [ -z "$console_url" ]; then
 echo_ts "Please enter valid Prisma Cloud API URL in plugin in Input
 param. For details refer to :plugin link"
 update_job_status "$job_json" "Please enter valid Prisma Cloud API URL
 in plugin in Input param. For details refer to plugin link"
 exit 1;
 fi
 echo "executing login api"
 local login_url="${console_url}/login"
 local req_cmd=$(curl -k -i -o -X POST $login_url -H "ContentType:application/json" --user-agent "AWS-CodePipeline-CustomAction/2.0.0"
 -d "{\"username\":\"${access_key}\",\"password\":\"${secret_key}\"}" -x
 http://127.0.0.1:8080 ) || update_job_status "$job_json" "$err_500"
 local err_400="Invalid credentials please verify that API URL, Access
 Key and Secret Key in Prisma Cloud plugin settings are valid For details
 refer to Extension link https://docs.paloaltonetworks.com/prisma/prismacloud/prisma-cloud-admin/prisma-cloud-devops-security/use-the-prisma-cloudextension-for-aws-codepipeline.html"
 local err_500="Oops! Something went wrong, please try again or refer to
 documentation on https://docs.paloaltonetworks.com/prisma/prisma-cloud/
prisma-cloud-admin/prisma-cloud-devops-security/use-the-prisma-cloudextension-for-aws-codepipeline.html"

 http_status=$(echo "$req_cmd" | grep HTTP | awk '{print $2}')
 echo "http status: $http_status"
 if [[ -z "$http_status" ]]; then
 echo_ts '$err_500' >&2
 update_job_status "$job_json" "error"
 exit 1;
 fi
 if [[ "$http_status" == 400 || "$http_status" == 401 ]] ; then
 echo_ts '$err_400' >&2
 update_job_status "$job_json" "error"
 exit 1

 fi
 if [[ "$http_status" == 500 || "$http_status" > 500 ]] ; then
 echo "http_status: $http_status"
 echo_ts '$err_500' >&2
 update_job_status "$job_json" "error"
 exit 1
 fi
 #echo "req cmd: $req_cmd"
 output_response=$(echo "$req_cmd" | grep token)

 local token="$(echo "$output_response" | jq .token | tr -d '"')"
 #echo "token: $token"

 local scan_location="$(echo $bucketName/$object_key)"
 aws s3 cp s3://$scan_location . || update_job_status "$job_json" "Copy
 Object from S3 bucket failed"

 local file=( *.zip )
 #echo "file: $file"
 mv $file artifact.zip
 file_size="$(wc -c artifact.zip | awk '{print $1}')"
 #echo "$file_size"
 file_size_limit=1000000
 if [[ "$file_size" -gt "$file_size_limit" ]]
 then
 printf "\nDirectory size $project_name more than 2 MB is not supported."
 exit 1;
 fi
 mkdir .prismaCloud
 unzip -p artifact.zip .prismaCloud/config.yml >.prismaCloud/config.yml
 if [[ ! -f .prismaCloud/config.yml ]]
 then
 echo "File nt present"
 fi

 iacAPI=${console_url}/iac_scan
 echo "executing scan api: $iacAPI"
 ##################################################################
 # Generate the url and the headers
 ##################################################################
 #echo "m here"
 if [[ ! -f .prismaCloud/config.yml ]]
 then
 echo "Can not find config.yml under .prismaCloud folder in repo
 $project_name. Please make sure the file is present in correct format
 (refer: https://docs.paloaltonetworks.com/prisma/prisma-cloud/prisma-cloudadmin/prisma-cloud-devops-security/use-the-prisma-cloud-extension-for-awsdevops.html) at the root of repo under .prismaCloud folder."
 exit 1;
 fi
 headers=""
 url=""
 fileContents=$(yq read -j .prismaCloud/config.yml)
 #echo "file contents are: " $fileContents
 templateType="$(echo "$fileContents" | jq -r '.template_type')"
 #echo "template type: " $templateType
 if [[ ! -z "$templateType" && ( "$templateType" == "TF" ||
 "$templateType" == "tf" ) ]]
 then
 url="$console_url/iac/tf/v1/scan"
 terraformVersion="$(echo "$fileContents" | jq -r '.terraform_version')"
 if [[ ! -z "$terraformVersion" && ( "$terraformVersion" == 0.12 ||
 "$terraformVersion" > 0.12 ) ]]
 then
 headers+=" -H terraform-version:0.12"
 isTerraform12ParamsPresent="$(echo "$fileContents" | jq -r
 '.terraform_012_parameters')"
 if [[ "$isTerraform12ParamsPresent" != null ]]
 then
 terraformContents="$(echo "$fileContents" | jq -r
 '.terraform_012_parameters[] |= with_entries( .key |= gsub("root_module";  "root-module") )' | jq -r '.terraform_012_parameters[] |=
 with_entries( .key |= gsub("variable_files"; "variable-files") )' )"
 terraform012Parameters="$(echo "$terraformContents" | jq -r
 '.terraform_012_parameters' | tr -d '\n\t' | tr -d '[:blank:]')"
 if [[ "$terraform012Parameters" != null ]]
 then
 headers+=" -H terraform-012-parameters:$terraform012Parameters"
 fi
 fi
 else
 headers+=" -H terraform-version:0.11"
 #read terraform 0.11 parameters
 variableFiles="$(echo "$fileContents" | jq -r
 '.terraform_011_parameters.variable_files' | tr -d '\n\t' | tr -d
 '[:blank:]')"
 variableValues="$(echo "$fileContents" | jq -r
 '.terraform_011_parameters.variable_values' | tr -d '\n\t' | tr -d
 '[:blank:]')"
 if [[ "$variableFiles" != null ]]
 then
 headers+=" -H rl-variable-file-names:$variableFiles"
 fi
 if [[ "$variableValues" != null ]]
 then
 headers+=" -H rl-parameters:$variableValues"
 fi
 fi
 elif [[ ! -z "$templateType" && ( "$templateType" == "CFT" ||
 "$templateType" == "cft" ) ]]
 then
 url="$console_url/iac/cft/v1/scan"
 variableValues="$(echo "$fileContents" | jq -r
 '.cft_parameters.variable_values' | tr -d '\n\t' | tr -d '[:blank:]')"
 if [[ "$variableValues" != null ]]
 then
 headers+=" -H rl-parameters:$variableValues"
 fi
 elif [[ ! -z "$templateType" && ( "$templateType" == "K8S" ||
 "$templateType" == "k8s" || "$templateType" == "K8s" ) ]]
 then
 url="$console_url/iac/k8s/v1/scan"
 else
 echo "No valid template-type found in config.yml file in repo
 $project_name. Please specify either of these values: TF, CFT or K8s as  template-type variable in the config.yml"
 exit 1;
 fi
 #echo url: "$url"
 #echo header: "$headers"

 #########################################################################################
 # Metadata Structure

 #########################################################################################
 # Tags
 task_tags="$(echo "$job_json" | jq -r
 ".data.actionConfiguration.configuration.Tags")"
 repo_tags="$(echo "$fileContents" | jq -r '.tags' |tr -d '\n\t' | tr -d
 '[:blank:]')"
 prisma_tags=""  if [[ ! -z "$repo_tags" ]]
 then
 prisma_tags+="\"repo_tags\":$repo_tags"
 fi
 if [[ ! -z "$task_tags" ]]
 then
 temp="\"$(sed 's/,/","/g' \<<< "$task_tags")\""
 if [[ "$prisma_tags" == "" ]]
 then
 prisma_tags+="\"task_tags\":[$temp]"
 else
 prisma_tags+=", \"task_tags\":[$temp]"
 fi
 fi
 aws codepipeline get-pipeline --name "$pipelineName" >
 pipelineDetails.json
 jq '.pipeline.stages[] | select(.name == "Source")' pipelineDetails.json
 > source.json
 cat source.json
 local user_id="$(cat source.json | jq -r
 ".actions[].configuration.Owner")"
 local project_name="$(cat source.json | jq -r
 ".actions[].configuration.Repo")"
 local stage_name="$(echo "$job_json" | jq -r
 ".data.pipelineContext.stage.name")"
 local action_name="$(echo "$job_json" | jq -r
 ".data.pipelineContext.action.name")"
 local asset_name="$(echo "$job_json" | jq -r
 ".data.actionConfiguration.configuration.Asset_Name")"
 #############################################################
 # Check failure criteria exists, if not default 0,0,0,or
 ############################################################

 local failure_criteria="$(echo "$job_json" | jq -r
 ".data.actionConfiguration.configuration.Failure_Criteria")"
 if [[ -z "$failure_criteria" ]];then
 failure_criteria_high_severity=0
 failure_criteria_medium_severity=0
 failure_criteria_low_severity=0
 failure_criteria_operator="or"
 else
 echo "failure criteria:" $failure_criteria
 failure_criteria_removed_spaces=$(printf '%s' $failure_criteria)
 delimiter=,  s=$failure_criteria_removed_spaces$delimiter
 array=();
 while [[ $s ]]; do
 array+=( "${s%%"$delimiter"*}" );
 s=${s#*"$delimiter"};
 done;
 #declare -p array
 failure_criteria_high_severity=$(awk -F':' '{print
 $2}'<<<"${array[0]}")
 failure_criteria_medium_severity=$(awk -F':' '{print $2}' <<<
 "${array[1]}")
 failure_criteria_low_severity=$(awk -F':' '{print $2}' <<<
 "${array[2]}")
 failure_criteria_operator=$(awk -F':' '{print $2}' <<< "${array[3]}")  #echo "Failure Criteria:" $failure_criteria_high_severity
 $failure_criteria_medium_severity $failure_criteria_low_severity
 $failure_criteria_operator
 fi
 # Metadata  metadata_json={"asset-name":"$asset_name","asset-type":"AWSCodePipeline","user-id":"${user_id}","prisma_tags":{"$prisma_tags"},"scanattributes":{"project-name":"${project_name}","pipeline-details":
{"pipeline-name":"$pipelineName","job-id":"$job_id","stagename":"$stage_name","action-name":"$action_name"}},"failure-criteria":
{"high":"$failure_criteria_high_severity","medium":"$failure_criteria_medium_severity","low":"$failure_criteria_low_severity","operator":"$failure_criteria_operator"}}
 echo metadata "$metadata_json"

 #################################################################################################
 # IaC Scan Execution

 #################################################################################################

 echo "Executing the scan api"
 local response="$(curl -k -X POST $url -H "x-redlock-auth:${token}"
 --user-agent "AWS-CodePipeline-CustomAction/2.0.0" $headers -H "xredlock-iac-metadata:${metadata_json}" -F templateFile=@artifact.zip -x
 http://127.0.0.1:8080)" || update_job_status "$job_json" "Call from API
 failed"
 #echo "response: $response"
 local result="$(echo "$response" | jq -r '.result.is_successful')"
 #echo "result: $result"
 if [[ $result ]]
 then
 local partial_failure="$(echo "$response" | jq -r
 '.result.partial_failure')"
 local matched="$(echo "$response" | jq -r '.result.rules_matched')"
 if [[ $matched != null ]] ;then
 local stats="$(echo "$response" | jq -r '.result.severity_stats')"
 decide_job_status "$job_json" "$stats"
 "$failure_criteria_high_severity" "$failure_criteria_medium_severity"
 "$failure_criteria_low_severity" "$failure_criteria_operator"
 display="$(echo "$matched" | jq -r 'sort_by(.severity) |
 (["SEVERITY" ,"NAME" ,"FILES"] | (., map(length*"-")) ), (.[] |
 [.severity , .name, .files[0] ]) | join(",")' | column -t -s ",")" ||
 update_job_status "$job_json" "Unknown Error "
 if [[ ! -z "$partial_failure" ]]
 then
 display+="\n$partial_failure"
 fi
 else
 echo_ts "Good job! Prisma Cloud did not detect any issues."
 fi
 else
 local error_message="$(echo "$response" | jq -r
 '.result.error_details')"
 echo_ts "$error_message"
 update_job_status "$job_json" "$error_message"
 exit 1
 fi

 echo_ts "$display" >&2
   aws s3 cp Prisma_Cloud_IaC_Scan.log s3://$s3_bucket/
Prisma_Cloud_IaC_Scan_$job_id.log || update_job_status "$job_json" "upload
 results to S3 bucket failed"
 rm -fr $job_id
}
run "$1" Use the Prisma Cloud Extension for Azure
DevOps
Use the Prisma Cloud extension to scan IaC templates, container images, and serverless functions in the build or release phase of the Azure DevOps pipeline. After install this extension from the Azure Visual
Studio Marketplace, set up the service connections for Prisma Cloud Iac Scan and Prisma Cloud
Compute Scan, and then use custom tasks in the build or release pipeline for scanning IaC templates—AWS
CloudFormation Templates, Terraform templates (version 0.11 and 0.12), Kubernetes app deployment
YAML files— container images, or serverless zip files. When create a custom task, specify the build or pipeline failure criteria based on severity of the security issues that the extension identifies.
When set up the Prisma Cloud extension to scan, specify the tags at different stages. Prisma Cloud tags enable visibility on the Prisma Cloud administrator console, and are different from Azure
DevOps tags or cloud tags that may have included within IaC templates. include these
tags as key:value pairs in a comma separated list when set up the service connection, and within
the.prismaCloud/config.yml at the repository-level, or where define the failure criteria for a Prisma Cloud IaC scan at the task level, and use it as a filter on Prisma Cloud (coming soon).
- Install and Configure the Prisma Cloud Extensions
- Set up a Custom Task for IaC Scanning
- Set Up Container Image Scanning
- (Required with nested virtualization only) Set Up RASP Defender
- Sample YAML File
Install and Configure the Prisma Cloud Extensions
need to add the prisma-cloud-config.yml in the root directory of repository branch, and get the Prisma Cloud extension from the Visual Studio Marketplace, set up service connections to authenticate with
Prisma Cloud and start scanning IaC templates, container images, and serverless functions.
- STEP 1 | Set up Azure DevOps organization and pipeline.
If are just getting started with Azure Pipeline, refer to the Azure documentation.
1. Create a project.
2. Create a new pipeline.
3. Select code repository, configure, and save the pipeline.
- STEP 2 | Set Up Prisma Cloud Configuration File for IaC Scan file.
Create the .prismaCloud/config.yml and add it to the root directory of repository branch. The file is
required, and it must include the template type, version, and the template specific parameters and tags
use in environment.
- STEP 3 | Install the extension.
1. Search for Prisma Cloud in the Visual Studio Marketplace. 2. Install the extension in Azure DevOps organization.
Select Organization settings > Extensions to verify that the extensions displays in the list of Installed
extensions. - STEP 4 | Add service connections to authenticate to Prisma Cloud.
must create a new service connection for each type of scan— one for IaC scanning and one for scanning container image or serverless functions.
1. Select Project Settings > Service Connections > New Service Connection > Prisma Cloud IaC
Console.
2. Enter the following information for the Prisma Cloud for IaC scanning and save changes.
- Enter the Prisma Cloud API URL as Server URL.
The URL for Prisma Cloud varies depending on the region and cluster on which tenant is
deployed. The tenant provisioned for is, for example, https://app2.prismacloud.io or https://
app.eu.prismacloud.io. Replace app in the URL with api and enter it here. Refer to the Prisma Cloud REST API Reference for more details.
- Enter Prisma CloudAccess Key.
The access key enables programmatic access. If do not have a key, must Create and
Manage Access Keys.
- Enter Prisma CloudSecret Key.
should have saved this key when generated it. cannot view it on the Prisma Cloud
web interface.
- Enter an Asset Name to identify the repository want to scan.
- Enter the Tags to organize the templates that are scanned with this service connection, for visibility on Prisma Cloud.
- Provide a Service connection name.
- Verify that Grant access permission to all pipelines is selected and Save changes.
3. Continue to the next step if want to set up another service connection for container image
scanning. If not, go to Set up a Custom Task for IaC Scanning.
4. Select Project Settings > Service Connections > New Service Connection > Prisma Cloud Compute
Console.
5. Enter the following information for Prisma Cloud Compute Console and save changes. -Server URL.
need to copy the server URL from the Prisma Cloud interface, Compute > Manage >
System > Downloads > Path to Console. For Prisma Cloud Compute Edition, get the URL from
Manage > System > Downloads > Path to Console
- Username and password.
These credentials are required for the service connection to authenticate with Prisma Cloud. If
are using Prisma Cloud Compute Edition (self-hosted), create a role and enter username
and password.
If are using Prisma Cloud Compute, must first Create Prisma Cloud Roleswith the Build
and Deploy Security permission group and assign this role to the administrative user so that they
can create an access key. The access key is the username and the secret key is password.
If password has special characters, make sure to escape any special
characters when enter password.
- Optional CA certificate if are using certificate-based authentication.
- Add a Name for the service connection.
- Verify that Grant access permission to all pipelines is selected.
- STEP 5 | Continue with Set up a Custom Task for IaC Scanning.
Set up a Custom Task for IaC Scanning
Use the following instructions to add a custom task for IaC scanning and container image and serverless
functions scanning in azure-pipelines.yml. In each task, define the pipeline failure criteria
based on the severity of the issues that are detected during the scan.
- STEP 1 | Under Pipelines, select pipeline and Edit to add custom task.
- STEP 2 | Add a custom task for IaC scanning. 1. Under Task, search for Prisma Cloud IaC Scan created earlier.
2. Enter the path for the directory want to scan.
If want to scan the entire repository, use.or $(System.DefaultWorkingDirectory).
3. Select the Service Endpoint, which is the service connection created in the previous task.
4. Enter the Tags want to apply to the templates that are being scanned.
The tags format is name:value, and add multiple tags that are separated using commas.
5. Select the Failure Criteria for the scan.
set the count for High, Medium, Low severity issues and decide whether want to use
the AND or OR operator to specify criteria. For example, if have a very strict threshold and
set the failure criteria to 0,0,0 with the OR operator build will fail if the policy checks detect any
issues.
6. Add to yml file, and Save the task.
7. Enable system diagnostics and Run . - STEP 3 | Run the task.
1. In Azure DevOps, click Queue to execute task on the next available build agent.
If task configuration is incomplete, a red status message displays Some settings need
attention just below Run build.
2. Check the results.
- If the IaC Scan finds no issues the pipeline task result is successful.
- If the IaC Scan finds issues but the failure criteria threshold defined is not met, the job is
successful but it displays the list of issues that were detected. If the failure criteria defined is more stringent that the default scan threshold, the job will fail
and review results in the log file.
Set Up Container Image Scanning
On Windows and Linux OS, scan container images and serverless functions when enable
twistcli, add a vulnerability scan rule where define the criteria to fail the build, and set up a task to scan
the image or function in the pipeline. - STEP 1 | Add a vulnerability scan rule on the Prisma Cloud Compute Console.
1. Select Compute > Defender > Vulnerabilities > Images > CI.
2. Add Rule and enter a Rule name.
3. Specify the Alert and Failure thresholds.
set the vulnerability scan to fail on critical, high, medium, low severity. The failure threshold
must be equal to or greater than the alert threshold.
4. (Optional) Specify the Images to scan.
The image or function zip file name is required later when add the scan task to the pipeline in
- Step 3.
5. (Optional) Select Apply rule when vendor fixes are available, if want to scan only for vulnerabilities that have fixes available.
6. Specify the Grace period.
The grace period is the number of days for which want to ignore a vulnerability. The time
frame is measured in days starting at the date from the first vendor publish. For more details on the advanced settings, see the Prisma Cloud Compute guide.
- STEP 2 | Add a pipeline task to scan container images using twistcli. 1. Select Pipelines, and Edit pipeline and to add custom task.
2. Search for Prisma in the task list and select Prisma Cloud Compute twistcli scan.
3. Select the Scan type—Images or Serverless.
4. Select the Prisma Cloud Compute Console service connection name that created earlier, from
the drop-down.
5. Specify the Image name or serverless Function zip file name.
The image name enter here must match the name of the image are building in the pipeline, if
it doesn’t the scan will fail.
- STEP 3 | View the results of the scan.
See the results in
To see results on Prisma Cloud, select Compute > Monitor > Vulnerabilities > Twistcli Scans Set Up RASP Defender
If are using Docker-in-Docker, where have a Docker container that itself has Docker installed, and
from within the container use Docker to pull images, build images, run containers, have to set up
RASP Defenders to secure containers at runtime.
Update the Dockerfile and embed the RASP defender as part of the Azure DevOps build.
1. Select Pipelines, and Edit pipeline and to add custom task
2. Search for Prisma in the task list and select Prisma Cloud Compute embed RASP.
3. Select the Scan type—Images or Serverless.
4. Select the Service connection created earlier for Prisma Cloud Compute Console.
5. Provide a unique Application ID for the RASP defender.
For example, <company>-<app> 6. Enter the Console Host, which is the DNS name or IP address of Prisma Cloud Compute
Console.
7. Specify the Data Folder, which is the read-write directory in the container file system.
For example, /twistlock/.
8. Enter the Dockerfile path of the container image to which want to add the RASP defender.
Sample YAML File
The following is a sample azure-pipeline.yml when enable both the Prisma Cloud IaC scan and Prisma Cloud Compute scan. This file autogenerates is referenced below as an example.
 # Starter pipeline
# Start with a minimal pipeline that customize to build and deploy
 code.
# Add steps that build, run tests, deploy, and more:
# https://aka.ms/yaml
trigger:
 branches:
 include:
 - master
pool:
 vmImage: 'ubuntu-latest'
steps:
- task: Palo-Alto-Networks.build-release-task.custom-build-releasetask.prisma-cloud-compute-scan@1
 displayName: 'Prisma Cloud Compute Scan'
 inputs:
 twistlockService: 'NewEnv Connection'
 artifact: 'nginx:latest'
- task: Prisma Cloud IaC Scan@1
 inputs:
 Path: 'repo'
 prismaCloudService: 'Prisma Cloud Scan'
 High: '0'
 Medium: '0'
 Low: '0'
 Operator: 'or'
- script: |
 echo Add other tasks to build, test, and deploy project.
 echo See https://aka.ms/yaml
 displayName: 'Run a multi-line script'
  Use the Prisma Cloud Plugin for CircleCI
Use the Prisma Cloud orb for CircleCI to scan IaC templates and container images during CircleCI pipelines.
In order to use Prisma Cloud IaC scan functionality, need to have a connection to a Prisma Cloud API
server and have details for that connection specified as environment variables. Similarly, in order to perform
container image vulnerability scans, need a connection to the Prisma Cloud Compute console. When
create a custom task to embed this functionality in CircleCI pipeline, specify the build or
pipeline failure criteria based on the severity of the security issues that are identified.
- STEP 1 | Verify the prerequisites.
- Verify the .circleci/config.yml is in project root directory.
CircleCI uses this file each time it runs a build.
- Set CircleCI org permissions to allow orbs that are not part of the Certified and Partner list. Your
CircleCI org admin can opt in to use uncertified third-party orbs by navigating to Settings > Security
and selecting the opt-in setting.
- For IaC scan, get the details for enabling authentication to Prisma Cloud.
- Prisma Cloud API URL. The URL for Prisma Cloud varies depending on the region and cluster on which tenant is
deployed. The tenant provisioned for is, for example, https://app2.prismacloud.io or https://
app.eu.prismacloud.io. Replace app in the URL with api and enter it here. Refer to the Prisma Cloud REST API Reference, which is accessible from the Help Center within the Prisma Cloud web
interface for more details.
- Access Key.
The access key enables programmatic access to Prisma Cloud. If do not have a key, must
Create and Manage Access Keys.
- Secret Key.
should have saved this secret key when generated it. cannot view it on the Prisma Cloud web interface.
- For image scan, get the details for authenticating to the Prisma Cloud Compute.
- Prisma Cloud Compute URL.
need to copy the URL from the Prisma Cloud interface, Manage > Defenders > Deploy.
- Prisma Cloud Compute username and password.
- STEP 2 | Add the environment variables for enabling authentication to Prisma Cloud.
On CircleCI, must add the environment variables as name value pairs. The following table lists the environment variables, and the figure below shows an example of environment variable settings for IaC
scans.
Name Value Notes
prisma_cloud_asset_name CircleCI server. Examples:
creditapp_server, ConsumerBU_server
Used to track specific results
in Prisma Cloud. For IaC scan
only. Required.
prisma_cloud_secret_key Prisma Cloud secret key See Create and Manage Access
Keys for details about the secret key. For IaC scan only.
Required.
prisma_cloud_compute_pass Prisma Cloud Compute
password
The Prisma Cloud Compute
users’s password.
prisma_cloud_tags Comma-separated list of key/
value pairs. Examples: project:x, owner:mr.y, compliance:pci
Used for visibility in Prisma Cloud UI. For IaC scan only.
Optional. 1. For IaC scan, enter the Name and Value for the Prisma Cloud secret key.
The default name is prisma_cloud_secret_key. If enter a different name, make sure to
match this name in the config.yml file in the next step.
2. For IaC scan, enter the Name and Value for asset name.
Prisma Cloud uses prisma_cloud_asset_name to track specific scan results.
3. For IaC scan, enter the Name and Value for all the prisma_cloud_tags want to define.
The Value is a comma-separated list of key/value pairs for tags that want to define. Use an
equals sign to assign tag values to tag keys. Tags are optional but enable visibility in the Prisma Cloud
UI.
4. For image scan, enter the Name and Value for the Prisma Cloud Compute password.
The default name is prisma_cloud_compute_pass. If enter a different name, make sure to
match this name in the config.yml file in the next step.
- STEP 3 | Add the Prisma Cloud configuration file.
The Prisma Cloud configuration file supports various IaC scan features. To add this file, create a subdirectory and file .prismacloud/config.yml in the root folder of project or repository
branch. See Set Up Prisma Cloud Configuration File for IaC Scan for details. Note that this file
is different from .circleci/config.yml, and subsequent references to config.yml in these steps
indicate the .circleci/config.yml file.
- STEP 4 | Add the Prisma Cloud orb to the config.yml.
1. Modify the config.yml to include the orb named prisma_cloud/devops_security for IaC and image
scanning.
Note that, through the orb, customize the IaC scan failure criteria and the vulnerability
thresholds for image scanning based on security needs.
More details about the Prisma Cloud orb are available at Prisma Cloud Orb Quick Start Guide.
The following table lists the parameters specify to customize the Prisma Cloud IaC scan job
in orb. Parameter Description Required Default Type
access_key Prisma Cloud
access key
no $prisma_cloud_access_key String
secret_key Prisma Cloud
secret key
no prisma_cloud_secret_key Environment
variable
prisma_cloud_api_url Prisma Cloud
server URL
no $prisma_cloud_console_url String
terraform_variable_filenames Commaseparated
list of file
names
containing
Terraform
variables
no ‘’ String
templates_directory_path Directory
path
where IaC
templates
are stored.
Note: The total size
of the IaC
templates
in this
directory
cannot
exceed 9
MB.
no . String
faiure_criteria_high_severity Provides
failure
threshold
for high
severity
security
issues
no 0 Integer
failure_criteria_medium_severity Provides
failure
threshold
for medium
severity
security
issues
no 0 Integer Parameter Description Required Default Type
failure_criteria_low_severity Provides
failure
threshold
for low
severity
security
issues
no 0 Integer
failure_criteria_operator Provides
operator
for high, medium, low severity
failure
thresholds
no or String
tags Commaseparated
list of tags
for your
task. Used
for visibility
in Prisma Cloud UI.
For IaC
scan only.
Optional.
no ‘’ String
The following table lists the parameters specify to customize the Prisma Cloud Compute
container image scanning job in orb.
Parameter Description RequiredDefault Type
prisma_cloud_compute_user The Prisma Cloud
Compute
user with the CI User role
no $prisma_cloud_compute_user String
prisma_cloud_compute_pass The Prisma Cloud
Compute
user's
password
no prisma_cloud_compute_pass Environment
variable
prisma_cloud_compute_url The base
URL for the console --
e.g. http://
console.<abc>.com:8083
no $prisma_cloud_compute_url String Parameter Description RequiredDefault Type
- - without a trailing /
workspace_name Name of workspace
to “docker
save” the image-tar
into so it can
be scanned
by orb
no workspace String
image_tar The name
of the image
tar file
stored in the workspace
no image.tar String
image The name
of the image
to scan --
myimage
or myorg/
myimage
or myorg/
myimage:latest
yes String
The following script is an example that shows how to add the details required to set up both IaC
and container image scanning.
Make sure that have defined the secret key (prisma_cloud_secret_key
for IaC scan) and the Prisma Cloud compute password
(prisma_cloud_compute_pass for container image scanning), each as an
environment variable.
version: 2.1
orbs:
 scan: prisma_cloud/devops_security@2.0.0
jobs:
 scan_iac: scan/prisma_cloud
 docker_build_and_save:
 executor: scan/compute
 steps:
 - checkout
 - run: docker pull nginx
 - run: mkdir -p workspace
 - run: docker image
 - run: 'docker save nginx:latest -o workspace/image.tar'
 - persist_to_workspace:
 root: workspace
 paths:
 - image.tar
workflows:  scan:
 jobs:
 - scan_iac:
 # Default env var for below: prisma_cloud_console_url
 prisma_cloud_api_url:<prisma cloud api url>
 # Default env var for below: prisma_cloud_access_key
 access_key: <prisma cloud access key>
 # Default env var for below: prisma_cloud_secret_key
 secret_key: prisma_cloud_secret_key
 failure_criteria_high_severity: 1
 failure_criteria_medium_severity: 2
 failure_criteria_low_severity: 3
 failure_criteria_operator: and
 tags: env:development, team:devOps
 - docker_build_and_save
 - scan/scan_image:
 requires:
 - docker_build_and_save
 # Default env var for below: prisma_cloud_compute_url
 prisma_cloud_compute_url: <prisma cloud compute console url>
 # Default env var for below: prisma_cloud_compute_user
 prisma_cloud_compute_user: <prisma cloud compute username>
 # Default env var for below: prisma_cloud_compute_pass
 prisma_cloud_compute_pass: prisma_cloud_compute_pass
 image: 'myrepo/myimage:tag'
 image_tar: image.tar
 vulnerability_threshold: critical
 compliance_threshold: ''
 only_fixed: true

2. Check the scan results.
After update the config.yml, whenever a PR is created, the Prisma Cloud orb checks for any
potential issues. The build is a Success or Failure depending on whether the number of the of issues
detected is lower than or more than the specified threshold.
When the scan starts, view the status:
In the following image view the status of the checks. The IaC scan reports as successful, while the image scan has completed the prerequisite check and is pending completion.
The ability to merge code is enabled only when the result is successful. Click Details to view more information. When any of the checks are unsuccessful, the results are
uploaded in a file named scan.csv in the Artifacts tab.
The following is an example of IaC scan results when the scan is successful and have no detected
security issues.
The following is an example of IaC scan results when the result was successful but with issues. The following is an example of IaC scan results that returned a failure because of the number and
type of security issues it found.
The following shows an example of container image scan results that failed because the IaC scan
found security issues in the image. Use the Prisma Cloud Plugin for IntelliJ IDEA
With the Prisma Cloud Enterprise edition license, install the IntelliJ IDEA plugin that enables to
check Infrastructure-as-Code (IaC) templates and deployment files against Prisma Cloud IaC policies, within
integrated development environment (IDE). The following steps show how simple it is to install and
check IaC templates and files for potential security misconfigurations.
If were using version 1.2 or earlier of the Prisma Cloud plugin for IntelliJ IDEA, must
update the plugin to version 1.3 or later. Use the instructions in this section to set up the plugin with the updated Prisma Cloud API URL and enter the credentials that are required to
authenticate to Prisma Cloud.
1. Install the Prisma Cloud Plugin for IntelliJ
2. Configure the Prisma Cloud Plugin for IntelliJ
3. Scan Using the Prisma Cloud Plugin for IntelliJ
Install the Prisma Cloud Plugin for IntelliJ
The Prisma Cloud plugin supports IntelliJ IDEA version 2016.2 and above.
- STEP 1 | In IntelliJ IDEA, select File > Settings > Plugins (on macOS, select Preferences > Plugins).
- STEP 2 | On the Plugins page, select Marketplace and search for Prisma Cloud.
- STEP 3 | Install the plugin.
Restart the IDE and verify that the Prisma Cloud plugin displays in the list of Installed plugins. Configure the Prisma Cloud Plugin for IntelliJ
After install the plugin, must provide the Prisma Cloud API URL and Prisma Cloud access key
information to authenticate and start scanning IaC templates. If access key changes, you’ll need to
update the access key information in this configuration.
- STEP 1 | In IntelliJ IDEA, select Settings > Tools > Prisma Cloud Plugin (on macOS, select Preferences >
Tools > Prisma Cloud Plugin).
- STEP 2 | Enter the following information to set up the plugin.
- Prisma Cloud API URL.
The URL for Prisma Cloud varies depending on the region and cluster on which tenant is
deployed. The tenant provisioned for is, for example, https://app2.prismacloud.io or https://
app.eu.prismacloud.io. Replace app in the URL with api and enter it here. Refer to the Prisma Cloud
REST API Reference, which is accessible from the Help Center within the Prisma Cloud web interface
for more details.
- Access Key.
The access key enables programmatic access to Prisma Cloud. If do not have a key, must
Create and Manage Access Keys.
- Secret Key.
should have saved this secret key when generated it. cannot view it on the Prisma Cloud web interface.
- Asset Name
Enter an asset name to identify the repository want to scan.
- Tags.
Define tags to organize the templates that are scanned with this service connection, for visibility on
Prisma Cloud. - STEP 3 | Add the Prisma Cloud configuration file.
The Prisma Cloud configuration file supports IaC scanning of complex module structures and variable
formats. To add this file, create a subdirectory and file .prismaCloud/config.yml in the root folder of your
project or repository branch. See Set Up Prisma Cloud Configuration File for IaC Scan for details.
Scan Using the Prisma Cloud Plugin for IntelliJ
Now, are ready to scan templates and view the results before check it in to the repository or
pipeline.
must have a Prisma Cloud Enterprise edition license and valid credentials to scan IaC
templates.
- STEP 1 | Scan the files for insecure configurations.
Right-click to scan template file or folder in the IDEA Project window and select Prisma Scan. - STEP 2 | View the results of the scan in the Scan Result tool window.
The title of the Scan Result window includes the date and time of the scan. For each scan, a new scan
result window is added. The tab situated farthest to the right displays the results of the latest scan.
If the scan detects no potential issues, the message displays as follows: If the scan detects any policy violations, the scan result displays the following details for each violation.
- Name of the violated policy -Description of the violated policy -Severity of the violation
- Name of the file with the issue
By default, the results are sorted by severity. sort the Scan Result using the policy name also.
The following examples show scan results for various template types. The first example shows the result
of scanning a Kubernetes deployment file with content that violates policies. will need to change
content of Prisma Cloud configuration file, .prismaCloud/config.yml, depending on the template
types and variables in project. The following example shows the result of scanning a folder with CloudFormation templates that have
policy violations. The example below shows the result of scanning a folder with Terraform 0.12 templates that contain a policy violation. Use the Prisma Cloud App for GitHub
This Prisma Cloud app for GitHub enables to scan IaC templates to check them against security policies
when open a pull request. For each pull request, define the pass criteria and view the scan
results directly on GitHub. When the defined criteria are not met, the pull request fails and view
all the checks that failed. In addition, the Prisma Cloud app creates an issue and adds the scan results as comments, so that fix all the issues reported before the changes are merged to the repository.
Use this app for scanning files in a private GitHub repository that has enabled restricted access. Be sure
to create a read-only role on Prisma Cloud and to generate a secret key and access key for a user. will need to provide these credentials to authenticate to Prisma Cloud for API access for the scanning
capabilities.
1. Set up the Prisma Cloud App Files for GitHub
2. Install the Prisma Cloud App for GitHub
Recent versions of the app capture the Prisma Cloud credentials as part of the installation
process and no longer require the credentials to be hard-coded in configuration file .github/
prisma-cloud-config.yml. Existing customers should remove the credentials from this file after
the app upgrade.
Set up the Prisma Cloud App Files for GitHub
To set up for IaC scans for a repository, need to create IaC scan configuration files. These files enable
to control the behavior of scans to meet needs for that repository. For example, depending
on the thresholds defined in these files, the Prisma Cloud app will perform checks that allow or fail
requests to merge or commit changes.
Creating these files before install the Prisma Cloud app for GitHub enables the app
installation itself to run a full IaC scan of selected repositories as part of the installation.
The three new files are:
- The Prisma Cloud configuration file .prismaCloud/config.yml
This file identifies the templates types wish to scan.
- .github/prisma-cloud-config.yml
This file includes the criteria that defines whether or not allow the commit for the pull request.
- .github/prisma-template-for-scan-results.yml
This file specifies how the scan results are made available to the person who created the pull request.
- STEP 1 | Set Up Prisma Cloud Configuration File for IaC Scan.
Create the .prismaCloud/config.yml in the root directory of repository branch. This file is required, and it must include the template type, version, and the template specific parameters and tags use in
environment.
- STEP 2 | Create the prisma-cloud-config.yml file to support the ability to scan IaC templates.
1. Select Create new file.
Add a new folder called .github, and name the file prisma-cloud-config.yml. The path should be
<repository name>/.github/prisma-cloud-config.yml. 2. Copy the template for this new file.
Copy and paste the following contents into .github/prisma-cloud-config.yml.
# Please update with the respective environment values and commit
# to master branch under the .github folder before performing scans
# Define the failure criteria for creating checks. If the criteria
# matches a check will be created. The template for the checks can
# becustomized in the "/.github/prisma-template-for-scan-results"
# file.
failure_criteria_for_creating_checks:
 high: 1
 medium: 1
 low: 1
 operator: or
# Define the failure criteria for creating issues. If the criteria
# matches an issue will be created. The template for issues can be
# customized in the "/.github/prisma-template-for-scan-results"
# file.
failure_criteria_for_creating_issues:
 high: 1
 medium: 1
 low: 1
 operator: or
# Define github asset name
github_asset_name: "Github Asset Dev"
# Define tags
tags:
- phase:testing
- env:QA 3. Define the parameter values in prisma-cloud-config.yml.
The parameters in prisma-cloud-config.yml define the failure criteria for pull requests. set
failure_criteria_for_creating_checks to define the number and severity of security
policy check failures that need to occur to trigger a merge request failure. The syntax for the failure_criteria_for_creating_checks value is as follows.
 high: x
 medium: y
 low: z
 operator: op
In the syntax above, x is a count of high-severity policy check failure, y is a count of medium-severity
policy check failures, and z is a count of low-severity policy check failures. The operator value
determines what combination of High/Medium/Low counts should result in a merge request failure.
The default for each count is 0. The value for operator, op, can be either OR or AND. The default
is OR. Some examples of settings for failure_critieria_for_creatings_checks are as follows.
- The setting below would result in a failed merge request security check for any detected policy check failure
 high: 0
 medium: 0
 low: 0
 operator: OR
- The setting below would result in merge requests never failing a security check.
 high: 1000
 medium: 1000
 low: 1000
 operator: AND
also use failure_criteria_for_creating_issues to define the number and severity of security
policy check failures that need to occur to trigger creation of a GitHub issue, during a pull request.
The syntax of the variable value is the same as that for failure_criteria_for_creating_checks. The value
includes high, medium, and low counts and includes an operator whose possible values are AND
and OR.
Prisma Cloud uses the asset name to track results. Some example names are creditapp_server and
ConsumerBU_server.
Prisma Cloud tags enable visibility on the Prisma Cloud administrator console.
- STEP 3 | Create the .github/prisma-template-for-scan-results.yml file to support how the scan results
are displayed.
Create the file .github/prisma-template-for-scan-results.yml with the same steps used to
create .github/prisma-cloud-config.yml.
1. Select Create new file and add file .github/prisma-template-for-scan-results.yml just as created .github/prisma-cloud-config.yml earlier.
2. Copy the template for the newly created prisma-template-for-scan-results.yml file. Copy and paste the following contents into .github/prisma-template-for-scan-results.yml.
# Prisma custom template for scan results
# Please update with the template and commit to master branch
# under the .github folder before performing scans
table_header_template : "Rule Name | Severity | Files | Description
\n------------ | ------------- | ------------ | -------------\n"
table_content_template : "{RuleName}|{Severity}|{Files}|{Description}\n"
#Template for defining title of issues created.
issues_title_template : "Vulnerabilities Found : {NumberOfVulnerability}"
issues_title_template_full_repo_scan : "Issues found during full repo
 scan"
#Template for defining title of checks created
checks_title_template : "Vulnerabilities Found : {NumberOfVulnerability}"
Update this file further only if want to customize the text in GitHub issues that the Prisma Cloud
app creates.
Install the Prisma Cloud App for GitHub
must set up the app to authenticate to Prisma Cloud, and optionally customize the scan
settings.
- STEP 1 | Search for Prisma Cloud on the GitHub marketplace.
- STEP 2 | Select Settings > Integrations &services > Add service and add Prisma Cloud.
This app requires the following permissions:
- Read access to code, to perform scan on template files.
- Read/write access to check for issues and open pull requests.
- Read access to metadata. - STEP 3 | Specify where want to install the app.
choose to install the Prisma Cloud app for GitHub on all repositories or only on selected
repositories. change this setting later to include more repositories for scanning.
- STEP 4 | Specify the Prisma Cloud API URL, Prisma Cloud access key ID, and corresponding secret key
to use for the integration.
The Prisma Cloud API URL specify depends on the region and cluster of Prisma Cloud tenant.
For example, if Prisma Cloud admin console URL is https://app.prismacloud.io, then Prisma Cloud API URL is https://api.prismacloud.io. See the Prisma Cloud REST API Reference for a list of Prisma Cloud API URLs.
See Create and Manage Access Keys for details about Prisma Cloud access keys.
Once you’ve entered settings, select Validate. If the settings are valid, a Save button appears, which
enables to save settings.
- STEP 5 | To add other repositories or to modify the configuration, select Settings > Integrations
& services > Prisma Cloud to Configure the app.
Whenever use this option to add repositories, the addition will result in an IaC scan of the repository
if all the configuration files for the Prisma Cloud app are set up. Use the Prisma Cloud Extension for GitLab
Use the Prisma Cloud extension to scan IaC templates in the build or release phase of the GitLab CI/CD
pipeline or SCM when create or merge a request. Container image or serverless function scanning is not
available with this plugin, currently.
- Use the Prisma Cloud Extension for the GitLab CI/CD Pipeline
- Use the Prisma Cloud Extension for GitLab SCM
Use the Prisma Cloud Extension for the GitLab CI/CD Pipeline
use the Prisma Cloud extension for GitLab CI/CD to scan IaC templates to check against Prisma Cloud policies or to scan container images to check for vulnerabilities.
To scan IaC templates in the build or release phase of the GitLab CI/CD pipeline, need to configure
the Prisma Cloud extension. The first step is to set up a connection to the Prisma Cloud API server and
configure the details—as environment variables—for that connection. Then, the IaC scanning capability
becomes available as a script that embed as a custom task in GitLab pipelines. can
trigger the custom task to scan on every commit (pull request) or on a schedule, and specify the build or
release pipeline failure criteria based on the severity of the security issues that it identifies. The scan uses
the failure thresholds specify to pass or fail the check. When the scan is successful, the code can be
merged. If the scan is unsuccessful, the security issues must be fixed in order to merge code changes.
The list of inputs that are required for scanning IaC templates in the build or release phase of the GitLab CI/
CD pipeline are:
- Connection settings as environment variables to enable communication between the Prisma Cloud API
server and GitLab repository.
- The .gitlab-ci.yml at the root level in repository.
For any commit or push to repository, this file start jobs on GitLab runners according to the contents of the file. must have a shared runner or a project-specific/custom runner for the job to
run successfully.
- iac_scan.sh script.
This script uses the values that provided in the environment variables to call the Prisma Cloud API
endpoint. have the flexibility either to provide the path where this iac_scan.sh script resides within
repository in the .gitlab-ci.yml file or to copy the script itself into the gitlab-ci.yml file.
When the script runs, if have any missing or incorrect environment variables, an error message
displays on the pipeline console.
- config.yml file at the root-level within the project under the .prismaCloud directory.
The path for this file must be .prismaCloud/config.yml. Prisma Cloud requires this configuration file to
learn about IaC module structure, runtime variables, and tags so that it can scan the IaC templates
in repository.
The Prisma Cloud extension also enables to scan container images for vulnerabilities. The steps to
configure the extension for container image scans are similar to those for IaC scans in that need to
connect to Prisma Cloud Compute through environment variables and that container image scanning is
available through a script that invoke through a job.
The list of resources that are required for scanning container images are: -Connection settings as environment variables to enable communication between the Prisma Cloud
Compute console and GitLab repository.
- The .gitlab-ci.yml at the root level in repository.
- container_scan.sh
This script uses the values that provide in the environment variables to call the Prisma Cloud
Compute endpoint. have the flexibility either to provide the path where this container_scan.sh script
resides within repository in the .gitlab-ci.yml file or to copy the script itself into the gitlab-ci.yml file.
- config.yml file at the root-level within the project under the .prismaCloud directory.
To set up the Prisma Cloud GitLab extension for CI/CD:
- Configure the Prisma Cloud Extension for GitLab CI/CD
- Set Up a Custom Job for IaC Scan
- Set Up a Custom Job for Container Image Scan
- Prisma Cloud Custom Script— iac_scan.sh
Configure the Prisma Cloud Extension for GitLab CI/CD
The GitLab extension is made up of scripts, and configuration includes defining environment variables and
creating scripts.
Much of the configuration involves defining environment variables in GitLab project settings. The table
below summarizes the environment variables will set to configure project for both IaC scans and
container image scans.
Key Description
prisma_cloud_api_url Prisma Cloud base API URL (for IaC scan)
prisma_cloud_access_key Prisma Cloud access key for API access (for IaC
scan)
prisma_cloud_secret_key Secret key that corresponds to Prisma Cloud
access key (for IaC scan)
prisma_cloud_cicd_asset_name GitLab server (for IaC scan)
prisma_cloud_cicd_failure_criteria String that defines criteria that should cause a pipeline failure (for IaC scan)
prisma_cloud_cicd_tags Prisma Cloud tags for future use (for IaC scan)
prisma_cloud_compute_url Base URL for the Prisma Cloud Compute console
(for container image scan)
prisma_cloud_compute_username Prisma Cloud Compute user with the CI User role
(for container image scan)
prisma_cloud_compute_password Prisma Cloud Compute user password (for container image scan) - STEP 1 | Set up the connection to the Prisma Cloud API.
1. Add the connection settings as environment variables to Project > Settings > CICD > Variables.
2. Set the Prisma Cloud API URL as the value for the prisma_cloud_api_url environment variable.
The API URL for Prisma Cloud varies depending on the region and cluster on which tenant is
deployed. If the tenant provisioned for is, for example, https://app2.prismacloud.io or https://
app.eu.prismacloud.io, replace app in the URL with api and enter it here. Refer to the Prisma Cloud
REST API Reference, for more details.
3. Add Prisma Cloud access key as the value for the prisma_cloud_access_key environment
variable.
The access key enables programmatic access. If do not have a key, see Create and Manage
Access Keys.
4. Add GitLab server name as the value for the prisma_cloud_cicd_asset_name environment
variable.
On Prisma Cloud, the asset name is used to track results. Some examples names are -
creditapp_server, ConsumerBU_server. etc
- STEP 2 | Set up environment variables for container image scans.
Set up the following environment variables only if want to run container image scans. As with the environment variables that support IaC scans, navigate to Project > Settings > CICD > Variables to
add new environment variables. 1. Add prisma_cloud_compute_url, whose value is the base URL for Prisma Cloud Compute
console. An example value is http://console<example>.com:8083.
2. Add prisma_cloud_compute_username, whose value is the Prisma Cloud Compute user with a CI user role.
3. Add prisma_cloud_compute_password, whose value is the password for the Prisma Cloud
Compute user.
- STEP 3 | Set Up Prisma Cloud Configuration File for IaC Scan.
Create the .prismaCloud/config.yml file and add it to the root directory of repository branch. The file is required, and it must include the template type, version, and the template specific parameters and
tags use in environment.
Set Up a Custom Job for IaC Scan
- STEP 1 | Create the iac_scan.sh custom script.
Use the sample Prisma Cloud Custom Script— iac_scan.sh to create the file. Then, add the file to a folder
from where it can be accessed in GitLab pipeline. This file enables to view the scan results.
- STEP 2 | Add Prisma Cloud IaC scan job to the GitLab CI configuration.
The GitLab CI configuration is stored in the .gitlab-ci.yml file.
1. Make the script executable.
chmod +x ./prismacloud-scripts/iac_scan.sh
2. Add the path to IaC templates in the gitlab-ci.yml file.
Add the command ./prismacloud-scripts/iac_scan.sh $CI_BUILDS_DIR/prisma_scan
to gitlab-ci.yml file. $CI_BUILDS_DIR/prisma_scan is the path to IaC templates
location.
The IaC templates in this directory must not exceed the 1 MB size limit.
Refer to the GitLab documentation to learn about the gitlab-ci.yml file. A sample file is included here.
#variables are specific to environment, please change accordingly. variables:
 GIT_STRATEGY: fetch
 GIT_CHECKOUT: "true"
 GIT_CLONE_PATH: $CI_BUILDS_DIR/prisma_scan

prisma-cloud-scan:
 stage: build
 before_script:
 - apt-get update -qy
 - apt-get install -y jq
 - wget https://github.com/mikefarah/yq/releases/download/3.2.1/
yq_linux_386
 - mv yq_linux_386 /usr/local/bin/yq
 - chmod +x /usr/local/bin/yq
 - apt-get install bsdmainutils #needed for displaying file output in
 column format on console
 - apt-get -y install zip unzip
 script:
 # If wish to pull code of project using git clone before
 next steps if wish to clone at different stage than build.

 # Here ./prismacloud-scripts/iac_scan.sh is the location of the  script in the gitlab repo or wherever it is stored. $CI_BUILDS_DIR/
prisma_scan is the argument to file which is 'full cloned repository
 path'.
 # prisma_scan is the project(repository) name.
 # The chmod +x ./prismacloud-scripts/iac_scan.sh is required to make
 the script executable.
 - chmod +x ./prismacloud-scripts/iac_scan.sh
 - ./prismacloud-scripts/iac_scan.sh $CI_BUILDS_DIR/prisma_scan
 # also pass the absolute repo path to the script as shown
 below
 #- ./prismacloud-scripts/iac_scan.sh /build/prisma_scan
 artifacts:
 when : always
 paths:
 - report/scan_results.csv
- STEP 3 | Set up the failure criteria for the Prisma Cloud IaC scan.
Define the number of issues by severity in the prisma_cloud_cicd_failure_criteria
environment variable. Set the High : x, Medium : y, Low : z, Operator: O, where, x,y,z is the number of issues of each severity, and the operator is OR, AND.
For example:
- To fail the pipeline for any security issue detected, prisma_cloud_cicd_failue_criteria = High : 0, Medium : 0, Low : 0, Operator: OR
- To never fail the pipeline, prisma_cloud_cicd_failure_criteria = High : 1000, Medium : 1000, Low :
1000, Operator: AND
- STEP 4 | Set up the Prisma Cloud tags.
Prisma Cloud tags are different from GitLab tags or cloud tags that may have included within your
IaC templates. Prisma Cloud tags enable visibility on the Prisma Cloud administrator console. Provide the values as a comma separated list of tags in the prisma_cloud_cicd_tags environment
variable. For example, prisma_cloud_cicd_tag:project:x,owner:mr.y,compliance:pci.
- STEP 5 | View IaC scan results.
The Prisma Cloud IaC scan uses the failure criteria defined in the prisma_cloud_cicd_failure_criteria environment variable to pass or fail a scan. When it
detects a security issue, it generates an artifact.
- To download the artifact, select Project > CI/CD > Pipeline, select the job and Download artifacts for the job.
IaC Scan result when the scan is successful and have no security issues.
- IaC Scan result when the scan fails.
- The Prisma Cloud artifact is a .csv file that lists the security issues detected. Download the artifact
and open report/scan_results.csv to view the list of issues.
- View the Prisma Cloud IaC scan results on the console of CI/CD pipeline log output (Project >
Setting > Pipeline, select the job and view Log console output.
Set Up a Custom Job for Container Image Scan
add container image scans to the extension have already configured for IaC scans, or can
configure the extension to perform just container image scans.
- STEP 1 | Create the custom script to scan container images. Use the sample Prisma Cloud custom script below, container_scan.sh, to create custom script.
Add script to a folder where GitLab pipeline can access it. This file enables to invoke the container image scan and to view the results of the scan.
#!/bin/bash
#######container_scan.sh - Container scan###########
if [[ "$1" != "" ]];then
 image=$1
 echo "image name:"$image
else
 echo "Please enter the image name to be scanned as an argument to bash
 script."
 exit 1;
fi
mkdir -p securethecloud
docker save $image -o securethecloud/image.tar
docker load -i securethecloud/image.tar
curl -u $prisma_cloud_compute_username:$prisma_cloud_compute_password --
output ./twistcli $prisma_cloud_compute_url/api/v1/util/twistcli
chmod +x ./twistcli
./twistcli --version
IMAGEID=`docker images $image --format "{{.ID}}"`
./twistcli images scan --details --address $prisma_cloud_compute_url -u
 $prisma_cloud_compute_username -p $prisma_cloud_compute_password $IMAGEID
if [ "$?" == "1" ]; then
 exit 1;
fi
- STEP 2 | Add a job to GitLab CI configuration, that invokes custom script.
The sample below shows a .gitlab-ci.yaml file that includes the details for both IaC and container image
scanning. Refer to the GitLab documentation to learn about the gitlab-ci.yml file. In this sample, the job
prisma-cloud-container-scan invokes the container image scan.
#variables are specific to environment, please change accordingly.
variables:
 GIT_STRATEGY: fetch
 GIT_CHECKOUT: "true"
 GIT_CLONE_PATH: $CI_BUILDS_DIR/prisma_scan

prisma-cloud-iac-scan:
 stage: build
 before_script:
 - apt-get update -qy
 - apt-get install -y jq
 - wget https://github.com/mikefarah/yq/releases/download/3.2.1/
yq_linux_386
 - mv yq_linux_386 /usr/local/bin/yq
 - chmod +x /usr/local/bin/yq
 - apt-get install bsdmainutils #needed for displaying file output in
 column format on console
 - apt-get -y install zip unzip
 script:
 # If necessary, use git clone in a different step to
 # pull project code
   # Below, ./prismacloud-scripts/iac_scan.sh is the fully qualified
 # script path.
 # Argument: $CI_BUILDS_DIR/prisma_scan is the fully qualified
 # cloned repository path where prisma_scan is the  # project(repository) name
 - ./prismacloud-scripts/iac_scan.sh $CI_BUILDS_DIR/prisma_scan
 # also pass the absolute repo path to the script
 # instead of using an environment variable.
 #- ./prismacloud-scripts/iac_scan.sh /build/prisma_scan
 artifacts:
 when : always
 paths:
 - report/scan_results.csv
prisma-cloud-container-scan:
 stage: twistlock #this could be any default stage or customized stage
 before_script:
 - docker version
 - docker info
 script:
 # ./prismacloud-scripts/container_scan.sh is the location of the  container scan
 # executable script in system, and
 # argument {image_name} is the location and name of the image want
 to scan
 - ./prismacloud-scripts/container_scan.sh {image_name} #If wish
 also copy the content of container_scan.sh here.
 #example
 #- ./prismacloud-scripts/container_scan.sh nginx:latest
- STEP 3 | Add a vulnerability scan rule on the Prisma Cloud Compute Console.
1. In the Prisma Cloud UI, select Compute > Defender > Vulnerabilities > CI.
2. Select Add Rule and enter a Rule name. 3. Specify the Alert and Failure thresholds.
4. (Optional) Specify the image to scan.
The alternative is to specify the image to scan in GitLab job, as in step 2 above.
5. (Optional) Select Advanced Settings, for options to refine rule further.
For more details on the advanced settings, see the Prisma Cloud Compute guide
- STEP 4 | Run pipeline and view the job results.
view the results of the container image scan on the CI/CD pipeline log console output. In
GitLab, navigate to Project > Settings > Pipeline, select job, and view Log console output. Prisma Cloud Custom Script— iac_scan.sh
iac_scan.sh script for Prisma Cloud IaC scan GitLab CI/CD extension.
 #!/bin/bash
#######Perform IaC scan###########
#echo "Entered full cloned repo path:" $1
if [[ "$1" != "" ]];then
 repo_path=$1
else
 echo "Please enter the full cloned repository path on build server/
runner. For details refer to https://docs.paloaltonetworks.com/prisma/
prisma-cloud/prisma-cloud-admin/prisma-cloud-devops-security/use-the-prismacloud-app-for-gitlab.html"
 exit 1;
fi
#echo "repo_path:" $repo_path
#ls -al $repo_path
#read ENV variables
echo $prisma_cloud_api_url $prisma_cloud_access_key #
$prisma_cloud_secret_key
if [[ -z "$prisma_cloud_api_url" ]];then  echo "Please enter a valid URL. For details refer to https://
docs.paloaltonetworks.com/prisma/prisma-cloud/prisma-cloud-admin/prismacloud-devops-security/use-the-prisma-cloud-app-for-gitlab.html"
 exit 1;
fi
if [[ -z "$prisma_cloud_access_key" || -z "$prisma_cloud_secret_key" ]];then
 echo "Invalid credentials, verify that access key and secret key
 in environment variables are valid. For details refer to https://
docs.paloaltonetworks.com/prisma/prisma-cloud/prisma-cloud-admin/prismacloud-devops-security/use-the-prisma-cloud-app-for-gitlab.html"
 exit 1;
fi
if [[ ! -f $repo_path/.prismaCloud/config.yml ]]; then
 echo "Can not find config.yml under .prismaCloud folder in repo
 $CI_PROJECT_TITLE. Please make sure the file is present in correct format
 https://docs.paloaltonetworks.com/prisma/prisma-cloud/prisma-cloud-admin/
prisma-cloud-devops-security/use-the-prisma-cloud-app-for-gitlab.html at the  root of repo under .prismaCloud folder."
 exit 1;
fi
if [[ -z "$prisma_cloud_cicd_asset_name" ]]; then
 echo "Please enter a valid cicd asset name. For details refer to https://
docs.paloaltonetworks.com/prisma/prisma-cloud/prisma-cloud-admin/prismacloud-devops-security/use-the-prisma-cloud-app-for-gitlab.html"
 exit 1;
fi
#####Compress the repo and check if compressed zip file
 size>5MB#############
#echo "current path:"
#pwd
cd $repo_path
#ls -al .
zip -r $repo_path/iacscan.zip . -x \*.git\* #here cd inside repo_path and
 '.' as source is mandatory else while zipping copies else it will zip from
 root instead of files inside repo
#echo "after zip content of repo_path"
#ls -al $repo_path
file_size="$(wc -c $repo_path/iacscan.zip | awk '{print $1}')"
#echo "file_size:" $file_size
file_size_limit=5242880
if [[ "$file_size" -gt "$file_size_limit" ]];then
 echo "Directory size $repo_path more than 8MB is not supported"
 exit 1;
fi
#$CI_PROJECT_DIR is default inbuilt dir used to upload the artifacts but  can change to any one the job has access to.
#echo "view content of CI_PROJECT_DIR"
#ls -al $CI_PROJECT_DIR
#############Check failure criteria exists, if not default 0,0,0,or#########
if [[ -z "$prisma_cloud_cicd_failure_criteria" ]];then
 failure_criteria_high_severity=0
 failure_criteria_medium_severity=0
 failure_criteria_low_severity=0
 failure_criteria_operator="or" else
 echo "failure criteria:" $prisma_cloud_cicd_failure_criteria
 cicd_failure_criteria_removed_spaces=$(printf '%s'
 $prisma_cloud_cicd_failure_criteria)
#- echo $cicd_failure_criteria_removed_spaces
 delimiter=,  s=$cicd_failure_criteria_removed_spaces$delimiter
 array=();
 while [[ $s ]]; do
 array+=( "${s%%"$delimiter"*}" );
 s=${s#*"$delimiter"};
 done;
#- declare -p array
 failure_criteria_high_severity=$(awk -F':' '{print $2}' <<< "${array[0]}")
 failure_criteria_medium_severity=$(awk -F':' '{print $2}' <<<
 "${array[1]}")
 failure_criteria_low_severity=$(awk -F':' '{print $2}' <<< "${array[2]}")
 failure_criteria_operator=$(awk -F':' '{print $2}' <<< "${array[3]}")
 #echo "Failure Criteria:" $failure_criteria_high_severity
 $failure_criteria_medium_severity $failure_criteria_low_severity
 $failure_criteria_operator
fi
#################################################
# Read .prismaCloud/config.yml and form headers for scan
################################################
fileContents=$(yq read -j $repo_path/.prismaCloud/config.yml)
#echo "file contents are:" $fileContents
t_Type="$(echo "$fileContents" | jq -r '.template_type')"
#echo "template type:" $t_Type
headers=""
url=""
if [[ ! -z "$t_Type" ]]; then
 templateType=${t_Type^^}
 #echo $templateType
else
 echo "No valid template-type found in config.yml file in repo
 $CI_PROJECT_TITLE. Please specify either of these values: TF, CFT or K8s as  template-type variable in the config.yml"
 exit 1;
fi
if [[ "$templateType" == "TF" ]]; then
 url="$prisma_cloud_api_url/iac/tf/v1/scan"
 terraformVersion="$(echo "$fileContents" | jq -r '.terraform_version')"
 if [[ ! -z "$terraformVersion" && "$terraformVersion" == "0.12" ]];then
 headers+=" -H terraform-version:$terraformVersion"
#read terraform 0.12 parameters
 isTerraform12ParamsPresent="$(echo "$fileContents" | jq -r
 '.terraform_012_parameters')"
 if [[ "$isTerraform12ParamsPresent" != null ]]; then
 terraformContents="$(echo "$fileContents" | jq -r
 '.terraform_012_parameters[] |= with_entries( .key |= gsub("root_module";
 "root-module") )' | jq -r '.terraform_012_parameters[] |=
 with_entries( .key |= gsub("variable_files"; "variable-files") )' )"
 terraform012Parameters="$(echo "$terraformContents" | jq -r
 '.terraform_012_parameters' | tr -d '\n\t' | tr -d '[:blank:]')"
 if [[ "$terraform012Parameters" != null ]]; then
 headers+=" -H terraform-012-parameters:$terraform012Parameters"  fi
 fi
 else
#- headers+=" -H terraform-version:0.11" no version header needed for  0.11
#- read terraform 0.11 parameters
 variableFiles="$(echo "$fileContents" | jq -r
 '.terraform_011_parameters.variable_files')"
 variableValues="$(echo "$fileContents" | jq -r
 '.terraform_011_parameters.variable_values')"
 if [[ "$variableFiles" != null ]]; then
 headers+=" -H rl-variable-file-names:$variableFiles"
 fi
 if [[ "$variableValues" != null ]]; then
 headers+=" -H rl-parameters:$variableValues"
 fi
 fi
elif [[ "$templateType" == "CFT" ]]; then
 url="$prisma_cloud_api_url/iac/cft/v1/scan"
 variableValues="$(echo "$fileContents" | jq -r
 '.cft_parameters.variable_values' | tr -d '\n\t' | tr -d '[:blank:]')"
 if [[ "$variableValues" != null ]]; then
 headers+=" -H 'rl-parameters:$variableValues'"
 fi
elif [[ "$templateType" == "K8S" ]]; then
 url="$prisma_cloud_api_url/iac/k8s/v1/scan"
else
 echo "No valid template-type found in config.yml file in repo
 $CI_PROJECT_TITLE. Please specify either of these values: TF, CFT or K8s as  template-type variable in the config.yml"
 exit 1;
fi
###################################################
# LOGIN TO GET TOKEN
##################################################
#echo "Get token using login api"
result=$(curl -k -i -X POST $prisma_cloud_api_url/login --user-agent
 "GitLab PrismaCloud/DevOpsSecurity-1.0.0" -H 'Content-Type:application/
json' -d "{\"username\":\"${prisma_cloud_access_key}\",\"password\":
\"${prisma_cloud_secret_key}\"}")
#echo $result
code=$(echo "$result" |grep HTTP | awk '{print $2}')
echo $code
if [[ "$code" -eq 400 || "$code" -eq 401 || "$code" -eq 403 ]]; then
 echo "Invalid credentials, verify that access key and secret key
 in environment variables are valid. For details refer to https://
docs.paloaltonetworks.com/prisma/prisma-cloud/prisma-cloud-admin/prismacloud-devops-security/use-the-prisma-cloud-app-for-gitlab.html"
 exit 1;
elif [[ "$code" -eq 500 || "$code" -eq 501 || "$code" -eq 503 ]];then
 echo "Oops! Something went wrong, please try again or refer to
 documentation https://docs.paloaltonetworks.com/prisma/prisma-cloud/prismacloud-admin/prisma-cloud-devops-security/use-the-prisma-cloud-app-forgitlab.html"
 exit 1;
elif [[ "$code" -ne 200 ]];then
 echo "Oops! Something went wrong, please try again or refer to
 documentation https://docs.paloaltonetworks.com/prisma/prisma-cloud/prisma- cloud-admin/prisma-cloud-devops-security/use-the-prisma-cloud-app-forgitlab.html"
 exit 1;
fi
output_response=$(echo "$result" | grep token)
token="$(echo "$output_response" | jq .token | tr -d '"')"
####################################################
# Start PROCESSING PRISM CLOUD IAC SCAN
###################################################
#echo url:"$url"
echo header:"$headers"
#form prisma-tags
prisma_tags=""
if [[ ! -z "$prisma_cloud_cicd_tags" ]]; then
 temp_str=$(printf '%s' $prisma_cloud_cicd_tags)
 if [[ ! -z "$temp_str" ]]; then
 settings_tags=\"$(sed 's/,/","/g' <<< "$temp_str")\"
 prisma_tags="\"settings-tags\":[$settings_tags]"
 fi
fi
#tags from config.yml
repo_tags="$(echo "$fileContents" | jq -r '.tags' |tr -d '\n\t' | tr -d
 '[:blank:]')"
if [[ $repo_tags != null ]]; then
 prisma_tags+=",\"repo-tags\":$repo_tags"
fi
##################################################################
# creating metadata structure
metadata_json={"asset-name":"$prisma_cloud_cicd_asset_name","assettype":"Gitlab","user-id":"${GITLAB_USER_LOGIN}","prisma-tags":
{"$prisma_tags"},"scan-attributes":{"build-number":"${CI_JOB_ID}","projectname":"${CI_PROJECT_TITLE}"},"failure-criteria":
{"high":"$failure_criteria_high_severity","medium":"$failure_criteria_medium_severity","low":"$failure_criteria_low_severity","operator":"$failure_criteria_operator"}}
#echo metadata "$metadata_json"
#################################################################
cd $CI_BUILDS_DIR
#ls
cp $repo_path/iacscan.zip .
response="$(curl -k -X POST $url -H "x-redlock-auth:${token}" --user-agent
 "GitlabCI PrismaCloud/DevOpsSecurity-1.0.0" $headers -H "x-redlock-iacmetadata:${metadata_json}" -F templateFile=@iacscan.zip)"
#echo $response
result="$(echo "$response" | jq -r '.result.is_successful')"
mkdir results
if [[ "$result" == true ]];then
 matched="$(echo "$response" | jq -r '.result.rules_matched')"
 if [[ $matched != null ]];then
 stats="$(echo "$response" | jq -r '.result.severity_stats')"
 echo $matched | jq '["Severity","Name","Description",  "Files"], (map({severity, name, description, files} ) | .[] |
 [.severity, .name, .description, (.files|join(";"))]) | @csv' | tr -d '\
\"'> results/scan.csv
 awk -F'\t' -v OFS='\t' '  NR == 1 {print "Index", $0; next}
 {print (NR-1), $0}
 ' results/scan.csv > results/scan_results.csv
 #format console output file to display
 echo $matched | jq '["Severity","Name","Files"], (map({severity, name,  files} ) | .[] | [.severity, .name, (.files|join(";"))]) | @csv'| column -t
 -s "," | tr -d '\\"' > results/formatted.csv
 awk -F'\t' -v OFS='\t' '
 NR == 1 {print "\nIndex", $0; print
 "------------------------------------------------------------------------------------------------------------------------------------------------------" ;
 next}
 {print (NR-1), $0}
 ' results/formatted.csv > results/console_output.csv
 #show result on console
 cat results/console_output.csv
 #echo $CI_PROJECT_DIR
 mkdir $CI_PROJECT_DIR/report
 cp -r results/scan_results.csv $CI_PROJECT_DIR/report
 #ls -la $CI_PROJECT_DIR/report
 high="$(echo "$stats" | jq -r '.high')"
 med="$(echo "$stats" | jq -r '.medium')"
 low="$(echo "$stats" | jq -r '.low')"
 if [[ ( ( $failure_criteria_operator == "or" ) &&
 ( "$high" -ge $failure_criteria_high_severity) || ( "$medium"
 -ge $failure_criteria_medium_severity ) || ( "$low" -ge
 $failure_criteria_low_severity ) ) || ( ($failure_criteria_operator
 == "and") && ( "$high" -ge $failure_criteria_high_severity ) &&
 ( "$medium" -ge $failure_criteria_medium_severity ) && ( "$low" -ge
 $failure_criteria_low_severity ) ) ]];then
 echo "Prisma Cloud IaC scan failed with issues as security issues
 count (high:$high , medium:$med , low:$low) meets or exceeds the  failure criteria (high:$failure_criteria_high_severity, medium:
$failure_criteria_medium_severity, low:$failure_criteria_low_severity,  operator:$failure_criteria_operator) "
 exit 1;
 else
 echo "Prisma Cloud IaC Scan has been successful as security
 issues count (high:$high, medium:$med, low:$low) does not exceed
 the failure criteria (high:$failure_criteria_high_severity, medium:
$failure_criteria_medium_severity, low:$failure_criteria_low_severity,  operator:$failure_criteria_operator)"
 exit 0;
 fi
 else
 echo "Good job! Prisma Cloud did not detect any issues."
 fi
else
 error_message="$(echo "$response" | jq -r '.result.error_details')"
 echo "$error_message"
 exit 1;
fi

Use the Prisma Cloud Extension for GitLab SCM
Use the Prisma Cloud extension to scan IaC templates when create or update a merge request. can
define failure criteria for each GitLab project and view the scan results directly in the GitLab user interface. In addition, the Prisma Cloud extension can create GitLab issues that report details from IaC scans for checks against security policies. This ability enables to fix all the reported issues before changes
are merged into the repository.
The sections below describe how to set up the Prisma Cloud extension and how to use it.
- Configure the Prisma Cloud Extension for GitLab SCM
- Run an IaC Scan in a Merge Request
Configure the Prisma Cloud Extension for GitLab SCM
The Prisma Cloud Extension for GitLab SCM does not require a separate software installation, but does
require the following configuration steps.
Much of the configuration involves setting environment variables in GitLab project settings. The image
below summarizes the environment variables will set to configure project for IaC scans.
If want to run IaC scans for both GitLab SCM and GitLab CICD in a single project, set environment variables for both in project settings.
- STEP 1 | Set environment variables to support a connection to the Prisma Cloud API.
1. In GitLab, navigate to Project > Settings > CICD > Variables, and add the connection settings as environment variables. 2. Set the Prisma Cloud API URL as the value for the prisma_cloud_api_url environment variable.
The API URL for Prisma Cloud varies depending on the region and cluster on which tenant is
deployed. If the tenant provisioned for is, for example, https://app2.prismacloud.io or https://
app.eu.prismacloud.io, replace app in the URL with api and enter it here. Refer to the Prisma Cloud
REST API Reference for more details.
3. Add Prisma Cloud access key as the value for the prisma_cloud_access_key environment
variable.
The access key enables programmatic access. If do not have a key, see Create and Manage
Access Keys.
4. Add GitLab server name as the value for the prisma_cloud_scm_asset_name environment
variable.
Prisma Cloud uses the asset name to track results. Some example names are creditapp_server and
ConsumerBU_server.
5. Create a GitLab access token by navigating to User settings > access tokens and creating a new
GitLab access token with the following permissions: api, read_user, and read_repository. Use a bot account or service account that you’d generally use for webhooks or pipeline integration
to create this access token. The account should have a project Maintainer or Owner role or an
Administrator role to ensure the Prisma Cloud extension can read the environment variables.
These permissions are necessary to enable the webhook to send necessary data to the Prisma Cloud
IaC service to perform the checks against security policies.
- STEP 2 | Set up the failure criteria for merge request checks.
set the environment variable prisma_cloud_scm_failure_mr_criteria to define the number and severity of security policy check failures that need to occur to trigger a merge request
failure. The syntax for the prisma_cloud_scm_failure_mr_criteria value is as follows.
 High: x, Medium: y, Low: z, Operator: op
In the syntax above, x is a count of high-severity policy check failure, y is a count of medium-severity
policy check failures, and z is a count of low-severity policy check failures. The Operator value
determines what combination of High/Medium/Low counts should result in a merge request failure. The default for each count is 0. The value for Operator, op, can be either OR or AND. The default is OR.
Some examples of settings for prisma_cloud_scm_failure_mr_criteria are as follows.
- The setting below would result in a failed merge request security check for any detected policy check
failure
 High: 0, Medium: 0, Low: 0, Operator: OR
- The setting below would result in merge requests never failing a security check.
 High: 1000, Medium: 1000, Low: 1000, Operator: AND
- STEP 3 | Set up the failure criteria for GitLab issue creation.
set the environment variable prisma_cloud_scm_failure_issue_criteria to define
the number and severity of security policy check failures that need to occur to trigger creation
of a GitLab issue, during a merge request. The syntax of the variable value is the same as that for prisma_cloud_scm_failure_mr_criteria. The value includes High, Medium, and Low counts and includes an
Operator whose possible values are AND and OR. - STEP 4 | Set up the Prisma Cloud tags.
Prisma Cloud tags are different from GitLab tags or cloud tags that might have included within your
IaC templates. Prisma Cloud tags enable visibility on the Prisma Cloud administrator console.
Provide the value for this environment variable as a comma-separated list of tags that define. An
example is: prisma_cloud_scm_tags=project x, owner=mr.y, compliance=pci.
- STEP 5 | Set up a webhook to perform the IaC scan during merge request operations.
1. Navigate to Project > Settings > Webhooks
2. Specify https://scan.api.redlock.io/gitlab/v1 in the URL field.
This URL is the Prisma Cloud SaaS API that supports IaC scanning for GitLab.
3. Provide the GitLab access token that generated earlier in the Secret Token field.
4. Select Merge request events as the trigger.
5. Select Enable SSL verification. 6. Select the Add webhook button to add the webhook just configured.
newly added webhook should appear under the Project Hooks list on the same page.
- STEP 6 | Set Up Prisma Cloud Configuration File for IaC Scan file.
Create the .prismaCloud/config.yml and add it to the root directory of repository branch. The file is
required, and it must include the template type, version, and the template specific parameters and tags
use in environment.
Run an IaC Scan in a Merge Request
When create, update, or reopen a merge request with added or modified files, this set up will trigger a merge request event to invoke a Prisma Cloud IaC scan for all files in the merge request. The scan does not
include deleted files.
see the results of the IaC scan through a comment on the merge request. If
the scan results meets or exceeds the failure criteria set in the environment variable
prisma_cloud_scm_failure_mr_criteria, then the results will show that the security check failed.
The following shows the result of an IaC scan for a merge request. In this example, the IaC scan resulted in
some security policy check failures. Since the number and severity of the failures did not meet the failure
criteria set in the environment variable prisma_cloud_scm_failure_mr_criteria, the security check
passed, and the merge request succeeded.
The following is an example of output that occurs when the failure criteria in the environment variable
prisma_cloud_scm_failure_issue_criteria is met or exceeded. Use the Prisma Cloud Extension for Visual
Studio Code
With the Prisma Cloud Enterprise edition license, install the Prisma Cloud extension for Visual
Studio (VS) Code to detect issues in Infrastructure-as-Code (IaC) templates and deployment files
against Prisma Cloud security policies early in the software development process, directly within VS
Code editor. The following steps show how simple it is to install and check templates and files for potential security misconfigurations.
1. Install Prisma Cloud Extension for Visual Studio Code
2. Configure the Prisma Cloud Extension for VS Code
3. Scan Using the Prisma Cloud VS Code Extension
Install Prisma Cloud Extension for Visual Studio Code
The Prisma Cloud extension supports VS Code version 1.36.0 and later.
- STEP 1 | In VS Code, navigate to Extensions.
- STEP 2 | Enter Prisma Cloud in search.
- STEP 3 | Install the extension.
Configure the Prisma Cloud Extension for VS Code
Before use the Prisma Cloud extension for VS Code, you’ll need to configure the extension to
include API access key, secret key, and Prisma Cloud API URL. If access keys change, must
update the details in the extension settings. - STEP 1 | In VS Code, navigate to Settings > Extensions > Prisma Cloud.
- STEP 2 | Enter the following information for the Prisma Cloud extension:
- Prisma Cloud API URL.
The URL for Prisma Cloud varies depending on the region and cluster on which tenant is
deployed. The tenant provisioned for is, for example, https://app2.prismacloud.io or https://
app.eu.prismacloud.io. Replace app in the URL with api and enter it here. Refer to the Prisma Cloud
REST API Reference, which is accessible from the Help Center within the Prisma Cloud web interface
for more details.
- Access Key.
The Prisma Cloud access key enables programmatic access. If do not have a key, must Create
and Manage Access Keys.
- Secret Key.
should have saved this key when generated Prisma Cloud access key and
corresponding secret key. cannot view the secret key on the Prisma Cloud web interface.
- Asset Name
Give VSCode instance an asset name. choose an arbitrary name. Prisma Cloud uses the asset name to track results. Some examples of names are appteam_vscode or johndoe_vscode.
- Prisma Cloud Tags
Prisma Cloud tags are different from cloud tags that may have included within IaC
templates. Prisma Cloud tags enable visibility in the Prisma Cloud administrator console.
Provide the values as a comma-separated list of tags. in the Prisma Cloud Tags field. An example list
is: owner:johndoe, team:creditapp, env:dev.
- STEP 3 | Set Up Prisma Cloud Configuration File for IaC Scan
Create the .prismaCloud/config.yml file and add it to the root directory of repository branch. The file is required, and it must include the template type, version, and the template specific parameters and
tags use in environment. Scan Using the Prisma Cloud VS Code Extension
Now, are ready to scan templates and view the results within the VS Code editor.
- STEP 1 | Scan a file.
Right-click on template file in the VS Code Explorer and select Prisma Scan to check template
against Prisma Cloud IaC policies.
- STEP 2 | View the scan results.
Select the Prisma Cloud icon on the Activity Bar. The results of the check will appear in the Prisma Cloud Result window. If the extension discovers any
policy violations, the Prisma Cloud Result window sorts the results by severity and displays the following
details for each violation:
- Name of the violated policy -Severity of the violation
- Names of the module or files that have issues
- Timestamp of the scan
When scan a different template, the result window refreshes to display the latest scan results. Use the Prisma Cloud IaC Scan REST API
Prisma Cloud makes the IaC scanning functionality available as a SaaS solution through a REST API.
The Prisma Cloud IaC scan service supports Terraform templates, Terraform plan files in JSON format, CloudFormation templates, and Kubernetes app manifests. While should take advantage of the IaC scan
plugins that are available, have the option to use the IaC scan API directly.
- Use the IaC Scan API Version 2
- Deprecated - IaC Scan API Version 1
- Scan API Version 1 for Terraform Files (Deprecated)
- Scan API Version 1 for AWS CloudFormation Templates (Deprecated)
- Scan API Version 1 for Kubernetes Templates (Deprecated)
The Prisma Cloud IaC scan service API version 2 enables to scan templates against
policies and display scan results asynchronously for a better user experience. Version 1
of the IaC Scan API is deprecated and will continue to work until January 31, 2021. Make
sure to update scripts or integrations that make requests to version 1 resources as soon as possible, to prevent any disruption.
Use the IaC Scan API Version 2
With version 2 of the IaC scan API, initiate IaC scans asynchronously and integrate IaC scan
results with Prisma Cloud. The new asynchronous IaC scan API solves timeout issues, increases the file size
limit to 300MB, and supports Terraform v0.13. Also, this new API detects Terraform module structures and
variable files automatically, in most cases.
To scan templates, identify scan to the Prisma Cloud IaC scan service, upload the files to be
scanned, and submit a job to perform the scan. After submit job, make API requests to check
the status of the job and to view the results.
Before make first IaC Scan v2 request, must complete the following prerequisites:
Know the base API URL for Prisma Cloud tenant, which will serve as the base URL of all IaC
scan API requests.
Prisma Cloud API base URL depends on the region and cluster of Prisma Cloud tenant. For example, if Prisma Cloud admin console URL is https://app.prismacloud.io, then your
Prisma Cloud API base URL is https://api.prismacloud.io. See the Prisma Cloud REST API
Reference for a list of Prisma Cloud API URLs.
Get the Prisma Cloud JSON web token (JWT) token for authentication.
Prisma Cloud administrator normally assigns an API access key, which use with a Prisma Cloud login API request to obtain a JWT token. will include the JWT token in the header of IaC
scan API requests. See Access the Prisma Cloud REST API for details.
Know the Content-Type for request header, which is in the table below.
The following table shows the required request-header fields for all the IaC scan V2 API requests.
Request-Header Field Value
x-redlock-auth JWT token as described above Request-Header Field Value
Content-Type application/vnd.api+json
If want to scan Terraform plan files, an additional prerequisite is to use the Terraform plan and
Terraform show commands to convert Terraform modules to JSON-formatted plan files.
Version 2 of the IaC scan API is JSON API compliant.
The examples below illustrate the API requests need to run and manage an asynchronous IaC scan job
in Prisma Cloud.
- STEP 1 | Create an IaC scan asset in Prisma Cloud.
The following API enables to create an IaC scan asset. An IaC scan asset represents a collection of one or more templates whose contents have been scanned or will be scanned by the Prisma Cloud scan
service to check against Prisma Cloud IaC policies.The Prisma Cloud scan service performs this scan
asynchronously.
Method Endpoint URL
POST https://<Prisma Cloud API base URL>/iac/v2/scans
The following is an example of a cURL request to create an IaC scan asset.
curl -X POST 'https:/<Prisma Cloud API URL>/iac/v2/scans' \
 --header 'x-redlock-auth: <JWT Token> \
 --header 'Content-Type: application/vnd.api+json' \
 --data-raw '
{
 "data": {
 "type": "async-scan",  "attributes": {
 "assetName": "my-asset",  "assetType": "IaC-API",  "tags": {
 "env": "dev"
 },  "scanAttributes": {
 "projectName": "my-project"
 },  "failureCriteria": {
 "high": 1,  "medium": 10,  "low": 30,  "operator": "or"
 }
 }
 }
}'
The request body parameters both provide metadata about scan and define failure criteria for an IaC scan job. The required data.type value must currently be async-scan. The parameters in this
example includes the following attributes: -assetName: Can be a project name or any identifier want to attach to the scan. Some examples
are a CI/CD project name or a Git repository name.
- assetType: Identifies where the IaC scan is being done. For the IaC scan API the standard value
is IaC-API. are allowed to use other values as well. For example, if are requesting a scan
through a GitHub action and using the IaC scan API, set the assetType to GitHub. See the Async Scan API in the Prisma Cloud API Reference for a list of supported asset types.
- tags: Prisma Cloud tags are different from cloud tags that might have included in IaC
templates. Prisma Cloud tags will facilitate use of upcoming Prisma Cloud features like role-based
access control and policy selection. An example tag list is owner:johndoe, team:creditapp, env:dev.
- scanAttributes: Can be any additional details that want to send as key/value pairs with each scan.
Some examples are Git pull request numbers or specific build numbers inside a CI/CD project.
- failureCriteria: Enables to evaluate scan results against set failure criteria to obtain failed or
passed verdicts. set the count for high, medium, and low severity issues and use and or or
operators to refine criteria. The IaC scan API checks each severity violation number separately
against scan results and applies the operator to each evaluation. The scan triggers a failure if the number of violations is greater than or equal to the failureCriteria values. For example, if want
a scan to fail when it detects any type of issue, set the failure criteria to “high”: 1, “medium”: 1, “low”: 1 and”operator”: “or”. With this failureCriteria setting, if the scan
finds any type of issue, the job will fail. An example of failureCriteria that causes the IaC scan API to
run checks in warning mode is “high”: 1000, “medium”: 1000, “low”: 1000, “operator”:
and.
Details about all the body parameters for this request are in Async Scan API in the Prisma Cloud API
Reference.
The following example shows the response of a successful request. In this example, the data.id value
is a scan id that will use in subsequent requests to manage scan job. The data.url value is a presigned URL will use to upload the files want scanned to Prisma Cloud.
{
 "data": {
 "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",  "links": {
 "url": "https://s3.amazonaws.com/s3sign2-bucket-hchq3nwuo8ns/s3-
sign-demo.json?X-Amz-Security-Token=FQ…&X-Amz-Credential=ASIAJF3BXG…&XAmz-Date=20170125T044127Z&X-Amz-Expires=60&X-Amz-Signature=24db0"
 }
 }
}
- STEP 2 | Use the presigned URL from the scan asset creation to upload the templates to be scanned.
Prisma Cloud uses a presigned URL that gives temporary access to upload an object. In the example
above, the presigned URL is the value of the key data.url. The following is an example of a cURL request
to upload a file to the presigned URL.
curl -X PUT '<presigned URL>' --form 'file=@<path and file name of file to
 be uploaded>'
upload a single file or multiple files as a zip archive to the presigned URL.
Some example templates and plan files are available at Prisma Cloud IaC samples on GitHub; use
one of these files to experiment with an upload request. - STEP 3 | Start a job to perform a scan of uploaded templates.
The following API enables start an asynchronous job to perform a scan of uploaded file. The path parameter is the scan ID from the response of earlier request to create a scan asset.
Method Endpoint URL
POST https://<Prisma Cloud API base URL>/iac/v2/scans/{scanID]
The path parameter scanID is the scan ID from the response of earlier request to create a scan
asset.
The following is an example of a cURL request to start a job that scans the file you’ve already uploaded.
curl -X POST 'https://<Prisma Cloud API URL>/iac/v2/
scans/3fa85f64-5717-4562-b3fc-2c963f66afa6' \
 --header 'x-redlock-auth: <JWT Token> \
 --header 'Content-Type: application/vnd.api+json' \
 --data-raw '
{
 "data": {
 "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",  "attributes": {
 "templateType": "tf",  }
 }
}'
The data.id in the request body parameter is the same as the path parameter scanID and is optional.
Optional request parameters are available to handle more complex scenarios, including specification of the following:
- Runtime variables for templates.
- Variable files if the IaC scan API cannot automatically detect the files.
- A list of files to which the scan will be limited.
- A list of folders to which the scan will be limited. With this specification, the IaC scan service will scan
the content of the listed folders, including files and sub-folders.
Details about the API request are in the Initiate Scan Job API in the Prisma Cloud API Reference
A successful request will return an HTTP code of 200.
- STEP 4 | Query job status.
The following API enables to query the status of asynchronous IaC scan job.
Method Endpoint URL
GET https://<Prisma Cloud API base URL>/iac/v2/{scanID}/status
The scanID path parameter is the id received in the response object from the request to create
a scan asset and which also used as a path parameter to start the scan job. The following is an
example cURL request to request the job status. curl -X POST 'https://<Prisma Cloud API URL>/iac/v2/
scans/3fa85f64-5717-4562-b3fc-2c963f66afa6/status' \
- -header 'x-redlock-auth: <JWT Token> \
- -header 'Content-Type: application/vnd.api+json'
The following example shows the response of a successful request.
{
 "data": {
 "id": 12345678-5717-4562-b3fc-2c963f66afa6",  "attributes": {
 "status": "processing"
 }
 }
}
- STEP 5 | Request IaC scan results.
The following API enables to request IaC scan results after the scan job is done.
Method Endpoint URL
GET https://<Prisma Cloud API base URL>/iac/v2/scans/{scanID}/results
The scanID path parameter is the id received in the response object from the request to create
a scan asset and which also used as a path parameter to start the scan job. The following is an
example cURL request to request the job status.
curl -X POST 'https://<Prisma Cloud API URL>/iac/v2/
scans/12345678-5717-4562-b3fc-2c963f66afa6/results' \
- -header 'x-redlock-auth: <JWT Token> \
- -header 'Content-Type: application/vnd.api+json'
The following example shows the response of a successful request.
{
 "meta": {
 "matchedPoliciesSummary": {
 "high": 1,  "medium": 0,  "low": 0
 },  "errorDetails": []
 },  "data": [
 {
 "id": "12345678-5717-4562-b3fc-2c963f66afa6",  "attributes": {
 "severity": "high",  "name": "AWS Security Groups allow internet traffic to SSH port
 (22)",  "rule": "$.resource[*].aws_security_group exists and
 ($.resource.aws_security_group[*].ingress[?(@.protocol == 'tcp'&&
 @.from_port<23 && @.to_port>21 )].cidr_blocks contains 0.0.0.0/0", 462 PRISMA™ CLOUD ADMINISTRATOR'S  "desc": "This policy identifies AWS Security Groups which do allow
 inbound traffic on SSH port (22) from public internet",  "files": [
 "./main.tf"
 ],  "policyId": "617b9138-584b-4e8e-ad15-7fbabafbed1a",  "docUrl": "https://docs.paloaltonetworks.com/prisma/prisma-cloud/
prisma-cloud-policy-reference/configuration-policies/configurationpolicies-build-phase/amazon-web-services-configuration-policies/
policy_617b9138-584b-4e8e-ad15-7fbabafbed1a.html"
 }
 }
 ]
}
Scan API Version 1 for Terraform Files (Deprecated)
Method Endpoint URL
POST https://<Prisma Cloud API base URL>/iac/tf/v1/scan
This REST API request scans a Terraform file or a zip archive that contains multiple Terraform files for comparison against Prisma Cloud security policies. The body of the API request contains the file or zip
archive to be scanned.The module scan can have either Terraform 0.12 or prior version templates.The request-header fields differ, depending on the type of Terraform module want to scan.
The IaC scan service cannot scan any files that are not valid Terraform files, either in the incorrect or invalid .tf format.
Terraform 0.12
The following table shows request-header fields required to request a scan of Terraform 0.12 modules.
Request-Header Field Value Notes
x-redlock-auth JWT token Required
content-type To scan single files: text/
plain. To scan zip archives:
multipart/form-data Required
terraform-version 0.12 Required
terraform-012-parameters An array of key/value pairs that
describe the variables in your
module. See details below.
Required
The value of terraform-012-parameters differs, depending on whether Terraform 0.12 module
has (1) standard variables or (2) custom variable file names and/or external variables. -If the Terraform module has variable files but no external variables, then the array elements that make
up the value of terraform-012-parameters is as follows.
Key Value
root-module Terraform 0.12 root module
The following example shows a cURL request to scan a Terraform 0.12 module that has standard
variables.
curl -X POST ’https://<Prisma Cloud API URL>/iac/tf/v1/scan' \
- -header 'x-redlock-auth: '<JWT token> \
- -header 'Content-Type: multipart/form-data' \
- -header 'terraform-version: 0.12' \
- -header 'terraform-012-parameters: [{"root-module":"/scan/rich-valuetypes/"},{"root-module":"/scan/rich-value-types/network/"}]' \
- -form 'templateFile=@<path and file name of single Terraform file or zip
 archive>'
The following example shows the response of a successful request.
{
 "result": {
 "is_successful": true,  "rules_matched": [
 {
 "severity": "medium",  "name": "AWS S3 Object Versioning is disabled",  "rule": " $.resource[*].aws_s3_bucket exists and
 ($.resource[*].aws_s3_bucket.*[*].*.versioning[*].enabled does not exist
 or $.resource[*].aws_s3_bucket.*[*].*.versioning[*].enabled anyFalse)",  "description":"Thispolicy identifies the S3 buckets
 which have Object Versioning disabled. S3 Object Versioning is an
 important capability in protecting data within a bucket. Once  enable Object Versioning, cannot remove it; suspend Object
 Versioning at any time on a bucket if do not wish for it to persist.
 It is recommended to enable Object Versioning on S3.",  "files": [
 "/scan/for-expressions"
 ],  "id": "89ea62c1-3845-4134-b337-cc82203b8ff9"
 }
 ],  "severity_stats": {
 "high": 0,  "low": 0,  "medium": 1
 }
 },  "response_id": "bb3ba05a-2e31-4fc3-9a8e-91b31f673500"
}
- set the value of terraform-012-parameters to enable a scan of Terraform variable files
with custom names or Terraform external variables. If Terraform module has either of these
variable uses, then the value of terraform-012-parameters is an array of key/value pairs where the key/value pairs can be one or more of the following. Key Value
root-module Terraform 0.12 root module
variable-files An array of custom variable file names. The path
of each file is relative to root module.
variables An array of key/value pairs. Each array
element has a name and value that
together identify an input variable (e.g.
[{“name”:”varName1”,”value”:”varValue1}, {“name”:”varName2”,”value”:”varValue2”}]
The following example shows a cURL request to scan a Terraform 0.12 zip archive that has custom
variable file names.
curl -X POST 'https://<Prisma Cloud API URL>/iac/tf/v1/scan' \
- -header 'x-redlock-auth: <JWT token>' \
- -header 'Content-Type: multipart/form-data' \
- -header 'terraform-version: 0.12' \
- -header 'terraform-012-parameters: [{"root-module":"/scan/rich-valuetypes/"},{"root-module":"/scan/rich-value-types/network/","variablefiles":["/scan/rich-value-types/network/variables.tf"]},{"root-module":"/
scan/for-expressions/"}]' \
- -form 'templateFile=@<absolute file path of template or zip>'
The following example shows a cURL request to scan a Terraform 0.12 zip archive that has external
variables.
curl -X POST 'https://<Prisma Cloud API URL>/iac/tf/v1/scan' \
- -header 'x-redlock-auth: <JWT token>' \
- -header 'Content-Type: multipart/form-data' \
- -header 'terraform-version: 0.12' \
- -header 'terraform-012-parameters: [{"root-module":"/", "variables":
 [ {"name": "region", "value": "us-west-1" }, { "name" : "bucket",  "value": "testbucket"}] } ]' \
- -form 'templateFile=@<absolute file path of template or zip>'
- The following example shows a cURL request to scan a Terraform 0.12 plan file that is in JSON format.
curl --location --request POST 'https://api.prismacloud.io/iac/tf/v1/scan'
 \
- -header 'x-redlock-auth: '<JWT token> \
- -form 'templateFile=@<absolute file path of plan JSON file>'
Note that, as the request above shows, the only required header is x-redlock-auth. The following is
an example of successful response to this request.
{
 "result": {
  "rules_matched": [
 {
 "severity": "medium",  "name": "AWS S3 Object Versioning is disabled",  "rule": " $.resource[*].aws_s3_bucket exists and
 ($.resource[*].aws_s3_bucket.*[*].*.versioning[*].enabled does not exist
 or $.resource[*].aws_s3_bucket.*[*].*.versioning[*].enabled anyFalse)",  "description": "This policy identifies the S3 buckets which
 have Object Versioning disabled. S3 Object Versioning is an important
 capability in protecting data within a bucket. Once enable
 Object Versioning, cannot remove it; suspend Object
 Versioning at any time on a bucket if do not wish for it to persist.
 It is recommended to enable Object Versioning on S3.",  "id":"89ea62c1-3845-4134-b337-cc82203b8ff9"
 }
 ],  "severity_stats": {
 "high": 0,  "low": 0,  "medium": 1
 }
 },  "response_id": "35760530-70d3-4652-b4d2-2a06a9eb776e"
}
Terraform 0.11
The following table shows request-header fields required to request a scan of Terraform 0.11 modules that
have only standard variables.
Request-header Field Value Notes
x-redlock-auth JWT token Required
content-type To scan single files: text/
plain. To scan zip archives:
multipart/form-data Required
The following is an example of a cURL request to scan a Terraform 0.11 module that has only standard
variable file names.
curl -X POST 'https://<Prisma Cloud API URL>/iac/tf/v1/scan' \
- -header 'x-redlock-auth: <JWT token>' \
- -header 'Content-Type: multipart/form-data' \
- -form 'templateFile=@<absolute file path of template or zip>'
The following table shows request-header fields required to request a scan of Terraform 0.11 modules that
have custom variable file names or external variables.
Request-header Field Value Notes
x-redlock-auth JWT token Required Request-header Field Value Notes
content-type To scan single files: text/
plain. To scan zip archives:
multipart/form-data Required
rl-parameters An array of key/value pairs.
Each array element has a name and value that together
identify an input variable (e.g.
[{“name”:”varName1”,”value”:”varValue1}, {“name”:”varName2”,”value”:”varValue2”}])
Required for input variables
rl-variable-file-names An array of variable file names.
The path of each file is relative
to repository branch root
directory
Required for variable files
The following is an example of a cURL request to scan a Terraform 0.11 module that has custom file names
and external variables.
curl -X POST 'https://<Prisma Cloud API URL>/iac/tf/v1/scan' \
- -header 'x-redlock-auth: <JWT token>' \
- -header 'Content-Type: multipart/form-data' \
- -header ‘rl-parameters: “[{"name":"varName1","value":"varValue1"}, {"name":"varName2","value":"varValue2"}]”’ \
- -header 'rl-variable-file-names: ["vars.tf.json", "file1.tf"]' \
- -form 'templateFile=@<absolute file path of template or zip>'
The following is an example of a successful response to the request above.
{
 "result": {
 "is_successful": true,  "rules_matched": [
 {
 "severity": "high",  "name": "AWS Security Groups allow internet traffic to SSH
 port (22)",  "rule": "$.resource[*].aws_security_group exists and
 ($.resource[*].aws_security_group[*].*[*].ingress[?( @.protocol == 'tcp'&&
 @.from_port<23 && @.to_port>21 )].cidr_blocks[*] contains 0.0.0.0/0 or
 $.resource[*].aws_security_group[*].*[*].ingress[?( @.protocol == 'tcp' &&
 @.from_port<23 && @.to_port>21 )].ipv6_cidr_blocks[*] contains ::/0)",  "description": "This policy identifies AWS Security Groups
 which do allow inbound traffic on SSH port (22) from public internet. Doing
 so, may allow a bad actor to brute force their way into the system and
 potentially get access to the entire network.",  "files": [
 "demo/securitygroup22.tf"
 ],  "id": "617b9138-584b-4e8e-ad15-7fbabafbed1a"
 },  "severity_stats": {  "high": 0,  "low": 0,  "medium": 1
 }
 },  "response_id": "bb3ba05a-2e31-4fc3-9a8e-91b31f673500"
}
Scan API Version 1 for AWS CloudFormation Templates
(Deprecated)
Method Endpoint URL
POST https://<Prisma Cloud API base URL>/iac/cft/v1/scan
This REST API request scans AWS CloudFormation template files for comparison against Prisma Cloud
security policies. Support exists for both JSON and YAML formats. Prisma Cloud IaC API also supports
parameters for CloudFormation templates. also scan either a single template or a zip archive
of template files with a single API request. Note that scan support does not currently exist for nested
references, macros, or intrinsic functions in CloudFormation templates.
The following table shows the request-header fields. The body of the API request contains the file or zip
archive to be scanned.
Request-header Field Value Notes
x-redlock-auth JWT token Required
content-type To scan single files: text/
plain. To scan zip archives:
multipart/form-data Required
rl-parameters An array of key/value pairs.
Each array element has a name and value that together
identify a parameter (e.g.
[{“name”:”varName1”,”value”:”varValue1}, {“name”:”varName2”,”value”:”varValue2”}])
Required for parameters
The following example shows a cURL request to scan an AWS CloudFormation template with external
variables.
curl-X POST ’https://<Prisma Cloud API URL>/iac/cft/v1/scan' \
- -header 'x-redlock-auth: <JWT token>' \
- -header 'Content-Type: multipart/form-data' \
- -header ‘rl-parameters: “[{"name":"varName1","value":"varValue1"}, {"name":"varName2","value":"varValue2"}]”’ \
- -form 'templateFile=@<absolute file path of template or zip>'  The following is an example of a successful response to this request.
{
 "result": {
 "is_successful": true,  "rules_matched": [
 {
 "severity": "high",  "name": "AWS Security Groups allow internet traffic to SSH
 port (22)",  "rule": "$.resource[*].aws_security_group exists and
 ($.resource[*].aws_security_group[*].*[*].ingress[?( @.protocol == 'tcp'&&
 @.from_port<23 && @.to_port>21 )].cidr_blocks[*] contains 0.0.0.0/0 or
 $.resource[*].aws_security_group[*].*[*].ingress[?( @.protocol == 'tcp' &&
 @.from_port<23 && @.to_port>21 )].ipv6_cidr_blocks[*] contains ::/0)",  "description": "This policy identifies AWS Security Groups
 which do allow inbound traffic on SSH port (22) from public internet. Doing
 so, may allow a bad actor to brute force their way into the system and
 potentially get access to the entire network.",  "files": [
 "cftdemo/cft_sg.json"
 ],  "id": "617b9138-584b-4e8e-ad15-7fbabafbed1a"
 },  "severity_stats": {
 "high": 0,  "low": 0,  "medium": 1
 }
 },  "response_id": "bb3ba05a-2e31-4fc3-9a8e-91b31f673500"
}
Scan API Version 1 for Kubernetes Templates (Deprecated)
Method Endpoint URL
POST https://<Prisma Cloud API base URL>/iac/k8s/v1/scan
This REST API request scans Kubernetes manifests to compare against Prisma Cloud security policies, including manifests that generate from Helm charts. scan either a single manifest or a zip
archive of manifest files with a single API request.
The following table shows the request-header fields. The body of the API request contains the file or zip
archive to be scanned.
Request-header Field Value Notes
x-redlock-auth JWT token Required
content-type To scan single files: text/
plain. To scan zip archives:
multipart/form-data Required The following example shows a cURL request to scan a single Kubernetes manifest file.
curl --location --request POST 'https://<Prisma Cloud API URL>/iac/k8s/v1/
scan' \
- -header 'x-redlock-auth: <JWT token>' \
- -header 'Content-Type: multipart/form-data' \
- -form 'templateFile=@<absolute file path of template or zip>'
The following is an example of a successful response object.
{
 "result": {
 "is_successful": true,  "rules_matched": [
 {
 "severity": "high",  "name": "All capabilities should be dropped",  "rule":
 "$.spec.template.spec.containers[*].securityContext.capabilities.drop exists
 and not
 $.spec.templates.spec.containers[*].securityContext.capabilities.drop[*]
 contains ALL",  "description": "Ensure that all capabilities are dropped.",  "id": "4682a6f1-2a1b-4f5a-938c-cdd3fa421a63"
 },  {
 "severity": "medium",  "name": "Do not run containers with dangerous capabilities",  "rule":
 "$.spec.template.spec.containers[*].securityContext.capabilities exists and
 $.spec.template.spec.containers[*].securityContext.capabilities.add[*] is
 member of (FSETID, SETUID,  SETGID,SYS_CHROOT,SYS_PTRACE,CHOWN,NET_RAW,NET_ADMIN,SYS_ADMIN,NET_BIND_SERVICE)",  "description": "Ensure not running containers with dangerous
 capabilities.",  "id": "135420a6-3206-4c29-b944-846f65cea43e"
 }
 ],  "severity_stats": {
 "high": 1,  "low": 0,  "medium": 1
 }
 },  "response_id": "ddd6d597-e560-4e67-abd1-4bc2cedee062"
} 471
Prisma Cloud Data Security
Prisma Cloud Data Security is Limited GA available to select Prisma Cloud Enterprise customers
only. The Data Security capabilities on Prisma Cloud enable to discover and classify data stored in AWS S3 buckets and protect accidental exposure, misuse, or sharing of sensitive
data. To identify and detect confidential and sensitive data, Prisma Cloud Data Security
integrates with Palo Alto Networks’ Enterprise DLP service and provides built-in data profiles, which include data patterns that match sensitive information such as PII, health care, financial
information and Intellectual Property. In addition to protecting confidential and sensitive
data, data is also protected against threats—known and unknown (zero-day) malware—using the Palo Alto Networks’ WildFire service.
> What is Included with Prisma Cloud Data Security
> Enable the Prisma Cloud Data Security Module
> Monitor Data Security Scan Results on Prisma Cloud
> Disable Prisma Cloud Data Security and Offboard AWS account
> Guidelines for Optimizing Data Security Cost on Prisma Cloud What is Included with Prisma Cloud Data Security
- AWS S3 support for USA regions.
- Ability to scan all or selected S3 buckets when onboard AWS account(s) on Prisma Cloud. choose to enable a forward or backward scan when add the cloud account. The scan quota for each tenant is 10TB; this quota allows to limit how much data is scanned. It is adjustable and contact Prisma Cloud Customer Success to increase it for needs.
Prisma Cloud Data Security needs to read objects stored on AWS S3 buckets for scanning them.
- Visibility, exposure, and classification of S3 buckets & objects on the new Data Dashboard, Data Inventory, and Object Explorer.
- S3 objects in standard storage class only are ingested for scanning.
- File sizes and scanning:
- For data classification and malware scanning, the uncompressed file size must be less than 20MB.
For example, if the file size is 25MB, but was compressed to under 20MB the file will not be
successfully scanned.
- Malware detection of objects (only Windows executables & Linux binaries).
- For ML-based classification scanning, the file size must be less than 1MB.
- For backward scan, each tenant has a daily limit of 300GB.
For forward scan each tenant has 10GB per hour. When this threshold is met, the scanning is
slower for the files in queue until the hour is reset.
- Default Data policies to detect public exposure of sensitive information. The data policies, currently
five, generate alerts on Prisma Cloud and set up notification to external integration channels
supported on Prisma Cloud.
- Freemium experience that offers 300GB per tenant, before are charged for using the Data Security
module. When data exceeds the freemium threshold use credits from the Prisma Cloud
Enterprise Edition license.
- Integration with Config RQL to show all objects in an S3 bucket, including exposure, Data Profile &
malware detection in Resource Explorer. Enable the Prisma Cloud Data Security Module
Prisma Cloud Data Security requires to configure an AWS CloudTrail bucket. To save cost, ensure that
follow the instructions to only select Write events instead of Read and Write events.
- STEP 1 | Log in to the Prisma Cloud administrative console.
1. Launch a browser and go to the following url: https://<Prisma Cloud Instance>
- STEP 2 | On the portal page enter Username and Password and then click Sign In.
- STEP 3 | Select Subscription to Learn More on Data Security.
- STEP 4 | Enable Data Security
1. Click the Settings tab and click Cloud Accounts and click Add New. Click AWS from the popup and
then click the Data Security check box. - STEP 5 | Edit An existing account or onboard a new account to get started with scanning the data for a specific AWS account.
- If select “Onboard a New Account”, click AWS in the new popup screen. Then go to the Configuration > Add a new AWS Account section in this document for instructions on how to
onboard a new AWS account and enable Prisma Cloud Data Security for it.
- If select “Edit An Existing Account”, move to the Configuration > Edit an existing AWS Account
section in this document for instructions on how to enable Prisma Cloud Data Security for an already
onboarded AWS account. Add a New AWS Account and Enable Data Security
1. Go to Settings > Cloud Accounts > Add New.
2. Select AWS and add a Cloud Account Name.
3. Select Data Security.
Only Monitor mode is supported.
4. Select Next.
5. Create a stack in AWS account.
6. Select Create Stack.
NOTE: Log in to AWS account in a separate tab. The CloudFormation template defaults to N.
Virginia. must change it to the region where AWS Account is before create the stack. 1. Select I acknowledge that AWS CloudFormation might create IAM resources with custom names
and Create stack .
Wait for the CREATE_COMPLETE status.
2. Copy Role: ARN & SNS Topic: ARN from the Outputs tab in the AWS Management Console.
3. Paste Role: ARN & SNS Topic: ARN into Prisma Cloud Onboarding Setup screen
7. Setup AWS CloudTrail & SNS. 1. Create new CloudTrail or use an existing CloudTrail
2. Select Write-only events to save cost.
3. Select all S3 buckets in account or Add S3 bucket for only specific buckets. Select Write events
only.
4. Create New or Use an existing S3 bucket
Click Advanced 5. Select Send SNS notification for every log file delivery - Yes, and select SNS topic - PrismaCloudSNS.
This was created earlier when created the stack.
6. Click Create.
Confirm that the CloudTrail bucket is created.
7. . Create bucket policy to enable Prisma Cloud to read from CloudTrail bucket. 8. Click Next.
8. Select the S3 buckets in which want to scan data.
choose to scan All or Custom storage buckets. Scanning all buckets is an expensive operation
depending on the number of files in all buckets and scan type.
NOTE: Currently active and old CloudTrail buckets will be skipped because they contain AWS CloudTrail
generated logs instead of customer data. Objects containing ELB access logs, VPC flow logs and S3
access logs are also automatically skipped.
- Select whether want to scan data Forward only or Forward and Backward.
Forward scan is enabled by default and cannot be disabled. Prisma Cloud scans all new files added or
edited by a user in the bucket near real time.
When select Backward scan, Prisma Cloud starts scanning all existing files in the bucket in a batch operation. Depending on number of files in bucket, backward scan can be expensive. -Custom buckets will list all buckets in AWS account.
Custom option lets choose individual buckets to scan based on scan type (recommended)
- Choose buckets and select a scan type for each bucket.
9. Click Next and select Account Groups
10.Click Next and wait for a few seconds for the configuration to complete. Edit an AWS Account Onboarded on Prisma Cloud to Enable Data Security
1. Go to Settings > Cloud Accounts.
2. Select an AWS account that is currently onboarded on Prisma Cloud.
3. Select Data Security and Next.
NOTE: In the first release, only Monitor model is offered for Data Security.
4. Update stack using instructions below
1. Download CFT here and login to AWS Console to update stack and go through steps 1 - 9 in the Cloud Onboarding Setup window. 2. Go to AWS Management Console -> Stacks. Select PrismaCloudApp Stack (if have previously
used CFT to deploy PrismaCloudApp Stack) and Click “Update” button. If not, pick the stack manually created for Prisma Cloud.
3. Click Update
4. Select “Replace current template” and “Upload a template file”. Then upload the CFT downloaded earlier from step a) and click the “Next” button. 5. Copy the “Callback URL” from Prisma Cloud - Cloud Accounts - Configure Account page
6. Paste the “Callback URL” in the SNSEndpoint field in the “Specify stack details” page in the AWS
Management Console. Click “Next”.
7. Click “Next” to go through the next couple of screens until get to this page to complete the Update operation. Click “Update stack”. 8. The update CFT operation internally will also create a PrismaCloudSNS Topic and will be used to
monitor CloudTraill data events
9. Copy RoleARN from AWS Management Console’s Outputs tab in the stack
1. Paste Role ARN into the Prisma Cloud Cloud - Cloud Accounts - Configure Account page, replacing
the previous Role ARN
1. Copy SNS ARN from AWS Management Console’s Outputs tab in the stack 2. Paste the SNS Topic: ARN into Prisma Cloud - Cloud Accounts - Configure Account page. Click
“Next” to continue.
5. Setup AWS CloudTrail & SNS.
1. Create new CloudTrail or use an existing CloudTrail
2. Select Write-only events to save cost. 3. Select all S3 buckets in account or Add S3 bucket for only specific buckets. Select Write events
only.
4. Create New or Use an existing S3 bucket
Click Advanced
5. Select Send SNS notification for every log file delivery - Yes, and select SNS topic - PrismaCloudSNS.
This was created earlier when created the stack. 6. Click Create.
Confirm that the CloudTrail bucket is created.
7. . Create bucket policy to enable Prisma Cloud to read from CloudTrail bucket.
8. Click Next. 6. Select the S3 buckets in which want to scan data.
choose to scan All or Custom storage buckets. Scanning all buckets is an expensive operation
depending on the number of files in all buckets and scan type.
NOTE: Currently active and old CloudTrail buckets will be skipped because they contain AWS CloudTrail
generated logs instead of customer data. Objects containing ELB access logs, VPC flow logs and S3
access logs are also automatically skipped.
- Select whether want to scan data Forward only or Forward and Backward.
Forward scan is enabled by default and cannot be disabled. Prisma Cloud scans all new files added or
edited by a user in the bucket near real time.
When select Backward scan, Prisma Cloud starts scanning all existing files in the bucket in a batch operation. Depending on number of files in bucket, backward scan can be expensive. -Custom buckets will list all buckets in AWS account.
Custom option lets choose individual buckets to scan based on scan type (recommended)
- Choose buckets and select a scan type for each bucket.
7. Click Nextand select Account Groups.
8. Click Next and wait for a few seconds for the configuration to complete Monitor Data Security Scan Results on Prisma Cloud
- Resource Explorer
- Data Inventory
- Data Dashboard
- Object Explorer
- Exposure Evaluation
- Supported File Extensions—Prisma Cloud Data Security
- Supported Data Profiles & Patterns
Use the Data Policies to Scan
Prisma Cloud includes default data policies to help start scanning.
1. Create a new alert rule or edit an existing rule.
1. Select the Data policies that want to scan against S3 buckets.
- Healthcare information public exposed
- Intellectual Property public exposed
- Objects containing Financial Information publicly exposed
- Objects containing PII data public exposed
- Objects containing malware
2. Select the notification channels.
Prisma Cloud Data Security only supports—Amazon SQS, Splunk, and Webhook integration. See
Configure External Integrations on Prisma Cloud.
3. Confirm to save the alert rule.
2. Optional Create a custom data policy.
must first onboard an account and enable Data Security before create a custom data policy.
1. Select Policies > New Policy > Data.
Enter a Policy Name, Description, Severity, and Labels for the new policy.
2. Select a data classification profile. select one of the four data profiles—Financial Information, Healthcare Information, Intellectual Property, PII.
3. Select the File Exposure.
Private, Conditional, or Public. See Exposure Evaluation.
4. Select the file extensions that want to scan for sensitive information.
For example, txt. If select Financial Information, Public, and txt, the policy will generate an alert if
a publicly exposed .txt file has Financial Information. Do not use a dot before the extension; if do, an error message displays.
5. Save changes.If would like to define a rule to scan for information specific to your
requirements, create a custom policy, and add it to an alert rule to generate alerts.
3. View Data Policy alerts.
1. Select Alerts > Overview.
2. Filter on Policy Type—Data, to see all alerts related to Data policies.
3. Select an alert to view details on:
Click Bucket Name to see bucket information in the Data Inventory.
Click Object Name to see object information in Data Inventory, Object Explorer.
Click on Alert Rule to see the Alert Rule that generates this particular Alert instance
Data Security settings
1. Click on Settings -> Data 2. See Scan Settings & Data Profiles tabs
Note: The Scan Settings tab is not enabled yet in V1. It will be in the future to allow the user to control
which file extensions to ignore for scanning and enable/disable Malware scanning.
3. Click on Data Profiles tab to show the list of Data Profiles enabled - Financial Information, Healthcare, Intellectual Property & PII.
4. Click on any of the Profiles to see all the Data Patterns supported. 5. See all the Data Patterns supported for any of the four Data Profiles supported - Financial Information, Healthcare, Intellectual Property & PII.
6. Click on any of the Data Patterns to see additional details.
7. See details of any specific Data Pattern. Data Dashboard
The new Data Dashboard tab provides complete visibility into S3 storage. The dashboard widgets
below give insight into how many storage buckets and objects have, what kind of data is stored
in those objects, across which regions, who owns what and what is the exposure of the objects. This tab is
available under the Dashboard menu.
1. Total Buckets
This widget shows the total number of buckets (except empty buckets) discovered in AWS
Account. Buckets are categorized into Private and Public.
- Private buckets are internal and not publicly accessible.
- Public Buckets are accessible to everyone. See Exposure Evaluation to learn how exposure is
calculated.
click on either the Private or Public circle in the widget to view those buckets in the Data Inventory.
2. Total Objects
This widget shows the total number of objects discovered in all S3 storage buckets. Objects are
categorized into Public, Sensitive and Malware. Public objects are accessible to everyone. Sensitive
objects contain data such as Financial Information, Healthcare, PII and Intellectual Property. Malware
objects contain malicious code.
Click on either the Public, Sensitive of Malware circle in the widget to see those objects in the Data Inventory view. 3. Top Publicly Exposed Objects By Profile
1. This widget shows top 5 publicly exposed objects with Data Profiles of Financial Information, Healthcare, PII and Intellectual Property. Click on any of the bars in the widget to view those objects
in the Data Inventory.
4. Top Object Owners by Exposure
This widget shows top 5 objects owner with exposure (Public, Private or Conditional). click
on any of the bars in the widget to view objects in the Data Inventory. 5. Data Alerts by Severity
This widget shows the breakdown of Data Alerts by Severity (High, Medium, Low). click
on any particular severity segment in the circle to see those Alerts in the Alerts view. The data for this
widget is based on the timestamp of when the alert was generated, while data on other widgets use the objects created/updated timestamp.
6. Top Data Policy Violations
1. This widget shows the top Data Policy violated by objects in S3 buckets. The data for this
widget is based on the timestamp of when the alert was generated, while data on other widgets use
the objects created/updated timestamp.click on any bar in the widget to see those Alerts in
the Alerts view. 7. Object Profile by Region
1. This chart shows object profiles such as Financial Information, Healthcare, PII and Intellectual
Property across AWS Regions. Click on any region in the widget to see those objects in the Data Inventory view.
Data Inventory
The new Data Inventory tab provides detailed insights into S3 storage objects. The Data Inventory
page comprises data cards, objects insights and a detailed inventory view of objects across accounts, account groups, buckets and regions. It also provides a number of filters such as Time Range, Bucket Name, Bucket Exposure, Object Profile, Object Pattern etc. to allow the user to filter the specific buckets/objects
they are interested in. This tab is available under the Inventory menu. 1. The Data Inventory page shows 6 data cards on top :
- Total Buckets
- Total number of buckets discovered in AWS (except empty buckets)
- Public Buckets
- Total number of buckets identified as public based on exposure
- Total Objects
- Total number of files discovered in buckets
- Public Objects
- Total objects with exposure public.
- Sensitive Objects
- Total number of objects containing sensitive data such as Financial Information, Healthcare, PII
and Intellectual Property
- Malware Object
- Total number of objects identified by Wildfire as Malware
1. The Inventory table at the bottom of this page represents a hierarchical view of data grouped by account
name, service name and region name. There are 4 views available on this table with the default view
under cloud type followed by service name followed by bucket view followed by object view.
- View 1 (Cloud View)
- View 2 (Service View) -View 3 (Bucket View)
- View 4 (Object View)
The Object View above includes the following information
- Object Name -Name of the file as discovered in the bucket
- Object Exposure
- Private, public or conditional
- Data Profile
- Object content is classified under one of the following profiles: Financial Information, Healthcare, PII
and Intellectual Property
- If an object belongs to any of the above categories, it is identified as sensitive data -Object Patterns
- Profiles are categorized by one or more patterns; e.g., an object with the PII profile may have patterns
like Driver’s License #, SSN, etc.
- Malware
- Yes/No/Unknown
- User
- Owner of the object
- Bucket Name
- Name of the bucket that the object belongs to
- Account Name
- Name of the account that the object belongs to
- Region Name
- Name of the region that the object belongs to
- Service Name
- Name of cloud storage service (e.g. S3)
- Last Modified
- Object creation time or last updated time in S3.
Resource Explorer
The Resource Explorer provides visibility into each cloud resource monitored by Prisma Cloud. The Resource Explorer for S3 bucket has been extended to show all objects that belong to a particular
bucket. now view all objects with object name, object profile, object pattern, malware and
exposure for each object. Type the following RQL or config where cloud.service = 'Amazon S3'
See Resource Explorer including the new Objects tab. Object Explorer
The new Object Explorer provides granular visibility into each object. view the metadata for each
object along with exposure and Data Profile.
1. Select any object in Data Inventory -> Level 4 (Object View) or Alerts -> Overview -> Click on specific
Alert -> Expand on an Alert instance to open Object Explorer or click on any object in the Resource
Explorer’s Objects tab to access the Object Explorer.
2. Objects that contain malware display a malware icon -Optionally download the malware report (currently available only in XML).
3. The top left pane of the object explorer presents object metadata details.
- Type—The type of file
- Owner—The owner of the file
- Created On—Timestamp of when the file was created on AWS
- Last Updated On—Timestamp of when the file was last updated on AWS
- Region—The region where the object is stored.
- Bucket Name—Name of the bucket where the object is stored.
- Tags—AWS tags that help in identifying the file.
- URL—AWS URL for accessing the file.
4. The top left pane of the object explorer presents object exposure details.
- Exposure Type—Exposure derived based on bucket policy, object ACL and bucket ACL. It can Public, Private or Conditional.
- Last Matched—Most recent exposure derivation.
- Object ACL—JSON attribute that defines the object ACL
5. The right side of the page represents Data Profile
- Object Profile—Object content is classified under one of the following profiles: Financial Information, Healthcare, PII and Intellectual Property.
- Object Patterns—List the pattern and the profile with which it is associated. For example, a profile
can be PII with pattern that matches Driver’s License number or SSN.
- Frequency—Number of times the pattern occurred in the file.
- Detection Time—Timestamp of when the pattern was detected.
Exposure Evaluation
Prisma Cloud and the Cloud Service Provider both monitor the configuration of a bucket and access to the objects within. Exposure evaluation on Prisma Cloud Data Security determines the access level defined
for an object within a bucket, that is who has access to the object, and whether the user(s) can download
or exfiltrate content from a bucket. While on AWS, exposure is an attribute of the bucket or object in the Cloud Service Provider's (CSP) Storage service which allows applications and end users to access structured
data for their use cases.
Prisma Cloud Data Security categorizes exposure levels as follows: Exposure Level Description
Public Is a file/bucket accessible to everyone? If so this may
be a potential risk and needs to be reviewed by the Customer.
Private The file/bucket is internal and not publicly accessible.
This is the safest state to be in for the customer.
However there may be legit reasons why some files
are Public e.g. CDN web templates hosted by a server
etc.
Conditional This usually applies to resources that have a Bucket
policy attached to it allowing access when some set of conditions are met. These conditions are contextual
and cannot be deterministically resolved as they may
be specific to the customer’s environment.
Some examples include:
- access to a bucket is only allowed/denied within a time window
- access to a bucket is only allowed when there is a match on user principal or a specific set of users access this bucket.
- Access to a bucket is allowed/denied if the client
request IP matches a certain range mentioned as a whitelist.
How is Exposure Evaluated?
This is again dependent on each service provider but here we will look closer at AWS and S3 service. S3
access evaluation performs a least-privilege based evaluation. This means a default Deny, explicit Deny
takes precedence over any Allow.
Bucket Exposure Evaluation
- Normalize all controls into a Policy Document (i.e. Bucket ACL, Bucket Policy)
- Evaluate all policy documents normalized above following the steps outlined above in the diagram. The evaluation is checked against a known set of S3 specific API methods/Actions to check for allow and/or
deny.
Supported Bucket Events are:
- DeleteBucket
- CreateBucket -DeleteBucketPolicy -PutBucketAcl
- PutBucketPolicy -If the final result comes out to be that the bucket is publicly accessible i.e. either the whitelisted set of actions are allowed for everyone globally then the verdict is presented as Public.
- If the final result is a Deny for the set of known actions against all policy documents for public users -
then the verdict is considered Private.
- If any of the policy document contains Conditional tags indicative of access to the resource under
specific conditions, the verdict returned is Conditional. Here we expect feedback from the customer to
evaluate the risk posture for the bucket.
Object Exposure Evaluation
- The same steps are followed again as bucket exposure influences object exposure. In addition to the normalized bucket ACL and bucket policy we also normalize the object ACL and factor it into the evaluation.
Supported Object Events are:
- DeleteObject (from CLI, limitation of AWS https://forums.aws.amazon.com/thread.jspa?
threadID=261594)
- PutObject
- PubObjectAcl
- PutObjectTagging
- DeleteObjectTagging
- All steps for Bucket policy evaluation is followed again to determine the eventual exposure verdict of the file/object.
Supported File Extensions—Prisma Cloud Data Security
Review the file extensions that Prisma Cloud scans on storage buckets.
File extensions supported for Data Profile / Data Patterns
File extensions supported for Malware scanning
.JAR .exe
.DOC, .DOCX, .XLS, .XLSX, .PPT, .PPTX .dll
.pdf .so
.tar, .rar, .zip, .7zip, .gz .o
BAT, JS, VBS, PSI, Shell Script, HTA .a
.HTML, .XHTML .elf
.xml .pl
.c, .js, .py, .pyc, .r, .rb, .v, .vhdl, .java, .asm, .ps1, .vb .ko
.mbox, .msg, .pst
.odt, .ods , .odp, .ott, .ots, .otp .numbers, .pages, .keynote
.ibooks, .epub
.rss
.chm
.otf, .ttf
.docm
.dotm
.xlm
.xlsm
.xltm
.pptm
.potm
.ppsm
.sldm
.odt
.rtf
.text
.txt
.json
Supported Data Profiles & Patterns
Object Profiles Object Patterns
PII Driver License - Austria
Driver License - Belgium
Driver License - Bulgaria
Driver License - Cyprus
Driver License - CzechRepublic Driver License - Germany
Driver License - Denmark
Driver License - Estonia
Driver License - Spain
Driver License - Finland
Driver License - France
Driver License - UK
Driver License - Greece
Driver License - Croatia
Driver License - Hungary
Driver License - Ireland
Driver License - Iceland
Driver License - Italy
Driver License - Liechtenstein
Driver License - Lithuania
Driver License - Luxembourg
Driver License - Latvia
Driver License - Malta
Driver License - Netherlands
Driver License - Norway
Driver License - Poland
Driver License - Portugal
Driver License - Romania
Driver License - Sweden
Driver License - Slovenia
Driver License - Slovakia
Driver License - Canada Driver License - US
Passport - Austria
Passport - Belgium
Passport - Bulgaria
Passport - Cyprus
Passport - CzechRepublic
Passport - Germany
Passport - Denmark
Passport - Estonia
Passport - Spain
Passport - Finland
Passport - France
Passport - UK
Passport - Greece
Passport - Croatia
Passport - Hungary
Passport - Ireland
Passport - Iceland
Passport - Italy
Passport - Liechtenstein
Passport - Lithuania
Passport - Luxembourg
Passport - Latvia
Passport - Malta
Passport - Netherlands
Passport - Norway
Passport - Poland Passport - Portugal
Passport - Romania
Passport - Sweden
Passport - Slovenia
Passport - Slovakia
Passport - Canada
Tax Id - US
Tax Id - Austria
Tax Id - Belgium
Tax Id - Bulgaria
Tax Id - Cyprus
Tax Id - CzechRepublic
Tax Id - Germany
Tax Id - Denmark
Tax Id - Estonia
Tax Id - Spain
Tax Id - Finland
Tax Id - France
Tax Id - UK
Tax Id - Greece
Tax Id - Croatia
Tax Id - Hungary
Tax Id - Ireland
Tax Id - Iceland
Tax Id - Italy
Tax Id - Liechtenstein
Tax Id - Lithuania Tax Id - Luxembourg
Tax Id - Latvia
Tax Id - Malta
Tax Id - Netherlands
Tax Id - Norway
Tax Id - Poland
Tax Id - Portugal
Tax Id - Romania
Tax Id - Sweden
Tax Id - Slovenia
Tax Id - Slovakia
Tax Id - Australia
National Id - Austria -Central Register of Residents
National Id - Austria Social Security Card (e-card)
National Id - Belgium -National Registration Number
National Id - Belgium- Citizen Service Number (BSN)
National Id - Bulgaria - Uniform Civil Number
National Id - Cyprus - Identity Card
National Id - Czech - Birth Number
National Id - Czech - National eID Card
National Id - Germany
National Id - Denmark - CPR Number
National Id - Estonia - Personal Identification Code
National Id - Spain -National Identity Document
(Documento Nacional de Identidad)
National Id - Spanish NIE Number
National Id - Finland - Personal Identity Code National Id - UK National Insurance Number
National Id - Greece
National Id - Croatia - Personal Identification Number
National Id - Hungary - Personal Identification Number
National Id - Ireland - Personal Public Service Number
(PPSN)
National Id - Iceland
National Id - Italy - Fiscal Code Card (Codice Fiscale)
National Id - Liechtenstein
National Id - Lithuania
National Id - Luxembourg
National Id - Latvia - Personal Public Service Number
(PPSN)
National Id - Malta
National Id - Netherlands - Citizen Service Number (BSN)
National Id - Norway - Identification Number
(Fødselsnummer)
National Id - Poland
National Id - Portugal
National Id - Romania - Identity Card (CNP)
National Id - Sweden - Personal Identity Number
National Id - Slovenia
National Id - Slovakia
National Id - Brazil - CPF
National Id - Brazil - CNPJ
National Id - Japan Corporate Number
National Id - Japan My Number
National Id - France - INSEE National Id - France - Social Security Number (NIR)
National Id - Canada Social Insurance Number
National Id - US Social Security Number
Malware
Financial info Bank - American Bankers Association Routing Number
Bank - Bankruptcy Filings
Bank - International Bank Account Number
Bank - Statements
Committee on Uniform Securities Identification
Procedures number
Credit card number [Supports Strict Check]
Financial - Financial Accounting
Financial - Personal Finance
Financial - Invoice
Financial - Others
Magnetic Stripe Information
Healthcare Health - CLIA
Health - DEA
Health - Document Others
Intellectual Property Secret Info - AWS Access Key ID
Secret Info - AWS Secret Access Key
Secret Info - RSA Private Key
Company Confidential
Source Code - Cfamily
Source Code - java Source Code - javascript
Source Code - perl
Source Code - python
Source Code - ruby
Source Code - r
Source Code - verilog
Source Code - vhdl
Source Code - x86_assembly
Source Code - powershell
Source Code - vbs
Source Code - others
Non Sensitive Doesn’t contain any of the above patterns Disable Prisma Cloud Data Security and
Offboard AWS account
- Disable Prisma Cloud Data Security
- On Settings > Cloud Accounts, and edit the AWS account. Select the AWS account on which want to disable Prisma Cloud Data Security and clear the Data Security option. Prisma Cloud stops
ingestion for monitoring data stored in S3 buckets for the account. Data Security information from
the earlier scans are not deleted on Prisma Cloud, and have access to the alerts for the period
when Data Security was enabled (using Time Query).
If do not plan to re-enable Prisma Cloud Data Security for this AWS account, also delete
the SNS topic to avoid the additional cost of that SNS topic sending messages to Prisma Cloud Data Security. also stop sending S3 events to the CloudTrail bucket, if had set it up only for Prisma Cloud.
- easily enable Prisma Cloud Data Security again on Settings > Cloud Accounts. Select an AWS
account and check the Data Security checkbox. Prisma Cloud will start ingestion again and will
resume being charged for usage. All the data (before disabled) and new data (after re-enable
Data Security) will be available for you.
- Offboard and onboard an AWS account within 24 hours
- If had onboarded an AWS account and enabled Prisma Cloud Data Security enabled, all the scanned data results including alerts can no longer using the administrative console or API. Prisma Cloud Data Security stops further ingestion to monitor the data stored within the account.
- If add the same AWS account within 24 hours and enable Prisma Cloud Data Security, all the previously scanned data will be available from the administrative console and the API again. can
view all previously generated alerts as reported before offboarded the account, for example open
alerts will remain open, resolved will remain resolved. Prisma Cloud Data Security resumes ingestion
for this account.
- Offboard and onboard an AWS account after 24 hours
- Customer offboards a previously onboarded AWS account that has Prisma Cloud Data Security
enabled with scanned data results. All the scanned data results, including Alerts will stop being
available in the UI or via API. Prisma Cloud Data Security stops doing any more ingestion for Prisma Cloud Data Security for this account.
- After the account has been offboarded for more than 24 hours, Prisma Cloud Data Security will
delete all the previously scanned data related to Prisma Cloud Data Security in the customer
tenant.
- Because Prisma Cloud Data Security has stopped ingestion of new S3 data, Prisma Cloud Data Security will also stop incurring additional charges for Prisma Cloud Data Security.
- Customer onboards the same AWS account after 24 hours and enable Prisma Cloud Data Security. Prisma Cloud Data Security will start ingestion again for the account from scratch and
start charging for usage. Only the new data (after re-enable Prisma Cloud Data Security) will
be there. All previously scanned data will be gone, except for alerts that were generated.
The original alerts would not have been resolved and new alerts will be created on
the same objects, creating duplicate alerts on the same objects.
- Offboard an AWS account that was previously enabled for Prisma Cloud Data Security
After the AWS account has been offboarded for more than 24 hours, all the previously scanned data and alerts will be deleted. If do not plan to enable Prisma Cloud Data Security for this AWS account in the future, can
also delete the SNS topic to avoid the additional cost of that SNS topic sending messages to Prisma Cloud Data Security. also stop sending S3 events to the CloudTrail bucket, if had set it
up only for Prisma Cloud. Guidelines for Optimizing Data Security Cost
on Prisma Cloud
Cost Implications and Control
- Prisma Cloud leverages the Data Events published by CloudTrail to keep track of any changes. These
are exported as compressed log files to the customer's S3 bucket. To ensure these log files are not
contributing to unnecessary storage cost we would recommend enforcing a Bucket Lifecycle policy on this bucket for 1 month TTL (time to live). This will ensure that the files don’t contribute to the per
month pricing model.
- The Cloud Trail should be enabled for Write Data events instead of Read and Write. The Read events
volume in general is orders of magnitude more than Write events and AWS provides the flexibility to
customers to only enable either or both. For current Prisma Cloud features around Data security, read
events are not leveraged.
- Prisma Cloud will automatically skip active CloudTrail buckets and skip inactive CT logs, ELB access
logs, VPC flow logs, and S3 access logs on a best-effort basis based on documentation from AWS on
how to distinguish these logs from other objects. Prisma Cloud is skipping all these because they usually
don’t contain sensitive information and there is no point for Prisma Cloud to handle these and charge
customers for scanning them. HOWEVER, there is no guarantee that Prisma Cloud can catch all possible
logs because AWS may change their log format.
API Throttling and Egress Implications
- The solution is cloud service provider (CSP) API driven and undergoes the same throttling as any other
CSP API.
- The solution performs client side API throttling to ensure we don't overuse/abuse the API rate limits
enforced by the CSP.
- The client side rate limiting feature also ensures the full quota of API limit is not consumed by default to
ensure this does not come in the way of customer’s processes’ or applications’ API usage.
- The data that is downloaded from customer’s storage systems are NEVER persisted anywhere on Prisma Cloud infrastructure and is only held for the duration of processing of the content for Data Profile
analysis or limited by a maximum time out limit (24 hrs) whichever is hit earlier.
- There will be egress cost implications for the customer as the solution seeks to evaluate all content
in customer’s buckets. The customer can choose to optimize on cost by only selecting those buckets
requiring scans and filtering out any known good files that would not require any Data Profile analysis or
malware analysis. E.g. Database backup files etc
