



[toc]

- ref
  - [AWS科普系列：使用EC2 Systems Manager](https://www.jianshu.com/p/7ba51af0a268)


---

# AWS Systems Manager


- a management service
- view and control of the infrastructure on AWS.
  - a unified user interface to view operational data from multiple AWS services
  - and <font color=red> automate operational tasks </font> across the AWS resources.
  - maintain security and compliance
    - by scanning the managed instances
    - and reporting on (or taking corrective action on) any policy violations it detects.

- Systems Manager helps configure and maintain the managed instances.
  - managed instance, machine that has been configured for use with Systems Manager.
  - Supported machine types: Amazon EC2 instances, on-premises servers, and VMs, including VMs in other cloud environments.
  - Supported operating system types: Windows Server, macOS, Raspbian, Linux distributions.

- associate AWS resources together
  - applying the same identifying resource tag to each of the associate AWS resources.
  - then view operational data for these resources as a <fonr color=blue> resource group </font>
  - help monitor and troubleshoot.
  - example
    - assign a resource tag of "Operation=North Region OS Patching" to following resources:
    - A group of `EC2 instances`
    - A group of `on-premises servers` in the own facility
    - A `Systems Manager patch baseline`, specifies which patches to apply to the managed instances
    - An `S3 bucket` to store patching operation log output
    - A `Systems Manager maintenance window`, specifies the schedule for the patching operation
  - After tagging the resources, view a consolidated dashboard in Systems Manager
    - reports the status of all the resources that are part of the patching operation in the North region.
    - If a problem arises with any of these resources, take corrective action immediately.


- Systems Manager supported AWS Regions
  - AWS Systems Manager is available in the AWS Regions listed in Systems Manager service endpoints in the Amazon Web Services General Reference.
    - Before start Systems Manager configuration process, ensure the service is available in each of the AWS Regions you want to use it in.
  - For on-premises servers and VMs in the hybrid environment, choose the Region closest to the data center or computing environment.


> 公司一般会采取内网+域账号+动态口令+跳板机的方式来保证生产环境的安全。这套机制确实好用，但是也有一些问题，比如需要单独的服务器，耗费资源；大家共用跳板机，使得跳板机可能成为瓶颈。在云计算时代，跳板机不能很好地满足公有云环境的安全了。



![1376176-e508e38c19e24e2d](https://i.imgur.com/e7UPUxb.png)

![1376176-3ae22c933900df6c](https://i.imgur.com/HumTX2q.png)


![INSTALLREMOTEAGENT_2](https://i.imgur.com/pmLG3TQ.png)



---

## Capabilities in Systems Manager

- Systems Manager is comprised of individual capabilities, which are grouped into five categories:
  - Operations Management
  - Application Management
  - Change Management
  - Node Management
  - Shared Resources

- This collection of capabilities is a powerful set of tools and features to perform operational tasks.
  - For example:
    - Group AWS resources together by any purpose or activity you choose
      - such as application, environment, region, project, campaign, business unit, or software lifecycle.
    - Centrally define the configuration options and policies for the managed instances.
    - Centrally view, investigate, and resolve operational work items related to AWS resources.
    - Automate or schedule a variety of maintenance and deployment tasks.
    - Use and create runbook-style SSM documents that define the actions to perform on the managed instances.
    - Run a command, with rate and error controls, that targets an entire fleet of managed instances.
    - Securely connect to a managed instance with a single click, without having to open an inbound port or manage SSH keys.
    - Separate the secrets and configuration data from the code by using parameters, with or without encryption, and then reference those parameters from a number of other AWS services.
    - Perform automated inventory by collecting metadata about the Amazon EC2 and on-premises managed instances. Metadata can include information about applications, network configurations, and more.
    - View consolidated inventory data from multiple AWS Regions and accounts that you manage.
    - Quickly see which resources in the account are out of compliance and take corrective action from a centralized dashboard.
    - View active summaries of metrics and alarms for the AWS resources.
    - Systems Manager simplifies resource and application management, shortens the time to detect and resolve operational problems, and helps you operate and manage the AWS infrastructure securely at scale.

---

## Operations Management
- a suite of capabilities that help you manage the AWS resources.

- Explorer
  - a customizable operations dashboard that reports information about the AWS resources.
  - displays an aggregated view of operations data (OpsData) for the AWS accounts and across Regions.
  - OpsData includes metadata about the EC2 instances, patch compliance details, and operational work items (OpsItems).
  - Explorer provides context about how OpsItems are distributed across the business units or applications, how they trend over time, and how they vary by category.
  - group and filter information in Explorer to focus on items that are relevant to you and that require action.
  - When you identify high priority issues, use Systems Manager OpsCenter to run Automation runbooks and quickly resolve those issues.

- OpsCenter
  - provides a central location where operations engineers and IT professionals can view, investigate, and resolve operational work items (OpsItems) related to AWS resources.
  - reduce mean time to resolution for issues impacting AWS resources.
  - aggregates and standardizes OpsItems across services while providing contextual investigation data about each OpsItem, related OpsItems, and related resources.
  - OpsCenter also provides Systems Manager Automation documents (runbooks) that you can use to quickly resolve issues.
  - You can specify searchable, custom data for each OpsItem.
  - You can also view automatically-generated summary reports about OpsItems by status and source.

- CloudWatch Dashboards
  - customizable home pages in the CloudWatch console
  - monitor the resources in a single view, even those resources that are spread across different regions.
  - create customized views of the metrics and alarms for the AWS resources.

- Trusted Advisor & Personal Health Dashboard (PHD)
  - Systems Manager hosts two online tools to help you provision the resources and monitor the account for health events.
  - Trusted Advisor
    - real time guidance to help you provision the resources following AWS best practices.
  - The AWS Personal Health Dashboard
    - provides information about AWS Health events that can affect the account.
    - The information is presented in two ways:
      - a dashboard that shows recent and upcoming events organized by category,
      - and a full event log that shows all events from the past 90 days.

---

## Application Management
- a suite of capabilities that help you manage the applications running in AWS

- Application Manager
  - helps you investigate and remediate issues with the AWS resources in the context of the applications.
  - aggregates operations information from multiple AWS services and Systems Manager capabilities to a single AWS Management Console.

- Resource Groups
  - An AWS resource
    - an entity you can work with in AWS,
    - such as Systems Manager SSM documents, patch baselines, maintenance windows, parameters, and managed instances; an EC2 instance; an Amazon EBS volume; a security group; or an VPC.
  - A resource group
    - a collection of AWS resources that are all in the same AWS Region, and that match criteria provided in a query.
      - build queries in the Resource Groups console,
      - or pass them as arguments to Resource Groups commands in the AWS CLI.
    - With Resource Groups, create a custom console that organizes and consolidates information based on criteria that you specify in tags.
    - You can also use groups as the basis for viewing monitoring and configuration insights in AWS Systems Manager.

- AppConfig
  - create, manage, and quickly deploy application configurations.
  - AppConfig supports controlled deployments to applications of any size.
    - use AppConfig with applications hosted on EC2 instances, AWS Lambda, containers, mobile applications, or IoT devices.
  - AppConfig includes validators.
    - To prevent errors when deploying application configurations,
    - provides a syntactic or semantic check to ensure that the configuration you want to deploy works as intended.
    - During a configuration deployment, AppConfig monitors the application to ensure that the deployment is successful.
    - If the system encounters an error or if the deployment triggers an alarm, AppConfig rolls back the change to minimize impact for the application users.


- Parameter Store
  - provides secure, hierarchical storage for configuration data and secrets management.
  - store data such as passwords, database strings, EC2 instance IDs and Amazon Machine Image (AMI) IDs, and license codes as parameter values. You can store values as plain text or encrypted data. You can then reference values by using the unique name you specified when you created the parameter.


---


## Change Management

- taking action against or changing the AWS resources.

- Change Manager
  - an enterprise change management framework
  - requesting, approving, implementing, and reporting on operational changes to the application configuration and infrastructure.
  - managing changes to both AWS resources and on-premises resources.
  - single delegated administrator account using AWS Organizations, manage changes across multiple AWS accounts in multiple AWS Regions.
  - local account, manage changes for a single AWS account.

- Systems Manager Automation
  - automate common maintenance and deployment tasks.
    - create and update Amazon Machine Images,
    - apply driver and agent updates,
    - reset passwords on Windows Server instance,
    - reset SSH keys on Linux instances,
    - and apply OS patches or application updates.

- Change Calendar
  - set up date and time ranges when actions you specify may or may not be performed in the AWS account.
    - for example, in Systems Manager Automation documents
  - In Change Calendar, these ranges are called events.
  - create a Change Calendar entry
    - a Systems Manager document of the type `ChangeCalendar`.
    - the document stores [iCalendar 2.0](https://icalendar.org/) data in plaintext format.
  - Events that you add to the Change Calendar entry become part of the document.

- Maintenance Windows
  - set up recurring schedules for managed instances to run administrative tasks like installing patches and updates without interrupting business-critical operations.



---


## Node Management

- managing the EC2 instances, the on-premises servers and VMs in the hybrid environment, and other types of AWS resources (nodes).

- Systems Manager Configuration Compliance
  - scan the fleet of managed instances for patch compliance and configuration inconsistencies.
    - collect and aggregate data from multiple AWS accounts and Regions,
    - and then drill down into specific resources that aren’t compliant.
  - By default, displays compliance data about Patch Manager patching and State Manager associations.
  - You can also customize the service and create the own compliance types based on the IT or business requirements.

- Fleet Manager
  - a capability of AWS Systems Manage
  - a unified user interface (UI) experience that remotely manage the server fleet running on AWS, or on-premises.
  - view the health and performance status of the entire server fleet from one console.
  - gather data from individual instances to perform common troubleshooting and management tasks from the console.
    - This includes viewing folder and file contents, Windows registry management, operating system user management, and more.

### Managed Instances
  - 可以看到透過 AWS SSM 管理的 EC2 Instance（需要在 Instance 上安裝 SSM Agent）。
    - Managed Instances
    - any EC2 instance or on-premises machine (server/VM) in the hybrid environment that is configured for Systems Manager.
  - set up managed instances
    - install SSM Agent on the machines (if not installed by default)
    - and configure IAM permissions.
    - On-premises machines also require an activation code.

- Inventory
  - automates the process of collecting software inventory from managed instances.
  - gather metadata about applications, files, components, patches, and more on the managed instances.


---

### Session Manager:
- AWS Systems Manager provides you safe, secure remote management of the instances at scale
  - without logging into the servers, replacing the need for bastion hosts, SSH, or remote PowerShell.
  - without the need to open inbound ports, maintain bastion hosts, or manage SSH keys.
  - automating common administrative tasks across groups of instances
    - such as registry edits, user management, and software and patch installations.
  - manage the EC2 instances through an interactive one-click browser-based shell or through the AWS CLI.
- integration with AWS IAM
  - can apply granular permissions to control the actions users can perform on instances.
- provides secure and auditable instance management
  - All actions taken with Systems Manager are recorded by AWS CloudTrail, allowing you to audit changes throughout the environment.

---

#### Systems Manager Run Command
  - 以遠端的方式，透過 Command 安全地管理 Instance 的設定。
  - remotely and securely manage the configuration of the managed instances at scale.
  - perform on-demand changes like updating applications or running Linux shell scripts and Windows PowerShell commands on a target set of dozens or hundreds of instances.

---


- Systems Manager State Manager
  - automate the process of keeping the managed instances in a defined state.
    - ensure that the instances are bootstrapped with specific software at startup,
    - joined to a Windows domain (Windows Server instances only),
    - or patched with specific software updates.
  - 透過自動化程序讓 Instance 維持在定義的狀態，可定期去追蹤 Instance 是否有依照設置狀態下執行。


- Patch Manager
  - 自動修補 Instance 系統及應用程式的更新。
  - automate the process of patching the managed instances with both security related and other types of updates.
  - apply patches for both operating systems and applications. (On Windows Server, application support is limited to updates for Microsoft applications.) This capability enables you to scan instances for missing patches and apply missing patches individually or to large groups of instances by using EC2 instance tags. Patch Manager uses patch baselines_, which can include rules for auto-approving patches within days of their release, as well as a list of approved and rejected patches. You can install security patches on a regular basis by scheduling patching to run as a Systems Manager maintenance window task, or you can patch the managed instances on demand at any time. For Linux operating systems, you can define the repositories that should be used for patching operations as part of the patch baseline. This allows you to ensure that updates are installed only from trusted repositories regardless of what repositories are configured on the instance. For Linux, you also have the ability to update any package on the instance, not just those that are classified as operating system security updates. For Windows Server, you can also use Patch Manager to update supported Microsoft applications.

- Distributor
  - Use Distributor to create and deploy packages to managed instances. Distributor lets you package the own software—or find AWS-provided agent software packages, such as **AmazonCloudWatchAgent**—to install on AWS Systems Manager managed instances. After you install a package for the first time, you can use Distributor to completely uninstall and reinstall a new package version, or perform an in-place update that adds new or changed files only. Distributor publishes resources, such as software packages, to AWS Systems Manager managed instances.

- Hybrid Activations
  - To set up servers and VMs in the hybrid environment as managed instances, you need to create a managed instance activation . After you complete the activation, you receive an activation code and ID. This code/ID combination functions like an Amazon EC2 access ID and secret key to provide secure access to the Systems Manager service from the managed instances.



---


## Shared Resources

Systems Manager uses the following shared resources for managing and configuring the AWS resources. Choose the tabs to learn more.

Documents

A Systems Manager document (SSM document) defines the actions that Systems Manager performs. SSM document types include Command documents, which are used by State Manager and Run Command, and Automation documents, which are used by Systems Manager Automation. Systems Manager includes dozens of pre-configured documents that you can use by specifying parameters at runtime. Documents can be expressed in JSON or YAML, and include steps and parameters that you specify.


---



---



- straightforward to use.
  - Access AWS Systems Manager from the Amazon EC2 console, select the instances that you want to manage, and define the management tasks you want to perform.
  - no cost to manage both EC2 and on-premises resources.
- Enables automate configuration and ongoing management of systems at scale through a set of capablity.
  - automatically collect software inventory, apply OS patches, create system images, and configure Windows and Linux operating systems.
  - These capabilities help you define and track system configurations, prevent drift, and maintain software compliance of EC2 and on-premises configurations.
- providing a management approach that is designed for the scale and agility of the cloud but extends to the on-premises data center
  - AWS Systems Manager allows you to bridge the existing infrastructure with AWS.

- centralize operational data from multiple AWS services and automate tasks across the AWS resources.
- can create logical groups of resources
  - such as applications, different layers of an application stack, or production versus development environments.
  - With Systems Manager, can select a resource group and view its recent API activity, resource configuration changes, related notifications, operational alerts, software inventory, and patch compliance status.
  - can also take action on each resource group depending on operational needs.
- provides a central place to view and manage the AWS resources, can have complete visibility and control over operations.
  - Centralized console and toolset for a wide variety of system management tasks.
  - Designed for managing a large fleet of systems – tens or hundreds.
- SSM Agent enables System Manager features
  - supports all OSs supported by OS as well as back to Windows Server 2003 and Raspbian .
  - SSM Agent installed by default on recent AWS-provided base AMIs for Linux and Windows.
  - Manages AWS-based and on-premises based systems via the agent.
- The AWS Systems Manager console integrates with AWS Resource Groups, and it offers grouping capabilities in addition to other native integrations.



AWS Systems Manager feature

Systems Manager Inventory:
- AWS Systems Manager collects information about instances and the software installed on them,
  - helping you to understand the system configurations and installed applications.
  - can collect data about applications, files, network configurations, Windows services, registries, server roles, updates, and any other system properties.
- The gathered data enables you to manage application assets, track licenses, monitor file integrity, discover applications not installed by a traditional installer, and more.


Configuration Compliance.
- AWS Systems Manager lets you scan the managed instances for patch compliance and configuration inconsistencies.
- can collect and aggregate data from multiple AWS accounts and Regions, and then drill down into specific resources that aren’t compliant.
- By default, AWS Systems Manager displays data about patching and associations.
- You can also customize the service and create the own compliance types based on the requirements. .

Automation:
- AWS Systems Manager allows you to safely automate common and repetitive IT operations and management tasks across AWS resources.
- With Systems Manager, you can create JSON documents that specify a specific list of tasks or use community published documents.
- These doc can be executed directly through the AWS Management Console, CLIs, and SDKs, scheduled in a maintenance window, or triggered based on changes to AWS resources through Amazon CloudWatch Events.
- You can track the execution of each step in the documents as well as require approvals for each step.
- You can also incrementally roll out changes and automatically halt when errors occur.


Run Command:
- Use Systems Manager Run Command to remotely and securely manage the configuration of the managed instances at scale.
- Use Run Command to perform on-demand changes like updating applications or running Linux shell scripts and Windows PowerShell commands on a target set of dozens or hundreds of instances.

---

Patch Manager:
- AWS Systems Manager helps you select and deploy operating system and software patches automatically across large groups of Amazon EC2 or on-premises instances.
- Through patch baselines, can set rules to auto-approve select categories of patches to be installed, such as operating system or high severity patches, and you can specify a list of patches that override these rules and are automatically approved or rejected.
- can also schedule maintenance windows for the patches so that they are only applied during preset times.
- Systems Manager helps ensure that the software is up-to-date and meets the compliance policies.

Maintenance Windows:
- AWS Systems Manager lets you schedule windows of time to run administrative and maintenance tasks across the instances.
- This ensures that you can select a convenient and safe time to install patches and updates or make other configuration changes, improving the availability and reliability of the services and applications.


Distributor:
- Distributor enables to securely store and distribute software packages in the organization.
- can use Distributor with existing Systems Manager features like Run Command and State Manager to control the lifecycle of the packages running on the instances.


State Manager:
- AWS Systems Manager provides configuration management, helps maintain consistent configuration of Amazon EC2 or on-premises instances.
- can control configuration details
  - such as server configurations, anti-virus definitions, firewall settings, and more.
- can define configuration policies for the servers through the AWS Management Console or use existing scripts, PowerShell modules, or Ansible playbooks directly from GitHub or Amazon S3 buckets.
- Systems Manager automatically applies the configurations across the instances at a time and frequency that you define.
- You can query Systems Manager at any time to view the status of the instance configurations, giving you on-demand visibility into the compliance status.

Parameter Store:
- provides a centralized store to manage the configuration data, whether plain-text data such as database strings or secrets such as passwords.
- This allows you to separate the secrets and configuration data from the code. Parameters can be tagged and organized into hierarchies, helping you manage parameters more easily.
- For example, you can use the same parameter name, “db-string”, with a different hierarchical path, “dev/db-string” or “prod/db-string”, to store different values.
- Systems Manager is integrated with AWS Key Management Service (KMS), allowing you to automatically encrypt the data you store.
- can also control user and resource access to parameters using IAM. Parameters can be referenced through other AWS services, such as Amazon Elastic Container Service, AWS Lambda, and AWS CloudFormation.


















.
