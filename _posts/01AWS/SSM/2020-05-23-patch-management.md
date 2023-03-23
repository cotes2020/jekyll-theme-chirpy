---
title: AWS - SSM - Patch Manager
date: 2020-05-23 11:11:11 -0400
categories: [01AWS, SSM]
tags: [AWS, SSM]
toc: true
image:
---

- [AWS Systems Manager Patch Manager](#aws-systems-manager-patch-manager)
  - [Patch Manager prerequisites](#patch-manager-prerequisites)
    - [SSM Agent version](#ssm-agent-version)
    - [Connectivity to the patch source](#connectivity-to-the-patch-source)
    - [S3 endpoint access](#s3-endpoint-access)
    - [Supported OSs](#supported-oss)
  - [How security patches are selected](#how-security-patches-are-selected)
    - [[ Amazon Linux and Amazon Linux 2 ]](#-amazon-linux-and-amazon-linux-2-)
    - [[ CentOS ]](#-centos-)
    - [[ Debian Server ]](#-debian-server-)
    - [[ Oracle Linux ]](#-oracle-linux-)
    - [[ RHEL ]](#-rhel-)
    - [[ SLES ]](#-sles-)
    - [[ Ubuntu Server ]](#-ubuntu-server-)
    - [[ Windows ]](#-windows-)
  - [Linux: specify an alternative patch source repository](#linux-specify-an-alternative-patch-source-repository)
    - [considerations for alternative repositories](#considerations-for-alternative-repositories)
    - [Sample uses for alternative patch source repositories](#sample-uses-for-alternative-patch-source-repositories)
  - [How patches are installed](#how-patches-are-installed)
    - [[ Amazon Linux and Amazon Linux 2 ]](#-amazon-linux-and-amazon-linux-2--1)
  - [predefined and custom patch baselines](#predefined-and-custom-patch-baselines)
  - [How patch baseline rules work on Linux-based systems](#how-patch-baseline-rules-work-on-linux-based-systems)
    - [patch baseline rules work on Amazon Linux and Amazon Linux 2](#patch-baseline-rules-work-on-amazon-linux-and-amazon-linux-2)
    - [Key differences between Linux and Windows patching](#key-differences-between-linux-and-windows-patching)
- [patching operations](#patching-operations)
  - [patching configurations](#patching-configurations)
  - [SSM documents for patching instances](#ssm-documents-for-patching-instances)
    - [SSM documents recommended for patching instances](#ssm-documents-recommended-for-patching-instances)
      - [AWS-ConfigureWindowsUpdate](#aws-configurewindowsupdate)
      - [AWS-InstallWindowsUpdates](#aws-installwindowsupdates)
      - [AWS-RunPatchBaseline](#aws-runpatchbaseline)
      - [AWS-RunPatchBaselineAssociation](#aws-runpatchbaselineassociation)
      - [AWS-RunPatchBaselineWithHooks](#aws-runpatchbaselinewithhooks)
    - [Legacy SSM documents for patching instances](#legacy-ssm-documents-for-patching-instances)
      - [AWS-ApplyPatchBaseline](#aws-applypatchbaseline)
      - [AWS-InstallMissingWindowsUpdates](#aws-installmissingwindowsupdates)
      - [AWS-InstallSpecificWindowsUpdates](#aws-installspecificwindowsupdates)
- [Patch baselines](#patch-baselines)
  - [predefined and custom patch baselines](#predefined-and-custom-patch-baselines-1)
  - [predefined baselines](#predefined-baselines)
  - [custom baselines](#custom-baselines)
- [patch groups](#patch-groups)
  - [How it works](#how-it-works)

---


# AWS Systems Manager Patch Manager

Patch Manager

- a capability of AWS Systems Manager
- automates the process of patching managed instances with both security related and other types of updates.
- can use Patch Manager to
  - **apply patches for both OSs and applications**
    - On Windows Server, application support is limited to updates for Microsoft applications.
  - install **Service Packs** on Windows instances and perform **minor version upgrades** on Linux instances.
  - **patch fleets of Amazon EC2 instances, on-premises servers and VMs** by OS type.
    - This includes supported versions of Amazon Linux, Amazon Linux 2, CentOS, Debian Server, macOS, Oracle Linux, Red Hat Enterprise Linux (RHEL), SUSE Linux Enterprise Server (SLES), Ubuntu Server, and Windows Server.
  - **scan instances** to see only a report of missing patches, or automatically install all missing patches.
    - Patch Manager provides options to
      - scan the instances, report compliance, install available patches on a schedule
      - or patch or scan instances on demand
    - generate patch compliance reports that are sent to an S3 bucket
      - generate one-time reports, or generate reports on a regular schedule.
      - For a single instance, reports include details of all patches for the instance.
      - For a report on all instances, only a summary of how many patches are missing is provided.

**Important**

- AWS does not test patches for Windows Server or Linux <font color=blue> before making them available in Patch Manager </font>.
- Patch Manager doesn't support <font color=blue> upgrading major versions of OSs </font>
  - such as Windows Server 2016 to Windows Server 2019,
  - or SUSE Linux Enterprise Server (SLES) 12.0 to SLES 15.0.

Patch Manager uses <font color=red> patch baselines </font>
- which include rules for `auto-approving patches` within days of their release, as well as a `list of approved and rejected patches`.
- You can install patches
  - on a regular basis by scheduling patching to run as a Systems Manager maintenance window task.
  - or individually or to large groups of instances by using Amazon EC2 tags.
    - Tags are keys that help identify and sort the resources within the organization.
    - add tags to the patch baselines themselves when you create or update them.

> Patch Manager integrates with IAM, AWS CloudTrail, and Amazon EventBridge to provide a secure patching experience that includes event notifications and the ability to audit usage.


## Patch Manager prerequisites

### SSM Agent version
Version 2.0.834.0 or later of SSM Agent is running on the instances you want to manage with Patch Manager.
- An updated version of SSM Agent
- If an older version of the agent is running on an instance, some SSM Agent processes can fail.
- recommend automate install SSM Agent up-to-date.
- Subscribe to the [SSM Agent Release Notes](https://github.com/aws/amazon-ssm-agent/blob/mainline/RELEASENOTES.md) page on GitHub to get notifications about SSM Agent updates.

### Connectivity to the patch source
If the instances don't have a direct connection to the Internet and you are using an Amazon VPC with a VPC endpoint, you must ensure that the instances have access to the **source patch repositories (repos)**.

1. On Linux instances
   - patch updates are typically downloaded from the **remote repos** configured on the instance.
   - instance must be able to connect to the repos so the patching can be performed.
   - For more information, see **How security patches are selected**

2. Windows Server instances
   - must be able to connect to the **Windows Update Catalog or Windows Server Update Services (WSUS)**.
   - Confirm that the instances have connectivity to the [Microsoft Update Catalog](https://www.catalog.update.microsoft.com/home.aspx) through an `internet gateway, NAT gateway, or NAT instance`.
   - If you are using WSUS, confirm that the instance has connectivity to the WSUS server in the environment. For more information, see [Troubleshooting instance does not have access to Windows Update Catalog or WSUS].

### S3 endpoint access
Whether the instances operate in a private or public network, without access to the required AWS managed Amazon S3 buckets, patching operations fail.

- For information about the S3 buckets the managed instances must be able to access, see [About minimum S3 Bucket permissions for SSM Agent] and [Step 6: (Optional) Create a Virtual Private Cloud endpoint].


### Supported OSs
The Patch Manager capability does not support all the same OSs versions that are supported by other Systems Manager capabilities.

For example, Patch Manager does not support CentOS 6.3 or Raspbian Stretch.
- full list of Systems Manager-supported OSs, see [Systems Manager prerequisites](https://github.com/awsdocs/aws-systems-manager-user-guide/blob/main/doc_source/systems-manager-prereqs.md)
- Therefore, ensure that the instances you want to use with Patch Manager are running one of the OSs listed in the following table.


OS

1. Linux
   - Amazon Linux 2012.03 - 2018.03
   - Amazon Linux 2 2 - 2.0
   - CentOS 6.5 - 7.9, 8.0-8.2
   - Debian Server 8.x, 9.x, and 10.x
   - Oracle Linux 7.5 - 8.3
   - Red Hat Enterprise Linux (RHEL) 6.5 - 8.3
   - SUSE Linux Enterprise Server (SLES) 12.0 and later 12.x versions, 15.0 and 15.1
   - Ubuntu Server 14.04 LTS, 16.04 LTS, 18.04 LTS, 20.04 LTS, and 20.10 STR
   - Instances created from an Amazon Linux AMI that are using a proxy must be running a current version of the Python `requests` module in order to support Patch Manager operations.
   - [Upgrade the Python requests module on Amazon Linux instances that use a proxy server](https://github.com/awsdocs/aws-systems-manager-user-guide/blob/main/doc_source/sysman-proxy-with-ssm-agent-al-python-requests.md)

2. macOS
   - macOS 10.14.x (Mojave) and 10.15.x (Catalina)

3. Windows
   - Windows Server 2008 through Windows Server 2019, including R2 versions.
   - As of January 14, 2020, Windows Server 2008 is no longer supported for feature or security updates from Microsoft.
   - Legacy AMIs for Windows Server 2008 and 2008 R2 still include version 2 of SSM Agent preinstalled
     - but Systems Manager no longer officially supports 2008 versions
     - and no longer updates the agent for these versions of Windows Server.
     - SSM Agent version 3.0 may not be compatible with all operations on Windows Server 2008 and 2008 R2.
     - The final officially supported version of SSM Agent for Windows Server 2008 versions is 2.3.1644.0



## How security patches are selected

The primary focus of Patch Manager is on installing OSs **security-related updates** on instances.
- By default, Patch Manager doesn't install all available patches, but rather a smaller set of patches focused on security.
  - On all Linux-based systems supported by Patch Manager, you can choose a different source repository configured for the instance, typically to install nonsecurity updates.

### [ Amazon Linux and Amazon Linux 2 ]

On Amazon Linux and Amazon Linux 2, the Systems Manager patch baseline service uses preconfigured repositories on the instance.

There are usually two **preconfigured repositories (repos)** on an instance:

| **Repo ID**           | **Repo name**       |
| --------------------- | ------------------- |
| `amzn-main/latest`    | `amzn-main-Base`    |
| `amzn-updates/latest` | `amzn-updates-Base` |


**Note**
- All updates are downloaded from the remote repos configured on the instance.
  - so the instance must be able to connect to the repos so the patching can be performed.

- Amazon Linux and Amazon Linux 2 instances use Yum as the package manager, and Yum uses the concept of an **update notice as a file** named `updateinfo.xml`.
  - An **update notice**: a collection of packages that fix specific problems.
  - All packages that are in an update notice are considered Security by Patch Manager.
  - Individual packages are not assigned classifications or severity levels.
    - For this reason, Patch Manager assigns the attributes of an update notice to the related packages.

  - select the **Approved patches include non-security updates** check box in the **Create patch baseline** page
    - packages that are not classified in an `updateinfo.xml` file (package that contains a file without properly formatted Classification, Severity, and Date values) can be included in the prefiltered list of patches.
    - However, in order for a patch to be applied, the patch must still meet the user-specified patch baseline rules.


### [ CentOS ]

On CentOS, the Systems Manager patch baseline service uses preconfigured repositories (repos) on the instance. The following list provides examples for a fictitious CentOS 8.2 Amazon Machine Image (AMI):

| **Repo ID**                      | **Repo name**                                |
| -------------------------------- | -------------------------------------------- |
| `example-centos-8.2-base`        | `Example CentOS-8.2 - Base`                  |
| `example-centos-8.2-extras`      | `Example CentOS-8.2 - Extras`                |
| `example-centos-8.2-updates`     | `Example CentOS-8.2 - Updates`               |
| `example-centos-8.x-examplerepo` | `Example CentOS-8.x – Example Repo Packages` |

- All updates are downloaded from the remote repos configured on the instance.
  - so the instance must be able to connect to the repos so the patching can be performed.

- CentOS 6 and 7 instances use Yum as the package manager.
- CentOS 8 instances use DNF as the package manager.
- Both package managers use the concept of an **update notice**.
- However, CentOS default repos aren't configured with an update notice.
  - This means that Patch Manager does not detect packages on a default CentOS repo.
  - To enable Patch Manager to process packages that aren't contained in an update notice, must enable the `EnableNonSecurity` flag in the patch baseline rules.

- CentOS update notices are supported. Repos with update notices can be downloaded after launch.


### [ Debian Server ]

On Debian Server, the Systems Manager patch baseline service uses preconfigured repositories (repos) on the instance.
- These preconfigured repos are used to pull an **updated list of available package upgrades**.
  - For this, Systems Manager performs the equivalent of a `sudo apt-get update` command
- Packages are then filtered from `debian-security codename` repos. This means that
  - on Debian Server 8, Patch Manager only identifies upgrades that are part of `debian-security jessie`.
    - On Debian Server 8 only: Because some Debian `Server 8.*` instances refer to an obsolete package repository (`jessie-backports`), Patch Manager performs additional steps to ensure that patching operations succeed.
  - On Debian Server 9, only upgrades that are part of `debian-security stretch` are identified.
  - On Debian Server 10, only upgrades that are part of `debian-security buster` are identified.




### [ Oracle Linux ]

On Oracle Linux, the Systems Manager patch baseline service uses preconfigured repositories (repos) on the instance.

There are usually two preconfigured repos on an instance.

**Oracle Linux 7**:

| **Repo ID**         | **Repo name**                                                                      |
| ------------------- | ---------------------------------------------------------------------------------- |
| `ol7_UEKR5/x86_64`  | `Latest Unbreakable Enterprise Kernel Release 5 for Oracle Linux 7Server (x86_64)` |
| `ol7_latest/x86_64` | `Oracle Linux 7Server Latest (x86_64)`                                             |

**Oracle Linux 8**:

| **Repo ID**         | **Repo name**                                                                |
| ------------------- | ---------------------------------------------------------------------------- |
| `ol8_baseos_latest` | `Oracle Linux 8 BaseOS Latest (x86_64)`                                      |
| `ol8_appstream`     | `Oracle Linux 8 Application Stream (x86_64)`                                 |
| `ol8_UEKR6`         | `Latest Unbreakable Enterprise Kernel Release 6 for Oracle Linux 8 (x86_64)` |

- All updates are downloaded from the remote repos configured on the instance. so the instance must be able to connect to the repos so the patching can be performed.
- Oracle Linux instances use Yum as the package manager, and Yum uses update notice `updateinfo.xml`.
  - Individual packages are not assigned classifications or severity levels.
  - For this reason, Patch Manager assigns the attributes of an update notice to the related packages and installs packages based on the Classification filters specified in the patch baseline.
  - select the **Approved patches include non-security updates** check box in the **Create patch baseline** page
    - then packages that are not classified in an `updateinfo.xml` file (or a package that contains a file without properly formatted Classification, Severity, and Date values) can be included in the prefiltered list of patches.
    - However, in order for a patch to be applied, the patch must still meet the **user-specified patch baseline rules**.


### [ RHEL ]

On Red Hat Enterprise Linux, the Systems Manager patch baseline service uses preconfigured repositories (repos) on the instance. There are usually three preconfigured repos on an instance.

- All updates are downloaded from the remote repos configured on the instance. so the instance must be able to connect to the repos so the patching can be performed.

- Red Hat Enterprise Linux 7 instances use Yum as the package manager.
- Red Hat Enterprise Linux 8 instances use DNF as the package manager.
- Both package managers use the concept of an update notice as a file named `updateinfo.xml`.
  - Individual packages are not assigned classifications or severity levels. For this reason, Patch Manager assigns the attributes of an update notice to the related packages and installs packages based on the Classification filters specified in the patch baseline.
- select the **Approved patches include non-security updates** check box in the **Create patch baseline** page
    - then packages that are not classified in an `updateinfo.xml` file (or a package that contains a file without properly formatted Classification, Severity, and Date values) can be included in the prefiltered list of patches.
    - However, in order for a patch to be applied, the patch must still meet the **user-specified patch baseline rules**.


Note that repo locations differ between RHEL 7 and RHEL 8:

RHEL 7

The following repo IDs are associated with RHUI 2.
- RHUI 3 launched in December 2019 and introduced a different naming scheme for Yum repository IDs.
- Depending on the RHEL-7 AMI you create the instances from, you might need to update the commands.

| **Repo ID**                                        | **Repo name**                                                     |
| -------------------------------------------------- | ----------------------------------------------------------------- |
| `rhui-REGION-client-config-server-7/x86_64`        | `Red Hat Update Infrastructure 2.0 Client Configuration Server 7` |
| `rhui-REGION-rhel-server-releases/7Server/x86_64`  | `Red Hat Enterprise Linux Server 7 (RPMs)`                        |
| `rhui-REGION-rhel-server-rh-common/7Server/x86_64` | `Red Hat Enterprise Linux Server 7 RH Common (RPMs)`              |

RHEL 8

| **Repo ID**                   | **Repo name**                                                        |
| ----------------------------- | -------------------------------------------------------------------- |
| `rhel-8-appstream-rhui-rpms`  | `Red Hat Enterprise Linux 8 for x86_64 - AppStream from RHUI (RPMs)` |
| `rhel-8-baseos-rhui-rpms`     | `Red Hat Enterprise Linux 8 for x86_64 - BaseOS from RHUI (RPMs)`    |
| `rhui-client-config-server-8` | `Red Hat Update Infrastructure 3 Client Configuration Server 8`      |


### [ SLES ]

On SUSE Linux Enterprise Server (SLES) instances, the ZYPP library gets the list of available patches (a collection of packages) from the following locations:
- List of repositories: `etc/zypp/repos.d/*`
- Package information: `/var/cache/zypp/raw/*`

SLES instances use Zypper as the package manager, and Zypper uses the concept of a patch. A patch is simply a collection of packages that fix a specific problem. Patch Manager handles all packages referenced in a patch as security-related. Because individual packages aren't given classifications or severity, Patch Manager assigns the packages the attributes of the patch that they belong to.


### [ Ubuntu Server ]

On Ubuntu Server, the Systems Manager patch baseline service uses preconfigured repositories (repos) on the instance. These preconfigured repos are used to pull an updated list of available package upgrades. For this, Systems Manager performs the equivalent of a `sudo apt-get update` command.

Packages are then filtered from `codename-security` repos, where the codename is unique to the release version, such as `trusty` for Ubuntu Server 14. Patch Manager only identifies upgrades that are part of these repos:
- Ubuntu Server 14.04 LTS: `trusty-security`
- Ubuntu Server 16.04 LTS: `xenial-security`
- Ubuntu Server 18.04 LTS: `bionic-security`
- Ubuntu Server 20.04 LTS: `focal-security`
- Ubuntu Server 20.10 STR: `groovy-gorilla`


### [ Windows ]

On Microsoft Windows OSs, Patch Manager retrieves a list of available updates that Microsoft publishes to Microsoft Update and are automatically available to Windows Server Update Services (WSUS).

Patch Manager continuously monitors for new updates in every AWS Region. The list of available updates is refreshed in each Region at least once per day. When the patch information from Microsoft is processed, Patch Manager removes updates that were replaced by later updates from its patch list . Therefore, only the most recent update is displayed and made available for installation. For example, if `KB4012214` replaces `KB3135456`, only `KB4012214` is made available as an update in Patch Manager.

**Note**
Patch Manager only makes available patches for Windows Server OS versions that are supported for Patch Manager. For example, Patch Manager can't be used to patch Windows RT.


---


## Linux: specify an alternative patch source repository

**use the default repositories configured** on an instance for patching operations
- Patch Manager scans for or installs security-related patches. (the default behavior for Patch Manager)

On Linux systems
- however, you can also use Patch Manager to
  - install patches that are not related to security, or in a different source repository.
  - specify alternative patch source repositories when create a <font color=red> custom patch baseline </font>.
  - In each custom patch baseline, specify **patch source configurations for up to 20 versions of a supported Linux OS**.
- Running a **custom patch baseline** that `specifies alternative patch repositories` on an instance doesn't change the **default repository** configured for the instance.

> For example,
> Ubuntu Server fleet includes both `Ubuntu Server 14.04` and `Ubuntu Server 16.04` instances.
> specify alternate repositories for each version in the **same custom patch baseline**.
> For each version, you provide a name, specify the OS version type (product), and provide a repository configuration.
> You can also specify a single alternative source repository that applies to all versions of a supported OS.

To specify alternative patch source repositories
1. **Example: Using the console**
   - use the **Patch sources** section on the **Create patch baseline** page.
   - For information about using the **Patch sources** options, see [Creating a custom patch baseline](https://docs.aws.amazon.com/systems-manager/latest/userguide/create-baseline-console-linux.html).

2. **Example: Using the AWS CLI**
   - using the `--sources` option
   - see [Create a patch baseline with custom repositories for different OS versions](https://docs.aws.amazon.com/systems-manager/latest/userguide/patch-manager-cli-commands.html#patch-manager-cli-commands-create-patch-baseline-mult-sources).
   - `aws ssm create-patch-baseline --cli-input-json file://my-patch-repository.json`


### considerations for alternative repositories

1. **Only specified repositories are used for patching**
   - Specifying alternative repositories doesn't mean specifying *additional* repositories.
   - you must also specify the default repositories as part of the **alternative patch source configuration** if you want their updates to be applied.
   - Running a **custom patch baseline** that specifies **alternative patch repositories** for an instance doesn't make those repositories the new default repositories.
     - After the patching operation completes
     - the repositories previously defined as the defaults still remain the default repository configured for the instance.

> For example
> on Amazon Linux 2 instances, the default repositories are `amzn-main` and `amzn-update`.
> If you want to include the **Extra Packages for Enterprise Linux (EPEL) repository** in the patching operations, you must specify all three repositories as alternative repositories.


2. **Patching behavior for YUM-based distributions depends on the updateinfo.xml manifest**
   - When you specify **alternative patch repositories** for `YUM-based distributions`,
     - such as Amazon Linux or Amazon Linux 2, Red Hat Enterprise Linux, or CentOS,
   - patching behavior depends on whether the repository includes an update manifest in the form of a complete and correctly formatted `updateinfo.xml` file.
   - This file specifies the r`elease date, classifications, and severities` of the various packages.
   - Any of the following will affect the patching behavior:
     - If you filter on **Classification** and **Severity**, but they aren't specified in `updateinfo.xml`
       - the package will not be included by the filter.
       - This also means that packages without an `updateinfo.xml` file won't be included in patching.
     - If you filter on **ApprovalAfterDays**, but the package release date isn't in Unix Epoch format (or has no release date specified)
       - the package will not be included by the filter.
     - Exception:
       - select the **Approved patches include non-security updates** check box in the **Create patch baseline** page.
       - In this case, packages without an `updateinfo.xml` file or contains this file but without properly formatted **Classification**, **Severity**, and **Date** values *will* be included in the prefiltered list of patches.
       - They must still meet the other patch baseline rule requirements in order to be installed.

### Sample uses for alternative patch source repositories

**Example 1 – Nonsecurity Updates for Ubuntu Server**
- using Patch Manager to install security patches on a fleet of Ubuntu Server instances using the `AWS-provided predefined patch baseline` **AWS-UbuntuDefaultPatchBaseline**.
- create a `new patch baseline` that is based on this default, but specify in the approval rules that you want `nonsecurity related updates` that are part of the default distribution to be installed as well.
- When this patch baseline is run against the instances, patches for both security and nonsecurity issues are applied.
- You can also choose to approve nonsecurity patches in the patch exceptions you specify for a baseline.

**Example 2 - Personal Package Archives (PPA) for Ubuntu Server**
- Ubuntu Server instances are running software that is distributed through a [Personal Package Archives (PPA) for Ubuntu](https://launchpad.net/ubuntu/+ppas).
- create a patch baseline that specifies a PPA repository that you have configured on the instance as the source repository for the patching operation.
- Then use Run Command to run the `patch baseline document` on the instances.

**Example 3 – Internal Corporate Applications on Amazon Linux**
- to run some applications needed for industry regulatory compliance on the Amazon Linux instances.
- configure a repository for these applications on the instances, use YUM to initially install the applications, and then update or create a new patch baseline to include this new corporate repository.
- After this you can use Run Command to run the **AWS-RunPatchBaseline** document with the `Scan` option to see if the corporate package is listed among the installed packages and is up to date on the instance.
- If it isn't up to date, you can run the document again using the `Install` option to update the applications.



---

## How patches are installed

Patch Manager uses the appropriate built-in mechanism for an OS type to install updates on an instance.
- on Windows Server, the `Windows Update API` is used
- on Amazon Linux the `yum` package manager is used.

### [ Amazon Linux and Amazon Linux 2 ]

On Amazon Linux and Amazon Linux 2 instances, the patch installation workflow is as follows:

1. If a list of patches is specified using an `https/S3 URL` using the `InstallOverrideList` parameter for the `AWS-RunPatchBaseline` or `AWS-RunPatchBaselineAssociation` documents, the listed patches are installed and steps 2-7 are skipped.

2. Apply [GlobalFilters](https://docs.aws.amazon.com/systems-manager/latest/APIReference/API_CreatePatchBaseline.html#systemsmanager-CreatePatchBaseline-request-GlobalFilters) as specified in the **patch baseline**
   - keeping <font color=red> only the qualified packages for further processing </font>.

3. Apply [ApprovalRules](https://docs.aws.amazon.com/systems-manager/latest/APIReference/API_CreatePatchBaseline.html#EC2-CreatePatchBaseline-request-ApprovalRules) as specified in the **patch baseline**.
   - <font color=red> Each approval rule can define a package as approved </font>.
   - Approval rules are also subject to whether the **Include nonsecurity updates** check box was selected when create/update the patch baseline.
   - If `nonsecurity updates` are excluded
     - an implicit rule is applied in order to select only packages with upgrades in security repos.
     - For each package, the candidate version of the package (which is typically the latest version) must be part of a security repo.
   - If `nonsecurity updates` are included
     - patches from other repositories are considered as well.

4. Apply [ApprovedPatches](https://docs.aws.amazon.com/systems-manager/latest/APIReference/API_CreatePatchBaseline.html#EC2-CreatePatchBaseline-request-ApprovedPatches) as specified in the **patch baseline**.
   - The approved patches are approved for update
   - even if they are discarded by `GlobalFilters` or if no approval rule specified in `ApprovalRules` grants it approval.

5. Apply [RejectedPatches](https://docs.aws.amazon.com/systems-manager/latest/APIReference/API_CreatePatchBaseline.html#EC2-CreatePatchBaseline-request-RejectedPatches) as specified in the **patch baseline**
   - The rejected patches are removed from the list of approved patches and will not be applied.

6. If multiple versions of a patch are approved, the latest version is applied.

7. The YUM update API is applied to approved patches as follows:
   - For **predefined default patch baselines** provided by AWS
   - and for **custom patch baselines** where the `Approved patches include non-security updates` check box is *not* selected
   - only patches specified in `updateinfo.xml` are applied (security updates only).

```bash
# The equivalent yum command for this workflow is:
sudo yum update-minimal --sec-severity=critical,important --bugfix -y
```

   - For **custom patch baselines** where the **Approved patches include non-security updates** check box *is* selected
   - both patches in `updateinfo.xml` and those not in `updateinfo.xml` are applied (security and nonsecurity updates).


```bash
# The equivalent yum command for this workflow is:
sudo yum update --security --bugfix
```

8. The instance is rebooted if any updates were installed.
   - (Exception: If the `RebootOption` parameter is set to `NoReboot` in the `AWS-RunPatchBaseline` document, the instance is not rebooted after Patch Manager runs.)

---

## predefined and custom patch baselines


---

## How patch baseline rules work on Linux-based systems

**The rules in a patch baseline** for Linux distributions operate differently based on the distribution type.
- Unlike patch updates on Windows Server instances, rules are evaluated on each instance to take the configured repos on the instance into consideration.
- Patch Manager uses the native package manager to drive the installation of patches approved by the patch baseline.

### patch baseline rules work on Amazon Linux and Amazon Linux 2

On Amazon Linux and Amazon Linux 2, the patch selection process is as follows:

1. On the instance, the YUM library accesses the `updateinfo.xml` file for each configured repo.
   - If no `updateinfo.xml` file is found, whether patches are installed depend on settings for **Approved patches include non-security updates** and **Auto-approval**.
   - For example, if non-security updates are permitted, they are installed when the auto-approval time arrives.

2. Each **update notice** in `updateinfo.xml` includes `several attributes`
   - Update notice attributes: denote the properties of the packages in the notice, as described in the following table.
   - list of supported values: `describe-patch-properties`
   - <font color=blue> type </font>
     - Corresponds to the value of the `Classification key attribute` in the patch baseline's PatchFilter data type.
     - Denotes the type of package included in the update notice.
   - <font color=blue> severity </font>
     - Corresponds to the value of the `Severity key attribute` patch baseline's PatchFilter data type.
     - Denotes the severity of the packages included in the update notice. Usually only applicable for Security update notices.
   - <font color=blue> update_id 	</font>
     - Denotes the advisory ID, such as ALAS-2017-867.
     - The advisory ID can be used in the ApprovedPatches or RejectedPatches attribute in the patch baseline.
   - <font color=blue> references </font>
     - Contains additional information about the update notice, such as a CVE ID (format: CVE-2017-1234567).
     - The CVE ID can be used in the ApprovedPatches or RejectedPatches attribute in the patch baseline.
   - <font color=blue> updated </font>
     - Corresponds to `ApproveAfterDays` in the patch baseline.
     - Denotes the released date (updated date) of the packages included in the update notice.
     - A comparison between the current timestamp and the value of this attribute plus the ApproveAfterDays is used to determine if the patch is approved for deployment.

3. The product of the instance is determined by SSM Agent.
   - This attribute corresponds to the value of the Product key attribute in the patch baseline's [PatchFilter](https://docs.aws.amazon.com/systems-manager/latest/APIReference/API_PatchFilter.html) data type.

4. Packages are selected for the update according to the following guidelines.
   - **Pre-defined default patch baselines** provided by AWS and **custom patch baselines** where the `Approved patches include non-security updates is not selected`
     - For each update notice in `updateinfo.xml`
     - the patch baseline is used as a filter, allowing only the qualified packages to be included in the update.
     - If multiple packages are applicable after applying the patch baseline definition, the latest version is used.
     - The equivalent yum command for this workflow is:
     - `sudo yum update-minimal --sec-severity=critical,important --bugfix -y`
   - **Custom patch baselines** where the `Approved patches include non-security updates check box is selected`
     - In addition to applying the security updates that were selected from `updateinfo.xml`,
     - Patch Manager applies nonsecurity updates that otherwise meet the patch filtering rules.
     - The equivalent yum command for this workflow is:
     - `sudo yum update --security --bugfix -y`



### Key differences between Linux and Windows patching

**Difference 1: Patch evaluation**
- **Linux**
  - Systems Manager evaluates `patch baseline rules` and the `list of approved and rejected patches` on *each* managed instance.
  - Systems Manager must evaluate patching on each instance because the service `retrieves the list of known patches and updates from the repositories` that are configured on the instance.
- **Windows**
  - For Windows patching, Systems Manager evaluates `patch baseline rules` and the `list of approved and rejected patches` *directly in the service*.
  - It can do this because Windows patches are pulled from a single repository (Windows Update).

**Difference 2: `Not Applicable` patches**
- Due to the large number of available packages for Linux OSs, Systems Manager does not report details about patches in the *Not Applicable* state.
- A `Not Applicable` patch is,
- for example,
- a patch for Apache software when the instance does not have Apache installed.
- Systems Manager does report the number of `Not Applicable` patches in the summary, but if you call the [DescribeInstancePatches](https://docs.aws.amazon.com/systems-manager/latest/APIReference/API_DescribeInstancePatches.html) API for an instance, the returned data does not include patches with a state of `Not Applicable`. This behavior is different from Windows.


**Difference 3: SSM document support**
- The `AWS-ApplyPatchBaseline` Systems Manager document (SSM document) doesn't support Linux instances.
- For applying patch baselines to **Linux, macOS, and Windows Server managed instances**, the recommended SSM document is `AWS-RunPatchBaseline`.

**Difference 4: Application patches**
- The primary focus of Patch Manager is applying patches to OSs.
- However, you can also use Patch Manager to apply patches to some applications on the instances.
- **Linux**
  - Patch Manager uses the configured repositories for updates, and <font color=blue> does not differentiate between OSs and application patches </font>.
  - use Patch Manager to define which repositories to fetch updates from.
- **Windows**
  - apply approval rules, as well as *Approved* and *Rejected* patch exceptions, for applications released by Microsoft, s
  - uch as Microsoft Word 2016 and Microsoft Exchange Server 2016.

---

# patching operations

## patching configurations

A **patching configuration** defines a `unique patching operation`.
- The configuration specifies
  - the **instances** for patching,
  - which **patch baseline** is to be applied,
  - the **schedule** for patching,
  - and typically, the **maintenance window** that the configuration is to be associated with.

To create a patching configuration
- use the Configure patching page
- or run a one-time manual patching operation on a set of instances.


## SSM documents for patching instances

8 Systems Manager documents (SSM documents) currently available to keep the managed instances patched with the latest security-related updates.

recommend using just five of these documents in the patching operations.
- these five SSM documents provide you with a full range of patching options using AWS Systems Manager.
- Four of these documents were released later than the four legacy SSM documents they replace and represent expansions or consolidations of functionality.

The five recommended SSM documents include:
- **AWS-ConfigureWindowsUpdate**
- **AWS-InstallWindowsUpdates**
- **AWS-RunPatchBaseline**
- **AWS-RunPatchBaselineAssociation**
- **AWS-RunPatchBaselineWithHooks**

The four legacy SSM documents that are still available for use in some AWS Regions, but might be deprecated in the future, include:
- **AWS-ApplyPatchBaseline**
- **AWS-FindWindowsUpdates**
- **AWS-InstallMissingWindowsUpdates**
- **AWS-InstallSpecificWindowsUpdates**


### SSM documents recommended for patching instances

#### AWS-ConfigureWindowsUpdate

This SSM document prompts **Windows Update to download and install the specified updates and reboot instances as needed**.
- Supports configuring basic Windows Update functions and using them to install updates automatically (or to disable automatic updates).
  - Use this document with State Manager to ensure Windows Update maintains its configuration.
  - run it manually using Run Command to change the Windows Update configuration.
- Available in all AWS Regions.
- The available parameters in this document support
  - specifying a category of updates to install (or whether to disable automatic updates),
  - specifying the day of the week and time of day to run patching operations.

This SSM document is most useful if you <font color=red> don't need strict control over Windows updates and don't need to collect compliance information </font>


#### AWS-InstallWindowsUpdates

This SSM document provides **basic patching functionality**
- in cases where you either want to install a specific update (using the `Include Kbs` parameter),
- or install patches with specific classifications or categories but don't need patch compliance information.

**Replaces legacy SSM documents:**
- **AWS-FindWindowsUpdates**
- **AWS-InstallMissingWindowsUpdates**
- **AWS-InstallSpecificWindowsUpdates**

> The three legacy documents perform different functions
> - can be achieve by using different parameter settings with the SSM document **AWS-InstallWindowsUpdates**.

---

#### AWS-RunPatchBaseline

> Replaces legacy documents:
> AWS-ApplyPatchBaseline
> The legacy document applies only to Windows Server instances
> and does not provide support for application patching.

1. Available in all AWS Regions.
2. This SSM document **control patch approvals using the patch baseline currently specified as the "default" for an OS type**.
   - Installs patches on the instances or scans instances to determine whether any qualified patches are missing.
     - This document supports Linux, macOS, and Windows Server instances.
     - The document will perform the appropriate actions for each platform.
   - can apply patches for both OSs and applications.
     - On Windows Server, application support is limited to updates for Microsoft applications.
     - For Linux OSs, compliance information is provided for patches from both the default source repository configured on an instance and from any alternative source repositories you specify in a custom patch baseline.

3. This SSM document performs patching operations on instances for <font color=red> both security related and other types of updates </font>.
   - When the document is run, it uses the **patch baseline currently specified as the "default"** for an OS type if no **patch group** is specified.
   - Otherwise, it uses the **patch baseline that is associated with the patch group**

4. Reports patch compliance information that you can view using the Systems Manager Compliance tools.
   - These tools provide you with insights on the patch compliance state of the instances,
     - such as which instances are missing patches and what those patches are.
5. When you use `AWS-RunPatchBaseline`, patch compliance information is recorded using the `PutInventory` API command.


**[ Windows ]**

On Windows Server instances, the **AWS-RunPatchBaseline** document
- downloads and invokes a PowerShell module
- downloads a **snapshot of the patch baseline** that applies to the instance.
- This **patch baseline snapshot** contains a `list of approved patches` that is compiled by querying the patch baseline against a Windows Server Update Services (WSUS) server.
- this list is passed to the Windows Update API, which controls downloading and installing the approved patches as appropriate.


**[ Linux ]**

On Linux instances, the **AWS-RunPatchBaseline** document
- invokes a Python module
- downloads a **snapshot of the patch baseline** that applies to the instance.
- This **patch baseline snapshot** uses the `defined rules and lists of approved and blocked patches` to drive the appropriate package manager for each instance type:
  - Amazon Linux, Amazon Linux 2, CentOS, Oracle Linux, and RHEL 7 instances use YUM. For YUM operations, Patch Manager requires `Python 2.6` or later.
  - RHEL 8 instances use DNF. For DNF operations, Patch Manager requires `Python 2` or `Python 3`. (Neither version is installed by default on RHEL 8. You must install one or the other manually.)
  - Debian Server and Ubuntu Server instances use APT. For APT operations, Patch Manager requires `Python 3`.
  - SUSE Linux Enterprise Server instances use Zypper. For Zypper operations, Patch Manager requires `Python 2.6` or later.


**[ macOS ]**

On macOS instances, the `AWS-RunPatchBaseline` document
- invokes a Python module
- downloads a **snapshot of the patch baseline** that applies to the instance.
- Next, a Python subprocess invokes the AWS Command Line Interface (AWS CLI) on the instance to `retrieve the installation and update information` for the specified package managers and to drive the appropriate package manager for each update package.


**snapshot**
- Each snapshot is specific to an AWS account, patch group, OS, and snapshot ID.
- The snapshot is delivered through a presigned S3 URL, which expires 24 hours after the snapshot is created.
- After the URL expires, to apply the same snapshot content to other instances, generate a new presigned Amazon S3 URL up to three days after the snapshot was created.

After all approved and applicable updates have been installed, with reboots performed as necessary, patch compliance information is generated on an instance and reported back to Patch Manager.
- `RebootOption`: `NoReboot` in the `AWS-RunPatchBaseline` document, the instance is not rebooted after Patch Manager runs.


**AWS-RunPatchBaseline parameters**
- supports five parameters.
- The `Operation` parameter is required.
- The `InstallOverrideList`, `BaselineOverride`, and `RebootOption` parameters are optional.
- `Snapshot-ID` is technically optional, but we recommend that you supply a custom value for it when you run `AWS-RunPatchBaseline` outside of a maintenance window.
- Patch Manager can supply the custom value automatically when the document is run as part of a maintenance window operation.
-

Parameter name: `Operation`
- **Usage**: Required.
- **Options**:
  - `Scan`
    - `AWS-RunPatchBaseline` determines the patch compliance state of the instance and reports this information back to Patch Manager.
    - `Scan` does not prompt updates to be installed or instances to be rebooted.
    - Instead, the operation identifies where updates are missing that are approved and applicable to the instance.
  - `Install`
    - `AWS-RunPatchBaseline` attempts to install the approved and applicable updates that are missing from the instance.
    - Patch compliance information generated as part of an `Install` operation does not list any missing updates, but might report updates that are in a failed state if the installation of the update did not succeed for any reason.
    - Whenever an update is installed on an instance, the instance is rebooted to ensure the update is both installed and active.
    - Exception: If the `RebootOption` parameter is set to `NoReboot` in the `AWS-RunPatchBaseline` document, the instance is not rebooted after Patch Manager runs.

If a patch specified by the baseline rules is installed *before* Patch Manager updates the instance, the system might not reboot as expected. This can happen when a patch is installed manually by a user or installed automatically by another program, such as the `unattended-upgrades` package on Ubuntu Server.



Parameter name: `Snapshot ID`
- **Usage**: Optional.
- `Snapshot ID` is a unique ID (GUID) used by Patch Manager
- to ensure that <font color=red> a set of instances that are patched in a single operation all have the exact same set of approved patches </font>
- Although the parameter is defined as optional, our best practice recommendation depends on whether or not you are running **AWS-RunPatchBaseline** in a maintenance window, as described in the following table.

**AWS-RunPatchBaseline best practices**

Running AWS-RunPatchBaseline inside a maintenance window
- Do not supply a Snapshot ID. Patch Manager will supply it for you.
- Systems Manager provides a GUID value based on the `maintenance window execution ID`.
- This ensures that a correct ID is used for all the invocations of `AWS-RunPatchBaseline` in that maintenance window.
- If you do specify a value in this scenario, note that the snapshot of the patch baseline might not remain in place for more than three days.
- After that, a new snapshot will be generated even if you specify the same ID after the snapshot expires.


Running AWS-RunPatchBaseline outside of a maintenance window
- Generate and specify a custom GUID value for the Snapshot ID.
- When you are not using a maintenance window to run `AWS-RunPatchBaseline`, we recommend that you generate and specify a unique Snapshot ID for each patch baseline, particularly if you are running the `AWS-RunPatchBaseline` document on multiple instances in the same operation. If you do not specify an ID in this scenario, Systems Manager generates a different Snapshot ID for each instance the command is sent to. This might result in varying sets of patches being specified among the instances. For instance, say that you are running the `AWS-RunPatchBaseline` document directly via Run Command, a capability of AWS Systems Manager, and targeting a group of 50 instances. Specifying a custom Snapshot ID results in the generation of a single baseline snapshot that is used to evaluate and patch all the instances, ensuring that they end up in a consistent state.

> You can use any tool capable of generating a GUID to generate a value for the Snapshot ID parameter.
> For example, in PowerShell, you can use the `New-Guid` cmdlet to generate a GUID in the format of `12345699-9405-4f69-bc5e-9315aEXAMPLE`.  |



Parameter name: `InstallOverrideList`
- **Usage**: Optional.
- `InstallOverrideList` lets you specify an `https/S3 path-style URL` to a list of patches to be installed.
- This patch installation list, in YAML format, <font color=blue> overrides the patches specified by the current default patch baseline. </font>
- provides more granular control over which patches are installed on the instances.
- Be aware that **compliance reports** reflect `patch states` according to what’s specified in the patch baseline, not what you specify in an `InstallOverrideList` list of patches.
  - so Scan operations ignore the `InstallOverrideList` parameter.
  - This is to ensure that compliance reports consistently reflect patch states according to policy rather than what was approved for a specific patching operation.

**Valid URL formats**
- **https URL format**:

  ```
  https://s3.amazonaws.com/DOC-EXAMPLE-BUCKET/my-windows-override-list.yaml
  ```
- **Amazon S3 path-style URL**:

  ```
  s3://DOC-EXAMPLE-BUCKET/my-windows-override-list.yaml
  ```

- Valid YAML content formats
  - The formats to specify patches in the list depends on the OS of the instance.
  - The general format, however, is as follows:

	```
	patches:
		-
			id: '{patch-d}'
			title: '{patch-title}'
			{additional-fields}:{values}
	```


[ Linux ]

**id**
The **id** field is required. Use it to specify patches using the package name and architecture. For example: `'dhclient.x86_64'`. You can use wildcards in id to indicate multiple packages. For example: `'dhcp*'` and `'dhcp*1.*'`.

**Title**
The **title** field is optional, but on Linux systems it does provide additional filtering capabilities. If you use **title**, it should contain the package version information in the one of the following formats:

YUM/SUSE Linux Enterprise Server (SLES):

```
{name}.{architecture}:{epoch}:{version}-{release}
```

APT

```
{name}.{architecture}:{version}
```

For Linux patch titles, you can use one or more wildcards in any position to expand the number of package matches. For example: `'*32:9.8.2-0.*.rc1.57.amzn1'`.

For example:
- apt package version 1.2.25 is currently installed on the instance, but version 1.2.27 is now available.
- You add apt.amd64 version 1.2.27 to the patch list. It depends on apt utils.amd64 version 1.2.27, but apt-utils.amd64 version 1.2.25 is specified in the list.

In this case, apt version 1.2.27 will be blocked from installation and reported as “Failed-NonCompliant.”



Parameter name: `RebootOption`
- **Usage**: Optional.
- **Options**:
  - `RebootIfNeeded`
    - the instance is rebooted if
      - Patch Manager installed new patches,
      - or if it detected any patches with a status of `INSTALLED_PENDING_REBOOT` during the `Install` operation.
    - The `INSTALLED_PENDING_REBOOT` status can mean that the option `NoReboot` was selected the last time the `Install` operation was run.
    - (Patches installed outside of Patch Manager are never given a status of `INSTALLED_PENDING_REBOOT`.)
    - When you choose the `RebootIfNeeded` option, Patch Manager does not evaluate whether a reboot is *required* by the patch.
    - A reboot occurs whenever there are missing packages or packages with a status of `INSTALLED_PENDING_REBOOT`.

  - `NoReboot`
    - Patch Manager does not reboot an instance even if it installed patches during the `Install` operation.
    - This option is useful if you know that the instances don't require rebooting after patches are applied, or you have applications or processes running on an instance that should not be disrupted by a patching operation reboot.
    - It is also useful when you want more control over the timing of instance reboots, such as by using a maintenance window.
    - If you choose the `NoReboot` option and a patch is installed, the patch is assigned a status of `InstalledPendingReboot`. The instance itself, however, is marked as `Non-Compliant`. After a reboot occurs and a `Scan` operation is run, the instance status is updated to `Compliant`.

**Patch installation tracking file**:
- To track patch installation, especially patches that were installed since the last system reboot,
- Systems Manager maintains a file on the managed instance.
- Do not delete or modify the tracking file. If this file is deleted or corrupted, the patch compliance report for the instance is inaccurate.
- If this happens, reboot the instance and <font color=red> run a patch Scan operation to restore the file </font>

This tracking file is stored in the following locations on the managed instances:
- Linux OSs:
  - `/var/log/amazon/ssm/patch-configuration/patch-states-configuration.json`
  - `/var/log/amazon/ssm/patch-configuration/patch-inventory-from-last-operation.json`
- Windows Server OS:
  - `C:\ProgramData\Amazon\PatchBaselineOperations\State\PatchStatesConfiguration.json`
  - `C:\ProgramData\Amazon\PatchBaselineOperations\State\PatchInventoryFromLastOperation.json`

Parameter name: `BaselineOverride`
- **Usage**: Optional.
- define patching preferences at runtime using the `BaselineOverride` parameter.
- This baseline override is maintained as a JSON object in an S3 bucket.
- It ensures patching operations use the provided baselines that match the host OS instead of applying the rules from the default patch baseline

---

#### AWS-RunPatchBaselineAssociation

This SSM document **Installs patches on the instances or scans instances** to determine whether any qualified patches are missing.
- Available in all commercial AWS Regions.
- differs from **AWS-RunPatchBaseline** as it supports the use of tags
  - to identify which patch baseline to use with a set of targets when it runs.
- In addition, patch compliance data is compiled in terms of a specific State Manager association.
- The patch compliance data collected by **AWS-RunPatchBaselineAssociation** is recorded using the `PutComplianceItems` API command instead of the `PutInventory` command.
- This prevents compliance data that isn't associated with this particular association from being overwritten.
- For Linux OSs, compliance information is provided for patches from both the default source repository configured on an instance and from any alternative source repositories you specify in a custom patch baseline.

**Replaces legacy documents:**
- **None**


#### AWS-RunPatchBaselineWithHooks

This SSM document, with optional hooks, can be used to run at three points during the patching cycle.
- Available in all commercial AWS Regions.
- differs from **AWS-RunPatchBaseline** in its Install operation.
- supports **lifecycle hooks** that run at designated points during instance patching.
  - Because patch installations sometimes require instances to reboot, the patching operation is divided into two events, for a total of three hooks that support custom functionality.
  - The first hook is before the `Install with NoReboot` operation.
  - The second hook is after the `Install with NoReboot` operation.
  - The third hook is available after the reboot of the instance.

**Replaces legacy documents:**
- **None**


### Legacy SSM documents for patching instances

The following four SSM documents are still available for use in the patching operations in some AWS Regions. However, they might be deprecated in the future, so we do not recommend their use.


#### AWS-ApplyPatchBaseline

#### AWS-InstallMissingWindowsUpdates

#### AWS-InstallSpecificWindowsUpdates


---

# Patch baselines


## predefined and custom patch baselines

<font color=red> patch baseline </font>
- defines which patches are approved for installation on the instances.
- You can
  - specify approved or rejected patches one by one.
  - create auto-approval rules to specify that certain types of updates (for example, critical updates) should be automatically approved.
  - The rejected list overrides both the rules and the approve list.

To use a list of approved patches to install specific packages
- first remove all auto-approval rules.
- If you explicitly identify a patch as rejected, it will not be approved or installed, even if it matches all of the criteria in an auto-approval rule.
- Also, a patch is installed on an instance only if it applies to the software on the instance, even if the patch has otherwise been approved for the instance.


Patch Manager provides
1. <font color=red> predefined patch baselines </font>
   - for each of the OSs supported by Patch Manager.
   - use these baselines as they are currently configured (you can't customize them)
2. <font color=red> create the own custom patch baselines </font>
   - for greater control over which patches are approved or rejected for the environment.
   - Also, the predefined baselines assign a compliance level of `Unspecified` to all patches installed using those baselines.
   - For compliance values to be assigned, you can create a copy of a predefined baseline and specify the compliance values you want to assign to patches.


## predefined baselines

The following table describes the predefined patch baselines provided with Patch Manager.

1. `AWS-AmazonLinuxDefaultPatchBaseline` Amazon Linux
   - Approves all OS patches that are
     - classified as "Security" or "Bugfix"
     - and that have a severity level of "Critical" or "Important".
   - Patches are **auto-approved seven days** after release.

2. `AWS-AmazonLinux2DefaultPatchBaseline` Amazon Linux 2
   - Approves all OS patches that are
     - classified as "Security" or "Bugfix"
     - and that have a severity level of "Critical" or "Important".
   - Patches are **auto-approved seven days** after release.

3. `AWS-CentOSDefaultPatchBaseline` CentOS
   - including <font color=red> nonsecurity updates </font>
   - Approves all updates **seven days after they become available**


4. `AWS-DebianDefaultPatchBaseline` Debian Server
   - **Immediately approves** all OS security-related patches that have a priority of "Required", "Important", "Standard," "Optional," or "Extra."
   - There is no wait before approval because reliable release dates are not available in the repos.


5. `AWS-MacOSDefaultPatchBaseline` macOS
   - Approves all OS patches that are classified as "Security". Also approves all packages with a current update.


6. `AWS-OracleLinuxDefaultPatchBaseline` Oracle Linux
   - Approves all OS patches that are
     - classified as "Security" or "Bugfix"
     - have a severity level of "Important" or "Moderate".
   - Patches are **auto-approved seven days** after release.

7. `AWS-RedHatDefaultPatchBaseline`   Red Hat Enterprise Linux (RHEL)
   - Approves all OS patches that are
     - classified as "Security" or "Bugfix"
     - and that have a severity level of "Critical" or "Important".
   - Patches are **auto-approved seven days** after release.


8. `AWS-SuseDefaultPatchBaseline` SUSE Linux Enterprise Server (SLES)
   - Approves all OS patches that are classified as "Security" and with a severity of "Critical" or "Important".
   - Patches are **auto-approved seven days** after release.


9. `AWS-UbuntuDefaultPatchBaseline`  Ubuntu Server
    - Immediately approves all OS security-related patches that have a priority of "Required", "Important", "Standard," "Optional," or "Extra."
    - There is no wait before approval because reliable release dates are not available in the repos.


10. `AWS-DefaultPatchBaseline`  Windows Server
    - Approves all Windows Server OS patches that are classified as "CriticalUpdates" or "SecurityUpdates" and that have an MSRC severity of "Critical" or "Important".
    - Patches are **auto-approved seven days** after release.


11. `AWS-WindowsPredefinedPatchBaseline-OS`  Windows Server
    - Approves all Windows Server OS patches that are classified as "CriticalUpdates" or "SecurityUpdates" and that have an MSRC severity of "Critical" or "Important".
    - Patches are **auto-approved seven days** after release.

12. `AWS-WindowsPredefinedPatchBaseline-OS-Applications` Windows Server
    - For the Windows Server OS,
      - approves all patches that are classified as "CriticalUpdates" or "SecurityUpdates"
      - and that have an MSRC severity of "Critical" or "Important".
    - For Microsoft applications,
      - approves all patches.
    - Patches for both OS and applications are **auto-approved seven days** after release.


## custom baselines

create the own patch baseline
- you can <font color=red> choose which patches to auto-approve by using the following categories </font>.
  - <font color=red> OS </font>:
    - Windows, Amazon Linux, Ubuntu Server, and so on.
  - <font color=red> Product name (for OSs) </font>:
    - For example, RHEL 6.5, Amazon Linux 2014.09, Windows Server 2012, Windows Server 2012 R2, and so on.
  - <font color=red> Product name (for Microsoft applications on Windows Server only) </font>:
    - For example, Word 2016, BizTalk Server, and so on.
  - <font color=red> Classification </font>:
    - For example, critical updates, security updates, and so on.
  - <font color=red> Severity </font>:
    - For example, critical, important, and so on.

- For each approval rule, you can choose to specify an **auto-approval delay** or specify a **patch approval cutoff date**.
  - Because it's not possible to reliably determine the release dates of update packages for Ubuntu Server, the auto-approval options are not supported for this OS.
  - An **auto-approval delay**:
    - the number of days to wait after the patch was released, before the patch is automatically approved for patching.
    - For example
      - create a rule using the `CriticalUpdates` classification
      - and configure it for seven days auto-approval delay,
      - then a new critical patch released on July 7 is automatically approved on July 14.
    - If a Linux repository doesn’t provide release date information for packages,
      - Systems Manager uses the build time of the package as the auto-approval delay for Amazon Linux, Amazon Linux 2, RHEL, and CentOS.
      - If the system isn't able to find the build time of the package, Systems Manager treats the auto-approval delay as having a value of zero.

  - **auto-approval cutoff date**,
    - Patch Manager automatically applies all patches released on or before that date.
    - For example
      - specify July 7, 2020, as the cutoff date,
      - no patches released on or after July 8, 2020, are installed automatically.

- specify a compliance severity level.
  - If an approved patch is reported as missing, `Compliance Level` is the severity of the compliance violation.

> multiple **patch baselines** - different **auto-approval delays or cutoff dates**
> deploy patches at different rates to different instances.
> For example
> create separate patch baselines, auto-approval delays, andcutoff dates for development and production environments.
> This enables you to test patches in the development environment before they get deployed in the production environment.

to create a patch baseline:
- Patch Manager provides one predefined patch baseline for each supported OS.
  - These predefined patch baselines are used as the default patch baselines for each OS type
  - unless you create the own patch baseline and designate it as the default for the corresponding OS type.

- For Windows Server, three predefined patch baselines are provided.
  - The configuration settings in these two patch baselines are the same.
    - `AWS-DefaultPatchBaseline`: the default patch baseline for Windows Server instances, unless specify a different patch baseline.
    - `AWS-WindowsPredefinedPatchBaseline-OS`, was created to distinguish it from the third predefined patch baseline for Windows Server.
    - <font color=red> support only OS updates on the Windows OS itself </font>.
  - `AWS-WindowsPredefinedPatchBaseline-OS-Applications`, can be used to apply patches to both the Windows Server OS and supported Microsoft applications.

- For on-premises servers and virtual machines (VMs)
  - Patch Manager attempts to use the custom default patch baseline.
  - If no custom default patch baseline exists, the system uses the **predefined patch baseline** for the corresponding OS.


- If a patch is listed as both approved and rejected in the same patch baseline, the patch is rejected.

- <font color=red> An instance can have only one patch baseline defined for it </font>

- The formats of **package names add to lists of approved/rejected patches for a patch baseline** depend on the type of OS you are patching.

---


# patch groups

use a **patch group** to associate instances with a specific patch baseline in Patch Manager.
- ensure deploying the appropriate patches, based on the associated patch baseline rules, to the correct set of instances.
- avoid deploying patches before they have been adequately tested.
- For example
  - create patch groups for different environments (such as Development, Test, and Production)
  - and register each patch group to an appropriate patch baseline.
- A patch group can be registered with only one patch baseline for each operating system type.
- An instance can only be in one patch group.


run `AWS-RunPatchBaseline`
- target managed instances using **instance ID or tags**.
- SSM Agent and Patch Manager then evaluate which patch baseline to use based on the patch group value that you added to the instance.

create a patch group by using EC2 tags.
- unlike other tagging scenarios across Systems Manager, a patch group *must* be defined with the tag key: **Patch Group**.
- Note that the key is case-sensitive.
- You can specify any value, for example "web servers," but the key must be **Patch Group**.


After you create a patch group and tag instances
- register the patch group with a patch baseline.
- Registering the patch group with a patch baseline ensures that the instances within the patch group use the rules defined in the associated patch baseline.


## How it works

when a maintenance window is configured to send a command to patch using Patch Manager.
- the system `runs the task to apply a patch baseline to an instance`
- **SSM Agent** verifies that a **patch group** value is defined for the instance.
- If the instance is assigned to a patch group
  - Patch Manager then verifies which patch baseline is registered to that group.
  - if a patch baseline is found for that group, Patch Manager notifies SSM Agent to use the associated patch baseline.
- if an instance isn't configured for a patch group
  - Patch Manager automatically notifies SSM Agent to use the currently configured default patch baseline.


![\[Diagram showing how patch baselines are determined when performing patching operations.\]](http://docs.aws.amazon.com/systems-manager/latest/userguide/images/patch-groups-how-it-works.png)

three groups of EC2 instances:

| EC2 instances group | Tags                                               |
| ------------------- | -------------------------------------------------- |
| Group 1             | `key=OS,value=Windows` `key=Patch Group,value=DEV` |
| Group 2             | `key=OS,value=Windows`                             |
| Group 3             | `key=OS,value=Windows` `key=Patch Group,value=QA`  |


two Windows Server patch baselines:

| Patch baseline ID      | Default | Associated patch group |
| ---------------------- | ------- | ---------------------- |
| `pb-0123456789abcdef0` | Yes     | `Default`              |
| `pb-9876543210abcdef0` | No      | `DEV`                  |


**patching operations process flow**

The general process to scan or install patches using Run Command:

1. **Send a command to patch**:
   - Use the Systems Manager console, SDK, AWS Command Line Interface (AWS CLI), or AWS Tools for Windows PowerShell
   - to send a Run Command task using the document `AWS-RunPatchBaseline`.
   - The diagram shows a **Run Command task**: to patch managed instances by targeting the tag `key=OS,value=Windows`.

2. **Patch baseline determination**:
   - SSM Agent verifies the patch group tags applied to the EC2 instance and queries Patch Manager for the corresponding patch baseline.

   - **Matching patch group value associated with patch baseline:**

     1. SSM Agent receives the command issued in Step 1 to begin a patching operation.
     2. SSM Agent validates that the EC2 instances have the **patch group tag-value** `DEV` applied
     3. queries Patch Manager for an associated patch baseline.
     4. Patch Manager verifies that patch baseline `pb-9876543210abcdef0` has the patch group `DEV` associated and notifies SSM Agent.
     5. SSM Agent `retrieves a patch baseline snapshot` from Patch Manager based on the **approval rules and exceptions** configured in `pb-9876543210abcdef0` and proceeds to the next step.

   - **No patch group tag added to instance:**

     1. SSM Agent receives the command issued in Step 1 to begin a patching operation.
     2. SSM Agent validates that the EC2 instances don't have a `Patch Group` tag applied
     3. SSM Agent queries Patch Manager for the **default Windows patch baseline**.
     4. Patch Manager verifies that the default Windows Server patch baseline is `pb-0123456789abcdef0` and notifies SSM Agent.
     5. SSM Agent `retrieves a patch baseline snapshot` from Patch Manager based on the approval rules and exceptions configured in the default patch baseline `pb-0123456789abcdef0` and proceeds to the next step.

   - **No matching patch group value associated with a patch baseline:**

     1. SSM Agent receives the command issued in Step 1 to begin a patching operation.
     2. SSM Agent validates that the EC2 instances have the patch group tag-value `QA` applied
     3. queries Patch Manager for an associated patch baseline.
     4. Patch Manager does not find a patch baseline that has the patch group `QA` associated.
     5. Patch Manager notifies SSM Agent to use the **default Windows patch baseline** `pb-0123456789abcdef0`.
     6. SSM Agent `retrieves a patch baseline snapshot` from Patch Manager based on the approval rules and exceptions configured in the default patch baseline `pb-0123456789abcdef0` and proceeds to the next step.

3. **Patch scan or install**:
   - After determining the appropriate patch baseline to use
   - SSM Agent begins either `scanning for or installing patches` based on the operation value specified in Step 1.
   - The patches that are scanned for or installed are determined by the approval rules and patch exceptions defined in the **patch baseline snapshot** provided by Patch Manager.












.
