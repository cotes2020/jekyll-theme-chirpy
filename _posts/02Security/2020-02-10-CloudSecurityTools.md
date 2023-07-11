---
title: Cloud Security Tools
date: 2020-02-10 11:11:11 -0400
categories: [02Security]
tags: [02Security]
math: true
image:
---

- [Cloud Security Tools](#cloud-security-tools)
  - [Cloud Infrastructure Entitlement Management (CIEM)](#cloud-infrastructure-entitlement-management-ciem)
  - [Cloud Security Posture Mgmt (CSPM)](#cloud-security-posture-mgmt-cspm)
  - [Cloud Detection and Response (CDR)](#cloud-detection-and-response-cdr)
  - [Cloud Workload Protection Platform (CWPP)](#cloud-workload-protection-platform-cwpp)

---

# Cloud Security Tools

云安全解决方案可以帮助减轻企业安全团队的一些云安全负担；但是，有许多解决方案可用，很难确定哪个最能满足组织的需求。公司应考虑的两个主要云安全解决方案是云基础设施授权管理 (CIEM) 和云安全状态管理 (CSPM)。

- CIEM 解决方案旨在管理对云资源的访问，使组织能够实施最小权限原则和零信任安全模型。
- CSPM 提供对云安全配置的关键可见性，使组织能够识别和解决将基于云的资源置于风险之中的错误配置。

![Screenshot 2023-06-27 at 11.49.24](/assets/img/Screenshot%202023-06-27%20at%2011.49.24.png)

![Screenshot 2023-06-27 at 11.49.29](/assets/img/Screenshot%202023-06-27%20at%2011.49.29.png)

![Screenshot 2023-06-27 at 11.49.35](/assets/img/Screenshot%202023-06-27%20at%2011.49.35.png)

![Screenshot 2023-06-27 at 11.49.40](/assets/img/Screenshot%202023-06-27%20at%2011.49.40.png)




---

## Cloud Infrastructure Entitlement Management (CIEM)

随着公司采用云基础架构，许多公司正在部署多云环境，跨多个提供商的平台分发数据和应用程序。这些平台中的每一个都有自己的安全控制和方法来`管理对公司基于云的资源的访问`。

- 零信任安全模型和最小特权原则规定用户、应用程序和系统应该只拥有完成工作所需的访问和权限。
- 跨多个云平台实施授权可能很复杂且不可扩展。
- 云基础设施授权管理(CIEM) 可以让组织跨多云部署自动化授权管理过程，使组织能够在其整个环境中保持一致的访问控制。

CIEM 的一些主要功能包括：
- 发现：
  - 应识别所有人类和非人类身份、帐户活动和资源。
  - 应该评估所有类型的权利策略，并为本地和联合身份提供支持。
- 跨云关联：
  - 在多云环境中，通过本机支持所有主要公共云平台来简化授权管理。
- 可见性：
  - 如果没有图形视图等可视化支持，人们很难理解复杂的权利关系。此图形视图应创建身份和资源之间的映射，并支持基于自然语言的权利信息查询。
  - 组织还应该能够在仪表板上跟踪行为、权利消耗和类似指标。
- 权利优化：
  - 未充分使用、过度使用或无效的权利会产生风险并为组织提供有限的价值。
  - 识别这些权利并提供建议以提高效率和有效性。
- 权利保护：
  - 应有助于识别和纠正异常且具有潜在风险的权利。
  - 应根据预建规则或通过创建支持票证自动完成这些权利的补救。
- 威胁检测和响应：
  - 用户行为监控是 CIEM 解决方案的重要组成部分。
  - 异常行为应在企业 SIEM 中生成警报，并分析感兴趣的异常、模式和趋势。
- 安全态势分析：
  - 适用的安全最佳实践、法规和行业标准应集成到云授权创建过程中。
  - 应自动将政策与这些要求进行比较，生成差距分析和建议的修改。
- 权利记录和报告：
  - 有关组织权利的信息是合规性报告中的一项要求，对安全事件的调查至关重要。
  - 应自动生成日志并使用相关授权数据填充内置合规性报告模板。

---

## Cloud Security Posture Mgmt (CSPM)

Goal:

- 安全配置错误是导致云数据泄露的主要原因。为了有效地保护云环境，组织需要正确配置一系列供应商提供的安全控制。对于多个云环境，所有环境都有自己特定于供应商的安全设置，配置管理变得更加复杂。云安全态势管理(CSPM) 使组织能够监控云安全配置并识别云安全控制的潜在错误配置。

- monitor cloud infrastructure to ensure that all cloud applications and services are securely configured.

- accurately detect and remediate your greatest configuration and permission risks.

CSPM 解决方案的一些关键特性包括：
- 持续配置监控：
  - 持续监控云配置是否符合法规和其他政策违规行为。
- 资产跟踪：
    - 验证新资产是否符合公司安全策略并检查对组织云安全状况的威胁。
- 事件响应管理：
  - 实现威胁检测、隔离和补救的集中监控和管理。
- 风险识别：
  - 识别云安全威胁并对其进行分类。
- 资产清单和分类：
  - 提供对云资产及其配置设置的可见性。


![Screenshot 2023-06-27 at 11.40.04](/assets/img/Screenshot%202023-06-27%20at%2011.40.04.png)

![Screenshot 2023-06-27 at 11.40.29](/assets/img/Screenshot%202023-06-27%20at%2011.40.29.png)

![Screenshot 2023-06-27 at 11.47.44](/assets/img/Screenshot%202023-06-27%20at%2011.47.44.png)

![Screenshot 2023-06-27 at 11.48.00](/assets/img/Screenshot%202023-06-27%20at%2011.48.00.png)

![Screenshot 2023-06-27 at 11.48.05](/assets/img/Screenshot%202023-06-27%20at%2011.48.05.png)

![Screenshot 2023-06-27 at 11.48.09](/assets/img/Screenshot%202023-06-27%20at%2011.48.09.png)


---


## Cloud Detection and Response (CDR)

- Unified security for collaborative threat response

- Complex cloud innovations can result in gaps in your security. unifies visibility across workloads, cloud logs, and threat intelligence feeds, so teams are better prepared to detect quickly and respond together.

![Screenshot 2023-06-27 at 11.51.14](/assets/img/Screenshot%202023-06-27%20at%2011.51.14.png)

![Screenshot 2023-06-27 at 11.51.26](/assets/img/Screenshot%202023-06-27%20at%2011.51.26.png)

![Screenshot 2023-06-27 at 11.51.30](/assets/img/Screenshot%202023-06-27%20at%2011.51.30.png)


---

## Cloud Workload Protection Platform (CWPP)

- Complete security for your cloud workloads

- delivers full visibility across hosts, VMs, serverless functions, and hybrid environments, ensuring seamless workload management in both private and public clouds.


- Real-time detection of malicious behavior

- Continuous runtime security with ATT&CK-mapped behavioral detection

- YARA scans of in-memory processes and files

- Live and historical query investigations


![Screenshot 2023-06-27 at 11.52.29](/assets/img/Screenshot%202023-06-27%20at%2011.52.29.png)

![Screenshot 2023-06-27 at 11.53.11](/assets/img/Screenshot%202023-06-27%20at%2011.53.11.png)






.
