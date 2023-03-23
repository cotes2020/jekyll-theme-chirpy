---
title: SecConcept - Single Tenant vs Multi-Tenant
date: 2020-11-11 11:11:11 -0400
categories: [10SecConcept]
tags: [SecConcept]
toc: true
image:
---

[toc]

- ref
  - [SaaS: Single Tenant vs Multi-Tenant - What's the Difference?](https://digitalguardian.com/blog/saas-single-tenant-vs-multi-tenant-whats-difference)

---

## Saas

![1_H5NOx0tvsAnCIWRjfXtf4w](https://i.imgur.com/k7CZkyL.png)


![google-computre-engine-instance-placement-multi-tenant-vs-single-tenant](https://i.imgur.com/pMGXKqh.png)

---


## Single Tenant

- A single instance of the software and supporting infrastructure serve a single customer.
- each customer has his or her own independent database and instance of the software.
 no sharing happening with this option.

**Potential benefits:**
- <font color=blue> Security </font>:
  - A single customer and a single server is often contained on secure hardware being used by a limited number of people.
- <font color=blue> Dependability </font>:
  - With an entire environment dedicated to one client, resources are abundant and available anytime.
- <font color=blue> Customization </font>:
  - Control over the entire environment allows for customization and added functionality, if desired.

**Potential drawbacks:**
- <font color=blue> Maintenance </font>:
  - more tasks and regular maintenance to keep things running smoothly and efficiently.
- <font color=blue> Setup/Management </font>:
  - multi-tenant environments are quick to setup and manage.
- <font color=blue> Cost </font>:
  - Single-tenant typically allows for more resources, but at a premium price given that there is only one customer for the entire environment.



## multi-tenant architecture

- CORE BENEFIT OF SAAS
- Multi-tenancy is just one of multiple benefits of SaaS.
- a single instance of the software and its supporting infrastructure serves multiple customers.
- Each customer shares the software application and also shares a single database.
- Each tenant’s data is isolated and remains invisible to other tenants.

**Potential benefits**
- <font color=blue> Affordable Cost </font>:
  - the cost for the environment is shared, and those savings (from the SaaS vendor) are typically transferred to the cost of the software.
  - Lower costs through economies of scale
  - scaling has far fewer infrastructure implications than with a single-tenancy-hosted solution because new users get access to the same basic software.
  - Shared infrastructure leads to lower costs: SaaS allows companies of all sizes to share infrastructure and data center operational costs. There is no need to add applications and more hardware to their environment. Not having to provision or manage any infrastructure or software above and beyond internal resources enables businesses to focus on everyday tasks.
- <font color=blue> Integrations </font>:
  - Cloud environments allow for easier integration with other applications through the use of APIs.
  - Multi-tenant solutions are designed to be highly configurable so that businesses can make the application perform the way they want.
  - There is no changing the code or data structure, making the upgrade process easy.
- <font color=blue> “Hands-free” Maintenance </font>:
  - The server technically belongs to the SaaS vendor, meaning that a certain level of database maintenance is handled by the vendor, instead of you maintaining the environment yourself.
  - Ongoing maintenance and updates: Customers don’t need to pay costly maintenance fees to keep their software up to date. Vendors roll out new features and updates. These are often included with a SaaS subscription.
- Configuration can be done while leaving the underlying codebase unchanged:
  - Single-tenant-hosted solutions are often customized, requiring changes to an application’s code.
  - This customization can be costly and can make upgrades time-consuming because the upgrade might not be compatible with your environment.

**Potential drawbacks**
- <font color=blue> Limited Management/Customization: </font>
  - While you do have added integration benefits, custom changes to the database aren’t typically an option.
- <font color=blue> Security </font>:
  - Other tenants won’t see your data.
  - However, multiple users (not associated with your organization) are allowed on the same database.
  - This broader access reduces control of security.
  - Multi-Tenancy的架構，在系統設計時的要考量的點可不少，包含獨立的設定、客製化、安全性、獨立頻寬、獨立CPU/記憶體資源等
  - 獨立而不會受到其他租戶的影響，這些對SaaS服務都非常的重要
  - 否則只要一個租戶發生狀況，其他租戶可能就會連帶的受到影響，一個安全性漏洞就連帶影響到其他租戶。
- <font color=blue> Updates/Changes </font>:
  - If you’re reliant on integrations with other SaaS products and one updates their system, it may cause issues with those connecting apps.
