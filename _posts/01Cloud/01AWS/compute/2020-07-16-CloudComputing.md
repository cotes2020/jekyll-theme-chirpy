---
title: AWS - Cloud computing
date: 2020-07-16 11:11:11 -0400
categories: [01AWS, Compute]
tags: [AWS]
math: true
image:
---


# AWS - Cloud computing

- [AWS - Cloud computing](#aws---cloud-computing)
  - [overall](#overall)
    - [traditional computing model](#traditional-computing-model)
    - [IaaS - Infrastructure as a Service](#iaas---infrastructure-as-a-service)
    - [PaaS - Platform as a Service](#paas---platform-as-a-service)
    - [SaaS - Software as a Service](#saas---software-as-a-service)

---

## overall

cloud computing:
- Cloud computing is the `on-demand delivery` of compute power, database, storage, applications, and other IT resources via the internet with `pay-as-you-go pricing`.
- These resources run on server computers that are located in large data centers in different locations around the world.
- use a cloud service, service provider owns the computers that you are using. These resources can be used together like building blocks to build solutions that help meet business goals and satisfy technology requirements.

different types of cloud computing models

### traditional computing model
- infrastructure is hardware. Hardware solutions are physical
- require space, staff, physical security, planning, and capital expenditure
- the `long hardware procurement 采购 cycle` that involves acquiring, provisioning, and maintaining on-premises infrastructure.
  - enough resource capacity or sufficient storage?
  - provision capacity by guessing theoretical maximum peaks.(pay resources stay idle / don’t have sufficient capacity to meet the needs).
  - if the needs change, then spend the time, effort, and money required to implement a new solution.
  - For example, if you wanted to provision a new website, you would need to buy the hardware, rack and stack it, put it in a data center, and then manage it or have someone else manage it.
- This approach is `expensive and time-consuming`.

### IaaS - Infrastructure as a Service
- contains the `basic building blocks for cloud IT`. 
  - Networking features
  - Computers (virtual or on dedicated hardware)
  - Data storage space
- `software solutions`
  - Compared to hardware solutions, software solutions can change much more quickly, easily, and cost-effectively.
- `highest flexibility and management control` over IT resources.
  - most similar to the existing IT resources with which many IT departments and developers are familiar.
  - select the cloud services best match the needs and provision, terminate those resources on-demand, pay as use.
  - elastically scale resources up and down in an automated fashion.
  - `treat resources as temporary and disposable`
  - enables businesses to implement new solutions quickly and with low upfront costs.
- eliminate undifferentiated heavy-lifting tasks like procurement, maintenance, and capacity planning, thus enabling them to focus on what matters most.
- several different service models and deployment strategies emerged to meet the specific needs of different users. Each type of cloud service model and deployment strategy provides you with a different level of control, flexibility, and management. Understanding the differences between these cloud service models and deployment strategies can help you decide what set of services is right for the needs.

### PaaS - Platform as a Service
- `no management for underlying infrastructure` (usually hardware and operating systems), and allows you to `focus on the deployment and management of the applications`.
- be more efficient, no worry about resource procurement, capacity planning, software maintenance, patching, or any of the other undifferentiated heavy lifting involved in running the application.

### SaaS - Software as a Service
- provides a complete product that is run and managed by the service provider.
  - end-user applications (such as web-based email).
- no maintain or underlying infrastructure is managed.
- Only think about how to use particular software.
