---
title: Palo Alto Networks - Prisma Cloud - 7
# author: Grace JyL
date: 2020-10-18 11:11:11 -0400
description:
excerpt_separator:
categories: [SOC, PaloAlto]
tags: [SOC, Prisma]
math: true
# pin: true
toc: true
image: /assets/img/note/prisma.png
---

[toc]

---

# Prisma Cloud - Compute

--

## Overview of Prisma Cloud Compute

Prisma Cloud Compute supports an architecture that requires no changes to your host, container engine, or applications.

Prisma Cloud Compute
- Prisma Cloud Compute is the `SaaS version` of the `full Cloud Native Security Platform` hosted by Palo Alto Networks.
- Prisma Cloud protects your cloud native assets anywhere they operate, whether you’re running containers, serverless functions, non-container hosts, or any combination of them.
- Advanced threat intelligence and machine learning enable protection of your entire cloud native stack, whether it runs in the public cloud, private, or hybrid cloud.
- Prisma Cloud offers a rich set of `cloud workload protection capabilities`. These features are called `Compute`.
- For environments that do not support deployment of Prisma Cloud Compute as a privileged peer, we offer `runtime application self protection (RASP) capabilities`.
- **Cloud Environment**
  - Upon deployment, Prisma Cloud immediately begins working to secure your container and cloud native assets.
  - Prisma Cloud Compute discovers assets within your cloud environment, identify assets which are not protected.
- **Scanning Capabilities**
  - Prisma Cloud Compute is easily integrated into your container build process with support for continuous integration continuous deployment (CI/CD) systems, registry, and serverless repository scanning capabilities.
- **Serverless Functions**
  - Prisma Cloud Compute supports the full stack and lifecycle of your cloud native workloads.
  - With Prisma Cloud Compute, protect mixed workload environments.
  - Whether running standalone hosts, containers, serverless functions, or combination of the above, Prisma Cloud Compute allows to manage the environment with a single interface across the entirety of the lifecycle from development to runtime.


**Accessing Compute**
- Compute has a dedicated management interface called `Compute Console`, that can be accessed in one of two ways
- <kbd>Prisma Cloud Compute</kbd>
  - Hosted by Palo Alto Networks
  - SaaS offering
  - Provides `full Cloud Native Security Platform capability`
  - Access the `Compute Console` from the `Compute tab` in the `Prisma Cloud user interface`
- <kbd>Prisma Cloud Compute Edition</kbd>
  - Self-hosted offering, deployed and managed by the enterprise
  - Provides `Cloud Workload Protection Platform (CWPP) capability`
  - Software can be downloaded from the Palo Alto Networks Customer Support Portal


Prisma Cloud Compute Core Concepts
- Compute software consists of two components: `Console` and `Defender`.
- **Console**
  - Palo Alto Networks hosts the `Console` in <kbd>Prisma Cloud Compute</kbd>.
  - Use Prisma Cloud Compute's `management console` to define policies and monitor the cloud environment.
- **Defender**
  - `Prisma Cloud Defenders` are deployed in your cloud native workloads
  - enforce the policies defined in Console
  - and send event data up to the Console for correlation.
  - There are several types of Defenders, and depending on the assets in your environment that require protection you may end up deploying all of them or a subset.
  - Defenders support the full variety of workloads in your cloud environment.
  - The following types of defenders can be deployed.
  - All Defenders, regardless of their type, report back to the Console, letting you secure a hybrid environment with a single tool.
  - The main requirement for installing Defender is that it can connect to the Console.
  - Defender connects to the Console using a websocket over port 8084 to retrieve policies and send data.
  - The following diagram shows the key connections.
  - When Defender is installed, it automatically starts scanning images, containers, and hosts for vulnerabilities.
  - **Container Defenders**:
    - Deploy Container Defenders on every host that runs containers in your cloud native environment.
  - **host Defenders**:
    - This Defender type is deployed for Virtual Machines that do not run containers.
  - **App Embedded Defenders**:
    - offer runtime protection for containers.
    - Deploy App Embedded Defender anywhere can run a container, but can’t run Container Defender.
  - **Serverless Defenders**
    - offer runtime protection for AWS Lambda functions.
    - Serverless Defender must be embedded inside your functions.
    - Deploy one Serverless Defender per function.

How Prisma Cloud Compute Works
- **Prisma Cloud Advanced Threat Protection (TATP)** is a collection of malware signatures and IP reputation lists curated from commercial threat feeds, open source threat feeds, and Prisma Cloud Labs.
- It is delivered to your installation via the `Prisma Cloud Intelligence Stream`.
- **Default Rules**
  - The TATP is enabled in the default rules that ship with the product, with the effect set to alert.
  - You can impose more stringent control by setting the action to `prevent` or `block`.
  - Runtime defense for file systems lets you actively block any container that tries to download malware.
- **Network Intelligence**
  - With app-specific network intelligence, Prisma Cloud can learn about the settings for apps from their configuration files, and use this knowledge to detect runtime anomalies.
  - No special configuration is required to enable this feature.
- **Port Settings**
  - In addition to identifying ports that are exposed via the EXPOSE directive in a Dockerfile, or the –p argument passed to docker run, Compute can identify port settings from an app’s configuration file.
  - This enables Compute to detect, for example, if the app has been commandeered to listen on an unexpected port, or if a malicious process has managed to listen on the app’s port to steal data.


<kbd>Demo: Accessing the Compute Console</kbd>

require: `prisma cloud system admin role`

prisma cloud > compute:

![Screen Shot 2020-10-22 at 19.16.21](https://i.imgur.com/R2uEfQa.png)

manage > system > download > tools:

![Screen Shot 2020-10-22 at 19.17.55](https://i.imgur.com/Bb0dJAy.png)


<kbd>Demo: Compute Console</kbd>

prisma cloud > compute:

refresh: 24h

![Screen Shot 2020-10-22 at 19.19.43](https://i.imgur.com/vQYc6RD.png)

![Screen Shot 2020-10-22 at 19.20.54](https://i.imgur.com/FxqCUGZ.png)

---

## Prisma Cloud Compute Operations

Prisma Cloud Compute is deployed in stages. Initially learn about the capabilities of Compute and plan your deployment, making considerations for accessing the console and for deploying defenders.

Prisma Cloud Compute Operations Overview
- Palo Alto Networks operates the `Console` for you, and you must deploy `Defenders` into your environment to secure hosts, containers, and serverless functions running in any cloud, including on-premises.
- After deployment, observe the results of data that are reported from your applications. When you are ready, it is time to operationalize Compute. Prisma Cloud Compute supports the following operations.


Vulnerability Management
- Vulnerability management is the optimization of your rule and policy configurations.


Compliance
- Compliance enforces the compliance checks that are built into Prisma Cloud Compute and tuning the default compliance policies for your environment.


Runtime Defense
- Runtime defense automatically models the intent of a container image so that it can be secured at runtime.


Native Firewalls
- `Prisma Cloud Native Firewalls` learn the topology of your applications and provide micro-segmentation for all your microservices.

CNAF
- CNAF is a layer 7 web application firewall (WAF).
- If a container handles web requests, configure CNAF to protect it.

CNNF
- CNNF is a layer 3 firewall that automatically models inter-container traffic.
- As part of the automatic behavioral learning at runtime, Prisma Cloud builds out a topology of connections from one container to another.


Vulnerability Management
- In Prisma Cloud Compute, when Defenders are installed, it automatically starts scanning images, containers, and hosts for vulnerabilities.
- **Vulnerability Policies**:
  - composed of discrete 分离的 rules.
- **Rules Declaration**:
  - Rules declare the actions to take when vulnerabilities are found in the resources in your environment.
  - They also control the data surfaced in Prisma Cloud Console, including scan reports and Radar visualizations.
- **Actions**
  - Rules let you `target segments of your environment` and `specify actions to take` when vulnerabilities of a given type are found.
  - For example,
  - block images with critical severity vulnerabilities from being deployed to production environment hosts.
  - There are separate vulnerability policies for containers and hosts.
  - Host rules offer a subset of the capabilities of container rules.
  - The big difference is that container rules support `blocking`.


<kbd>Demo: Vulnerability Rules</kbd>
how Prisma Cloud Compute ships with a simple default vulnerability policy for both containers and hosts.

compute > defend > vulnerability

rules: top -> bottom

![Screen Shot 2020-10-22 at 19.32.50](https://i.imgur.com/HcXYvxq.png)

![Screen Shot 2020-10-22 at 19.33.21](https://i.imgur.com/uQ8XhuU.png)

![Screen Shot 2020-10-22 at 19.34.02](https://i.imgur.com/QUMCIFQ.png)


compute > monitor > vulnerability

![Screen Shot 2020-10-22 at 19.34.55](https://i.imgur.com/HzMwllf.png)


<kbd>Demo: Compliance</kbd>
how Prisma Cloud helps enterprises monitor and enforce compliance for hosts, containers, and serverless environments.

compute > defend > compliance

![Screen Shot 2020-10-22 at 19.36.29](https://i.imgur.com/h1ZIxs8.png)

![Screen Shot 2020-10-22 at 19.36.54](https://i.imgur.com/TrD74iw.png)



<kbd>Demo: Runtime Defense</kbd>
how to set the features that provide both predictive and threat-based active protection for running containers.

compute > defend > Runtime


![Screen Shot 2020-10-22 at 19.39.34](https://i.imgur.com/2UzaedK.png)

![Screen Shot 2020-10-22 at 19.40.16](https://i.imgur.com/nQx6H7p.png)

![Screen Shot 2020-10-22 at 19.40.48](https://i.imgur.com/sOcsqlW.png)

![Screen Shot 2020-10-22 at 19.40.30](https://i.imgur.com/9TnXLj6.png)



<kbd>Demo: Firewalls</kbd>
how Prisma Cloud provides layer 4 and layer 7 firewalls that automatically learn the network topology of applications and provide application-tailored micro-segmentation for all microservices.

![Screen Shot 2020-10-22 at 19.42.04](https://i.imgur.com/eEQlgx3.png)

![Screen Shot 2020-10-22 at 19.43.21](https://i.imgur.com/1w6oaho.png)

![Screen Shot 2020-10-22 at 19.42.51](https://i.imgur.com/x4QFCQc.png)


---

## Prisma Cloud Compute Monitoring

Prisma Cloud combines vulnerability detection with an always up-to-date threat feed and knowledge about your runtime deployments to prioritize risks specifically for your environment.

the types of Compute Monitoring.
- Monitor Applications
  - In Prisma Cloud Compute, monitor applications once they have been discovered and identify and prevent vulnerabilities across the entire application lifecycle while prioritizing risk for your cloud native environments.
- Integrate Vulnerability Management
  - Integrate vulnerability management into any CI/CD process, while continuously monitoring, identifying, and preventing risks to all the hosts, images, and functions in your environment.


<kbd>Demo: Vulnerability Explorer</kbd>
how Vulnerability Explorer takes it a step further by analyzing the data within the context of your environment.

compute > monitor > vulnerability:

![Screen Shot 2020-10-22 at 19.46.46](https://i.imgur.com/XLEWx5X.png)

![Screen Shot 2020-10-22 at 19.47.40](https://i.imgur.com/pLmlsOR.png)

![Screen Shot 2020-10-22 at 19.47.52](https://i.imgur.com/uud9RxM.png)


<kbd>Demo: Search CVEs</kbd>
how to search for exposure to specific CVEs and determine if Prisma Cloud offers coverage for a specific CVE by using the search interface in the Console.

compute > monitor > vulnerability:

CVE-yyyy-nnnn

![Screen Shot 2020-10-22 at 19.49.07](https://i.imgur.com/wxjssS0.png)

![Screen Shot 2020-10-22 at 19.49.19](https://i.imgur.com/jt2yxVf.png)

---

## Prisma Cloud Compute Reports

Prisma Cloud enables you to view, assess, report, monitor, and review your cloud infrastructure health and compliance posture.

Prisma Cloud Scan Reports
- After `Defender` is installed, it automatically starts scanning images on the host.
- After the initial scan, subsequent scans are triggered periodically, according to the `scan interval` configured in the Console.
- By default, images are scanned every 24 hours.

Scanning Process
Prisma Cloud scans are also triggered when `new images are created, pushed, or pulled` onto the host, `when images change`, and `when scans are forced with the Scan button` in console.
- Step 1
  - Prisma Cloud scans all `Docker images` on all `hosts` that run `Defender`.
- Step 2
  - After Defender is installed, it automatically starts scanning `images` on the `host`.
- Step 3
  - After the **initial scan**, **subsequent scans** are triggered.


What Defender Scans For
- The `Prisma Cloud Intelligence Stream` keeps `Console` up to date with the latest vulnerabilities.
- The data in this feed is distributed to your Defenders, and employed in subsequent scans.
- Through Console, Defender can be extended to scan images for custom components.
- For example
- configure Defender to scan for an internally developed library named libexample.so, and set a policy to block a container from running if version 1.9.9 or earlier is installed.

Defender scans Docker images for:


<kbd>Demo: View Scan Reports</kbd>


compute > monitor > vulnerability > images

![Screen Shot 2020-10-22 at 20.02.37](https://i.imgur.com/88TyNO5.png)


![Screen Shot 2020-10-22 at 20.02.59](https://i.imgur.com/s6oGK9r.png)


compute > monitor > vulnerability > hosts

![Screen Shot 2020-10-22 at 20.03.18](https://i.imgur.com/45wdbcD.png)

![Screen Shot 2020-10-22 at 20.03.29](https://i.imgur.com/XeK9BtZ.png)




---

## check

Compute supports which two Defender types? (Choose two.)
- Host
- Container


What does the Intelligence Stream provide to the Console?
- Delivery of real-time threat feed


How can the Compute Console be accessed in Prisma Cloud?
- Click the Compute tab in Prisma Cloud



.
