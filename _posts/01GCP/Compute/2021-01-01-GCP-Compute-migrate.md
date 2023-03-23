---
title: GCP - Compute migrate
date: 2021-01-01 11:11:11 -0400
categories: [01GCP, Compute]
tags: [GCP]
toc: true
image:
---

- [Compute migrate](#compute-migrate)
  - [modern Hybrid on Multi-Cloud Computing](#modern-hybrid-on-multi-cloud-computing)
    - [Traditional: on-premises distributed systems architecture](#traditional-on-premises-distributed-systems-architecture)
    - [Modern: hybrid on multi-cloud architecture](#modern-hybrid-on-multi-cloud-architecture)
  - [Migration](#migration)
    - [Migrate for VM](#migrate-for-vm)
    - [Migrate for Compute Engine](#migrate-for-compute-engine)
    - [Anthos (Migrate for Container / from VM to Container)](#anthos-migrate-for-container--from-vm-to-container)
      - [build modern hybrid infrastructure stack](#build-modern-hybrid-infrastructure-stack)
    - [Transfer Appliance (Migrate for big data)](#transfer-appliance-migrate-for-big-data)
    - [Migrate for IAM](#migrate-for-iam)
  - [Deployment methods](#deployment-methods)

---

# Compute migrate



---


## modern Hybrid on Multi-Cloud Computing

---

### Traditional: on-premises distributed systems architecture

![Screen Shot 2021-02-07 at 23.20.20](https://i.imgur.com/dX9TDe3.png)

> how business is traditionally made the enterprise computing needs before cloud computing.

- most enterprise scale applications are designed as distributed systems.
  - Spreading the computing workload required to provide services over two or more network servers.
  - containers can break these workloads down into microservices,
  - more easily maintained and expanded.

- <font color=red> on-premises systems </font>
  - Enterprise systems and workloads, containerized or not, have been housed on-premises,
  - a set of high-capacity servers running in the company's network or data center.
  - When an application's computing needs begin to outstrip its available computing resources
    - need to procure more powerful servers.
    - Install them on the company network after any necessary network changes or expansions.
    - Configure the new servers
    - and finally load the application and it's dependencies onto the new servers before resource bottlenecks could be resolved.
  - shortcut
    - The time required to complete an on-premises upgrade could be from months to years.
    - also costly, the useful lifespan of the average server is only three to five years.


> what if you need more computing power now, not months from now?
>
> What if the company wants to begin to relocate some workloads away from on-premises to the Cloud to take advantage of lower cost and higher availability, but is unwilling or unable to move the enterprise application from the on-premises network?
>
> What if you want to use specialized products and services that only available in the Cloud?
>
> This is where a modern hybrid or multi-cloud architecture can help.

---

### Modern: hybrid on multi-cloud architecture

- creating an environment uniquely suited to the company's needs.
  - keep parts of the systems infrastructure on-premises
  - Move only specific workloads to the Cloud at the own pace
    - because a full scale migration is not required for it to work.

- benefits:
  - Take advantage of the cloud services for running the workloads you decide to migrate.
    - <font color=red> flexibility, scalability, and lower computing costs </font>
  - Add specialized services to the computing resources tool kit.
    - such as <font color=red> machine learning, content caching, data analysis, long-term storage, and IoT </font>

- the adoption of hybrid architecture for powering distributed systems and services.


---

## Migration


---



### Migrate for VM

![Screen Shot 2022-08-16 at 23.14.33](https://i.imgur.com/iIJ4hkf.png)

---


### Migrate for Compute Engine

![Screen Shot 2022-08-16 at 23.14.48](https://i.imgur.com/UI4omjt.png)

![Screen Shot 2022-08-16 at 23.15.17](https://i.imgur.com/pOHzmTN.png)


---

### Anthos (Migrate for Container / from VM to Container)


![Screen Shot 2022-08-16 at 23.15.26](https://i.imgur.com/CQimUze.png)

![Screen Shot 2022-08-16 at 23.15.37](https://i.imgur.com/qoT2fh4.png)

- modern solution for <font color=blue> hybrid and multi-cloud distributed systems and service management </font>
  - powered by the latest innovations in distributed systems, and service management software from Google.
- On-permises and Cloud environments stay in sync
  - The Anthos framework rests on Kubernetes and GKE on-prem.
- provides
  - the foundation for an architecture
    - the foundation that is fully integrated with centralized management through a central control plane that supports <font color=blue> policy based application lifecycle </font> delivery across <font color=blue> hybrid and multi-cloud environments </font>
  - a rich set of tools
    - Manage sevices on-permises and in the cloud
    - monitor systems and services
      - for monitoring and maintaining the consistency of the applications across all network (on-premises, Cloud, multiple clouds)
    - migrate application from VMs into the clusters
    - maintain consistent policies across across all network (on-premises, Cloud, multiple clouds)

---

#### build modern hybrid infrastructure stack

![Screen Shot 2021-02-07 at 23.50.31](https://i.imgur.com/7LTuSeN.png)


- <font color=red> Google Kubernetes Engine on the Cloud site </font> of the hybrid network.
  - Google Kubernetes Engine is a managed production-ready environment for <font color=blue> deploying containerized applications </font>
  - Operates seamlessly with high availability and an SLA.
  - Runs certified Kubernetes ensuring portability across clouds and on-premises.
  - Includes auto-node repair, and auto-upgrade, and auto-scaling.
  - Uses regional clusters for high availability with multiple masters.
  - Node storage replication across multiple zones.

- <font color=red> Google Kubernetes Engine deployed ON-PREM </font>
  - a turn-key production-grade conformed version of Kubernetes
  - with the best practice configuration already pre-loaded.
  - Provides
    - <font color=blue> easy upgrade path to the latest validated Kubernetes releases </font> by Google.
    - <font color=blue> Provides access to container services </font> on Google Cloud platform,
      - such as Cloud build, container registry, audit logging, and more.
    - <font color=blue> integrates with Istio, Knative and Marketplace Solutions </font>
  - Ensures a consistent Kubernetes version and experience across Cloud and on-premises environments.

- <font color=red> Marketplace </font>
  - both <font color=blue> Google Kubernetes Engine in the Cloud </font> and <font color=blue> Google Kubernetes Engine deployed on-premises </font> integrate with <font color=blue> Marketplace </font>
  - so all of the clusters in network (on-premises or in the Cloud), have access to the same repository of containerized applications.
  - benefits:
    - use the same configurations on both the sides of the network,
    - reducing the time spent developing applications.
    - use ones replicate anywhere
    - maintaining conformity between the clusters.

> Enterprise applications may use hundreds of microservices to handle computing workloads.
> Keeping track of all of these services and monitoring their health can quickly become a challenge.



- <font color=red> Anthos </font>
  - an Istio Open Source service mesh
  - take these guesswork out of managing and securing the microservices.

- <font color=red> Cloud interconnect </font>
  - These service mesh layers communicate across the hybrid network by Cloud interconnect
  - to sync and pass their data.

- <font color=red> Stackdriver </font>
  - the <font color=blue> built-in logging and monitoring solution </font> for Google Cloud.
    - offers a fully managed logging, metrics collection, monitoring dashboarding, and alerting solution that watches all sides of the hybrid on multi-cloud network.
  - the ideal solution for <font color=blue> single easy configure powerful cloud-based observability solution </font>
  - a single pane of class dashboard to monitor all of the environments.

- <font color=red> Anthos Configuration Management </font>
  - provides
    - a single source of truth for the clusters configuration.
      - source of truth is kept in the policy repository, a git repository.
      - this repository can be located on-premises or in the Cloud.
    - deploy code changes with a single repository commit.
    - implement configuration inheritance, by using namespaces.

- <font color=red> Anthos Configuration Management agents </font>
  - use the policy repository to enforce configurations locally in each environment,
  - managing the complexity of owning clusters across environments.


---


### Transfer Appliance (Migrate for big data)



![Screen Shot 2022-08-16 at 23.15.56](https://i.imgur.com/ft78tg5.png)


---



### Migrate for IAM


![Screen Shot 2022-08-16 at 23.16.34](https://i.imgur.com/I5KqbRd.jpg)

![Screen Shot 2022-08-16 at 23.16.43](https://i.imgur.com/4C6tS1O.jpg)



---

## Deployment methods

Blue green deployments

- 2 environments, arbitrarily called blue, and green
- toggle between the two of them.
- Imagine
  - green environment is currently serving traffic,
  - use the blue environment to test out the latest version,
  - and once you're happy with the results, just switch the traffic over to blue,
  - and you just repeat this process.

![Screen Shot 2022-08-25 at 00.31.08](https://i.imgur.com/RY6hVRf.png)



Rolling deployments

- Rolling deployments progressively replace a resource with another version until everything has been updated.
- Imagine
  - have five resources all on version 100, and you wanna roll out version 101, without impacting users,
  - so you update the resources one at a time, making sure there are no failures until everything is up to date.

![Screen Shot 2022-08-25 at 00.39.39](https://i.imgur.com/k26ROyK.png)




canary deployments

- Canary deployments get their name from a mining practice, which involved bringing canaries into coalmines, because their death was an indicator of lethal gases.
- The process is similar in software though, without the potential ethical debates.
- Imagine
  - A new version is introduced into the current group of resources and it's monitored.
  - If there are problems, then only a small portion of the total user base are going to experience those problems.
  - Once everything's working as it should, that version can be fully deployed.


![Screen Shot 2022-08-25 at 00.40.02](https://i.imgur.com/syhSKBf.png)

Traffic splitting deployments

- Traffic splitting diverts traffic to a different version of a resource
- use cases for this.
  - the classic A/B testing use case.
    - You have two versions that you want to see how users respond to,
    - and so you split the traffic between those two versions,
    - you monitor for whatever it is you're looking to see,
    - once you know which one is more successful, that is the one that you can actually deploy.

![Screen Shot 2022-08-25 at 00.40.25](https://i.imgur.com/dErLBSq.png)





.
