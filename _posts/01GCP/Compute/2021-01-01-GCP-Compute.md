---
title: GCP - Google Cloud Computing Solutions
date: 2021-01-01 11:11:11 -0400
categories: [01GCP, Compute]
tags: [GCP]
toc: true
image:
---

- [Google Cloud Computing Solutions](#google-cloud-computing-solutions)
  - [Cloud Computing](#cloud-computing)
  - [Google Cloud Computing Solutions](#google-cloud-computing-solutions-1)
    - [Compute type](#compute-type)
    - [compare](#compare)
    - [IaaS vs PaaS vs Serverless](#iaas-vs-paas-vs-serverless)
  - [IaaS](#iaas)
    - [Compute Engine](#compute-engine)
  - [PaaS](#paas)
    - [App Engine](#app-engine)
  - [Serverless](#serverless)
    - [Cloud Function](#cloud-function)
  - [container - Stateless](#container---stateless)
    - [Cloud Run](#cloud-run)
  - [container - Hybrid](#container---hybrid)
    - [GKE Kubernetes Engine](#gke-kubernetes-engine)
  - [which compute service to you adopt](#which-compute-service-to-you-adopt)

---


# Google Cloud Computing Solutions


---

## Cloud Computing


![Screen](https://i.imgur.com/7vMDIFw.png)

![Screen Shot 2021-02-03 at 14.32.13](https://i.imgur.com/6p8blb8.png)

5 fundamental attributes.
1. on-demand and self-service.
   1. automated interface and get the processing power, storage, and network they need with no human intervention.
2. broad network access
   1. resources are accessible over a network from any location.
3. Resource pooling
   1. Providers allocate resources to customers from a large pool,
   2. allowing them to benefit from economies of scale.
   3. Customers don't have to know or care about the exact physical location of these resources.
4. Rapid elasticity
   1. Resources themselves are elastic.
   2. Customers who need more resources can get them rapidly,
   3. when they need less they can scale back.
5. Measured service
   1. customers pay for only what they use or reserve as they go.
   2. stop using resources, stop paying.

---

## Google Cloud Computing Solutions


> GCP products that provide the compute infrastructure for applications

### Compute type


![Screen Shot 2022-08-14 at 23.53.19](https://i.imgur.com/zqLCHjW.jpg)

![Screen Shot 2022-08-14 at 23.54.09](https://i.imgur.com/GrvWJPN.jpg)

![Screen Shot 2022-08-15 at 00.11.35](https://i.imgur.com/7tbu3te.png)


### compare


![Screen Shot 2022-08-15 at 00.11.25](https://i.imgur.com/QauFUMT.png)

![Screen Shot 2022-08-28 at 16.41.39](https://i.imgur.com/jny3hPU.png)


### IaaS vs PaaS vs Serverless

![Screen Shot 2021-06-27 at 1.20.06 AM](https://i.imgur.com/tXqN8CH.png)

![Screen Shot 2021-02-12 at 13.25.46](https://i.imgur.com/uuTClRK.png)

![Screen Shot 2021-02-03 at 14.34.02](https://i.imgur.com/e2nAsAC.png)

![Screen Shot 2021-02-09 at 23.26.11](https://i.imgur.com/Zghiw6i.png)

---


## IaaS

### Compute Engine

![Screen Shot 2021-02-14 at 21.27.28](https://i.imgur.com/pKrLZLF.png)

- <font color=red> Compute Engine </font>
  - [detailed page](https://ocholuo.github.io/posts/Compute-engine/)
  - <font color=blue> Infrastructure as a Service </font>
  - A managed environment for deploying virtual machines
  - Fully customizable **VMs**
    - Compute Engine offers virtual machines that run on GCP
    - create and run virtual machines on Google infrastructure.
    - run virtual machines on demand in the Cloud.
      - select predefined VM configurations
      - create customized configurations
  - no upfront investments
  - run thousands of virtual CPUs on a system that is designed to be fast and to offer consistent performance.
  - choice:
    - have complete control over your infrastructure
      - maximum flexibility
      - for people who prefer to manage those server instances themselves.
      - customize operating systems and even run applications that rely on a mix of operating systems.
    - best option when other computing options don't support your applications or requirements
      - easily lift and shift your on-premises workloads into GCP without rewriting the applications or making any changes.


---


## PaaS


### App Engine

![Screen Shot 2021-02-14 at 21.28.45](https://i.imgur.com/0ngmPQq.png)


- <font color=red> App Engine </font>
  - [detailed page](https://ocholuo.github.io/posts/app-engine/)
  - <font color=blue> Platform as a Service </font>
  - fully managed serverless application framework.
    - deploy an application on App Engine
      - hand App Engine the code
      - and the App Engine service takes care of the rest.
    - focus on code and run code in the Cloud
      - without worry about infrastructure.
      - focus on building applications instead of deploying and managing the environment.
      - Google deal with all the provisioning and resource management.
        - no worry about building the highly reliable and scalable infrastructure
        - zero server management or configuration deployments for deploying applications
        - The App Engine platform manages the hardware and networking infrastructure for the code.
  - provides built-in services that many web applications need.
    - code the application to take advantage of these services and App Engine provides them.
    - `NoSQL databases, in-memory caching, load balancing, health checks, logging` and a `way to authenticate users`.
    - could also `run container workloads`.
    - `Stackdriver monitoring, logging, and diagnostics`
      - such as debugging and error reporting are also tightly integrated with App Engine.
      - use Stackdriver's real time debugging features to analyze and debug your source code.
      - Stackdriver integrates with tools such as Cloud SDK, cloud source repositories, IntelliJ, Visual Studio, and PowerShell.
    - App Engine also supports `version control and traffic splitting`.
  - scale the application automatically in response to the amount of traffic it receives.
  - only pay for those resources you use.
    - no servers to provision or maintain.
  - App Engine offers two environments:
    - standard and flexible
  - App Engine supports popular languages like Java and Node.js, Python, PHP, C#, .NET, Ruby, and Go.
  - especially suited for applications
    - where the workload is highly variable or unpredictable
    - like web applications and mobile backend.
    - for websites, mobile apps, gaming backends,
    - and as a way to present a RESTful API to the Internet
      - an application program interface
      - resembles the way a web browser interacts with the web server.
      - RESTful APIs are easy for developers to work with and extend.
      - And App Engine makes them easy to operate

---


## Serverless

---

### Cloud Function

![Screen Shot 2021-02-14 at 21.33.08](https://i.imgur.com/Hf2hreB.png)

- <font color=red> Cloud Function </font>
  - <font color=blue> functions as a Service </font>
  - A managed serverless platform/environment for deploying event-driven functions
    - an event-driven, serverless compute service
    - for simple single purpose functions that are attached to events.
    - It executes the code in response to events,
      - whether those occur once a day or many times per second.

  - create single-purpose functions that respond to events without servers or runtime binaries.
    - just write code in JavaScript for a Node.js environment that GCP provides
    - upload the code written in JavaScript or Python, or Go
    - configure when it should fire
      - setting up a Cloud Function works.
      - choose which events you care about.
      - triggers: For each event type, you tell Cloud Functions you're interested in it.
      - attach JavaScript functions to the triggers.
    - and then GCP will automatically deploy the appropriate computing capacity to run that code.
    - the functions will respond whenever the events happen.

  - Google scales resources as required, but you only pay for the service while the code runs.
    - no pay for servers
    - charged for the time that the code/functions runs.
    - For each function, invocation memory and CPU use is measured in the 100 millisecond increments, rounded up to the nearest increment.
    - provides a perpetual free tier.
    - So many cloud function use cases could be free of charge.

  - the code is triggered within a few milliseconds based on events.
    - can trigger on events in Cloud Storage, Cloud Pub/Sub,
      - file is uploaded to Google cloud storage
      - or a message is received from Cloud Pub/Sub.
    - or in HTTP call
      - triggered based on HTTP endpoints define,
    - and events in the fire based mobile application back end.

  - to enhance existing applications without having to worry about scaling.

  - These servers are automatically scaled and are deployed from highly available and a fault-tolerant design.

  - use cases
    - used as part of a microservices application architecture.
      - Some applications, especially those that have microservices architecture, can be implemented entirely in Cloud Functions.
    - build symbols, serverless,
      - mobile IoT backends
      - integrate with third party services and APIs.

    - Files uploaded into the GCS bucket can be processed in real time.
    - the data can be extracted, transformed and loaded for querying in analysis.
    - intelligent applications
      - such as virtual assistance, chat bots
      - video or image analysis, and sentiment analysis.

---


## container - Stateless


---

### Cloud Run

![Screen Shot 2021-02-14 at 21.32.19](https://i.imgur.com/PWNsn3v.png)

- <font color=red> Cloud Run </font>
  - serverless
  - builds, deploys, and manages modern stateless workloads.
    - can build the applications in any language using whatever frameworks and tools
    - deploy them in seconds without manage and maintain the server infrastructure.
      - distracts way all the infrastructure management
      - such as provisioning, configuring, managing those servers
      - only focus on developing applications.
    - run request or event driven stateless workloads
      - without having to worry bout servers.
  - automatically scales up and down from zero
    - depending upon traffic almost instantaneously
    - no worry about scale configuration.
  - pay for only the resources used
    - calculated down to the nearest 100 milliseconds.
    - no pay for those over provisioned resources.
  - gives the choice of running the containers
    - with fully managed or in the own GKE cluster.
    - deploy the stateless containers with a consistent developer experience to a fully managed environment or to the own GKE cluster.
    - This common experiences enabled by Knative
      - Cloud Run is built on Knative
        - an open source Kubernetes based platform.
        - an open API and runtime environment built on top of Kubernetes.
  - gives the freedom to move the workloads across different environments and platforms,
    - either fully managed on GCP, on GKE
    - or anywhere a Knative runs.
  - enables you to deploy stateless containers
    - that listen for requests or events delivered via HTTP requests.

---


## container - Hybrid


---


### GKE Kubernetes Engine

![Screen Shot 2021-02-14 at 21.30.41](https://i.imgur.com/JZUUrNV.png)

- <font color=red> GKE Kubernetes Engine </font>
  - [detailed page](https://ocholuo.github.io/posts/kubernete-engine/)
  - A managed environment for deploying containerized applications
  - to run containerized applications on a Cloud environment that Google Cloud manages for you under the administrative control.
  - containerization, a way to package code that's designed to be highly portable and to use resources very efficiently.
  - Kubernetes, a way to orchestrate code in those containers.


---


## which compute service to you adopt

![Screen Shot 2021-02-14 at 21.36.35](https://i.imgur.com/8e7pec1.png)


- compute engine
  - running applications on physical server hardware
  - running applications in long-lived virtual machines in which each VM is managed and maintained
    - moving to compute engine is the quickest GCP services for getting the applications to the cloud.

- What do you want to to think about operations at all? Well, App Engine and Cloud Functions are good choices.

- Containerization is the most efficient, importable way to package you an application.
- The popularity of containerization is growing very fast.

- both Compute Engine and App Engine can launch containers for you.
  - Compute Engine
    - accept the container image and launch a virtual machine instance that contains it.
    - use Compute Engine technologies to scale and manage the resulting VM.
  - App Engine flexible environment
    - accept the container image and then run it with the same No-ops environment that App Engine delivers for code.

- GKE:
  - if you're already running Kubernetes in the on-premises data centers
    - you'll be able to bring along both the workloads and the management approach.
  - want more control over the containerized workloads than what App Engine offers
  - And denser packing than what Compute Engine offers
  - The Kubernetes paradigm of container orchestration is incredibly powerful, and its vendor neutral, and a abroad and vibrant community is developed all around it.
  - Using Kubernetes as a managed service from GCP saves you work and let's you benefit from all the other GCP resources too.

- Cloud Run
  - run stateless containers on a managed compute platform.
