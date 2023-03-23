---
title: GCP - Google Cloud Computing - App Engine
date: 2021-01-01 11:11:11 -0400
categories: [01GCP, Compute]
tags: [GCP]
toc: true
image:
---

- [Google Cloud Computing - App Engine](#google-cloud-computing---app-engine)
  - [B asic](#b-asic)
  - [App Engine Environments](#app-engine-environments)
    - [Google App Engine Standard Environment](#google-app-engine-standard-environment)
  - [use App Engine Standard Environment in practice](#use-app-engine-standard-environment-in-practice)
    - [Google App Engine flexible Environment](#google-app-engine-flexible-environment)
  - [comparison](#comparison)
    - [Standard and Flexible.](#standard-and-flexible)
    - [App Engine and Kubernetes Engine.](#app-engine-and-kubernetes-engine)

---


# Google Cloud Computing - App Engine


---


## B asic

compute infrastructure for applications:
- Compute Engine and Kubernetes Engine.
- choose the infrastructure in which the application runs.
- Based on virtual machines for Compute Engine and containers for Kubernetes Engine.

> when don't want to focus on the infrastructure at all, but focus on the code.

App Engine
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

## App Engine Environments

---

### Google App Engine Standard Environment

<font color=red> Google App Engine Standard Environment </font>

- simpler deployment experience than the Flexible environment and fine-grained auto-scale.
- a free daily usage quota for the use of some services.
    - low utilization applications might be able to run at no charge.
- autoscale workloads
- usage based pricing



<font color=red> App Engine software development kits </font>

- Google provides App Engine software development kits in several languages
- can test the application locally before you upload it to the real App Engine service.
- The SDKs also provide simple commands for deployment.


> what does my code actually run on?
> what exactly is the executable binary?


<font color=red> runtime </font>

- App Engine's term for this kind of binary is the runtime.
- In App Engine Standard Environment, use a runtime provided by Google.
- App Engine Standard Environment provides runtimes for specific versions of Java, Python, PHP and Go.
- The runtimes also include libraries that support App Engine APIs.
- for many applications, the Standard Environment runtimes and libraries may be all you need.

> If you want to code in another language, Standard Environment is not right for you.
> consider the Flexible Environment.


<font color=red> Sandbox </font>

- The Standard Environment also enforces restrictions on the code by making it run in Sandbox
- a software construct that's independent of the hardware, operating system, or physical location of the server it runs on.
- The Sandbox is one of the reasons why App Engine Standard Environment can scale and manage the application in a very fine-grained way.
- Like all Sandboxes, it imposes some constraints.
- example
  - application can't write to the local file system.
    - have to write to a database service to make data persistent.
  - all the requests the application receives has a 60-second timeout
  - can't install arbitrary third party software.

> If these constraints don't work, choose the Flexible Environment.


use App Engine Standard Environment in practice
---

![Screen Shot 2021-02-08 at 00.47.35](https://i.imgur.com/4uWzf1a.png)


1. develop the application and run a test version locally using the App Engine SDK.
2. use the SDK to deploy it.
3. App Engine automatically scales and reliably serves the web application
   - Each App Engine application runs in a GCP project.
     - Project > App Engine > App servers > App instances
   - automatically provisions server instances and scales and load balances them.
4. the application can make calls to a variety of services using dedicated APIs.
   - examples:
   - a NoSQL data store to make data persistent, caching of that data using <font color=blue> Memcache </font>
   - searching
   - user logging,
   - launch actions triggered by direct user requests, like <font color=blue> task queues and a task scheduler </font>



### Google App Engine flexible Environment


<font color=red> App Engine flexible environment </font>

- build and deploy containerized apps with a click
- not sandbox constraints
  - App Engine flexible environment lets you specify the container your App Engine runs in.
  - Your application runs inside Docker containers on Google Compute Engine Virtual Machines, VMs.
- App Engine manages these Compute Engine machines for you.
  - health checked, healed as necessary,
  - critical backward-compatible updates to their operating systems are automatically applied.
- you
  - choose which geographical region they run in
  - and focus on your code.
- App Engine flexible environment apps use standard run times,
- can access App Engine services
  - such as data store, memcached, task queues, and so on.

---


## comparison

### Standard and Flexible.

| term                       | Standard                 | Flexble                                                |
| -------------------------- | ------------------------ | ------------------------------------------------------ |
| instance startup           | Milliseconds              | Minutes                                                |
| SSH access                 | No                       | Yes (not default)                                      |
| Write to local disk        | No                       | Yes (not default)                                      |
| Support 3rd party binaries | No                       | Yes                                                    |
| Network access             | Via App Engine services  | Yes                                                    |
| Pricing model              | free daily user, pay per instance class, auto shutdown | pay for resource allocation per hour, no auto shutdown |


Standard environment
- starts up instances of your application faster,
  - but get less access to the infrastructure in which the application runs.
- Google provides and maintains runtime binaries
- Scaling is finer-grained
- billing can drop to zero for the completely idle application.
  - free daily user, pay per instance class, auto shutdown

Flexible environment
- SSH into the virtual machines on which your application runs.
- use local disk for scratch base
- install third-party software
- lets your application make calls to the network without going through App Engine.


---


### App Engine and Kubernetes Engine.

![Screen Shot 2021-02-09 at 00.23.50](https://i.imgur.com/vFIOA1G.png)

**App Engine standard environment**
- who want the service to take maximum control of their application's deployment and scaling.

**Kubernetes Engine**
- gives the application owner the full flexibility of Kubernetes.

App Engine flexible edition is somewhere in between.
Also, App Engine environment treats containers as a means to an end, but for Kubernetes Engine, containers are a fundamental organizing principle.



---

















.
