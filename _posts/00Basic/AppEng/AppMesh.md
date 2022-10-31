---
title: App Eng - Service Mesh
# author: Grace JyL
date: 2019-08-25 11:11:11 -0400
description:
excerpt_separator:
categories: [04AppEng]
tags: [Microservices, DistributedSystems, ServiceMesh, Patterns]
math: true
# pin: true
toc: true
# image: /assets/img/sample/devices-mockup.png
---


- [Service Mesh](#service-mesh)
  - [basic](#basic)
  - [path](#path)
    - [**PC 1v1** networking computers](#pc-1v1-networking-computers)
    - [时代1：原始通信时代](#时代1原始通信时代)
    - [时代2：TCP时代](#时代2tcp时代)
    - [first started with microservices](#first-started-with-microservices)
    - [时代3：第一代微服务](#时代3第一代微服务)
    - [时代4：第二代微服务](#时代4第二代微服务)
    - [The next logical step](#the-next-logical-step)
    - [Sidecars: 时代5：第一代Service Mesh 代理模式（边车模式）](#sidecars-时代5第一代service-mesh-代理模式边车模式)
    - [时代6：第二代 Service Mesh](#时代6第二代-service-mesh)
  - [benefits](#benefits)
- [AWS App Mesh](#aws-app-mesh)
  - [basic](#basic-1)
  - [Introduction to AWS App Mesh](#introduction-to-aws-app-mesh)


- ref
  - [https://philcalcado.com/2017/08/03/pattern_service_mesh.html](https://philcalcado.com/2017/08/03/pattern_service_mesh.html)




---


# Service Mesh

---


## basic

- distributed systems enable lots use cases, but also introduce all sorts of new issues.

When these systems were rare and simple
- engineers dealt with the added complexity by minimising the number of remote interactions.
- The safest way to handle distribution has been to avoid it as much as possible, even if that meant duplicated logic and data across various systems.

from a few larger central computers to hundreds and thousands of small services

---


## path


---



### **PC 1v1** networking computers



> Variations of the model above have been in use since the 1950s.

开发人员想象中，不同服务间通信的方式，抽象表示如下：


![1](https://i.imgur.com/59mrbG8.png)


- getting two or more computers to talk to each other
  - A service talks to another to accomplish some goal for an end-user.

- more detail by showing the networking stack
  - many layers that translate between the bytes your code manipulates and the electric signals that are sent and received over a wire


- In the beginning, computers were rare and expensive, so each link between two nodes was carefully crafted and maintained.



---


### 时代1：原始通信时代

![Screen Shot 2022-03-22 at 14.14.14](https://i.imgur.com/Cuaxu6v.png)


- As computers became less expensive and more popular, the number of connections and the amount of data going through them increased drastically.
  - With people relying more and more on networked systems, engineers needed to make sure that the software they built was up to the quality of service required by their users.
  - many questions that needed to be answered to get to the desired quality levels.
  - People needed to find ways for machines to find each other,
  - to handle multiple simultaneous connections over the same wire,
  - to allow for to machines to talk to each other when not connected directly,
  - to route packets across networks, encrypt traffic, etc.

- Amongst those, there is something called **flow control**
  - a mechanism that prevents one server from sending more packets than the downstream server can process.
  - It is necessary because in a networked system you have at least two distinct, independent computers that don’t know much about each other.
  - Computer A sends bytes at a given rate to Computer B, but there is no guarantee that B will process the received bytes at a consistent and fast-enough speed.
  - For example, B might be busy running other tasks in parallel, or the packets may arrive out-of-order, and B is blocked waiting for packets that should have arrived first.
  - This means that not only A wouldn’t have the expected performance from B, but it could also be making things worse, as it might overload B that now has to queue up all these incoming packets for processing.

  - For a while, it was expected that the people `building networked services and applications would deal with the challenges` presented above.
    - In our flow control example, it meant that the application itself had to contain logic to make sure we did not overload a service with packets.
    - This networking-heavy logic sat side by side with your business logic.


---

### 时代2：TCP时代



- Fortunately, technology quickly evolved and soon enough **standards like TCP/IP** incorporated solutions to flow control and many other problems into the network stack itself.
  - This means that that piece of code still exists, but it has been extracted from your application to the underlying networking layer provided by your operating system
  - This model has been wildly successful. There are very few organisations that can’t just use the TCP/IP stack that comes with a commodity operating system to drive their business, even when high-performance and reliability are required.


![4](https://i.imgur.com/qCT5fre.png)

---


### first started with microservices

> extreme distribution brought up a lot of higher-level use cases and benefits,
> but it also surfaced several challenges.

- Over the years, computers became even cheaper and more omnipresent, and networking stack described above has proven itself as the de-facto toolset to reliably connect systems.

- With more nodes and stable connections, the industry has played with various flavours of networked systems, from `fine-grained distributed agents and objects` to `Service-Oriented Architectures composed of larger but still heavily distributed components`


- In the 90s, Peter Deutsch and his fellow engineers at Sun Microsystems compiled “**The 8 Fallacies of Distributed Computing**”, in which he lists some assumptions people tend to make when working with distributed systems.
  - Peter’s point is that these, might have been true in more primitive networking architectures or the theoretical models, but they don’t hold true in the modern world:
    - The network is reliable
    - Latency is zero
    - Bandwidth is infinite
    - The network is secure
    - Topology doesn’t change
    - There is one administrator
    - Transport cost is zero
    - The network is homogeneous

  - engineers cannot just ignore these issues, they have to explicitly deal with them.


- moving to even more distributed systems **microservices architecture**
  - has introduced new needs on the operability side:

    - Rapid provisioning of compute resources
    - Basic monitoring
    - Rapid deployment
    - Easy to provision storage
    - Easy access to the edge
    - Authentication/Authorisation
    - Standardised RPC



---


### 时代3：第一代微服务


在TCP出现之后，机器之间的网络通信不再是一个难题
- 以GFS/BigTable/MapReduce为代表的分布式系统得以蓬勃发展。
- 这时，分布式系统特有的通信语义又出现了，如`熔断策略、负载均衡、服务发现、认证和授权、quota限制、trace和监控`等等，于是服务根据业务需求来实现一部分所需的通信语义。


> So while the TCP/IP stack and general networking model is still a powerful tool in making computers talk to each other, the more sophisticated architectures introduced another layer of requirements that have to be fulfilled by engineers working in such architectures.

- the first organisations building systems based on microservices followed a strategy very similar to those of the first few generations networked computers.
  - the responsibility of dealing with the requirements listed above was left to the engineer writing the services.



![5](https://i.imgur.com/MPJhGSq.png)

  - As an example, consider **service discovery** and **circuit breakers**,
    - two techniques used to tackle several of the resiliency and distribution challenges listed above.

    - **Service discovery**
      - the process of automatically finding what instances of service fulfil a given query,
      - e.g. a service called `Teams` needs to find instances of a service called `Players` with the attribute environment set to production.
      - You will invoke some service discovery process which will return a list of suitable servers.
          - For more monolithic architectures, this is a simple task usually implemented using DNS, load balancers, and some convention over port numbers (e.g. all services bind their HTTP servers to port 8080).
          - In more distributed environments, the task starts to get more complex, and services that previously could blindly trust on their DNS lookups to find dependencies now have to deal with things like `client-side load-balancing`, `multiple different environments (e.g. staging vs. production)`, `geographically distributed servers`, etc.
          - If before all you needed was a single line of code to resolve hostnames, now your services need many lines of boilerplate to deal with various corner cases introduced by higher distribution.

    - **Circuit breakers**
      - a pattern catalogued by Michael Nygard
      - Wrap a protected function call in a circuit breaker object, which monitors for failures.
      - Once the failures reach a certain threshold, the circuit breaker trips, and all further calls to the circuit breaker return with an error, without the protected call being made at all.
      - Usually you’ll also want some kind of monitor alert if the circuit breaker trips.


- These are great simple devices to add more reliability to interactions between your services. Nevertheless, just like everything else they tend to get much more complicated as the level of distribution increases.
  - The likelihood of something going wrong in a system raises exponentially with distribution, so even simple things like “some kind of monitor alert if the circuit breaker trips” aren’t necessarily straightforward anymore.
  - One failure in one component can create a cascade of effects across many clients, and clients of clients, triggering thousands of circuits to trip at the same time.
  - Once more what used to be just a few lines of code now requires loads of boilerplate to handle situations that only exist in this new world.


---


### 时代4：第二代微服务


![5-a](https://i.imgur.com/d2mqdVj.png)

为了避免每个服务都需要自己实现一套分布式系统通信的语义功能，随着技术的发展，一些面向微服务架构的开发框架出现了
- 如Twitter的Finagle、Facebook的Proxygen以及Spring Cloud等等
- 这些框架实现了分布式系统通信需要的各种通用语义功能：如负载均衡和服务发现等，因此一定程度上屏蔽了这些通信细节，使得开发人员使用较少的框架代码就能开发出健壮的分布式系统。


- In fact, the two examples listed above can be so hard to implement correctly that large,

- sophisticated libraries like `Twitter’s Finagle and Facebook’s Proxygen` became very popular as means to **avoid rewriting the same logic in every service**.
  - The model depicted above was followed by the majority of the organisations that pioneered the microservices architecture, like Netflix, Twitter, and SoundCloud. As the number of services in their systems grew, they also stumbled upon various drawbacks of this approach.


1. even when using a library like Finagle, is that an organisation will still need to invest time from its engineering team in building the glue that links the libraries with the rest of their ecosystem.
   - Based on my experiences at SoundCloud and DigitalOcean I would estimate that following this strategy in a 100-250 engineers organisation, one would need to dedicate 1/10 of the staff to building tooling.
   - Sometimes this cost is explicit as engineers are assigned to teams dedicated to building tooling, but more often the price tag is invisible as it manifests itself as time taken away from working on your products.

2. the setup above limits the `tools, runtimes, and languages` you can use for your microservices.
   - Libraries for microservices are often written for a specific platform, be it a programming language or a runtime like the JVM. If an organisation uses platforms other than the one supported by the library, it often needs to port the code to the new platform itself.
   - This steals scarce engineering time. Instead of working on their core business and products, engineers have to, once again, build tools and infrastructure.
   - That is why some medium-sized organisations like SoundCloud and DigitalOcean decided to support only one platform for their internal services—Scala and Go respectively.


3. governance. The library model might abstract the implementation of the features required to tackle the needs of the microservices architecture, but it is still in itself a component that needs to be maintained.
   - Making sure that thousands of instances of services are using the same or at least compatible versions of your library isn’t trivial, and every update means integrating, testing, and re-deploying all services—even if the service itself didn’t suffer any change.


---


### The next logical step


> Similarly to what we saw in the networking stack

> to extract the features required by massively distributed services into an underlying platform.


- People write very sophisticated applications and services using higher level protocols like HTTP without even thinking about how TCP controls the packets on their network.

- engineers working on services can focus on their business logic and avoid wasting time in writing their own services infrastructure code or managing libraries and frameworks across the whole fleet.


![6](https://i.imgur.com/kRCInaB.png)



- Unfortunately, changing the networking stack to add this layer isn’t a feasible task. The solution found by many practitioners was to **implement it as a set of proxies**.
  - The idea here is that a service won’t connect directly to its downstream dependencies, but instead all of the traffic will go through a small piece of software that transparently adds the desired features.


---

### Sidecars: 时代5：第一代Service Mesh 代理模式（边车模式）

它将分布式服务的通信抽象为单独一层
- 在这一层中实现`负载均衡、服务发现、认证授权、监控追踪、流量控制`等分布式系统所需要的功能
- 作为一个和服务对等的代理服务，和服务部署在一起，接管服务的流量，通过代理之间的通信间接完成服务之间的通信请求，这样上边所说的三个问题也迎刃而解。

- The first documented developments in this space used the concept of **sidecars**.
  - an auxiliary process that runs aside your application and provides it with extra features.
  - In 2013, Airbnb wrote about Synapse and Nerve, their open-source implementation of a sidecar.
  -  One year later, Netflix introduced Prana, a sidecar dedicated to allowing for non-JVM applications to benefit from their NetflixOSS ecosystem.
  -  At SoundCloud, we built sidecars that enabled our Ruby legacy to use the infrastructure we had built for JVM microservices.


![6-a](https://i.imgur.com/ZLoLIoP.png)



- While there are several of these open-source proxy implementations, they tend to be designed to **work with specific infrastructure components**.
  - As an example, when it comes to service discovery Airbnb’s Nerve & Synapse assume that services are registered in Zookeeper, while for Prana one should use Netflix’s own Eureka service registry for that.


- With the increasing popularity of microservices architecture, we have recently seen a **new wave of proxies that are flexible enough** to adapt to different infrastructure components and preferences.
  - The first widely known system on this space was `Linkerd`, created by Buoyant based on their engineers’ prior work on Twitter’s microservices platform. Soon enough, the engineering team at Lyft announced `Envoy` which follows a similar principle.



![mesh1](https://i.imgur.com/MF9jYB1.png)



> Buoyant’s CEO William Morgan made the observation that the the interconnection between proxies form a mesh network. In early 2017, William wrote a definition for this platform, and called it a Service Mesh:

- In such model, each of your services will have a **companion proxy sidecar**. Given that services communicate with each other only through the sidecar proxy, we end up with a deployment similar to the diagram below:

service mesh
- a dedicated infrastructure layer for handling **service-to-service communication**.
- It’s responsible for the reliable delivery of requests through the complex topology of services that comprise a modern, cloud native application.
- In practice, the service mesh is typically implemented as an array of `lightweight network proxies` that are deployed alongside application code, without the application needing to be aware.
- it moves away from thinking of proxies as isolated components and acknowledges the network they form as something valuable in itself.


---


### 时代6：第二代 Service Mesh

以Istio为代表的第二代Service Mesh
- 第一代Service Mesh由一系列`独立运行的单机代理服务`构成
- 为了提供统一的上层运维入口，演化出了集中式的控制面板
- 所有的单机代理组件通过和`控制面板`交互进行`网络拓扑策略`的更新和单机数据的汇报。


![6-b](https://i.imgur.com/y6hS7Bz.png)

As organisations move their microservices deployments to more sophisticated runtimes like Kubernetes and Mesos, people and organisations have started using the tools made available by those platforms to implement this idea of a mesh network properly.
- They are moving away from a set of independent proxies working in isolation to a proper, somewhat centralised, control plane.

- Looking at our bird’s eye view diagram, we see that the actual service traffic still flows from proxy to proxy directly, but the control plane knows about each proxy instance.
- The control plane enables the proxies to implement things like `access control and metrics collection`, which requires cooperation:

![mesh3](https://i.imgur.com/eMGusFI.png)


- The recently announced `Istio project` is the most prominent example of such system.


---



## benefits

- not having to write custom software to deal with what are ultimately commodity code for microservices architecture will allow for many smaller organisations to enjoy features previously only available to large enterprises, creating all sorts of interesting use cases.

- this architecture might allow us to finally realise the dream of using the best tool/language for the job without worrying about the availability of libraries and patterns for every single platform.





---


# AWS App Mesh

---


## basic

- a service mesh that provides application-level networking to make it easy for your services to communicate with each other across multiple types of compute infrastructure. App Mesh gives end-to-end visibility and high-availability for your applications.

Modern applications are typically composed of multiple services. Each service may be built using multiple types of compute infrastructure such as Amazon EC2, Amazon ECS, Amazon EKS, and AWS Fargate. As the number of services grow within an application, it becomes difficult to pinpoint the exact location of errors, re-route traffic after failures, and safely deploy code changes. Previously, this has required you to build monitoring and control logic directly into your code and redeploy your service every time there are changes.

AWS App Mesh makes it easy to run services by providing consistent visibility and network traffic controls, and helping you deliver secure services. App Mesh removes the need to update application code to change how monitoring data is collected or traffic is routed between services. App Mesh configures each service to export monitoring data and implements consistent communications control logic across your application.

You can use App Mesh with AWS Fargate, Amazon EC2, Amazon ECS, Amazon EKS, and Kubernetes running on AWS, to better run your application at scale. App Mesh also integrates with AWS Outposts for your applications running on-premises. App Mesh uses the open source Envoy proxy, making it compatible with a wide range of AWS partner and open source tools.


---


## Introduction to AWS App Mesh



Benefits

Get end-to-end visibility

App Mesh captures metrics, logs, and traces from all of your applications. You can combine and export this data to Amazon CloudWatch, AWS X-Ray, and compatible AWS partner and community tools for monitoring and tracing. This lets you quickly identify and isolate issues with any service to optimize your entire application.
Streamline your operations

App Mesh provides controls to configure and standardize how traffic flows between your services. You can easily implement custom traffic routing rules so that your service is highly available during deployments, after failures, and as your application scales. This removes the need to configure communication protocols for each service, write custom code, or implement libraries to operate your application.
Enhance network security

App Mesh helps encrypt all requests between services even when they are in your private networks. Further, you can add authentication controls to ensure that only services that you allow interconnect.
How it works
