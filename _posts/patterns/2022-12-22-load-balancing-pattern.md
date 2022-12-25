---
title: Load Balancing Pattern
description: description
tags: ["cloud", "load balancing", "design", "scaling"]
category: ["architecture", "patterns"]
date: 2022-12-22
permalink: '/patterns/load-balancing/'
counterlink: 'patterns-load-balancing/'
image:
  path: https://raw.githubusercontent.com/Gaur4vGaur/traveller/master/images/patterns/2022-12-22-load-balancing-pattern.jpg
  width: 800
  height: 500
---

## Introduction
Any modern website on the internet today receives thousands of hits, if not millions. Without any scalability strategy, the website is either going to crash or significantly degrade in performance. A situation we want to avoid. As a known fact, adding more powerful hardware or scaling vertically will only delay the problem. However, adding multiple servers or scaling horizontally, without a well-thought-through approach, may not reap the benefits to their full extent.


The recipe for creating a highly scalable system in any domain is to use proven software architecture patterns. Software architecture patterns enable us to create cost-effective systems that can handle billions of requests and petabytes of data. The article describes the most basic and popular scalability pattern known as Load Balancing. The concept of Load Balancing is essential for any developer building a high-volume traffic site in the cloud. The article first introduces the Load balancer, then discusses the type of Load balancers, next is load balancing in the cloud, followed by Open-source options and finally a few pointers to choose load balancers.

## What is a Load Balancer?
A load balancer is a traffic manager that distributes incoming client requests across all servers that can process them. The pattern helps us realize the full potential of cloud computing by minimizing the request processing time and maximizing capacity utilization. The traffic manager dispatches the request only to the available servers, and hence, the pattern works well with scalable cloud systems. Whenever a new server is added to the group, the load balancer starts dispatching requests to it and scales up. On the contrary, if a server goes down, the dispatcher redirects requests to other available servers in the group and scales down, which helps us save money.

## Types of Load Balancers
After getting the basics of Load Balancer, the next is to familiarize with the load balancing algorithms. There are broadly 2 types of load balancing algorithms.

### Static Load Balancers
Static load balancers distribute the incoming traffic equally as per the algorithms. 
-	<strong>Round Robin</strong> is the most fundamental and default algorithm to perform load balancing. It distributes the traffic sequentially to a list of servers in a group. The algorithm assumes that the application is stateless and each request from the client can be handled in isolation. Whenever a new request comes in it goes to the next available server in the sequence. As the algorithm is basic, it is not suited for most cases.
-	<strong>Weighted Round Robin</strong> is a variant of round robin where administrators can assign weightage to servers. A server with a higher capacity will receive more traffic than others. The algorithm can address the scenario where a group has servers of varying capacities.
-	<strong>Sticky Session</strong> also known as the Session Affinity algorithm is best suited when all the requests from a client need to be served by a specific server. The algorithm works by identifying the requests coming in from a particular client. The client can be identified either by using the cookies or by the IP address. The algorithm is more efficient in terms of data, memory and using cache but can degrade heavily if a server start getting stuck with excessively long sessions. Moreover, if a server goes down, the session data will be lost.
-	<strong>IP Hash</strong> is another way to route the requests to the same server. The algorithm uses the IP address of the client as a hashing key and dispatches the request based on the key. Another variant of this algorithm uses the request URL to determine the hash key.


### Dynamic Load Balancers
Dynamic load balancers, as the name suggests, consider the current state of each server and dispatch incoming requests accordingly.
-	<strong>Least Connection</strong> dispatches the traffic to the server with the fewest number of connections. The assumption is that all the servers are equal and the server having a minimum number of connections would have the maximum resources available.
-	<strong>Weighted Least Connection</strong> is another variant of least connection. It provides an ability for an administrator to assign weightage to servers with higher capacity so that requests can be distributed based on the capacity.
-	<strong>Least Response Time</strong> considers the response time along with the number of connections. The requests are dispatched to the server with the fewest connections and minimum average response time. The principle is to ensure the best service to the client.
-	<strong>Adaptive or Resource-based</strong> dispatches the load and makes decisions based on the resources i.e., CPU and memory available on the server. A dedicated program or agent runs on each server that measures the available resources on a server. The load balancer queries the agent to decide and allocate the incoming request.


## Load Balancing in Cloud
A successful cloud strategy is to use load balancers with Auto Scaling. Typically, cloud applications are monitored for network traffic, memory consumption and CPU utilization. These metrics and trends can help define the scaling policies to add or remove the application instances dynamically. A load balancer in the cloud considers the dynamic resizing and dispatches the traffic based on available servers. The section below describes a few of the popularly known solutions in the cloud:

### AWS - Elastic Load Balancing (ELB)
[Amazon ELB](https://aws.amazon.com/elasticloadbalancing/){:target="_blank"} is highly available and scalable load balancing solution. It is ideal for applications running in AWS. Below are 4 different choices of Amazon ELB to pick from:
-	[Application Load Balancer](https://aws.amazon.com/elasticloadbalancing/application-load-balancer/){:target="_blank"} used for load balancing of HTTP and HTTPS traffic.
-	[Network Load Balancer](https://aws.amazon.com/elasticloadbalancing/network-load-balancer/){:target="_blank"} is used for load balancing both TCP, UDP and TLS traffic. 
-	[Gateway Load Balancer](https://aws.amazon.com/elasticloadbalancing/gateway-load-balancer/){:target="_blank"} is used to deploy, scale, and manage third-party virtual appliances. 
-	[Classic Load Balancer](https://aws.amazon.com/elasticloadbalancing/classic-load-balancer/){:target="_blank"} is used for load balancing across multiple EC2 instances. 

### GCP – Cloud Load Balancing
[Google Cloud Load Balancing](https://cloud.google.com/load-balancing){:target="_blank"} is a highly performant and scalable offering from Google. It can support up to 1 million+ queries per second. It can be divided into 2 major categories i.e., internal, and external. Each major category is further classified based on the incoming traffic. Below are a few load balancer types.
-	[Internal HTTP(S) Load Balancing](https://cloud.google.com/load-balancing/docs/l7-internal){:target="_blank"}
-	[Internal TCP/UDP Load Balancing](https://cloud.google.com/load-balancing/docs/internal){:target="_blank"}
-	[External HTTP(S) Load Balancing](https://cloud.google.com/load-balancing/docs/https){:target="_blank"}
-	[External TCP/UDP Network Load Balancing](https://cloud.google.com/load-balancing/docs/network){:target="_blank"}

A complete guide to compare all the available load balancers can be found on the [Google load balancer page](https://cloud.google.com/load-balancing/docs/choosing-load-balancer){:target="_blank"}.

### Microsoft Azure Load Balancer
[Microsoft Azure load balancing](https://azure.microsoft.com/en-us/services/load-balancer/){:target="_blank"} solution provides 3 different types of load balancers:
- [Standard Load Balancer](https://docs.microsoft.com/en-us/azure/load-balancer/load-balancer-overview){:target="_blank"} - Public and internal Layer 4 load balancer
- [Gateway Load Balancer](https://learn.microsoft.com/en-us/azure/load-balancer/gateway-overview){:target="_blank"} - High performance and high availability load balancer for third-party Network Virtual Appliances.
- [Basic Load Balancer](https://learn.microsoft.com/en-us/azure/load-balancer/skus){:target="_blank"} - Ideal for small scale application

## Open-Source Load Balancing Solution
Although a default choice is always to use the vendor specific cloud load balancer, there are a few open-source load balancer options available. Below is a couple of those.


### NGINX
NGINX provides [NGINX Plus](https://www.nginx.com/products/nginx/){:target="_blank"} and [NGINX](https://nginx.org/en/){:target="_blank"}, modern load balancing solutions. There are many popular websites including Dropbox, Netflix and Zynga, that are using load balancers from NGINX. The NGINX load balancing solutions are high performance and can help improve the efficiency and reliability of a high traffic website.

### Cloudflare
[Cloudflare](https://www.cloudflare.com/load-balancing/){:target="_blank"} is another popular load balancing solution. It offers different tiers of load balancer to meet specific customer needs. Pricing plans are based on the services, health checks and security provided.
-	Zero Trust platform plans
-	Websites & application services plans
-	Developer platform plans
-	Enterprise plan
-	Network services


## Choosing Load Balancer
It is evident from the sections above that a load balancer can have a big impact on the applications. Thus, picking up the right solution is essential. Below are a few considerations to make the decision.
-	Identifying the short term and long-term goals of a business can help drive the decision. The business requirements should help identify the expected traffic, growth regions and region of the service. Business considerations should also include the level of availability, the necessity of encryptions or any other security concerns that need to be addressed.
-	There are ample options available in the market. Identifying the necessary features for the application can help pick the right solution. As an example, the load balancer should be able to handle the incoming traffic for the application such as HTTP/HTTPS or SSL or TCP. Another example is a load balancer used for internal traffic has different security concerns than external load balancers.
-	Cloud vendors provide various support tiers and pricing plans. A detailed comparison of the total cost of ownership, features and support tier can help identify the right choice for the project. 


Most experts agree that it is a best practice to use a load balancer to manage the traffic which is critical to cloud applications. With the use of a load balancer, applications can serve the requests better and also save costs. 


