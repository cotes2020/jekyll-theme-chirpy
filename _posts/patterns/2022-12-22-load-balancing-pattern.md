---
title: Load Balancing Pattern
description: description
tags: ["cloud", "load balancing", "design", "scaling"]
category: ["architecture", "pattern"]
date: 2022-12-22
permalink: '/patterns/load-balancing/'
counterlink: 'patterns-load-balancing/'
image:
  path: https://raw.githubusercontent.com/Gaur4vGaur/traveller/master/images/patterns/2022-12-22-load-balancing-pattern.jpg
  width: 800
  height: 500
  alt: Load Balancing
---

## Introduction
Any modern website on the internet today receives thousands of hits, if not millions. Without any scalability strategy, the website is either going to crash or significantly degrade in performance. A situation we want to avoid. As a known fact, adding more powerful hardware or scaling vertically will only delay the problem. However, adding multiple servers or scaling horizontally, without a well-thought-through approach, may not reap the benefits to their full extent.


The recipe for creating a highly scalable system in any domain is to use proven software architecture patterns. Software architecture patterns enable us to create cost-effective systems that can handle billions of requests and petabytes of data. The article describes the most basic and popular scalability pattern known as Load Balancing. The concept of Load Balancing is essential for any developer building a high-volume traffic site in the cloud. The article first introduces the Load balancer, then discusses the type of Load balancers, next is load balancing in the cloud, followed by Open-source options and finally a few pointers to choose load balancers.

## What is a Load Balancer?
A load balancer is a traffic manager that distributes incoming client requests across all servers that can process them. The pattern helps us realize the full potential of cloud computing by minimizing the request processing time and maximizing capacity utilization. The traffic manager dispatches the request only to the available servers, and hence, the pattern works well with scalable cloud systems. Whenever a new server is added to the group, the load balancer starts dispatching requests to it and scales up. On the contrary, if a server goes down, the dispatcher redirects requests to other available servers in the group and scales down, which helps us save money.





[APIs](https://en.wikipedia.org/wiki/API){:target="_blank"}






