---
title: Introduction to Microservices
description: What are Microservices? A quick introduction to microservices architecture 
tags: ["microservices", "design"]
category: ["architecture", "microservices"]
date: 2017-05-21
permalink: '/microservices/introduction-to-microservices/'
counterlink: 'microservices-introduction-to-microservices/'
---

Microservices are gaining popularity everywhere. The digital platform that I am currently working on is built on Microservices architecture. Microservices refers to an architectural approach that is intended to decompose application into finely grained, highly cohesive and loosely coupled services. These microservices are language-agnostic and platform-agnostic, enabling different systems to talk to each other.

The microservice architecture can be used to break down a big monolithic application into smaller, simpler and manageable services. These smaller services can be then managed by cross-functional teams, without stepping on each other's toes. The respective teams can then deliver the functionality based on their velocity and on their business needs.

![Microservice Architecture](https://raw.githubusercontent.com/Gaur4vGaur/traveller/master/images/microservices/2017-05-21-introduction-to-microservices.png)

The microservice architecture is best suited for a complex application that has potential to grow into a large code base over a period of time. It is sensible to initially have a large system with modules, that may decompose into microservices, when operations need it. Microservice architecture goes hand in hand with Agile and DevOps culture.

A platform may only start with a single functionality of collating data from different sources and serving to the customers. But soon Product Owners can decide to extend and provide more functionalities such as inserting/updating/deleting the records, subscribe for new updates, reconciliation services etc. It provides business flexibility to include different modules without impacting the existing functionalities on the platform.

Microservices enable a team to have quick turnaround and continuous deliveries. However, these benefits come at a cost of increased complexity. Before choosing microservices, teams must justify the benefits against cost of complexity.



