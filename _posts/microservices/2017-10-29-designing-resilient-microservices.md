---
title: Designing Resilient Microservices
description: Resilience is must. Designing resilient microservices is must.
tags: ["microservices", "design"]
category: ["architecture", "microservices"]
date: 2017-10-29
permalink: '/microservices/designing-resilient-microservices/'
counterlink: 'microservices-designing-resilient-microservices/'
---

Resilience is an essential consideration while designing microservices. A microservice should embrace failure, whenever it happens. A microservice should be designed for all known failures. A few of the common known failures are:

* Timeout from downstream microservice, or a third party system failed to respond
* An exception occurred within a microservice or one of the downstream
* There can be network delays or network failure etc.

![Designing Automated Microservice](https://raw.githubusercontent.com/Gaur4vGaur/traveller/master/images/microservices/2017-10-29-designing-resilient-microservices.png)

Whatever is the type of failure, a well-designed microservice needs to embrace the failure. There are many ways in which a microservice can handle failures:

* microservice can fallback to default functionality. For eg consider a `suggestion` service for an e-commerce website that provides suggestions on the side pane based on the user’s past searches and order history. The `suggestion` service analyses the user data and feeds the side pane. If the `suggestion` service is down, then an alternative is to provide the most popular products from the `global suggestion` service rather than an error message on the side pane.
* another option is to degrade the functionality or direct users with alternatives. For eg, consider a microservice that aggregate user information from different sources. If one of the sources is getting overwhelmed with requests, one of the strategies could be to terminate all the requests to the failing source, providing it time to recover.  However, microservice can continue to gather data from other sources. The application can render incomplete information to users with an error message to come back for full details. Thus, avoiding complete dissatisfaction.

Another way to make whole distributed systems, having multiple instances of microservices, resilient is by making the failed microservice(if it cannot be recovered) deregister itself, so the system is only aware of fully functioning microservices.

We can observe all these errors/exceptions, using logging and monitoring tools, to make the service more resilient.