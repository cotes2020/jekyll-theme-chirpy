---
title: Designing Autonomous Microservice
description: Microservices must be self-sufficient and self-governing. Designing an autonomous microservices. The article discusses why and how to design self-sufficient and self-governing services
tags: ["microservices", "design"]
category: ["architecture", "microservices"]
date: 2017-09-28
permalink: '/microservices/designing-autonomous-microservices/'
counterlink: 'microservices-designing-autonomous-microservices/'
---
    
A microservice must be __autonomous__. A microservice interacts with many other microservices or external systems to serve the user. But, any change in external systems should not force a change in the microservice. Similarly, any change in the microservice should not impact other microservices. All the microservices must adhere to the contracts and interfaces to their clients. Clearly defined contracts between services enable multiple teams to develop microservices in parallel and to deploy them independently.

![Designing Observable Microservice](https://raw.githubusercontent.com/Gaur4vGaur/traveller/master/images/microservices/2017-09-28-designing-autonomous-microservices.png)

One way to make a microservice autonomous, i.e., independently deployable and changeable, is by making them loosely coupled. The microservice should interact with other microservices over the network and should be stateless. We can use Open Communication Protocols such as REST over HTTP and architecture components such as message brokers, queues, a publish-subscribe mechanism to minimize the dependency between the services. The standard formats such as XMLs and JSONs can help to standardize the communication between the services. Teams should also avoid using shared resources such as shared libraries and shared databases.

Another way to make microservices autonomous is by introducing versioning. The systems that we are working on today are increasing in size and complexity. It is not possible to upgrade each part of the system in a single release. As a result, there are scenarios when one of the microservice is upgraded while its client is still waiting for business decisions. In such scenarios, we need to make sure that even if there are changes in microservices, then those changes should be backwards compatible. The team working on a microservice should avoid making any breaking changes. One way to deal with this problem is to introduce versions. The clients that have upgraded themselves can use later versions while others can keep using the older version of the microservice. This allows consumers to slowly migrate over time from the old version to the new version. You may have to deploy both old and new version of the microservice. Another way is to include both old endpoints and new endpoints in the service. These old points are no more than a wrapper around the new endpoints. This will prevent breaking any existing contract. However, over a while, all the clients should move to new endpoints.
