---
title: Designing Business Centric Microservice
description: Designing a domain specific and business focussed microservice. Designing a domain specific microservices. The article discussed couple of examples of live and well-defined business centric microservices
tags: ["microservices", "design"]
category: ["architecture", "microservices"]
date: 2017-06-16
permalink: '/microservices/designing-business-centric-microservices/'
counterlink: 'microservices-designing-business-centric-microservices/'
---


A microservice should be business-centric or domain-centric. It should be designed to represent a business function. A business-centric service helps to scope the service and to control the size. The idea is borrowed from the domain-driven design where the primary focus is on core domain logic with a bounded context. For eg HMRC is a huge organization and it needs to send many updates to all taxpayers across the UK. Hence, HMRC needs email functionality. It would be sensible to have a service that can handle all email related functionalities such as verifying an email, drafting an email for updates, maintaining templates based on users, etc. Thus, we can say that email itself is a business function. So HMRC created microservice to address the same. The code for the service can be found at <a href="https://github.com/hmrc/email-verification" target="_blank">hmrc/email-verification</a>.

![Designing Business Centric Microservice](https://raw.githubusercontent.com/Gaur4vGaur/traveller/master/images/microservices/2017-06-16-designing-business-centric-microservices.png)

The first step towards designing a microservice is to coarsely analyze a  business domain. The business domain is then further split to represent separate business functions or bounded context. Every bounded context can then be made as a microservice. These extracted microservices can interact with each other using well-defined APIs.

As the various microservices will be in constant interaction, we need to think about the inputs and outputs for each microservice. The microservices are sometimes designed based on technical boundaries for eg a common piece of functionality is extracted as a separate service as the same calculation is used by various modules, or a microservice is designed to access common reference data for all other microservices.

As an example let us have a look at <a href="https://github.com/hmrc/gmp-frontend" target="_blank">hmrc/gmp-frontend</a> and <a href="https://github.com/hmrc/gmp" target="_blank">hmrc/gmp</a> services. The services are used to calculate Guaranteed Minimum Pension calculation for a member of a contracted-out scheme under certain circumstances. These services are split into two business functions, where gmp-frontend takes care of all the user interactions and gmp handles all the calculations. The two services constantly talk to each other to serve the users.


