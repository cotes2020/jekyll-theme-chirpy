---
title: Designing High Cohesion Microservice
description: Designing a highly cohesive and self aggregated microservice. Designing a highly cohesive microservices. The article discussed couple of examples of live and well-defined self aggregated microservices
tags: ["microservices", "design"]
category: ["architecture", "microservices"]
date: 2017-08-05
permalink: '/microservices/designing-high-cohesion-microservices/'
counterlink: 'microservices-designing-high-cohesion-microservices/'
---

A microservice must have __high cohesion__. Microservice should focus on single functionality and should be consistent in terms of functionality, input and outputs. Consider it as a self-contained unit managing its own lifecycle.

![Designing High Cohesive Microservice](https://raw.githubusercontent.com/Gaur4vGaur/traveller/master/images/microservices/2017-08-05-designing-high-cohesion-microservices.png)

Let us refer to <a href="https://github.com/hmrc/email-verification" target="_blank">hmrc/email-verification</a> as an example. The service has a single focus on verifying the emails. The microservice verifies the email address by firing the verification link to the provided email address. This service encapsulates all the logic/functionality of email verification. The microservice has only one reason to change. The service will be modified only if it needs to be upgraded or if there is a change in the business logic to verify the email.

The idea of the Single Responsibility Principle is adopted from the object-oriented world. The principle of the service to be focused on specific business reason and to be highly cohesive. As the microservice is cohesive it is __scalable, flexible and reliable__.

The first of designing a microservice is to identify its focussed use case. We can determine the responsibility of the microservice either: by business requirements for eg email-verification needs to verify the email, or by a domain, where the focus is to create, update, fetch or delete the data.
Let us refer to another service <a href="https://github.com/hmrc/tamc" target="_blank">hmrc/tamc</a>. The service allows a married couple to check if they are eligible for Marriage Allowance. The service concentrates on creating/updating/listing the relationships. The service requires the changes when government rules regarding the marriage allowance change for eg allowances going up or down. The microservice deals only with the calculation of marriage allowance, however, there is a separate microservice written to display marriage allowance data to users.

This enables the marriage allowance team to enhance and deploy specific parts of the system.
