---
title: Creating a Microservice with .NET Core 3.1
date: 2020-07-21T18:53:20+02:00
author: Wolfgang Ofner
categories: [ASP.NET]
tags: [NET Core 3.1, 'C#', CQRS, MediatR, microservice, Swagger, xUnit]
---
Using a microservice architecture is a topic that is present at every conference for quite some time already. In my work as a consultant, I often have to train development teams about the basics of microservices, why to use them, do&#8217;s and don&#8217;ts, and best practices. Therefore, I want to start a new series where I will create a microservice and the following parts deploy it to Kubernetes using AKS, implementing CI/CD pipelines using Azure DevOps, using Helm charts and automated unit tests including code coverage.

## What is a Microservice?

A microservice is an independently deployable service that is modeled around a business domain. An application uses many microservices that work together. A simple example would be an online shop. Potential microservices could be for customers, orders, products, and the search.

As the name already suggests, a microservice is very small. The opinions on how small vary. Some say not more than a hundred lines, some say that it should do one thing. My opinion is that a microservice should something in the same context. This can also be several methods. Take a customer service for example. This service could offer methods to do the registration, login, and changing the user&#8217;s password.

For more details on microservices, I recommend my post &#8220;<a href="/microservices-getting-started/" target="_blank" rel="noopener noreferrer">Microservices &#8211; Getting Started</a>&#8220;.

## Creating your first Microservice

You can find the code of the demo on <a href="https://github.com/WolfgangOfner/.NetCoreMicroserviceCiCdAks/tree/createMicroservice" target="_blank" rel="noopener noreferrer">Github</a> on the createMicroservice branch.

A microservice is an application that offers operations for a specific context. In my example, the application offers operations to read, create, and update customers. To keep it simple, I use an in-memory database. A bit special is that I am using CQRS and MediatR to read and write data. You can find a detailed description of the application in my post &#8220;<a href="/programming-microservices-net-core-3-1" target="_blank" rel="noopener noreferrer">Programming Microservices with .NET Core 3.1</a>&#8220;.

When you start the application, you will see the Swagger UI and can also try all the available methods.

<div class="col-12 col-sm-10 aligncenter">
  <a href="/assets/img/posts/2020/07/Swagger-UI-of-the-Microservice.jpg"><img loading="lazy" src="/assets/img/posts/2020/07/Swagger-UI-of-the-Microservice.jpg" alt="Swagger UI of the Microservice" /></a>
  
  <p>
    Swagger UI of the Microservice
  </p>
</div>

## Conclusion

This post was a short introduction into the microservice that I will use in the following posts to create automatic builds, deployments to Kubernetes, DevOps workflows, and much more. Check out the <a href="/build-net-core-in-a-ci-pipeline-in-azure-devops/" target="_blank" rel="noopener noreferrer">next post of this series</a> where I create a CI pipeline in Azure DevOps to build the .NET Core solution and run all unit tests.

You can find the code of the demo on <a href="https://github.com/WolfgangOfner/.NetCoreMicroserviceCiCdAks/tree/createMicroservice" target="_blank" rel="noopener noreferrer">Github</a> on the createMicroservice branch.