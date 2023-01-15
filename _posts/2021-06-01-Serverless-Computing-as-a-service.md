---
title: Serverless Computing or Function as a Service(FaaS) - Cloud Computing
author: Dhruv Doshi
date: 2021-06-01 11:33:00 +0800
categories: [Cloud Computing]
tags: [Cloud Computing]
math: true
mermaid: true
# image:
#   path: /blogs/Blockchain.jpg
#   width: 800
#   height: 500
#   alt: Representation of Blockchain through Image.
  
---

**`FAAS - Function as a Service`**<br>

Function-as-a-Service (FaaS) is a serverless way to execute modular pieces of code on the edge. FaaS lets developers write and update a piece of code on the fly, which can then be executed in response to an event, such as a user clicking on an element in a web application. This makes it easy to scale code and is a cost-efficient way to implement microservices.

*`How FAAS works?`*

FaaS gives developers an abstraction for running web applications in response to events, without managing servers. For example, uploading a file could trigger custom code that transcodes the file into a variety of formats.

FaaS infrastructure is usually metered on-demand by the service provider, primarily through an event-driven execution model, so it’s there when you need it but it doesn’t require any server processes to be running constantly in the background, like platform-as-a-service (PaaS) would. 

Modern PaaS solutions offer serverless capabilities as part of common workflows that developers can use to deploy applications, blurring the lines between PaaS and FaaS. 

In reality, entire applications will be composed of a mix of these solutions: functions, microservices, and long running services

*What are Microservices ?`*

If a web application were a work of visual art, using microservice architecture would be like making the art out of a collection of mosaic tiles. The artist can easily add, replace, and repair one tile at a time. Monolithic architecture would be like painting the entire work on a single piece of canvas.

<center><img src="https://i.imgur.com/4e84fm8.png" style="height:40%; width:80%;"></center><br>

This approach of building an application out of a set of modular components is known as microservice architecture. Dividing an application into microservices is appealing to developers because it means they can create and modify small pieces of code which can be easily implemented into their codebases. This is in contrast to monolithic architecture, in which all the code is interwoven into one large system. With large monolithic systems, even a minor changes to the application requires a hefty deploy process. FaaS eliminates this deploy complexity.

Using serverless code like FaaS, web developers can focus on writing application code, while the serverless provider takes care of server allocation and backend services.

*`Popular FAAS providers`<br>*

1. <a href="https://cloud.ibm.com/functions/">IBM Cloud Functions</a>
2. <a href="https://aws.amazon.com/lambda/">Amazon AWS Lambda</a>
3. <a href="https://cloud.google.com/functions">Google Cloud Function</a> 
4. <a href="https://docs.microsoft.com/en-us/azure/azure-functions/">Microsoft Azure function</a> 
5. <a href="https://www.openfaas.com/">OpenFaaS (OpenSource)</a>

*`Advantages of FAAS`<br>*
Taken in consideration there are many points which makes this model best for the cloud computing. Those are listed below.

1. Improved developer velocity
2. Builtin scalablity
3. Cost efficiency


As listed, MBAAS is better in all the terms compared to all other traditional models and all comparative models like PAAS and IAAS.


*`Disadvantages of FAAS`<br>*

1. Less System Control
2. More difficult to test the system


