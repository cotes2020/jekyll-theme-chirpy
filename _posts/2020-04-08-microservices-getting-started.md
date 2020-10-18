---
title: 'Microservices - Getting Started'
date: 2020-04-08T16:49:59+02:00
author: Wolfgang Ofner
categories: [Microservice]
tags: [docker, high availability, kubernetes, microservice]
---
Many developers have heard about microservices and that it&#8217;s the next great thing. But for many developers I have talked to, microservices is just another buzzword like DevOps. I have been working on different projects using microservices for a bit more than a year and in this post, I want to talk about the theory and the ideas behind the concept. <a href="https://www.programmingwithwolfgang.com/programming-microservices-net-core-3-1/" target="_blank" rel="noopener noreferrer">In my next posts</a>, I will show how to implement a microservice using ASP .NET Core 3.1.

## What is a Microservice?

As the name already suggests, a microservice is very small. The opinions on how small vary. Some say not more than a hundred lines, some say that it should do one thing. My opinion is that a microservice should offer one or more methods in the same context. Take a customer service for example. This service could offer methods to do the registration, login and changing the user&#8217;s password.

For your application, take the microservices you need to compose the wanted functionality. For example, if you have an online shop, you could have microservices for products, search, wishlist, customers, and orders.

### Why are Microservices so popular?

The most important aspect of a microservice is that it works completely independently. This means that a microservice has its own database (or other storage). This is very important because this guarantees that changes in other services won&#8217;t break the microservice. It might sound strange in the beginning that a microservice is a small application itself, especially with its own database but this makes it way easier to change or deploy new features. This is the same principle as KISS (Keep it simple stupid) and SRP (Single Responsibility Principle) in programming. Both principles strive to keep things small and simple.

A big reason why microservices became so popular is the fact that it helps to achieve high availability for your application. To achieve this high availability you must not couple services together and keep the connections loose. This loose connection can be achieved by using message systems like RabbitMQ, Azure Queue or Azure Service Bus. A service sends a message to the queue and the other services can process this message. If a service is offline, the message will stay in the queue and the service can process all messages once it is back online. The downside of this approach is that it leads to a lot more complexity and problems like latency, consistency and debugging. I will talk more about that in the Disadvantages of Microservices section further down.

## Advantages of Microservices

Using a microservice architecture can bring the following advantages:

  * Easy to build and maintain
  * Easy deployment
  * New features can be implemented fast
  * Usage of different technologies
  * Teams can focus and specialize on one domain
  * Better scalability and resource usage

### Microservices are easy to build and maintain

Microservices are supposed to be small. Therefore they can be built, maintained and understood easily. Tests are less complicated and new team members can learn the functionality quicker.

### Microservices can be easily deployed

Since microservices are small without dependencies on other projects, they are easy to deploy. A microservice can often be deployed with a couple of lines in a Dockerfile.

### New features can be implemented fast

As already mentioned, microservices are small and can be understood quicker than monoliths. Therefore it is also easier and faster to implement and deploy new features.

### Usage of different technologies

Another big advantage, especially in bigger projects is that you can use different technologies for each service. The microservices often talk with each other over HTTP or gRPC. Therefore each service can be written in a different language, for example, one service can be .NET Core, one can be Java and the third one can be Go.

### Teams can focus and specialize on one domain

Microservices also help teams to focus on their domain since they don&#8217;t have to care about unrelated services. For example, the team which provides the search service for an online shop only has to focus on features for the search. They don&#8217;t have to care how the shop shows products or processes orders. This also helps the team to specialize in advanced search techniques which can lead to an even better product.

### Better scalability and resource usage

Microservices are small and often run in a Docker container in a Kubernetes cluster. If you have a monolithic online shop and it&#8217;s Black Friday, you have to scale your whole application, no matter if a feature is highly used or not used at all. With a microservice architecture, you can scale the services which are in high demand. For example, if a lot of people are placing orders, you can scale the order service but you don&#8217;t have to scale the customer service. This helps to save resources and therefore decreases the costs of running your application. Another advantage of the small services is that you can place them better on a server which increases the utilization which also helps to reduce the costs.

## Disadvantages of Microservices

It&#8217;s not all great when using a microservice architecture and there are some downsides to it:

  * Increased complexity
  * Latency
  * Data consistency
  * Debugging is harder
  * Cascading of problems to other services
  * Handling of the messages in the queue

### Increased complexity

Earlier I said that microservices are small and easy to understand. But when your application consists of hundreds or thousands of microservices, the application becomes really complex. Especially when the services talk to each other. Dave Hahn gave a great talk about the architecture of Netflix in 2015 and admitted that no one in the company understands the whole architecture.

<div id="attachment_1857" style="width: 710px" class="wp-caption aligncenter">
  <a href="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/04/Netflix-Architecture-in-2015.jpg"><img aria-describedby="caption-attachment-1857" loading="lazy" class="wp-image-1857" src="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/04/Netflix-Architecture-in-2015.jpg" alt="Netflix Microservices Architecture in 2015" width="700" height="501" srcset="https://www.programmingwithwolfgang.com/wp-content/uploads/2020/04/Netflix-Architecture-in-2015.jpg 997w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/04/Netflix-Architecture-in-2015-300x215.jpg 300w, https://www.programmingwithwolfgang.com/wp-content/uploads/2020/04/Netflix-Architecture-in-2015-768x550.jpg 768w" sizes="(max-width: 700px) 100vw, 700px" /></a>
  
  <p id="caption-attachment-1857" class="wp-caption-text">
    Netflix Microservices Architecture in 2015 (<a href="https://www.youtube.com/watch?v=-mL3zT1iIKw" target="_blank" rel="noopener noreferrer">Source Youtube</a>)
  </p>
</div>

### Latency

Every call to another service adds some latency and the user has to wait longer until he or she gets a result. Microservices should only call other services which are really needed and services should be placed as close together as possible.

### Data consistency

Microservices often exchange data asynchronously and also have their own data storage. This can lead to data inconsistency, for example, the customer services updates a customer but the order service hasn&#8217;t updated the customer yet. The data will be synchronized eventually but you can&#8217;t say exactly when.

### Debugging is harder

Debugging can be harder, especially when you have problems that only occur in production. It is essential to have a good monitoring and logging strategy.

### Cascading of problems to other services

A failing service can bring down a lot of other services. For example, if the product service has a problem and times out, every service calling the product service has to wait until it gets an error message back. The service calling the product service might be called by another service and quickly you have a lot of affected services. To prevent your whole system from failing, you should implement a <a href="https://microservices.io/patterns/reliability/circuit-breaker.html" target="_blank" rel="noopener noreferrer">circuit breaker</a>.

### Handling of the messages in the queue

Messages between services are often sent via queue so messages can be handled, even if a service is offline at the time of publishing. The messages in the queue can get problematic if they can&#8217;t be processed. A service takes it, can&#8217;t process it and puts it back on the queue. This costs resources and time. You could set a time to live and remove the message from the queue when the time is reached.

## Conclusion

Today, I tried to give a short overview of the theory of microservices. Microservices have advantages but also disadvantages compared to monolithic applications. In the end, there is no perfect solution and you should think about the requirements of your project and then decide if microservices are the way to go. I would also recommend starting with a small project and gain some experience in running them.

This post was all about microservices in theory, <a href="https://www.programmingwithwolfgang.com/programming-microservices-net-core-3-1/" target="_blank" rel="noopener noreferrer">in my next posts</a>, I will implement two microservices using ASP .NET Core, CQRS, mediator, RabbitMQ and Docker.