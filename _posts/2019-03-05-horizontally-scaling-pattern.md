---
title: Horizontally Scaling Pattern
date: 2019-03-05T09:18:50+01:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [Architecture Pattern, Cloud, Scaling]
---
In <a href="/vertically-scaling-pattern/" target="_blank" rel="noopener noreferrer">my last post</a>, I talked about vertically scaling vs. horizontally scaling with a focus on vertically scaling. Today, I will talk about it again, but I will focus on horizontally scaling this time.

## Why use Scaling?

Surveys say the faster a website is, the higher its generated revenue. Therefore scaling is a business concern. Google observed that a 500 millisecond longer response time on a website reduces the traffic by 20%. Amazon claims that a 100 milliseconds delay costs them 1% revenue. Microsoft reported that if an engineer can increase the speed of their search platform Bing by 20 milliseconds, he or she increased the revenue by his yearly salary.

Additionally, to the revenue loss of a slow website, Google is also including the page speed in its index algorithm. The slower your website is, the lower your ranking will be.

## Vertically Scaling Up vs. Horizontally Scaling out

Vertically scaling means increasing the performance by increases the power of the hardware. This can be a faster CPU, more RAM or using SSDs instead of HDDs. This was a common approach in the past. If your computer was too slow, replace it with a faster one and the problem is solved. The problem with vertically scaling is that the available hardware is limited, therefore the scaling possibilities are limited.

Horizontally scaling increases the performance of the application by adding more computing nodes. This means instead of a single web server you have multiple web server delivering your website to your visitors. The requests are handled by a load balancer which spreads the requests evenly to all server. The advantage of this approach is that you are not limited by hardware. If you need more performance, add more computing nodes. Horizontally scaling got popular with the rise of cloud providers like Azure or AWS.

The advantage of horizontally scaling is that you can use several cheap new servers to handle the workload whereas a high-end CPU or SSD might cost way more than 10 average server, which still can provide more performance.

## Describing Scalability Goals

Scalability goals help to formulate the requirements for an SLA. These could be being able to serve 100 concurrent users or that the response time is always below two seconds. Another requirement could be that the infrastructure automatically scales up and down with the goal that the CPU usage is always between 50 and 70 percent. The goals depend on the kind of application and its criticality.

## Limitations

Scaling horizontally is almost limitless, especially when you are using a cloud provider. Your architecture changes if you are using a horizontally scaling approach. To distribute the workload to your servers you will need one or more load balancer and you also have to figure out a way to handle the session if you operate a website since a user can be processed by one server his or her next request is processed by a different server.

## Scaling Advantages of Cloud Solutions

As already previously mentioned, the scaling capabilities in the cloud are almost endless. You might not be able to scale immediately if you need an immense amount of performance like Facebook or Google would need but you will get the needed extra hardware provided by your cloud provider. Cloud provider like Azure also offer load balancer to distribute the workload to different data centers around the world, depending on the location of the user and also can distribute the workload within a data center.

Handling the user session is also very simple with Azure. You can either use sticky session which configures the load balancer to send a request from a user always to the same server or even better, you can save the session in a database or more preferably in a faster cache server like Redis. Either option can be configured within minutes.

One of the biggest advantages with cloud providers is that you can easily scale down if you don&#8217;t need the extra performance anymore and therefore don&#8217;t have to pay for it. You can turn off your on-premise server but you still have the storage or initial investment costs.

## Conclusion

This post gave a quick introduction into vertical scaling, compared it with horizontal scaling and discussed the differences between scaling your on-premise server versus scaling a server in the cloud. If you want to learn more about cloud architecture patterns, I can recommend the book &#8220;[Cloud Architecture Pattern](https://www.oreilly.com/library/view/cloud-architecture-patterns/9781449357979/)&#8221; from O&#8217;Reilly.