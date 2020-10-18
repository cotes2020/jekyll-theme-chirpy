---
title: Vertically Scaling Pattern
date: 2019-02-28T19:40:33+01:00
author: Wolfgang Ofner
categories: [Cloud]
tags: [Architecture Pattern, Cloud, Scaling]
---
Scalability of a system is a measure of the number of users it can handle at the same time. For a web server, for example, would this mean how many concurrent users it can serve. Serving more users can be achieved by vertically scaling up or horizontally scaling out. In this post, I will explain how vertically scaling up work. If you want to learn more about horizontally scaling out, you can read about it [here](https://www.programmingwithwolfgang.com/horizontally-scaling-pattern/).

## Why use Scaling?

Surveys say the faster a website is, the higher its generated revenue. Therefore scaling is a business concern. Google observed that a 500 millisecond longer response time on a website reduces the traffic by 20%. Amazon claims that a 100 milliseconds delay costs them 1% revenue. Microsoft reported that if an engineer can increase the speed of their search platform Bing by 20 milliseconds, he or she increased the revenue by his yearly salary.

Additionally, to the revenue loss of a slow website, Google is also including the page speed in its index algorithm. The slower your website is, the lower your ranking will be.

## Vertically Scaling Up vs. Horizontally Scaling out

Vertically scaling means increasing the performance by increases the power of the hardware. This can be a faster CPU, more RAM or using SSDs instead of HDDs. This was a common approach in the past. If your computer was too slow, replace it with a faster one and the problem is solved. The problem with vertically scaling is that the available hardware is limited, therefore the scaling possibilities are limited.

Horizontally scaling increases the performance of the application by adding more computing nodes. This means instead of a single web server you have multiple web server delivering your website to your visitors. The requests are handled by a load balancer which spreads the requests evenly to all server. The advantage of this approach is that you are not limited by hardware. If you need more performance, add more computing nodes. Horizontally scaling got popular with the rise of cloud providers like Azure or AWS.

## Describing Scalability Goals

Scalability goals help to formulate the requirements for an SLA. These could be being able to serve 100 concurrent users or that the response time is always below two seconds. Another requirement could be that the infrastructure automatically scales up and down with the goal that the CPU usage is always between 50 and 70 percent. The goals depend on the kind of application and its criticality.

## Limitations

Scaling vertically is limited by the available hardware. Modern CPUs have up to 24 cores and you can maybe stack a terabyte of ram into your server. No matter how powerful your server is, it will always too little to handle all the users of Google or Facebook.

Sometimes you have only certain peak times, like Black Friday or Christmas and the server is idle for the rest of the year. This is were hosting your application in the cloud comes in handy since it enables you to scale down to decrease the costs of your infrastructure when you don&#8217;t need it.

## Scaling Advantages of Cloud Solutions

Scaling a cloud-native application is as easy is it possible could be. Cloud provider like Azure provides you with the option to automatically scale your server to a higher grade if a specific event occurred. This could be the CPU or RAM usage was higher than 80% for five minutes. The big advantage of applications in the cloud is that you only have to do some mouse clicks and you have a more powerful server. To scale your on-premise server, you have to order new hardware and install it or move your whole application to a more powerful server. This also means that you have the investment costs.

## Conclusion

This post gave a quick introduction into horizontal scaling, compared it with vertical scaling and discussed the differences between scaling your on-premise server versus scaling a server in the cloud. If you want to learn more about cloud architecture patterns, I can recommend the book &#8220;[Cloud Architecture Pattern](https://www.oreilly.com/library/view/cloud-architecture-patterns/9781449357979/)&#8221; from O&#8217;Reilly.