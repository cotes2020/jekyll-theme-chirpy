---
title: Attending My first Tech Meetup in Bangalore
date: 2023-08-27 02:00:00 +/-0530
categories: [TeChronicles, Meetup]
tags: [tech, chronicle, sre, meetup, blr, bangalore, atlassian, chaos, platformization]     # TAG names should always be lowercase
---

> “The first step towards getting somewhere is to decide you’re not going to stay where you are.” — JP Morgan



Three months after moving to the tech capital of India, It was about time I re-indulge in my devops / tech passion.  I attended a SRE (site reliability engineering) Meetup organized at [Atlassian](https://www.atlassian.com/) Bangalore office.

![Its about time](/assets/img/memes/its_abt_time.jpeg)

Since it was the first time I was going to EGL ([Embassy Golf Links](https://goo.gl/maps/M5XKBNPM4zdLocna8)), I started a little early and consequentially arrived there nearly 15-20 mins early. That gave me a chance to connect with the organizer and an early bird ([Pranav A](https://www.linkedin.com/in/pranav-a-a029921b1/)) such as myself.  Pranav was a 2023 grad with two years of experience.  He is also an instructor on [whizlabs](https://www.whizlabs.com/), having created a course on AWS.

The event started with an informal meet-and-greet, followed by a more formal self-introduction routine, where we gave a brief of who we are and why we were here. The majority of the congregation was made of techies in the early walks of their career, and also few experienced folks.  We then swiftly transitioned into the First talk of the session. 

## Platformization

The first talk of the day was delivered by [Bhagvatula Nipun](https://www.linkedin.com/in/bhagvatulanipun/), A senior Devops (InfraPlatform) Engineer at Groww.  He talked about the concept of platformization and how it made the lives of people easier. It resonated with my experience as an ex-ML-Platform Engineer.  GitOps and declarative provisioning is their game. Nipun's Team uses Git to manage the configs and a suite of tools (terraform to provision, GH workflows in case of triggers and Argo for CI/CD.). New-relic observability also plays an important role in their dat-to-days. Grafana Tempo is used as Span and Trace backend, while Grafana Loki for log aggregation.  The metrics are exported via open-telemetry[^otel].  [Growthbook](https://www.growthbook.io/) is used for feature flagging.  Now this is a soiree.

![just a few devops tools](/assets/img/memes/lgoos_sre_8-23.jpg)
_The SRE starter pack_

Groww is an online broker, that makes investing/trading easier.  These services see voluminous traffic during trading hours and very little traffic otherwise.  The infra is similarly expected to scale on schedule, during market hours.  Service owners write the scaling schedule as the part of the spec, which on deployment becomes a kubernetes CRD and a (custom built) kubernetes controller reads the CRD and updates the deployment's HPA config, effectively scaling the service on a schedule.

Additionally, they have servers in multiple AZs, even across India. The easiness of deploying stacks in multiple AZs using IAC (infrastructure as code, ex: Terraform) was also emphasized.  He also recounted a few challenges faced in the adaption of the tool, and why it was still necessary form them to maintain the legacy processes. This was followed by a Q&A session and a short break.  
<!-- (Note to self, I really should read more about multi-AZ deployments and reliability + consistency guarantees for the same ) -->

Atlassian was generous enough to make its cafeteria available to us, and I had a Soy Chocolate milk during this short commercial break.


## Chaos Engineering at Atlassian

[Chethan](https://www.linkedin.com/in/chethan-c-94639178/), a seasoned developer who used to work at Nokia, VMware and currently working at Atlassian, took the stage for his talk on Chaos Engineering.  I was familiar with a what chaos engineering was, thanks to my peers at one of my employer (OYO).

> Chaos engineering is a method of testing distributed software that deliberately introduces failure and faulty scenarios to verify its resilience in the face of random disruptions. These disruptions can cause applications to respond unpredictably and break under pressure. Chaos engineers ask why. [^ref1] 


Chethan's team at Atlassian had built a platform to do exactly that. The platform takes as input a config file, defined by the service owners, which has a bunch of test suites to disrupt the service.  Chethan's team had also built a library of disruptions that can be included in the test suite.  Then, the platform will use the information to disrupt the service, at random times.  The service owner doesn't (and shouldn't) know if the error was deliberated by the Chaos Platform or if it was a legitimate issue. So, the service team will address this disruption as if it is a severity use-case.  

![Chaos is'nt a pit. its a ladder](/assets/img/memes/chaos_GOT_pit_ladder.jpg)
_But only for people working on Chaos Engineering_


Before the Chaos platform's introduction, various team had their own approach to chaos and usually performed a few experiments (manufactured disruptions) on their respective game-days to assure themselves of their resilience.  This involved significant manpower to plan, implement the disruptions,analyze and fix the same.  We would rather have developers do what they are good at, developing products / features and generate value to the customer. Unification of the effort and reducing the overhead to manufacture and run experiments seem like an easy way to ensure the time is well spent.  The platform at Atlassian is available to run experiments, but fixing of the services is rightfully, the responsibility of the service owners.   



When Chethan's team had started building the platform, they had explored multiple tools to define and run their experiments (manufactured disruptions). Gremlin, Mangle and chaos-toolkit were in the running, to name a few.  [Gremlin](https://www.gremlin.com/) has an [article](https://www.gremlin.com/community/tutorials/chaos-engineering-tools-comparison/) highlighting simple pros and cons of various tools, but it is always good to evaluate one's own use-cases through these lenses yourself.  They settled on  Chaos toolkit for the extensibility it provides.  


## Experiments

Atlassian's Chaos platform has a wide suite of experiments to run, starting from latency of upstream, downstream sources and destinations, deleting nodes, resource throttling, blackholes, I/O latency, packet losses, DNS resolution errors, certificate expiry etc.  

## Running experiments

![you made the mess, I clean it](/assets/img/memes/clean_up_dog_mess.png)
_Chaos engineers to Service Owners_

They run their experiments on their staging cluster.  Staging clusters' service layout is usually a clone of the Production service layout, which reduced capacity and resource allocation, usually for the sake of cost.  I personally know quite a few companies who strive really hard to maintain their staging stack as a production clone in a cost effective way. (for non-critical services, pre-prod is simply too expensive.)  Even though running on staging cluster might not directly yield the expected resilience on the prod cluster, it does forces the service owners to think about the possibility of failure, which I would call a win.  

Atlassian sometimes also uses load-generation tools at their disposal to generate traffic, but the load itself is something the service owners have to define and configure.Atlassian also has tools for continuous verification, which ensures new deployments don't break the upstream or downstream services / sources. Atlassian's experiments library also includes AZ (availability zone) failures and automated disaster recovery.


## Challenges

The main challenges to platforms are their adaption.  Onboarding a service onto a new platform, and claiming they are resilient to chaos attacks requires **effort**. New platforms have their own **learning curves** as well.  Analyzing and fixing manufactured disruptions is an additional overhead for most teams. Atlassian's tech management has compliance in place for high priority services to be resilient, and thus have regular (quarterly) game days for the same. 

![Use it. Use the Platform](/assets/img/memes/le_platform_use_it.jpg)
_Chaos engineers to Service Owners_

Another concern that arises during enforcing compliance is the relevance of experiments. The common subset of experiments relevant to all services is small.  Any centralized platform team would find it near impossible to analyze each service and curate a experiment suite.  This is again left to the service owners.  The service owners would be able to make a decision only if they have knowledge of the experiments and Atlassian's Chaos team regularly conducts sessions in various forums to bridge the gap. 


If you are a fellow Bengalurian and are interested ti delve deeper into chaos Engineering do join the [meetup group](https://www.meetup.com/chaos-engineering-meetup-group/)


## Time to leave..

After a very engaging networking session and a very sumptuous meal, courtesy Atlassian, we bid our farewells.  Half a day well spent indeed. 

![group pic. Where's Waldo?](/assets/img/memories/highres_515396783.webp)
_Where's Waldo?_

## Footnotes & references
[^otel]:  It provides a collection of tools, APIs, and SDKs for capturing metrics, distributed traces and logs from applications.

[^ref1]: https://www.dynatrace.com/news/blog/what-is-chaos-engineering/
