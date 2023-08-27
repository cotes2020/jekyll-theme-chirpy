---
title: My first Tech Meetup in Bangalore
date: 2023-08-27 02:00:00 +/-0530
categories: [TeChronicles, Meetup]
tags: [tech, chronicle, sre, meetup, blr, bangalore, atlassian, chaos, platformization]     # TAG names should always be lowercase
---

> “The first step towards getting somewhere is to decide you’re not going to stay where you are.” — JP Morgan



Three months after moving to the tech capital of India, It was about time I re-indulge in my devops / tech passion.  I attended a SRE (site reliability engineering) Meetup organized at [Atlassian](https://www.atlassian.com/) Bangalore office.

Since it was the first time I was going to EGL ([Embassy Golf Links](https://goo.gl/maps/M5XKBNPM4zdLocna8)), I started a little early and consequentially arrived there nearly 15-20 mins early. That gave me a chance to connect with the organizer and an early bird ([Pranav A](https://www.linkedin.com/in/pranav-a-a029921b1/)) such as myself.  Pranav was a 2023 grad with two years of experience.  He is also an instructor on [whizlabs](https://www.whizlabs.com/), having created a course on AWS.

The event started with an informal meet-and-greet, followed by a more formal self-introduction routine, where we gave a brief of who we are and why we were here. The majority of the congregation was made of techies in the early walks of their career, and also few experienced folks.  We then swiftly transitioned into the First talk of the session. 

The first talk of the day was delivered by [Bhagvatula Nipun](https://www.linkedin.com/in/bhagvatulanipun/), A senior Devops (InfraPlatform) Engineer at Groww.  He talked about the concept of platformization and how it made the lives of people easier. It resonated with my experience as an ex-ML-Platform Engineer.  GitOps and declarative provisioning is their game. Nipun's Team uses Git to manage the configs and a suite of tools (terraform to provision, GH workflows in case of triggers and Argo for CI/CD.). New-relic observability also plays an important role in their dat-to-days. Grafana Tempo is used as Span and Trace backend, while Grafana Loki for log aggregation.  The metrics are exported via open-telemetry[^otel].  [Growthbook](https://www.growthbook.io/) is used for feature flagging.  

Groww is an online broker, that makes investing/trading easier.  These services see voluminous traffic during trading hours and very little traffic otherwise.  The infra is similarly expected to scale on schedule, during market hours.  Service owners write the scaling schedule as the part of the spec, which on deployment becomes a kubernetes CRD and a (custom built) kubernetes controller reads the CRD and updates the deployment's HPA config, effectively scaling the service on a schedule.

Additionally, they have servers in multiple AZs, even across India. The easiness of deploying stacks in multiple AZs using IAC (infrastructure as code, ex: Terraform) was also emphasized.  This was followed by a Q&A session and a short break.  
<!-- (Note to self, I really should read more about multi-AZ deployments and reliability + consistency guarantees for the same ) -->

Atlassian was generous enough to make its cafeteria available to us, and I had a Soy Chocolate milk during this short commercial break.








## Footnotes
[^otel]:  It provides a collection of tools, APIs, and SDKs for capturing metrics, distributed traces and logs from applications.
