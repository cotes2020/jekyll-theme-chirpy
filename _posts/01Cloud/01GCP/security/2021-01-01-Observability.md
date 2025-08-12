---
title: GCP - Gcloud
date: 2021-01-01 11:11:11 -0400
categories: [01GCP, Security]
tags: [GCP]
toc: true
image:
---

# Google Cloud Observability

- [Google Cloud Observability](#google-cloud-observability)
  - [Overview](#overview)
  - [Logging](#logging)
  - [Monitoring](#monitoring)
    - [error](#error)
  - [GCP service](#gcp-service)
    - [Logs Explorer](#logs-explorer)
    - [Error reporting](#error-reporting)
  - [Cloud profiler](#cloud-profiler)
  - [Cloud trace](#cloud-trace)
  - [Application Performance Management](#application-performance-management)

ref:
- [Logging and Monitoring in Google Cloud]()

---

## Overview

Application Performance Management

![alt text](./assets/img/post/mcl6grha83.png)

Visibility into system health: Users want to understand what is happening with their application and system. They rely on a service that provides a clear mental model for how their application is working on Google Cloud. They need a report on the overall health of systems. The services should help answer questions such as “are my systems functioning?” or “”do my systems have sufficient resources available?”

Error reporting and alerting: Users want to monitor their service at a glance through healthy/unhealthy status icons or red/green indicators. Customers appreciate any proactive alerting, anomaly detection, or guidance on issues. Ideally, they want to avoid connecting the dots themselves.

Efficient troubleshooting: Users don’t want multiple tabs open. They need a system that can proactively correlate relevant signals and make it easy to search across different data sources, like logs and metrics. If possible, the service needs to be opinionated about the potential cause of the issue and recommend a meaningful direction for the customer to start their investigation. It should allow users to immediately act on what they discover.

Performance improvement: Users need a service that can perform retrospective analysis. Generally, help them plan intelligently by analyzing trends and understand how changes in the system affect its performance

![alt text](./assets/img/post/mcl6grhb23.png)


---

## Logging

![alt text](./assets/img/post/mcl6grhc72.png)

![alt text](./assets/img/post/mcl6grhc09.png)

![alt text](./assets/img/post/mcl6grhc22.png)

![alt text](./assets/img/post/mcl6grhc11.png)

![alt text](./assets/img/post/mcl6grhd54.png)

---

## Monitoring

![alt text](./assets/img/post/mcl6grhd68.png)


![alt text](./assets/img/post/mcl6grhd26.png)

- Great products also need thorough testing, preferably automated testing, and a refined continuous integration/continuous development (CI/CD) release pipeline.

![alt text](./assets/img/post/mcl6grhd87.png)

![alt text](./assets/img/post/mcl6grhd69.png)


![alt text](./assets/img/post/mcl6grhe88.png)

![alt text](./assets/img/post/mcl6grhe68.png)


![alt text](./assets/img/post/mcl6grhf84.png)

- measure a system’s performance and reliability
- latency, traffic, saturation, and errors.

![alt text](./assets/img/post/mcl6grhf61.png)

- Changes in latency could indicate emerging issues. Its values may be tied to capacity demands.
- `measure system improvements`.
- Sample latency metrics include:
  - page load latency,
  - number of requests waiting for a thread,
  - query duration,
  - service response time,
  - transaction duration,
  - time to first response and time to complete data return.

![alt text](./assets/img/post/mcl6grhf21.png)

- The next signal is traffic, which measures how many requests are reaching your system.
- it’s an indicator of current system demand. Its historical trends are used for capacity planning. - It’s a core measure when `calculating infrastructure spend`.
- Sample traffic metrics include:
  - number of HTTP requests per second,
  - number of requests for static vs. dynamic content,
  - number of concurrent sessions, and many more.

### error

![alt text](./assets/img/post/mcl6grhf21.png)

- measures how close to capacity a service is. It’s important to note, though, that capacity is often a subjective measure, that depends on the underlying service or application.
- Saturation is important because it's an indicator of how full the service is. It focuses on the most constrained resources. It’s frequently tied to degrading performance as capacity is reached.
- Sample capacity metrics include
  - percentage memory utilization,
  - percentage of thread pool utilization,
  - percentage of cache utilization and many more.

![alt text](./assets/img/post/mcl6grhg97.png)

- measure system failures or other issues.
- Errors are often raised when a flaw, failure, or fault in a computer program or system causes it to produce incorrect or unexpected results, or behave in unintended ways.
- Errors might indicate configuration or capacity issues or service level objective violations.
- Sample error metrics include
  - wrong answers or incorrect content,
  - number of 400/500 HTTP codes,
  - number of failed requests,
  - number of exceptions and many more.

---

## GCP service

### Logs Explorer

- examine messages generated by running code

### Error reporting

![alt text](./assets/img/post/mcl6grhg15.png)

![alt text](./assets/img/post/mcl6grhg91.png)

## Cloud profiler

![alt text](./assets/img/post/mcl6grhg92.png)

![alt text](./assets/img/post/mcl6grhg04.png)

## Cloud trace

![alt text](./assets/img/post/mcl6grhh63.png)

- see the latency of requests for a web application deployed to Cloud Run

---

## Application Performance Management




---
