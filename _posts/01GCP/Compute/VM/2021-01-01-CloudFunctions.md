---
title: GCP - Google Cloud Computing - Cloud Functions
date: 2021-01-01 11:11:11 -0400
categories: [01GCP, Compute]
tags: [GCP]
toc: true
image:
---

- [Google Cloud Computing - Cloud Functions](#google-cloud-computing---cloud-functions)
  - [basic](#basic)

---

# Google Cloud Computing - Cloud Functions

---

## basic


![Screen Shot 2021-02-12 at 13.25.46](https://i.imgur.com/uuTClRK.png)

> Many applications contain event-driven parts.
> - example,
> - application that lets users upload images.
>   - need to process that image in various ways:
>   - convert it to a standard image format,
>   - thumbnail into various sizes,
>   - and store each in a repository.
> - integrate this function in application, then you have to worry about providing compute resources for it, no matter whether it happens once a day or once a millisecond.
> - What if you could just make that provisioning problem go away? write a single purpose function that did the necessary image manipulations and then arrange for it to automatically run whenever a new image gets uploaded.


Cloud Functions

- an <font color=red> event-driven </font>, serverless compute service
  - for simple single purpose functions that are attached to events.
  - <font color=blue> event-driven, the function gets executed when a particular event occurs. </font>

- create single-purpose functions that respond to events without servers or runtime binaries.
  - just write code in JavaScript for a Node.js environment that GCP provides
  - upload the code written in JavaScript or Python, or Go
  - configure when it should fire
    - setting up a Cloud Function works.
    - choose which events you care about.
    - triggers: For each event type, you tell Cloud Functions you're interested in it.
    - attach JavaScript functions to the triggers.
  - and then GCP will automatically deploy the appropriate computing capacity to run that code.
  - the functions will respond whenever the events happen.

- no pay for servers
  - charged for the time that the code/functions runs.
  - For each function, invocation memory and CPU use is measured in the 100 millisecond increments, rounded up to the nearest increment.
  - provides a perpetual free tier.
  - So many cloud function use cases could be free of charge.

- the code is triggered within a few milliseconds based on events.
  - can trigger on events in Cloud Storage, Cloud Pub/Sub,
    - file is uploaded to Google cloud storage
    - or a message is received from Cloud Pub/Sub.
  - or in HTTP call
    - triggered based on HTTP endpoints define,
  - and events in the fire based mobile application back end.

- to enhance existing applications without having to worry about scaling.

- These servers are automatically scaled and are deployed from highly available and a fault-tolerant design.

- use cases
  - generally used as part of a microservices application architecture.
    - Some applications, especially those that have microservices architecture, can be implemented entirely in Cloud Functions.
  - build symbols, serverless, mobile IoT backends, or integrate with third party services and APIs.

  - Files uploaded into the GCS bucket can be processed in real time.
  - the data can be extracted, transformed and loaded for querying in analysis.
  - use Cloud Functions as part of intelligent applications
    - such as virtual assistance, video or image analysis, and sentiment analysis.





- Min Instance: keep app warm
- Loneger processung: 60min
- Larger instances
- More regions
