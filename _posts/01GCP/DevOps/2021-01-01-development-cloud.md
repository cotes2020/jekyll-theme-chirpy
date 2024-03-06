---
title: GCP - Development in the cloud
date: 2021-01-01 11:11:11 -0400
categories: [01GCP, DevOps]
tags: [GCP]
toc: true
image:
---

- [Development in the cloud](#development-in-the-cloud)
  - [devOps](#devops)
  - [Cloud Build](#cloud-build)
  - [development.](#development)
    - [Cloud Source Repositories](#cloud-source-repositories)
  - [Infrastructure as code](#infrastructure-as-code)
    - [Deployment Manager `declarative rather than imperative`](#deployment-manager-declarative-rather-than-imperative)
  - [Proactive instrumentation: Stackdriver](#proactive-instrumentation-stackdriver)
    - [Stackdriver](#stackdriver)
    - [Stackdriver Monitoring](#stackdriver-monitoring)
    - [Stackdriver Logging](#stackdriver-logging)
    - [Stackdriver Error Reporting](#stackdriver-error-reporting)
    - [Stackdriver Trace](#stackdriver-trace)
    - [Stackdriver Debugger](#stackdriver-debugger)

---


# Development in the cloud


![Screen Shot 2022-08-16 at 23.32.22](https://i.imgur.com/izRxtgt.jpg)

![Screen Shot 2022-08-16 at 23.32.42](https://i.imgur.com/Nt0AZ8o.png)



---

## devOps


## Cloud Build


![Screen Shot 2022-08-16 at 23.33.12](https://i.imgur.com/oNtXBE6.png)




---


## development.

- Git
  - use Git to store and manage the source code trees.
  - running their own Git instances or using a hosted Git provider.
    - Running the own: have total control.
    - Using a hosted Git provider: less work.
  - Cloud Source Repositories
    - keep code private to a GCP project
    - use IAM permissions to protect it,
    - and not have to maintain the Git instance theself.

---

### Cloud Source Repositories

- Cloud Source Repositories
  - provides Git version control
    - to support the team's development of any application or service,
    - including those that run on App Engine, Compute Engine, and Kubernetes Engine.
  - can have any number of private Git repositories
    - to organize the code associated with the cloud project in whatever way works best for you.
  - contains a source viewer
    - browse and view repository files from within the GCP console.


---



## Infrastructure as code

- Setting up the environment in GCP can entail many steps:
  - setting up compute network and storage resources,
  - and keeping track of their configurations.

- do it all by hand `imperative`
  - figure out the commands to set up the environment
  - to change the environment
    - figure out the commands to change it from the old state to the new.
  - to clone the environment,
    - do all those commands again.

---

### Deployment Manager `declarative rather than imperative`

- an Infrastructure Management Service for GCP resources.

- use a template.
  - a specification of what the environment should look like.

- automates the creation and management of the Google Cloud Platform resources

- To use it
  - create a template file
    - using either the YAML markup language or Python
    - describes the components of the environment
  - give the template to Deployment Manager
    - figures out and does the actions needed to create the environment the template describes.

- to change the environment
  - edit the template and then tell Deployment Manager to update the environment to match the change.

- can store and version control the Deployment Manager templates in Cloud Source repositories.


```bash
export MY_ZONE=us-central1-f
echo $DEVSHELL_PROJECT_ID
vim mydeploy.yaml
# resources:
# - name: my-vm
#   type: compute.va.instance
#   properties:
#     zone: ZONE
#     machineType: zones/ZONE/machineTypes/na-standard-1

#     metadata:
#      items:
#      - key: startup_script
#        value: "apt-get update"

#     disks:
#     - deviceName: boot
#     type: PERSISTENT
#     boot: true
#     autoDelete: true
#     initializeParams:
#         sourceImage: https://www.googleapis.com/compute/v1/projects/debian-cloud/global/images/debian-9-stretch-v201709$

#     networkInterfaces:
#     - network: https://www.googleapis.com/compute/v1/projects/PROJECT_ID/global/networks/default
#     accessConfigs:
#     - name: External NAT
#         type: ONE_TO_ONE_NAT

sed -i -e 's/PROJECT_ID/'$DEVSHELL_PROJECT_ID/ mydeploy.yaml
sed -i -e 's/ZONE/'$MY_ZONE/ mydeploy.yaml

gcloud deployment-manager deployments create my-dep1 \
    --config mydeploy.yaml

gcloud deployment-manager deployments list


gcloud deployment-manager deployments update my-dep1 \
    --config mydeploy.yaml
```





---

## Proactive instrumentation: Stackdriver

![Screen Shot 2021-06-30 at 1.11.02 AM](https://i.imgur.com/9sLtQFa.png)

- Monitoring
  - lets you figure out whether the changes you made were good or bad.
  - lets you respond with information rather than with panic, when one of the end users complains that the application is down.



### Stackdriver



- GCP's tool for monitoring, logging and diagnostics (debug, error reporting, trace)
  - gives access to many different kinds of signals from the infrastructure platforms, virtual machines, containers, middleware and application tier, logs, metrics and traces.
  - gives insight into the application's health, performance and availability.

- core components of Stackdriver:
  - Monitoring, Logging, Trace, Error Reporting and Debugging.


![Screen Shot 2021-06-30 at 1.13.54 AM](https://i.imgur.com/NlEdSpl.png)


![Screen Shot 2021-02-09 at 01.38.29](https://i.imgur.com/rRIi5gV.png)


![Screen Shot 2021-06-30 at 1.14.23 AM](https://i.imgur.com/UvolPWA.png)




### Stackdriver Monitoring

![Screen Shot 2021-06-30 at 1.14.50 AM](https://i.imgur.com/F9RrJ3c.png)

![Screen Shot 2021-06-30 at 1.15.14 AM](https://i.imgur.com/G4ZLaXt.png)

![Screen Shot 2021-06-30 at 1.15.35 AM](https://i.imgur.com/olL8dTe.png)

![Screen Shot 2021-06-30 at 1.16.24 AM](https://i.imgur.com/K39mhuB.png)

![Screen Shot 2021-06-30 at 1.16.52 AM](https://i.imgur.com/2fqQzkE.png)

![Screen Shot 2021-06-30 at 1.17.23 AM](https://i.imgur.com/OFwsCQ5.png)

![Screen Shot 2021-06-30 at 1.18.07 AM](https://i.imgur.com/pO5CT99.png)

![Screen Shot 2021-06-30 at 1.18.51 AM](https://i.imgur.com/5TyCIe8.png)


![Screen Shot 2021-06-30 at 1.19.25 AM](https://i.imgur.com/ZKSuj9X.png)

![Screen Shot 2021-06-30 at 1.19.52 AM](https://i.imgur.com/NSWwXF2.png)

- checks the endpoints of web applications and other Internet accessible services running on the cloud environment.
- configure uptime checks associated with URLs, groups or resources such as Instances and load balancers.
- set up alerts on interesting criteria,
  - like when health check results or uptimes fall into levels that need action.
- use Monitoring with a lot of popular notification tools.
- create dashboards to help visualize the state of the application.


### Stackdriver Logging

![Screen Shot 2021-06-30 at 1.36.12 AM](https://i.imgur.com/m1p8cOr.png)

![Screen Shot 2021-06-30 at 1.36.38 AM](https://i.imgur.com/PaX7hvz.png)


![Screen Shot 2021-06-30 at 1.37.17 AM](https://i.imgur.com/DjbicRa.png)

- view logs from the applications and filter and search on them.
- define metrics based on log
  - based on log contents that are incorporated into dashboards and alerts.
- export logs to BigQuery, Cloud Storage and Cloud PubSub.



### Stackdriver Error Reporting

![Screen Shot 2021-06-30 at 1.37.46 AM](https://i.imgur.com/3lwE35H.png)

- tracks and groups the errors in the cloud applications.
- notifies you when new errors are detected.


### Stackdriver Trace

![Screen Shot 2021-06-30 at 1.38.11 AM](https://i.imgur.com/gAFEdwa.png)

- sample the latency of app engine applications and report Per-URL statistics.


### Stackdriver Debugger

![Screen Shot 2021-06-30 at 1.38.53 AM](https://i.imgur.com/gFT2uFY.png)

> debugging
> go back into it and add lots of logging statements.

Stackdriver Debugger
- offers a different way.
- It connects the applications production data to the source code.
- inspect the state of the application at any code location in production.
- view the application stage without adding logging statements.

- works best when the application source code is available, such as in Cloud Source repositories.
- it can be in other repositories too.

```bash
dd if=/dev/urandom | gzip -9 >> /dev/null &

# Google CLoud Platform > Stackdriver Monitoring
```





























.
