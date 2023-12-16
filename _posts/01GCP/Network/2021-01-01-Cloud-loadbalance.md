---
title: GCP - Cloud Load balance
date: 2021-01-01 11:11:11 -0400
<<<<<<< HEAD
categories: [01GCP, network]
=======
categories: [01GCP, GCPNetwork]
>>>>>>> 1a148b47672b35d180699fc905d033785c8bbe28
tags: [GCP]
toc: true
image:
---

- [Cloud Load balance](#cloud-load-balance)
  - [overview](#overview)
    - [global load balancers](#global-load-balancers)
    - [regional load balancers](#regional-load-balancers)
    - [managed instance group](#managed-instance-group)
    - [autoscaling and health checks](#autoscaling-and-health-checks)
    - [health check](#health-check)
  - [HTTP(S) load balancing](#https-load-balancing)
    - [architecture](#architecture)


---


# Cloud Load balance


![Screen Shot 2021-07-31 at 1.50.06 AM](https://i.imgur.com/FuGbPf6.png)

- Cloud load balancing gives you the ability to distribute load balanced compute resources, in single or multiple regions to meet the high availability requirements.
- put the resources behind a single anycast iP address,
- scale the resources up or down with intelligent autoscaling.
- serve content as close as possible to the users on a system that can respond to over one million queries per second.

Cloud load balancing
- a fully distributed software to find managed service.
- not instance or device based, so you do not need to manage a physical load balancing infrastructure.


## overview

### global load balancers

- HTTP/HTTP(s), SSL proxy, and TCP proxy load balancers.
- These load balancers leveraged the Google front ends, which are software defined distributed systems that sit in Google's point of presence, and are distributed globally.
- use a global load balancer when
  - the users and instances are globally distributed,
  - the users need access to the same application and content
  - you want to provide access using a single anycast iP address.


### regional load balancers

- internal and network load balancers
  - distribute traffic to instances that are in a single GCP region.
  - The internal load balancer uses Andromeda, which is GCP's software defined network virtualization stack.
  - And the network load balancer uses Maglev, which is a large distributed software system.
- Internal load balancer HTTPs traffic.
  - proxy based regional layer seven load balancer
  - enables you to run and scale the services behind a private load balancing iP address that is accessible only in the load balancers region, in the VPC network.

---


### managed instance group

- a collection of identical VM instances control as a single entity using an instance template.
- can easily update all the instances in the group by specifying a new template in a rolling update.
- Also, when applications require additional compute resources, managed instance groups can scale automatically to the number of instances in the group.
- Managed instance groups can work with load balancing services to distribute network traffic to all of the instances in the group.
- If an instance in the group stops, crushes, or is deleted by an action other than the instance group's commands, the managed instance group automatically recreates the instance so it can resume its processing tasks.
  - The recreated instance uses the same name and the same instance template as the previous instance.
- Managed instance groups can automatically identify and recreate unhealthy instances in a group to ensure that all instances are running optimally.
- **Regional managed instance groups** are generally recommended over **zonal managed instance groups** because they allow you to `spread the application load across multiple zones` instead of confining application to a single zone or having you manage multiple instance groups across different zones.
  - This replication protects against zonal failures and unforeseen scenarios where an entire group of instances in a single zone malfunctions.
  - If that happens, application can continue serving traffic from instances running in another zone in the same region.

- to create a managed instance group
  - create an `instance template`
  - create a managed instance group of N specified instances.
  - The instance group manager then automatically populates the instance group based on the instance template.
  - define the specific rules for that instance group.
    - decide what type of managed instance group you want to create.
    - You can use managed instance groups for
      - stateless serving or batch workloads, such as website front-end or image processing from a queue,
      - or for stateful applications, such as databases or legacy applications.
  - provide a name for the instance group
  - decide whether the instance group is going to be single or multi-zoned and where those locations will be.
  - optionally provide port name mapping details.
  - select the instance template
  - decide auto-scale and under what circumstances.
  - creating a health check to determine which instances are healthy and should receive traffic. Essentially, you're creating virtual machines, but you're applying more rules to that instance group.



### autoscaling and health checks

managed instance groups offer autoscaling capabilities
- automatically add or remove instances from a **managed instance group** based on increase or decrease in load.
- Autoscaling helps applications gracefully handle increase in traffic and reduces cost when the need for resource is lower.
- define the autoscaling policy, and the autoscaler performs automatic scaling based on the measured load.
- Applicable autoscaling policies include scaling based on `CPU utilization, load balancing capacity, or monitoring metrics, or by a queue-based workload` like Cloud Pub/Sub.
- if the overall load is much lower than the target, the autoscaler will remove instances as long as that keeps the overall utilization below the target.


### health check

- similar to an Uptime check in Stackdriver.
- just define a protocol, port, and health criteria
- Based on this configuration, GCP computes a health state for each instance.
- The health criteria defines how often to check whether an instance is healthy. That's the check interval. How long to wait for a response? That's the timeout. How many successful attempts are decisive? That's the healthy threshold. How many failed attempts are decisive? That the unhealthy threshold.
- In the example on this slide, the health check would have to fill twice over a total of 15 seconds before an instance is considered unhealthy.


---


## HTTP(S) load balancing

- acts at layer seven of the OSI model. application layer
  - deals with the actual content of each message
  - allowing for routing decisions based on the URL.
- GCP HTTPS load balancing provides **global load balancing** for HTTPS requests destined for the instances.
- This means that the applications are available to the customers at a single anycast IP address, which simplifies the DNS setup.
- HTTPS load balancing balances `HTTP and HTTPS traffic` across multiple backend instances and across multiple regions.
  - HTTP requests are load balanced on port 80 or 8080,
  - HTTPS requests are load balanced on port 443.
- This load balancers supports both IPv4 and IPv6 clients,
- scalable, requires no pre-warming,
- enables content-based and cross-regional load balancing.
- configure own maps that route some URLs to one set of instances and route other URLs to other instances.
- Requests are generally routed to the instance group that is closest to the user.
- If the closest instance group does not have sufficient capacity, the request is sent to the next closest instance group that does have the capacity.


### architecture

- A **Global Forwarding Rule** direct incoming requests from the Internet to a target **HTTP proxy**.
- The target HTTP proxy checks each request against a `URL map` to determine the appropriate **backend service** for the request.
- For example,
  - send requests for www.example.com/audio to one backend service, which contains instances configured to deliver audio files,
  - send request for www.example.com/video to another backend service which contains instances configured to deliver video files.
- The **backend service** directs each request to an appropriate **backend** based on solving capacity zone and instance held of its attached backends.
- **The backend services** contain `a health check, session affinity, a timeout setting, and one or more backends.`
  - A health check
    - pulls instances attached to the backend service at configured intervals.
    - Instances that pass the health check are allowed to receive new requests.
    - Unhealthy instances are not sent requests until they are healthy again.
  - session affinity
    - Normally, HTTPS load balancing uses a **round robin algorithm** to distribute requests among available instances.
    - This can be overridden with session affinity.
    - Session affinity attempts to send all requests from the same client to the same Virtual Machine Instance.
  - timeout setting
    - Backend services also have a timeout setting, 30 sec by default.
    - the amount of time the backend service will wait on the backend before considering the request a failure.
    - This is a fixed timeout not an idle timeout.
    - If you require longer lived connections, set this value appropriately.
- **The backends** themselves contain an instance group, a balancing mode, and a capacity scalar.
  - An instance group
    - contains Virtual Machine Instances.
    - may be a `managed instance group with or without autoscaling` or an `unmanaged instance group`.
  - A balancing mode
    - tells the load balancing system how to determine when the backend is at full usage.
    - based on `CPU utilization` or `requests per second`.
    - If older backends for the backend service in a region are at the full usage, new requests are automatically routed to the nearest region that can still handle requests.
  - A capacity setting
    - an additional control that interacts with the balancing mode setting.
    - For example,
    - want the instances to operate at a maximum of 80% CPU utilization, you would set the balancing mode to 80% CPU utilization and the capacity to 100%.
    - to cut instance utilization in half, leave the balancing mode at 80% CPU utilization and set capacity to 50%.






。
