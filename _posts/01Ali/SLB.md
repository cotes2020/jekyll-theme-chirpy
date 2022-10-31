



- [SLB](#slb)
  - [Server Load Balancing](#server-load-balancing)
    - [server load balancer SLB](#server-load-balancer-slb)
  - [Server Load Balancer Components](#server-load-balancer-components)
    - [charges](#charges)
    - [listener](#listener)
    - [backend servers](#backend-servers)
    - [health checks](#health-checks)
    - [consideratiyon](#consideratiyon)
  - [highly available](#highly-available)


---

# SLB


setup:
- select a region to deploy a server load balancer to.
- Choose whether it is to be public or private.
- Select the instance specification, either shared or guaranteed.
- Create a listener,
  - selecting the protocol and port required.
  - select between TCP, UDP, HTTP, and HTTPS.
- Accept the default algorithm or change it.
  - select between `weighted round robin` (default),
  - weighted least connection,
  - round Robin
  - and consistent hash.
- Then allocate the listener to a group of backend servers.
  - choose between default group,
  - a primary/secondary group
  - or a VServer group.
- And lastly, set the parameters for the health check or accept the defaults.
- Once these steps have been followed, you will have a running server load balancer.


---

## Server Load Balancing

why we need something like a load balancer:
- There are potential issues when creating a web-based application service that needs to be considered.

for website
- How popular will it be?
- How many requests or hits is it going to have on it?
- The first potential issue is:
  - what hardware platform are we gonna put it on?
  - How powerful does your web server need to be to cope with all of the potential requests?
  - Do you go with, build one big and powerful server to cope with any kind of load, which can be very expensive?
  - Or, do you go with a less powerful platform, cheaper, and hope it doesn't get overloaded with too many requests, and either slow down, or even crash?
  - what happens to the web service if the hardware or server that it's running on, fails. This becomes a single point of failure.

could provision two servers to alleviate the single point of failure scenario. And at the same time, use the cheaper platform to save the cost of one big or powerful server.
  - Then this creates another potential problem:
  - How to `route requests to the different servers` so that one server does not become overloaded with all of the requests, while the other sits idle?
  - And at the same time, keep this complexity transparent to the users who are trying to access the website?


the **domain naming service/DNS** has a function called, `DNS Round Robin` that could be used in this scenario.
- this is where the request for a website's fully qualified domain name will be sequentially forwarded to each server in turn,
- but this also has potential problem.
  - The DNS servers cannot tell if a server is down.
  - if one of the two servers fail, half of the requests for the website will be sent to a server that is offline and will not respond.


In this case, we could use a **server load balancer** instead.

---

### server load balancer SLB
- a traffic distribution and control service that automatically distributes inbound traffic across multiple web-based applications, microservices or containers hosted on Alibaba ECS instances.
- It provides high availability when utilizing multiple availability zones.
- It prevents single point of failure when using more than one ECS instance in the same zone, and at the same time, provides high availability in the zone.
- It can be set up to elastically expand capacity, according to service loading.

Now, this requires autoscaling, which is a subject of another set of sessions and will not be covered here.
- And by default, SLB defends against denial of service attacks, preventing different kinds of flood attacks on the services running behind it.

SLB components.
- The server load balancer consists of three major sets of components
  - a server load balancer instance,
  - one or more listeners,
  - and at least two backend servers.
- A server load balancer instance receives and distributes incoming traffic to backend servers, using one or more listeners, that checks the client request and does a health check on the backend servers before forwarding the request.

---



## Server Load Balancer Components


A server load balancer instance which includes instance network types and instance specifications. Creating one or more listeners. Backend servers to forward traffic to. And backend server health checks.

The first component then is the server load balancer instance. A server load balancer or SLB instance is a virtual machine in which the SLB service runs. You must first select a region to create an SLB instance in, recommended best practice is to choose a region that supports the multi-zone zone type. This provisions two copies of the SLB. One in the primary selected zone, and one in the secondary selected zone, which becomes the backup zone for fail over functionality.

You must then select the instance network type and instance specification.
There are two instance network types available.
- They are internet SLB instances and intranet SLB instances.

internet SLB
- An internet SLB instance distributes client requests from the internet to backend servers according to configured forwarding rules on listeners.
- When you create an internet SLB instance, it's allocated an public IP address.
- You can resolve a domain name to the public IP address and provide public services.

Internet SLB instances
- can only be used inside Alibaba Cloud and can only forward requests from clients that can access the intranet of the SLB instance.
- When you create an intranet SLB instance, it's allocated a private IP address.

Like the network types, there are two types of instance specifications available, shared-performance instances and guaranteed-performance instances,
- shared-performance instances share other Alibaba SLB resources in the same region,
  - which means their performance cannot be guaranteed.
- Guaranteed-performance instances are set according to their selected performance specification.
  - Six different levels of performance are currently available and are based around three key performance indicators,
  - max connections, which is the maximum number of connections allowed before new connection requests are dropped.
  - Connections per second, which is the rate at which new connections are established per second before new connection requests are dropped.
  - And `queries per second`, which is the number of HTTP or HTTPS requests that can be processed per second before new connection requests are dropped.
    - The queries per second metric is available only for Layer-7 SLB listeners.


### charges
The server load balancer service can incur charges depending on which solution is selected. The following diagram depicts which of these services incur charges. You can see from the diagram that the internal SLB using the shared performance instance is the only free offering.
- All other offerings incur a usage charge.
- At present, pay as you go is the only method that supports the payment, so there's no upfront costs or longterm commitments.

Public-facing SLB instances are charged based on the charge type that is selected. And there are two charge types available,
- by traffic and by bandwidth.
- It's worth noting that internal-facing SLB instances are charged by traffic only.


### listener

After you've created the server load balancer instance, the next component to configure is the listener.

For SLB to work, a minimum of one listener is a mandatory requirement.
- The listener checks connection requests, and then distributes the request to backend servers after carrying out a health check on the server to make sure that it's running and healthy.


- A listener comprises of two main components,
  - selecting a `listener protocol and port number `
  - and selecting a `scheduling algorithm`.
  - advanced settings:
    - Session persistence, access control, and peak bandwidth settings.

- For the port forwarding rules,
  - a separate listener protocol is required for each port,
  - and the rules can be either TCP, UDP, HTTP, or HTTPS.
- For the `scheduling algorithm`,
  - there are four types of algorithm to choose from.
  - The default selection is a weighted round round.

- Backend servers can have a weight/number set against them,
  - the default is 100.
  - A backend server with a higher weight than another backend server would receive more requests.
    - For example
    - 2 backend servers named EC1 and EC2,
    - and EC1 has a weight of 100 and EC2 has a weight of 50,
    - then twice as many requests would be forwarded to EC1 than EC2.

    ![Screen Shot 2021-09-16 at 11.59.28 PM](https://i.imgur.com/iOmtV1x.png)

Weighted Round Robin
- Round Robin is where requests are evenly and sequentially distributed to all backend servers.
- It's worth noting that if the default setting of weighted round robin is selected and all backend servers have the same weight, then it's the same as selecting round robin.

![Screen Shot 2021-09-17 at 12.00.26 AM](https://i.imgur.com/xvgYG2I.png)



weighted least connection
- is the same as weighted round robin where a server with a higher weight receives more requests.
- But when the weight values of two backend servers are the same, the backend server with the least number of connections will be used to forward traffic to.


Consistent hash
- And the last one, consistent hash, which is only available for TCP and UDP rules, is where requests from the same source IP address are scheduled to the same backend server.

![Screen Shot 2021-09-17 at 12.01.53 AM](https://i.imgur.com/BBnWxva.png)


### backend servers

Before you use the SLB service, you must add one or more ECS instances as backend servers to an SLB instance, to process distributed client requests.

SLB virtualizes the added group of ECS instances in the same region into an application pool,
- you can manage backend servers through either
  - the default server group,
  - a primary server group,
  - or VServer groups.


The default server group
- contains ECS instances that are not associated with a VServer group or a primary/secondary server group.
- By default requests are forwarded to ECS instances in the default server group.

![Screen Shot 2021-09-17 at 12.04.55 AM](https://i.imgur.com/jXHLSnG.png)

---

A primary/secondary server group
- only contains two ECS instances.
- One acts as the primary or active server and the other acts as a secondary or standby server.
- No health check is performed on the secondary server.
- And when the primary server is declared as unhealthy, the system forwards traffic to the secondary server.
- When the primary server is declared as healthy again, and it restores service, the traffic is forwarded to the primary server once again.
- Note that `only TCP and UDP listeners` support configuring primary/secondary server groups.

![Screen Shot 2021-09-17 at 12.05.29 AM](https://i.imgur.com/rdPhyVE.png)

---

VServer groups
- to distribute different requests to different backend servers or configured domain name based or URL based forwarding rules.
- A single ECS instance can be a member of multiple VServer groups.

![Screen Shot 2021-09-17 at 12.03.57 AM](https://i.imgur.com/avknDCq.png)



### health checks
Before passing any requests to backend servers, SLB checks the service availability of the backend server ECS instances by performing health checks.
- Health checks improve the overall availability of your front-end service and avoid sending requests for a service to a backend server that's not online.
- The health check function is enabled by default when you create a listener, but can be turned off if required. not recommended.

- With the health check function enabled, SLB stops distributing requests to any instance that is discovered as unhealthy and restarts forwarding requests to the instance only when it's declared healthy again.


### consideratiyon

- Before creating an SLB instance, it's important to know that SLB does not support cross-region deployment.
  - Therefore ECS instances that are being used as backend servers must be in the same region as the deployed SLB instance.

- SLB also does not limit which operating system is used on the ECS instances in the pool of backend servers.
  - As long as the applications deployed in the ECS instances is the same and the data is consistent.


---


## highly available

SLB
- fully managed, scalable, and highly available load balancing service.
- Its `content-based routing` using **listeners** allows requests to be routed to different applications behind a single **load balancer**,
- saving the cost of having to build a web server per application.
- By utilizing the multi-zone feature for SLB, it supports multi-zone disaster tolerance. If the primary zone becomes unavailable, SLB rapidly switches to the backup zone.
- And it can support multi-region disaster tolerance when used in conjunction with DNS.



Welcome to session three, SLB high availability. In this session, we will look at the following topics. Provisioning high availability of SLB in a single zone, provisioning high availability of SLB in multiple zones, and provisioning cross-region disaster tolerance of SLB.

Provisioning high availability of SLB in a single zone. In session one, an introduction to load balancing, I talked about the reasoning behind providing multiple backend servers when providing a web service. Basically, it was to prevent a single point of failure by spreading the service across multiple web servers, and at the same time, providing a request balancing service so that one web server doesn't become overloaded with requests.

> In essence, providing a highly available service by utilizing multiple servers. Whilst everything is okay, a normal service is achieved.

High availability in a single zone is achieved by health checks.
- If the hardware that a server resides on fails, then SLB detects that the server is no longer available due to the health checks that are carried out on the listener in the SLB.
- However, by placing all of the servers in the same zone, we are now introducing another point of failure, and that is failure at the zone level.
- If the zone itself fails, then we lose all backend servers and the load balancer itself.

Provisioning high availability of SLB in multiple zones.
- To overcome the problem of a single zone failure, Alibaba automatically provisions a backup server load balancer in another zone, but in the same region.
- One zone is set up as the primary zone and another as the backup zone. This is known as a multi-zone SLB.
- When you create the SLB instance, you can choose which zone is the primary zone and which is the backup zone.
- there are 21 regions available for the provisioning of a server load balancer solution, and currently the UAE region is the only one with a single zone and therefore does not support multi-zone. All other regions by default are multi-zone regions.


Recommended best practice is to provision a server load balancer in a region that supports multi-zones, thereby leveraging automatic high availability across two zones in the same region.

To achieve this,
- you will have to provision at least one backend server in the backup zone.
- If you have at least two servers in each zone, you will then have high availability at the zone level and at the region level.
- While the service is running and both zones are in a healthy state, traffic is distributed across all backend servers based on the rules applied to the listeners.
- If a failure occurs at the primary zone level, SLB will detect the zone failure and within 30 seconds switch to the backup SLB instance to keep the service running.
- If the primary zone is up but the backup zone becomes unavailable, the health check on the listeners will detect the backend servers are not responding and stop sending traffic to them.
- When the primary region becomes available again, SLB switches back and normal service resumes.

Provisioning cross-region disaster tolerance of SLB. The SLB service spans across two zones in a region for high availability, but a single SLB instance cannot span across multiple regions.
- To protect against a complete regional failure, you can configure multiple SLB instances in different regions.
- You can then use Alibaba's cloud DNS service to schedule requests to achieve `cross-region disaster tolerance` through global SLB.
- You can use DNS to resolve domain names to the IP addresses of multiple SLB instances running in different regions. In the event of a region outage, DNS can then stop DNS resolution for the effective domain, thereby creating cross-region disaster tolerance by still being able to forward traffic to the second region's server load balancer.

![Screen Shot 2021-09-17 at 12.14.46 AM](https://i.imgur.com/tm7U2fI.png)
