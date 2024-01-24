---
title: AWS - Session Affinity, Load-Balanced, Session Fail Over, Sticky Sessions
date: 2020-07-18 11:11:11 -0400-
categories: [01AWS, Balancing]
tags: [AWS, Balancing]
toc: true
image:
---

- [Session Affinity 类同, Load-Balanced, Session Fail Over, Sticky Sessions](#session-affinity-类同-load-balanced-session-fail-over-sticky-sessions)
  - [The application can’t remember who the client is](#the-application-cant-remember-who-the-client-is)
  - [session location](#session-location)
  - [Load balanced](#load-balanced)
    - [1. Session information stored in client-side cookies only](#1-session-information-stored-in-client-side-cookies-only)
    - [2. Load balancer directs the user to the same machine:](#2-load-balancer-directs-the-user-to-the-same-machine)
    - [3. Shared backend database or memcached or key/value store:](#3-shared-backend-database-or-memcached-or-keyvalue-store)
  - [example](#example)
  - [Stickiness vs Sticky sessions](#stickiness-vs-sticky-sessions)
    - [Sticky sessions / session affinity](#sticky-sessions--session-affinity)
      - [Duration-based session stickiness](#duration-based-session-stickiness)
      - [Application-controlled session stickiness](#application-controlled-session-stickiness)

---

# Session Affinity 类同, Load-Balanced, Session Fail Over, Sticky Sessions

---


## The application can’t remember who the client is
- On a technical level:
- **Each HTTP request-response pair between the client and app happens (most often) on a different TCP connection**.
  - This is especially true when a load balancer sits between the client and the app
  - So the application can’t use the TCP connection as a way to remember the conversational context.
- **HTTP itself is stateless**:
  - any request can be sent at any time, in any sequence, regardless of the preceding requests.
  - app may demand a particular pattern of interaction – like logging in before accessing certain resources – but that application-level state is enforced by the application, not by HTTP.
  - So HTTP cannot be relied on to maintain conversational context between the client and the application.


no transparent session fail-over:
- The OpenPages application can be configured for a multi-server (node) configuration.
- If one node (or server) goes down while a user is connected, the user will need to close the browser and re-login to connect to one of the other available nodes (or servers) in the environment.
- if one node goes down, users will NOT automatically be rerouted to one of the available nodes.

Use-Case 1:
> If Production Server A goes down, the admin service and the OP/IBPM server service will go down.
> For users using the server, will need to close the browser and manually re-login to utilize Production Server B.
> All in-flight transactions which are getting processed will be lost and users have to reiterate the task.
> From an end user perspective
> - was performing an action or task on Production Server A
> - may need to re-login to complete the task on Production Server B.
> - no loss in functionality, however have to repeat the task again that been interrupted.
> - All in-flight transactions getting processed will be lost and have to reiterate the task.

Use-Case 2:
> If both Production Server A and Production Server B are unavailable (due to various reasons),
> an administrator can startup the disaster recovery system(s). This assumes that 3rd party mechanisms (ie: database mirroring; data replication) are in place.
> The IT administrator would also have to update the load-balancer to indicate the disaster recovery servers can be used.
> If the load balancer has sticky-IP time out configured then users may have to wait until the time out threshold is reached and re-access the URL.


---

## session location


goal: <font color=red> Manage user session </font>
- storing those sessions locally to the node responding to the HTTP request
- design a layer in architecture which can store those sessions in a scalable and robust manner.


2 ways to solve this problem of forgetting the context.

1. <font color=red> the client remind the application of the context every time </font> he requests something

2. <font color=red> the application remember the context </font> by creating an associated memento
   - This memento is given to the client and returned to the application on subsequent requests.
     1. via URL
        - `https://www.example.com/products/awesomeDoohickey.html?sessionID=0123456789ABCDEFGH`
     2. via cookies
        - placed within the HTTP request
        - so they can be discovered by the application even if a load balancer intervenes.



Use-Case:
> Large websites may be "load balanced" across multiple machines.
> - a user may hit any of the backend machines during a session.
> - several methods exist to allow many machines to share user sessions.
> - The method chosen will depend on the style of load balancing employed, as well as the availability/capacity of backend storage:



---


## Load balanced
- a user may hit any of the backend machines during a session.
- several methods exist to allow many machines to share user sessions.
- The method chosen will depend on the style of load balancing employed, as well as the availability/capacity of backend storage:


way in which the Application Session State is stored.
- Stateful
  - the application session state is stored locally on the same server as the application.
    - This is also referred to as a stateful server
  - to scale up/down the application server
    - there <font color=red> would be user interruption </font>

- Stateless
  - the application session state is stored remotely on another server rather than locally on the application server.
    - This is also referred to as a stateless server
  - to scale up/down the application server
    - <font color=red> no user interruption </font>



---


### 1. Session information stored in client-side cookies only

> least suitable for most applications:

- <font color=red> session identifier + Session information </font> is stored in a user's <font color=red> cookie </font>
  - example: the user's cookie might contain the contents of their shopping basket.

- <font color=red> No backend storage </font> is required
  - the session data is not stored server-side
  - more difficult for developers to debug
  - The amount of data that can be stored in the session is limited (by the 4K cookie size limit)

- The user does not need to hit the same machine each time, so DNS load balancing can be employed

- <font color=red> no latency </font> associated with retrieving the session information from a database machine
  - (as it is provided with the HTTP request).
  - Useful if your site is load-balanced by machines on different continents.


- <font color=red> Encryption </font> has to be employed
  - if a user should not be able to see the contents of their session
  - <font color=blue> HMAC (or similar) has to be employed </font> to prevent user tampering of session data


---


### 2. Load balancer directs the user to the same machine:

> may be good in some situations:

- load balancers may set <font color=red> session cookie </font>
  - indicating which backend machine a user is making requests from
  - and direct them to that machine in the future.

- An `existing application's session handling may not need to be changed` to become multiple machines aware

- <font color=red> No shared database system (or similar) is required </font> for storing sessions
  - possibly increasing reliability
  - but at the cost of complexity

- <font color=red> A backend machine going down will take down user sessions started on/with it </font>
  - Because the user is always directed to the same machine, <font color=red> session sharing between multiple machines is not required. </font>

- <font color=red> Taking machines out of service is more difficult </font>
  - Users with sessions on a machine to be taken down for maintenance should be allowed to complete their tasks before the machine is turned off.
  - To support this, web load balancers may have a feature to "drain" requests to a certain backend machine.


---


### 3. Shared backend database or memcached or key/value store:

> probably the cleanest method of the three:

- <font color=red> Session information is stored in a backend database </font>
  - The user's <font color=blue> browser stores a cookie containing an identifier (like session ID), pointing to the session information </font>
  - The user never needs to be exposed to the stored session information.

- all web servers have access to query and update.
  - The user does not need to hit the same machine each time, so DNS load balancing can be employed

- Session information may be expired and backed up consistently.

- One disadvantage is `the bottleneck that can be placed on whichever backend storage system is employed`.

- most dynamic web applications perform <font color=red> several database queries or key/value store requests </font>
  - so the database or key/value store is the logical storage location of session data.



---

## example

![Screen Shot 2020-06-22 at 15.07.33](https://i.imgur.com/gE2TVx5.png)

A cloud design pattern that uses multiple load balancers
- 2 separate ELB going to a set of servers.
  - a load balancer that is separated by a certificate
  - another load balancer that is keeping the <font color=red> session sticky </font>

- When a website is served by <font color=red> only one web server </font>
  - for each client-server pair
    - a session object is created and remains in the memory of the web server.
  - All requests from the client go to this web server and update this session object.

- When a website is served by <font color=red> multiple web servers behind a load balancer </font>
  - the load balancer decides which web server the request goes to.
  - load balancer use <font color=blue> sticky sessions </font> or <font color=blue> Stickiness sessions </font>


---

cache

![Screen Shot 2020-06-20 at 23.23.50](https://i.imgur.com/bTDwJPr.png)
￼
![Screen Shot 2020-06-20 at 23.24.41](https://i.imgur.com/M9FQKnN.png)

---


## Stickiness vs Sticky sessions

> Stickiness vs Sticky sessions

<img alt="pic" src="https://i.imgur.com/b2u9xtx.png" width="400">

<img alt="pic" src="https://i.imgur.com/0HkpZ99.png" width="400">


1. If the load balancer use <font color=red> sticky sessions </font>
   - <font color=blue> all interactions happen with the same physical server </font>
   - the new sticky session feature instruct the load balancer
     - to <font color=blue> route repeated requests to the same EC2 instance whenever possible </font>
       - A series of requests from the user will be routed to the same EC2 instance if possible.
     - If the instance has been terminated or has failed a recent health check
       - the load balancer will route the request to another instance.
   - the instances can cache user data locally for better performance.


1. If the load balancer use <font color=red> Stickiness sessions </font>
   - important because mobile applications need to keep sticky sessions
   - For desktop users, common not require sticky sessions
   - load balancer had the freedom to forward each incoming HTTP or TCP request to any of the EC2 instances under its purview.
   - even load on each instance,
   - but also meant that each instance would have to retrieve, manipulate, and store session data for each request without any possible benefit from locality of reference.


---


### Sticky sessions / session affinity

- By default, load balancer routes each request independently to the registered instance with the smallest load.
   - Stickiness sessions
   - even load on each instance


by sticky session

- enables the load balancer to <font color=red> bind user's session to a specific instance </font>
  - all requests from the user during the session are sent to the same server instance.
  - can use `sticky sessions` for only `HTTP/HTTPS load balancer listeners`

- <font color=red> limit application’s scalability </font>
  - the load balancer is unable to truly balance the load each time it receives request from a client.
  - send all the requests to their original server where the session state was created
    - even that server might be heavily loaded
    - and another less-loaded server is available to take on this request.

- allow to <font color=red> route user to the particular web server </font> which is managing that individual user’s session.
  - better user experience.


The <font color=red> session’s validity </font> can be determined by:
- a client-side cookies
- via configurable duration parameters that set at the load balancer
  - which routes requests to the web servers.



#### Duration-based session stickiness

- The load balancer uses a special `load balancer–generated cookie` to <font color=red> track the application instance for each request </font>
- When the load balancer receives a request
  - first <font color=blue> checks whether this cookie is present in the request </font>
  - If there is a cookie
    - the request is sent to the application instance specified in the cookie.
  - If there is no cookie
    - the load balancer chooses an application instance based on the existing load balancing algorithm.
    - A cookie is inserted into the response
      - for binding subsequent requests from the same user to that application instance.

- The stickiness policy configuration
  - <font color=red> defines a cookie expiration </font>
  - establishes the duration of validity for each cookie.
  - The cookie is <font color=blue> automatically updated after its duration expires </font>



#### Application-controlled session stickiness
- The load balancer uses a special cookie to <font color=red> associate the session with the original server that handled the request </font>

- The stickiness policy configuration
  - follows the lifetime of the application-generated cookie corresponding to the cookie name specified in the policy configuration.
  - The load balancer only inserts a new `stickiness cookie` <font color=blue> if the application response includes a new application cookie </font>


- The load balancer stickiness cookie does not update with each request.
- If the application cookie is explicitly removed or expires the session stops being sticky until a new application cookie is issued.
  - This means that can perform maintenance without affecting customers’ experience.
  - such as deploying software upgrades or replacing backend instances,


- Applications often store session data in memory, but this approach doesn’t scale well.
  - Options available to manage session data without `sticky sessions` include:
  - Using ElastiCache or DynamoDB to store session data.




---

Multiple load balancers, based on the types of devices that access the web site.

![Screen Shot 2020-06-22 at 15.07.33](https://i.imgur.com/gE2TVx5.png)

When a web application is multi-device compatible (access from PCs and smart phones)

1. perform a `setup for SSL/TLS` or to `assign sessions for individual access devices`, if the setup is performed by the EC2 instances themselves,
   - any change to the settings would become extremely laborious as the number of servers increases.


2. **solve this problem**: assign multiple virtual load balancers with different settings.
   - rather than modifying the `servers`
   - changing the `virtual load balancer` for routing the access.
   - change the behavior relative to access by the different devices
   - For example
     - apply this to settings such as for sessions, health checks, and HTTPS.
     - To implement, `assign multiple virtual load balancers to a single EC2 instance`.
     - use the SSL Termination function of the load balancer to perform the HTTPS (SSL) process.
     - Place EC2 instance under the control of the load balancers
     - And prepare load balancers with different settings for sessions, health checks, HTTPS, etc., and switch between them for the same EC2 instance.


- some benefits.
  - The behavior on the load balancer level for mobile sites and PC sites can be different, even use the same EC2 instance.
  - Even when multiple SSLs (HTTPS) are used by the same EC2 instance, can prepare load balancers for each SSL (HTTP).
  - when cut off an EC2 instance from a load balancer to perform maintenance, have to cut off the EC2 instance from all of the load balancers.
  - When use the SSL Termination function of a load balancer, the EC2 instance will be able to receive requests via HTTP, making it difficult to evaluate the HTTPS connection by the applications.


---


ref
- [New Elastic Load Balancing Feature: `Sticky Sessions`](https://aws.amazon.com/blogs/aws/new-elastic-load-balancing-feature-sticky-sessions/)
- [Elastic Load Balancing with Sticky Sessions](https://shlomoswidler.com/2010/04/08/elastic-load-balancing-with-sticky-sessions/)
- [IBM - OpenPages Load-Balanced Configuration vs Session Fail Over](https://www.ibm.com/support/pages/openpages-load-balanced-configuration-vs-session-fail-over)


.
