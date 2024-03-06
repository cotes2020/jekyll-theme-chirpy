---
title: NetworkSec - Layer 3 - Basic Routing Processes
# author: Grace JyL
date: 2018-05-18 11:11:11 -0400
description:
excerpt_separator:
categories: [15NetworkSec]
tags: [NetworkSec]
math: true
# pin: true
toc: true
# image: /assets/img/note/tls-ssl-handshake.png
---

[toc]

---


# Basic Routing Processes

> From N+ Chapter 6

- **PC1** needs to send traffic to **Server1**:
  - devices are on different networks.
  - How packet from source IP address 192.168.1.2 get routed to destination IP address 192.168.3.2.

---

## process step by step:

- step 1: **PC1** to **Router1**
  - **PC1**
    - compare two `IP` and `subnet mask`, 192.168.1.2/24, 192.168.3.2/24.
    - concludes that the destination IP address resides on a remote subnet.
    - send the packet to its **default gateway Router1** 192.168.1.1

    - To construct a Layer 2 frame, **PC1** needs **Router1**’s MAC address.
      - been `manually configured` on **PC1**
      - or `dynamically learned via DHCP (Dynamic Host Configuration Protoco`

    - **PC1** sends an `Address Resolution Protocol (ARP)`
      - broadcast-based protocol, request for **Router1**’s MAC address.
    - receives an ARP reply from **Router1**
    - adds **Router1**’s MAC address to its ARP cache.
    - now sends its data in a frame destined for **Server1**.

- step 2. **Router1** to **Router2**
  - **Router1**
    - receives frame from **PC1**
    - interrogates the IP header.
      - IP header contains a Time to Live (TTL) field:
        - decremented once for each router hop.
        - When reduced to 0:
          - the router discards the frame
          - and sends a time exceeded Internet Control Message Protocol (ICMP) message back to the source.
  - Assuming the TTL is not decremented to 0
  - **Router1**
    - checks its routing table to determine the best path to reach `network 192.168.3.0/24`.
    - network 192.168.3.0/24 is accessible via `interface Serial 1/1`.
    - forwards the frame out of its Serial 1/1 interface.
- step 3. **Router2**
  - **Router2**
    - receives the frame
      - decrements the TTL in the IP header, like **Router1** did.
      - Assuming the TTL != 0
    - interrogates the IP header to determine the destination network.
  - the destination network: 192.168.3.0/24
    - directly attached to router R2’s Fast Ethernet 0/0 interface.
  - R2
    - sends an `ARP request` to determine the MAC address of **Server1**.
    - After an ARP Reply is received from **Server1**
    - forwards the frame out of its Fast Ethernet 0/0 interface to **Server1**.

---

## Router table:
- allows a router to quickly look up the best path that can be used to send the data.
- `Layer 3 to Layer 2 mapping information`:
  - ARP cache mapping
  - `MAC address -> IP address`.
- `Routers rely on internal routing table` to make packet forwarding decisions.
  - consulted its routing table to find the best match.
  - The best match: the route that has the longest prefix.
- updated on a regular schedule
  - to ensure that info is accurate
  - to account for changing network conditions.

- Example:
  - a router has an entry for network 10.0.0.0/8 and for network 10.1.1.0/24.
  - the router is seeking the best match for a destination address of 10.1.1.1/24.
  - The router would select the 10.1.1.0/24 route entry as the best entry,
  - because that route entry has the longest prefix.


---

## Routing Protocol Characteristics

- This section looks at routing protocol characteristics,
  - Like how believable a routing protocol is versus other routing protocols.
  - Also, in multiple routes, different routing protocols use different metrics to determine the best path.
  - A distinction is made between Interior Gateway Protocols (IGP) and Exterior Gateway Protocols (EGP).


---


## Believability of a Route
When networks be reachable via more than one path / a routing protocol knows multiple paths to reach such a network. (maybe as a result of a corporate merger)

which route does the routing protocol select?


### Metrics

If a routing protocol knows of more than one route to reach a destination network and those routes have equal metrics:
- some routing protocols support load balancing across equal-cost paths.
- EIGRP can even be configured to load balance across unequal-cost paths.

Different routing protocols can use different parameters in their calculation of a metric.

It varies on the routing protocol and the metric that routing protocol uses.
- a value assigned to a route.
- lower metrics are preferred over higher metrics.


- **administrative distance** (AD) 通告距离:
  - The index of believability
  - lower AD values are more believable than higher AD values.

| Routing Information Source                            | Administrative Distance            |
| ----------------------------------------------------- | ---------------------------------- |
| `Directly` connected network                          | 0                                  |
| `Statically` configured network                       | 1                                  |
| `EIGRP` (Enhanced Interior Gateway Routing Protocol ) | 90                                 |
| `OSPF` (Open Shortest Path First)                     | 110                                |
| `RIP` (Routing Information Protocol)                  | 120                                |
| `External EIGRP`                                      | 170                                |
| Unknown of unbelievable                               | 255 (considered to be unreachable) |



---


# Metrics

> From 528CN - chapter 3.3.4

ways to calculate link costs that have proven effective in practice.

> One example:
> quite reasonable and very simple, assign a cost of 1 to all links—the least-cost route will then be the one with the fewest hops.

Such an approach has several drawbacks:
1. does not distinguish between links on a latency basis. Thus, a satellite link with 250-ms latency looks just as attractive to the routing protocol as a terrestrial link with 1-ms latency.
2. does not distinguish between routes on a capacity basis, making a 9.6-kbps link look just as good as a 45-Mbps link.
3. does not distinguish between links based on their current load, making it impossible to route around overloaded links.(the hardest because you are trying to capture the complex and dynamic characteristics of a link in a single scalar cost.)

The `ARPANET (Advanced Research Projects Agency Network) 阿帕网` was the testing ground for a number of different approaches to link-cost calculation.
- (It was also the place where the superior stability of link-state over distance-vector routing was demonstrated; the original mechanism used distance vector while the later version used link state.)
- The evolution of the ARPANET routing metric explores the subtle aspects of the problem.


The original ARPANET routing metric:
- measured `the number of packets that were queued waiting to be transmitted on each link`
  - a link with 10 packets queued waiting to be transmitted was assigned a larger cost weight than a link with 5 packets queued for transmission.
- Using queue length as routing metric did not work well
  - queue length is an artificial measure of load
  - it moves packets toward the shortest queue rather than toward the destination
  - like hop from line to line at the grocery store.
- State more precisely, it did not considerate the bandwidth / latency of the link.


A second version of the ARPANET routing algorithm
- `took both link bandwidth and latency into consideration`
- used delay, not just queue length, as a measure of load.
- This was done as follows:
  1. each incoming packet was times-tamped with its time of arrival at the router `(ArrivalTime)`; and its departure time from the router `(DepartTime)`
  2. when the link-level ACK was received from the other side, the node computed the delay for that packet as:
     - `Delay = (DepartTime − ArrivalTime) + TransmissionTime + Latency`
     - `TransmissionTime and Latency`: statically defined for the link and captured the link’s bandwidth and latency, respectively.
     - `DepartTime − ArrivalTime:` the amount of time the packet was delayed (queued) in the node due to load.
     - If the ACK did not arrive, but instead the packet timed out, then DepartTime was reset to the time the packet was retransmitted.
     - `DepartTime − ArrivalTime`: captures the reliability of the link
       - the more frequent the retransmission of packets, the less reliable the link, the more to avoid it.
  3. the weight assigned to each link was derived from the average delay experienced by the packets recently sent over that link.

Although an improvement over the original mechanism, this approach also had a lot of problems. Under light load, it worked reasonably well,


---

## Difference between a routing protocol and a routed protocol.

- **routing protocol**
  - (like `RIP, OSPF, or EIGRP`):
  - a protocol that `advertises route information between routers`.

- **routed protocol**
  - (only IP, no other room, a static way):
  - a protocol with an addressing scheme that `defines different network addresses`.
  - Traffic can then be routed between defined networks, perhaps with the assistance of a routing protocol.


---

# Different routing way

A router’s routing table can be populated from various sources.
- `statically configure` a route entry.
- learned via a `dynamic routing protocol` (example, OSPF or EIGRP),
- or the router is `physically attached to that network`.

---

## Directly Connected Routes

how to reach a specific destination network:
- the router has an interface directly participating in network.

---

## Static Routes
- `Routing table` been crated manually
- used mainly on small networks.
- losses utility on larger networks because the manual updates hard to keep it up to date.

Static Routes
- Router does not need knowledge of each individual route on the Internet:
  - R1 knows: devices on its locally attached networks.
  - R1 needs to know: get out to the rest of the world.
  - R1 could be configured with a default static route,
  - “If traffic's destined network is not currently in the routing table, send that traffic out of interface Serial 1/1.”
  - Any traffic destined for a non-local network can simply be sent to router R2.
  - Because R2 is the next router hop along the path to reach all those other networks,
  - R2 can reach the Internet by sending traffic out of its Serial 1/0 interface.
  - a static route, pointing to X.X.X.X, can be statically added to router R2’s routing table.
- static route does not always reference a local interface.
  - Instead, static route might point to a next-hop IP address (an interface’s IP address on the next router to which traffic should be forwarded).
  - The network address of a default route is 0.0.0.0/0：
  - default route 默认路由:
  - 路由表里的一个表项, 指定 next hop , default route (默认网关).
  - 所有在路由表里没有对应表项的数据包都发到这个网关.
  - 在路由表中查找"对应"表项：
  - 把路由表表项的IP地址的子网掩码与目的地址的子网掩码进行比较
  - 所以只要把"默认路由"的子网掩码设为0(在路由表中即系0.0.0.0/0这一项),则"默认路由"一定可以目的地址"对应".

---

## Interior vs Exterior Gateway Protocols

Routing protocols can categorized based on the scope of their operation:

- **Interior Gateway Protocols (IGP)**
  - operate within an autonomous system (AS)
  - used to `exchange routing information`.
    - Link-state routing protocols: OSPF, IS-IS
    - Distance-vector routing protocols: RIPv1, RIPv2, IGRP, EIGRP,

- **Exterior Gateway Protocols (EGP)**
  - operate between autonomous systems.
  - Used for `exchanging routing information between autonomous systems`.
    - BGP (Border Gateway Protocol) ,
    - path vector routing protocol,

> example:
- `R1 and R2` are in one AS (AS 65002).
- `R3 and R4` are in one AS (AS 65003).
- `router ISP1` is a router in a separate autonomous system (AS 65001), run by a service provider.
- **EGP Border Gateway Protocol** is used to exchange routing information between the service provider’s AS and each of the other autonomous systems.



















.
