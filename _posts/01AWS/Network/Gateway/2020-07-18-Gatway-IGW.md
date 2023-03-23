---
title: AWS - VPC Gateway - IGW
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, AWSNetwork]
tags: [AWS, Network, VPC]
toc: true
image:
---

- [Internet gateway (IGWs)](#internet-gateway-igws)

---

# Internet gateway (IGWs)

![IGW](https://i.imgur.com/9jacyfO.png)


<img src="https://i.imgur.com/bkzz2pL.png" width="400">

<img src="https://i.imgur.com/mU8bKIE.png" width="400">


- The key: whether it can access internet


- IGW is <font color=red> resilient by design </font>
  - a scalable, redundant, and highly available VPC component
    - horizontally scaled out, redundant, and highly available by default.
  - provide all subnets in all AZs with <font color=blue> resilient internet connectivity </font>
- <font color=red> one IGW one VPC </font>
  - Default VPC already has IGW
  - cannot assign more than one IGW to a VPC.
- No network riskdor bandwidth constraints on network traffic.

- <font color=red> allows communication between instances in your VPC and the internet </font>
  - provide a way to get access to the internet
  - <font color=blue> allow traffic on the internet to come by providing a target in the subnet route tables </font> for internet-routable traffic.
    - Because the instance has a public IP address, the internet can access the public instance with the public IP address.
  - IGW has to <font color=blue> add routing rules to the route table </font> for resources in a public subnet to reach the internet
  - rules are not automatically created.


  - <font color=red> IGW two purposes </font> :
    - to <font color=red> provide a target in VPC route tables for internet-routable traffic </font>
      - To make subnet public
      - attach an internet gateway
      - add a route to the route table:
        - send non-local traffic through the internet gateway to the internet (0.0.0.0/0).
        - Public IPv4 addresses are never attached to the resource's network interface.

    - to <font color=red> perform SNAT network address translation for instances that were assigned public IPv4 addresses </font>
      - a record has the mappings of private to public IPs,
      - and the IGW performs SNAT on the associated resource.
        - When the IGW receives a packet from a resource with a public IP
        - it will adjust the packets.
        - It <font color=red> replaces the private IP with the associated public IP address </font>
        - This process is known as SNAT.



- enable access to or from the internet for instances in a VPC subnet, you must ensure:
  - Create an internet gateway
  - <font color=red> Attach an internet gateway to VPC </font>
  - <font color=red> subnet's route table points to the internet gateway </font>
    - Add a route to your subnet's route table that directs internet-bound traffic to the internet gateway.
  - instances in subnet have <font color=red> public / Elastic IP addresses </font>
    - (public IPv4 address, Elastic IP address, or IPv6 address)
  - <font color=red> NACLs and security groups </font> allow the relevant traffic to flow to and from your instance.




.
