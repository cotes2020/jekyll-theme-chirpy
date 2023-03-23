---
title: AWS - VPC Gateway - NAT
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, Network]
tags: [AWS, Network, VPC]
toc: true
image:
---

- [Network address translation (NAT)](#network-address-translation-nat)
  - [NAT instances](#nat-instances)
  - [Network address translation (NAT) gateway](#network-address-translation-nat-gateway)
  - [difference between the VPC NAT gateway and a NAT instance](#difference-between-the-vpc-nat-gateway-and-a-nat-instance)

---

# Network address translation (NAT)

1. instances <font color=red> connect to the internet </font> but <font color=red> prevents the internet initial connection </font>
2. <font color=red> Do not support port forwarding </font>


3. enable instances in the private subnet
   - to <font color=blue> initiate outbound traffic to the internet or to other AWS services </font>
   - <font color=blue> prevent receiving inbound traffic from the Internet. </font>
     - example:
     - have a database that want to keep in the private subnet
     - but still let it access database patches.
     - NAT service allows the instance to reach the internet to download patches without letting traffic come back in and access the instance.

4. <font color=blue> fault-tolerant and can scale in response to load. </font>

5. DHCP
   - can have multiple sets of DHCP options,
     - but only can associate one set of DHCP options with a VPC at a time.
   - The DHCP option sets element of an Amazon VPC allows to direct Amazon EC2 hostname assignments to your own resources.

6. <font color=red> Dynamic / Static NAT gateway </font>
   - <font color=red> Static NAT: SNAT </font>
     - A private IP is mapped to a public IP. 
     - translates private to public IPs at a <font color=blue> 1:1 ratio </font>
   - <font color=red> Dynamic NAT: DNAT </font>
     - A range of private addresses, are mapped onto one or more public IPs.
     - <font color=blue> translate a range of private IPs to public IPs </font>
     - example:

     - when private instances only need internet access for an update
       - Dynamic NATs support session traffic,
       - provides outbound internet access to private instances for security updates.

7. When a NAT gateway has an Elastic IP it can send outbound traffic from a private subnet to the internet gateway using the NAT gateway’s Elastic IP address as the source IP address.
   - when several private instances need to share an Elastic IP
   - assign an Elastic IP to a Dynamic NAT
   - several instances can use the same Elastic IP.

8. AWS offers two primary options for using NAT services:
   - <font color=red> NAT instance </font>
     - An Amazon EC2 instance that set up as a NAT service in a public subnet
   - <font color=red> NAT Gateway </font>

![Screen Shot 2020-06-22 at 01.21.47](https://i.imgur.com/5iVOvIL.png)

---

## NAT instances

1. NAT instances are managed by you.
1. Used to <font color=red> enable private subnet instances to access the Internet </font>
   - be a route from a private subnet to the NAT instance for it to work.
   - setup
     - NAT instance must live on a <font color=blue> single public subnet with a route to an Internet Gateway </font>
     - Private instances in private subnets must <font color=blue> have a route to the NAT instance </font>
       - usually the default route destination of 0.0.0.0/0.
   - use as a <font color=red> bastion (jump) host </font>
2. Can <font color=red> monitor traffic metrics </font>
3. <font color=red> disable the source/destination check on the instance </font>
4. NAT instances need to be assigned to security groups.
   - Security groups for NAT instances must allow
     - <font color=blue> HTTP/HTTPS inbound from the private subnet </font>
     - and <font color=blue> outbound to 0.0.0.0/0. </font>
5. Using a NAT instance <font color=red> can lead to bottlenecks (not HA) </font>
   - HA can be achieved by using Auto Scaling groups, multiple subnets in different AZ’s and a script to automate failover.
6. The amount of traffic a NAT instance can support is based on the instance type.
   - Performance is dependent on instance size.
   - Can <font color=blue> scale up instance size or use enhanced networking. </font>
   - Can <font color=blue> scale out by using multiple NATs in multiple subnets. </font>
7. <font color=red> Not supported for IPv6 </font> (use Egress-Only Internet Gateway).
8. <font color=red> stateful </font>
   - NAT gateway understands the session
   - will allow inbound information because the request was a response to the private resource's request.


---

## Network address translation (NAT) gateway

1. NAT gateways are managed by AWS
   - replaces the need for NAT instances on EC2.
   - fully scaled, redundant and highly available.
   - No need to patch.
   - Not associated with any security groups.
   - Automatically assigned a public IP address.

2. limitation
   - Port forwarding is not supported.
   - Using the NAT Gateway as a Bastion host server is not supported.
   - Traffic metrics are not supported.

3. Must be created in a public subnet.
   - create a NAT gateway
     - specify the public subnet the NAT gateway should reside
     - associate the NAT gateway an Elastic IP address
       - Uses an Elastic IP address for the public IP.
   - update the route table
     - associated private subnets to the route table
     - point internet-bound traffic to the NAT gateway.
   - Thus, instances in private subnets can communicate with the internet.

4. NAT can handle 5 Gbps of bandwidth.
   - Add more IGWs, and it can scale up to 45 Gbps.

5. cannot privately route traffic to a NAT gateway through
   - a VPC peering connection, a Site-to-Site VPN connection, or AWS Direct Connect.
   - so be sure to include specific routes to those in your route table.
   - NAT gateway cannot be used by resources on the other side of these connections.


6. More secure
   - cannot access with SSH and there are no security groups to maintain
   - No need to disable source/destination checks.


---
￼
## difference between the VPC NAT gateway and a NAT instance

![Pasted Graphic](https://i.imgur.com/j0mIsQF.jpg)


1. port forwarding.
   - The VPC NAT gateway does not support Port forwarding.
2. cost differences
3. NAT gateway is a managed NAT service
   - provides better availability, higher bandwidth, and less administrative effort.
   - inherently highly available
     - might not provide the exact level of control that your application needs.
     - when you need more than 10GB of bandwidth, that is the maximum amount of bandwidth that the NAT gateway can handle.
   - NAT gateways do not have management overhead like NAT instances do.
