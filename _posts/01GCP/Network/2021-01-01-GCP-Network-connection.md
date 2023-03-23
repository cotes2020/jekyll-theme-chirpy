---
title: GCP - GCP connection
date: 2021-01-01 11:11:11 -0400
categories: [01GCP, GCPNetwork]
tags: [GCP]
toc: true
image:
---

- [GCP Network](#gcp-network)
  - [basic](#basic)
- [Cloud Interconnect and Peering](#cloud-interconnect-and-peering)
  - [Dedicated connections and Partner Interconnect](#dedicated-connections-and-partner-interconnect)
    - [IPsec VPN tunnels](#ipsec-vpn-tunnels)
    - [Dedicated connections](#dedicated-connections)
    - [Partner Interconnect](#partner-interconnect)
  - [Direct Peering and Carrier Peering](#direct-peering-and-carrier-peering)
    - [Direct Peering](#direct-peering)
    - [Carrier Peering](#carrier-peering)
- [Cloud VPN](#cloud-vpn)
  - [Cloud VPN gateway](#cloud-vpn-gateway)
  - [Classic VPN](#classic-vpn)
    - [dynamic routes](#dynamic-routes)
  - [HA VPN. Alternative Cloud VPN Gateway](#ha-vpn-alternative-cloud-vpn-gateway)
  - [shared VPC](#shared-vpc)


---

# GCP Network

## basic


![Screen Shot 2022-08-15 at 00.24.46](https://i.imgur.com/sBbiUs9.png)

![Screen Shot 2022-08-15 at 00.24.55](https://i.imgur.com/ikiYBca.jpg)

![Screen Shot 2022-08-15 at 00.25.05](https://i.imgur.com/gFb0cHL.png)



![Screen Shot 2022-08-15 at 00.25.30](https://i.imgur.com/j1FEI7v.jpg)


![Screen Shot 2022-08-15 at 00.25.45](https://i.imgur.com/Qu3FE9e.png)



---



# Cloud Interconnect and Peering

![Screen Shot 2021-07-31 at 1.38.32 AM](https://i.imgur.com/jT3nmfU.png)

![Screen Shot 2021-07-31 at 1.39.39 AM](https://i.imgur.com/mVa1yoW.png)
xcd0opAKI

different Cloud Interconnect and Peering services available to connect the infrastructure to Google's network.
- Direct Peering, Carrier Peering, Dedicated Interconnect, and Partner Interconnect.


- Dedicated connections
  - provide a direct connection to Google's network.
- shared connections
  - provide a connection to Google's network through a partner.
- Layer two connections
  - use a VLAN that pipes directly into the GCP environment, providing connectivity to internal IP addresses in the RFC 1918 address space.
- Layer three connections
  - provide access to G Suite services, YouTube and Google Cloud APIs using public IP addresses.
- Google also offers its own Virtual Private Network service called Cloud VPN.
  - uses the public Internet
  - but traffic is encrypted and provides access to internal IP addresses.
  - Cloud VPN is a useful addition to Direct Peering and Carrier Peering.


---

## Dedicated connections and Partner Interconnect


![Screen Shot 2021-07-31 at 1.25.49 AM](https://i.imgur.com/EfgjzSu.png)

- All of these options provide **internal IP address access** between resources in the on-premise network and in the VPC network.
- The main differences are the connection capacity and the requirements for using a service.


### IPsec VPN tunnels

- <font color=red> IPsec VPN tunnels </font>
  - Cloud VPN offers
  - capacity of 1.5-3 Gbps per tunnel
  - require a VPN device on the on-premise network.
  - The 1.5 Gbps capacity applies to the traffic that traverses the public Internet, and the 3 Gbps capacity applies to the traffic that is traversing a direct peering link.
  - can configure multiple tunnels if you want to scale this capacity.


### Dedicated connections


![Screen Shot 2021-07-31 at 1.16.20 AM](https://i.imgur.com/DIbFWPL.png)


- <font color=red> Dedicated Interconnect </font>
  - capacity of 10 Gbps per link
  - requires a connection in a Google-supported co-location facility.
  - You can have up to eight links to achieve multiples of 10 Gbps, but 10 Gbps is the minimum capacity. As of this recording, there is a Beta feature that provides 100 Gbps per link with a maximum of two links.


- provides direct physical connections between the on-premise network and Google's network.
- enables to transfer large amount of data between networks
  - more cost-effective than purchasing additional bandwidth over the public Internet.


- Dedicated Interconnect
  - allow user traffic from the on-premises network to reach GCP resources on the VPC network and vice-versa.
  - can be configured to offer a 99.9% or a 99.99% uptime SLA.
- to use Dedicated Interconnect
  - provision a cross-connect between the Google network and the own router in a common co-location facility,
  - To exchange routes between the networks, configure a BGP session over the Interconnect between the `Cloud router` and the `on-premise router.`
  - the network must physically meet Google's network in a supported co-location facility.
    - full list of these locations

### Partner Interconnect

![Screen Shot 2021-07-31 at 1.25.22 AM](https://i.imgur.com/UEvvgNo.png)


- <font color=red> Partner Interconnect </font>
  - capacity of 50 Mbps to 10 Gbps per connection,
  - requirements depend on the service provider.


- provides connectivity between the on-premises network and the VPC network through a supported service provider.
- This is useful if the data center is in the physical location that cannot reach a Dedicated Interconnect co-location facility or if the data needs don't warrant a Dedicated Interconnect.
- In order to use Partner Interconnect,
- work with a supported service provider to connect the VPC and on-premise networks.
  - full list of providers,

- These service providers have existing physical connections to Google's network that they make available for their customers to use.
- After you establish connectivity with the service provider, you can request a **Partner Interconnect connection** from the service provider,
- then establish a **BGP session** between the `Cloud router` and `on-premise router` to start passing traffic between the networks.
- can be configured to offer a 99.9% or a 99.99% uptime SLA between Google and the service provider.


> recommendation
> start with VPN tunnels.
> When need enterprise-grade connection to GCP, switch to Dedicated Interconnect or Partner Interconnect, depending on the proximity to a co-location facility and the capacity requirements.

---

## Direct Peering and Carrier Peering

![Screen Shot 2021-07-31 at 1.37.19 AM](https://i.imgur.com/J4DCgJg.png)

- all provide **public IP address access** to all of Google's services.
- The main differences are capacity and the requirements for using a service.
- Direct Peering has a capacity of 10 Gbps per link and requires you to have a connection in a GCP edge point of presence.
- Carrier Peerings, capacity and requirements depending on the service provider that you work with.


### Direct Peering

- useful when you require access to Google and Google cloud properties.
- Google allows you to establish a direct peering connection between the business network and Google's.
- With this connection, you will be able to exchange internet traffic between the network and Google's at one of the Googles broad reaching edge network locations.


- Direct Peering
  - exchanging BGP route between Google and peering entity.
  - use it to reach all the Google services, including the full suite of Google cloud platform products.
  - Direct Peering does not have an SLA.

- In order to use direct peering
  - need to satisfy the peering requirements
  - GPS edge `Points of Presence` or PoPs are where Google's network connects to the rest of the internet via peering.
  - PoPs are present on over 90 Internet exchanges and at over 100 interconnection facilities around the world.


### Carrier Peering
- nowhere near one of these locations, consider Carrier Peering.
- If you require access to Google public infrastructure and cannot satisfy Google's peering requirements, you can connect with a Carrier Peering partner.
- Work directly with the service provider to get the connection you need and to understand the partners requirements.
- Carrier Pearing also does not have an SLA.

---


# Cloud VPN


- securely connects the on-premise network to the GCP VPC network through an **IPSec VPN tunnel**.
- Traffic traveling between the two networks is encrypted by one VPN gateway. Then decrypted by the other VPN gateway.
  - protects the data as it travels over the public internet.
  - That's why Cloud VPN is useful for low volume data connections.
- managed service, SLA of `99.9%` service availability
- supports site to site VPN static and dynamic routes, and IKEv1 and IKEv2 ciphers.
- Cloud VPN doesn't support new cases where a client computers need to dial in to a VPN using client VPN software.
- Also, dynamic routes are configured with **Cloud Router**


![Screen Shot 2021-07-31 at 12.16.33 AM](https://i.imgur.com/GK97AFt.png)

a VPN connection between the `VPC `and `on-premise network`.
- the VPC network has subnets in US-east one and US-west one.
  - With GCP resources in each of those regions.
  - These resources are able to communicate using internal IP addresses\
  - routing within a network is automatically configured, as that firewall rules allow it.
- to connect to the on-premise network and its resources
  - configure the `Cloud VPN gateway`, `on-premise VPN gateway` and to `VPN tunnels`.

- The <font color=red> Cloud VPN gateway </font>
  - a regional resource, uses **regional external IP address**.
- the <font color=red> on-premise VPN gateway </font>
  - can be a physical device in the data center or a physical or software based VPN offering in another Cloud providers network.
  - This VPN gateway also has an **external IP address**.
- A VPN tunnel
  - A VPN tunnel then connects the VPN gateways
  - and serves as the virtual medium through which encrypted traffic is passed.
  - to create a connection between two VPN gateways, two VPN tunnels is needed.
    - Each tunnel defines the connection from the perspective of its gateway and traffic can only pass when the pair of tunnels established.

- Now, one thing to remember when using Cloud VPN is that the maximum transmission unit, MTU for the **on-premises VPN gateway** cannot be greater than `1,460 bytes`.
  - because of the encryption and encapsulation of packets.

---

## Cloud VPN gateway

Classic VPN | High-availability (HA) VPN
---|---
Supports dynamic routing and static routing | Supports dynamic routing (BGP) only
No high availability | high availability (99.99 SLA, within region)



## Classic VPN

- support static and dynamic routes.

### dynamic routes

> need to configure **Cloud Router**.

Cloud Router
- manage routes from Cloud VPN tunnel using <font color=blue> border gateway protocol, BGP </font>.
  - routing method
  - allows for routes to be updated and exchanged without changing the tunnel configuration.


For example

![Screen Shot 2021-07-31 at 12.25.57 AM](https://i.imgur.com/l8jfnFD.png)

- two different regional subnets in a VPC network
- The on-premise network has 29 subnets
- the two networks are connected through `Cloud VPN tunnels`.

To automatically propagate network configuration
- changes the VPN tunnel uses **Cloud Router** to establish a `BGP session` between the VPC and the on-premise VPN gateway which must support BGP.
- The new subnets are then seamlessly advertised between networks.
- This means that instances in the new subnets can start sending and receiving traffic immediately
- To set up BGP
  - an additional IP address has to be assigned to each end of the VPN tunnel.
  - These two IP addresses must be link-local IP addresses Belonging to the IP address range `169.254.0.0/16`
  - These addresses are not part of IP address space of either network
  - are used exclusively for establishing a BGP session.



## HA VPN. Alternative Cloud VPN Gateway

- a high availability Cloud VPN solution
- securely connect the on-premises network to the Virtual Private Cloud (VPC) network through an `IPsec VPN connection` in a **single region**
- HA VPN provides an SLA of `99.99%` service availability.


---


## shared VPC


![Screen Shot 2021-07-31 at 1.42.24 AM](https://i.imgur.com/WT7fKCM.png)

![Screen Shot 2021-07-31 at 1.43.32 AM](https://i.imgur.com/Lne7UHT.png)











.
