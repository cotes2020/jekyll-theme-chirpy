


configure your VPCs in several ways, and take advantage of numerous connectivity options and gateways.
- These options and gateways include 
  - AWS Direct Connect (via DX gateways), 
  - NAT gateways, 
  - internet gateways, 
  - VPC peering, etc.


hundreds of VPCs distributed across AWS accounts and Regions to serve multiple lines of business, teams, projects, and get complex to set up connectivity between VPCs.


VPC peering 
- All the connectivity options are strictly point-to-point, so the number of VPC-to-VPC connections can grow quickly.
  - As grow the number of workloads run on AWS
  - scale your networks across multiple accounts and VPCs to keep up with the growth.
  - Though you can use VPC peering to connect pairs of VPCs, 
    - managing point-to-point connectivity across many VPCs 
    - without the ability to centrally manage the connectivity policies 
    - operationally costly and difficult.

  - For on-premises connectivity, you must attach your VPN to each individual VPC. This solution can be time-consuming to build and difficult to manage when the number of VPCs grows into the hundreds.

AWS Transit Gateway
simplify your networking model.
- only need to create and manage a single connection from 
  - the central gateway into each VPC, 
  - on-premises data center, 
  - or remote office across your network.
- acts as a hub, hub-and-spoke model
  - controls how traffic is routed among all the connected networks, which act like spokes.
  - simplifies management and reduces operational costs 
    - because each network only needs to connect to the transit gateway and not to every other network.
    - Any new VPC is connected to the transit gateway, then automatically available to other network connected to the transit gateway.
  - easier to scale your network as you grow.
￼
![AWS Transit Gateway](https://i.imgur.com/4QfbqaR.png)
