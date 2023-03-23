---
title: AWS - Network - BadNetwork
date: 2020-07-18 11:11:11 -0400
categories: [01AWS, AWSNetwork]
tags: [AWS, Network, VPC]
toc: true
image:
---

- [BadNetwork](#badnetwork)
  - [Misconfiguration 1: Unnecessary Service Exposure](#misconfiguration-1-unnecessary-service-exposure)
  - [Misconfiguration 2: Soft Center](#misconfiguration-2-soft-center)
  - [Misconfiguration 3: Bad Failover](#misconfiguration-3-bad-failover)
  - [Misconfiguration 4: Typo in Security Group](#misconfiguration-4-typo-in-security-group)

- ref
  - [Spotting Misconfigurations With CloudMapper](https://duo.com/blog/spotting-misconfigurations-with-cloudmapper)

---

# BadNetwork


---


## Misconfiguration 1: Unnecessary Service Exposure

![m1-demo-all-exposed](https://i.imgur.com/nG6RPDH.png)

**all EC2s can connect to the ELBs which are public.**
- the "internal" web servers and databases can all be reached from 0.0.0.0/0, which means the public internet or anywhere.
- the bastion host really isn't providing much value here, because you can just connect directly to any of the systems.
- all of the EC2 instances can connect back to the ELBs. needlessly complex.
- In reality, everything can talk to everything, so a more accurate representation of the graph would be

![m1-mod-demo-all-exposed](https://i.imgur.com/HeqH8f7.png)

![m1-ec2-demo](https://i.imgur.com/G5gymM4.png)

---

## Misconfiguration 2: Soft Center

![m2-demo-soft-center](https://i.imgur.com/crDO901.png)

**All resources can communicate with each other**
- default security group that allows access from that same security group and then applied this to all resources.
- The result is that only a few resources are public, which is good, but everything inside the network can talk to everything else.
- This network configuration can be bad because if an attacker gets inside the network they may be able to more easily move laterally to any other system. This rat's nest looking diagram can usually be spotted before the visualization is even generated because the "prepare" step of CloudMapper will show "n" nodes and roughly "2(n2)" connections.


---

## Misconfiguration 3: Bad Failover

![m3-demo-bad-failover](https://i.imgur.com/Apk6wdr.png)

**Architecture that will not be resilient to AZ failover**
- have availability zone failover, but part of the architecture will not be resilient.
- Multiple ELBs and RDS instances were set up, one in each AZ, but the EC2 running the web server only exists in one AZ.
- an "unbalanced" architecture that straddles multiple AZs or regions can sometimes be more easily spotted visually.

---

## Misconfiguration 4: Typo in Security Group

**Security Group is accidentally open to a /2 instead of a /32**
- instead of an external CIDR being labeled "SF Office", it has been labeled "1.1.1.1/2". The reason is that although the known CIDR for the SF Office was configured as "1.1.1.1/32", the Security Group has a typo that accidentally allows in anything in the whole "/2".

- The result is that instead of 1 IP being granted access, roughly one billion IP addresses have been granted access.


![m4-demo-typo](https://i.imgur.com/9187m0q.png)








.
