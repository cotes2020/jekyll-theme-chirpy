---
title: NetworkSec - Layer 3 - Autonomous System (AS)
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

# Autonomous System (AS)

`AS` autonomous system 自治系統
- a network under a single administrative control.
- Assumes the Internet is an arbitrarily interconnected set of AS's.
  - **local traffic**: traffic that `originates at or terminates on nodes within an AS`
  - **transit traffic**: traffic that `passes through an AS`.

classify AS's into 3 types:
- **Stub AS**:
  - an AS that has only a single connection to one other AS;
  - such an AS will only carry local traffic
  - (small corporation)
- **Multihomed AS**:
  - an AS that has connections to more than one other AS, but refuses to carry transit traffic
  - (large corporation at the top)
- **Transit AS**:
  - an AS that has connections to more than one other AS,
  - and is designed to carry both transit and local traffic
  - (backbone providers)


- 在每一個自治系統中，
  - 會有一個 `Backbone Area` 網路
    - 與外部路由網路互相連接的區段
    - 負責外部路由網路與自治系統內部其他網路的溝通，
    - 也是Transition Area。 
  - 除了Backbone Area外，自治系統中其他的網路區段就是 `Non-Backbone Area`。
    - 自治系統中所有的Non-Backbone Area都必須連接到Backbone Area上。

- 各個路由器扮演的角色 
  - `Backbone Area` 中:
    - `Autonomous System Boundary Router (ASBR)`
      - 自治系統邊界router.
      - 用來連接外部路由網域和自治系統。 
      - OSPF: Backbone Router.
      - IS-IS: L2 Router。
    - Normal Router
      - 提供不同Area之間的連接性。 

  - `Non-Backbone Area`:
    - Normal Router
      - 用於連接不同的Area，
      - 維護所連接的Area的Link-State路由資料庫，
      - 也負責轉送封包到其他的Area
    - OSPF: Area Border Routers(ABRs)
    - IS-IS: L1/L2 Routers。

自治系統編號 
- 分辨不同的自治系統
  - 透過 `自治系統編號(ASN)` 來區別。
  - 世界上多個組織可以使用自己私有的自治系統編號，以便於和他們的ISP業者之間透過BGP協定連線，
- 因此，自治系統編號又分為:
  - 私有的自治系統編號,
  - 公有並透過註冊之後的自治系統編號.
- 每個ISP業者必須登記公開至少一個ASN，用於BGP協定。
  - ASN極重要，是等一下用於路由協定設定時辨識自治系統的重要條件, 是識別各個網路的指標。 
- 而如何分辨公開註冊ASN和私有ASN呢？
  - `IANA（Internet Assigned Number Authority)` 使用16位元的長度來儲存自治系統編號（2的16次方).
    - 64512到65535之間的編號: 保留給私有自治系統所使用，
    - 1到64511之間的號碼: 公開註冊的自治系統編號


---


# Different Routing Protocols

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









.
