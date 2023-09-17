---
title: Gateway에 대한 개념과 이해 
date: 2023-09-14 20:34:51 +0900
author: kkankkandev
categories: [Network]
tags: [network, gateway, default gateway, centos, flooding, routing, route]     # TAG names should always be lowercase
comments: true
---

Gateway란 Local Network에서 다른 Network로 이동하기 위해 반드시 거쳐야 하는 지점을 말합니다. 

## 2. Gateway의 개념
Gateway는 OSI 모델의 3계층에서 작동하며, IP 패킷의 헤더 정보를 확인하여 목적지 IP주소가 자신의 네트워크에 속하는지 여부를 판단합니다. 목적지 IP 주소가 자신의 네트워크에 속하는 경우 Gateway는 패킷을 목적지 호스트로 직접 전달하고. 목적지 IP 주소가 자신의 네트워크에 속하지 않는 경우 Gateway는 패킷을 다른 네트워크로 전달합니다.
  
  
따라서 Local Network에 있는 Host가 외부네트워크와 통신하기 위해서는 Gateway를 통과해야 하며. Gateway는 서로 다른 네트워크를 연결해줍니다.
  
라우터, 브리지, 터널링 등이 Gateway에 해당하는 장비에 속하며 컴퓨터에서 인터넷으로 데이터를 전송할 때, 컴퓨터는 먼저 Default Gateway 주소로 데이터를 전송합니다.

## 2. Gateway와 Default Gateway는 다른걸까?

네 Gateway와 Default Gateway는 약간 다릅니다. Gateway는 서로 다른 네트워크 간 통신을 가능하게 하고, Dafault Gateway는 로컬 네트워크에서 인터넷으로 데이터를 전송할 때 사용됩니다.

```
# Gateway
서로 다른 네트워크를 연결하는 장치
- 네트워크를 연결하고, 패킷을 전달

# Default Gateway
특정 네트워크의 컴퓨터가 외부 네트워크로 가는 패킷을 전송할 때 사용하는 게이트웨이
- 패킷을 외부 네트워크로 전달
```

<br>

<strong>궁금하신점이나 추가해야할 부분은 댓글이나 아래의 링크를 통해 문의해주세요.</strong>   
Written with [KKam.\_\.Ji](https://www.instagram.com/kkam._.ji/)
