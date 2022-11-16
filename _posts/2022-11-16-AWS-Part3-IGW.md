---
title: AWS Infra / Part3. IGW(인터넷 게이트웨이)
date: 2022-11-16 14:00:00 +0900
categories: [AWS, 2Tier 아키텍쳐 구축]
tags: [AWS, IGW]     # TAG names should always be lowercase
typora-root-url: ../
---
# Intro

[저번 글](/posts/AWS-Part2-Subnet/)에서는 만든 VPC에 구역을 나누는 서브네팅을 진행하였습니다.

이번에는 VPC와 외부와 통신하기 위한 IGW(인터넷 게이트웨이)를 배치해 보겠습니다.

<br>

# IGW(인터넷 게이트웨이)이란?

> *인터넷 게이트웨이*는 수평 확장되고 가용성이 높은 중복 VPC 구성 요소로, VPC와 인터넷 간에 통신할 수 있게 해줍니다. IPv4 트래픽 및 IPv6 트래픽을 지원합니다. 네트워크 트래픽에 가용성 위험이나 대역폭 제약이 발생하지 않습니다. - *AWS Document*

인터넷 게이트웨이는 VPC와 인터넷 환경을 이어주는 라우터(집으로 따지면 공유기)의 역할을 하는 요소 입니다.

퍼블릭 IP가 부여된 VPC내의 요소들은 IGW를 통해 외부 인터넷과 통신 할 수 있습니다.

<br>

# 실전

## IGW 생성

![01](/assets/post/2022-11-16-AWS-Part3-IGW/01.png)

VPC 대시보드에서 **인터넷 게이트웨이**를 클릭 합니다.

<br>

![02](/assets/post/2022-11-16-AWS-Part3-IGW/02.png)

새로 생성한 VPC에는 인터넷 게이트웨이가 없습니다. **인터넷 게이트웨이 생성**을 클릭합니다.

<br>

![03](/assets/post/2022-11-16-AWS-Part3-IGW/03.png)

인터넷 게이트웨이의 이름을  지정하고 **인터넷 게이트웨이 생성** 을 클릭합니다.

<br>

![04](/assets/post/2022-11-16-AWS-Part3-IGW/04.png)

다시 인터넷 게이트웨이 메뉴로 돌아가 확인을 해보면 IGW가 만들어져 있습니다.

하지만 VPC와 실질적인 연결이 안된 상태 입니다.

위 사진을 보시면 "Detached(연결이 안된)"라는 상태를 확인 할 수 있습니다.

만들어진 인터넷 게이트웨이를 클릭후, 위의 작업 > **VPC에 연결**을 클릭 합니다.

<br>

![05](/assets/post/2022-11-16-AWS-Part3-IGW/05.png)

연결할 VPC의 이름과 ID를 확인 후 알맞는 VPC를 고르고, **인터넷 게이트웨이 연결**을 클릭합니다.

<br>

## 결과 확인

![06](/assets/post/2022-11-16-AWS-Part3-IGW/06.png)

상단에 인터넷 게이트웨이와 VPC가 연결 되었다는 메세지가 뜨게 됩니다.

<br>

![07](/assets/post/2022-11-16-AWS-Part3-IGW/07.png)

현재는 위와 같이 배치가 되었습니다.

<br>

# Outro

이번에는 VPC와 외부 인터넷이 연결되기 위한 인터넷 게이트웨이를 배치하였습니다.

[지난 파트인 서브넷 파트](/posts/AWS-Part2-Subnet/)에서 서브넷을 Public/Private로 나누었던것 기억 하실 텐데요.

[다음 파트인 라우팅 테이블 설정](/posts/AWS-Part4-RoutingTable/)에서 실질적으로 구분을 하게 됩니다.