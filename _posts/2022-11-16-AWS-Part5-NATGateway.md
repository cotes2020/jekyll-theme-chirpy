---
title: AWS Infra / Part5. NAT Gateway
date: 2022-11-16 15:50:00 +0900
categories: [AWS, 2Tier 아키텍쳐 구축]
tags: [AWS, NAT Gateway]     # TAG names should always be lowercase
typora-root-url: ../
---
# Intro

[저번 파트인 라우팅 테이블](/posts/AWS-Part4-RoutingTable/)에서는 논리적으로 서브넷들을 구분하는 작업을 하였습니다.

하지만 Private 서브넷이 한정적인 외부통신을 할 수 있게 하기 위해서는 NAT Gateway가 필요합니다.

<br>

# NAT Gateway란?

> *NAT 게이트웨이*는 NAT(네트워크 주소 변환) 서비스입니다. 프라이빗 서브넷의 인스턴스가 VPC 외부의 서비스에 연결할 수 있지만 외부 서비스에서 이러한 인스턴스와의 연결을 시작할 수 없도록 NAT 게이트웨이를 사용할 수 있습니다. - *AWS Document*

NAT Gateway는 [공유기의 NAT 기술과 비슷한 역할](https://namu.wiki/w/NAT)(나무위키 링크)을 수행 합니다.

공유기에 물린 PC는 외부 통신을 할 수 있지만, 외부에서 직접적인 접근은 불가능 합니다.

위 처럼 외부 접근을 한정적으로만 가능하게 하여 서브넷을 외부로부터 격리할 수 있게 해주는게 목적 입니다.

그럼 이처럼 생각 할 수도 있습니다.

NAT Gateway와 인터넷 게이트웨이를 생성하고 나서 라우팅 테이블을 만들면 되지 않느냐

라고 말씀하실 수 도 있는데요. 맞습니다. 그렇게 하셔도 됩니다.

하지만 이번에는 Private 서브넷을 만들어야 할때는 NAT Gateway가 꼭 필요하다는 것을 강조하기 위해

순서를 위와 같이 작성 하였습니다.

<br>

# 실전

## NAT Gateway 생성

![01](/assets/post/2022-11-16-AWS-Part5-NATGateway/01.png)

 VPC 대시보드에서 **NAT 게이트웨이**를 클릭 합니다.

<br>

![02](/assets/post/2022-11-16-AWS-Part5-NATGateway/02.png)

**NAT 게이트웨이 생성** 을 클릭합니다.

<br>

![03](/assets/post/2022-11-16-AWS-Part5-NATGateway/03.png)

이름을 기입하고 서브넷을 선택합니다. 저는 아래와 같이 지정 하였습니다.

<br>

|     이름     |    서브넷    |
| :----------: | :----------: |
| KGUnivNAT-01 | 2a-PubSubNet |
| KGUnivNAT-02 | 2c-PubSubNet |

<br>

연결 유형은 퍼블릭로 지정하고, **탄력적 IP 할당**을 클릭 합니다.

NAT Gateway가 바깥과 통신을 하기 위한 공인 IP 지정을 하는 단계 입니다.

그 후 아래의 **NAT 게이트웨이 생성**을 누릅니다.

이 과정으로 총 두개의 NAT Gateway를 생성 합니다.

두개를 만드는 이유는 모종의 이유로 하나의 가용영역(AZ)이 사용 불가능할 때, 가용성을 위해 두개를 배치했습니다.

<br>![04](/assets/post/2022-11-16-AWS-Part5-NATGateway/04.png)

위와 같이 두개의 NAT Gateway가 생성 되었습니다.

상태에 Pending이 뜰 경우에는 탄력적 IP가 아직 연동이 덜 되서 그런것 입니다.

시간이 지나면 위와 같이 Available로 표시 됩니다.

<br>

## Private 서브넷 라우팅 테이블과 연동하기

![05](/assets/post/2022-11-16-AWS-Part5-NATGateway/05.png)

라우팅 테이블로 돌아가 **PrivateSNa 라우팅 테이블의 ID**를 클릭 합니다.

<br>

![06](/assets/post/2022-11-16-AWS-Part5-NATGateway/06.png)

**라우팅** 탭의 **라우팅 편집** 버튼을 누릅니다.

<br>

![07](/assets/post/2022-11-16-AWS-Part5-NATGateway/07.png)

처음 대상(Destination)에서는 **0.0.0.0/0** 을 선택 합니다.

두번째 대상(Target)에서는 **NAT 게이트웨이를 선택** 후 **2a 가용영역에 있는** NAT Gateway를 선택 합니다.

그 후 **변경 사항 저장**을 클릭 합니다.

**PrivateSNc의 라우팅 테이블**도 똑같이 진행(NAT Gateway는 **2c 가용영역에 있는것을 선택**) 합니다.

<br>

## 결과 확인

![08](/assets/post/2022-11-16-AWS-Part5-NATGateway/08.png)

위와 같이 라우팅 테이블과 NAT Gateway와의 연동도 끝났습니다.

<br>

![09](/assets/post/2022-11-16-AWS-Part5-NATGateway/09.png)

현재 위와 같이 진행이 되었습니다.

<br>

# Outro

이번 파트에서는 Private 서브넷에서 한정적인 외부통신을 위해서 NAT Gateway를 설치 하였습니다.

인터넷 게이트웨이와는 다르게 **NAT Gateway를 통한 트래픽은 과금이 진행 되는 부분**입니다.

사용한 양 만큼 과금이 되니 알아두셔야 됩니다.

다음 파트는 [배스천 호스트 만들기](/posts/AWS-Part6-BastionHost/) 입니다.