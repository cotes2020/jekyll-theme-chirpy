---
title: AWS Infra / Part8. Elasticashe
date: 2022-11-17 21:00:00 +0900
categories: [AWS, 2Tier 아키텍쳐 구축]
tags: [AWS, Elasticache, Redis]     # TAG names should always be lowercase
typora-root-url: ../
---
# Intro

[이전 파트](/posts/AWS-Part7-AmazonRDS/)에서는 RDS 서비스를 이용해 DB 인스턴스를 설치 해보았습니다.

이번에는 캐싱 서비스인 Elasticache에서 제공하는 Redis 클러스터를 설치해보는 파트 입니다.

<br>

# Elasticache란?

> *Amazon ElastiCache*는 유연한 실시간 사용 사례를 지원하는 완전관리형 인 메모리 캐싱 서비스입니다. [캐싱](https://aws.amazon.com/caching/)에 ElastiCache를 사용하면 애플리케이션 및 데이터베이스 성능을 가속화할 수 있으며, 세션 스토어, 게임 리더보드, 스트리밍 및 분석과 같이 내구성이 필요하지 않는 사용 사례에서는 기본 데이터 스토어로 사용할 수 있습니다. ElastiCache는 Redis 및 Memcached와 호환 가능합니다. - *AWS Document*

Elasticache는 아마존의 캐싱 서비스이며, Redis와 Memcached 호환 클러스터를 제공 합니다.

DB의 캐싱 뿐만 아니라 세션 저장등 여러가지 기능이 있고,

부하 절감 및 응답속도 개선 등의 효과를 가져올 수 있습니다.

이번에는 Elasticache의 Redis 클러스터를 설치해 보겠습니다.

<br>

# 실전

## Redis용 보안 그룹 생성

![01](/assets/post/2022-11-17-AWS-Part8-Elasticache/01.png)

이번에는 위와 같은 보안 그룹을 생성 합니다. **Redis의 기본 포트는 6379** 입니다.

<br>

## Redis용 서브넷 그룹 생성

![02](/assets/post/2022-11-17-AWS-Part8-Elasticache/02.png)상단 검색바에서 **Elasticache**를 검색해서 들어 갑니다.

<br>

![03](/assets/post/2022-11-17-AWS-Part8-Elasticache/03.png)

좌측 사이드바에서 **서브넷 그룹** 을 클릭합니다.

<br>

![04](/assets/post/2022-11-17-AWS-Part8-Elasticache/04.png)

**서브넷 그룹 생성**을 클릭 합니다.

<br>

![05](/assets/post/2022-11-17-AWS-Part8-Elasticache/05.png)

이름을 지정하고 VPC ID를 체크 합니다.

<br>

![06](/assets/post/2022-11-17-AWS-Part8-Elasticache/06.png)

선택된 서브넷이 모든 서브넷인것을 알 수 있습니다.

Redis용으로 만든 서브넷으로 바꿔 보겠습니다.

우측 상단의 **관리** 를 누릅니다.

<br>

![07](/assets/post/2022-11-17-AWS-Part8-Elasticache/07.png)

**x.x.13.x 서브넷**과 **x.x.23.x 서브넷**을 2*-RedisSubNet으로 만들었던 것 기억 하시나요?

팝업창에서 그 두개의 서브넷만 선택한 것을 확인 하고 **선택** 버튼을 누릅니다.

<br>

![08](/assets/post/2022-11-17-AWS-Part8-Elasticache/08.png)

위와 같이 두개의 서브넷이 선택 되어 있으면 됩니다.

**생성** 버튼을 눌러 생성 합니다.

<br>

## Redis Cluster 만들기

![09](/assets/post/2022-11-17-AWS-Part8-Elasticache/09.png)

Elasticache 대시보드에서 ElastiCache 시작하기 란의 클러스터 생성 > **Redis 클러스터 생성**을 클릭 합니다.

<br>

![10](/assets/post/2022-11-17-AWS-Part8-Elasticache/10.png)

생성 방법은 새 클러스터 구성 및 생성을 선택 하고, 클러스터 모드는 이번엔 비활성화됨으로 만들겠습니다.

*(클러스터를 만든다고는 했지만 우선 검증이 목표기 때문에 최대한 돈을 덜 쓰는 쪽으로 옵션을 넣겠습니다.)*

<br>

![11](/assets/post/2022-11-17-AWS-Part8-Elasticache/11.png)

클러스터의 이름을 지정 합니다.

<br>

![12](/assets/post/2022-11-17-AWS-Part8-Elasticache/12.png)

위치는 당연히 클라우드 상에 설치 할 것이므로 AWS 클라우드를 선택 합니다.

<br>

![13](/assets/post/2022-11-17-AWS-Part8-Elasticache/13.png)

클러스터 설정 입니다.

여기서 포트를 변경 할 수 도 있지만 기본 값인 6379로 두겠습니다.

처음 노드 유형은 성능이 좋은 인스턴스로 배치가 되어있는데요, 

낮은 성능의 cache.t2.micro를 선택 합니다.

복제본 갯수는 1로(메인 1개, 복제본 1개. 총 2개가 만들어 집니다.) 지정하겠습니다.

<br>

![14](/assets/post/2022-11-17-AWS-Part8-Elasticache/14.png)

서브넷 그룹은 아까 만들어둔 서브넷을 지정하면 아래와 같이 선택했던 서브넷이 표시 됩니다.

<br>

![15](/assets/post/2022-11-17-AWS-Part8-Elasticache/15.png)

다음 가용영역 배치 입니다. 

가용 영역이 처음에 자동으로 지정 으로 되있기 때문에, 가용 영역 지정을 선택 합니다.

여기서는 프라이머리를 2a, 복제본을 2c로 배치 하겠습니다.

그 후 스크롤을 내려 **다음** 버튼을 누릅니다.

<br>

![16](/assets/post/2022-11-17-AWS-Part8-Elasticache/16.png)

보안 그룹 선택 입니다. 우상단의 관리를 눌러 팝업이 뜨면

아까 만들어 두었던 Redis용 보안 그룹을 고르고 **선택** 버튼을 누릅니다.

<br>

![17](/assets/post/2022-11-17-AWS-Part8-Elasticache/17.png)

이번에는 테스트 구축이 목표므로 최대한 과금 될만한 옵션은 끄도록 하겠습니다.

그후 스크롤을 내려 **다음** 버튼을 누릅니다.

<br>

![18](/assets/post/2022-11-17-AWS-Part8-Elasticache/18.png)

검토 및 생성 단계 입니다. 이제까지 설정했던 내역이 뜨게 됩니다.

문제가 없으면 맨 밑으로 스크롤을 내려 **생성**을 클릭 합니다.

<br>

# 결과 확인

![19](/assets/post/2022-11-17-AWS-Part8-Elasticache/19.png)

Redis 클러스터가 생성 되었습니다. 이름을 눌러 세부 정보로 들어 갑니다.

<br>

![20](/assets/post/2022-11-17-AWS-Part8-Elasticache/20.png)

RDS와 마찬가지로 IP가 아닌 주소로 엔드포인트가 표시 됩니다.

읽기 전용 주소와 일반 주소로 분리되어 있는 것을 알 수 있습니다.

백엔드 연동시 위의 주소를 기반으로 작업 하시면 됩니다.

<br>

![21](/assets/post/2022-11-17-AWS-Part8-Elasticache/21.png)

이번에는 위와 같이 배치 하였습니다. 

추후의 파트를 위해 2c-PrivSubNet 구역에 만들어 두었던 인스턴스는 삭제하도록 하겠습니다. 

<br>

# Outro

이번 파트에서는 Amazon Elasticache의 서비스를 이용하여 Redis 클러스터를 생성 하였습니다.

[다음 파트](/posts/AWS-Part9-InstanceSetting/)는 Private 서브넷에 있는 EC2 인스턴스에 대한 사전 설정 파트 입니다.
