---
title: AWS Infra / Part10. Amazon EFS
date: 2022-12-02 19:00:00 +0900
categories: [AWS, 2Tier 아키텍쳐 구축]
tags: [AWS, EFS, NFS]     # TAG names should always be lowercase
typora-root-url: ../
---
# Intro

[저번 파트](/posts/AWS-Part9-InstanceSetting/) 에서는 인스턴스에 설정을 위한 예제 git을 클론해와서 기초적인 설정 일부를 진행하였습니다.

이번에는 Amazon EFS를 배치해 보겠습니다.

<br>

# Amazon EFS?

> *Amazon Elastic File System(Amazon EFS)*은 AWS 클라우드 서비스 또는 온프레미스 리소스와 함께 사용되는 범용 워크로드를 위한 단순하고, 조정 가능하며, 탄력적인 파일 시스템을 제공합니다.  \- *AWS Document*

Amazon EFS는 저장소 역할을 수행합니다.

추후 진행 시 인스턴스가 자동으로 늘고 줄어드는 Auto Scaling이 진행 되는데,

변동되는 파일들은 같은 저장소를 바라보게 하여 동일한 경험을 하게 하고자 합니다.

EFS에는 동적인 파일을 담을 예정 입니다.

<br>

# 실전

![01](/assets/post/2022-12-02-AWS-Part10-AmazonEFS/01.png)

내부의 인스턴스를 바라보는 **NFS** 정책 보안그룹을 위와 같이 생성 합니다.

EFS는 **NFS 프로토콜**을 사용 합니다.

<br>

![02](/assets/post/2022-12-02-AWS-Part10-AmazonEFS/02.png)

상단 검색바에서 **EFS**를 검색해 나오는 메뉴를 클릭 합니다.

<br>

![03](/assets/post/2022-12-02-AWS-Part10-AmazonEFS/03.png)

좌측 사이드바의 **파일 시스템**을 클릭 합니다.

<br>

![04](/assets/post/2022-12-02-AWS-Part10-AmazonEFS/04.png)

**파일 시스템 생성**을 클릭 합니다.

<br>

![05](/assets/post/2022-12-02-AWS-Part10-AmazonEFS/05.png)

이후 나오는 창에서 **사용자 지정** 을 클릭 합니다.

<br>

![06](/assets/post/2022-12-02-AWS-Part10-AmazonEFS/06.png)

이름을 지정 하고 다음을 클릭 합니다.

저 같은 경우에는 혹시나 모를 과금이 있을까봐 자동 백업 활성화를 체크 해제 했습니다.

<br>

![07](/assets/post/2022-12-02-AWS-Part10-AmazonEFS/07.png)

보안 그룹은 처음에 생성해두었던 **EFS용 보안 그룹을 선택** 합니다.

EFS는 멀티리전으로 생성 되기 때문에 서브넷도 선택해야 됩니다.

여기서는 각 가용영역의 **PrivSubNet으로 지정** 하고 다음을 누릅니다.

<br>

![08](/assets/post/2022-12-02-AWS-Part10-AmazonEFS/08.png)

추가로 정책을 설정 할 수 있습니다.

필요시 적용 후 **다음** 버튼을 누릅니다.

<br>

![09](/assets/post/2022-12-02-AWS-Part10-AmazonEFS/09.png)

검토 및 생성 단계로 이동하게 됩니다.

설정 확인 후 스크롤을 내려 **생성** 버튼을 누릅니다.

<br>

![10](/assets/post/2022-12-02-AWS-Part10-AmazonEFS/10.png)

생성이 된 모습 입니다. 이름의 링크를 눌러서 상세정보 보기로 이동 합니다.

<br>

![11](/assets/post/2022-12-02-AWS-Part10-AmazonEFS/11.png)

여기서 나오는 DNS 이름의 주소로 접근 할 수 있습니다.

바로 접근은 안되고 DNS로 접근 할 수 있게 설정을 변경 해 주어야 합니다.

<br>

![12](/assets/post/2022-12-02-AWS-Part10-AmazonEFS/12.png)

VPC 대시보드로 이동해 **VPC 메뉴**로 이동 합니다.

그 후 현재 VPC를 선택 하고 작업 > **VPC 설정 편집** 으로 이동 합니다.

<br>

![13](/assets/post/2022-12-02-AWS-Part10-AmazonEFS/13.png)

DNS 설정의 두가지 체크 박스를 선택한 상태로 **저장** 버튼을 누릅니다.

<br>

# 결과 확인

![14](/assets/post/2022-12-02-AWS-Part10-AmazonEFS/14.png)

위와 같이 EFS 파일 시스템을 배치 하였습니다.

<br>

# Outro

이번 파트 에서는 Amazon EFS를 생성하였습니다. [다음 파트는 Amazon S3 설정](/posts/AWS-Part11-AmazonS3/) 입니다.

EFS와 S3을 생성 한 다음 인스턴스에 실제로 반영하는 작업을 진행 할 예정 입니다.
