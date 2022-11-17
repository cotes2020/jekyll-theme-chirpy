---
title: AWS Infra / Part7. Amazon RDS
date: 2022-11-17 17:00:00 +0900
categories: [AWS, 2Tier 아키텍쳐 구축]
tags: [AWS, Amazon RDS, Mysql]     # TAG names should always be lowercase
typora-root-url: ../
---
# Intro

[이전 파트인 Bastion Host 만들기 파트](/posts/AWS-Part6-BastionHost/)에서 Bastion Host 및 테스트용 인스턴스를 만들어보고,

연결 테스트까지 진행 하였습니다. 이번에는 Amazon RDS를 생성해보도록 하겠습니다.

<br>

# Amazon RDS란?

> *Amazon Relational Database Service(RDS)*는 클라우드에서 간편하게 데이터베이스를 설치, 운영 및 확장할 수 있는 관리형 서비스 모음입니다. [Amazon Aurora(MySQL 호환)](https://aws.amazon.com/ko/rds/aurora/?pg=ln&sec=hiw), [Amazon Aurora(PostgreSQL 호환)](https://aws.amazon.com/ko/rds/aurora/?pg=ln&sec=hiw), [MySQL](https://aws.amazon.com/ko/rds/mysql/?pg=ln&sec=hiw), [MariaDB](https://aws.amazon.com/ko/rds/mariadb/?pg=ln&sec=hiw), [PostgreSQL](https://aws.amazon.com/ko/rds/postgresql/?pg=ln&sec=hiw), [Oracle](https://aws.amazon.com/ko/rds/oracle/?pg=ln&sec=hiw) 및 [SQL Server](https://aws.amazon.com/ko/rds/sqlserver/?pg=ln&sec=hiw)의 7가지 주요 엔진 중에서 선택하고 [Amazon RDS on AWS Outposts](https://aws.amazon.com/ko/rds/outposts/?pg=ln&sec=hiw)를 통해 온프레미스에 배포할 수 있습니다. - *AWS Document*

Amazon RDS(이후 부터는 RDS라 칭하겠습니다.)는 DB서버 입니다.

여러가지 옵션들을 지원 합니다. 이번 파트에서는 MySQL 서버를 만들 것 입니다.

<br>

# 실전

## RDS용 보안 그룹 생성

![01](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/01.png)

위 사진 같은 인바운드 규칙을 가지는 DBSecGroup이라는 보안 그룹을 생성 하겠습니다.

소스는 각각 Bastion Host와 Private Host용으로 만든 보안 그룹을 지정 하면 됩니다.

보안 그룹 만드는 법을 잊으셨나요? [이전 글에서 확인](/posts/AWS-Part6-BastionHost/#보안그룹-생성하기) 해 주세요

<br>

## RDS용 서브넷 그룹 생성

상단 검색바에서 **RDS**를 검색해서 들어 갑니다.

<br>

![03](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/03.png)

RDS 를 생성하기 전에 서브넷 그룹을 만들어야 됩니다. 왼쪽 사이드바의 **서브넷 그룹**을 클릭 합니다.

<br>

![04](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/04.png)

**DB 서브넷 그룹 생성**을 클릭 합니다.

<br>

![05](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/05.png)

이름과 설명을 작성 하고 VPC를 잘 확인 합니다.

<br>

![06](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/06.png)

가용 영역은 **2a와 2c를 선택** 합니다.

서브넷은 DB용 서브넷으로 만들었던 서브넷을 선택합니다. 

앞서 만들었던 환경에서는 **x.x.12.x 서브넷**과 **x.x.22.x 서브넷**이 해당 됩니다.

여기서는 서브넷 이름이 안보여서 CIDR 지정값을 보고 선택해야되네요.

이후 **생성** 버튼을 눌러 DB 서브넷 그룹을 생성 합니다.

<br>

## 메인 RDS 생성

![02](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/02.png)

![07](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/07.png)

이후 대시보드로 다시 돌아가 **데이터베이스 생성**을 클릭 합니다.

<br>

![08](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/08.png)

이번에는 MySQL 서버로 만들어보도록 하겠습니다.

Aurora도 사용해보고 싶지만 이번엔 프리티어 환경에서 구현하는게 목표기 때문에 MySQL로 선정했습니다.

<br>

![09](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/09.png)

템플릿은 프리티어로 선택 합니다.

<br>

![10](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/10.png)

인스턴스 식별자는 RDS 서버의 이름입니다. 하나를 더 만들기 때문에 1이라는 식별자를 추가하였습니다.

자격 증명 설정에서 사용되는 사용자 이름과 암호는 **실제로 DB에 로그인 하기위한 마스터 유저 정보** 입니다.

작성하고 넘어갑니다.

<br>

![11](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/11.png)

프리티어의 경우에는 RDS로 생성할 수 있는 인스턴스에 제한을 받습니다.

테스트만 할 것이라 제일 낮은 등급의 프리티어 인스턴스를 사용하였습니다.

<br>

![12](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/12.png)

마찬가지로 스토리지도 제일 낮은 용량을 넣었습니다.

저같은 경우에는 스토리지 자동 조정 활성화 기능도 체크 해제 했습니다.

<br>

![13](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/13.png)

연결 파트 입니다. 먼저 새로고침을 누르면 어느정도 만들어둔 환경에 대해 미리 선택이 되는것 같습습니다.

VPC를 확인 합니다. 그후 DB 서브넷 그룹이 아까 만들어 둔 그룹인지 확인 합니다.

아니라면 선택 합니다.

<br>

![14](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/14.png)

퍼블릭 엑세스는 아니요를 선택 합니다. 

예를 누르면 공개된 서브넷의 경우 자동발급되는 DNS 주소를 통해 접근이 가능합니다.

보안 그룹은 기존 항목 선택에서 아까 만든 RDS 용 그룹을 선택 합니다.

기본으로 선택 되어 있는 default 그룹은 X를 눌러 제거 합니다.

지금 만드는 RDS 인스턴스는 가용영역 2a에 배치해 보겠습니다.

<br>

![15](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/15.png)

DB서버에 접근 할 수 있는 방법을 지정하는곳 입니다.

이번에는 암호 인증으로 접근을 해보겠습니다.

암호 인증을 체크 한 후 스크롤을 내려 **데이터베이스 생성**을 클릭 합니다.

<br>

## 읽기 전용 RDS 생성

![16](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/16.png)

만들어진 RDS 인스턴스를 선택 후 작업 > **읽기 전용 복제본 생성**을 클릭 합니다.

<br>

![17](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/17.png)

여러곳의 AZ에 배포할 수 있는 옵션이 있습니다만 이번에는 그냥 넘어가겠습니다.

<br>

![18](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/18.png)

서브넷 그룹을 확인 해서 아까 만든 서브넷 그룹인지 확인 합니다.

가용영역은 이번엔 2c를 선택 하겠습니다.

퍼블릭 액세스는 당연히 아니오를 선택 합니다.

보안 그룹도 제대로 되어있는지 확인 합니다.

(부모 DB의 정보를 계승하기 때문에 왠만하면 다 맞긴 합니다.)

<br>

![19](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/19.png)

이름은 이번엔 2번 식별자를 부여했습니다.

나머지 옵션은 건드리지 말고 스크롤을 내려 **읽기 전용 복제본 생성**을 클릭 하여 생성 합니다.

<br>

# 결과 확인

![20](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/20.png)

위와 같이 Read Replica 구성이 되어 있습니다. 1번 RDS 인스턴스를 클릭 해보겠습니다.

<br>

![21](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/21.png)

상세정보를 볼수있는 창으로 전환 됩니다. 

연결 & 보안 탭에서 **엔드포인트에 있는 주소**가 DB로 접근 할 수 있는 주소 입니다.

접속을 시도해 보겠습니다.

<br>

![22](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/22.png)

```shell
mysql -h [RDS Endpoint Address] -u [앞서 생성한 마스터 계정명] -p
Enter password : [앞서 생성한 마스터 패스워드]
```

Bastion Host에서 Private 서브넷에 있는 인스턴스에 원격 접속하여 접속을 시도해 봤습니다.

RDS로 접속이 잘 되는 것을 볼 수 있습니다.

<br>

![23](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/23.png)

이번에도 접속 시도를 해보았습니다. 차이가 보이시나요?

Bastion Host에게도 접속이 개방되어 있어 Bastion Host로도 접속이 가능 한 것을 볼 수 있습니다.

<br>

![24](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/24.png)

이번엔 1번 DB에서 테스트용 DB를 생성 해보겠습니다.

```mysql
create database TEST;
show databases;
```

간단한 명령어로 TEST라는 데이터베이스를 생성 했습니다.

<br>

![25](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/25.png)

이번엔 읽기 전용 복제본인 2번 인스턴스에 접속 하여 데이터베이스를 조회해보면 

TEST라는 Database가 존재하는걸 볼 수 있습니다.

<br>

![26](/assets/post/2022-11-17-AWS-Part7-AmazonRDS/26.png)

현재 이렇게 배치 되었습니다.

<br>

# Outro

이번 파트에서는 Amazon RDS라는 DB 서버 인스턴스를 생성하여 배치 했습니다.

Read Replica 구성을 하여 서비스 분산에 대한 대비(완벽하지는 않습니다)도 해보았습니다.

[다음 파트는 Redis 클러스터를 생성하여 배치하기 위한 Elasticache](/posts/AWS-Part8-Elasticache/)를 알아보겠습니다.