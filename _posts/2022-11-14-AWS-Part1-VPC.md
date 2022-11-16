---
title: AWS Infra / Part1. VPC
date: 2022-11-14 15:00:00 +0900
categories: [AWS, 2Tier 아키텍쳐 구축]
tags: [AWS, VPC]     # TAG names should always be lowercase
typora-root-url: ../
---
# 글에 앞서..

안녕하세요.  클라우드 프로젝트를 하면서 단계별로 찬찬히 만드는법을 정리하고자 본글을 쓰게 됬습니다.

VPC 생성으로 2 Tier 구축을 해보고, 다음 CI/CD 환경을 구성하는것 까지 파트별로 구성하였습니다.

기본적으로 서울리전에서 최대한 프리티어 옵션으로 작업하는것을 기본으로 하겠습니다.

<br>

# VPC란?

> 직접 정의 가능한 가상 네트워크(Private Network)에서 AWS 리소스를 구동할 수 있는 논리적으로 격리된 네트워크 환경을 제공한다.  - *AWS Document*

라고 정의되어 있습니다.  

쉽게 말하자면 건물이 올라가기 위해 필요한 토지를 만드는 것이라고 생각하면 될거 같습니다.

<br>

# 실전

## VPC 생성

![01](/assets/post/2022-11-14-AWS-Part1-VPC/01.png)

상단의 VPC를 검색하시면 VPC라는 메뉴가 뜨게 됩니다. 클릭해서 들어갑니다.

<br>

![02](/assets/post/2022-11-14-AWS-Part1-VPC/02.png)

위와 같은 대시보드가 뜨게 됩니다. VPC 생성을 누릅니다.

<br>

![03](/assets/post/2022-11-14-AWS-Part1-VPC/03.png)

저같은 경우엔 글을 작성하기 위해 모든 VPC를 삭제한 상태이지만, 기본적으로 하나의 VPC가 생성 되어 있습니다.

지워서 만들어도 되고 그냥 만드셔도됩니다. 대신 다음 항목을 진행할 때 VPC의 Name과 ID를 확인하여야 됩니다.

<br>

![04](/assets/post/2022-11-14-AWS-Part1-VPC/04.png)

이름은 진행했던 클라우드 프로젝트 이름으로 이름을 지정하였습니다. 원하는 이름을 지정해도 상관 없습니다.

CIDR는 규모에 맞게 지정하시면 됩니다. 서브넷 범위(/16)는 꼭 기입해주셔야 됩니다.

<br>

## 결과 확인

![05](/assets/post/2022-11-14-AWS-Part1-VPC/05.png)

VPC는 쉽게 생성이 가능합니다. 위와같이 방금 만든 VPC를 확인할 수 있습니다.

<br>

![06](/assets/post/2022-11-14-AWS-Part1-VPC/06.png)

<br>

# Outro

이제 이렇게 기본적인 VPC를 만드셨습니다. 다음은 [서브넷 생성](/posts/AWS-Part2-Subnet/) 입니다.

