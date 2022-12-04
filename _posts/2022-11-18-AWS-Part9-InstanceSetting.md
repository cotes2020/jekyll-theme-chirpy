---
title: AWS Infra / Part9. Instance 세팅 1부
date: 2022-11-18 15:00:00 +0900
categories: [AWS, 2Tier 아키텍쳐 구축]
tags: [AWS, Docker]     # TAG names should always be lowercase
typora-root-url: ../
---
# Intro

[이전 파트](/posts/AWS-Part8-Elasticache/)에서는 Elasticache 의 서비스를 이용해 Redis 클러스터를 설치해 보았습니다.

![01](/assets/post/2022-11-18-AWS-Part9-InstanceSetting/01.png)

시작 하기 전에 지난 파트 끝에 말한대로 가용 영역 2c에 있는 PrivSubNet의 인스턴스는 삭제하겠습니다.

이번 파트에서는 인스턴스에 도커 환경을 구성하고, EFS 연동을 위한 컴포넌트를 설치 할 것입니다.

파트 진행 순서는 Instance 세팅 1부 - EFS 구성 - S3 구성 - Instance 세팅 2부 순으로 진행 될 예정 입니다.

<br>

# 어떻게 구성 하지?

2a-PrivSubNet에 있는 인스턴스를 웹 서버로 만들기 위해,

이번에는 도커를 설치하고, 커스텀 django 이미지를 만들 것 입니다.

그 후 Amazon의 저장 서비스인 Amazon S3와 Amazon EFS에 대한 연동 준비를 하겠습니다.

인스턴스 세팅은 실제로 다음 파트에서 이어질 S3과 EFS에서 계속 이어질 예정이기 때문에

이번에는 제가 짜놓은 스크립트에 대해서 설명 위주로 설명 됩니다.

**참고 : 모든 작업은 Ubuntu OS 기반에서 작업 되었습니다.**

<br>

# 실전

![02](/assets/post/2022-11-18-AWS-Part9-InstanceSetting/02.png)

우선 이 파트에서는 [저의 레포지토리 기반](https://github.com/gitryk/portfolio-docker)으로 설명이 진행 됩니다.

<br>

![03](/assets/post/2022-11-18-AWS-Part9-InstanceSetting/03.png)

```shell
git clone https://github.com/tryklab/portfolio-docker.git
```

Bastion Host로 내부 인스턴스로 접속하고, 위 의 명령어를 이용해 레포지토리를 복사해 오겠습니다.

완료 되면 위와 같이 레포지토리가 복제 됩니다.

<br>

![04](/assets/post/2022-11-18-AWS-Part9-InstanceSetting/04.png)

만들어진 폴더로 접근하면 레포지토리의 파일 그대로 복사가 완료 된 것을 볼 수 있습니다.

```shell
chmod + run_me.sh
./run_me.sh
```

위의 명령어를 쳐서 스크립트를 실행 합니다.

1번 코드는 실행 권한을, 두번째 라인은 실행하는 명령어 입니다.

<br>

![05](/assets/post/2022-11-18-AWS-Part9-InstanceSetting/05.png)

가끔씩 이런 화면이 뜨면 엔터를 누르시면 됩니다.

<br>

![06](/assets/post/2022-11-18-AWS-Part9-InstanceSetting/06.png)

뭔가 쭉 지나가더니 끝날 겁니다(마지막 빨간 글씨는 신경 안 쓰셔도 됩니다). 필요한 기본 세팅은 끝났습니다.

이번 파트에서는 실질적으로 할 것은 여기까지 입니다.

하지만 이렇게만 넘어가면 안되겠죠. 스크립트 코드를 보고 무엇을 했는지 알아보는 시간을 가지겠습니다.

<br>

# 스크립트 확인

```shell
mkdir -p portainer/data
mkdir django
sudo apt update && sudo apt upgrade -y
sudo apt-get install -y net-tools vim make binutils
sudo apt-get install ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o \
        /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
sudo apt-get install -y docker-compose
sudo systemctl enable docker
sudo systemctl start docker
sudo docker network create web-server
git clone https://github.com/aws/efs-utils
cd efs-utils
./build-deb.sh
sudo apt install -y ./build/amazon-efs-utils*deb
cd ..
```

run_me.sh의 내용물 입니다. 천천히 볼까요?

<br>

```shell
mkdir -p portainer/data
mkdir django
```

첫 두줄은 작업 공간을 위한 폴더를 생성 하는 코드 입니다.

<br>

```shell
sudo apt update && sudo apt upgrade -y
sudo apt-get install -y net-tools vim make binutils
sudo apt-get install ca-certificates curl gnupg lsb-release
```

다음 세줄은 인스턴스를 최신화 하고 필요한 구성요소를 설치하는 명령어 입니다.

<br>

```shell
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o \
        /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
sudo apt-get install -y docker-compose
sudo systemctl enable docker
sudo systemctl start docker
sudo docker network create web-server
```

다음 코드들은 도커를 설치하기 위한 keyring을 받아 설치하고,  

도커와 docker-compose를 실제로 설치하는 명령어 입니다.

그 후 web-server라는 도커 네트워크(컨테이너들이 여기서 묶일 예정 입니다)를 만들었습니다.

<br>

```shell
git clone https://github.com/aws/efs-utils
cd efs-utils
./build-deb.sh
sudo apt install -y ./build/amazon-efs-utils*deb
cd ..
```

다음 마지막 명령어는 Amazon EFS를 사용하기 위해 EFS-utils이라는 구성요소를 설치하는 작업 입니다.

Amazon Linux가 아닌 Ubuntu에서 작업 하기 때문에 위의 작업이 필요 합니다.

위에 대한 작업을 스크립트로 한번에 적용 하였습니다.

<br>

# Outro

이번 파트 에서는 인스턴스에 대해 기초적인 세팅 일부를 진행 하였습니다.

다음 파트는 [Amazon EFS](/posts/AWS-Part10-AmazonEFS/) 설정 입니다.
