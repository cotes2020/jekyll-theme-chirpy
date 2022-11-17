---
title: AWS Infra / Part6. Bastion Host
date: 2022-11-16 20:50:00 +0900
categories: [AWS, 2Tier 아키텍쳐 구축]
tags: [AWS, Bastion Host, 보안 그룹, EC2 Instance]     # TAG names should always be lowercase
typora-root-url: ../
---
# Intro

파트 1 ~ 5를 거치면서 기본적인 토목 작업(?)이 끝났습니다.

이번엔 실제로 작동하는 인스턴스를 배치하려고 합니다.

제목은 배스천 호스트 만들기 지만, 내용에는 보안 그룹과 EC2 Instance에 대한 내용도 들어 있습니다.

<br>

# EC2 Instance?

> *Amazon Elastic Compute Cloud(Amazon EC2)*는 Amazon Web Services(AWS) 클라우드에서 확장 가능 컴퓨팅 용량을 제공합니다. Amazon EC2를 사용하면 하드웨어에 선투자할 필요가 없어 더 빠르게 애플리케이션을 개발하고 배포할 수 있습니다. Amazon EC2를 사용하여 원하는 수의 가상 서버를 구축하고 보안 및 네트워킹을 구성하며 스토리지를 관리할 수 있습니다. Amazon EC2에서는 확장 또는 축소를 통해 요구 사항 변경 또는 사용량 스파이크를 처리할 수 있으므로 트래픽을 예측할 필요성이 줄어듭니다. - *AWS Document*

간단하게 말하자면 클라우드상에서 구현된 서버 라고 볼 수 있습니다.

원하는 OS를 선택하고 원하는 서브넷에 배치할 수 있습니다.

<br>

# 보안 그룹이란?

> *보안 그룹*은 EC2 인스턴스에 대한 수신 및 발신 트래픽을 제어하는 가상 방화벽 역할을 합니다. 인바운드 규칙은 인스턴스로 들어오는 트래픽을 제어하고 아웃바운드 규칙은 인스턴스에서 나가는 트래픽을 제어합니다. 인스턴스를 시작할 때 하나 이상의 보안 그룹을 지정할 수 있습니다. 보안 그룹을 지정하지 않을 경우 Amazon EC2에서 기본 보안 그룹이 사용됩니다. 연결된 인스턴스에서 트래픽을 주고 받을 수 있도록 하는 규칙을 각 보안 그룹에 추가할 수 있습니다. 언제든지 보안 그룹에 대한 규칙을 수정할 수 있습니다. 새 규칙 및 수정된 규칙은 보안 그룹에 연결된 모든 인스턴스에 자동으로 적용됩니다. Amazon EC2는 트래픽이 인스턴스에 도달하도록 허용할지 여부를 결정할 때 인스턴스와 연결된 모든 보안 그룹에서 모든 규칙을 평가합니다. - *AWS Document*

보안 그룹은 일종의 방화벽 역할을 수행한다고 쓰여져 있습니다.

그 역할에 맞게 특정 호스트에게만 특정 포트 또는 서비스만 개방하여 안전하게 통신할 수 있도록 해 줍니다.

특정 호스트는 아이피 지정만이 아니고 보안 그룹으로도 지정할 수 있습니다.

소스를 보안 그룹으로 지정 했을 경우, 해당 인스턴스나 요소가 지정한 보안 그룹일때, 통신을 허용하게 됩니다.

<br>

# Bastion Host?

> *배스천 호스트*는 가상 프라이빗 클라우드(VPC)의 프라이빗 및 퍼블릭 서브넷에 위치한 Linux 인스턴스에 대한 보안 액세스를 제공합니다. - *AWS Document*

일반적으로 Private 서브넷에 존재하는 인스턴스에 대해 직접적으로는 통신이 불가능 합니다.

그래서 Public 서브넷에 Bastion Host라는 인스턴스를 배치한 다음, 

Bastion Host만 접근 할 수 있는 정책(보안 그룹)을 Private 서브넷에 있는 인스턴스(또는 요소)에 적용하면,

Bastion Host를 경유해서 해당 인스턴스 들을 관리 할 수 있습니다.

물론 이를 위해서는 Bastion Host에 대한 보안도 충분히 강구 해야 됩니다.

<br>

# 실전

## 보안그룹 생성하기

![01](/assets/post/2022-11-16-AWS-Part6-BastionHost/01.png)

상단 검색창에서 **EC2를 검색**해서 서비스 란의 **EC2** 를 클릭 합니다.

<br>

![02](/assets/post/2022-11-16-AWS-Part6-BastionHost/02.png)

EC2 대시보드에서 **보안 그룹**을 클릭 합니다.

<br>

![03](/assets/post/2022-11-16-AWS-Part6-BastionHost/03.png)

보안 그룹 생성을 위해 **보안 그룹 생성** 버튼을 클릭 합니다.

이번에는 Bastion Host용과 테스트를 해볼 Private Host용 보안 그룹을 생성 하겠습니다.

<br>

### Bastion Host용 보안 그룹 생성

![04](/assets/post/2022-11-16-AWS-Part6-BastionHost/04.png)

보안 그룹의 이름과 설명을 작성 합니다.

차후에 봐도 알아볼 수 있게 작성하는것이 좋습니다.

VPC가 여러개 일 경우 VPC를 꼭 확인해 봐야 합니다.

<br>

![05](/assets/post/2022-11-16-AWS-Part6-BastionHost/05.png)

이번에는 Bastion Host 접근을 SSH를 통한 접속으로 할 것이기 때문에 **유형을 SSH로** 선택 합니다.

Bastion Host는 Public 서브넷에 배치될 거기 때문에 **Anywhere-IPv4(0.0.0.0/0)을 선택**하면

Bastion Host의 공인 IP를 아는 사람이라면 **누구나 접근** 할 수 있습니다.

여기서는 편의상 0.0.0.0/0을 선택 했습니다만, 필요에 따라서 IP를 지정하거나 할 수 있습니다.

아웃바운드 규칙은 별도로 안건드리셔도 무방 합니다.

<br>

![06](/assets/post/2022-11-16-AWS-Part6-BastionHost/06.png)

보안 그룹의 태그는 선택사항 입니다만 Name이라는 키를 추가해서 설명을 쓰시는것을 추천 합니다.

다 작성하였으면 **보안 그룹 생성** 버튼을 누릅니다.

<br>

### Private Host용 보안 그룹 생성

![07](/assets/post/2022-11-16-AWS-Part6-BastionHost/07.png)

통신을 테스트할 Private Host용 보안 그룹의 규칙은 2개를 지정 합니다.

첫 줄은 **Bastion Host만 접근을 허용**하는 SSH 규칙 입니다.

소스칸의 공란을 클릭하면 이전 단계에서 만들어둔 보안 그룹이 표시 됩니다.

이 보안 그룹을 소스로 선택 하면, 다음 단계에서 만들 Bastion Host의 보안 그룹이 ForBastionHost기 때문에,

Bastion Host에서 SSH로 접근 하는 것을 허용하게 됩니다.

보안 그룹으로 지정하게 되면 IP에 대해 신경 쓰지 않고 원하는 인스턴스에 대해 서비스를 개방 할 수 있습니다.

두번째 줄은 **내부 Ping(ICMP)을 허용(현 VPC의 CIDR 대역이 172.31.0.0/16 입니다.)**하는 규칙 입니다.

이 외의 것은 전 단계에서 했던것 처럼 진행 하면 됩니다.

<br>

![08](/assets/post/2022-11-16-AWS-Part6-BastionHost/08.png)

현재 위와 같이 보안 그룹이 두개가 생성 되었습니다.

태그 단계에서 Name을 지정하지 않으면, 기본 보안 그룹 처럼 -로 표시 됩니다.

<br>

## EC2 Instance 생성하기

EC2 인스턴스는 총 4개를 생성해 보겠습니다.

각 Public 서브넷에 Bastion Host 한개씩, 각 Private 서브넷(2*-**PrivSubNet**)에 테스트용 인스턴스 한개씩 입니다.

<br>

![09](/assets/post/2022-11-16-AWS-Part6-BastionHost/09.png)

인스턴스를 만들기 위해 EC2 대시보드에서 **인스턴스**를 클릭 합니다.

<br>

![10](/assets/post/2022-11-16-AWS-Part6-BastionHost/10.png)

다음 **인스턴스 시작**을 누릅니다.

<br>

![11](/assets/post/2022-11-16-AWS-Part6-BastionHost/11.png)

인스턴스의 이름을 지정하고, OS를 선택 합니다.

원하는대로 만들어도 됩니다. 저는 ubuntu로 만들어 보겠습니다.

<br>

![12](/assets/post/2022-11-16-AWS-Part6-BastionHost/12.png)

대부분은 프리 티어로 구축을 해보실 텐데요, **t2.micro는 서울 리전 기준 가용영역 2a와 2c에서 사용이 가능** 합니다.

본인이 원하는 경우 더 좋은 성능의 인스턴스를 고르셔도 상관 없습니다.

<br>

![13](/assets/post/2022-11-16-AWS-Part6-BastionHost/13.png)

인스턴스에 안전하게 접속하기 위한 키 페어를 생성 할 수 있습니다. **새 키 페어 생성**을 클릭 합니다.

<br>

![14](/assets/post/2022-11-16-AWS-Part6-BastionHost/14.png)

키 페어 이름을 지정 합니다.

그 후 프라이빗 키 파일 형식을 지정해야 합니다.

저 같은 경우에는 MobaXterm을 사용하기 때문에 .pem을 선택 했습니다.

PuTTY를 사용하신다면 ppk를 선택 해도 됩니다.

**키 페어 생성**을 클릭 합니다.

<br>

![15](/assets/post/2022-11-16-AWS-Part6-BastionHost/15.png)

그럼 키가 자동으로 다운로드 됩니다.

같은 과정으로 PrivHostKey 키도 만들겠습니다.

지금은 BastionHost 생성 단계니 키 페어 이름은 BastionKey로 선택 합니다.

<br>

![16](/assets/post/2022-11-16-AWS-Part6-BastionHost/16.png)

네트워크 설정 입니다. 편집 버튼을 누르면 편집 가능한 상태가 됩니다.

VPC와 서브넷을 선택 합니다. 그후 BastionHost의 경우에는 퍼블릭 IP 자동 할당을 **활성화** 합니다.

그 외에 경우에는 활성화 하지 않습니다.

그 다음 보안 그룹은 **기존 보안 그룹 선택** 을 클릭 하고 ForBastionHost 보안 그룹을 선택 합니다.

<br>

![17](/assets/post/2022-11-16-AWS-Part6-BastionHost/17.png)

스토리지 구성은 필요할 경우 크기를 선택 할 수 있습니다. 

테스트로 만드는 것이기 때문에 따로 추가는 하지 않았습니다.

그 후 우측 사이드바의 **인스턴스 시작** 버튼을 누르면 생성이 완료 됩니다.

<br>

| 인스턴스 이름 |   키 페어   |    서브넷     | 퍼블릭 IP |   보안 그룹    |
| :-----------: | :---------: | :-----------: | :-------: | :------------: |
| BastionHost1  | BastionKey  | 2a-PubSubNet  |  활성화   | ForBastionHost |
| BastionHost2  | BastionKey  | 2c-PubSubNet  |  활성화   | ForBastionHost |
|   PrivHost1   | PrivHostKey | 2a-PrivSubNet | 비활성화  |  ForPrivTEST   |
|   PrivHost2   | PrivHostKey | 2c-PrivSubNet | 비활성화  |  ForPrivTEST   |

나머지 3개의 인스턴스는 위 표를 참고해서 추가로 생성 합니다.

<br>

![18](/assets/post/2022-11-16-AWS-Part6-BastionHost/18.png)

위와 같이 4개의 인스턴스를 생성 하셨나요? 다음은 결과를 확인 해보겠습니다.

<br>

# 결과 확인

### 연결 테스트

![19](/assets/post/2022-11-16-AWS-Part6-BastionHost/19.png)

인스턴스 중 Bastion Host인 것을 클릭 합니다.

클릭 하면 브라우저 하단에 정보창이 새로 뜨게 됩니다.

세부정보 탭의 퍼블릭 IPv4 주소를 확인 합니다. 인스턴스 생성 시 퍼블릭 IP 할당을 활성화 했을때만 부여 됩니다.

IP옆의 아이콘을 클릭하면 클립보드로 복사도 됩니다. 클릭 하여 복사 합니다.

<br>

![20](/assets/post/2022-11-16-AWS-Part6-BastionHost/20.png)

여기서부터는 MobaXterm 기준으로 설명 드립니다.

User Sessions를 우클릭 하고 **New session**을 클릭 합니다.

<br>

![21](/assets/post/2022-11-16-AWS-Part6-BastionHost/21.png)

Remote host에는 아까 복사한 IP를 넣습니다.

username은 ubuntu를 넣습니다.

username은 선택한 OS에 따라 달라집니다. 

[이 링크를 클릭](https://docs.aws.amazon.com/ko_kr/AWSEC2/latest/UserGuide/managing-users.html#ami-default-user-names)해서 다른 OS를 선택했을때 기본 Username을 확인하세요.

<br>

![22](/assets/post/2022-11-16-AWS-Part6-BastionHost/22.png)

다음 하단의 탭중 Advanced SSH settings를 클릭 하면 개인키를 지정할 수 있는 란이 있습니다.

Use private key를 클릭 후 공란의 문서 버튼을 클릭하면 개인키 파일을 선택 할 수 있습니다.

Bastion Host에 연결 할 것이므로 BastionKey.pem을 선택합니다.

그 후 OK를 누르면 접속을 시도 합니다.

<br>

![23](/assets/post/2022-11-16-AWS-Part6-BastionHost/23.png)

Bastion Host는 Public 서브넷에 개방된 설정으로 배치되었기 때문에 접속이 가능합니다.

<br>

![24](/assets/post/2022-11-16-AWS-Part6-BastionHost/24.png)

다음 Private 서브넷에 있는 PrivHost로 접속하기 위해 키 파일을 업로드를 할 필요가 있습니다.

MobaXterm은 파일을 드래그하면 쉽게 업로드가 가능 합니다.

<br>

![25](/assets/post/2022-11-16-AWS-Part6-BastionHost/25.png)

업로드가 된것을 확인 할 수 있습니다.

<br>

![26](/assets/post/2022-11-16-AWS-Part6-BastionHost/26.png)

키를 이용해 접속하기 전, 키 파일의 권한을 400(r--)으로 설정할 필요가 있습니다.

```shell
chmod 400 {키 파일 이름}
```

명령어로 소유권을 수정 해 줍니다.

<br>

![27](/assets/post/2022-11-16-AWS-Part6-BastionHost/27.png)

그후 Private 서브넷에 있는 인스턴스 하나의 프라이빗 주소를 확인 합니다.

PrivHost*의 경우 퍼블릭 IP 부여 옵션을 비활성화 했기 때문에 퍼블릭 IP주소가 없습니다.

<br>

![28](/assets/post/2022-11-16-AWS-Part6-BastionHost/28.png)

```shell
ssh -i [키 파일] [접속할 호스트의 기본 유저네임]@[IP]
```

를 입력하면 키를 이용하여 호스트에 접속 할 수 있습니다.

외부에서는 접속할 수 없지만, Bastion Host의 보안 그룹을 대상으로는 개방이 되어 있기 때문에,

접속이 되는것을 확인 할 수 있습니다.

<br>

![29](/assets/post/2022-11-16-AWS-Part6-BastionHost/29.png)

핑 명령을 통해 현재 원격으로 접속한 PrivHost에서 다른 PrivHost로 핑을 날리면 잘 통신 되는 것을 볼 수 있습니다.

현재는 SSH와 ICMP(Ping)만 보안 그룹에 명시되어 있지만, 

인바운드 규칙을 수정해서 다른 방법으로도 접근할 수 있게 할 수 있습니다.

<br>

![31](/assets/post/2022-11-16-AWS-Part6-BastionHost/31.png)

NAT Gateway가 설치되어 있기 때문에 Private 내부 인스턴스가 apt 명령어로 업데이트가 잘 되는걸 볼 수 있습니다.

<br>

## 인프라 배치 현황

![30](/assets/post/2022-11-16-AWS-Part6-BastionHost/30.png)

현재 까지 이렇게 배치되었습니다.

<br>

# Outro

이번 파트에서는 배스천 호스트를 배치하고, 테스트를 위한 임시 인스턴스를 배치 하였습니다.

서로간의 통신을 위해 적절한 보안 그룹을 만들고 EC2 Instance도 생성하고 테스트까지 해보았습니다.

[다음 파트는 Amazon RDS](/posts/AWS-Part7-AmazonRDS/)입니다. DB 인스턴스를 생성하여 전용 Private 서브넷에 배치해 볼 것 입니다.