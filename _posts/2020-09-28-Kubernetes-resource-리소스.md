---
title: Kubernetes resource (리소스)
authors: jongin_kim
date: 2020-09-28 00:00:00 +0900
categories: [kubernetes]
tags: [kubernetes]
# image:
#   path: /commons/devices-mockup.png
#   lqip: data:image/webp;base64,UklGRpoAAABXRUJQVlA4WAoAAAAQAAAADwAABwAAQUxQSDIAAAARL0AmbZurmr57yyIiqE8oiG0bejIYEQTgqiDA9vqnsUSI6H+oAERp2HZ65qP/VIAWAFZQOCBCAAAA8AEAnQEqEAAIAAVAfCWkAALp8sF8rgRgAP7o9FDvMCkMde9PK7euH5M1m6VWoDXf2FkP3BqV0ZYbO6NA/VFIAAAA
#   alt: Responsive rendering of Chirpy theme on multiple devices.
---
### Role(역할)로 전환하는 이유

-   EC2 에서 실행되는 응용 프로그램이 자동 생성, 배포 되는 임시 보안 자격 증명을 사용할 수 있다.
-   임시자격 증명을 사용한다는 것은 인스턴스에서 직접 키 관리를 하지 않아도 된다는 뜻
-   즉, EC2 IAM Role(역할)기능을 적용해 장기적인 AWS 엑세스키(Access Key)를 수동 혹은 프로그램에서 직접 관리할 필요가 없어지는 것
-   그리고 2018년? 정도 부터 기존 EC2 인스턴스에 IAM 역할을 손쉽게 변경하는 방법이 생김

### EC2 / Docker(EC2)

`인스턴스 프로파일`

-   인스턴스 프로파일은 IAM Role(역할)을 위한 컨테이너로서 인스턴스 시작 시 EC2 인스턴스에 역할 정보를 전달하는 데 사용됩니다.
-   하나의 인스턴스 프로파일은 하나의 IAM 역할만 포함할 수 있습니다.
-   하지만 한 역할이 여러 인스턴스 프로파일에 포함될 수 있습니다. (어떤 방법으로도 늘릴 수 없음)

### 방법

1.  AWS IAM Role(역할) 생성하기
2.  해당 서버(EC2)에 기존 IAM 계정의 키로 관리되어지던 AWS Credentials를 제거
3.  EC2 인스턴스에 기존 IAM Role(역할) 변경하기 (당연히 AWS CLI는 설치되어 있어야 한다)
4.  1개의 EC2에서는 1개의 Role만 매핑이된다. 그러므로 그 EC2에 맞는 Role을 잘 생성해야 하며 여러가지 Role의 조합으로 Role을 만들자 해당 서버내의 어플리케이션이 필요한 모든 AWS 리소스 권한을 주는 것을 추천한다.
5.  당연하지만 EC2에 Docker를 띄운애들도 결국 그 EC2의 기본 Role을 따라간다.