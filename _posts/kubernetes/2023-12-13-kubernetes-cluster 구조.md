---
title: Kubernetes Cluster 구성 및 개념
date: 2023-12-13 22:54:32 +0900
author: kkankkandev
categories: [Kubernetes]
tags: [kubernetes, k8s, k8s-cluster, cluster, api-server, etcd, controller-management, scheduler, kubelet, kube-proxy, containerd, container]     # TAG names should always be lowercase
comments: true
image:
  path: https://github.com/War-Oxi/war-oxi.github.io/assets/72260110/c39504c6-3de4-4b41-919b-5ef1b132106c
---
# Concept

> 쿠버네티스 클러스터는 애플리케이션  컨테이너를 실행하기 위한 일련의 노드 머신들의 집합입니다.


- 클러스터는 **Control-Plane** 및 하나 이상의 컴퓨팅 머신 또는 노드를 포함합니다.
- 컨트롤플레인이 어느 Application을 실행하고 Application이 어느 Conatiner Image를 사용할지와 같이 클러스터를 원하는 상태로 유지 관리합니다. Node는 Control Plane에서 요청을 받아 Appliation과 Workload를 실제로 실행합니다
- 따라서 물리 머신, 가상 머신, 온프레미스, 클라우드에 구애받지 않고 머신 그룹 전체에서 컨테이너를 예약하고 실행할 수 있습니다.

# Master Node

> 클러스터의 전반적인 관리와 조정을 담당합니다. 여러 개의 마스터 노드로 구성될 수 있어 고가용성(HA)를 지원합니다.



## etcd

<aside>
🔥 키 값 형식으로 클러스터의 모든 상태와 정보를 저장하는 데이터베이스입니다.
데이터베이스, 분산 시스템을 위한 신뢰할 수 있는 키-값 스토어. 단순하고 안전하며 신속합니다

⇒ k8s클러스터의 모든 상태 정보와 메타데이터가 저장됩니다.

</aside>

- etcd 데이터 저장소에 저장되는 Cluster에 관한 정보는 다음과 같습니다.
    - Nodes
    - Pods
    - Configs
    - Secrets
    - Accounts
    - Roles
    - Bindings
    - Others

### etcd의 역할과 중요성

1. 클러스터 상태 저장
2. 데이터의 일관성 유지
    -  Raft 합의 알고리즘을 사용해 클러스터 전반에 걸쳐 데이터의 강력한 일관성을 보장
3. 고가용성
4. 변경 감지 및 감시 기능
    - key-value 데이터에 대한 병경 사항을 감시하는 기능을 제공함.
    - etcd를 통해 k8s는 클러스터의 상태 변화를 실시간으로 감지하고, 필요한 조치를 취할 수 있음

### k8s와 etcd의 상호 작용

- **API 서버와의 통신:** Kubernetes의 API 서버는 클러스터의 모든 정보를 etcd에 읽고 쓰는 주된 방법입니다. API 서버는 etcd와의 상호작용을 통해 클러스터 상태의 변경 사항을 반영하고 조회합니다.
- **클러스터 복원력:** etcd의 높은 가용성과 일관성은 Kubernetes 클러스터의 전반적인 복원력과 안정성을 강화합니다. 예를 들어, 마스터 노드가 실패할 경우, etcd 데이터를 사용하여 클러스터 상태를 복원할 수 있습니다.
- **스케일링과 업데이트:** 클러스터의 스케일링이나 업데이트 시, etcd는 새로운 노드나 서비스의 정보를 저장하고, 이를 클러스터 전체와 동기화합니다.

## api-server

<aside>
🔥 쿠버네티스 API를 제공하는 컴포넌트로, 사용자 클러스터 내의 다양한 컴포넌트와 통신을 담당합니다

</aside>

- api-server를 통해 사용자, 클러스터 내부의 다양한 컴포넌트, 그리고 외부 시스템들이 클러스터와 상호 작용할 수 있습니다.
- **RESTful Interface**
    - 클라이언트가 HTTP 메서드를 사용해 리소스(Pod, Service 등)를 생성, 수정, 삭제할 수 있습니다.
- **인증 및 권한 부여**
    - 클라이언트 요청에 대한 인증과 권한 부여 절차를 처리합니다.
- **클러스터 상태 관리**
    - api-server는 클러스터의 상태를 etcd와 같은 분산 키-값 저장소에 저장하고 관리합니다.
- 리소스 유효성 검사 및 적용
    - 클라이언트로부터 리소스에 대한 요청을 받으면, 해당 요청의 유효성을 검사하고, 규칙에 맞는 경우에만 시스템에 적용합니다

## controller-manager

> 쿠버네티스 컨트롤러는 종류가 매우 다양합니다. 예시로 아래 두 개의 컨트롤러가 있습니다. 이 컨트롤러 들은 Kube Controller Manager라는 하나의 프로세스로 패키지화되어 관리됩니다


- Kube Controller Manager를 설치하면 나머지 다른 컨트롤러도 같이 설치됨

## scheduler

> Node Pod를 Scheduling


- Pod를 어디에 배치시킬지 결정

# Worker Node

## kubelet

> 클러스터의 각 노드에서 실행되는 Agent



<aside>
🔥 API 서버로부터 Pod 명세를 받아, 이를 해석 후 컨테이너 실행합니다.

</aside>

- Kube API 서버의 지시를 듣고 필요한대로 노드에서 컨테이너를 배포하거나 파괴합니다
- Kube API 서버는 주기적으로 Kubelet으로부터 상태 보고서를 가져옵니다
- 클러스터의 각 노드의 kubelet은 pod의 상태를 확인 후 k8s api 서버에 보고합니다.
    - ⇒ 노드와 컨테이너의 상태를 모니터링

### Worker Node가 Master Node와 연결이 끊겼을 경우

- kube-apiserver와 kube-scheduler가 없기 때문에 kubelet은 외부에서 명령을 받을 수 없습니다
- 해당 Worker Node는 독립적으로 존재하며 해당 Worker Node의 kubelet은 독립적으로 Pods를 관리합니다
- 이 경우 kubelet은 Pods를 어떻게 관리할까?
    - Pod에 관한 정보를 저장하는 서버 디렉토리를 통해 Pod 정의 파일을 읽을 수 있습니다.
        - `/etc/kubernetes/menifests` - 서버 디렉토리
        - 서버디렉토리에 Pod 정의 파일을 넣어두면 kubelet은 주기적으로 서버 디렉토리를 확인한 후 Host에 Pod를 생성합니다

## kube-proxy

> K8s 클러스터 내에서 네트워킹을 관리하는 주요 컴포넌트


- 각 K8s 노드에 설치되며, 클러스터 내의 네트워크 통신 및 라우팅 규칙을 처리
- Pod간 네트워크 통신을 가능하게 하고, 외부 네트워크로부터의 접근을 관리

### Kube-Proxy의 주요기능

1. 서비스 추상화
    1. K8s의 서비스 추상화를 구현
    2. 서비스 ⇒ Pod 그룹에 대한 네트워크 접근을 제공하는 추상적인 개념
2. 로드밸런싱
    1. 클러이언트 요청을 서비스에 연결된 여러 포드 중 하나로 분산시키는 역할을 합니다
3. 네트워크 규칙 관리
    1. 각 노드의 iptabels, ipvs 또는 사용자 공간 프록시를 통해 네트워크 규칙을 설정하고 관리함
    ⇒ Pod, Service 및 Endpoint 간의 네트워크 통신을 가능하게 해줍니다
4. 외부 접근 처리
    1. 외부에서 클러스터 내의 서비스에 접근할 수 있도록 NodePort LoadBalancer 또는 ClusterIP 서비스 타입을 통해 트래픽을 적절한 Pod로 라우팅합니다.

## Containerd

> 컨테이너 런타임으로 사용되는 컴포넌트


- K8s 클러스터 내에서 컨테이너를 생성하고 실행하는 역할을 합니다

### Containerd의 주요 기능과 특징

1. 컨테이너 실행 및 관리
    1. 컨테이너의 생명주기를 관리합니다
    ⇒ 컨테이너의 생성, 실행, 일시 정지, 재개 및 종료
2. 이미지 관리
    1. 컨테이너 이미지를 다운로드, 저장, 관리하는 기능을 제공합니다
    2. Docker이미지 레지스트리 또는 OCI(Open Conatiner Initiative) 호환 레지스트리에서 이미지를 가져오는 것을 포함
3. 네트워킹 및 스토리지
    1. 컨테이너에 대한 네트워킹 및 스토리지 인터페이스를 제공
    2. CNI(Container Network Interface)와 CSI(Container Storage Interface) 플러그인을 통해 확장 가능
4. 보안
    1. 컨테이너의 격리 및 보안을 위해 Namespace, Cgroups, AppArmor, SELinux 등을 사용함
5. 가벼움 및 성능
    1. Docker 보다 가벼운 대안으로 시스템 리소스를 적게 사용하며, 더 빠른 시작 시간과 성능을 제공
6. 표준화 및 호환성
    1. OCI 표준을 준수 ⇒ 다양한 컨테이너 도구 및 시스템과의 호환성 보장


<strong>궁금하신점이나 추가해야할 부분은 댓글이나 아래의 링크를 통해 문의해주세요.</strong>   
Written with [KKam.\_\.Ji](https://www.instagram.com/kkam._.ji/)
