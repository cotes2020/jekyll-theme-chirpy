---
title: 도커교과서 도커의 기본적인 사용법
date: 2023-10-06
categories: [blog]
tags: [docker, container]
---

## 📦 컨테이너란 무엇인가?

각각의 운영체제를 따로 사용하는 가상머신(VM)과 달리,
컨테이너는 운영체제(cpu, 메모리 리소스)를 호스트 컴퓨터와 유동적으로 공유하는 가상의 애플리케이션 실행 환경을 말한다.


## 📦 컨테이너의 장점

컨테이너를 사용하면 호스트 컴퓨터와 운영체제를 공유하기 때문에 밀집<sup>[1](#footnote_1)</sup>과 격리<sup>[2](#footnote_2)</sup> 를 동시에 달성할 수 있고 매우 효율적이다.

<a name="footnote_1">[1]</a> 밀집 : 컴퓨터에 CPU, 메모리가 허용하는 한 최대한 많은 수의 애플리케이션을 실행 하는 것

<a name="footnote_2">[2]</a>: 격리 : 서로 다른 애플리케이션을 동시에 실행할 때 특정 app 때문에 다른 앱이 꺼지는 상황을 방지 하기 위해 app 당 독립된 환경을 제공하는 것을 말한다.

 >  (ex) 버전이 달라서, 호환되지 않아서, 과다한 리소스를 특정 앱이 소모해서



## 🐳도커의 장점

### 1. 컨테이너를 사용하기 때문에 효율성을 갖춘다.

### 2. 배우기 쉽다.

도커를 사용하면 npm run ~ , python3 uvicorn ~ 등 여러 패키지매니저, 스크립트 언어를 배우지 않고도 도커만의 명령어로 바로 실행할 수 있어 매우 편리하고 효율적이다.

15년 전 닷넷, 10년전 자바, 오늘 GO로 만든 애플리케이션도 모두 똑같은 명령어로 관리할 수 있다.

** 도커 컨테이너 실행 명령어 **


```bash
$ docker container run --interactive --tty 이미지명
```

> `--interactive` 컨테이너와 상호작용할 수 있게 접속 <br/>
> `--tty` 터미널 세션으로 조작하겠다는 뜻

<br/>
 
```sh
$ docker container ls
```

> 실행중인 도커 컨테이너 리스트를 보여줘

<br/>
 
```sh
$ docker container ls --all
```

> 실행 여부에 관계 없이 모든 도커 컨테이너 리스트를 보여줘

<br/>
 
```sh
$ docker container top 
```

> 도커 컨테이너 에서 실행 중인 프로세스 목록을 표를 보여줘 <br/>
> (table of processes)

<br/>
 
```sh
$ docker container logs
```

> 도커 컨테이너에서 나온 모든 로그를 출력해줘

<br/>
 
```sh
$ docker container inspect
```

> 도커 컨테이너에 대한 상세한 정보를 줘

<br/>
 
```sh
$ docker container run --detach --publish 8080:80 이미지명
```

> 도커야 백그라운드에서 동작하면서 네트워크를 주시(listen)해줘
> --detach 백그라운드에서 실행해줘 <br/>
> --publish 컨테이너 포트를 호스트 컴퓨터에 공개해줘 

<br/>

## 🐳도커의 네트워크 주시

도커를 설치하면 호스트 컴퓨터의 네트워크 계층에 도커가 끼어든다.

그러면 호스트 컴퓨터에 들고 나는 네트워크 트래픽을 모두 도커가 가로채서 그 중 필요한 것을 컨테이너에 전달한다. 

`-- publish` 혹은 `-p`  태그를 붙이면 호스트 컴퓨터의 해당 포트로 들어오는 트래픽을 도커가 호스트와 컨테이너 간에 중계하게 된다.

 <div markdown="block" style="width: 80%;">
![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-01-learn-docker-in-a-month-of-lunches/01.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-10-01-learn-docker-in-a-month-of-lunches/01.png)
</div>

## 📎 Related articles

| 이슈명               | 링크                                                                                           |
| -------------------- | ---------------------------------------------------------------------------------------------- |
| 교보문고 도커 교과서 | [https://www.yes24.com/Product/Goods/111408749](https://www.yes24.com/Product/Goods/111408749) |
