---
title: "Networking Solution"
# description: ""
categories: [컴퓨터, 🌚Computer-General]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-09-25. 01:22
# last_modified_at: 2024-09-25. 01:22
---

## 머리말

---

Networking Solution, Networking Framework, Network Library, Networking Engine, ...  

아바타 모캡이나 페이셜 데이터 전달은 Transport Layer부터 쓸 수 있으면 좋다.  
왜냐하면 원하는대로 커스터마이징이 필요하고, 실시간 데이터 전송이 필요하기 때문에 성능이 중요하기 때문이다.  

카메라 조작 기능 같은 건 HLAPI(High Level API)로도 충분하다.  

일반적으로 서버/클라이언트 모델의 온라인 게임을 만들고 운영하려면 게임 서버 빌드를 만들고 호스팅해야 합니다. 그 서버 빌드에는 서버 엔진 파트와 게임 컨텐츠 파트가 구현되어야 합니다. 엔진 파트는 Transport 설계, 소켓 통신, API 등 전반적인 네트워킹과 관련된 것들을 구현하며 컨텐츠 파트는 말그대로 게임의 온라인 컨텐츠 기능을 구현하는 부분입니다. 그렇다면 엔진 파트는 어떻게 구현할 수 있을까요?

먼저 성능과 필요한 기능 등을 고려해 서버 엔진을 팀에서 자체적으로 제작하는 방법이 있습니다. 이 방법의 장점은 서버 기능의 확장, 유지 보수가 용이합니다. 구현하고자 하는 게임에 최적화도 가능합니다. 그리고 구현 방식을 자유롭게 선택할 수 있습니다. 하지만 서버 엔진을 구현하기 위해 필요한 네트워크 지식을 모두 알아야 되며 제작에 드는 시간과 비용을 고려하면 결코 쉬운 작업이 아닐 수 있습니다. 그래서 소규모 팀에서는 그다지 추천하지 않는 방식입니다.

또 다른 방법은 이미 만들어진 게임 서버 라이브러리와 제공되는 API를 이용하는 방법이 있습니다. 이미 많은 솔루션이 있으며 각 솔루션마다 각기 다른 특징을 가지고 있기 때문에 구현하고자 하는 게임에 맞게 선택을 해서 사용하면 됩니다. 이 방법의 장점은 서버 엔진을 구현하는 시간과 비용을 절약할 수 있다는 것, 경우에 따라서는 네트워크 프로그래밍에 관한 지식이 없어도 API 사용 방법만 익히면 멀티플레이를 구현할 수 있다는 것입니다. 단점은 라이브러리 내에 구현된 서버 엔진에 따라서 원하는 기능의 추가나 성능 최적화가 제한될 수 있다는 점입니다. 예시로 Unity 엔진에서는 Unet이라고 불리는 네트워킹 전용 고수준 API (HLAPI)와 네트워크 라이브러리를 자체적으로 지원하며 이를 이용해서 멀티플레이 게임을 구현할 수 있습니다. (다만 Unet은 유니티에서 제거될 예정이기 때문에, 새로운 프로젝트는 Unet으로 구현하지 않는 것을 추천합니다.)

메세징 프레임워크
https://thebook.io/007035/0190/
https://medium.com/wardgames/unity-%EB%A9%80%ED%8B%B0%ED%94%8C%EB%A0%88%EC%9D%B4-%EA%B2%8C%EC%9E%84%EC%9D%84-%EB%A7%8C%EB%93%A4%EA%B8%B0-%EC%9C%84%ED%95%9C-mirror-mirage-%EC%86%8C%EA%B0%9C-a74b58bc115f
hlapi
https://docs.unity3d.com/kr/2019.4/Manual/UNetUsingHLAPI.html

### _

## [Photon](https://www.photonengine.com/ko-kr)

---

멀티플레이어 서비스  

## [Riptide](https://github.com/RiptideNetworking/Riptide)

---

<https://riptide.tomweiland.net/manual/overview/about-riptide.html>  

Unity Networking Library  
주로 온라인 게임에서 사용할 수 있도록 설계되었지만, Unity 뿐만 아니라 .Net Core 및 .Net Framework에서도 사용할 수 있다.  

유니티 데디케이트 서버 개발 등에 사용하기에 적합하지만, 객체 동기화 등의 기능은 없고, 순순하게 네트워크 메시지 전달 기능만 지원한다.  

## Mirror

---

## LiteNetLib

---

## Unity Multiplayer (Netcode)

---

## Mirage

---

## 메모

---

- [[Unity] 게임 서버/네트워크 라이브러리 Mirror & Mirage 소개](https://medium.com/wardgames/unity-멀티플레이-게임을-만들기-위한-mirror-mirage-소개-a74b58bc115f)