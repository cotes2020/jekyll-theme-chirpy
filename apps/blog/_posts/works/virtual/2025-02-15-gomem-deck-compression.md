---
title: "VRChat - 고멤 덱 압축"
description: "우왁굳님의 고멤유치원 (고멤 하드코어) 결과 발표 컨텐츠"
categories: [작업물, 버추얼]
tags: [작업물, VRChat, 유니티]
image: "/assets/img/post/works/gomem-deck-compression/250215-000000.png"
hidden: true

date: 2025-02-15. 00:00
last_modified_at: 2025-03-09. 21:05 # 전
---

드디어 고멤 유치원 최종 합격자 발표  
{% include embed/youtube.html id = "Q-NpTQu_FD0" %}

## 머리말

---

우왁굳님의 '고멤 유치원 합격/탈락 발표' 컨텐츠입니다.  

25년 02월 15일 방송이 진행됐습니다.  

### 참여 / 담당

본 컨텐츠는 VRChat 게임 내에서 진행되었습니다.  

해당 컨텐츠에 사용된 VRChat World 내 기능 프로그래밍을 담당했습니다.  

- 카메라 시스템
  - 에디터 타임에 카메라 구도를 미리 설정
  - `CinemachineVirtualCamera.Priority`를 조정하여 구도 전환

### 사용한 툴

- Unity 2022.3.22f1
- [U# (C# + VRChat SDK)](https://udonsharp.docs.vrchat.com/)
- [Woodon](https://github.com/wrchat/Woodon)

Discord를 통해 팀원/클라이언트와 소통했습니다.  

## 시작

---

## 과정

---

### _

## 구현

---

02월 06일 프로젝트에 참가했다.  

강화 대상을 맵 중앙으로 옮겨야 했는데,  
이를 위해 고멤 유치원 분들의 VRChat 닉네임을 정리해야했다.  

- 303특공대 강요셉: [`303특공대 강요셉`](https://vrchat.com/home/user/usr_0e7841b3-1b71-4342-8a3e-769cffab63d8)
- 금뱅기: [`금 뱅기`](https://vrchat.com/home/user/usr_50863740-8225-499d-9733-bd345d5b1566)
- 남미버드: [`남미bird`](https://vrchat.com/home/user/usr_a5f3dc59-0d41-464c-8205-3f5ed4934152)
- 도깨비공주 루딘: [`ROODIN`](https://vrchat.com/home/user/usr_6dbbaf57-1760-48fa-95bd-f458eb928fec)
- 마리 블랙로즈: [`마리 블랙로즈`](https://vrchat.com/home/user/usr_311110ef-98c7-4363-b91b-c5bbaf9c61fb)
- 마익호: [`마익호_`](https://vrchat.com/home/user/usr_65841063-4556-4d08-b55e-afbb28892f5f)
- 바룬상: [`바룬상~`](https://vrchat.com/home/user/usr_21950d9e-fec3-4916-9fd4-f3a5c0daa098)
- 선도위원강무: [`선도위원 강무`](https://vrchat.com/home/user/usr_2678cebb-71de-461d-8e62-627fe72212ad)
- 세르게이 드라구노프: [`세르게이 드라구노프`](https://vrchat.com/home/user/usr_8f08f05b-7762-41ec-afbe-f4274c832b1a)
- 알렉스 퀀턴: [`알렉스 퀀턴`](https://vrchat.com/home/user/usr_75932048-7ae7-4e7d-90ca-b1b5a9cce3be)
- 야구자: [`야 구 자`](https://vrchat.com/home/user/usr_3c30cfe5-b1ad-4b52-b8ad-16cf136d917d)
- 에밀리: [`emily00_`](https://vrchat.com/home/user/usr_e6c432b4-3358-4518-8a3c-406911833cd7)
- 인디언 빔밥: [`인디언 빔밥`](https://vrchat.com/home/user/usr_3e7731f0-e8b4-4d25-8b59-6a318dd00524)
- 제갈 통: [`제갈 통`](https://vrchat.com/home/user/usr_c45bba6b-31de-4d6c-bb4d-05db2cc3ee9f)
- 촌장 고봉: [`촌장 고봉`](https://vrchat.com/home/user/usr_6674cb0d-bc87-4015-a198-71acda756a8e)
- 코드네임 로즈: [`o로즈o`](https://vrchat.com/home/user/usr_04b8479e-797f-45bc-b17a-94be7c5db813)
- 강풍: [`_강풍_`](https://vrchat.com/home/user/usr_8432eae7-5423-4473-b0d5-3e770d3f614c)
- 구호선: [`구호선_`](https://vrchat.com/home/user/usr_6e678b37-fa23-4b67-8292-e70cdb8edb93)
- 길앞잡이 광수: [`길앞잡이 광수`](https://vrchat.com/home/user/usr_2eeb2071-1fd1-4820-a1d2-f6d6361f651d)
- 망찡: [`망찡_manggjjing`](https://vrchat.com/home/user/usr_b9f63647-2f4f-47f8-87e1-58c9532c7b78)
- 무녀 사야: [`무녀 사야`](https://vrchat.com/home/user/usr_9bd51d75-6576-4341-972d-30470683952b)
- 상 가르마: [`상 가르마`](https://vrchat.com/home/user/usr_c795c5b0-f38b-45ad-a98c-fe6726cb8311)
- 서애라: [`서애라_`](https://vrchat.com/home/user/usr-1.1da3ec1-1848-4c2d-a839-c37dfa219397)
- 프리스트 엘리나: [`엘리나_`](https://vrchat.com/home/user/usr_db489b8c-4a76-499b-9724-d55d6e63138a)
- 하율 옹: [`하율 옹`](https://vrchat.com/home/user/usr_fa63ac62-fb96-4cb5-9cab-33da24106d2a)

그림자 군도, 롤 시네마틱 관련해서 아이디어를 던졌다.  

디자인으로 쩌나님께서 참여해주셨다.  

앞부분에 스포되는 걸 중간에 한 번 인지했었는데  
까먹고 고치지 못했다.  

방송 당일이 돼서야 뒤늦게 알아차렸다.  
아쉬운 부분.  

## 기록

---

![250211-222111](/assets/img/post/works/gomem-deck-compression/250211-222111.png)
![250211-223055](/assets/img/post/works/gomem-deck-compression/250211-223055.png)
![250211-224730](/assets/img/post/works/gomem-deck-compression/250211-224730.png)
![250215-000000](/assets/img/post/works/gomem-deck-compression/250215-000000.png)
![250215-050358](/assets/img/post/works/gomem-deck-compression/250215-050358.png)
![250215-234434](/assets/img/post/works/gomem-deck-compression/250215-234434.png)
![250309-201659](/assets/img/post/works/gomem-deck-compression/250309-201659.png)
![250309-201836](/assets/img/post/works/gomem-deck-compression/250309-201836.png)
![250309-201850](/assets/img/post/works/gomem-deck-compression/250309-201850.png)

- 무조건 강화
  - 프리미엄 중간계
