---
title: 디스코드로 깃허브 알림봇 만들기 대작전
date: 2023-07-19
categories: [troubleshooting]
layout: post
tags: [github, discord, bot]
---

## 🤔 Problem

매 번 푸시 할때마다 팀원들에게 일일히 pull 해달라고 말하는것이 번거로웠다. 특히 프론트 분들은 인원수가 많다보니 서로 최신화하는 주기를 알려주는 봇이 있으면 좋겠다고 생각하게 되었다.

## 🌱 Solution

예전에 회사 다닐때 자주 봤던 슬랙 봇 처럼 디스코드에도 깃허브에도 봇을 달 수 있었다.

1. 알림 봇이 이용할 채널을 새롭게 만든다.
   <div markdown="block" style="width: 80%;">
      ![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-07-19-github-bot-with-discord/1.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-07-19-github-bot-with-discord/1.png)
   </div>

2. 채널 편집을 누르고 연동 탭에서 웹후크를 생성한 뒤 나온 링크를 복사 한다.
   <div markdown="block" style="width: 80%;">
   ![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-07-19-github-bot-with-discord/2.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-07-19-github-bot-with-discord/2.png)
   </div>

3. 깃허브 리포지토리 설정에서 웹후크를 클릭한다.
   <div markdown="block" style="width: 80%;">
   ![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-07-19-github-bot-with-discord/3.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-07-19-github-bot-with-discord/3.png)
   </div>

4. 디스코드에서 복사한 주소 + **/github** 를 입력하고, Content type 을 application/json 으로 변경한 뒤 알림 받을 항목을 체크한다.
   <div markdown="block" style="width: 70%;">
   ![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-07-19-github-bot-with-discord/4.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-07-19-github-bot-with-discord/4.png)
   </div>
5. 저장소에서 하나 푸시해보면 알림이 정상적으로 도착하는 것을 알 수 있다.
   <div markdown="block" style="width: 80%;">
   ![https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-07-19-github-bot-with-discord/5.png](https://yubinshin.s3.ap-northeast-2.amazonaws.com/2023-07-19-github-bot-with-discord/5.png)
   </div>

## 📎 Related articles

| 이슈명        | 링크                                                         |
| ------------- | ------------------------------------------------------------ |
| 디스코드 웹훅 | [https://eddori.tistory.com/6](https://eddori.tistory.com/6) |
