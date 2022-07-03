---
title: "GitHub organization 만들기"
date: 2022-05-24 00:56:00 +0900
categories: [GitHub]
tags: [GitHub, Organization, sw마에스트로]
---

# GitHub organization 만들기

### 참고링크

[GitHub에서 협업 용 단체(Organization) 만드는 방법](https://www.lainyzine.com/ko/article/how-to-create-an-organization-for-collaboration-on-github/)

## 0. 소개

sw마에스트로 팀빌딩이 끝났고 멘토 매칭도 한분만 남겨 둔 상황이다. 본격적으로 개발에 들어가기 전, github organization을 만들어 보려 한다.

## 1. organization 만들기

위 참고링크 따라서 만들었다. 쉬우므로 생략.
![1](https://user-images.githubusercontent.com/64428916/169953320-0dde2839-8d25-4472-8fef-a24f4d902185.png)

## 2. 기존 레포 가져오기

첨에 아래와 같은 에러가 발생했다. private 레포를 가져오려고 한게 문제였던 것 같다.

cf) 에러 문구 <br>
`Your old project requires credentials for read-only access. We will only temporarily store them for importing.`

![2](https://user-images.githubusercontent.com/64428916/169953358-1635c31d-bacc-40f1-aa18-a7f43eb812b6.png)

private을 public으로 바꿔주고 다시 로그인하니 제대로 클론하는 것을 확인할 수 있었다!
![3](https://user-images.githubusercontent.com/64428916/169953381-e84447c0-63f7-4142-b78c-8aa1655deec6.png)
![4](https://user-images.githubusercontent.com/64428916/169953400-a00fa9c1-4a84-4b0f-84ed-9372e12f3cb0.png)

아 참고로 저 레포는 임시 레포여서 커밋 컨벤션은 없다..ㅋㅋㅋㅋㅋ
