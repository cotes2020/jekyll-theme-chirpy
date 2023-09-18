---
title: NestJS로 쉽고 재미있게 REST API 만들기
date: 2023-06-08 20:00:00 +0900
categories:
  - JavaScript
tags:
  - NextJS
  - RESTAPI
---

안녕하세요, 여러분! 오늘은 모두의 친구 NestJS와 함께 REST API를 만들어 볼거예요. REST API는 한마디로 말해서 다양한 프로그램이 서로 소통할 수 있게 해주는 중간 다리 같은 역할을 해요. NestJS는 이런 API를 만드는 데 정말 유용한 친구랍니다! 😄

## 실습 프로젝트 시작하기 💻

우리는 먼저 NestJS 프로젝트를 시작해야해요. 터미널에서 아래와 같은 명령어를 입력해서 프로젝트를 만들어 봅시다!

```bash
$ nest new our-nestjs
```

이제 NestJS의 세상으로 한 발자국 더 다가갔어요! 🌟

## 코드 자동 생성 🛠

이제 본격적으로 코드를 작성해볼 차례에요. 근데 코드를 모두 손으로 작성하려면 시간이 너무 오래 걸리죠? NestJS는 이런 번거로움을 덜어주기 위해 자동으로 코드를 생성해주는 기능을 제공해요. 아래 명령어를 통해 필요한 파일들을 한 번에 생성해봅시다!

```bash
$ nest generate resource users
```

와! 이제 필요한 파일들이 모두 생성되었어요. 굉장히 편리하죠? 🥳

## 엔티티 클래스 작성하기 🏗

이제 데이터를 어떻게 다룰지 정해야 할 차례에요. 우리가 만들 유저 정보를 관리하는 엔티티 클래스를 작성해봅시다. 아래와 같이 파일을 작성해볼까요?

```typescript
export class User {
  id: number;
  name: string;
  email: string;
  phone?: string;
  createdAt: Date;
  updatedAt?: Date;
}
```

엔티티 클래스에는 유저의 정보를 나타내는 여러 속성들이 있는데요, 이 중 `createdAt`과 `updatedAt` 속성은 데이터가 언제 생성되었는지, 수정되었는지를 알려주는 친구들이에요. 👀

## DTO 클래스 꾸미기 🎨

이제 데이터를 주고 받을 때 사용할 DTO 클래스를 작성해볼게요. DTO 클래스는 데이터를 안전하게 전달해주는 역할을 해요. 우리가 만들 `CreateUserDto`와 `UpdateUserDto` 클래스에는 유저 정보를 받을 속성들을 정의해봅시다. 이렇게 해보세요!

```typescript
export class CreateUserDto {
  name: string;
  email: string;
  phone?: string;
}
```

이렇게 작성하면 유저를 생성할 때 필요한 정보만 담을 수 있어요. 우리가 직접 관리할 필요가 없는 `id`, `createdAt`, `updatedAt` 속성들은 제외했어요. 즉, 불필요한 정보는 받지 않겠다는 거죠! 🚀

## 마무리 🎉

여러분들도 NestJS를 사용해서 멋진 프로젝트를 시작해보세요! NestJS와 함께라면 누구든지 쉽고 빠르게 API를 개발할 수 있어요. 그럼 모두 NestJS와 함께 행복한 코딩 시간 되세요! 👋
