---
title: "[TIL] react query - useQuery, useMutation 이해하기"
date: 2022-07-13 00:56:00 +0900
categories: [TIL]
tags: [react, react-query]
---

# [TIL] react query - useQuery, useMutation 이해하기

## 참고자료

[react-query 공식 문서](https://react-query-beta.tanstack.com/) <br>
[카카오페이 프론트엔드 개발자들이 React Query를 선택한 이유](https://tech.kakaopay.com/post/react-query-1/) <br>
[React Query 사용법](https://velog.io/@leehyunho2001/React-Query-uuj3rjo7)

## 0. 소개

이번 프로젝트에 react-query 라이브러리를 도입했다. 처음 써보는거라 간단히 정리해보고자 한다.

## 1. react query란?

- 전역 상태 변경 없이 데이터를 가져오고 캐싱하고 업데이트 할 수 있는 라이브러리
- 서버 상태를 가져오고 캐싱하고 동기화하고 업데이트 하는 작업을 쉽게 만들어 줌
  - 비동기 함수와 그것의 리턴값에 대한 데이터 캐싱 등의 기능.
  - 단, promise를 리턴하는 코드는 직접 구현해야 함
- 나온지 오래되지 않아서 커뮤니티 규모가 크진 않음
- 특징
  - 알아서 해주는 데이터 패칭 로직
  - 데이터와 에러를 처리할 함수만 알려주면 됨 (간단)
  - 대충 promise만 다룰 수 있으면 됨 (친숙)
  - 쿼리의 모든 observer 객체마다 특별한 설정 추가 가능

<br>

> 「if(kakao)2021 - 카카오페이 프론트엔드 개발자들이 React Query를 선택한 이유」 세줄요약

1. React Query는 React Application에서 서버 상태를 불러오고, 캐싱하며, 지속적으로 동기화하고 업데이트하는 작업을 도와주는 라이브러리입니다.
2. 복잡하고 장황한 코드가 필요한 다른 데이터 불러오기 방식과 달리 React Component 내부에서 간단하고 직관적으로 API를 사용할 수 있습니다.
3. 더 나아가 React Query에서 제공하는 캐싱, Window Focus Refetching 등 다양한 기능을 활용하여 API 요청과 관련된 번잡한 작업 없이 "핵심 로직"에 집중할 수 있습니다.

## 2. TIL

- 상황

  - post, put 요청을 보내 user 정보 생성/수정
  - 요청이 success이면 modal 지우고 navigate
  - react query는 query와 mutation 두 가지 유형이 있음
  - 나는 여기서 사용자 정보를 입력(post)하고 수정(put)해야 하므로 mutation 사용
  - 확인 버튼을 누르면 form이 submit되어 mutate요청을 보내야 하는데 안됨
  - 확인 버튼을 한번 더 눌러야 했음
  - 콜백함수 안써서 생긴 문제..

- 오류 코드

```typescript
const {mutate, isSuccess} = useMutation(editUser);
...

if (isSuccess) {
      modalHandler();
      navigate('/main');
}
```

- 수정 코드

```typescript
const { mutate } = useMutation(editUser, {
  onSuccess: () => {
    modalHandler();
    navigate("/main");
  },
});
```
