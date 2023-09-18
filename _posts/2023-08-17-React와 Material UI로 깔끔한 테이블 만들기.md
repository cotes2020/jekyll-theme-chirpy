---
title: React와 Material UI로 깔끔한 테이블 만들기
date: 2023-08-17 20:00:00 +0900
categories:
  - React
tags:
  - MaterialUI
---

## 테이블을 만들어보자! 🚀
저희는 여러분에게 React와 Material UI로 완벽하게 빛나는 테이블을 만드는 방법을 알려드릴 겁니다. 그래서 여러분은 이 글을 읽고 나면, 테이블 UI를 마스터 할 수 있을 거예요! 🌟

## 필수 구성 요소
테이블을 만들려면 어떤 컴포넌트가 필요할까요? 그건 바로 `Table`, `TableBody`, `TableCell`, `TableContainer`, `TableHead`, `TableRow`입니다. 이 컴포넌트들을 모아서 @material-ui/core 패키지에서 한 번에 불러옵니다. 코드는 이렇게 생겼어요:

```javascript
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from "@material-ui/core";
```

## 테스트 데이터 만들기: faker의 마법 🎩
여러분은 테스트 데이터가 필요합니다. 그래서 `faker` 라이브러리를 이용해서 다양한 사용자 정보를 생성합니다. `faker`가 없다면 테스트 데이터를 직접 입력해야 하죠, 그건 너무 지루해요! 🙄 코드는 아래와 같습니다.

```javascript
import faker from "faker/locale/ko";
faker.seed(123);
const users = Array(53).fill().map(() => ({
  id: faker.random.uuid(),
  name: faker.name.lastName() + faker.name.firstName(),
  email: faker.internet.email(),
  phone: faker.phone.phoneNumber(),
}));
```

## 기본 테이블 작성하기: 먼저 기초부터! 🏗️
기본적인 테이블은 어떻게 만들까요? 그건 바로 `TableHead`와 `TableBody`를 이용해서 각각 레이블과 데이터를 표시하면 돼요. 아, 너무 쉬워서 놀랐죠? 😲 코드는 이렇게 작성합니다.

```javascript
function UserTable() {
  return (
    <TableContainer component={Paper}>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>No</TableCell>
            <TableCell align="right">Name</TableCell>
            <TableCell align="right">Email</TableCell>
            <TableCell align="right">Phone</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {users.map(({ id, name, email, phone }, i) => (
            <TableRow key={id}>
              <TableCell>{i + 1}</TableCell>
              <TableCell align="right">{name}</TableCell>
              <TableCell align="right">{email}</TableCell>
              <TableCell align="right">{phone}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
```

## 페이징도 있어요: 페이지 넘기기로 편안함을! 📖
많은 데이터를 보여주려면 페이징이 필요해요. Material UI에서는 `TablePagination` 컴포넌트를 제공합니다. 덕분에 페이징도 쉬워요. 헤헤 😄

```javascript
import {
  TableFooter,
  TablePagination,
} from "@material-ui/core";

// ...이하 생략
```


이렇게 하면 여러분도 테이블 마스터가 될 수 있습니다! React와 Material UI로 만든 테이블, 어때요? 나머지 활용은 여러분의 손에 달렸어요! 🎉🎉
