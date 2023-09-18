---
title: React Table로 데이터를 쉽고 멋지게
date: 2023-08-06 20:00:00 +0900
categories:
  - React
tags:
  - ReactTable
---

## 소개: React Table이란 무엇인가요?

안녕하세요, 여러분! 오늘은 '테이블'을 다룰 예정이에요. 리액트를 사용하여 편리하고 멋진 테이블을 만들고 싶다면, React Table을 꼭 알아봐야 합니다! 이 라이브러리는 정렬, 검색 같은 여러 기능을 쉽게 추가할 수 있게 도와줍니다. ✨

## 시작하기 전에: 필요한 준비물 📦

먼저, 터미널에서 `react-table` 패키지를 설치해 줍니다. 그리고 `faker` 라이브러리도 설치해 주세요. 이건 랜덤한 테스트 데이터를 만들어주는 역할을 합니다.

```javascript
npm install react-table faker
```

## 첫 번째 단계: 랜덤 데이터 생성하기 🎲

랜덤 데이터를 만드는 건 정말 쉬워요. `faker` 라이브러리를 이용하면 몇 줄 안에 끝납니다.

```javascript
import faker from "faker/locale/ko";

faker.seed(100);
const data = Array(53).fill().map(() => ({
  name: faker.name.lastName() + faker.name.firstName(),
  email: faker.internet.email(),
  phone: faker.phone.phoneNumber(),
}));
```

이렇게 하면 이름, 이메일, 전화번호가 들어간 53개의 랜덤 데이터가 생깁니다.

## 두 번째 단계: 테이블 만들기 🛠️

### 기본 테이블 만들기

React로 기본 테이블을 만들어 볼게요. `Table` 컴포넌트를 만들어서 `columns`과 `data`를 props로 받습니다.

```javascript
import React from "react";

function Table({ columns, data }) {
  return (
    <table>
      <thead>
        <tr>
          {columns.map((column) => (
            <th key={column}>{column}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.map(({ name, email, phone }) => (
          <tr key={name + email + phone}>
            <td>{name}</td>
            <td>{email}</td>
            <td>{phone}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default Table;
```

### React Table로 심화 테이블 만들기

이제 본격적으로 React Table을 이용해서 테이블을 만들어볼게요. 다음과 같이 코드를 작성해 주세요.

```javascript
import React from "react";
import { useTable } from "react-table";

function Table({ columns, data }) {
  const {
    getTableProps,
    getTableBodyProps,
    headerGroups,
    rows,
    prepareRow,
  } = useTable({ columns, data });

  return (
    <table {...getTableProps()}>
      <thead>
        {headerGroups.map((headerGroup) => (
          <tr {...headerGroup.getHeaderGroupProps()}>
            {headerGroup.headers.map((column) => (
              <th {...column.getHeaderProps()}>{column.render("Header")}</th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody {...getTableBodyProps()}>
        {rows.map(row => {
          prepareRow(row);
          return (
            <tr {...row.getRowProps()}>
              {row.cells.map(cell => {
                return <td {...cell.getCellProps()}>{cell.render('Cell')}</td>
              })}
            </tr>
          )
        })}
      </tbody>
    </table>
  );
}
```

와우, 이제 여러분도 실제 프로젝트에서 이를 활용해보세요! 😎🎉
