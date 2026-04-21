---
title: "Boolean 크기가 1바이트인 이유"
# description: ""
categories: [컴퓨터, 시스템]
tags: []
image: "/assets/img/background/kururu-lab.jpg"

date: 2023-03-16. 10:51
# last_modified_at: 2023-11-08. 14:55
# last_modified_at: 2024-08-29. 21:27
last_modified_at: 2024-10-20. 13:11 # 정리
---

## Boolean 크기가 1바이트인 이유

---

1과 0, 참과 거짓.  
딱 두 가지 상태만을 가지는 `Boolean`.  

`Boolean`이 1-`Bit`가 아니라 1-`Byte` 씩이나 용량을 차지하는 이유는,  

현대 컴퓨터가 데이터나 주소에 접근하는 최소 단위가 `Bit`가 아니라 `Byte` 라 그렇다.  
다시 말해, 컴퓨터가 데이터를 `Bit` 가 아니라 최소 `Byte` 단위로 저장하기 때문이다.  

CPU 구현에 따른 컴퓨터의 최소 단위를 `Word`라고 하는데,  
현대 컴퓨터 구조 상 일반적으로 1-`Word`는 1-`Byte` = 8-`Bit`다.

## 같이 읽으면 좋은 글

---

- [Bit, Byte, Word](/posts/bit-byte-word/)

## 메모

---

- [참고 - 스택오버플로우](https://stackoverflow.com/questions/2064550/)
- [참고 - 1바이트는 왜 8 비트인가](https://zepeh.tistory.com/313)
