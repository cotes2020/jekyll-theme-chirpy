---
title: "연산 순서에 따른 퍼포먼스 차이"
# description: ""
categories: [컴퓨터, 프로그래밍, Convention]
tags: []
image: "/assets/img/background/kururu-lab.jpg"

date: 2022-01-02. 12:07
# last_modified_at: 2023-11-26. 06:53
last_modified_at: 2024-08-29. 21:33
---

## 연산 순서에 따른 퍼포먼스 차이

---

### 시간 복잡도 차이

@ VS 2022 에서 제안하는 것들 중 도움되는 것이 참 많은 것 같다.  

![참고](/assets/img/post/stone/2022/220102-0000.png)

연산 순서에 따라 퍼포먼스 차이가 발생하는 경우.  

### 결과 차이

```txt
( 3.14 + 1e20 ) - 1e20 = 0.1 - 1e20
3.14 + ( 1e20 - 1e20 ) = 3.14
```

연산 순서에 따라 결과 차이가 발생하는 경우. (Overflow)  

## 메모

---

### 참고 - [Short-Circuit Evaluation](/posts/short-circuit-evaluation/)

_230109.  
Short-Circuit Evaluation 에 따른 연산 순서 상의 속도 차이도 존재한다.  
요약하면 비용이 높은 함수를 뒤쪽에 배치하면 좋다.  
