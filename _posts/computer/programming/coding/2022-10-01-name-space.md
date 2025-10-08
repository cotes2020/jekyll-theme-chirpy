---
title: "네임스페이스"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2022-10-01. 11:03
# last_modified_at: 2022-10-01. 11:03
---

## Namespace

---
CPP의 경우

네임스페이스는 식별자 (자료형, 함수, 변수 등의 이름)의 영역

:: 연산자  
네임스페이스 (이름공간) 을 지정하는 연산자

```cpp
double Circle::CalcArea()
{
    // ... //
}
```

함수가 정의된 이름공간이 클래스임을 지정하는 것이라도 생각해도 된다

이름 공간은 코드를 논리적 그룹으로 구성  
특히 코드에 여러 라이브러리가 포함되어 잇을 때 발생할 수 잇는 이름 충돌을 방지하는데 사용

모든 변수 이름 앞에 이름 공간을 붙일 수도 있고 아니면  
`using namespace std;` 와 같은 선언문을 사용하여서 현재의 이름공간을 지정하여도 된다

하나의 프로그램에서 여러 개의 이름 공간을 사용할 수 있다

## `::`

---

[https://stackoverflow.com/questions/4269034/what-is-the-meaning-of-prepended-double-colon/4269060#4269060](https://stackoverflow.com/questions/4269034/what-is-the-meaning-of-prepended-double-colon/4269060#4269060)

네임스페이스 없이 `::` 를 쓰면 Global Namespace를 사용  
