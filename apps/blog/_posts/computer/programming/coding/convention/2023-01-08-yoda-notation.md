---
title: "Yoda Notation"
# description: ""
categories: [컴퓨터, 프로그래밍, Convention]
tags: []
image: "/assets/img/background/kururu-lab.jpg"

date: 2023-01-08. 18:08
# last_modified_at: 2023-01-08. 18:08
---

## Yoda Notation(Condition)

---

[참고](https://en.wikipedia.org/wiki/Yoda_conditions)  

```cs
// 1
if (length >= 10)

// 2
if (10 <= length)
```

"180보다 내 키가 컸다면.." 은 아무래도 부자연스럽다.  
같은 이유로 2번 조건문도 부자연스럽다.  
"10이 length보다 작거나 같다면" 은 아무래도 부자연스럽다.  

이처럼 비교 조건문을 적을 때는, "내 키가 180보다 컸다면.." 같은 어순과 동일하게,  
왼쪽에 변하는 값(질문을 받는), 오른쪽에 정해진 값(비교 대상으로 사용 되는) 으로 적는 것이 일반적이다.  
"length가 10보다 작거나 같다면" 은 상대적으로 자연스럽다.  

반면, 의도적으로 2번 조건문처럼 적는 경우가 있다.  

```cs
if (intVar = 10)
// intVar == 10 이라고 적었어야 했는데 !
```

위처럼 == 를 써야 하는 상황에서,  
실수로 = 만 적어버리는 실수를 할 때가 있다.  

이는 분명 논리적인 오류가 존재하지만, 코드 상의 오류는 존재하지 않기 때문에,  
컴파일 에러가 발생하지 않는다.  

때문에 이를 놓치는 경우,  
직접 버그를 찾아야 하는 경우가 발생할 수 있다.  

```cs
if (10 = intVar)
// 컴파일 에러 !
```

이때, 위처럼 피연산자들의 순서를 바꾸게 된다면,  
상수에 변수를 대입할 수는 없기에, 컴파일 에러가 발생하게 된다.  
이를 통해 실수를 사전에 방지할 수 있다!  

이처럼 의도적으로 읽기에 부자연스러운 순서로 비교 조건문을 적는 방법을,  
스타워즈의 등장인물, 요다가 어색한 어순으로 말한다는 특징에서 따와,  
[요다](https://namu.wiki/w/%EC%9A%94%EB%8B%A4#s-2) 표기법(조건) 이라고 한다.  

요다 표기법에는 또 다른 이점도 있다.  

```java
// 1
String myString = null;
if (myString.equals("foobar")) { /* ... */ }
// This causes a NullPointerException in Java

// 2
String myString = null;
if ("foobar".equals(myString)) { /* ... */ }
// This resolves to false without throwing a NullPointerException
```

위는 Wikipedia의 Yoda_conditions 문서의 예제다.  
위같은 상황에서는 예외 발생을 피할 수 있게 해주는 역할도 하게 된다.  

이러한 이점들이 분명 존재하기는 하지만,  

![앗](/assets/img/post/stone/2023/230108-0000.jpg)

최신 컴파일러/IDE 에서는 알잘딱으로 실수에 대해 경고를 표기해주기도 하고,  
애초에 할당 연산 시 반환값이 존재하지 않거나, 조건문에 할당 연산문을 허용하지 않는 언어들도 있고,  
무엇보다 가독성을 해치기 때문에,  

'요다 표기법은 불필요한 과거의 것이 되어 가고 있다.'  
라는 '읽기 좋은 코드가 좋은 코드가' 저자의 의견처럼,  
요즘 들어서는 그렇게 쓰이지 않는 모양인 것 같다.  
