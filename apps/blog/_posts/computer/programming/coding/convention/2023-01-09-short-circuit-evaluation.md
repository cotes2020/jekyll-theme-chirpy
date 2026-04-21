---
title: "Short-Circuit Evaluation"
# description: ""
categories: [컴퓨터, 프로그래밍, Convention]
tags: []
image: "/assets/img/background/kururu-lab.jpg"

date: 2023-01-09. 22:02
last_modified_at: 2023-01-09. 22:02
---

## Short-Circuit Evaluation

---

단락 방식  

[참고 - 위키피디아](https://en.wikipedia.org/wiki/Short-circuit_evaluation), [참고 - 제로니모님의 글](https://code-lab1.tistory.com/68)  

대부분의 프로그래밍 언어에서는, 불 연산에 Short-Circuit Evaluation 이 적용된다.  

![그림](/assets/img/post/stone/2023/230109-0000.jpg)

위 그림처럼,  
순서대로 연산하는 도중 이미 결과가 결정되는 경우,  
굳이 남은 피연산자들까지 계산할 필요가 없기 때문에, 결과가 결정된 시점에서 계산을 마친다.  

```cs
SomeObject a = null;
if (a != null && a.SomeFunc())
```

C# 에서 `&&` 은 Short-Circuit Evaluation 이 적용되기 때문에,  
위 코드에서는 `a != null` 에서 계산이 끝나고, `a.SomeFunc()` 가 실행되지 않아 런타임 에러가 발생하지 않는다.  
이런식으로 활용할 수 있다.  

만약 Short-Circuit Evaluation 이 적용되지 않았다면,  
위 코드에서 `a` 가 `null` 인채로 `a.SomeFunc()` 가 실행되어 `NullReferenceException` 이 발생했을 것이다.  

```cs
// 1: 최소 1ms
if (takes1ms() || takes1s())
// 2: 최소 1s
if (takes1s() || takes1ms())

// a: 최소 O(1)
if (boolVariable || take1s())
// b: 최소 1s
if (takes1s() || boolVariable)
```

또, 만약 함수의 반환 값을 피연산자로 사용하는 경우,  
상대적으로 비용이 높은 함수를 뒷쪽에 배치하여 시간 복잡도를 줄일 수도 있다.  

```cs
if (false & SomeFunc())
if (true | SomeFunc())
```

반면, 결과와 상관없이 피연산자로 존재하는 모든 함수들을 실행시키고 싶은 경우가 있을 수 있다.  
(예시가 딱 떠오르지는 않지만)  

이 경우, C# 에서는 `&` 이나 `|` 를 사용할 수 있다.  
`&` 와 `|` 는 비트 연산자로 쓰이기도 하지만, 불 연산 식에 사용될 경우, 불 논리 연산자로써 사용된다.  

`&` 이나 `|` 를 불 논리 연산자로써 사용하면, 결과와 상관없이 모든 피연산자들을 계산한다.  
때문에 위 두 줄의 코드는, 첫 번째 피연산자에서 이미 결과가 결정되어버리지만, 계산을 끝내지 않고 SomeFunc() 함수를 실행시킨다.  

이때, 이런 `&` 와 `|` 를 `&&` 와 `||` 에 구분지어,  
Eager Operators (`&`, `|`) 와 Short-Circuit Operators (`&&`, `||`) 로 부를 수 있다.  

---

C# 튜플 비교 시, Short-Circuit Evaluation 이 적용된다.  
