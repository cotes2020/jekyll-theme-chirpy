---
title: Python MRO(Method Resolution Order)
date: 2022-05-22 09:32 +0900
categories: [Programming, Python]
tags: [Method Resolution Order, MRO]
---

## MRO (Method Resolution Order)
<hr style="border-top: 1px solid;"><br>

파이썬의 상속과 관련 있는 개념으로, 메소드 결정 순서이다.

수 많은 부모 클래스로부터 상속받은 자식 클래스가 있을 때, ```만약 상속받은 메소드를 실행한다면 어느 부모 클래스에 있는 메소드를 호출해야 하는가?``` 라는 문제가 있는데 이를 ```죽음의 다이아몬드 문제``` 라고 한다.

이러한 문제를 피하고자 Python에서는 MRO 즉, 메소드 결정 순서를 도입하였다. 

<br>

MRO는 자식과 부모 클래스를 전부 포함하여 메소드의 실행 순서를 지정하는 것이다. 

따라서 동일한 이름의 메소드가 호출되어도 자식 클래스부터 부모 클래스까지 지정된 순서대로 호출하면 되므로 문제가 발생하지 않는다.

<br>

동작 방식은 호출된 자식 클래스를 먼저 확인하고, 그 다음에는 상속된 클래스들을 나열한 순서대로 확인한다.

```class.__mro__``` 또는 ```class.mro()```로 지정된 순서를 확인할 수 있다.

자세한 내용은 <a href="https://xo.dev/python-method-resolution-order/" target="_blank">xo.dev/python-method-resolution-order/</a> 여기서 확인
: 정리 잘 되어 있음.

<br>

주의할 점은 중복 상속은 허용되나, 교차 상속은 허용되지 않는다는 점이다.

<br>

```python
class A()

class B()

class C(A,B)

class D(B,A)
```

<br>

위와 같은 코드가 있을 때, C와 D 클래스는 모두 A,B 클래스를 상속받으나 순서를 다르게 해서 상속을 하였다.

이는 교차 상속이 되는데 이렇게 되면 어떤 부모 클래스를 먼저 방문해야 하는지 순서를 지정하기 모호하게 되므로 오류가 발생한다.
: ```>>> TypeError: Cannot create a consistent method resolution```
: ```>>> order (MRO) for bases A, B```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 참고
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://xo.dev/python-method-resolution-order/" target="_blank">xo.dev/python-method-resolution-order/</a>
: <a href="https://tibetsandfox.tistory.com/26" target="_blank">tibetsandfox.tistory.com/26</a>
: <a href="https://dev-navill.tistory.com/4" target="_blank">dev-navill.tistory.com/4</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
