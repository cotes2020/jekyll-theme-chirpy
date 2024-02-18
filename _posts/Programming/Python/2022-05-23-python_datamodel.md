---
title: Python Data Model
date: 2022-05-23 19:13 +0900
categories: [Programming, Python]
tags: [python datamodel, instance method object]
---

## 객체
<hr style="border-top: 1px solid;"><br>

객체(Objects)는 파이썬이 데이터(data)를 추상화한 것(abstraction)이다. 

파이썬 프로그램의 모든 데이터는 객체나 객체 간의 관계로 표현된다고 한다.

모든 객체는 아이덴티티(id), 형(type), 값(value)를 갖는다고 한다.

<br>

파이썬에는 내장된 표준 형(type)들이 있는데 대표적으로 ```int, float, str, tuple, bytes, list, bytearray, set, dictionary, callable type, 모듈, 사용자 정의 클래스, 클래스 인스턴스``` 등 다양한 것들이 있다.

웬만해서 파이썬에 있는 모든 것이 객체라고 생각하면 된다.

<br><br>
<hr style="border: 2px solid;">
<br><br>

## callable types
### 사용자 정의 함수
<hr style="border-top: 1px solid;"><br>

함수 객체는 임의의 어트리뷰트를 읽고 쓸 수 있도록 지원한다.

어트리뷰트를 읽거나 쓸 때는 일반적인 점 표현법(dot-notation)이 사용된다. 

아직은 오직 사용자 정의 함수만 함수 어트리뷰트를 지원한다는 점을 알아둬야 한다. 

<br>

특수 어트리뷰트는 아래와 같다.

<br>

![image](https://user-images.githubusercontent.com/52172169/169804687-f399b227-b591-4b03-9a62-f9e41d5a497c.png)

<br><br>

### 인스턴스 메서드(Instance methods)
<hr style="border-top: 1px solid;"><br>

인스턴스 메서드는 클래스, 클래스 인스턴스와 모든 콜러블 객체 (보통 사용자 정의 함수)을 결합

<br>

+ 클래스 객체
  + 클래스 정의가 정상적으로 끝날 때, 클래스 객체가 만들어진다. 즉, 클래스 자체(클래스 이름)를 뜻한다. 
  + 클래스 객체는 두 종류의 연산을 지원한다. (어트리뷰트 참조, 인스턴스 생성)
  + 어트리뷰트 참조 : ```ex) Myclass.i, Myclass.f()```
  + 인스턴스 생성 : ```ex) x = Myclass()```

<br>

+ 인스턴스 객체
  + 인스턴스 객체는 한 가지 연산만 지원한다. (어트리뷰트 참조)
    + 어트리뷰트에는 두 가지 종류가 있다. (데이터 어트리뷰트, 메서드)
  + ```ex) x.f()는 함수 객체가 아니라 메서드 객체```
 
<br>

+ 읽기 전용 특수 어트리뷰트

  + ```__self__``` : 클래스 인스턴스 객체

  + ```__func__``` : 함수 객체

  + ```__doc__``` : 메서드 설명 == ```__func__.__doc__```

  + ```__name__``` : 메서드 이름 ==  ```__func__.__name__```

  + ```__module__```: 메서드가 정의된 모듈의 이름, 없는 경우 ```None```

<br>

인스턴스 메서드 객체가 호출될 때, 기반을 두는 함수 ```(__func__)``` 가 호출되는데, 인자 목록의 앞에 클래스 인스턴스 ```(__self__)``` 가 삽입된다.

예를 들어, class A 가 함수 ```test()``` 의 정의를 포함하는 클래스이고, b 가 A 의 인스턴스일 때, ```b.test()``` 를 호출하는 것은 ```A.test(b)``` 을 호출하는 것과 같다.

<br>

```python
class A():
	def test(self) :
		pass
	
b = A()

print(A.__dict__)
print(A.__name__) # A 
# print(a.__name__ ) --> error
print(b.__dict__) # {}
print(b.test.__name__) # test
# print(a.test.__func__.__name__) --> error
print(b.test.__func__) 
print(b.test.__self__)
print(b.test.__module__) # __main__
```

<br><br>

## 모듈
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://ind2x.github.io/posts/python_builtins_and_special_attributes/#object__dict__" target="_blank">ind2x.github.io/posts/python_builtins_and_special_attributes/#object__dict__</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>

이 외에도 ssti나 jailbreak 공격을 할 때, 알아둬야 할 것들이 너무나 많이 있으므로 틈 날때마다 공부해야 할 듯 하다..

<br><br>

## 출처
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://docs.python.org/ko/3/reference/datamodel.html" target="_blank">docs.python.org/ko/3/reference/datamodel.html</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
