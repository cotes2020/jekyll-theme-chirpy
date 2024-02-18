---
title: Python builtins 모듈과 특수 어트리뷰트
date: 2022-05-22 22:06 +0900
categories: [Programming, Python]
tags: [python built-in, python special attributes, __builtins__, dir(), __dict__]
---

## builtins module
<hr style="border-top: 1px solid;"><br>

Python Built-in
: <a href="https://docs.python.org/ko/3/library/builtins.html" target="_blank">docs.python.org/ko/3/library/builtins.html</a>

<br>

파이썬에는 내장 함수, 내장 상수, 내장 모듈 등 내장 객체들이 있는데, 이런 내장 식별자들을 ```builtins```으로 모두 접근 가능하다.
: ```import builtins```

<br>

대부분 모듈은 전역 변수로 ```__builtins__``` 라는 변수를 가지고 있다.

```builtins``` 또한 모듈이므로 ```__builtins__```로 사용할 수 있다.

<br>

```dir()```을 사용하면 객체에 저장된 유효한 attribute(클래스의 메소드와 변수) 리스트를 반환한다.
: <a href="https://docs.python.org/ko/3/library/functions.html#dir" target="_blank">docs.python.org/ko/3/library/functions.html#dir</a>

<br>

따라서 ```dir(__builtins__) 또는 __builtins__.__dir__()```를 하면 내장된 함수, 객체, 상수 등을 확인할 수 있다.

<br><br>
<hr style="border: 2px solid;">
<br><br>


## Special Attribute
<hr style="border-top: 1px solid;"><br>

Python Special Attributes
: <a href="https://docs.python.org/ko/3/reference/datamodel.html" target="_blank">docs.python.org/ko/3/reference/datamodel.html</a>
: <a href="https://docs.python.org/ko/3/library/stdtypes.html#special-attributes" target="_blank">docs.python.org/ko/3/library/stdtypes.html#special-attributes</a>

<br>

위의 링크에서 읽어내려가다 보면 특수 어트리뷰트에 대한 설명이 있다. 

특수 어트리뷰트 뿐만 아니라 파이썬의 어트리뷰트에 대해 이해하고자 하면 위의 문서를 숙지해야 한다!!

<br>

![image](https://user-images.githubusercontent.com/52172169/169731304-6ae98924-61a2-4557-a15d-5f23f1f5a00e.png)

<br>

![image](https://user-images.githubusercontent.com/52172169/169847109-f79c4513-680a-464a-b539-8cdaace31762.png)

<br>

### ```object.__dict__```
<hr style="border-top: 1px solid;"><br>

특수 attribute 중 ```__dict__```가 있는데, ```__dict__```는 객체의 사용 가능한 attribute를 저장한 딕셔너리나 또는 기타 매핑 객체를 리턴한다.
: <a href="https://docs.python.org/ko/3/tutorial/classes.html#id2" target="_blank">docs.python.org/ko/3/tutorial/classes.html#id2</a>
: <a href="https://docs.python.org/ko/3/library/stdtypes.html#object.__dict__" target="_blank">docs.python.org/ko/3/library/stdtypes.html#object.__dict__</a>

<br>

![image](https://user-images.githubusercontent.com/52172169/169746216-8f1df6c6-ef2f-441a-9482-a72b48bb61b3.png)

<br>

사진 출처
: <a href="https://docs.python.org/ko/3/reference/datamodel.html" target="_blank">docs.python.org/ko/3/reference/datamodel.html</a>

<br>

사진을 보면 모듈을 호출할 때, import 문 또는 ```__import__()``` 함수를 호출한다고 한다.

```__import__()```는 내장함수로 <a href="https://docs.python.org/ko/3/library/functions.html#import__" target="_blank">파이썬 내장함수 문서</a>에서 ```__import__()``` 를 보면 모듈을 임포트한 뒤 ```__builtins__.__import__```에 대입한다고 한다.

그러면 ```__builtins__.__import__```에 내가 임포트하고자 하는 모듈명을 넣어주면 된다.
: ```ex) __builtins__.__import__('os')```

<br>

만약 우리가 os 모듈의 system 메소드를 호출하고 싶다고 하면 아래와 같이 코드를 짜거나 한 줄로 호출할 수 있다.

<br>

```python
import os

system('ls')

# 또는

__import__('os').system('ls')

__builtins__.__import__('os').system('ls')

__builtins__.__import__('os').__dict__['system']('ls')

__builtins__.__dict__['__import__']('os').system('ls')
```

<br>

또한 모듈 ```os```이 있고 어트리뷰트로 ```system```이 있을 때, ```os.system```은 ```os.__dict__["system"]```와 같다. 

<br>

```python
import os

print(dir(os))            # os 모듈에 저장된 attribute 출력

print(dir(os.__dict__))   # os.__dict__는 dictionary를 리턴하므로 dir(dict())와 동일한 결과 출력
                          # 즉, dictionary 클래스의 메소드가 출력됨

print(os.__dict__.keys()) # os 모듈의 사용 가능한 attribute를 저장한 딕셔너리의 키를 출력


os.system('ls')
os.__dict__['system']('ls')
```

<br>

위와 같은 코드를 실행시켜보면 os 모듈의 사용 가능한 attribute가 출력되는 걸 확인할 수 있다.

두 개의 print문의 결과는 확인해보면 비슷하다는 걸 알 수 있다.

<br><br>

### ```instance.__class__```
<hr style="border-top: 1px solid;"><br>

클래스 인스턴스가 속한 클래스를 리턴한다.
: 어떤 객체로 부터 생성됬는지 판단할 때 사용하는 attribute

<br>

```python
class A():
	def test(self) :
		pass
	
b = A()

print(b.__class__)    # <class '__main__.A'>

print(''.__class__)   # <class 'str'>
```

<br>

<br><br>

### ```class.__bases__```
<hr style="border-top: 1px solid;"><br>

클래스 객체의 베이스 클래스들의 튜플 리턴.

<br>

```python
class A():
	def test(self) :
		pass
	
b = A()

print(A.__bases__)    # (<class 'object'>,)
print(str.__bases__)  # (<class 'object'>,)

# 모든 형은 최상위 클래스가 object 클래스이다.
```

<br>

<br><br>

### ```definition.__name__```
<hr style="border-top: 1px solid;"><br>

클래스, 함수, 메서드, 디스크립터 또는 제너레이터 인스턴스의 이름을 리턴.

<br>

```python
class A():
	def test(self) :
		pass
	
b = A()

print(A.__name__)     # A
```

<br>

```python
def f() :
	pass

print(f.__name__)     # f
```

<br>

<br><br>

### ```class.__mro__```, ```class.mro()```
<hr style="border-top: 1px solid;"><br>

+ ```class.__mro__``` 
  + 이 어트리뷰트는 메서드 결정 중에 베이스 클래스를 찾을 때 고려되는 클래스들의 **튜플**이다.

<br>

+ ```class.mro()```
  + 이 메서드는 인스턴스의 메서드 결정 순서를 사용자 정의하기 위해 메타 클래스가 재정의할 수 있다. 
  + 클래스 인스턴스를 만들 때 호출되며 그 결과는 ```__mro__``` 에 저장됩니다.

<br>

```python
class A():
	def test(self) :
		pass
	
b = A()

print(A.__mro__)     # (<class '__main__.A'>, <class 'object'>)
                     # __mro__는 튜플

print(A.mro())       # [<class '__main__.A'>, <class 'object'>] 
                     # 리스트 반환


print(str.__mro__)   # (<class 'str'>, <class 'object'>)
print(str.mro())     # [<class 'str'>, <class 'object'>]
```

<br>

<br><br>

### ```class.__subclasses__()```
<hr style="border-top: 1px solid;"><br>

각 클래스는 직계 서브 클래스에 대한 약한 참조의 리스트를 유지합니다. 

이 메서드는 아직 살아있는 모든 참조의 리스트를 돌려줍니다. 

리스트는 정의 순서대로 되어 있습니다.

<br>

```python
class A():
	pass
		
class B(A) :
	pass	

class C(B):
	pass


print(A.__subclasses__())     # [<class '__main__.B'>]
print(B.__subclasses__())     # [<class '__main__.C'>]
print(C.__subclasses__())     # []
```

<br>

```python
int.__subclasses__()    # [<class 'bool'>]

print(object.__subclasses__()) 
# [<class 'type'>, <class 'weakref'>, ..., <class 'rlcompleter.Completer'> ]
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 참고
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://dokhakdubini.tistory.com/471" target="_blank">dokhakdubini.tistory.com/471</a>
: <a href="https://docs.python.org/ko/3/reference/datamodel.html" target="_blank">docs.python.org/ko/3/reference/datamodel.html</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>
