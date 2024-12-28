---
title: function vs method
date: "2024-12-28T23:46:26+09:00"
categories: [Knowledge, IT]
tags: [class,method,function,용어]
description: 함수? 메소드? 뭐가 다르지?
author: hoon
---
## 메소드는 함수의 한 종류
함수는 `any block of code designed to perform a task`라고 정의할 수 있다.

함수는 
1. 어떤 오브젝트나 클래스에 종속될 수도 있고,
2. 종속되지 않을 수 도 있다.

메소드는 위에서 1번에 해당하는 경우의 함수를 특별히 부르는 단어인 것이다.

따라서 **메소드는 함수의 한 종류**라고 이해해도 무리가 아니다.

## 예시
`add()`함수가 어떻게 정의되었는지에 따라 함수와 메소드로 분류할 수 있다.

### 함수
```python
# This is a standalone function
def add(a, b):
    return a + b

# Calling the function directly
result = add(3, 4)
print(result)  # Output: 7
```
### 메소드
```python
# This is a class with a method
class Calculator:
    def add(self, a, b):
        return a + b

# Creating an instance of the class
calc = Calculator()

# Calling the method on the instance
result = calc.add(3, 4)
print(result)  # Output: 7
```