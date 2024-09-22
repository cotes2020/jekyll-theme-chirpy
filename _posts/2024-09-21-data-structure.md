---
title: 자료구조
author: jinwoo
date: 2024-09-22 08:30:00 +0900
categories: [Study, Data structure]
tags: [data structure]
render_with_liquid: false
description: 자료구조 정의, 성능
---


## 정의

### 데이터 (Data)

- 모든 유형의 정보를 망라하는 용어
- 일반적으로 수와 문자열의 조합으로 이루어짐

```python
x = "Hello "
y = "world"
z = "!"
print(x+y+z)
```

### 자료구조 (Data structure)

- 데이터를 구조화하여 조작하는 방법
- 구조에 따라 코드의 실행속도에 미치는 영향이 큼

```python
array = ["Hello ", "world", "!"]
for a in array:  
  print(a, end="") # 개행없이 print
```

## 연산

자료구조의 성능을 확인하기 위해서는 코드와 자료구조가 어떻게 상호작용하는지 분석이 필요하다.

대부분의 자료구조는 아래에 기술한 네 가지 기본방법을 주로 사용한다.

### 읽기

- 자료구조의 특정 위치(index)를 찾아보는 것
- index 제공한 다음 값을 반환받는 구조
- 일반적으로 index는 컴퓨터의 메모리 주소와 mapping 되어있다고 볼 수 있음

### 검색

- 자료구조에서 특정 값이 있는지 알아본 후, 어떤 위치(index)에 있는지 찾기
- 값을 제공한 다음 index를 반환받은 구조
- 컴퓨터는 각 메모리 주소에 할당된 값을 바로 알지 못하기에 각 셀을 하나씩 조사하는 방법밖에 없음

### 삽입

- 자료구조에 새로운 값을 추가하는 것
- 삽입되는 위치에 따라 성능이 달라짐

### 삭제

- 자료구조에서 값을 제거하는 것
- 삭제되는 위치에 따라 성능이 달라짐

## 성능 

시간은 연산을 담당하는 하드웨어 성능에 따라 다르기에 절대적인 수치가 될 수 없다.

성능을 측정할 때는 순수하게 `시간` 관점에서 연산이 얼마나 빠른가가 아니라 얼마나 많은 `단계`가 필요한 지를 검토해야한다.

100 이하의 짝수를 출력하는 함수 2개를 예시로 들어본다.

```python
# 100 단계 진행
def print_even_v1:
    n = 1
    while n <= 100:
        if n % 2 == 0:
          print(n)
        n += 1
        
# 50 단계 진행
def print_even_v2:
    n = 2
    while n <= 100:
      print(n)
      n += 2
```

`print_even_v1`, `print_even_v2` 모두 같은 결과를 출력한다. 

하지만, 루프를 보면 `print_even_v1`은 `print_even_v2` 보다 2배의 단계가 필요하다.

그렇기에 `print_even_v2`가더 성능이 좋은 것으로 볼 수 있다.
