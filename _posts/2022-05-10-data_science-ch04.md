---
title: "데이터과학 ch04. 선형대수"
excerpt: "책 제목: 밑바닥부터 시작하는 데이터과학"

categories:
- DataScience
tags:
- [DataScience]
use_math: true

permalink: /categories/DataScience/Ch04

toc: true
toc_sticky: true

date: 2022-05-10
last_modified_at: 2022-05-11
---

# 벡터
- 벡터란, 벡터끼리 더하거나 상수끼리 곱해지면 새로운 벡터를 생성하는 개념적인 도구
- 어떤 유한한 차원의 공간에 존재하는 점
- 벡터를 가장 간단하게 표현하는 방법은 `List`이다.
- ex. 키, 몸무게, 나이를 3차원의 벡터로 표현가능함. 
    ```python
    from typing import List
    Vector = List[float]

    height_weight_age = [70, # 인치,
                        170, # 파운드,
                        40 ] #나이
    ```

## Python code로 나타내기
- 벡터의 산술연산은 `zip`을 사용하여 두 벡터를 묶은 뒤, 각 성분에 리스트 컴프티헨션을 적용

### 더하기
```python
# 더하기
def add(v: Vector, w:Vector) -> Vector:
    assert len(v) == len(w), "vectors must be the same length"
    return [v_i + w_i for v_i, w_i in zip(v, w)]

assert add([1, 2, 3], [4, 5, 6] == [5, 7, 9]) 
```

### 빼기
```python
def substract(v: Vector, w:Vector) -> Vector:
    assert len(v) == len(w), "vectors must be the same length"
    return [v_i - w_i for v_i, w_i in zip(v, w)]

assert add([5, 7, 9], [4, 5, 6]) == [1, 2, 3] 
```

### 모든 성분 더하기
```python
def vector_sum(vectors: List[Vector]) -> Vector:
    assert vectors, "no vectors provieded!"

    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]
```

### 모든 성분에 scalar c 곱하기
```python
def scalar_multiply(c: float, v: Vector) -> Vector:
    return [c* v_i for v_i in v]

assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]
```

### 모든 성분 평균 구하기
```python
def vector_mean(vectors: List[Vector]) -> Vector:
    n = len(vectors)
    return scalar_multiply(1/n, vector_msum(vectors))

assert vector_mean([1, 2], [3, 4], [5, 6]) == [3, 4]
```

## 내적(dot product)
```python
def dot(v: Vector, w: Vector) -> float:
    # v_1 * w_1 + ... v_n * w_n 
    assert len(v) == len(w), "vectors must be same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))
    
assert dot([1, 2, 3], [4, 5, 6]) == 32
```

### 내적을 사용하여 벡터의 크기 계산하기
```python
import math

def sum_of_squares(v: Vector) -> float:
    return dot(v, v)

def magnitude(v: Vector) -> float:
    return math.sqrt(sum_of_squares(v)) # math.sqrt()는 제곱근을 계산해주는 함수

assert magnitude([3, 4]) == 5
```
### 두 벡터간의 거리
```python
def squared_distance(v: Vector, w: Vector)-> float:
    return magnitude(substract(v, w))
```

# 행렬(Matrix)
- 2차원으로 구성된 숫자의 집합
- 리스트의 리스트로 표현 가능 
- 수학에서는 1행, 1열이 첫번째 행, 열이지만 python에서는 0행, 0열이 첫번째 행, 열임
    ```python
    Matrix = List[List[float]]

    A = [[1, 2, 3],
        [4, 5, 6]] # 2행 3열

    B = [[1, 2],
        [3, 4],
        [5, 6]] # 3행 2열
    ```

## Python 코드로 나타내기

### (열의 개수, 행의 개수)를 반환

```python
from typing import Tuple
def shape(A: Matrix) -> Tuple[int, int]:
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols

assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)
```

###  Identity Matrix
```python
from typing import Callable

def make_matrix(num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]) -> Matrix:
    return [[entry_fn(i, j)
            for j in range(num_cols)]
            for i in range(num_rows)]

def identity_matrix(n: int) -> Matrix:
    return make_matrix(n, n, lambda i, j: 1 if i==j else 0)

assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 1]]
```

### 행렬이 중요한 이유
1. 각 벡터를 행렬의 행으로 나타내어 여러 백터로 구성된 데이터셋을 행렬로 표현 가능함
2. k차원의 벡터를 n차원의 벡터로 변환해주는 선형함수를 $n \times k$ 행렬로 표현 가능함
3. 이진관계(binary relationship)을 행렬로 나타낼 수 있음

