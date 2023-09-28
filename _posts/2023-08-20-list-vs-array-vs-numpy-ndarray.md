---
layout: post
title: List vs Array vs Numpy ndarray
date: 2023-08-20 16:19 +0900
category: [Language]
tag: [Python]
---

### 구현 원리

list
: 항목들이 메모리 공간에 흩어져있고 각 항목의 레퍼런스를 저장하는 방식으로 이루어져 있다. 이 방식은 cache의 locality를 떨어뜨릴 수 있다.

array
: 모든 항목들이 메모리 공간에 연속적으로 붙어있는 형태이다. cache locality를 보장하지만 각 항목의 자료형이 primitive 해야한다.

numpy ndarray
: array와 비슷하게 연속된 메모리 공간에 항목을 저장하지만, 모든 자료형을 저장할 수 있는 dtype='O'인 경우 list처럼 레퍼런스를 저장하는 방식이 된다.

### 실험

```python

import timeit
import array
import numpy as np

# Parameters for testing
N = 10000       # number of rows and columns in the array
M = 1000   # number of iterations

# Initialize the arrays
my_list = [i for i in range(N)]
my_array = array.array('i', my_list)
my_ndarray = np.array(my_list)

def test_list_sum():
    sum(my_list)

def test_list_npsum():
    np.sum(my_list)

def test_array_sum():
    sum(my_array)

def test_array_npsum():
    np.sum(my_array)

def test_numpy_sum():
    sum(my_ndarray)

def test_numpy_npsum():
    np.sum(my_ndarray)

# Perform the benchmarks
list_sum_time = timeit.timeit(test_list_sum, number=M)
list_npsum_time = timeit.timeit(test_list_npsum, number=M)
array_sum_time = timeit.timeit(test_array_sum, number=M)
array_npsum_time = timeit.timeit(test_array_npsum, number=M)
ndarray_sum_time = timeit.timeit(test_numpy_sum, number=M)
ndarray_npsum_time = timeit.timeit(test_numpy_npsum, number=M)

# Output the results
print(f'list (sum): {list_sum_time * 1000:.1f} ms')
print(f'list (np.sum): {list_npsum_time * 1000:.1f} ms')
print(f'array (sum): {array_sum_time * 1000:.1f} ms')
print(f'array (np.sum): {array_npsum_time * 1000:.1f} ms')
print(f'ndarray (sum): {ndarray_sum_time * 1000:.1f} ms')
print(f'ndarray (np.sum): {ndarray_npsum_time * 1000:.1f} ms')

```

수행시간(ms)|list    |array     |ndarray
-----------|-------:|---------:|--------:
sum        |96.3    |186.3     |586.6
np.sum     |527.9   |14.7      |6.2

list인 경우 sum이 빠르고 array, ndarray인 경우 np.sum이 빠르다.

`sum`은 항목들을 하나씩 <kbd>참조하기</kbd> > <kbd>더하기</kbd> 과정을 수행한다. 이 때 사용되는 덧셈 연산은 두 파이썬 객체를 더하는 연산이다. `array`와 `ndarray`는 list와 달리 모든 항목을 파이썬 객체로 만들어 보관하지 않기 때문에 모든 항목에 대해 <kbd>참조하기</kbd> > <kbd>파이썬 객체로 변환</kbd> > <kbd>더하기</kbd>과정을 수행하게 되기 때문에 발생하는 오버헤드로 시간이 오래 걸린다.

ndarray의 성능을 최대한 발휘하기 위해서는 numpy에서 제공해주는 함수만 사용하거나 vectorize된 연산자를 적용해야 한다.

### Ref.

<https://hyperconnect.github.io/2023/05/30/Python-Performance-Tips.html>