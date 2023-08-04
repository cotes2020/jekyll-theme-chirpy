---
title : 퀵 정렬 [Quick Sort]
categories: [Programming, Algorithm]
tags : [Quick Sort]
---

## 참고
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://freedeveloper.tistory.com/377" target="_blank">freedeveloper.tistory.com/377</a>

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Quick Sort
<hr style="border-top: 1px solid;"><br>

퀵 정렬에서는 pivot이라는 것을 정렬의 기준으로 사용함.

pivot은 데이터 배열에서 아무 값을 기준으로 삼는데 보통 처음, 중간, 끝에 위치한 값을 선정함.

pivot을 기준으로 작거나 같은 값은 왼쪽으로 넘김. **그러면 왼쪽에는 pivot보다 작거나 같은 값, 오른쪽은 큰 값이 있게 됨.**

**중요한 점은 이런 식으로 했을 때, pivot의 위치가 정해진 다는 것임.**

<br>

시간복잡도는 보통 ```O(nlogn)```이지만 단점은 최악의 경우(정렬 또는 역정렬된 배열) ```O(n^2)```

**그에 반해 병합 정렬은 정확히 반절로 나눈다는 점에서 최악의 경우에도 시간복잡도는 ```O(nlogn)```임.**

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Code
<hr style="border-top: 1px solid;"><br>

일반적인 방식

```python
array = [5, 7, 9, 0, 3, 1, 6, 2, 4, 8]

def quick_sort(array, start, end):
    if start >= end: # 원소가 1개인 경우 종료
        return
        
    pivot = start # 피벗은 첫 번째 원소
    left = start + 1
    right = end
    
    while(left <= right):
        # 피벗보다 큰 데이터를 찾을 때까지 반복 
        while(left <= end and array[left] <= array[pivot]):
            left += 1
            
        # 피벗보다 작은 데이터를 찾을 때까지 반복
        while(right > start and array[right] >= array[pivot]):
            right -= 1
            
        if(left > right): # 엇갈렸다면 작은 데이터와 피벗을 교체
            array[right], array[pivot] = array[pivot], array[right]
            
        else: # 엇갈리지 않았다면 작은 데이터와 큰 데이터를 교체
            array[left], array[right] = array[right], array[left]
            
    # 분할 이후 왼쪽 부분과 오른쪽 부분에서 각각 정렬 수행
    quick_sort(array, start, right - 1)
    quick_sort(array, right + 1, end)

quick_sort(array, 0, len(array) - 1)
print(array)
```

<br>

파이썬의 장점을 이용한 방식 

```python
array = [5, 7, 9, 0, 3, 1, 6, 2, 4, 8]

def quick_sort(array):
    # 리스트가 하나 이하의 원소만을 담고 있다면 종료
    if len(array) <= 1:
        return array

    pivot = array[0] # 피벗은 첫 번째 원소
    tail = array[1:] # 피벗을 제외한 리스트

    left_side = [x for x in tail if x <= pivot] # 분할된 왼쪽 부분
    right_side = [x for x in tail if x > pivot] # 분할된 오른쪽 부분

    # 분할 이후 왼쪽 부분과 오른쪽 부분에서 각각 정렬을 수행하고, 전체 리스트를 반환
    return quick_sort(left_side) + [pivot] + quick_sort(right_side)

print(quick_sort(array))
```

<br><br>
<hr style="border: 2px solid;">
<br><br>
