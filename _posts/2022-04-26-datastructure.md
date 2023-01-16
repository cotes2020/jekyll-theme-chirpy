---
title: "Priority Queues"
excerpt: "2022.04.26 Data structure 수업 정리"

categories:
- DataStructure
tags:
- [DataStructure, PriorityQueue, Queue]
use_math: true

permalink: /categories/Data-Structure/Priority-Queue

toc: true
toc_sticky: true

date: 2022-04-26
last_modified_at: 2022-04-26
---

# Priority Queue
- 별도의 우선순위에 대해 연산을 수행해줌

## min-priority queue

- 이걸 기준으로 수업함

## max-priority queue

## ADT

- `entry`: key와 value pair로 구성되어있음(queue는 entry를 저장)
- Main method
    - `insert`: entry 삽입을 수행하는 연산
    - `removeMin`: 가장 높은 우선순위를 갖는 entry를 삭제(remove smallest key)
- Additional method
    - `min()`: 가장 높은 우선순위를 갖는 entry를 return
    - `empty(), size()`

## Sorting

removeMin을 n번 수행해줌

```cpp
Algorithm PS-Sort(S,C) // C: 비교자
	**Input** sequences S, comparator C for rhe elements of S
	**Output** sequence S sorted in increasing order according to C
	p <- priority queue with comparator C
	while ㄱS.empty() // **phase 1**
		e <- S.front(); S.eraseFront()
		P.insert(e, ø) // e: key, ø: value
	while ㄱ.P.empty() // **phase 2**
		e <- P.removeMin()
		S.insertBack(e)
```

- 참고
    
    4 < 12
    
    “4” > “12” → 우선순위를 앞부터 비교하므로 “4”와 “1”을 비교하면 “4”가 더 큰게 됨
    

## Prority Queue 구현 방법

1. Unsorted Sequence를 이용
    
    ![image](https://user-images.githubusercontent.com/63302432/165286105-25a9c039-b9c8-4d28-b733-211908d90152.png)
    
    - 데이터를 저장할 때 정렬되지 않은 상태로 저장
    - insert: 정렬되지 않은 상태로 저장하기 때문에 맨 뒤 or 맨 앞으로 삽입 → `O(1) time`
    - removeMin: 가장 작은 값을 찾아서 삭제해주는 것 → `O(n) time`
    - min: 가장 작은 값을 return 해주는 것 → `O(n) time`
    
     **Selection Sort(선택 정렬)**
    
    1. n번의 insert: priority queue에 삽입해줘야됨 → `O(n) time`
    2. removeMin → 한 번 할 때마다 size가 1씩 줄어듦
        1. 처음에 할 때 n개에서 removeMin → n
        2. 두번째는 n-1개에서 removeMin
        
        ...
        
        →  $1+2+\dots + n  = O(n^2)\ time$
        
    
    ![image](https://user-images.githubusercontent.com/63302432/165286136-06fe15c6-c605-47f4-ba76-64fba9e46ca1.png)
    
2. Sorted Sequence를 이용
    
    ![image](https://user-images.githubusercontent.com/63302432/165286153-7fe3e521-7392-43bb-8b94-70f89e6a8ca1.png)

    
    - 데이터를 저장할 때 정렬된 상태로 저장
    - insert의 경우 최악의 경우(위의 경우에서 0이 삽입되었을 때 가장 앞까지 와야됨)에 `O(n) time`
    - removeMin, Min → head로 바로 접근 가능 → `O(1) time`
    
    **Insertion Sort(삽입 정렬)**
    
    1. insert를 할 때마다 정렬된 상태를 유지함 
        1. 첫번째 insert → 1
        2. 두번째 insert → 2
        
        ...
        
        → $1+2+\dots \ + n = O(n^2)\ time$
        
    2. removeMin → 정렬되어있는 상태이므로 `O(n) time`
    
    ![image](https://user-images.githubusercontent.com/63302432/165286192-ce9c5dae-9b8f-4f40-8c61-f12a7004109d.png)
    
3. Heap을 이용 → 목요일에..

## In-place Insertion-Sort

앞에서부터 정렬을 시켜줌

![image](https://user-images.githubusercontent.com/63302432/165286216-6be78fa1-9be2-4558-8e2a-909d34147cde.png)

- 파란색: sorted-sequence
- `swaps` 이용하여 바꿔줌
- $O(n^2)\ time$

## 구현방법에 따른 Algorithm 수행시간 비교

|  | 1. unsorted sequence | 2. sorted sequence | 3. heap |
| --- | --- | --- | --- |
| insert(e) | O(1) | O(n) | $O(log_2 n)$ |
| removeMin() or removeMax() | O(n) | O(1) | $O(log_2 n)$ |
| min() or max() | O(n) | O(1) | O(1) |
- 삽입이 많이 필요한 경우: unsorted sequence가 빠름
- min을 많이 필요로하는 경우: 장담 X 
( $\because$ sorted sequence는 insert를 한 후에 remove가 진행되어야 하므로 전체적으로 보면 
       O(n) time이 소요됨 )