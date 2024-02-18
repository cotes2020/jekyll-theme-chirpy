---
title : 이진 탐색 [Binary Search]
categories: [Programming, Algorithm]
tags : [Binary Search, 이진 탐색]
---

## 이진 탐색
<hr style="border-top: 1px solid;"><br>

데이터가 정렬된 배열에서 특정한 값을 찾는 알고리즘.

데이터의 중간에 있는 값을 선택하여 특정 값 x와 비교하고
  1. x가 중간 값보다 작으면 중간 값을 기준으로 좌측의 데이터들을 대상으로 탐색
  2. x가 중간값보다 크면 배열의 우측을 대상으로 탐색

탐색할 범위를 찾았으면 동일한 방법으로 다시 중간의 값을 임의로 선택하고 비교. 해당 값을 찾을 때까지 이 과정을 반복.

시간복잡도 : O(log N)

<br>

![image](https://user-images.githubusercontent.com/52172169/165547556-48eb04a6-3daa-4726-9731-caee791a686b.png)

<br><br>
<hr style="border: 2px solid;">
<br><br>

## Code
<hr style="border-top: 1px solid;"><br>

```c
// 반복문을 이용
int BSearch(int arr[], int target) {
    int start = 0;
    int end = arr.length - 1;
    int mid;

    while(start <= end) {
        mid = (start + end) / 2;

        if (arr[mid] == target)
            return mid;
        else if (arr[mid] > target)
            end = mid - 1;
        else
            start = mid + 1;
    }
    return -1;
}
```

<br>

```c
// 재귀문
int BSearchRecursive(int arr[], int target, int start, int end) {
    if (start > end)
        return -1;

    int mid = (start + end) / 2;
    if (arr[mid] == target)
        return mid;
    else if (arr[mid] > target)
        return BSearchRecursive(arr, target, start, mid-1);
    else
        return BSearchRecursive(arr, target, mid+1, end);
}
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 백준 1920번 : 수 찾기
<hr style="border-top: 1px solid;"><br>

Link 
: <a href="https://www.acmicpc.net/problem/1920" target="_blank">www.acmicpc.net/problem/1920</a>

<br>

```python
from sys import stdin
n=int(stdin.readline())
A=stdin.readline().split()
m=int(stdin.readline())
check=stdin.readline().split()
A.sort()

for i in check :
    no=True
    start=0
    end=n-1
    while start <= end :
        mid=(start+end)//2
        if A[mid] == i : 
            print(1)
            no=False
            break
        elif A[mid] > i :
            end=mid-1
        else :
            start=mid+1
    if no : print(0)
```

<br><br>
<hr style="border: 2px solid;">
<br><br>

## 출처
<hr style="border-top: 1px solid;"><br>

Link
: <a href="https://cjh5414.github.io/binary-search/" target="_blank">cjh5414.github.io/binary-search/</a>  
: <a href="https://kosaf04pyh.tistory.com/94" target="_blank">kosaf04pyh.tistory.com/94</a>  

<br><br>
<hr style="border: 2px solid;">
<br><br>
