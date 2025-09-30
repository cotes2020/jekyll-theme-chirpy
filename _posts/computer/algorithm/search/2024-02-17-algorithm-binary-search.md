---
title: "Binary-Search"
# description: ""
categories: [컴퓨터, 알고리즘]
tags: [알고리듬, Search]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-02-17. 14:44
# last_modified_at: 2024-06-19. 02:25
last_modified_at: 2024-11-16. 05:15 # 정리
---

## Binary-Search

---

이진탐색, 이분탐색.  
**정렬된** 리스트에 대해, 특정 원소를 Log N 안에 찾아내는 알고리듬  

## Use

---

- [Data-Structure-Binary-Search-Tree](/posts/data-structure-binary-search-tree/)

- 전화번호부에서 이름이 K로 시작하는 사람의 번호 찾기
  - 책 한가운데를 펼치고 찾기 시작하는 방법
    - 알파벳 순서대로라면 중간쯤에 K로 시작하는 이름이 있을 테니까
- 사전에서 단어 찾기
- 페이스북에 로그인하기 (페이스북이 내 계정이 실존하는 계정인지 확인하기 위해 데이터베이스에서 아이디 찾기)

## 왜 Why

---

Updown 게임.  
처음부터 끝까지 숫자 하나하나 순서대로 검증하는 것 보다는, 더 좋은 방법.  

| 탐색     | 시간     | 설명                                               |
| -------- | -------- | -------------------------------------------------- |
| 선형탐색 | O(n)     | 리스트의 처음부터 끝까지 순차적으로 탐색           |
| 이진탐색 | O(log n) | 정렬된 리스트에서 탐색 범위를 절반씩 줄여가며 탐색 |

- 리스트에 숫자 N개 있다면
  - 선형탐색: 최악의 경우 N개의 숫자를 모두 확인해야 함.
  - 이진탐색: 최악의 경우 log<sub>N</sub>만 확인하면 됨.

- if N = 1,024
  - 선형탐색: 최악의 경우 1,024
  - 이진탐색: 최악의 경우 log<sub>2</sub>1,024 = 10

## 구현

---

### 찾는 값이 하나

low/high (or start/end).  
찾고자 하는 값이 포함되어 있을 거라고 예상하는 범위.  
(값이 리스트 자체에 없을 수도 있기 때문에 예상 범위)  

```cpp
int binarySearch(int arr[], int target, int size)
{
    // 처음에는 모든 범위를 탐색
    int low = 0;
    int high = size - 1;

    // 서로 위치가 역전되면, 예상 범위가 없다는 것.
    while (low <= high)
    {
        // 절반씩 범위를 줄여 나가는
        // 홀수일 경우에는 내림
        int mid = (low + high) / 2;

        if (arr[mid] == target)
            return mid;
        else if (arr[mid] < target)
            low = mid + 1;
        else
            high = mid - 1;
    }

    return -1;
}
```

### 찾는 값이 여러 개

이분 탐색 후 찾은 위치에서 양 옆으로 확장해도 되지만,  
최악의 경우 O(N)이 될 수 있음. (전부 다 같은 값일 경우)  

대신,  
`가장 마지막에 나오는 값의 인덱스 - 가장 먼저 나오는 값 인덱스 + 1 = 개수`  

다르게 생각하면,  
찾고자 하는 값을 삽입했을 때 오름차순이 유지되는 위치를 찾는 것.  
`오름차순이 유지되는 가장 오른쪽 위치 - 오름차순이 유지되는 가장 왼쪽 위치 = 개수`  

![바킹독님 이미지](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb3XsQb%2Fbtranij1dB5%2FeWIvm0dDzBRao8vTxmm1K1%2Fimg.png)

가장 먼저 나오는 값의 인덱스  
= 삽입했을 때 오름차순이 유지되는 가장 왼쪽 위치.  

```cpp
int lowerBound(int arr[], int target, int size)
{
    int low = 0;
    int high = size;

    while (low < high)
    {
        int mid = (low + high) / 2;

        if (arr[mid] >= target)
            high = mid;
        else
            low = mid + 1;
    }

    return low;
}
```

![바킹독님 이미지](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FckG40B%2FbtranhemVXf%2FGJcaPNmKFdqr8Ky9F2N3Tk%2Fimg.png)

가장 마지막에 나오는 값의 인덱스  
= 삽입했을 때 오름차순이 유지되는 가장 오른쪽 위치.  

```cpp
int upperBound(int arr[], int target, int size)
{
    int low = 0;
    int high = size;

    while (low < high)
    {
        int mid = (low + high) / 2;

        if (arr[mid] > target)
            high = mid;
        else
            low = mid + 1;
    }

    return low;
}
```

## 메모

---

- C++ STL
  - `binary_search` 함수
    - 리스트가 반드시 오름차순으로 정렬되어 있어야 함.
    - 요소 유무를 true/false로 반환
  - `lower_bound` 함수
    - 찾고자 하는 값 이상이 처음 나타나는 위치
  - `upper_bound` 함수
    - 찾고자 하는 값 초과가 처음 나타나는 위치
  - `equal_range` 함수
    - `lower_bound`와 `upper_bound`를 한 번에 호출
    - `pair`로 반환

- Parametric Search 처럼 STL가 도움이 안되고 직접 이분탐색을 해야하는 경우가 있다.
  - 무한 루프에 빠지는 경우가 있기 때문에 주의

- 무한 루프에 빠지지 않으려면
  - 오버플로우 방지: `int mid = low + (high - low) / 2;`
  - 균등하게 나누기: `mid = (high - low + 1) / 2;`

- [좌표 압축 ~](https://blog.encrypted.gg/985)
