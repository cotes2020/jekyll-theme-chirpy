---
title: "Array 배열"
# description: ""
categories: [컴퓨터, 자료구조]
tags: [자료구조, Array]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-02-18. 11:34
# last_modified_at: 2024-02-19. 00:30
# last_modified_at: 2024-02-20. 00:04
# last_modified_at: 2024-02-21. 18:19
# last_modified_at: 2024-02-21. 19:32
last_modified_at: 2024-08-29. 22:00
---

## 정의

---

메모리 상에 원소를 연속하게 배치한 자료구조

## 성질

---

### 메모리에 연속하게 배치

`임의접근 Random-Access`  

특정 위치를 알고 싶다면? 그냥 시작 위치에서 데이터 크기 * 위치를 더해주면 바로 알 수 있음  
메모리 상에 '연속하게' 위치해 있으니까  

Like 영화관에서 지인들과 나란히 좌석을 예매할 때  
다른 지인이 더 온다면, 나란히 않기 위해서 연속된 자리를 찾아 이동해야 함  
더 많은 지인이 올 수록, 예매는 비용도 커지고 이동도 힘들어짐  
실패할 수도 있음! 연속된 자리가 없다면 ! (메모리가 부족하면 실패할 수 있는 건 당연하지만, 여기서 말하고자 하는건 '연속된' 메모리 공간이 없다면!)  

배열에 새 원소를 추가하는 일은 쉽지 않은 일  

미리 충분한 좌석을 예매한다면  
추가할 일이 생기지 않는 다면 메모리 낭비 (배열에서 쓰지도 않지만, 다른 쪽에서도 못씀)  
만약에 지인이 미리 예매해둔 양보다 많아진다면 어차피 다시 예매해야 함  

좋은 해결책이지만, 완벽하지는 않음  
-> 연결 리스트  

- O(1)에 k번째 원소를 확인/변경 가능
- 추가적으로 소모되는 메모리의 양(=overhead)가 거의 없음
- Cache hit rate가 높음
- 메모리 상에 연속한 구간을 잡아야 해서 할당에 제약이 걸림

- 자료구조로써의 배열, 길이를 마음대로 늘리거나 줄이는게 가능하다고 생각

### 시간복잡도

- 임의의 위치에 있는 원소를 확인/변경, O(1)
- 원소를 끝에 추가, O(1)
- 마지막 원소를 제거, O(1)
- 임의의 위치에 원소를 추가/제거, O(N)

## 사용

---

### C++ Array

컴파일 시간에 크기가 결정되고 더 이상 크기의 변경이 불가능한 정적 배열  

```cpp
// Declaration & Definition & Init
int a[21]; // [21]: Index 색인/Subscript 첨자
int b[21][21];
int c[21] = { 1, 2, 3 }; // { 1, 2, 3, 0, 0 }
int d[] = { 1, 2, 3 };
int d[] { 1, 2, 3 }; // Universal Initialization

// 1. memset
#include <cstring>
memset(a, 0, sizeof a);
memset(b, 0, sizeof b);

// 2. for
for (int i = 0; i < 21; i++)
    a[i] = 0;
for (int i = 0; i < 21; i++)
    for (int j = 0; j < 21; j++)
        b[i][j] = 0;

// 3. fill
#include <algorithm>
fill(a, a + 21, 0);
for (int i = 0; i < 21; i++)
    fill(b[i], b[i] + 21, 0);
```

### C++ STL vector

실행시간에 크기를 변경할 수 있는 동적 배열  

메모리 연속하게 저장  
크기를 자유자재로 늘리거나 줄이거나  

그래프의 인접 리스트라는 것을 저장할 때에는 vector를 쓰는게 많이 편해서 vector가 필요하게 되지만,  
그 전까지는 사실 굳이 배열대신 vector를 써야하는 상황은 없다.  

생성과 소멸을 하는데 상당한 시간이 소요된다. (성능)  

```cpp
// Declaration & Definition & Init
#include <vector>

vector<int> v1;
vector<int> v1 = { 0, 0, 0 };
vector<int> v1 { 0, 0, 0 };
vector<int> v1(1); // { 0 };
vector<int> v1(size); // 변수, 동적 생성 시
vector<int> v1(3, 4); // { 4, 4, 4 };
vector<int> v1.assign(3, 4); // { 4, 4, 4 };
```

```cpp
// Use

v1.size();
v1.push_back(1);
v1.pop_back();
v1.clear();

// iterator
v1.begin(); // 시작 주소
v1.end(); // 끝 주소 + 1
v1.rbegin(); // (끝 주소)를 시작으로
v1.rend(); // (시작 주소 + 1) 을 끝으로

v1.insert(v1.begin() + 1, 1); // { 0, 3, 0 };
v1.erase(v1.begin() + 1); // { 1, 2, 4 };

// Deep Copy (이것은 연산자 중복 정의)
v1 = v2;

// 요소의 개수와 값이 모두 일치하면 true (이것 역시 연산자 중복 정의)
(v1 == v2)
```

```cpp
// Traversal

// 1. range-based for loop (since C++11)
for (int e: v1)
for (int& e: v1)
// & <- list, map, set 등에서도 모두 사용할 수 있는

// 2. for (not bad)
for (int i = 0; i < v1.size(); i++)

// 2-1. ***WRONG***
// size()가 unsigned int이기 때문에,
// 빈 vector의 size() 0에서 -1을 해버리면
// (unsigned int)0 - (int)1 = 4294967295
for (int i = 0; i <= v1.size() - 1; i++)
```

### C++ STL array

일부 함수를 제공하는 정적 배열  

```cpp
// Declaration & Definition & Init
#include <array>

array<int, 3> a { 1, 2, 3 };
```

```cpp
// Use
a.size();
a.fill(1);
a.empty();
a.at();
a.front();
a.back();
```

## 메모

---

- @ TODO
  - [C++ vector](https://hwan-shell.tistory.com/119)
