---
title: "Deque"
# description: ""
categories: [컴퓨터, 자료구조]
tags: [자료구조, Deck]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-02-19. 16:54
# last_modified_at: 2024-02-25. 02:11
last_modified_at: 2024-08-29. 22:00
---

## @TODO

---

Double Ended Queue  

양쪽 끝에서 삽입 삭제가 전부 가능  

`Restricted Structure`  
특정 위치에서만 원소를 넣거나 뺄 수 있는 제한된 자료구조 (스택, 큐, 덱)  

- 성질
  - 원소 추가 O(1)
  - 원소 제거 O(1)
  - 제일 앞/뒤 원소 확인 O(1)
  - 제일 앞/뒤가 아닌 나머지 원소들의 확인/변경이 '원칙적으로' 불가능
    - 독특하게도 STL deque에서는 인덱스로 원소에 접근할 수 있음

마찬가지로 연결리스트보다 배열 구현이 더 쉬움

```cpp
const int MAX = 1000005;
int dat[2 * MAX + 1];
int head = MAX, tail = MAX;

void push_front(int x)
{
    dat[--head] = x;
}

void push_back(int x)
{
    dat[tail++] = x;
}

void pop_front()
{
    head++;
}

void pop_back()
{
    tail--;
}

int front()
{
    return dat[head];
}

int back()
{
    return dat[tail-1];
}

```

```cpp
#include <bits/stdc++.h>
using namespace std;

int main(void)
{
    // STL deque는 deque보다도 vector랑 비슷한데
    // vector가 front에서도 O(1)로 추가 제거가 가능한 느낌

    // insert erase 인덱스접근
    // stl vector에서 제공되는 기능을 stl deque에서도 다 제공

    // 단, vector와 달리 deque는 모든 원소들이 메모리상에 연속하게 배치되어 있지 않음
    // 궁금하다면 c++ deque vs vector

    deque<int> DQ;
    DQ.push_front(10); // 10
    DQ.push_back(50); // 10 50
    DQ.push_front(24); // 24 10 50
    for(auto x: DQ) cout << x << ' ';
    cout << DQ.size() << '\n'; // 3
    if(DQ.empty()) cout << "DQ is empty\n";
    else cout << "DQ is not empty\n"; // DQ is not empty
    DQ.pop_front(); // 10 50
    DQ.pop_back(); // 10
    cout << DQ.back() << '\n'; // 10
    DQ.push_back(72); // 10 72
    cout << DQ.front() << '\n'; // 10
    DQ.push_back(12); // 10 72 12
    DQ[2] = 17; // 10 72 17
    DQ.insert(DQ.begin()+1, 33); // 10 33 72 17
    DQ.insert(DQ.begin()+4, 60); // 10 33 72 17 60
    for(auto x: DQ) cout << x << ' ';
    cout << '\n';
    DQ.erase(DQ.begin()+3); // 10 33 72 60
    cout << DQ[3] << '\n'; // 60
    DQ.clear(); // DQ의 모든 원소 제거
    }
```

BFS  
Flood Fill  

둘 다 단골 문제  

- 운영체제 작업 스케줄링 알고리듬
- 앱의 undo 리스트

### _
