---
title: "Queue"
# description: ""
categories: [컴퓨터, 자료구조]
tags: [자료구조, Queue]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-02-19. 16:45
last_modified_at: 2024-08-29. 22:07
---

## @TODO

---

한쪽 끝에서 원소를 넣고, 반대쪽 끝에서 원소를 뺄 수 있는 구조  
FIFO First in First out  

`Restricted Structure`  
특정 위치에서만 원소를 넣거나 뺄 수 있는 제한된 자료구조 (스택, 큐, 덱)  

대기 행렬  

- 성질
  - 원소 추가 O(1)
  - 원소 제거 O(1)
  - 제일 앞/뒤 원소 확인 O(1)
  - 제일 앞/뒤가 아닌 나머지 원소들의 확인/변경이 '원칙적으로' 불가능

추가되는 쪽을 rear 뒤쪽  
제거되는 쪽을 front 앞쪽  

마찬가지로 연결리스트보다 배열 구현이 더 쉬움

원형 큐  

```cpp
const int MAX = 1000005;
int dat[MAX];
int head = 0, tail = 0;

void push(int x)
{
    dat[tail++] = x;
}

void pop()
{
    head++;
}

int front()
{
    return dat[head];
}

int back()
{
    return dat[tail-1];
}

int size()
{
    return tail - head;
}

```

```cpp
#include <bits/stdc++.h>
using namespace std;

int main(void)
{
    queue<int> Q;
    Q.push(10); // 10
    Q.push(20); // 10 20
    Q.push(30); // 10 20 30
    cout << Q.size() << '\n'; // 3
    if(Q.empty()) cout << "Q is empty\n";
    else cout << "Q is not empty\n"; // Q is not empty
    Q.pop(); // 20 30
    cout << Q.front() << '\n'; // 20
    cout << Q.back() << '\n'; // 30
    Q.push(40); // 20 30 40
    Q.pop(); // 30 40
    cout << Q.front() << '\n'; // 30
}
```

BFS  
Flood Fill  

둘 다 단골 문제  

### _
