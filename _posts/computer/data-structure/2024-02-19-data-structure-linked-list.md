---
title: "Linked-List"
# description: ""
categories: [컴퓨터, 자료구조]
tags: [자료구조, Linked-List]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-02-19. 00:30
# last_modified_at: 2024-02-19. 02:30
# last_modified_at: 2024-02-19. 19:56
# last_modified_at: 2024-02-25. 02:11
last_modified_at: 2024-08-29. 22:07
---

## @TODO

---

Linked-List 연결 리스트  
(혹은 List 리스트)  

원소를 저장할 때 그 다음 원소가 있는 위치를 포함시키는 방식으로 저장하는 자료구조  
원소들은 이곳 저곳에 흩어져있다.  

흩어져있기에 원하는 원소가 어디에 있는지 알 수 없음  
이를테면 마지막 원소가 뭔지 바로 알 수 없음  
모든 원소를 한 번 씩 읽어야 한다면 연결 리스트가 좋지만,  
특정한 원소 하나만 알고 싶다면 연결 리스트는 최악  

순차 접근 Sequential Access  

극장에서 서로 흩어져서 영화보기  
서로 어디있는지는 앎  

### _

- 성질
  - k번째 원소를 확인/변경하기 위해 O(k)가 필요함
  - 임의의 위치에 원소를 추가/임의 위치의 원소 제거는 O(1)
    - 추가, 제거할 위치를 알고 있다는 (이미 접근해 있다는) 것을 전제로
    - 아니라면 read하는 만큼의 시간 복잡도가 더 들겠죠 (+O(k))
  - 원소들이 메모리 상에 연속해있지 않아 Cache hit rate가 낮지만 할당이 다소 쉬움  

- 종류
  - 단일 연결 리스트 Singly Linked List
  - 이중 연결 리스트 Doubly Linked List (STL list)
  - 원형 연결 리스트 Circular Linked List
    - 환형 "
    - 순환 "

첫 번째 요소, 두 번째 요소를 식별할 수 있음  
배열과 마찬가지로 선형 자료구조  
트리, 그래프, 해쉬 등은 비선형 자료구조  

| | 배열 | 연결 리스트 |
| k번째 원소에 접근 | O(1) | O(k) |
| 임의 위치에 원소 추가/제거 | O(N) | O(1) (추가/제거할 위치를 알고 있는 경우) |
| 메모리 상의 배치 | 연속 | 불연속 |
| 추가적으로 필요한 공간 (Overhead) | - (없음, 굳이 만든다면 배열 길이) | O(N) (주소값, 32bit 4N바이트, 64bit 8N바이트, N에 비례하는 만큼의 메모리) |

```cpp
// 야매 연결 리스트
// 메모리 누수 문제로 실무에서는 절대 쓸 수 없는 방식이지만,
// 코테에서는 구현 난이도가 일반적인 구현보다 낮은데 시간복잡도는 동일해서 애용할 수 있다

const int MX = 1000005;
int dat[MX], pre[MX], nxt[MX];
int unused = 1; // top?

// 배열 순서대로 저장하는 것이 아님, pre nxt에 원소 index를 저장

// 단, 0번지는 고정적인 단순 시작지로써, 데이터를 저장하지 않음, 맨 처음 원소 앞에 0번지 원소가 있다고 생각
// 이런 Dummy node를 두지 않으면, 삽입삭제 등을 구현할 때 원소가 아예 없는 경우에 대한 예외처리를 해야 하는 번거로움이

// @ stl list의 경우 dummy node는 맨 마지막에 있음

fill(pre, pre + MX, -1);
fill(nxt, nxt + MX, -1);

// traverse
// 0번지에서 출발해 nxt를 따라가는
void traverse()
{
    int cur = nxt[0];
    while(cur != -1)
    {
        cout << dat[cur] << ' ';
        cur = nxt[cur];
    }
    cout << "\n\n";
}

// insert (int addr추가할위치주소, int num)
// 새로운 원소 생성
// 새로운 원소 pre, nxt 대입
// 삽입할 위치의 nxt 값과 삽입할 위치의 다음 원소의 pre 값을 새 원소로 변경
// unused 1 증가
void insert(int addr, int num)
{
    dat[unused] = num;
    pre[unused] = addr;
    nxt[unused] = nxt[addr];
    if (nxt[addr] != -1) pre[nxt[addr]] = unused;
    nxt[addr] = unused;
    unused++;
}

// erase
// 이전 위치의 nxt를 삭제할 위치의 nxt로 변경
// 다음 위치의 pre를 삭제할 위치의 pre로 변경

// 야매 방법으로는 제거를 해도 프로그램이 종료될 때 까지 메모리를 점유하고 있음
// 그래서 실무에서는 못씀
void erase(int addr)
{
    // dummy node의 존재로 인해 그 어떤 노드를 지우더라도 pre[addr]은 -1이 아님이 보장됨
    nxt[pre[addr]] = nxt[addr];
    if (nxt[addr] != -1) pre[nxt[addr]] = pre[addr];

    // 남은 데이터를 일부러 지울 필요는 없음
    // 어차피 접근할 일도 없고, 새로 값이 들어올 때 덮어씌워질 것이기 때문
}
```

```cpp
// STL list
#include <bits/stdc++.h>
using namespace std;

// push_back, pop_back, push_front, pop_front는 모두 O(1)
int main(void)
{
    list<int> L = { 1, 2 }; // 1 2
    list<int>::iterator t = L.begin(); // t는 1을 가리키는 중
    // c++ 11 이상 일때 auto t = L.begin() 가능
    L.push_front(10); // 10 1 2
    cout << *t << '\n'; // t가 가리키는 값 = 1을 출력
    L.push_back(5); // 10 1 2 5
    L.insert(t, 6); // t가 가리키는 곳 앞에 6을 삽입, 10 6 1 2 5
    t++; // t를 1칸 앞으로 전진, 현재 t가 가리키는 값은 2
    t = L.erase(t); // t가 가리키는 값을 제거, 그 다음 원소인 5의 위치를 반환
                    // 10 6 1 5, t가 가리키는 값은 5
    cout << *t << '\n'; // 5
    for(auto i: L) cout << i << ' '; // C++ 11 이상
    cout << '\n';
    for(list<int>::iterator it = L.begin(); it != L.end(); it++)
    cout << *it << ' ';
}// remove unique merge reverse sort splice
// []를 지원하지 않음, 따라서 임의 접근 반복자를 필요로 하는 binary_search 알고리듬 적용할 수 없다
```

중간에 만나는 두 연결 리스트의 시작점들이 주어졌을 때 만나는 지점을 구하는 방법?  

둘 다 끝까지 진행시킨 후에 그 길이를 기록  
다시 시작점들로 돌아와서 짧은 쪽을 길이 차이만큼 먼저 진행  
이후 동시에 하나씩 전진해나가며 만나는 지점 찾기  

공간 복잡도 O(1), 시간 복잡도 O(A + B)  

---

주어진 연결 리스트 안에 사이클이 있는지 판단하라  

Floyd's cycle-finding algorithm, 공간 복잡도 O(1), 시간 복잡도 O(N)  

---

- 운영 체제에서 동적 메모리를 관리할 때
- 희소 행렬 Sparse Matrix을 표현할때
- 덱이나 스택, 큐와 같은 다른 자료구조를 표현할 때 기초 자료 구조로
- 텍스트 에디터도 내부적으로 컨텐츠를 저장할 때 연결리스트를 사용. 텍스트 파일의 중간에서 문자가 입력될 수 있기 때문
