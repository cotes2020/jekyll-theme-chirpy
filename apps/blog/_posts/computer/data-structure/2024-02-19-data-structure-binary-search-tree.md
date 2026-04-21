---
title: "Binary (Search) Tree"
# description: ""
categories: [컴퓨터, 자료구조]
tags: [자료구조, Tree]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-02-19. 20:52
# last_modified_at: 2024-07-15. 06:12
# last_modified_at: 2024-08-01. 10:48
# last_modified_at: 2024-08-29. 21:35
# last_modified_at: 2024-09-02. 11:25
last_modified_at: 2024-10-04. 17:37
---

{% include embed/youtube.html id='IKnjzmyk70U' %}
{% include embed/youtube.html id='nehRy6hAJsA' %}

## 이진 트리 (Binary-Tree)

---

각 노드의 자식이 2개 이하인 트리  
자식이 2개 이하이기 때문에 자식을 왼쪽과 오른쪽으로 구분할 수 있다.  

### 이진 트리 구현

![이진 트리 구현](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb1mDuH%2FbtrodS0jcTD%2FXNIcWM2daVsr1bMA8b7FcK%2Fimg.png)  

```cpp
int lc[] = {};
int rc[] = {};
int parent[] = {};
```

## 이진 트리의 순회

---

- [BFS](/posts/algorithm-bfs/#트리에서의-bfs)
- [DFS](/posts/algorithm-dfs/#트리에서의-dfs)
- 말고도 특별히 이진 트리에 대해 레벨/전위/중위/후위 순회가 있다.

### 레벨 순회 (Level-order Traversal)

![레벨 순회](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdBBT4a%2FbtrnJdwJJYO%2Fdg9ZQa2aevyGennKFkq0L1%2Fimg.png)  

레벨, 즉 높이 순으로 방문하는 순회  

루트에서 BFS를 돌리면 자연스럽게 레벨 순회가 됨  
이때 lc, rc로 이진 트리를 표현하고 있는 상황이기에 이에 맞는 BFS 코드가 필요.  

### 전위 순회 (Preorder Traversal)

![전위 순회](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FHR7q8%2FbtrnOQgIjOR%2Fukzq4wrCVKGTHv5ikKI2F1%2Fimg.png)  

현재 -> 왼쪽 -> 오른쪽  

1. 현재 정점을 방문한다.
2. 왼쪽 서브 트리를 전위 순회한다.
3. 오른쪽 서브 트리를 전위 순회한다.

→ DFS와 방문 순서가 동일, DFS는 자기 자신을 방문한 후 첫 번째 자식부터 들어가 거기에서 DFS를 다시 시작하는 방식  
재귀를 이용해 자기 자신을 출력한 후 왼쪽 자식과 오른쪽 자식 각각에 대해 전위 순회  

### 중위 순회 (Inorder Traversal)

![중위 순회](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FkNSNN%2FbtrnEJbSo6H%2FrxhXktyJhEW41uKHKlRES1%2Fimg.png)  

왼쪽 -> 현재 -> 오른쪽  

1. 왼쪽 서브 트리를 중위 순회한다.
2. 현재 정점을 방문한다.
3. 오른쪽 서브 트리를 중위 순회한다.

### 후위 순회 (Postorder Traversal)

![후위 순회](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FpKHG0%2FbtrnPLffDRi%2FdAjvuXYnm2NLKzFGGEySVk%2Fimg.png)  

왼쪽 -> 오른쪽 -> 현재  

1. 왼쪽 서브 트리를 후위 순회한다.
2. 오른쪽 서브 트리를 후위 순회한다.
3. 현재 정점을 방문한다.

### 이진 트리의 순회 메모

![서로 다른 트리라고 하더라도 순회 결과가 일치할 수 있다](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FesAGSl%2FbtrnN1iz6v7%2FiSMoZtqH69KlwqJDSW86Kk%2Fimg.png)

서로 다른 트리라고 하더라도 순회 결과가 일치할 수 있다.  

사진 속 두 트리는 중위 순회 결과가 2 1 3 으로 동일  
다른 순회에 대해서도 쉽게 이러한 예시들을 찾을 수 있다.  

2개의 순회 결과가 주어졌을 때에는 그러한 트리가 유일할까요?  
[만약 중위 순회(Inorder)와 다른 순회가 주어진다면 유일하지만 중위 순회가 포함되어있지 않다면 유일하지 않는다.](https://www.geeksforgeeks.org/if-you-are-given-two-traversal-sequences-can-you-construct-the-binary-tree/)  

## 이진 검색 트리 (Binary-Search-Tree)

---

자가 균형 트리가 아니라면 이진 검색 트리는 시간복잡도가 안 좋아서 써먹을 수 없다.  
그런데, 자가 균형 트리는 구현이 어렵다.  
→ 직접 구현하기 보다는, STL을 이용하자.  

만약 STL을 쓰지 못하는 상황이라면?  
어떻게든 해시나 이분 탐색 같은, 이진 검색 트리를 쓰지 않는 다른 풀이를 찾아야 한다.  
구현 자체도 실수할 여지가 많고, 자가 균형 트리가 아니라면 시간복잡도가 안좋아서 높은 확률로 시간 초과가 발생한다.  

Binary-Search-Tree | 이진 탐색 트리  

왼쪽 서브트리의 모든 값은 부모의 값보다 작고,  
오른쪽 서브트리의 모든 값은 부모의 값보다 큰,  
이진 트리  

이진 탐색 개념을 그래프의 트리 구조 사용하여 표현  
마찬가지로 각 노드는 최대 두 개의 자식 노드를 가짐  

### 왜 이진 검색 트리를 쓰냐?

탐색, 삽입, 삭제 연산을 O(logN)에 할 수 있기 때문  

배열은 insert는 제일 뒤에 붙이면 되니 O(1)이라고 해도,  
erase는 배열 중간에 있는 원소가 제거될 상황이 나올 수 있으니 O(N), find, update도 O(N)이다.  

그런데 이진 탐색 트리는 O(logN)이다.  
때문에 erase, find, update가 많은 상황에서는 이진 탐색 트리를 쓰는 것이 좋다.  

해시는 비록 충돌 때문에 성능이 안 좋아질 수 있지만, 저 4개의 연산이 O(1)이긴한데,  
해시에는 없는 이진 검색 트리의 강력한 특징은 원소가 크기 순으로 정렬되어 있다는 것이다.  

1, 3, 5, 7, 9 중애서 5보다 큰 최초의 원소가 무엇인지 찾으라고 하면  
해시는 O(N), 그러나 이진 검색 트리는 O(logN)이다.  

때문에 insert, erase, find, update 등이 빈번하면서, 동시에 뭔가 원소의 대소와 관련한 성질이 필요한 경우에는 이진 검색 트리를 사용해야 한다.  

### 성질

- 모든 노드는 왼쪽 가지에 포함되는 어떤 숫자보다 큰 숫자
- 모든 노드는 오른쪽 가지에 포함되는 어떤 숫자보다 작은 숫자
- -> 따라서 최상단 노드로부터 왼쪽 가지만 쭉 따라가면 최소노드 (최솟값)이 나옴
- -> 따라서 최상단 노드로부터 오른쪽 가지만 쭉 따라가면 최대노드 (최댓값)이 나옴

### Binary-Search-Tree 구현

#### 추가/삽입 Insert

1. 루트부터
2. 추가하려는 노드가 현재 노드 값과 비교해서 작으면 왼쪽, 크면 오른쪽으로 진행
3. 더 이상 비교할 수 없으면 정착 (비어있는 곳에 새로 추가해야 하는 상황)

#### 탐색 Find/Search

1. 루트부터
2. 찾으려는 숫자가 현재 노드와 비교해서 작으면 왼쪽으로, 크면 오른쪽으로 진행

#### 삭제 Erase

1. 자식 노드가 없으면 대상 노드만 삭제하면 끝 (리프 노드)
2. 자식 노드가 하나면 자식 노드를 기존 노드 위치로 이동
3. 자식 노드가 두 개 이상이면
   - 삭제한 노드의 왼쪽 가지에서 최대 (기존 노드 값보다 작은 것들 중 가장 큰) 노드를 찾아 기존 노드 위치로 이동
   - 혹은 삭제한 노드의 오른쪽 가지에서 최소 (기존 노드 값보다 큰 것들 중 가장 작은) 노드를 찾아 기존 노드 위치로 이동
   - 이동한 자식 노드가 자식 노드를 가지고 있으면, 해당 노드에 대해 3번 재귀적 반복

### 문제점

![문제점](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FAsFh1%2Fbtrj3Wq2wSl%2Fx0uT0lByKesX8MFjcLWmA0%2Fimg.png)

트리의 삽입, 검색, 삭제는 모두 트리의 높이가 얼마인지에 따라서 시간 복잡도가 정해진다.  

만약 왼쪽 트리처럼 각 정점이 대부분 2개의 자식을 가지고 있다면, 높이가 하나 내려갈 때 마다 자식의 수가 1, 2, 4, 8, ... 이렇게 2배씩 증가하기 때문에 정점이 N개 있다고 하면 높이가 대략 log<sub>2</sub>N이 된다.  
이 경우 삽입, 검색, 삭제 모두 O(logN)이 된다.  

반면 오른쪽 트리처럼 트리가 편향되어 있다면 높이가 O(N)에 가깝기 때문에,  
이 경우 삽입, 검색, 삭제 모두 O(N)이 된다.  

각 연산을 O(logN)로 쓰려고 이진 검색 트리를 쓰는 건데, 만역 트리가 오른쪽 트리처럼 편향되어 있다면 이진 검색 트리를 쓰는 의미가 없어진다.  
그리고 1, 2, 3, 4, ... 이렇게 크기 순으로 주어진 원소를 삽입한다면 1이 루트이고 나머지 원소들이 오른쪽에 일직서느올 연결되는 편향된 트리가 되니, 편향된 트리가 만들어지는 상황은 아주 잘 발생할 수 있다.  

## Self-Balancing Tree

---

대표적으로 AVL-Tree, Red-Black-Tree 등이 있다.  
구현은 AVL이 상대적으로 쉬운 편인데, 성능 자체는 Red-Black이 더 좋아서 STL에서는 Red-Black을 쓴다.  

![간단하게 설명하면](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F8HccL%2Fbtrj6deGDl3%2FtKUQbYJA9KYoKW1fKK0zO0%2Fimg.png)

간단하게 설명하면 불균형이 발생했을 때 트리를 꺾어버린다.  
이러면 불균형을 없앨 수 있다.  

이렇게 편향성을 해소해주는 자가 균형 트리를 사용할 때 비로소 이진 검색 트리에서 삽입, 검색, 삭제가 모두 O(lg N)이 됩니다.  

## STL

---

- 해쉬: `unordered_set`, `unordered_multiset`, `unordered_map`
- 이진 검색 트리: `set`, `multiset`, `map`
  - 내부적으로 원소가 크기 순으로 저장

- 공통 | 런타임 에러를 유발하는 주요 원인
  - `iterator`가 `end()`를 가리키고 있을 때 값 참조

### set

![set](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbLouBY%2Fbtrj3hoNORk%2FAN6nrrpe1PhKDGEVoxgGjK%2Fimg.png)  

14번째 줄 이전의 내용은 unordered_set이랑 차이가 없습니다.  
15번째 줄부터 나오는 저 부분이 set이 unordered_set과 차별화되는 점입니다.  

unordered_set에서는 "-40보다 크면서 가장 작은 원소가 무엇인가?" 라는 문제를 풀기 위해서는 그냥 그냥 모든 원소를 다 살펴보는 방법 밖에는 없었습니다.  
set에는 원소가 정렬되어 있기 때문에 O(lg N)에 가능합니다.  

- iterator
  - prev
  - next
  - advance와 같은 멤버 함수를 이용해서 좌우로 움직일 수 있습니다.
  - begin()은 처음 원소의 iterator를 반환하고
  - end()는 마지막 원소의 한칸 뒤의 iterator를 반환합니다.
  - find  

- lower_bound는 이분탐색 단원에서 나왔던 그 lower_bound랑 기능이 동일한데, 특정 원소가 삽입되어도 오름차순 순서가 그대로 유지되는 가장 왼쪽 위치를 나타냅니다.
- 순서가 그대로 유지되는 가장 오른쪽 위치를 나타내는 upper_bound랑
- lower_bound와 upper_bound 쌍을 반환하는 equal_range도 그대로 있습니다.

이 멤버 함수들의 시간복잡도를 보면, set에서는 진짜 그냥 온갖 연산이 다 O(lg N)이라고 생각하면 됩니다. size, end, begin 함수는 그냥 멤버 변수로 가지고 있던 값을 바로 반환할테니 O(1)이지만 insert, erase, find, lower_bound, next, prev 등은 모두 O(lg N)입니다.  

단 advance의 경우에는 한 칸을 움직이는게 O(lg N)이기 때문에 advance(it, 100); 이라고 하면 이게 그냥 O(lg N)이 아니고 한 칸 움직이는 O(lg N) 연산을 100번 수행하는 상황이란건 기억을 하고 계셔야 합니다.

한편 next, prev의 경우에는 정확히는 최악의 경우 O(lg N)이지만 amortized O(1)입니다. 운이 정말 안좋다면 1번 next나 prev 연산을 하는건 O(lg N)일 수 있지만, 예를 들어 K번 next나 prev를 한다고 하면 amortized O(1)이기 때문에 O(K)의 시간이 필요하게 됩니다.

### multiset

![multiset](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FqEMGv%2Fbtrj1Yp4ihm%2FdtmdE3oKovxDnWFdfOvjK1%2Fimg.png)  

원소의 중복이 허용되는 STL이고 그 점만 유의해서 보면 set이랑 큰 차이 없이 이해가가능합니다.

- erase: 15를 1개 지우는게 아니라 모든 15를 지운다는 점을 유의하시고,
- find
  - 같은 원소가 여러 개 있을 수 있습니다.
  - 그러면 그 중에서 뭘 반환해줄지가 좀 애매한데 표준을 보면 그 중에서 아무거나 준다고 되어 있습니다.
  - 바킹독 센세 실험 gcc 버전 기준 항상 제일 먼저 등장하는 원소의 iterator를 주긴 했다만, 일단은 기본적으로는 아무거나  
- 만약 제일 먼저 등장하는 원소의 iterator가 필요한 상황이라면 find를 쓰는 대신 lower_bound를 써야 합니다.

그리고 it2를 보면 100의 upper_bound는 100이 들어갔을 때 오름차순이 유지되는 가장 오른쪽 위치니까 시작점(-10)으로부터 3칸 떨어진 곳,
즉 multiset의 마지막 원소가 있는 곳에서 한 칸 오른쪽으로 간 곳입니다.
여기는 ms.end()입니다.

그리고 여기는 가리키는 원소가 없기 때문에 만약 *it2를 출력하게끔 했다면 런타임 에러가 발생합니다.
또한 unordered_multiset과 마찬가지로 count 함수는 O(lg N)이 아닌 O(원소의 개수)만큼의 시간이 걸림에 유의하세요.

### map

![map](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcwVBO7%2Fbtrj5c8ajqZ%2Fo1ypN9u9ZjfjZNip0Ia771%2Fimg.png)  

- prev
- next
- lower_bound
- upper_bound

이런건 앞에서 설명을 잘 했어서 14번째 줄만 한 번 보고 가겠습니다.

it1이 가리키는 대상은 `pair<string, int>`이기 때문에 `it1->first`, `it1->second`로 key와 value를 가져올 수 있습니다.

### 메모

#### 1

만약 문제를 풀다가 뭔가 set, map 느낌의 성질이 필요하면서 특히 lower_bound나 prev, next 이런걸 사용해야만 풀리는 문제라면 꼭 STL set, map으로 해결을 해야 합니다.
반면에 그냥 key로 value를 빠르게 찾거나, 원소의 삽입/검색/삭제만 빠르게 처리를 해주어야 할 경우라면 STL unordered_set, unordered_map을 사용해도 상관이 없습니다.

평균적으로는 set/map보다 unordered_set/unordered_map이 빠릅니다.

바킹독 센세는 그럼에도 불구하고 set/map을 쓰는걸 선호  
왜 그렇냐면 unordered~은 평균적으로는 빠를지언정 충돌이 얼마나 빈번한가에 따라서 속도의 저하가 발생할 수 있어서 항상 빠르게 동작한다는 보장을 할 수 없다는 치명적인 단점이 있습니다.
흔치는 않겠지만 충돌을 유발하는 데이터에 대해 각 연산이 O(1)이 아닌 O(N)에 동작하게 되어서 꼼짝없이 시간초과가 발생.
반면 set/map은 평균적으로는 느릴지언정 항상 O(lg N)이기 때문에 데이터가 어떻게 들어있던간에 실행 시간이 가늠 가능

#### 2

이진 검색 트리의 연산은 O(lg N)이 되는건 맞지만 같은 O(lg N)중에서도 상당히 느립니다.
시간복잡도에 로그가 붙는걸 생각해보면 이분탐색이나 정렬 알고리즘을 떠올릴 수 있습니다.  

이분탐색에서는 lg N번의 연산 동안 인덱스의 값만 왔다갔다하면 되지만  
이진 검색 트리에서는 새로운 노드를 동적할당으로 생성하거나, 편향성을 해소해주기 위해 노드를 뗐다 붙였다 하거나 하는 식으로 다소 무거운 연산을 해야 할 일이 많습니다.  

그렇기 때문에 이분탐색이나 정렬에서는 N = 100만이라고 할 때 N개의 데이터에서 이분 탐색을 N번 하거나 정렬을 해서 O(NlgN)짜리 연산을 수행한다고 하면 크게 부담스럽지 않게 통과가 되겠다 하고 짐작을 할 수 있는 반면
이진 검색 트리에서는 N = 100만일 때 N개의 데이터에서 연산을 N번 수행해야 한다고 하면 조금 부담이 됩니다.  

상황에 따라 차이는 좀 있지만 보통 저런 상황에서 1초 제한이라고 하면 간당간당할 수가 있습니다.  
그래서 이진 검색 트리는 O(lg N)이지만 좀 느리다는걸 기억해두면 좋습니다.  

- 이진 검색 트리를 쓰는 문제인 것 같긴 한데, N = 100만과 같이 N이 좀 큰 상황에서 풀이는 O(NlgN) 같고 시간 제한이 좀 넉넉하지 않은 상황을 마주한다면 STL set/map으로 풀었을 때 시간초과가 날 가능성
  1. 구현을 해보고 TLE가 난다면
  2. STL set/map -> STL unordered_set/unordered_map 교체
  3. 이분 탐색, 정렬, 아니면 배열의 인덱스를 가지고 푸는 다른 풀이를 고민

## 메모

---

- `B-Tree`: 자식 수를 m개로 확장해서 자식 수에 제한을 두지 않고 트리의 균형을 도모한 B트리
- @TODO: 이진 검색 트리 연습문제
