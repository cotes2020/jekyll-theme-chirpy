---
title: "Graph, Tree | 그래프, 트리"
# description: ""
categories: [컴퓨터, 자료구조]
tags: [자료구조, Graph]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-07-14. 23:37
# last_modified_at: 2024-07-15. 06:12
# last_modified_at: 2024-08-29. 21:35
last_modified_at: 2024-09-04. 12:58
---

## Graph | 그래프

---

그래프: 정점과 간선으로 이루어진 자료구조  

각 원소를 정점(Vertex) 또는 노드(Node: 마디)라고 부르고, 간선(Edge)은 두 정점을 연결하는 선이다.  

## Tree | 트리

---

무방향이면서 사이클이 없는 연결 그래프 (Undirected Acyclic Connected Graph)  

각 노드가 서로 부모-자식 관계로 연결된다.  

- 부모-자식 관계
  - 부모(Parent): 자식을 가지는 노드
  - 자식(Child): 부모를 가지는 노드

- 루트(Root: 뿌리) 노드: 트리의 최상위 노드
- 리프(Leaf: 잎), 단말(Terminal) 노드: 자식이 없는 말단 노드

- 서브트리(Subtree): 트리의 일부분, 어떤 한 정점에 대해 그 정점과 그 정점의 자손들로 이루어진 트리

- 높이/레벨(Height/Level): 루트에서 해당 노드까지의 거리
  - 노드가 1개만 있을 때의 높이를 1로 두느냐 0으로 두느냐에 따라 높이가 달라질 수 있음

### 다른 말로 (성질)

- 연결 그래프이면서 임의의 간선을 제거하면 연결 그래프가 아니면 되는 그래프
- ⭐ 임의의 두 점을 연결하는 simple path(정점이 중복해서 나오지 않는 경로)가 유일한 그래프
- ⭐ V개의 정점을 가지고 V-1개의 간선을 가지는 연결 그래프
- 사이클이 없는 연결 그래프이면서 임의의 간선을 추가하면 사이클이 생기는 그래프
- V개의 정점을 가지고 V-1개의 간선은 가지는 Anyclic(=사이클이 없는) 그래프

### 임의의 노드를 루트로 만들 수 있다

![임의의 노드를 루트로 만들 수 있다](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcA1oan%2FbtrnP7oHGaL%2Fgy8pI7h721aCcXHD16Ns0k%2Fimg.png)  

트리를 구슬과 실로 연결된 모양이라고 생각할 때,  
아무 구슬을 잡고 위로 올려도 그 모양은 여전히 트리가 된다.  

단, 루트가 달라지면 각 노드의 부모가 달라진다. (부모-자식 관계가 달라진다.)  

### Binary Search Tree

[Binary Search Tree](/posts/data-structure-binary-search-tree/)  
