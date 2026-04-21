---
title: "Subset Sum Problem"
# description: ""
categories: [컴퓨터, 알고리즘]
tags: [알고리듬, Back-Tracking]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true
math: true

date: 2023-12-07. 11:38
# last_modified_at: 2023-12-13. 10:43
# last_modified_at: 2023-12-19. 02:57
last_modified_at: 2024-08-29. 21:36
---

## Subset Sum Problem (Sum of Subsets Problem)

---

부분집합의 합 문제  

유한개의 정수를 원소로 하는 집합이 있을 때 원소의 합이 0이 되는 부분집합이 존재하는지 알아내는 문제(단 공집합은 제외)  
i.e. 집합{-4, -2, 1, 3}의 경우 부분집합 {-4, 1, 3}의 원소 합이 0  
원소 합에 관한 조건을 임의의 정수로 일반화 가능 (합 0 → 합 n)  

물건의 단위 무게당 이익이 동일한 [0-1 배낭 문제](/posts/0-1-knapsack-problem/)  
0-1 배낭 문제에서 최대 이익은 배낭 용량 M을 꽉 채울 경우 = 원소를 양수로 한정하는 부분집합의 합 문제와 동일  

@ 자료 Subset0000  
@ 주의: w<sub>i</sub> < w<sub>i+1</sub>  

각 물건의 이익에 대한 정보는 더 이상 필요 없음  
주의: 물건을 무게에 따라 오름차순으로 정렬  

@ 무게 M이 꼭 0이 아니여도 됨 (임의의 정수로 일반화 가능)  

@ SSP_0000  

## Solve By [BackTracking](/posts/algorithm-back-tracking/)

---

### 상태 공간 트리

[N-Queen](/posts/n-queen/), [K-Graph-Coloring](/posts/k-graph-coloring/)과 다르게, 노드에 위치 정보가 저장되는 것이 아니고, 각 노드 기준 무게가 저장됨  

### 알고리듬

- 레벨 `i`의 노드가 유망하지 않음을 판별하는 기준
  - - 현재 `i`번째 물건까지 결정된 상태이고 현재까지 배낭에 넣은 물건들의 무게의 합을 `weight`라 하면 다음 경우에 유망하지 않은 노드
    - 규칙 1) 현재까지의 무게가 배낭의 무게를 초과: $weight + rest > M$
    - 규칙 2) 아직은 배낭에 여유가 있지만 어떤 물건을 추가로 넣더라도 초과
      - $weight + w_{i+1} > M$
      - 물건을 무게에 따라 오름차순으로 정렬한 이유
      - 규칙 2를 조사하여 가지치기를 하게 되면 규칙 1에 의한 유망하지 않은 경우는 절대로 발생 불가능 → 규칙 1은 검사할 필요 없음
    - 규칙 3) 남은 물건을 모두 배낭에 넣어도 배낭 용량에 미치지 못함
      - 남은 물건들의 무게 합을 `rest`라 할 때, $weight + rest < M$

@ SSP_0001  

- 배열을 이용하여 상태 공간 트리를 관리하는 방법
  - 기억해야 할 정보는 “걸어온 길”
  - 레벨 `i` 노드는 `i` 개 물건가운데 배낭에 넣은 물건들을 기억해야 함
    - 해당 노드가 유망하여 자식 노드로 내려간다면, 다음 단계에 `i + 1`개 물건 가운데 배낭에 넣은 물건들을 기억
    - 해당 노드가 유망하지 않아 부모 노드로 되돌아간다면, 다음 단계에 `i - 1`개 물건 가운데 배낭에 넣은 물건들을 기억

| 인덱스 | 1 | 2 | 3 | 4 |
| 포함여부 | false | true | false | true |

단말 노드가 아닌 노드가 해답이라면 해답 노드의 레벨까지 저장된 값만이 유효  

- 배열 인덱스: 물건
- 배열 값: 물건의 포함 여부

∴ n 개의 값(true/false)색을 저장할 수 있는 1차원 배열이 필요  

### 구현

`SumOfSubsetsBT(0, 0, rest)`를 호출함으로써,  
시작 변수 `rest`의 초기 값은 모든 물건의 무게 합 = 배열 `W[ ]`의 값을 모두 더한 값  

```cs
void SumOfSubsetsBT(int i, int weight, int rest)
{
    if (!Promising(i, weight, rest)) // 유망하지 않으면 패스
        return;

    if (weight == M) // 찾으면 즉시 끝 (결정 문제)
    {
        OutputSolution(i);
        Exit(); // 대충 끝나는 함수
    }
    else
    {
        // 왼쪽, 오른쪽 두 개의 노드만 보면 됨 (이진트리)
        X[i + 1] = true;
        SumOfSubsetsBT(i + 1, weight + W[i + 1], rest - W[i + 1]);
        X[i + 1] = false;
        SumOfSubsetsBT(i + 1, weight, rest - W[i + 1]);
    }
}

bool Promising(int i, int weight, int rest)
{
    // 유망하지 않은 경우
    // 1. 남은 걸 다 더해도 목표 무게에 '도달'할 수 없음 (레퀴엠 ㄷㄷ)
    // 2. 무게가 넘침
    if (weight + rest < M)
        return false;
    
    // (weight == M) → 해답
    if ((weight != M) && (weight + W[i + 1] > M))
        return false;
    
    return true;
}
```

### 분석

#### 상태 공간 트리의 노드 수

$$ 1 + 2 + 2^1 + ... + 2^n = 2^{n+1} - 1 $$

2-그래프 채색 문제와 같은 노드수  
NP-완전 문제 중에는 비교적 노드수가 적지만 여전히 지수 복잡도  

#### 유망한 노드 수

- 해밀턴 사이클 문제처럼 유망한 노드 개수를 계산할 수 없음
- 같은 개수의 정점이라 하더라도 배낭 용량이나 물건 무게에 따라 유망한 노드의 수가 달라짐
- 아주 빨리 해답을 얻을 수도 있고 상태 공간 트리를 모두 뒤져야 할 수도 있음
  - 가장 무거운 물건을 제외한 나머지 물건들의 무게 합이 배낭의 용량보다 작고, 가장 무거운 물건의 무게가 배낭의 용량과 같다면 해답을 찾기 위해 모든 노드를 탐색하여야 함
