---
title: "Fibonacci | 피보나치 수열"
# description: ""
categories: [컴퓨터, 알고리즘]
tags: [알고리듬, Fibonacci]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-11-13. 03:57
last_modified_at: 2024-11-13. 03:57 # Init
---

## Fibonacci | 피보나치 수열

---

각 항이 그 앞의 두 항의 합인 수열.  
만든 사람이 이탈리아 수학자 피보나치(Fibonacci).  

- 피보나치 수열
  - F(0) = 0
  - F(1) = 1
  - F(n) = F(n-1) + F(n-2) (n ≥ 2)
- 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...

## Recursion | 재귀

---

```cs
int RecFib(int n)
{
    if (n == 0) return 0;
    if (n == 1) return 1;
    return RecFib(n - 1) + RecFib(n - 2);
}
```

중복된 연산이 계속 발생해서 O(1.618^N)의 시간복잡도를 가진다.  
재귀 호출이 많아질수록 (N이 클수록) 스택 오버플로우(Stack Overflow) 위험이 증가한다.  

## DP

---

```cs
int DpFib(int n)
{
    int[] D = new int[n + 1];
    D[0] = 0;
    D[1] = 1;

    for (int i = 2; i <= n; i++)
        D[i] = D[i - 1] + D[i - 2];

    return D[n];
}
```

미리 배열을 만들어두고 0~N까지 하나씩 채워가는 방식.  
N+1칸을 채우고 나면 답(N)을 알 수 있으니 O(N)의 시간복잡도를 가진다.  

## Iterative | 반복문

---

```cs
int IterativeFib(int n)
{
    if (n == 0) return 0;
    if (n == 1) return 1;

    int a = 0, b = 1, c;
    for (int i = 2; i <= n; i++)
    {
        c = a + b;
        a = b;
        b = c;
    }
    return b;
}
```

반복문을 이용한 방식.  
마찬가지로 O(N)의 시간복잡도를 가지지만, 배열을 사용하지 않아 메모리를 덜 사용한다.  
O(1)의 공간복잡도를 가진다.  

## Comparsion

---

| 방법   | 시간       | 공간 | _                                        |
| ------ | ---------- | ---- | ---------------------------------------- |
| 재귀   | O(1.618^N) | O(N) | 중복 계산이 많아 비효율적                |
| DP     | O(N)       | O(N) | 중복 계산을 피하고 효율적                |
| 반복문 | O(N)       | O(1) | 공간 복잡도를 줄일 수 있는 효율적인 방법 |

중간 결과 저장/이용 유무에 따라 극적인 시간복잡도의 차이가 발생한다.  
