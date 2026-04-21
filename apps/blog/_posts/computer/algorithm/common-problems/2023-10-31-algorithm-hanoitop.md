---
title: "알고리듬, 하노이탑"
# description: ""
categories: [컴퓨터, 알고리즘]
tags: [알고리듬]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2023-10-31. 14:11
last_modified_at: 2024-08-29. 22:19
---

## 하노이탑

---

재귀적인/순환적인 풀이로도 유명하지만,  
DC로 풀 수 있다  

규칙  
→ 원판은 한 번에 맨 위에 있는 한 개씩 옮겨야  
→ 작은 원판 위에 큰 원판을 올려놓을 수 없음  

부문제가 문제 크기 1만큼만 줄어들기 때문에  
시원하게 DC이다 ! 말하기는 내키지 않지만  
무튼 조금씩 분할하는 DC이다  
