---
title: "C++11 constexpr"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2025-09-29. 19:04 # Init
# last_modified_at: 2025-09-29. 19:04
---

## 머리말

---

## constexpr

---

constexpr는 컴파일 타임에 계산되는 상수.  
const expression  

```cpp
int i = 1;
int j = 1;
int a = i + j;

// 실제 코드에서는 a만 쓰고 싶어

// 윗 3줄을 굳이 exe에 넣어주지 않고, a를 계산해서 2라는 값을 a가 쓰이는 곳에 다 넣어주는
```

예전에도 preprocessor 써서 만드는 방식도 있고, 컴파일러 최적화 플래그 써서 되는 경우도 있고  
