---
title: "Pipe"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-08-29. 22:39
# last_modified_at: 2024-08-29. 22:39
---

## Pipe

---

하나의 함수를 여러 함수로 분리하고자 할 때,  
좀 더 아름답게 분리하는 방법.  

함수형 프로그래밍에서 사용되는 기법으로,  
여러 함수를 연결하여 데이터를 순차적으로 처리하는 기법이다.  

## 예시 (C#)

---

### `Pipe`를 사용하지 않는 경우

```cs
// 예제 함수들
Func<int, int> add = x => x + 1;
Func<int, int> multiply = x => x * 2;
Func<int, int> subtract = x => x - 3;

// 각 함수를 순차적으로 호출하여 데이터를 처리
int result = 5;
result = add(result);
result = multiply(result);
result = subtract(result);

Console.WriteLine(result); // (5 + 1) * 2 - 3 = 9
```

### `Pipe`를 사용하는 경우

```cs
// 예제 함수들
Func<int, int> add = x => x + 1;
Func<int, int> multiply = x => x * 2;
Func<int, int> subtract = x => x - 3;

// pipe 함수 구현
Func<T, T> Pipe<T>(params Func<T, T>[] funcs) => 
    input => funcs.Aggregate(input, (acc, func) => func(acc));

// pipe를 사용하여 함수들을 연결
var process = Pipe
(
    add,
    multiply,
    subtract
);

Console.WriteLine(process(5)); // (5 + 1) * 2 - 3 = 9
```

여러 함수를 연결하는 `Pipe`를 만들었다.  

결과를 저장하는 대신 `Pipe`를 만들어 저장하고 필요할 때 사용한다.  
이렇게 만든 `Pipe`는 재사용 가능하다.  
