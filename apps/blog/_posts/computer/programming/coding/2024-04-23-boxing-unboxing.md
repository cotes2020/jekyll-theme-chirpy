---
title: "Boxing, Unboxing"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: []
image: "/assets/img/background/kururu-lab.jpg"

date: 2024-04-23. 05:25
# last_modified_at: 2024-04-27. 20:02
# last_modified_at: 2024-08-29. 21:56
last_modified_at: 2024-11-19. 11:56 # 정리
---

- 먼저
  - 힙과 스택에 대해 알아야 한다.
  - 값과 참조에 대해 알아야 한다.
  - 값 형식은 스택에 저장되며, 참조 형식은 힙에 저장됩니다.
  - 값 형식을 힙에 저장되도록 하려면 박싱이 필요하다.

## Q

---

- `Object` 타입에 `Value` 타입을 대입하면
  - `Value`를 레퍼런스 타입으로 `Boxing`
  - `Stack`에 있던 `Value`를 `Heap`으로 복사 후 주소값을 할당
- `박싱과 언박싱`에 대해 설명해 주세요.
- 가비지에 대하여

## Boxing, Unboxing

---

### Boxing | 박싱

값 형식의 인스턴스 -> 참조 형식으로 변환.  

```cs
int n = 4615; // 값
object someObject = n; //값 -> 참조
```

- Heap에 새 개체를 만들고 (새 메모리 할당)
  - 박싱된 값은 기존보다 더 많은 메모리 공간을 사용할 수 있다.
- Stack에 있던 값 타입의 값을 개체로 복사 (복사)
- 개체에 대한 참조를 반환 (참조)

### Unboxing | 언박싱

박싱된 값 -> 원래의 값 형식으로 변환.

```cs
int n = (int)someObject; //참조 -> 값
```

- Heap에 있던 데이터를 Stack으로 복사 (복사)

### 성능

- 박싱
  - 객체를 생성 -> 사용하지 않을 때 가비지 생성
- 언박싱
  - 박싱된 객체를 다시 값 형식으로 변환 -> 메모리 복사 잡업 -> 성능

### 대신

- Generic \| 제네릭 (C# 2.0)을 사용하여 박싱과 언박싱을 피할 수 있다.
