---
title: "[C#] stackalloc"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2025-10-11. 16:57 # Init
# last_modified_at: 2025-10-11. 16:57 # E 초고 (Gemini)
---

## 머리말

---

## stackalloc

--

`stackalloc`은 배열과 같은 메모리 블록을 힙(Heap)이 아닌 스택(Stack)에 직접 할당하는 C# 키워드로, 가비지 컬렉션(GC)을 회피하여 극단적인 성능 최적화를 이끌어내기 위해 사용됩니다.  

### 핵심 개념: 왜 굳이 스택에 할당하는가?

C#에서 우리가 일반적으로 `new int[10]`와 같이 배열을 만드는 것은 힙(Heap) 메모리에 공간을 할당하는 것입니다. 힙은 크고 유연하며, 객체가 오랫동안 살아남을 수 있는 '장기 보관 창고'입니다. 하지만 이 창고는 가비지 컬렉터(GC)가 주기적으로 정리해야 하므로, 아주 작은 성능 저하(GC Overhead)가 발생할 수 있습니다.  

반면, **스택(Stack)**은 함수가 호출될 때 잠깐 생겼다가 함수가 끝나면 즉시 사라지는 '임시 작업대'입니다. 이 작업대는 매우 빠르고 GC가 전혀 관여하지 않습니다.  

`stackalloc`은 바로 이 '임시 작업대' 위에 메모리를 할당하는 방법입니다.  

목적: 아주 짧은 시간 동안만 사용할 작은 메모리 버퍼가 필요한데, 이 작업이 초당 수천, 수만 번씩 반복된다면? `new` 키워드로 매번 힙에 할당하는 것은 GC에게 큰 부담을 줍니다. `stackalloc`을 사용하면 이 부담을 0으로 만들 수 있습니다.  

### stackalloc의 두 가지 형태: 안전한 최신 방식 vs 위험한 레거시 방식

`stackalloc`은 사용하는 방식에 따라 안전성이 크게 달라집니다. 반드시 최신 방식을 사용하는 것을 권장합니다.

#### 1. 안전한 최신 방식 (`Span<T>` 활용) - 강력 추천

C# 7.2부터 도입된 이 방식은 stackalloc을 훨씬 안전하고 쉽게 만들어 줍니다.

- 문법: `Span<T> buffer = stackalloc T[size];`
- 특징:
  - 할당된 스택 메모리를 `Span<T>` 또는 `ReadOnlySpan<T>` 타입으로 받습니다.
  - `Span<T>`는 배열처럼 인덱서(`buffer[i]`)를 사용할 수 있고, 경계를 벗어나는 접근을 하면 예외를 발생시켜 메모리 안전성(Type and Memory Safety)을 보장합니다.
  - `unsafe` 키워드가 필요 없습니다.

```cs
public int SumSmallArray()
{
    // 스택에 정수 10개를 저장할 공간을 할당하고 Span<int>로 참조합니다.
    Span<int> numbers = stackalloc int[10];

    // 일반 배열처럼 사용 가능
    for (int i = 0; i < numbers.Length; i++)
    {
        numbers[i] = i + 1;
    }

    // numbers[10] = 5; // 이 코드는 IndexOutOfRangeException 예외를 발생시켜 안전합니다.

    int sum = 0;
    foreach (int num in numbers)
    {
        sum += num;
    }

    return sum; 
} // 함수가 여기서 끝나면, 'numbers'가 사용하던 스택 메모리는 자동으로 소멸됩니다.
```

#### 2. 위험한 레거시 방식 (포인터 활용) - 특별한 경우에만 사용

과거 C# 버전에서 사용하던 방식으로, unsafe 컨텍스트 안에서만 사용할 수 있습니다.

- 문법: `T* buffer = stackalloc T[size];`
- 특징:
  - 할당된 스택 메모리를 **포인터(*)**로 직접 받습니다.
  - 포인터 연산(`buffer + i`)이 가능하지만, 경계 검사(Bounds Checking)가 전혀 없습니다. 할당된 크기를 넘어서 메모리를 읽거나 쓰면 프로그램이 즉시 충돌하거나 예측할 수 없는 오류를 일으킵니다.
  - `unsafe` 키워드와 컴파일 옵션이 필요합니다.

```cs
public unsafe int SumWithPointers()
{
    // unsafe 블록 안에서만 사용 가능
    int* numbers = stackalloc int[10];

    for (int i = 0; i < 10; i++)
    {
        numbers[i] = i + 1; // C#이 편의를 위해 인덱서 문법을 제공
        // *(numbers + i) = i + 1; // C/C++ 스타일의 포인터 연산도 가능
    }

    // numbers[10] = 5; // 경고나 예외 없이 메모리를 침범합니다! 매우 위험!

    int sum = 0;
    for (int i = 0; i < 10; i++)
    {
        sum += numbers[i];
    }
    return sum;
}
```

### 핵심 규칙 및 제약사항

`stackalloc`은 스택 메모리의 특성상 매우 엄격한 규칙을 따릅니다.

함수를 탈출할 수 없다: `stackalloc`으로 할당된 메모리는 해당 함수가 끝나는 즉시 사라집니다. 따라서 `return` 문으로 `stackalloc`으로 만든 `Span<T>`나 포인터를 반환할 수 없습니다. 컴파일 에러가 발생합니다.

크기 제한: 스택의 크기는 힙에 비해 매우 작습니다 (보통 1MB). 너무 큰 메모리를 `stackalloc`으로 할당하면 **StackOverflowException**이 발생하여 프로그램이 즉시 종료됩니다. 수백 KB 이상의 큰 버퍼에는 절대 사용하면 안 됩니다.

`catch`, `finally` 블록 내 사용 불가: 예외 처리 블록 안에서는 `stackalloc`을 사용할 수 없습니다.

### 언제 사용해야 하는가?

`stackalloc`은 일반적인 애플리케이션 개발이 아닌, 성능이 한계까지 요구되는 특정 시나리오를 위한 도구입니다.  

- 고성능 루프: 반복문 안에서 작은 임시 버퍼가 계속 필요한 경우.
- 데이터 파싱: 바이트 스트림이나 문자열을 파싱할 때, 불필요한 `string`이나 `byte[]` 객체 생성을 피하고 싶을 때.
- 네이티브 라이브러리 호출 (P/Invoke): C/C++로 만들어진 함수에 잠시 사용할 메모리 블록의 포인터를 넘겨줘야 할 때.

### 요약 비교

- 특징:
  - `new T[]` (힙 할당)
  - `stackalloc T[]` (스택 할당)
- 메모리 위치
  - 힙 (Heap)
  - 스택 (Stack)
- 관리 주체
  - 가비지 컬렉터 (GC)
  - 함수 호출 스택 (자동 소멸)
- 성능
  - 상대적으로 느림 (GC 부담 존재)
  - 매우 빠름 (GC 부담 없음)
- 수명
  - GC가 수거할 때까지
  - 함수가 끝날 때까지 (매우 짧음)
- 크기 제한
  - 매우 큼 (시스템 메모리 한도)
  - 매우 작음 (약 1MB)
- 안전성
  - 안전함 (배열)
  - 안전함 (`Span<T>` 사용 시), 위험함 (포인터 사용 시)
- 주 사용처
  - 대부분의 일반적인 경우
  - 극단적인 성능 최적화가 필요한 경우

결론적으로, `stackalloc`은 C#에게 C++ 수준의 저수준 메모리 제어 능력을 부여하는 강력한 도구이지만, 그에 따르는 책임과 제약을 명확히 이해하고 `Span<T>`와 함께 안전하게 사용해야 합니다.  

## 메모

---

- **참고**
- **키워드:**
  - `stackalloc`
- **TODO:**
