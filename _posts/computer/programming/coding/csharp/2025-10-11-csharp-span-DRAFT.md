---
title: "[C#] Span"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2025-10-11. 14:48 # Init
# last_modified_at: 2025-10-11. 14:48 # E 초고 (Gemini)
---

## 머리말

---

> Provides a type-safe and memory-safe representation of a contiguous region of arbitrary memory.

배열에 대한 참조 뷰(View)를 제공하는 타입. (`Span` --> Array를 바라보는 View, 참조자, 포인터)  
원형(not circle) Array 없이 혼자 요소를 가질 수 없음.  

`Span`이 다루는 배열은 `System.Array` 파생 타입 뿐만 아니라 스택 상의 배열과 관리되지 않는 힙을 포함한 낮을 수준의 선형 데이터 구조 전반을 가리킨다.  

왜 Why, 힙 메모리 할당을 줄이기 위해 사용.  

`Span<T>`는 `readonly ref struct`이다.  
(`ref struct`는 오로지 스택에만 생성이 가능한 구조체이다.)  

`IReadOnlyList`처럼 `ReadOnly` 콜렉션을 저장하고 싶은데, 대상이 배열이라 `IReadOnlyList`를 쓰면 불필요한 낭비가 생기는 경우  

`Span`은 어쨌든 사용하는 수준의 것이 아니지만, 속도가 중요한 프로그래밍에 확실히 힘을 실어 준다. 만약 당신이 I/O나 통신의 핵심 라이브러리를 쓰는 일에 관련하고 있다면, 꼭 사용하기 바란다.  

## ref struct

--

구조체 선언 시 `ref` 제한자를 쓸 수 있다.  
스택 메모리에만 존재하도록 컴파일러가 강제로 제한하는 특별한 구조체가 된다.  

GC의 부담을 극도로 피하고, C++의 스택 변수처럼 빠른 성능을 내기 위해 사용되는 기능.  

### struct

일반 `struct`는 `Span<T>`나 `ref int`와 같이 메모리의 특정 위치를 직접 가리키는 '진짜 참조'를 필드(멤버 변수)로 가질 수 없다.  
일반 `struct`는 `string`이나 `object`처럼 힙에 있는 객체를 가리키는 '객체 참조'는 가질 수 있지만, 스택에 있는 변수나 배열의 특정 요소를 직접 가리키는 '메모리 참조'는 가질 수 없다.  

왜? 메모리 안전성 문제 (댕글링 참조 Dangling Reference: 이미 사라진 메모리를 계속 가리키고 있는 무효한 참조(포인트))  

```cs
// 이런 struct는 현재 C#에서 허용되지 않습니다!
public struct BadStruct 
{
    // [!] 'ref int' 필드를 가질 수 있다고 가정해 봅시다.
    public ref int NumberReference; 
}

public class Program
{
    public static BadStruct GetBadReference()
    {
        int local_variable_on_stack = 100; // 이 변수는 GetBadReference 함수가 끝나면 스택에서 사라집니다.

        BadStruct bad_instance = new BadStruct();
        bad_instance.NumberReference = ref local_variable_on_stack; // 사라질 변수를 참조합니다.

        return bad_instance; // 이 구조체를 밖으로 반환합니다.
    }

    public static void Main()
    {
        BadStruct returned_struct = GetBadReference();
        
        // --- [!] ---
        // returned_struct.NumberReference는 이미 사라진 'local_variable_on_stack'의 메모리 공간을 가리킵니다.
        // 그 메모리 공간에는 이제 어떤 쓰레기 값이 들어있을지 아무도 모릅니다.
        // 이 값을 읽거나 쓰려고 하면, 프로그램이 알 수 없는 오류를 일으키거나 즉시 충돌합니다.
        Console.WriteLine(returned_struct.NumberReference); // ??? (Undefined Behavior)
    }
}
```

위 코드에서 `returned_struct`는 스택을 탈출하여 `Main` 함수에서도 살아남았지만, 그 안에 담긴 `NumberReference`는 자신이 가리키던 대상(`local_variable_on_stack`)의 죽음을 모른 채 허공을 가리키게 됩니다. 이것이 바로 '댕글링 참조'입니다.  

C# 컴파일러는 이러한 위험을 원천적으로 차단하기 위해, "일반 struct는 힙으로 이동할 수도 있으니, 수명이 짧은 스택 변수를 참조하는 ref 필드를 절대 가질 수 없다"는 규칙을 만든 것입니다.  

#### 클래스도 똑같은 문제 있는거 아닌가요?

- C#의 철학:
  - 클래스: 클래스는 태생부터 힙에 살면서 오랫동안 살아남도록 설계되었습니다. 이런 클래스에게 "너는 이제부터 스택에서만 살아야 해!" (ref class를 만든다면) 라고 제약을 거는 것은, 클래스의 정체성을 완전히 부정하고 시스템을 혼란스럽게 만드는 일입니다. 마치 "자동차는 이제부터 하늘로만 다녀야 한다"고 법을 만드는 것과 같습니다.
  - 구조체: 구조체는 태생부터 스택에 살면서 짧은 생을 살도록 설계되었습니다. 이런 구조체에게 "너는 스택에서만 살아야 하는 규칙을 더욱 엄격하게 지켜야 하는 특별한 버전(ref struct)이 될 수 있다"고 하는 것은, 구조체의 원래 정체성(가볍고, 빠르고, 스택 친화적인)을 더욱 강화하는 자연스러운 확장입니다.

결론적으로, class와 struct 모두 ref 필드를 가지면 위험한 것은 맞지만, 그 위험을 막기 위해 '스택에 가두는' 제약을 거는 것이 합리적인 대상은 struct뿐이기 때문에 ref struct만 존재하는 것입니다.  

### ref struct 사용 이유

C#의 가장 큰 장점 중 하나는 가비지 컬렉터(GC)가 알아서 메모리를 관리해준다는 것입니다. 하지만 초당 수백만 번씩 데이터를 처리해야 하는 극단적인 고성능 환경에서는, 이 GC가 잠시 멈추는 것(GC Stop-the-world)조차 성능 저하의 원인이 될 수 있습니다.  

**문제:** "GC의 도움 없이, 내가 직접 관리하는 메모리(예: 배열의 특정 부분)를 빠르고 안전하게 참조하고 싶은데, 이걸 힙(Heap)에 할당되는 class로 만들면 GC 부담이 생기고, 일반 struct로 만들면 참조를 담는 데 한계가 있다."  

**해결책:** C# 7.2에서 ref struct가 등장했습니다. 이 키워드가 붙은 구조체는 힙 할당이 원천적으로 불가능하며, 따라서 GC의 관리 대상에서 완전히 벗어납니다. 대신, 이 구조체는 `Span<T>`와 같은 메모리 참조 자체를 필드로 가질 수 있는 강력한 능력을 얻습니다.  

### 두 가지 핵심 특징

#### 1. 스택 할당 강제 (Forced Stack Allocation)

이것이 `ref struct`의 가장 중요한 규칙이자 제약입니다. 컴파일러는 `ref struct`가 스택을 벗어나 힙으로 '탈출(escape)'할 수 있는 모든 가능성을 막아버립니다.  

따라서, `ref struct`는 다음과 같은 작업을 할 수 없습니다:  

- 박싱(Boxing)될 수 없습니다: `object`나 `dynamic` 타입, 또는 인터페이스 타입의 변수에 할당할 수 없습니다.
- 일반 클래스나 구조체의 필드가 될 수 없습니다: 클래스나 일반 구조체는 힙에 존재할 수 있으므로, 그 안에 `ref struct`를 담을 수 없습니다. (단, 다른 `ref struct`의 필드는 될 수 있습니다.)
- 배열의 요소가 될 수 없습니다: 배열은 항상 힙에 생성되므로 불가능합니다.
- `async` 메서드에서 사용할 수 없습니다: `async`/`await는` 내부적으로 상태 머신 클래스를 힙에 생성하므로, 그 안에 `ref struct` 지역 변수를 담을 수 없습니다.
- 람다나 로컬 함수의 클로저가 될 수 없습니다.

이 모든 제약은 **"이 타입은 절대로, 어떤 상황에서도 힙에 올라가선 안 된다"**는 하나의 대원칙을 지키기 위함입니다.

(댕글링 참조 문제를 방지, 스택 탈출을 막아 참조 대상보다 오래 사는 것을 불가능하게 함)  

#### 2. 참조(ref) 필드 포함 가능

엄청난 제약을 감수한 대가로, `ref struct`는 일반 `struct`가 할 수 없는 특별한 능력을 얻습니다. 바로 메모리에 대한 직접적인 참조를 필드로 가질 수 있다는 것입니다.

- `Span<T>`
- `ReadOnlySpan<T>`
- `ref T` (예: `ref int`)

이것이 가능한 이유는 `ref struct` 자체가 스택에만 존재하여 수명이 명확하게 관리되기 때문입니다. 즉, 자신이 참조하는 메모리보다 먼저 사라질 일이 없도록 컴파일러가 보장해 줄 수 있습니다.

네, C#의 ref struct에 대해 아주 명확하고 실용적인 관점에서 설명해 드리겠습니다. 공식 문서는 정확하지만 때로는 그 배경과 실제 사용 사례를 파악하기 어려울 수 있습니다.

### 주요 사용 사례 및 예제

`ref struct`의 대표적인 사용 예는 단연코 `Span<T>` 입니다. `Span<T>` 자체가 `ref struct`로 선언되어 있습니다.

하지만 우리도 직접 `ref struct`를 만들어 유용하게 사용할 수 있습니다. 문자열을 분리할 때, 추가적인 메모리 할당 없이 처리하는 예제를 보여드리겠습니다.

일반적인 `string.Split()`은 분리된 각 문자열을 힙에 새로 할당하여 배열로 반환합니다. 이는 매우 편리하지만, 고성능이 요구되는 반복문 안에서는 GC 부담을 유발합니다.

```cs
// StringSplitter.cs
// 추가적인 힙 할당 없이 문자열을 순회하며 분리하는 열거자(Enumerator)
public ref struct StringSplitter
{
    private ReadOnlySpan<char> _source;
    private readonly char _separator;

    public StringSplitter(ReadOnlySpan<char> source, char separator)
    {
        _source = source;
        _separator = separator;
        Current = default;
    }

    public ReadOnlySpan<char> Current { get; private set; }

    // C#의 foreach 구문이 인식하는 GetEnumerator() 패턴
    public StringSplitter GetEnumerator() => this;

    public bool MoveNext()
    {
        if (_source.IsEmpty)
        {
            return false;
        }

        int separatorIndex = _source.IndexOf(_separator);

        if (separatorIndex == -1) // 마지막 단어
        {
            Current = _source;
            _source = ReadOnlySpan<char>.Empty; // 소스를 비워서 루프 종료
        }
        else
        {
            Current = _source.Slice(0, separatorIndex);
            _source = _source.Slice(separatorIndex + 1);
        }

        return true;
    }
}

// 사용 예시
public class Program
{
    public static void Main()
    {
        string text = "apple,banana,cherry,date";
        char separator = ',';

        Console.WriteLine("--- StringSplitter (no heap allocation) ---");
        // StringSplitter는 ref struct이므로 foreach 구문에 직접 사용 가능
        foreach (ReadOnlySpan<char> word in new StringSplitter(text, separator))
        {
            // word.ToString()을 호출하는 순간 힙 할당이 일어나므로 주의!
            // 여기서는 ReadOnlySpan<char>를 그대로 사용
            Console.WriteLine(word.ToString()); 
        }

        Console.WriteLine("\n--- string.Split() (heap allocation) ---");
        // string.Split()은 힙에 string 배열과 각 문자열을 할당
        foreach (string word in text.Split(separator))
        {
            Console.WriteLine(word);
        }
    }
}
```

핵심: `StringSplitter`는 `foreach` 루프를 도는 동안, 분리된 각 단어(`"apple"`, `"banana"` 등)에 대한 새로운 문자열 객체를 힙에 만들지 않습니다. 대신, 원본 문자열의 메모리를 가리키는`ReadOnlySpan<char>`(메모리의 창문)만 계속해서 반환합니다. 이로써 GC 압박을 0으로 만들 수 있습니다.

#### 결론: 언제 사용해야 하는가?

- 사용해야 할 때:
  - 성능이 극도로 중요하고, GC로 인한 지연을 최소화해야 할 때.
  - 대용량의 바이트, 텍스트, 메모리 블록을 파싱하거나 처리하는 라이브러리를 만들 때.
  - `Span<T>`과 같은 메모리 참조 타입을 안전하게 캡슐화해야 할 때.
- 사용하지 말아야 할 때:
  - 일반적인 애플리케이션 로직을 작성할 때. `class`나 일반 `struct`가 훨씬 유연하고 사용하기 쉽습니다.
  - `ref struct`는 성능 최적화를 위한 '최후의 수단' 중 하나로, 명확한 성능상의 이점이 입증될 때만 사용하는 것이 좋습니다. 복잡성과 제약사항이 크기 때문입니다.

## Memory\<T\>

---

`Span<T>`: 스택(Stack)에만 살 수 있는 '초고속 단기 메모리 접근자'. 동기(Synchronous) 작업에서 실제 메모리를 직접 처리할 때 사용합니다.  
`Memory<T>`: 힙(Heap)에도 살 수 있는 '메모리 소유권 증서'. 비동기(Asynchronous) 작업에서 메모리의 위치를 저장하고 전달할 때 사용합니다.  

핵심 비교: `ref struct` vs `struct`  
두 타입의 모든 차이점은 이 한 가지 근본적인 선언의 차이에서 비롯됩니다.  

- 타입:
  - ref struct
  - 일반 struct
- 메모리 위치:
  - 오직 스택(Stack)에만 존재
  - 스택 또는 **힙(Heap)에 존재 가능
- 사용처 제약
  - 클래스의 필드가 될 수 없음 async/await를 넘나들 수 없음**
  - 클래스의 필드가 될 수 있음 async/await와 함께 사용 가능
- 핵심 역할
  - 실제 메모리 처리 (Worker) 메모리를 직접 읽고 쓰는 작업
  - 메모리 참조 전달 (Handle/Wrapper) 메모리의 위치를 저장하고 소유
- 성능
  - 매우 빠름 (추가 할당 없음)
  - Span으로 변환 시 오버헤드 거의 없음

### 비유를 통한 이해

**`Span<T>`**는 **'도서관 당일 이용권'**과 같습니다.  

도서관 안에 있는 동안(하나의 동기 함수 내에서) 어떤 책이든 매우 빠르게 찾아보고 읽을 수 있습니다.  

하지만 이 이용권은 도서관 밖으로 절대 가지고 나갈 수 없습니다. (힙에 저장하거나 `async` 메소드 밖으로 전달할 수 없음)  

수명이 매우 짧지만, 그 안에서는 최고의 성능을 보장합니다.  

**`Memory<T>`**는 **'도서관 회원증'**과 같습니다.  

이 회원증은 지갑(클래스)에 넣어 다닐 수 있고, 내일 다시 와서 사용할 수도 있습니다 (`async` 메소드).  

회원증 자체로는 책을 바로 읽을 수 없습니다. 책을 읽으려면, 이 회원증을 사서에게 보여주고 '당일 이용권(`Span<T>`)'을 발급받아야 합니다.  

메모리의 소유권을 나타내며, 필요할 때마다 실제 작업을 위한 `Span<T>`를 생성할 수 있는 능력을 가집니다.  

둘의 관계: `Memory<T>`에서 `Span<T>`으로  
이것이 둘의 관계를 보여주는 가장 중요한 포인트입니다.  

`Memory<T>`가 있으면 언제든지 `Span<T>`를 얻을 수 있지만, `Span<T>`만으로는 `Memory<T>`를 얻을 수 없습니다.  

도서관 회원증이 있으면 당일 이용권을 발급받을 수 있지만, 당일 이용권만으로는 회원증을 만들 수 없는 것과 같습니다.  

```cs
// 1. 힙에 배열을 할당합니다.
byte[] arrayOnHeap = new byte[1024];

// 2. 이 배열 전체를 가리키는 '회원증'(Memory<T>)을 만듭니다.
// 이 memory 변수는 클래스의 필드가 되거나 async 메소드를 넘나들 수 있습니다.
Memory<byte> memory = new Memory<byte>(arrayOnHeap);

// 3. 비동기 작업을 시뮬레이션
await Task.Delay(100);

// 4. 실제 데이터 처리가 필요한 시점에, '회원증'으로 '당일 이용권'(Span<T>)을 발급받습니다.
// memory.Span 프로퍼티는 거의 비용 없이 Span<T>를 생성합니다.
Span<byte> span = memory.Span;

// 5. 이제 Span<T>를 사용하여 초고속으로 메모리에 직접 접근하여 작업합니다.
for (int i = 0; i < span.Length; i++)
{
    span[i] = 0xFF;
}

// Span<byte> someSpan = ...
// Memory<byte> someMemory = someSpan; // 컴파일 에러! Span은 수명이 짧아 Memory로 변환될 수 없습니다.
```

언제 무엇을 사용해야 할까요?
API의 파라미터로는 `Span<T>` 또는 `ReadOnlySpan<T>`를 사용하세요.

함수의 역할이 데이터를 '처리'하는 것이라면, `Span<T>`를 받는 것이 가장 효율적이고 유연합니다. 호출하는 쪽에서 배열, `stackalloc`, `Memory<T>` 등 어떤 형태의 데이터든 `Span`을 통해 넘겨줄 수 있기 때문입니다.

```cs
// 이 함수는 어떤 종류의 메모리든 처리할 수 있는 매우 유연한 함수입니다.
public void ProcessData(Span<byte> data)
{
    // ... 데이터 처리 로직 ...
}
```

`async` 작업이나 클래스 내에서 메모리 조각을 저장해야 할 때는 `Memory<T>`를 사용하세요.

네트워크에서 데이터를 수신하여 버퍼에 저장해두었다가 나중에 처리해야 하는 경우, 이 버퍼를 `Memory<T>` 타입의 필드로 저장해두는 것이 정석입니다.

```cs
public class NetworkProcessor
{
    // 수신한 데이터 패킷을 '회원증' 형태로 저장해 둡니다.
    private Memory<byte> _packetBuffer;

    public async Task ReceiveDataAsync()
    {
        _packetBuffer = await _networkStream.ReadAsync(...);

        // 나중에 이 _packetBuffer.Span을 통해 실제 데이터를 처리합니다.
    }
}
```

결론적으로, 두 타입은 서로 경쟁하는 관계가 아니라, `Memory<T>`가 데이터의 생명 주기를 관리하고, `Span<T>`가 실제 작업을 수행하는 완벽한 협력 관계에 있습니다.  

## 예제

---

### 배열에 대한 View를 제공

```cs
// 코드 출처: ['tsyang' - '(C# 7.2) Span\<T\>'](https://tsyang.tistory.com/m/121)
{
    var arr = new int[] { 0, 1, 2, 3 };

    // [!] 힙 영역에 메모리 할당
    {
        // 뭔본 배열과 똑같은 데이터를 힙 영역에 메모리를 할당하고 초기화
        var left = arr.Take(arr.Length / 2).ToArray(); // 앞에서부터 절반을 배열로
        var right = arr.Skip(arr.Length / 2).ToArray(); // 절반 건너뛰고 배열로
    }

    // [!] Span<T>를 이용하면 힙 영역에 메모리를 생성하지 않고도 원본 배열에 대한 참조를 제공할 수 있다.
    {
        // T[] => Span<T>로의 암시적 형변환
        Span<int> view = arr;

        // [!] Span<T>
        {
            // 추가적인 힙 메모리 할당 없음
            Span<int> leftView = view.Slice(0, arr.Length / 2);
            Span<int> rightView = view.Slice(arr.Length / 2);

            // 원본 배열 수정됨
            rightView[0] = 99;
        }

        // [!] ReadOnlySpan<T>
        {
            // 추가적인 힙 메모리 할당 없음
            ReadOnlySpan<int> leftViewReadOnly = view.Slice(0, arr.Length / 2);
            ReadOnlySpan<int> rightViewReadOnly = view.Slice(arr.Length / 2);
            
            // [!] 컴파일 에러: 원본 배열 수정 불가능
            // rightView[0] = 99;
        }
    }
}
```

`Span<T>`를 활용하면 힙 메모리에 대한 할당과 초기화가 없으므로 성능이 향상되고 가비지를 만들지도 않는다.  

### stackalloc

본디 `stackalloc`을 사용하려면 `unsafe` 구문 안에서 포인터와 함께 사용해야 했지만 `Span<T>`를 사용하면 그러지 않아도 된다.  

가령 Merge Sort를 구현하기 위해 내부적으로 임시 공간을 사용한다고 해보자. (실제로 Merge Sort는 임시 메모리 공간 없이 동작 가능하다.) 이런 임시 공간들은 가비지가 되기 때문에 성능에 좋지 않다.

```cs
{
    var leftTempArray = new int[leftArrayLength];
    var rightTempArray = new int[rightArrayLength];
}
```

그러나 `stackalloc`을 이용하면 스택 영역에 배열을 만들 수 있기 때문에 게임 제작에서는 이러한 방법이 꽤 활용되는데,  
`Span<T>`를 사용한다면 `unsafe` 구문을 사용하지 않고도 스택에 배열을 생성한 뒤 이를 다룰 수 있다.

```cs
{
    Span<int> leftTempArray = stackalloc int[leftArrayLength];
    Span<int> rightTempArray = stackalloc int[rightArrayLength];
}
```

다만 스택에 배열을 할당할 때는 아래와 같이 최대 사용 용량을 지정해서 사용해야 스택 오버플로를 방지할 수 있다.

```cs
const int MaxStackLimit = 1024;
Span<byte> buffer = inputLength <= MaxStackLimit ? stackalloc byte[MaxStackLimit] : new byte[inputLength];
```

### vs ArraySegment

종래 .Net의 배열 뷰라고 하면 System.ArraySegment 타입이었다. Span에는 ArraySegment에 비해 다음과 같은 장점이 있다.  

- 성능이 좋다.
- 읽기 전용 버전(ReadOnlySpan)가 준비되어 있다.
- System.Array뿐만 아니라 스택 배열과 관리되지 않는 힙에 대해서도 이용 가능하다.
- T가 관리되지 않는 타입일 때, MemoryMarshal에 따라 타입을 넘는 유연한 읽고 쓰기가 가능하다.
- IList으로 캐스팅 하지 않아도 인덱서를 사용할 수 있다.

대충 이것만으로도 Span의 존재 의의가 전해질 것이다. 물론 ArraySegment은 할 수 있지만 Span는 할 수 없는 경우도 있지만, Span쪽이 뛰어난 부분이 많다.  

### No more unsafe

C#과 같은 객체 지향 언어에서는 외부에서 본 행동을 인터페이스로 정의하고 구현을 은폐함으로써 복잡한 구현에서 추상화된 기능을 꺼내왔다. 이러한 객체 지향의 실현 기구는 어느 정도의 계산 능력을 필요로 하고 있지만, 오늘날의 개발은 거의 필수라고 말해도 좋을 정도로 성공을 거두고 있다.  

그런데 C#은 상호 운용성이나 성능 이라든지에 적당히 신경을 쓰고 있는 언어이므로, 부분적으로 제한을 걸면서 낮은 수준의 데이터 구조를 지원하고 있다. 즉 `System.Array`, 스택 배열, 관리되지 않는 힙 이라는 세 종류의 배열이 있다.  

기존의 안전한 컨텍스트에서 사용할 수 있는 것은 `System.Array`뿐으로 다른 2개의 이용에는 `unsafe`가 필요했다. 3종의 배열은 모두 첫 번째 요소에 대한 참조와 배열의 길이를 가지며, 내부적으로 메모리에 연속하고, 색인 사용이 가능하다는 그야말로 추상화 할 것 같은 공통된 기능을 가지지만, 포인터 없이는 실현 될 수 없다. 라이브러리라면 몰라도 응용 프로그램 코드를 안전하지 않은 컨텍스트로 쓴다는 것은 마음이 내키지 않는다.  

그러던 중 `Span`이 구현 되었다. 앞서 언급했듯이, `Span`은 3종의 배열 구조에서도 만들 수 있을뿐만 아니라 같게 다룬다. 인터페이스 타입과는 다르지만, "배열 타입"으로 기대되는 행동을 멋지게 추상화 되어 있다고 할 수 있겠다.  

### The type is the document, the type is the contract

내가 생각하는 정적 타이핑의 가장 큰 장점은 형식 자체가 문서화 되고, 또한 계약이 되는 것이다.  
타입이 명시된 코드는 컴파일 시에 검증된 일종의 코딩 실수는 논리적 보증에서 검출된다.  
예를 들어, `IReadOnlyList` 타입 개체의 인덱서에 값을 `set` 하려고 하면 튕겨진다. 정적 타입 언어로 프로그래밍을 하는데 이 장점을 살리는 코드를 작성하는 것은 소중히 하고 싶다는 지침이다.  

기존 배열에 대해서 타입에 의한 계약을 베푸는 수단은 부족했다.  
예를 들어 다음의 코드를 보자.  

```cs
public class StreamReadingBuffer
{
    public int CurrentPosition { get; private set; }

    public byte[] Buffer => _buffer;
    private readonly byte[] _buffer;

    // ～～～
}
```

이 클래스는 이름에서 알 수 있듯이 바이너리 스트림 읽기를 효율화 하기 위해 버퍼를 감싼 것이다.  
현재의 읽기 버퍼를 Buffer 속성으로 공개하고 있지만 `byte[]` 형식은 내용의 변경을 방지 할 수 없다.  
게다가 보지 않으면 안되는 위치도 `CurrentPosition` 속성으로 분리되어 버렸다.  
보통 이러한 경우에는 원시 형식 대신 `IReadOnlyList` 등의 형태로 읽기 전용 제약을 거는 것을 권장되지만, 일부러 배열 형을 사용하고 있기 때문에 상상할 수 있듯이, 이 녀석은 꽤 성능에 신경을 쓴 케이스로 사용되는 것을 상정하고 있다. 불필요한 인터페이스를 사이에 두어서 실행 속도를 악화시키는 시그니쳐는 안전하다 해도 미움 받는다.  

`Span` / `ReadOnlySpan`이 도입 된 것으로, 이 같은 장면은 효율성과 안전성을 양립 할 수 있게 되었다.  

```cs
public class StreamReadingBuffer
{
    private int _currentPosition;

    public ReadOnlySpan<byte> Buffer => _buffer.Slice(_currentPosition);
    private readonly byte[] _buffer;

    // ～～～
}
```

위의 코드에서는 `Buffer`의 내용을 클래스의 사용자가 다시 작성할 수 없으며, 수신 버퍼 자체가 현재 위치에 따라서 분리해 낸 것으로 되어 있다. 이 변경에 따른 배열에 대한 오버 헤드도 극히 소량이다.  
읽기 전용인 것과 봐야할 장소가 확보된 메모리 전체가 아닌 일부라는 것이 `ReadOnlySpan`로 있다는 사실에 의해 코드에서 표현되고 있는 것이다.  

### 그래도 나는 바이너리를 읽고 싶다

보통의 배열에 대해서도 그 표현력에 따라 역할을 가질 `Span`이지만 이 진가는 `T`가 관리되지 않는 형식 일 때 발휘된다.  

관리되지 않는 형식은 .Net의 관리 참조를 포함하지 않는 형태이다.  
가비지 컬렉터가 신경 쓰지 않는 타입 또는 C로 동등한 구조체를 만들 수 있는 형태라고 생각해도 좋다.  

모든 비 관리 타입 객체는 등가인 바이트 배열을 생각할 수 있다.  
`Span`은 `MemoryMarshal`을 사용하여 임의의 관리 되지 않는 형식 간에 변환 할 수 있다.  

```cs
using  System.Runtime.InteropServices;

// float 배열을 int 배열로서 읽고 쓸 수 있도록 한다. 
public void Foo(Span<float> bufferAsFloat)
{
    Span<int> bufferAsInt = MemoryMarshal.Cast<float, int>(bufferAsFloat);
    // ～～～
}
```

이 기능을 사용하려면 최대의 유스 케이스는 바이너리 스트림의 읽고 쓰기이다.  
독자적인(또는 독자적인 아닌 특수한) 프로토콜과 파일 형식을 구현할 때 구조체를 정의 해두면 빠르고 간단한 읽고 쓰기가 가능하게 된다. 예를 들어, 다음 코드는 `PortableExecutable` 파일의 헤더를 읽는 코드이다.  

```cs
// < PE 파일 헤더를 읽는다 >
using System;
using System.Runtime.InteropServices;

// DosHeader, ImageFileHeader, ImageOptionalHeaderなどは割愛
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct NtHeader
{
    public readonly uint Signature;
    public readonly ImageFileHeader FileHeader;
    public readonly ImageOptionalHeader OptionalHeader;
}

public static class Sample
{
    public static NtHeader ReadHeader(byte[] buffer)
    {
        var dosHeader = MemoryMarshal.Cast<byte, DosHeader>(buffer)[0];
        return MemoryMarshal.Cast<byte, NtHeader>(buffer.AsSpan(dosHeader.e_lfanew))[0];
    }
}
```

이른바 C++의 `reinterpret_cast` 같은 동작이지만, 이것을 손쉽게 할 수 있게 된 셈이다.

### API 충분

`ArraySegment`은 BCL 내부에서의 대응도 짧고, 표준 API로 다루기에 미묘했다. 어떤 타입이 사용될지 여부에 대해 API의 충족은 중요한 요소이다. 전달도 받을 수도 없다면 결국 직접 변환이 필요하게 되고, 볼 기회도 훨씬 줄어들 것이다.  

`Span`의 도입에서, 다음의 타입 등에 대응이 담겨 있다.  

- 기본 숫자 형식
- `System.BitConverter`
- `System.IO.Stream`
- `System.Text.Encoding`

즉, 지금까지 낮은 수준 용도로 배열을 받는 API에 상당 부분 `Span`이 대응된 것이다. 전술 한 바와 같이 `Span`는 `stackalloc`도 사용할 수 있으므로 힙을 사용하지 않고 스트림을 읽을 수 있다.  

## Span를 사용하면 안 되는 3개의 케이스

뭐, 여기까지 `Span` 선전을 해 왔지만, 모든 경우에 최적의 솔루션이라는 것은 없다. 오히려 어떤 면에서의 편리성을 추구한 결과, 다른 방면에서는 불편한 형태가 된다. 여기에서는 이 성격에서 `Span` 사용이 적합하지 않거나 사용할 수 없는 케이스를 소개한다.  

### 클래스의 멤버로 사용할 수 없던데?

A. 스펙이다.  

앞서 언급했듯이 `Span`은 `ref` 구조체라는 타입이다. `ref` 구조체의 객체는 반드시 스택에 놓여 있는 것을 보증하고 있다. 반복하면 힙에 실릴 수 있는 것은 일절 할 수 없다는 것이다.  

```cs
// < Span이 할 수 없는 것 >
public class Hoge
{
    private Span<object> _span; // NG: ref 구조체 타입의 필드로 되지 앟는다.

    public static Span<int> Foo()
    {
        Span<int> span = stackalloc int[16];
        Bar(span); // NG: 제널릭 타입인수로 할 수 없다.
        var obj = (object)span; // NG: object로 업캐스트 할 수 없다
        Func<int> baz = () => span[0]; // NG: 델리게이트 캡쳐할 수 없다.

        // True. 업캐스트 할 수 없지만 내부적으로는 object의 파생 타입.
        // 어디까지나 C# 컴파일러가 정적 검증으로 금지하고 있을뿐이므로 당연하다면 당연하다.
        Console.WriteLine($"Span<T> is a subtype of object: {typeof(Span<>).IsSubclassOf(typeof(object))}");

        return span; // NG: stackalloc은 메소드 영역에서 나오면 죽으므로 반환값으로 할 수 없다.
    }

    public static void Bar<T>(T value){}
}

public ref struct Fuga : IEnumerable<int> // NG: 인터페이스를 구현할 수 없다
{
}
```

당연하지만, 클래스와 인터페이스를 통해 객체 지향을 실현하는 C#에서 이것은 매우 엄격한 제한이다.  
여기서 제한에 걸리는 용도라면, 배열을 사용하거나 `System.Memory`라는 형식을 사용하자.  

### `IList`/ `IReadOnlyList`/ `ICollection`/ `IEnumerable`로 충분?

만약 그렇게 생각한다면, 대개의 경우 그 감각이 옳은 것 같다.  
이미 언급 한 바와 같이, `Span`는 저수준 프로그래밍을 위해 존재한다. 대규모 응용 프로그램도 적당히 쉽게 보수성 좋게 쓸 수 있는 C#에서 원시 바이너리에 주목하는 케이스는 결코 많지 않다. 실체는 `List`가 되는것을 사용으로 두어서 적절한 인터페이스를 공개하도록 하는 것이 .Net 타입 시스템에서 솔직하고, 사용 측에 매우 편리하고 직관적인 것이 사상의 대부분이다. 성능이 매우 중요하다든가, 레거시 프로토콜을 구현하고 싶다든가, 명확한 이유가 없다면 `Span`을 쓸 필요는 없다.  

## 메모

---

- **참고**
  - ['tsyang' - '(C# 7.2) Span\<T\>'](https://tsyang.tistory.com/m/121)
    - ['MSDN' - 'Span'](https://learn.microsoft.com/ko-kr/dotnet/api/system.span-1?view=net-10.0)
    - ['MSDN' - 'stackalloc'](https://learn.microsoft.com/en-us/dotnet/csharp/language-reference/operators/stackalloc)
    - ['MSDN' - 'Memory-related and span types'](https://learn.microsoft.com/en-us/dotnet/standard/memory-and-spans/)
    - ['MSDN' - 'Memory\<T\> and Span\<T\> usage guidelines'](https://learn.microsoft.com/en-us/dotnet/standard/memory-and-spans/memory-t-usage-guidelines)
  - ['jacking75' - 'Span 를 사용해야 할 5가지 이유'](https://jacking75.github.io/NET_Span_5_Reasons_to_Use/)
    - ['aka-nse - 'Span\<T\>を使うべき5つの理由'](https://qiita.com/aka-nse/items/cea3c6f91413c3582b5f)
  - ['dotnetdev' - 'Span\<T\>을 실무에 적용해보자'](https://forum.dotnetdev.kr/t/span-t-slog/530)
    - ['corefxlab' - 'span'](https://github.com/dotnet/corefxlab/blob/archive/docs/specs/span.md#requirements)
    - ['정성태 ' - 'C# - 고성능이 필요한 환경에서 GC가 발생하지 않는 네이티브 힙 사용'](https://www.sysnet.pe.kr/2/0/12036)
- **키워드:**
  - `Span<T>`
  - `Memory<T>`
  - `ref`, `ref struct`
  - `stackalloc`
- **TODO:**
  - 꼬리질문
  - ['tearsinrain' - 'Span\<T\>, Memory\<T\> 소개와 활용법](https://tearsinrain.tistory.com/22)
