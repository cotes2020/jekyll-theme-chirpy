---
title: "Garbage-Collection"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: []
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2024-04-23. 05:21
# last_modified_at: 2024-08-29. 21:56
last_modified_at: 2024-11-19. 11:40 # Garbage, C#
---

## Q

---

- 가비지에 대하여

- GC에 대해 설명해봐라
- GC 최적화를 위해 할수있는일은 무엇이 있을지 설명해봐라
- GC의 장점과 단점에 대해 설명해봐라
- 세대별 가비지 컬렉션
- Unity
  - `Instantiate/Destroy`를 반복적으로 사용하면 메모리 증가
  - -> 오브젝트 풀 사용

## Garbage

---

프로그램이 동적으로 할당한 메모리 중에서 더 이상 사용하지 않는 메모리를 가비지라고 한다.  
가비지는 프로그램의 성능을 저하시키고 메모리 누수를 발생시킬 수 있으므로 가비지 컬렉션을 통해 메모리를 해제해주어야 한다.  

## Garbage가 생기는 상황

---

### 문자열

```csharp
string str = "Hello";
str = "World";
```

`World`가 새로운 메모리에 할당되고 `Hello`는 더 이상 사용되지 않기 때문에 가비지가 된다.  

```csharp
string str = "Hello";
str += "World";
```

`HelloWorld`가 새로운 메모리에 할당되고 `Hello`는 더 이상 사용되지 않기 때문에 가비지가 된다.  

많은 문자열을 조합하는 경우에는 `StringBuilder`를 사용하여 가비지를 줄일 수 있다.  
`Append`를 통해 미리 할당된 메모리에 문자열을 복사만 해뒀다가, `ToString`을 통해 한 번에 문자열 객체로 변환한다.  

```csharp
StringBuilder sb = new StringBuilder();
sb.Append("Hello");
sb.Append("World");
string str = sb.ToString();
```

### 객체 (new)

```csharp
class MyClass
{
    public int value;
}

MyClass myClass = new MyClass();
myClass = new MyClass();
```

- 클래스를 인스턴싱하려면 `new`를 사용해야 한다.
- `new`로 생성된 객체는 Heap에 할당되고 Stack에는 Heap에 할당된 객체의 주소값이 저장된다.
- 메서드 안에서 new로 생성한 객체는 메서드가 종료되면 가비지가 된다.
- -> 메서드가 자주 호출되는 경우에는 가비지가 빠르게 쌓일 수 있다.

- 클래스 대신 구조체를 사용하면 (가능하다면) Stack에 할당되기 때문에 가비지가 발생하지 않는다.

### Boxing, Unboxing

```csharp
int num = 10;
object obj = num;
int num2 = (int)obj;
```

- 참고: ['Boxing, Unboxing'](/posts/boxing-unboxing)

### 이중 참조

```csharp
MyClass myClass = new MyClass();
MyClass myClass2 = myClass;
```

- `myClass`와 `myClass2`는 같은 객체를 참조하고 있기 때문에 둘 중 하나만 참조하고 있어도 가비지가 되지 않는다.

## GC

---

동적으로 `Garbage`를 자동으로 탐지하고 해제하는 메모리 기법.  
이는 메모리 누수와 같은 문제점을 방지하여 프로그램의 안정성과 성능을 향상하는 데 사용한다.  

### 세대별 GC

세대가 낮은 메모리부터 메모리 해제를 해준 다음 메모리 컴펙션을 해준다.

- 0세대: GC를 한번도 겪지 않은 갓 생성된 객체가 대상
- 1세대: GC를 1회 겪은 객체가 대상
- 2세대: GC를 2회 이상 겪은 객체가 대상(전체를 의미)
  - 2세대 GC를 할 시 Full Garbage Collection이라 하고 전체 Heap에 대하여 GC하는 것을 의미한다.

### 세대를 나누는 근거

- 최근에 생성된 객체일수록 생명주기가 짧을 가능성이높고, 오래된 객체일수록 생명주기가 길 가능성이 높습니다.
- 최근에 생성된 객체끼리는 서로 연관성이 높을 수 있으며, 비슷한 시점에 자주 액세스 됩니다.
- 일부분 Heap에 대해 GC를 하는 것이 전체 GC를 하는 것 보다 빠릅니다.

### 언어에서

- 매니지드 언어 (GC 지원)
  - C#: .NET 프레임워크에서 실행.

- 언매니지드 언어
  - C++: 직접 메모리를 할당하고 해제.

## **C#**

---

```csharp
GC.Collect(); // 가비지 컬렉션 강제 수행
GC.Collect(0); // 0세대 가비지 컬렉션 강제 수행
GC.Collect(0, GCCollectionMode.Forced); // 0세대 가비지 컬렉션 강제 수행
GC.WaitForPendingFinalizers(); // 가비지 컬렉션 완료 대기
GC.WaitForFullGCComplete(); // 전체 가비지 컬렉션 완료 대기
GC.GetTotalMemory(true); // 전체 메모리 사용량 반환
```

### Dispose Pattern

```csharp
public class MyClass: IDisposable
{
    private bool disposed = false;

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!disposed)
        {
            if (disposing)
            {
                // 관리되는 자원 해제
            }

            // 비관리 자원 해제
            disposed = true;
        }
    }

    ~MyClass()
    {
        Dispose(false);
    }
}
```

- `IDisposable`: 관리되는 자원과 비관리 자원을 해제하는 메서드를 정의하는 인터페이스
  - 직접 메서드를 만들어 쓸 수도 있지만, `IDisposable` 인터페이스를 상속받아 사용하는 것이 좋다.
    - 서로 다른 Type의 객체를 사용해도, 동일한 코드/방법으로 메모리를 해제할 수 있다.
    - `Dispose` 메서드만 보고도 '아, 클래스 사용이 끝나면 `Dispose`를 호출해야겠구나' 라는 것을 알 수 있다.

- `Dispose`: 관리되는 자원과 비관리 자원을 해제
  - `Dispose(true)`: 관리되는 자원을 해제
  - `Dispose(false)`: 비관리 자원을 해제
- `GC.SuppressFinalize(this)`: 파괴자를 호출하지 않도록 설정

`FileStream` 관련 객체에서 많이 볼 수 있다.  

### WeakReference

```csharp
WeakReference weakReference = new WeakReference(new MyClass());
```

- `WeakReference`: 약한 참조
- `WeakReference`를 사용하면 가비지 컬렉션 대상이 되지 않는다.
