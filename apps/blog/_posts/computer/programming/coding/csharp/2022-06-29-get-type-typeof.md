---
title: "C# GetType() typeof()"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: [CSharp]
image: "/assets/img/background/kururu-lab.jpg"

date: 2022-06-29. 13:01
# last_modified_at: 2023-10-26. 13:08
last_modified_at: 2024-11-19. 13:04 # 정리
---

@ TODO: 글 정리

## 요약

---

- 둘 다 **Meta-Information**을 포함한 `System.Type`을 가져옴
- `GetType()`: 실행 시점 평가
- `typeof()`: 컴파일 시점 평가, 정적으로

```csharp
// Testing Types | 타입 식별
public class SomeDerivedClass: SomeClass { }
SomeDerivedClass someInstance = new SomeDerivedClass();

if (someInstance is SomeClass)                       // ==> true
// is는 상속된 클래스도 포함하여 검사하지만, 

if (someInstance.GetType() == typeof(SomeClass))     // ==> false
// GetType()은 정확한 타입을 반환
// mycontrol이 TextBox에게 derived 파생되어있을지도
```

## typeof()

---

() 안에 타입 이름을(문자열이 아니라 식별자 `Identifier`) 넣고, 타입을 가져오는 키워드 (컴파일 타임 시점 => 정적인 타입)  

```csharp
typeof(int); // => Int32
typeof(string); // => string
```

## GetType()

---

() 안에 인스턴스를 넣고, 타입을 가져오는 함수 (런타임 시점)  

```csharp
int temp = 0;
GetType(temp); // => Int32

TempClass temp = new();
GetType(temp); // => TempClass
```

## 비교

---

```csharp
string s = "Hi";
Type t1 = typeof(string);
Type t2 = s.GetType();

t1 == t2 // => true
```

```csharp
object obj = "Hi";
Type t1 = typeof(object);  // ==> object
Type t2 = obj.GetType();   // ==> string

t1 == t2 // => false
```

## 메모

---

- 참고: [When and where to use GetType() or typeof()? [duplicate]](https://stackoverflow.com/questions/11312111/when-and-where-to-use-gettype-or-typeof)
- 참고: [20200219[C#] GetType메서드와 typeof연산자](https://funfunhanblog.tistory.com/313)
