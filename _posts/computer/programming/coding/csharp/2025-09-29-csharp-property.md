---
title: "Property (프로퍼티)"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: []
image: "/assets/img/background/kururu-lab.jpg"

date: 2025-09-29. 07:26 # S Init
last_modified_at: 2025-09-29. 23:42 # E +정리: get/set
---

## 머리말

---

## Property (프로퍼티)

---

- 클래스 속성
  - 흔히 get, set을 정의 해주는 간편한 예약어이다.
- 프로퍼티는 근본적으로 메서드 ?

## get/set

---

`public` 필드 쓰지 않고, get/set 메서드 쓰는 이유 -> OOP **정보 은닉** 지키기 위함.  
데이터를 외부로부터 보호, 클래스가 허용한 안전한 방식으로만 상호작용하도록 제어.  

- 이점
  - 데이터 무결성
  - 유지보수
  - 구현의 자유

### 1. 데이터 무결성

`public` 필드는 외부에서 아무 제약 없이 값 수정 가능  
객체의 상태를 논리적으로 허용되지 않는 값으로 설정할 수 있음. (예: 체력 음수)  

반면, `set`에서 유효성 검사하면 데이터의 무결성 보장 가능.

```csharp
public void SetHealth(int value)
{
    this.health = Mathf.Max(0, value);
}
```

### 2. 유지보수

값이 변경될 때마다 추가 작업이 필요해진다면 (예: UI 업데이트, 로그)  
`public` 필드는 값 설정하는 모든 코드 찾아 수정해야 함.  

반면, `set`은 내부만 수정하면 돼서 쉽고 안전함.  

```csharp
public void SetHealth(int value)
{
    this.health = value;
    UpdateHealthUI(); // UI 업데이트 추가
    Log.Write($"Health changed to {this.health}"); // 로그 추가
}
```

### 3. 구현의 자유

`get`, 단순히 필드 값을 반환하는 것 이상의 작업 가능.  

- 여러 필드를 조합해 계산된 값을 반환
- 지연 초기화, 필요할 때만 데이터 생성

## 왜 프로퍼티

---

get/set 이쁘게 만들 수 있음.  
또한, 객체의 상태, 접근 제어된다는 의도를 명확히 드러냄.  

당장 `get`, `set` 필요없어도, 프로퍼티로 먼저 만들어둔다면..  
나중에 요구 사항 바뀔 때, API 외관 그대로 유지하고 내부 구현만 수정 가능.  

```csharp
// 처음엔 자동 구현 프로퍼티
public int Health { get; set; }
```

```csharp
// 이후 기능 추가
private int _health;
public int Health
{
    get => _health;
    set
    {
        _health = Math.Max(0, value); // 유효성 검사
        OnHealthChanged();            // 이벤트 호출
    }
}
```

또, 소소한 이점.  
IDE 기능을 통해 해당 변수가 어디에서 어떻게 사용되고 있는지 알고 싶을 때에도 유용.  
`get`, `set` 분리되어 있어서, 어디서 '참조'하고 있는지, 어디서 '설정'하고 있는지 별개로 확인할 수가 있다.  
`public` 필드는 값 접근/설정이 한 번에 보여서, 큰 프로젝트에서는 원하는 것 찾기 힘듦.  

## 응용

---

```cs
public int SomeValue0 { get; set; }
public int SomeValue1 { get; private set; }
[field: SerializeField] public int SomeValue2 { get; private set; }
```

`get`, `set` 마다 접근제한자를 따로 지정해줄 수 있다.  

위처럼 `get`은 `public`, `set`은 `private`로 만들어게 되면, 해당 `property`는 선언된 클래스 안에서는 `set`이 가능하지만, 외부에서는 `get`만 가능하도록 만들어줄 수 있다.  

Unity에서는 `[field: SerializeField]`를 통해 인스펙터에서는 값 설정이 가능하도록 만들 수 있다. 물론 외부 클래스에서는 여전히 `set`이 불가능하다.  

## 메모

---

- 키워드
  - Property (프로퍼티)
- TODO: 꼬리질문
  - How `SerializeField` works
