---
title: "함수, 메서드"
# description: ""
categories: [컴퓨터, 프로그래밍]
tags: []
image: "/assets/img/background/kururu-lab.jpg"

date: 2025-09-28. 12:10 # Init
# last_modified_at: 2025-09-28. 12:10
---

## 말머리

---

- 요약
  - 함수가 더 넓은 개념.
  - 메서드는 클래스/객체에 소속되어, 객체의 행위를 정의.

## Function

---

특정 작업을 수행하는 코드 묶음.  
독립적으로 호출될 수 있으며, 클래스나 객체에 묶여있지 않은 더 넓은 개념.  

## Method

---

**클래스 또는 객체에 소속된 함수**.  
반드시 객체를 통해 호출되며, 해당 객체의 데이터(멤버 변수)에 접근하고 조작. 즉, 객체의 '행위'를 정의.  

## 언어 별 차이

---

- C++:
  - 독립적인 함수를 지원함
  - --> '함수 오버로딩'과 '메서드 오버로딩' 구분 가능
- C#, Java:
  - 모든 함수가 클래스 내에 존재해야 함
  - --> 엄밀히 말하면 모든 오버로딩은 '메서드 오버로딩'

```cs
public static class MathUtils
{
    // C# 정적 메서드
    // - 특정 객체의 상태와 무관하게 입력 값만으로 결과를 반환
    // - 개념적으로는 함수에 가깝지만, C#에서 모든 함수는 클래스 내에 있어야 하므로, 정적 메서드로 표현됨
    public static float CalculateDistance(Vector3 a, Vector3 b)
    {
        return Vector3.Distance(a, b);
    }
}
```

## 메모

--

- 키워드
  - Function (함수)
  - Method (메소드, 메서드)
