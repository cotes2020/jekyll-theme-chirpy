---
title: "React Native 리액트 네이티브 - Hook"
# description: ""
categories: [컴퓨터, 🌒Programming]
tags: [Mobile, React-Native]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2023-10-31. 15:31
# last_modified_at: 2023-11-14. 15:14
# last_modified_at: 2023-11-21. 15:09
# last_modified_at: 2023-11-22. 13:07
# last_modified_at: 2023-12-05. 15:55
# last_modified_at: 2023-12-06. 10:33
last_modified_at: 2024-08-29. 21:26
---

## 리액트 훅 - React Hooks

---

@ U 중간고사 출제: 리액트 훅 & 상태 개념  

React 프레임워크,  

컴포넌트 기법으로 가상 DOM 객체를 만들고나서,  
가상 DOM 객체에 어떤 변화가 감지되면, 해당 변화만 재렌더링하여 전체 렌더링 속도를 빠르게  

Old  
객체 지향 언어의 상속 개념에 맞춘, 클래스 형태로 제작되었음  
→ 클래스 컴포넌트 기술 ← 코드 작성 복잡  
→ 60/1s 같은 빠른 재렌더링 시 정상적인 렌더링이 안되는 버그  

New  
구현 복잡함을 덜어내고자 컴포넌트를 함수 형태로 만들 수 있도록  
함수 컴포넌트가 어떤 값을 Persistence - 유지 할 수 있도록, New 데이터 Cache 시스템  
Cache 시스템 쉽게 사용할 수 있도록 use~ 로 시작하는 여러 `API` 제공  
→ React Hooks - `리액트 훅`  

즉, 함수 내 로컬변수를 클래스의 전역변수처럼 사용하기 위해  

로컬변수가 실제 데이터를 가지지 않고,  
실제 데이터는 어디인가 캐시하고서  
로컬변수는 필요할 때 캐시한 데이터를 찾을 수 있는 일종의 키를 저장하는 방법을 고안 → Reference  

@ 포인터, 참조타입 유사한 개념인듯?  

원래 사용자가 직접 구현해 사용했지만,  
리액트에서 제공한 useMemo 훅을 통해 사용  

↓ 아래는 기존 구현 방법

```TypeScript
const cache: Record<string, any> = {}

export const createOrUse = <T>(key: string, callback: () => T) =>
{
    if (!cache[key])
        cache[key] = callback()
    return cache[key]
}

const temp = createOrUse('Temp', () => createTemp)

/*
    Record<Key, Type>
    TypeScript에서 제공하는 객체 타입
    Type은 아무 타입이나 올 수 있음

    있으면 꺼내서 반환  
    없으면 callback으로 초기화 후 반환  
*/
```

i.e.  
컴포넌트 데이터 관리 - useMemo, useCallback, useState, useReducer  
컴포넌트 생명주기 대응 - useEffect, useLayoutEffect  
컴포넌트 간의 정보 공유 - useContext  
컴포넌트 메소드 호출 - useRef, useImperativeHandle  

## 의존성 - Dependency

---

캐시를 갱신해야하는 특정 조건/상황들  

`의존성 목록` 중 하나라도 충족되면,  
자동으로 캐시 갱신 (콜백) → 재렌더링  

## useMemo

---

@ U 기말고사 출제: useMemo  

```js
const 캐시된_데이터 = useMemo(초기값, [의존성1, 의존성2, ...])

const 캐시된_데이터 = useMemo(콜백, [의존성1, 의존성2, ...])
콜백 = () => 초기값
```

값(, 함수)을 [메모이제이션](/posts/algorithm-memoization/)  
→ 함수: () => 콜백  

useCallback이 있는데, useMemo로 함수를 메모이제이션 하는 경우?  
→ useMemo를 쓰면 함수와 그 `결과 값`을 함께 메모이제이션  

@ 222p, `useMemo(() => fibonacci, [])...` 동일한 입력값에 대해 함수가 반복해서 호출되는 것을 방지 (기말고사 X)  

## useCallback

---

@ U 기말고사 출제: useCallback  

```js
const 캐시콜백 = useCallback(초기콜백, [의존성1, 의존성2, ...])
```

함수를 [메모이제이션](/posts/algorithm-memoization/)  

재렌더링마다 지속적으로 만들어질 수 있는 콜백 함수를, useCallback을 통해 저장해서 재사용  

## 상태

---

@ U 중간고사 출제: 리액트 훅 & 상태 개념  

시간이 지나도 값이 유지되며, 필요에 따라서는 값을 바꿀 수 있는 어떤 것  
클래스의 멤버 속성이나 전역 변수 형태로 만들게 되며, 보통 함수 몸통의 지역 변수 형태로는 상태를 만들지 못함  

## useState

---

@ U 중간고사 출제: 리액트 훅 & 상태 개념  
@ U 기말고사 출제: useState  

```js
import React, {useState} from 'react'

{ /* 타입 정의, S - State Type */ }
function useState<S>(initialState: S | (() => S)): [S, Dispatch<SetStateAction<S>>]

{ /* 사용 방법 */ }
const [값, Setter] = useState(초깃값)
const [값, Setter] = useState<S>(초깃값)
{ /* 값을 변경하는 함수 (Setter)를 호출하면 값 부분을 변경하고, 컴포넌트를 재렌더링 */}
{ /* 재렌더링 때문에 useState 훅은 값과 Setter 함수를 Tuple 형태의 배열로 반환 */}
```

`함수 컴포넌트 내부에 클래스의 멤버 속성처럼 값을 유지하고 변경할 수 있는 상태를 만들 수 있게 한다`  

상태 저장 (지역변수를 전역변수처럼)  

@ 배열에 적용하는 비구조화 할당 구문  
할당 받는 변수 이름을 자유롭게  

@ 객체에 적용하는 비구조화 할당 구문  

## 컴포넌트의 생명주기

---

컴포넌트를 생성하여 최초 렌더링 과정을 끝마침: 컴포넌트를 마운트했다. - Mount  

마운트된 컴포넌트는 구현 로직에 따라 재렌더링을 거듭하다, 어떤 시점에서 구현 로직에 의해 파괴되어 사라짐  

컴포넌트가 파괴되어 더는 렌더링 과정에 참여하지 않음: 컴포넌트가 언마운트됐다. - Unmount  

마운트 ~ 언마운트 과정을 합하여, 컴포넌트의 생명주기라 표현한다. - Lifecycle  

## useEffect, useLayoutEffect

---

@ U 중간고사 출제: useEffect  
@ U 기말고사 출제: useEffect (setInterval?)  

컴포넌트의 생명주기와 관련있는 생명주기 훅.  
컴포넌트 마운트, 의존 목록 조건, 언마운트 시 처리 할 작업.  

```js
{ /* React Hook, React가 제공 */}
import React, {useEffect} from 'react'

{ /* 의존성 목록에 있는 조건 중 어느 하나라도 충족되면 그때마다 콜백 함수 실행 */}
useEffect(콜백, 의존성목록)
useLayoutEffect(콜백, 의존성목록)

콜백 = () => { }
{ /* 콜백 함수도 함수 반환 가능, 컴포넌트를 언마운트 할 때 단 한 번 실행 됨*/}
콜백 = () => { return 반환함수 }

{ /* 컴포넌트 생성할 때 한 번만 실행하려면 의존성 목록에 빈 배열 [] */}
useEffect(() => {}, [])

{ /* 함수 반환 가능 */}
useEffect(() => { /* some */ return () => {} }, [])

{ /* setInterval 한 번만 실행되어야 하는데, 재렌더링마다 실행되기에 */}
{ /* useEffect(() => {}, [])를 통해 처음 만들어질 때만 한 번 실행되도록 */}
```

콜백에서 함수를 반환할 수 있는데, 이 반환 함수는 컴포넌트를 언마운트할 때 단 한 번만 실행한다.  

의존성이 변화하면 콜백이 반환한 종료 함수를 호출하여 콜백을 파괴하고, 자신의 매개변수로 입력한 콜백을 다시 호출  
@ TODO: 이해 못함  

리액트 네이티브 코어 컴포넌트는 onLayout 이벤트 속성을 제공한다, 'react-native' 패키지는 LayoutChangeEvent 타입을 제공한다.  

LayoutChangeEvent는 onLayout 이벤트 속성에 설정하는 이벤트 처리기의 입력 매개변수 타입이다.  

LayoutChangeEvent의 nativeEvent 속성을 통해 특정 타입 layout을 얻을 수 있다.  

```js
const onLayout = (e: LayoutChangeEvent) => { /* e.nativeEvent ~ */}

export interface LayoutChangeEvent
{
    nativeEvent:
    {
        layout: LayoutRectangle;
    };
}
```

onLayout 이벤트를 호출했다는 것은 컴포넌트의 렌더링이 끝났다는 것을 의미한다.  

마운트 과정  
컴포넌트 렌더링 시작 → useLayoutEffect → 화면에 나타남 → useEffect → onLayout  

언마운트 과정  
컴포넌트 언마운트 시작 → useEffect 반환 함수 호출 → useLayoutEffect 반환 함수 호출 → 컴포넌트 파괴  

useLayoutEffect 훅은 동기 Synchronous로 실행하고, useEffect 훅은 비동기 Asynchronous로 실행한다.  

useLayoutEffect 훅은 콜백 함수가 끝날 때까지 프레임워크가 기다리고, useEffect 훅은 콜백 함수를 기다리지 않는다.  

가능하면 useEffect 훅을 사용하는 것이 좋다. (리액트 공식 문서 권장 사항)  

## 커스텀 훅 - Custom Hook

---

Something like Design Pattern  

여러 리액트 훅과 커스텀 훅을 조합하여 재사용할 수 있는 새로운 훅 함수를 만드는 기능이다.  

컴포넌트의 훅 함수 코드 패턴이 비슷하기에, 이런 훅 호출을 조합하여 간결하게 표현할 수 있다. (추상화, 긴 코드 함수로 만들어버리는 것처럼)  

리액트 훅과 마찬가지로, 함수 이름은 항상 'use~'로 시작해야 한다.  

[useHooks](https://usehooks.com/), [useHooks-ts](https://usehooks-ts.com/) 같은 사이트에서 다른 사람들이 만든 훅, 훅 라이브러리를 참고할 수 있다.  

## useRef, useImperativeHandle

---

[Context, Ref](/posts/react-native-context/)
