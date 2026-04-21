---
title: "React Native 리액트 네이티브 - Context"
# description: ""
categories: [컴퓨터, 🌒Programming]
tags: [Mobile, React-Native]
image: "/assets/img/background/kururu-lab.jpg"
hidden: true

date: 2023-11-15. 13:16
# last_modified_at: 2023-11-22. 14:54
# last_modified_at: 2023-12-05. 15:13
# last_modified_at: 2023-12-06. 10:33
last_modified_at: 2024-08-29. 21:26
---

## Context

---

@ U 기말고사 출제: Context, 306p, useToggleTheme, useContext  

컴포넌트의 속성은 부모 컴포넌트가 자식 컴포넌트로 어떤 정보를 전달하려고 할 때 사용하는 메커니즘이다.  

그런데 부모 컴포넌트가 직계 자식이 아닌 손자나 중손자 컴포넌트에 속성을 전달하려고 하면 '지속적인 속성 전달'을 해야 한다. (비효율적)  

```js
<자식컴포넌트 shared_props={shared_props} />
({shared_props}) => <손자컴포넌트 shared_props={shared_props} />
({shared_props}) => <증손자컴포넌트 shared_props={shared_props} />

// shared_props를 단지 증손자 컴포넌트에서만 쓰더라도 자식 컴포넌트와 손자 컴포넌트 모두에게 속성을 전달해야만 함
```

이런 속성 전달을 대신하기 위해 사용되는 것이 컨텍스트이다.  

컨텍스트는, react 패키지에서 제공하는 createContext와 useContext 훅을 통해 이루어진다.  

createContext 함수 호출로 얻은 값이 컨텍스트다.  

```js
<Provider value={} />  // 부모, by createContext
useContext ~ // 자식
useContext ~ // 손자
useContext ~ // 증손자
```

컨텍스트를 쓰는 컴포넌트는 이름에 'Provider'를, 커스텀 훅에는 이름에 'use~'를 쓴다.  

컨텍스트 기능을 구현한 react-native-paper와 같은 패키지 또한 'Provider' 가 들어간 컴포넌트와 Provider가 제공하는 정보를 사용할 수 있게 하는 useTheme와 같은 커스텀 훅을 제공한다.  

## Theme

---

리액트 네이티브 앱에서 테마 기능을 제공하려면 반드시 컨텍스트를 이용해야 한다.  

react-native-paper 패키지는 Provider, AppearanceProvider 컴포넌트와 useColorScheme 커스텀 훅을 제공한다.  

리액트 / 리액트 네이티브에서 제공자를 뜻하는 'Provider'가 들어간 컴포넌트는 항상 최상위 컴포넌트 (Root Component)로 동작해야 한다.  

이런 컴포넌트는 콘텍스트 기능을 사용하여 만든다.  

'Provider' 컴포넌트를 제공하는 패키지는 항상 이 컴포넌트가 제공하는 기능을 사용할 수 있도록 'use~' 형태의 커스텀 훅을 제공한다.  

(AppearanceProvider는 useColorScheme 커스텀 훅을 제공한다.)  

```js
const scheme = useColorScheme() // 'dark' or 'light'
```

useColorScheme는 현재 폰이 다크모드로 동작하고 있는지를 확인한다.  

useColorScheme는 단순히 모드만 확인하는 것이고, 실제로 모드에 따라 바탕색과 글자 색을 바꾸려면 또 다른 'Provider' 컴포넌트가 필요하다.  

```js
import {Provider as 특정이름} from 'react-native-paper'
```

Provider는 너무 일반적인 이름이라, 특정 이름으로 바꿔 사용하는 것이 좋다.  

Provider는 반드시 상위 컴포넌트로 동작해야 하며, 이때 서로 다른 Provider를 중첩시켜야 할 때 순서는 자유롭게 설정할 수 있다.  

컨텍스트 기능은 완전히 독립적으로 동작하기에, 다른 컨텍스트 기능에 전혀 영향을 주지 않기 때문이다.  

Provider는 theme 라는 선택 속성을 제공하는데, 이는 react-native-paper 패키지에서 제공하는 DefaultTheme, DarkTheme 속성으로 설정할 수 있다.  

```js
const theme = useTheme()
const {fonts, colors} = theme
```

useTheme 커스텀 훅은 Provider의 theme 속성에 설정된 값을 컨텍스트로 불러온다.

theme 객체에 비구조화 할당 구문을 적용하여 fonts와 colors 속성을 얻을 수 있다.  

```js
type SomeProps =
{
    theme: any
}

const Some: FC<SomeProps> = ({theme}) => {}
```

원래 같으면 위 같은 코드를 직접 구현하고, 속성을 넘겨 받아 사용해야 했다.  

```js
export type ContextType = {  /*공유 속성*/ }
const defaultContextType: ContextType = { /* 공유 속성 초깃값 */ }
const SomeContext = createContext<ContextType>(defaultContextType)
```

컨텍스트를 사용하려면 가장 먼저 컨텍스트 객체를 만들어야 하고, 컨텍스트 객체는 createContext를 통해 만들 수 있다.  

createContext를 통해 만든 컨텍스트 객체는 Provider와 Consumer 컴포넌트를 제공한다.  

Provider는 앞서 언급한 Provider들과 같은 역할을 하는 컴포넌트고, Consumer는 Provider가 제공하는 기능을 사용하는 클래스 컴포넌트를 위한 컴포넌트이다.  

[참고: 클래스 컴포넌트](/posts/web-browser/)  

Provider 컴포넌트는 value와 children 속성이 있는 ProviderProps 속성을 제공한다.  

```js
/* 타입 변수 T == createContext<T> */
interface ProviderProps<T>
{
    /* 컨텍스트 Provider가 제공하는 기능 */
    value: T;
    /* 컴포넌트의 children과 같은 의미 */
    children?: ReactNode;
}
```

react 패키지가 제공하는 useContext 훅은, 매개변수로 전달받은 컨텍스트 객체가 제공하는 Provider 컴포넌트의 value 속성값을 반환하는 훅이다.  

useContext 훅을 사용하는 코드 패턴은 아래와 같으며, useColorScheme, useTheme 커스텀 훅도 이런 코드 패턴으로 만들어진 커스텀 훅이다.  

```js
export const useSome = () =>
{
    const value = uesContext(SomeContext)
    return value;
}
```

@ TODO: 322p Switch  

## useRef, useImperativeHandle

---

@ U 기말고사 출제: useRef, 306p  

useRef와 useImperativeHandle 훅은 ref 속성에 적용하는 값을 만드는 훅이다.  

리액트와 리액트 네이티브가 제공하는 컴포넌트, App 같은 사용자 컴포넌트에는 모두 ref 속성이 있다.  

코어 컴포넌트에서는 그대로 사용할 수 있는데, ref속성이 있는 사용자 컴포넌트는 forwardRef 함수로 생성해야 한다는 조건이 있다.  

```js
function useRef<T>(initialValue: T): MutableRefObject<T>;
function useRef<T>(initialValue: T | null): RefObject<T>;
```

### ref 속성

리액트/리액트 네이티브에서 제공하는 코어 컴포넌트 중에는 메소드를 제공하는 것이 있다.  

- TextInput 컴포넌트: focus(), blur()
- ScrollView 컴포넌트 & FlatList 컴포넌트: scrollToTop(), scrollToEnd()  

컴포넌트의 메소드를 호출하려면 컴포넌트의 리액트 요소 (React Element, 개체 지향 언어에서 클래스의 인스턴스와 같은 개념) 을 얻을 수 있어야, 개체.메소드() 형태로 호출할 수 있다.  

리액트와 리액트 네이티브는 컴포넌트가 제공하는 메소드를 호출할 수 있도록 ref 속성을 제공한다. 컴포넌트의 인스턴스를 얻을 수 있으며 이를 이용하여 ref.메소드() 형태로 호출할 수 있다.  

### 구현

```js
// T는 FlatList, ScrollView, TextInput 같은 컴포넌트
interface RefAttributes<T> extends Attributes
{ ref?: Ref<T> }
interface RefObject<T>
{ readonly current: T | null; }

function useRef<T>(initialValue: T): MutableRefObject<T>;
function useRef<T>(initialValue: T | null): RefObject<T>;

// i.e.
const someRef = useRef<Some | null>(null)
<Some ref={someRef} />

someRef.current?.someMethod()
```
