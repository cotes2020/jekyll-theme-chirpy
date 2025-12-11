---
title: "React State Hooks：useContext && useReducer"
date: 2021-01-10
permalink: /2021-01-10-react-state-hook/
tags: [函数式编程, 状态管理]
---
_**Hook 实现原理请参考**_ [一文彻底搞懂react hooks的原理和实现](https://www.notion.so/a3380898e55e49b98a7ec7aae960cb0a) 


## 前言


对于 react 函数组件，管理组件状态最常用的 hooks 就是`useState`。


但在某些场景下，例如需要跨多层组件传递状态，或者组件中状态过多（一堆 useState），react hooks 提供了更优雅的解决方法。


## useContext


1、可以用来传递状态和方法（例如修改状态的方法）


2、能够避免多级组件，层层通过 props 来嵌套传递。可以直接在任意组件中直接使用 context。


3、一般将所有的 context 都收归到一个文件夹下进行统一管理，对外暴露提供引用。方便维护和定位问题。


下面的代码是在子组件中使用状态，并且调用上级组件状态更新方法，触发状态更新：


```typescript
/*
 * @Author: dongyuanxin
 * @Date: 2021-01-05 23:44:05
 * @Github: https://github.com/dongyuanxin/blog
 * @Blog: https://0x98k.com/
 * @Description: useContext
 */

import React, { useReducer, createContext, useContext, useState } from "react";
import ReactDOM from "react-dom";

// step1: 使用createContext创建上下文
const UserContext = createContext({});

// step2: 通过给Context.Provider的props.value赋值来指定上下文的值
const App = () => {
    const [name, setName] = useState("xin-tan");
    const [age, setAge] = useState(22);

    return (
        <UserContext.Provider value={{ name, age, setAge, setName }}>
            <div
                style={{ cursor: "pointer" }}
                onClick={() => {
                    setAge(age + 1);
                }}
            >
                （APP组件）点我增加年龄
            </div>
            <hr></hr>
            <B />
        </UserContext.Provider>
    );
};

const B = () => {
    return <C />;
};

// step3: 在任意子组件中，调用useContext。参数是context，返回值是具体的值。
const C = () => {
    const { age, setAge } = useContext(UserContext);

    return (
        <>
            <div>姓名：{name}</div>
            <div>年龄：{age}</div>
            <div
                style={{ cursor: "pointer" }}
                onClick={() => {
                    setAge(age + 1);
                }}
            >
                （C组件）点我增加年龄
            </div>
        </>
    );
};

const rootElement = document.getElementById("app");
ReactDOM.render(<App />, rootElement);
```


## useReducer


useReducer 和 redux 的使用基本一致：


1、返回中有 state，上面挂着状态


2、Reducer 用来执行状态更新


3、通过 dispatch 来发起状态更新


4、useReducer 的第三个参数接受一个函数作为参数，并把第二个参数当作函数的参数执行，返回的值作为初始值


下面是 useReducer 的使用案例：


```typescript
/*
 * @Author: dongyuanxin
 * @Date: 2021-01-06 00:18:34
 * @Github: https://github.com/dongyuanxin/blog
 * @Blog: https://0x98k.com/
 * @Description: useReducer 使用
 */

import React, { useReducer } from "react";
import ReactDOM from "react-dom";

const initialState = {
    count: 0,
};

function reducer(state, action) {
    switch (action.type) {
        case "increment":
            return { count: state.count + action.payload };
        case "decrement":
            return { count: state.count - action.payload };
        default:
            throw new Error();
    }
}

function App() {
    const [state, dispatch] = useReducer(reducer, initialState);
    const [state2, dispatch2] = useReducer(reducer, initialState);
    return (
        <div>
            Count: {state.count}
            <button onClick={() => dispatch({ type: "increment", payload: 5 })}>
                +
            </button>
            <button onClick={() => dispatch({ type: "decrement", payload: 5 })}>
                -
            </button>
            Count2: {state2.count}
            <button
                onClick={() => dispatch2({ type: "increment", payload: 5 })}
            >
                +
            </button>
            <button onClick={() => dispatch2({ type: "decrement", payload: 5 })}>-</button>
        </div>
    );
}

const rootElement = document.getElementById("app");
ReactDOM.render(<App />, rootElement);

```


## 参考链接

- [十个案例学会 react hooks](https://zhuanlan.zhihu.com/p/60925430)
- [创建自己的 hooks](http://www.ruanyifeng.com/blog/2019/09/react-hooks.html)

