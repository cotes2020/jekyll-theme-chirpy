---
title: "封装自己的React Hooks"
date: 2021-01-12
permalink: /2021-01-12-react-diy-hook/
categories: ["C工作实践分享"]
---
_**Hook 实现原理请参考**_ [一文彻底搞懂react hooks的原理和实现](https://www.notion.so/a3380898e55e49b98a7ec7aae960cb0a) 


## 背景


挑选了 2 个最常用以及有代表性的自定义 hooks：

- useRequest：和数据有关，用于处理异步请求
- useScroll：和操作有关，用于监听鼠标滚动，并且实时拿到最新的滚动数据

## DIY 一个简易的 useRequest


这里模仿 ahooks.js，简单封装下 useRequest 方法


参数是要执行的异步请求函数，返回字段如下：

- loading：异步请求是否在执行中
- error：异步请求失败后，里面有值
- data：异步请求成功后，里面有值
- run：包装作为参数传入的异步请求函数，有更新组件状态的附加逻辑

```typescript
/*
 * @Author: dongyuanxin
 * @Date: 2021-01-09 23:09:19
 * @Github: https://github.com/dongyuanxin/blog
 * @Blog: https://dongyuanxin.github.io/
 * @Description: 自定义hooks -- useRequest
 */

import React, { useState, useEffect, useRef } from "react";
import ReactDOM from "react-dom";

// 自定义 useRequest Hooks
const useRequest = (req) => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [data, setData] = useState(null);

    const run = async () => {
        if (loading) {
            return;
        }

        setLoading(true);
        try {
            const data = await req();
            setData(data);
        } catch (error) {
            setError(error);
        } finally {
            setLoading(false);
        }
    };

    return {
        loading,
        error,
        data,
        run,
    };
};

```


下面是使用这个 Hook 的示例：


```typescript
// 模拟异步请求
const mockData = () => {
    return new Promise((resolve) => {
        const waitMs = Math.floor(Math.random() * 1000);
        console.log(">>> waitMs ", waitMs);
        setTimeout(() => {
            resolve(waitMs);
        }, waitMs);
    });
};

// 下面使用了useRequest来包装异步请求
const App = () => {
    const { run, loading, error, data } = useRequest(mockData);

    useEffect(() => {
        run();
    }, []);

    if (error) {
        return <span>发生错误: {error.message}</span>;
    }

    if (loading || !data) {
        return <span>加载中...</span>;
    }

    return <span>加载完成，数据是：{data}</span>;
};

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);

```


可以看出来，借助 hooks 的特性，可以将状态和数据的逻辑单独封装，作为可复用的函数提供给开发者使用。


封装的思路也很简单，就是将被认为可以复用的逻辑，提炼到一个 hooks 函数（use 开通命名）中即可。在 UI 函数组件中，直接使用这个 hooks 函数获取状态/动作，进行渲染。


**这点有点像 saga-duck.js, 在使用 saga-duck.js 的时候，数据、状态是放在 Duck 文件中维护的，一个 Duck 文件一个类（继承自 BaseDuck）。而 UI 的逻辑是放在 .jsx 中使用。其可以直接使用 Duck 文件中维护的数据和状态。并且 Duck 之间是通过继承和组合的方式，实现逻辑复用。**


但使用 saga-duck.js 毕竟对代码有侵入性，现在借助 hooks，就可以简单快速实现同样的逻辑，并且不引入额外维护和学习成本。


## DIY 一个简易的 useScroll


除了可以封装数据和状态，其它逻辑也可以用 hooks 来封装。例如节流、防抖、状态管理（使用 useContext 和 useReducer）、用户交互


下面就是一个监听指定 dom 滚动的 hooks，当指定 dom 滚动时，会实时计算外界 UI 组件使用后，从 useScroll 返回的值，就是当前滚动的最新位置


**代码实现：**


```typescript
import React, { useState, useEffect, useRef } from "react";
import ReactDom from "react-dom";

const useScroll = (domRef) => {
    const [position, setPosition] = useState([0, 0]);

    useEffect(() => {
        function handleScroll() {
            setPosition([domRef.current.scrollLeft, domRef.current.scrollTop]);
        }

        domRef.current.addEventListener("scroll", handleScroll, false);
        return () =>
            domRef.current.removeEventListener("scroll", handleScroll, false);
    }, []);

    return {
        position,
    };
};

```


在滚动时，能看到界面上的数字一直在变：


```typescript
const App = () => {
    const divRef = useRef(null);
    const { position } = useScroll(divRef);

    return (
        <div>
            <div
                style={{
                    width: "100px",
                    height: "100px",
                    overflow: "scroll",
                    border: "2px solid grey",
                }}
                ref={divRef}
            >
                <div style={{ width: "500px", height: "500px" }}>滚动一下</div>
            </div>
            <div>
                滚动位置：({position[0]}, {position[1]})
            </div>
        </div>
    );
};

const rootElement = document.getElementById("root");
ReactDom.render(<App />, rootElement);

```


## 参考链接

- [创建自己的 hooks](http://www.ruanyifeng.com/blog/2019/09/react-hooks.html)
- [10 分钟教你手写 8 个常用的自定义 hooks](https://juejin.cn/post/6844904074433789959)

