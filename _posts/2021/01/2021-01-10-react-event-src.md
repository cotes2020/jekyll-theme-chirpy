---
title: "深挖 React.js 合成事件原理"
date: 2021-01-10
permalink: /2021-01-10-react-event-src/
categories: ["B源码精读", "React"]
---
> 阅读 [《React 事件代理与 stopImmediatePropagation》](https://github.com/youngwind/blog)的笔记。


## React 的事件实现：合成事件


在 reactjs 中，react 没有使用浏览器原生的事件模型，而是使用了「合成事件」。


**「合成事件」是什么？**


例如为 div 绑定 onClick 回调函数，这个回调函数并不是通过 div.addEventListener 绑定上去的。


而是放在 reactjs 内部维护的一个回调数组中，reactjs 只在 document 上绑定了一个事件 dispatchEvent，用来识别事件类型，并且执行对应的回调函数。


当用户点击 div 时，那么点击事件就会冒泡到 document。此时触发 reactjs 为 document 绑定的 dispatchEvent 回调函数，然后执行用户绑定在 div 上的 onClick 回调函数。


reactjs 还会包装原生事件对象，包装后的对象作为参数传给 onClick 函数。但根据 reactjs 文档，可以通过 event.nativeEvent 来查看原生事件对象。


## React 合成事件的好处与不足


**「合成事件」的好处是什么？**


1、跨浏览器兼容：开发者不需要关心浏览器差异，reactjs 暴露给开发者的接口和包装后的事件对象都是一样的


2、统一管理：react 可以在组件卸载时，移除其上的绑定事件，防止事件无限绑定下去


3、内存更优：第 2 点就可以看出来


**「合成事件」的不足是什么？**


1、回调函数中，使用 e.stopPropagation() 只能阻止 reactjs 的合成事件的冒泡，而不能组织原生事件的冒泡
2、支持事件类型不如浏览器本身多：[https://zh-hans.reactjs.org/docs/events.html](https://zh-hans.reactjs.org/docs/events.html)


## 示例：混用 Reactjs 合成事件和原生 DOM 事件


一段很简单的示例代码：


```typescript
/*
 * @Author: dongyuanxin
 * @Date: 2021-01-10 20:22:44
 * @Github: https://github.com/dongyuanxin/blog
 * @Blog: https://dongyuanxin.github.io/
 * @Description: 探究reactjs的事件机制
 */
import React, { useEffect } from "react";
import ReactDom from "react-dom";

const App = () => {
    const borderStyle = {
        border: "2px solid grey",
    };

    useEffect(() => {
        document.addEventListener("click", () => {
            console.log("[browser event] click document");
        });

        document.querySelector(".father").addEventListener("click", () => {
            console.log("[browser event] click father");
        });
    }, []);

    return (
        <div
            className="father"
            onClick={() => {
                console.log("[react event] click father");
            }}
            style={{ ...borderStyle, height: "200px", width: "200px" }}
        >
            parent
            <div
                className="child"
                onClick={(e) => {
                    console.log("[react event] click child");
                    e.stopPropagation();
                }}
                style={{ ...borderStyle, height: "100px", width: "100px" }}
            >
                child
            </div>
        </div>
    );
};

const rootElement = document.getElementById("root");
ReactDom.render(<App />, rootElement);
```


在上面的例子中，点击 child 组件，输出是


```shell
[browser event] click father
[react event] click child
```


**为什么输出是这样的？**


在类名为 child 的 div 上的 onClick 中，调用`e.stopPropagation()`阻止冒泡仅阻止了合成事件冒泡，也就是类名为 father 的 div 的 onClick 触发；却没有组织调用原生 addEventListener 绑定在 father 上的事件，输出了“[browser event] click father”。


document 上的原生绑定的事件没触发，猜测是 reactjs 做了些特殊处理，在 hooks 没出来之前的一些版本，这个也会触发：[https://github.com/youngwind/blog/issues/107](https://github.com/youngwind/blog/issues/107)


触发的流程是：


1、child 被点击，事件冒泡到 parent


2、parent 上有 addEventListener 绑定的事件，执行调用，先输出：[browser event] click father


3、继续冒泡，直到 document


4、执行 reactjs 为 document 绑定的回调，分发通过 onClick 绑定的合成事件，再输出：[react event] click child


**dispatchEvent 模拟 click 事件冒泡的大概思路（伪代码）：**


```javascript
const map = new Map(); // 存储对应元素绑定的 click 回调函数
function dispatchEvent(event) {
    let target = event.target;
    let callback = map.get(target);
    callback && callback(); // 执行回调函数
    while (target.parentNode) {
        // 模拟冒泡：沿 DOM 向上回溯，遍历父节点
        target = target.parentNode;
        callback = map.get(target);
        callback && callback();
    }
}
```


## 参考链接

- 推荐：[React 事件代理与 stopImmediatePropagation](https://github.com/youngwind/blog)
- 官方文档：[React 合成事件](https://zh-hans.reactjs.org/docs/events.html)

