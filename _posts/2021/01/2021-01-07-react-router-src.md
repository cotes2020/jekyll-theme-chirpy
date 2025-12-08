---
title: "手搓 React Router 简易源码"
date: 2021-01-07
permalink: /2021-01-07-react-router-src/
categories: ["B源码精读", "React"]
---
## Hash Router 实现


**实现思路**：监听路由 hash 的变化，调用路由对应的回调函数。


**实现代码**：


```typescript
/*
 * @Author: dongyuanxin
 * @Date: 2021-01-07 23:15:50
 * @Github: https://github.com/dongyuanxin/blog
 * @Blog: https://dongyuanxin.github.io/
 * @Description: react-router hash形式实现
 */

class HashRouter {
    constructor() {
        this.routes = {};
        this.currentUrl = "";
        this._listenLoadAndHashChange();
    }

    /**
     * 为什么要监听load事件？
     * 当页面初次进入的时候，会触发load事件。之后路由上hash值改变，会触发hashchange事件。
     */
    _listenLoadAndHashChange() {
        window.addEventListener("load", () => this.refresh(), false);
        window.addEventListener("hashchange", () => this.refresh(), false);
    }

    /**
     * 匹配当前路由，执行对应路由的回调函数，来进行页面渲染的操作
     */
    refresh() {
        // 切掉hash上的#字符：'#123' => '123'
        this.currentUrl = window.location.hash.slice(1);

        const callback = this.routes[this.currentUrl];
        if (typeof callback === "function") {
            callback();
        }
    }

    /**
     * 为路由指定回调函数
     */
    registerRoute(path, callback) {
        this.routes[path] =
            typeof callback === "function" ? callback : () => {};
    }
}

```


**使用例子（分为 js 和 html）**


在 js 中：

- 初始化 hash router
- 注册路由对应的回调函数（回调函数中进行页面的渲染等逻辑）

```typescript
const hashRouter = new HashRouter();

hashRouter.registerRoute("/", () => {
    console.log("加载 #/ 对应的页面逻辑");
    document.querySelector("body").style.backgroundColor = "white";
});

hashRouter.registerRoute("/blue", () => {
    console.log("加载 #/blue 对应的页面逻辑");
    document.querySelector("body").style.backgroundColor = "blue";
});

```


在 html 中：当点击「首页」或者「blue 页面」链接时，执行上方绑定的回调函数。


```html
<!DOCTYPE html>
<html>
    <head>
        <title>Parcel Sandbox</title>
        <meta charset="UTF-8" />
        <script src="./script.js"></script>
    </head>

    <body>
        <ul>
            <li><a href="#/">首页</a></li>
            <li><a href="#/blue">blue页面</a></li>
        </ul>
    </body>
</html>

```


## History Router 实现


这里模拟 react-router-dom 的 API，封装自己的`<Link />`和`<Router />`。


在 Hash Router 中，通过监听 hashchange 和 load 的事件，可以响应所有的路由 hash 变化。**在 History Router 中，history 模式单页路由和 hash 模式的单页路由类似，都是通过事件回调，来监听路由（hash 或者 url）的变化，进而进行路由的渲染工作。**


但是对于 history 模式的路由，要考虑的情况有些复杂：

- 通过 js 代码，直接调用方法，进行跳转
- 用户点击 a 标签，进行跳转
- 用户点击浏览器的前进/后退按钮

但在 history 路由中，情况有些不同。根据上面三种情况，解决方案细节分别是：

- 暴露内置的 js 方法，用来进行全部组件的重新渲染（匹配当前路由的组件才会重新渲染）
- 包装浏览器的 a 标签，阻止 a 标签点击的默认行为，而是执行全部组件的重新渲染（匹配当前路由的组件才会重新渲染）
- 前进/后退会触发 popstate 事件，在事件回调中执行全部组件的重新渲染（匹配当前路由的组件才会重新渲染）

**代码实现**：分别实现了`<MiniHistoryRoute />`和`<MiniHistoryLink />`这 2 个函数组件，以及 1 个用于 js 路由跳转的`push()`方法。


```typescript
/*
 * @Author: dongyuanxin
 * @Date: 2021-01-08 19:09:34
 * @Github: https://github.com/dongyuanxin/blog
 * @Blog: https://dongyuanxin.github.io/
 * @Description: 封装自己的Router和Link组件
 */
import React, { useEffect, useState } from "react";

const strictMatch = (path) => window.location.pathname === path;

// 存储每个路由对应的强制刷新函数
const routeForceUpdateMap = {};
// 强制刷新所有的组件
const forceUpdateRoutes = () => {
    Reflect.ownKeys(routeForceUpdateMap).forEach((route) =>
        routeForceUpdateMap[route]()
    );
};

window.addEventListener("popstate", () => {
    console.log("触发 popstate");
    forceUpdateRoutes();
});

export const MiniHistoryRoute = (props) => {
    const { path, component } = props || {};

    const [_, forceUpdate] = useState();

    // 在第一次渲染的时候，将此路由的强制刷新函数保存下来
    // 以便在路由变化的时候调用，从而触发当前MiniHistoryRoute的更新
    useEffect(() => {
        routeForceUpdateMap[path] = () => forceUpdate(Date.now());

        return () => {
            delete routeForceUpdateMap[path];
        };
    }, []);

    // 为了方便演示，这里仅支持最简单的严格匹配
    return strictMatch(path) ? React.createElement(component) : null;
};

export const MiniHistoryLink = (props) => {
    const { to } = props;

    // 拦截 a 标签的默认操作
    // 调用 pushState 将url推入浏览器路由栈，并且强制重新渲染
    const handleLinkClick = (event) => {
        event.preventDefault();
        window.history.pushState({}, null, to);
        forceUpdateRoutes();
    };

    return (
        <a href={to} target="_self" onClick={handleLinkClick}>
            {props.children}
        </a>
    );
};

export const push = (to) => {
    window.history.pushState({}, null, to);
    forceUpdateRoutes();
};

```


**使用案例**：当点击 a 链接的时候，会渲染组件 A；同理，b 和 c 链接渲染对应的 B 和 C 组件。


```typescript
import React from "react";
import ReactDOM from "react-dom";
import { MiniHistoryRoute, MiniHistoryLink } from "./mini-history-router";

const A = () => <div>A</div>;

const B = () => <div>B</div>;

const C = () => <div>C</div>;

const App = () => {
    return (
        <>
            <ul>
                <li>
                    <MiniHistoryLink to="/a">a</MiniHistoryLink>
                </li>
                <li>
                    <MiniHistoryLink to="/b">b</MiniHistoryLink>
                </li>
                <li>
                    <MiniHistoryLink to="/c">c</MiniHistoryLink>
                </li>
            </ul>
            <MiniHistoryRoute path="/" component={A} />
            <MiniHistoryRoute path="/a" component={A} />
            <MiniHistoryRoute path="/b" component={B} />
            <MiniHistoryRoute path="/c" component={C} />
        </>
    );
};

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);

```


### 题外话：怎么在 react hooks 中强制重新更新组件？


1、在 Class 时代，可以通过调用实例上的 forceUpdate()方法，来触发组件的强制更新


2、在 Hooks 时代，组件状态的改变，就会触发更新。所以通过改变状态属性即可。


## 参考链接

- Hash Router 实现：[前端路由实现与 react-router 源码分析](https://github.com/joeyguo/blog/issues/2)
- History Router 实现：[单页面应用路由实现原理：以 React-Router 为例](https://github.com/youngwind/blog/issues/109)
- 函数组件如何模拟 class forceUpdate：[React Docs](https://zh-hans.reactjs.org/docs/hooks-faq.html#is-there-something-like-forceupdate)
- [理解 forceUpdate()](https://juejin.cn/post/6844903505145282568)

