---
title: "React 组件通信方案"
url: "2019-04-08-react-components-communication"
date: 2019-04-08
---

最近在做 react 开发的时候，需要在多级组件之间进行有效通信。这里所说的多级组件，可能是父子组件、兄弟组件，还可能是两个完全不相关的组件。


**那么问题是：有没有一种方法可以统一实现组件之间的通信？不借助 redux 等第三方库，降低依赖性，保证代码简洁，还要保证通用性**。


## 场景介绍


最近在做 react 开发的时候，需要在多级组件之间进行有效通信。这里所说的多级组件，可能是父子组件、兄弟组件，还可能是两个完全不相关的组件。


**那么问题是：有没有一种方法可以统一实现组件之间的通信？不借助 redux 等第三方库，降低依赖性，保证代码简洁，还要保证通用性**。


## 解决方案：订阅-发布模式


这里利用“订阅-发布模式”实现一个事件类：Event。它与 `window.addEventListener`  、 `window.removeEventListener`  类似。**为了方便演示，Event 的实现放在文章最后，下面将展示在具体场景中的应用。**


假设现在有 2 个 react 组件（A 与 B）需要进行通信，组件 A 用户点击后，组件 B 能接收到消息。


组件 A 的大致业务逻辑：


```typescript
import Event from 'event-proxy'
export default ComponentA {
  // ...
  render() {
    return (
      <div>
        {/*
          被点击的时候, 触发 click-event 事件
          注意: 被触发事件的名称, 需要由两个组件进行约定
        */}
        <button onClick={(e) => Event.trigger('click-event')}></button>
      </div>
    )
  }
}

```


组件 B 的大致业务逻辑：


```typescript
import Event from 'event-proxy'
export default ComponentB {
  componentDidMount() {
    // 监听click-event事件, 并且指定 handleClick 为其处理函数
    Event.on('click-event', this.handleClick)
  }
  componentWillUnmount() {
    // 在组件即将卸载的时候, 移除事件监听
    Event.remove('click-event')
  }
  handleClick = () => {
    console.log('组件A被点击了')
  }
  // ...
}

```


## 代码实现


最后附上`event-proxy.js`代码的基本实现：


```typescript
const cache = Symbol("cache");
class EventProxy {
    constructor() {
        this[cache] = {};
    }
    // 绑定事件key以及它的回调函数fn
    on(key, fn) {
        if (!Array.isArray(this[cache][key])) {
            this[cache][key] = [];
        }
        const fns = this[cache][key];
        if (typeof fn === "function" && !fns.includes(fn)) {
            fns.push(fn);
        }
        return this;
    }
    // 触发事件key的回调函数
    trigger(key) {
        const fns = this[cache][key] || [];
        for (let fn of fns) {
            fn(key);
        }
        return this;
    }
    // 移除事件key的回调函数fn
    remove(key, fn) {
        const fns = this[cache][key];
        if (!fns) {
            return this;
        }
        if (typeof fn !== "function") {
            this[cache][key] = null;
            return this;
        }
        for (let i = 0; i < fns.length; ++i) {
            if (fns[i] === fn) {
                fns.splice(i, 1);
                return this;
            }
        }
    }
    clear() {
        this[cache] = null;
        this[cache] = {};
    }
}
const event = new EventProxy();
export default event;

```


## 参考链接

- [设计模式手册之订阅-发布模式](https://godbmw.com/passages/2018-11-18-publish-subscribe-pattern/)
- 淘宝前端团队：[《React 组件通信》](http://taobaofed.org/blog/2016/11/17/react-components-communication/)

