---
title: "Node.js events 模块研究"
url: "2020-01-10-nodejs-events"
date: 2020-01-10
---

## 概述


读了 `events` 模块的文档，研究了几个有意思的问题：

- 🤔️ 事件驱动模型
- 🤔️ 优雅的错误处理
- 🤔️ 监听器器队列顺序处理
- 🤔️ 内存管理与防止泄漏
- 🔨 配合 Promise 使用

## 事件驱动模型


Nodejs 使用了一个事件驱动、非阻塞 IO 的模型。`events`模块是事件驱动的核心模块。很多内置模块都继承了`events.EventEmitter`。


自己无需手动实现这种设计模式，直接继承`EventEmitter`即可。代码如下：


```typescript
const { EventEmitter } = require("events");
class MyEmitter extends EventEmitter {}
const ins = new MyEmitter();
ins.on("test", () => {
    console.log("emit test event");
});
ins.emit("test");
```


## 优雅的错误处理


根据文档，应该 EventEmitter 实例的`error`事件是个特殊事件。**推荐做法是**：在创建实例后，应该立即注册`error`事件。


```typescript
const ins = new MyEmitter();
ins.on("error", error => {
    console.log("error msg is", error.message);
});
```


注册`error`事件后，**我原本的理解**是，所有事件回掉逻辑中的错误都会在 EventEmitter 内部被捕获，并且在内部触发 `error` 事件。


也就是说下面代码，会打印："error msg is a is not defined"。


```typescript
ins.on("test", () => {
    console.log(a);
});
ins.emit("test");
```


然而，错误并没有捕获，直接抛出了异常。由此可见，EventEmitter 在执行内部逻辑的时候，并没有`try-catch`。这个原因，请见[Node Issue](https://github.com/nodejs/node/issues/21002)。简单来讲，Error 和 Exception 并不完全一样。


如果按照正常想法，不想每一次都在外面套一层`try-catch`，那应该怎么做呢？我的做法是在 EventEmitter 原型链上新增一个`safeEmit`函数。


```typescript
EventEmitter.prototype.safeEmit = function(name, ...args) {
    try {
        return this.emit(name, ...args);
    } catch (error) {
        return this.emit("error", error);
    }
};
```


如此一来，运行前一段代码的 Exception 就会被捕获到，并且触发`error`事件。前一段代码的输出就变成了：


```shell
error msg is a is not defined
```


## 监听器队列顺序处理


对于同一个事件，触发它的时候，函数的执行顺序就是函数绑定时候的顺序。官方库提供了`emitter.prependListener()`和 `emitter.prependOnceListener()` 两个接口，可以让新的监听器直接添加到队列头部。


但是如果想让新的监听器放入任何监听器队列的任何位置呢？在原型链上封装了 `insertListener` 方法。


```typescript
EventEmitter.prototype.insertListener = function(
    name,
    index,
    callback,
    once = false
) {
    // 如果是once监听器，其数据结构是 {listener: Function}
    // 正常监听器，直接是 Function
    const listeners = ins.rawListeners(name);
    const that = this;
    // 下标不合法
    if (index > listeners.length || index < 0) {
        return false;
    }
    // 绑定监听器数量已达上限
    if (listeners.length >= this.getMaxListeners()) {
        return false;
    }
    listeners.splice(index, 0, once ? { listener: callback } : callback);
    this.removeAllListeners(name);
    listeners.forEach(function(item) {
        if (typeof item === "function") {
            that.on(name, item);
        } else {
            const { listener } = item;
            that.once(name, listener);
        }
    });
    return true;
};
```


使用起来，效果如下：


```typescript
const ins = new MyEmitter();
ins.on("error", error => {
    console.log("error msg is", error.message);
});
ins.on("test", () => {
    console.log("test 1");
});
ins.on("test", () => {
    console.log("test 2");
});
// 监听器队列中插入新的监听器，一个是once类型，一个不是once类型
ins.insertListener(
    "test",
    0,
    () => {
        console.log("once test insert");
    },
    true
);
ins.insertListener("test", 1, () => {
    console.log("test insert");
});
```


连续调用两次`ins.emit("test")`，结果输出如下：


```shell
# 第一次
once test insert
test insert
test 1
test 2
# 第二次: once 类型的监听器调用一次后销毁
test insert
test 1
test 2
```


## 内存管理与防止泄漏


在绑定事件给监听器（全局实例，不会被自动回收销毁）的时候，如果事件没有被 remove，那么存在内存泄漏的风险。


我知道的常见做法如下：

- 经常 CR，移除不需要的事件监听器
- 通过`once`绑定监听器，调用一次后，监听器被自动移除
- [推荐]hack 一个更安全的`EventEmitter`

## TODO: 配合 Promise 使用


## 参考链接

- [NodeJS Issue](https://github.com/nodejs/node/issues/21002)
- [Docs: events](http://nodejs.cn/api/events.html)

